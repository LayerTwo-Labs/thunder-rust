use std::{
    collections::{HashMap, HashSet},
    fmt::Debug,
    net::SocketAddr,
    path::Path,
    sync::Arc,
};

use bip300301::bitcoin;
use futures::{stream, StreamExt, TryFutureExt};
use rustreexo::accumulator::pollard::Pollard;
use tokio::{spawn, task::JoinHandle};
use tokio_stream::StreamNotifyClose;

use crate::{
    archive::{self, Archive},
    mempool::{self, MemPool},
    net::{
        self, Net, PeerConnectionInfo, PeerInfoRx, PeerRequest, PeerResponse,
    },
    state::{self, State},
    types::*,
};

pub const THIS_SIDECHAIN: u8 = 9;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("address parse error")]
    AddrParse(#[from] std::net::AddrParseError),
    #[error("archive error")]
    Archive(#[from] archive::Error),
    #[error("bincode error")]
    Bincode(#[from] bincode::Error),
    #[error("drivechain error")]
    Drivechain(#[from] bip300301::Error),
    #[error("heed error")]
    Heed(#[from] heed::Error),
    #[error("quinn error")]
    Io(#[from] std::io::Error),
    #[error("mempool error")]
    MemPool(#[from] mempool::Error),
    #[error("net error")]
    Net(#[from] net::Error),
    #[error("peer info stream closed")]
    PeerInfoRxClosed,
    #[error("state error")]
    State(#[from] state::Error),
    #[error("Utreexo error: {0}")]
    Utreexo(String),
}

async fn submit_block(
    env: &heed::Env,
    archive: &Archive,
    drivechain: &bip300301::Drivechain,
    mempool: &MemPool,
    state: &State,
    header: &Header,
    body: &Body,
) -> Result<(), Error> {
    let last_deposit_block_hash = {
        let txn = env.read_txn()?;
        state.get_last_deposit_block_hash(&txn)?
    };
    let bundle = {
        let two_way_peg_data = drivechain
            .get_two_way_peg_data(
                header.prev_main_hash,
                last_deposit_block_hash,
            )
            .await?;
        let mut txn = env.write_txn()?;
        let height = archive.get_height(&txn)?;
        state.validate_body(&txn, &header.roots, body, height)?;
        if tracing::enabled!(tracing::Level::DEBUG) {
            let block_hash = header.hash();
            let merkle_root = body.compute_merkle_root();
            state.connect_body(&mut txn, body)?;
            tracing::debug!(%height, %merkle_root, %block_hash,
                                "connected body")
        } else {
            state.connect_body(&mut txn, body)?;
        }
        state.connect_two_way_peg_data(&mut txn, &two_way_peg_data, height)?;
        let bundle = state.get_pending_withdrawal_bundle(&txn)?;
        archive.append_header(&mut txn, header)?;
        archive.put_body(&mut txn, header, body)?;
        for transaction in &body.transactions {
            mempool.delete(&mut txn, &transaction.txid())?;
        }
        let accumulator = state.get_accumulator(&txn)?;
        mempool.regenerate_proofs(&mut txn, &accumulator)?;
        txn.commit()?;
        bundle
    };
    if let Some(bundle) = bundle {
        let () = drivechain
            .broadcast_withdrawal_bundle(bundle.transaction)
            .await?;
    }
    Ok(())
}

struct NetTaskContext {
    env: heed::Env,
    archive: Archive,
    drivechain: bip300301::Drivechain,
    mempool: MemPool,
    net: Net,
    state: State,
}

struct NetTask {
    ctxt: NetTaskContext,
    peer_info_rx: PeerInfoRx,
}

impl NetTask {
    async fn handle_response(
        ctxt: &NetTaskContext,
        addr: SocketAddr,
        resp: PeerResponse,
        req: PeerRequest,
    ) -> Result<(), Error> {
        match (req, resp) {
            (
                req @ PeerRequest::GetBlock { height },
                ref resp @ PeerResponse::Block {
                    ref header,
                    ref body,
                },
            ) => {
                let (tip_hash, tip_height) = {
                    let rotxn = ctxt.env.read_txn()?;
                    let tip_height = ctxt.archive.get_height(&rotxn)?;
                    let tip_hash = ctxt.archive.get_best_hash(&rotxn)?;
                    (tip_hash, tip_height)
                };
                if height != tip_height + 1 {
                    return Ok(());
                }
                if header.prev_side_hash != tip_hash {
                    tracing::warn!(%addr, ?req, ?resp, "Cannot handle reorg");
                    return Ok(());
                }
                submit_block(
                    &ctxt.env,
                    &ctxt.archive,
                    &ctxt.drivechain,
                    &ctxt.mempool,
                    &ctxt.state,
                    header,
                    body,
                )
                .await
            }
            (
                PeerRequest::GetBlock { height: req_height },
                PeerResponse::NoBlock {
                    height: resp_height,
                },
            ) if req_height == resp_height => Ok(()),
            (
                PeerRequest::PushTransaction { transaction: _ },
                PeerResponse::TransactionAccepted(_),
            ) => Ok(()),
            (
                PeerRequest::PushTransaction { transaction: _ },
                PeerResponse::TransactionRejected(_),
            ) => Ok(()),
            (
                req @ (PeerRequest::GetBlock { .. }
                | PeerRequest::Heartbeat(_)
                | PeerRequest::PushTransaction { .. }),
                resp,
            ) => {
                // Invalid response
                tracing::warn!(%addr, ?req, ?resp,"Invalid response from peer");
                let () = ctxt.net.remove_active_peer(addr);
                Ok(())
            }
        }
    }

    async fn run(self) -> Result<(), Error> {
        enum MailboxItem {
            AcceptConnection(Result<(), Error>),
            PeerInfo(Option<(SocketAddr, Option<PeerConnectionInfo>)>),
        }
        let accept_connections = stream::try_unfold((), |()| {
            let env = self.ctxt.env.clone();
            let net = self.ctxt.net.clone();
            let fut = async move {
                let () = net.accept_incoming(env).await?;
                Result::<_, Error>::Ok(Some(((), ())))
            };
            Box::pin(fut)
        })
        .map(MailboxItem::AcceptConnection);
        let peer_info_stream = StreamNotifyClose::new(self.peer_info_rx)
            .map(MailboxItem::PeerInfo);
        let mut mailbox_stream =
            stream::select(accept_connections, peer_info_stream);
        while let Some(mailbox_item) = mailbox_stream.next().await {
            match mailbox_item {
                MailboxItem::AcceptConnection(res) => res?,
                MailboxItem::PeerInfo(None) => {
                    return Err(Error::PeerInfoRxClosed)
                }
                MailboxItem::PeerInfo(Some((addr, None))) => {
                    // peer connection is closed, remove it
                    tracing::warn!(%addr, "Connection to peer closed");
                    let () = self.ctxt.net.remove_active_peer(addr);
                    continue;
                }
                MailboxItem::PeerInfo(Some((addr, Some(peer_info)))) => {
                    match peer_info {
                        PeerConnectionInfo::Error(err) => {
                            tracing::error!(%addr, %err, "Peer connection error");
                            let () = self.ctxt.net.remove_active_peer(addr);
                        }
                        PeerConnectionInfo::NewTransaction(mut new_tx) => {
                            let mut rwtxn = self.ctxt.env.write_txn()?;
                            let () = self.ctxt.state.regenerate_proof(
                                &rwtxn,
                                &mut new_tx.transaction,
                            )?;
                            self.ctxt.mempool.put(&mut rwtxn, &new_tx)?;
                            rwtxn.commit()?;
                            // broadcast
                            let () = self
                                .ctxt
                                .net
                                .push_tx(HashSet::from_iter([addr]), new_tx);
                        }
                        PeerConnectionInfo::Response(resp, req) => {
                            let () = Self::handle_response(
                                &self.ctxt, addr, resp, req,
                            )
                            .await?;
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct Node {
    archive: Archive,
    drivechain: bip300301::Drivechain,
    env: heed::Env,
    mempool: MemPool,
    net: Net,
    net_task: Arc<JoinHandle<()>>,
    state: State,
}

impl Node {
    pub fn new(
        datadir: &Path,
        bind_addr: SocketAddr,
        main_addr: SocketAddr,
        user: &str,
        password: &str,
    ) -> Result<Self, Error> {
        let env_path = datadir.join("data.mdb");
        // let _ = std::fs::remove_dir_all(&env_path);
        std::fs::create_dir_all(&env_path)?;
        let env = heed::EnvOpenOptions::new()
            .map_size(10 * 1024 * 1024) // 10MB
            .max_dbs(
                crate::state::State::NUM_DBS
                    + crate::archive::Archive::NUM_DBS
                    + crate::mempool::MemPool::NUM_DBS,
            )
            .open(env_path)?;
        let state = State::new(&env)?;
        let archive = Archive::new(&env)?;
        let mempool = MemPool::new(&env)?;
        let drivechain = bip300301::Drivechain::new(
            THIS_SIDECHAIN,
            main_addr,
            user,
            password,
        )?;
        let (net, peer_info_rx) =
            Net::new(&env, archive.clone(), state.clone(), bind_addr)?;
        let net_task = spawn({
            let ctxt = NetTaskContext {
                env: env.clone(),
                archive: archive.clone(),
                drivechain: drivechain.clone(),
                mempool: mempool.clone(),
                net: net.clone(),
                state: state.clone(),
            };
            NetTask { ctxt, peer_info_rx }
                .run()
                .unwrap_or_else(|err| tracing::error!(%err))
        });
        Ok(Self {
            archive,
            drivechain,
            env,
            mempool,
            net,
            net_task: Arc::new(net_task),
            state,
        })
    }

    pub fn drivechain(&self) -> &bip300301::Drivechain {
        &self.drivechain
    }

    pub fn get_height(&self) -> Result<u32, Error> {
        let txn = self.env.read_txn()?;
        Ok(self.archive.get_height(&txn)?)
    }

    pub fn get_best_hash(&self) -> Result<BlockHash, Error> {
        let txn = self.env.read_txn()?;
        Ok(self.archive.get_best_hash(&txn)?)
    }

    pub async fn get_best_parentchain_hash(
        &self,
    ) -> Result<bitcoin::BlockHash, Error> {
        use bip300301::MainClient;
        let res = self
            .drivechain
            .client
            .getbestblockhash()
            .await
            .map_err(bip300301::Error::Jsonrpsee)?;
        Ok(res)
    }

    pub fn submit_transaction(
        &self,
        transaction: AuthorizedTransaction,
    ) -> Result<(), Error> {
        {
            let mut txn = self.env.write_txn()?;
            self.state.validate_transaction(&txn, &transaction)?;
            self.mempool.put(&mut txn, &transaction)?;
            txn.commit()?;
        }
        self.net.push_tx(Default::default(), transaction);
        Ok(())
    }

    pub fn get_spent_utxos(
        &self,
        outpoints: &[OutPoint],
    ) -> Result<Vec<(OutPoint, SpentOutput)>, Error> {
        let txn = self.env.read_txn()?;
        let mut spent = vec![];
        for outpoint in outpoints {
            if let Some(output) = self.state.stxos.get(&txn, outpoint)? {
                spent.push((*outpoint, output));
            }
        }
        Ok(spent)
    }

    pub fn get_utxos_by_addresses(
        &self,
        addresses: &HashSet<Address>,
    ) -> Result<HashMap<OutPoint, Output>, Error> {
        let txn = self.env.read_txn()?;
        let utxos = self.state.get_utxos_by_addresses(&txn, addresses)?;
        Ok(utxos)
    }

    pub fn get_accumulator(&self) -> Result<Pollard, Error> {
        let rotxn = self.env.read_txn()?;
        Ok(self.state.get_accumulator(&rotxn)?)
    }

    pub fn regenerate_proof(&self, tx: &mut Transaction) -> Result<(), Error> {
        let rotxn = self.env.read_txn()?;
        let () = self.state.regenerate_proof(&rotxn, tx)?;
        Ok(())
    }

    pub fn get_header(&self, height: u32) -> Result<Option<Header>, Error> {
        let txn = self.env.read_txn()?;
        Ok(self.archive.get_header(&txn, height)?)
    }

    pub fn get_body(&self, height: u32) -> Result<Option<Body>, Error> {
        let txn = self.env.read_txn()?;
        Ok(self.archive.get_body(&txn, height)?)
    }

    pub fn get_all_transactions(
        &self,
    ) -> Result<Vec<AuthorizedTransaction>, Error> {
        let txn = self.env.read_txn()?;
        let transactions = self.mempool.take_all(&txn)?;
        Ok(transactions)
    }

    /// Get total sidechain wealth in Bitcoin
    pub fn get_sidechain_wealth(&self) -> Result<bitcoin::Amount, Error> {
        let txn = self.env.read_txn()?;
        Ok(self.state.sidechain_wealth(&txn)?)
    }

    pub fn get_transactions(
        &self,
        number: usize,
    ) -> Result<(Vec<AuthorizedTransaction>, u64), Error> {
        let mut txn = self.env.write_txn()?;
        let transactions = self.mempool.take(&txn, number)?;
        let mut fee: u64 = 0;
        let mut returned_transactions = vec![];
        let mut spent_utxos = HashSet::new();
        for transaction in &transactions {
            let inputs: HashSet<_> =
                transaction.transaction.inputs.iter().copied().collect();
            if !spent_utxos.is_disjoint(&inputs) {
                println!("UTXO double spent");
                self.mempool
                    .delete(&mut txn, &transaction.transaction.txid())?;
                continue;
            }
            if self.state.validate_transaction(&txn, transaction).is_err() {
                self.mempool
                    .delete(&mut txn, &transaction.transaction.txid())?;
                continue;
            }
            let filled_transaction = self
                .state
                .fill_transaction(&txn, &transaction.transaction)?;
            let value_in: u64 = filled_transaction
                .spent_utxos
                .iter()
                .map(GetValue::get_value)
                .sum();
            let value_out: u64 = filled_transaction
                .transaction
                .outputs
                .iter()
                .map(GetValue::get_value)
                .sum();
            fee += value_in - value_out;
            returned_transactions.push(transaction.clone());
            spent_utxos.extend(transaction.transaction.inputs.clone());
        }
        txn.commit()?;
        Ok((returned_transactions, fee))
    }

    pub fn get_pending_withdrawal_bundle(
        &self,
    ) -> Result<Option<WithdrawalBundle>, Error> {
        let txn = self.env.read_txn()?;
        Ok(self.state.get_pending_withdrawal_bundle(&txn)?)
    }

    pub fn remove_from_mempool(&self, txid: Txid) -> Result<(), Error> {
        let mut rwtxn = self.env.write_txn()?;
        let () = self.mempool.delete(&mut rwtxn, &txid)?;
        rwtxn.commit()?;
        Ok(())
    }

    pub fn connect_peer(&self, addr: SocketAddr) -> Result<(), Error> {
        self.net
            .connect_peer(self.env.clone(), addr)
            .map_err(Error::from)
    }

    pub async fn submit_block(
        &self,
        header: &Header,
        body: &Body,
    ) -> Result<(), Error> {
        submit_block(
            &self.env,
            &self.archive,
            &self.drivechain,
            &self.mempool,
            &self.state,
            header,
            body,
        )
        .await
    }
}

impl Drop for Node {
    // If only one reference exists (ie. within self), abort the net task.
    fn drop(&mut self) {
        // use `Arc::get_mut` since `Arc::into_inner` requires ownership of the
        // Arc, and cloning would increase the reference count
        if let Some(task) = Arc::get_mut(&mut self.net_task) {
            task.abort()
        }
    }
}
