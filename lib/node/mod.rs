use std::{
    collections::{HashMap, HashSet},
    fmt::Debug,
    net::SocketAddr,
    path::Path,
    sync::Arc,
};

use bitcoin::amount::CheckedSum;
use fallible_iterator::FallibleIterator as _;
use futures::{Stream, future::BoxFuture};
use sneed::{DbError, Env, EnvError, RoTxn, RwTxnError, env};
use tokio::sync::Mutex;
use tonic::transport::Channel;

use crate::{
    archive::{self, Archive},
    mempool::{self, MemPool},
    net::{self, Net, Peer},
    state::{self, State},
    types::{
        Accumulator, AmountOverflowError, AmountUnderflowError,
        AuthorizedTransaction, BlockHash, BmmResult, Body, GetValue, Header,
        OutPoint, Output, SpentOutput, Tip, Transaction, TransparentAddress,
        Txid, WithdrawalBundle,
        proto::{self, mainchain},
    },
    util::Watchable,
};

mod mainchain_task;
mod net_task;

use mainchain_task::MainchainTaskHandle;

use self::net_task::NetTaskHandle;

#[derive(Debug, thiserror::Error, transitive::Transitive)]
#[transitive(from(env::error::ReadTxn, EnvError))]
pub enum Error {
    #[error("address parse error")]
    AddrParse(#[from] std::net::AddrParseError),
    #[error(transparent)]
    AmountOverflow(#[from] AmountOverflowError),
    #[error(transparent)]
    AmountUnderflow(#[from] AmountUnderflowError),
    #[error("archive error")]
    Archive(#[from] archive::Error),
    #[error("CUSF mainchain proto error")]
    CusfMainchain(#[from] proto::Error),
    #[error(transparent)]
    Db(#[from] DbError),
    #[error("Database env error")]
    DbEnv(#[from] EnvError),
    #[error("Database write error")]
    DbWrite(#[from] RwTxnError),
    #[error("I/O error")]
    Io(#[from] std::io::Error),
    #[error("error requesting mainchain ancestors")]
    MainchainAncestors(#[source] mainchain_task::ResponseError),
    #[error("mempool error")]
    MemPool(#[from] mempool::Error),
    #[error("net error")]
    Net(#[from] net::Error),
    #[error("net task error")]
    NetTask(#[source] Box<net_task::Error>),
    #[error("No CUSF mainchain wallet client")]
    NoCusfMainchainWalletClient,
    #[error("peer info stream closed")]
    PeerInfoRxClosed,
    #[error("Receive mainchain task response cancelled")]
    ReceiveMainchainTaskResponse,
    #[error("Send mainchain task request failed")]
    SendMainchainTaskRequest,
    #[error("state error")]
    State(#[source] Box<state::Error>),
    #[error("Utreexo error: {0}")]
    Utreexo(String),
    #[error("Verify BMM error")]
    VerifyBmm(anyhow::Error),
}

impl From<net_task::Error> for Error {
    fn from(err: net_task::Error) -> Self {
        Self::NetTask(Box::new(err))
    }
}

impl From<state::Error> for Error {
    fn from(err: state::Error) -> Self {
        Self::State(Box::new(err))
    }
}

#[derive(Clone)]
pub struct Node<MainchainTransport = Channel> {
    archive: Archive,
    cusf_mainchain: Arc<Mutex<mainchain::ValidatorClient<MainchainTransport>>>,
    cusf_mainchain_wallet:
        Option<Arc<Mutex<mainchain::WalletClient<MainchainTransport>>>>,
    env: sneed::Env,
    mainchain_task: MainchainTaskHandle,
    mempool: MemPool,
    net: Net,
    net_task: NetTaskHandle,
    state: State,
}

impl<MainchainTransport> Node<MainchainTransport>
where
    MainchainTransport: proto::Transport,
{
    pub fn new(
        datadir: &Path,
        bind_addr: SocketAddr,
        cusf_mainchain: mainchain::ValidatorClient<MainchainTransport>,
        cusf_mainchain_wallet: Option<
            mainchain::WalletClient<MainchainTransport>,
        >,
        runtime: &tokio::runtime::Runtime,
    ) -> Result<Self, Error>
    where
        mainchain::ValidatorClient<MainchainTransport>: Clone,
        MainchainTransport: Send + 'static,
        <MainchainTransport as tonic::client::GrpcService<
            tonic::body::BoxBody,
        >>::Future: Send,
    {
        let env_path = datadir.join("data.mdb");
        // let _ = std::fs::remove_dir_all(&env_path);
        std::fs::create_dir_all(&env_path)?;
        let env = {
            let mut env_open_opts = heed::EnvOpenOptions::new();
            env_open_opts
                .map_size(1024 * 1024 * 1024) // 1GB
                .max_dbs(
                    State::NUM_DBS
                        + Archive::NUM_DBS
                        + MemPool::NUM_DBS
                        + Net::NUM_DBS,
                );
            unsafe { Env::open(&env_open_opts, &env_path) }
                .map_err(EnvError::from)?
        };
        let state = State::new(&env)?;
        let archive = Archive::new(&env)?;
        let mempool = MemPool::new(&env)?;
        let (mainchain_task, mainchain_task_response_rx) =
            MainchainTaskHandle::new(
                env.clone(),
                archive.clone(),
                cusf_mainchain.clone(),
            );
        let (net, peer_info_rx) =
            Net::new(&env, archive.clone(), state.clone(), bind_addr)?;

        let net_task = NetTaskHandle::new(
            runtime,
            env.clone(),
            archive.clone(),
            mainchain_task.clone(),
            mainchain_task_response_rx,
            mempool.clone(),
            net.clone(),
            peer_info_rx,
            state.clone(),
        );
        let cusf_mainchain_wallet =
            cusf_mainchain_wallet.map(|wallet| Arc::new(Mutex::new(wallet)));
        Ok(Self {
            archive,
            cusf_mainchain: Arc::new(Mutex::new(cusf_mainchain)),
            cusf_mainchain_wallet,
            env,
            mainchain_task,
            mempool,
            net,
            net_task,
            state,
        })
    }

    pub fn env(&self) -> &Env {
        &self.env
    }

    pub fn archive(&self) -> &Archive {
        &self.archive
    }

    pub fn state(&self) -> &State {
        &self.state
    }

    /// Borrow the CUSF mainchain client, and execute the provided future.
    /// The CUSF mainchain client will be locked while the future is running.
    pub async fn with_cusf_mainchain<F, Output>(&self, f: F) -> Output
    where
        F: for<'cusf_mainchain> FnOnce(
            &'cusf_mainchain mut mainchain::ValidatorClient<MainchainTransport>,
        )
            -> BoxFuture<'cusf_mainchain, Output>,
    {
        let mut cusf_mainchain_lock = self.cusf_mainchain.lock().await;
        let res = f(&mut cusf_mainchain_lock).await;
        drop(cusf_mainchain_lock);
        res
    }

    pub fn try_get_height(&self) -> Result<Option<u32>, Error> {
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        Ok(self.state.try_get_height(&rotxn)?)
    }

    pub fn try_get_best_hash(&self) -> Result<Option<BlockHash>, Error> {
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        Ok(self.state.try_get_tip(&rotxn)?)
    }

    pub fn submit_transaction(
        &self,
        transaction: AuthorizedTransaction,
    ) -> Result<(), Error> {
        {
            let mut rotxn = self.env.write_txn().map_err(EnvError::from)?;
            self.state.validate_transaction(&rotxn, &transaction)?;
            self.mempool.put(&mut rotxn, &transaction)?;
            rotxn.commit().map_err(RwTxnError::from)?;
        }
        self.net.push_tx(Default::default(), transaction);
        Ok(())
    }

    pub fn get_all_utxos(&self) -> Result<HashMap<OutPoint, Output>, Error> {
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        self.state
            .get_utxos(&rotxn)
            .map_err(|err| DbError::from(err).into())
    }

    pub fn get_latest_failed_withdrawal_bundle_height(
        &self,
    ) -> Result<Option<u32>, Error> {
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        let res = self
            .state
            .get_latest_failed_withdrawal_bundle(&rotxn)
            .map_err(DbError::from)?
            .map(|(height, _)| height);
        Ok(res)
    }

    pub fn get_spent_utxos(
        &self,
        rotxn: &RoTxn,
        outpoints: &[OutPoint],
    ) -> Result<Vec<(OutPoint, SpentOutput)>, Error> {
        let mut spent = vec![];
        for outpoint in outpoints {
            if let Some(output) = self
                .state
                .stxos
                .try_get(rotxn, outpoint)
                .map_err(DbError::from)?
            {
                spent.push((*outpoint, output));
            }
        }
        Ok(spent)
    }

    pub fn get_utxos_by_addresses(
        &self,
        addresses: &HashSet<TransparentAddress>,
    ) -> Result<HashMap<OutPoint, Output>, Error> {
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        let utxos = self
            .state
            .get_utxos_by_addresses(&rotxn, addresses)
            .map_err(DbError::from)?;
        Ok(utxos)
    }

    pub fn try_get_tip(
        &self,
        rotxn: &RoTxn,
    ) -> Result<Option<BlockHash>, Error> {
        let tip = self.state.try_get_tip(rotxn)?;
        Ok(tip)
    }

    pub fn get_tip_accumulator(&self) -> Result<Accumulator, Error> {
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        Ok(self.state.get_accumulator(&rotxn)?)
    }

    pub fn regenerate_proof(&self, tx: &mut Transaction) -> Result<(), Error> {
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        let () = self.state.regenerate_proof(&rotxn, tx)?;
        Ok(())
    }

    pub fn try_get_accumulator(
        &self,
        block_hash: &Option<BlockHash>,
    ) -> Result<Option<Accumulator>, Error> {
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        Ok(self.archive.try_get_accumulator(&rotxn, block_hash)?)
    }

    pub fn get_accumulator(
        &self,
        block_hash: &Option<BlockHash>,
    ) -> Result<Accumulator, Error> {
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        Ok(self.archive.get_accumulator(&rotxn, block_hash)?)
    }

    pub fn try_get_header(
        &self,
        block_hash: BlockHash,
    ) -> Result<Option<Header>, Error> {
        let txn = self.env.read_txn().map_err(EnvError::from)?;
        Ok(self.archive.try_get_header(&txn, block_hash)?)
    }

    pub fn get_header(&self, block_hash: BlockHash) -> Result<Header, Error> {
        let txn = self.env.read_txn().map_err(EnvError::from)?;
        Ok(self.archive.get_header(&txn, block_hash)?)
    }

    /// Get the block hash at the specified height in the current chain,
    /// if it exists
    pub fn try_get_block_hash(
        &self,
        height: u32,
    ) -> Result<Option<BlockHash>, Error> {
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        let Some(tip) = self.state.try_get_tip(&rotxn)? else {
            return Ok(None);
        };
        let Some(tip_height) = self.state.try_get_height(&rotxn)? else {
            return Ok(None);
        };
        if tip_height >= height {
            self.archive
                .ancestors(&rotxn, tip)
                .nth((tip_height - height) as usize)
                .map_err(Error::from)
        } else {
            Ok(None)
        }
    }

    pub fn try_get_body(
        &self,
        block_hash: BlockHash,
    ) -> Result<Option<Body>, Error> {
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        Ok(self.archive.try_get_body(&rotxn, block_hash)?)
    }

    pub fn get_body(&self, block_hash: BlockHash) -> Result<Body, Error> {
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        Ok(self.archive.get_body(&rotxn, block_hash)?)
    }

    pub fn get_best_main_verification(
        &self,
        hash: BlockHash,
    ) -> Result<bitcoin::BlockHash, Error> {
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        let hash = self.archive.get_best_main_verification(&rotxn, hash)?;
        Ok(hash)
    }

    pub fn get_bmm_inclusions(
        &self,
        block_hash: BlockHash,
    ) -> Result<Vec<bitcoin::BlockHash>, Error> {
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        let bmm_inclusions = self
            .archive
            .get_bmm_results(&rotxn, block_hash)?
            .into_iter()
            .filter_map(|(block_hash, bmm_res)| match bmm_res {
                BmmResult::Verified => Some(block_hash),
                BmmResult::Failed => None,
            })
            .collect();
        Ok(bmm_inclusions)
    }

    pub fn get_all_transactions(
        &self,
    ) -> Result<Vec<AuthorizedTransaction>, Error> {
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        let transactions = self.mempool.take_all(&rotxn)?;
        Ok(transactions)
    }

    /// Get total sidechain wealth in Bitcoin
    pub fn get_sidechain_wealth(&self) -> Result<bitcoin::Amount, Error> {
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        Ok(self.state.sidechain_wealth(&rotxn)?)
    }

    pub fn get_transactions(
        &self,
        number: usize,
    ) -> Result<(Vec<AuthorizedTransaction>, bitcoin::Amount), Error> {
        let mut txn = self.env.write_txn().map_err(EnvError::from)?;
        let transactions = self.mempool.take(&txn, number)?;
        let mut fee = bitcoin::Amount::ZERO;
        let mut returned_transactions = vec![];
        let mut spent_utxos = HashSet::new();
        for transaction in &transactions {
            let inputs: HashSet<_> =
                transaction.transaction.inputs.iter().copied().collect();
            if !spent_utxos.is_disjoint(&inputs) {
                // UTXO double spent
                self.mempool
                    .delete(&mut txn, transaction.transaction.txid())?;
                continue;
            }
            if self.state.validate_transaction(&txn, transaction).is_err() {
                self.mempool
                    .delete(&mut txn, transaction.transaction.txid())?;
                continue;
            }
            let filled_transaction = self
                .state
                .fill_transaction(&txn, &transaction.transaction)?;
            let mut value_in: bitcoin::Amount = filled_transaction
                .spent_utxos
                .iter()
                .map(GetValue::get_value)
                .checked_sum()
                .ok_or(AmountOverflowError)?;
            let mut value_out: bitcoin::Amount = filled_transaction
                .transaction
                .outputs
                .iter()
                .map(GetValue::get_value)
                .checked_sum()
                .ok_or(AmountOverflowError)?;
            if let Some(orchard_bundle) =
                filled_transaction.transaction.orchard_bundle.as_ref()
            {
                let value_balance = orchard_bundle.value_balance();
                if value_balance.is_positive() {
                    value_in = value_in
                        .checked_add(value_balance.unsigned_abs())
                        .ok_or(AmountOverflowError)?;
                } else {
                    value_out = value_out
                        .checked_add(value_balance.unsigned_abs())
                        .ok_or(AmountOverflowError)?;
                }
            }
            fee = fee
                .checked_add(
                    value_in
                        .checked_sub(value_out)
                        .ok_or(AmountOverflowError)?,
                )
                .ok_or(AmountUnderflowError)?;
            returned_transactions.push(transaction.clone());
            spent_utxos.extend(transaction.transaction.inputs.clone());
        }
        txn.commit().map_err(RwTxnError::from)?;
        Ok((returned_transactions, fee))
    }

    pub fn get_pending_withdrawal_bundle(
        &self,
    ) -> Result<Option<WithdrawalBundle>, Error> {
        let txn = self.env.read_txn().map_err(EnvError::from)?;
        let bundle = self
            .state
            .get_pending_withdrawal_bundle(&txn)?
            .map(|(bundle, _)| bundle);
        Ok(bundle)
    }

    pub fn remove_from_mempool(&self, txid: Txid) -> Result<(), Error> {
        let mut rwtxn = self.env.write_txn().map_err(EnvError::from)?;
        let () = self.mempool.delete(&mut rwtxn, txid)?;
        rwtxn.commit().map_err(RwTxnError::from)?;
        Ok(())
    }

    pub fn connect_peer(&self, addr: SocketAddr) -> Result<(), Error> {
        self.net
            .connect_peer(self.env.clone(), addr)
            .map_err(Error::from)
    }

    pub fn get_active_peers(&self) -> Vec<Peer> {
        self.net.get_active_peers()
    }

    /// Attempt to submit a block.
    /// Returns `Ok(true)` if the block was accepted successfully as the new tip.
    /// Returns `Ok(false)` if the block could not be submitted for some reason,
    /// or was rejected as the new tip.
    pub async fn submit_block(
        &self,
        main_block_hash: bitcoin::BlockHash,
        header: &Header,
        body: &Body,
    ) -> Result<bool, Error> {
        let Some(cusf_mainchain_wallet) = self.cusf_mainchain_wallet.as_ref()
        else {
            return Err(Error::NoCusfMainchainWalletClient);
        };
        let block_hash = header.hash();
        // Store the header, if ancestors exist
        if let Some(parent) = header.prev_side_hash
            && self.try_get_header(parent)?.is_none()
        {
            tracing::error!(%block_hash,
                "Rejecting block {block_hash} due to missing ancestor headers",
            );
            return Ok(false);
        }
        // Request mainchain header/infos if they do not exist
        let mainchain_task::Response::AncestorInfos(_, res): mainchain_task::Response = self
            .mainchain_task
            .request_oneshot(mainchain_task::Request::AncestorInfos(
                main_block_hash,
            ))
            .map_err(|_| Error::SendMainchainTaskRequest)?
            .await
            .map_err(|_| Error::ReceiveMainchainTaskResponse)?;
        if !res.map_err(Error::MainchainAncestors)? {
            return Ok(false);
        };
        // Write header
        tracing::trace!("Storing header: {block_hash}");
        {
            let mut rwtxn = self.env.write_txn().map_err(EnvError::from)?;
            let () = self.archive.put_header(&mut rwtxn, header)?;
            rwtxn.commit().map_err(RwTxnError::from)?;
        }
        tracing::trace!("Stored header: {block_hash}");
        // Check BMM
        {
            let rotxn = self.env.read_txn().map_err(EnvError::from)?;
            if self.archive.get_bmm_result(
                &rotxn,
                block_hash,
                main_block_hash,
            )? == BmmResult::Failed
            {
                tracing::error!(%block_hash,
                    "Rejecting block {block_hash} due to failing BMM verification",
                );
                return Ok(false);
            }
        }
        // Check that ancestor bodies exist, and store body
        {
            let rotxn = self.env.read_txn().map_err(EnvError::from)?;
            let tip = self.state.try_get_tip(&rotxn)?;
            let common_ancestor = if let Some(tip) = tip {
                self.archive.last_common_ancestor(&rotxn, tip, block_hash)?
            } else {
                None
            };
            let missing_bodies = self.archive.get_missing_bodies(
                &rotxn,
                block_hash,
                common_ancestor,
            )?;
            if !(missing_bodies.is_empty()
                || missing_bodies == vec![block_hash])
            {
                tracing::error!(%block_hash,
                    "Rejecting block {block_hash} due to missing ancestor bodies",
                );
                return Ok(false);
            }
            drop(rotxn);
            if missing_bodies == vec![block_hash] {
                let mut rwtxn = self.env.write_txn().map_err(EnvError::from)?;
                let () = self.archive.put_body(&mut rwtxn, block_hash, body)?;
                rwtxn.commit().map_err(RwTxnError::from)?;
            }
        }
        // Submit new tip
        let new_tip = Tip {
            block_hash,
            main_block_hash,
        };
        if !self.net_task.new_tip_ready_confirm(new_tip).await? {
            tracing::warn!(%block_hash, "Not ready to reorg");
            return Ok(false);
        };
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        let bundle = self.state.get_pending_withdrawal_bundle(&rotxn)?;
        if let Some((bundle, _)) = bundle {
            let m6id = bundle.compute_m6id();
            let mut cusf_mainchain_wallet_lock =
                cusf_mainchain_wallet.lock().await;
            let () = cusf_mainchain_wallet_lock
                .broadcast_withdrawal_bundle(bundle.tx())
                .await?;
            drop(cusf_mainchain_wallet_lock);
            tracing::trace!(%m6id, "Broadcast withdrawal bundle");
        }
        Ok(true)
    }

    /// Get a notification whenever the tip changes
    pub fn watch_state(&self) -> impl Stream<Item = ()> {
        self.state.watch()
    }
}
