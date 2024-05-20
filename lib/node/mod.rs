use std::{
    collections::{HashMap, HashSet},
    fmt::Debug,
    net::SocketAddr,
    path::Path,
};

use bip300301::{bitcoin, DepositInfo};
use fallible_iterator::FallibleIterator;
use tokio_util::task::LocalPoolHandle;

use crate::{
    archive::{self, Archive},
    mempool::{self, MemPool},
    net::{self, Net},
    state::{self, State},
    types::{
        Accumulator, Address, AuthorizedTransaction, BlockHash, BmmResult,
        Body, GetValue, Header, OutPoint, Output, SpentOutput, Tip,
        Transaction, Txid, WithdrawalBundle,
    },
};

mod mainchain_task;
mod net_task;

use mainchain_task::MainchainTaskHandle;

use self::net_task::NetTaskHandle;

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
    #[error("net task error")]
    NetTask(#[from] net_task::Error),
    #[error("peer info stream closed")]
    PeerInfoRxClosed,
    #[error("Receive mainchain task response cancelled")]
    ReceiveMainchainTaskResponse,
    #[error("Send mainchain task request failed")]
    SendMainchainTaskRequest,
    #[error("state error")]
    State(#[from] state::Error),
    #[error("Utreexo error: {0}")]
    Utreexo(String),
}

/// Request any missing two way peg data up to the specified block hash.
/// All ancestor headers must exist in the archive.
// TODO: deposits only for now
#[allow(dead_code)]
async fn request_two_way_peg_data(
    env: &heed::Env,
    archive: &Archive,
    drivechain: &bip300301::Drivechain,
    block_hash: bitcoin::BlockHash,
) -> Result<(), Error> {
    // last block for which deposit info is known
    let last_known_deposit_info = {
        let rotxn = env.read_txn()?;
        #[allow(clippy::let_and_return)]
        let last_known_deposit_info = archive
            .main_ancestors(&rotxn, block_hash)
            .find(|block_hash| {
                let deposits = archive.try_get_deposits(&rotxn, *block_hash)?;
                Ok(deposits.is_some())
            })?;
        last_known_deposit_info
    };
    if last_known_deposit_info == Some(block_hash) {
        return Ok(());
    }
    let two_way_peg_data = drivechain
        .get_two_way_peg_data(block_hash, last_known_deposit_info)
        .await?;
    let mut rwtxn = env.write_txn()?;
    // Deposits by block, first-to-last within each block
    let deposits_by_block: HashMap<bitcoin::BlockHash, Vec<DepositInfo>> = {
        let mut deposits = HashMap::<_, Vec<_>>::new();
        two_way_peg_data.deposits.into_iter().for_each(|deposit| {
            deposits
                .entry(deposit.block_hash)
                .or_default()
                .push(deposit)
        });
        let () = archive
            .main_ancestors(&rwtxn, block_hash)
            .take_while(|block_hash| {
                Ok(last_known_deposit_info != Some(*block_hash))
            })
            .for_each(|block_hash| {
                let _ = deposits.entry(block_hash).or_default();
                Ok(())
            })?;
        deposits
    };
    deposits_by_block
        .into_iter()
        .try_for_each(|(block_hash, deposits)| {
            archive.put_deposits(&mut rwtxn, block_hash, deposits)
        })?;
    rwtxn.commit()?;
    Ok(())
}

#[derive(Clone)]
pub struct Node {
    archive: Archive,
    drivechain: bip300301::Drivechain,
    env: heed::Env,
    _local_pool: LocalPoolHandle,
    mainchain_task: MainchainTaskHandle,
    mempool: MemPool,
    net: Net,
    net_task: NetTaskHandle,
    state: State,
}

impl Node {
    pub fn new(
        datadir: &Path,
        bind_addr: SocketAddr,
        main_addr: SocketAddr,
        user: &str,
        password: &str,
        local_pool: LocalPoolHandle,
    ) -> Result<Self, Error> {
        let env_path = datadir.join("data.mdb");
        // let _ = std::fs::remove_dir_all(&env_path);
        std::fs::create_dir_all(&env_path)?;
        let env = unsafe {
            heed::EnvOpenOptions::new()
                .map_size(1024 * 1024 * 1024) // 1GB
                .max_dbs(
                    State::NUM_DBS
                        + Archive::NUM_DBS
                        + MemPool::NUM_DBS
                        + Net::NUM_DBS,
                )
                .open(env_path)?
        };
        let state = State::new(&env)?;
        let archive = Archive::new(&env)?;
        let mempool = MemPool::new(&env)?;
        let drivechain = bip300301::Drivechain::new(
            THIS_SIDECHAIN,
            main_addr,
            user,
            password,
        )?;
        let (mainchain_task, mainchain_task_response_rx) =
            MainchainTaskHandle::new(
                env.clone(),
                archive.clone(),
                drivechain.clone(),
            );
        let (net, peer_info_rx) =
            Net::new(&env, archive.clone(), state.clone(), bind_addr)?;

        let net_task = NetTaskHandle::new(
            local_pool.clone(),
            env.clone(),
            archive.clone(),
            drivechain.clone(),
            mainchain_task.clone(),
            mainchain_task_response_rx,
            mempool.clone(),
            net.clone(),
            peer_info_rx,
            state.clone(),
        );
        Ok(Self {
            archive,
            drivechain,
            env,
            _local_pool: local_pool,
            mainchain_task,
            mempool,
            net,
            net_task,
            state,
        })
    }

    pub fn drivechain(&self) -> &bip300301::Drivechain {
        &self.drivechain
    }

    pub fn get_height(&self) -> Result<u32, Error> {
        let txn = self.env.read_txn()?;
        Ok(self.state.get_height(&txn)?)
    }

    pub fn get_best_hash(&self) -> Result<BlockHash, Error> {
        let txn = self.env.read_txn()?;
        Ok(self.state.get_tip(&txn)?)
    }

    pub async fn get_best_parentchain_hash(
        &self,
    ) -> Result<bitcoin::BlockHash, Error> {
        use bip300301::MainClient;
        let res = self.drivechain.client.getbestblockhash().await.map_err(
            |source| bip300301::Error::Jsonrpsee {
                source,
                main_addr: self.drivechain.main_addr,
            },
        )?;
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

    pub fn get_all_utxos(&self) -> Result<HashMap<OutPoint, Output>, Error> {
        let rotxn = self.env.read_txn()?;
        self.state.get_utxos(&rotxn).map_err(Error::from)
    }

    pub fn get_spent_utxos(
        &self,
        outpoints: &[OutPoint],
    ) -> Result<Vec<(OutPoint, SpentOutput)>, Error> {
        let rotxn = self.env.read_txn()?;
        let mut spent = vec![];
        for outpoint in outpoints {
            if let Some(output) = self.state.stxos.get(&rotxn, outpoint)? {
                spent.push((*outpoint, output));
            }
        }
        Ok(spent)
    }

    pub fn get_utxos_by_addresses(
        &self,
        addresses: &HashSet<Address>,
    ) -> Result<HashMap<OutPoint, Output>, Error> {
        let rotxn = self.env.read_txn()?;
        let utxos = self.state.get_utxos_by_addresses(&rotxn, addresses)?;
        Ok(utxos)
    }

    pub fn get_tip_accumulator(&self) -> Result<Accumulator, Error> {
        let rotxn = self.env.read_txn()?;
        Ok(self.state.get_accumulator(&rotxn)?)
    }

    pub fn regenerate_proof(&self, tx: &mut Transaction) -> Result<(), Error> {
        let rotxn = self.env.read_txn()?;
        let () = self.state.regenerate_proof(&rotxn, tx)?;
        Ok(())
    }

    pub fn try_get_accumulator(
        &self,
        block_hash: BlockHash,
    ) -> Result<Option<Accumulator>, Error> {
        let rotxn = self.env.read_txn()?;
        Ok(self.archive.try_get_accumulator(&rotxn, block_hash)?)
    }

    pub fn get_accumulator(
        &self,
        block_hash: BlockHash,
    ) -> Result<Accumulator, Error> {
        let rotxn = self.env.read_txn()?;
        Ok(self.archive.get_accumulator(&rotxn, block_hash)?)
    }

    pub fn try_get_header(
        &self,
        block_hash: BlockHash,
    ) -> Result<Option<Header>, Error> {
        let txn = self.env.read_txn()?;
        Ok(self.archive.try_get_header(&txn, block_hash)?)
    }

    pub fn get_header(&self, block_hash: BlockHash) -> Result<Header, Error> {
        let txn = self.env.read_txn()?;
        Ok(self.archive.get_header(&txn, block_hash)?)
    }

    /// Get the block hash at the specified height in the current chain,
    /// if it exists
    pub fn try_get_block_hash(
        &self,
        height: u32,
    ) -> Result<Option<BlockHash>, Error> {
        let rotxn = self.env.read_txn()?;
        let tip = self.state.get_tip(&rotxn)?;
        let tip_height = self.state.get_height(&rotxn)?;
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
        let txn = self.env.read_txn()?;
        Ok(self.archive.try_get_body(&txn, block_hash)?)
    }

    pub fn get_body(&self, block_hash: BlockHash) -> Result<Body, Error> {
        let txn = self.env.read_txn()?;
        Ok(self.archive.get_body(&txn, block_hash)?)
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
        let bundle = self
            .state
            .get_pending_withdrawal_bundle(&txn)?
            .map(|(bundle, _)| bundle);
        Ok(bundle)
    }

    pub fn remove_from_mempool(&self, txid: Txid) -> Result<(), Error> {
        let mut rwtxn = self.env.write_txn()?;
        let () = self.mempool.delete(&mut rwtxn, txid)?;
        rwtxn.commit()?;
        Ok(())
    }

    pub fn connect_peer(&self, addr: SocketAddr) -> Result<(), Error> {
        self.net
            .connect_peer(self.env.clone(), addr)
            .map_err(Error::from)
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
        let block_hash = header.hash();
        // Store the header, if ancestors exist
        if header.prev_side_hash != BlockHash::default()
            && self.try_get_header(header.prev_side_hash)?.is_none()
        {
            tracing::error!(%block_hash,
                "Rejecting block {block_hash} due to missing ancestor headers",
            );
            return Ok(false);
        }
        // Request mainchain headers if they do not exist
        let _: mainchain_task::Response = self
            .mainchain_task
            .request_oneshot(mainchain_task::Request::AncestorHeaders(
                main_block_hash,
            ))
            .map_err(|_| Error::SendMainchainTaskRequest)?
            .await
            .map_err(|_| Error::ReceiveMainchainTaskResponse)?;
        // Verify BMM
        let mainchain_task::Response::VerifyBmm(_, res) = self
            .mainchain_task
            .request_oneshot(mainchain_task::Request::VerifyBmm(
                main_block_hash,
            ))
            .map_err(|_| Error::SendMainchainTaskRequest)?
            .await
            .map_err(|_| Error::ReceiveMainchainTaskResponse)?
        else {
            panic!("should be impossible")
        };
        if let Err(bip300301::BlockNotFoundError(missing_block)) = res {
            tracing::error!(%block_hash,
                "Rejecting block {block_hash} due to missing mainchain block {missing_block}",
            );
            return Ok(false);
        }
        // Write header
        tracing::trace!("Storing header: {block_hash}");
        {
            let mut rwtxn = self.env.write_txn()?;
            let () = self.archive.put_header(&mut rwtxn, header)?;
            rwtxn.commit()?;
        }
        tracing::trace!("Stored header: {block_hash}");
        // Check BMM
        {
            let rotxn = self.env.read_txn()?;
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
            rotxn.commit()?;
        }
        // Check that ancestor bodies exist, and store body
        {
            let rotxn = self.env.read_txn()?;
            let tip = self.state.get_tip(&rotxn)?;
            let common_ancestor =
                self.archive.last_common_ancestor(&rotxn, tip, block_hash)?;
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
            rotxn.commit()?;
            if missing_bodies == vec![block_hash] {
                let mut rwtxn = self.env.write_txn()?;
                let () = self.archive.put_body(&mut rwtxn, block_hash, body)?;
                rwtxn.commit()?;
            }
        }
        // Submit new tip
        let new_tip = Tip {
            block_hash,
            main_block_hash,
        };
        if !self.net_task.new_tip_ready_confirm(new_tip).await? {
            return Ok(false);
        };
        let rotxn = self.env.read_txn()?;
        let bundle = self.state.get_pending_withdrawal_bundle(&rotxn)?;
        if let Some((bundle, _)) = bundle {
            let () = self
                .drivechain
                .broadcast_withdrawal_bundle(bundle.transaction)
                .await?;
        }
        Ok(true)
    }
}
