use std::{
    collections::{HashMap, HashSet},
    fmt::Debug,
    net::SocketAddr,
    path::Path,
    sync::Arc,
};

use bip300301::bitcoin;
use fallible_iterator::{FallibleIterator, IteratorExt};
use futures::future::JoinAll;
use heed::{RoTxn, RwTxn};
use rustreexo::accumulator::pollard::Pollard;
use tokio::sync::RwLock;
use tokio_util::task::LocalPoolHandle;

use crate::{
    authorization::Authorization,
    net::{PeerState, Request, Response},
    types::*,
};

pub const THIS_SIDECHAIN: u8 = 9;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("address parse error")]
    AddrParse(#[from] std::net::AddrParseError),
    #[error("archive error")]
    Archive(#[from] crate::archive::Error),
    #[error("bincode error")]
    Bincode(#[from] bincode::Error),
    #[error("drivechain error")]
    Drivechain(#[from] bip300301::Error),
    #[error("heed error")]
    Heed(#[from] heed::Error),
    #[error("quinn error")]
    Io(#[from] std::io::Error),
    #[error("mempool error")]
    MemPool(#[from] crate::mempool::Error),
    #[error("net error")]
    Net(#[from] crate::net::Error),
    #[error("state error")]
    State(#[from] crate::state::Error),
    #[error("Utreexo error: {0}")]
    Utreexo(String),
}

#[derive(Clone)]
pub struct Node {
    net: crate::net::Net,
    state: crate::state::State,
    archive: crate::archive::Archive,
    mempool: crate::mempool::MemPool,
    drivechain: bip300301::Drivechain,
    env: heed::Env,
    local_pool: LocalPoolHandle,
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
        let env = heed::EnvOpenOptions::new()
            .map_size(10 * 1024 * 1024) // 10MB
            .max_dbs(
                crate::state::State::NUM_DBS
                    + crate::archive::Archive::NUM_DBS
                    + crate::mempool::MemPool::NUM_DBS,
            )
            .open(env_path)?;
        let state = crate::state::State::new(&env)?;
        let archive = crate::archive::Archive::new(&env)?;
        let mempool = crate::mempool::MemPool::new(&env)?;
        let drivechain = bip300301::Drivechain::new(
            THIS_SIDECHAIN,
            main_addr,
            user,
            password,
        )?;
        let net = crate::net::Net::new(bind_addr)?;
        Ok(Self {
            net,
            state,
            archive,
            mempool,
            drivechain,
            env,
            local_pool,
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
        let res = self
            .drivechain
            .client
            .getbestblockhash()
            .await
            .map_err(bip300301::Error::Jsonrpsee)?;
        Ok(res)
    }

    pub fn validate_transaction(
        &self,
        txn: &RoTxn,
        transaction: &AuthorizedTransaction,
    ) -> Result<u64, Error> {
        let filled_transaction =
            self.state.fill_transaction(txn, &transaction.transaction)?;
        for (authorization, spent_utxo) in transaction
            .authorizations
            .iter()
            .zip(filled_transaction.spent_utxos.iter())
        {
            if authorization.get_address() != spent_utxo.address {
                return Err(crate::state::Error::WrongPubKeyForAddress.into());
            }
        }
        if Authorization::verify_transaction(transaction).is_err() {
            return Err(crate::state::Error::AuthorizationError.into());
        }
        let fee = self
            .state
            .validate_filled_transaction(&filled_transaction)?;
        Ok(fee)
    }

    pub async fn submit_transaction(
        &self,
        transaction: &AuthorizedTransaction,
    ) -> Result<(), Error> {
        {
            let mut txn = self.env.write_txn()?;
            self.validate_transaction(&txn, transaction)?;
            self.mempool.put(&mut txn, transaction)?;
            txn.commit()?;
        }
        for peer in self.net.peers.read().await.values() {
            peer.request(&Request::PushTransaction {
                transaction: transaction.clone(),
            })
            .await?;
        }
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

    /// Regenerate utreexo proof for a tx
    pub fn regenerate_proof(&self, tx: &mut Transaction) -> Result<(), Error> {
        let accumulator = self.get_accumulator()?;
        let targets: Vec<_> = tx
            .inputs
            .iter()
            .map(|(_, utxo_hash)| utxo_hash.into())
            .collect();
        let (proof, _) = accumulator.prove(&targets).map_err(Error::Utreexo)?;
        tx.proof = proof;
        Ok(())
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
            if self.validate_transaction(&txn, transaction).is_err() {
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

    async fn connect_tip_(
        &self,
        rwtxn: &mut RwTxn<'_, '_>,
        header: &Header,
        body: &Body,
    ) -> Result<(), Error> {
        let last_deposit_block_hash =
            self.state.get_last_deposit_block_hash(rwtxn)?;
        let two_way_peg_data = self
            .drivechain
            .get_two_way_peg_data(
                header.prev_main_hash,
                last_deposit_block_hash,
            )
            .await?;
        let block_hash = header.hash();
        self.state.validate_block(rwtxn, header, body)?;
        if tracing::enabled!(tracing::Level::DEBUG) {
            let merkle_root = body.compute_merkle_root();
            let height = self.state.get_height(rwtxn)?;
            self.state.connect_block(rwtxn, header, body)?;
            tracing::debug!(%height, %merkle_root, %block_hash,
                                "connected body")
        } else {
            self.state.connect_block(rwtxn, header, body)?;
        }
        self.state
            .connect_two_way_peg_data(rwtxn, &two_way_peg_data)?;
        self.archive.put_header(rwtxn, header)?;
        self.archive.put_body(rwtxn, block_hash, body)?;
        for transaction in &body.transactions {
            self.mempool.delete(rwtxn, transaction.txid())?;
        }
        let accumulator = self.state.get_accumulator(rwtxn)?;
        self.mempool.regenerate_proofs(rwtxn, &accumulator)?;
        Ok(())
    }

    pub async fn connect_tip(
        &self,
        header: &Header,
        body: &Body,
    ) -> Result<(), Error> {
        let mut rwtxn = self.env.write_txn()?;
        let () = self.connect_tip_(&mut rwtxn, header, body).await?;
        rwtxn.commit()?;
        Ok(())
    }

    pub async fn submit_block(
        &self,
        header: &Header,
        body: &Body,
    ) -> Result<(), Error> {
        let mut rwtxn = self.env.write_txn()?;
        let () = self.connect_tip_(&mut rwtxn, header, body).await?;
        let bundle = self.state.get_pending_withdrawal_bundle(&rwtxn)?;
        rwtxn.commit()?;
        if let Some((bundle, _)) = bundle {
            let () = self
                .drivechain
                .broadcast_withdrawal_bundle(bundle.transaction)
                .await?;
        }
        Ok(())
    }

    async fn disconnect_tip_(
        &self,
        rwtxn: &mut RwTxn<'_, '_>,
    ) -> Result<(), Error> {
        let tip_block_hash = self.state.get_tip(rwtxn)?;
        let tip_header = self.archive.get_header(rwtxn, tip_block_hash)?;
        let tip_body = self.archive.get_body(rwtxn, tip_block_hash)?;
        let height = self.state.get_height(rwtxn)?;
        let two_way_peg_data = {
            let start_block_hash = self
                .state
                .deposit_blocks
                .rev_iter(rwtxn)?
                .transpose_into_fallible()
                .find_map(|(_, (block_hash, applied_height))| {
                    if applied_height < height {
                        Ok(Some(block_hash))
                    } else {
                        Ok(None)
                    }
                })?;
            self.drivechain
                .get_two_way_peg_data(
                    tip_header.prev_main_hash,
                    start_block_hash,
                )
                .await?
        };
        let () = self
            .state
            .disconnect_two_way_peg_data(rwtxn, &two_way_peg_data)?;
        let () = self.state.disconnect_tip(rwtxn, &tip_header, &tip_body)?;
        for transaction in tip_body.authorized_transactions().iter().rev() {
            self.mempool.put(rwtxn, transaction)?;
        }
        let accumulator = self.state.get_accumulator(rwtxn)?;
        self.mempool.regenerate_proofs(rwtxn, &accumulator)?;
        Ok(())
    }

    pub async fn disconnect_tip(&self) -> Result<(), Error> {
        let mut rwtxn = self.env.write_txn()?;
        let () = self.disconnect_tip_(&mut rwtxn).await?;
        rwtxn.commit()?;
        Ok(())
    }

    /// Re-org to the specified tip. The new tip block and all ancestor blocks
    /// must exist in the node's archive.
    pub async fn reorg_to_tip(&self, new_tip: BlockHash) -> Result<(), Error> {
        let mut rwtxn = self.env.write_txn()?;
        let tip = self.state.get_tip(&rwtxn)?;
        let tip_height = self.state.get_height(&rwtxn)?;
        let common_ancestor =
            self.archive.last_common_ancestor(&rwtxn, tip, new_tip)?;
        // Check that all necessary bodies exist before disconnecting tip
        let blocks_to_apply: Vec<(Header, Body)> = self
            .archive
            .ancestors(&rwtxn, tip)
            .take_while(|block_hash| Ok(*block_hash != tip))
            .map(|block_hash| {
                let header = self.archive.get_header(&rwtxn, block_hash)?;
                let body = self.archive.get_body(&rwtxn, block_hash)?;
                Ok((header, body))
            })
            .collect()?;
        // Disconnect tip until common ancestor is reached
        let common_ancestor_height =
            self.archive.get_height(&rwtxn, common_ancestor)?;
        for _ in 0..tip_height - common_ancestor_height {
            let () = self.disconnect_tip_(&mut rwtxn).await?;
        }
        let tip = self.state.get_tip(&rwtxn)?;
        assert_eq!(tip, common_ancestor);
        // Apply blocks until new tip is reached
        for (header, body) in blocks_to_apply.into_iter().rev() {
            let () = self.connect_tip_(&mut rwtxn, &header, &body).await?;
        }
        let tip = self.state.get_tip(&rwtxn)?;
        assert_eq!(tip, new_tip);
        rwtxn.commit()?;
        Ok(())
    }

    pub async fn heart_beat_listen(
        &self,
        peer: &crate::net::Peer,
    ) -> Result<(), Error> {
        let message = match peer.connection.read_datagram().await {
            Ok(message) => message,
            Err(err) => {
                self.net
                    .peers
                    .write()
                    .await
                    .remove(&peer.connection.stable_id());
                let addr = peer.connection.stable_id();
                println!("connection {addr} closed");
                return Err(crate::net::Error::from(err).into());
            }
        };
        let state: PeerState = bincode::deserialize(&message)?;
        *peer.state.write().await = Some(state);
        Ok(())
    }

    pub async fn peer_listen(
        &self,
        peer: &crate::net::Peer,
    ) -> Result<(), Error> {
        let (mut send, mut recv) = peer
            .connection
            .accept_bi()
            .await
            .map_err(crate::net::Error::from)?;
        let data = recv
            .read_to_end(crate::net::READ_LIMIT)
            .await
            .map_err(crate::net::Error::from)?;
        let message: Request = bincode::deserialize(&data)?;
        match message {
            Request::GetBlock { block_hash } => {
                let (header, body) = {
                    let txn = self.env.read_txn()?;
                    (
                        self.archive.try_get_header(&txn, block_hash)?,
                        self.archive.try_get_body(&txn, block_hash)?,
                    )
                };
                let response = match (header, body) {
                    (Some(header), Some(body)) => {
                        Response::Block { header, body }
                    }
                    (_, _) => Response::NoBlock,
                };
                let response = bincode::serialize(&response)?;
                send.write_all(&response)
                    .await
                    .map_err(crate::net::Error::from)?;
                send.finish().await.map_err(crate::net::Error::from)?;
            }
            Request::GetHeaders { end } => {
                let response = {
                    let rotxn = self.env.read_txn()?;
                    if self.archive.try_get_header(&rotxn, end)?.is_some() {
                        let mut headers: Vec<Header> = self
                            .archive
                            .ancestors(&rotxn, end)
                            .take_while(|block_hash| {
                                Ok(*block_hash != BlockHash::default())
                            })
                            .map(|block_hash| {
                                self.archive.get_header(&rotxn, block_hash)
                            })
                            .collect()?;
                        headers.reverse();
                        Response::Headers(headers)
                    } else {
                        Response::NoHeader
                    }
                };
                let response_bytes = bincode::serialize(&response)?;
                send.write_all(&response_bytes)
                    .await
                    .map_err(crate::net::Error::from)?;
                send.finish().await.map_err(crate::net::Error::from)?;
            }
            Request::PushTransaction { transaction } => {
                let valid = {
                    let txn = self.env.read_txn()?;
                    self.validate_transaction(&txn, &transaction)
                };
                match valid {
                    Err(err) => {
                        let response = Response::TransactionRejected;
                        let response = bincode::serialize(&response)?;
                        send.write_all(&response)
                            .await
                            .map_err(crate::net::Error::from)?;
                        return Err(err);
                    }
                    Ok(_) => {
                        {
                            let mut txn = self.env.write_txn()?;
                            println!(
                                "adding transaction to mempool: {:?}",
                                &transaction
                            );
                            self.mempool.put(&mut txn, &transaction)?;
                            txn.commit()?;
                        }
                        for peer0 in self.net.peers.read().await.values() {
                            if peer0.connection.stable_id()
                                == peer.connection.stable_id()
                            {
                                continue;
                            }
                            peer0
                                .request(&Request::PushTransaction {
                                    transaction: transaction.clone(),
                                })
                                .await?;
                        }
                        let response = Response::TransactionAccepted;
                        let response = bincode::serialize(&response)?;
                        send.write_all(&response)
                            .await
                            .map_err(crate::net::Error::from)?;
                        return Ok(());
                    }
                }
            }
        };
        Ok(())
    }

    pub async fn connect_peer(&self, addr: SocketAddr) -> Result<(), Error> {
        let peer = self.net.connect_peer(addr).await?;
        tokio::spawn({
            let node = self.clone();
            let peer = peer.clone();
            async move {
                loop {
                    match node.peer_listen(&peer).await {
                        Ok(_) => {}
                        Err(err) => {
                            println!("{:?}", err);
                            break;
                        }
                    }
                }
            }
        });
        tokio::spawn({
            let node = self.clone();
            let peer = peer.clone();
            async move {
                loop {
                    match node.heart_beat_listen(&peer).await {
                        Ok(_) => {}
                        Err(err) => {
                            println!("{:?}", err);
                            break;
                        }
                    }
                }
            }
        });
        Ok(())
    }

    pub fn run(&mut self) -> Result<(), Error> {
        // Listening to connections.
        let node = self.clone();
        tokio::spawn(async move {
            loop {
                let incoming_conn = node.net.server.accept().await.unwrap();
                let connection = incoming_conn.await.unwrap();
                for peer in node.net.peers.read().await.values() {
                    if peer.connection.remote_address()
                        == connection.remote_address()
                    {
                        println!(
                            "already connected to {} refusing duplicate connection",
                            connection.remote_address()
                        );
                        connection.close(
                            quinn::VarInt::from_u32(1),
                            b"already connected",
                        );
                    }
                }
                if connection.close_reason().is_some() {
                    continue;
                }
                println!(
                    "[server] connection accepted: addr={} id={}",
                    connection.remote_address(),
                    connection.stable_id(),
                );
                let peer = crate::net::Peer {
                    state: Arc::new(RwLock::new(None)),
                    connection,
                };
                tokio::spawn({
                    let node = node.clone();
                    let peer = peer.clone();
                    async move {
                        loop {
                            match node.peer_listen(&peer).await {
                                Ok(_) => {}
                                Err(err) => {
                                    println!("{:?}", err);
                                    break;
                                }
                            }
                        }
                    }
                });
                tokio::spawn({
                    let node = node.clone();
                    let peer = peer.clone();
                    async move {
                        loop {
                            match node.heart_beat_listen(&peer).await {
                                Ok(_) => {}
                                Err(err) => {
                                    println!("{:?}", err);
                                    break;
                                }
                            }
                        }
                    }
                });
                node.net
                    .peers
                    .write()
                    .await
                    .insert(peer.connection.stable_id(), peer);
            }
        });

        // Heart beat.
        let node = self.clone();
        tokio::spawn(async move {
            loop {
                for peer in node.net.peers.read().await.values() {
                    let (block_height, tip) = {
                        let txn = node.env.read_txn().unwrap();
                        let block_height = node.state.get_height(&txn).unwrap();
                        let tip = node.state.get_tip(&txn).unwrap();
                        (block_height, tip)
                    };
                    let state = PeerState { block_height, tip };
                    peer.heart_beat(&state).unwrap();
                }
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            }
        });

        // Request missing headers.
        let node = self.clone();
        tokio::spawn(async move {
            loop {
                for peer in node.net.peers.read().await.values() {
                    if let Some(peer_state) = &peer.state.read().await.as_ref()
                    {
                        let (block_height, tip) = {
                            let rotxn = node.env.read_txn().unwrap();
                            let block_height =
                                node.state.get_height(&rotxn).unwrap();
                            let tip = node.state.get_tip(&rotxn).unwrap();
                            (block_height, tip)
                        };
                        if peer_state.block_height > block_height {
                            // If we have the header, request any missing bodies and apply them
                            let header_exists = {
                                let rotxn = node.env.read_txn().unwrap();
                                node.archive
                                    .try_get_header(&rotxn, peer_state.tip)
                                    .unwrap()
                                    .is_some()
                            };
                            if header_exists {
                                let missing_bodies: Vec<_> = {
                                    let rotxn = node.env.read_txn().unwrap();
                                    let last_common_ancestor = node
                                        .archive
                                        .last_common_ancestor(
                                            &rotxn,
                                            tip,
                                            peer_state.tip,
                                        )
                                        .unwrap();
                                    node.archive
                                        .ancestors(&rotxn, peer_state.tip)
                                        .take_while(|block_hash| {
                                            Ok(*block_hash
                                                != last_common_ancestor)
                                        })
                                        .filter_map(|block_hash| {
                                            match node.archive.try_get_body(
                                                &rotxn, block_hash,
                                            )? {
                                                Some(_) => Ok(None),
                                                None => Ok(Some(block_hash)),
                                            }
                                        })
                                        .collect()
                                        .unwrap()
                                };
                                // Reorg If no bodies are missing
                                if missing_bodies.is_empty() {
                                    node.local_pool
                                        .spawn_pinned({
                                            let node = node.clone();
                                            let new_tip = peer_state.tip;
                                            move || async move {
                                                node.reorg_to_tip(new_tip).await
                                            }
                                        })
                                        .await
                                        .unwrap()
                                        .unwrap();
                                    continue;
                                }
                                // Request missing bodies
                                let _res: Vec<()> = missing_bodies.into_iter().map(|block_hash| {
                                    let node = node.clone();
                                    async move {
                                    let response = peer
                                    .request(&Request::GetBlock {
                                        block_hash
                                    }).await.unwrap();
                                    match response {
                                        Response::Block { header, body } => {
                                            let mut rwtxn = node.env.write_txn().unwrap();
                                            let block_hash = header.hash();
                                            node.archive.put_body(&mut rwtxn, block_hash, &body).unwrap();
                                            rwtxn.commit().unwrap();
                                        }
                                        Response::Headers(_) => {}
                                        Response::NoBlock => {}
                                        Response::NoHeader => {}
                                        Response::TransactionAccepted => {}
                                        Response::TransactionRejected => {}
                                    };
                                }}).collect::<JoinAll<_>>().await;
                            } else {
                                // Request headers
                                let response = peer
                                    .request(&Request::GetHeaders {
                                        end: peer_state.tip,
                                    })
                                    .await
                                    .unwrap();
                                match response {
                                    Response::Block { .. } => {}
                                    Response::Headers(headers) => {
                                        let mut rwtxn =
                                            node.env.write_txn().unwrap();
                                        // Store new headers
                                        for header in headers {
                                            let block_hash = header.hash();
                                            if node
                                                .archive
                                                .try_get_header(
                                                    &rwtxn, block_hash,
                                                )
                                                .unwrap()
                                                .is_none()
                                            {
                                                if node
                                                    .archive
                                                    .try_get_header(
                                                        &rwtxn,
                                                        header.prev_side_hash,
                                                    )
                                                    .unwrap()
                                                    .is_some()
                                                {
                                                    node.archive
                                                        .put_header(
                                                            &mut rwtxn, &header,
                                                        )
                                                        .unwrap();
                                                } else {
                                                    break;
                                                }
                                            }
                                        }
                                        rwtxn.commit().unwrap();
                                    }
                                    Response::NoBlock => {}
                                    Response::NoHeader => {}
                                    Response::TransactionAccepted => {}
                                    Response::TransactionRejected => {}
                                };
                            }
                        }
                    }
                }
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            }
        });
        Ok(())
    }
}
