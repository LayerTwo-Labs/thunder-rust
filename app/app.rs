use std::{collections::HashMap, sync::Arc};

use futures::{StreamExt, TryFutureExt};
use parking_lot::RwLock;
use rustreexo::accumulator::proof::Proof;
use thunder::{
    format_deposit_address,
    miner::{self, Miner},
    node::{self, Node},
    types::{
        self, proto::mainchain, OutPoint, Output, Transaction, THIS_SIDECHAIN,
    },
    wallet::{self, Wallet},
};
use tokio::{spawn, sync::RwLock as TokioRwLock, task::JoinHandle};
use tokio_util::task::LocalPoolHandle;

use crate::cli::Config;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("CUSF mainchain error")]
    CusfMainchain(#[from] mainchain::Error),
    #[error("io error")]
    Io(#[from] std::io::Error),
    #[error("miner error")]
    Miner(#[from] miner::Error),
    #[error("node error")]
    Node(#[from] node::Error),
    #[error("Utreexo error: {0}")]
    Utreexo(String),
    #[error("wallet error")]
    Wallet(#[from] wallet::Error),
}

fn update_wallet(node: &Node, wallet: &Wallet) -> Result<(), Error> {
    let addresses = wallet.get_addresses()?;
    let utxos = node.get_utxos_by_addresses(&addresses)?;
    let outpoints: Vec<_> = wallet.get_utxos()?.into_keys().collect();
    let spent: Vec<_> = node
        .get_spent_utxos(&outpoints)?
        .into_iter()
        .map(|(outpoint, spent_output)| (outpoint, spent_output.inpoint))
        .collect();
    wallet.put_utxos(&utxos)?;
    wallet.spend_utxos(&spent)?;
    Ok(())
}

/// Update utxos & wallet
fn update(
    node: &Node,
    utxos: &mut HashMap<OutPoint, Output>,
    wallet: &Wallet,
) -> Result<(), Error> {
    let () = update_wallet(node, wallet)?;
    *utxos = wallet.get_utxos()?;
    Ok(())
}

#[derive(Clone)]
pub struct App {
    pub node: Arc<Node>,
    pub wallet: Wallet,
    pub miner: Arc<TokioRwLock<Miner>>,
    pub utxos: Arc<RwLock<HashMap<OutPoint, Output>>>,
    task: Arc<JoinHandle<()>>,
    pub transaction: Arc<RwLock<Transaction>>,
    pub runtime: Arc<tokio::runtime::Runtime>,
    pub local_pool: LocalPoolHandle,
}

impl App {
    async fn task(
        node: Arc<Node>,
        utxos: Arc<RwLock<HashMap<OutPoint, Output>>>,
        wallet: Wallet,
    ) -> Result<(), Error> {
        let mut state_changes = node.watch_state();
        while let Some(()) = state_changes.next().await {
            let () = update(&node, &mut utxos.write(), &wallet)?;
        }
        Ok(())
    }

    fn spawn_task(
        node: Arc<Node>,
        utxos: Arc<RwLock<HashMap<OutPoint, Output>>>,
        wallet: Wallet,
    ) -> JoinHandle<()> {
        spawn(Self::task(node, utxos, wallet).unwrap_or_else(|err| {
            let err = anyhow::Error::from(err);
            tracing::error!("{err:#}")
        }))
    }

    pub fn new(config: &Config) -> Result<Self, Error> {
        // Node launches some tokio tasks for p2p networking, that is why we need a tokio runtime
        // here.
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;
        let wallet = Wallet::new(&config.datadir.join("wallet.mdb"))?;
        if let Some(seed_phrase_path) = &config.mnemonic_seed_phrase_path {
            let mnemonic = std::fs::read_to_string(seed_phrase_path)?;
            let () = wallet.set_seed_from_mnemonic(mnemonic.as_str())?;
        }
        let cusf_mainchain = {
            let transport = tonic::transport::channel::Channel::from_shared(
                format!("https://{}", config.main_addr),
            )
            .unwrap()
            .connect_lazy();
            mainchain::Client::new(transport)
        };
        let miner = Miner::new(cusf_mainchain.clone())?;
        let rt_guard = runtime.enter();
        let local_pool = LocalPoolHandle::new(1);
        let node = Node::new(
            &config.datadir,
            config.net_addr,
            cusf_mainchain,
            local_pool.clone(),
        )?;
        let utxos = {
            let mut utxos = wallet.get_utxos()?;
            let transactions = node.get_all_transactions()?;
            for transaction in &transactions {
                for (outpoint, _) in &transaction.transaction.inputs {
                    utxos.remove(outpoint);
                }
            }
            Arc::new(RwLock::new(utxos))
        };
        let node = Arc::new(node);
        let task =
            Self::spawn_task(node.clone(), utxos.clone(), wallet.clone());
        drop(rt_guard);
        Ok(Self {
            node,
            wallet,
            miner: Arc::new(TokioRwLock::new(miner)),
            utxos,
            task: Arc::new(task),
            transaction: Arc::new(RwLock::new(Transaction {
                inputs: vec![],
                proof: Proof::default(),
                outputs: vec![],
            })),
            runtime: Arc::new(runtime),
            local_pool,
        })
    }

    /// Update utxos & wallet
    fn update(&self) -> Result<(), Error> {
        update(self.node.as_ref(), &mut self.utxos.write(), &self.wallet)
    }

    pub fn sign_and_send(&self, tx: Transaction) -> Result<(), Error> {
        let authorized_transaction = self.wallet.authorize(tx)?;
        self.node.submit_transaction(authorized_transaction)?;
        let () = self.update()?;
        Ok(())
    }

    pub fn get_new_main_address(
        &self,
    ) -> Result<bitcoin::Address<bitcoin::address::NetworkChecked>, Error> {
        let address = self.runtime.block_on({
            let miner = self.miner.clone();
            async move {
                let mut miner_write = miner.write().await;
                let cusf_mainchain = &mut miner_write.cusf_mainchain;
                let mainchain_info = cusf_mainchain.get_chain_info().await?;
                let res = cusf_mainchain
                    .create_new_address(None, mainchain::AddressType::Legacy)
                    .await?
                    .require_network(mainchain_info.network)
                    .unwrap();
                drop(miner_write);
                Result::<_, Error>::Ok(res)
            }
        })?;
        Ok(address)
    }

    const EMPTY_BLOCK_BMM_BRIBE: bitcoin::Amount =
        bitcoin::Amount::from_sat(1000);

    pub async fn mine(
        &self,
        fee: Option<bitcoin::Amount>,
    ) -> Result<(), Error> {
        const NUM_TRANSACTIONS: usize = 1000;
        let (txs, tx_fees) = self.node.get_transactions(NUM_TRANSACTIONS)?;
        let coinbase = match tx_fees {
            0 => vec![],
            _ => vec![types::Output {
                address: self.wallet.get_new_address()?,
                content: types::OutputContent::Value(tx_fees),
            }],
        };
        let body = types::Body::new(txs, coinbase);
        let prev_side_hash = self.node.get_best_hash()?;
        let prev_main_hash = {
            let mut miner_write = self.miner.write().await;
            let prev_main_hash =
                miner_write.cusf_mainchain.get_chain_tip().await?.block_hash;
            drop(miner_write);
            prev_main_hash
        };
        let roots = {
            let mut accumulator = self.node.get_tip_accumulator()?;
            body.modify_pollard(&mut accumulator.0)
                .map_err(Error::Utreexo)?;
            accumulator
                .0
                .get_roots()
                .iter()
                .map(|root| root.get_data())
                .collect()
        };
        let header = types::Header {
            merkle_root: body.compute_merkle_root(),
            roots,
            prev_side_hash,
            prev_main_hash,
        };
        let bribe = fee.unwrap_or_else(|| {
            if tx_fees > 0 {
                bitcoin::Amount::from_sat(tx_fees)
            } else {
                Self::EMPTY_BLOCK_BMM_BRIBE
            }
        });
        let mut miner_write = self.miner.write().await;
        miner_write
            .attempt_bmm(bribe.to_sat(), 0, header, body)
            .await?;
        // miner_write.generate().await?;
        tracing::trace!("confirming bmm...");
        if let Some((main_hash, header, body)) =
            miner_write.confirm_bmm().await?
        {
            tracing::trace!(
                "confirmed bmm, submitting block {}",
                header.hash()
            );
            self.node.submit_block(main_hash, &header, &body).await?;
        }
        drop(miner_write);
        let () = self.update()?;
        self.node.regenerate_proof(&mut self.transaction.write())?;
        Ok(())
    }

    pub fn deposit(
        &mut self,
        amount: bitcoin::Amount,
        fee: bitcoin::Amount,
    ) -> Result<(), Error> {
        self.runtime.block_on(async {
            let address = self.wallet.get_new_address()?;
            let address =
                format_deposit_address(THIS_SIDECHAIN, &format!("{address}"));
            let mut miner_write = self.miner.write().await;
            let _txid = miner_write
                .cusf_mainchain
                .create_deposit_tx(address, amount.to_sat(), fee.to_sat())
                .await?;
            drop(miner_write);
            Ok(())
        })
    }
}

impl Drop for App {
    fn drop(&mut self) {
        self.task.abort()
    }
}
