use std::{collections::HashMap, sync::Arc};

use futures::{StreamExt, TryFutureExt};
use parking_lot::RwLock;
use rustreexo::accumulator::proof::Proof;
use thunder::{
    miner::{self, Miner},
    node::{self, Node},
    types::{
        self, Address, OutPoint, Output, Transaction,
        proto::mainchain::{
            self,
            generated::{validator_service_server, wallet_service_server},
        },
    },
    wallet::{self, Wallet},
};
use tokio::{spawn, sync::RwLock as TokioRwLock, task::JoinHandle};
use tokio_util::task::LocalPoolHandle;
use tonic_health::{
    ServingStatus,
    pb::{HealthCheckRequest, health_client::HealthClient},
};

use crate::cli::Config;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("CUSF mainchain proto error")]
    CusfMainchain(#[from] thunder::types::proto::Error),
    #[error("io error")]
    Io(#[from] std::io::Error),
    #[error("miner error")]
    Miner(#[from] miner::Error),
    #[error("node error")]
    Node(#[source] Box<node::Error>),
    #[error("No CUSF mainchain wallet client")]
    NoCusfMainchainWalletClient,
    #[error("Utreexo error: {0}")]
    Utreexo(String),
    #[error("Unable to verify existence of CUSF mainchain service(s) at {url}")]
    VerifyMainchainServices {
        url: Box<url::Url>,
        source: Box<tonic::Status>,
    },
    #[error("wallet error")]
    Wallet(#[from] wallet::Error),
}

impl From<node::Error> for Error {
    fn from(err: node::Error) -> Self {
        Self::Node(Box::new(err))
    }
}

fn update_wallet(node: &Node, wallet: &Wallet) -> Result<(), Error> {
    tracing::trace!("starting wallet update");
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

    tracing::debug!("finished wallet update");
    Ok(())
}

/// Update utxos & wallet
fn update(
    node: &Node,
    utxos: &mut HashMap<OutPoint, Output>,
    wallet: &Wallet,
) -> Result<(), Error> {
    tracing::trace!("Updating wallet");
    let () = update_wallet(node, wallet)?;
    *utxos = wallet.get_utxos()?;
    tracing::trace!("Updated wallet");
    Ok(())
}

#[derive(Clone)]
pub struct App {
    pub node: Arc<Node>,
    pub wallet: Wallet,
    pub miner: Option<Arc<TokioRwLock<Miner>>>,
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

    async fn check_status_serving(
        client: &mut HealthClient<tonic::transport::Channel>,
        service_name: &str,
    ) -> Result<bool, tonic::Status> {
        match client
            .check(HealthCheckRequest {
                service: service_name.to_string(),
            })
            .await
        {
            Ok(res) => {
                let expected_status = ServingStatus::Serving;
                let status = res.into_inner().status;

                let as_expected = status == expected_status as i32;
                if !as_expected {
                    tracing::warn!(
                        "Expected status {} for {}, got {}",
                        expected_status,
                        service_name,
                        status
                    );
                }
                Ok(as_expected)
            }
            Err(status) if status.code() == tonic::Code::NotFound => Ok(false),
            Err(e) => Err(e),
        }
    }

    /// Returns `true` if validator service AND wallet service are available,
    /// `false` if only validator service is available, and error if validator
    /// service is unavailable.
    async fn check_proto_support(
        transport: tonic::transport::channel::Channel,
    ) -> Result<bool, tonic::Status> {
        let mut client = HealthClient::new(transport);

        let validator_service_name = validator_service_server::SERVICE_NAME;
        let wallet_service_name = wallet_service_server::SERVICE_NAME;

        // The validator service MUST exist. We therefore error out here directly.
        if !Self::check_status_serving(&mut client, validator_service_name)
            .await?
        {
            return Err(tonic::Status::aborted(format!(
                "{validator_service_name} is not supported in mainchain client",
            )));
        }

        tracing::info!("Verified existence of {}", validator_service_name);

        // The wallet service is optional.
        let has_wallet_service =
            Self::check_status_serving(&mut client, wallet_service_name)
                .await?;

        tracing::info!(
            "Checked existence of {}: {}",
            wallet_service_name,
            has_wallet_service
        );
        Ok(has_wallet_service)
    }

    pub fn new(config: &Config) -> Result<Self, Error> {
        // Node launches some tokio tasks for p2p networking, that is why we need a tokio runtime
        // here.
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;

        tracing::info!(
            "Instantiating wallet with data directory: {}",
            config.datadir.display()
        );

        let wallet = Wallet::new(&config.datadir.join("wallet.mdb"))?;
        if let Some(seed_phrase_path) = &config.mnemonic_seed_phrase_path {
            let mnemonic = std::fs::read_to_string(seed_phrase_path)?;
            let () = wallet.set_seed_from_mnemonic(mnemonic.as_str())?;
        }

        tracing::info!(
            "Connecting to mainchain at {}",
            config.mainchain_grpc_url
        );
        let rt_guard = runtime.enter();
        let transport = tonic::transport::channel::Channel::from_shared(
            format!("{}", config.mainchain_grpc_url),
        )
        .unwrap()
        .concurrency_limit(256)
        .connect_lazy();
        let (cusf_mainchain, cusf_mainchain_wallet) = if runtime
            .block_on(Self::check_proto_support(transport.clone()))
            .map_err(|err| Error::VerifyMainchainServices {
                url: Box::new(config.mainchain_grpc_url.clone()),
                source: Box::new(err),
            })? {
            (
                mainchain::ValidatorClient::new(transport.clone()),
                Some(mainchain::WalletClient::new(transport)),
            )
        } else {
            (mainchain::ValidatorClient::new(transport), None)
        };
        let miner = cusf_mainchain_wallet
            .clone()
            .map(|wallet| Miner::new(cusf_mainchain.clone(), wallet))
            .transpose()?;
        let local_pool = LocalPoolHandle::new(1);

        tracing::debug!("Instantiating node struct");
        let node = Node::new(
            &config.datadir,
            config.net_addr,
            cusf_mainchain,
            cusf_mainchain_wallet,
            &runtime,
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
        let miner = miner.map(|miner| Arc::new(TokioRwLock::new(miner)));
        let task =
            Self::spawn_task(node.clone(), utxos.clone(), wallet.clone());
        drop(rt_guard);
        Ok(Self {
            node,
            wallet,
            miner,
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
        let Some(miner) = self.miner.as_ref() else {
            return Err(Error::NoCusfMainchainWalletClient);
        };
        let address = self.runtime.block_on({
            let miner = miner.clone();
            async move {
                let mut miner_write = miner.write().await;
                let cusf_mainchain = &mut miner_write.cusf_mainchain;
                let mainchain_info = cusf_mainchain.get_chain_info().await?;
                let cusf_mainchain_wallet =
                    &mut miner_write.cusf_mainchain_wallet;
                let res = cusf_mainchain_wallet
                    .create_new_address()
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
        let Some(miner) = self.miner.as_ref() else {
            return Err(Error::NoCusfMainchainWalletClient);
        };
        const NUM_TRANSACTIONS: usize = 1000;
        let (txs, tx_fees) = self.node.get_transactions(NUM_TRANSACTIONS)?;
        let coinbase = match tx_fees {
            bitcoin::Amount::ZERO => vec![],
            _ => vec![types::Output {
                address: self.wallet.get_new_address()?,
                content: types::OutputContent::Value(tx_fees),
            }],
        };
        let body = types::Body::new(txs, coinbase);
        let prev_side_hash = self.node.try_get_best_hash()?;
        let prev_main_hash = {
            let mut miner_write = miner.write().await;
            let prev_main_hash =
                miner_write.cusf_mainchain.get_chain_tip().await?.block_hash;
            drop(miner_write);
            prev_main_hash
        };
        let roots = {
            let mut accumulator = self.node.get_tip_accumulator()?;
            body.modify_memforest(&mut accumulator.0)
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
            if tx_fees > bitcoin::Amount::ZERO {
                tx_fees
            } else {
                Self::EMPTY_BLOCK_BMM_BRIBE
            }
        });

        let mut miner_write = miner.write().await;
        let bmm_txid = miner_write
            .attempt_bmm(bribe.to_sat(), 0, header, body)
            .await?;

        tracing::debug!(%bmm_txid, "mine: confirming BMM...");
        if let Some((main_hash, header, body)) =
            miner_write.confirm_bmm().await?
        {
            tracing::debug!(
                %main_hash, side_hash = %header.hash(), "mine: confirmed BMM, submitting block",
            );
            match self.node.submit_block(main_hash, &header, &body).await? {
                true => {
                    tracing::debug!(
                         %main_hash, "mine: BMM accepted as new tip",
                    );
                }
                false => {
                    tracing::warn!(
                        %main_hash, "mine: BMM not accepted as new tip",
                    );
                }
            }
        }

        drop(miner_write);
        let () = self.update()?;

        self.node
            .regenerate_proof(&mut self.transaction.write())
            .inspect_err(|err| {
                tracing::error!("mine: unable to regenerate proof: {err:#}");
            })?;
        Ok(())
    }

    pub fn deposit(
        &self,
        address: Address,
        amount: bitcoin::Amount,
        fee: bitcoin::Amount,
    ) -> Result<bitcoin::Txid, Error> {
        let Some(miner) = self.miner.as_ref() else {
            return Err(Error::NoCusfMainchainWalletClient);
        };
        self.runtime.block_on(async {
            let mut miner_write = miner.write().await;
            let txid = miner_write
                .cusf_mainchain_wallet
                .create_deposit_tx(address, amount.to_sat(), fee.to_sat())
                .await?;
            drop(miner_write);
            Ok(txid)
        })
    }
}

impl Drop for App {
    fn drop(&mut self) {
        self.task.abort()
    }
}
