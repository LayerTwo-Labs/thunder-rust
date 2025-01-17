use std::{
    collections::{HashMap, HashSet},
    net::SocketAddr,
    sync::Arc,
};

use futures::{StreamExt, TryFutureExt, TryStreamExt as _};
use parking_lot::RwLock;
use rustreexo::accumulator::proof::Proof;
use serde::Deserialize;
use thunder::{
    miner::{self, Miner},
    node::{self, Node},
    types::{
        self,
        proto::mainchain::{
            self,
            generated::{validator_service_server, wallet_service_server},
        },
        Address, OutPoint, Output, Transaction,
    },
    wallet::{self, Wallet},
};
use tokio::{spawn, sync::RwLock as TokioRwLock, task::JoinHandle};
use tokio_util::task::LocalPoolHandle;
use tonic_reflection::pb::v1::{
    self as server_reflection,
    server_reflection_client::ServerReflectionClient,
    server_reflection_request, server_reflection_response,
};

use crate::cli::Config;

#[derive(Debug, Deserialize)]
struct StarterFile {
    mnemonic: String,
}

impl StarterFile {
    fn validate(&self) -> bool {
        bip39::Mnemonic::from_phrase(&self.mnemonic, bip39::Language::English)
            .is_ok()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("CUSF mainchain proto error")]
    CusfMainchain(#[from] thunder::types::proto::Error),
    #[error("gRPC reflection client error")]
    GrpcReflection(#[from] tonic::Status),
    #[error("io error")]
    Io(#[from] std::io::Error),
    #[error("miner error")]
    Miner(#[from] miner::Error),
    #[error("node error")]
    Node(#[from] node::Error),
    #[error("No CUSF mainchain wallet client")]
    NoCusfMainchainWalletClient,
    #[error("Utreexo error: {0}")]
    Utreexo(String),
    #[error("wallet error")]
    Wallet(#[from] wallet::Error),
    #[error("other error: {0}")]
    Other(#[from] anyhow::Error),
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

    /// Returns `true` if validator service AND wallet service are available,
    /// `false` if only validator service is available, and error if validator
    /// service is unavailable.
    async fn check_proto_support(
        host: SocketAddr,
        transport: tonic::transport::channel::Channel,
    ) -> Result<bool, tonic::Status> {
        let mut reflection_client = ServerReflectionClient::new(transport);
        let request = server_reflection::ServerReflectionRequest {
            host: host.to_string(),
            message_request: Some(
                server_reflection_request::MessageRequest::ListServices(
                    String::new(),
                ),
            ),
        };
        let mut resp_stream = reflection_client
            .server_reflection_info(futures::stream::once(
                futures::future::ready(request),
            ))
            .await?
            .into_inner();
        let resp = resp_stream.try_next().await?.ok_or_else(|| {
            tonic::Status::aborted(
                "reflection server closed response stream unexpectedly",
            )
        })?;
        let resp = resp.message_response.ok_or_else(|| {
            tonic::Status::aborted("Missing message response")
        })?;
        match resp {
            server_reflection_response::MessageResponse::ListServicesResponse(
                server_reflection::ListServiceResponse {
                    service
                }
            ) => {
                let services: HashSet<String> =
                    service.into_iter().map(|resp| resp.name).collect();
                if !services.contains(validator_service_server::SERVICE_NAME) {
                    let err_msg = format!(
                        "{} is not supported",
                        validator_service_server::SERVICE_NAME
                    );
                    Err(tonic::Status::aborted(err_msg))
                } else {
                    Ok(services.contains(wallet_service_server::SERVICE_NAME))
                }
            }
            server_reflection_response::MessageResponse::ErrorResponse(
                err_resp
            ) => {
                let err_msg = format!(
                    "Received error from reflection server: `{}`",
                    err_resp.error_message
                );
                Err(tonic::Status::aborted(err_msg))
            }
            _ => Err(tonic::Status::aborted(
                    "unexpected response from reflection server"
                ))
        }
    }

    pub fn new(config: &Config) -> Result<Self, Error> {
        // Node launches some tokio tasks for p2p networking, that is why we need a tokio runtime
        // here.
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;
        let wallet = Wallet::new(&config.datadir.join("wallet.mdb"))?;

        // Handle wallet reset first if requested
        if config.reset_wallet {
            wallet.reset_wallet()?;
        }

        // Then handle setting new seed if provided
        if let Some(mnemonic_seed_phrase_path) =
            &config.mnemonic_seed_phrase_path
        {
            let content = std::fs::read_to_string(mnemonic_seed_phrase_path)
                .map_err(|e| {
                    Error::Other(anyhow::anyhow!(
                        "Failed to read mnemonic seed phrase file: {}",
                        e
                    ))
                })?;

            let starter: StarterFile =
                serde_json::from_str(&content).map_err(|e| {
                    Error::Other(anyhow::anyhow!(
                        "Failed to parse mnemonic seed phrase file JSON: {}",
                        e
                    ))
                })?;

            if !starter.validate() {
                return Err(Error::Other(anyhow::anyhow!(
                    "Invalid mnemonic in seed phrase file"
                )));
            }

            let () = wallet.set_seed_from_mnemonic(&starter.mnemonic)?;
        }

        let rt_guard = runtime.enter();
        let transport = tonic::transport::channel::Channel::from_shared(
            format!("https://{}", config.main_addr),
        )
        .unwrap()
        .concurrency_limit(256)
        .connect_lazy();
        let (cusf_mainchain, cusf_mainchain_wallet) = if runtime.block_on(
            Self::check_proto_support(config.main_addr, transport.clone()),
        )? {
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
        let node = Node::new(
            &config.datadir,
            config.net_addr,
            cusf_mainchain,
            cusf_mainchain_wallet,
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
        let prev_side_hash = self.node.get_best_hash()?;
        let prev_main_hash = {
            let mut miner_write = miner.write().await;
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
            if tx_fees > bitcoin::Amount::ZERO {
                tx_fees
            } else {
                Self::EMPTY_BLOCK_BMM_BRIBE
            }
        });
        let mut miner_write = miner.write().await;
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
