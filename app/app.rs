use std::{borrow::BorrowMut, collections::HashMap, sync::Arc};

use fallible_iterator::FallibleIterator as _;
use futures::{StreamExt, TryFutureExt};
use parking_lot::RwLock;
use rustreexo::accumulator::proof::Proof;
use thunder::{
    miner::{self, Miner},
    node::{self, Node},
    types::{
        self, Address, FilledTransaction, OutPoint, Output, Transaction,
        proto::mainchain::{
            self,
            generated::{
                mining_service_server, validator_service_server,
                wallet_service_server,
            },
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
    #[error(transparent)]
    ModifyMemForest(#[from] thunder::types::ModifyMemForestError),
    #[error("node error")]
    Node(#[source] Box<node::Error>),
    #[error("No CUSF mainchain wallet client")]
    NoCusfMainchainWalletClient,
    #[error("Failed to request mainchain ancestor info for {block_hash}")]
    RequestMainchainAncestorInfos { block_hash: bitcoin::BlockHash },
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

struct ProtoSupport {
    miner: bool,
    wallet: bool,
}

/// A block that is ready to be blind merged mined
pub struct BlockTemplate {
    /// Bribe to offer for this block
    pub bribe: bitcoin::Amount,
    pub header: types::Header,
    pub body: types::Body,
    /// Fees collected by the transactions in the block
    pub fees: bitcoin::Amount,
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

    /// Returns `Ok(_)` iff validator service is available, and error if validator
    /// service is unavailable.
    async fn check_proto_support(
        transport: tonic::transport::channel::Channel,
    ) -> Result<ProtoSupport, tonic::Status> {
        let mut client = HealthClient::new(transport);

        let mining_service_name = mining_service_server::SERVICE_NAME;
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

        // The mining and wallet services are optional.
        let has_mining_service =
            Self::check_status_serving(&mut client, mining_service_name)
                .await?;
        let has_wallet_service =
            Self::check_status_serving(&mut client, wallet_service_name)
                .await?;

        tracing::info!(
            %has_mining_service,
            %has_wallet_service,
            "Checked existence of {}, {}",
            mining_service_name,
            wallet_service_name,
        );
        let res = ProtoSupport {
            miner: has_mining_service,
            wallet: has_wallet_service,
        };
        Ok(res)
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
        let (cusf_mainchain, cusf_mainchain_miner, cusf_mainchain_wallet) = {
            let ProtoSupport { miner, wallet } = runtime
                .block_on(Self::check_proto_support(transport.clone()))
                .map_err(|err| Error::VerifyMainchainServices {
                    url: Box::new(config.mainchain_grpc_url.clone()),
                    source: Box::new(err),
                })?;
            let mining_client = if miner {
                Some(mainchain::MiningClient::new(transport.clone()))
            } else {
                None
            };
            let wallet_client = if wallet {
                Some(mainchain::WalletClient::new(transport.clone()))
            } else {
                None
            };
            let validator_client = mainchain::ValidatorClient::new(transport);
            (validator_client, mining_client, wallet_client)
        };
        let miner = cusf_mainchain_wallet
            .clone()
            .map(|wallet| {
                Miner::new(
                    cusf_mainchain.clone(),
                    cusf_mainchain_miner.clone(),
                    wallet,
                )
            })
            .transpose()?;
        let local_pool = LocalPoolHandle::new(1);

        tracing::debug!("Instantiating node struct");
        let node = Node::new(
            thunder::node::Config {
                datadir: &config.datadir,
                bind_addr: config.net_addr,
                magic_bytes_override: config.network_magic_override,
                peers: &config.peers,
                network: config.network,
            },
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

    /// Regenerate proofs and submit transaction
    pub fn submit_transaction<Tx>(&self, tx: Tx) -> Result<(), Error>
    where
        Tx: BorrowMut<thunder::types::AuthorizedTransaction>,
    {
        self.node.submit_transaction(tx)?;
        let () = self.update()?;
        Ok(())
    }

    pub fn sign_and_send(&self, tx: Transaction) -> Result<(), Error> {
        let authorized_transaction = self.wallet.authorize(tx)?;
        self.submit_transaction(authorized_transaction)
    }

    pub async fn get_new_main_address(
        &self,
    ) -> Result<bitcoin::Address<bitcoin::address::NetworkChecked>, Error> {
        let Some(miner) = self.miner.as_ref() else {
            return Err(Error::NoCusfMainchainWalletClient);
        };
        let mut miner_write = miner.write().await;
        let cusf_mainchain = &mut miner_write.cusf_mainchain;
        let mainchain_info = cusf_mainchain.get_chain_info().await?;
        let cusf_mainchain_wallet = &mut miner_write.cusf_mainchain_wallet;
        let res = cusf_mainchain_wallet
            .create_new_address()
            .await?
            .require_network(mainchain_info.network)
            .unwrap();
        drop(miner_write);
        Ok(res)
    }

    pub fn get_new_main_address_blocking(
        &self,
    ) -> Result<bitcoin::Address<bitcoin::address::NetworkChecked>, Error> {
        self.runtime.block_on(self.get_new_main_address())
    }

    const EMPTY_BLOCK_BMM_BRIBE: bitcoin::Amount =
        bitcoin::Amount::from_sat(1000);

    /// Assemble a block to blind merge mine, without requesting BMM for it
    async fn build_block_template(
        &self,
        fee: Option<bitcoin::Amount>,
    ) -> Result<BlockTemplate, Error> {
        let prev_main_hash = self
            .node
            .with_cusf_mainchain(|cusf_mainchain| cusf_mainchain.clone())
            .get_chain_tip()
            .await?
            .block_hash;
        let tip_hash = self.node.try_get_best_hash()?;
        // If `prev_side_hash` is not the best tip to mine on, then mine an
        // empty block.
        // This is a temporary fix, ideally we always choose the best tip to
        // mine on
        let prev_side_hash = if let Some(tip_hash) = tip_hash {
            let tip_header = self.node.get_header(tip_hash)?;
            let archive = self.node.archive();
            let prev_main_hash_header_in_archive = {
                let rotxn =
                    self.node.env().read_txn().map_err(node::Error::from)?;
                archive
                    .try_get_main_header_info(&rotxn, &prev_main_hash)
                    .map_err(node::Error::from)?
                    .is_some()
            };
            if !prev_main_hash_header_in_archive {
                // Request mainchain header info
                if !self
                    .node
                    .request_mainchain_ancestor_infos(prev_main_hash)
                    .await?
                {
                    return Err(Error::RequestMainchainAncestorInfos {
                        block_hash: prev_main_hash,
                    });
                }
            }
            let rotxn =
                self.node.env().read_txn().map_err(node::Error::from)?;
            let last_common_main_ancestor = archive
                .last_common_main_ancestor(
                    &rotxn,
                    prev_main_hash,
                    tip_header.prev_main_hash,
                )
                .map_err(node::Error::from)?;
            if last_common_main_ancestor == tip_header.prev_main_hash {
                Some(tip_hash)
            } else {
                // Find a tip to mine on
                archive
                    .ancestor_headers(&rotxn, tip_hash)
                    .find_map(|(block_hash, header)| {
                        if header.prev_main_hash == last_common_main_ancestor {
                            Ok(None)
                        } else if archive.is_main_descendant(
                            &rotxn,
                            header.prev_main_hash,
                            last_common_main_ancestor,
                        )? {
                            Ok(Some(block_hash))
                        } else {
                            Ok(None)
                        }
                    })
                    .map_err(node::Error::from)?
            }
        } else {
            None
        };
        let (bribe, header, body, fees) = if prev_side_hash == tip_hash {
            const NUM_TRANSACTIONS: usize = 1000;
            let (txs, tx_fees) =
                self.node.get_transactions(NUM_TRANSACTIONS)?;
            let coinbase = match tx_fees {
                bitcoin::Amount::ZERO => Vec::new(),
                _ => vec![types::Output {
                    // A template is built on every poll and mostly thrown
                    // away, so it must not derive an address each time.
                    address: self.wallet.get_receive_address()?,
                    content: types::OutputContent::Value(tx_fees),
                }],
            };
            let (merkle_root, roots) = {
                let mut accumulator = if let Some(tip_hash) = tip_hash {
                    let rotxn = self
                        .node
                        .env()
                        .read_txn()
                        .map_err(node::Error::from)?;
                    self.node
                        .archive()
                        .get_accumulator(&rotxn, tip_hash)
                        .map_err(node::Error::from)?
                } else {
                    types::Accumulator::default()
                };
                let merkle_root = thunder::types::Body::modify_memforest(
                    &coinbase,
                    &txs,
                    &mut accumulator.0,
                )?;
                let roots = accumulator
                    .0
                    .get_roots()
                    .iter()
                    .map(|root| root.get_data())
                    .collect();
                (merkle_root, roots)
            };
            let body = types::Body::new(
                txs.into_iter().map(|tx| tx.into()).collect(),
                coinbase,
            );
            let header = types::Header {
                merkle_root,
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
            (bribe, header, body, tx_fees)
        } else {
            let coinbase = Vec::new();
            let (merkle_root, roots) = {
                let mut accumulator =
                    if let Some(prev_side_hash) = prev_side_hash {
                        let rotxn = self
                            .node
                            .env()
                            .read_txn()
                            .map_err(node::Error::from)?;
                        self.node
                            .archive()
                            .get_accumulator(&rotxn, prev_side_hash)
                            .map_err(node::Error::from)?
                    } else {
                        types::Accumulator::default()
                    };
                let merkle_root = thunder::types::Body::modify_memforest::<
                    FilledTransaction,
                >(
                    &coinbase, &[], &mut accumulator.0
                )?;
                let roots = accumulator
                    .0
                    .get_roots()
                    .iter()
                    .map(|root| root.get_data())
                    .collect();
                (merkle_root, roots)
            };
            let body = types::Body::new(Vec::new(), coinbase);
            let header = types::Header {
                merkle_root,
                roots,
                prev_side_hash,
                prev_main_hash,
            };
            let bribe = Self::EMPTY_BLOCK_BMM_BRIBE;
            (bribe, header, body, bitcoin::Amount::ZERO)
        };
        Ok(BlockTemplate {
            bribe,
            header,
            body,
            fees,
        })
    }

    /// Assemble a block to blind merge mine. The caller requests BMM for
    /// `header.hash()` itself, and submits the block via `connect_block` once
    /// its BMM request is included in a mainchain block.
    pub async fn get_block_template(&self) -> Result<BlockTemplate, Error> {
        self.build_block_template(None).await
    }

    /// Connect a block for which a BMM request was included in the specified
    /// mainchain block. Returns `true` if it was accepted as the new tip.
    pub async fn connect_block(
        &self,
        block: types::Block,
        main_block_hash: bitcoin::BlockHash,
    ) -> Result<bool, Error> {
        let types::Block { header, body } = block;
        let accepted = self
            .node
            .submit_block(main_block_hash, &header, &body)
            .await?;
        if accepted {
            let () = self.update()?;
        }
        Ok(accepted)
    }

    /// Attempt to mine a sidechain block
    pub async fn mine(
        &self,
        fee: Option<bitcoin::Amount>,
    ) -> Result<(), Error> {
        let Some(miner) = self.miner.as_ref() else {
            return Err(Error::NoCusfMainchainWalletClient);
        };
        let BlockTemplate {
            bribe,
            header,
            body,
            ..
        } = self.build_block_template(fee).await?;
        let mut miner_write = miner.write().await;
        let bmm_txid = miner_write
            .attempt_bmm(bribe.to_sat(), 0, header, body)
            .await?;

        tracing::debug!(%bmm_txid, "mine: confirming BMM...");
        if let Some((main_hash, header, body)) =
            miner_write.confirm_bmm().await.inspect_err(|err| {
                tracing::error!("{:#}", thunder::util::ErrorChain::new(err))
            })?
        {
            tracing::debug!(
                %main_hash, side_hash = %header.hash(), "mine: confirmed BMM, submitting block",
            );
            match self
                .node
                .submit_block(main_hash, &header, &body)
                .await
                .inspect_err(|err| {
                    tracing::error!("{:#}", thunder::util::ErrorChain::new(err))
                })? {
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

    pub async fn deposit(
        &self,
        address: Address,
        amount: bitcoin::Amount,
        fee: bitcoin::Amount,
    ) -> Result<bitcoin::Txid, Error> {
        let Some(miner) = self.miner.as_ref() else {
            return Err(Error::NoCusfMainchainWalletClient);
        };
        let mut miner_write = miner.write().await;
        let txid = miner_write
            .cusf_mainchain_wallet
            .create_deposit_tx(address, amount.to_sat(), fee.to_sat())
            .await?;
        drop(miner_write);
        Ok(txid)
    }

    pub fn deposit_blocking(
        &self,
        address: Address,
        amount: bitcoin::Amount,
        fee: bitcoin::Amount,
    ) -> Result<bitcoin::Txid, Error> {
        self.runtime.block_on(self.deposit(address, amount, fee))
    }
}

impl Drop for App {
    fn drop(&mut self) {
        self.task.abort()
    }
}
