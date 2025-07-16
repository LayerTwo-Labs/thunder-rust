use std::{
    net::{Ipv4Addr, SocketAddrV4},
    path::PathBuf,
    time::Duration,
};

use bip300301_enforcer_integration_tests::{
    setup::{PostSetup as EnforcerPostSetup, Sidechain},
    util::AbortOnDrop,
};
use bip300301_enforcer_lib::types::SidechainNumber;
use futures::{TryFutureExt as _, channel::mpsc, future};
use reserve_port::ReservedPort;
use thiserror::Error;
use thunder::types::{OutputContent, PointedOutput};
use thunder_app_rpc_api::RpcClient as _;
use tokio::time::sleep;

use crate::util::ThunderApp;

#[derive(Debug)]
pub struct ReservedPorts {
    pub net: ReservedPort,
    pub rpc: ReservedPort,
}

impl ReservedPorts {
    pub fn new() -> Result<Self, reserve_port::Error> {
        Ok(Self {
            net: ReservedPort::random()?,
            rpc: ReservedPort::random()?,
        })
    }
}

#[derive(Debug)]
pub struct Init {
    pub thunder_app: PathBuf,
    pub data_dir_suffix: Option<String>,
}

#[derive(Debug, Error)]
pub enum BmmError {
    #[error(transparent)]
    Mine(#[from] bip300301_enforcer_integration_tests::mine::MineError),
    #[error(transparent)]
    RpcClient(#[from] jsonrpsee::core::ClientError),
}

#[derive(Debug, Error)]
pub enum SetupError {
    #[error("Failed to create thunder dir")]
    CreateThunderDir(#[source] std::io::Error),
    #[error(transparent)]
    ReservePort(#[from] reserve_port::Error),
    #[error(transparent)]
    RpcClient(#[from] jsonrpsee::core::ClientError),
    #[error("Timeout: Thunder RPC server not responding")]
    RpcTimeout,
}

#[derive(Debug, Error)]
pub enum ConfirmDepositError {
    #[error(transparent)]
    Bmm(#[from] BmmError),
    #[error("Deposit not found with txid: `{txid}`")]
    DepositNotFound { txid: bitcoin::Txid },
    #[error(transparent)]
    RpcClient(#[from] jsonrpsee::core::ClientError),
}

#[derive(Debug, Error)]
pub enum CreateWithdrawalError {
    #[error(transparent)]
    Bmm(#[from] BmmError),
    #[error("Pending withdrawal bundle not found")]
    PendingWithdrawalBundleNotFound,
    #[error(transparent)]
    RpcClient(#[from] jsonrpsee::core::ClientError),
}

#[derive(Debug)]
pub struct PostSetup {
    // MUST occur before temp dirs and reserved ports in order to ensure that processes are dropped
    // before reserved ports are freed and temp dirs are cleared
    pub _thunder_app_task: AbortOnDrop<()>,
    /// RPC client for thunder_app
    pub rpc_client: jsonrpsee::http_client::HttpClient,
    /// Address for receiving deposits
    pub deposit_address: thunder::types::Address,
    // MUST occur after tasks in order to ensure that tasks are dropped
    // before reserved ports are freed
    pub reserved_ports: ReservedPorts,
}

impl PostSetup {
    /// BMM a block
    pub async fn bmm_single(
        &self,
        post_setup: &mut EnforcerPostSetup,
    ) -> Result<(), BmmError> {
        use bip300301_enforcer_integration_tests::mine::mine;
        let ((), ()) = future::try_join(
            self.rpc_client.mine(None).map_err(BmmError::from),
            async {
                // debug: Sleep 1 sec way to long.
                sleep(Duration::from_millis(30)).await;
                mine::<Self>(post_setup, 1, Some(true))
                    .await
                    .map_err(BmmError::from)
            },
        )
        .await?;
        Ok(())
    }

    /// BMM blocks
    pub async fn bmm(
        &self,
        post_setup: &mut EnforcerPostSetup,
        blocks: u32,
    ) -> Result<(), BmmError> {
        for i in 0..blocks {
            tracing::debug!("BMM block {}/{blocks}", i + 1);
            let () = self.bmm_single(post_setup).await?;
        }
        Ok(())
    }

    pub fn net_port(&self) -> u16 {
        self.reserved_ports.net.port()
    }

    pub fn net_addr(&self) -> SocketAddrV4 {
        SocketAddrV4::new(Ipv4Addr::LOCALHOST, self.net_port())
    }
}

impl Sidechain for PostSetup {
    const SIDECHAIN_NUMBER: SidechainNumber =
        SidechainNumber(thunder::types::THIS_SIDECHAIN);

    type Init = Init;

    type SetupError = SetupError;

    async fn setup(
        init: Self::Init,
        post_setup: &EnforcerPostSetup,
        res_tx: mpsc::UnboundedSender<anyhow::Result<()>>,
    ) -> Result<Self, Self::SetupError> {
        let reserved_ports = ReservedPorts::new()?;
        let thunder_dir = if let Some(suffix) = init.data_dir_suffix {
            post_setup.out_dir.path().join(format!("thunder-{suffix}"))
        } else {
            post_setup.out_dir.path().join("thunder")
        };
        std::fs::create_dir(&thunder_dir)
            .map_err(Self::SetupError::CreateThunderDir)?;
        let thunder_app = ThunderApp {
            path: init.thunder_app,
            data_dir: thunder_dir,
            log_level: Some(tracing::Level::TRACE),
            mainchain_grpc_port: post_setup
                .reserved_ports
                .enforcer_serve_grpc
                .port(),
            net_port: reserved_ports.net.port(),
            rpc_port: reserved_ports.rpc.port(),
        };
        let thunder_app_task = thunder_app
            .spawn_command_with_args::<String, String, _, _, _>([], [], {
                let res_tx = res_tx.clone();
                move |err| {
                    let _err: Result<(), _> = res_tx.unbounded_send(Err(err));
                }
            });
        tracing::debug!("Started thunder");
        // sleep(Duration::from_secs(1)).await;
        // let rpc_client = jsonrpsee::http_client::HttpClient::builder()
        //     .build(format!("http://127.0.0.1:{}", reserved_ports.rpc.port()))?;

        // debug: Wait until the RPC server is ready
        let rpc_client = jsonrpsee::http_client::HttpClient::builder()
            .build(format!("http://127.0.0.1:{}", reserved_ports.rpc.port()))?;

        const MAX_WAIT_MS: u64 = 5000; // Max 5 sec
        const POLL_INTERVAL_MS: u64 = 200;

        let mut waited_ms = 0;
        loop {
            match rpc_client.getblockcount().await {
                Ok(_) => break, // RPC ready
                Err(_) if waited_ms >= MAX_WAIT_MS => {
                    return Err(Self::SetupError::RpcTimeout);
                }
                Err(_) => {
                    sleep(Duration::from_millis(POLL_INTERVAL_MS)).await;
                    waited_ms += POLL_INTERVAL_MS;
                }
            }
        }

        tracing::debug!("Generating mnemonic seed phrase");
        let mnemonic = rpc_client.generate_mnemonic().await?;
        tracing::debug!("Setting mnemonic seed phrase");
        let () = rpc_client.set_seed_from_mnemonic(mnemonic).await?;
        tracing::debug!("Generating deposit address");
        let deposit_address = rpc_client.get_new_address().await?;
        Ok(Self {
            _thunder_app_task: thunder_app_task,
            rpc_client,
            deposit_address,
            reserved_ports,
        })
    }

    type GetDepositAddressError = std::convert::Infallible;

    async fn get_deposit_address(
        &self,
    ) -> Result<String, Self::GetDepositAddressError> {
        Ok(self.deposit_address.to_string())
    }

    type ConfirmDepositError = ConfirmDepositError;

    async fn confirm_deposit(
        &mut self,
        post_setup: &mut EnforcerPostSetup,
        address: &str,
        value: bitcoin::Amount,
        txid: bitcoin::Txid,
    ) -> Result<(), Self::ConfirmDepositError> {
        let is_expected = |utxo: &PointedOutput| {
            utxo.output.address.to_string() == address
                && match utxo.output.content {
                    OutputContent::Value(utxo_value) => utxo_value == value,
                    OutputContent::Withdrawal { .. } => false,
                }
                && match utxo.outpoint {
                    thunder::types::OutPoint::Deposit(outpoint) => {
                        outpoint.txid == txid
                    }
                    _ => false,
                }
        };
        let utxos = self.rpc_client.list_utxos().await?;
        if utxos.iter().any(is_expected) {
            return Ok(());
        }
        tracing::debug!("Deposit not found, BMM 1 block...");
        let () = self.bmm_single(post_setup).await?;
        let utxos = self.rpc_client.list_utxos().await?;
        if utxos.iter().any(is_expected) {
            Ok(())
        } else {
            Err(Self::ConfirmDepositError::DepositNotFound { txid })
        }
    }

    type CreateWithdrawalError = CreateWithdrawalError;

    async fn create_withdrawal(
        &mut self,
        post_setup: &mut EnforcerPostSetup,
        receive_address: &bitcoin::Address,
        value: bitcoin::Amount,
        fee: bitcoin::Amount,
    ) -> Result<bip300301_enforcer_lib::types::M6id, Self::CreateWithdrawalError>
    {
        let _txid = self
            .rpc_client
            .withdraw(
                receive_address.as_unchecked().clone(),
                value.to_sat(),
                0,
                fee.to_sat(),
            )
            .await?;
        let blocks_to_mine = 'blocks_to_mine: {
            use thunder::state::WITHDRAWAL_BUNDLE_FAILURE_GAP;
            let block_count = self.rpc_client.getblockcount().await?;
            let Some(block_height) = block_count.checked_sub(1) else {
                break 'blocks_to_mine WITHDRAWAL_BUNDLE_FAILURE_GAP;
            };
            let latest_failed_withdrawal_bundle_height = self
                .rpc_client
                .latest_failed_withdrawal_bundle_height()
                .await?
                .unwrap_or(0);
            match WITHDRAWAL_BUNDLE_FAILURE_GAP.saturating_sub(
                block_height - latest_failed_withdrawal_bundle_height,
            ) {
                0 => WITHDRAWAL_BUNDLE_FAILURE_GAP + 1,
                blocks_to_mine => blocks_to_mine,
            }
        };
        tracing::debug!(
            "Mining thunder blocks until withdrawal bundle is broadcast"
        );
        let () = self.bmm(post_setup, blocks_to_mine).await?;
        let pending_withdrawal_bundle =
            self.rpc_client.pending_withdrawal_bundle().await?.ok_or(
                Self::CreateWithdrawalError::PendingWithdrawalBundleNotFound,
            )?;
        let m6id = pending_withdrawal_bundle.compute_m6id();
        Ok(bip300301_enforcer_lib::types::M6id(m6id.0))
    }
}
