//! Test that the wallet reuses its receive address until it receives

use bip300301_enforcer_integration_tests::{
    integration_test::{
        activate_sidechain, deposit, fund_enforcer, propose_sidechain,
    },
    setup::{
        Mode, Network, PostSetup as EnforcerPostSetup,
        PreSetup as EnforcerPreSetup, SetupOpts as EnforcerSetupOpts,
        Sidechain as _,
    },
    util::{
        AbortOnDrop, AsyncTrial, BinPaths as EnforcerBinPaths,
        TestFailureCollector, TestFileRegistry,
    },
};
use bitcoin::Amount;
use futures::{
    FutureExt as _, StreamExt as _, channel::mpsc, future::BoxFuture,
};
use thunder_app_rpc_api::wallet::RpcClient as _;
use tokio::time::sleep;
use tracing::Instrument as _;

use crate::{
    setup::{Init, PostSetup},
    util::BinPaths,
};

const DEPOSIT_AMOUNT: Amount = Amount::from_sat(21_000_000);
const DEPOSIT_FEE: Amount = Amount::from_sat(1_000_000);

/// Initial setup for the test
async fn setup(
    enforcer_bin_paths: &EnforcerBinPaths,
    res_tx: mpsc::UnboundedSender<anyhow::Result<()>>,
) -> anyhow::Result<EnforcerPostSetup> {
    let enforcer_pre_setup =
        EnforcerPreSetup::new(enforcer_bin_paths, Network::Regtest)?;
    let mut enforcer_post_setup = {
        let setup_opts: EnforcerSetupOpts = Default::default();
        enforcer_pre_setup
            .setup(Mode::Mempool, setup_opts, res_tx.clone())
            .await?
    };
    let () = propose_sidechain::<PostSetup>(&mut enforcer_post_setup).await?;
    let () = activate_sidechain::<PostSetup>(&mut enforcer_post_setup).await?;
    let () = fund_enforcer::<PostSetup>(&mut enforcer_post_setup).await?;
    Ok(enforcer_post_setup)
}

async fn receive_address_task(
    bin_paths: BinPaths,
    res_tx: mpsc::UnboundedSender<anyhow::Result<()>>,
) -> anyhow::Result<()> {
    let mut enforcer_post_setup =
        setup(&bin_paths.others, res_tx.clone()).await?;
    let mut sidechain = PostSetup::setup(
        Init {
            thunder_app: bin_paths.thunder()?.clone(),
            data_dir_suffix: None,
        },
        &enforcer_post_setup,
        res_tx,
    )
    .await?;
    tracing::info!("Setup thunder node successfully");

    // Setup asks for one address and keeps it as the deposit address.
    let deposit_address = sidechain.get_deposit_address().await?;
    let before = sidechain.rpc_client.get_wallet_addresses().await?.len();

    tracing::debug!("Checking that a template asks for no new address");
    for _ in 0..10 {
        let _template = sidechain.rpc_client.get_block_template().await?;
    }
    anyhow::ensure!(
        sidechain.rpc_client.get_wallet_addresses().await?.len() == before
    );

    tracing::debug!("Checking that a fresh address is still fresh");
    let fresh = sidechain.rpc_client.get_new_address().await?;
    anyhow::ensure!(fresh.to_string() != deposit_address);
    anyhow::ensure!(
        sidechain.rpc_client.get_wallet_addresses().await?.len() == before + 1
    );

    tracing::debug!("Depositing, so the receive address receives");
    let () = deposit(
        &mut enforcer_post_setup,
        &mut sidechain,
        &deposit_address,
        DEPOSIT_AMOUNT,
        DEPOSIT_FEE,
    )
    .await?;
    let after_deposit =
        sidechain.rpc_client.get_wallet_addresses().await?.len();

    tracing::debug!("Checking that a template still asks for no new address");
    for _ in 0..10 {
        let _template = sidechain.rpc_client.get_block_template().await?;
    }
    anyhow::ensure!(
        sidechain.rpc_client.get_wallet_addresses().await?.len()
            == after_deposit
    );

    drop(sidechain);
    tracing::info!(
        "Removing {}",
        enforcer_post_setup.directories.base_dir.path().display()
    );
    drop(enforcer_post_setup.tasks);
    // Wait for tasks to die
    sleep(std::time::Duration::from_secs(1)).await;
    enforcer_post_setup.directories.base_dir.cleanup()?;
    Ok(())
}

async fn receive_address(bin_paths: BinPaths) -> anyhow::Result<()> {
    let (res_tx, mut res_rx) = mpsc::unbounded();
    let _test_task: AbortOnDrop<()> = tokio::task::spawn({
        let res_tx = res_tx.clone();
        async move {
            let res = receive_address_task(bin_paths, res_tx.clone()).await;
            let _send_err: Result<(), _> = res_tx.unbounded_send(res);
        }
        .in_current_span()
    })
    .into();
    res_rx.next().await.ok_or_else(|| {
        anyhow::anyhow!("Unexpected end of test task result stream")
    })?
}

pub fn receive_address_trial(
    bin_paths: BinPaths,
    file_registry: TestFileRegistry,
    failure_collector: TestFailureCollector,
) -> AsyncTrial<BoxFuture<'static, anyhow::Result<()>>> {
    AsyncTrial::new(
        "receive_address",
        receive_address(bin_paths).boxed(),
        file_registry,
        failure_collector,
    )
}
