//! Test assembling block templates without a mainchain wallet

use bip300301_enforcer_integration_tests::{
    setup::{
        EnforcerWallet, Mode, Network, PostSetup as EnforcerPostSetup,
        PreSetup as EnforcerPreSetup, SetupOpts as EnforcerSetupOpts,
        Sidechain as _,
    },
    util::{
        AbortOnDrop, AsyncTrial, BinPaths as EnforcerBinPaths,
        TestFailureCollector, TestFileRegistry,
    },
};
use futures::{
    FutureExt as _, StreamExt as _, channel::mpsc, future::BoxFuture,
};
use thunder_app_rpc_api::{node::RpcClient as _, wallet::RpcClient as _};
use tokio::time::sleep;
use tracing::Instrument as _;

use crate::{
    setup::{Init, PostSetup},
    util::BinPaths,
};

/// Initial setup for the test. The enforcer serves no `WalletService`, and the
/// sidechain is neither proposed nor activated: both steps need services that a
/// wallet-less enforcer does not serve.
async fn setup(
    enforcer_bin_paths: &EnforcerBinPaths,
    res_tx: mpsc::UnboundedSender<anyhow::Result<()>>,
) -> anyhow::Result<EnforcerPostSetup> {
    let enforcer_pre_setup =
        EnforcerPreSetup::new(enforcer_bin_paths, Network::Regtest)?;
    let setup_opts: EnforcerSetupOpts = EnforcerSetupOpts {
        enforcer_wallet: EnforcerWallet::Disabled,
        ..Default::default()
    };
    enforcer_pre_setup
        .setup(Mode::Mempool, setup_opts, res_tx)
        .await
}

async fn wallet_less_block_template_task(
    bin_paths: BinPaths,
    res_tx: mpsc::UnboundedSender<anyhow::Result<()>>,
) -> anyhow::Result<()> {
    let enforcer_post_setup = setup(&bin_paths.others, res_tx.clone()).await?;
    let sidechain = PostSetup::setup(
        Init {
            thunder_app: bin_paths.thunder()?.clone(),
            data_dir_suffix: None,
        },
        &enforcer_post_setup,
        res_tx,
    )
    .await?;
    tracing::info!("Setup thunder node successfully");

    // Guards the premise. Were a wallet served here, the template check below
    // would cover nothing.
    tracing::debug!("Checking that mining needs a mainchain wallet client");
    anyhow::ensure!(sidechain.rpc_client.mine(None).await.is_err());

    tracing::debug!("Checking that a template is built without a wallet");
    let template = sidechain.rpc_client.get_block_template().await?;
    anyhow::ensure!(template.block.header.prev_side_hash.is_none());
    anyhow::ensure!(template.block.body.transactions.is_empty());
    anyhow::ensure!(template.fees_sats == 0);
    anyhow::ensure!(template.block.header.hash() == template.critical_hash);

    tracing::debug!("Checking that a template is stable while the chain is");
    let template_repeat = sidechain.rpc_client.get_block_template().await?;
    anyhow::ensure!(template_repeat.critical_hash == template.critical_hash);

    // The block carries no BMM request, so it must not connect. It must fail
    // on that, and never on the missing wallet client.
    tracing::debug!("Checking that connecting a block reaches the BMM check");
    let connect_err = sidechain
        .rpc_client
        .connect_block(
            template.block.clone(),
            template.block.header.prev_main_hash,
        )
        .await;
    match connect_err {
        Ok(accepted) => anyhow::ensure!(!accepted),
        Err(err) => anyhow::ensure!(
            !format!("{err:#}").contains("No CUSF mainchain wallet client"),
            "connecting a block asked for a mainchain wallet: {err:#}"
        ),
    }

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

async fn wallet_less_block_template(bin_paths: BinPaths) -> anyhow::Result<()> {
    let (res_tx, mut res_rx) = mpsc::unbounded();
    let _test_task: AbortOnDrop<()> = tokio::task::spawn({
        let res_tx = res_tx.clone();
        async move {
            let res =
                wallet_less_block_template_task(bin_paths, res_tx.clone())
                    .await;
            let _send_err: Result<(), _> = res_tx.unbounded_send(res);
        }
        .in_current_span()
    })
    .into();
    res_rx.next().await.ok_or_else(|| {
        anyhow::anyhow!("Unexpected end of test task result stream")
    })?
}

pub fn wallet_less_block_template_trial(
    bin_paths: BinPaths,
    file_registry: TestFileRegistry,
    failure_collector: TestFailureCollector,
) -> AsyncTrial<BoxFuture<'static, anyhow::Result<()>>> {
    AsyncTrial::new(
        "wallet_less_block_template",
        wallet_less_block_template(bin_paths).boxed(),
        file_registry,
        failure_collector,
    )
}
