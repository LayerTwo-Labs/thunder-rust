//! Test assembling block templates

use bip300301_enforcer_integration_tests::{
    integration_test::{activate_sidechain, fund_enforcer, propose_sidechain},
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
    tracing::info!("Proposed sidechain successfully");
    let () = activate_sidechain::<PostSetup>(&mut enforcer_post_setup).await?;
    tracing::info!("Activated sidechain successfully");
    let () = fund_enforcer::<PostSetup>(&mut enforcer_post_setup).await?;
    Ok(enforcer_post_setup)
}

async fn block_template_task(
    bin_paths: BinPaths,
    res_tx: mpsc::UnboundedSender<anyhow::Result<()>>,
) -> anyhow::Result<()> {
    let mut enforcer_post_setup =
        setup(&bin_paths.others, res_tx.clone()).await?;
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

    tracing::debug!("Checking that the first template is empty");
    let template = sidechain.rpc_client.get_block_template().await?;
    anyhow::ensure!(template.block.header.prev_side_hash.is_none());
    anyhow::ensure!(template.block.body.transactions.is_empty());
    anyhow::ensure!(template.fees_sats == 0);

    tracing::debug!("Checking that a template is stable while the chain is");
    let template_repeat = sidechain.rpc_client.get_block_template().await?;
    anyhow::ensure!(template_repeat.critical_hash == template.critical_hash);

    tracing::debug!("Checking that a template is not connected by itself");
    anyhow::ensure!(sidechain.rpc_client.getblockcount().await? == 0);

    tracing::debug!("BMM 1 block");
    let () = sidechain.bmm_single(&mut enforcer_post_setup).await?;
    anyhow::ensure!(sidechain.rpc_client.getblockcount().await? == 1);

    tracing::debug!("Checking that a template builds on the new tips");
    let template_connected = sidechain.rpc_client.get_block_template().await?;
    let best_side_hash =
        sidechain.rpc_client.get_best_sidechain_block_hash().await?;
    let best_main_hash =
        sidechain.rpc_client.get_best_mainchain_block_hash().await?;
    anyhow::ensure!(best_side_hash.is_some());
    anyhow::ensure!(
        template_connected.block.header.prev_side_hash == best_side_hash
    );
    anyhow::ensure!(
        Some(template_connected.block.header.prev_main_hash) == best_main_hash
    );
    anyhow::ensure!(
        template_connected.critical_hash != template_repeat.critical_hash
    );

    tracing::debug!("Checking that a template commits to its own block");
    anyhow::ensure!(
        template_connected.block.header.hash()
            == template_connected.critical_hash
    );

    tracing::debug!("Checking that a block without BMM is not connected");
    let connect_without_bmm = sidechain
        .rpc_client
        .connect_block(
            template_connected.block.clone(),
            template_connected.block.header.prev_main_hash,
        )
        .await;
    anyhow::ensure!(matches!(connect_without_bmm, Err(_) | Ok(false)));
    anyhow::ensure!(sidechain.rpc_client.getblockcount().await? == 1);

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

async fn block_template(bin_paths: BinPaths) -> anyhow::Result<()> {
    let (res_tx, mut res_rx) = mpsc::unbounded();
    let _test_task: AbortOnDrop<()> = tokio::task::spawn({
        let res_tx = res_tx.clone();
        async move {
            let res = block_template_task(bin_paths, res_tx.clone()).await;
            let _send_err: Result<(), _> = res_tx.unbounded_send(res);
        }
        .in_current_span()
    })
    .into();
    res_rx.next().await.ok_or_else(|| {
        anyhow::anyhow!("Unexpected end of test task result stream")
    })?
}

pub fn block_template_trial(
    bin_paths: BinPaths,
    file_registry: TestFileRegistry,
    failure_collector: TestFailureCollector,
) -> AsyncTrial<BoxFuture<'static, anyhow::Result<()>>> {
    AsyncTrial::new(
        "block_template",
        block_template(bin_paths).boxed(),
        file_registry,
        failure_collector,
    )
}
