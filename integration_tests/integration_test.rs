use bip300301_enforcer_integration_tests::{
    integration_test as bip300301_enforcer_integration_test,
    setup::{
        Mode, Network, PostSetup as EnforcerPostSetup,
        PreSetup as EnforcerPreSetup, SetupOpts as EnforcerSetupOpts,
        Sidechain as _,
    },
    util::{AsyncTrial, TestFailureCollector, TestFileRegistry},
};
use bip300301_enforcer_lib::bins::CommandExt;
use futures::{FutureExt, channel::mpsc::UnboundedSender, future::BoxFuture};
use thunder_app_rpc_api::RpcClient as _;

use crate::{
    ibd::ibd_trial,
    setup::{Init, PostSetup},
    unknown_withdrawal::unknown_withdrawal_trial,
    util::BinPaths,
};

#[allow(clippy::significant_drop_tightening, reason = "false positive")]
pub async fn deposit_withdraw_roundtrip_task(
    post_setup: &mut EnforcerPostSetup,
    res_tx: UnboundedSender<anyhow::Result<()>>,
    init: Init,
) -> anyhow::Result<PostSetup> {
    use bip300301_enforcer_integration_test::{
        activate_sidechain, deposit, fund_enforcer, propose_sidechain,
        wait_for_wallet_sync, withdraw_succeed,
    };
    use bitcoin::Amount;

    const DEPOSIT_AMOUNT: Amount = Amount::from_sat(21_000_000);
    const DEPOSIT_FEE: Amount = Amount::from_sat(1_000_000);
    const WITHDRAW_AMOUNT: Amount = Amount::from_sat(18_000_000);
    const WITHDRAW_FEE: Amount = Amount::from_sat(1_000_000);

    let mut sidechain = PostSetup::setup(init, post_setup, res_tx).await?;
    tracing::info!("Setup successfully");
    let () = propose_sidechain::<PostSetup>(post_setup).await?;
    tracing::info!("Proposed sidechain successfully");
    let () = activate_sidechain::<PostSetup>(post_setup).await?;
    tracing::info!("Activated sidechain successfully");
    let () = fund_enforcer::<PostSetup>(post_setup).await?;
    tracing::info!("Funded enforcer successfully");
    let deposit_address = sidechain.get_deposit_address().await?;
    let () = deposit(
        post_setup,
        &mut sidechain,
        &deposit_address,
        DEPOSIT_AMOUNT,
        DEPOSIT_FEE,
    )
    .await?;
    tracing::info!("Deposited to sidechain successfully");
    // Wait for mempool to catch up before attempting second deposit
    tracing::debug!("Waiting for wallet sync...");
    let () = wait_for_wallet_sync().await?;
    tracing::info!("Attempting second deposit");
    let () = deposit(
        post_setup,
        &mut sidechain,
        &deposit_address,
        DEPOSIT_AMOUNT,
        DEPOSIT_FEE,
    )
    .await?;
    tracing::info!("Deposited to sidechain successfully");
    let sidechain_block_count = sidechain.rpc_client.getblockcount().await?;
    let target_sidechain_block_height = 5;
    tracing::info!(
        sidechain_block_count,
        target_sidechain_block_height,
        "BMMing sidechain blocks..."
    );
    sidechain
        .bmm(
            post_setup,
            target_sidechain_block_height - sidechain_block_count,
        )
        .await?;
    let () = withdraw_succeed(
        post_setup,
        &mut sidechain,
        WITHDRAW_AMOUNT,
        WITHDRAW_FEE,
        Amount::ZERO,
    )
    .await?;
    tracing::info!("Withdrawal succeeded");
    let mainchain_block_count = post_setup
        .bitcoin_cli
        .command::<String, _, String, _, _>([], "getblockcount", [])
        .run_utf8()
        .await?;
    let sidechain_block_count = sidechain.rpc_client.getblockcount().await?;
    tracing::info!(%mainchain_block_count, sidechain_block_count);
    Ok(sidechain)
}

async fn deposit_withdraw_roundtrip(
    mut post_setup: EnforcerPostSetup,
    init: Init,
    res_tx: UnboundedSender<anyhow::Result<()>>,
) -> anyhow::Result<()> {
    let sidechain_post_setup =
        deposit_withdraw_roundtrip_task(&mut post_setup, res_tx, init).await?;
    // check that everything is ok after BMM'ing 3 blocks
    let mut block_count_pre =
        sidechain_post_setup.rpc_client.getblockcount().await?;
    sidechain_post_setup.bmm_single(&mut post_setup).await?;
    let mut block_count_post =
        sidechain_post_setup.rpc_client.getblockcount().await?;
    anyhow::ensure!(block_count_post == block_count_pre + 1);
    block_count_pre = block_count_post;
    sidechain_post_setup.bmm_single(&mut post_setup).await?;
    block_count_post = sidechain_post_setup.rpc_client.getblockcount().await?;
    anyhow::ensure!(block_count_post == block_count_pre + 1);
    block_count_pre = block_count_post;
    sidechain_post_setup.bmm_single(&mut post_setup).await?;
    block_count_post = sidechain_post_setup.rpc_client.getblockcount().await?;
    anyhow::ensure!(block_count_post == block_count_pre + 1);
    Ok(())
}

fn deposit_withdraw_roundtrip_trial(
    bin_paths: BinPaths,
    file_registry: TestFileRegistry,
    failure_collector: TestFailureCollector,
) -> AsyncTrial<BoxFuture<'static, anyhow::Result<()>>> {
    AsyncTrial::new(
        "deposit_withdraw_roundtrip",
        async move {
            let (res_tx, _) = futures::channel::mpsc::unbounded();
            let pre_setup =
                EnforcerPreSetup::new(bin_paths.others, Network::Regtest)?;
            let post_setup = {
                let setup_opts: EnforcerSetupOpts = Default::default();
                pre_setup
                    .setup(Mode::Mempool, setup_opts, res_tx.clone())
                    .await?
            };
            deposit_withdraw_roundtrip(
                post_setup,
                Init {
                    thunder_app: bin_paths.thunder,
                    data_dir_suffix: None,
                },
                res_tx,
            )
            .await
        }
        .boxed(),
        file_registry,
        failure_collector,
    )
}

pub fn tests(
    bin_paths: BinPaths,
    file_registry: TestFileRegistry,
    failure_collector: TestFailureCollector,
) -> Vec<AsyncTrial<BoxFuture<'static, anyhow::Result<()>>>> {
    vec![
        deposit_withdraw_roundtrip_trial(
            bin_paths.clone(),
            file_registry.clone(),
            failure_collector.clone(),
        ),
        ibd_trial(
            bin_paths.clone(),
            file_registry.clone(),
            failure_collector.clone(),
        ),
        unknown_withdrawal_trial(bin_paths, file_registry, failure_collector),
    ]
}
