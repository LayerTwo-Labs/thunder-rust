//! The wallet must follow a block that a peer mined

use std::time::Duration;

use bip300301_enforcer_integration_tests::{
    integration_test::{activate_sidechain, fund_enforcer, propose_sidechain},
    mine::{self, MiningPolicy},
    setup::{
        Mode, Network, PreSetup as EnforcerPreSetup,
        SetupOpts as EnforcerSetupOpts, Sidechain as _,
    },
    util::{AbortOnDrop, AsyncTrial, TestFailureCollector, TestFileRegistry},
};
use futures::{
    FutureExt as _, StreamExt as _, channel::mpsc, future::BoxFuture,
};
use thunder_app_rpc_api::{
    node::{PrivateRpcClient as _, RpcClient as _},
    wallet::RpcClient as _,
};
use tokio::time::sleep;
use tracing::Instrument as _;

use crate::{
    setup::{Init, PostSetup},
    util::BinPaths,
};

const DEPOSIT_AMOUNT: bitcoin::Amount = bitcoin::Amount::from_sat(21_000_000);
const DEPOSIT_FEE: bitcoin::Amount = bitcoin::Amount::from_sat(1_000_000);

/// How long the wallet gets to report the deposit the miner put in a block
const BALANCE_TIMEOUT: Duration = Duration::from_secs(30);

async fn wallet_sync_task(
    bin_paths: BinPaths,
    res_tx: mpsc::UnboundedSender<anyhow::Result<()>>,
) -> anyhow::Result<()> {
    let enforcer_pre_setup =
        EnforcerPreSetup::new(&bin_paths.others, Network::Regtest)?;
    let mut enforcer_post_setup = {
        let setup_opts: EnforcerSetupOpts = Default::default();
        enforcer_pre_setup
            .setup(Mode::Mempool, setup_opts, res_tx.clone())
            .await?
    };
    let miner = PostSetup::setup(
        Init {
            thunder_app: bin_paths.thunder()?.clone(),
            data_dir_suffix: Some("miner".to_owned()),
        },
        &enforcer_post_setup,
        res_tx.clone(),
    )
    .await?;
    let wallet = PostSetup::setup(
        Init {
            thunder_app: bin_paths.thunder()?.clone(),
            data_dir_suffix: Some("wallet".to_owned()),
        },
        &enforcer_post_setup,
        res_tx,
    )
    .await?;
    let () = propose_sidechain::<PostSetup>(&mut enforcer_post_setup).await?;
    let () = activate_sidechain::<PostSetup>(&mut enforcer_post_setup).await?;
    let () = fund_enforcer::<PostSetup>(&mut enforcer_post_setup).await?;
    tracing::info!("Setup successfully");

    let () = wallet
        .rpc_client
        .connect_peer(miner.net_addr().into())
        .await?;
    sleep(Duration::from_secs(1)).await;

    // The wallet node asks for the deposit itself, which is what the GUI does.
    tracing::info!("Wallet node: create the deposit");
    let _txid = wallet
        .rpc_client
        .create_deposit(
            wallet.deposit_address,
            DEPOSIT_AMOUNT.to_sat(),
            DEPOSIT_FEE.to_sat(),
        )
        .await?;
    let () = mine::mine::<PostSetup>(
        &mut enforcer_post_setup,
        1,
        MiningPolicy::VOTE,
    )
    .await?;

    tracing::info!("Miner node: BMM the block that applies the deposit");
    let () = miner.bmm_single(&mut enforcer_post_setup).await?;

    let deadline = tokio::time::Instant::now() + BALANCE_TIMEOUT;
    let mut balance = wallet.rpc_client.balance().await?;
    while balance.total != DEPOSIT_AMOUNT
        && tokio::time::Instant::now() < deadline
    {
        sleep(Duration::from_millis(500)).await;
        balance = wallet.rpc_client.balance().await?;
    }
    let blocks = wallet.rpc_client.getblockcount().await?;
    anyhow::ensure!(
        balance.total == DEPOSIT_AMOUNT,
        "wallet reports {} at {blocks} blocks, expected {DEPOSIT_AMOUNT}",
        balance.total
    );

    drop(wallet);
    drop(miner);
    drop(enforcer_post_setup.tasks);
    sleep(Duration::from_secs(1)).await;
    enforcer_post_setup.directories.base_dir.cleanup()?;
    Ok(())
}

async fn wallet_sync(bin_paths: BinPaths) -> anyhow::Result<()> {
    let (res_tx, mut res_rx) = mpsc::unbounded();
    let _test_task: AbortOnDrop<()> = tokio::task::spawn({
        let res_tx = res_tx.clone();
        async move {
            let res = wallet_sync_task(bin_paths, res_tx.clone()).await;
            let _send_err: Result<(), _> = res_tx.unbounded_send(res);
        }
        .in_current_span()
    })
    .into();
    res_rx.next().await.ok_or_else(|| {
        anyhow::anyhow!("Unexpected end of test task result stream")
    })?
}

pub fn wallet_sync_trial(
    bin_paths: BinPaths,
    file_registry: TestFileRegistry,
    failure_collector: TestFailureCollector,
) -> AsyncTrial<BoxFuture<'static, anyhow::Result<()>>> {
    AsyncTrial::new(
        "wallet_sync",
        wallet_sync(bin_paths).boxed(),
        file_registry,
        failure_collector,
    )
}
