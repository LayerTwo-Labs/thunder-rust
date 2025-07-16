//! Test an unknown withdrawal event

use bip300301_enforcer_integration_tests::{
    integration_test::{
        activate_sidechain, deposit, fund_enforcer, propose_sidechain,
        withdraw_succeed,
    },
    setup::{
        Mode, Network, PostSetup as EnforcerPostSetup, Sidechain as _,
        setup as setup_enforcer,
    },
    util::{AbortOnDrop, AsyncTrial},
};
use futures::{
    FutureExt as _, StreamExt as _, channel::mpsc, future::BoxFuture,
};
use thunder::types::OutPoint;
use thunder_app_rpc_api::RpcClient as _;
use tokio::time::sleep;
use tracing::Instrument as _;

use crate::{
    setup::{Init, PostSetup},
    util::BinPaths,
};

/// Initial setup for the test
async fn setup(
    bin_paths: &BinPaths,
    res_tx: mpsc::UnboundedSender<anyhow::Result<()>>,
) -> anyhow::Result<EnforcerPostSetup> {
    let mut enforcer_post_setup = setup_enforcer(
        &bin_paths.others,
        Network::Regtest,
        Mode::Mempool,
        res_tx.clone(),
    )
    .await?;
    let () = propose_sidechain::<PostSetup>(&mut enforcer_post_setup).await?;
    tracing::info!("Proposed sidechain successfully");
    let () = activate_sidechain::<PostSetup>(&mut enforcer_post_setup).await?;
    tracing::info!("Activated sidechain successfully");
    let () = fund_enforcer::<PostSetup>(&mut enforcer_post_setup).await?;
    Ok(enforcer_post_setup)
}

const DEPOSIT_AMOUNT: bitcoin::Amount = bitcoin::Amount::from_sat(21_000_000);
const DEPOSIT_FEE: bitcoin::Amount = bitcoin::Amount::from_sat(1_000_000);
const WITHDRAW_AMOUNT: bitcoin::Amount = bitcoin::Amount::from_sat(18_000_000);
const WITHDRAW_FEE: bitcoin::Amount = bitcoin::Amount::from_sat(1_000_000);

async fn unknown_withdrawal_task(
    bin_paths: BinPaths,
    res_tx: mpsc::UnboundedSender<anyhow::Result<()>>,
) -> anyhow::Result<()> {
    let mut enforcer_post_setup = setup(&bin_paths, res_tx.clone()).await?;
    let mut sidechain_withdrawer = PostSetup::setup(
        Init {
            thunder_app: bin_paths.thunder.clone(),
            data_dir_suffix: Some("withdrawer".to_owned()),
        },
        &enforcer_post_setup,
        res_tx.clone(),
    )
    .await?;
    tracing::info!("Setup thunder withdrawer node successfully");
    let withdrawer_deposit_address =
        sidechain_withdrawer.get_deposit_address().await?;
    let () = deposit(
        &mut enforcer_post_setup,
        &mut sidechain_withdrawer,
        &withdrawer_deposit_address,
        DEPOSIT_AMOUNT,
        DEPOSIT_FEE,
    )
    .await?;
    tracing::info!("Deposited to sidechain successfully");
    let () = withdraw_succeed(
        &mut enforcer_post_setup,
        &mut sidechain_withdrawer,
        WITHDRAW_AMOUNT,
        WITHDRAW_FEE,
        bitcoin::Amount::ZERO,
    )
    .await?;
    tracing::info!("Withdrawal succeeded");
    // New sidechain node, starting from scratch
    let mut sidechain_successor = PostSetup::setup(
        Init {
            thunder_app: bin_paths.thunder,
            data_dir_suffix: Some("successor".to_owned()),
        },
        &enforcer_post_setup,
        res_tx,
    )
    .await?;
    tracing::info!("Setup thunder successor node successfully");
    tracing::debug!("BMM 1 block");
    sidechain_successor
        .bmm_single(&mut enforcer_post_setup)
        .await?;
    tracing::debug!("Checking that successor sidechain has exactly 1 block");
    let successor_block_count =
        sidechain_successor.rpc_client.getblockcount().await?;
    anyhow::ensure!(successor_block_count == 1);
    tracing::debug!("Checking that successor sidechain has no deposit UTXOs");
    let successor_utxos = sidechain_successor.rpc_client.list_utxos().await?;
    anyhow::ensure!(successor_utxos.is_empty());
    let successor_deposit_address =
        sidechain_successor.get_deposit_address().await?;
    let () = deposit(
        &mut enforcer_post_setup,
        &mut sidechain_successor,
        &successor_deposit_address,
        DEPOSIT_AMOUNT,
        DEPOSIT_FEE,
    )
    .await?;
    tracing::info!("Deposited to sidechain successfully");
    tracing::debug!("Checking that withdrawer sidechain recognizes deposit");
    {
        let withdrawer_deposit_utxos_count = sidechain_withdrawer
            .rpc_client
            .list_utxos()
            .await?
            .into_iter()
            .filter(|utxo| matches!(utxo.outpoint, OutPoint::Deposit(_)))
            .count();
        sidechain_withdrawer
            .bmm_single(&mut enforcer_post_setup)
            .await?;
        let deposit_utxos_count_delta = sidechain_withdrawer
            .rpc_client
            .list_utxos()
            .await?
            .into_iter()
            .filter(|utxo| matches!(utxo.outpoint, OutPoint::Deposit(_)))
            .count()
            .checked_sub(withdrawer_deposit_utxos_count);
        anyhow::ensure!(deposit_utxos_count_delta == Some(1));
    }
    drop(sidechain_successor);
    drop(sidechain_withdrawer);
    tracing::info!("Removing {}", enforcer_post_setup.out_dir.path().display());
    drop(enforcer_post_setup.tasks);
    //debug Wait for tasks to die
    sleep(std::time::Duration::from_millis(200)).await;
    enforcer_post_setup.out_dir.cleanup()?;
    Ok(())
}

async fn unknown_withdrawal(bin_paths: BinPaths) -> anyhow::Result<()> {
    let (res_tx, mut res_rx) = mpsc::unbounded();
    let _test_task: AbortOnDrop<()> = tokio::task::spawn({
        let res_tx = res_tx.clone();
        async move {
            let res = unknown_withdrawal_task(bin_paths, res_tx.clone()).await;
            let _send_err: Result<(), _> = res_tx.unbounded_send(res);
        }
        .in_current_span()
    })
    .into();
    res_rx.next().await.ok_or_else(|| {
        anyhow::anyhow!("Unexpected end of test task result stream")
    })?
}

pub fn unknown_withdrawal_trial(
    bin_paths: BinPaths,
) -> AsyncTrial<BoxFuture<'static, anyhow::Result<()>>> {
    AsyncTrial::new("unknown_withdrawal", unknown_withdrawal(bin_paths).boxed())
}
