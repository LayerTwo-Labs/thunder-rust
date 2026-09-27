//! Test that the wallet spends an unconfirmed output, and that one block
//! carries the whole chain.

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
use std::future::Future;

use bitcoin::Amount;
use futures::{
    FutureExt as _, StreamExt as _, channel::mpsc, future::BoxFuture,
};
use thunder_app_rpc_api::{node::RpcClient as _, wallet::RpcClient as _};
use tracing::Instrument as _;

use crate::{
    setup::{Init, PostSetup},
    util::BinPaths,
};

const DEPOSIT_AMOUNT: Amount = Amount::from_sat(21_000_000);
const DEPOSIT_FEE: Amount = Amount::from_sat(1_000_000);
const TRANSFER_FEE: Amount = Amount::from_sat(1_000);

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

async fn spend_unconfirmed_task(
    bin_paths: BinPaths,
    res_tx: mpsc::UnboundedSender<anyhow::Result<()>>,
) -> anyhow::Result<()> {
    let mut enforcer_post_setup =
        setup(&bin_paths.others, res_tx.clone()).await?;
    let mut sidechain = PostSetup::setup(
        Init {
            thunder_app: bin_paths.thunder()?.clone(),
            data_dir_suffix: None,
            extra_args: Vec::new(),
        },
        &enforcer_post_setup,
        res_tx,
    )
    .await?;

    let deposit_address = sidechain.get_deposit_address().await?;
    let () = deposit(
        &mut enforcer_post_setup,
        &mut sidechain,
        &deposit_address,
        DEPOSIT_AMOUNT,
        DEPOSIT_FEE,
    )
    .await?;
    tracing::info!("Deposited to the sidechain");

    let confirmed = sidechain.rpc_client.balance().await?;
    anyhow::ensure!(confirmed.total == DEPOSIT_AMOUNT);
    anyhow::ensure!(confirmed.unconfirmed == Amount::ZERO);

    // The deposit is one coin, so the first transfer takes all of it and the
    // second one must spend the first one's change.
    let dest = sidechain.rpc_client.get_new_address().await?;
    let half = DEPOSIT_AMOUNT.to_sat() / 2;
    let block_count = sidechain.rpc_client.getblockcount().await?;

    tracing::debug!("Sending the parent transfer");
    let parent = sidechain
        .rpc_client
        .create_transfer(dest, half, TRANSFER_FEE.to_sat())
        .await?;
    anyhow::ensure!(
        sidechain.rpc_client.getblockcount().await? == block_count,
        "the parent must stay in the mempool",
    );

    let pending = sidechain.rpc_client.balance().await?;
    anyhow::ensure!(
        pending.unconfirmed > Amount::ZERO,
        "the change of the parent must read as unconfirmed",
    );
    anyhow::ensure!(
        !sidechain
            .rpc_client
            .get_unconfirmed_wallet_utxos()
            .await?
            .is_empty(),
        "the wallet must hold an unconfirmed output",
    );

    tracing::debug!("Sending the child transfer, which spends the change");
    let child = sidechain
        .rpc_client
        .create_transfer(dest, half / 2, TRANSFER_FEE.to_sat())
        .await?;
    anyhow::ensure!(child != parent);

    for txid in [parent, child] {
        let found = sidechain.rpc_client.get_transaction(txid).await?;
        anyhow::ensure!(
            found.is_some_and(|found| found.block_hash.is_none()),
            "{txid} must sit in the mempool",
        );
    }

    tracing::debug!("Checking that one template carries the whole chain");
    let template = sidechain.rpc_client.get_block_template().await?;
    let body: Vec<_> = template
        .block
        .body
        .transactions
        .iter()
        .map(|tx| tx.txid())
        .collect();
    anyhow::ensure!(
        body == vec![parent, child],
        "the parent must come first, the body holds {body:?}",
    );

    tracing::debug!("BMM one block, which must carry the parent and the child");
    let () = sidechain.bmm_single(&mut enforcer_post_setup).await?;
    anyhow::ensure!(
        sidechain.rpc_client.getblockcount().await? == block_count + 1
    );
    for txid in [parent, child] {
        let found = sidechain.rpc_client.get_transaction(txid).await?;
        anyhow::ensure!(
            found.is_some_and(|found| found.block_hash.is_some()),
            "the block must carry {txid}",
        );
    }
    let settled = sidechain.rpc_client.balance().await?;
    anyhow::ensure!(
        settled.unconfirmed == Amount::ZERO,
        "nothing stays unconfirmed after the block",
    );
    // The wallet pays itself, and the coinbase of the block it mines pays the
    // fees back to it, so the whole deposit stays.
    anyhow::ensure!(
        settled.total == DEPOSIT_AMOUNT,
        "the wallet paid itself and took the fees back: {settled:?}",
    );

    drop(sidechain);
    drop(enforcer_post_setup.tasks);
    tracing::info!(
        "Removing {}",
        enforcer_post_setup.directories.base_dir.path().display()
    );
    enforcer_post_setup.directories.base_dir.cleanup()?;
    Ok(())
}

/// With `--spend-zero-conf-change false` the wallet must still show the value
/// of its own unconfirmed change, and must refuse to spend it. Bitcoin Core
/// reports such value and refuses it under the same option.
async fn spend_unconfirmed_off_task(
    bin_paths: BinPaths,
    res_tx: mpsc::UnboundedSender<anyhow::Result<()>>,
) -> anyhow::Result<()> {
    let mut enforcer_post_setup =
        setup(&bin_paths.others, res_tx.clone()).await?;
    let mut sidechain = PostSetup::setup(
        Init {
            thunder_app: bin_paths.thunder()?.clone(),
            data_dir_suffix: None,
            extra_args: vec!["--spend-zero-conf-change=false".to_owned()],
        },
        &enforcer_post_setup,
        res_tx,
    )
    .await?;

    let deposit_address = sidechain.get_deposit_address().await?;
    let () = deposit(
        &mut enforcer_post_setup,
        &mut sidechain,
        &deposit_address,
        DEPOSIT_AMOUNT,
        DEPOSIT_FEE,
    )
    .await?;

    let confirmed = sidechain.rpc_client.balance().await?;
    anyhow::ensure!(confirmed.total == DEPOSIT_AMOUNT);
    anyhow::ensure!(confirmed.unconfirmed == Amount::ZERO);

    let dest = sidechain.rpc_client.get_new_address().await?;
    let half = DEPOSIT_AMOUNT.to_sat() / 2;
    let _parent = sidechain
        .rpc_client
        .create_transfer(dest, half, TRANSFER_FEE.to_sat())
        .await?;

    let pending = sidechain.rpc_client.balance().await?;
    anyhow::ensure!(
        pending.unconfirmed > Amount::ZERO,
        "the change must stay visible, got {pending:?}",
    );
    anyhow::ensure!(
        pending.total == DEPOSIT_AMOUNT - TRANSFER_FEE,
        "the wallet still holds the deposit less the fee, got {pending:?}",
    );
    anyhow::ensure!(
        pending.available == Amount::ZERO,
        "the wallet may take nothing, got {pending:?}",
    );
    let refused = sidechain
        .rpc_client
        .create_transfer(dest, half / 2, TRANSFER_FEE.to_sat())
        .await;
    let Err(err) = refused else {
        anyhow::bail!("the wallet must refuse to spend its unconfirmed change");
    };
    anyhow::ensure!(
        err.to_string().contains("not enough funds"),
        "the wallet refused for the wrong reason: {err}",
    );

    drop(sidechain);
    drop(enforcer_post_setup.tasks);
    enforcer_post_setup.directories.base_dir.cleanup()?;
    Ok(())
}

/// Run one task under its own result channel, so an early failure inside the
/// task reaches the trial rather than the spawned task.
async fn run_task<Fut>(
    bin_paths: BinPaths,
    task: impl FnOnce(BinPaths, mpsc::UnboundedSender<anyhow::Result<()>>) -> Fut
    + Send
    + 'static,
) -> anyhow::Result<()>
where
    Fut: Future<Output = anyhow::Result<()>> + Send,
{
    let (res_tx, mut res_rx) = mpsc::unbounded();
    let _test_task: AbortOnDrop<()> = tokio::task::spawn({
        let res_tx = res_tx.clone();
        async move {
            let res = task(bin_paths, res_tx.clone()).await;
            let _send_err: Result<(), _> = res_tx.unbounded_send(res);
        }
        .in_current_span()
    })
    .into();
    res_rx.next().await.ok_or_else(|| {
        anyhow::anyhow!("Unexpected end of test task result stream")
    })?
}

pub fn spend_unconfirmed_trial(
    bin_paths: BinPaths,
    file_registry: TestFileRegistry,
    failure_collector: TestFailureCollector,
) -> AsyncTrial<BoxFuture<'static, anyhow::Result<()>>> {
    AsyncTrial::new(
        "spend_unconfirmed",
        run_task(bin_paths, spend_unconfirmed_task).boxed(),
        file_registry,
        failure_collector,
    )
}

pub fn spend_unconfirmed_off_trial(
    bin_paths: BinPaths,
    file_registry: TestFileRegistry,
    failure_collector: TestFailureCollector,
) -> AsyncTrial<BoxFuture<'static, anyhow::Result<()>>> {
    AsyncTrial::new(
        "spend_unconfirmed_off",
        run_task(bin_paths, spend_unconfirmed_off_task).boxed(),
        file_registry,
        failure_collector,
    )
}
