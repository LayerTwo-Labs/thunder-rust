//! Test that a node serves the mempool and the block index that an Esplora
//! index reads

use bip300301_enforcer_integration_tests::{
    integration_test::{
        activate_sidechain, deposit, fund_enforcer, propose_sidechain,
    },
    setup::{
        Mode, Network, PreSetup as EnforcerPreSetup,
        SetupOpts as EnforcerSetupOpts, Sidechain as _,
    },
    util::{AbortOnDrop, AsyncTrial, TestFailureCollector, TestFileRegistry},
};
use bitcoin::Amount;
use futures::{
    FutureExt as _, StreamExt as _, channel::mpsc, future::BoxFuture,
};
use thunder::types::{
    BlockIndex, BlockIndexTx, GetValue as _, OutPoint,
    transaction::outpoint::DEPOSIT_MATURITY,
};
use thunder_app_rpc_api::{node::RpcClient as _, wallet::RpcClient as _};
use tokio::time::sleep;
use tracing::Instrument as _;

use crate::{
    setup::{Init, PostSetup},
    util::BinPaths,
};

const DEPOSIT_AMOUNT: Amount = Amount::from_sat(21_000_000);
const DEPOSIT_FEE: Amount = Amount::from_sat(1_000_000);
const TRANSFER_AMOUNT: u64 = 1_000_000;
const TRANSFER_FEE: u64 = 1_000;

/// The block index of each block in the active chain, from genesis to the tip
async fn block_indexes(
    sidechain: &PostSetup,
) -> anyhow::Result<Vec<BlockIndex>> {
    let block_count = sidechain.rpc_client.getblockcount().await?;
    let mut res = Vec::new();
    for height in 0..block_count {
        let block_hash = sidechain
            .rpc_client
            .get_block_hash(height)
            .await?
            .ok_or_else(|| anyhow::anyhow!("no block at height {height}"))?;
        res.push(sidechain.rpc_client.get_block_index(block_hash).await?);
    }
    Ok(res)
}

async fn block_index_task(
    bin_paths: BinPaths,
    res_tx: mpsc::UnboundedSender<anyhow::Result<()>>,
) -> anyhow::Result<()> {
    let enforcer_pre_setup =
        EnforcerPreSetup::new(&bin_paths.others, Network::Regtest)?;
    let setup_opts: EnforcerSetupOpts = Default::default();
    let mut enforcer_post_setup = enforcer_pre_setup
        .setup(Mode::Mempool, setup_opts, res_tx.clone())
        .await?;
    let mut sidechain = PostSetup::setup(
        Init::new(bin_paths.thunder()?.clone()),
        &enforcer_post_setup,
        res_tx,
    )
    .await?;
    let () = propose_sidechain::<PostSetup>(&mut enforcer_post_setup).await?;
    let () = activate_sidechain::<PostSetup>(&mut enforcer_post_setup).await?;
    let () = fund_enforcer::<PostSetup>(&mut enforcer_post_setup).await?;
    tracing::info!("Setup successfully");

    anyhow::ensure!(sidechain.rpc_client.list_mempool().await?.is_empty());

    let deposit_address = sidechain.get_deposit_address().await?;
    let () = deposit(
        &mut enforcer_post_setup,
        &mut sidechain,
        &deposit_address,
        DEPOSIT_AMOUNT,
        DEPOSIT_FEE,
    )
    .await?;
    tracing::info!("Deposited to sidechain successfully");

    tracing::debug!("Checking that one block lists the deposit");
    let deposits: Vec<_> = block_indexes(&sidechain)
        .await?
        .into_iter()
        .flat_map(|block_index| block_index.deposits)
        .collect();
    anyhow::ensure!(deposits.len() == 1);
    anyhow::ensure!(matches!(deposits[0].outpoint, OutPoint::Deposit(_)));
    anyhow::ensure!(deposits[0].output.get_value() == DEPOSIT_AMOUNT);
    anyhow::ensure!(sidechain.rpc_client.list_mempool().await?.is_empty());

    tracing::debug!("BMMing sidechain blocks until the deposit is mature");
    let () = sidechain
        .bmm(&mut enforcer_post_setup, DEPOSIT_MATURITY)
        .await?;

    let dest = sidechain.rpc_client.get_new_address().await?;
    let txid = sidechain
        .rpc_client
        .create_transfer(dest, TRANSFER_AMOUNT, TRANSFER_FEE)
        .await?;
    tracing::info!(%txid, "Created a transfer");

    tracing::debug!("Checking that the mempool holds the transfer");
    let mempool = sidechain.rpc_client.list_mempool().await?;
    anyhow::ensure!(mempool.len() == 1);
    let entry = &mempool[0];
    anyhow::ensure!(entry.txid == txid);
    let from_tx = BlockIndexTx::new(&entry.tx)?;
    anyhow::ensure!(from_tx.txid == txid);
    anyhow::ensure!(from_tx.size == entry.size);
    anyhow::ensure!(from_tx.raw == entry.raw);

    tracing::debug!("Checking that the next block lists the transfer");
    let () = sidechain.bmm_single(&mut enforcer_post_setup).await?;
    anyhow::ensure!(sidechain.rpc_client.list_mempool().await?.is_empty());
    let tip = sidechain
        .rpc_client
        .get_best_sidechain_block_hash()
        .await?
        .ok_or_else(|| anyhow::anyhow!("no sidechain tip"))?;
    let tip_index = sidechain.rpc_client.get_block_index(tip).await?;
    anyhow::ensure!(tip_index.txs.len() == 1);
    anyhow::ensure!(tip_index.txs[0].txid == entry.txid);
    anyhow::ensure!(tip_index.txs[0].size == entry.size);
    anyhow::ensure!(tip_index.txs[0].raw == entry.raw);
    anyhow::ensure!(tip_index.deposits.is_empty());

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

async fn block_index(bin_paths: BinPaths) -> anyhow::Result<()> {
    let (res_tx, mut res_rx) = mpsc::unbounded();
    let _test_task: AbortOnDrop<()> = tokio::task::spawn({
        let res_tx = res_tx.clone();
        async move {
            let res = block_index_task(bin_paths, res_tx.clone()).await;
            let _send_err: Result<(), _> = res_tx.unbounded_send(res);
        }
        .in_current_span()
    })
    .into();
    res_rx.next().await.ok_or_else(|| {
        anyhow::anyhow!("Unexpected end of test task result stream")
    })?
}

pub fn block_index_trial(
    bin_paths: BinPaths,
    file_registry: TestFileRegistry,
    failure_collector: TestFailureCollector,
) -> AsyncTrial<BoxFuture<'static, anyhow::Result<()>>> {
    AsyncTrial::new(
        "block_index",
        block_index(bin_paths).boxed(),
        file_registry,
        failure_collector,
    )
}
