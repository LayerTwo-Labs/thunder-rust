//! Initial block download tests

use std::net::SocketAddr;

use bip300301_enforcer_integration_tests::{
    integration_test::{
        activate_sidechain, deposit, fund_enforcer, propose_sidechain,
    },
    setup::{
        Mode, Network, PostSetup as EnforcerPostSetup,
        PreSetup as EnforcerPreSetup, SetupOpts as EnforcerSetupOpts,
        Sidechain as _,
    },
    util::{AbortOnDrop, AsyncTrial, TestFailureCollector, TestFileRegistry},
};
use futures::{FutureExt, StreamExt as _, channel::mpsc, future::BoxFuture};
use thunder_app_rpc_api::node::{PrivateRpcClient as _, RpcClient as _};
use tokio::time::sleep;
use tracing::Instrument as _;

use crate::{
    setup::{Init, PostSetup},
    util::BinPaths,
};

#[derive(Debug)]
struct ThunderNodes {
    /// Sidechain process that will be sending blocks
    sender: PostSetup,
    /// The sidechain instance that will be syncing blocks
    syncer: PostSetup,
}

/// Initial setup for the test
async fn setup(
    bin_paths: BinPaths,
    res_tx: mpsc::UnboundedSender<anyhow::Result<()>>,
) -> anyhow::Result<(EnforcerPostSetup, ThunderNodes)> {
    let enforcer_pre_setup =
        EnforcerPreSetup::new(&bin_paths.others, Network::Regtest)?;
    let mut enforcer_post_setup = {
        let setup_opts: EnforcerSetupOpts = Default::default();
        enforcer_pre_setup
            .setup(Mode::Mempool, setup_opts, res_tx.clone())
            .await?
    };
    let sidechain_sender = PostSetup::setup(
        Init {
            thunder_app: bin_paths.thunder()?.clone(),
            data_dir_suffix: Some("sender".to_owned()),
        },
        &enforcer_post_setup,
        res_tx.clone(),
    )
    .await?;
    tracing::info!("Setup thunder send node successfully");
    let sidechain_syncer = PostSetup::setup(
        Init {
            thunder_app: bin_paths.thunder()?.clone(),
            data_dir_suffix: Some("syncer".to_owned()),
        },
        &enforcer_post_setup,
        res_tx,
    )
    .await?;
    tracing::info!("Setup thunder sync node successfully");
    let thunder_nodes = ThunderNodes {
        sender: sidechain_sender,
        syncer: sidechain_syncer,
    };
    tracing::info!("Setup successfully");
    let () = propose_sidechain::<PostSetup>(&mut enforcer_post_setup).await?;
    tracing::info!("Proposed sidechain successfully");
    let () = activate_sidechain::<PostSetup>(&mut enforcer_post_setup).await?;
    tracing::info!("Activated sidechain successfully");
    let () = fund_enforcer::<PostSetup>(&mut enforcer_post_setup).await?;
    Ok((enforcer_post_setup, thunder_nodes))
}

/// Check that a Thunder node is connected to the specified peer
async fn check_peer_connection(
    thunder_setup: &PostSetup,
    expected_peer: SocketAddr,
) -> anyhow::Result<()> {
    let peers = thunder_setup
        .rpc_client
        .list_peers()
        .await?
        .iter()
        .map(|p| p.address)
        .collect::<Vec<_>>();

    if peers.contains(&expected_peer) {
        Ok(())
    } else {
        Err(anyhow::anyhow!(
            "Expected connection to {expected_peer}, found {peers:?}"
        ))
    }
}

/// What the syncer holds before it meets the sender.
#[derive(Clone, Copy, Debug)]
enum SyncerStart {
    /// Fresh node: plain IBD.
    Empty,
    /// The syncer already BMM'd its own chain of three blocks, with a deposit
    /// that lands in the second one and nothing in the third. Adopting the
    /// sender's chain then has to disconnect a tip whose parent is the most
    /// recent deposit block, which is the shape that panicked the alphanet
    /// seed in `State::disconnect` (`two_way_peg_data.rs`, deposit-height
    /// assert) on 2026-09-09.
    OwnChainWithDeposit,
}

/// Number of blocks the syncer holds under [`SyncerStart::OwnChainWithDeposit`]
const OWN_CHAIN_BLOCKS: u32 = 3;

async fn initial_block_download_task(
    bin_paths: BinPaths,
    res_tx: mpsc::UnboundedSender<anyhow::Result<()>>,
    syncer_start: SyncerStart,
) -> anyhow::Result<()> {
    use bitcoin::Amount;
    const DEPOSIT_AMOUNT: Amount = Amount::from_sat(21_000_000);
    const DEPOSIT_FEE: Amount = Amount::from_sat(1_000_000);

    let (mut enforcer_post_setup, mut thunder_nodes) =
        setup(bin_paths, res_tx).await?;
    let expected_syncer_blocks = match syncer_start {
        SyncerStart::Empty => 0,
        SyncerStart::OwnChainWithDeposit => {
            tracing::info!("Syncer: BMM block 1 (no deposit)");
            thunder_nodes
                .syncer
                .bmm(&mut enforcer_post_setup, 1)
                .await?;
            let deposit_address =
                thunder_nodes.syncer.get_deposit_address().await?;
            // `deposit` mines the mainchain deposit block, then
            // `confirm_deposit` BMMs syncer block 2 to apply it.
            tracing::info!("Syncer: deposit, applied by BMM block 2");
            let () = deposit(
                &mut enforcer_post_setup,
                &mut thunder_nodes.syncer,
                &deposit_address,
                DEPOSIT_AMOUNT,
                DEPOSIT_FEE,
            )
            .await?;
            tracing::info!("Syncer: BMM block 3 (no deposit)");
            thunder_nodes
                .syncer
                .bmm(&mut enforcer_post_setup, 1)
                .await?;
            let syncer_blocks =
                thunder_nodes.syncer.rpc_client.getblockcount().await?;
            anyhow::ensure!(
                syncer_blocks == OWN_CHAIN_BLOCKS,
                "syncer should hold {OWN_CHAIN_BLOCKS} blocks, has {syncer_blocks}"
            );
            OWN_CHAIN_BLOCKS
        }
    };
    const BMM_BLOCKS: u32 = 16;
    tracing::info!(blocks = %BMM_BLOCKS, "Attempting BMM");
    thunder_nodes
        .sender
        .bmm(&mut enforcer_post_setup, BMM_BLOCKS)
        .await?;
    // Check that sender has all blocks, and syncer only its own
    {
        let sender_blocks =
            thunder_nodes.sender.rpc_client.getblockcount().await?;
        anyhow::ensure!(sender_blocks == BMM_BLOCKS);
        let syncer_blocks =
            thunder_nodes.syncer.rpc_client.getblockcount().await?;
        anyhow::ensure!(syncer_blocks == expected_syncer_blocks);
    }
    tracing::info!("Attempting sync");
    tracing::debug!(
        sender_addr = %thunder_nodes.sender.net_addr(),
        syncer_addr = %thunder_nodes.syncer.net_addr(),
        "Connecting syncer to sender");
    let () = thunder_nodes
        .syncer
        .rpc_client
        .connect_peer(thunder_nodes.sender.net_addr().into())
        .await?;
    // Wait for connection to be established
    sleep(std::time::Duration::from_secs(1)).await;
    tracing::debug!("Checking peer connections");
    // Check peer connections
    let () = check_peer_connection(
        &thunder_nodes.syncer,
        thunder_nodes.sender.net_addr().into(),
    )
    .await?;
    tracing::debug!("Syncer has connection to sender");
    let () = check_peer_connection(
        &thunder_nodes.sender,
        thunder_nodes.syncer.net_addr().into(),
    )
    .await?;
    tracing::debug!("Sender has connection to syncer");
    // Wait for sync to occur
    sleep(std::time::Duration::from_secs(10)).await;
    // Check peer connections
    let () = check_peer_connection(
        &thunder_nodes.syncer,
        thunder_nodes.sender.net_addr().into(),
    )
    .await?;
    tracing::debug!("Syncer still has connection to sender");
    // Check that sender and syncer have all blocks, on the same tip
    {
        let sender_blocks =
            thunder_nodes.sender.rpc_client.getblockcount().await?;
        anyhow::ensure!(sender_blocks == BMM_BLOCKS);
        let syncer_blocks =
            thunder_nodes.syncer.rpc_client.getblockcount().await?;
        anyhow::ensure!(
            syncer_blocks == BMM_BLOCKS,
            "syncer stuck at {syncer_blocks} blocks, sender at {sender_blocks}"
        );
        let sender_tip = thunder_nodes
            .sender
            .rpc_client
            .get_best_sidechain_block_hash()
            .await?;
        let syncer_tip = thunder_nodes
            .syncer
            .rpc_client
            .get_best_sidechain_block_hash()
            .await?;
        anyhow::ensure!(
            sender_tip == syncer_tip,
            "syncer tip {syncer_tip:?} != sender tip {sender_tip:?}"
        );
    }
    drop(thunder_nodes.syncer);
    drop(thunder_nodes.sender);
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

async fn ibd(
    bin_paths: BinPaths,
    syncer_start: SyncerStart,
) -> anyhow::Result<()> {
    let (res_tx, mut res_rx) = mpsc::unbounded();
    let _test_task: AbortOnDrop<()> = tokio::task::spawn({
        let res_tx = res_tx.clone();
        async move {
            let res = initial_block_download_task(
                bin_paths,
                res_tx.clone(),
                syncer_start,
            )
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

pub fn ibd_trial(
    bin_paths: BinPaths,
    file_registry: TestFileRegistry,
    failure_collector: TestFailureCollector,
) -> AsyncTrial<BoxFuture<'static, anyhow::Result<()>>> {
    AsyncTrial::new(
        "initial_block_download",
        ibd(bin_paths, SyncerStart::Empty).boxed(),
        file_registry,
        failure_collector,
    )
}

/// IBD onto a node that must first reorg its own chain away, disconnecting a
/// tip whose parent carries the latest deposit.
pub fn reorg_across_deposit_trial(
    bin_paths: BinPaths,
    file_registry: TestFileRegistry,
    failure_collector: TestFailureCollector,
) -> AsyncTrial<BoxFuture<'static, anyhow::Result<()>>> {
    AsyncTrial::new(
        "reorg_across_deposit",
        ibd(bin_paths, SyncerStart::OwnChainWithDeposit).boxed(),
        file_registry,
        failure_collector,
    )
}
