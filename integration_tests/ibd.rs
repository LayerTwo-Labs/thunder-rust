//! Initial block download tests

use std::net::SocketAddr;

use bip300301_enforcer_integration_tests::{
    integration_test::{activate_sidechain, fund_enforcer, propose_sidechain},
    setup::{
        Mode, Network, PostSetup as EnforcerPostSetup, Sidechain as _,
        setup as setup_enforcer,
    },
    util::{AbortOnDrop, AsyncTrial},
};
use futures::{FutureExt, StreamExt as _, channel::mpsc, future::BoxFuture};
use thunder_app_rpc_api::RpcClient as _;
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
    let mut enforcer_post_setup = setup_enforcer(
        &bin_paths.others,
        Network::Regtest,
        Mode::Mempool,
        res_tx.clone(),
    )
    .await?;
    let sidechain_sender = PostSetup::setup(
        Init {
            thunder_app: bin_paths.thunder.clone(),
            data_dir_suffix: Some("sender".to_owned()),
        },
        &enforcer_post_setup,
        res_tx.clone(),
    )
    .await?;
    tracing::info!("Setup thunder send node successfully");
    let sidechain_syncer = PostSetup::setup(
        Init {
            thunder_app: bin_paths.thunder.clone(),
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

async fn initial_block_download_task(
    bin_paths: BinPaths,
    res_tx: mpsc::UnboundedSender<anyhow::Result<()>>,
) -> anyhow::Result<()> {
    let (mut enforcer_post_setup, thunder_nodes) =
        setup(bin_paths, res_tx).await?;
    const BMM_BLOCKS: u32 = 16;
    tracing::info!(blocks = %BMM_BLOCKS, "Attempting BMM");
    thunder_nodes
        .sender
        .bmm(&mut enforcer_post_setup, BMM_BLOCKS)
        .await?;
    // Check that sender has all blocks, and syncer has 0
    {
        let sender_blocks =
            thunder_nodes.sender.rpc_client.getblockcount().await?;
        anyhow::ensure!(sender_blocks == BMM_BLOCKS);
        let syncer_blocks =
            thunder_nodes.syncer.rpc_client.getblockcount().await?;
        anyhow::ensure!(syncer_blocks == 0);
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
    // Check that sender and syncer have all blocks
    {
        let sender_blocks =
            thunder_nodes.sender.rpc_client.getblockcount().await?;
        anyhow::ensure!(sender_blocks == BMM_BLOCKS);
        let syncer_blocks =
            thunder_nodes.syncer.rpc_client.getblockcount().await?;
        anyhow::ensure!(syncer_blocks == BMM_BLOCKS);
    }
    drop(thunder_nodes.syncer);
    drop(thunder_nodes.sender);
    tracing::info!("Removing {}", enforcer_post_setup.out_dir.path().display());
    drop(enforcer_post_setup.tasks);
    // Wait for tasks to die
    sleep(std::time::Duration::from_secs(1)).await;
    enforcer_post_setup.out_dir.cleanup()?;
    Ok(())
}

async fn ibd(bin_paths: BinPaths) -> anyhow::Result<()> {
    let (res_tx, mut res_rx) = mpsc::unbounded();
    let _test_task: AbortOnDrop<()> = tokio::task::spawn({
        let res_tx = res_tx.clone();
        async move {
            let res =
                initial_block_download_task(bin_paths, res_tx.clone()).await;
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
) -> AsyncTrial<BoxFuture<'static, anyhow::Result<()>>> {
    AsyncTrial::new("initial_block_download", ibd(bin_paths).boxed())
}
