//! A current node refuses peers older than [`MIN_PEER_VERSION`], in both
//! directions, using a real 0.18 release binary as the old peer

use std::{
    net::SocketAddr,
    time::{Duration, Instant},
};

use bip300301_enforcer_integration_tests::{
    integration_test::{activate_sidechain, fund_enforcer, propose_sidechain},
    setup::{
        Mode, Network, PostSetup as EnforcerPostSetup,
        PreSetup as EnforcerPreSetup, SetupOpts as EnforcerSetupOpts,
        Sidechain as _,
    },
    util::{AbortOnDrop, AsyncTrial, TestFailureCollector, TestFileRegistry},
};
use futures::{FutureExt, StreamExt as _, channel::mpsc, future::BoxFuture};
use thunder::net::MIN_PEER_VERSION;
use thunder_app_rpc_api::node::{PrivateRpcClient as _, RpcClient as _};
use tokio::time::sleep;
use tracing::Instrument as _;

use crate::{
    setup::{Init, PostSetup},
    util::BinPaths,
};

/// Blocks the old node BMMs, so that it has a tip the current node would
/// otherwise sync
const OLD_BLOCKS: u32 = 3;

/// Logged by the current node each time it refuses a peer
const REFUSAL_LOG: &str = "older than the minimum compatible version";

/// Long enough for several heartbeats, and for a reconnect after the base
/// reconnect delay of 10s
const OBSERVE: Duration = Duration::from_secs(15);

struct Nodes {
    current: PostSetup,
    old: PostSetup,
}

async fn setup(
    bin_paths: &BinPaths,
    res_tx: mpsc::UnboundedSender<anyhow::Result<()>>,
) -> anyhow::Result<(EnforcerPostSetup, Nodes)> {
    let enforcer_pre_setup =
        EnforcerPreSetup::new(&bin_paths.others, Network::Regtest)?;
    let setup_opts: EnforcerSetupOpts = Default::default();
    let mut enforcer_post_setup = enforcer_pre_setup
        .setup(Mode::Mempool, setup_opts, res_tx.clone())
        .await?;
    let current = PostSetup::setup(
        Init {
            data_dir_suffix: Some("current".to_owned()),
            ..Init::new(bin_paths.thunder()?.clone())
        },
        &enforcer_post_setup,
        res_tx.clone(),
    )
    .await?;
    let old = PostSetup::setup(
        Init {
            data_dir_suffix: Some("old".to_owned()),
            ..Init::new(bin_paths.thunder_0_18()?.clone())
        },
        &enforcer_post_setup,
        res_tx,
    )
    .await?;
    tracing::info!("Started current and old thunder nodes");
    let () = propose_sidechain::<PostSetup>(&mut enforcer_post_setup).await?;
    let () = activate_sidechain::<PostSetup>(&mut enforcer_post_setup).await?;
    let () = fund_enforcer::<PostSetup>(&mut enforcer_post_setup).await?;
    Ok((enforcer_post_setup, Nodes { current, old }))
}

/// Number of times the node has logged refusing a peer
fn refusals(node: &PostSetup) -> anyhow::Result<usize> {
    let stdout = std::fs::read_to_string(node.data_dir.join("stdout.txt"))?;
    Ok(stdout.matches(REFUSAL_LOG).count())
}

/// Wait until the node has logged more than `seen` refusals
async fn wait_for_refusal(node: &PostSetup, seen: usize) -> anyhow::Result<()> {
    let deadline = Instant::now() + OBSERVE;
    while refusals(node)? <= seen {
        anyhow::ensure!(
            Instant::now() < deadline,
            "no peer refusal logged within {OBSERVE:?}"
        );
        sleep(Duration::from_millis(500)).await;
    }
    Ok(())
}

async fn is_connected(
    node: &PostSetup,
    peer: SocketAddr,
) -> anyhow::Result<bool> {
    let peers = node.rpc_client.list_peers().await?;
    Ok(peers.iter().any(|p| p.address == peer))
}

/// The current node must not have synced anything from the old node
async fn ensure_not_synced(current: &PostSetup) -> anyhow::Result<()> {
    let blocks = current.rpc_client.getblockcount().await?;
    anyhow::ensure!(
        blocks == 0,
        "current node synced {blocks} blocks from an old peer"
    );
    Ok(())
}

async fn refuse_old_peer_task(
    bin_paths: BinPaths,
    res_tx: mpsc::UnboundedSender<anyhow::Result<()>>,
) -> anyhow::Result<()> {
    let (mut enforcer_post_setup, nodes) = setup(&bin_paths, res_tx).await?;
    let old_addr: SocketAddr = nodes.old.net_addr().into();
    let current_addr: SocketAddr = nodes.current.net_addr().into();
    tracing::info!(blocks = OLD_BLOCKS, "Old node: BMM");
    let () = nodes.old.bmm(&mut enforcer_post_setup, OLD_BLOCKS).await?;
    let old_blocks = nodes.old.rpc_client.getblockcount().await?;
    anyhow::ensure!(old_blocks == OLD_BLOCKS);

    tracing::info!(%MIN_PEER_VERSION, "Current node dials old node");
    let () = nodes
        .current
        .rpc_client
        .connect_peer(old_addr.into())
        .await?;
    let () = wait_for_refusal(&nodes.current, 0).await?;
    let seen = refusals(&nodes.current)?;
    // An outbound peer that was refused is not dialed again
    sleep(OBSERVE).await;
    anyhow::ensure!(
        refusals(&nodes.current)? == seen,
        "current node redialed the old node after refusing it"
    );
    anyhow::ensure!(
        !is_connected(&nodes.current, old_addr).await?,
        "current node still connected to the old node"
    );
    let () = ensure_not_synced(&nodes.current).await?;

    tracing::info!("Old node dials current node");
    let () = nodes
        .old
        .rpc_client
        .connect_peer(current_addr.into())
        .await?;
    let () = wait_for_refusal(&nodes.current, seen).await?;
    // The old node keeps redialing, but each connection is refused
    sleep(OBSERVE).await;
    let () = ensure_not_synced(&nodes.current).await?;

    drop(nodes);
    tracing::info!(
        "Removing {}",
        enforcer_post_setup.directories.base_dir.path().display()
    );
    drop(enforcer_post_setup.tasks);
    // Wait for tasks to die
    sleep(Duration::from_secs(1)).await;
    enforcer_post_setup.directories.base_dir.cleanup()?;
    Ok(())
}

async fn refuse_old_peer(bin_paths: BinPaths) -> anyhow::Result<()> {
    let (res_tx, mut res_rx) = mpsc::unbounded();
    let _test_task: AbortOnDrop<()> = tokio::task::spawn({
        let res_tx = res_tx.clone();
        async move {
            let res = refuse_old_peer_task(bin_paths, res_tx.clone()).await;
            let _send_err: Result<(), _> = res_tx.unbounded_send(res);
        }
        .in_current_span()
    })
    .into();
    res_rx.next().await.ok_or_else(|| {
        anyhow::anyhow!("Unexpected end of test task result stream")
    })?
}

pub fn refuse_old_peer_trial(
    bin_paths: BinPaths,
    file_registry: TestFileRegistry,
    failure_collector: TestFailureCollector,
) -> AsyncTrial<BoxFuture<'static, anyhow::Result<()>>> {
    AsyncTrial::new(
        "refuse_old_peer",
        refuse_old_peer(bin_paths).boxed(),
        file_registry,
        failure_collector,
    )
}
