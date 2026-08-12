//! Tests for which peers are written to the `known_peers` DB.
//!
//! A peer is only worth remembering once it has proven it is on our network by
//! sending a message bearing our magic bytes; the QUIC handshake proves nothing
//! on its own, since certificates are not verified.

use std::net::SocketAddr;

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
use thunder_app_rpc_api::node::{PrivateRpcClient as _, RpcClient as _};
use tokio::time::{Duration, sleep};
use tracing::Instrument as _;

use crate::{
    setup::{Init, PostSetup},
    util::BinPaths,
};

/// Upper bound on any single wait below. Generous, since it is only reached
/// when something is actually broken.
const WAIT_TIMEOUT: Duration = Duration::from_secs(60);

const POLL_INTERVAL: Duration = Duration::from_millis(100);

/// Addresses currently held in the node's active peer set.
async fn peer_addrs(node: &PostSetup) -> anyhow::Result<Vec<SocketAddr>> {
    let peers = node
        .rpc_client
        .list_peers()
        .await?
        .iter()
        .map(|peer| peer.address)
        .collect();
    Ok(peers)
}

/// Poll `node`'s active peer set until `addr` is present or absent, as asked.
///
/// Waiting on the condition itself keeps the tests fast when things go well,
/// and makes them fail naming what was awaited rather than tripping some
/// downstream assertion when they do not.
async fn wait_for_peer_presence(
    node: &PostSetup,
    addr: SocketAddr,
    want_present: bool,
) -> anyhow::Result<()> {
    let deadline = tokio::time::Instant::now() + WAIT_TIMEOUT;
    loop {
        let addrs = peer_addrs(node).await?;
        if addrs.contains(&addr) == want_present {
            return Ok(());
        }
        anyhow::ensure!(
            tokio::time::Instant::now() < deadline,
            "timed out after {WAIT_TIMEOUT:?} waiting for {addr} to be \
             {} the peer set, found {addrs:?}",
            if want_present { "in" } else { "absent from" },
        );
        sleep(POLL_INTERVAL).await;
    }
}

async fn wait_for_peer(
    node: &PostSetup,
    addr: SocketAddr,
) -> anyhow::Result<()> {
    wait_for_peer_presence(node, addr, true).await
}

async fn wait_for_no_peer(
    node: &PostSetup,
    addr: SocketAddr,
) -> anyhow::Result<()> {
    wait_for_peer_presence(node, addr, false).await
}

/// Poll the node's stdout log until a line containing all of `needles`
/// appears, for at most `timeout`.
///
/// Only for states no RPC reports; `list_peers` answers the question wherever
/// it can.
async fn wait_for_log(
    node: &PostSetup,
    what: &str,
    needles: &[&str],
    timeout: Duration,
) -> anyhow::Result<()> {
    let path = node.thunder_app.data_dir.join("stdout.txt");
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        match std::fs::read_to_string(&path) {
            Ok(contents) => {
                if contents.lines().any(|line| {
                    needles.iter().all(|needle| line.contains(needle))
                }) {
                    return Ok(());
                }
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => return Err(err.into()),
        }
        anyhow::ensure!(
            tokio::time::Instant::now() < deadline,
            "timed out after {timeout:?} waiting for {what} in {}",
            path.display()
        );
        sleep(POLL_INTERVAL).await;
    }
}

/// Two thunder nodes against a single enforcer.
async fn setup(
    bin_paths: BinPaths,
    res_tx: mpsc::UnboundedSender<anyhow::Result<()>>,
) -> anyhow::Result<(EnforcerPostSetup, PostSetup, PostSetup)> {
    let enforcer_pre_setup =
        EnforcerPreSetup::new(&bin_paths.others, Network::Regtest)?;
    let mut enforcer_post_setup = {
        let setup_opts: EnforcerSetupOpts = Default::default();
        enforcer_pre_setup
            .setup(Mode::Mempool, setup_opts, res_tx.clone())
            .await?
    };
    let peer = PostSetup::setup(
        Init {
            thunder_app: bin_paths.thunder()?.clone(),
            data_dir_suffix: Some("peer".to_owned()),
        },
        &enforcer_post_setup,
        res_tx.clone(),
    )
    .await?;
    let node = PostSetup::setup(
        Init {
            thunder_app: bin_paths.thunder()?.clone(),
            data_dir_suffix: Some("node".to_owned()),
        },
        &enforcer_post_setup,
        res_tx,
    )
    .await?;
    let () = propose_sidechain::<PostSetup>(&mut enforcer_post_setup).await?;
    let () = activate_sidechain::<PostSetup>(&mut enforcer_post_setup).await?;
    let () = fund_enforcer::<PostSetup>(&mut enforcer_post_setup).await?;
    Ok((enforcer_post_setup, peer, node))
}

/// A peer that completes a handshake is remembered across a restart; one that
/// never answers is not. Both are checked against the same restart, so the
/// unreachable one is held to the same bar as the reachable one.
async fn peer_persistence_task(
    bin_paths: BinPaths,
    res_tx: mpsc::UnboundedSender<anyhow::Result<()>>,
) -> anyhow::Result<()> {
    let (enforcer_post_setup, peer, mut node) =
        setup(bin_paths, res_tx.clone()).await?;
    let peer_addr: SocketAddr = peer.net_addr().into();

    // Loopback with nothing bound. Reserved so nothing else can answer.
    let unreachable_port = reserve_port::ReservedPort::random()?;
    let unreachable_addr: SocketAddr = std::net::SocketAddrV4::new(
        std::net::Ipv4Addr::LOCALHOST,
        unreachable_port.port(),
    )
    .into();
    anyhow::ensure!(
        peer_addr != unreachable_addr,
        "unreachable address must not collide with the live peer"
    );

    tracing::info!(%peer_addr, %unreachable_addr, "Connecting to both peers");
    // Not an error: the dial only fails later, when the handshake times out.
    let () = node.rpc_client.connect_peer(unreachable_addr).await?;
    let () = node.rpc_client.connect_peer(peer_addr).await?;

    // Both are in the active set, having both been dialed -- which is why the
    // restart below, not this, is the real check. Wait for the live peer to be
    // validated, since that, not the dial, is what writes it to `known_peers`;
    // killing the node any earlier would race that write. No RPC reports it.
    let () = wait_for_log(
        &node,
        &format!("{peer_addr} to be validated"),
        &["peer validated", &peer_addr.to_string()],
        WAIT_TIMEOUT,
    )
    .await?;
    let () = wait_for_peer(&node, peer_addr).await?;

    // Let the peer notice before bringing the node back: until its heartbeat
    // times out it refuses the reconnection as a duplicate, which a startup
    // dial does not retry. Orthogonal to what is under test.
    tracing::info!("Restarting node");
    let () = node.kill();
    let node_addr: SocketAddr = node.net_addr().into();
    let () = wait_for_no_peer(&peer, node_addr).await?;
    let () = node.start(res_tx).await?;

    // Validated before the restart, so written to `known_peers` and re-dialed.
    let () = wait_for_peer(&node, peer_addr).await?;
    let addrs = peer_addrs(&node).await?;
    // Never sent a message, so never validated, so never persisted. Both dials
    // are issued by `Net::new` before the RPC server answers, so had it been
    // persisted it would already be listed alongside the live peer.
    anyhow::ensure!(
        !addrs.contains(&unreachable_addr),
        "unreachable peer at {unreachable_addr} was never validated and must \
         not survive a restart, found {addrs:?}"
    );

    drop(node);
    drop(peer);
    drop(unreachable_port);
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

async fn peer_persistence(bin_paths: BinPaths) -> anyhow::Result<()> {
    let (res_tx, mut res_rx) = mpsc::unbounded();
    let _test_task: AbortOnDrop<()> = tokio::task::spawn({
        let res_tx = res_tx.clone();
        async move {
            let res = peer_persistence_task(bin_paths, res_tx.clone()).await;
            let _send_err: Result<(), _> = res_tx.unbounded_send(res);
        }
        .in_current_span()
    })
    .into();
    res_rx.next().await.ok_or_else(|| {
        anyhow::anyhow!("Unexpected end of test result stream")
    })?
}

pub fn peer_persistence_trial(
    bin_paths: BinPaths,
    file_registry: TestFileRegistry,
    failure_collector: TestFailureCollector,
) -> AsyncTrial<BoxFuture<'static, anyhow::Result<()>>> {
    AsyncTrial::new(
        "peer_persistence",
        peer_persistence(bin_paths).boxed(),
        file_registry,
        failure_collector,
    )
}
