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

/// Two thunder nodes against a single enforcer. `peer_init` may adjust the
/// first node's setup; `node_init` the second's, and is also passed the
/// first node's P2P port -- not known until it is set up.
async fn setup_with_args<P, F>(
    bin_paths: BinPaths,
    res_tx: mpsc::UnboundedSender<anyhow::Result<()>>,
    peer_init: P,
    node_init: F,
) -> anyhow::Result<(EnforcerPostSetup, PostSetup, PostSetup)>
where
    P: FnOnce(Init) -> Init,
    F: FnOnce(Init, u16) -> Init,
{
    let enforcer_pre_setup =
        EnforcerPreSetup::new(&bin_paths.others, Network::Regtest)?;
    let mut enforcer_post_setup = {
        let setup_opts: EnforcerSetupOpts = Default::default();
        enforcer_pre_setup
            .setup(Mode::Mempool, setup_opts, res_tx.clone())
            .await?
    };
    let peer = PostSetup::setup(
        peer_init(Init::new(bin_paths.thunder()?.clone(), Some("peer"))),
        &enforcer_post_setup,
        res_tx.clone(),
    )
    .await?;
    // Started second, so the first is listening when its seeds are dialed.
    let node = PostSetup::setup(
        node_init(
            Init::new(bin_paths.thunder()?.clone(), Some("node")),
            peer.net_port(),
        ),
        &enforcer_post_setup,
        res_tx,
    )
    .await?;
    let () = propose_sidechain::<PostSetup>(&mut enforcer_post_setup).await?;
    let () = activate_sidechain::<PostSetup>(&mut enforcer_post_setup).await?;
    let () = fund_enforcer::<PostSetup>(&mut enforcer_post_setup).await?;
    Ok((enforcer_post_setup, peer, node))
}

/// Initial setup: two thunder nodes against a single enforcer.
async fn setup(
    bin_paths: BinPaths,
    res_tx: mpsc::UnboundedSender<anyhow::Result<()>>,
) -> anyhow::Result<(EnforcerPostSetup, PostSetup, PostSetup)> {
    setup_with_args(bin_paths, res_tx, |init| init, |init, _| init).await
}

/// Active connections `node` holds to `port` on either loopback address.
async fn loopback_conns_to_port(
    node: &PostSetup,
    port: u16,
) -> anyhow::Result<Vec<SocketAddr>> {
    let conns = peer_addrs(node)
        .await?
        .into_iter()
        .filter(|addr| addr.ip().is_loopback() && addr.port() == port)
        .collect();
    Ok(conns)
}

/// A hostname seed is resolved on startup and dialed, with no `connect_peer`,
/// and a host that resolves to several addresses gets exactly one connection.
///
/// `localhost` resolves from the hosts file on every supported platform, so
/// this exercises the real resolver without needing network access or an
/// external zone. A numeric address would not distinguish resolution from
/// parsing. Both nodes bind dual-stack, so the peer is reachable at both of
/// `localhost`'s addresses; dialing every resolved address instead of trying
/// them one at a time would thus connect twice.
async fn seed_node_resolution_task(
    bin_paths: BinPaths,
    res_tx: mpsc::UnboundedSender<anyhow::Result<()>>,
) -> anyhow::Result<()> {
    // Loopback with nothing bound, reserved so nothing else can answer. Its
    // dials fail no matter which of `localhost`'s addresses is tried, so the
    // node must walk through all of them and report exhaustion.
    let dead_port = reserve_port::ReservedPort::random()?;
    let (enforcer_post_setup, peer, node) = setup_with_args(
        bin_paths,
        res_tx,
        // Bound dual-stack, so it is reachable at both of `localhost`'s
        // addresses
        |init| init.with_net_host("[::]"),
        // Also bound dual-stack: an IPv4-bound QUIC endpoint cannot dial
        // `::1` at all, which would leave only one usable address per host
        // and nothing for the single-connection check below to detect
        |init, peer_port| {
            init.with_net_host("[::]").with_extra_args([
                "--seed-node".to_owned(),
                format!("localhost:{peer_port}"),
                "--seed-node".to_owned(),
                format!("localhost:{}", dead_port.port()),
            ])
        },
    )
    .await?;
    let peer_port = peer.net_port();

    tracing::info!(
        %peer_port,
        "Waiting for the node to reach its seed node by hostname"
    );
    // Seed resolution runs in the background, so may not be done when RPC is
    // up. The connection may use either loopback address, depending on
    // resolution order, so it is awaited by port. Waiting on the validation
    // makes the connection count below final: once a connection to the seed
    // has validated, its other addresses are never dialed.
    let () = wait_for_log(
        &node,
        "the seed node to be validated",
        &["peer validated", &format!("localhost:{peer_port}")],
        WAIT_TIMEOUT,
    )
    .await?;
    // Give a second dial time to land, were one ever issued; the addresses
    // of a peer are dialed within moments of each other.
    sleep(Duration::from_secs(1)).await;
    let conns = loopback_conns_to_port(&node, peer_port).await?;
    anyhow::ensure!(
        conns.len() == 1,
        "the node should hold exactly one connection to its seed node, \
         found {conns:?}"
    );
    // A real peering, not just a half-open dial recorded locally.
    let peer_side = peer_addrs(&peer).await?;
    anyhow::ensure!(
        peer_side.len() == 1,
        "peer should have accepted exactly one connection from the node, \
         found {peer_side:?}"
    );

    // The dead seed's addresses are tried one at a time, each dial failing,
    // until none remain. Each attempt takes a full 30s handshake timeout:
    // nothing answers, and ICMP unreachable does not cancel a QUIC dial. The
    // dials began at node startup, so some of that has already passed, but
    // the wait must still cover most of two timeouts.
    const EXHAUSTION_TIMEOUT: Duration = Duration::from_secs(90);
    let () = wait_for_log(
        &node,
        "the dead seed node's addresses to be exhausted",
        &[
            "no more addresses to try",
            &format!("localhost:{}", dead_port.port()),
        ],
        EXHAUSTION_TIMEOUT,
    )
    .await?;

    drop(node);
    drop(peer);
    drop(dead_port);
    tracing::info!(
        "Removing {}",
        enforcer_post_setup.directories.base_dir.path().display()
    );
    drop(enforcer_post_setup.tasks);
    sleep(Duration::from_secs(1)).await;
    enforcer_post_setup.directories.base_dir.cleanup()?;
    Ok(())
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
    // Startup dials are issued by the net task in one batch, but only
    // near-simultaneously. Give it a moment, so that absence below means
    // "not persisted", not "not dialed yet".
    sleep(Duration::from_secs(1)).await;
    let addrs = peer_addrs(&node).await?;
    // Never sent a message, so never validated, so never persisted.
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

async fn seed_node_resolution(bin_paths: BinPaths) -> anyhow::Result<()> {
    let (res_tx, mut res_rx) = mpsc::unbounded();
    let _test_task: AbortOnDrop<()> = tokio::task::spawn({
        let res_tx = res_tx.clone();
        async move {
            let res =
                seed_node_resolution_task(bin_paths, res_tx.clone()).await;
            let _send_err: Result<(), _> = res_tx.unbounded_send(res);
        }
        .in_current_span()
    })
    .into();
    res_rx.next().await.ok_or_else(|| {
        anyhow::anyhow!("Unexpected end of test result stream")
    })?
}

pub fn seed_node_resolution_trial(
    bin_paths: BinPaths,
    file_registry: TestFileRegistry,
    failure_collector: TestFailureCollector,
) -> AsyncTrial<BoxFuture<'static, anyhow::Result<()>>> {
    AsyncTrial::new(
        "seed_node_resolution",
        seed_node_resolution(bin_paths).boxed(),
        file_registry,
        failure_collector,
    )
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
