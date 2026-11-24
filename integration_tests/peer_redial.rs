//! A node must keep dialing a seed peer that did not answer at startup.

use std::time::Duration;

use bip300301_enforcer_integration_tests::{
    setup::{
        Mode, Network, PreSetup as EnforcerPreSetup,
        SetupOpts as EnforcerSetupOpts, Sidechain as _,
    },
    util::{AbortOnDrop, AsyncTrial, TestFailureCollector, TestFileRegistry},
};
use futures::{FutureExt, StreamExt as _, channel::mpsc, future::BoxFuture};
use thunder::types::net::PeerConnectionStatus;
use thunder_app_rpc_api::node::RpcClient as _;
use tokio::time::{Instant, sleep};
use tracing::Instrument as _;

use crate::{
    setup::{Init, PostSetup, ReservedPorts},
    util::BinPaths,
};

/// Longer than a QUIC handshake to an unreachable address takes to fail
/// (quinn's default idle timeout, 30s).
const FIRST_DIAL_FAILED_AFTER: Duration = Duration::from_secs(40);

/// How long the dialer gets to reach the seed once the seed is up.
const REDIAL_DEADLINE: Duration = Duration::from_secs(120);

async fn connected_to(
    node: &PostSetup,
    peer: std::net::SocketAddr,
) -> anyhow::Result<bool> {
    Ok(node.rpc_client.list_peers().await?.iter().any(|p| {
        p.address == peer && p.status == PeerConnectionStatus::Connected
    }))
}

async fn redial_unanswered_seed_task(
    bin_paths: BinPaths,
    res_tx: mpsc::UnboundedSender<anyhow::Result<()>>,
) -> anyhow::Result<()> {
    let enforcer_pre_setup =
        EnforcerPreSetup::new(&bin_paths.others, Network::Regtest)?;
    let enforcer_post_setup = {
        let setup_opts: EnforcerSetupOpts = Default::default();
        enforcer_pre_setup
            .setup(Mode::Mempool, setup_opts, res_tx.clone())
            .await?
    };
    // The seed's address is fixed now, but nothing listens on it yet.
    let seed_ports = ReservedPorts::new()?;
    let seed_addr = std::net::SocketAddrV4::new(
        std::net::Ipv4Addr::LOCALHOST,
        seed_ports.net.port(),
    );
    tracing::info!(%seed_addr, "Starting dialer with an unreachable seed");
    let dialer = PostSetup::setup(
        Init {
            data_dir_suffix: Some("dialer".to_owned()),
            add_peers: vec![seed_addr],
            ..Init::new(bin_paths.thunder()?.clone())
        },
        &enforcer_post_setup,
        res_tx.clone(),
    )
    .await?;
    let dialing = dialer.rpc_client.list_peers().await?;
    anyhow::ensure!(
        dialing.iter().any(|p| p.address == seed_addr.into()
            && p.status == PeerConnectionStatus::Connecting),
        "dialer is not dialing {seed_addr} at startup (peers: {:?})",
        dialing.iter().map(|p| p.address).collect::<Vec<_>>(),
    );
    // By now the first dial has timed out. A redial may already be under
    // way, so the seed can be listed again, but only as connecting.
    sleep(FIRST_DIAL_FAILED_AFTER).await;
    anyhow::ensure!(
        !connected_to(&dialer, seed_addr.into()).await?,
        "dialer connected to {seed_addr} before the seed was started"
    );
    tracing::info!("Starting seed");
    let seed = PostSetup::setup(
        Init {
            data_dir_suffix: Some("seed".to_owned()),
            reserved_ports: Some(seed_ports),
            ..Init::new(bin_paths.thunder()?.clone())
        },
        &enforcer_post_setup,
        res_tx,
    )
    .await?;
    let seed_up = Instant::now();
    let connected = loop {
        if connected_to(&dialer, seed_addr.into()).await? {
            break true;
        }
        if seed_up.elapsed() > REDIAL_DEADLINE {
            break false;
        }
        sleep(Duration::from_secs(2)).await;
    };
    let peers = dialer.rpc_client.list_peers().await?;
    drop(seed);
    drop(dialer);
    drop(enforcer_post_setup.tasks);
    sleep(Duration::from_secs(1)).await;
    enforcer_post_setup.directories.base_dir.cleanup()?;
    anyhow::ensure!(
        connected,
        "dialer never redialed its seed {seed_addr}: not connected {}s after \
         the seed came up (peers: {:?})",
        REDIAL_DEADLINE.as_secs(),
        peers.iter().map(|p| p.address).collect::<Vec<_>>(),
    );
    Ok(())
}

async fn redial_unanswered_seed(bin_paths: BinPaths) -> anyhow::Result<()> {
    let (res_tx, mut res_rx) = mpsc::unbounded();
    let _test_task: AbortOnDrop<()> = tokio::task::spawn({
        let res_tx = res_tx.clone();
        async move {
            let res =
                redial_unanswered_seed_task(bin_paths, res_tx.clone()).await;
            let _send_err: Result<(), _> = res_tx.unbounded_send(res);
        }
        .in_current_span()
    })
    .into();
    res_rx.next().await.ok_or_else(|| {
        anyhow::anyhow!("Unexpected end of test task result stream")
    })?
}

pub fn redial_unanswered_seed_trial(
    bin_paths: BinPaths,
    file_registry: TestFileRegistry,
    failure_collector: TestFailureCollector,
) -> AsyncTrial<BoxFuture<'static, anyhow::Result<()>>> {
    AsyncTrial::new(
        "redial_unanswered_seed",
        redial_unanswered_seed(bin_paths).boxed(),
        file_registry,
        failure_collector,
    )
}
