//! A client that abandons its handshake must not keep other peers out.

use std::{
    net::{Ipv4Addr, SocketAddr},
    sync::Arc,
    time::Duration,
};

use bip300301_enforcer_integration_tests::{
    setup::{
        Mode, Network, PreSetup as EnforcerPreSetup,
        SetupOpts as EnforcerSetupOpts, Sidechain as _,
    },
    util::{AbortOnDrop, AsyncTrial, TestFailureCollector, TestFileRegistry},
};
use futures::{FutureExt, StreamExt as _, channel::mpsc, future::BoxFuture};
use thunder::types::net::PeerConnectionStatus;
use thunder_app_rpc_api::node::{PrivateRpcClient as _, RpcClient as _};
use tokio::{
    net::UdpSocket,
    time::{Instant, sleep, timeout},
};
use tracing::Instrument as _;

use crate::{
    setup::{Init, PostSetup},
    util::BinPaths,
};

/// Each one holds a handshake open until the server gives up on it.
const STALLED_HANDSHAKES: usize = 3;

/// Less than the stalled handshakes would hold up a node that completes one
/// handshake at a time, even with handshakes cut short at 10s.
const CONNECT_DEADLINE: Duration = Duration::from_secs(15);

fn quic_client() -> anyhow::Result<quinn::Endpoint> {
    // The handshake never gets as far as checking the server's certificate.
    let crypto = rustls::ClientConfig::builder_with_provider(Arc::new(
        rustls::crypto::ring::default_provider(),
    ))
    .with_safe_default_protocol_versions()?
    .with_root_certificates(rustls::RootCertStore::empty())
    .with_no_client_auth();
    let mut endpoint =
        quinn::Endpoint::client((Ipv4Addr::LOCALHOST, 0).into())?;
    endpoint.set_default_client_config(quinn::ClientConfig::new(Arc::new(
        quinn::crypto::rustls::QuicClientConfig::try_from(crypto)?,
    )));
    Ok(endpoint)
}

/// Start a handshake with `server` and never answer it: a relay passes the
/// client's Initial packet on, then drops everything the server sends back.
/// Returns the relay, which must be kept alive while the handshake stalls.
async fn stall_handshake(server: SocketAddr) -> anyhow::Result<UdpSocket> {
    let relay = UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).await?;
    let client = quic_client()?;
    let _connecting = client.connect(relay.local_addr()?, "localhost")?;
    let mut initial = vec![0; 65_536];
    let (len, _) =
        timeout(Duration::from_secs(5), relay.recv_from(&mut initial))
            .await??;
    relay.send_to(&initial[..len], server).await?;
    Ok(relay)
}

async fn connected_to(
    node: &PostSetup,
    peer: SocketAddr,
) -> anyhow::Result<bool> {
    Ok(node.rpc_client.list_peers().await?.iter().any(|p| {
        p.address == peer && p.status == PeerConnectionStatus::Connected
    }))
}

async fn stalled_handshake_task(
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
    let server = PostSetup::setup(
        Init {
            data_dir_suffix: Some("server".to_owned()),
            ..Init::new(bin_paths.thunder()?.clone())
        },
        &enforcer_post_setup,
        res_tx.clone(),
    )
    .await?;
    let dialer = PostSetup::setup(
        Init {
            data_dir_suffix: Some("dialer".to_owned()),
            ..Init::new(bin_paths.thunder()?.clone())
        },
        &enforcer_post_setup,
        res_tx,
    )
    .await?;
    let server_addr: SocketAddr = server.net_addr().into();
    let mut relays = Vec::with_capacity(STALLED_HANDSHAKES);
    for _ in 0..STALLED_HANDSHAKES {
        relays.push(stall_handshake(server_addr).await?);
    }
    tracing::info!(
        stalled = STALLED_HANDSHAKES,
        "Stalled handshakes queued, connecting dialer"
    );
    let () = dialer.rpc_client.connect_peer(server_addr.into()).await?;
    let started = Instant::now();
    let connected = loop {
        if connected_to(&dialer, server_addr).await? {
            break true;
        }
        if started.elapsed() > CONNECT_DEADLINE {
            break false;
        }
        sleep(Duration::from_millis(500)).await;
    };
    let elapsed = started.elapsed();
    let peers: Vec<_> = dialer
        .rpc_client
        .list_peers()
        .await?
        .into_iter()
        .map(|p| (p.address, p.status == PeerConnectionStatus::Connected))
        .collect();
    drop(relays);
    drop(dialer);
    drop(server);
    drop(enforcer_post_setup.tasks);
    sleep(Duration::from_secs(1)).await;
    enforcer_post_setup.directories.base_dir.cleanup()?;
    anyhow::ensure!(
        connected,
        "dialer could not connect within {}s while {STALLED_HANDSHAKES} \
         handshakes were stalled (peers: {peers:?})",
        CONNECT_DEADLINE.as_secs(),
    );
    tracing::info!(?elapsed, "Dialer connected despite stalled handshakes");
    Ok(())
}

async fn stalled_handshake(bin_paths: BinPaths) -> anyhow::Result<()> {
    let (res_tx, mut res_rx) = mpsc::unbounded();
    let _test_task: AbortOnDrop<()> = tokio::task::spawn({
        let res_tx = res_tx.clone();
        async move {
            let res = stalled_handshake_task(bin_paths, res_tx.clone()).await;
            let _send_err: Result<(), _> = res_tx.unbounded_send(res);
        }
        .in_current_span()
    })
    .into();
    res_rx.next().await.ok_or_else(|| {
        anyhow::anyhow!("Unexpected end of test task result stream")
    })?
}

pub fn stalled_handshake_trial(
    bin_paths: BinPaths,
    file_registry: TestFileRegistry,
    failure_collector: TestFailureCollector,
) -> AsyncTrial<BoxFuture<'static, anyhow::Result<()>>> {
    AsyncTrial::new(
        "stalled_handshake",
        stalled_handshake(bin_paths).boxed(),
        file_registry,
        failure_collector,
    )
}
