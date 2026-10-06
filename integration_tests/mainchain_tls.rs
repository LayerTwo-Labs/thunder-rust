//! Test following the enforcer over TLS: an `https` mainchain gRPC URL,
//! verified against a CA the node is configured to trust

use std::{net::SocketAddr, path::Path, sync::Arc, time::Duration};

use bip300301_enforcer_integration_tests::{
    integration_test::{activate_sidechain, fund_enforcer, propose_sidechain},
    setup::{
        Mode, Network, PreSetup as EnforcerPreSetup,
        SetupOpts as EnforcerSetupOpts, Sidechain as _,
    },
    util::{AbortOnDrop, AsyncTrial, TestFailureCollector, TestFileRegistry},
};
use futures::{
    FutureExt as _, StreamExt as _, channel::mpsc, future::BoxFuture,
};
use reserve_port::ReservedPort;
use thunder_app_rpc_api::node::RpcClient as _;
use tokio::time::sleep;
use tokio_rustls::{TlsAcceptor, rustls};
use tracing::Instrument as _;

use crate::{
    setup::{Init, MainchainTls, PostSetup},
    util::{BinPaths, ThunderApp},
};

/// A fresh CA (returned as PEM), and a TLS acceptor serving a certificate for
/// `localhost` signed by it, over HTTP/2 only
fn test_pki() -> anyhow::Result<(String, TlsAcceptor)> {
    let ca_key = rcgen::KeyPair::generate()?;
    let mut ca_params = rcgen::CertificateParams::new(Vec::<String>::new())?;
    ca_params.is_ca = rcgen::IsCa::Ca(rcgen::BasicConstraints::Unconstrained);
    let ca_cert = ca_params.self_signed(&ca_key)?;
    let key = rcgen::KeyPair::generate()?;
    let cert = rcgen::CertificateParams::new(vec!["localhost".to_owned()])?
        .signed_by(&key, &ca_cert, &ca_key)?;
    let mut config = rustls::ServerConfig::builder_with_provider(Arc::new(
        rustls::crypto::ring::default_provider(),
    ))
    .with_safe_default_protocol_versions()?
    .with_no_client_auth()
    .with_single_cert(
        vec![cert.der().clone()],
        rustls::pki_types::PrivatePkcs8KeyDer::from(key.serialize_der()).into(),
    )?;
    config.alpn_protocols = vec![b"h2".to_vec()];
    Ok((ca_cert.pem(), TlsAcceptor::from(Arc::new(config))))
}

/// Terminates TLS and forwards the plain h2c stream to `upstream`, like a
/// TLS-terminating proxy in front of a remote enforcer
async fn spawn_tls_proxy(
    acceptor: TlsAcceptor,
    upstream: SocketAddr,
) -> anyhow::Result<(u16, AbortOnDrop<()>)> {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let port = listener.local_addr()?.port();
    let task = tokio::spawn(async move {
        while let Ok((inbound, _)) = listener.accept().await {
            let acceptor = acceptor.clone();
            tokio::spawn(async move {
                let (Ok(mut inbound), Ok(mut outbound)) = (
                    acceptor.accept(inbound).await,
                    tokio::net::TcpStream::connect(upstream).await,
                ) else {
                    return;
                };
                let _res =
                    tokio::io::copy_bidirectional(&mut inbound, &mut outbound)
                        .await;
            });
        }
    });
    Ok((port, task.into()))
}

/// Starts a node that must exit at startup, with an error containing
/// `expected`, ie. for that reason and not another one
async fn expect_startup_failure(
    bin_paths: &BinPaths,
    data_dir: &Path,
    mainchain_grpc_url: String,
    mainchain_grpc_ca_cert: Option<&Path>,
    expected: &str,
) -> anyhow::Result<()> {
    std::fs::create_dir(data_dir)?;
    let (net, rpc) = (ReservedPort::random()?, ReservedPort::random()?);
    let thunder_app = ThunderApp {
        path: bin_paths.thunder()?.clone(),
        data_dir: data_dir.to_owned(),
        log_level: Some(tracing::Level::DEBUG),
        mainchain_grpc_url,
        mainchain_grpc_ca_cert: mainchain_grpc_ca_cert.map(Path::to_owned),
        net_port: net.port(),
        network: thunder::types::Network::Regtest,
        rpc_port: rpc.port(),
    };
    let (exit_tx, exit_rx) = futures::channel::oneshot::channel();
    let _task = thunder_app.spawn_command_with_args::<String, String, _, _, _>(
        [],
        [],
        move |err| {
            let _send: Result<(), _> = exit_tx.send(err);
        },
    );
    let err = tokio::time::timeout(Duration::from_secs(60), exit_rx)
        .await
        .map_err(|_| {
            anyhow::anyhow!("node started; expected `{expected}`")
        })??;
    anyhow::ensure!(
        format!("{err:#}").contains(expected),
        "node exited, but not with `{expected}`: {err:#}"
    );
    Ok(())
}

async fn mainchain_tls_task(
    bin_paths: BinPaths,
    res_tx: mpsc::UnboundedSender<anyhow::Result<()>>,
) -> anyhow::Result<()> {
    let setup_opts: EnforcerSetupOpts = Default::default();
    let mut enforcer_post_setup =
        EnforcerPreSetup::new(&bin_paths.others, Network::Regtest)?
            .setup(Mode::Mempool, setup_opts, res_tx.clone())
            .await?;
    let () = propose_sidechain::<PostSetup>(&mut enforcer_post_setup).await?;
    let () = activate_sidechain::<PostSetup>(&mut enforcer_post_setup).await?;
    let () = fund_enforcer::<PostSetup>(&mut enforcer_post_setup).await?;
    let enforcer_port = enforcer_post_setup
        .reserved_ports
        .enforcer_serve_grpc
        .port();
    let base_dir = enforcer_post_setup.directories.base_dir.path().to_owned();
    let (ca_cert_pem, acceptor) = test_pki()?;
    let ca_cert = base_dir.join("mainchain-ca.pem");
    std::fs::write(&ca_cert, ca_cert_pem)?;
    let (proxy_port, _proxy) = spawn_tls_proxy(
        acceptor,
        SocketAddr::from(([127, 0, 0, 1], enforcer_port)),
    )
    .await?;
    let url = format!("https://localhost:{proxy_port}");

    tracing::info!("Checking that certificates are verified");
    let () = expect_startup_failure(
        &bin_paths,
        &base_dir.join("thunder-untrusting"),
        url.clone(),
        None,
        "UnknownIssuer",
    )
    .await?;
    // The certificate is for `localhost`
    let () = expect_startup_failure(
        &bin_paths,
        &base_dir.join("thunder-wrong-name"),
        format!("https://127.0.0.1:{proxy_port}"),
        Some(&ca_cert),
        "NotValidForName",
    )
    .await?;
    let () = expect_startup_failure(
        &bin_paths,
        &base_dir.join("thunder-plain"),
        format!("http://127.0.0.1:{enforcer_port}"),
        Some(&ca_cert),
        "needs an https",
    )
    .await?;

    tracing::info!("Following the enforcer over TLS");
    let sidechain = PostSetup::setup(
        Init {
            mainchain_tls: Some(MainchainTls { url, ca_cert }),
            ..Init::new(bin_paths.thunder()?.clone())
        },
        &enforcer_post_setup,
        res_tx,
    )
    .await?;
    // BMM needs the mainchain tip and block events over the TLS connection
    let () = sidechain.bmm(&mut enforcer_post_setup, 2).await?;
    anyhow::ensure!(sidechain.rpc_client.getblockcount().await? == 2);

    drop(sidechain);
    drop(enforcer_post_setup.tasks);
    // Wait for tasks to die
    sleep(Duration::from_secs(1)).await;
    enforcer_post_setup.directories.base_dir.cleanup()?;
    Ok(())
}

async fn mainchain_tls(bin_paths: BinPaths) -> anyhow::Result<()> {
    let (res_tx, mut res_rx) = mpsc::unbounded();
    let _test_task: AbortOnDrop<()> = tokio::task::spawn({
        let res_tx = res_tx.clone();
        async move {
            let res = mainchain_tls_task(bin_paths, res_tx.clone()).await;
            let _send_err: Result<(), _> = res_tx.unbounded_send(res);
        }
        .in_current_span()
    })
    .into();
    res_rx.next().await.ok_or_else(|| {
        anyhow::anyhow!("Unexpected end of test task result stream")
    })?
}

pub fn mainchain_tls_trial(
    bin_paths: BinPaths,
    file_registry: TestFileRegistry,
    failure_collector: TestFailureCollector,
) -> AsyncTrial<BoxFuture<'static, anyhow::Result<()>>> {
    AsyncTrial::new(
        "mainchain_tls",
        mainchain_tls(bin_paths).boxed(),
        file_registry,
        failure_collector,
    )
}
