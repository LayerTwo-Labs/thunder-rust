use std::{
    collections::{HashMap, HashSet, hash_map},
    net::{IpAddr, Ipv4Addr, SocketAddr},
    sync::Arc,
};

use fallible_iterator::FallibleIterator;
use futures::{StreamExt, channel::mpsc};
use heed::types::{SerdeBincode, Unit};
use parking_lot::RwLock;
use quinn::{ClientConfig, Endpoint, ServerConfig};
use sneed::{
    DatabaseUnique, Env, EnvError, RwTxn, RwTxnError, UnitKey,
    db::error::Error as DbError,
};
use tokio_stream::StreamNotifyClose;
use tracing::instrument;

use crate::{
    archive::Archive,
    state::State,
    types::{
        AuthorizedTransaction, Network, VERSION, Version,
        net::{
            DEFAULT_PORT, Peer, PeerAddress, PeerConnectionStatus,
            ResolvedPeerAddress,
        },
    },
    util::ErrorChain,
};

pub mod error;
mod peer;

pub use error::Error;
use peer::{
    Connection, ConnectionContext as PeerConnectionCtxt,
    ConnectionHandle as PeerConnectionHandle,
};
pub use peer::{
    ConnectionError as PeerConnectionError, Info as PeerConnectionInfo,
    InternalMessage as PeerConnectionMessage, PeerStateId,
    Request as PeerRequest, ResponseMessage as PeerResponse,
    message as peer_message,
};

/// Dummy certificate verifier that treats any certificate as valid.
/// NOTE, such verification is vulnerable to MITM attacks, but convenient for testing.
#[derive(Debug)]
struct SkipServerVerification;

impl SkipServerVerification {
    fn new() -> Arc<Self> {
        Arc::new(Self)
    }
}

impl rustls::client::danger::ServerCertVerifier for SkipServerVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &rustls::pki_types::CertificateDer,
        _intermediates: &[rustls::pki_types::CertificateDer],
        _server_name: &rustls::pki_types::ServerName,
        _ocsp_response: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &rustls::pki_types::CertificateDer<'_>,
        dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error>
    {
        rustls::crypto::verify_tls12_signature(
            message,
            cert,
            dss,
            &rustls::crypto::ring::default_provider()
                .signature_verification_algorithms,
        )
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &rustls::pki_types::CertificateDer<'_>,
        dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error>
    {
        rustls::crypto::verify_tls13_signature(
            message,
            cert,
            dss,
            &rustls::crypto::ring::default_provider()
                .signature_verification_algorithms,
        )
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        rustls::crypto::ring::default_provider()
            .signature_verification_algorithms
            .supported_schemes()
    }
}

fn configure_client() -> Result<ClientConfig, error::ConfigureClient> {
    let crypto_provider = Arc::new(rustls::crypto::ring::default_provider());
    let crypto = rustls::ClientConfig::builder_with_provider(crypto_provider)
        .with_safe_default_protocol_versions()
        .map_err(error::configure_client::Inner::Rustls)?
        .dangerous()
        .with_custom_certificate_verifier(SkipServerVerification::new())
        .with_no_client_auth();
    let client_config =
        quinn::crypto::rustls::QuicClientConfig::try_from(crypto)?;
    Ok(ClientConfig::new(Arc::new(client_config)))
}

/// Returns default server configuration along with its certificate.
fn configure_server() -> Result<(ServerConfig, Vec<u8>), Error> {
    let cert_key =
        rcgen::generate_simple_self_signed(vec!["localhost".into()])?;
    let keypair_der = cert_key.key_pair.serialize_der();
    let priv_key = rustls::pki_types::PrivateKeyDer::Pkcs8(keypair_der.into());
    let cert_der = cert_key.cert.der().to_vec();
    let cert_chain = vec![cert_key.cert.into()];

    let mut server_config =
        ServerConfig::with_single_cert(cert_chain, priv_key)?;
    let transport_config = Arc::get_mut(&mut server_config.transport).unwrap();
    transport_config.max_concurrent_uni_streams(1_u8.into());

    Ok((server_config, cert_der))
}

/// Constructs a QUIC endpoint configured to listen for incoming connections on a certain address
/// and port.
///
/// ## Returns
///
/// - a stream of incoming QUIC connections
/// - server certificate serialized into DER format
pub fn make_server_endpoint(
    bind_addr: SocketAddr,
) -> Result<(Endpoint, Vec<u8>), Error> {
    let (server_config, server_cert) = configure_server()?;

    tracing::info!("creating server endpoint: binding to {bind_addr}",);

    let mut endpoint =
        Endpoint::server(server_config, bind_addr).map_err(Error::Quinn)?;
    let client_cfg = configure_client()?;
    endpoint.set_default_client_config(client_cfg);
    Ok((endpoint, server_cert))
}

// None indicates that the stream has ended
pub type PeerInfoRx =
    mpsc::UnboundedReceiver<(SocketAddr, Option<PeerConnectionInfo>)>;

const ALPHANET_SEED_PEER_ADDRS: &[PeerAddress<&'static str>] = {
    // seed.alpha.ecash.drivecha.in
    const DRIVECHA_IN: PeerAddress<&'static str> = PeerAddress {
        host: url::Host::Ipv4(Ipv4Addr::new(157, 180, 96, 24)),
        port: DEFAULT_PORT,
    };
    // seed.alpha.ecash.ninja
    const ECASH_NINJA: PeerAddress<&'static str> = PeerAddress {
        host: url::Host::Domain("seed.alpha.ecash.ninja"),
        port: DEFAULT_PORT,
    };
    &[DRIVECHA_IN, ECASH_NINJA]
};

const SIGNET_SEED_PEER_ADDRS: &[PeerAddress<&'static str>] = {
    const SIGNET_MINING_SERVER: PeerAddress<&'static str> = PeerAddress {
        host: url::Host::Ipv4(Ipv4Addr::new(172, 105, 148, 135)),
        port: DEFAULT_PORT,
    };
    const BIP300_XYZ: PeerAddress<&'static str> = PeerAddress {
        host: url::Host::Domain("thunder.bip300.xyz"),
        port: DEFAULT_PORT,
    };
    &[SIGNET_MINING_SERVER, BIP300_XYZ]
};

const FORKNET_SEED_PEER_ADDRS: &[PeerAddress<&'static str>] = {
    const BIP300_XYZ: PeerAddress<&'static str> = PeerAddress {
        host: url::Host::Domain("explorer.bip300.xyz"),
        port: DEFAULT_PORT,
    };
    &[
        BIP300_XYZ,
        PeerAddress {
            host: url::Host::Ipv4(Ipv4Addr::new(157, 180, 8, 224)),
            port: DEFAULT_PORT,
        },
    ]
};

const fn seed_peer_addrs(
    network: Network,
) -> &'static [PeerAddress<&'static str>] {
    match network {
        Network::Alphanet => ALPHANET_SEED_PEER_ADDRS,
        Network::Signet => SIGNET_SEED_PEER_ADDRS,
        Network::Regtest => &[],
        Network::Forknet => FORKNET_SEED_PEER_ADDRS,
    }
}

pub async fn resolve_peer_address<S>(
    peer_addr: PeerAddress<S>,
) -> std::io::Result<ResolvedPeerAddress<S>>
where
    S: std::fmt::Display + tokio::net::ToSocketAddrs,
{
    match peer_addr.host {
        url::Host::Ipv4(ipv4) => Ok(ResolvedPeerAddress::Static(
            SocketAddr::new(IpAddr::V4(ipv4), peer_addr.port),
        )),
        url::Host::Ipv6(ipv6) => Ok(ResolvedPeerAddress::Static(
            SocketAddr::new(IpAddr::V6(ipv6), peer_addr.port),
        )),
        url::Host::Domain(domain) => {
            let mut addrs: Vec<_> = tokio::net::lookup_host(&domain)
                .await?
                .filter_map(|addr| {
                    if addr.ip().is_unspecified() {
                        None
                    } else {
                        Some(addr.ip())
                    }
                })
                .collect();
            if let Some(last_addr) = addrs.pop() {
                addrs.reverse();
                let addrs = nonempty::NonEmpty {
                    head: last_addr,
                    tail: addrs,
                };
                Ok(ResolvedPeerAddress::Domain {
                    port: peer_addr.port,
                    addrs,
                    domain,
                })
            } else {
                tracing::warn!(%domain, "unable to resolve host");
                let err_msg =
                    format!("unable to resolve host for domain ({domain})");
                Err(std::io::Error::other(err_msg))
            }
        }
    }
}

/// Handle to tasks that dial known peers. Tasks are aborted on drop.
#[repr(transparent)]
pub struct DialKnownPeersHandle(
    tokio_util::task::JoinMap<PeerAddress, Result<(), error::DialKnownPeer>>,
);

impl DialKnownPeersHandle {
    pub async fn join_next(
        &mut self,
    ) -> Option<(
        PeerAddress,
        Result<Result<(), error::DialKnownPeer>, tokio::task::JoinError>,
    )> {
        self.0.join_next().await
    }
}

// Keep track of peer state
// Exchange metadata
// Bulk download
// Propagation
//
// Initial block download
//
// 1. Download headers
// 2. Download blocks
// 3. Update the state
#[derive(Clone)]
pub struct Net {
    pub server: Endpoint,
    archive: Archive,
    magic_bytes: peer_message::MagicBytes,
    state: State,
    active_peers: Arc<RwLock<HashMap<SocketAddr, PeerConnectionHandle>>>,
    // None indicates that the stream has ended
    peer_info_tx:
        mpsc::UnboundedSender<(SocketAddr, Option<PeerConnectionInfo>)>,
    known_peers: DatabaseUnique<SerdeBincode<PeerAddress>, Unit>,
    _version: DatabaseUnique<UnitKey, SerdeBincode<Version>>,
}

impl Net {
    pub const NUM_DBS: u32 = 2;

    fn add_active_peer(
        &self,
        addr: SocketAddr,
        peer_connection_handle: PeerConnectionHandle,
    ) -> Result<(), error::AlreadyConnected> {
        tracing::trace!(%addr, "add active peer: starting");
        let mut active_peers_write = self.active_peers.write();
        match active_peers_write.entry(addr) {
            hash_map::Entry::Occupied(_) => {
                tracing::error!(%addr, "add active peer: already connected");
                Err(error::AlreadyConnected(addr))
            }
            hash_map::Entry::Vacant(active_peer_entry) => {
                active_peer_entry.insert(peer_connection_handle);
                Ok(())
            }
        }
    }

    pub fn remove_active_peer(&self, addr: SocketAddr) {
        tracing::trace!(%addr, "remove active peer: starting");
        let mut active_peers_write = self.active_peers.write();
        if let Some(peer_connection) = active_peers_write.remove(&addr) {
            drop(peer_connection);
            tracing::info!(%addr, "remove active peer: disconnected");
        }
    }

    /// Apply the provided function to the peer connection handle,
    /// if it exists.
    pub fn try_with_active_peer_connection<F, T>(
        &self,
        addr: SocketAddr,
        f: F,
    ) -> Option<T>
    where
        F: FnMut(&PeerConnectionHandle) -> T,
    {
        let active_peers_read = self.active_peers.read();
        active_peers_read.get(&addr).map(f)
    }

    // TODO: This should have more context.
    // Last received message, connection state, etc.
    pub fn get_active_peers(&self) -> Vec<Peer> {
        self.active_peers
            .read()
            .iter()
            .map(|(addr, conn_handle)| Peer {
                address: *addr,
                status: conn_handle.connection_status(),
            })
            .collect()
    }

    #[instrument(skip_all, fields(addr), err(Debug))]
    pub fn connect_peer(
        &self,
        env: Env<heed::WithoutTls>,
        resolved_addr: ResolvedPeerAddress,
    ) -> Result<(), error::ConnectPeer> {
        {
            let mut rwtxn = env.write_txn()?;
            self.known_peers.put(
                &mut rwtxn,
                &resolved_addr.as_peer_address().to_owned(),
                &(),
            )?;
            rwtxn.commit()?;
        }
        {
            let active_peers = self.active_peers.read();
            for ip_addr in resolved_addr.ip_addrs() {
                let addr = SocketAddr::new(ip_addr, resolved_addr.port());
                if active_peers.contains_key(&addr) {
                    tracing::error!("already connected to peer");
                    return Err(error::AlreadyConnected(addr).into());
                }
            }
        }
        let addr = SocketAddr::new(
            resolved_addr.first_ip_addr(),
            resolved_addr.port(),
        );
        let peer_addr = resolved_addr.as_peer_address().to_owned();

        // This check happens within Quinn with a
        // generic "invalid remote address". We run the
        // same check, and provide a friendlier error
        // message.
        if addr.ip().is_unspecified() {
            return Err(error::ConnectPeer::UnspecfiedPeerIP(addr.ip()));
        }
        let connecting = self.server.connect(addr, "localhost")?;
        let connection_ctxt = PeerConnectionCtxt {
            env,
            archive: self.archive.clone(),
            magic_bytes: self.magic_bytes,
            resolved_address: resolved_addr,
            state: self.state.clone(),
        };

        let (connection_handle, info_rx) =
            peer::connect(connecting, connection_ctxt);
        tracing::trace!("connect peer: spawning info rx");
        tokio::spawn({
            let info_rx = StreamNotifyClose::new(info_rx)
                .map(move |info| Ok((addr, info)));
            let peer_info_tx = self.peer_info_tx.clone();
            async move {
                if let Err(_send_err) = info_rx.forward(peer_info_tx).await {
                    tracing::error!(
                        addr = %peer_addr,
                        "Failed to send peer connection info",
                    );
                }
            }
        });

        tracing::trace!("connect peer: adding to active peers");
        self.add_active_peer(addr, connection_handle)?;
        Ok(())
    }

    /// Delete peer from known_peers DB.
    /// Connections to the peer are not terminated.
    pub fn forget_peer(
        &self,
        rwtxn: &mut RwTxn,
        peer_address: &PeerAddress,
    ) -> Result<bool, Error> {
        self.known_peers
            .delete(rwtxn, peer_address)
            .map_err(|err| DbError::from(err).into())
    }

    async fn dial_known_peer(
        &self,
        env: Env<heed::WithoutTls>,
        peer_addr: PeerAddress,
    ) -> Result<(), error::DialKnownPeer> {
        tracing::trace!("connecting to already known peer at {peer_addr}");
        let resolved_peer_addr = resolve_peer_address(peer_addr)
            .await
            .map_err(error::DialKnownPeer::DnsResolve)?;
        let () = self.connect_peer(env, resolved_peer_addr)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        runtime: &tokio::runtime::Handle,
        env: &Env<heed::WithoutTls>,
        archive: Archive,
        magic_bytes_override: Option<peer_message::MagicBytes>,
        network: Network,
        state: State,
        bind_addr: SocketAddr,
        add_peers: HashSet<PeerAddress>,
    ) -> Result<(Self, PeerInfoRx, DialKnownPeersHandle), Error> {
        let (server, _) = make_server_endpoint(bind_addr)?;
        let active_peers = Arc::new(RwLock::new(HashMap::new()));
        let mut rwtxn = env.write_txn()?;
        let known_peers =
            match DatabaseUnique::open(env, &rwtxn, "known_peers")? {
                Some(known_peers) => known_peers,
                None => {
                    let known_peers =
                        DatabaseUnique::create(env, &mut rwtxn, "known_peers")?;
                    for seed_peer_addr in seed_peer_addrs(network) {
                        let seed_peer_addr =
                            PeerAddress::to_owned(seed_peer_addr);
                        known_peers.put(
                            &mut rwtxn,
                            &(seed_peer_addr.to_owned()),
                            &(),
                        )?;
                    }
                    known_peers
                }
            };
        for peer in add_peers {
            known_peers.put(&mut rwtxn, &peer, &())?;
        }
        let version = DatabaseUnique::create(env, &mut rwtxn, "net_version")?;
        match version.try_get(&rwtxn, &())? {
            Some(db_version)
                if db_version
                    < Version {
                        major: 0,
                        minor: 17,
                        patch: 6,
                    } =>
            {
                // types for `known_peers` db changed in v0.17.6
                return Err(Error::IncompatibleVersion {
                    version: db_version,
                    db_path: env.path().to_path_buf(),
                });
            }
            Some(_) => (),
            None => version.put(&mut rwtxn, &(), &*VERSION)?,
        }
        rwtxn.commit().map_err(RwTxnError::from)?;
        let magic_bytes = magic_bytes_override
            .unwrap_or_else(|| peer_message::magic_bytes(network));
        let (peer_info_tx, peer_info_rx) = mpsc::unbounded();
        let net = Net {
            server,
            archive,
            magic_bytes,
            state,
            active_peers,
            peer_info_tx,
            known_peers: known_peers.clone(),
            _version: version,
        };
        let known_peers: Vec<_> = {
            let rotxn = env.read_txn().map_err(EnvError::from)?;
            known_peers
                .iter_keys(&rotxn)
                .map_err(DbError::from)?
                .collect()
                .map_err(DbError::from)?
        };
        let dial_known_peers_handle = {
            let mut join_map = tokio_util::task::JoinMap::new();
            for peer_addr in known_peers {
                let env = Env::clone(env);
                let net = net.clone();
                join_map.spawn_on(
                    peer_addr.clone(),
                    async move {
                        net.dial_known_peer(env, peer_addr)
                            .await
                            .inspect_err(|err| tracing::error!(message = %ErrorChain::new(err)))
                    },
                    runtime
                );
            }
            DialKnownPeersHandle(join_map)
        };
        Ok((net, peer_info_rx, dial_known_peers_handle))
    }

    /// Accept the next incoming connection. Returns Some(addr) if a connection was accepted
    /// and a new peer was added.
    pub async fn accept_incoming(
        &self,
        env: Env<heed::WithoutTls>,
    ) -> Result<Option<SocketAddr>, error::AcceptConnection> {
        tracing::debug!(
            "accept incoming: listening for connections on `{}`",
            self.server
                .local_addr()
                .map(|socket| socket.to_string())
                .unwrap_or("unknown address".into())
        );
        let connection = match self.server.accept().await {
            Some(conn) => {
                let remote_address = conn.remote_address();
                tracing::trace!("accepting connection from {remote_address}",);

                let raw_conn = conn.await.map_err(|error| {
                    error::AcceptConnection::Connection {
                        error,
                        remote_address,
                    }
                })?;
                Connection::new(raw_conn, self.magic_bytes)
            }
            None => {
                tracing::debug!("server endpoint closed");
                return Err(error::AcceptConnection::ServerEndpointClosed);
            }
        };
        let addr = connection.addr();

        tracing::trace!(%addr, "accepted incoming connection");
        if self.active_peers.read().contains_key(&addr) {
            tracing::info!(
                %addr, "incoming connection: already peered, refusing duplicate",
            );
            connection
                .inner
                .close(quinn::VarInt::from_u32(1), b"already connected");
        }
        if connection.inner.close_reason().is_some() {
            return Ok(None);
        }
        tracing::info!(%addr, "connected to new peer");
        let connection_ctxt = PeerConnectionCtxt {
            env,
            archive: self.archive.clone(),
            magic_bytes: self.magic_bytes,
            resolved_address: addr.into(),
            state: self.state.clone(),
        };
        let (connection_handle, info_rx) =
            peer::handle(connection_ctxt, connection);
        tokio::spawn({
            let info_rx = StreamNotifyClose::new(info_rx)
                .map(move |info| Ok((addr, info)));
            let peer_info_tx = self.peer_info_tx.clone();
            async move {
                if let Err(_send_err) = info_rx.forward(peer_info_tx).await {
                    tracing::error!(%addr, "Failed to send peer connection info");
                }
            }
        });
        // TODO: is this the right state?
        self.add_active_peer(addr, connection_handle)?;
        Ok(Some(addr))
    }

    /// Attempt to push an internal message to the specified peer
    /// Returns `true` if successful
    pub fn push_internal_message(
        &self,
        message: PeerConnectionMessage,
        addr: SocketAddr,
    ) -> bool {
        let active_peers_read = self.active_peers.read();
        let Some(peer_connection_handle) = active_peers_read.get(&addr) else {
            let err = Error::MissingPeerConnection(addr);
            tracing::warn!("{:#}", ErrorChain::new(&err));
            return false;
        };

        if let Err(send_err) = peer_connection_handle
            .internal_message_tx
            .unbounded_send(message)
        {
            let message = send_err.into_inner();
            tracing::warn!(
                "Failed to push internal message to peer connection {addr}: {message:?}"
            );
            return false;
        }
        true
    }

    /// Push a tx to all active peers, except those in the provided set
    pub fn push_tx(
        &self,
        exclude: HashSet<SocketAddr>,
        tx: &AuthorizedTransaction,
    ) {
        self.active_peers
            .read()
            .iter()
            .filter(|(addr, _)| !exclude.contains(addr))
            .for_each(|(addr, peer_connection_handle)| {
                match peer_connection_handle.connection_status() {
                    PeerConnectionStatus::Connecting => {
                        tracing::trace!(%addr, "skipping peer at {addr} because it is not fully connected");
                        return;
                    }
                    PeerConnectionStatus::Connected => {}
                }
                let request: PeerRequest = peer::message::PushTransactionRequest {
                    transaction: tx.clone(),
                }.into();
                if let Err(_send_err) = peer_connection_handle
                    .internal_message_tx
                    .unbounded_send(request.into())
                {
                    let txid = tx.transaction.txid();
                    tracing::warn!("Failed to push tx {txid} to peer at {addr}")
                }
            })
    }
}
