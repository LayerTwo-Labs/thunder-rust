use std::{
    collections::{HashMap, HashSet, hash_map},
    net::SocketAddr,
    sync::Arc,
};

use fallible_iterator::FallibleIterator;
use futures::{StreamExt, channel::mpsc};
use heed::types::{SerdeBincode, Unit};
use parking_lot::RwLock;
use quinn::{ClientConfig, Endpoint, ServerConfig};
use sneed::{
    DatabaseUnique, EnvError, RwTxn, RwTxnError, UnitKey,
    db::error::Error as DbError,
};
use tokio_stream::StreamNotifyClose;
use tracing::instrument;

use crate::{
    archive::Archive,
    state::State,
    types::{
        AuthorizedTransaction, Network, THIS_SIDECHAIN, VERSION, Version,
        net::{Peer, PeerConnectionStatus},
    },
    util::ErrorChain,
};

pub mod error;
mod peer;

pub use error::Error;
pub(crate) use peer::error::mailbox::Error as PeerConnectionMailboxError;
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

    let mut endpoint = Endpoint::server(server_config, bind_addr)?;
    let client_cfg = configure_client()?;
    endpoint.set_default_client_config(client_cfg);
    Ok((endpoint, server_cert))
}

// None indicates that the stream has ended
pub type PeerInfoRx =
    mpsc::UnboundedReceiver<(SocketAddr, Option<PeerConnectionInfo>)>;

const DEFAULT_SEED_NODE_PORT: u16 = 4000 + THIS_SIDECHAIN as u16;

const SIGNET_SEED_NODES: &[(&str, u16)] = &[
    // Signet mining server.
    ("172.105.148.135", DEFAULT_SEED_NODE_PORT),
];

const FORKNET_SEED_NODES: &[(&str, u16)] = &[
    ("157.180.8.224", DEFAULT_SEED_NODE_PORT),
    ("explorer.bip300.xyz", DEFAULT_SEED_NODE_PORT),
];

/// Built-in seed peers for the provided network.
pub fn builtin_seed_peers(network: Network) -> Vec<PeerAddress> {
    let seeds: &[(&str, u16)] = match network {
        Network::Signet => SIGNET_SEED_NODES,
        Network::Regtest => &[],
        Network::Forknet => FORKNET_SEED_NODES,
    };
    seeds
        .iter()
        .map(|(host, port)| PeerAddress {
            host: url::Host::parse(host)
                .expect("builtin seed host should parse"),
            port: *port,
        })
        .collect()
}

/// A peer's address, as a host and port. The host is kept unresolved, so
/// that a peer whose name resolves to several socket addresses is still
/// identified (and persisted) as a single peer. Resolution happens when the
/// peer is dialed.
#[derive(
    Clone, Debug, Eq, Hash, PartialEq, serde::Deserialize, serde::Serialize,
)]
pub struct PeerAddress {
    pub host: url::Host,
    pub port: u16,
}

impl PeerAddress {
    /// Resolve to socket addresses. IP hosts resolve to themselves without
    /// a lookup. Failures are logged and yield no addresses, so a peer that
    /// cannot be resolved never prevents others from being dialed.
    pub async fn resolve(&self) -> Vec<SocketAddr> {
        let domain = match &self.host {
            url::Host::Ipv4(ipv4) => {
                return vec![SocketAddr::from((*ipv4, self.port))];
            }
            url::Host::Ipv6(ipv6) => {
                return vec![SocketAddr::from((*ipv6, self.port))];
            }
            url::Host::Domain(domain) => domain,
        };
        match tokio::net::lookup_host((domain.as_str(), self.port)).await {
            Ok(socket_addrs) => {
                let socket_addrs: Vec<SocketAddr> = socket_addrs.collect();
                if socket_addrs.is_empty() {
                    tracing::warn!(
                        peer_address = %self,
                        "resolve peer: host resolved to no addresses"
                    );
                } else {
                    tracing::debug!(
                        peer_address = %self,
                        "resolve peer: resolved host to {socket_addrs:?}"
                    );
                }
                socket_addrs
            }
            Err(err) => {
                tracing::warn!(
                    peer_address = %self,
                    "resolve peer: failed to resolve host: {err:#}"
                );
                Vec::new()
            }
        }
    }
}

impl From<SocketAddr> for PeerAddress {
    fn from(socket_addr: SocketAddr) -> Self {
        let host = match socket_addr.ip() {
            std::net::IpAddr::V4(ipv4) => url::Host::Ipv4(ipv4),
            std::net::IpAddr::V6(ipv6) => url::Host::Ipv6(ipv6),
        };
        Self {
            host,
            port: socket_addr.port(),
        }
    }
}

impl std::fmt::Display for PeerAddress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `url::Host` displays IPv6 addresses bracketed, so this
        // round-trips through the `FromStr` impl.
        write!(f, "{}:{}", self.host, self.port)
    }
}

/// Parses `host:port`. IPv6 literals must be bracketed, as in `[::1]:4009`, so
/// their colons are not read as the port separator.
impl std::str::FromStr for PeerAddress {
    type Err = error::ParsePeerAddress;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (host, port) = s
            .rsplit_once(':')
            .ok_or_else(|| error::ParsePeerAddress::new(s, "has no port"))?;
        let host = url::Host::parse(host).map_err(|_| {
            error::ParsePeerAddress::new(s, "has an invalid host")
        })?;
        let port = port.parse().map_err(|_| {
            error::ParsePeerAddress::new(s, "has an invalid port")
        })?;
        Ok(Self { host, port })
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

    /// The peer is not recorded in `known_peers` here, see
    /// [`Self::remember_peer`].
    #[instrument(skip_all, fields(addr), err(Debug))]
    pub fn connect_peer(
        &self,
        env: sneed::Env<heed::WithoutTls>,
        addr: SocketAddr,
    ) -> Result<(), Error> {
        if self.active_peers.read().contains_key(&addr) {
            tracing::error!("connect peer: already connected");
            return Err(error::AlreadyConnected(addr).into());
        }

        // This check happens within Quinn with a
        // generic "invalid remote address". We run the
        // same check, and provide a friendlier error
        // message.
        if addr.ip().is_unspecified() {
            return Err(Error::UnspecfiedPeerIP(addr.ip()));
        }
        let connecting = self.server.connect(addr, "localhost")?;
        let connection_ctxt = PeerConnectionCtxt {
            env,
            archive: self.archive.clone(),
            magic_bytes: self.magic_bytes,
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
                    tracing::error!(%addr, "Failed to send peer connection info");
                }
            }
        });

        tracing::trace!("connect peer: adding to active peers");
        self.add_active_peer(addr, connection_handle)?;
        Ok(())
    }

    /// Record a peer in `known_peers`, so that it is dialed again on the next
    /// startup.
    ///
    /// Only for peers that have emitted [`PeerConnectionInfo::Validated`].
    /// Recording one any earlier persists nodes from other networks, which are
    /// then re-dialed on every startup forever.
    pub fn remember_peer(
        &self,
        rwtxn: &mut RwTxn,
        peer_address: &PeerAddress,
    ) -> Result<(), Error> {
        self.known_peers
            .put(rwtxn, peer_address, &())
            .map_err(|err| DbError::from(err).into())
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

    /// All peers recorded in `known_peers`, to be dialed on startup.
    ///
    /// If the records cannot be read -- e.g. they were written by an older
    /// version, which stored socket addresses -- the DB is cleared instead.
    /// It is only a cache of validated peers, which will be re-learned.
    pub fn get_known_peers(
        &self,
        env: &sneed::Env<heed::WithoutTls>,
    ) -> Result<Vec<PeerAddress>, Error> {
        let known_peers: Result<Vec<(PeerAddress, ())>, DbError> = {
            let rotxn = env.read_txn().map_err(EnvError::from)?;
            self.known_peers
                .iter(&rotxn)
                .map_err(DbError::from)
                .and_then(|it| it.collect().map_err(DbError::from))
        };
        match known_peers {
            Ok(known_peers) => Ok(known_peers
                .into_iter()
                .map(|(peer_address, ())| peer_address)
                .collect()),
            Err(err) => {
                tracing::warn!(
                    "clearing unreadable known peers DB: {:#}",
                    ErrorChain::new(&err)
                );
                let mut rwtxn = env.write_txn().map_err(EnvError::from)?;
                let () = self
                    .known_peers
                    .clear(&mut rwtxn)
                    .map_err(DbError::from)?;
                rwtxn.commit().map_err(RwTxnError::from)?;
                Ok(Vec::new())
            }
        }
    }

    /// Known peers and seed nodes are not dialed here: the net task reads
    /// them via [`Self::get_known_peers`], resolves them, and dials them, so
    /// that node startup never waits on a resolver.
    pub fn new(
        env: &sneed::Env<heed::WithoutTls>,
        archive: Archive,
        magic_bytes_override: Option<peer_message::MagicBytes>,
        network: Network,
        state: State,
        bind_addr: SocketAddr,
    ) -> Result<(Self, PeerInfoRx), Error> {
        let (server, _) = make_server_endpoint(bind_addr)?;
        let active_peers = Arc::new(RwLock::new(HashMap::new()));
        let mut rwtxn = env.write_txn()?;
        // Seed nodes are resolved and dialed on every startup instead of being
        // written here, so a relocated seed is picked up from DNS.
        let known_peers =
            match DatabaseUnique::open(env, &rwtxn, "known_peers")? {
                Some(known_peers) => known_peers,
                None => DatabaseUnique::create(env, &mut rwtxn, "known_peers")?,
            };
        let version = DatabaseUnique::create(env, &mut rwtxn, "net_version")?;
        if version.try_get(&rwtxn, &())?.is_none() {
            version.put(&mut rwtxn, &(), &*VERSION)?;
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
            known_peers,
            _version: version,
        };
        Ok((net, peer_info_rx))
    }

    /// Accept the next incoming connection. Returns Some(addr) if a connection was accepted
    /// and a new peer was added.
    pub async fn accept_incoming(
        &self,
        env: sneed::Env<heed::WithoutTls>,
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
        // Not written to `known_peers` here: the handshake proves nothing about
        // which network the peer is on. Recorded once it emits `Validated`.
        let connection_ctxt = PeerConnectionCtxt {
            env,
            archive: self.archive.clone(),
            magic_bytes: self.magic_bytes,
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
