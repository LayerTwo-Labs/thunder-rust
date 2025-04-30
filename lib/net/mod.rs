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
    DatabaseUnique, EnvError, RwTxnError, UnitKey, db::error::Error as DbError,
};
use tokio_stream::StreamNotifyClose;
use tracing::instrument;

use crate::{
    archive::Archive,
    state::State,
    types::{AuthorizedTransaction, THIS_SIDECHAIN, VERSION, Version},
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
    InternalMessage as PeerConnectionMessage, Peer, PeerConnectionStatus,
    PeerStateId, Request as PeerRequest, ResponseMessage as PeerResponse,
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

fn configure_client()
-> Result<ClientConfig, quinn::crypto::rustls::NoInitialCipherSuite> {
    let crypto = rustls::ClientConfig::builder()
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
    state: State,
    active_peers: Arc<RwLock<HashMap<SocketAddr, PeerConnectionHandle>>>,
    // None indicates that the stream has ended
    peer_info_tx:
        mpsc::UnboundedSender<(SocketAddr, Option<PeerConnectionInfo>)>,
    known_peers: DatabaseUnique<SerdeBincode<SocketAddr>, Unit>,
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
        env: sneed::Env,
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
        let mut rwtxn = env.write_txn().map_err(EnvError::from)?;
        self.known_peers
            .put(&mut rwtxn, &addr, &())
            .map_err(DbError::from)?;
        rwtxn.commit().map_err(RwTxnError::from)?;
        let connection_ctxt = PeerConnectionCtxt {
            env,
            archive: self.archive.clone(),
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

    pub fn new(
        env: &sneed::Env,
        archive: Archive,
        state: State,
        bind_addr: SocketAddr,
    ) -> Result<(Self, PeerInfoRx), Error> {
        let (server, _) = make_server_endpoint(bind_addr)?;
        let active_peers = Arc::new(RwLock::new(HashMap::new()));
        let mut rwtxn = env.write_txn().map_err(EnvError::from)?;
        let known_peers = match DatabaseUnique::open(env, &rwtxn, "known_peers")
            .map_err(EnvError::from)?
        {
            Some(known_peers) => known_peers,
            None => {
                let known_peers =
                    DatabaseUnique::create(env, &mut rwtxn, "known_peers")
                        .map_err(EnvError::from)?;
                const SEED_NODE_ADDR: SocketAddr = SocketAddr::new(
                    std::net::IpAddr::V4(std::net::Ipv4Addr::new(
                        172, 105, 148, 135,
                    )),
                    4000 + THIS_SIDECHAIN as u16,
                );
                known_peers
                    .put(&mut rwtxn, &SEED_NODE_ADDR, &())
                    .map_err(DbError::from)?;
                known_peers
            }
        };
        let version = DatabaseUnique::create(env, &mut rwtxn, "net_version")
            .map_err(EnvError::from)?;
        if version
            .try_get(&rwtxn, &())
            .map_err(DbError::from)?
            .is_none()
        {
            version
                .put(&mut rwtxn, &(), &*VERSION)
                .map_err(DbError::from)?;
        }
        rwtxn.commit().map_err(RwTxnError::from)?;
        let (peer_info_tx, peer_info_rx) = mpsc::unbounded();
        let net = Net {
            server,
            archive,
            state,
            active_peers,
            peer_info_tx,
            known_peers,
            _version: version,
        };
        #[allow(clippy::let_and_return)]
        let known_peers: Vec<_> = {
            let rotxn = env.read_txn().map_err(EnvError::from)?;
            let known_peers = net
                .known_peers
                .iter(&rotxn)
                .map_err(DbError::from)?
                .collect()
                .map_err(DbError::from)?;
            known_peers
        };
        let () = known_peers.into_iter().try_for_each(|(peer_addr, _)| {
            tracing::trace!(
                "new net: connecting to already known peer at {peer_addr}"
            );
            match net.connect_peer(env.clone(), peer_addr) {
                Err(Error::Connect(
                    quinn::ConnectError::InvalidRemoteAddress(addr),
                )) => {
                    tracing::warn!(
                        %addr, "new net: known peer with invalid remote address, removing"
                    );
                    let mut tx = env.write_txn().map_err(EnvError::from)?;
                    net.known_peers.delete(&mut tx, &peer_addr).map_err(DbError::from)?;
                    tx.commit().map_err(RwTxnError::from)?;

                    tracing::info!(
                        %addr,
                        "new net: removed known peer with invalid remote address"
                    );
                    Ok(())
                }
                res => res,
            }
        })
        // TODO: would be better to indicate this in the return error? tbh I want to scrap
        // the typed error out of here, and just use anyhow
        .inspect_err(|err| {
            tracing::error!("unable to connect to known peers during net construction: {err:#}");
        })?;
        Ok((net, peer_info_rx))
    }

    /// Accept the next incoming connection. Returns Some(addr) if a connection was accepted
    /// and a new peer was added.
    pub async fn accept_incoming(
        &self,
        env: sneed::Env,
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
                Connection::from(raw_conn)
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
        let mut rwtxn = env.write_txn().map_err(EnvError::from)?;
        self.known_peers
            .put(&mut rwtxn, &addr, &())
            .map_err(DbError::from)?;
        rwtxn.commit().map_err(RwTxnError::from)?;

        tracing::trace!(%addr, "wrote peer to database");
        let connection_ctxt = PeerConnectionCtxt {
            env,
            archive: self.archive.clone(),
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
            tracing::warn!("{:#}", anyhow::Error::from(err));
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
        tx: AuthorizedTransaction,
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
