use std::{
    collections::{hash_map, HashMap, HashSet},
    net::SocketAddr,
    sync::Arc,
};

use fallible_iterator::{FallibleIterator, IteratorExt};
use futures::{channel::mpsc, StreamExt};
use heed::{
    types::{SerdeBincode, Unit},
    Database,
};
use parking_lot::RwLock;
use quinn::{ClientConfig, Endpoint, ServerConfig};
use strum;
use tokio_stream::StreamNotifyClose;
use tracing::instrument;

use crate::{
    archive::Archive,
    state::State,
    types::{AuthorizedTransaction, THIS_SIDECHAIN},
};

pub mod peer;

use peer::{
    Connection, ConnectionContext as PeerConnectionCtxt,
    ConnectionHandle as PeerConnectionHandle,
};
pub use peer::{
    ConnectionError as PeerConnectionError, Info as PeerConnectionInfo,
    InternalMessage as PeerConnectionMessage, PeerStateId,
    Request as PeerRequest, Response as PeerResponse,
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("accept error")]
    AcceptError,
    #[error("already connected to peer at {0}")]
    AlreadyConnected(SocketAddr),
    #[error("bincode error")]
    Bincode(#[from] bincode::Error),
    #[error("connect error")]
    Connect(#[from] quinn::ConnectError),
    #[error("connection error (remote address: {remote_address})")]
    Connection {
        #[source]
        error: quinn::ConnectionError,
        remote_address: SocketAddr,
    },
    #[error("heed error")]
    Heed(#[from] heed::Error),
    #[error("quinn error")]
    Io(#[from] std::io::Error),
    #[error("peer connection not found for {0}")]
    MissingPeerConnection(SocketAddr),
    #[error("peer connection is not fully connected for {0}")]
    PeerNotConnected(SocketAddr),
    /// Unspecified peer IP addresses cannot be connected to.
    /// `0.0.0.0` is one example of an "unspecified" IP.
    #[error("unspecified peer ip address (cannot connect to '0.0.0.0')")]
    UnspecfiedPeerIP,
    #[error(transparent)]
    NoInitialCipherSuite(#[from] quinn::crypto::rustls::NoInitialCipherSuite),
    #[error("peer connection")]
    PeerConnection(#[from] PeerConnectionError),
    #[error("quinn rustls error")]
    QuinnRustls(#[from] quinn::crypto::rustls::Error),
    #[error("rcgen")]
    RcGen(#[from] rcgen::Error),
    #[error("read to end error")]
    ReadToEnd(#[from] quinn::ReadToEndError),
    #[error("send datagram error")]
    SendDatagram(#[from] quinn::SendDatagramError),
    #[error("server endpoint closed")]
    ServerEndpointClosed,
    #[error("write error")]
    Write(#[from] quinn::WriteError),
}

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

fn configure_client(
) -> Result<ClientConfig, quinn::crypto::rustls::NoInitialCipherSuite> {
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

    tracing::info!(
        "creating server endpoint: binding to {bind_addr} ({})",
        if bind_addr.is_ipv6() { "ipv6" } else { "ipv4" }
    );

    let mut endpoint = Endpoint::server(server_config, bind_addr)?;
    let client_cfg = configure_client()?;
    endpoint.set_default_client_config(client_cfg);
    Ok((endpoint, server_cert))
}

#[derive(
    Clone,
    Copy,
    Eq,
    PartialEq,
    serde::Serialize,
    serde::Deserialize,
    strum::Display,
    utoipa::ToSchema,
)]
pub enum PeerConnectionState {
    /// We're still in the process of initializing the peer connection
    Connecting,

    /// The connection is successfully established
    Connected,
}

// None indicates that the stream has ended
pub type PeerInfoRx =
    mpsc::UnboundedReceiver<(SocketAddr, Option<PeerConnectionInfo>)>;

// State.
// Archive.

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
    active_peers: Arc<
        RwLock<
            HashMap<SocketAddr, (PeerConnectionHandle, PeerConnectionState)>,
        >,
    >,
    // None indicates that the stream has ended
    peer_info_tx:
        mpsc::UnboundedSender<(SocketAddr, Option<PeerConnectionInfo>)>,
    known_peers: Database<SerdeBincode<SocketAddr>, Unit>,
}

impl Net {
    pub const NUM_DBS: u32 = 1;

    fn update_active_peer_state(
        &self,
        addr: SocketAddr,
        new_state: PeerConnectionState,
    ) -> Result<(), Error> {
        let mut active_peers_write = self.active_peers.write();
        active_peers_write
            .entry(addr)
            .and_modify(|(_, state)| *state = new_state);
        Ok(())
    }

    fn add_active_peer(
        &self,
        addr: SocketAddr,
        peer_connection_handle: PeerConnectionHandle,
        state: PeerConnectionState,
    ) -> Result<(), Error> {
        tracing::trace!(%addr, "add active peer: starting");

        let mut active_peers_write = self.active_peers.write();
        match active_peers_write.entry(addr) {
            hash_map::Entry::Occupied(_) => {
                tracing::error!(%addr, "add active peer: already connected");
                Err(Error::AlreadyConnected(addr))
            }
            hash_map::Entry::Vacant(active_peer_entry) => {
                active_peer_entry.insert((peer_connection_handle, state));
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

    // TODO: This should have more context. Last received message, connection state, etc.
    pub fn get_active_peers(&self) -> Vec<(SocketAddr, PeerConnectionState)> {
        self.active_peers
            .read()
            .iter()
            .map(|(addr, (_, state))| (*addr, state.to_owned()))
            .collect()
    }

    #[instrument(skip_all, fields(addr), err(Debug))]
    pub fn connect_peer(
        &self,
        env: heed::Env,
        addr: SocketAddr,
    ) -> Result<(), Error> {
        if self.active_peers.read().contains_key(&addr) {
            tracing::error!("connect peer: already connected");
            return Err(Error::AlreadyConnected(addr));
        }

        // This check happens within Quinn with a
        // generic "invalid remote address". We run the
        // same check, and provide a friendlier error
        // message.
        if addr.ip().is_unspecified() {
            return Err(Error::UnspecfiedPeerIP);
        }
        let connecting = self.server.connect(addr, "localhost")?;
        let mut rwtxn = env.write_txn()?;
        self.known_peers.put(&mut rwtxn, &addr, &())?;
        rwtxn.commit()?;
        let connection_ctxt = PeerConnectionCtxt {
            env,
            archive: self.archive.clone(),
            state: self.state.clone(),
        };

        let (connected_tx, mut connected_rx) = mpsc::unbounded();
        let (connection_handle, info_rx) =
            peer::connect(connecting, connection_ctxt, connected_tx);
        tracing::trace!("connect peer: spawning info rx");
        tokio::spawn({
            let info_rx = StreamNotifyClose::new(info_rx)
                .map(move |info| Ok((addr, info)));
            let peer_info_tx = self.peer_info_tx.clone();
            let net = self.clone();

            async move {
                let update_peer_state = async move {
                    connected_rx.next().await.inspect(|_| {
                        match net.update_active_peer_state(addr, PeerConnectionState::Connected) {
                            Ok(_) => {
                                tracing::info!(%addr, "connect peer: updated state to connected");
                            }
                            Err(err) => {
                                tracing::error!(
                                    "failed to update active peer state: {err:#}"
                                );
                            }
                        }
                    })
                };

                let forward_peer_info = async move {
                    if let Err(_send_err) = info_rx.forward(peer_info_tx).await
                    {
                        tracing::error!("Failed to send peer connection info");
                    };
                };

                tokio::join!(forward_peer_info, update_peer_state)
            }
        });

        tracing::trace!("connect peer: adding to active peers");
        self.add_active_peer(
            addr,
            connection_handle,
            PeerConnectionState::Connecting,
        )?;
        Ok(())
    }

    pub fn new(
        env: &heed::Env,
        archive: Archive,
        state: State,
        bind_addr: SocketAddr,
    ) -> Result<(Self, PeerInfoRx), Error> {
        let (server, _) = make_server_endpoint(bind_addr)?;
        let active_peers = Arc::new(RwLock::new(HashMap::new()));
        let mut rwtxn = env.write_txn()?;
        let known_peers =
            match env.open_database(&rwtxn, Some("known_peers"))? {
                Some(known_peers) => known_peers,
                None => {
                    let known_peers =
                        env.create_database(&mut rwtxn, Some("known_peers"))?;
                    const SEED_NODE_ADDR: SocketAddr = SocketAddr::new(
                        std::net::IpAddr::V4(std::net::Ipv4Addr::new(
                            172, 105, 148, 135,
                        )),
                        4000 + THIS_SIDECHAIN as u16,
                    );
                    known_peers.put(&mut rwtxn, &SEED_NODE_ADDR, &())?;
                    known_peers
                }
            };
        rwtxn.commit()?;
        let (peer_info_tx, peer_info_rx) = mpsc::unbounded();
        let net = Net {
            server,
            archive,
            state,
            active_peers,
            peer_info_tx,
            known_peers,
        };
        #[allow(clippy::let_and_return)]
        let known_peers: Vec<_> = {
            let rotxn = env.read_txn()?;
            let known_peers = net
                .known_peers
                .iter(&rotxn)?
                .transpose_into_fallible()
                .collect()?;
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
                    let mut tx = env.write_txn()?;
                    net.known_peers.delete(&mut tx, &peer_addr)?;
                    tx.commit()?;

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
        env: heed::Env,
    ) -> Result<Option<SocketAddr>, Error> {
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

                let raw_conn =
                    conn.await.map_err(|error| Error::Connection {
                        error,
                        remote_address,
                    })?;
                Connection(raw_conn)
            }
            None => {
                tracing::debug!("server endpoint closed");
                return Err(Error::ServerEndpointClosed);
            }
        };
        let addr = connection.addr();

        tracing::trace!(%addr, "accepted incoming connection");
        if self.active_peers.read().contains_key(&addr) {
            tracing::info!(
                %addr, "incoming connection: already peered, refusing duplicate",
            );
            connection
                .0
                .close(quinn::VarInt::from_u32(1), b"already connected");
        }
        if connection.0.close_reason().is_some() {
            return Ok(None);
        }
        tracing::info!(%addr, "connected to new peer");
        let mut rwtxn = env.write_txn()?;
        self.known_peers.put(&mut rwtxn, &addr, &())?;
        rwtxn.commit()?;

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
        self.add_active_peer(
            addr,
            connection_handle,
            PeerConnectionState::Connected,
        )?;
        Ok(Some(addr))
    }

    // Push an internal message to the specified peer
    pub fn push_internal_message(
        &self,
        message: PeerConnectionMessage,
        addr: SocketAddr,
    ) -> Result<(), Error> {
        let active_peers_read = self.active_peers.read();
        let Some((peer_connection_handle, state)) =
            active_peers_read.get(&addr)
        else {
            return Err(Error::MissingPeerConnection(addr));
        };

        match state {
            PeerConnectionState::Connecting => {
                return Err(Error::PeerNotConnected(addr));
            }
            PeerConnectionState::Connected => {}
        }

        if let Err(send_err) = peer_connection_handle
            .internal_message_tx
            .unbounded_send(message)
        {
            let message = send_err.into_inner();
            tracing::error!("Failed to push internal message to peer connection {addr}: {message:?}")
        }
        Ok(())
    }

    // Push a request to the specified peers
    pub fn push_request(
        &self,
        request: PeerRequest,
        peers: &HashSet<SocketAddr>,
    ) {
        let active_peers_read = self.active_peers.read();
        for addr in peers {
            let Some((peer_connection_handle, PeerConnectionState::Connected)) =
                active_peers_read.get(addr)
            else {
                continue;
            };
            if let Err(_send_err) = peer_connection_handle
                .internal_message_tx
                .unbounded_send(request.clone().into())
            {
                tracing::warn!(
                    "Failed to push request to peer at {addr}: {request:?}"
                )
            }
        }
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
            .for_each(|(addr, (peer_connection_handle, state))| {
                match state {
                    PeerConnectionState::Connecting => {
                        tracing::trace!(%addr, "skipping peer at {addr} because it is not fully connected");
                        return;
                    }
                    PeerConnectionState::Connected => {}
                }
                let request = PeerRequest::PushTransaction {
                    transaction: tx.clone(),
                };
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
