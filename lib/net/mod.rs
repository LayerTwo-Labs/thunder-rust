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
use tokio_stream::StreamNotifyClose;

use crate::{archive::Archive, state::State, types::AuthorizedTransaction};

mod peer;

use peer::{
    Connection, ConnectionContext as PeerConnectionCtxt,
    ConnectionHandle as PeerConnectionHandle,
};
pub use peer::{
    ConnectionError as PeerConnectionError, Info as PeerConnectionInfo,
    Request as PeerRequest, Response as PeerResponse,
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("accept error")]
    AcceptError,
    #[error("address parse error")]
    AddrParse(#[from] std::net::AddrParseError),
    #[error("already connected to peer at {0}")]
    AlreadyConnected(SocketAddr),
    #[error("bincode error")]
    Bincode(#[from] bincode::Error),
    #[error("connect error")]
    Connect(#[from] quinn::ConnectError),
    #[error("connection error")]
    Connection(#[from] quinn::ConnectionError),
    #[error("heed error")]
    Heed(#[from] heed::Error),
    #[error("quinn error")]
    Io(#[from] std::io::Error),
    #[error("peer connection")]
    PeerConnection(#[from] PeerConnectionError),
    #[error("quinn rustls error")]
    QuinnRustls(#[from] quinn::crypto::rustls::Error),
    #[error("rcgen")]
    RcGen(#[from] rcgen::RcgenError),
    #[error("read to end error")]
    ReadToEnd(#[from] quinn::ReadToEndError),
    #[error("send datagram error")]
    SendDatagram(#[from] quinn::SendDatagramError),
    #[error("server endpoint closed")]
    ServerEndpointClosed,
    #[error("write error")]
    Write(#[from] quinn::WriteError),
}

pub fn make_client_endpoint(bind_addr: SocketAddr) -> Result<Endpoint, Error> {
    let client_cfg = configure_client();
    let mut endpoint = Endpoint::client(bind_addr)?;
    endpoint.set_default_client_config(client_cfg);
    Ok(endpoint)
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
    pub client: Endpoint,
    pub server: Endpoint,
    archive: Archive,
    state: State,
    active_peers: Arc<RwLock<HashMap<SocketAddr, PeerConnectionHandle>>>,
    // None indicates that the stream has ended
    peer_info_tx:
        mpsc::UnboundedSender<(SocketAddr, Option<PeerConnectionInfo>)>,
    known_peers: Database<SerdeBincode<SocketAddr>, Unit>,
}

impl Net {
    pub const NUM_DBS: u32 = 1;

    fn add_active_peer(
        &self,
        addr: SocketAddr,
        peer_connection_handle: PeerConnectionHandle,
    ) -> Result<(), Error> {
        let mut active_peers_write = self.active_peers.write();
        match active_peers_write.entry(addr) {
            hash_map::Entry::Occupied(_) => Err(Error::AlreadyConnected(addr)),
            hash_map::Entry::Vacant(active_peer_entry) => {
                active_peer_entry.insert(peer_connection_handle);
                Ok(())
            }
        }
    }

    pub fn remove_active_peer(&self, addr: SocketAddr) {
        let mut active_peers_write = self.active_peers.write();
        if let Some(peer_connection) = active_peers_write.remove(&addr) {
            drop(peer_connection);
            tracing::info!("Disconnected from peer at {addr}")
        }
    }

    pub fn connect_peer(
        &self,
        env: heed::Env,
        addr: SocketAddr,
    ) -> Result<(), Error> {
        if self.active_peers.read().contains_key(&addr) {
            return Err(Error::AlreadyConnected(addr));
        }
        let mut rwtxn = env.write_txn()?;
        self.known_peers.put(&mut rwtxn, &addr, &())?;
        rwtxn.commit()?;
        let connection_ctxt = PeerConnectionCtxt {
            env,
            archive: self.archive.clone(),
            state: self.state.clone(),
        };
        let (connection_handle, info_rx) =
            peer::connect(self.client.clone(), addr, connection_ctxt);
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
        self.add_active_peer(addr, connection_handle)?;
        Ok(())
    }

    pub fn new(
        env: &heed::Env,
        archive: Archive,
        state: State,
        bind_addr: SocketAddr,
    ) -> Result<(Self, PeerInfoRx), Error> {
        let (server, _) = make_server_endpoint(bind_addr)?;
        let client = make_client_endpoint("0.0.0.0:0".parse()?)?;
        let active_peers = Arc::new(RwLock::new(HashMap::new()));
        let mut rwtxn = env.write_txn()?;
        let known_peers =
            env.create_database(&mut rwtxn, Some("known_peers"))?;
        rwtxn.commit()?;
        let (peer_info_tx, peer_info_rx) = mpsc::unbounded();
        let net = Net {
            server,
            client,
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
            net.connect_peer(env.clone(), peer_addr)
        })?;
        Ok((net, peer_info_rx))
    }

    /// Accept the next incoming connection
    pub async fn accept_incoming(&self, env: heed::Env) -> Result<(), Error> {
        let connection = match self.server.accept().await {
            Some(conn) => Connection(conn.await?),
            None => return Err(Error::ServerEndpointClosed),
        };
        let addr = connection.addr();
        if self.active_peers.read().contains_key(&addr) {
            tracing::info!(
                "already connected to {addr}, refusing duplicate connection",
            );
            connection
                .0
                .close(quinn::VarInt::from_u32(1), b"already connected");
        }
        if connection.0.close_reason().is_some() {
            return Ok(());
        }
        tracing::info!("connected to peer at {addr}");
        let mut rwtxn = env.write_txn()?;
        self.known_peers.put(&mut rwtxn, &addr, &())?;
        rwtxn.commit()?;
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
        self.add_active_peer(addr, connection_handle)?;
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
            let Some(peer_connection_handle) = active_peers_read.get(addr)
            else {
                continue;
            };
            if let Err(_send_err) = peer_connection_handle
                .forward_request_tx
                .unbounded_send(request.clone())
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
            .for_each(|(addr, peer_connection_handle)| {
                if let Err(_send_err) = peer_connection_handle
                    .forward_request_tx
                    .unbounded_send(PeerRequest::PushTransaction {
                        transaction: tx.clone(),
                    })
                {
                    let txid = tx.transaction.txid();
                    tracing::warn!("Failed to push tx {txid} to peer at {addr}")
                }
            })
    }
}

/// Constructs a QUIC endpoint configured to listen for incoming connections on a certain address
/// and port.
///
/// ## Returns
///
/// - a stream of incoming QUIC connections
/// - server certificate serialized into DER format
#[allow(unused)]
pub fn make_server_endpoint(
    bind_addr: SocketAddr,
) -> Result<(Endpoint, Vec<u8>), Error> {
    let (server_config, server_cert) = configure_server()?;
    let endpoint = Endpoint::server(server_config, bind_addr)?;
    Ok((endpoint, server_cert))
}

/// Returns default server configuration along with its certificate.
fn configure_server() -> Result<(ServerConfig, Vec<u8>), Error> {
    let cert = rcgen::generate_simple_self_signed(vec!["localhost".into()])?;
    let cert_der = cert.serialize_der()?;
    let priv_key = cert.serialize_private_key_der();
    let priv_key = rustls::PrivateKey(priv_key);
    let cert_chain = vec![rustls::Certificate(cert_der.clone())];

    let mut server_config =
        ServerConfig::with_single_cert(cert_chain, priv_key)?;
    let transport_config = Arc::get_mut(&mut server_config.transport).unwrap();
    transport_config.max_concurrent_uni_streams(1_u8.into());

    Ok((server_config, cert_der))
}

/// Dummy certificate verifier that treats any certificate as valid.
/// NOTE, such verification is vulnerable to MITM attacks, but convenient for testing.
struct SkipServerVerification;

impl SkipServerVerification {
    fn new() -> Arc<Self> {
        Arc::new(Self)
    }
}

impl rustls::client::ServerCertVerifier for SkipServerVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &rustls::Certificate,
        _intermediates: &[rustls::Certificate],
        _server_name: &rustls::ServerName,
        _scts: &mut dyn Iterator<Item = &[u8]>,
        _ocsp_response: &[u8],
        _now: std::time::SystemTime,
    ) -> Result<rustls::client::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::ServerCertVerified::assertion())
    }
}

fn configure_client() -> ClientConfig {
    let crypto = rustls::ClientConfig::builder()
        .with_safe_defaults()
        .with_custom_certificate_verifier(SkipServerVerification::new())
        .with_no_client_auth();

    ClientConfig::new(Arc::new(crypto))
}
