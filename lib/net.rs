use crate::types::{AuthorizedTransaction, Body, Header};
use quinn::{ClientConfig, Connection, Endpoint, ServerConfig};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

pub use quinn;
use std::collections::HashMap;
use std::{net::SocketAddr, sync::Arc};

pub const READ_LIMIT: usize = 1024;

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
    pub peers: Arc<RwLock<HashMap<usize, Peer>>>,
}

#[derive(Clone)]
pub struct Peer {
    pub state: Arc<RwLock<Option<PeerState>>>,
    pub connection: Connection,
}

impl Peer {
    pub fn heart_beat(&self, state: &PeerState) -> Result<(), Error> {
        let message = bincode::serialize(state)?;
        self.connection.send_datagram(bytes::Bytes::from(message))?;
        Ok(())
    }

    pub async fn request(&self, message: &Request) -> Result<Response, Error> {
        let (mut send, mut recv) = self.connection.open_bi().await?;
        let message = bincode::serialize(message)?;
        send.write_all(&message).await?;
        send.finish().await?;
        let response = recv.read_to_end(READ_LIMIT).await?;
        let response: Response = bincode::deserialize(&response)?;
        Ok(response)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub enum Request {
    GetBlock { height: u32 },
    PushTransaction { transaction: AuthorizedTransaction },
}

#[derive(Debug, Serialize, Deserialize)]
pub enum Response {
    Block { header: Header, body: Body },
    NoBlock,
    TransactionAccepted,
    TransactionRejected,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct PeerState {
    pub block_height: u32,
}

impl Net {
    pub fn new(bind_addr: SocketAddr) -> Result<Self, Error> {
        let (server, _) = make_server_endpoint(bind_addr)?;
        let client = make_client_endpoint("0.0.0.0:0".parse()?)?;
        let peers = Arc::new(RwLock::new(HashMap::new()));
        Ok(Net {
            server,
            client,
            peers,
        })
    }
    pub async fn connect(&self, addr: SocketAddr) -> Result<Peer, Error> {
        for peer in self.peers.read().await.values() {
            if peer.connection.remote_address() == addr {
                return Err(Error::AlreadyConnected(addr));
            }
        }
        let connection = self.client.connect(addr, "localhost")?.await?;
        let peer = Peer {
            state: Arc::new(RwLock::new(None)),
            connection,
        };
        self.peers
            .write()
            .await
            .insert(peer.connection.stable_id(), peer.clone());
        Ok(peer)
    }

    pub async fn disconnect(
        &self,
        stable_id: usize,
    ) -> Result<Option<Peer>, Error> {
        let peer = self.peers.write().await.remove(&stable_id);
        Ok(peer)
    }
}

#[allow(unused)]
pub fn make_client_endpoint(bind_addr: SocketAddr) -> Result<Endpoint, Error> {
    let client_cfg = configure_client();
    let mut endpoint = Endpoint::client(bind_addr)?;
    endpoint.set_default_client_config(client_cfg);
    Ok(endpoint)
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

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("address parse error")]
    AddrParse(#[from] std::net::AddrParseError),
    #[error("quinn error")]
    Io(#[from] std::io::Error),
    #[error("connect error")]
    Connect(#[from] quinn::ConnectError),
    #[error("connection error")]
    Connection(#[from] quinn::ConnectionError),
    #[error("rcgen")]
    RcGen(#[from] rcgen::RcgenError),
    #[error("accept error")]
    AcceptError,
    #[error("read to end error")]
    ReadToEnd(#[from] quinn::ReadToEndError),
    #[error("write error")]
    Write(#[from] quinn::WriteError),
    #[error("send datagram error")]
    SendDatagram(#[from] quinn::SendDatagramError),
    #[error("quinn rustls error")]
    QuinnRustls(#[from] quinn::crypto::rustls::Error),
    #[error("bincode error")]
    Bincode(#[from] bincode::Error),
    #[error("already connected to peer at {0}")]
    AlreadyConnected(SocketAddr),
}
