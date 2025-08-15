use std::net::{IpAddr, SocketAddr};

use fatality::fatality;
use sneed::{db, env, rwtxn};
use thiserror::Error;
use transitive::Transitive;

use crate::net::PeerConnectionError;

#[derive(Debug, Error)]
#[error("already connected to peer at {0}")]
pub struct AlreadyConnected(pub SocketAddr);

/// Another connection can be accepted after a non-fatal error
#[allow(clippy::duplicated_attributes)]
#[derive(transitive::Transitive)]
#[fatality(splitable)]
#[transitive(
    from(sneed::db::error::Put, sneed::DbError),
    from(sneed::DbError, sneed::Error),
    from(sneed::env::error::WriteTxn, sneed::EnvError),
    from(sneed::EnvError, sneed::Error),
    from(sneed::RwTxnError, sneed::Error)
)]
pub enum AcceptConnection {
    #[error(transparent)]
    AlreadyConnected(#[from] AlreadyConnected),
    #[error("connection error (remote address: {remote_address})")]
    Connection {
        #[source]
        error: quinn::ConnectionError,
        remote_address: SocketAddr,
    },
    #[error(transparent)]
    #[fatal]
    Db(#[from] sneed::Error),
    #[error("server endpoint closed")]
    #[fatal]
    ServerEndpointClosed,
}

#[allow(clippy::duplicated_attributes)]
#[derive(Debug, Error, Transitive)]
#[transitive(from(db::error::Put, db::Error))]
#[transitive(from(db::error::TryGet, db::Error))]
#[transitive(from(env::error::CreateDb, env::Error))]
#[transitive(from(env::error::OpenDb, env::Error))]
#[transitive(from(env::error::WriteTxn, env::Error))]
#[transitive(from(rwtxn::error::Commit, rwtxn::Error))]
pub enum Error {
    #[error(transparent)]
    AcceptConnection(#[from] <AcceptConnection as fatality::Split>::Fatal),
    #[error("accept error")]
    AcceptError,
    #[error(transparent)]
    AlreadyConnected(#[from] AlreadyConnected),
    #[error("bincode error")]
    Bincode(#[from] bincode::Error),
    #[error("connect error")]
    Connect(#[from] quinn::ConnectError),
    #[error(transparent)]
    Db(#[from] db::Error),
    #[error("Database env error")]
    DbEnv(#[from] env::Error),
    #[error("Database write error")]
    DbWrite(#[from] rwtxn::Error),
    #[error("quinn error")]
    Io(#[from] std::io::Error),
    #[error("peer connection not found for {0}")]
    MissingPeerConnection(SocketAddr),
    /// Unspecified peer IP addresses cannot be connected to.
    /// `0.0.0.0` is one example of an "unspecified" IP.
    #[error("unspecified peer ip address (cannot connect to '{0}')")]
    UnspecfiedPeerIP(IpAddr),
    #[error(transparent)]
    NoInitialCipherSuite(#[from] quinn::crypto::rustls::NoInitialCipherSuite),
    #[error("peer connection")]
    PeerConnection(#[source] Box<PeerConnectionError>),
    #[error("quinn rustls error")]
    QuinnRustls(#[from] quinn::crypto::rustls::Error),
    #[error("rcgen")]
    RcGen(#[from] rcgen::Error),
    #[error("read to end error")]
    ReadToEnd(#[from] quinn::ReadToEndError),
    #[error("send datagram error")]
    SendDatagram(#[from] quinn::SendDatagramError),
    #[error("write error")]
    Write(#[from] quinn::WriteError),
}

impl From<PeerConnectionError> for Error {
    fn from(err: PeerConnectionError) -> Self {
        Self::PeerConnection(Box::new(err))
    }
}
