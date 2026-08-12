use std::{
    net::{IpAddr, SocketAddr},
    path::PathBuf,
};

use error_fatality::{Fatality, Split};
use sneed::{db::error as db, env::error as env, rwtxn::error as rwtxn};
use thiserror::Error;
use transitive::Transitive;

pub(crate) use crate::net::peer::error as peer;
use crate::{net::PeerConnectionError, types::Version};

#[derive(Debug, Error)]
#[error("already connected to peer at {0}")]
pub struct AlreadyConnected(pub SocketAddr);

pub mod parse_peer_address {
    use thiserror::Error;

    #[derive(Debug, Error)]
    pub enum Inner {}

    #[derive(Debug, Error)]
    #[error("failed to parse peer address")]
    #[repr(transparent)]
    pub struct Error(#[from] Inner);
}
pub use parse_peer_address::Error as ParsePeerAddress;

/// Another connection can be accepted after a non-fatal error
#[allow(clippy::duplicated_attributes)]
#[derive(Debug, Error, Fatality, Split, Transitive)]
#[split(attrs(derive(Debug, Error)))]
#[transitive(
    from(db::Error, sneed::Error),
    from(db::Put, db::Error),
    from(env::WriteTxn, env::Error),
    from(env::Error, sneed::Error),
    from(rwtxn::Error, sneed::Error)
)]
pub enum AcceptConnection {
    #[error(transparent)]
    #[fatal(false)]
    AlreadyConnected(#[from] AlreadyConnected),
    #[error("connection error (remote address: {remote_address})")]
    #[fatal(false)]
    Connection {
        #[source]
        error: quinn::ConnectionError,
        remote_address: SocketAddr,
    },
    #[error(transparent)]
    #[fatal(true)]
    Db(#[from] sneed::Error),
    #[error("server endpoint closed")]
    #[fatal(true)]
    ServerEndpointClosed,
}

pub(in crate::net) mod configure_client {
    use thiserror::Error;

    #[derive(Debug, Error)]
    pub(in crate::net) enum Inner {
        #[error(transparent)]
        NoInitialCipherSuite(
            #[from] quinn::crypto::rustls::NoInitialCipherSuite,
        ),
        #[error("rustls error")]
        Rustls(#[source] rustls::Error),
    }

    #[derive(Debug, Error)]
    #[error("failed to configure p2p client")]
    #[repr(transparent)]
    pub struct Error(#[source] Inner);

    impl<E> From<E> for Error
    where
        Inner: From<E>,
    {
        fn from(err: E) -> Self {
            Self(err.into())
        }
    }
}
pub use configure_client::Error as ConfigureClient;

#[derive(Debug, Error)]
pub enum ConnectPeer {
    #[error(transparent)]
    AlreadyConnected(#[from] AlreadyConnected),
    #[error("failed to commit db write txn")]
    DbCommit(#[source] Box<rwtxn::Commit>),
    #[error("database error")]
    DbPut(#[source] Box<db::Put>),
    #[error("failed to create db write txn")]
    DbWriteTxn(#[source] Box<env::WriteTxn>),
    #[error("quinn connect error")]
    QuinnConnect(#[from] quinn::ConnectError),
    /// Unspecified peer IP addresses cannot be connected to.
    /// `0.0.0.0` is one example of an "unspecified" IP.
    #[error("unspecified peer ip address (cannot connect to '{0}')")]
    UnspecfiedPeerIP(IpAddr),
}

impl From<db::Put> for ConnectPeer {
    fn from(err: db::Put) -> Self {
        Self::DbPut(Box::new(err))
    }
}

impl From<env::WriteTxn> for ConnectPeer {
    fn from(err: env::WriteTxn) -> Self {
        Self::DbWriteTxn(Box::new(err))
    }
}

impl From<rwtxn::Commit> for ConnectPeer {
    fn from(err: rwtxn::Commit) -> Self {
        Self::DbCommit(Box::new(err))
    }
}

#[derive(Debug, Error)]
pub enum DialKnownPeer {
    #[error("failed to connect to peer")]
    ConnectPeer(#[from] ConnectPeer),
    #[error("DNS resolution for hostname failed")]
    DnsResolve(#[source] std::io::Error),
}

#[allow(clippy::duplicated_attributes)]
#[derive(Debug, Error, Transitive)]
#[transitive(from(db::Put, db::Error))]
#[transitive(from(db::TryGet, db::Error))]
#[transitive(from(env::CreateDb, env::Error))]
#[transitive(from(env::OpenDb, env::Error))]
#[transitive(from(env::WriteTxn, env::Error))]
#[transitive(from(rwtxn::Commit, rwtxn::Error))]
pub enum Error {
    #[error(transparent)]
    AcceptConnection(#[from] <AcceptConnection as Split>::Fatal),
    #[error("accept error")]
    AcceptError,
    #[error(transparent)]
    AlreadyConnected(#[from] AlreadyConnected),
    #[error("bincode error")]
    Bincode(#[from] bincode::Error),
    #[error("failed to connect to peer ({peer_addr})")]
    ConnectPeer {
        peer_addr: crate::types::net::PeerAddress,
        source: ConnectPeer,
    },
    #[error(transparent)]
    ConfigureClient(#[from] ConfigureClient),
    #[error(transparent)]
    Db(#[from] db::Error),
    #[error("Database env error")]
    DbEnv(#[from] env::Error),
    #[error("Database write error")]
    DbWrite(#[from] rwtxn::Error),
    #[error(
        "Incompatible DB version ({}). Please clear the DB (`{}`) and re-sync",
        .version,
        .db_path.display()
    )]
    IncompatibleVersion { version: Version, db_path: PathBuf },
    #[error("quinn error")]
    Quinn(#[source] std::io::Error),
    #[error("peer connection not found for {0}")]
    MissingPeerConnection(SocketAddr),
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
