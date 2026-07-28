use sneed::{DbError, RwTxnError, env::error as env};
use thiserror::Error;
use transitive::Transitive;

use crate::{
    archive, mempool, net, state,
    types::{AmountOverflowError, AmountUnderflowError, proto},
};

pub mod mainchain_task {
    use sneed::EnvError;
    use thiserror::Error;

    use crate::{archive, types::proto};

    /// Error included in a response
    #[derive(Debug, Error)]
    pub enum Response {
        #[error("Archive error")]
        Archive(#[from] archive::Error),
        #[error("Database env error")]
        DbEnv(#[from] EnvError),
        #[error("Database write error")]
        DbWrite(#[from] sneed::rwtxn::Error),
        #[error("CUSF Mainchain proto error")]
        Mainchain(#[from] proto::Error),
    }
}

pub mod net_task {
    use std::sync::Arc;

    use futures::channel::{mpsc, oneshot};
    use sneed::{db::error as db, env::error as env, rwtxn::error as rwtxn};
    use thiserror::Error;
    use transitive::Transitive;

    use crate::{
        archive, mempool, net, node::error::mainchain_task, state, types::proto,
    };

    #[derive(Debug, Error)]
    pub enum MainchainAncestors {
        #[error("Requested block was not available: {block_hash}")]
        BlockNotAvailable { block_hash: bitcoin::BlockHash },
        #[error(transparent)]
        MainchainTaskResponse(#[from] Arc<mainchain_task::Response>),
    }

    #[allow(clippy::duplicated_attributes)]
    #[derive(Debug, Error, Transitive)]
    #[transitive(
        from(db::IterInit, db::Error),
        from(db::IterItem, db::Error),
        from(env::WriteTxn, env::Error),
        from(rwtxn::Commit, rwtxn::Error)
    )]
    pub enum Error {
        #[error("archive error")]
        Archive(#[from] archive::Error),
        #[error("CUSF mainchain proto error")]
        CusfMainchain(#[from] proto::Error),
        #[error(transparent)]
        Db(#[from] db::Error),
        #[error("Database env error")]
        DbEnv(#[from] env::Error),
        #[error("Database write error")]
        DbWrite(#[from] rwtxn::Error),
        #[error("Forward mainchain task request failed")]
        ForwardMainchainTaskRequest,
        #[error("mempool error")]
        MemPool(#[from] mempool::Error),
        #[error("Net error")]
        Net(#[from] Box<net::Error>),
        #[error("peer info stream closed")]
        PeerInfoRxClosed,
        #[error("Receive mainchain task response cancelled")]
        ReceiveMainchainTaskResponse,
        #[error("Receive reorg result cancelled (oneshot)")]
        ReceiveReorgResultOneshot(#[source] oneshot::Canceled),
        #[error("Send mainchain task request failed")]
        SendMainchainTaskRequest,
        #[error("Send new tip ready failed")]
        SendNewTipReady(#[source] mpsc::SendError),
        #[error("Send reorg result error (oneshot)")]
        SendReorgResultOneshot,
        #[error("state error")]
        State(#[from] state::Error),
    }

    impl From<net::Error> for Error {
        fn from(err: net::Error) -> Self {
            Self::Net(Box::new(err))
        }
    }
}

#[derive(Debug, Error, Transitive)]
#[transitive(from(env::ReadTxn, env::Error))]
pub enum Error {
    #[error("address parse error")]
    AddrParse(#[from] std::net::AddrParseError),
    #[error(transparent)]
    AmountOverflow(#[from] AmountOverflowError),
    #[error(transparent)]
    AmountUnderflow(#[from] AmountUnderflowError),
    #[error("archive error")]
    Archive(#[from] archive::Error),
    #[error("CUSF mainchain proto error")]
    CusfMainchain(#[from] proto::Error),
    #[error(transparent)]
    Db(#[from] DbError),
    #[error("Database env error")]
    DbEnv(#[from] env::Error),
    #[error("Database write error")]
    DbWrite(#[from] RwTxnError),
    #[error("I/O error")]
    Io(#[from] std::io::Error),
    #[error("error requesting mainchain ancestors")]
    MainchainAncestors(#[source] mainchain_task::Response),
    #[error("mempool error")]
    MemPool(#[from] mempool::Error),
    #[error("net error")]
    Net(#[from] Box<net::Error>),
    #[error("net task error")]
    NetTask(#[source] Box<net_task::Error>),
    #[error("No CUSF mainchain wallet client")]
    NoCusfMainchainWalletClient,
    #[error("peer info stream closed")]
    PeerInfoRxClosed,
    #[error("Receive mainchain task response cancelled")]
    ReceiveMainchainTaskResponse,
    #[error("Send mainchain task request failed")]
    SendMainchainTaskRequest,
    #[error("state error")]
    State(#[source] Box<state::Error>),
    #[error("Utreexo error: {0}")]
    Utreexo(String),
}

impl From<net::Error> for Error {
    fn from(err: net::Error) -> Self {
        Self::Net(Box::new(err))
    }
}

impl From<net_task::Error> for Error {
    fn from(err: net_task::Error) -> Self {
        Self::NetTask(Box::new(err))
    }
}

impl From<state::Error> for Error {
    fn from(err: state::Error) -> Self {
        Self::State(Box::new(err))
    }
}
