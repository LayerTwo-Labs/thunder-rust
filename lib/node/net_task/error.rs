use fatality::{Fatality, Split, fatality};
use futures::channel::{mpsc, oneshot};
use sneed::{db::error as db, env::error as env, rwtxn::error as rwtxn};
use thiserror::Error;
use transitive::Transitive;

use crate::{
    archive, mempool, net, node::net_task, state::error as state, types::proto,
    util::FatalitySplitWrappers,
};

#[fatality(splitable)]
pub(in crate::node::net_task) enum ConnectTip {
    #[error(transparent)]
    #[fatal]
    Archive(#[from] archive::Error),
    #[error(transparent)]
    #[fatal(forward)]
    ConnectBlock(#[from] state::ConnectBlock),
    #[error(transparent)]
    #[fatal]
    Connect2wpd(#[from] state::Connect2wpd),
    #[error(transparent)]
    #[fatal]
    DbTryGet(#[from] Box<db::TryGet>),
    #[error(transparent)]
    #[fatal]
    Mempool(#[from] mempool::Error),
    #[error(transparent)]
    #[fatal(forward)]
    ValidateBlock(#[from] state::ValidateBlock),
}

impl From<db::TryGet> for ConnectTip {
    fn from(err: db::TryGet) -> Self {
        Self::DbTryGet(Box::new(err))
    }
}

#[derive(Debug, Error, Transitive)]
#[transitive(
    from(db::IterInit, db::Error),
    from(db::IterItem, db::Error),
    from(db::TryGet, db::Error),
    from(state::NoTip, state::Error)
)]
pub(in crate::node::net_task) enum DisconnectTip {
    #[error(transparent)]
    Archive(#[from] archive::Error),
    #[error(transparent)]
    Db(#[from] Box<db::Error>),
    #[error(transparent)]
    Disconnect2wpd(#[from] state::Disconnect2wpd),
    #[error(transparent)]
    DisconnectTip(#[from] state::DisconnectTip),
    #[error(transparent)]
    Mempool(#[from] mempool::Error),
    #[error("state error")]
    State(#[from] state::Error),
}

impl From<db::Error> for DisconnectTip {
    fn from(err: db::Error) -> Self {
        Self::Db(Box::new(err))
    }
}

#[derive(Transitive)]
#[fatality(splitable)]
#[transitive(from(db::TryGet, db::Error))]
pub(in crate::node::net_task) enum ReorgToTipInner {
    #[error(transparent)]
    #[fatal]
    Archive(#[from] archive::Error),
    #[error("failed to connect tip")]
    #[fatal(forward)]
    ConnectTip(#[from] ConnectTip),
    #[error("failed to disconnect tip")]
    DisconnectTip(#[from] DisconnectTip),
    #[error(transparent)]
    #[fatal]
    Db(#[from] Box<db::Error>),
    #[error("Database commit error")]
    #[fatal]
    DbCommit(#[from] rwtxn::Commit),
    #[error("Failed to open database write txn")]
    #[fatal]
    DbWrite(#[from] env::WriteTxn),
}

impl From<db::Error> for ReorgToTipInner {
    fn from(err: db::Error) -> Self {
        Self::Db(Box::new(err))
    }
}

#[derive(Debug, Error)]
#[error("failed to reorg to tip")]
#[repr(transparent)]
pub(in crate::node::net_task) struct ReorgToTipWrapper<Inner>(
    #[from] pub(in crate::node::net_task) Inner,
);

impl Fatality for ReorgToTipWrapper<ReorgToTipInner> {
    fn is_fatal(&self) -> bool {
        self.0.is_fatal()
    }
}

impl Split for ReorgToTipWrapper<ReorgToTipInner> {
    type Fatal = ReorgToTipWrapper<<ReorgToTipInner as Split>::Fatal>;
    type Jfyi = ReorgToTipWrapper<<ReorgToTipInner as Split>::Jfyi>;
    fn split(self) -> std::result::Result<Self::Jfyi, Self::Fatal> {
        match self.0.split() {
            Ok(jfyi) => Ok(jfyi.into()),
            Err(fatal) => Err(fatal.into()),
        }
    }
}

FatalitySplitWrappers!(
    pub ReorgToTip,
    pub(in crate::node::net_task) ReorgToTipWrapper<ReorgToTipInner>
);

impl<Err> From<Err> for ReorgToTip
where
    ReorgToTipInner: From<Err>,
{
    fn from(err: Err) -> Self {
        Self(ReorgToTipWrapper(err.into()))
    }
}

#[derive(Debug, Error, Transitive)]
#[transitive(
    from(db::IterInit, db::Error),
    from(db::IterItem, db::Error),
    from(db::TryGet, db::Error),
    from(state::NoTip, state::Error)
)]
pub enum Error {
    #[error(transparent)]
    Archive(#[from] archive::Error),
    #[error("CUSF mainchain proto error")]
    CusfMainchain(#[from] proto::Error),
    #[error(transparent)]
    Db(#[from] Box<db::Error>),
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
    #[error(transparent)]
    ReorgToTip(#[from] <ReorgToTip as Split>::Fatal),
    #[error("Receive mainchain task response cancelled")]
    ReceiveMainchainTaskResponse,
    #[error("Receive reorg result cancelled (oneshot)")]
    ReceiveReorgResultOneshot(#[source] oneshot::Canceled),
    #[error("Send mainchain task request failed")]
    SendMainchainTaskRequest,
    #[error("Send new tip ready failed")]
    SendNewTipReady(#[source] mpsc::TrySendError<net_task::NewTipReadyMessage>),
    #[error("Send reorg result error (oneshot)")]
    SendReorgResultOneshot,
    #[error("state error")]
    State(#[from] state::Error),
}

impl From<db::Error> for Error {
    fn from(err: db::Error) -> Self {
        Self::Db(Box::new(err))
    }
}

impl From<net::Error> for Error {
    fn from(err: net::Error) -> Self {
        Self::Net(Box::new(err))
    }
}
