use sneed::{db::error as db, env::error as env};
use thiserror::Error;
use transitive::Transitive;

#[derive(Debug, Error)]
pub enum Create {
    #[error(transparent)]
    CreateDb(#[from] env::CreateDb),
    #[error(transparent)]
    DbPut(#[from] db::Put),
    #[error(transparent)]
    DbTryGet(#[from] db::TryGet),
}

#[derive(Debug, Error)]
#[error("missing sidechain parent ({0})")]
#[repr(transparent)]
pub struct MissingSidechainParent(pub(crate) crate::types::BlockHash);

#[derive(Debug, Error)]
pub enum ConnectSidechainTip {
    #[error(transparent)]
    DbPut(Box<db::Put>),
    #[error(transparent)]
    DbTryGet(Box<db::TryGet>),
    #[error(transparent)]
    MissingSidechainParent(#[from] MissingSidechainParent),
}

impl From<db::Put> for ConnectSidechainTip {
    fn from(err: db::Put) -> Self {
        Self::DbPut(Box::new(err))
    }
}

impl From<db::TryGet> for ConnectSidechainTip {
    fn from(err: db::TryGet) -> Self {
        Self::DbTryGet(Box::new(err))
    }
}

#[derive(Debug, Error)]
pub enum DisconnectSidechainTip {
    #[error(transparent)]
    DbDelete(Box<db::Delete>),
    #[error(transparent)]
    DbTryGet(Box<db::TryGet>),
}

impl From<db::Delete> for DisconnectSidechainTip {
    fn from(err: db::Delete) -> Self {
        Self::DbDelete(Box::new(err))
    }
}

impl From<db::TryGet> for DisconnectSidechainTip {
    fn from(err: db::TryGet) -> Self {
        Self::DbTryGet(Box::new(err))
    }
}

#[derive(Debug, Error)]
pub enum ConnectMainchainTip {
    #[error(transparent)]
    DbGet(Box<db::Get>),
    #[error(transparent)]
    DbPut(Box<db::Put>),
    #[error(transparent)]
    DbTryGet(Box<db::TryGet>),
    #[error("invalid mainchain parent, expected ({expected})")]
    InvalidMainchainParent { expected: bitcoin::BlockHash },
    #[error("invalid mainchain tip height, expected ({expected})")]
    InvalidTipHeight { expected: u32 },
    #[error(transparent)]
    MissingSidechainParent(MissingSidechainParent),
}

impl From<db::Get> for ConnectMainchainTip {
    fn from(err: db::Get) -> Self {
        Self::DbGet(Box::new(err))
    }
}

impl From<db::Put> for ConnectMainchainTip {
    fn from(err: db::Put) -> Self {
        Self::DbPut(Box::new(err))
    }
}

impl From<db::TryGet> for ConnectMainchainTip {
    fn from(err: db::TryGet) -> Self {
        Self::DbTryGet(Box::new(err))
    }
}

impl From<ConnectSidechainTip> for ConnectMainchainTip {
    fn from(err: ConnectSidechainTip) -> Self {
        match err {
            ConnectSidechainTip::DbPut(err) => Self::DbPut(err),
            ConnectSidechainTip::DbTryGet(err) => Self::DbTryGet(err),
            ConnectSidechainTip::MissingSidechainParent(err) => {
                Self::MissingSidechainParent(err)
            }
        }
    }
}

#[derive(Debug, Error)]
#[error("missing sidechain tip ({0})")]
#[repr(transparent)]
pub struct MissingSidechainTip(pub(crate) crate::types::BlockHash);

#[derive(Debug, Error)]
pub enum DisconnectMainchainTip {
    #[error(transparent)]
    DbDelete(Box<db::Delete>),
    #[error(transparent)]
    DbGet(Box<db::Get>),
    #[error(transparent)]
    DbPut(Box<db::Put>),
    #[error(transparent)]
    DbTryGet(Box<db::TryGet>),
    #[error("invalid mainchain parent, expected ({expected})")]
    InvalidMainchainParent { expected: bitcoin::BlockHash },
    #[error("invalid mainchain parent height, expected ({tip_height} - 1)")]
    InvalidMainchainParentHeight { tip_height: u32 },
    #[error(transparent)]
    MissingSidechainTip(#[from] MissingSidechainTip),
    #[error("no mainchain tip to disconnect")]
    NoMainchainTip,
}

impl From<db::Delete> for DisconnectMainchainTip {
    fn from(err: db::Delete) -> Self {
        Self::DbDelete(Box::new(err))
    }
}

impl From<db::Get> for DisconnectMainchainTip {
    fn from(err: db::Get) -> Self {
        Self::DbGet(Box::new(err))
    }
}

impl From<db::Put> for DisconnectMainchainTip {
    fn from(err: db::Put) -> Self {
        Self::DbPut(Box::new(err))
    }
}

impl From<db::TryGet> for DisconnectMainchainTip {
    fn from(err: db::TryGet) -> Self {
        Self::DbTryGet(Box::new(err))
    }
}

#[allow(clippy::duplicated_attributes)]
#[derive(Debug, Error, Transitive)]
#[transitive(
    from(db::Get, db::Error),
    from(db::Put, db::Error),
    from(db::TryGet, db::Error)
)]
pub enum Error {
    #[error("failed to connect mainchain tip ({tip})")]
    ConnectMainchainTip {
        tip: bitcoin::BlockHash,
        source: ConnectMainchainTip,
    },
    #[error("failed to connect sidechain tip")]
    ConnectSidechainTip(#[from] ConnectSidechainTip),
    #[error("failed to create dbs")]
    Create(#[from] Create),
    #[error("failed to disconnect mainchain tip")]
    DisconnectMainchainTip(#[source] Box<DisconnectMainchainTip>),
    #[error("failed to disconnect sidechain tip")]
    DisconnectSidechainTip(#[from] DisconnectSidechainTip),
    #[error(transparent)]
    Db(Box<db::Error>),
}

impl From<DisconnectMainchainTip> for Error {
    fn from(err: DisconnectMainchainTip) -> Self {
        Self::DisconnectMainchainTip(Box::new(err))
    }
}

impl From<db::Error> for Error {
    fn from(err: db::Error) -> Self {
        Self::Db(Box::new(err))
    }
}
