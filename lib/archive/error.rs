use std::path::PathBuf;

use sneed::{db::error as db, env::error as env, rwtxn::error as rwtxn};
use thiserror::Error;
use transitive::Transitive;

use crate::types::{BlockHash, Version};

#[allow(clippy::duplicated_attributes)]
#[derive(Debug, Error, Transitive)]
#[transitive(
    from(db::Delete, db::Error),
    from(db::Put, db::Error),
    from(db::TryGet, db::Error),
    from(env::CreateDb, env::Error)
)]
pub enum Error {
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
    #[error("invalid merkle root")]
    InvalidMerkleRoot,
    #[error("invalid previous side hash")]
    InvalidPrevSideHash,
    #[error("no accumulator for block {0}")]
    NoAccumulator(BlockHash),
    #[error("no ancestor with depth {depth} for block {block_hash}")]
    NoAncestor { block_hash: BlockHash, depth: u32 },
    #[error("no mainchain ancestor with depth {depth} for block {block_hash}")]
    NoMainAncestor {
        block_hash: bitcoin::BlockHash,
        depth: u32,
    },
    #[error("unknown block hash: {0}")]
    NoBlockHash(BlockHash),
    #[error("no BMM result with block {0}")]
    NoBmmResult(BlockHash),
    #[error("no block body with hash {0}")]
    NoBody(BlockHash),
    #[error("no deposits info for block {0}")]
    NoDepositsInfo(bitcoin::BlockHash),
    #[error("no header with hash {0}")]
    NoHeader(BlockHash),
    #[error("no height info for block hash {0}")]
    NoHeight(BlockHash),
    #[error("unknown mainchain block hash: {0}")]
    NoMainBlockHash(bitcoin::BlockHash),
    #[error("no mainchain block info for block hash {0}")]
    NoMainBlockInfo(bitcoin::BlockHash),
    #[error("no mainchain header info for block hash {0}")]
    NoMainHeaderInfo(bitcoin::BlockHash),
    #[error("no height info for mainchain block hash {0}")]
    NoMainHeight(bitcoin::BlockHash),
}
