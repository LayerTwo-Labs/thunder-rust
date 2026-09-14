use borsh::BorshSerialize;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::{
    hashes::{self, BlockHash, MerkleRoot, UtreexoNodeHash},
    schema, util,
};

pub mod body;
pub use body::Body;

#[derive(
    BorshSerialize,
    Clone,
    Debug,
    Deserialize,
    Eq,
    Hash,
    PartialEq,
    Serialize,
    ToSchema,
)]
pub struct Header {
    pub merkle_root: MerkleRoot,
    pub prev_side_hash: Option<BlockHash>,
    #[borsh(serialize_with = "util::borsh::serialize::bitcoin_block_hash")]
    #[schema(value_type = schema::BitcoinBlockHash)]
    pub prev_main_hash: bitcoin::BlockHash,
    /// Utreexo roots
    #[borsh(serialize_with = "util::borsh::serialize::utreexo_roots")]
    #[schema(value_type = Vec<schema::UtreexoNodeHash>)]
    pub roots: Vec<UtreexoNodeHash>,
}

impl Header {
    pub fn hash(&self) -> BlockHash {
        hashes::hash_with_scratch_buffer(self).into()
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct Block {
    pub header: Header,
    pub body: Body,
}
