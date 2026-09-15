use borsh::BorshSerialize;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::{
    hashes::{self, BlockHash, CoinbaseMerkleRoot, CoinbaseTxid, MerkleRoot},
    transaction::outputs::{self, Outputs},
    util,
};

#[derive(
    BorshSerialize, Clone, Debug, Default, Deserialize, Serialize, ToSchema,
)]
pub struct Coinbase {
    pub memo: Vec<u8>,
    pub outputs: Outputs,
}

impl Coinbase {
    pub(crate) fn compute_merkle_root(
        &self,
    ) -> Result<CoinbaseMerkleRoot, outputs::error::ComputeMerkleRoot> {
        let Self { memo, outputs } = self;
        let outputs_commitment = outputs.compute_merkle_root()?;
        Ok(hashes::hash(&(memo, outputs_commitment)).into())
    }

    /// [`CoinbaseTxid`]s are computed by hashing the concatenation of
    /// * The merkle root of the block that contains the coinbase tx
    /// * The previous mainchain hash for the block that contains the
    ///   coinbase tx
    /// * The previous sidechain hash for the block that contains the
    ///   coinbase tx
    pub fn compute_txid(
        merkle_root: &MerkleRoot,
        prev_main_hash: &bitcoin::BlockHash,
        prev_side_hash: Option<&BlockHash>,
    ) -> CoinbaseTxid {
        // Borsh encoding for hashing
        #[derive(BorshSerialize)]
        struct HashComponents<'a> {
            merkle_root: &'a MerkleRoot,
            #[borsh(
                serialize_with = "util::borsh::serialize::bitcoin_block_hash"
            )]
            prev_main_hash: &'a bitcoin::BlockHash,
            prev_side_hash: Option<&'a BlockHash>,
        }

        let digest = hashes::hash_with_scratch_buffer(&HashComponents {
            merkle_root,
            prev_main_hash,
            prev_side_hash,
        });
        CoinbaseTxid(digest)
    }
}
