use borsh::BorshSerialize;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::{
    hashes::{self, MerkleRoot},
    transaction::outputs::{self, Outputs},
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
    ) -> Result<MerkleRoot, outputs::error::ComputeMerkleRoot> {
        let Self { memo, outputs } = self;
        let outputs_commitment = outputs.compute_merkle_root()?;
        Ok(hashes::hash(&(memo, outputs_commitment)).into())
    }
}
