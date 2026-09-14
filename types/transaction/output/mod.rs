use borsh::BorshSerialize;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::{
    address::Address,
    hashes::{UtreexoNodeHash, hash},
    transaction::{GetValue, outpoint::OutPoint},
};

mod content;
pub use content::Content;

#[derive(
    BorshSerialize,
    Clone,
    Debug,
    Deserialize,
    Eq,
    PartialEq,
    Serialize,
    ToSchema,
)]
pub struct Output {
    pub address: Address,
    pub content: Content,
}

impl GetValue for Output {
    #[inline(always)]
    fn get_value(&self) -> bitcoin::Amount {
        self.content.get_value()
    }
}

#[derive(
    BorshSerialize,
    Clone,
    Debug,
    Deserialize,
    Eq,
    PartialEq,
    Serialize,
    ToSchema,
)]
pub struct Pointed<Output = crate::transaction::output::Output> {
    pub outpoint: OutPoint,
    pub output: Output,
}

impl From<&Pointed> for UtreexoNodeHash {
    fn from(pointed_output: &Pointed) -> Self {
        Self::new(hash(pointed_output))
    }
}

/// Useful when computing hashes for Utreexo,
/// without needing to clone an output
#[derive(BorshSerialize, Clone, Copy, Debug)]
pub struct PointedOutputRef<'a> {
    pub outpoint: OutPoint,
    pub output: &'a Output,
}

impl From<PointedOutputRef<'_>> for UtreexoNodeHash {
    fn from(pointed_output: PointedOutputRef) -> Self {
        Self::new(hash(&pointed_output))
    }
}
