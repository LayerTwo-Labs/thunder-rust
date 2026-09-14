use std::cmp::Ordering;

use borsh::BorshSerialize;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::{
    MerkleRoot,
    hashes::{self, Hash},
    transaction::{GetValue, output::Output},
    util,
};

pub mod error {
    use thiserror::Error;

    use crate::error::AmountOverflow;

    #[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
    pub enum MergeCbmtNodes {
        #[error("canonical size overflow")]
        SizeOverflow,
        #[error(transparent)]
        ValueOverflow(#[from] AmountOverflow),
    }

    #[derive(Debug, Error)]
    pub(crate) enum ComputeMerkleRoot {
        #[error("failed to merge CBMT nodes")]
        MergeCbmtNodes(#[from] MergeCbmtNodes),
        #[error("failed to compute canonical size for output ({index})")]
        TxCanonicalSize {
            index: usize,
            source: borsh::io::Error,
        },
    }
}

/// Hash to get a [`CmbtNode`] inner commitment for a leaf value
#[derive(BorshSerialize, Debug)]
struct CbmtLeafPreCommitment<'a> {
    #[borsh(serialize_with = "util::borsh::serialize::bitcoin_amount")]
    value: bitcoin::Amount,
    canonical_size: u64,
    output: &'a Output,
}

/// Hash to get a [`CmbtNode`] inner commitment for a non-leaf value
#[derive(BorshSerialize, Debug)]
struct CbmtNodePreCommitment {
    /// left child inner commitment
    left_commitment: Hash,
    /// Sum of child values
    #[borsh(serialize_with = "util::borsh::serialize::bitcoin_amount")]
    value: bitcoin::Amount,
    /// Sum of canonical sizes of children
    canonical_size: u64,
    /// right child inner commitment
    right_commitment: Hash,
}

// Internal node of a CBMT
#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct CbmtNode {
    // Commitment to child nodes or leaf value
    commitment: Hash,
    // Sum of values for child nodes or leaf value
    value: bitcoin::Amount,
    // Sum of canonical sizes for child nodes or leaf value
    canonical_size: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CmbtNodeResult {
    value: Result<CbmtNode, error::MergeCbmtNodes>,
    // CBT index, see https://github.com/nervosnetwork/merkle-tree/blob/5d1898263e7167560fdaa62f09e8d52991a1c712/README.md#tree-struct
    // This is required so that `CbmtNode` can be `Ord` correctly
    index: usize,
}

impl Default for CmbtNodeResult {
    fn default() -> Self {
        Self {
            value: Ok(CbmtNode::default()),
            index: 0,
        }
    }
}

impl PartialOrd for CmbtNodeResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CmbtNodeResult {
    fn cmp(&self, other: &Self) -> Ordering {
        self.index.cmp(&other.index)
    }
}

// Marker type for merging branch commitments with
// * branch value totals
// * branch canonical size totals
struct MergeValueSizeTotal;

impl merkle_cbt::merkle_tree::Merge for MergeValueSizeTotal {
    type Item = CmbtNodeResult;

    fn merge(lnode: &Self::Item, rnode: &Self::Item) -> Self::Item {
        // see https://github.com/nervosnetwork/merkle-tree/blob/5d1898263e7167560fdaa62f09e8d52991a1c712/README.md#tree-struct
        assert_eq!(lnode.index + 1, rnode.index);
        let index = (lnode.index - 1) / 2;
        let lnode = match lnode.value.as_ref() {
            Ok(lnode) => lnode,
            Err(err) => {
                return CmbtNodeResult {
                    value: Err(*err),
                    index,
                };
            }
        };
        let rnode = match rnode.value.as_ref() {
            Ok(rnode) => rnode,
            Err(err) => {
                return CmbtNodeResult {
                    value: Err(*err),
                    index,
                };
            }
        };
        let Some(value) = lnode.value.checked_add(rnode.value) else {
            return CmbtNodeResult {
                value: Err(crate::error::AmountOverflow.into()),
                index,
            };
        };
        let Some(canonical_size) =
            lnode.canonical_size.checked_add(rnode.canonical_size)
        else {
            return CmbtNodeResult {
                value: Err(error::MergeCbmtNodes::SizeOverflow),
                index,
            };
        };
        let commitment = hashes::hash(&CbmtNodePreCommitment {
            left_commitment: lnode.commitment,
            value,
            canonical_size,
            right_commitment: rnode.commitment,
        });
        CmbtNodeResult {
            value: Ok(CbmtNode {
                commitment,
                value,
                canonical_size,
            }),
            index,
        }
    }
}

// Complete binary merkle tree with annotated value and canonical size totals
type CbmtWithValueSizeTotal =
    merkle_cbt::CBMT<CmbtNodeResult, MergeValueSizeTotal>;

#[derive(
    BorshSerialize, Clone, Debug, Default, Deserialize, Serialize, ToSchema,
)]
#[repr(transparent)]
#[serde(transparent)]
pub struct Outputs(pub Vec<Output>);

impl Outputs {
    #[inline(always)]
    pub fn as_slice(&self) -> &[Output] {
        self.0.as_slice()
    }

    pub(crate) fn compute_merkle_root(
        &self,
    ) -> Result<MerkleRoot, error::ComputeMerkleRoot> {
        let CbmtNode { commitment, .. } = {
            let n_outputs = self.len();
            let leaves: Vec<CmbtNodeResult> = self
                .iter()
                .enumerate()
                .map(|(idx, output)| -> Result<_, error::ComputeMerkleRoot> {
                    let value = output.get_value();
                    let canonical_size =
                        output.canonical_size().map_err(|err| {
                            error::ComputeMerkleRoot::TxCanonicalSize {
                                index: idx,
                                source: err,
                            }
                        })?;
                    let leaf_pre_commitment = CbmtLeafPreCommitment {
                        value,
                        canonical_size,
                        output,
                    };
                    Ok(CmbtNodeResult {
                        value: Ok(CbmtNode {
                            commitment: hashes::hash(&leaf_pre_commitment),
                            value,
                            canonical_size,
                        }),
                        // see https://github.com/nervosnetwork/merkle-tree/blob/5d1898263e7167560fdaa62f09e8d52991a1c712/README.md#tree-struct
                        index: (idx + n_outputs) - 1,
                    })
                })
                .collect::<Result<_, _>>()?;
            CbmtWithValueSizeTotal::build_merkle_root(leaves.as_slice()).value?
        };
        Ok(commitment.into())
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    #[inline(always)]
    pub fn iter(&self) -> std::slice::Iter<'_, Output> {
        self.0.iter()
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[inline(always)]
    pub fn push(&mut self, output: Output) {
        self.0.push(output)
    }

    #[inline(always)]
    pub fn remove(&mut self, index: usize) -> Output {
        self.0.remove(index)
    }
}

impl From<Vec<Output>> for Outputs {
    #[inline(always)]
    fn from(outputs: Vec<Output>) -> Self {
        Self(outputs)
    }
}

impl IntoIterator for Outputs {
    type IntoIter = <Vec<Output> as IntoIterator>::IntoIter;
    type Item = Output;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a> IntoIterator for &'a Outputs {
    type IntoIter = <&'a Vec<Output> as IntoIterator>::IntoIter;
    type Item = &'a Output;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}
