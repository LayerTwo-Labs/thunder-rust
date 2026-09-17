use std::cmp::Ordering;

use borsh::BorshSerialize;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::{
    hashes::{self, Hash, InputsMerkleRoot},
    transaction::outpoint::OutPoint,
};

// Internal node of a CBMT
#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct CbmtNode {
    // Commitment to child nodes or leaf value
    commitment: Hash,
    // CBT index, see https://github.com/nervosnetwork/merkle-tree/blob/5d1898263e7167560fdaa62f09e8d52991a1c712/README.md#tree-struct
    // This is required so that `CbmtNode` can be `Ord` correctly
    index: usize,
}

impl PartialOrd for CbmtNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CbmtNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.index.cmp(&other.index)
    }
}

// Marker type for merging branch commitments
struct Merge;

impl merkle_cbt::merkle_tree::Merge for Merge {
    type Item = CbmtNode;

    fn merge(lnode: &Self::Item, rnode: &Self::Item) -> Self::Item {
        // see https://github.com/nervosnetwork/merkle-tree/blob/5d1898263e7167560fdaa62f09e8d52991a1c712/README.md#tree-struct
        assert_eq!(lnode.index + 1, rnode.index);
        let index = (lnode.index - 1) / 2;
        let commitment = hashes::hash(&(&lnode.commitment, &rnode.commitment));
        CbmtNode { commitment, index }
    }
}

// Complete binary merkle tree
type Cbmt = merkle_cbt::CBMT<CbmtNode, Merge>;

#[derive(BorshSerialize, Clone, Debug, Deserialize, Serialize, ToSchema)]
#[repr(transparent)]
#[serde(transparent)]
pub struct Inputs<Input>(pub Vec<Input>);

impl<Input> Inputs<Input> {
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    #[inline(always)]
    pub fn iter(&self) -> std::slice::Iter<'_, Input> {
        self.0.iter()
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[inline(always)]
    pub fn push(&mut self, input: Input) {
        self.0.push(input)
    }

    #[inline(always)]
    pub fn remove(&mut self, index: usize) -> Input {
        self.0.remove(index)
    }
}

impl Inputs<(OutPoint, Hash)> {
    fn merkle_leaves(&self) -> Vec<CbmtNode> {
        let n_inputs = self.len();
        self.iter()
            .enumerate()
            .map(|(idx, (outpoint, output_hash))| CbmtNode {
                commitment: hashes::hash(&(outpoint, output_hash)),
                // see https://github.com/nervosnetwork/merkle-tree/blob/5d1898263e7167560fdaa62f09e8d52991a1c712/README.md#tree-struct
                index: (idx + n_inputs) - 1,
            })
            .collect()
    }

    pub(crate) fn compute_merkle_root(&self) -> InputsMerkleRoot {
        let CbmtNode { commitment, .. } =
            { Cbmt::build_merkle_root(self.merkle_leaves().as_slice()) };
        commitment.into()
    }
}

impl<Input> Default for Inputs<Input> {
    #[inline(always)]
    fn default() -> Self {
        Self(Vec::default())
    }
}

impl<Input> From<Vec<Input>> for Inputs<Input> {
    #[inline(always)]
    fn from(inputs: Vec<Input>) -> Self {
        Self(inputs)
    }
}

impl<'a, Input> IntoIterator for &'a Inputs<Input> {
    type IntoIter = <&'a Vec<Input> as IntoIterator>::IntoIter;
    type Item = &'a Input;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

#[cfg(test)]
mod test {
    use crate::{
        hashes::Txid,
        transaction::{
            inputs::{Cbmt, Inputs},
            outpoint::OutPoint,
        },
    };

    #[test]
    fn test_merkle_proof() -> anyhow::Result<()> {
        use rand::{
            Rng as _, RngExt as _, SeedableRng as _, rngs::ChaCha20Rng,
        };
        let mut rng = ChaCha20Rng::from_rng(&mut rand::rng());
        let inputs: Inputs<_> = (0..10)
            .map(|_| {
                let mut txid = Txid([0; 32]);
                rng.fill_bytes(&mut txid.0);
                let outpoint = OutPoint::Regular {
                    txid,
                    vout: rng.next_u32(),
                };
                let mut output_hash = [0; 32];
                rng.fill_bytes(&mut output_hash);
                (outpoint, output_hash)
            })
            .collect::<Vec<_>>()
            .into();
        let merkle_leaves = inputs.merkle_leaves();
        let merkle_tree = Cbmt::build_merkle_tree(merkle_leaves.as_slice());
        let merkle_root = merkle_tree.root();
        anyhow::ensure!(
            merkle_root.commitment == inputs.compute_merkle_root().0
        );
        // select a random input
        let input_idx = rng.random_range(0..inputs.len());
        let input = merkle_leaves[input_idx].clone();
        let merkle_proof = merkle_tree
            .build_proof(&[input_idx as u32])
            .ok_or_else(|| anyhow::anyhow!("generating merkle proof failed"))?;
        if !merkle_proof.verify(&merkle_root, &[input]) {
            anyhow::bail!("verifying merkle proof failed")
        }
        Ok(())
    }
}
