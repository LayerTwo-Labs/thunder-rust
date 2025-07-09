//! Proof generation and verification for the arena forest.

use super::super::node_hash::AccumulatorHash;
use super::forest::ArenaForest;

impl<Hash: AccumulatorHash> ArenaForest<Hash> {
    /// Generate proofs for target hashes.
    pub fn prove(&self, targets: &[Hash]) -> Result<super::super::proof::Proof<Hash>, String> {
        let mut positions = Vec::new();

        if targets.is_empty() {
            return Ok(super::super::proof::Proof::new_with_hash(
                Vec::new(),
                Vec::new(),
            ));
        }

        // Get positions for all target hashes
        for target in targets {
            let node_idx = self.hash_to_node.get(target).ok_or_else(|| {
                format!(
                    "Could not find target hash: {:?}. Available hashes: {} total, leaves: {}",
                    target,
                    self.hash_to_node.len(),
                    self.leaves
                )
            })?;

            let position = self.get_pos(*node_idx).map_err(|e| {
                format!("Could not calculate position for hash {:?}: {}", target, e)
            })?;
            positions.push(position);
        }

        // Calculate proof positions
        let tree_height = super::super::util::tree_rows(self.leaves);
        let needed_positions =
            super::super::util::get_proof_positions(&positions, self.leaves, tree_height);

        let mut proof_hashes = Vec::new();
        for pos in needed_positions {
            let hash = self
                .get_hash_at_position(pos)
                .ok_or_else(|| format!("Could not get hash at position {}", pos))?;
            proof_hashes.push(hash);
        }

        Ok(super::super::proof::Proof::new_with_hash(
            positions,
            proof_hashes,
        ))
    }

    /// Verify proofs against the accumulator.
    pub fn verify(
        &self,
        proof: &super::super::proof::Proof<Hash>,
        del_hashes: &[Hash],
    ) -> Result<bool, String> {
        // Get current root hashes
        let roots = self
            .get_roots()
            .iter()
            .map(|root_wrapper| root_wrapper.get_data())
            .collect::<Vec<_>>();

        proof.verify(del_hashes, &roots, self.leaves)
    }
}
