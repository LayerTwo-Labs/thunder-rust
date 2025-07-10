//! Proof generation and verification for the arena forest.

use super::super::node_hash::AccumulatorHash;
use super::forest::ArenaForest;

impl<Hash: AccumulatorHash + std::ops::Deref<Target = [u8; 32]>> ArenaForest<Hash> {
    /// Generate proofs for target hashes.
    pub fn prove(&self, targets: &[Hash]) -> Result<super::super::proof::Proof<Hash>, String> {
        let mut positions = Vec::new();

        if targets.is_empty() {
            return Ok(super::super::proof::Proof::new_with_hash(
                Vec::new(),
                Vec::new(),
            ));
        }

        for target in targets {
            let key = super::forest::hash_key(target);
            let node_idx = self
                .hash_to_node
                .get(&key)
                .and_then(|entries| {
                    entries
                        .iter()
                        .find(|(stored_hash, _)| stored_hash == target)
                        .map(|(_, idx)| *idx)
                })
                .ok_or_else(|| format!("Target hash not found: {:?}", target))?;

            let position = self
                .get_pos(node_idx)
                .map_err(|e| format!("Position calculation failed: {}", e))?;
            positions.push(position);
        }

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
        let roots = self
            .get_roots()
            .iter()
            .map(|root_wrapper| root_wrapper.get_data())
            .collect::<Vec<_>>();

        proof.verify(del_hashes, &roots, self.leaves)
    }
}
