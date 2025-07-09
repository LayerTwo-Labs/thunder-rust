//! Tree construction operations for ArenaForest.

use super::super::node_hash::AccumulatorHash;
use super::forest::ArenaForest;
use super::types::ArenaNode;

impl<Hash: AccumulatorHash + Send + Sync> ArenaForest<Hash> {
    /// Add a single leaf to the forest.
    /// Implements arena-based leaf addition with dirty tracking following the utreexo algorithm.
    pub fn add_single(&mut self, hash: Hash) -> u32 {
        // Create initial leaf node
        let leaf_node = ArenaNode::new_leaf(hash);
        let mut current_idx = self.allocate_node(leaf_node);

        self.hash_to_node.insert(hash, current_idx);

        let mut leaves = self.leaves;

        // Combine while leaves & 1 != 0 (utreexo algorithm)
        while leaves & 1 != 0 {
            // Pop root from end
            let root_idx = if let Some(root_option) = self.roots.pop() {
                if let Some(idx) = root_option {
                    // Check for empty root
                    if (idx as usize) < self.nodes.len()
                        && self.nodes[idx as usize].hash == Hash::empty()
                    {
                        leaves >>= 1;
                        continue;
                    }
                    idx
                } else {
                    // Skip None entries
                    leaves >>= 1;
                    continue;
                }
            } else {
                // No more roots
                break;
            };

            // Create parent: parent_hash(root, current)
            let root_hash = self.nodes[root_idx as usize].hash;
            let current_hash = self.nodes[current_idx as usize].hash;
            let parent_hash = Hash::parent_hash(&root_hash, &current_hash);

            let root_level = self.nodes[root_idx as usize].level;
            let current_level = self.nodes[current_idx as usize].level;
            let parent_level = std::cmp::max(root_level, current_level) + 1;

            // Create parent node with root=left, current=right
            let parent_idx = self.allocate_node(ArenaNode::new_internal(
                parent_hash,
                root_idx,
                current_idx,
                parent_level,
            ));

            // Maintain parent pointers for O(1) lookup
            self.nodes[root_idx as usize].parent = parent_idx;
            self.nodes[current_idx as usize].parent = parent_idx;

            self.mark_dirty(parent_idx, parent_level as usize);

            current_idx = parent_idx;
            leaves >>= 1;
        }

        // Push final node as root
        self.roots.push(Some(current_idx));

        // Increment leaf count
        self.leaves += 1;

        current_idx
    }

    /// Modify the forest by adding new leaves and deleting existing ones.
    /// This is the main public interface for forest modifications.
    pub fn modify(&mut self, add: &[Hash], del: &[Hash]) -> Result<(), String> {
        // Process deletions first
        self.delete_leaves(del)?;

        // Add new leaves
        for &hash in add {
            self.add_single(hash);
        }

        // Recompute dirty hashes
        self.recompute_dirty_hashes_adaptive();

        Ok(())
    }
}
