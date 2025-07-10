//! Tree construction operations for ArenaForest.

use smallvec::SmallVec;

use super::super::node_hash::AccumulatorHash;
use super::forest::ArenaForest;
use super::types::ArenaNode;

impl<Hash: AccumulatorHash + Send + Sync + std::ops::Deref<Target = [u8; 32]>> ArenaForest<Hash> {
    /// Add a single leaf to the forest.
    pub fn add_single(&mut self, hash: Hash) -> u32 {
        self.add_single_queued(hash)
    }

    /// Unified implementation for adding a single leaf.
    fn add_single_impl(&mut self, hash: Hash, use_queue: bool) -> u32 {
        // Create initial leaf node
        let leaf_node = ArenaNode::new_leaf(hash);
        let mut current_idx = self.allocate_node(leaf_node);

        // Hash-to-node insertion with collision handling
        let key = super::forest::hash_key(&hash);
        self.hash_to_node
            .entry(key)
            .or_insert_with(SmallVec::new)
            .push((hash, current_idx));

        let mut leaves = self.leaves;

        // Combine while leaves & 1 != 0 (utreexo algorithm)
        while leaves & 1 != 0 {
            // Pop root from end
            let root_idx = if let Some(root_option) = self.roots.pop() {
                if let Some(idx) = root_option {
                    // Check for empty root
                    if (idx as usize) < self.hashes.len()
                        && self.hashes[idx as usize] == Hash::empty()
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
            let root_hash = self.hashes[root_idx as usize];
            let current_hash = self.hashes[current_idx as usize];
            let parent_hash = Hash::parent_hash(&root_hash, &current_hash);

            let root_level = self.level[root_idx as usize];
            let current_level = self.level[current_idx as usize];
            let parent_level = std::cmp::max(root_level, current_level) + 1;

            // Create parent node with root=left, current=right
            let parent_idx = self.allocate_node(ArenaNode::new_internal(
                parent_hash,
                root_idx,
                current_idx,
                parent_level,
            ));

            // Maintain parent pointers
            self.parent[root_idx as usize] = parent_idx;
            self.parent[current_idx as usize] = parent_idx;

            // Handle dirty propagation based on strategy
            if use_queue {
                self.queue_for_dirty_propagation(parent_idx);
            } else {
                self.mark_dirty_ancestors(parent_idx);
            }

            current_idx = parent_idx;
            leaves >>= 1;
        }

        // Push final node as root
        self.roots.push(Some(current_idx));

        // Increment leaf count
        self.leaves += 1;

        current_idx
    }

    /// Add a single leaf with queue-based dirty propagation (for large workloads).
    fn add_single_queued(&mut self, hash: Hash) -> u32 {
        self.add_single_impl(hash, true)
    }

    /// Add a single leaf with direct dirty propagation (for small workloads).
    fn add_single_direct(&mut self, hash: Hash) -> u32 {
        self.add_single_impl(hash, false)
    }

    /// Modify the forest by adding new leaves and deleting existing ones.
    pub fn modify(&mut self, add: &[Hash], del: &[Hash]) -> Result<(), String> {
        // Use queue optimization for large operations
        let total_operations = add.len() + del.len();
        const QUEUE_THRESHOLD: usize = 32;

        if total_operations >= QUEUE_THRESHOLD {
            // Large workload: use queue-based processing
            self.modify_with_queue(add, del)
        } else {
            // Small workload: use direct marking
            self.modify_direct(add, del)
        }
    }

    /// Modify with queue-based batch processing (for large workloads)
    fn modify_with_queue(&mut self, add: &[Hash], del: &[Hash]) -> Result<(), String> {
        // Process deletions first
        self.delete_leaves(del)?;

        // Add new leaves with queue
        for &hash in add {
            self.add_single_queued(hash);
        }

        // Flush any remaining zombies
        if !self.zombies.is_empty() {
            self.flush_zombies()?;
        }

        // Flush dirty queue before hash recomputation
        self.flush_dirty_queue()?;

        // Recompute dirty hashes
        self.recompute_dirty_hashes_adaptive();

        Ok(())
    }

    /// Modify with direct marking (for small workloads)
    fn modify_direct(&mut self, add: &[Hash], del: &[Hash]) -> Result<(), String> {
        // Process deletions first
        self.delete_leaves(del)?;

        // Add new leaves with direct marking
        for &hash in add {
            self.add_single_direct(hash);
        }

        // Flush any remaining zombies
        if !self.zombies.is_empty() {
            self.flush_zombies()?;
        }

        // Recompute dirty hashes
        self.recompute_dirty_hashes_adaptive();

        Ok(())
    }
}
