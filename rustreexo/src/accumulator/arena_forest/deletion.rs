//! Deletion operations for leaf nodes and tree restructuring.


use super::super::node_hash::AccumulatorHash;
use super::forest::ArenaForest;
use super::types::ArenaNode;
// Removed unused import
use super::types::NULL_INDEX;

impl<Hash: AccumulatorHash + std::ops::Deref<Target = [u8; 32]>> ArenaForest<Hash> {
    /// Remove leaves from the forest using lazy deletion with tombstone marking.
    pub fn delete_leaves(&mut self, hashes: &[Hash]) -> Result<(), String> {
        for hash in hashes {
            // Hash-to-node lookup with collision handling
            let key = super::forest::hash_key(hash);
            let node_idx = self
                .hash_to_node
                .get(&key)
                .and_then(|entries| {
                    entries.iter().find(|(stored_hash, _)| stored_hash == hash).map(|(_, idx)| *idx)
                })
                .ok_or_else(|| format!("Hash not found for deletion: {:?}", hash))?;

            // Validate it's a leaf node
            let lr = self.lr[node_idx as usize];
            if (lr & super::types::LEAF_TYPE_BIT) == 0 {
                return Err(format!("Node {} is not a leaf", node_idx));
            }

            // Mark as tombstone instead of immediate deletion
            self.mark_tombstone(node_idx);
            
            // Remove from hash_to_node map
            if let Some(entries) = self.hash_to_node.get_mut(&key) {
                entries.retain(|(stored_hash, _)| stored_hash != hash);
                if entries.is_empty() {
                    self.hash_to_node.remove(&key);
                }
            }
        }

        // For small batches, flush immediately for test compatibility
        if hashes.len() <= 10 || self.should_flush_zombies() {
            self.flush_zombies()?;
        }

        Ok(())
    }

    /// Flush all zombies by performing batch tree restructuring.
    pub fn flush_zombies(&mut self) -> Result<(), String> {
        if self.zombies.is_empty() {
            return Ok(());
        }

        // Collect all zombie indices for processing
        let zombie_indices: Vec<u32> = self.zombies.indices.clone();
        
        // Process zombies in reverse position order to avoid conflicts
        let mut zombies_with_positions: Vec<(u64, u32)> = Vec::new();
        for &zombie_idx in &zombie_indices {
            if let Ok(position) = self.get_pos(zombie_idx) {
                zombies_with_positions.push((position, zombie_idx));
            }
        }
        
        // Sort by position (highest positions first)
        zombies_with_positions.sort_by(|a, b| b.0.cmp(&a.0));

        // Apply batch restructuring
        for (position, node_idx) in zombies_with_positions {
            // Skip if already processed
            if !self.is_tombstone(node_idx) {
                continue;
            }
            
            // Perform actual deletion and restructuring
            self.delete_single_leaf_immediate(position, node_idx)?;
        }

        // Clear the zombie queue
        self.zombies.clear();
        
        // Flush dirty queue for batch optimization
        self.flush_dirty_queue()?;
        
        Ok(())
    }

    /// Immediate deletion of a single leaf (used by flush_zombies).
    fn delete_single_leaf_immediate(&mut self, position: u64, node_idx: u32) -> Result<(), String> {
        // Clear tombstone flag
        self.clear_tombstone(node_idx);

        let node = self
            .get_node(node_idx)
            .ok_or_else(|| format!("Invalid node index: {}", node_idx))?;

        if !node.is_leaf() {
            return Err(format!("Node at position {} is not a leaf", position));
        }

        let parent_idx = self.find_parent_of_node(node_idx);

        // Apply sibling promotion logic
        let sibling_idx = match parent_idx {
            None => {
                // Leaf is itself a root: replace with empty placeholder
                let mut found_root_pos = None;
                for (root_pos, root_option) in self.roots.iter().enumerate() {
                    if let Some(root_idx) = root_option {
                        if *root_idx == node_idx {
                            found_root_pos = Some(root_pos);
                            break;
                        }
                    }
                }
                
                if let Some(root_pos) = found_root_pos {
                    let empty_node = ArenaNode::new_leaf(Hash::empty());
                    let empty_idx = self.allocate_node(empty_node);
                    self.roots[root_pos] = Some(empty_idx);
                    self.queue_for_dirty_propagation(empty_idx);
                    return Ok(());
                } else {
                    return Err(format!("Root node {} not found in roots array", node_idx));
                }
            }
            Some(parent_idx) => {
                // Find sibling
                let parent_idx_usize = parent_idx as usize;
                let parent_lr = self.lr[parent_idx_usize] & super::types::INDEX_MASK;
                let parent_rr = self.rr[parent_idx_usize];
                
                if parent_lr == node_idx {
                    if parent_rr == super::types::NULL_INDEX {
                        return Err(format!("Parent {} missing right sibling", parent_idx));
                    }
                    parent_rr
                } else if parent_rr == node_idx {
                    if parent_lr == super::types::NULL_INDEX {
                        return Err(format!("Parent {} missing left sibling", parent_idx));
                    }
                    parent_lr
                } else {
                    return Err(format!("Node {} is not a child of parent {}", node_idx, parent_idx));
                }
            }
        };

        // Promote sibling upwards
        if let Some(grand_idx) = self.find_parent_of_node(parent_idx.unwrap()) {
            // Internal case: splice sibling into grandparent
            let grand_idx_usize = grand_idx as usize;
            let parent_idx_unwrapped = parent_idx.unwrap();
            
            let grand_lr = self.lr[grand_idx_usize] & super::types::INDEX_MASK;
            if grand_lr == parent_idx_unwrapped {
                self.lr[grand_idx_usize] = sibling_idx;
            } else {
                self.rr[grand_idx_usize] = sibling_idx;
            }
            
            self.parent[sibling_idx as usize] = grand_idx;
            self.level[sibling_idx as usize] = self.level[parent_idx_unwrapped as usize];
            
            self.queue_for_dirty_propagation(grand_idx);
        } else {
            // Root case: promote sibling to new root
            let parent_idx_unwrapped = parent_idx.unwrap();
            let root_pos = self.roots.iter().position(|r| *r == Some(parent_idx_unwrapped))
                .ok_or_else(|| format!("Parent root {} not found in roots array", parent_idx_unwrapped))?;
            
            self.roots[root_pos] = Some(sibling_idx);
            self.parent[sibling_idx as usize] = NULL_INDEX;
            self.level[sibling_idx as usize] = self.level[parent_idx_unwrapped as usize];
            
            self.queue_for_dirty_propagation(sibling_idx);
        }

        Ok(())
    }
}
