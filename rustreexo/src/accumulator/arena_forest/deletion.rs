//! Deletion operations for leaf nodes and tree restructuring.

use super::super::node_hash::AccumulatorHash;
use super::forest::ArenaForest;
use super::types::ArenaNode;
use super::types::INDEX_MASK;
use super::types::NULL_INDEX;

impl<Hash: AccumulatorHash> ArenaForest<Hash> {
    /// Remove leaves from the forest with proper tree restructuring.
    pub fn delete_leaves(&mut self, hashes: &[Hash]) -> Result<(), String> {
        let mut nodes_to_delete = Vec::new();

        for hash in hashes {
            let node_idx = self
                .hash_to_node
                .get(hash)
                .ok_or_else(|| format!("Hash not found for deletion: {:?}", hash))?;

            let position = self
                .get_pos(*node_idx)
                .map_err(|e| format!("Could not calculate position for hash {:?}: {}", hash, e))?;

            nodes_to_delete.push((position, *hash, *node_idx));
        }

        // Sort by position (required for correct deletion order)
        nodes_to_delete.sort_by(|a, b| a.0.cmp(&b.0));

        for (position, hash, node_idx) in nodes_to_delete {
            self.delete_single_leaf(position, hash, node_idx)?;
        }

        Ok(())
    }

    /// Deletes a single leaf node and restructures the tree
    fn delete_single_leaf(
        &mut self,
        position: u64,
        hash: Hash,
        node_idx: u32,
    ) -> Result<(), String> {
        self.hash_to_node.remove(&hash);

        let node = self
            .get_node(node_idx)
            .ok_or_else(|| format!("Invalid node index: {}", node_idx))?;

        if !node.is_leaf() {
            return Err(format!("Node at position {} is not a leaf", position));
        }

        let parent_idx = self.find_parent_of_node(node_idx);

        match parent_idx {
            None => {
                self.delete_root_node(node_idx)?;
            }
            Some(parent_idx) => {
                self.delete_internal_leaf(node_idx, parent_idx)?;
            }
        }

        // NOTE: In utreexo, the leaves count is monotonic and never decreases
        // It represents the total number of leaves ever added to the forest

        Ok(())
    }

    /// Deletes a root node by replacing it with an empty node
    fn delete_root_node(&mut self, node_idx: u32) -> Result<(), String> {
        for (root_pos, &root_idx) in self.roots.iter().enumerate() {
            if root_idx == Some(node_idx) {
                let empty_node = ArenaNode {
                    hash: Hash::empty(),
                    lr: INDEX_MASK,
                    rr: NULL_INDEX,
                    parent: NULL_INDEX,
                    level: root_pos as u32,
                };
                let empty_idx = self.allocate_node(empty_node);

                self.roots[root_pos] = Some(empty_idx);
                self.mark_dirty_ancestors(empty_idx);
                return Ok(());
            }
        }

        Err(format!("Root node {} not found in any level", node_idx))
    }

    /// Deletes an internal leaf by promoting its sibling
    fn delete_internal_leaf(&mut self, leaf_idx: u32, parent_idx: u32) -> Result<(), String> {
        let parent_node = self
            .get_node(parent_idx)
            .ok_or_else(|| format!("Invalid parent index: {}", parent_idx))?;

        let sibling_idx = if parent_node.left_child() == Some(leaf_idx) {
            parent_node.right_child()
        } else if parent_node.right_child() == Some(leaf_idx) {
            parent_node.left_child()
        } else {
            return Err(format!(
                "Node {} is not a child of parent {}",
                leaf_idx, parent_idx
            ));
        };

        let sibling_idx = sibling_idx.ok_or_else(|| {
            format!(
                "Parent {} has no sibling for child {}",
                parent_idx, leaf_idx
            )
        })?;

        let grandparent_idx = self.find_parent_of_node(parent_idx);

        match grandparent_idx {
            None => {
                let mut found_root_pos = None;
                for (root_pos, root_entry) in self.roots.iter().enumerate() {
                    if *root_entry == Some(parent_idx) {
                        found_root_pos = Some(root_pos);
                        break;
                    }
                }

                if let Some(root_pos) = found_root_pos {
                    let row_still_populated = ((self.leaves >> root_pos) & 1) == 1;

                    if row_still_populated {
                        self.roots[root_pos] = Some(sibling_idx);
                        self.nodes[sibling_idx as usize].parent = NULL_INDEX;
                        self.nodes[sibling_idx as usize].level =
                            self.nodes[parent_idx as usize].level;
                        self.fix_sibling_children_parent_pointers(sibling_idx)?;
                        self.mark_dirty_ancestors(sibling_idx);
                    } else {
                        let empty_idx = self.allocate_node(ArenaNode {
                            hash: Hash::empty(),
                            lr: INDEX_MASK,
                            rr: NULL_INDEX,
                            parent: NULL_INDEX,
                            level: root_pos as u32,
                        });
                        self.roots[root_pos] = Some(empty_idx);
                        self.mark_dirty_ancestors(empty_idx);
                    }
                }
            }
            Some(grandparent_idx) => {
                let grandparent_node = self
                    .get_node_mut(grandparent_idx)
                    .ok_or_else(|| format!("Invalid grandparent index: {}", grandparent_idx))?;

                if grandparent_node.left_child() == Some(parent_idx) {
                    grandparent_node.set_left_child(Some(sibling_idx));
                } else if grandparent_node.right_child() == Some(parent_idx) {
                    grandparent_node.set_right_child(Some(sibling_idx));
                } else {
                    return Err(format!(
                        "Parent {} is not a child of grandparent {}",
                        parent_idx, grandparent_idx
                    ));
                }

                if (sibling_idx as usize) < self.nodes.len() {
                    self.nodes[sibling_idx as usize].parent = grandparent_idx;
                    let parent_level = self.nodes[parent_idx as usize].level;
                    self.nodes[sibling_idx as usize].level = parent_level;
                }

                self.fix_sibling_children_parent_pointers(sibling_idx)?;
                self.mark_dirty_ancestors(grandparent_idx);
            }
        }

        Ok(())
    }

    /// Updates promoted sibling's children parent pointers.
    fn fix_sibling_children_parent_pointers(&mut self, sibling_idx: u32) -> Result<(), String> {
        let (left_child_idx, right_child_idx) = {
            let sibling_node = self
                .get_node(sibling_idx)
                .ok_or_else(|| format!("Invalid sibling index: {}", sibling_idx))?;
            (sibling_node.left_child(), sibling_node.right_child())
        };

        if let Some(left_child_idx) = left_child_idx {
            if (left_child_idx as usize) < self.nodes.len() {
                self.nodes[left_child_idx as usize].parent = sibling_idx;
            }
        }

        if let Some(right_child_idx) = right_child_idx {
            if (right_child_idx as usize) < self.nodes.len() {
                self.nodes[right_child_idx as usize].parent = sibling_idx;
            }
        }

        Ok(())
    }
}
