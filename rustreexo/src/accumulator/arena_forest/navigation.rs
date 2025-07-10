//! Tree navigation and position calculation utilities.

use super::super::node_hash::AccumulatorHash;
use super::compatibility::SimpleNodeWrapper;
use super::forest::ArenaForest;
use super::types::NULL_INDEX;

impl<Hash: AccumulatorHash> ArenaForest<Hash> {
    /// Position calculation for ArenaForest nodes.
    pub fn get_pos(&self, node_idx: u32) -> Result<u64, String> {
        if node_idx as usize >= self.hashes.len() {
            return Err(format!("Invalid node index: {}", node_idx));
        }

        let mut left_child_indicator = 0_u64;
        let mut rows_to_top = 0;
        let mut current_idx = node_idx;

        while let Some(parent_idx) = self.find_parent_of_node(current_idx) {
            let parent_node = self
                .get_node(parent_idx)
                .ok_or_else(|| format!("Invalid parent index: {}", parent_idx))?;

            if parent_node.left_child() == Some(current_idx) {
                left_child_indicator <<= 1;
            } else if parent_node.right_child() == Some(current_idx) {
                left_child_indicator <<= 1;
                left_child_indicator |= 1;
            } else {
                return Err(format!(
                    "Node {} is not a child of parent {}",
                    current_idx, parent_idx
                ));
            }

            rows_to_top += 1;
            current_idx = parent_idx;
        }

        let mut slot_index = None;
        for (slot, &root_idx) in self.roots.iter().enumerate() {
            if root_idx == Some(current_idx) {
                slot_index = Some(slot);
                break;
            }
        }

        let slot_index = slot_index
            .ok_or_else(|| format!("Could not find slot for root node {}", current_idx))?;

        let tree_index = self.slot_to_tree(slot_index);
        let (_, _, _) = super::super::util::detect_offset(tree_index as u64, self.leaves);
        let forest_rows = super::super::util::tree_rows(self.leaves);

        let mut root_row = None;
        let mut root_vec_idx = self.roots.len().saturating_sub(1);

        for row in 0..=forest_rows {
            if super::super::util::is_root_populated(row, self.leaves) {
                if let Some(Some(idx)) = self.roots.get(root_vec_idx) {
                    if *idx == current_idx {
                        root_row = Some(row);
                        break;
                    }
                }
                root_vec_idx = root_vec_idx.saturating_sub(1);
            }
        }

        let root_row =
            root_row.ok_or_else(|| format!("Could not find root row for node {}", node_idx))?;

        let mut pos = super::super::util::root_position(self.leaves, root_row, forest_rows);

        for _ in 0..rows_to_top {
            match left_child_indicator & 1 {
                0 => {
                    pos = super::super::util::left_child(pos, forest_rows);
                }
                1 => {
                    pos = super::super::util::right_child(pos, forest_rows);
                }
                _ => unreachable!(),
            }
            left_child_indicator >>= 1;
        }

        Ok(pos)
    }

    /// Traverse down the tree following bit pattern
    fn traverse_to_position(&self, position: u64) -> Option<u32> {
        let (tree, branch_len, bits) = super::super::util::detect_offset(position, self.leaves);
        if tree as usize >= self.roots.len() {
            return None;
        }
        let mut current_idx = self.roots[tree as usize]?;

        for row in (0..branch_len).rev() {
            let current_node = self.get_node(current_idx)?;
            let niece_pos = ((bits >> row) & 1) as u8;

            if super::super::util::is_left_niece(niece_pos as u64) {
                current_idx = current_node.right_child()?;
            } else {
                current_idx = current_node.left_child()?;
            }
        }

        Some(current_idx)
    }

    /// Finds the arena index of the node at a specific tree position
    pub(crate) fn find_node_at_position(&self, position: u64) -> Option<u32> {
        self.traverse_to_position(position)
    }

    /// Gets the hash at a specific tree position
    pub(crate) fn get_hash_at_position(&self, position: u64) -> Option<Hash> {
        self.traverse_to_position(position)
            .and_then(|idx| self.get_node(idx))
            .map(|node| node.hash)
    }

    #[inline]
    pub(crate) fn slot_to_tree(&self, slot: usize) -> usize {
        self.roots.len() - 1 - slot
    }

    /// O(1) parent lookup using parent pointers
    #[inline]
    pub(crate) fn find_parent_of_node(&self, node_idx: u32) -> Option<u32> {
        let idx = node_idx as usize;
        if idx >= self.parent.len() {
            return None;
        }

        let parent_idx = self.parent[idx];
        if parent_idx == NULL_INDEX {
            None
        } else {
            Some(parent_idx)
        }
    }

    /// Returns the hash of a specific leaf by position
    pub fn get_leaf(&self, position: u64) -> Option<Hash> {
        self.find_node_at_position(position)
            .and_then(|idx| self.get_node(idx))
            .map(|node| node.hash)
    }

    /// Returns root hashes for verification
    pub fn get_roots(&self) -> Vec<SimpleNodeWrapper<Hash>> {
        self.roots
            .iter()
            .filter_map(|&root_idx| {
                root_idx.and_then(|idx| {
                    self.get_node(idx)
                        .map(|node| SimpleNodeWrapper::new(node.hash))
                })
            })
            .collect()
    }

    /// Finds which root subtree a node belongs to
    #[inline]
    pub(crate) fn find_root_of_node(&self, node_idx: u32) -> Option<usize> {
        let mut current_idx = node_idx;
        
        while let Some(parent_idx) = self.find_parent_of_node(current_idx) {
            current_idx = parent_idx;
        }
        
        for (root_position, &root_option) in self.roots.iter().enumerate() {
            if let Some(root_idx) = root_option {
                if root_idx == current_idx {
                    return Some(root_position);
                }
            }
        }
        
        None
    }

}
