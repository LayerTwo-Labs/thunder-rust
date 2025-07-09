//! ArenaNode implementation with all node operations.

use super::super::node_hash::AccumulatorHash;
use super::types::ArenaNode;
use super::types::INDEX_MASK;
use super::types::LEAF_TYPE_BIT;
use super::types::NULL_INDEX;

impl<Hash: AccumulatorHash> ArenaNode<Hash> {
    // --- Node Constructors ---

    /// Creates a new leaf node
    pub fn new_leaf(hash: Hash) -> Self {
        Self {
            hash,
            lr: LEAF_TYPE_BIT,
            rr: NULL_INDEX,
            parent: NULL_INDEX,
            level: 0,
        }
    }

    /// Creates a new internal node with left and right children
    pub fn new_internal(hash: Hash, left_idx: u32, right_idx: u32, level: u32) -> Self {
        Self {
            hash,
            lr: left_idx & INDEX_MASK,
            rr: right_idx,
            parent: NULL_INDEX,
            level,
        }
    }

    /// Creates a new internal node with optional children
    pub fn new_internal_with_optional_children(
        hash: Hash,
        left_idx: Option<u32>,
        right_idx: Option<u32>,
        level: u32,
    ) -> Self {
        let lr = match left_idx {
            Some(idx) => idx & INDEX_MASK,
            None => NULL_INDEX & INDEX_MASK,
        };

        let rr = match right_idx {
            Some(idx) => idx,
            None => NULL_INDEX,
        };

        Self {
            hash,
            lr,
            rr,
            parent: NULL_INDEX,
            level,
        }
    }

    // --- Node Type and Child Access ---

    /// Returns true if this node is a leaf
    #[inline]
    pub fn is_leaf(&self) -> bool {
        (self.lr & LEAF_TYPE_BIT) != 0
    }

    /// Returns the left child index, or None if no left child
    #[inline]
    pub fn left_child(&self) -> Option<u32> {
        if self.is_leaf() {
            None
        } else {
            let idx = self.lr & INDEX_MASK;
            if idx == NULL_INDEX {
                None
            } else {
                Some(idx)
            }
        }
    }

    /// Returns the right child index, or None if no right child
    #[inline]
    pub fn right_child(&self) -> Option<u32> {
        if self.is_leaf() || self.rr == NULL_INDEX {
            None
        } else {
            Some(self.rr)
        }
    }

    /// Returns both children as a tuple (left, right)
    #[inline]
    pub fn children(&self) -> (Option<u32>, Option<u32>) {
        (self.left_child(), self.right_child())
    }

    // --- Node Mutation ---

    /// Sets the left child index
    #[inline]
    pub fn set_left_child(&mut self, idx: Option<u32>) {
        match idx {
            Some(child_idx) => {
                self.lr = (self.lr & LEAF_TYPE_BIT) | (child_idx & INDEX_MASK);
            }
            None => {
                self.lr = (self.lr & LEAF_TYPE_BIT) | INDEX_MASK;
            }
        }
    }

    /// Sets the right child index  
    #[inline]
    pub fn set_right_child(&mut self, idx: Option<u32>) {
        self.rr = idx.unwrap_or(NULL_INDEX);
    }

    /// Sets both children at once
    #[inline]
    pub fn set_children(&mut self, left: Option<u32>, right: Option<u32>) {
        self.set_left_child(left);
        self.set_right_child(right);
    }

    // --- Node Type Conversion ---

    /// Converts this node to a leaf (typically during deletion)
    #[inline]
    pub fn make_leaf(&mut self) {
        self.lr = LEAF_TYPE_BIT;
        self.rr = NULL_INDEX;
    }

    /// Converts this node to an internal node
    #[inline]
    pub fn make_internal(&mut self, left: u32, right: u32) {
        self.lr = left & INDEX_MASK;
        self.rr = right;
    }
}
