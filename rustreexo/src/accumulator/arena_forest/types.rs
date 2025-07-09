//! Core types and constants for the arena-based forest implementation.

use super::super::node_hash::AccumulatorHash;

/// Node type encoded in the high bit of the left-child index
pub const LEAF_TYPE_BIT: u32 = 1u32 << 31;
/// Mask to extract the actual index from lr field
pub const INDEX_MASK: u32 = !(1u32 << 31);
/// Invalid/null index marker
pub const NULL_INDEX: u32 = u32::MAX;

/// Compact arena node with O(1) parent and level lookups.
///
/// The parent and level fields enable efficient tree traversal and dirty tracking.
#[repr(C)]
#[derive(Clone, Debug)]
pub struct ArenaNode<Hash: AccumulatorHash> {
    /// The hash stored in this node
    pub hash: Hash,
    /// Left child index (low 31 bits) + is_leaf flag (high bit)
    pub lr: u32,
    /// Right child index (for internal nodes)
    pub rr: u32,
    /// Parent node index (NULL_INDEX for roots)
    pub parent: u32,
    /// Level/height of this node (distance from leaves, leaves=0)
    pub level: u32,
}
