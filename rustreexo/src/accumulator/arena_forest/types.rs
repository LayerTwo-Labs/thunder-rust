//! Core types and constants for the arena-based forest implementation.

use super::super::node_hash::AccumulatorHash;

/// Node type bit (bit 31)
pub const LEAF_TYPE_BIT: u32 = 1u32 << 31;
/// Tombstone flag (bit 30)
pub const TOMBSTONE_BIT: u32 = 1u32 << 30;
/// Index mask (bits 29-0)
pub const INDEX_MASK: u32 = !(LEAF_TYPE_BIT | TOMBSTONE_BIT);
/// Null index marker
pub const NULL_INDEX: u32 = u32::MAX;

/// Zombie queue for lazy deletion management
#[derive(Clone, Debug)]
pub struct ZombieQueue {
    /// Indices of tombstoned nodes awaiting cleanup
    pub indices: Vec<u32>,
    /// Threshold for automatic flushing
    pub threshold: usize,
    /// Total count of zombies ever added (for metrics)
    pub total_tombstoned: u64,
}

impl ZombieQueue {
    /// Create new zombie queue with given threshold
    pub fn new(threshold: usize) -> Self {
        Self {
            indices: Vec::new(),
            threshold,
            total_tombstoned: 0,
        }
    }

    /// Add a tombstoned node to the queue
    pub fn add_zombie(&mut self, node_idx: u32) {
        self.indices.push(node_idx);
        self.total_tombstoned += 1;
    }

    /// Check if zombie queue needs flushing
    pub fn needs_flush(&self) -> bool {
        self.indices.len() >= self.threshold
    }

    /// Clear the zombie queue (called after flushing)
    pub fn clear(&mut self) {
        self.indices.clear();
    }

    /// Get current zombie count
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Check if queue is empty
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

/// Fast dirty queue for two-phase dirty propagation.
#[derive(Clone, Debug, Default)]
pub struct DirtyQueue {
    /// Indices collected since the previous flush
    pub inner: Vec<u32>,
}

impl DirtyQueue {
    /// Create new dirty queue
    #[inline]
    pub fn new() -> Self {
        Self { inner: Vec::new() }
    }

    /// Create new dirty queue with pre-allocated capacity
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: Vec::with_capacity(capacity),
        }
    }

    /// Push a node for dirty propagation
    #[inline]
    pub fn push(&mut self, node_idx: u32) {
        self.inner.push(node_idx);
    }

    /// Check if queue is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Get current queue length
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Clear the queue
    #[inline]
    pub fn clear(&mut self) {
        self.inner.clear();
    }
}

impl<Hash: AccumulatorHash> ArenaNode<Hash> {
    /// Check if node is marked as tombstone
    pub fn is_tombstone(&self) -> bool {
        (self.lr & TOMBSTONE_BIT) != 0
    }

    /// Mark node as tombstone
    pub fn mark_tombstone(&mut self) {
        self.lr |= TOMBSTONE_BIT;
    }

    /// Clear tombstone flag
    pub fn clear_tombstone(&mut self) {
        self.lr &= !TOMBSTONE_BIT;
    }
}

/// Compact arena node with bit-packed metadata.
#[repr(C)]
#[derive(Clone, Debug)]
pub struct ArenaNode<Hash: AccumulatorHash> {
    /// Node hash
    pub hash: Hash,
    /// Left child index with type/tombstone flags
    pub lr: u32,
    /// Right child index
    pub rr: u32,
    /// Parent node index
    pub parent: u32,
    /// Node level/height
    pub level: u32,
}
