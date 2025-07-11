//! ArenaForest struct definition and core operations.

use std::convert::TryInto;
use std::fmt;
use std::ops::Deref;

use bitvec::vec::BitVec;
use hashbrown::HashMap;
use smallvec::SmallVec;

use super::super::node_hash::AccumulatorHash;
use super::super::node_hash::BitcoinNodeHash;
use super::types::ArenaNode;
use super::types::DirtyQueue;
use super::types::ZombieQueue;
use super::types::TOMBSTONE_BIT;

/// Fast hash-to-node map using 8-byte keys with collision handling
type FastMap<Hash> = HashMap<u64, SmallVec<[(Hash, u32); 2]>>;

/// Extract 8-byte key from 32-byte hash for fast lookup
#[inline]
pub(crate) fn hash_key<Hash: AccumulatorHash + Deref<Target = [u8; 32]>>(hash: &Hash) -> u64 {
    u64::from_le_bytes(hash[0..8].try_into().unwrap())
}

/// Arena-based utreexo accumulator with SoA layout for cache optimization.
#[derive(Clone, Debug)]
pub struct ArenaForest<Hash: AccumulatorHash = BitcoinNodeHash> {
    /// Node hashes
    pub(crate) hashes: Vec<Hash>,
    /// Left child indices with leaf type bit
    pub(crate) lr: Vec<u32>,
    /// Right child indices
    pub(crate) rr: Vec<u32>,
    /// Parent node indices
    pub(crate) parent: Vec<u32>,
    /// Node levels/heights
    pub(crate) level: Vec<u32>,
    /// Root indices at each tree height
    pub(crate) roots: Vec<Option<u32>>,
    /// Total number of leaves (monotonic)
    pub(crate) leaves: u64,
    /// Hash to node index mapping
    pub(crate) hash_to_node: FastMap<Hash>,
    /// Bit-packed dirty flags per level
    pub(crate) dirty: Vec<BitVec>,
    /// Maximum tree height
    pub(crate) max_height: usize,
    /// Zombie queue for lazy deletion
    pub(crate) zombies: ZombieQueue,
    /// Dirty queue for batch propagation
    pub(crate) dirty_queue: DirtyQueue,
    /// Per-root recomputation buckets
    pub(crate) root_levels: Vec<Vec<Vec<u32>>>,
    /// Root cache for optimization
    pub(crate) root_cache: Vec<Option<usize>>,
}

impl<Hash: AccumulatorHash + Send + Sync> Default for ArenaForest<Hash> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Hash: AccumulatorHash> fmt::Display for ArenaForest<Hash> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ArenaForest {{ leaves: {}, nodes: {}, roots: {} }}",
            self.leaves(),
            self.len(),
            self.roots.len()
        )
    }
}

impl<Hash: AccumulatorHash> ArenaForest<Hash> {
    /// Creates a new empty ArenaForest
    pub fn new_sequential() -> Self {
        Self::with_capacity_sequential(0)
    }

    /// Creates a new ArenaForest with pre-allocated capacity
    pub fn with_capacity_sequential(capacity: usize) -> Self {
        Self {
            hashes: Vec::with_capacity(capacity),
            lr: Vec::with_capacity(capacity),
            rr: Vec::with_capacity(capacity),
            parent: Vec::with_capacity(capacity),
            level: Vec::with_capacity(capacity),
            roots: Vec::new(),
            hash_to_node: HashMap::with_capacity(capacity.max(1) / 2),
            dirty: vec![BitVec::new()],
            max_height: 0,
            leaves: 0,
            zombies: ZombieQueue::new(capacity.max(100) / 10),
            dirty_queue: DirtyQueue::with_capacity(capacity.max(5) / 5),
            root_levels: Vec::new(),
            root_cache: Vec::with_capacity(capacity),
        }
    }

    /// Mark a node as tombstone for lazy deletion
    #[inline]
    pub(crate) fn mark_tombstone(&mut self, node_idx: u32) {
        let idx = node_idx as usize;
        if idx < self.lr.len() {
            self.lr[idx] |= TOMBSTONE_BIT;
            self.zombies.add_zombie(node_idx);
        }
    }

    /// Check if a node is marked as tombstone
    #[inline]
    pub(crate) fn is_tombstone(&self, node_idx: u32) -> bool {
        let idx = node_idx as usize;
        idx < self.lr.len() && (self.lr[idx] & TOMBSTONE_BIT) != 0
    }

    /// Clear tombstone flag from a node
    #[inline]
    pub(crate) fn clear_tombstone(&mut self, node_idx: u32) {
        let idx = node_idx as usize;
        if idx < self.lr.len() {
            self.lr[idx] &= !TOMBSTONE_BIT;
        }
    }

    /// Check if zombie queue needs flushing
    #[inline]
    pub(crate) fn should_flush_zombies(&self) -> bool {
        self.zombies.needs_flush()
    }

    /// Get current zombie count
    #[inline]
    pub fn zombie_count(&self) -> usize {
        self.zombies.len()
    }

    /// Queue a node for dirty propagation
    #[inline]
    pub(crate) fn queue_for_dirty_propagation(&mut self, node_idx: u32) {
        self.dirty_queue.push(node_idx);
    }

    /// Get current dirty queue length
    #[inline]
    pub fn dirty_queue_len(&self) -> usize {
        self.dirty_queue.len()
    }

    /// Returns the number of leaves in the forest
    #[inline]
    pub fn leaves(&self) -> u64 {
        self.leaves
    }

    /// Returns the total number of nodes in the arena
    #[inline]
    pub fn len(&self) -> usize {
        self.hashes.len()
    }

    /// Returns true if the forest is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.hashes.is_empty()
    }

    /// Gets a node by index
    #[inline]
    pub fn get_node(&self, idx: u32) -> Option<ArenaNode<Hash>> {
        let idx = idx as usize;
        if idx < self.hashes.len() {
            Some(ArenaNode {
                hash: self.hashes[idx],
                lr: self.lr[idx],
                rr: self.rr[idx],
                parent: self.parent[idx],
                level: self.level[idx],
            })
        } else {
            None
        }
    }

    /// Gets hash by index
    #[inline]
    pub fn get_hash(&self, idx: u32) -> Option<Hash> {
        self.hashes.get(idx as usize).copied()
    }

    /// Gets parent index by index
    #[inline]
    pub fn get_parent(&self, idx: u32) -> Option<u32> {
        self.parent.get(idx as usize).copied()
    }

    /// Gets level by index
    #[inline]
    pub fn get_level(&self, idx: u32) -> Option<u32> {
        self.level.get(idx as usize).copied()
    }

    /// Allocates a new node in the arena using fast raw pointer writes
    ///
    /// This eliminates bounds checks and reduces cache misses by doing sequential
    /// writes instead of 5 separate Vec::push operations.
    #[inline]
    pub(crate) fn allocate_node(&mut self, node: ArenaNode<Hash>) -> u32 {
        let idx = self.hashes.len() as u32;
        let new_len = idx as usize + 1;

        unsafe {
            // Reserve capacity only once (usually a no-op after initial bulk reserve)
            self.hashes.reserve(1);
            self.lr.reserve(1);
            self.rr.reserve(1);
            self.parent.reserve(1);
            self.level.reserve(1);

            // Get raw pointers at the end of each vec
            let h_ptr = self.hashes.as_mut_ptr().add(idx as usize);
            let lr_ptr = self.lr.as_mut_ptr().add(idx as usize);
            let rr_ptr = self.rr.as_mut_ptr().add(idx as usize);
            let p_ptr = self.parent.as_mut_ptr().add(idx as usize);
            let lvl_ptr = self.level.as_mut_ptr().add(idx as usize);

            // Single sequential write phase - no bounds checks, no branch penalties
            h_ptr.write(node.hash);
            lr_ptr.write(node.lr);
            rr_ptr.write(node.rr);
            p_ptr.write(node.parent);
            lvl_ptr.write(node.level);

            // Bump all vector lengths just once
            self.hashes.set_len(new_len);
            self.lr.set_len(new_len);
            self.rr.set_len(new_len);
            self.parent.set_len(new_len);
            self.level.set_len(new_len);
        }

        idx
    }

    /// Allocates multiple nodes in the arena using batch writes for optimal cache performance.
    ///
    /// This method uses a single reserve operation followed by unsafe sequential writes
    /// to eliminate bounds checks and reduce cache misses from 5N to ~5 total.
    ///
    /// # Safety
    ///
    /// This method uses unsafe pointer arithmetic for performance. It maintains the invariant
    /// that all Vec columns remain synchronized in length and capacity.
    ///
    /// # Performance
    ///
    /// For N nodes, this reduces:
    /// - Cache misses: from 5N to 5 sequential column writes
    /// - Bounds checks: from 5N to 5 total
    /// - Memory allocations: coordinated growth instead of independent reallocations
    ///
    /// # Arguments
    ///
    /// * `nodes` - Slice of nodes to allocate
    ///
    /// # Returns
    ///
    /// The starting index of the allocated nodes. Subsequent nodes are at consecutive indices.
    pub(crate) fn allocate_nodes_batch(&mut self, nodes: &[ArenaNode<Hash>]) -> u32 {
        if nodes.is_empty() {
            return self.hashes.len() as u32;
        }

        let start_idx = self.hashes.len() as u32;
        let count = nodes.len();

        // Reserve exact space in all vectors to avoid reallocations
        self.hashes.reserve_exact(count);
        self.lr.reserve_exact(count);
        self.rr.reserve_exact(count);
        self.parent.reserve_exact(count);
        self.level.reserve_exact(count);

        unsafe {
            // Get raw pointers to the end of each vector
            let hashes_ptr = self.hashes.as_mut_ptr().add(self.hashes.len());
            let lr_ptr = self.lr.as_mut_ptr().add(self.lr.len());
            let rr_ptr = self.rr.as_mut_ptr().add(self.rr.len());
            let parent_ptr = self.parent.as_mut_ptr().add(self.parent.len());
            let level_ptr = self.level.as_mut_ptr().add(self.level.len());

            // Column-wise sequential writes for optimal cache behavior
            // Write all hashes sequentially
            for (i, node) in nodes.iter().enumerate() {
                hashes_ptr.add(i).write(node.hash);
            }

            // Write all lr values sequentially
            for (i, node) in nodes.iter().enumerate() {
                lr_ptr.add(i).write(node.lr);
            }

            // Write all rr values sequentially
            for (i, node) in nodes.iter().enumerate() {
                rr_ptr.add(i).write(node.rr);
            }

            // Write all parent values sequentially
            for (i, node) in nodes.iter().enumerate() {
                parent_ptr.add(i).write(node.parent);
            }

            // Write all level values sequentially
            for (i, node) in nodes.iter().enumerate() {
                level_ptr.add(i).write(node.level);
            }

            // Update lengths atomically to maintain vector invariants
            self.hashes.set_len(self.hashes.len() + count);
            self.lr.set_len(self.lr.len() + count);
            self.rr.set_len(self.rr.len() + count);
            self.parent.set_len(self.parent.len() + count);
            self.level.set_len(self.level.len() + count);
        }

        start_idx
    }
}

impl<Hash: AccumulatorHash + Send + Sync> ArenaForest<Hash> {
    /// Creates a new empty ArenaForest
    pub fn new() -> Self {
        Self::new_sequential()
    }

    /// Creates a new ArenaForest with pre-allocated capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_sequential(capacity)
    }
}
