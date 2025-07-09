//! ArenaForest struct definition and core operations.

use std::collections::HashMap;
use std::fmt;

use smallvec::SmallVec;

use super::super::node_hash::AccumulatorHash;
use super::super::node_hash::BitcoinNodeHash;
use super::types::ArenaNode;

/// Arena-based MemForest implementing the utreexo accumulator.
///
/// Stores all nodes in a single Vec for improved cache locality and uses
/// dirty tracking for efficient hash recomputation.
#[derive(Clone, Debug)]
pub struct ArenaForest<Hash: AccumulatorHash = BitcoinNodeHash> {
    /// Arena containing all nodes indexed by u32
    pub(crate) nodes: Vec<ArenaNode<Hash>>,
    /// Root indices at each tree height (sparse array)
    pub(crate) roots: Vec<Option<u32>>,
    /// Total number of leaves ever added to the forest (monotonic)
    pub(crate) leaves: u64,
    /// Hash to arena node index mapping for proof generation
    pub(crate) hash_to_node: HashMap<Hash, u32>,
    /// Dirty nodes per level for efficient recomputation
    pub(crate) dirty_levels: Vec<SmallVec<[u32; 16]>>,
    /// Maximum tree height for level iteration
    pub(crate) max_height: usize,
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
    // --- Construction ---

    /// Creates a new empty ArenaForest
    pub fn new_sequential() -> Self {
        Self {
            nodes: Vec::new(),
            roots: Vec::new(),
            leaves: 0,
            hash_to_node: HashMap::new(),
            dirty_levels: vec![SmallVec::new()],
            max_height: 0,
        }
    }

    /// Creates a new ArenaForest with pre-allocated capacity
    pub fn with_capacity_sequential(capacity: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(capacity),
            roots: Vec::new(),
            hash_to_node: HashMap::with_capacity(capacity / 2),
            dirty_levels: vec![SmallVec::new()],
            max_height: 0,
            leaves: 0,
        }
    }

    // --- Basic Accessors ---

    /// Returns the number of leaves in the forest
    #[inline]
    pub fn leaves(&self) -> u64 {
        self.leaves
    }

    /// Returns the total number of nodes in the arena
    #[inline]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns true if the forest is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Gets a node by index
    #[inline]
    pub fn get_node(&self, idx: u32) -> Option<&ArenaNode<Hash>> {
        self.nodes.get(idx as usize)
    }

    /// Gets a mutable node by index
    #[inline]
    pub fn get_node_mut(&mut self, idx: u32) -> Option<&mut ArenaNode<Hash>> {
        self.nodes.get_mut(idx as usize)
    }

    // --- Memory Management ---

    /// Allocates a new node in the arena and returns its index
    pub(crate) fn allocate_node(&mut self, node: ArenaNode<Hash>) -> u32 {
        let idx = self.nodes.len() as u32;
        self.nodes.push(node);
        idx
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
