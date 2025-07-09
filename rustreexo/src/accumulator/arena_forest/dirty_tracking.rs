//! Dirty node tracking and hash recomputation system.

use rayon::prelude::*;
use smallvec::SmallVec;

use super::super::node_hash::AccumulatorHash;
use super::forest::ArenaForest;

impl<Hash: AccumulatorHash> ArenaForest<Hash> {
    /// Marks a node as dirty with dynamic level growth.
    /// Grows dirty_levels Vec on demand to prevent silent corruption.
    #[inline]
    pub(crate) fn mark_dirty(&mut self, node_idx: u32, level: usize) {
        // Grow Vec if level exceeds current capacity
        if level >= self.dirty_levels.len() {
            self.dirty_levels.resize_with(level + 1, SmallVec::new);
        }

        self.dirty_levels[level].push(node_idx);
        self.max_height = self.max_height.max(level);
    }

    /// Mark node as dirty only if not already marked.
    /// Returns true if newly marked, false if already dirty.
    #[inline]
    fn mark_if_new(&mut self, node_idx: u32, level: usize) -> bool {
        // Grow Vec if level exceeds current capacity
        if level >= self.dirty_levels.len() {
            self.dirty_levels.resize_with(level + 1, SmallVec::new);
        }

        if self.dirty_levels[level].contains(&node_idx) {
            return false;
        }

        self.dirty_levels[level].push(node_idx);
        self.max_height = self.max_height.max(level);
        true
    }

    /// Mark ancestors with early termination.
    /// Walks up the ancestor chain and marks each node as dirty, stopping
    /// when hitting a node that's already marked.
    pub(crate) fn mark_dirty_ancestors(&mut self, mut node_idx: u32) {
        loop {
            let level = self.nodes[node_idx as usize].level as usize;
            if !self.mark_if_new(node_idx, level) {
                break;
            }
            match self.find_parent_of_node(node_idx) {
                Some(p) => node_idx = p,
                None => break,
            }
        }
    }

    /// Fast hash recomputation for all AccumulatorHash types.
    /// Processes all dirty nodes bottom-up in a single pass.
    pub fn recompute_dirty_hashes_sequential(&mut self) {
        // Process levels from bottom to top
        for level in 0..=self.max_height {
            let dirty_nodes = &self.dirty_levels[level];
            if dirty_nodes.is_empty() {
                continue;
            }

            for &node_idx in dirty_nodes.iter() {
                if node_idx as usize >= self.nodes.len() {
                    continue;
                }

                // Get children indices to avoid borrowing conflicts
                let (left_idx, right_idx) = {
                    let node = &self.nodes[node_idx as usize];
                    if node.is_leaf() {
                        continue;
                    }
                    node.children()
                };

                if let (Some(left_idx), Some(right_idx)) = (left_idx, right_idx) {
                    if (left_idx as usize) < self.nodes.len()
                        && (right_idx as usize) < self.nodes.len()
                    {
                        let left_hash = self.nodes[left_idx as usize].hash;
                        let right_hash = self.nodes[right_idx as usize].hash;
                        let new_hash = Hash::parent_hash(&left_hash, &right_hash);

                        self.nodes[node_idx as usize].hash = new_hash;
                    }
                }
            }
        }

        // Clear dirty tracking
        for nodes in &mut self.dirty_levels[..=self.max_height] {
            nodes.clear();
        }
        self.max_height = 0;
    }
}

impl<Hash: AccumulatorHash + Send + Sync> ArenaForest<Hash> {
    /// Parallel hash recomputation with level-wise parallelism.
    /// Provides speedup on multi-core systems over sequential processing.
    pub fn recompute_dirty_hashes_parallel(&mut self) {
        // Process levels from bottom to top
        for level in 0..=self.max_height {
            let dirty_nodes = &self.dirty_levels[level];
            if dirty_nodes.is_empty() {
                continue;
            }

            // Collect all hash computations for this level first to avoid borrowing issues
            let hash_updates: Vec<(u32, Hash)> = dirty_nodes
                .par_iter()
                .filter_map(|&node_idx| {
                    if node_idx as usize >= self.nodes.len() {
                        return None;
                    }

                    let node = &self.nodes[node_idx as usize];
                    if node.is_leaf() {
                        return None;
                    }

                    let (left_idx, right_idx) = node.children();

                    if let (Some(left_idx), Some(right_idx)) = (left_idx, right_idx) {
                        if (left_idx as usize) < self.nodes.len()
                            && (right_idx as usize) < self.nodes.len()
                        {
                            let left_hash = self.nodes[left_idx as usize].hash;
                            let right_hash = self.nodes[right_idx as usize].hash;
                            let new_hash = Hash::parent_hash(&left_hash, &right_hash);

                            Some((node_idx, new_hash))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();

            // Apply all hash updates sequentially
            for (node_idx, new_hash) in hash_updates {
                if (node_idx as usize) < self.nodes.len() {
                    self.nodes[node_idx as usize].hash = new_hash;
                }
            }
        }

        // Clear dirty tracking
        for nodes in &mut self.dirty_levels[..=self.max_height] {
            nodes.clear();
        }
        self.max_height = 0;
    }

    /// Choose optimal recomputation strategy based on workload size.
    /// Uses dynamic thresholds to prefer sequential processing for small workloads.
    pub fn recompute_dirty_hashes_adaptive(&mut self) {
        let total_dirty: usize = self.dirty_levels.iter().map(|level| level.len()).sum();

        let phys_cores = num_cpus::get_physical().max(1);
        let threshold = 12 * phys_cores;

        if total_dirty >= threshold {
            self.recompute_dirty_hashes_parallel();
        } else {
            self.recompute_dirty_hashes_sequential();
        }
    }

    /// Force sequential recomputation regardless of workload size.
    pub fn recompute_dirty_hashes_force_sequential(&mut self) {
        self.recompute_dirty_hashes_sequential();
    }

    /// Force parallel recomputation regardless of workload size.
    pub fn recompute_dirty_hashes_force_parallel(&mut self) {
        self.recompute_dirty_hashes_parallel();
    }

    /// Inject a specific set of dirty nodes for testing and benchmarking.
    pub fn inject_dirty(&mut self, node_indices: &[u32]) {
        // Clear existing dirty state
        for nodes in &mut self.dirty_levels {
            nodes.clear();
        }
        self.max_height = 0;

        // Mark specified nodes as dirty
        for &idx in node_indices {
            if (idx as usize) < self.nodes.len() {
                let level = self.nodes[idx as usize].level as usize;
                self.mark_dirty(idx, level);
            }
        }
    }

    /// Get the current total number of dirty nodes across all levels.
    pub fn dirty_count(&self) -> usize {
        self.dirty_levels.iter().map(|level| level.len()).sum()
    }
}
