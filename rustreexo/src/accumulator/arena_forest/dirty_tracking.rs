//! Dirty node tracking and hash recomputation system.

use bitvec::vec::BitVec;
use rayon::prelude::*;

use super::super::node_hash::AccumulatorHash;
use super::forest::ArenaForest;

impl<Hash: AccumulatorHash> ArenaForest<Hash> {
    /// Marks a node as dirty with dynamic level growth.
    #[inline]
    pub(crate) fn mark_dirty(&mut self, node_idx: u32, level: usize) {
        // Grow dirty levels if needed
        if level >= self.dirty.len() {
            self.dirty.resize_with(level + 1, BitVec::new);
        }

        // Ensure capacity for node index
        let node_idx_usize = node_idx as usize;
        if node_idx_usize >= self.dirty[level].len() {
            self.dirty[level].resize(node_idx_usize + 1, false);
        }

        // Set dirty bit
        self.dirty[level].set(node_idx_usize, true);
        self.max_height = self.max_height.max(level);
    }

    /// Mark node as dirty only if not already marked.
    #[inline]
    fn mark_if_new(&mut self, node_idx: u32, level: usize) -> bool {
        // Grow dirty levels if needed
        if level >= self.dirty.len() {
            self.dirty.resize_with(level + 1, BitVec::new);
        }

        let node_idx_usize = node_idx as usize;

        // Check if already dirty
        if node_idx_usize < self.dirty[level].len() && self.dirty[level][node_idx_usize] {
            return false;
        }

        // Ensure capacity for node index
        if node_idx_usize >= self.dirty[level].len() {
            self.dirty[level].resize(node_idx_usize + 1, false);
        }

        // Set dirty bit
        self.dirty[level].set(node_idx_usize, true);
        self.max_height = self.max_height.max(level);
        true
    }

    /// Mark ancestors with early termination.
    pub(crate) fn mark_dirty_ancestors(&mut self, mut node_idx: u32) {
        loop {
            let idx = node_idx as usize;
            if idx >= self.level.len() {
                break;
            }

            // Direct level access
            let level = self.level[idx] as usize;
            if !self.mark_if_new(node_idx, level) {
                break;
            }
            match self.find_parent_of_node(node_idx) {
                Some(p) => node_idx = p,
                None => break,
            }
        }
    }

    /// Fast hash recomputation using bottom-up traversal.
    pub fn recompute_dirty_hashes_sequential(&mut self) {
        // Process levels from bottom to top
        for level in 0..=self.max_height {
            // Skip empty levels
            if level >= self.dirty.len() || self.dirty[level].is_empty() {
                continue;
            }

            // Iterate over dirty nodes
            for node_idx_usize in self.dirty[level].iter_ones() {
                let idx = node_idx_usize;
                if idx >= self.hashes.len() {
                    continue;
                }

                // Check if node is leaf
                let lr = self.lr[idx];
                if lr & super::types::LEAF_TYPE_BIT != 0 {
                    continue;
                }

                // Extract children indices
                let left_idx = (lr & super::types::INDEX_MASK) as usize;
                let right_idx = self.rr[idx] as usize;

                // Bounds check
                if left_idx < self.hashes.len() && right_idx < self.hashes.len() {
                    // Get child hashes
                    let left_hash = self.hashes[left_idx];
                    let right_hash = self.hashes[right_idx];
                    let new_hash = Hash::parent_hash(&left_hash, &right_hash);

                    // Update hash
                    self.hashes[idx] = new_hash;
                }
            }
        }

        // Clear dirty tracking
        for bitvec in &mut self.dirty[..=self.max_height] {
            bitvec.clear();
        }
        self.max_height = 0;
    }

    /// O(k) bottom-up dirty propagation using level-by-level processing.
    pub(crate) fn flush_dirty_queue(&mut self) -> Result<(), String> {
        use smallvec::SmallVec;

        if self.dirty_queue.is_empty() {
            return Ok(());
        }

        // Bucket queued nodes by level
        let mut frontier: SmallVec<[Vec<u32>; 8]> = SmallVec::new();
        for &n in &self.dirty_queue.inner {
            let idx = n as usize;
            if idx >= self.level.len() {
                continue;
            }

            let lvl = self.level[idx] as usize;
            if lvl >= frontier.len() {
                frontier.resize_with(lvl + 1, Vec::new);
            }
            frontier[lvl].push(n);
        }
        self.dirty_queue.clear();

        // Bottom-up sweep with dynamic frontier
        let mut level = 0;
        while level < frontier.len() {
            let dirty_here = std::mem::take(&mut frontier[level]);
            if !dirty_here.is_empty() {
                for idx in dirty_here {
                    // Mark dirty and propagate if newly marked
                    if self.mark_if_new(idx, level) {
                        // Enqueue parent
                        if let Some(p) = self.find_parent_of_node(idx) {
                            let p_idx = p as usize;
                            if p_idx < self.level.len() {
                                let p_lvl = self.level[p_idx] as usize;
                                if p_lvl >= frontier.len() {
                                    frontier.resize_with(p_lvl + 1, Vec::new);
                                }
                                frontier[p_lvl].push(p);
                            }
                        }
                    }
                }
            }
            level += 1;
        }

        Ok(())
    }

    /// Zero-allocation bucketing of dirty nodes by root and level.
    fn bucket_dirty_once(&mut self) -> usize {
        // Clear existing buckets
        for root_levels in &mut self.root_levels {
            for level_vec in root_levels {
                level_vec.clear();
            }
        }

        // Reuse persistent root cache
        if self.root_cache.len() < self.hashes.len() {
            self.root_cache.resize(self.hashes.len(), None);
        }
        self.root_cache.fill(None);

        let mut active_roots = 0;

        // Process dirty nodes
        for level in 0..=self.max_height {
            if level >= self.dirty.len() || self.dirty[level].is_empty() {
                continue;
            }

            for node_idx_usize in self.dirty[level].iter_ones() {
                if node_idx_usize >= self.hashes.len() {
                    continue;
                }

                // Skip leaf nodes
                let lr = self.lr[node_idx_usize];
                if lr & super::types::LEAF_TYPE_BIT != 0 {
                    continue;
                }

                // Find root for this node
                let root_position = if let Some(cached_root) = self.root_cache[node_idx_usize] {
                    cached_root
                } else {
                    if let Some(root_pos) = self.find_root_of_node(node_idx_usize as u32) {
                        self.root_cache[node_idx_usize] = Some(root_pos);
                        root_pos
                    } else {
                        continue;
                    }
                };

                // Ensure enough root buckets
                while self.root_levels.len() <= root_position {
                    self.root_levels.push(Vec::new());
                }

                // Ensure enough levels for this root
                if self.root_levels[root_position].len() <= level {
                    self.root_levels[root_position]
                        .resize_with(level + 1, || Vec::with_capacity(32));
                }

                // Check if root was empty before adding node
                let was_empty = self.root_levels[root_position]
                    .iter()
                    .all(|lvl| lvl.is_empty());

                // Add node to bucket
                self.root_levels[root_position][level].push(node_idx_usize as u32);

                // Track active root count
                if was_empty {
                    active_roots += 1;
                }
            }
        }

        active_roots
    }
}

impl<Hash: AccumulatorHash + Send + Sync> ArenaForest<Hash> {
    /// Parallel hash recomputation with level-wise parallelism.
    pub fn recompute_dirty_hashes_parallel(&mut self) {
        // Process levels from bottom to top
        for level in 0..=self.max_height {
            // Skip empty levels
            if level >= self.dirty.len() || self.dirty[level].is_empty() {
                continue;
            }

            // Collect dirty nodes for parallel processing
            let dirty_indices: Vec<usize> = self.dirty[level].iter_ones().collect();

            // Parallel hash computation
            let hash_updates: Vec<(usize, Hash)> = dirty_indices
                .par_iter()
                .filter_map(|&node_idx_usize| {
                    let idx = node_idx_usize;
                    if idx >= self.hashes.len() {
                        return None;
                    }

                    // Check if node is leaf
                    let lr = self.lr[idx];
                    if lr & super::types::LEAF_TYPE_BIT != 0 {
                        return None;
                    }

                    // Extract children indices
                    let left_idx = (lr & super::types::INDEX_MASK) as usize;
                    let right_idx = self.rr[idx] as usize;

                    // Bounds check
                    if left_idx < self.hashes.len() && right_idx < self.hashes.len() {
                        // Get child hashes
                        let left_hash = self.hashes[left_idx];
                        let right_hash = self.hashes[right_idx];
                        let new_hash = Hash::parent_hash(&left_hash, &right_hash);

                        Some((node_idx_usize, new_hash))
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();

            // Apply hash updates
            for (node_idx_usize, new_hash) in hash_updates {
                if node_idx_usize < self.hashes.len() {
                    self.hashes[node_idx_usize] = new_hash;
                }
            }
        }

        // Clear dirty tracking
        for bitvec in &mut self.dirty[..=self.max_height] {
            bitvec.clear();
        }
        self.max_height = 0;
    }

    /// Per-root parallel hash recomputation using pre-built buckets.
    /// SAFETY: Root subtrees are disjoint, enabling safe parallel writes.
    pub fn recompute_dirty_hashes_per_root(&mut self) {
        let active_roots = self.bucket_dirty_once();
        self.recompute_dirty_hashes_per_root_impl(active_roots);
    }

    /// Shared implementation for per-root recomputation.
    fn recompute_dirty_hashes_per_root_impl(&mut self, active_roots: usize) {
        if active_roots == 0 {
            return;
        }

        // Process each root in parallel
        self.root_levels.par_iter().for_each(|root_levels| {
            if root_levels.is_empty() {
                return;
            }

            // Process levels sequentially within this root
            for level_nodes in root_levels.iter() {
                if level_nodes.is_empty() {
                    continue;
                }

                // Process nodes sequentially to maintain dependency order
                for &node_idx in level_nodes.iter() {
                    let idx = node_idx as usize;
                    if idx >= self.hashes.len() {
                        continue;
                    }

                    // Safe Rust for most logic
                    let lr = self.lr[idx];
                    if lr & super::types::LEAF_TYPE_BIT != 0 {
                        continue;
                    }

                    let left_idx = (lr & super::types::INDEX_MASK) as usize;
                    let right_idx = self.rr[idx] as usize;

                    if left_idx < self.hashes.len() && right_idx < self.hashes.len() {
                        // Compute new hash from children
                        let left_hash = self.hashes[left_idx];
                        let right_hash = self.hashes[right_idx];
                        let new_hash = Hash::parent_hash(&left_hash, &right_hash);

                        // SAFETY: Root subtrees are disjoint, no data races
                        unsafe {
                            *self.hashes.as_ptr().add(idx).cast_mut() = new_hash;
                        }
                    }
                }
            }
        });

        self.clear_dirty();
    }

    /// Clear dirty tracking structures.
    fn clear_dirty(&mut self) {
        // Clear dirty tracking
        for bitvec in &mut self.dirty[..=self.max_height] {
            bitvec.clear();
        }
        self.max_height = 0;
    }

    /// Choose optimal recomputation strategy based on heuristics.
    pub fn recompute_dirty_hashes_adaptive(&mut self) {
        let total_dirty: usize = self.dirty.iter().map(|bitvec| bitvec.count_ones()).sum();

        if total_dirty == 0 {
            return;
        }

        let phys_cores = num_cpus::get_physical().max(1);
        let sequential_threshold = 12 * phys_cores;

        // For small workloads, use sequential processing
        if total_dirty < sequential_threshold {
            self.recompute_dirty_hashes_sequential();
            return;
        }

        // For medium workloads, try level-wise parallel first
        let per_root_threshold = 64 * phys_cores;

        if total_dirty >= per_root_threshold {
            // For large workloads, try per-root parallelism
            let active_roots = self.bucket_dirty_once();

            // Use per-root when we have many distinct roots
            if active_roots >= phys_cores {
                // Use pre-built buckets to avoid double bucketing
                self.recompute_dirty_hashes_per_root_impl(active_roots);
                return;
            }
        }

        // Default to level-wise parallelism
        self.recompute_dirty_hashes_parallel();
    }

    /// Force sequential recomputation regardless of workload size.
    pub fn recompute_dirty_hashes_force_sequential(&mut self) {
        self.recompute_dirty_hashes_sequential();
    }

    /// Force parallel recomputation regardless of workload size.
    pub fn recompute_dirty_hashes_force_parallel(&mut self) {
        self.recompute_dirty_hashes_parallel();
    }

    /// Force per-root parallel recomputation regardless of workload size.
    pub fn recompute_dirty_hashes_force_per_root(&mut self) {
        self.recompute_dirty_hashes_per_root();
    }

    /// Inject dirty nodes for testing and benchmarking.
    pub fn inject_dirty(&mut self, node_indices: &[u32]) {
        // Clear existing dirty state
        for bitvec in &mut self.dirty {
            bitvec.clear();
        }
        self.max_height = 0;

        // Mark nodes as dirty
        for &idx in node_indices {
            let idx_usize = idx as usize;
            if idx_usize < self.level.len() {
                // Get node level
                let level = self.level[idx_usize] as usize;
                self.mark_dirty(idx, level);
            }
        }
    }

    /// Get the current total number of dirty nodes across all levels.
    pub fn dirty_count(&self) -> usize {
        self.dirty.iter().map(|bitvec| bitvec.count_ones()).sum()
    }
}
