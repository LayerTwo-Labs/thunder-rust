//! API compatibility layer for Thunder integration.

use super::super::node_hash::AccumulatorHash;

/// API Compatibility Wrapper
///
/// Provides the same interface as the original Node for Thunder integration.
#[derive(Debug, Clone)]
pub struct SimpleNodeWrapper<Hash: AccumulatorHash> {
    data: Hash,
}

impl<Hash: AccumulatorHash> SimpleNodeWrapper<Hash> {
    pub fn new(data: Hash) -> Self {
        Self { data }
    }

    /// Returns the node's hash data
    pub fn get_data(&self) -> Hash {
        self.data
    }
}
