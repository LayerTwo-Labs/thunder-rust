//! Arena-based utreexo accumulator with SoA layout for cache optimization.

mod compatibility;
mod construction;
mod deletion;
mod dirty_tracking;
mod forest;
mod navigation;
mod node;
mod proofs;
mod serialization;
mod types;

#[cfg(test)]
mod tests;

// Re-export the main types for public API
pub use compatibility::SimpleNodeWrapper;
pub use forest::ArenaForest;
pub use types::ArenaNode;
pub use types::INDEX_MASK;
pub use types::LEAF_TYPE_BIT;
pub use types::NULL_INDEX;
