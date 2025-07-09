//! Arena-based MemForest Implementation
//!
//! Implements an arena-based version of MemForest with compact nodes stored in a single Vec
//! for improved cache locality and efficient hash recomputation via dirty tracking.
//!
//! ## Module Structure
//!
//! - `types`: Core types and constants
//! - `node`: ArenaNode implementation
//! - `forest`: ArenaForest struct and core operations
//! - `construction`: Tree construction and leaf addition
//! - `dirty_tracking`: Hash recomputation
//! - `navigation`: Tree navigation and position calculation
//! - `deletion`: Leaf deletion and tree restructuring
//! - `proofs`: Proof generation and verification
//! - `serialization`: State persistence
//! - `compatibility`: API compatibility layer

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
