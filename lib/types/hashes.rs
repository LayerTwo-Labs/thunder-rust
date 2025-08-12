use std::str::FromStr;

use bitcoin::hashes::Hash as _;
use borsh::{BorshDeserialize, BorshSerialize};
use hex::FromHex;
use serde::{Deserialize, Serialize};

use super::serde_hexstr_human_readable;

const BLAKE3_LENGTH: usize = 32;

pub type Hash = [u8; BLAKE3_LENGTH];

#[derive(
    BorshSerialize,
    BorshDeserialize,
    Clone,
    Copy,
    Deserialize,
    Eq,
    Hash,
    Ord,
    PartialEq,
    PartialOrd,
    Serialize,
)]
pub struct BlockHash(#[serde(with = "serde_hexstr_human_readable")] pub Hash);

impl From<Hash> for BlockHash {
    fn from(other: Hash) -> Self {
        Self(other)
    }
}

impl From<BlockHash> for Hash {
    fn from(other: BlockHash) -> Self {
        other.0
    }
}

impl From<BlockHash> for Vec<u8> {
    fn from(other: BlockHash) -> Self {
        other.0.into()
    }
}

impl From<BlockHash> for bitcoin::BlockHash {
    fn from(other: BlockHash) -> Self {
        let inner: [u8; 32] = other.into();
        Self::from_byte_array(inner)
    }
}

impl std::fmt::Display for BlockHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", hex::encode(self.0))
    }
}

impl std::fmt::Debug for BlockHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", hex::encode(self.0))
    }
}

impl FromStr for BlockHash {
    type Err = hex::FromHexError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Hash::from_hex(s).map(Self)
    }
}

impl utoipa::PartialSchema for BlockHash {
    fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
        let obj =
            utoipa::openapi::Object::with_type(utoipa::openapi::Type::String);
        utoipa::openapi::RefOr::T(utoipa::openapi::Schema::Object(obj))
    }
}

impl utoipa::ToSchema for BlockHash {
    fn name() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("BlockHash")
    }
}

#[derive(
    BorshSerialize,
    Clone,
    Copy,
    Default,
    Deserialize,
    Eq,
    Hash,
    Ord,
    PartialEq,
    PartialOrd,
    Serialize,
)]
pub struct MerkleRoot(#[serde(with = "serde_hexstr_human_readable")] Hash);

impl From<Hash> for MerkleRoot {
    fn from(other: Hash) -> Self {
        Self(other)
    }
}

impl From<MerkleRoot> for Hash {
    fn from(other: MerkleRoot) -> Self {
        other.0
    }
}

impl std::fmt::Display for MerkleRoot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", hex::encode(self.0))
    }
}

impl std::fmt::Debug for MerkleRoot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", hex::encode(self.0))
    }
}

impl utoipa::PartialSchema for MerkleRoot {
    fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
        let obj =
            utoipa::openapi::Object::with_type(utoipa::openapi::Type::String);
        utoipa::openapi::RefOr::T(utoipa::openapi::Schema::Object(obj))
    }
}

impl utoipa::ToSchema for MerkleRoot {
    fn name() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("MerkleRoot")
    }
}

#[derive(
    BorshSerialize,
    Clone,
    Copy,
    Default,
    Deserialize,
    Eq,
    Hash,
    Serialize,
    Ord,
    PartialEq,
    PartialOrd,
)]
#[repr(transparent)]
#[serde(transparent)]
pub struct Txid(#[serde(with = "serde_hexstr_human_readable")] pub Hash);

impl Txid {
    pub fn as_slice(&self) -> &[u8] {
        self.0.as_slice()
    }
}

impl From<Hash> for Txid {
    fn from(other: Hash) -> Self {
        Self(other)
    }
}

impl From<Txid> for Hash {
    fn from(other: Txid) -> Self {
        other.0
    }
}

impl<'a> From<&'a Txid> for &'a Hash {
    fn from(other: &'a Txid) -> Self {
        &other.0
    }
}

impl std::fmt::Display for Txid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", hex::encode(self.0))
    }
}

impl std::fmt::Debug for Txid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", hex::encode(self.0))
    }
}

impl FromStr for Txid {
    type Err = hex::FromHexError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Hash::from_hex(s).map(Self)
    }
}

impl utoipa::PartialSchema for Txid {
    fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
        let obj =
            utoipa::openapi::Object::with_type(utoipa::openapi::Type::String);
        utoipa::openapi::RefOr::T(utoipa::openapi::Schema::Object(obj))
    }
}

impl utoipa::ToSchema for Txid {
    fn name() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("Txid")
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[repr(transparent)]
#[serde(transparent)]
pub struct M6id(pub bitcoin::Txid);

impl std::fmt::Display for M6id {
    #[inline(always)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

pub fn hash<T>(data: &T) -> Hash
where
    T: BorshSerialize + ?Sized,
{
    let data_serialized = borsh::to_vec(data)
        .expect("failed to serialize with borsh to compute a hash");
    blake3::hash(&data_serialized).into()
}

/// Cross-platform SIMD-optimized batch hashing for merkle computations
/// Uses BLAKE3's built-in vectorization which auto-detects available SIMD instructions
pub fn hash_batch_optimized<T>(data_batch: &[T]) -> Vec<Hash>
where
    T: BorshSerialize + Sync,
{
    if data_batch.is_empty() {
        return Vec::new();
    }
    
    // For small batches, use regular hashing
    if data_batch.len() < 4 {
        return data_batch.iter().map(hash).collect();
    }
    
    // BLAKE3 has built-in SIMD optimization that adapts to the platform
    // Use parallel hashing for better throughput on multi-core systems
    use rayon::prelude::*;
    
    // Optimal chunk size for SIMD processing (based on BLAKE3's internal buffering)
    let chunk_size = std::cmp::max(4, data_batch.len() / rayon::current_num_threads().max(1));
    
    data_batch
        .par_chunks(chunk_size)
        .flat_map(|chunk| {
            // BLAKE3 automatically uses available SIMD instructions (SSE2, SSE4.1, AVX2, AVX-512, NEON)
            // based on CPU feature detection at runtime
            chunk.iter().map(|item| {
                let data_serialized = borsh::to_vec(item)
                    .expect("failed to serialize with borsh to compute a hash");
                blake3::hash(&data_serialized).into()
            }).collect::<Vec<_>>()
        })
        .collect()
}

/// SIMD-optimized hash computation for pre-serialized data
/// Leverages BLAKE3's cross-platform SIMD capabilities
pub fn hash_preserialized_batch(serialized_data: &[Vec<u8>]) -> Vec<Hash> {
    if serialized_data.is_empty() {
        return Vec::new();
    }
    
    if serialized_data.len() < 4 {
        return serialized_data.iter().map(|data| blake3::hash(data).into()).collect();
    }
    
    // Use BLAKE3's multi-threading capability for large batches
    use rayon::prelude::*;
    
    serialized_data
        .par_iter()
        .map(|data| {
            // BLAKE3 automatically uses the best SIMD implementation available:
            // - x86/x86_64: AVX-512, AVX2, SSE4.1, SSE2 (runtime detection)
            // - ARM: NEON (runtime detection) 
            // - Other: scalar fallback
            blake3::hash(data).into()
        })
        .collect()
}
