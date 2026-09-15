use std::str::FromStr;

use bitcoin::hashes::Hash as _;
use blake3::Hasher;
use borsh::{BorshDeserialize, BorshSerialize};
use const_hex::FromHex;
use serde::{Deserialize, Serialize};

use crate::util::serde::hexstr_human_readable;

const BLAKE3_LENGTH: usize = 32;

pub type Hash = [u8; BLAKE3_LENGTH];

pub type UtreexoNodeHash = rustreexo::accumulator::node_hash::BitcoinNodeHash;

macro_rules! new_hash_wrapper {
    ($vis:vis $ident:ident) => {
        #[derive(
            BorshSerialize,
            BorshDeserialize,
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
        #[repr(transparent)]
        #[serde(transparent)]
        $vis struct $ident(#[serde(with = "hexstr_human_readable")] pub Hash);

        impl From<Hash> for $ident {
            fn from(inner: Hash) -> Self {
                Self(inner)
            }
        }

        impl From< $ident > for Hash {
            fn from(wrapped: $ident) -> Self {
                wrapped.0
            }
        }

        impl<'a> From<&'a $ident> for &'a Hash {
            fn from(wrapped: &'a $ident) -> Self {
                &wrapped.0
            }
        }

        impl FromStr for $ident {
            type Err = const_hex::FromHexError;
            fn from_str(s: &str) -> Result<Self, Self::Err> {
                Hash::from_hex(s).map(Self)
            }
        }

        impl std::fmt::Debug for $ident {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
            {
                write!(f, "{}", const_hex::encode(self.0))
            }
        }

        impl std::fmt::Display for $ident {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
            {
                write!(f, "{}", const_hex::encode(self.0))
            }
        }

        impl utoipa::PartialSchema for $ident {
            fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
                let obj =
                    utoipa::openapi::Object::with_type(utoipa::openapi::Type::String);
                utoipa::openapi::RefOr::T(utoipa::openapi::Schema::Object(obj))
            }
        }

        impl utoipa::ToSchema for $ident {
            fn name() -> std::borrow::Cow<'static, str> {
                std::borrow::Cow::Borrowed(stringify!($ident))
            }
        }
    }
}

new_hash_wrapper!(pub BlockHash);
new_hash_wrapper!(pub(crate) CoinbaseMerkleRoot);
new_hash_wrapper!(pub(crate) InputsMerkleRoot);
new_hash_wrapper!(pub(crate) OutputsMerkleRoot);
new_hash_wrapper!(pub(crate) TxMerkleRoot);
new_hash_wrapper!(pub MerkleRoot);
new_hash_wrapper!(pub CoinbaseTxid);
new_hash_wrapper!(pub Txid);

impl Txid {
    pub fn as_slice(&self) -> &[u8] {
        self.0.as_slice()
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

impl FromStr for M6id {
    type Err = <bitcoin::Txid as FromStr>::Err;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let inner = bitcoin::Txid::from_str(s)?;
        Ok(Self(inner))
    }
}

impl utoipa::PartialSchema for M6id {
    fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
        let obj =
            utoipa::openapi::Object::with_type(utoipa::openapi::Type::String);
        utoipa::openapi::RefOr::T(utoipa::openapi::Schema::Object(obj))
    }
}

impl utoipa::ToSchema for M6id {
    fn name() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("M6id")
    }
}

/// A block hash that is known to be non-zero. Bitcoin core often uses the
/// all-zeros block hash to represent `Option::<bitcoin::BlockHash>::None`.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[repr(transparent)]
#[serde(transparent)]
pub struct NonZeroBitcoinBlockHash(bitcoin::BlockHash);

impl NonZeroBitcoinBlockHash {
    pub fn new(block_hash: bitcoin::BlockHash) -> Option<Self> {
        if block_hash == bitcoin::BlockHash::all_zeros() {
            None
        } else {
            Some(Self(block_hash))
        }
    }
}

impl std::fmt::Display for NonZeroBitcoinBlockHash {
    #[inline(always)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

pub fn hash<T>(data: &T) -> Hash
where
    T: BorshSerialize + ?Sized,
{
    let mut hasher = blake3::Hasher::new();
    let () = borsh::to_writer(&mut hasher, data)
        .expect("failed to serialize with borsh to compute a hash");
    hasher.finalize().into()
}

/// Optimized hash function that reuses a thread-local scratch buffer
/// to avoid heap allocations for each hash operation. Useful for hashing many
/// small objects in tight loops.
pub fn hash_with_scratch_buffer<T>(data: &T) -> Hash
where
    T: BorshSerialize + ?Sized,
{
    thread_local! {
        static HASHER: std::cell::RefCell<blake3::Hasher> =
            std::cell::RefCell::new(Hasher::new());
    }

    HASHER.with(|hasher| {
        let mut hasher = hasher.borrow_mut();
        hasher.reset();
        borsh::to_writer(&mut *hasher, data)
            .expect("failed to serialize with borsh to compute a hash");
        hasher.finalize().into()
    })
}
