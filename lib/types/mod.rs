use bip300301::bitcoin::{self, hashes::Hash as _};
use borsh::BorshSerialize;
use rustreexo::accumulator::{node_hash::NodeHash, pollard::Pollard};
use serde::{Deserialize, Serialize};
use std::{
    cmp::Ordering,
    collections::{BTreeMap, HashMap},
};

mod address;
pub mod hashes;
mod transaction;

pub use address::Address;
pub use hashes::{hash, BlockHash, Hash, MerkleRoot, Txid};
pub use transaction::{
    AuthorizedTransaction, Body, Content as OutputContent, FilledTransaction,
    GetAddress, GetValue, InPoint, OutPoint, Output, PointedOutput,
    SpentOutput, Transaction, Verify,
};

/// (de)serialize as hex strings for human-readable forms like json,
/// and default serialization for non human-readable formats like bincode
mod serde_hexstr_human_readable {
    use hex::{FromHex, ToHex};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S, T>(data: T, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
        T: Serialize + ToHex,
    {
        if serializer.is_human_readable() {
            hex::serde::serialize(data, serializer)
        } else {
            data.serialize(serializer)
        }
    }

    pub fn deserialize<'de, D, T>(deserializer: D) -> Result<T, D::Error>
    where
        D: Deserializer<'de>,
        T: Deserialize<'de> + FromHex,
        <T as FromHex>::Error: std::fmt::Display,
    {
        if deserializer.is_human_readable() {
            hex::serde::deserialize(deserializer)
        } else {
            T::deserialize(deserializer)
        }
    }
}

fn borsh_serialize_utreexo_nodehash<W>(
    node_hash: &NodeHash,
    writer: &mut W,
) -> borsh::io::Result<()>
where
    W: borsh::io::Write,
{
    let bytes: &[u8; 32] = node_hash;
    borsh::BorshSerialize::serialize(bytes, writer)
}

fn borsh_serialize_utreexo_roots<W>(
    roots: &[NodeHash],
    writer: &mut W,
) -> borsh::io::Result<()>
where
    W: borsh::io::Write,
{
    #[derive(BorshSerialize)]
    #[repr(transparent)]
    struct SerializeNodeHash<'a>(
        #[borsh(serialize_with = "borsh_serialize_utreexo_nodehash")]
        &'a NodeHash,
    );
    let roots: Vec<SerializeNodeHash> =
        roots.iter().map(SerializeNodeHash).collect();
    borsh::BorshSerialize::serialize(&roots, writer)
}

fn borsh_serialize_bitcoin_block_hash<W>(
    block_hash: &bitcoin::BlockHash,
    writer: &mut W,
) -> borsh::io::Result<()>
where
    W: borsh::io::Write,
{
    let bytes: &[u8; 32] = block_hash.as_ref();
    borsh::BorshSerialize::serialize(bytes, writer)
}

#[derive(
    BorshSerialize, Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize,
)]
pub struct Header {
    pub merkle_root: MerkleRoot,
    pub prev_side_hash: BlockHash,
    #[borsh(serialize_with = "borsh_serialize_bitcoin_block_hash")]
    pub prev_main_hash: bitcoin::BlockHash,
    /// Utreexo roots
    #[borsh(serialize_with = "borsh_serialize_utreexo_roots")]
    pub roots: Vec<NodeHash>,
}

impl Header {
    pub fn hash(&self) -> BlockHash {
        hash(self).into()
    }
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum WithdrawalBundleStatus {
    Failed,
    Confirmed,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WithdrawalBundle {
    pub spend_utxos: BTreeMap<transaction::OutPoint, transaction::Output>,
    pub transaction: bitcoin::Transaction,
}

#[derive(Default, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TwoWayPegData {
    pub deposits: HashMap<transaction::OutPoint, transaction::Output>,
    pub deposit_block_hash: Option<bitcoin::BlockHash>,
    pub bundle_statuses: HashMap<bitcoin::Txid, WithdrawalBundleStatus>,
}

/*
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DisconnectData {
    pub spent_utxos: HashMap<types::OutPoint, Output>,
    pub deposits: Vec<types::OutPoint>,
    pub pending_bundles: Vec<bitcoin::Txid>,
    pub spent_bundles: HashMap<bitcoin::Txid, Vec<types::OutPoint>>,
    pub spent_withdrawals: HashMap<types::OutPoint, Output>,
    pub failed_withdrawals: Vec<bitcoin::Txid>,
}
*/

#[derive(Eq, PartialEq, Clone, Debug)]
pub struct AggregatedWithdrawal {
    pub spend_utxos: HashMap<OutPoint, transaction::Output>,
    pub main_address: bitcoin::Address<bitcoin::address::NetworkUnchecked>,
    pub value: u64,
    pub main_fee: u64,
}

impl Ord for AggregatedWithdrawal {
    fn cmp(&self, other: &Self) -> Ordering {
        if self == other {
            Ordering::Equal
        } else if self.main_fee > other.main_fee
            || self.value > other.value
            || self.main_address > other.main_address
        {
            Ordering::Greater
        } else {
            Ordering::Less
        }
    }
}

impl PartialOrd for AggregatedWithdrawal {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Default)]
#[repr(transparent)]
pub struct Accumulator(pub Pollard);

impl<'de> Deserialize<'de> for Accumulator {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let bytes: Vec<u8> =
            <Vec<_> as Deserialize>::deserialize(deserializer)?;
        let pollard = Pollard::deserialize(&*bytes)
            .inspect_err(|err| {
                tracing::debug!("deserialize err: {err}\n bytes: {bytes:?}")
            })
            .map_err(<D::Error as serde::de::Error>::custom)?;
        Ok(Self(pollard))
    }
}

impl Serialize for Accumulator {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut bytes = Vec::new();
        self.0
            .serialize(&mut bytes)
            .map_err(<S::Error as serde::ser::Error>::custom)?;
        <Vec<_> as Serialize>::serialize(&bytes, serializer)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum BmmResult {
    Verified,
    Failed,
}

/// A tip refers to both a sidechain block AND the mainchain block that commits
/// to it.
#[derive(
    BorshSerialize,
    Clone,
    Copy,
    Debug,
    Deserialize,
    Eq,
    Hash,
    PartialEq,
    Serialize,
)]
pub struct Tip {
    pub block_hash: BlockHash,
    #[borsh(serialize_with = "borsh_serialize_bitcoin_block_hash")]
    pub main_block_hash: bitcoin::BlockHash,
}

impl Default for Tip {
    fn default() -> Self {
        Self {
            block_hash: BlockHash::default(),
            main_block_hash: bitcoin::BlockHash::all_zeros(),
        }
    }
}
