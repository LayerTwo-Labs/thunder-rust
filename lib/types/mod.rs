use bitcoin::{self, hashes::Hash as _};
use borsh::BorshSerialize;
use rustreexo::accumulator::{node_hash::NodeHash, pollard::Pollard};
use serde::{Deserialize, Serialize};
use std::{
    cmp::Ordering,
    collections::{BTreeMap, HashMap},
    sync::LazyLock,
};
use thiserror::Error;

mod address;
pub mod hashes;
pub mod proto;
pub mod schema;
mod transaction;

pub use address::Address;
pub use hashes::{hash, BlockHash, Hash, M6id, MerkleRoot, Txid};
pub use transaction::{
    AuthorizedTransaction, Body, Content as OutputContent, FilledTransaction,
    GetAddress, GetValue, InPoint, OutPoint, Output, PointedOutput,
    SpentOutput, Transaction, Verify,
};

pub const THIS_SIDECHAIN: u8 = 9;

#[derive(Debug, Error)]
#[error("Bitcoin amount overflow")]
pub struct AmountOverflowError;

#[derive(Debug, Error)]
#[error("Bitcoin amount underflow")]
pub struct AmountUnderflowError;

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

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum WithdrawalBundleStatus {
    Confirmed,
    Failed,
    Submitted,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct WithdrawalBundleEvent {
    pub m6id: M6id,
    pub status: WithdrawalBundleStatus,
}

pub static OP_DRIVECHAIN_SCRIPT: LazyLock<bitcoin::ScriptBuf> =
    LazyLock::new(|| {
        let mut script = bitcoin::ScriptBuf::new();
        script.push_opcode(bitcoin::opcodes::all::OP_RETURN);
        script.push_instruction(bitcoin::script::Instruction::PushBytes(
            &bitcoin::script::PushBytesBuf::from([THIS_SIDECHAIN]),
        ));
        script.push_opcode(bitcoin::opcodes::OP_TRUE);
        script
    });

#[derive(Debug, Error)]
enum WithdrawalBundleErrorInner {
    #[error("bundle too heavy: weight `{weight}` > max weight `{max_weight}`")]
    BundleTooHeavy { weight: u64, max_weight: u64 },
}

#[derive(Debug, Error)]
#[error("Withdrawal bundle error")]
pub struct WithdrawalBundleError(#[from] WithdrawalBundleErrorInner);

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WithdrawalBundle {
    spend_utxos: BTreeMap<transaction::OutPoint, transaction::Output>,
    tx: bitcoin::Transaction,
}

impl WithdrawalBundle {
    pub fn new(
        block_height: u32,
        fee: bitcoin::Amount,
        spend_utxos: BTreeMap<transaction::OutPoint, transaction::Output>,
        bundle_outputs: Vec<bitcoin::TxOut>,
    ) -> Result<Self, WithdrawalBundleError> {
        let inputs_commitment_txout = {
            // Create inputs commitment.
            let inputs: Vec<OutPoint> = [
                // Commit to inputs.
                spend_utxos.keys().copied().collect(),
                // Commit to block height.
                vec![OutPoint::Regular {
                    txid: [0; 32].into(),
                    vout: block_height,
                }],
            ]
            .concat();
            let commitment = hash(&inputs);
            let script_pubkey = bitcoin::script::Builder::new()
                .push_opcode(bitcoin::opcodes::all::OP_RETURN)
                .push_slice(commitment)
                .into_script();
            bitcoin::TxOut {
                value: bitcoin::Amount::ZERO,
                script_pubkey,
            }
        };
        let mainchain_fee_txout = {
            let script_pubkey = bitcoin::script::Builder::new()
                .push_opcode(bitcoin::opcodes::all::OP_RETURN)
                .push_slice(fee.to_sat().to_be_bytes())
                .into_script();
            bitcoin::TxOut {
                value: bitcoin::Amount::ZERO,
                script_pubkey,
            }
        };
        let outputs = Vec::from_iter(
            [mainchain_fee_txout, inputs_commitment_txout]
                .into_iter()
                .chain(bundle_outputs),
        );
        let tx = bitcoin::Transaction {
            version: bitcoin::transaction::Version::TWO,
            lock_time: bitcoin::blockdata::locktime::absolute::LockTime::ZERO,
            input: Vec::new(),
            output: outputs,
        };
        if tx.weight().to_wu() > bitcoin::policy::MAX_STANDARD_TX_WEIGHT as u64
        {
            Err(WithdrawalBundleErrorInner::BundleTooHeavy {
                weight: tx.weight().to_wu(),
                max_weight: bitcoin::policy::MAX_STANDARD_TX_WEIGHT as u64,
            })?;
        }
        Ok(Self { spend_utxos, tx })
    }

    pub fn compute_m6id(&self) -> M6id {
        M6id(self.tx.compute_txid())
    }

    pub fn spend_utxos(
        &self,
    ) -> &BTreeMap<transaction::OutPoint, transaction::Output> {
        &self.spend_utxos
    }

    pub fn tx(&self) -> &bitcoin::Transaction {
        &self.tx
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct TwoWayPegData {
    pub deposits: HashMap<transaction::OutPoint, transaction::Output>,
    pub deposit_block_hash: Option<bitcoin::BlockHash>,
    pub bundle_statuses: HashMap<M6id, WithdrawalBundleEvent>,
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
    pub value: bitcoin::Amount,
    pub main_fee: bitcoin::Amount,
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
