use std::{
    cmp::Ordering,
    collections::{BTreeMap, HashMap},
    sync::LazyLock,
};

use borsh::BorshSerialize;
use hashlink::{LinkedHashMap, linked_hash_map};
use rustreexo::accumulator::mem_forest::MemForest;
use serde::{Deserialize, Serialize};
use serde_with::serde_as;
use utoipa::ToSchema;

mod address;
pub use address::Address;
pub mod authorization;
pub use authorization::Authorization;
pub mod block;
pub use block::{Block, Body, Coinbase, Header};
pub mod error;
pub use error::{
    AmountOverflow as AmountOverflowError,
    AmountUnderflow as AmountUnderflowError, ComputeFee as ComputeFeeError,
    ComputeMerkleRoot as ComputeMerkleRootError, Utreexo as UtreexoError,
    WithdrawalBundle as WithdrawalBundleError,
};
pub mod hashes;
pub use hashes::{
    BlockHash, CoinbaseTxid, Hash, M6id, MerkleRoot, NonZeroBitcoinBlockHash,
    Txid, UtreexoNodeHash, hash, hash_with_scratch_buffer,
};
pub mod net;
pub mod schema;
pub mod state;
pub mod transaction;
pub use transaction::{
    Authorized, AuthorizedTransaction, FilledTransaction, GetAddress, GetValue,
    InPoint, OutPoint, OutPointKey, Output, OutputContent, PointedOutput,
    PointedOutputRef, SpentOutput, Transaction,
};
mod util;
pub mod wallet;

pub const THIS_SIDECHAIN: u8 = 9;

pub type UtreexoProof = rustreexo::accumulator::proof::Proof<UtreexoNodeHash>;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum WithdrawalBundleEventStatus {
    Confirmed,
    Failed,
    Submitted,
}

#[derive(
    Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, ToSchema,
)]
pub enum WithdrawalBundleStatus {
    Confirmed,
    /// Formerly pending bundle
    Dropped,
    Failed,
    Pending,
    Submitted,
    /// Submitted, but unexpected due to previously being dropped or failing.
    /// It may not be possible to account for this withdrawal bundle, if it
    /// double-spends UTXOs.
    SubmittedUnexpected,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct WithdrawalBundleEvent {
    pub m6id: M6id,
    pub status: WithdrawalBundleEventStatus,
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

#[serde_as]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize, ToSchema)]
pub struct WithdrawalBundle {
    #[schema(value_type = Vec<(transaction::OutPoint, transaction::Output)>)]
    #[serde_as(as = "serde_with::IfIsHumanReadable<serde_with::Seq<(_, _)>>")]
    spend_utxos: BTreeMap<transaction::OutPoint, transaction::Output>,
    #[schema(value_type = schema::BitcoinTransaction)]
    tx: bitcoin::Transaction,
}

impl WithdrawalBundle {
    /// Compute the size of a single txout
    pub const fn txout_size(spk_size: u32) -> Option<u32> {
        let Some(size) = (bitcoin::Amount::SIZE as u32)
            .checked_add(bitcoin::VarInt(spk_size as u64).size() as u32)
        else {
            return None;
        };
        size.checked_add(spk_size)
    }

    /// Predict the weight of a withdrawal bundle, based on the number of
    /// outputs (not including the commitment/treasury outputs) and the
    /// sum of sizes of txouts (not including the commitment/treasury outputs).
    /// Returns None if the predicted weight exceeds the maximum tx weight.
    pub const fn predict_weight(
        n_outputs: u32,
        sum_txout_sizes: u32,
    ) -> Option<bitcoin::Weight> {
        use bitcoin::{VarInt, Weight};
        const fn txin_base_size(script_sig_size: u32) -> Option<u32> {
            const OUTPOINT_SIZE: u8 = 36;
            const SEQUENCE_SIZE: u8 = 4;
            let script_sig_len_size: u8 =
                VarInt(script_sig_size as u64).size() as u8;
            let Some(res) = ((OUTPOINT_SIZE + script_sig_len_size) as u32)
                .checked_add(script_sig_size)
            else {
                return None;
            };
            res.checked_add(SEQUENCE_SIZE as u32)
        }
        const fn tx_base_size(
            n_inputs: u32,
            sum_txin_base_sizes: u32,
            n_outputs: u32,
            sum_txout_sizes: u32,
        ) -> Option<u32> {
            const VERSION_SIZE: u8 = 4;
            const fn vin_base_size(
                n_inputs: u32,
                sum_txin_base_sizes: u32,
            ) -> Option<u32> {
                let len_size = VarInt(n_inputs as u64).size() as u8;
                (len_size as u32).checked_add(sum_txin_base_sizes)
            }
            const fn vout_size(
                n_outputs: u32,
                sum_txout_sizes: u32,
            ) -> Option<u32> {
                let len_size = VarInt(n_outputs as u64).size() as u8;
                (len_size as u32).checked_add(sum_txout_sizes)
            }
            const LOCKTIME_SIZE: u8 = bitcoin::absolute::LockTime::SIZE as u8;
            let res = VERSION_SIZE as u32;
            let Some(vin_base_size) =
                vin_base_size(n_inputs, sum_txin_base_sizes)
            else {
                return None;
            };
            let Some(res) = res.checked_add(vin_base_size) else {
                return None;
            };
            let Some(vout_size) = vout_size(n_outputs, sum_txout_sizes) else {
                return None;
            };
            let Some(res) = res.checked_add(vout_size) else {
                return None;
            };
            res.checked_add(LOCKTIME_SIZE as u32)
        }
        const N_INPUTS: u32 = 1;
        const SUM_TXIN_BASE_SIZES: u32 = {
            const TREASURY_TXIN_BASE_SIZE: u32 = {
                const TREASURY_SCRIPT_SIG_SIZE: u32 = 0;
                txin_base_size(TREASURY_SCRIPT_SIG_SIZE).unwrap()
            };
            TREASURY_TXIN_BASE_SIZE
        };
        let Some(n_outputs) = n_outputs.checked_add(2) else {
            return None;
        };
        let Some(sum_txout_sizes) = ({
            const INPUTS_COMMITMENT_TXOUT_SIZE: u32 = {
                const INPUTS_COMMITMENT_OUTPUT_SPK_SIZE: u8 = 34;
                WithdrawalBundle::txout_size(
                    INPUTS_COMMITMENT_OUTPUT_SPK_SIZE as u32,
                )
                .unwrap()
            };
            const MAINCHAIN_FEE_COMMITMENT_TXOUT_SIZE: u32 = {
                const MAINCHAIN_FEE_COMMITMENT_OUTPUT_SPK_SIZE: u8 = 10;
                WithdrawalBundle::txout_size(
                    MAINCHAIN_FEE_COMMITMENT_OUTPUT_SPK_SIZE as u32,
                )
                .unwrap()
            };
            (INPUTS_COMMITMENT_TXOUT_SIZE + MAINCHAIN_FEE_COMMITMENT_TXOUT_SIZE)
                .checked_add(sum_txout_sizes)
        }) else {
            return None;
        };
        let Some(tx_base_size) = tx_base_size(
            N_INPUTS,
            SUM_TXIN_BASE_SIZES,
            n_outputs,
            sum_txout_sizes,
        ) else {
            return None;
        };
        let Some(tx_weight_wu) =
            (tx_base_size as u64).checked_mul(Weight::WITNESS_SCALE_FACTOR)
        else {
            return None;
        };
        if tx_weight_wu <= bitcoin::Transaction::MAX_STANDARD_WEIGHT.to_wu() {
            Some(Weight::from_wu(tx_weight_wu))
        } else {
            None
        }
    }

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
            Err(error::withdrawal_bundle::Inner::BundleTooHeavy {
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
        // A *total* order (lexicographic by main_fee, value, main_address). The
        // previous `OR of >` was not antisymmetric/transitive, so the
        // withdrawal-bundle output order (and hence compute_m6id) depended on
        // HashMap iteration order and could differ across nodes. A real total order makes
        // the sorted bundle canonical regardless of aggregation order.
        (self.main_fee, self.value, &self.main_address).cmp(&(
            other.main_fee,
            other.value,
            &other.main_address,
        ))
    }
}

impl PartialOrd for AggregatedWithdrawal {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Manage accumulator diffs.
/// Insertions and removals 'cancel out' exactly once.
/// Inserting twice will cause one insertion.
/// Removing twice will cause one deletion.
/// Inserting and then removing will have no overall effect,
/// but a second removal will still cause a deletion.
#[derive(Clone, Debug)]
pub struct AccumulatorDiff {
    /// `true` indicates insertion, `false` indicates removal.
    diff: LinkedHashMap<UtreexoNodeHash, bool>,
    /// Total number of insertions still represented in `diff`.
    insertions: usize,
    /// Total number of deletions still represented in `diff`.
    deletions: usize,
}

impl Default for AccumulatorDiff {
    fn default() -> Self {
        Self {
            diff: LinkedHashMap::new(),
            insertions: 0,
            deletions: 0,
        }
    }
}

impl AccumulatorDiff {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            diff: LinkedHashMap::with_capacity(capacity),
            insertions: 0,
            deletions: 0,
        }
    }

    pub fn insert(&mut self, utxo_hash: UtreexoNodeHash) {
        match self.diff.entry(utxo_hash) {
            linked_hash_map::Entry::Occupied(entry) => {
                if !entry.get() {
                    entry.remove();
                    debug_assert!(self.deletions > 0);
                    self.deletions -= 1;
                }
            }
            linked_hash_map::Entry::Vacant(entry) => {
                entry.insert(true);
                self.insertions += 1;
            }
        }
    }

    pub fn remove(&mut self, utxo_hash: UtreexoNodeHash) {
        match self.diff.entry(utxo_hash) {
            linked_hash_map::Entry::Occupied(entry) => {
                if *entry.get() {
                    entry.remove();
                    debug_assert!(self.insertions > 0);
                    self.insertions -= 1;
                }
            }
            linked_hash_map::Entry::Vacant(entry) => {
                entry.insert(false);
                self.deletions += 1;
            }
        }
    }

    /// Returns the number of tracked insertions and deletions, in that order.
    pub fn counts(&self) -> (usize, usize) {
        (self.insertions, self.deletions)
    }

    pub fn is_empty(&self) -> bool {
        self.diff.is_empty()
    }
}

#[derive(Debug, Default)]
#[repr(transparent)]
pub struct Accumulator(pub MemForest<UtreexoNodeHash>);

impl Accumulator {
    pub fn apply_diff(
        &mut self,
        diff: AccumulatorDiff,
    ) -> Result<(), UtreexoError> {
        let AccumulatorDiff {
            diff,
            insertions: n_insertions,
            deletions: n_deletions,
        } = diff;
        let (mut insertions, mut deletions) = (
            Vec::with_capacity(n_insertions),
            Vec::with_capacity(n_deletions),
        );
        for (utxo_hash, insert) in diff {
            if insert {
                insertions.push(utxo_hash);
            } else {
                deletions.push(utxo_hash);
            }
        }
        tracing::trace!(
            leaves = %self.0.leaves,
            roots = ?self.get_roots(),
            insertions = ?insertions,
            deletions = ?deletions,
            "Applying diff"
        );
        let () = self
            .0
            .modify(&insertions, &deletions)
            .map_err(UtreexoError)?;
        tracing::debug!(
            leaves = %self.0.leaves,
            roots = ?self.get_roots(),
            "Applied diff"
        );
        Ok(())
    }

    pub fn get_roots(&self) -> Vec<UtreexoNodeHash> {
        self.0
            .get_roots()
            .iter()
            .map(|node| node.get_data())
            .collect()
    }

    pub fn prove(
        &self,
        targets: &[UtreexoNodeHash],
    ) -> Result<UtreexoProof, UtreexoError> {
        self.0.prove(targets).map_err(UtreexoError)
    }

    pub fn verify(
        &self,
        proof: &UtreexoProof,
        del_hashes: &[UtreexoNodeHash],
    ) -> Result<bool, UtreexoError> {
        self.0.verify(proof, del_hashes).map_err(UtreexoError)
    }
}

impl<'de> Deserialize<'de> for Accumulator {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let bytes: Vec<u8> =
            <Vec<_> as Deserialize>::deserialize(deserializer)?;
        let mem_forest = MemForest::deserialize(&*bytes)
            .inspect_err(|err| {
                tracing::debug!("deserialize err: {err}\n bytes: {bytes:?}")
            })
            .map_err(<D::Error as serde::de::Error>::custom)?;
        Ok(Self(mem_forest))
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
    #[borsh(serialize_with = "util::borsh::serialize::bitcoin_block_hash")]
    pub main_block_hash: bitcoin::BlockHash,
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
#[cfg_attr(
    feature = "clap",
    derive(clap::ValueEnum, strum::Display),
    strum(serialize_all = "lowercase")
)]
pub enum Network {
    #[default]
    Betanet,
    Forknet,
    Regtest,
    Signet,
}

/// Semver-compatible version
#[derive(
    BorshSerialize,
    Clone,
    Copy,
    Debug,
    Deserialize,
    Eq,
    Hash,
    Ord,
    PartialEq,
    PartialOrd,
    Serialize,
)]
pub struct Version {
    pub major: u64,
    pub minor: u64,
    pub patch: u64,
}

impl std::fmt::Display for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl From<semver::Version> for Version {
    fn from(version: semver::Version) -> Self {
        let semver::Version {
            major,
            minor,
            patch,
            pre: _,
            build: _,
        } = version;
        Self {
            major,
            minor,
            patch,
        }
    }
}

#[cfg(test)]
mod withdrawal_bundle_order_regression {
    use super::*;
    use std::collections::{BTreeMap, HashMap};

    use bitcoin::{Address, Amount, address::NetworkUnchecked};

    fn aw(value: u64, main_fee: u64) -> AggregatedWithdrawal {
        // value/main_fee drive the comparison; one address is enough to expose it.
        let addr: Address<NetworkUnchecked> =
            "bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4"
                .parse()
                .unwrap();
        AggregatedWithdrawal {
            spend_utxos: HashMap::new(),
            main_address: addr,
            value: Amount::from_sat(value),
            main_fee: Amount::from_sat(main_fee),
        }
    }

    // Build the bundle m6id exactly as `collect_withdrawal_bundle` does, for a given
    // (HashMap-determined) input order.
    fn bundle_m6id(mut aggregated: Vec<AggregatedWithdrawal>) -> M6id {
        aggregated.sort_by_key(|a| std::cmp::Reverse(a.clone()));
        let outputs: Vec<bitcoin::TxOut> = aggregated
            .iter()
            .map(|a| bitcoin::TxOut {
                value: a.value,
                script_pubkey: a
                    .main_address
                    .assume_checked_ref()
                    .script_pubkey(),
            })
            .collect();
        WithdrawalBundle::new(0, Amount::ZERO, BTreeMap::new(), outputs)
            .unwrap()
            .compute_m6id()
    }

    // The withdrawal bundle's m6id must not depend on the order in which withdrawals
    // were aggregated (HashMap iteration order is randomized per process). Before the
    // total-order fix, the comparator was non-transitive and this failed.
    #[test]
    fn m6id_is_independent_of_aggregation_order() {
        let a = aw(1, 3);
        let b = aw(3, 2);
        let c = aw(2, 1);
        let m = bundle_m6id(vec![a.clone(), b.clone(), c.clone()]);
        for perm in [
            vec![c.clone(), b.clone(), a.clone()],
            vec![b.clone(), a.clone(), c.clone()],
            vec![a.clone(), c.clone(), b.clone()],
        ] {
            assert_eq!(
                m,
                bundle_m6id(perm),
                "m6id must not depend on aggregation order"
            );
        }
    }
}
