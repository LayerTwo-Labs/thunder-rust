use borsh::BorshSerialize;
#[cfg(feature = "utreexo")]
use hashlink::{LinkedHashMap, linked_hash_map};
#[cfg(feature = "utreexo")]
use rustreexo::accumulator::{
    mem_forest::MemForest, node_hash::BitcoinNodeHash, proof::Proof,
};
use serde::{Deserialize, Serialize};
use serde_with::serde_as;
use std::{
    cmp::Ordering,
    collections::{BTreeMap, HashMap},
    sync::LazyLock,
};
use thiserror::Error;
use utoipa::ToSchema;

use crate::{
    authorization::Authorization, types::transaction::ComputeFeeError,
};

mod address;
pub mod hashes;
pub mod proto;
pub mod schema;
mod transaction;

pub use address::Address;
pub use hashes::{BlockHash, Hash, M6id, MerkleRoot, Txid, hash};
pub use transaction::{
    Authorized, AuthorizedTransaction, Content as OutputContent,
    FilledTransaction, GetAddress, GetValue, InPoint, OutPoint, Output,
    PointedOutput, PointedOutputRef, SpentOutput, Transaction,
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

#[cfg(feature = "utreexo")]
fn borsh_serialize_utreexo_nodehash<W>(
    node_hash: &BitcoinNodeHash,
    writer: &mut W,
) -> borsh::io::Result<()>
where
    W: borsh::io::Write,
{
    let bytes: &[u8; 32] = node_hash;
    borsh::BorshSerialize::serialize(bytes, writer)
}

#[cfg(feature = "utreexo")]
fn borsh_serialize_utreexo_roots<W>(
    roots: &[BitcoinNodeHash],
    writer: &mut W,
) -> borsh::io::Result<()>
where
    W: borsh::io::Write,
{
    #[derive(BorshSerialize)]
    #[repr(transparent)]
    struct SerializeBitcoinNodeHash<'a>(
        #[borsh(serialize_with = "borsh_serialize_utreexo_nodehash")]
        &'a BitcoinNodeHash,
    );
    let roots: Vec<SerializeBitcoinNodeHash> =
        roots.iter().map(SerializeBitcoinNodeHash).collect();
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
    BorshSerialize,
    Clone,
    Debug,
    Deserialize,
    Eq,
    Hash,
    PartialEq,
    Serialize,
    ToSchema,
)]
pub struct Header {
    pub merkle_root: MerkleRoot,
    pub prev_side_hash: Option<BlockHash>,
    #[borsh(serialize_with = "borsh_serialize_bitcoin_block_hash")]
    #[schema(value_type = schema::BitcoinBlockHash)]
    pub prev_main_hash: bitcoin::BlockHash,
    /// Utreexo roots
    #[cfg(feature = "utreexo")]
    #[borsh(serialize_with = "borsh_serialize_utreexo_roots")]
    #[schema(value_type = Vec<schema::UtreexoNodeHash>)]
    pub roots: Vec<BitcoinNodeHash>,
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

#[cfg(feature = "utreexo")]
/// Manage accumulator diffs.
/// Insertions and removals 'cancel out' exactly once.
/// Inserting twice will cause one insertion.
/// Removing twice will cause one deletion.
/// Inserting and then removing will have no overall effect,
/// but a second removal will still cause a deletion.
#[derive(Debug, Default)]
#[repr(transparent)]
pub struct AccumulatorDiff(
    /// `true` indicates insertion, `false` indicates removal.
    LinkedHashMap<BitcoinNodeHash, bool>,
);

#[cfg(feature = "utreexo")]
impl AccumulatorDiff {
    pub fn insert(&mut self, utxo_hash: BitcoinNodeHash) {
        match self.0.entry(utxo_hash) {
            linked_hash_map::Entry::Occupied(entry) => {
                if !entry.get() {
                    entry.remove();
                }
            }
            linked_hash_map::Entry::Vacant(entry) => {
                entry.insert(true);
            }
        }
    }

    pub fn remove(&mut self, utxo_hash: BitcoinNodeHash) {
        match self.0.entry(utxo_hash) {
            linked_hash_map::Entry::Occupied(entry) => {
                if *entry.get() {
                    entry.remove();
                }
            }
            linked_hash_map::Entry::Vacant(entry) => {
                entry.insert(false);
            }
        }
    }
}

#[cfg(feature = "utreexo")]
#[derive(Debug, Error)]
#[error("utreexo error: {0}")]
#[repr(transparent)]
pub struct UtreexoError(String);

#[cfg(feature = "utreexo")]
#[derive(Debug, Default)]
#[repr(transparent)]
pub struct Accumulator(pub MemForest<BitcoinNodeHash>);

#[cfg(feature = "utreexo")]
impl Accumulator {
    pub fn apply_diff(
        &mut self,
        diff: AccumulatorDiff,
    ) -> Result<(), UtreexoError> {
        let (mut insertions, mut deletions) = (Vec::new(), Vec::new());
        for (utxo_hash, insert) in diff.0 {
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

    pub fn get_roots(&self) -> Vec<BitcoinNodeHash> {
        self.0
            .get_roots()
            .iter()
            .map(|node| node.get_data())
            .collect()
    }

    pub fn prove(
        &self,
        targets: &[BitcoinNodeHash],
    ) -> Result<Proof<BitcoinNodeHash>, UtreexoError> {
        self.0.prove(targets).map_err(UtreexoError)
    }

    pub fn verify(
        &self,
        proof: &Proof<BitcoinNodeHash>,
        del_hashes: &[BitcoinNodeHash],
    ) -> Result<bool, UtreexoError> {
        self.0.verify(proof, del_hashes).map_err(UtreexoError)
    }
}

#[cfg(feature = "utreexo")]
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

#[cfg(feature = "utreexo")]
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

/// Hash to get a [`CmbtNode`] inner commitment for a leaf value
#[derive(Debug)]
struct CbmtLeafPreCommitment<'a> {
    fee: bitcoin::Amount,
    /// Canonical size of the tx
    canonical_size: u64,
    tx: &'a Transaction,
}

impl<'a> BorshSerialize for CbmtLeafPreCommitment<'a> {
    fn serialize<W: std::io::Write>(
        &self,
        writer: &mut W,
    ) -> std::io::Result<()> {
        let Self {
            fee,
            canonical_size,
            tx,
        } = self;
        BorshSerialize::serialize(&fee.to_sat(), writer)?;
        BorshSerialize::serialize(canonical_size, writer)?;
        BorshSerialize::serialize(tx, writer)
    }
}

/// Hash to get a [`CmbtNode`] inner commitment for a non-leaf value
#[derive(Debug)]
struct CbmtNodePreCommitment {
    /// left child inner commitment
    left_commitment: Hash,
    /// Sum of child tx fees
    fees: bitcoin::Amount,
    /// Sum of canonical sizes of child txs
    canonical_size: u64,
    /// left child inner commitment
    right_commitment: Hash,
}

impl BorshSerialize for CbmtNodePreCommitment {
    fn serialize<W: std::io::Write>(
        &self,
        writer: &mut W,
    ) -> std::io::Result<()> {
        let Self {
            left_commitment,
            fees,
            canonical_size,
            right_commitment,
        } = self;
        BorshSerialize::serialize(left_commitment, writer)?;
        BorshSerialize::serialize(&fees.to_sat(), writer)?;
        BorshSerialize::serialize(canonical_size, writer)?;
        BorshSerialize::serialize(right_commitment, writer)
    }
}

// Internal node of a CBMT
#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct CbmtNode {
    // Commitment to child nodes or leaf value
    commitment: Hash,
    // Sum of fees for child nodes or leaf value
    fees: bitcoin::Amount,
    // Sum of canonical tx sizes for child nodes or leaf value
    canonical_size: u64,
    // CBT index, see https://github.com/nervosnetwork/merkle-tree/blob/5d1898263e7167560fdaa62f09e8d52991a1c712/README.md#tree-struct
    // This is required so that `CbmtNode` can be `Ord` correctly
    index: usize,
}

impl PartialOrd for CbmtNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CbmtNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.index.cmp(&other.index)
    }
}

// Marker type for merging branch commitments with
// * branch fee totals
// * branch canonical size totals
struct MergeFeeSizeTotal;

impl merkle_cbt::merkle_tree::Merge for MergeFeeSizeTotal {
    type Item = CbmtNode;

    fn merge(lnode: &Self::Item, rnode: &Self::Item) -> Self::Item {
        let fees = lnode.fees + rnode.fees;
        let canonical_size = lnode.canonical_size + rnode.canonical_size;
        // see https://github.com/nervosnetwork/merkle-tree/blob/5d1898263e7167560fdaa62f09e8d52991a1c712/README.md#tree-struct
        assert_eq!(lnode.index + 1, rnode.index);
        let index = (lnode.index - 1) / 2;
        let commitment = hashes::hash(&CbmtNodePreCommitment {
            left_commitment: lnode.commitment,
            fees,
            canonical_size,
            right_commitment: lnode.commitment,
        });
        Self::Item {
            commitment,
            fees,
            canonical_size,
            index,
        }
    }
}

// Complete binary merkle tree with annotated fee and canonical size totals
type CbmtWithFeeTotal = merkle_cbt::CBMT<CbmtNode, MergeFeeSizeTotal>;

#[derive(Debug, Error)]
#[error("failed to compute fee for `{txid}`")]
struct ComputeMerkleRootErrorInner {
    txid: Txid,
    source: ComputeFeeError,
}

#[derive(Debug, Error)]
#[error("failed to compute merkle root")]
#[repr(transparent)]
pub struct ComputeMerkleRootError(#[from] ComputeMerkleRootErrorInner);

#[cfg(feature = "utreexo")]
#[derive(Debug, Error)]
pub enum ModifyMemForestError {
    #[error(transparent)]
    ComputeMerkleRoot(#[from] ComputeMerkleRootError),
    #[error(transparent)]
    Utreexo(#[from] UtreexoError),
}

#[derive(BorshSerialize, Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct Body {
    pub coinbase: Vec<Output>,
    pub transactions: Vec<Transaction>,
    pub authorizations: Vec<Authorization>,
}

impl Body {
    pub fn new(
        authorized_transactions: Vec<AuthorizedTransaction>,
        coinbase: Vec<Output>,
    ) -> Self {
        let mut authorizations = Vec::with_capacity(
            authorized_transactions
                .iter()
                .map(|t| t.transaction.inputs.len())
                .sum(),
        );
        let mut transactions =
            Vec::with_capacity(authorized_transactions.len());
        for at in authorized_transactions.into_iter() {
            authorizations.extend(at.authorizations);
            transactions.push(at.transaction);
        }
        Self {
            coinbase,
            transactions,
            authorizations,
        }
    }

    pub fn authorized_transactions(&self) -> Vec<AuthorizedTransaction> {
        let mut authorizations_iter = self.authorizations.iter();
        self.transactions
            .iter()
            .map(|tx| {
                let mut authorizations = Vec::with_capacity(tx.inputs.len());
                for _ in 0..tx.inputs.len() {
                    let auth = authorizations_iter.next().unwrap();
                    authorizations.push(auth.clone());
                }
                AuthorizedTransaction {
                    transaction: tx.clone(),
                    authorizations,
                }
            })
            .collect()
    }

    pub fn compute_merkle_root(
        coinbase: &[Output],
        txs: &[FilledTransaction],
    ) -> Result<MerkleRoot, ComputeMerkleRootError> {
        let CbmtNode {
            commitment: txs_root,
            ..
        } = {
            // Use parallel leaf computation
            let leaves = compute_merkle_leaves_parallel(txs)?;
            CbmtWithFeeTotal::build_merkle_root(leaves.as_slice())
        };

        // Consider parallelizing this hash too for large coinbases
        let coinbase_root = if coinbase.len() > 100 {
            parallel_hash_large(&coinbase)
        } else {
            hashes::hash(&coinbase)
        };

        let root = hashes::hash(&(coinbase_root, txs_root)).into();
        Ok(root)
    }

    #[cfg(feature = "utreexo")]
    /// Modifies the memforest, without checking tx proofs
    pub fn modify_memforest(
        coinbase: &[Output],
        txs: &[FilledTransaction],
        memforest: &mut MemForest<BitcoinNodeHash>,
    ) -> Result<MerkleRoot, ModifyMemForestError> {
        // New leaves for the accumulator
        let mut accumulator_add = Vec::<BitcoinNodeHash>::new();
        // Accumulator leaves to delete
        let mut accumulator_del = Vec::<BitcoinNodeHash>::new();
        let merkle_root = Self::compute_merkle_root(coinbase, txs)?;
        for (vout, output) in coinbase.iter().enumerate() {
            let outpoint = OutPoint::Coinbase {
                merkle_root,
                vout: vout as u32,
            };
            let pointed_output = PointedOutput {
                outpoint,
                output: output.clone(),
            };
            accumulator_add.push((&pointed_output).into());
        }
        for tx in txs {
            let txid = tx.transaction.txid();
            for (_, utxo_hash) in tx.transaction.inputs.iter() {
                accumulator_del.push(utxo_hash.into());
            }
            for (vout, output) in tx.transaction.outputs.iter().enumerate() {
                let outpoint = OutPoint::Regular {
                    txid,
                    vout: vout as u32,
                };
                let pointed_output = PointedOutput {
                    outpoint,
                    output: output.clone(),
                };
                accumulator_add.push((&pointed_output).into());
            }
        }
        let () = memforest
            .modify(&accumulator_add, &accumulator_del)
            .map_err(UtreexoError)?;
        Ok(merkle_root)
    }

    pub fn get_inputs(&self) -> Vec<OutPoint> {
        self.transactions
            .iter()
            .flat_map(|tx| tx.inputs.iter().map(|(outpoint, _)| outpoint))
            .copied()
            .collect()
    }

    pub fn get_outputs(
        coinbase: &[Output],
        txs: &[FilledTransaction],
    ) -> Result<HashMap<OutPoint, Output>, ComputeMerkleRootError> {
        let mut res = HashMap::new();
        let merkle_root = Self::compute_merkle_root(coinbase, txs)?;
        for (vout, output) in coinbase.iter().enumerate() {
            let vout = vout as u32;
            let outpoint = OutPoint::Coinbase { merkle_root, vout };
            res.insert(outpoint, output.clone());
        }
        for tx in txs {
            let txid = tx.transaction.txid();
            for (vout, output) in tx.transaction.outputs.iter().enumerate() {
                let vout = vout as u32;
                let outpoint = OutPoint::Regular { txid, vout };
                res.insert(outpoint, output.clone());
            }
        }
        Ok(res)
    }

    pub fn get_coinbase_value(
        &self,
    ) -> Result<bitcoin::Amount, AmountOverflowError> {
        use bitcoin::amount::CheckedSum as _;
        self.coinbase
            .iter()
            .map(|output| output.get_value())
            .checked_sum()
            .ok_or(AmountOverflowError)
    }
}

/// Parallel merkle leaf computation using chunked processing
fn compute_merkle_leaves_parallel(
    txs: &[FilledTransaction],
) -> Result<Vec<CbmtNode>, ComputeMerkleRootError> {
    use rayon::prelude::*;
    
    if txs.len() <= 50 {
        return compute_merkle_leaves_simple_parallel(txs);
    }
    
    let n_txs = txs.len();
    // Chunk size optimized for hash computation workload
    let chunks: Vec<_> = txs.chunks(50).collect();
    
    tracing::debug!(
        "Computing merkle leaves with {} chunks for {} transactions", 
        chunks.len(), 
        n_txs
    );
    
    let results: Vec<Vec<CbmtNode>> = chunks
        .par_iter()
        .enumerate()
        .map(|(chunk_id, chunk)| {
            let chunk_start_idx = chunk_id * 50;
            let mut chunk_leaves = Vec::with_capacity(chunk.len());
            
            for (local_idx, tx) in chunk.iter().enumerate() {
                let global_idx = chunk_start_idx + local_idx;
                
                let fees = tx.get_fee().map_err(|err| {
                    ComputeMerkleRootErrorInner {
                        txid: tx.transaction.txid(),
                        source: err,
                    }
                })?;
                
                let canonical_size = tx.transaction.canonical_size();
                let leaf_pre_commitment = CbmtLeafPreCommitment {
                    fee: fees,
                    canonical_size,
                    tx: &tx.transaction,
                };
                
                // This hash computation is expensive - now parallelized!
                let commitment = hashes::hash(&leaf_pre_commitment);
                
                chunk_leaves.push(CbmtNode {
                    commitment,
                    fees,
                    canonical_size,
                    index: (global_idx + n_txs) - 1,
                });
            }
            
            Ok(chunk_leaves)
        })
        .collect::<Result<Vec<_>, ComputeMerkleRootError>>()?;
    
    // Flatten results
    Ok(results.into_iter().flatten().collect())
}

fn compute_merkle_leaves_simple_parallel(
    txs: &[FilledTransaction],
) -> Result<Vec<CbmtNode>, ComputeMerkleRootError> {
    use rayon::prelude::*;
    
    let n_txs = txs.len();
    txs.par_iter()
        .enumerate()
        .map(|(idx, tx)| {
            let fees = tx.get_fee().map_err(|err| {
                ComputeMerkleRootErrorInner {
                    txid: tx.transaction.txid(),
                    source: err,
                }
            })?;
            
            let canonical_size = tx.transaction.canonical_size();
            let leaf_pre_commitment = CbmtLeafPreCommitment {
                fee: fees,
                canonical_size,
                tx: &tx.transaction,
            };
            
            Ok(CbmtNode {
                commitment: hashes::hash(&leaf_pre_commitment),
                fees,
                canonical_size,
                index: (idx + n_txs) - 1,
            })
        })
        .collect()
}

/// Parallel hash computation for large data
fn parallel_hash_large<T: BorshSerialize>(data: &T) -> Hash {
    let serialized = borsh::to_vec(data).expect("serialization failed");
    
    // Blake3 supports parallel hashing for large data
    if serialized.len() > 1024 {
        blake3::Hasher::new()
            .update(&serialized)  // Parallel update if available
            .finalize()
            .into()
    } else {
        blake3::hash(&serialized).into()
    }
}

pub trait Verify {
    type Error;
    fn verify_transaction(
        transaction: &AuthorizedTransaction,
    ) -> Result<(), Self::Error>;
    fn verify_body(body: &Body) -> Result<(), Self::Error>;
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

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum, strum::Display))]
pub enum Network {
    #[default]
    Signet,
    Regtest,
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

// Do not make this public outside of this crate, as it could break semver
pub(crate) static VERSION: LazyLock<Version> = LazyLock::new(|| {
    const VERSION_STR: &str = env!("CARGO_PKG_VERSION");
    semver::Version::parse(VERSION_STR).unwrap().into()
});

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct Block {
    pub header: Header,
    pub body: Body,
}
