use std::{borrow::Borrow, cmp::Ordering, collections::HashMap};

use borsh::BorshSerialize;
use rustreexo::accumulator::mem_forest::MemForest;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::{
    authorization::Authorization,
    block::coinbase::Coinbase,
    error,
    hashes::{self, Hash, MerkleRoot, UtreexoNodeHash},
    transaction::{
        AuthorizedTransaction, FilledTransaction, GetValue, OutPoint, Output,
        PointedOutput, Transaction,
    },
    util,
};

/// Hash to get a [`CmbtNode`] inner commitment for a leaf value
#[derive(BorshSerialize, Debug)]
struct CbmtLeafPreCommitment {
    #[borsh(serialize_with = "util::borsh::serialize::bitcoin_amount")]
    fee: bitcoin::Amount,
    /// Sum of canonical tx sizes for child txs
    canonical_size: u64,
    tx_merkle_root: MerkleRoot,
}

/// Hash to get a [`CmbtNode`] inner commitment for a non-leaf value
#[derive(BorshSerialize, Debug)]
struct CbmtNodePreCommitment {
    /// left child inner commitment
    left_commitment: Hash,
    /// Sum of child tx fees
    #[borsh(serialize_with = "util::borsh::serialize::bitcoin_amount")]
    fees: bitcoin::Amount,
    /// Sum of canonical sizes of child txs
    canonical_size: u64,
    /// right child inner commitment
    right_commitment: Hash,
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
            right_commitment: rnode.commitment,
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

#[derive(BorshSerialize, Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct Body {
    pub coinbase: Coinbase,
    pub transactions: Vec<Transaction>,
    pub authorizations: Vec<Authorization>,
}

impl Body {
    pub fn new(
        authorized_transactions: Vec<AuthorizedTransaction>,
        coinbase: Coinbase,
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

    pub fn compute_merkle_root<FilledTx>(
        coinbase: &Coinbase,
        txs: &[FilledTx],
    ) -> Result<MerkleRoot, error::ComputeMerkleRoot>
    where
        FilledTx: Borrow<FilledTransaction>,
    {
        let CbmtNode {
            commitment: txs_commitment,
            ..
        } = {
            let n_txs = txs.len();
            let leaves: Vec<_> = txs
                .iter()
                .enumerate()
                .map(|(idx, tx)| {
                    let tx = tx.borrow();
                    let fees = tx.get_fee().map_err(|err| {
                        error::compute_merkle_root::Inner::TxFee {
                            txid: tx.transaction.txid(),
                            source: err,
                        }
                    })?;
                    let canonical_size =
                        tx.transaction.canonical_size().map_err(|err| {
                            error::compute_merkle_root::Inner::TxCanonicalSize {
                                txid: tx.transaction.txid(),
                                source: err,
                            }
                        })?;
                    let tx_merkle_root = tx
                        .transaction
                        .compute_merkle_root()
                        .map_err(|err| {
                        error::compute_merkle_root::Inner::TxMerkleRoot {
                            txid: tx.transaction.txid(),
                            source: err,
                        }
                    })?;
                    let leaf_pre_commitment = CbmtLeafPreCommitment {
                        fee: fees,
                        canonical_size,
                        tx_merkle_root,
                    };
                    Ok::<_, error::ComputeMerkleRoot>(CbmtNode {
                        commitment: hashes::hash(&leaf_pre_commitment),
                        fees,
                        canonical_size,
                        // see https://github.com/nervosnetwork/merkle-tree/blob/5d1898263e7167560fdaa62f09e8d52991a1c712/README.md#tree-struct
                        index: (idx + n_txs) - 1,
                    })
                })
                .collect::<Result<_, _>>()?;
            CbmtWithFeeTotal::build_merkle_root(leaves.as_slice())
        };
        let coinbase_commitment = coinbase
            .compute_merkle_root()
            .map_err(error::compute_merkle_root::Inner::CoinbaseMerkleRoot)?;
        let root = hashes::hash_with_scratch_buffer(&(
            coinbase_commitment,
            txs_commitment,
        ))
        .into();
        Ok(root)
    }

    // Modifies the memforest, without checking tx proofs
    pub fn modify_memforest<FilledTx>(
        coinbase: &Coinbase,
        txs: &[FilledTx],
        memforest: &mut MemForest<UtreexoNodeHash>,
    ) -> Result<MerkleRoot, error::ModifyMemForest>
    where
        FilledTx: Borrow<FilledTransaction>,
    {
        // New leaves for the accumulator
        let mut accumulator_add = Vec::<UtreexoNodeHash>::new();
        // Accumulator leaves to delete
        let mut accumulator_del = Vec::<UtreexoNodeHash>::new();
        let merkle_root = Self::compute_merkle_root(coinbase, txs)?;
        for (vout, output) in coinbase.outputs.iter().enumerate() {
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
            let tx = tx.borrow();
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
            .map_err(error::Utreexo)?;
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
        coinbase: &Coinbase,
        txs: &[FilledTransaction],
    ) -> Result<HashMap<OutPoint, Output>, error::ComputeMerkleRoot> {
        let mut res = HashMap::new();
        let merkle_root = Self::compute_merkle_root(coinbase, txs)?;
        for (vout, output) in coinbase.outputs.iter().enumerate() {
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
    ) -> Result<bitcoin::Amount, error::AmountOverflow> {
        use bitcoin::amount::CheckedSum as _;
        self.coinbase
            .outputs
            .iter()
            .map(|output| output.get_value())
            .checked_sum()
            .ok_or(error::AmountOverflow)
    }

    /// Calculate total number of inputs across all transactions in a block body
    pub fn inputs_len(&self) -> usize {
        self.transactions.iter().map(|t| t.inputs.len()).sum()
    }
}
