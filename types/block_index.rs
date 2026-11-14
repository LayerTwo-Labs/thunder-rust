//! Coin movements and transaction facts that a block body does not carry

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::{
    hashes::{M6id, Txid},
    transaction::{OutPoint, Output, Transaction},
};

/// One output that the block created outside its body
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize, ToSchema)]
pub struct BlockIndexOutput {
    pub outpoint: OutPoint,
    pub output: Output,
}

/// One output that a withdrawal bundle spent, with the bundle that spent it
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize, ToSchema)]
pub struct BlockIndexSpend {
    pub outpoint: OutPoint,
    pub m6id: M6id,
}

/// Coin movements that a block applied outside its body
#[derive(
    Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize, ToSchema,
)]
pub struct BlockIndexEvents {
    /// Outputs that mainchain deposits created
    pub deposits: Vec<BlockIndexOutput>,
    /// Outputs that a new withdrawal bundle spent, or that an unknown
    /// confirmed bundle spent
    pub bundle_spends: Vec<BlockIndexSpend>,
    /// Outputs that a failed or dropped withdrawal bundle created again, at
    /// `OutPoint::UnsuccessfulWithdrawal` outpoints
    pub bundle_returns: Vec<BlockIndexOutput>,
}

impl BlockIndexEvents {
    /// True if the block moved no coins outside its body
    pub fn is_empty(&self) -> bool {
        self.deposits.is_empty()
            && self.bundle_spends.is_empty()
            && self.bundle_returns.is_empty()
    }

    /// Add the events of `other` after the events of `self`
    pub fn extend(&mut self, other: Self) {
        self.deposits.extend(other.deposits);
        self.bundle_spends.extend(other.bundle_spends);
        self.bundle_returns.extend(other.bundle_returns);
    }
}

/// One transaction of a block, with the fields that its body does not carry
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize, ToSchema)]
pub struct BlockIndexTx {
    pub txid: Txid,
    /// Canonical size in bytes
    pub size: u64,
    /// Canonical borsh encoding, as hex
    pub raw: String,
}

/// Everything about a block that its body does not carry
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize, ToSchema)]
pub struct BlockIndex {
    /// Transactions in body order
    pub txs: Vec<BlockIndexTx>,
    /// Outputs that mainchain deposits created
    pub deposits: Vec<BlockIndexOutput>,
    /// Outputs that a withdrawal bundle spent
    pub bundle_spends: Vec<BlockIndexSpend>,
    /// Outputs that a failed or dropped withdrawal bundle created again
    pub bundle_returns: Vec<BlockIndexOutput>,
}

impl BlockIndex {
    pub fn new(txs: Vec<BlockIndexTx>, events: BlockIndexEvents) -> Self {
        let BlockIndexEvents {
            deposits,
            bundle_spends,
            bundle_returns,
        } = events;
        Self {
            txs,
            deposits,
            bundle_spends,
            bundle_returns,
        }
    }
}

/// One transaction that the mempool holds
#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct MempoolTx {
    pub txid: Txid,
    /// Canonical size in bytes
    pub size: u64,
    /// Canonical borsh encoding, as hex
    pub raw: String,
    pub tx: Transaction,
}

impl BlockIndexTx {
    pub fn new(tx: &Transaction) -> borsh::io::Result<Self> {
        Ok(Self {
            txid: tx.txid(),
            size: tx.canonical_size()?,
            raw: const_hex::encode(tx.canonical_bytes()?),
        })
    }
}

impl MempoolTx {
    pub fn new(tx: Transaction) -> borsh::io::Result<Self> {
        let BlockIndexTx { txid, size, raw } = BlockIndexTx::new(&tx)?;
        Ok(Self {
            txid,
            size,
            raw,
            tx,
        })
    }
}
