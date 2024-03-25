use std::collections::HashMap;

use bip300301::bitcoin;
use borsh::BorshSerialize;
use rustreexo::accumulator::{
    node_hash::NodeHash, pollard::Pollard, proof::Proof,
};
use serde::{Deserialize, Serialize};

use super::{hash, Address, Hash, MerkleRoot, Txid};
use crate::authorization::Authorization;

fn borsh_serialize_bitcoin_outpoint<W>(
    block_hash: &bitcoin::OutPoint,
    writer: &mut W,
) -> borsh::io::Result<()>
where
    W: borsh::io::Write,
{
    let bitcoin::OutPoint { txid, vout } = block_hash;
    let txid_bytes: &[u8; 32] = txid.as_ref();
    borsh::BorshSerialize::serialize(&(txid_bytes, vout), writer)
}

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
pub enum OutPoint {
    // Created by transactions.
    Regular {
        txid: Txid,
        vout: u32,
    },
    // Created by block bodies.
    Coinbase {
        merkle_root: MerkleRoot,
        vout: u32,
    },
    // Created by mainchain deposits.
    Deposit(
        #[borsh(serialize_with = "borsh_serialize_bitcoin_outpoint")]
        bitcoin::OutPoint,
    ),
}

impl std::fmt::Display for OutPoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Regular { txid, vout } => write!(f, "regular {txid} {vout}"),
            Self::Coinbase { merkle_root, vout } => {
                write!(f, "coinbase {merkle_root} {vout}")
            }
            Self::Deposit(bitcoin::OutPoint { txid, vout }) => {
                write!(f, "deposit {txid} {vout}")
            }
        }
    }
}

/// Reference to a tx input.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub enum InPoint {
    /// Transaction input
    Regular {
        txid: Txid,
        // index of the spend in the inputs to spend_tx
        vin: u32,
    },
    // Created by mainchain withdrawals
    Withdrawal {
        txid: bitcoin::Txid,
    },
}

fn borsh_serialize_bitcoin_address<V, W>(
    bitcoin_address: &bitcoin::Address<V>,
    writer: &mut W,
) -> borsh::io::Result<()>
where
    V: bitcoin::address::NetworkValidation,
    W: borsh::io::Write,
{
    let spk = bitcoin_address
        .as_unchecked()
        .assume_checked_ref()
        .script_pubkey();
    borsh::BorshSerialize::serialize(spk.as_bytes(), writer)
}

#[derive(
    BorshSerialize, Clone, Debug, Deserialize, Eq, PartialEq, Serialize,
)]
pub enum Content {
    Value(u64),
    Withdrawal {
        value: u64,
        main_fee: u64,
        #[borsh(serialize_with = "borsh_serialize_bitcoin_address")]
        main_address: bitcoin::Address<bitcoin::address::NetworkUnchecked>,
    },
}

impl Content {
    pub fn is_value(&self) -> bool {
        matches!(self, Self::Value(_))
    }
    pub fn is_withdrawal(&self) -> bool {
        matches!(self, Self::Withdrawal { .. })
    }
}

impl GetValue for Content {
    #[inline(always)]
    fn get_value(&self) -> u64 {
        match self {
            Self::Value(value) => *value,
            Self::Withdrawal { value, .. } => *value,
        }
    }
}

#[derive(
    BorshSerialize, Clone, Debug, Deserialize, Eq, PartialEq, Serialize,
)]
pub struct Output {
    pub address: Address,
    pub content: Content,
}

impl GetValue for Output {
    #[inline(always)]
    fn get_value(&self) -> u64 {
        self.content.get_value()
    }
}

#[derive(
    BorshSerialize, Clone, Debug, Deserialize, Eq, PartialEq, Serialize,
)]
pub struct PointedOutput {
    pub outpoint: OutPoint,
    pub output: Output,
}

impl From<&PointedOutput> for NodeHash {
    fn from(pointed_output: &PointedOutput) -> Self {
        Self::new(hash(pointed_output))
    }
}

#[derive(BorshSerialize, Clone, Debug, Default, Deserialize, Serialize)]
pub struct Transaction {
    pub inputs: Vec<(OutPoint, Hash)>,
    /// Utreexo proof for inputs
    #[borsh(skip)]
    pub proof: Proof,
    pub outputs: Vec<Output>,
}

impl Transaction {
    pub fn txid(&self) -> Txid {
        hash(self).into()
    }
}

/// Representation of a spent output
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SpentOutput {
    pub output: Output,
    pub inpoint: InPoint,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilledTransaction {
    pub transaction: Transaction,
    pub spent_utxos: Vec<Output>,
}

impl FilledTransaction {
    pub fn get_value_in(&self) -> u64 {
        self.spent_utxos.iter().map(GetValue::get_value).sum()
    }

    pub fn get_value_out(&self) -> u64 {
        self.transaction
            .outputs
            .iter()
            .map(GetValue::get_value)
            .sum()
    }

    pub fn get_fee(&self) -> Option<u64> {
        let value_in = self.get_value_in();
        let value_out = self.get_value_out();
        if value_in < value_out {
            None
        } else {
            Some(value_in - value_out)
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthorizedTransaction {
    pub transaction: Transaction,
    /// Authorization is called witness in Bitcoin.
    pub authorizations: Vec<Authorization>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

    pub fn compute_merkle_root(&self) -> MerkleRoot {
        // FIXME: Compute actual merkle root instead of just a hash.
        hash(&(&self.coinbase, &self.transactions)).into()
    }

    // Modifies the pollard, without checking tx proofs
    pub fn modify_pollard(&self, pollard: &mut Pollard) -> Result<(), String> {
        // New leaves for the accumulator
        let mut accumulator_add = Vec::<NodeHash>::new();
        // Accumulator leaves to delete
        let mut accumulator_del = Vec::<NodeHash>::new();
        let merkle_root = self.compute_merkle_root();
        for (vout, output) in self.coinbase.iter().enumerate() {
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
        for transaction in &self.transactions {
            let txid = transaction.txid();
            for (_, utxo_hash) in transaction.inputs.iter() {
                accumulator_del.push(utxo_hash.into());
            }
            for (vout, output) in transaction.outputs.iter().enumerate() {
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
        pollard.modify(&accumulator_add, &accumulator_del)
    }

    pub fn get_inputs(&self) -> Vec<OutPoint> {
        self.transactions
            .iter()
            .flat_map(|tx| tx.inputs.iter().map(|(outpoint, _)| outpoint))
            .copied()
            .collect()
    }

    pub fn get_outputs(&self) -> HashMap<OutPoint, Output> {
        let mut outputs = HashMap::new();
        let merkle_root = self.compute_merkle_root();
        for (vout, output) in self.coinbase.iter().enumerate() {
            let vout = vout as u32;
            let outpoint = OutPoint::Coinbase { merkle_root, vout };
            outputs.insert(outpoint, output.clone());
        }
        for transaction in &self.transactions {
            let txid = transaction.txid();
            for (vout, output) in transaction.outputs.iter().enumerate() {
                let vout = vout as u32;
                let outpoint = OutPoint::Regular { txid, vout };
                outputs.insert(outpoint, output.clone());
            }
        }
        outputs
    }

    pub fn get_coinbase_value(&self) -> u64 {
        self.coinbase.iter().map(|output| output.get_value()).sum()
    }
}

pub trait GetAddress {
    fn get_address(&self) -> Address;
}

pub trait GetValue {
    fn get_value(&self) -> u64;
}

impl GetValue for () {
    fn get_value(&self) -> u64 {
        0
    }
}

pub trait Verify {
    type Error;
    fn verify_transaction(
        transaction: &AuthorizedTransaction,
    ) -> Result<(), Self::Error>;
    fn verify_body(body: &Body) -> Result<(), Self::Error>;
}
