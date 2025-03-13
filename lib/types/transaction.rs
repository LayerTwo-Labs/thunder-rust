use std::collections::HashMap;

use bitcoin::amount::CheckedSum;
use borsh::BorshSerialize;
use rustreexo::accumulator::{
    mem_forest::MemForest, node_hash::BitcoinNodeHash, proof::Proof,
};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use super::{Address, AmountOverflowError, Hash, M6id, MerkleRoot, Txid, hash};
use crate::authorization::Authorization;

pub trait GetAddress {
    fn get_address(&self) -> Address;
}

pub trait GetValue {
    fn get_value(&self) -> bitcoin::Amount;
}

impl GetValue for () {
    fn get_value(&self) -> bitcoin::Amount {
        bitcoin::Amount::ZERO
    }
}

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
    Ord,
    PartialEq,
    PartialOrd,
    Serialize,
    ToSchema,
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
    #[schema(value_type = crate::types::schema::BitcoinOutPoint)]
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
        m6id: M6id,
    },
}

fn borsh_serialize_bitcoin_amount<W>(
    bitcoin_amount: &bitcoin::Amount,
    writer: &mut W,
) -> borsh::io::Result<()>
where
    W: borsh::io::Write,
{
    borsh::BorshSerialize::serialize(&bitcoin_amount.to_sat(), writer)
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

mod content {
    use serde::{Deserialize, Serialize};
    use utoipa::{PartialSchema, ToSchema};

    /// Default representation for Serde
    #[derive(Deserialize, Serialize)]
    enum DefaultRepr {
        Value(bitcoin::Amount),
        Withdrawal {
            value: bitcoin::Amount,
            main_fee: bitcoin::Amount,
            main_address: bitcoin::Address<bitcoin::address::NetworkUnchecked>,
        },
    }

    /// Human-readable representation for Serde
    #[derive(Deserialize, Serialize, ToSchema)]
    #[schema(as = OutputContent, description = "")]
    enum HumanReadableRepr {
        #[schema(value_type = u64)]
        Value(
            #[serde(with = "bitcoin::amount::serde::as_sat")] bitcoin::Amount,
        ),
        Withdrawal {
            #[serde(with = "bitcoin::amount::serde::as_sat")]
            #[serde(rename = "value_sats")]
            #[schema(value_type = u64)]
            value: bitcoin::Amount,
            #[serde(with = "bitcoin::amount::serde::as_sat")]
            #[serde(rename = "main_fee_sats")]
            #[schema(value_type = u64)]
            main_fee: bitcoin::Amount,
            #[schema(value_type = crate::types::schema::BitcoinAddr)]
            main_address: bitcoin::Address<bitcoin::address::NetworkUnchecked>,
        },
    }

    type SerdeRepr = serde_with::IfIsHumanReadable<
        serde_with::FromInto<DefaultRepr>,
        serde_with::FromInto<HumanReadableRepr>,
    >;

    #[derive(borsh::BorshSerialize, Clone, Debug, Eq, PartialEq)]
    pub enum Content {
        Value(
            #[borsh(serialize_with = "super::borsh_serialize_bitcoin_amount")]
            bitcoin::Amount,
        ),
        Withdrawal {
            #[borsh(serialize_with = "super::borsh_serialize_bitcoin_amount")]
            value: bitcoin::Amount,
            #[borsh(serialize_with = "super::borsh_serialize_bitcoin_amount")]
            main_fee: bitcoin::Amount,
            #[borsh(serialize_with = "super::borsh_serialize_bitcoin_address")]
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

        pub(in crate::types) fn schema_ref() -> utoipa::openapi::Ref {
            utoipa::openapi::Ref::new("OutputContent")
        }
    }

    impl crate::wallet::GetValue for Content {
        #[inline(always)]
        fn get_value(&self) -> bitcoin::Amount {
            match self {
                Self::Value(value) => *value,
                Self::Withdrawal { value, .. } => *value,
            }
        }
    }

    impl From<Content> for DefaultRepr {
        fn from(content: Content) -> Self {
            match content {
                Content::Value(value) => Self::Value(value),
                Content::Withdrawal {
                    value,
                    main_fee,
                    main_address,
                } => Self::Withdrawal {
                    value,
                    main_fee,
                    main_address,
                },
            }
        }
    }

    impl From<Content> for HumanReadableRepr {
        fn from(content: Content) -> Self {
            match content {
                Content::Value(value) => Self::Value(value),
                Content::Withdrawal {
                    value,
                    main_fee,
                    main_address,
                } => Self::Withdrawal {
                    value,
                    main_fee,
                    main_address,
                },
            }
        }
    }

    impl From<DefaultRepr> for Content {
        fn from(repr: DefaultRepr) -> Self {
            match repr {
                DefaultRepr::Value(value) => Self::Value(value),
                DefaultRepr::Withdrawal {
                    value,
                    main_fee,
                    main_address,
                } => Self::Withdrawal {
                    value,
                    main_fee,
                    main_address,
                },
            }
        }
    }

    impl From<HumanReadableRepr> for Content {
        fn from(repr: HumanReadableRepr) -> Self {
            match repr {
                HumanReadableRepr::Value(value) => Self::Value(value),
                HumanReadableRepr::Withdrawal {
                    value,
                    main_fee,
                    main_address,
                } => Self::Withdrawal {
                    value,
                    main_fee,
                    main_address,
                },
            }
        }
    }

    impl<'de> Deserialize<'de> for Content {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            <SerdeRepr as serde_with::DeserializeAs<'de, _>>::deserialize_as(
                deserializer,
            )
        }
    }

    impl Serialize for Content {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
        {
            <SerdeRepr as serde_with::SerializeAs<_>>::serialize_as(
                self, serializer,
            )
        }
    }

    impl PartialSchema for Content {
        fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
            <HumanReadableRepr as PartialSchema>::schema()
        }
    }

    impl ToSchema for Content {
        fn name() -> std::borrow::Cow<'static, str> {
            <HumanReadableRepr as ToSchema>::name()
        }
    }
}
pub use content::Content;

#[derive(
    BorshSerialize,
    Clone,
    Debug,
    Deserialize,
    Eq,
    PartialEq,
    Serialize,
    ToSchema,
)]
pub struct Output {
    pub address: Address,
    #[schema(schema_with = Content::schema_ref)]
    pub content: Content,
}

impl GetValue for Output {
    #[inline(always)]
    fn get_value(&self) -> bitcoin::Amount {
        self.content.get_value()
    }
}

#[derive(
    BorshSerialize,
    Clone,
    Debug,
    Deserialize,
    Eq,
    PartialEq,
    Serialize,
    ToSchema,
)]
pub struct PointedOutput {
    pub outpoint: OutPoint,
    pub output: Output,
}

impl From<&PointedOutput> for BitcoinNodeHash {
    fn from(pointed_output: &PointedOutput) -> Self {
        Self::new(hash(pointed_output))
    }
}

#[derive(
    BorshSerialize, Clone, Debug, Default, Deserialize, Serialize, ToSchema,
)]
pub struct Transaction {
    #[schema(value_type = Vec<(OutPoint, String)>)]
    pub inputs: Vec<(OutPoint, Hash)>,
    /// Utreexo proof for inputs
    #[borsh(skip)]
    #[schema(value_type = crate::types::schema::UtreexoProof)]
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
    pub fn get_value_in(&self) -> Result<bitcoin::Amount, AmountOverflowError> {
        self.spent_utxos
            .iter()
            .map(GetValue::get_value)
            .checked_sum()
            .ok_or(AmountOverflowError)
    }

    pub fn get_value_out(
        &self,
    ) -> Result<bitcoin::Amount, AmountOverflowError> {
        self.transaction
            .outputs
            .iter()
            .map(GetValue::get_value)
            .checked_sum()
            .ok_or(AmountOverflowError)
    }

    pub fn get_fee(
        &self,
    ) -> Result<Option<bitcoin::Amount>, AmountOverflowError> {
        let value_in = self.get_value_in()?;
        let value_out = self.get_value_out()?;
        if value_in < value_out {
            Ok(None)
        } else {
            Ok(Some(value_in - value_out))
        }
    }
}

#[derive(BorshSerialize, Clone, Debug, Deserialize, Serialize)]
pub struct AuthorizedTransaction {
    pub transaction: Transaction,
    /// Authorization is called witness in Bitcoin.
    pub authorizations: Vec<Authorization>,
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

    pub fn compute_merkle_root(&self) -> MerkleRoot {
        // FIXME: Compute actual merkle root instead of just a hash.
        hash(&(&self.coinbase, &self.transactions)).into()
    }

    // Modifies the memforest, without checking tx proofs
    pub fn modify_memforest(
        &self,
        memforest: &mut MemForest<BitcoinNodeHash>,
    ) -> Result<(), String> {
        // New leaves for the accumulator
        let mut accumulator_add = Vec::<BitcoinNodeHash>::new();
        // Accumulator leaves to delete
        let mut accumulator_del = Vec::<BitcoinNodeHash>::new();
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
        memforest.modify(&accumulator_add, &accumulator_del)
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

    pub fn get_coinbase_value(
        &self,
    ) -> Result<bitcoin::Amount, AmountOverflowError> {
        self.coinbase
            .iter()
            .map(|output| output.get_value())
            .checked_sum()
            .ok_or(AmountOverflowError)
    }
}

pub trait Verify {
    type Error;
    fn verify_transaction(
        transaction: &AuthorizedTransaction,
    ) -> Result<(), Self::Error>;
    fn verify_body(body: &Body) -> Result<(), Self::Error>;
}
