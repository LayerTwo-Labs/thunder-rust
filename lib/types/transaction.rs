use bitcoin::amount::CheckedSum;
use borsh::{BorshDeserialize, BorshSerialize};
use heed::{BoxedError, BytesDecode, BytesEncode};
#[cfg(feature = "utreexo")]
use rustreexo::accumulator::{node_hash::BitcoinNodeHash, proof::Proof};
use serde::{Deserialize, Serialize};
use thiserror::Error;
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

/// Fixed-width lexicographically sortable key for OutPoint
/// Layout: [tag: u8][id: 32][vout: u32 BE]
/// - tag: 0 = Regular, 1 = Coinbase, 2 = Deposit
/// - id: txid/merkle_root/bitcoin::txid
/// - vout: big-endian for numeric order = lexicographic order
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct OutPointKey([u8; 37]);

impl OutPointKey {
    /// Encode an OutPoint into a fixed-width lexicographically sortable key
    #[inline]
    pub fn from_outpoint(op: &OutPoint) -> Self {
        let mut k = [0u8; 37];
        match *op {
            OutPoint::Regular {
                txid: Txid(id),
                vout,
            } => {
                k[0] = 0;
                k[1..33].copy_from_slice(&id);
                k[33..37].copy_from_slice(&vout.to_be_bytes());
            }
            OutPoint::Coinbase { merkle_root, vout } => {
                k[0] = 1;
                let id: Hash = merkle_root.into();
                k[1..33].copy_from_slice(&id);
                k[33..37].copy_from_slice(&vout.to_be_bytes());
            }
            OutPoint::Deposit(ref bop) => {
                k[0] = 2;
                k[1..33].copy_from_slice(bop.txid.as_ref());
                k[33..37].copy_from_slice(&bop.vout.to_be_bytes());
            }
        }
        Self(k)
    }

    /// Get the raw key bytes
    #[inline]
    pub fn as_bytes(&self) -> &[u8; 37] {
        &self.0
    }

    /// Decode OutPointKey back to OutPoint
    #[inline]
    pub fn to_outpoint(&self) -> OutPoint {
        let tag = self.0[0];
        let mut id = [0u8; 32];
        id.copy_from_slice(&self.0[1..33]);
        let vout = u32::from_be_bytes([
            self.0[33], self.0[34], self.0[35], self.0[36],
        ]);

        match tag {
            0 => OutPoint::Regular {
                txid: Txid(id),
                vout,
            },
            1 => OutPoint::Coinbase {
                merkle_root: MerkleRoot::from(Hash::from(id)),
                vout,
            },
            2 => {
                use bitcoin::hashes::Hash as BitcoinHash;
                let txid = bitcoin::Txid::from_byte_array(id);
                OutPoint::Deposit(bitcoin::OutPoint { txid, vout })
            }
            _ => unreachable!("Invalid OutPointKey tag"),
        }
    }
}

impl From<OutPoint> for OutPointKey {
    #[inline]
    fn from(op: OutPoint) -> Self {
        Self::from_outpoint(&op)
    }
}

impl From<&OutPoint> for OutPointKey {
    #[inline]
    fn from(op: &OutPoint) -> Self {
        Self::from_outpoint(op)
    }
}

impl From<OutPointKey> for OutPoint {
    #[inline]
    fn from(key: OutPointKey) -> Self {
        key.to_outpoint()
    }
}

impl From<&OutPointKey> for OutPoint {
    #[inline]
    fn from(key: &OutPointKey) -> Self {
        key.to_outpoint()
    }
}

impl Ord for OutPointKey {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

impl PartialOrd for OutPointKey {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl AsRef<[u8]> for OutPointKey {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

// Database key encoding traits for direct LMDB usage
impl<'a> BytesEncode<'a> for OutPointKey {
    type EItem = OutPointKey;

    #[inline]
    fn bytes_encode(
        item: &'a Self::EItem,
    ) -> Result<std::borrow::Cow<'a, [u8]>, BoxedError> {
        Ok(std::borrow::Cow::Borrowed(item.as_ref()))
    }
}

impl<'a> BytesDecode<'a> for OutPointKey {
    type DItem = OutPointKey;

    #[inline]
    fn bytes_decode(bytes: &'a [u8]) -> Result<Self::DItem, BoxedError> {
        if bytes.len() != 37 {
            return Err("OutPointKey must be exactly 37 bytes".into());
        }
        let mut key = [0u8; 37];
        key.copy_from_slice(bytes);
        Ok(OutPointKey(key))
    }
}

/// Reference to a tx input.
#[derive(
    BorshDeserialize,
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

    #[derive(Clone, Debug, Eq, PartialEq)]
    pub enum Content {
        Value(bitcoin::Amount),
        Withdrawal {
            value: bitcoin::Amount,
            main_fee: bitcoin::Amount,
            main_address: bitcoin::Address<bitcoin::address::NetworkUnchecked>,
        },
    }

    impl borsh::BorshSerialize for Content {
        fn serialize<W: borsh::io::Write>(
            &self,
            writer: &mut W,
        ) -> borsh::io::Result<()> {
            match self {
                Content::Value(amount) => {
                    borsh::BorshSerialize::serialize(&0u8, writer)?;
                    borsh::BorshSerialize::serialize(&amount.to_sat(), writer)
                }
                Content::Withdrawal {
                    value,
                    main_fee,
                    main_address,
                } => {
                    borsh::BorshSerialize::serialize(&1u8, writer)?;
                    borsh::BorshSerialize::serialize(&value.to_sat(), writer)?;
                    borsh::BorshSerialize::serialize(
                        &main_fee.to_sat(),
                        writer,
                    )?;
                    // Serialize address as script bytes
                    let script = main_address
                        .as_unchecked()
                        .assume_checked_ref()
                        .script_pubkey();
                    borsh::BorshSerialize::serialize(&script.as_bytes(), writer)
                }
            }
        }
    }

    impl borsh::BorshDeserialize for Content {
        fn deserialize(buf: &mut &[u8]) -> borsh::io::Result<Self> {
            let variant: u8 = borsh::BorshDeserialize::deserialize(buf)?;
            match variant {
                0 => {
                    let sats: u64 = borsh::BorshDeserialize::deserialize(buf)?;
                    Ok(Content::Value(bitcoin::Amount::from_sat(sats)))
                }
                1 => {
                    let value_sats: u64 =
                        borsh::BorshDeserialize::deserialize(buf)?;
                    let main_fee_sats: u64 =
                        borsh::BorshDeserialize::deserialize(buf)?;
                    let script_bytes: Vec<u8> =
                        borsh::BorshDeserialize::deserialize(buf)?;

                    let script = bitcoin::ScriptBuf::from_bytes(script_bytes);
                    let checked_address = bitcoin::Address::from_script(
                        &script,
                        bitcoin::Network::Bitcoin,
                    )
                    .map_err(|e| {
                        borsh::io::Error::new(
                            borsh::io::ErrorKind::InvalidData,
                            e,
                        )
                    })?;
                    let address = checked_address.as_unchecked().clone();

                    Ok(Content::Withdrawal {
                        value: bitcoin::Amount::from_sat(value_sats),
                        main_fee: bitcoin::Amount::from_sat(main_fee_sats),
                        main_address: address,
                    })
                }
                _ => Err(borsh::io::Error::new(
                    borsh::io::ErrorKind::InvalidData,
                    format!("Invalid Content variant: {}", variant),
                )),
            }
        }

        fn deserialize_reader<R: borsh::io::Read>(
            reader: &mut R,
        ) -> borsh::io::Result<Self> {
            let mut variant_buf = [0u8; 1];
            reader.read_exact(&mut variant_buf)?;
            let variant = variant_buf[0];

            match variant {
                0 => {
                    let mut sats_buf = [0u8; 8];
                    reader.read_exact(&mut sats_buf)?;
                    let sats = u64::from_le_bytes(sats_buf);
                    Ok(Content::Value(bitcoin::Amount::from_sat(sats)))
                }
                1 => {
                    let mut value_buf = [0u8; 8];
                    reader.read_exact(&mut value_buf)?;
                    let value_sats = u64::from_le_bytes(value_buf);

                    let mut fee_buf = [0u8; 8];
                    reader.read_exact(&mut fee_buf)?;
                    let main_fee_sats = u64::from_le_bytes(fee_buf);

                    // Read script length first
                    let mut len_buf = [0u8; 4];
                    reader.read_exact(&mut len_buf)?;
                    let script_len = u32::from_le_bytes(len_buf) as usize;

                    let mut script_bytes = vec![0u8; script_len];
                    reader.read_exact(&mut script_bytes)?;

                    let script = bitcoin::ScriptBuf::from_bytes(script_bytes);
                    let checked_address = bitcoin::Address::from_script(
                        &script,
                        bitcoin::Network::Bitcoin,
                    )
                    .map_err(|e| {
                        borsh::io::Error::new(
                            borsh::io::ErrorKind::InvalidData,
                            e,
                        )
                    })?;
                    let address = checked_address.as_unchecked().clone();

                    Ok(Content::Withdrawal {
                        value: bitcoin::Amount::from_sat(value_sats),
                        main_fee: bitcoin::Amount::from_sat(main_fee_sats),
                        main_address: address,
                    })
                }
                _ => Err(borsh::io::Error::new(
                    borsh::io::ErrorKind::InvalidData,
                    format!("Invalid Content variant: {}", variant),
                )),
            }
        }
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

    impl crate::types::GetValue for Content {
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
    BorshDeserialize,
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

#[cfg(feature = "utreexo")]
impl From<&PointedOutput> for BitcoinNodeHash {
    fn from(pointed_output: &PointedOutput) -> Self {
        Self::new(hash(pointed_output))
    }
}

/// Useful when computing hashes for Utreexo,
/// without needing to clone an output
#[derive(BorshSerialize, Clone, Copy, Debug)]
pub struct PointedOutputRef<'a> {
    pub outpoint: OutPoint,
    pub output: &'a Output,
}

#[cfg(feature = "utreexo")]
impl From<PointedOutputRef<'_>> for BitcoinNodeHash {
    fn from(pointed_output: PointedOutputRef) -> Self {
        Self::new(hash(&pointed_output))
    }
}

#[derive(
    BorshSerialize, Clone, Debug, Default, Deserialize, Serialize, ToSchema,
)]
pub struct Transaction {
    #[schema(value_type = Vec<(OutPoint, String)>)]
    pub inputs: Vec<(OutPoint, Hash)>,
    #[cfg(feature = "utreexo")]
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

    /// Canonical size in bytes. The canonical encoding is used for hashing,
    /// But other encodings may be used at eg. networking, rpc levels.
    pub fn canonical_size(&self) -> u64 {
        (borsh::object_length(self).unwrap() / 8) as u64
    }
}

/// Representation of a spent output
#[derive(
    BorshDeserialize,
    BorshSerialize,
    Clone,
    Debug,
    Deserialize,
    Eq,
    PartialEq,
    Serialize,
)]
pub struct SpentOutput {
    pub output: Output,
    pub inpoint: InPoint,
}

#[derive(Debug, Error)]
pub enum ComputeFeeError {
    #[error("underfunded (value in < value out)")]
    Underfunded,
    #[error("value in overflow")]
    ValueInOverflow(#[source] AmountOverflowError),
    #[error("value out overflow")]
    ValueOutOverflow(#[source] AmountOverflowError),
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

    pub fn get_fee(&self) -> Result<bitcoin::Amount, ComputeFeeError> {
        let value_in = self
            .get_value_in()
            .map_err(ComputeFeeError::ValueInOverflow)?;
        let value_out = self
            .get_value_out()
            .map_err(ComputeFeeError::ValueOutOverflow)?;
        if value_in < value_out {
            Err(ComputeFeeError::Underfunded)
        } else {
            Ok(value_in - value_out)
        }
    }
}

#[derive(BorshSerialize, Clone, Debug, Deserialize, Serialize)]
pub struct Authorized<T> {
    pub transaction: T,
    /// Authorizations are called witnesses in Bitcoin.
    pub authorizations: Vec<Authorization>,
}

pub type AuthorizedTransaction = Authorized<Transaction>;

impl From<Authorized<FilledTransaction>> for AuthorizedTransaction {
    fn from(tx: Authorized<FilledTransaction>) -> Self {
        Self {
            transaction: tx.transaction.transaction,
            authorizations: tx.authorizations,
        }
    }
}
