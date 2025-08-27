use bitcoin::amount::CheckedSum;
use borsh::{BorshDeserialize, BorshSerialize};
use heed::{BoxedError, BytesDecode, BytesEncode};
#[cfg(feature = "utreexo")]
use rustreexo::accumulator::{node_hash::BitcoinNodeHash, proof::Proof};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use utoipa::ToSchema;

use super::{Address, AmountOverflowError, Hash, M6id, MerkleRoot, Txid, hash};
use crate::types::Authorization;

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

fn borsh_deserialize_bitcoin_outpoint<R>(
    reader: &mut R,
) -> borsh::io::Result<bitcoin::OutPoint>
where
    R: borsh::io::Read,
{
    let (txid_bytes, vout): ([u8; 32], u32) =
        borsh::BorshDeserialize::deserialize_reader(reader)?;
    use bitcoin::hashes::Hash as BitcoinHash;
    let txid = bitcoin::Txid::from_byte_array(txid_bytes);
    Ok(bitcoin::OutPoint { txid, vout })
}

#[derive(
    BorshDeserialize,
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
        #[borsh(
            serialize_with = "borsh_serialize_bitcoin_outpoint",
            deserialize_with = "borsh_deserialize_bitcoin_outpoint"
        )]
        bitcoin::OutPoint,
    ),
}

impl BorshSerialize for OutPoint {
    fn serialize<W: borsh::io::Write>(
        &self,
        writer: &mut W,
    ) -> borsh::io::Result<()> {
        match self {
            OutPoint::Regular { txid, vout } => {
                BorshSerialize::serialize(&0u8, writer)?;
                BorshSerialize::serialize(txid, writer)?;
                BorshSerialize::serialize(vout, writer)
            }
            OutPoint::Coinbase { merkle_root, vout } => {
                BorshSerialize::serialize(&1u8, writer)?;
                BorshSerialize::serialize(merkle_root, writer)?;
                BorshSerialize::serialize(vout, writer)
            }
            OutPoint::Deposit(bitcoin_outpoint) => {
                BorshSerialize::serialize(&2u8, writer)?;
                borsh_serialize_bitcoin_outpoint(bitcoin_outpoint, writer)
            }
        }
    }
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
    /// Get the raw key bytes
    #[inline]
    pub fn as_bytes(&self) -> &[u8; 37] {
        &self.0
    }

    /// Fast comparison for same tag types (cache-friendly)
    #[inline]
    pub fn fast_cmp_same_tag(&self, other: &Self) -> std::cmp::Ordering {
        // Skip tag comparison when we know they're the same type
        // Compare ID first (32 bytes), then vout (4 bytes)
        match self.0[1..33].cmp(&other.0[1..33]) {
            std::cmp::Ordering::Equal => self.0[33..37].cmp(&other.0[33..37]),
            other => other,
        }
    }

    /// Get the tag byte for type checking
    #[inline]
    pub fn tag(&self) -> u8 {
        self.0[0]
    }

    /// Get the vout as u32 without full deserialization
    #[inline]
    pub fn vout(&self) -> u32 {
        u32::from_be_bytes([self.0[33], self.0[34], self.0[35], self.0[36]])
    }

    /// Get the ID bytes (txid/merkle_root)
    #[inline]
    pub fn id_bytes(&self) -> &[u8; 32] {
        // Safety: we know bytes 1..33 form a 32-byte array
        unsafe { &*(self.0[1..33].as_ptr() as *const [u8; 32]) }
    }
}

impl From<OutPoint> for OutPointKey {
    #[inline]
    fn from(op: OutPoint) -> Self {
        // Use uninitialized array for better performance
        let mut k = std::mem::MaybeUninit::<[u8; 37]>::uninit();
        let k_ptr = k.as_mut_ptr() as *mut u8;

        unsafe {
            match op {
                OutPoint::Regular {
                    txid: Txid(id),
                    vout,
                } => {
                    *k_ptr = 0;
                    std::ptr::copy_nonoverlapping(
                        id.as_ptr(),
                        k_ptr.add(1),
                        32,
                    );
                    let vout_bytes = vout.to_be_bytes();
                    std::ptr::copy_nonoverlapping(
                        vout_bytes.as_ptr(),
                        k_ptr.add(33),
                        4,
                    );
                }
                OutPoint::Coinbase { merkle_root, vout } => {
                    *k_ptr = 1;
                    let id: Hash = merkle_root.into();
                    std::ptr::copy_nonoverlapping(
                        id.as_ptr(),
                        k_ptr.add(1),
                        32,
                    );
                    let vout_bytes = vout.to_be_bytes();
                    std::ptr::copy_nonoverlapping(
                        vout_bytes.as_ptr(),
                        k_ptr.add(33),
                        4,
                    );
                }
                OutPoint::Deposit(ref bop) => {
                    *k_ptr = 2;
                    let txid_bytes: &[u8; 32] = bop.txid.as_ref();
                    std::ptr::copy_nonoverlapping(
                        txid_bytes.as_ptr(),
                        k_ptr.add(1),
                        32,
                    );
                    let vout_bytes = bop.vout.to_be_bytes();
                    std::ptr::copy_nonoverlapping(
                        vout_bytes.as_ptr(),
                        k_ptr.add(33),
                        4,
                    );
                }
            }
            Self(k.assume_init())
        }
    }
}

impl From<&OutPoint> for OutPointKey {
    #[inline]
    fn from(op: &OutPoint) -> Self {
        Self::from(*op)
    }
}

impl From<OutPointKey> for OutPoint {
    #[inline]
    fn from(key: OutPointKey) -> Self {
        let tag = key.0[0];
        // Use direct pointer access for better performance
        let id_bytes = key.id_bytes();
        let vout = key.vout();

        match tag {
            0 => OutPoint::Regular {
                txid: Txid(*id_bytes),
                vout,
            },
            1 => OutPoint::Coinbase {
                merkle_root: MerkleRoot::from(Hash::from(*id_bytes)),
                vout,
            },
            2 => {
                use bitcoin::hashes::Hash as BitcoinHash;
                let txid = bitcoin::Txid::from_byte_array(*id_bytes);
                OutPoint::Deposit(bitcoin::OutPoint { txid, vout })
            }
            _ => unreachable!("Invalid OutPointKey tag"),
        }
    }
}

impl From<&OutPointKey> for OutPoint {
    #[inline]
    fn from(key: &OutPointKey) -> Self {
        Self::from(*key)
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

fn borsh_serialize_bitcoin_amount<W>(
    bitcoin_amount: &bitcoin::Amount,
    writer: &mut W,
) -> borsh::io::Result<()>
where
    W: borsh::io::Write,
{
    borsh::BorshSerialize::serialize(&bitcoin_amount.to_sat(), writer)
}

fn borsh_deserialize_bitcoin_amount<R>(
    reader: &mut R,
) -> borsh::io::Result<bitcoin::Amount>
where
    R: borsh::io::Read,
{
    let sats: u64 = borsh::BorshDeserialize::deserialize_reader(reader)?;
    Ok(bitcoin::Amount::from_sat(sats))
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

fn borsh_deserialize_bitcoin_address<R>(
    reader: &mut R,
) -> borsh::io::Result<bitcoin::Address<bitcoin::address::NetworkUnchecked>>
where
    R: borsh::io::Read,
{
    let address_string: String =
        borsh::BorshDeserialize::deserialize_reader(reader)?;
    address_string.parse().map_err(|e| {
        borsh::io::Error::new(borsh::io::ErrorKind::InvalidData, e)
    })
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

    #[derive(
        borsh::BorshDeserialize,
        borsh::BorshSerialize,
        Clone,
        Debug,
        Eq,
        PartialEq,
    )]
    pub enum Content {
        Value(
            #[borsh(
                serialize_with = "super::borsh_serialize_bitcoin_amount",
                deserialize_with = "super::borsh_deserialize_bitcoin_amount"
            )]
            bitcoin::Amount,
        ),
        Withdrawal {
            #[borsh(
                serialize_with = "super::borsh_serialize_bitcoin_amount",
                deserialize_with = "super::borsh_deserialize_bitcoin_amount"
            )]
            value: bitcoin::Amount,
            #[borsh(
                serialize_with = "super::borsh_serialize_bitcoin_amount",
                deserialize_with = "super::borsh_deserialize_bitcoin_amount"
            )]
            main_fee: bitcoin::Amount,
            #[borsh(
                serialize_with = "super::borsh_serialize_bitcoin_address",
                deserialize_with = "super::borsh_deserialize_bitcoin_address"
            )]
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
    BorshDeserialize,
    BorshSerialize,
    Clone,
    Debug,
    Default,
    Deserialize,
    Serialize,
    ToSchema,
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

#[derive(
    BorshDeserialize, BorshSerialize, Clone, Debug, Deserialize, Serialize,
)]
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
