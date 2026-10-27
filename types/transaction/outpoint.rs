use std::io::Cursor;

use borsh::{BorshDeserialize, BorshSerialize};
#[cfg(feature = "heed")]
use heed::{BoxedError, BytesDecode, BytesEncode};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::{
    hashes::{CoinbaseTxid, M6id, Txid},
    schema, util,
};

/// Coinbase outputs become spendable if the sidechain tip height is at least
/// `COINBASE_MATURITY` greater than the height of the sidechain block in which
/// the coinbase tx was included.
pub const COINBASE_MATURITY: u32 = 50;

/// Deposits become spendable if the previous mainchain tip height is at least
/// `DEPOSIT_MATURITY` greater than the height of the mainchain block in which
/// the deposit tx was included.
pub const DEPOSIT_MATURITY: u32 = 50;

#[derive(
    BorshSerialize,
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
        txid: CoinbaseTxid,
        vout: u32,
    },
    // Created by mainchain deposits.
    #[schema(value_type = schema::BitcoinOutPoint)]
    Deposit(
        #[borsh(
            deserialize_with = "util::borsh::deserialize::bitcoin_outpoint",
            serialize_with = "util::borsh::serialize::bitcoin_outpoint"
        )]
        bitcoin::OutPoint,
    ),
    /// Created by recreating the outputs of an unsuccessful withdrawal
    /// bundle (one that failed, or a stale pending bundle that was
    /// dropped without ever being submitted).
    UnsuccessfulWithdrawal {
        /// M6ID of the unsuccessful withdrawal bundle
        m6id: M6id,
        /// Sidechain block height at which the outputs were recreated
        height: u32,
        /// Index of the spent output within the bundle's spend_utxos
        vout: u32,
    },
}

impl std::fmt::Display for OutPoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Regular { txid, vout } => write!(f, "regular {txid} {vout}"),
            Self::Coinbase { txid, vout } => {
                write!(f, "coinbase {txid} {vout}")
            }
            Self::Deposit(bitcoin::OutPoint { txid, vout }) => {
                write!(f, "deposit {txid} {vout}")
            }
            Self::UnsuccessfulWithdrawal { m6id, height, vout } => {
                write!(f, "unsuccessful withdrawal {m6id} {height} {vout}")
            }
        }
    }
}

/// Maximum borsh-serialized size of an `OutPoint`:
/// 1 (variant) + 32 (txid/m6id) + 4 (height) + 4 (vout).
/// Shorter variants are zero-padded in the fixed-width key.
const OUTPOINT_KEY_SIZE: usize = 41;

/// Fixed-width key for OutPoint based on its canonical Borsh encoding.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct OutPointKey([u8; OUTPOINT_KEY_SIZE]);

impl OutPointKey {
    /// Get the raw key bytes
    #[inline]
    pub fn as_bytes(&self) -> &[u8; OUTPOINT_KEY_SIZE] {
        &self.0
    }
}

impl From<OutPoint> for OutPointKey {
    #[inline]
    fn from(op: OutPoint) -> Self {
        let mut key = [0u8; OUTPOINT_KEY_SIZE];
        let mut cursor = Cursor::new(&mut key[..]);
        BorshSerialize::serialize(&op, &mut cursor)
            .expect("serializing OutPoint into key buffer should never fail");
        debug_assert!(cursor.position() as usize <= OUTPOINT_KEY_SIZE);
        Self(key)
    }
}

impl From<&OutPoint> for OutPointKey {
    #[inline]
    fn from(op: &OutPoint) -> Self {
        <Self as From<OutPoint>>::from(*op)
    }
}

impl From<OutPointKey> for OutPoint {
    #[inline]
    fn from(key: OutPointKey) -> Self {
        let mut cursor = Cursor::new(&key.0[..]);
        OutPoint::deserialize_reader(&mut cursor)
            .expect("deserializing OutPointKey should never fail")
    }
}

impl From<&OutPointKey> for OutPoint {
    #[inline]
    fn from(key: &OutPointKey) -> Self {
        <Self as From<OutPointKey>>::from(*key)
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

#[cfg(feature = "heed")]
impl<'a> BytesEncode<'a> for OutPointKey {
    type EItem = OutPointKey;

    #[inline]
    fn bytes_encode(
        item: &'a Self::EItem,
    ) -> Result<std::borrow::Cow<'a, [u8]>, BoxedError> {
        Ok(std::borrow::Cow::Borrowed(item.as_ref()))
    }
}

#[cfg(feature = "heed")]
impl<'a> BytesDecode<'a> for OutPointKey {
    type DItem = OutPointKey;

    #[inline]
    fn bytes_decode(bytes: &'a [u8]) -> Result<Self::DItem, BoxedError> {
        if bytes.len() != OUTPOINT_KEY_SIZE {
            return Err("OutPointKey must be exactly 37 bytes".into());
        }
        let mut key = [0u8; OUTPOINT_KEY_SIZE];
        key.copy_from_slice(bytes);
        let mut cursor = Cursor::new(&key[..]);
        let _ = OutPoint::deserialize_reader(&mut cursor)
            .map_err(|err| -> BoxedError { Box::new(err) })?;
        Ok(OutPointKey(key))
    }
}

#[cfg(test)]
mod test {
    use bitcoin::hashes::Hash as _;

    use crate::{
        hashes::M6id,
        transaction::outpoint::{OUTPOINT_KEY_SIZE, OutPoint, OutPointKey},
    };

    #[test]
    fn check_outpoint_key_size() -> anyhow::Result<()> {
        let variants = [
            OutPoint::Regular {
                txid: Default::default(),
                vout: u32::MAX,
            },
            OutPoint::Coinbase {
                txid: Default::default(),
                vout: u32::MAX,
            },
            OutPoint::Deposit(bitcoin::OutPoint {
                txid: bitcoin::Txid::from_byte_array([0; 32]),
                vout: u32::MAX,
            }),
            OutPoint::UnsuccessfulWithdrawal {
                m6id: M6id(bitcoin::Txid::from_byte_array([0; 32])),
                height: u32::MAX,
                vout: u32::MAX,
            },
        ];

        for op in variants {
            let serialized = borsh::to_vec(&op)?;
            anyhow::ensure!(
                serialized.len() <= OUTPOINT_KEY_SIZE,
                "unexpected serialized size: {}",
                serialized.len()
            );

            let key = OutPointKey::from(op);
            let decoded = OutPoint::from(key);
            anyhow::ensure!(decoded == op);
        }
        Ok(())
    }
}
