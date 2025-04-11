use std::str::FromStr;

use bitcoin::{
    bech32,
    hashes::{Hash as _, sha256},
};
use borsh::{BorshDeserialize, BorshSerialize};
use bytemuck::TransparentWrapper;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_with::{DeserializeAs, DisplayFromStr, SerializeAs};
use thiserror::Error;
use utoipa::ToSchema;

use crate::types::THIS_SIDECHAIN;

#[derive(Debug, thiserror::Error)]
pub enum TransparentAddressParseError {
    #[error("bs58 error")]
    Bs58(#[from] bitcoin::base58::InvalidCharacterError),
    #[error("wrong address length {0} != 20")]
    WrongLength(usize),
}

#[derive(
    BorshDeserialize, BorshSerialize, Clone, Copy, Eq, Hash, PartialEq, ToSchema,
)]
#[schema(value_type = String)]
pub struct TransparentAddress(pub [u8; 20]);

impl TransparentAddress {
    pub const ALL_ZEROS: Self = Self([0; 20]);

    pub fn as_base58(&self) -> String {
        bitcoin::base58::encode(&self.0)
    }

    /// Format with `s{sidechain_number}_` prefix and a checksum postfix
    pub fn format_for_deposit(&self) -> String {
        let prefix = format!("s{}_{}_", THIS_SIDECHAIN, self.as_base58());
        let prefix_digest =
            sha256::Hash::hash(prefix.as_bytes()).to_byte_array();
        format!("{prefix}{}", hex::encode(&prefix_digest[..3]))
    }
}

impl std::fmt::Display for TransparentAddress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_base58())
    }
}

impl std::fmt::Debug for TransparentAddress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_base58())
    }
}

impl From<[u8; 20]> for TransparentAddress {
    fn from(other: [u8; 20]) -> Self {
        Self(other)
    }
}

impl FromStr for TransparentAddress {
    type Err = TransparentAddressParseError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let address = bitcoin::base58::decode(s)?;
        Ok(TransparentAddress(address.try_into().map_err(
            |address: Vec<u8>| Self::Err::WrongLength(address.len()),
        )?))
    }
}

impl<'de> Deserialize<'de> for TransparentAddress {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        if deserializer.is_human_readable() {
            DisplayFromStr::deserialize_as(deserializer)
        } else {
            <[u8; 20] as Deserialize>::deserialize(deserializer).map(Self)
        }
    }
}

impl Serialize for TransparentAddress {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        if serializer.is_human_readable() {
            Serialize::serialize(&self.as_base58(), serializer)
        } else {
            Serialize::serialize(&self.0, serializer)
        }
    }
}

#[derive(Debug, Error)]
#[error("Wrong Bech32 HRP. Expected {expected} but decoded {decoded}")]
pub struct WrongHrpError {
    expected: bech32::Hrp,
    decoded: bech32::Hrp,
}

#[derive(Debug, Error)]
pub enum Bech32mDecodeError {
    #[error(transparent)]
    Bech32m(#[from] bech32::DecodeError),
    #[error("Invalid bytes: `{}`", hex::encode(.bytes))]
    InvalidBytes { bytes: [u8; 43] },
    #[error(transparent)]
    WrongHrp(#[from] Box<WrongHrpError>),
    #[error("Wrong decoded byte length. Must decode to 32 bytes of data.")]
    WrongSize,
    #[error("Wrong Bech32 variant. Only Bech32m is accepted.")]
    WrongVariant,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ToSchema, TransparentWrapper)]
#[repr(transparent)]
#[schema(value_type = String)]
pub struct ShieldedAddress(pub orchard::Address);

impl ShieldedAddress {
    const BECH32M_HRP: bech32::Hrp = bech32::Hrp::parse_unchecked("t-shld");

    /// Encode to Bech32m format
    pub fn bech32m_encode(&self) -> String {
        bech32::encode::<bech32::Bech32m>(
            Self::BECH32M_HRP,
            &self.0.to_raw_address_bytes(),
        )
        .expect("Bech32m Encoding should not fail")
    }

    /// Decode from Bech32m format
    pub fn bech32m_decode(s: &str) -> Result<Self, Bech32mDecodeError> {
        let (hrp, data) = bech32::decode(s)?;
        if hrp != Self::BECH32M_HRP {
            let err = WrongHrpError {
                expected: Self::BECH32M_HRP,
                decoded: hrp,
            };
            return Err(Box::new(err).into());
        }
        let Ok(bytes) = <[u8; 43]>::try_from(data) else {
            return Err(Bech32mDecodeError::WrongSize);
        };
        let res = match orchard::Address::from_raw_address_bytes(&bytes)
            .into_option()
        {
            Some(addr) => Self(addr),
            None => return Err(Bech32mDecodeError::InvalidBytes { bytes }),
        };
        if s != res.bech32m_encode() {
            return Err(Bech32mDecodeError::WrongVariant);
        }
        Ok(res)
    }
}

impl std::fmt::Display for ShieldedAddress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.bech32m_encode().fmt(f)
    }
}

impl std::hash::Hash for ShieldedAddress {
    fn hash<H>(&self, state: &mut H)
    where
        H: std::hash::Hasher,
    {
        self.0.to_raw_address_bytes().hash(state)
    }
}

impl<'de> Deserialize<'de> for ShieldedAddress {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        if deserializer.is_human_readable() {
            let s = <&'de str>::deserialize(deserializer)?;
            Self::bech32m_decode(s).map_err(|err| {
                let err = anyhow::anyhow!("{err:#}");
                <D::Error as serde::de::Error>::custom(err)
            })
        } else {
            let bytes: [u8; 43] =
                serde_with::Bytes::deserialize_as(deserializer)?;
            match orchard::Address::from_raw_address_bytes(&bytes).into_option()
            {
                Some(addr) => Ok(Self(addr)),
                None => {
                    let bytes_hex = hex::encode(bytes);
                    let err =
                        anyhow::anyhow!("Invalid address (`{bytes_hex}`)");
                    Err(<D::Error as serde::de::Error>::custom(err))
                }
            }
        }
    }
}

impl FromStr for ShieldedAddress {
    type Err = Bech32mDecodeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::bech32m_decode(s)
    }
}

impl Serialize for ShieldedAddress {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if serializer.is_human_readable() {
            Serialize::serialize(&self.bech32m_encode(), serializer)
        } else {
            let bytes: [u8; 43] = self.0.to_raw_address_bytes();
            serde_with::Bytes::serialize_as(&bytes, serializer)
        }
    }
}

#[derive(Debug, Error)]
#[error(
    "Failed to parse address: ({:#}), ({:#})",
    anyhow::anyhow!("{:#}", .shielded),
    anyhow::anyhow!("{:#}", .transparent),
)]
pub struct AddressParseError {
    shielded: <ShieldedAddress as FromStr>::Err,
    transparent: <TransparentAddress as FromStr>::Err,
}

/// Transparent or shielded address
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum Address {
    Shielded(ShieldedAddress),
    Transparent(TransparentAddress),
}

impl std::fmt::Display for Address {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Shielded(addr) => addr.fmt(f),
            Self::Transparent(addr) => addr.fmt(f),
        }
    }
}

impl FromStr for Address {
    type Err = AddressParseError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match ShieldedAddress::from_str(s) {
            Ok(addr) => Ok(Self::Shielded(addr)),
            Err(shielded) => match TransparentAddress::from_str(s) {
                Ok(addr) => Ok(Self::Transparent(addr)),
                Err(transparent) => Err(Self::Err {
                    shielded,
                    transparent,
                }),
            },
        }
    }
}
