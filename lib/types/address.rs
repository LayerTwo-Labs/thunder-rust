use borsh::{BorshDeserialize, BorshSerialize};
use serde::{Deserialize, Serialize};
use serde_with::{DeserializeAs, DisplayFromStr};
use utoipa::ToSchema;

#[derive(Debug, thiserror::Error)]
pub enum AddressParseError {
    #[error("bs58 error")]
    Bs58(#[from] bitcoin::base58::InvalidCharacterError),
    #[error("wrong address length {0} != 20")]
    WrongLength(usize),
}

#[derive(
    BorshDeserialize, BorshSerialize, Clone, Copy, Eq, Hash, PartialEq, ToSchema,
)]
#[schema(value_type = String)]
pub struct Address(pub [u8; 20]);

impl Address {
    pub fn to_base58(self) -> String {
        bitcoin::base58::encode(&self.0)
    }

    pub fn to_base58ck(self) -> String {
        bitcoin::base58::encode_check(&self.0)
    }
}

impl std::fmt::Display for Address {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_base58())
    }
}

impl std::fmt::Debug for Address {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_base58())
    }
}

impl From<[u8; 20]> for Address {
    fn from(other: [u8; 20]) -> Self {
        Self(other)
    }
}

impl std::str::FromStr for Address {
    type Err = AddressParseError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let address = bitcoin::base58::decode(s)?;
        Ok(Address(address.try_into().map_err(
            |address: Vec<u8>| AddressParseError::WrongLength(address.len()),
        )?))
    }
}

impl<'de> Deserialize<'de> for Address {
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

impl Serialize for Address {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        if serializer.is_human_readable() {
            Serialize::serialize(&self.to_base58(), serializer)
        } else {
            Serialize::serialize(&self.0, serializer)
        }
    }
}
