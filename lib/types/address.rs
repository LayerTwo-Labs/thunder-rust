use borsh::BorshSerialize;
use serde::{Deserialize, Serialize};
use serde_with::{DeserializeAs, DisplayFromStr};

#[derive(Debug, thiserror::Error)]
pub enum AddressParseError {
    #[error("bs58 error")]
    Bs58(#[from] bs58::decode::Error),
    #[error("wrong address length {0} != 20")]
    WrongLength(usize),
}

#[derive(BorshSerialize, Clone, Copy, Eq, PartialEq, Hash)]
pub struct Address(pub [u8; 20]);

impl Address {
    pub fn to_base58(self) -> String {
        bs58::encode(self.0)
            .with_alphabet(bs58::Alphabet::BITCOIN)
            .with_check()
            .into_string()
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
        let address = bs58::decode(s)
            .with_alphabet(bs58::Alphabet::BITCOIN)
            .with_check(None)
            .into_vec()?;
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
            <[u8; 20]>::deserialize(deserializer).map(Self)
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

impl utoipa::PartialSchema for Address {
    fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
        let obj = utoipa::openapi::Object::with_type(
            utoipa::openapi::SchemaType::String,
        );
        utoipa::openapi::RefOr::T(utoipa::openapi::Schema::Object(obj))
    }
}

impl utoipa::ToSchema<'static> for Address {
    fn schema() -> (
        &'static str,
        utoipa::openapi::RefOr<utoipa::openapi::schema::Schema>,
    ) {
        ("Address", <Address as utoipa::PartialSchema>::schema())
    }
}
