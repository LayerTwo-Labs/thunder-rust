use bitcoin::hashes::{Hash as _, sha256};
use borsh::{BorshDeserialize, BorshSerialize};
use serde::{Deserialize, Serialize};
use serde_with::{DeserializeAs, DisplayFromStr};
use utoipa::ToSchema;

use crate::{THIS_SIDECHAIN, error::ParseAddress as ParseAddressError};

#[derive(
    BorshDeserialize, BorshSerialize, Clone, Copy, Eq, Hash, PartialEq, ToSchema,
)]
#[schema(value_type = String)]
pub struct Address(pub [u8; 20]);

impl Address {
    pub const ALL_ZEROS: Self = Self([0; 20]);

    pub fn as_base58(&self) -> String {
        bitcoin::base58::encode(&self.0)
    }

    /// Format with `s{sidechain_number}_` prefix and a checksum postfix
    pub fn format_for_deposit(&self) -> String {
        let prefix = format!("s{}_{}_", THIS_SIDECHAIN, self.as_base58());
        let prefix_digest =
            sha256::Hash::hash(prefix.as_bytes()).to_byte_array();
        format!("{prefix}{}", const_hex::encode(&prefix_digest[..3]))
    }

    /// Parse the form that [`Self::format_for_deposit`] writes
    pub fn from_deposit_address(s: &str) -> Result<Self, ParseAddressError> {
        let address_str = s
            .strip_prefix(&format!("s{THIS_SIDECHAIN}_"))
            .and_then(|rest| rest.rsplit_once('_'))
            .map(|(address_str, _checksum)| address_str)
            .ok_or_else(|| {
                ParseAddressError::NotADepositAddress(s.to_owned())
            })?;
        let address: Self = address_str.parse()?;
        if !address.format_for_deposit().eq_ignore_ascii_case(s) {
            return Err(ParseAddressError::WrongDepositChecksum(s.to_owned()));
        }
        Ok(address)
    }
}

impl std::fmt::Display for Address {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_base58())
    }
}

impl std::fmt::Debug for Address {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_base58())
    }
}

impl From<[u8; 20]> for Address {
    fn from(other: [u8; 20]) -> Self {
        Self(other)
    }
}

impl std::str::FromStr for Address {
    type Err = ParseAddressError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let address = bitcoin::base58::decode(s)?;
        Ok(Address(address.try_into().map_err(
            |address: Vec<u8>| ParseAddressError::WrongLength(address.len()),
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
            Serialize::serialize(&self.as_base58(), serializer)
        } else {
            Serialize::serialize(&self.0, serializer)
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{THIS_SIDECHAIN, address::Address};

    #[test]
    fn deposit_address_round_trip() {
        let address = Address([7u8; 20]);
        let formatted = address.format_for_deposit();
        assert_eq!(Address::from_deposit_address(&formatted).unwrap(), address);
    }

    #[test]
    fn deposit_address_rejects_a_short_checksum() {
        let address = Address([9u8; 20]);
        let formatted = address.format_for_deposit();
        let short = &formatted[..formatted.len() - 1];
        assert!(Address::from_deposit_address(short).is_err());
    }

    #[test]
    fn deposit_address_rejects_a_wrong_checksum() {
        let address = Address([3u8; 20]);
        let formatted =
            format!("s{}_{}_ffffff", THIS_SIDECHAIN, address.as_base58());
        assert!(Address::from_deposit_address(&formatted).is_err());
    }

    #[test]
    fn deposit_address_rejects_another_sidechain() {
        let address = Address([3u8; 20]);
        let formatted =
            format!("s{}_{}_000000", THIS_SIDECHAIN + 1, address.as_base58());
        assert!(Address::from_deposit_address(&formatted).is_err());
    }

    #[test]
    fn deposit_address_rejects_a_bare_address() {
        let address = Address([3u8; 20]);
        assert!(Address::from_deposit_address(&address.as_base58()).is_err());
    }
}
