use borsh::{BorshDeserialize, BorshSerialize};
use rayon::{
    iter::{IntoParallelRefIterator as _, ParallelIterator as _},
    slice::ParallelSlice as _,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use utoipa::ToSchema;

use crate::types::{
    Address, AuthorizedTransaction, Body, GetAddress, Signature,
    SignatureError, Signer, SigningKey, Transaction, Verifier, VerifyingKey,
    get_address,
};

fn borsh_serialize_verifying_key<W>(
    vk: &VerifyingKey,
    writer: &mut W,
) -> borsh::io::Result<()>
where
    W: borsh::io::Write,
{
    BorshSerialize::serialize(&vk.to_bytes(), writer)
}

fn borsh_deserialize_verifying_key<R>(
    reader: &mut R,
) -> borsh::io::Result<VerifyingKey>
where
    R: borsh::io::Read,
{
    let bytes: [u8; 32] = BorshDeserialize::deserialize_reader(reader)?;
    VerifyingKey::from_bytes(&bytes).map_err(|e| {
        borsh::io::Error::new(borsh::io::ErrorKind::InvalidData, e)
    })
}

fn borsh_serialize_signature<W>(
    sig: &Signature,
    writer: &mut W,
) -> borsh::io::Result<()>
where
    W: borsh::io::Write,
{
    BorshSerialize::serialize(&sig.to_bytes(), writer)
}

fn borsh_deserialize_signature<R>(
    reader: &mut R,
) -> borsh::io::Result<Signature>
where
    R: borsh::io::Read,
{
    let bytes: [u8; 64] = BorshDeserialize::deserialize_reader(reader)?;
    Ok(Signature::from_bytes(&bytes))
}

fn serialize_verifying_key<S>(
    vk: &VerifyingKey,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    serializer.serialize_str(&hex::encode(vk.to_bytes()))
}

fn deserialize_verifying_key<'de, D>(
    deserializer: D,
) -> Result<VerifyingKey, D::Error>
where
    D: Deserializer<'de>,
{
    let hex_str = <String as serde::Deserialize>::deserialize(deserializer)?;
    let bytes = hex::decode(hex_str).map_err(serde::de::Error::custom)?;
    if bytes.len() != 32 {
        return Err(serde::de::Error::custom("Invalid verifying key length"));
    }
    let mut array = [0u8; 32];
    array.copy_from_slice(&bytes);
    VerifyingKey::from_bytes(&array).map_err(serde::de::Error::custom)
}

fn serialize_signature<S>(
    sig: &Signature,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    serializer.serialize_str(&hex::encode(sig.to_bytes()))
}

fn deserialize_signature<'de, D>(deserializer: D) -> Result<Signature, D::Error>
where
    D: Deserializer<'de>,
{
    let hex_str = <String as serde::Deserialize>::deserialize(deserializer)?;
    let bytes = hex::decode(hex_str).map_err(serde::de::Error::custom)?;
    if bytes.len() != 64 {
        return Err(serde::de::Error::custom("Invalid signature length"));
    }
    let mut array = [0u8; 64];
    array.copy_from_slice(&bytes);
    Ok(Signature::from_bytes(&array))
}

#[derive(Debug, Clone, Deserialize, Eq, PartialEq, Serialize)]
pub struct Authorization {
    #[serde(serialize_with = "serialize_verifying_key")]
    #[serde(deserialize_with = "deserialize_verifying_key")]
    pub verifying_key: VerifyingKey,
    #[serde(serialize_with = "serialize_signature")]
    #[serde(deserialize_with = "deserialize_signature")]
    pub signature: Signature,
}

impl BorshSerialize for Authorization {
    fn serialize<W: borsh::io::Write>(
        &self,
        writer: &mut W,
    ) -> borsh::io::Result<()> {
        borsh_serialize_verifying_key(&self.verifying_key, writer)?;
        borsh_serialize_signature(&self.signature, writer)?;
        Ok(())
    }
}

impl BorshDeserialize for Authorization {
    fn deserialize_reader<R: borsh::io::Read>(
        reader: &mut R,
    ) -> borsh::io::Result<Self> {
        let verifying_key = borsh_deserialize_verifying_key(reader)?;
        let signature = borsh_deserialize_signature(reader)?;
        Ok(Authorization {
            verifying_key,
            signature,
        })
    }
}

impl utoipa::ToSchema for Authorization {
    fn name() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("Authorization")
    }
}

impl utoipa::PartialSchema for Authorization {
    fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
        use utoipa::openapi::{Object, RefOr, Schema, schema};
        let obj = Object::builder()
            .property(
                "verifying_key",
                Object::builder().schema_type(schema::Type::String),
            )
            .property(
                "signature",
                Object::builder().schema_type(schema::Type::String),
            )
            .required("verifying_key")
            .required("signature")
            .build();
        RefOr::T(Schema::Object(obj))
    }
}

impl GetAddress for Authorization {
    fn get_address(&self) -> Address {
        get_address(&self.verifying_key)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AuthorizationError {
    #[error("borsh serialization error")]
    BorshSerialize(#[from] borsh::io::Error),
    #[error("ed25519_dalek error")]
    Dalek(#[from] SignatureError),
    #[error("not enough authorizations")]
    NotEnoughAuthorizations,
    #[error("too many authorizations")]
    TooManyAuthorizations,
    #[error(
        "wrong key for address: address = {address},
             hash(verifying_key) = {hash_verifying_key}"
    )]
    WrongKeyForAddress {
        address: Address,
        hash_verifying_key: Address,
    },
}

pub trait Verify {
    type Error;
    fn verify_transaction(
        transaction: &AuthorizedTransaction,
    ) -> Result<(), Self::Error>;
    fn verify_body(body: &Body) -> Result<(), Self::Error>;
}

impl Verify for Authorization {
    type Error = AuthorizationError;

    fn verify_transaction(
        transaction: &AuthorizedTransaction,
    ) -> Result<(), Self::Error> {
        use rayon::prelude::*;

        let serialized = borsh::to_vec(&transaction.transaction)?;

        transaction
            .authorizations
            .par_iter()
            .try_for_each(|authorization| {
                authorization
                    .verifying_key
                    .verify(&serialized, &authorization.signature)
                    .map_err(AuthorizationError::from)
            })
    }

    fn verify_body(body: &Body) -> Result<(), Self::Error> {
        verify_body(body).map_err(|e| match e {
            Error::BorshSerialize(e) => AuthorizationError::BorshSerialize(e),
            Error::Dalek(e) => AuthorizationError::Dalek(e),
            Error::NotEnoughAuthorizations => {
                AuthorizationError::NotEnoughAuthorizations
            }
            Error::TooManyAuthorizations => {
                AuthorizationError::TooManyAuthorizations
            }
            Error::WrongKeyForAddress {
                address,
                hash_verifying_key,
            } => AuthorizationError::WrongKeyForAddress {
                address,
                hash_verifying_key,
            },
        })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("borsh serialization error")]
    BorshSerialize(#[from] borsh::io::Error),
    #[error("ed25519_dalek error")]
    Dalek(#[from] SignatureError),
    #[error("not enough authorizations")]
    NotEnoughAuthorizations,
    #[error("too many authorizations")]
    TooManyAuthorizations,
    #[error(
        "wrong key for address: address = {address},
             hash(verifying_key) = {hash_verifying_key}"
    )]
    WrongKeyForAddress {
        address: Address,
        hash_verifying_key: Address,
    },
}

pub fn verify_body(body: &Body) -> Result<(), Error> {
    let serialized_transactions_inputs = body
        .transactions
        .iter()
        .map(|tx| {
            let serialized = borsh::to_vec(tx)?;
            Ok((serialized, tx.inputs.len()))
        })
        .collect::<Result<Vec<_>, Error>>()?;
    let messages =
        serialized_transactions_inputs
            .iter()
            .flat_map(|(tx, n_inputs)| {
                std::iter::repeat_n(tx.as_slice(), *n_inputs)
            });
    let pairs = body.authorizations.iter().zip(messages).collect::<Vec<_>>();
    assert_eq!(pairs.len(), body.authorizations.len());
    const CHUNK_SIZE: usize = 1 << 14;
    pairs.par_chunks(CHUNK_SIZE).try_for_each(|chunk| {
        let (signatures, verifying_keys, messages): (
            Vec<Signature>,
            Vec<VerifyingKey>,
            Vec<&[u8]>,
        ) = chunk
            .iter()
            .map(|(auth, msg)| (auth.signature, auth.verifying_key, msg))
            .collect();
        ed25519_dalek::verify_batch(&messages, &signatures, &verifying_keys)
    })?;
    Ok(())
}

pub fn sign(
    signing_key: &SigningKey,
    transaction: &Transaction,
) -> Result<Signature, Error> {
    let tx_bytes_canonical = borsh::to_vec(&transaction)?;
    Ok(signing_key.sign(&tx_bytes_canonical))
}

pub fn authorize(
    addresses_signing_keys: &[(Address, &SigningKey)],
    transaction: Transaction,
) -> Result<AuthorizedTransaction, Error> {
    let mut authorizations: Vec<Authorization> =
        Vec::with_capacity(addresses_signing_keys.len());
    let tx_bytes_canonical = borsh::to_vec(&transaction)?;
    for (address, signing_key) in addresses_signing_keys {
        let hash_verifying_key = get_address(&signing_key.verifying_key());
        if *address != hash_verifying_key {
            return Err(Error::WrongKeyForAddress {
                address: *address,
                hash_verifying_key,
            });
        }
        let authorization = Authorization {
            verifying_key: signing_key.verifying_key(),
            signature: signing_key.sign(&tx_bytes_canonical),
        };
        authorizations.push(authorization);
    }
    Ok(AuthorizedTransaction {
        authorizations,
        transaction,
    })
}
