use borsh::BorshSerialize;
// use rayon::iter::{IntoParallelRefIterator as _, ParallelIterator as _};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::types::{
    Address, AuthorizedTransaction, Body, GetAddress, Transaction, Verify,
};

pub use ed25519_dalek::{
    Signature, SignatureError, Signer, SigningKey, Verifier, VerifyingKey,
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("borsh serialization error")]
    BorshSerialize(#[from] borsh::io::Error),
    #[error("ed25519_dalek error")]
    DalekError(#[from] SignatureError),
    #[error(
        "wrong key for address: address = {address},
             hash(verifying_key) = {hash_verifying_key}"
    )]
    WrongKeyForAddress {
        address: Address,
        hash_verifying_key: Address,
    },
}

fn borsh_serialize_verifying_key<W>(
    vk: &VerifyingKey,
    writer: &mut W,
) -> borsh::io::Result<()>
where
    W: borsh::io::Write,
{
    borsh::BorshSerialize::serialize(&vk.to_bytes(), writer)
}

fn borsh_serialize_signature<W>(
    sig: &Signature,
    writer: &mut W,
) -> borsh::io::Result<()>
where
    W: borsh::io::Write,
{
    borsh::BorshSerialize::serialize(&sig.to_bytes(), writer)
}

#[derive(
    BorshSerialize,
    Debug,
    Clone,
    Deserialize,
    Eq,
    PartialEq,
    Serialize,
    ToSchema,
)]
pub struct Authorization {
    #[borsh(serialize_with = "borsh_serialize_verifying_key")]
    #[schema(value_type = String)]
    pub verifying_key: VerifyingKey,
    #[borsh(serialize_with = "borsh_serialize_signature")]
    #[schema(value_type = String)]
    pub signature: Signature,
}

impl GetAddress for Authorization {
    fn get_address(&self) -> Address {
        get_address(&self.verifying_key)
    }
}

impl Verify for Authorization {
    type Error = Error;
    fn verify_transaction(
        transaction: &AuthorizedTransaction,
    ) -> Result<(), Self::Error> {
        verify_authorized_transaction(transaction)?;
        Ok(())
    }

    fn verify_body(body: &Body) -> Result<(), Self::Error> {
        verify_authorizations(body)?;
        Ok(())
    }
}

pub fn get_address(verifying_key: &VerifyingKey) -> Address {
    let hash = blake3::hash(&verifying_key.to_bytes());
    let mut output: [u8; 20] = [0; 20];
    output.copy_from_slice(&hash.as_bytes()[..20]);
    Address(output)
}

struct Package<'a> {
    messages: Vec<&'a [u8]>,
    signatures: Vec<Signature>,
    verifying_keys: Vec<VerifyingKey>,
}

pub fn verify_authorized_transaction(
    transaction: &AuthorizedTransaction,
) -> Result<(), Error> {
    let tx_bytes_canonical = borsh::to_vec(&transaction.transaction)?;
    let messages: Vec<_> = std::iter::repeat_n(
        tx_bytes_canonical.as_slice(),
        transaction.authorizations.len(),
    )
    .collect();
    let (verifying_keys, signatures): (Vec<VerifyingKey>, Vec<Signature>) =
        transaction
            .authorizations
            .iter()
            .map(
                |Authorization {
                     verifying_key,
                     signature,
                 }| (verifying_key, signature),
            )
            .unzip();
    ed25519_dalek::verify_batch(&messages, &signatures, &verifying_keys)?;
    Ok(())
}

pub fn verify_authorizations(body: &Body) -> Result<(), Error> {
    if body.authorizations.is_empty() {
        return Ok(());
    }
    let serialized_transactions: Vec<Vec<u8>> = body
        .transactions
        .par_iter()
        .map(borsh::to_vec)
        .collect::<Result<_, _>>()?;

    let mut pairs: Vec<(&[u8], &Authorization)> =
        Vec::with_capacity(body.authorizations.len());
    let mut auth_iter = body.authorizations.iter();
    for (tx_idx, tx) in body.transactions.iter().enumerate() {
        let tx_bytes = &serialized_transactions[tx_idx];
        for _ in 0..tx.inputs.len() {
            pairs.push((tx_bytes, auth_iter.next().unwrap()));
        }
    }

    assert_eq!(pairs.len(), body.authorizations.len());

    pairs.par_chunks(16384).try_for_each(|chunk| {
        let (messages, authorizations): (Vec<&[u8]>, Vec<&Authorization>) =
            chunk.iter().copied().unzip();
        let signatures: Vec<Signature> =
            authorizations.iter().map(|a| a.signature).collect();
        let verifying_keys: Vec<VerifyingKey> =
            authorizations.iter().map(|a| a.verifying_key).collect();

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
