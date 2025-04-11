use borsh::BorshSerialize;
use rayon::iter::{IntoParallelRefIterator as _, ParallelIterator as _};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::types::{
    AuthorizedTransaction, Body, Transaction, TransparentAddress, orchard,
};

pub use ed25519_dalek::{
    Signature, SignatureError, Signer, SigningKey, Verifier, VerifyingKey,
};

pub fn get_address(verifying_key: &VerifyingKey) -> TransparentAddress {
    let mut hasher = blake3::Hasher::new();
    let mut reader = hasher.update(&verifying_key.to_bytes()).finalize_xof();
    let mut output: [u8; 20] = [0; 20];
    reader.fill(&mut output);
    TransparentAddress(output)
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("borsh serialization error")]
    BorshSerialize(#[from] borsh::io::Error),
    #[error("ed25519_dalek error")]
    DalekError(#[from] SignatureError),
    #[error("Orchard bundle proof verification error")]
    OrchardProof(#[from] orchard::BundleProofVerificationError),
    #[error("Orchard signature verification error")]
    OrchardSignature(#[from] orchard::SignatureVerificationError),
    #[error(
        "wrong key for address: address = {address},
             hash(verifying_key) = {hash_verifying_key}"
    )]
    WrongKeyForAddress {
        address: TransparentAddress,
        hash_verifying_key: TransparentAddress,
    },
}

// Verify orchard authorization
fn verify_orchard(transaction: &Transaction) -> Result<(), Error> {
    if let Some(orchard_bundle) = &transaction.orchard_bundle {
        let txid = transaction.txid();
        let bvk = orchard_bundle.binding_validating_key();
        let binding_sig = orchard_bundle.authorization().binding_signature();
        let () = bvk.verify(txid.as_slice(), binding_sig)?;
        let () = orchard_bundle.verify_proof()?;
    };
    Ok(())
}

pub fn verify_authorized_transaction(
    transaction: &AuthorizedTransaction,
) -> Result<(), Error> {
    let () = verify_orchard(&transaction.transaction)?;

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

struct Package<'a> {
    messages: Vec<&'a [u8]>,
    signatures: Vec<Signature>,
    verifying_keys: Vec<VerifyingKey>,
}

pub fn verify_authorizations(body: &Body) -> Result<(), Error> {
    // TODO: batch orchard verifications
    let () = body.transactions.par_iter().try_for_each(verify_orchard)?;
    let input_numbers = body
        .transactions
        .iter()
        .map(|transaction| transaction.inputs.len());
    let serialized_transactions: Vec<Vec<u8>> = body
        .transactions
        .par_iter()
        .map(borsh::to_vec)
        .collect::<Result<_, _>>()?;
    let serialized_transactions =
        serialized_transactions.iter().map(Vec::as_slice);
    let messages = input_numbers.zip(serialized_transactions).flat_map(
        |(input_number, serialized_transaction)| {
            std::iter::repeat_n(serialized_transaction, input_number)
        },
    );

    let pairs = body.authorizations.iter().zip(messages).collect::<Vec<_>>();

    let num_threads = rayon::current_num_threads();
    let num_authorizations = body.authorizations.len();
    let package_size = num_authorizations / num_threads;
    let mut packages: Vec<Package> = Vec::with_capacity(num_threads);
    for i in 0..num_threads {
        let mut package = Package {
            messages: Vec::with_capacity(package_size),
            signatures: Vec::with_capacity(package_size),
            verifying_keys: Vec::with_capacity(package_size),
        };
        for (authorization, message) in
            &pairs[i * package_size..(i + 1) * package_size]
        {
            package.messages.push(*message);
            package.signatures.push(authorization.signature);
            package.verifying_keys.push(authorization.verifying_key);
        }
        packages.push(package);
    }
    for (authorization, message) in &pairs[num_threads * package_size..] {
        packages[num_threads - 1].messages.push(*message);
        packages[num_threads - 1]
            .signatures
            .push(authorization.signature);
        packages[num_threads - 1]
            .verifying_keys
            .push(authorization.verifying_key);
    }
    assert_eq!(
        packages.iter().map(|p| p.signatures.len()).sum::<usize>(),
        body.authorizations.len()
    );
    packages
        .par_iter()
        .map(
            |Package {
                 messages,
                 signatures,
                 verifying_keys,
             }| {
                ed25519_dalek::verify_batch(
                    messages,
                    signatures,
                    verifying_keys,
                )
            },
        )
        .collect::<Result<(), SignatureError>>()?;
    Ok(())
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

impl Authorization {
    pub fn get_address(&self) -> TransparentAddress {
        get_address(&self.verifying_key)
    }

    pub fn verify_transaction(
        transaction: &AuthorizedTransaction,
    ) -> Result<(), Error> {
        verify_authorized_transaction(transaction)?;
        Ok(())
    }

    pub fn verify_body(body: &Body) -> Result<(), Error> {
        verify_authorizations(body)?;
        Ok(())
    }
}

pub fn sign_orchard(
    signing_keys: &[orchard::SpendAuthorizingKey],
    transaction: Transaction<
        orchard::InProgress<orchard::BundleProof, orchard::Unauthorized>,
    >,
) -> Result<Transaction, orchard::BuildError> {
    let sighash: [u8; 32] = transaction.txid().0;
    let Transaction {
        inputs,
        proof,
        outputs,
        orchard_bundle,
    } = transaction;
    let orchard_bundle = orchard_bundle
        .map(|bundle| {
            bundle.apply_signatures(rand::rngs::OsRng, sighash, signing_keys)
        })
        .transpose()?;
    let transaction = Transaction {
        inputs,
        proof,
        outputs,
        orchard_bundle,
    };
    Ok(transaction)
}

pub fn sign(
    signing_key: &SigningKey,
    transaction: &Transaction,
) -> Result<Signature, Error> {
    let tx_bytes_canonical = borsh::to_vec(&transaction)?;
    Ok(signing_key.sign(&tx_bytes_canonical))
}

pub fn authorize(
    addresses_signing_keys: &[(TransparentAddress, &SigningKey)],
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
