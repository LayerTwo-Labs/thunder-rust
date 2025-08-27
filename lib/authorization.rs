use borsh::BorshSerialize;
use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelIterator,
        IntoParallelRefIterator as _, ParallelIterator as _,
    },
    slice::ParallelSlice as _,
};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::types::{
    Address, AuthorizedTransaction, Body, GetAddress, Transaction, Verify,
};

pub use ed25519_dalek::{
    Signature, SignatureError, Signer, SigningKey, Verifier, VerifyingKey,
};

#[cfg(feature = "gpu-verification")]
use cuda_ed25519_verify::CudaEd25519Verifier;

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
        verify_authorized_transaction_hybrid(transaction)?;
        Ok(())
    }

    fn verify_body(body: &Body) -> Result<(), Self::Error> {
        verify_authorizations_hybrid(body)?;
        Ok(())
    }
}

pub fn get_address(verifying_key: &VerifyingKey) -> Address {
    let mut hasher = blake3::Hasher::new();
    let mut reader = hasher.update(&verifying_key.to_bytes()).finalize_xof();
    let mut output: [u8; 20] = [0; 20];
    reader.fill(&mut output);
    Address(output)
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
    let verifications_required =
        body.transactions.par_iter().map(|tx| tx.inputs.len()).sum();
    match body.authorizations.len().cmp(&verifications_required) {
        std::cmp::Ordering::Less => return Err(Error::NotEnoughAuthorizations),
        std::cmp::Ordering::Equal => (),
        std::cmp::Ordering::Greater => {
            return Err(Error::TooManyAuthorizations);
        }
    }
    if verifications_required == 0 {
        return Ok(());
    }
    // pairs of serialized txs, and the number of inputs
    let serialized_transactions_inputs: Vec<(Vec<u8>, usize)> = body
        .transactions
        .par_iter()
        .map(|tx| Ok((borsh::to_vec(tx)?, tx.inputs.len())))
        .collect::<Result<_, Error>>()?;
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

/// Hybrid CPU/GPU verification for authorized transactions
/// Splits work between GPU (individual verification) and CPU (batched) for maximum throughput
pub fn verify_authorized_transaction_hybrid(
    transaction: &AuthorizedTransaction,
) -> Result<(), Error> {
    #[cfg(feature = "gpu-verification")]
    {
        if let Ok(success) = try_verify_authorized_transaction_gpu(transaction)
        {
            return if success {
                Ok(())
            } else {
                Err(Error::Dalek(SignatureError::new()))
            };
        }
    }

    // Fallback to CPU verification
    verify_authorized_transaction(transaction)
}

/// Hybrid CPU/GPU verification for body authorizations
/// Splits signatures 50/50 between GPU and CPU - naively, ideally the split should be tuned
pub fn verify_authorizations_hybrid(body: &Body) -> Result<(), Error> {
    let needed: usize =
        body.transactions.iter().map(|tx| tx.inputs.len()).sum();
    match body.authorizations.len().cmp(&needed) {
        std::cmp::Ordering::Less => return Err(Error::NotEnoughAuthorizations),
        std::cmp::Ordering::Greater => {
            return Err(Error::TooManyAuthorizations);
        }
        std::cmp::Ordering::Equal => {}
    }
    if needed == 0 {
        return Ok(());
    }

    #[cfg(feature = "gpu-verification")]
    {
        if let Ok(success) = try_verify_authorizations_gpu_hybrid(body, needed)
        {
            return if success {
                Ok(())
            } else {
                Err(Error::Dalek(SignatureError::new()))
            };
        }
    }

    // Fallback to CPU-only verification
    verify_authorizations(body)
}

#[cfg(feature = "gpu-verification")]
fn try_verify_authorized_transaction_gpu(
    transaction: &AuthorizedTransaction,
) -> Result<bool, Error> {
    let mut verifier = CudaEd25519Verifier::new()
        .map_err(|_| Error::Dalek(SignatureError::new()))?;

    let tx_bytes_canonical = borsh::to_vec(&transaction.transaction)?;
    let messages: Vec<Vec<u8>> = std::iter::repeat_n(
        tx_bytes_canonical,
        transaction.authorizations.len(),
    )
    .collect();

    let signatures: Vec<[u8; 64]> = transaction
        .authorizations
        .iter()
        .map(|auth| auth.signature.to_bytes())
        .collect();

    let public_keys: Vec<[u8; 32]> = transaction
        .authorizations
        .iter()
        .map(|auth| auth.verifying_key.to_bytes())
        .collect();

    let (results, _perf) = verifier
        .verify_batch(&signatures, &messages, &public_keys)
        .map_err(|_| Error::Dalek(SignatureError::new()))?;

    Ok(results.iter().all(|&valid| valid))
}

#[cfg(feature = "gpu-verification")]
fn try_verify_authorizations_gpu_hybrid(
    body: &Body,
    needed: usize,
) -> Result<bool, Error> {
    let mut verifier = CudaEd25519Verifier::new()
        .map_err(|_| Error::Dalek(SignatureError::new()))?;

    // Serialize each tx exactly once (reuse existing optimized logic)
    let tx_bytes: Vec<Box<[u8]>> = body
        .transactions
        .par_iter()
        .map(|tx| {
            let len = borsh::object_length(tx).expect("len");
            let mut v = Vec::with_capacity(len);
            borsh::to_writer(&mut v, tx).expect("ser");
            v.into_boxed_slice()
        })
        .collect();

    // Prefix sums of input counts: offs[i]..offs[i+1] are the auth indices for tx i
    let mut offs = Vec::with_capacity(body.transactions.len() + 1);
    offs.push(0);
    for tx in &body.transactions {
        offs.push(offs.last().unwrap() + tx.inputs.len());
    }

    // Split work 50/50 between GPU and CPU - naively, ideally this should be tweaked
    let gpu_end = needed / 2;
    let cpu_start = gpu_end;

    // Prepare GPU data for first half
    let gpu_signatures: Vec<[u8; 64]> = body.authorizations[0..gpu_end]
        .iter()
        .map(|auth| auth.signature.to_bytes())
        .collect();

    let gpu_public_keys: Vec<[u8; 32]> = body.authorizations[0..gpu_end]
        .iter()
        .map(|auth| auth.verifying_key.to_bytes())
        .collect();

    // Build GPU messages by walking tx boundaries
    let mut gpu_messages = Vec::with_capacity(gpu_end);
    let mut txi = 0;
    let mut i = 0;
    while i < gpu_end {
        let next = offs[txi + 1].min(gpu_end);
        let reps = next - i;
        for _ in 0..reps {
            gpu_messages.push(tx_bytes[txi].to_vec());
        }
        i = next;
        txi += 1;
    }

    // Start GPU verification in parallel with CPU
    let gpu_handle = std::thread::spawn(move || {
        verifier.verify_batch(&gpu_signatures, &gpu_messages, &gpu_public_keys)
    });

    // CPU verification for second half using existing optimized chunked approach
    const CHUNK: usize = 1 << 14;
    let cpu_result: Result<(), SignatureError> = (cpu_start..needed)
        .into_par_iter()
        .step_by(CHUNK)
        .try_for_each(|start| {
            let end = (start + CHUNK).min(needed);

            let sigs: Vec<_> = body.authorizations[start..end]
                .iter()
                .map(|a| a.signature)
                .collect();
            let keys: Vec<_> = body.authorizations[start..end]
                .iter()
                .map(|a| a.verifying_key)
                .collect();

            // Build msgs by walking tx boundaries
            let mut msgs: Vec<&[u8]> = Vec::with_capacity(end - start);
            let mut txi = match offs.binary_search(&start) {
                Ok(i) => i,
                Err(i) => i - 1,
            };
            let mut i = start;
            while i < end {
                let next = offs[txi + 1].min(end);
                let reps = next - i;
                for _ in 0..reps {
                    msgs.push(&tx_bytes[txi]);
                }
                i = next;
                txi += 1;
            }

            ed25519_dalek::verify_batch(&msgs, &sigs, &keys)
        });

    // Wait for GPU results and combine
    let gpu_result = gpu_handle
        .join()
        .map_err(|_| Error::Dalek(SignatureError::new()))?
        .map_err(|_| Error::Dalek(SignatureError::new()))?;

    let cpu_success = cpu_result.is_ok();
    let gpu_success = gpu_result.0.iter().all(|&valid| valid);

    Ok(cpu_success && gpu_success)
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
