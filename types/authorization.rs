use borsh::BorshSerialize;
use rayon::{
    iter::{IntoParallelRefIterator as _, ParallelIterator as _},
    slice::ParallelSlice as _,
};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

pub use frost_ristretto255::rand_core;

use crate::{
    Address, AuthorizedTransaction, Body, GetAddress, Transaction,
    error::Authorization as Error, util::borsh::serialize as borsh_serialize,
};

pub type Signature = frost_ristretto255::Signature;
pub type SigningKey = frost_ristretto255::SigningKey;
pub type VerifyingKey = frost_ristretto255::VerifyingKey;

pub fn get_address(verifying_key: VerifyingKey) -> Address {
    let mut hasher = blake3::Hasher::new();
    let mut reader = hasher
        .update(verifying_key.to_element().compress().as_bytes())
        .finalize_xof();
    let mut output: [u8; 20] = [0; 20];
    reader.fill(&mut output);
    Address(output)
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
    #[borsh(serialize_with = "borsh_serialize::verifying_key")]
    #[schema(value_type = String)]
    pub verifying_key: VerifyingKey,
    #[borsh(serialize_with = "borsh_serialize::signature")]
    #[schema(value_type = String)]
    pub signature: Signature,
}

impl GetAddress for Authorization {
    fn get_address(&self) -> Address {
        get_address(self.verifying_key)
    }
}

/// Derives a CSPRNG seed for a single batch verification.
struct BatchVerifier {
    hasher: blake3::Hasher,
    inner: frost_core::batch::Verifier<frost_ristretto255::Ristretto255Sha512>,
    /// Item counter, added as a suffix to the hasher before verification
    items: usize,
}

impl BatchVerifier {
    pub fn queue_item<Msg>(
        mut self,
        verifying_key: VerifyingKey,
        signature: Signature,
        msg: Msg,
    ) -> Result<Self, frost_ristretto255::Error>
    where
        Msg: AsRef<[u8]>,
    {
        let Self {
            inner,
            items,
            hasher,
        } = &mut self;
        let msg_bytes = msg.as_ref();
        // Borsh encoding for hashing
        #[derive(BorshSerialize)]
        struct HashComponents<'a> {
            #[borsh(serialize_with = "borsh_serialize::verifying_key")]
            verifying_key: &'a VerifyingKey,
            #[borsh(serialize_with = "borsh_serialize::signature")]
            signature: &'a Signature,
            msg_bytes: &'a [u8],
        }
        borsh::to_writer(
            hasher,
            &HashComponents {
                verifying_key: &verifying_key,
                signature: &signature,
                msg_bytes,
            },
        )
        .expect("failed to serialize with borsh to compute a hash");
        *items += 1;
        inner.queue(frost_core::batch::Item::new(
            verifying_key,
            signature,
            msg_bytes,
        )?);
        Ok(self)
    }

    /// Performs batch verification, returning `Ok(_)` if all signatures were
    /// valid and the batch was non-empty, and `Err(_)` otherwise.
    pub fn verify(self) -> Result<(), frost_ristretto255::Error> {
        let Self {
            hasher,
            inner,
            items,
        } = self;
        let rng = {
            // move hasher so that it can be dropped early automatically
            let mut hasher = hasher;
            borsh::to_writer(&mut hasher, &items)
                .expect("failed to serialize with borsh to compute a hash");
            <rand::rngs::ChaCha20Rng as rand::SeedableRng>::from_seed(
                hasher.finalize().into(),
            )
        };
        inner.verify(rng)
    }
}

/// Required for batched verification.
/// It should be safe to re-use the same batch verification context for
/// several batched verifications.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct BatchVerificationContext {
    mac_key: [u8; blake3::KEY_LEN],
}

impl BatchVerificationContext {
    pub fn new<R>(rng: &mut R) -> Self
    where
        R: rand_core::CryptoRng,
    {
        let mut mac_key = [0; blake3::KEY_LEN];
        rng.fill_bytes(&mut mac_key);
        Self { mac_key }
    }

    /// Construct a new batch verifier
    fn verifier(&self) -> BatchVerifier {
        let Self { mac_key } = self;
        BatchVerifier {
            hasher: blake3::Hasher::new_keyed(mac_key),
            inner: frost_core::batch::Verifier::new(),
            items: 0,
        }
    }
}

pub fn verify_authorized_transaction(
    ctxt: &BatchVerificationContext,
    transaction: &AuthorizedTransaction,
) -> Result<(), Error> {
    let verifications_required = &transaction.transaction.inputs.len();
    match transaction.authorizations.len().cmp(verifications_required) {
        std::cmp::Ordering::Less => return Err(Error::NotEnoughAuthorizations),
        std::cmp::Ordering::Equal => (),
        std::cmp::Ordering::Greater => {
            return Err(Error::TooManyAuthorizations);
        }
    }
    let mut batch_verifier = ctxt.verifier();
    let tx_bytes_canonical = borsh::to_vec(&transaction.transaction)?;
    for auth in &transaction.authorizations {
        let Authorization {
            verifying_key,
            signature,
        } = auth;
        batch_verifier = batch_verifier.queue_item(
            *verifying_key,
            *signature,
            &tx_bytes_canonical,
        )?;
    }
    let () = batch_verifier.verify()?;
    Ok(())
}

pub fn verify_authorizations(
    ctxt: &BatchVerificationContext,
    body: &Body,
) -> Result<(), Error> {
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
        let mut batch_verifier = ctxt.verifier();
        for (auth, msg) in chunk {
            let Authorization {
                verifying_key,
                signature,
            } = auth;
            batch_verifier =
                batch_verifier.queue_item(*verifying_key, *signature, msg)?;
        }
        batch_verifier.verify()
    })?;
    Ok(())
}

pub fn sign<R>(
    rng: R,
    signing_key: &SigningKey,
    transaction: &Transaction,
) -> Result<Signature, Error>
where
    R: rand_core::CryptoRng,
{
    let tx_bytes_canonical = borsh::to_vec(&transaction)?;
    Ok(signing_key.sign(rng, &tx_bytes_canonical))
}

pub fn authorize<R>(
    mut rng: R,
    addresses_signing_keys: &[(Address, &SigningKey)],
    transaction: Transaction,
) -> Result<AuthorizedTransaction, Error>
where
    R: rand_core::CryptoRng,
{
    let mut authorizations: Vec<Authorization> =
        Vec::with_capacity(addresses_signing_keys.len());
    let tx_bytes_canonical = borsh::to_vec(&transaction)?;
    for (address, signing_key) in addresses_signing_keys {
        let verifying_key = VerifyingKey::from(*signing_key);
        let hash_verifying_key = get_address(verifying_key);
        if *address != hash_verifying_key {
            return Err(Error::WrongKeyForAddress {
                address: *address,
                hash_verifying_key,
            });
        }
        let authorization = Authorization {
            verifying_key,
            signature: signing_key.sign(&mut rng, &tx_bytes_canonical),
        };
        authorizations.push(authorization);
    }
    Ok(AuthorizedTransaction {
        authorizations,
        transaction,
    })
}
