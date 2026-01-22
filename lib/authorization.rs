use borsh::BorshSerialize;
use fips205::traits::{SerDes as _, Signer, Verifier};
use rayon::iter::{IntoParallelRefIterator as _, ParallelIterator as _};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_with::{DeserializeAs, IfIsHumanReadable, SerializeAs, serde_as};
use thiserror::Error;
use utoipa::ToSchema;

use crate::types::{
    Address, AuthorizedTransaction, Body, GetAddress, Transaction, Verify,
};

const FIPS205_PRE_HASH: fips205::Ph = fips205::Ph::SHAKE256;

const FIPS205_CTX: &[u8] = &[];

const FIPS205_HEDGED: bool = false;

#[serde_as]
#[derive(BorshSerialize, Clone, Deserialize, Eq, PartialEq, Serialize)]
#[repr(transparent)]
pub struct Signature(
    #[serde_as(
        as = "IfIsHumanReadable<serde_with::hex::Hex, serde_with::Bytes>"
    )]
    pub [u8; fips205::slh_dsa_shake_256s::SIG_LEN],
);

impl std::fmt::Debug for Signature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::LowerHex::fmt(&self, f)
    }
}

impl std::fmt::LowerHex for Signature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&hex::encode(self.0))
    }
}

impl std::fmt::UpperHex for Signature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&hex::encode_upper(self.0))
    }
}

#[derive(Clone)]
#[repr(transparent)]
pub struct VerifyingKey(pub fips205::slh_dsa_shake_256s::PublicKey);

impl VerifyingKey {
    fn to_bytes(&self) -> [u8; fips205::slh_dsa_shake_256s::PK_LEN] {
        self.0.clone().into_bytes()
    }

    // Hash message before verifying signature
    fn hash_verify(&self, msg: &[u8], sig: &Signature) -> bool {
        self.0
            .hash_verify(msg, &sig.0, FIPS205_CTX, &FIPS205_PRE_HASH)
    }
}

impl BorshSerialize for VerifyingKey {
    fn serialize<W: std::io::Write>(
        &self,
        writer: &mut W,
    ) -> std::io::Result<()> {
        BorshSerialize::serialize(&self.to_bytes(), writer)
    }
}

impl std::fmt::Debug for VerifyingKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::LowerHex::fmt(&self, f)
    }
}

impl<'de> Deserialize<'de> for VerifyingKey {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let bytes = IfIsHumanReadable::<
            serde_with::hex::Hex,
            serde_with::Bytes
        >::deserialize_as(deserializer)?;
        match fips205::slh_dsa_shake_256s::PublicKey::try_from_bytes(&bytes) {
            Ok(vk) => Ok(Self(vk)),
            Err(err) => Err(<D::Error as serde::de::Error>::custom(err)),
        }
    }
}

impl Eq for VerifyingKey {}

impl std::fmt::LowerHex for VerifyingKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&hex::encode(self.to_bytes()))
    }
}

impl PartialEq for VerifyingKey {
    fn eq(&self, other: &Self) -> bool {
        self.to_bytes().eq(&other.to_bytes())
    }
}

impl Serialize for VerifyingKey {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let bytes = self.0.clone().into_bytes();
        IfIsHumanReadable::<
            serde_with::hex::Hex,
            serde_with::Bytes
        >::serialize_as(&bytes, serializer)
    }
}

impl std::fmt::UpperHex for VerifyingKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&hex::encode_upper(self.to_bytes()))
    }
}

#[derive(Debug, Error)]
#[error("fips205 signing error: `{}`", .0)]
#[repr(transparent)]
pub struct Fips205SigningError(&'static str);

#[derive(Debug, Error)]
#[error("failed to decode signing key from bytes: `{}`", .0)]
#[repr(transparent)]
pub struct DecodeSigningKeyError(&'static str);

#[repr(transparent)]
pub struct SigningKey(pub fips205::slh_dsa_shake_256s::PrivateKey);

impl SigningKey {
    pub fn from_seeds(
        sk_seed: &[u8; fips205::slh_dsa_shake_256s::N],
        sk_prf: &[u8; fips205::slh_dsa_shake_256s::N],
        pk_seed: &[u8; fips205::slh_dsa_shake_256s::N],
    ) -> Self {
        use fips205::traits::KeyGen as _;
        let (_vk, sk) = fips205::slh_dsa_shake_256s::KG::keygen_with_seeds(
            sk_seed, sk_prf, pk_seed,
        );
        Self(sk)
    }

    pub fn verifying_key(&self) -> VerifyingKey {
        VerifyingKey(self.0.get_public_key())
    }

    // Hash and sign
    fn hash_sign_with_rng<R>(
        &self,
        rng: &mut R,
        msg: &[u8],
    ) -> Result<Signature, Fips205SigningError>
    where
        R: rand_core::CryptoRngCore,
    {
        match self.0.try_hash_sign_with_rng(
            rng,
            msg,
            FIPS205_CTX,
            &FIPS205_PRE_HASH,
            FIPS205_HEDGED,
        ) {
            Ok(sig) => Ok(Signature(sig)),
            Err(err) => Err(Fips205SigningError(err)),
        }
    }
}

impl TryFrom<&[u8; fips205::slh_dsa_shake_256s::SK_LEN]> for SigningKey {
    type Error = DecodeSigningKeyError;

    fn try_from(
        sk_bytes: &[u8; fips205::slh_dsa_shake_256s::SK_LEN],
    ) -> Result<Self, Self::Error> {
        match fips205::slh_dsa_shake_256s::PrivateKey::try_from_bytes(sk_bytes)
        {
            Ok(sk) => Ok(Self(sk)),
            Err(err) => Err(DecodeSigningKeyError(err)),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("borsh serialization error")]
    BorshSerialize(#[from] borsh::io::Error),
    #[error("invalid signature")]
    InvalidSignature,
    #[error("not enough authorizations")]
    NotEnoughAuthorizations,
    #[error(transparent)]
    Signing(#[from] Fips205SigningError),
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
    #[schema(value_type = String)]
    pub verifying_key: VerifyingKey,
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
    for auth in &transaction.authorizations {
        if !auth
            .verifying_key
            .hash_verify(&tx_bytes_canonical, &auth.signature)
        {
            return Err(Error::InvalidSignature);
        };
    }
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
    for (auth, msg) in pairs {
        if !auth.verifying_key.hash_verify(msg, &auth.signature) {
            return Err(Error::InvalidSignature);
        };
    }
    Ok(())
}

pub fn sign<R>(
    rng: &mut R,
    signing_key: &SigningKey,
    transaction: &Transaction,
) -> Result<Signature, Error>
where
    R: rand_core::CryptoRngCore,
{
    let tx_bytes_canonical = borsh::to_vec(&transaction)?;
    let sig = signing_key.hash_sign_with_rng(rng, &tx_bytes_canonical)?;
    Ok(sig)
}

pub fn authorize<R>(
    rng: &mut R,
    addresses_signing_keys: &[(Address, &SigningKey)],
    transaction: Transaction,
) -> Result<AuthorizedTransaction, Error>
where
    R: rand_core::CryptoRngCore,
{
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
            signature: signing_key
                .hash_sign_with_rng(rng, &tx_bytes_canonical)?,
        };
        authorizations.push(authorization);
    }
    Ok(AuthorizedTransaction {
        authorizations,
        transaction,
    })
}
