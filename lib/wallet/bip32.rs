use bip32ish::{
    Params,
    codec::{self, Codec},
    digest::{
        self, KeyInit, Update,
        array::{self, Array, ArrayN},
        common::KeySizeUser,
    },
    digest_traits::FixedOutputAs,
};
use bitcoin::hashes::{Hash, HashEngine as _, Hmac, HmacEngine, sha512};
use curve25519_dalek::Scalar;
use thiserror::Error;

#[derive(Clone, Debug)]
pub(in crate::wallet) struct SecretExtra;

impl std::ops::Add for SecretExtra {
    type Output = Self;

    #[inline(always)]
    fn add(self, _rhs: Self) -> Self::Output {
        Self
    }
}

pub(in crate::wallet) struct SecretExtraCodec;

impl Codec<SecretExtra> for SecretExtraCodec {
    type DecodeError = std::convert::Infallible;
    type EncodeError = std::convert::Infallible;

    #[inline(always)]
    fn decode<R>(_reader: R) -> Result<SecretExtra, Self::DecodeError>
    where
        R: std::io::Read,
    {
        Ok(SecretExtra)
    }

    #[inline(always)]
    fn encode<W>(_: &SecretExtra, _writer: W) -> Result<(), Self::EncodeError>
    where
        W: std::io::Write,
    {
        Ok(())
    }
}

#[repr(transparent)]
pub(in crate::wallet) struct Hasher<const PREFIX: bool>(
    HmacEngine<sha512::Hash>,
);

impl<const PREFIX: bool> KeySizeUser for Hasher<PREFIX> {
    type KeySize = array::sizes::U32;
}

impl KeyInit for Hasher<true> {
    fn new(key: &digest::Key<Self>) -> Self {
        let mut inner = HmacEngine::new(key);
        inner.input(&[0x00]);
        Self(inner)
    }

    fn new_from_slice(key: &[u8]) -> Result<Self, digest::InvalidLength> {
        let mut inner = HmacEngine::new(key);
        inner.input(&[0x00]);
        Ok(Self(inner))
    }
}

impl KeyInit for Hasher<false> {
    #[inline(always)]
    fn new(key: &digest::Key<Self>) -> Self {
        Self(HmacEngine::new(key))
    }

    #[inline(always)]
    fn new_from_slice(key: &[u8]) -> Result<Self, digest::InvalidLength> {
        Ok(Self(HmacEngine::new(key)))
    }
}

impl<const PREFIX: bool> Update for Hasher<PREFIX> {
    #[inline(always)]
    fn update(&mut self, data: &[u8]) {
        self.0.input(data)
    }

    #[inline(always)]
    fn chain(mut self, data: impl AsRef<[u8]>) -> Self
    where
        Self: Sized,
    {
        self.0.input(data.as_ref());
        self
    }
}

fn scalar_from_be_bytes(mut bytes: [u8; 32]) -> Scalar {
    bytes.reverse();
    Scalar::from_bytes_mod_order(bytes)
}

impl<const PREFIX: bool> FixedOutputAs<(Scalar, SecretExtra, ArrayN<u8, 32>)>
    for Hasher<PREFIX>
{
    fn finalize_as(self) -> (Scalar, SecretExtra, ArrayN<u8, 32>) {
        let full_digest: [u8; 64] = Hmac::from_engine(self.0).to_byte_array();
        let (zl, chaincode) = full_digest.split_first_chunk::<32>().unwrap();
        let zl = scalar_from_be_bytes(*zl);
        (zl, SecretExtra, Array::try_from(chaincode).unwrap())
    }
}

/// Marker for Bip32ish derivation over Ristretto25519
pub(in crate::wallet) struct Ristretto255;

impl Params for Ristretto255 {
    type ChaincodeSize = array::sizes::U32;

    type ChildNumberCodec = codec::BigEndian;

    type Group = curve25519_dalek::RistrettoPoint;

    type HardenedHasher = Hasher<true>;

    type NonHardenedHasher = Hasher<false>;

    type PubkeyCodec = codec::GroupEncoding;

    type SecretExtra = SecretExtra;

    type SecretExtraCodec = SecretExtraCodec;

    type SecretScalar = Scalar;

    type SecretScalarCodec = codec::PrimeField;

    const MAX_DEPTH: usize = usize::MAX;
}

pub(in crate::wallet) type HardenedDeriveError =
    bip32ish::HardenedDeriveError<Ristretto255>;

pub(in crate::wallet) type NonHardenedDeriveError =
    bip32ish::NonHardenedDeriveError<Ristretto255>;

pub(in crate::wallet) type Xpriv = bip32ish::Xpriv<Ristretto255>;

pub(in crate::wallet) fn new_master_xpriv(seed: &[u8]) -> Xpriv {
    let mut hmac_engine: HmacEngine<sha512::Hash> =
        HmacEngine::new(b"Bitcoin seed");
    hmac_engine.input(seed);
    let hmac_result: [u8; 64] = Hmac::from_engine(hmac_engine).to_byte_array();
    let (secret_bytes, chaincode) =
        hmac_result.split_first_chunk::<32>().unwrap();
    let secret_scalar = scalar_from_be_bytes(*secret_bytes);
    Xpriv::new_master(
        secret_scalar,
        SecretExtra,
        Array::try_from(chaincode).unwrap(),
    )
}

#[derive(Debug, Error)]
pub(in crate::wallet) enum Inner {
    #[error("bip32 hardened derivation error")]
    Hardened(#[from] HardenedDeriveError),
    #[error("bip32 non-hardened derivation error")]
    NonHardened(#[from] NonHardenedDeriveError),
}

#[derive(Debug, Error)]
#[error(transparent)]
#[repr(transparent)]
pub struct Error(pub(in crate::wallet) Inner);

impl<E> From<E> for Error
where
    Inner: From<E>,
{
    fn from(err: E) -> Self {
        Self(err.into())
    }
}
