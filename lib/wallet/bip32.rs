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
pub(in crate::wallet) struct Hasher<const PREFIX: bool>(blake3::Hasher);

impl<const PREFIX: bool> KeySizeUser for Hasher<PREFIX> {
    type KeySize = array::sizes::U32;
}

impl KeyInit for Hasher<true> {
    fn new(key: &digest::Key<Self>) -> Self {
        let mut inner = blake3::Hasher::new_keyed(&key.0);
        inner.update(&[0x00]);
        Self(inner)
    }

    fn new_from_slice(key: &[u8]) -> Result<Self, digest::InvalidLength> {
        let key: [u8; 32] =
            key.try_into().map_err(|_| digest::InvalidLength)?;
        let mut inner = blake3::Hasher::new_keyed(&key);
        inner.update(&[0x00]);
        Ok(Self(inner))
    }
}

impl KeyInit for Hasher<false> {
    #[inline(always)]
    fn new(key: &digest::Key<Self>) -> Self {
        Self(blake3::Hasher::new_keyed(&key.0))
    }

    #[inline(always)]
    fn new_from_slice(key: &[u8]) -> Result<Self, digest::InvalidLength> {
        let key: [u8; 32] =
            key.try_into().map_err(|_| digest::InvalidLength)?;
        Ok(Self(blake3::Hasher::new_keyed(&key)))
    }
}

impl<const PREFIX: bool> Update for Hasher<PREFIX> {
    #[inline(always)]
    fn update(&mut self, data: &[u8]) {
        self.0.update(data);
    }

    #[inline(always)]
    fn chain(mut self, data: impl AsRef<[u8]>) -> Self
    where
        Self: Sized,
    {
        self.0.update(data.as_ref());
        self
    }
}

/// Construct a scalar from uniformly random bytes, by interpreting as a
/// big-endian encoding of a 512-bit integer, and reducing modulo the group
/// order.
fn scalar_from_uniform_be_bytes(mut bytes: [u8; 64]) -> Scalar {
    bytes.reverse();
    Scalar::from_bytes_mod_order_wide(&bytes)
}

impl<const PREFIX: bool> FixedOutputAs<(Scalar, SecretExtra, ArrayN<u8, 32>)>
    for Hasher<PREFIX>
{
    fn finalize_as(self) -> (Scalar, SecretExtra, ArrayN<u8, 32>) {
        let mut output_reader = self.0.finalize_xof();
        let mut zl = [0; 64];
        output_reader.fill(&mut zl);
        let scalar = scalar_from_uniform_be_bytes(zl);
        let mut chaincode = [0; 32];
        output_reader.fill(&mut chaincode);
        (scalar, SecretExtra, Array(chaincode))
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
    let mut hasher = blake3::Hasher::new_derive_key("Thunder seed");
    hasher.update(seed);
    let mut output_reader = hasher.finalize_xof();
    let mut secret_bytes = [0u8; 64];
    output_reader.fill(&mut secret_bytes);
    let secret_scalar = scalar_from_uniform_be_bytes(secret_bytes);
    let mut chaincode = [0; 32];
    output_reader.fill(&mut chaincode);
    Xpriv::new_master(secret_scalar, SecretExtra, Array(chaincode))
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
