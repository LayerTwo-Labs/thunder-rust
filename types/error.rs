use thiserror::Error;

#[derive(Debug, Error)]
#[error("Bitcoin amount overflow")]
pub struct AmountOverflow;

#[derive(Debug, Error)]
#[error("Bitcoin amount underflow")]
pub struct AmountUnderflow;

#[derive(Debug, Error)]
pub enum Authorization {
    #[error("borsh serialization error")]
    BorshSerialize(#[from] borsh::io::Error),
    #[error("ed25519 error")]
    Ed25519(#[from] ed25519_dalek::SignatureError),
    #[error("not enough authorizations")]
    NotEnoughAuthorizations,
    #[error("too many authorizations")]
    TooManyAuthorizations,
    #[error(
        "wrong key for address: address = {address},
             hash(verifying_key) = {hash_verifying_key}"
    )]
    WrongKeyForAddress {
        address: crate::Address,
        hash_verifying_key: crate::Address,
    },
}

#[derive(Debug, Error)]
pub enum ComputeFee {
    #[error("underfunded (value in < value out)")]
    Underfunded,
    #[error("value in overflow")]
    ValueInOverflow(#[source] AmountOverflow),
    #[error("value out overflow")]
    ValueOutOverflow(#[source] AmountOverflow),
}

#[derive(Debug, Error)]
pub enum ParseAddress {
    #[error("bs58 error")]
    Bs58(#[from] bitcoin::base58::InvalidCharacterError),
    #[error("wrong address length {0} != 20")]
    WrongLength(usize),
}

#[derive(Debug, Error)]
#[error("utreexo error ({0})")]
#[repr(transparent)]
pub struct Utreexo(pub(crate) String);
