use thiserror::Error;

#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
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

#[derive(Debug, Error)]
pub enum ParsePeerAddress {
    #[error("missing port")]
    MissingPort,
    #[error(transparent)]
    Parse(#[from] url::ParseError),
}

pub mod compute_merkle_root {
    use thiserror::Error;

    use crate::{error::ComputeFee, hashes::Txid};

    #[derive(Debug, Error)]
    pub(crate) enum Inner {
        #[error("failed to compute merkle root for coinbase tx")]
        CoinbaseMerkleRoot(
            #[source] crate::transaction::outputs::error::ComputeMerkleRoot,
        ),
        #[error("failed to compute canonical size for tx ({txid})")]
        TxCanonicalSize {
            txid: Txid,
            source: borsh::io::Error,
        },
        #[error("failed to compute fee for tx ({txid})")]
        TxFee { txid: Txid, source: ComputeFee },
        #[error("failed to compute merkle root for tx ({txid})")]
        TxMerkleRoot {
            txid: Txid,
            source: crate::transaction::outputs::error::ComputeMerkleRoot,
        },
    }

    #[derive(Debug, Error)]
    #[error("failed to compute merkle root")]
    #[repr(transparent)]
    pub struct Error(Box<Inner>);

    impl From<Inner> for Error {
        fn from(err: Inner) -> Self {
            Self(Box::new(err))
        }
    }
}
pub use compute_merkle_root::Error as ComputeMerkleRoot;

#[derive(Debug, Error)]
pub enum ModifyMemForest {
    #[error(transparent)]
    ComputeMerkleRoot(#[from] ComputeMerkleRoot),
    #[error(transparent)]
    Utreexo(#[from] Utreexo),
}

pub mod withdrawal_bundle {
    use thiserror::Error;

    #[derive(Debug, Error)]
    pub(crate) enum Inner {
        #[error(
            "bundle too heavy: weight `{weight}` > max weight `{max_weight}`"
        )]
        BundleTooHeavy { weight: u64, max_weight: u64 },
    }

    #[derive(Debug, Error)]
    #[error("Withdrawal bundle error")]
    pub struct Error(#[from] Inner);
}
pub use withdrawal_bundle::Error as WithdrawalBundle;
