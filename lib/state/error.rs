use thiserror::Error;

use crate::types::{
    AmountOverflowError, AmountUnderflowError, BlockHash, M6id, MerkleRoot,
    OutPoint, Txid, UtreexoError, WithdrawalBundleError,
};

#[derive(Debug, Error)]
pub enum InvalidHeader {
    #[error("expected block hash {expected}, but computed {computed}")]
    BlockHash {
        expected: BlockHash,
        computed: BlockHash,
    },
    #[error("expected previous sidechain block hash {expected:?}, but received {received:?}")]
    PrevSideHash {
        expected: Option<BlockHash>,
        received: Option<BlockHash>,
    },
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("failed to verify authorization")]
    AuthorizationError,
    #[error(transparent)]
    AmountOverflow(#[from] AmountOverflowError),
    #[error(transparent)]
    AmountUnderflow(#[from] AmountUnderflowError),
    #[error("body too large")]
    BodyTooLarge,
    #[error(transparent)]
    BorshSerialize(borsh::io::Error),
    #[error("invalid body: expected merkle root {expected}, but computed {computed}")]
    InvalidBody {
        expected: MerkleRoot,
        computed: MerkleRoot,
    },
    #[error("invalid header: {0}")]
    InvalidHeader(InvalidHeader),
    #[error("heed error")]
    Heed(#[from] heed::Error),
    #[error("deposit block doesn't exist")]
    NoDepositBlock,
    #[error("total fees less than coinbase value")]
    NotEnoughFees,
    #[error("no tip")]
    NoTip,
    #[error("stxo {outpoint} doesn't exist")]
    NoStxo { outpoint: OutPoint },
    #[error("value in is less than value out")]
    NotEnoughValueIn,
    #[error("utxo {outpoint} doesn't exist")]
    NoUtxo { outpoint: OutPoint },
    #[error("Withdrawal bundle event block doesn't exist")]
    NoWithdrawalBundleEventBlock,
    #[error(transparent)]
    Utreexo(#[from] UtreexoError),
    #[error("Utreexo proof verification failed for tx {txid}")]
    UtreexoProofFailed { txid: Txid },
    #[error("Computed Utreexo roots do not match the header roots")]
    UtreexoRootsMismatch,
    #[error("utxo double spent")]
    UtxoDoubleSpent,
    #[error("too many sigops")]
    TooManySigops,
    #[error("Unknown withdrawal bundle: {m6id}")]
    UnknownWithdrawalBundle { m6id: M6id },
    #[error(
        "Unknown withdrawal bundle confirmed in {event_block_hash}: {m6id}"
    )]
    UnknownWithdrawalBundleConfirmed {
        event_block_hash: bitcoin::BlockHash,
        m6id: M6id,
    },
    #[error("wrong public key for address")]
    WrongPubKeyForAddress,
    #[error(transparent)]
    WithdrawalBundle(#[from] WithdrawalBundleError),
}
