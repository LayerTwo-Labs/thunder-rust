use fatality::{Fatality, Split, fatality};
use sneed::{db::error as db, env::error as env, rwtxn::error as rwtxn};
use thiserror::Error;
use transitive::Transitive;

use crate::{
    authorization,
    types::{
        Address, AmountOverflowError, AmountUnderflowError, BlockHash,
        ComputeMerkleRootError, M6id, MerkleRoot, OutPoint, Txid, UtreexoError,
        WithdrawalBundleError,
    },
    util::FatalitySplitWrappers,
};

#[derive(Debug, Error)]
#[error(
    "invalid body: expected merkle root {expected}, but computed {computed}"
)]
pub struct InvalidBody {
    pub expected: MerkleRoot,
    pub computed: MerkleRoot,
}

#[derive(Debug, Error)]
pub(in crate::state) enum InvalidHeaderInner {
    #[error("expected block hash {expected}, but computed {computed}")]
    BlockHash {
        expected: BlockHash,
        computed: BlockHash,
    },
    #[error(
        "expected previous sidechain block hash {expected:?}, but received {received:?}"
    )]
    PrevSideHash {
        expected: Option<BlockHash>,
        received: Option<BlockHash>,
    },
}

#[derive(Debug, Error)]
#[error("invalid header: {0}")]
#[repr(transparent)]
pub struct InvalidHeader(#[from] InvalidHeaderInner);

#[derive(Debug, Error)]
#[error("utxo {outpoint} doesn't exist")]
pub struct NoUtxo {
    pub outpoint: OutPoint,
}

#[fatality(splitable)]
pub enum FillTransaction {
    #[error(transparent)]
    NoUtxo(#[from] NoUtxo),
    #[error(transparent)]
    #[fatal]
    TryGet(#[from] Box<db::TryGet>),
}

impl From<db::TryGet> for FillTransaction {
    fn from(err: db::TryGet) -> Self {
        Self::TryGet(Box::new(err))
    }
}

#[derive(Debug, Error)]
pub enum ValidateCoinbase {
    #[error(transparent)]
    AmountOverflow(#[from] AmountOverflowError),
}

#[fatality(splitable)]
pub(in crate::state) enum ValidateFilledTransactionInner {
    #[error(transparent)]
    AmountOverflow(#[from] AmountOverflowError),
    #[error("value in is less than value out")]
    NotEnoughValueIn,
}

#[derive(Debug, Error)]
#[error(
    "wrong pubkey (`{}`) for address (`{address}`) at index (`{index}`)",
    hex::encode(.pubkey.as_bytes())
)]
pub(in crate::state) struct WrongPubKeyForAddress {
    pub pubkey: ed25519_dalek::VerifyingKey,
    pub address: Address,
    pub index: usize,
}

#[fatality(splitable)]
pub(in crate::state) enum ValidateAuthorizedFilledTransactionInner {
    #[error(transparent)]
    #[fatal(forward)]
    Authorization(#[from] authorization::Error),
    #[error(transparent)]
    #[fatal(forward)]
    FilledTransaction(#[from] ValidateFilledTransactionInner),
    #[error(transparent)]
    WrongPubKeyForAddress(#[from] Box<WrongPubKeyForAddress>),
}

impl From<WrongPubKeyForAddress> for ValidateAuthorizedFilledTransactionInner {
    fn from(err: WrongPubKeyForAddress) -> Self {
        Self::WrongPubKeyForAddress(Box::new(err))
    }
}

#[fatality(splitable)]
pub(in crate::state) enum ValidateAuthorizedTransactionInner {
    #[error(transparent)]
    #[fatal(forward)]
    AuthorizedFilledTransaction(
        #[from] ValidateAuthorizedFilledTransactionInner,
    ),
    #[error(transparent)]
    #[fatal(forward)]
    FillTransaction(#[from] FillTransaction),
}

#[derive(Debug, Error)]
#[error("failed to validate transaction (`{txid}`)")]
pub(in crate::state) struct ValidateTransaction<Source> {
    pub txid: Txid,
    pub(in crate::state) source: Source,
}

impl<Source> Fatality for ValidateTransaction<Source>
where
    Source: Fatality + 'static,
{
    fn is_fatal(&self) -> bool {
        self.source.is_fatal()
    }
}

impl<Source> Split for ValidateTransaction<Source>
where
    Source: Split + 'static,
{
    type Fatal = ValidateTransaction<Source::Fatal>;
    type Jfyi = ValidateTransaction<Source::Jfyi>;
    fn split(self) -> std::result::Result<Self::Jfyi, Self::Fatal> {
        let Self { txid, source } = self;
        match source.split() {
            Ok(jfyi) => Ok(ValidateTransaction { txid, source: jfyi }),
            Err(fatal) => Err(ValidateTransaction {
                txid,
                source: fatal,
            }),
        }
    }
}

#[derive(Debug, Error)]
#[error(transparent)]
#[repr(transparent)]
pub struct ValidateFilledTransaction {
    #[from]
    pub(in crate::state) inner:
        ValidateTransaction<ValidateFilledTransactionInner>,
}

#[derive(Debug, Error)]
#[error(transparent)]
#[repr(transparent)]
pub struct ValidateAuthorizedFilledTransaction {
    #[from]
    pub(in crate::state) inner:
        ValidateTransaction<ValidateAuthorizedFilledTransactionInner>,
}

impl From<ValidateFilledTransaction> for ValidateAuthorizedFilledTransaction {
    fn from(source_err: ValidateFilledTransaction) -> Self {
        Self {
            inner: ValidateTransaction {
                txid: source_err.inner.txid,
                source: source_err.inner.source.into(),
            },
        }
    }
}

#[derive(Debug, Error)]
#[error(transparent)]
#[repr(transparent)]
pub struct ValidateAuthorizedTransaction {
    #[from]
    pub(in crate::state) inner:
        ValidateTransaction<ValidateAuthorizedTransactionInner>,
}

impl From<ValidateAuthorizedFilledTransaction>
    for ValidateAuthorizedTransaction
{
    fn from(source_err: ValidateAuthorizedFilledTransaction) -> Self {
        Self {
            inner: ValidateTransaction {
                txid: source_err.inner.txid,
                source: source_err.inner.source.into(),
            },
        }
    }
}

#[derive(Debug, Error)]
#[error("utxo double spent")]
pub struct UtxoDoubleSpent;

#[fatality(splitable)]
pub(in crate::state) enum ValidateBlockInner {
    #[error(transparent)]
    AmountOverflow(#[from] AmountOverflowError),
    #[error(transparent)]
    #[fatal(forward)]
    Authorization(#[from] authorization::Error),
    #[error("body too large")]
    BodyTooLarge,
    #[error(transparent)]
    #[fatal]
    BorshSerialize(borsh::io::Error),
    #[error(transparent)]
    ComputeMerkleRoot(#[from] ComputeMerkleRootError),
    #[error(transparent)]
    #[fatal]
    DbTryGet(#[from] Box<db::TryGet>),
    #[error(transparent)]
    #[fatal(forward)]
    FillTransaction(#[from] ValidateTransaction<FillTransaction>),
    #[error(transparent)]
    InvalidBody(#[from] InvalidBody),
    #[error(transparent)]
    InvalidHeader(#[from] InvalidHeader),
    #[error("total fees less than coinbase value")]
    NotEnoughFees,
    #[error("too many sigops")]
    TooManySigops,
    #[error(transparent)]
    Utreexo(#[from] UtreexoError),
    #[error("Utreexo proof verification failed for tx {txid}")]
    UtreexoProofFailed { txid: Txid },
    #[error("Computed Utreexo roots do not match the header roots")]
    UtreexoRootsMismatch,
    #[error(transparent)]
    UtxoDoubleSpent(#[from] UtxoDoubleSpent),
    #[error(transparent)]
    #[fatal(forward)]
    ValidateFilledTransaction(
        #[from] ValidateTransaction<ValidateFilledTransactionInner>,
    ),
    #[error(transparent)]
    WrongPubKeyForAddress(#[from] Box<WrongPubKeyForAddress>),
}

impl From<db::TryGet> for ValidateBlockInner {
    fn from(err: db::TryGet) -> Self {
        Self::DbTryGet(Box::new(err))
    }
}

impl From<WrongPubKeyForAddress> for ValidateBlockInner {
    fn from(err: WrongPubKeyForAddress) -> Self {
        Self::WrongPubKeyForAddress(Box::new(err))
    }
}

#[derive(Debug, Error)]
#[error("failed to validate block")]
struct ValidateBlockWrapper<Inner>(#[from] Inner);

impl Fatality for ValidateBlockWrapper<ValidateBlockInner> {
    fn is_fatal(&self) -> bool {
        self.0.is_fatal()
    }
}

impl Split for ValidateBlockWrapper<ValidateBlockInner> {
    type Fatal = ValidateBlockWrapper<<ValidateBlockInner as Split>::Fatal>;
    type Jfyi = ValidateBlockWrapper<<ValidateBlockInner as Split>::Jfyi>;
    fn split(self) -> std::result::Result<Self::Jfyi, Self::Fatal> {
        match self.0.split() {
            Ok(jfyi) => Ok(jfyi.into()),
            Err(fatal) => Err(fatal.into()),
        }
    }
}

FatalitySplitWrappers!(
    pub ValidateBlock,
    ValidateBlockWrapper<ValidateBlockInner>
);

impl<Err> From<Err> for ValidateBlock
where
    ValidateBlockInner: From<Err>,
{
    fn from(err: Err) -> Self {
        Self(ValidateBlockWrapper(err.into()))
    }
}

#[derive(Transitive)]
#[fatality(splitable)]
#[transitive(
    from(db::Delete, db::Error),
    from(db::Put, db::Error),
    from(db::TryGet, db::Error)
)]
pub(in crate::state) enum ConnectTransactionInner {
    #[error(transparent)]
    #[fatal]
    Db(#[from] Box<db::Error>),
    #[error(transparent)]
    NoUtxo(#[from] NoUtxo),
}

impl From<db::Error> for ConnectTransactionInner {
    fn from(err: db::Error) -> Self {
        Self::Db(Box::new(err))
    }
}

#[derive(Debug, Error)]
#[error("failed to connect transaction (`{txid}`)")]
struct ConnectTransactionWrapper<Inner> {
    txid: Txid,
    source: Inner,
}

impl Fatality for ConnectTransactionWrapper<ConnectTransactionInner> {
    fn is_fatal(&self) -> bool {
        self.source.is_fatal()
    }
}

impl Split for ConnectTransactionWrapper<ConnectTransactionInner> {
    type Fatal =
        ConnectTransactionWrapper<<ConnectTransactionInner as Split>::Fatal>;
    type Jfyi =
        ConnectTransactionWrapper<<ConnectTransactionInner as Split>::Jfyi>;
    fn split(self) -> std::result::Result<Self::Jfyi, Self::Fatal> {
        let txid = self.txid;
        match self.source.split() {
            Ok(jfyi) => Ok(ConnectTransactionWrapper { txid, source: jfyi }),
            Err(fatal) => Err(ConnectTransactionWrapper {
                txid,
                source: fatal,
            }),
        }
    }
}

FatalitySplitWrappers!(
    pub ConnectTransaction,
    ConnectTransactionWrapper<ConnectTransactionInner>
);

impl ConnectTransaction {
    pub(in crate::state) fn new(
        txid: Txid,
        inner: ConnectTransactionInner,
    ) -> Self {
        Self(ConnectTransactionWrapper {
            txid,
            source: inner,
        })
    }
}

#[derive(Transitive)]
#[fatality(splitable)]
#[transitive(
    from(db::Delete, db::Error),
    from(db::Put, db::Error),
    from(db::TryGet, db::Error)
)]
pub(in crate::state) enum ConnectBlockInner {
    #[error(transparent)]
    ComputeMerkleRoot(#[from] ComputeMerkleRootError),
    #[error(transparent)]
    #[fatal(forward)]
    ConnectTransaction(#[from] ConnectTransaction),
    #[error(transparent)]
    #[fatal]
    Db(#[from] Box<db::Error>),
    #[error(transparent)]
    InvalidBody(#[from] InvalidBody),
    #[error(transparent)]
    InvalidHeader(InvalidHeader),
    #[error(transparent)]
    Utreexo(#[from] UtreexoError),
}

impl From<db::Error> for ConnectBlockInner {
    fn from(err: db::Error) -> Self {
        Self::Db(Box::new(err))
    }
}

#[derive(Debug, Error)]
#[error("failed to connect block")]
#[repr(transparent)]
struct ConnectBlockWrapper<Inner>(#[from] Inner);

impl Fatality for ConnectBlockWrapper<ConnectBlockInner> {
    fn is_fatal(&self) -> bool {
        self.0.is_fatal()
    }
}

impl Split for ConnectBlockWrapper<ConnectBlockInner> {
    type Fatal = ConnectBlockWrapper<<ConnectBlockInner as Split>::Fatal>;
    type Jfyi = ConnectBlockWrapper<<ConnectBlockInner as Split>::Jfyi>;
    fn split(self) -> std::result::Result<Self::Jfyi, Self::Fatal> {
        match self.0.split() {
            Ok(jfyi) => Ok(jfyi.into()),
            Err(fatal) => Err(fatal.into()),
        }
    }
}

FatalitySplitWrappers!(
    pub ConnectBlock,
    ConnectBlockWrapper<ConnectBlockInner>
);

impl<Err> From<Err> for ConnectBlock
where
    ConnectBlockInner: From<Err>,
{
    fn from(err: Err) -> Self {
        Self(ConnectBlockWrapper(err.into()))
    }
}

#[fatality(splitable)]
pub enum ApplyBlock {
    #[error(transparent)]
    #[fatal(forward)]
    Connect(#[from] ConnectBlock),
    #[error(transparent)]
    #[fatal(forward)]
    Validate(#[from] ValidateBlock),
}

#[derive(Debug, Error)]
#[error("stxo {outpoint} doesn't exist")]
pub(in crate::state) struct NoStxo {
    pub outpoint: OutPoint,
}

#[derive(Debug, Error)]
#[error("no tip")]
pub struct NoTip;

#[derive(Debug, Error, Transitive)]
#[transitive(
    from(db::Delete, db::Error),
    from(db::Put, db::Error),
    from(db::TryGet, db::Error)
)]
pub(in crate::state) enum DisconnectTipInner {
    #[error(transparent)]
    Db(#[from] Box<db::Error>),
    #[error(transparent)]
    InvalidHeader(InvalidHeader),
    #[error(transparent)]
    NoStxo(#[from] NoStxo),
    #[error(transparent)]
    NoTip(#[from] NoTip),
    #[error(transparent)]
    NoUtxo(#[from] NoUtxo),
    #[error(transparent)]
    Utreexo(#[from] UtreexoError),
}

impl From<db::Error> for DisconnectTipInner {
    fn from(err: db::Error) -> Self {
        Self::Db(Box::new(err))
    }
}

#[derive(Debug, Error)]
#[error("failed to disconnect tip")]
pub struct DisconnectTip(#[source] pub(in crate::state) DisconnectTipInner);

impl<Err> From<Err> for DisconnectTip
where
    DisconnectTipInner: From<Err>,
{
    fn from(err: Err) -> Self {
        Self(err.into())
    }
}

#[derive(Debug, Error, Transitive)]
#[transitive(from(db::IterInit, db::Iter), from(db::IterItem, db::Iter))]
pub(in crate::state) enum CollectWithdrawalBundle {
    #[error(transparent)]
    AmountOverflow(#[from] AmountOverflowError),
    #[error(transparent)]
    DbIter(#[from] Box<db::Iter>),
    #[error(transparent)]
    WithdrawalBundle(#[from] WithdrawalBundleError),
}

impl From<db::Iter> for CollectWithdrawalBundle {
    fn from(err: db::Iter) -> Self {
        Self::DbIter(Box::new(err))
    }
}

#[derive(Debug, Error)]
#[error("Unknown withdrawal bundle: {m6id}")]
pub(in crate::state) struct UnknownWithdrawalBundle {
    pub m6id: M6id,
}

#[derive(Debug, Error, Transitive)]
#[transitive(
    from(db::Clear, db::Error),
    from(db::Delete, db::Error),
    from(db::IterInit, db::Error),
    from(db::IterItem, db::Error),
    from(db::Last, db::Error),
    from(db::Put, db::Error),
    from(db::TryGet, db::Error)
)]
pub(in crate::state) enum Connect2wpdInner {
    #[error("failed to collect withdrawal bundle")]
    CollectWithdrawalBundle(#[from] CollectWithdrawalBundle),
    #[error(transparent)]
    Db(#[from] Box<db::Error>),
    #[error(transparent)]
    NoTip(#[from] NoTip),
    #[error(transparent)]
    UnknownWithdrawalBundle(#[from] UnknownWithdrawalBundle),
    #[error(
        "Unknown withdrawal bundle confirmed in {event_block_hash}: {m6id}"
    )]
    UnknownWithdrawalBundleConfirmed {
        event_block_hash: bitcoin::BlockHash,
        m6id: M6id,
    },
    #[error(transparent)]
    Utreexo(#[from] UtreexoError),
}

impl From<db::Error> for Connect2wpdInner {
    fn from(err: db::Error) -> Self {
        Self::Db(Box::new(err))
    }
}

#[derive(Debug, Error)]
#[error("failed to connect 2wpd")]
pub struct Connect2wpd(#[source] pub(in crate::state) Connect2wpdInner);

impl<Err> From<Err> for Connect2wpd
where
    Connect2wpdInner: From<Err>,
{
    fn from(err: Err) -> Self {
        Self(err.into())
    }
}

#[derive(Debug, Error, Transitive)]
#[transitive(
    from(db::Delete, db::Error),
    from(db::Last, db::Error),
    from(db::Put, db::Error),
    from(db::TryGet, db::Error)
)]
pub(in crate::state) enum Disconnect2wpdInner {
    #[error(transparent)]
    Db(#[from] Box<db::Error>),
    #[error("deposit block doesn't exist")]
    NoDepositBlock,
    #[error(transparent)]
    NoStxo(#[from] NoStxo),
    #[error(transparent)]
    NoUtxo(#[from] NoUtxo),
    #[error("Withdrawal bundle event block doesn't exist")]
    NoWithdrawalBundleEventBlock,
    #[error(transparent)]
    UnknownWithdrawalBundle(#[from] UnknownWithdrawalBundle),
    #[error(transparent)]
    Utreexo(#[from] UtreexoError),
}

impl From<db::Error> for Disconnect2wpdInner {
    fn from(err: db::Error) -> Self {
        Self::Db(Box::new(err))
    }
}

#[derive(Debug, Error)]
#[error("failed to disconnect 2wpd")]
pub struct Disconnect2wpd(#[source] pub(in crate::state) Disconnect2wpdInner);

impl<Err> From<Err> for Disconnect2wpd
where
    Disconnect2wpdInner: From<Err>,
{
    fn from(err: Err) -> Self {
        Self(err.into())
    }
}

#[derive(Debug, Error, Transitive)]
#[transitive(
    from(db::Clear, db::Error),
    from(db::Delete, db::Error),
    from(db::Error, sneed::Error),
    from(db::IterInit, db::Error),
    from(db::IterItem, db::Error),
    from(db::Last, db::Error),
    from(db::Put, db::Error),
    from(db::TryGet, db::Error),
    from(env::CreateDb, env::Error),
    from(env::Error, sneed::Error),
    from(env::WriteTxn, env::Error),
    from(rwtxn::Commit, rwtxn::Error),
    from(rwtxn::Error, sneed::Error)
)]
pub enum Error {
    #[error("failed to verify authorization")]
    Authorization,
    #[error(transparent)]
    AmountOverflow(#[from] AmountOverflowError),
    #[error(transparent)]
    AmountUnderflow(#[from] AmountUnderflowError),
    #[error(transparent)]
    ConnectBlock(#[from] ConnectBlock),
    #[error(transparent)]
    Db(#[from] Box<sneed::Error>),
    #[error(transparent)]
    InvalidBody(InvalidBody),
    #[error(transparent)]
    InvalidHeader(InvalidHeader),
    #[error("stxo {outpoint} doesn't exist")]
    NoStxo { outpoint: OutPoint },
    #[error(transparent)]
    NoTip(#[from] NoTip),
    #[error(transparent)]
    Utreexo(#[from] UtreexoError),
    #[error(transparent)]
    UtxoDoubleSpent(#[from] UtxoDoubleSpent),
    #[error(transparent)]
    WithdrawalBundle(#[from] WithdrawalBundleError),
}

impl From<sneed::Error> for Error {
    fn from(err: sneed::Error) -> Self {
        Self::Db(Box::new(err))
    }
}
