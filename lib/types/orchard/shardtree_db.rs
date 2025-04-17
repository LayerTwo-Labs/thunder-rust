//! Storage for a [`shardtree::ShardTree`]

use std::{collections::BTreeSet, rc::Weak, sync::Arc};

use bytemuck::TransparentWrapper;
use educe::Educe;
use fallible_iterator::FallibleIterator;
use heed::{
    byteorder::BE,
    types::{SerdeBincode, U32},
};
use incrementalmerkletree::Position;
use orchard::tree::MerkleHashOrchard;
use parking_lot::RwLock;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_with::{DeserializeAs, FromInto, serde_as};
use shardtree::{
    LocatedPrunableTree,
    store::{TreeState, caching::CachingShardStore},
};
use sneed::{
    DatabaseDup, DatabaseUnique, DbError, Env, RoTxn, RwTxn, UnitKey, db, env,
};
use thiserror::Error;
use transitive::Transitive;

use crate::types::{
    BlockHash,
    orchard::util::{
        Borrowed, Owned, Ownership, SerializeBorrow, SerializeWithRef, With,
    },
};

/// Serde wrapper for `incrementalmerkletree::Address`.
/// Levels are serialized as their bitwise complement, so that they are ordered
/// highest-to-lowest in a DB.
/// Indexes are serialized in big-endian, so that they are ordered
/// lowest-to-highest in a DB.
/// This ensures that the last element in the DB is the tip, since it is the
/// lowest level, at the highest index.
#[derive(Clone, Copy, Debug, TransparentWrapper)]
#[repr(transparent)]
struct Address(incrementalmerkletree::Address);

impl<'de> Deserialize<'de> for Address {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let (level_inv, idx_be) = <(u8, [u8; 8])>::deserialize(deserializer)?;
        let level = incrementalmerkletree::Level::new(!level_inv);
        let idx = u64::from_be_bytes(idx_be);
        let addr = incrementalmerkletree::Address::from_parts(level, idx);
        Ok(Self(addr))
    }
}

impl Serialize for Address {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let level: u8 = self.0.level().into();
        let level_inv: u8 = !level;
        let idx_be: [u8; 8] = self.0.index().to_be_bytes();
        (level_inv, idx_be).serialize(serializer)
    }
}

/// Serde wrapper for `shardtree::RetentionFlags`
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
struct RetentionFlags(shardtree::RetentionFlags);

impl<'de> Deserialize<'de> for RetentionFlags {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let bits = u8::deserialize(deserializer)?;
        Ok(Self(shardtree::RetentionFlags::from_bits_retain(bits)))
    }
}

impl Serialize for RetentionFlags {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.0.bits().serialize(serializer)
    }
}

/// Serde representation for `Node`
#[serde_as]
#[derive(Debug, Deserialize, Serialize)]
#[serde(bound(deserialize = "
    O::Value<Option<Arc<MerkleHashOrchard>>>: Deserialize<'de>,
    O::Value<Tree>: Deserialize<'de>,
"))]
enum NodeRepr<'a, O>
where
    O: Ownership<'a>,
{
    Parent {
        ann: O::Value<Option<Arc<MerkleHashOrchard>>>,
        left: O::Value<Tree>,
        right: O::Value<Tree>,
    },
    Leaf {
        value: (MerkleHashOrchard, RetentionFlags),
    },
    Nil,
}

/// Serde wrapper for `Tree`
#[derive(Debug, TransparentWrapper)]
#[repr(transparent)]
struct Tree(
    shardtree::Tree<
        Option<Arc<MerkleHashOrchard>>,
        (MerkleHashOrchard, shardtree::RetentionFlags),
    >,
);

impl From<NodeRepr<'_, Owned>> for Tree {
    fn from(node: NodeRepr<'_, Owned>) -> Self {
        match node {
            NodeRepr::Parent { ann, left, right } => {
                Self(shardtree::Tree::parent(
                    ann,
                    Tree::peel(left),
                    Tree::peel(right),
                ))
            }
            NodeRepr::Leaf {
                value: (hash, retention_flags),
            } => Self(shardtree::Tree::leaf((hash, retention_flags.0))),
            NodeRepr::Nil => Self(shardtree::Tree::empty()),
        }
    }
}

impl<'de> Deserialize<'de> for Tree {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let repr = NodeRepr::deserialize(deserializer)?;
        Ok(Self::from(repr))
    }
}

impl<'a> From<&'a Tree> for NodeRepr<'a, Borrowed<'a>> {
    fn from(tree: &'a Tree) -> Self {
        match <shardtree::Tree<_, _> as std::ops::Deref>::deref(&tree.0) {
            shardtree::Node::Parent { ann, left, right } => NodeRepr::Parent {
                ann,
                left: Tree::wrap_ref(left),
                right: Tree::wrap_ref(right),
            },
            shardtree::Node::Leaf {
                value: (hash, retention_flags),
            } => NodeRepr::Leaf {
                value: (*hash, RetentionFlags(*retention_flags)),
            },
            shardtree::Node::Nil => NodeRepr::Nil,
        }
    }
}

impl Serialize for Tree {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let repr = NodeRepr::from(self);
        repr.serialize(serializer)
    }
}

/// Serde wrapper for [`Position`].
/// Uses big-endian encoding, so that positions are sorted when used as DB keys
#[derive(Debug, TransparentWrapper)]
#[repr(transparent)]
pub struct PositionWrapper(pub incrementalmerkletree::Position);

impl<'de> Deserialize<'de> for PositionWrapper {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let pos_be = <[u8; 8]>::deserialize(deserializer)?;
        let pos = u64::from_be_bytes(pos_be);
        Ok(Self(pos.into()))
    }
}

impl Serialize for PositionWrapper {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let pos: u64 = self.0.into();
        let pos_be: [u8; 8] = pos.to_be_bytes();
        pos_be.serialize(serializer)
    }
}

/// Serde representation for [`TreeState`]
#[derive(Debug, Deserialize, Serialize)]
enum TreeStateRepr {
    Empty,
    AtPosition(u64),
}

impl From<TreeState> for TreeStateRepr {
    fn from(tree_state: TreeState) -> Self {
        match tree_state {
            TreeState::Empty => Self::Empty,
            TreeState::AtPosition(pos) => Self::AtPosition(pos.into()),
        }
    }
}

impl From<TreeStateRepr> for TreeState {
    fn from(repr: TreeStateRepr) -> Self {
        match repr {
            TreeStateRepr::Empty => Self::Empty,
            TreeStateRepr::AtPosition(pos) => Self::AtPosition(pos.into()),
        }
    }
}

/// Serde representation for [`Checkpoint`]
#[serde_as]
#[derive(Debug, Deserialize, Serialize)]
#[serde(bound(deserialize = "
    BTreeSet<FromInto<u64>>: DeserializeAs<'de, O::Value<BTreeSet<Position>>>
"))]
struct CheckpointRepr<'a, O>
where
    O: Ownership<'a>,
{
    tree_state: TreeStateRepr,
    #[serde_as(as = "With<
        BTreeSet<FromInto<u64>>,
        SerializeBorrow<
            BTreeSet<Position>,
            SerializeWithRef<BTreeSet<FromInto<u64>>>
        >
    >")]
    marks_removed: O::Value<BTreeSet<Position>>,
}

/// Serde wrapper for [`shardtree::store::Checkpoint`]
#[repr(transparent)]
struct Checkpoint(shardtree::store::Checkpoint);

impl From<CheckpointRepr<'_, Owned>> for Checkpoint {
    fn from(repr: CheckpointRepr<'_, Owned>) -> Self {
        let CheckpointRepr {
            tree_state,
            marks_removed,
        } = repr;
        Self(shardtree::store::Checkpoint::from_parts(
            tree_state.into(),
            marks_removed,
        ))
    }
}

impl<'de> Deserialize<'de> for Checkpoint {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let repr = CheckpointRepr::deserialize(deserializer)?;
        Ok(repr.into())
    }
}

impl<'a> From<&'a Checkpoint> for CheckpointRepr<'a, Borrowed<'a>> {
    fn from(checkpoint: &'a Checkpoint) -> Self {
        Self {
            tree_state: checkpoint.0.tree_state().into(),
            marks_removed: checkpoint.0.marks_removed(),
        }
    }
}

impl Serialize for Checkpoint {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let repr = CheckpointRepr::from(self);
        repr.serialize(serializer)
    }
}

#[allow(clippy::duplicated_attributes)]
#[derive(Debug, Error, Transitive)]
#[transitive(from(db::error::Put, DbError), from(db::error::TryGet, DbError))]
pub enum CreateShardTreeDbError {
    #[error(transparent)]
    CreateDb(#[from] env::error::CreateDb),
    #[error(transparent)]
    Db(#[from] DbError),
}

/// Store a [`ShardTree`] using LMDB
#[derive(Debug, Educe)]
#[educe(Clone(bound()))]
pub struct ShardTreeDb<Tag> {
    /// MUST always contain a value.
    cap: DatabaseUnique<UnitKey, SerdeBincode<Tree>, Tag>,
    /// Sequential order for tips, relative to a position.
    /// Higher sequence IDs *at a particular position* are more recent.
    /// Sequence IDs are not meaningfully comparable at different positions.
    /// Sequence IDs are not necessarily sequential at any given position, but
    /// should be monotonically increasing ie. there may be missing values.
    checkpoint_seq_to_tip:
        DatabaseUnique<U32<BE>, SerdeBincode<Option<BlockHash>>, Tag>,
    /// Maps each tip to the checkpoint and seq ID
    checkpoints: DatabaseUnique<
        SerdeBincode<Option<BlockHash>>,
        SerdeBincode<(Checkpoint, u32)>,
        Tag,
    >,
    /// Each position may correspond to multiple checkpoints.
    position_to_checkpoint_seq:
        DatabaseDup<SerdeBincode<Option<PositionWrapper>>, U32<BE>, Tag>,
    shards: DatabaseUnique<SerdeBincode<Address>, SerdeBincode<Tree>, Tag>,
}

impl<Tag> ShardTreeDb<Tag> {
    pub const NUM_DBS: u32 = 5;

    /// Creates/Opens a DB, does not commit the RwTxn.
    /// An optional prefix can be set for the DB names.
    /// If set, all DB names will have the prefix `PREFIX_` where `PREFIX` is
    /// the value of `db_name_prefix`.
    pub fn new(
        env: &Env<Tag>,
        rwtxn: &mut RwTxn<Tag>,
        db_name_prefix: Option<&str>,
    ) -> Result<Self, CreateShardTreeDbError> {
        let db_name = |db_name: &str| {
            if let Some(db_name_prefix) = db_name_prefix {
                format!("{db_name_prefix}_{db_name}")
            } else {
                db_name.to_owned()
            }
        };
        let cap = DatabaseUnique::create(env, rwtxn, &db_name("cap"))?;
        if !cap.contains_key(rwtxn, &())? {
            cap.put(rwtxn, &(), &Tree(shardtree::Tree::empty()))?;
        }
        let checkpoint_seq_to_tip = DatabaseUnique::create(
            env,
            rwtxn,
            &db_name("checkpoint_seq_to_tip"),
        )?;
        let checkpoints =
            DatabaseUnique::create(env, rwtxn, &db_name("checkpoints"))?;
        let position_to_checkpoint_seq = DatabaseDup::create(
            env,
            rwtxn,
            &db_name("position_to_checkpoint_seq"),
        )?;
        if !checkpoints.contains_key(rwtxn, &None)? {
            checkpoints.put(
                rwtxn,
                &None,
                &(Checkpoint(shardtree::store::Checkpoint::tree_empty()), 0),
            )?;
            checkpoint_seq_to_tip.put(rwtxn, &0, &None)?;
            position_to_checkpoint_seq.put(rwtxn, &None, &0)?;
        }
        let shards = DatabaseUnique::create(env, rwtxn, &db_name("shards"))?;
        Ok(Self {
            cap,
            checkpoint_seq_to_tip,
            checkpoints,
            position_to_checkpoint_seq,
            shards,
        })
    }

    /// Returns the checkpoint ID corresponding to a tip
    pub fn try_get_checkpoint_id(
        &self,
        rotxn: &RoTxn<Tag>,
        tip: Option<BlockHash>,
    ) -> Result<Option<CheckpointId>, db::error::TryGet> {
        if let Some((checkpoint, seq)) =
            self.checkpoints.try_get(rotxn, &tip)?
        {
            Ok(Some(CheckpointId {
                pos: checkpoint.0.position(),
                seq,
                tip,
            }))
        } else {
            Ok(None)
        }
    }

    /// Returns the checkpoint ID corresponding to a tip
    pub fn get_checkpoint_id(
        &self,
        rotxn: &RoTxn<Tag>,
        tip: Option<BlockHash>,
    ) -> Result<CheckpointId, db::error::Get> {
        let (checkpoint, seq) = self.checkpoints.get(rotxn, &tip)?;
        Ok(CheckpointId {
            pos: checkpoint.0.position(),
            seq,
            tip,
        })
    }

    /// Returns the highest existing checkpoint seq
    fn max_checkpoint_seq(
        &self,
        rotxn: &RoTxn<Tag>,
    ) -> Result<Option<u32>, db::error::Last> {
        let res = self
            .checkpoint_seq_to_tip
            .lazy_decode()
            .last(rotxn)?
            .map(|(seq, _)| seq);
        Ok(res)
    }

    pub fn next_checkpoint_seq(
        &self,
        rotxn: &RoTxn<Tag>,
    ) -> Result<u32, db::error::Last> {
        match self.max_checkpoint_seq(rotxn)? {
            Some(max_seq) => Ok(max_seq + 1),
            None => Ok(0),
        }
    }
}

#[allow(clippy::duplicated_attributes)]
#[derive(Debug, Error, Transitive)]
#[transitive(
    from(db::error::Delete, DbError),
    from(db::error::First, DbError),
    from(db::error::Get, DbError),
    from(db::error::Inconsistent, DbError),
    from(db::error::IterInit, DbError),
    from(db::error::IterItem, DbError),
    from(db::error::Last, DbError),
    from(db::error::Len, DbError),
    from(db::error::Put, DbError),
    from(db::error::TryGet, DbError)
)]
pub enum StoreError {
    #[error(transparent)]
    Db(#[from] DbError),
    #[error(
        "Invalid checkpoint ID: position ({:?}) must match checkpoint position ({:?})",
        .pos,
        .checkpoint_pos
    )]
    InvalidCheckpointIdPosition {
        pos: Option<Position>,
        checkpoint_pos: Option<Position>,
    },
    #[error(
        "Invalid checkpoint ID: seq ({:?}) must match checkpoint seq ({:?})",
        .seq,
        .checkpoint_seq,
    )]
    InvalidCheckpointIdSeq { seq: u32, checkpoint_seq: u32 },
    #[error("No DB write transaction available")]
    NoRwTxn,
    #[error("No DB transaction available")]
    NoTxn,
    #[error(
        "Failed to construct LocatedPrunableTree from parts (`{address:?}`)"
    )]
    LptFromParts {
        address: incrementalmerkletree::Address,
    },
    #[error("Cannot modify checkpoint position")]
    UpdateCheckpointPosition,
}

pub mod db_txn {
    use sneed::{RoTxn, RwTxn, rotxn, rwtxn};

    use crate::types::orchard::shardtree_db::StoreError;

    #[derive(Debug, thiserror::Error)]
    pub enum CommitError {
        #[error(transparent)]
        Ro(#[from] rotxn::error::Commit),
        #[error(transparent)]
        Rw(#[from] rwtxn::error::Commit),
    }

    /// Either RoTxn or RwTxn
    pub enum DbTxn<'a, Tag> {
        Ro(RoTxn<'a, Tag>),
        Rw(RwTxn<'a, Tag>),
    }

    impl<'a, Tag> DbTxn<'a, Tag> {
        /// Commits the txn
        pub fn commit(self) -> Result<(), CommitError> {
            match self {
                Self::Ro(ro) => ro.commit()?,
                Self::Rw(rw) => rw.commit()?,
            }
            Ok(())
        }

        pub(in crate::types::orchard::shardtree_db) fn rwtxn(
            &mut self,
        ) -> Result<&mut RwTxn<'a, Tag>, StoreError> {
            match self {
                Self::Ro(_) => Err(StoreError::NoRwTxn),
                Self::Rw(rwtxn) => Ok(rwtxn),
            }
        }
    }

    impl<'a, Tag> AsRef<RoTxn<'a, Tag>> for DbTxn<'a, Tag> {
        fn as_ref(&self) -> &RoTxn<'a, Tag> {
            match self {
                Self::Ro(rotxn) => rotxn,
                Self::Rw(rwtxn) => rwtxn,
            }
        }
    }
}
pub use db_txn::DbTxn;

/// Checkpoint IDs are horribly broken due to
/// [`shardtree::store::ShardStore::update_checkpoint_with`].
/// Checkpoint IDs are required to be ordered by position, however
/// [`shardtree::store::ShardStore::update_checkpoint_with`] can potentially
/// change the position of a checkpoint.
///
/// Our implementation of
/// [`shardtree::store::ShardStore::update_checkpoint_with`] guards against
/// this, but since it is used through
/// [`shardtree::store::caching::CachingShardStore`], it is still possible for
/// this invariant to break.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct CheckpointId {
    pub pos: Option<Position>,
    pub seq: u32,
    pub tip: Option<BlockHash>,
}

/// Used to implement [`shardtree::store::ShardStore`].
///
/// Does not enforce tree height invariants, so should only be used via
/// [`shardtree::store::caching::CachingShardStore`].
///
/// The `Ord` impl for checkpoint IDs does not behave as expected
/// (nor can it - [`shardtree::store::ShardStore::update_checkpoint_with`]
/// can mutate checkpoints, so checkpoint IDs cannot correspond to checkpoint
/// values, and therefore to checkpoint positions). Avoid methods with `Ord`
/// bounds as much as possible, they are almost guaranteed to be broken.
///
/// The `txn` field must be set in order to successfully load,
/// and the `txn` field must be set to a `RwTxn` in order to successfully
/// store.
#[must_use]
pub struct ShardTreeStore<'a, Tag> {
    pub txn: Weak<RwLock<Option<DbTxn<'a, Tag>>>>,
    pub db: ShardTreeDb<Tag>,
}

impl<'a, Tag> ShardTreeStore<'a, Tag> {
    fn read_with<F, T>(&self, f: F) -> Result<T, StoreError>
    where
        F: FnOnce(&RoTxn<'a, Tag>) -> Result<T, StoreError>,
    {
        match self.txn.upgrade() {
            Some(txn) => {
                f(txn.read().as_ref().ok_or(StoreError::NoTxn)?.as_ref())
            }
            None => Err(StoreError::NoTxn),
        }
    }

    fn write_with<F, T>(&self, f: F) -> Result<T, StoreError>
    where
        F: FnOnce(&mut RwTxn<'a, Tag>) -> Result<T, StoreError>,
    {
        match self.txn.upgrade() {
            Some(txn) => {
                f(txn.write().as_mut().ok_or(StoreError::NoTxn)?.rwtxn()?)
            }
            None => Err(StoreError::NoTxn),
        }
    }
}

impl<'a, Tag> shardtree::store::ShardStore for ShardTreeStore<'a, Tag> {
    type H = MerkleHashOrchard;

    type CheckpointId = CheckpointId;

    type Error = StoreError;

    fn get_shard(
        &self,
        shard_root: incrementalmerkletree::Address,
    ) -> Result<Option<LocatedPrunableTree<Self::H>>, Self::Error> {
        let tree = self.read_with(|rotxn| {
            let tree = self.db.shards.try_get(rotxn, &Address(shard_root))?;
            Ok(tree)
        })?;
        let res = tree
            .map(|tree| {
                LocatedPrunableTree::from_parts(shard_root, tree.0)
                    .map_err(|address| StoreError::LptFromParts { address })
            })
            .transpose()?;
        Ok(res)
    }

    fn last_shard(
        &self,
    ) -> Result<Option<LocatedPrunableTree<Self::H>>, Self::Error> {
        let last = self.read_with(|rotxn| {
            let last = self.db.shards.last(rotxn)?;
            Ok(last)
        })?;
        let res = last
            .map(|(addr, tree)| {
                LocatedPrunableTree::from_parts(addr.0, tree.0)
                    .map_err(|address| StoreError::LptFromParts { address })
            })
            .transpose()?;
        Ok(res)
    }

    fn put_shard(
        &mut self,
        subtree: LocatedPrunableTree<Self::H>,
    ) -> Result<(), Self::Error> {
        let addr = subtree.root_addr();
        let tree = Tree::wrap_ref(subtree.root());
        let () = self.write_with(|rwtxn| {
            let () = self.db.shards.put(rwtxn, &Address(addr), tree)?;
            Ok(())
        })?;
        Ok(())
    }

    fn get_shard_roots(
        &self,
    ) -> Result<Vec<incrementalmerkletree::Address>, Self::Error> {
        self.read_with(|rotxn| {
            let res = self
                .db
                .shards
                .iter_keys(rotxn)?
                .map(|addr| Ok(addr.0))
                .collect()?;
            Ok(res)
        })
    }

    fn truncate_shards(&mut self, shard_index: u64) -> Result<(), Self::Error> {
        self.write_with(|rwtxn| {
            let addrs: Vec<_> = self
                .db
                .shards
                .iter_keys(rwtxn)?
                .map(|addr| Ok(addr.0))
                .collect()?;
            for addr in addrs {
                if addr.index() >= shard_index {
                    self.db.shards.delete(rwtxn, &Address(addr))?;
                }
            }
            Ok(())
        })
    }

    fn get_cap(&self) -> Result<shardtree::PrunableTree<Self::H>, Self::Error> {
        self.read_with(|rotxn| {
            let res = self.db.cap.get(rotxn, &())?;
            Ok(res.0)
        })
    }

    fn put_cap(
        &mut self,
        cap: shardtree::PrunableTree<Self::H>,
    ) -> Result<(), Self::Error> {
        self.write_with(|rwtxn| {
            let () = self.db.cap.put(rwtxn, &(), &Tree(cap))?;
            Ok(())
        })
    }

    fn min_checkpoint_id(
        &self,
    ) -> Result<Option<Self::CheckpointId>, Self::Error> {
        self.read_with(|rotxn| 'res: {
            let Some((pos, seq)) =
                self.db.position_to_checkpoint_seq.first(rotxn)?
            else {
                break 'res Ok(None);
            };
            let tip = self
                .db
                .checkpoint_seq_to_tip
                .try_get(rotxn, &seq)?
                .ok_or_else(|| {
                    use db::error::inconsistent::{ByKey, ByValue, Error, Xor};
                    Error::from(Xor::new(
                        &seq,
                        ByValue(&*self.db.position_to_checkpoint_seq),
                        ByKey(&*self.db.checkpoint_seq_to_tip),
                    ))
                })?;
            Ok(Some(CheckpointId {
                pos: pos.map(PositionWrapper::peel),
                seq,
                tip,
            }))
        })
    }

    fn max_checkpoint_id(
        &self,
    ) -> Result<Option<Self::CheckpointId>, Self::Error> {
        self.read_with(|rotxn| 'res: {
            let Some((pos, seq)) =
                self.db.position_to_checkpoint_seq.last(rotxn)?
            else {
                break 'res Ok(None);
            };
            let tip = self
                .db
                .checkpoint_seq_to_tip
                .try_get(rotxn, &seq)?
                .ok_or_else(|| {
                    use db::error::inconsistent::{ByKey, ByValue, Error, Xor};
                    Error::from(Xor::new(
                        &seq,
                        ByValue(&*self.db.position_to_checkpoint_seq),
                        ByKey(&*self.db.checkpoint_seq_to_tip),
                    ))
                })?;
            Ok(Some(CheckpointId {
                pos: pos.map(PositionWrapper::peel),
                seq,
                tip,
            }))
        })
    }

    fn add_checkpoint(
        &mut self,
        checkpoint_id: Self::CheckpointId,
        checkpoint: shardtree::store::Checkpoint,
    ) -> Result<(), Self::Error> {
        if checkpoint.position() != checkpoint_id.pos {
            return Err(Self::Error::InvalidCheckpointIdPosition {
                pos: checkpoint_id.pos,
                checkpoint_pos: checkpoint.position(),
            });
        }
        self.write_with(|rwtxn| 'res: {
            // If there is already a checkpoint with the specified ID,
            // retain it if the checkpoint is unchanged. Otherwise, delete it.
            if let Some((original_checkpoint, original_seq)) =
                self.db.checkpoints.try_get(rwtxn, &checkpoint_id.tip)?
            {
                if original_checkpoint.0.tree_state() == checkpoint.tree_state()
                    && original_checkpoint.0.marks_removed()
                        == checkpoint.marks_removed()
                {
                    break 'res Ok(());
                } else {
                    let original_position = original_checkpoint.0.position();
                    if original_position != checkpoint_id.pos {
                        return Err(Self::Error::InvalidCheckpointIdPosition {
                            pos: checkpoint_id.pos,
                            checkpoint_pos: original_position,
                        });
                    }
                    if original_seq != checkpoint_id.seq {
                        return Err(Self::Error::InvalidCheckpointIdSeq {
                            seq: checkpoint_id.seq,
                            checkpoint_seq: original_seq,
                        });
                    }
                    self.db.checkpoints.delete(rwtxn, &checkpoint_id.tip)?;
                    self.db
                        .checkpoint_seq_to_tip
                        .delete(rwtxn, &original_seq)?;
                    self.db.position_to_checkpoint_seq.delete_one(
                        rwtxn,
                        &original_position.map(PositionWrapper),
                        &original_seq,
                    )?;
                }
            }
            self.db.checkpoints.put(
                rwtxn,
                &checkpoint_id.tip,
                &(Checkpoint(checkpoint), checkpoint_id.seq),
            )?;
            self.db.checkpoint_seq_to_tip.put(
                rwtxn,
                &checkpoint_id.seq,
                &checkpoint_id.tip,
            )?;
            self.db.position_to_checkpoint_seq.put(
                rwtxn,
                &checkpoint_id.pos.map(PositionWrapper),
                &checkpoint_id.seq,
            )?;
            Ok(())
        })
    }

    fn checkpoint_count(&self) -> Result<usize, Self::Error> {
        self.read_with(|rotxn| {
            let res = self.db.checkpoints.len(rotxn)?;
            Ok(res as usize)
        })
    }

    fn get_checkpoint_at_depth(
        &self,
        checkpoint_depth: usize,
    ) -> Result<
        Option<(Self::CheckpointId, shardtree::store::Checkpoint)>,
        Self::Error,
    > {
        self.read_with(|rotxn| 'res: {
            use db::error::inconsistent::{ByKey, ByValue, Error, Xor};
            let Some((pos, seq)) = self
                .db
                .position_to_checkpoint_seq
                .rev_iter_through_duplicate_values(rotxn)?
                .nth(checkpoint_depth)?
            else {
                break 'res Ok(None);
            };
            let tip = self
                .db
                .checkpoint_seq_to_tip
                .try_get(rotxn, &seq)?
                .ok_or_else(|| {
                    Error::from(Xor::new(
                        &seq,
                        ByValue(&*self.db.position_to_checkpoint_seq),
                        ByKey(&*self.db.checkpoint_seq_to_tip),
                    ))
                })?;
            let (checkpoint, _) =
                self.db.checkpoints.try_get(rotxn, &tip)?.ok_or_else(|| {
                    Error::from(Xor::new(
                        &tip,
                        ByValue(&*self.db.checkpoint_seq_to_tip),
                        ByKey(&*self.db.checkpoints),
                    ))
                })?;
            let checkpoint_id = CheckpointId {
                pos: pos.map(PositionWrapper::peel),
                seq,
                tip,
            };
            Ok(Some((checkpoint_id, checkpoint.0)))
        })
    }

    fn get_checkpoint(
        &self,
        checkpoint_id: &Self::CheckpointId,
    ) -> Result<Option<shardtree::store::Checkpoint>, Self::Error> {
        self.read_with(|rotxn| {
            let Some((checkpoint, seq)) =
                self.db.checkpoints.try_get(rotxn, &checkpoint_id.tip)?
            else {
                return Ok(None);
            };
            if seq != checkpoint_id.seq {
                return Err(Self::Error::InvalidCheckpointIdSeq {
                    seq: checkpoint_id.seq,
                    checkpoint_seq: seq,
                });
            }
            if checkpoint.0.position() != checkpoint_id.pos {
                return Err(Self::Error::InvalidCheckpointIdPosition {
                    pos: checkpoint_id.pos,
                    checkpoint_pos: checkpoint.0.position(),
                });
            }
            Ok(Some(checkpoint.0))
        })
    }

    fn with_checkpoints<F>(
        &mut self,
        limit: usize,
        mut callback: F,
    ) -> Result<(), Self::Error>
    where
        F: FnMut(
            &Self::CheckpointId,
            &shardtree::store::Checkpoint,
        ) -> Result<(), Self::Error>,
    {
        self.read_with(|rotxn| {
            self.db
                .position_to_checkpoint_seq
                .iter_through_duplicate_values(rotxn)?
                .take(limit)
                .map_err(Self::Error::from)
                .for_each(|(pos, seq)| {
                    use db::error::inconsistent::{ByKey, ByValue, Error, Xor};
                    let tip = self
                        .db
                        .checkpoint_seq_to_tip
                        .try_get(rotxn, &seq)?
                        .ok_or_else(|| {
                            Error::from(Xor::new(
                                &seq,
                                ByValue(&*self.db.position_to_checkpoint_seq),
                                ByKey(&*self.db.checkpoint_seq_to_tip),
                            ))
                        })?;
                    let (checkpoint, _) =
                        self.db.checkpoints.try_get(rotxn, &tip)?.ok_or_else(
                            || {
                                Error::from(Xor::new(
                                    &tip,
                                    ByValue(&*self.db.checkpoint_seq_to_tip),
                                    ByKey(&*self.db.checkpoints),
                                ))
                            },
                        )?;
                    let checkpoint_id = CheckpointId {
                        pos: pos.map(PositionWrapper::peel),
                        seq,
                        tip,
                    };
                    let () = callback(&checkpoint_id, &checkpoint.0)?;
                    Ok::<_, Self::Error>(())
                })
        })
    }

    fn for_each_checkpoint<F>(
        &self,
        limit: usize,
        mut callback: F,
    ) -> Result<(), Self::Error>
    where
        F: FnMut(
            &Self::CheckpointId,
            &shardtree::store::Checkpoint,
        ) -> Result<(), Self::Error>,
    {
        self.read_with(|rotxn| {
            self.db
                .position_to_checkpoint_seq
                .iter_through_duplicate_values(rotxn)?
                .take(limit)
                .map_err(Self::Error::from)
                .for_each(|(pos, seq)| {
                    use db::error::inconsistent::{ByKey, ByValue, Error, Xor};
                    let tip = self
                        .db
                        .checkpoint_seq_to_tip
                        .try_get(rotxn, &seq)?
                        .ok_or_else(|| {
                            Error::from(Xor::new(
                                &seq,
                                ByValue(&*self.db.position_to_checkpoint_seq),
                                ByKey(&*self.db.checkpoint_seq_to_tip),
                            ))
                        })?;
                    let (checkpoint, _) =
                        self.db.checkpoints.try_get(rotxn, &tip)?.ok_or_else(
                            || {
                                Error::from(Xor::new(
                                    &tip,
                                    ByValue(&*self.db.checkpoint_seq_to_tip),
                                    ByKey(&*self.db.checkpoints),
                                ))
                            },
                        )?;
                    let checkpoint_id = CheckpointId {
                        pos: pos.map(PositionWrapper::peel),
                        seq,
                        tip,
                    };
                    let () = callback(&checkpoint_id, &checkpoint.0)?;
                    Ok::<_, Self::Error>(())
                })
        })
    }

    fn update_checkpoint_with<F>(
        &mut self,
        checkpoint_id: &Self::CheckpointId,
        update: F,
    ) -> Result<bool, Self::Error>
    where
        F: Fn(&mut shardtree::store::Checkpoint) -> Result<(), Self::Error>,
    {
        self.write_with(|rwtxn| {
            let Some((mut checkpoint, seq)) =
                self.db.checkpoints.try_get(rwtxn, &checkpoint_id.tip)?
            else {
                return Ok(false);
            };
            if seq != checkpoint_id.seq {
                return Err(Self::Error::InvalidCheckpointIdSeq {
                    seq: checkpoint_id.seq,
                    checkpoint_seq: seq,
                });
            }
            let original_position = checkpoint.0.position();
            if original_position != checkpoint_id.pos {
                return Err(Self::Error::InvalidCheckpointIdPosition {
                    pos: checkpoint_id.pos,
                    checkpoint_pos: original_position,
                });
            }
            let () = update(&mut checkpoint.0)?;
            let new_position = checkpoint.0.position();
            if new_position != original_position {
                return Err(Self::Error::UpdateCheckpointPosition);
            };
            self.db.checkpoints.put(
                rwtxn,
                &checkpoint_id.tip,
                &(checkpoint, seq),
            )?;
            Ok(true)
        })
    }

    fn remove_checkpoint(
        &mut self,
        checkpoint_id: &Self::CheckpointId,
    ) -> Result<(), Self::Error> {
        self.write_with(|rwtxn| {
            let Some((checkpoint, seq)) =
                self.db.checkpoints.try_get(rwtxn, &checkpoint_id.tip)?
            else {
                return Ok(());
            };
            if seq != checkpoint_id.seq {
                return Err(Self::Error::InvalidCheckpointIdSeq {
                    seq: checkpoint_id.seq,
                    checkpoint_seq: seq,
                });
            }
            let pos = checkpoint.0.position();
            if pos != checkpoint_id.pos {
                return Err(Self::Error::InvalidCheckpointIdPosition {
                    pos: checkpoint_id.pos,
                    checkpoint_pos: pos,
                });
            }
            self.db.checkpoints.delete(rwtxn, &checkpoint_id.tip)?;
            self.db.position_to_checkpoint_seq.delete_one(
                rwtxn,
                &pos.map(PositionWrapper),
                &seq,
            )?;
            self.db.checkpoint_seq_to_tip.delete(rwtxn, &seq)?;
            Ok(())
        })
    }

    fn truncate_checkpoints_retaining(
        &mut self,
        checkpoint_id: &Self::CheckpointId,
    ) -> Result<(), Self::Error> {
        self.write_with(|rwtxn| {
            let (checkpoint, checkpoint_seq) =
                self.db.checkpoints.get(rwtxn, &checkpoint_id.tip)?;
            let checkpoint_pos = checkpoint.0.position();
            if checkpoint_id.pos != checkpoint_pos {
                return Err(Self::Error::InvalidCheckpointIdPosition {
                    pos: checkpoint_id.pos,
                    checkpoint_pos,
                });
            }
            if checkpoint_id.seq != checkpoint_seq {
                return Err(Self::Error::InvalidCheckpointIdSeq {
                    seq: checkpoint_id.seq,
                    checkpoint_seq,
                });
            }
            let checkpoints_to_delete: Vec<_> = self
                .db
                .position_to_checkpoint_seq
                .rev_iter_through_duplicate_values(rwtxn)?
                .take_while(|(pos, seq)| {
                    let pos = pos.as_ref().map(|pos| pos.0);
                    Ok(pos > checkpoint_pos
                        || (pos == checkpoint_pos && *seq >= checkpoint_seq))
                })
                .collect()?;
            for (pos, seq) in checkpoints_to_delete {
                self.db
                    .position_to_checkpoint_seq
                    .delete_one(rwtxn, &pos, &seq)?;
                let tip = self
                    .db
                    .checkpoint_seq_to_tip
                    .try_get(rwtxn, &seq)?
                    .ok_or_else(|| {
                    use db::error::inconsistent::{ByKey, ByValue, Error, Xor};
                    Error::from(Xor::new(
                        &seq,
                        ByValue(&*self.db.position_to_checkpoint_seq),
                        ByKey(&*self.db.checkpoint_seq_to_tip),
                    ))
                })?;
                self.db.checkpoint_seq_to_tip.delete(rwtxn, &seq)?;
                self.db.checkpoints.delete(rwtxn, &tip)?;
            }
            Ok(())
        })
    }
}

/// FIXME: This is arbitrary, unsure what it should be set to
const SHARD_HEIGHT: u8 = 2;

pub type ShardTree<'a, Tag> = shardtree::ShardTree<
    CachingShardStore<ShardTreeStore<'a, Tag>>,
    { orchard::NOTE_COMMITMENT_TREE_DEPTH as u8 },
    SHARD_HEIGHT,
>;

pub type ShardTreeError =
    shardtree::error::ShardTreeError<std::convert::Infallible>;

pub fn load_shard_tree<'a, Tag>(
    store: ShardTreeStore<'a, Tag>,
) -> Result<ShardTree<'a, Tag>, StoreError> {
    let caching_store = CachingShardStore::load(store)?;
    let tree = ShardTree::new(caching_store, usize::MAX);
    Ok(tree)
}

/// Store the shard tree to the DB. Returns the DB and the RwTxn.
pub fn store_shard_tree<'a, Tag>(
    tree: ShardTree<'a, Tag>,
) -> Result<ShardTreeStore<'a, Tag>, StoreError> {
    tree.into_store().flush()
}
