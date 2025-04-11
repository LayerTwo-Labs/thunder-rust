//! Storage for a [`shardtree::ShardTree`]

use std::{collections::BTreeSet, rc::Weak, sync::Arc};

use bytemuck::TransparentWrapper;
use educe::Educe;
use fallible_iterator::FallibleIterator;
use heed::types::SerdeBincode;
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
    checkpoints:
        DatabaseUnique<SerdeBincode<BlockHash>, SerdeBincode<Checkpoint>, Tag>,
    /// Each position may correspond to multiple checkpoints.
    position_to_checkpoint_id: DatabaseDup<
        SerdeBincode<Option<PositionWrapper>>,
        SerdeBincode<BlockHash>,
        Tag,
    >,
    shards: DatabaseUnique<SerdeBincode<Address>, SerdeBincode<Tree>, Tag>,
}

impl<Tag> ShardTreeDb<Tag> {
    pub const NUM_DBS: u32 = 4;

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
        let checkpoints =
            DatabaseUnique::create(env, rwtxn, &db_name("checkpoints"))?;
        let position_to_checkpoint_id = DatabaseDup::create(
            env,
            rwtxn,
            &db_name("position_to_checkpoint_id"),
        )?;
        let shards = DatabaseUnique::create(env, rwtxn, &db_name("shards"))?;
        Ok(Self {
            cap,
            checkpoints,
            position_to_checkpoint_id,
            shards,
        })
    }
}

#[allow(clippy::duplicated_attributes)]
#[derive(Debug, Error, Transitive)]
#[transitive(
    from(db::error::Delete, DbError),
    from(db::error::First, DbError),
    from(db::error::Get, DbError),
    from(db::error::IterInit, DbError),
    from(db::error::IterItem, DbError),
    from(db::error::Last, DbError),
    from(db::error::Len, DbError),
    from(db::error::Put, DbError),
    from(db::error::TryGet, DbError)
)]
pub enum StoreError {
    #[error(
        "Multiple checkpoints can exist at any depth, so this operation is undefined"
    )]
    CheckpointAtDepthUndefined,
    #[error(transparent)]
    Db(#[from] DbError),
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

/// Used to implement [`shardtree::store::ShardStore`].
///
/// Does not enforce tree height invariants, so should only be used via
/// [`shardtree::store::caching::CachingShardStore`].
///
/// Since multiple checkpoints can exist at any depth, methods such as
/// `get_checkpoint_at_depth` will return an error.
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

    type CheckpointId = BlockHash;

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
        self.read_with(|rotxn| {
            let res = self
                .db
                .position_to_checkpoint_id
                .first(rotxn)?
                .map(|(_pos, checkpoint)| checkpoint);
            Ok(res)
        })
    }

    fn max_checkpoint_id(
        &self,
    ) -> Result<Option<Self::CheckpointId>, Self::Error> {
        self.read_with(|rotxn| {
            let res = self
                .db
                .position_to_checkpoint_id
                .last(rotxn)?
                .map(|(_pos, checkpoint)| checkpoint);
            Ok(res)
        })
    }

    fn add_checkpoint(
        &mut self,
        checkpoint_id: Self::CheckpointId,
        checkpoint: shardtree::store::Checkpoint,
    ) -> Result<(), Self::Error> {
        let pos = checkpoint.position();
        self.write_with(|rwtxn| {
            self.db.checkpoints.put(
                rwtxn,
                &checkpoint_id,
                &Checkpoint(checkpoint),
            )?;
            self.db.position_to_checkpoint_id.put(
                rwtxn,
                &pos.map(PositionWrapper),
                &checkpoint_id,
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
        _checkpoint_depth: usize,
    ) -> Result<
        Option<(Self::CheckpointId, shardtree::store::Checkpoint)>,
        Self::Error,
    > {
        Err(StoreError::CheckpointAtDepthUndefined)
    }

    fn get_checkpoint(
        &self,
        checkpoint_id: &Self::CheckpointId,
    ) -> Result<Option<shardtree::store::Checkpoint>, Self::Error> {
        self.read_with(|rotxn| {
            let res = self
                .db
                .checkpoints
                .try_get(rotxn, checkpoint_id)?
                .map(|checkpoint| checkpoint.0);
            Ok(res)
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
                .checkpoints
                .iter(rotxn)?
                .take(limit)
                .map_err(Self::Error::from)
                .for_each(|(checkpoint_id, checkpoint)| {
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
                .checkpoints
                .iter(rotxn)?
                .take(limit)
                .map_err(Self::Error::from)
                .for_each(|(checkpoint_id, checkpoint)| {
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
            let Some(mut checkpoint) =
                self.db.checkpoints.try_get(rwtxn, checkpoint_id)?
            else {
                return Ok(false);
            };
            let original_position = checkpoint.0.position();
            let () = update(&mut checkpoint.0)?;
            let new_position = checkpoint.0.position();
            if new_position != original_position {
                self.db.position_to_checkpoint_id.delete_one(
                    rwtxn,
                    &original_position.map(PositionWrapper),
                    checkpoint_id,
                )?;
                self.db.position_to_checkpoint_id.put(
                    rwtxn,
                    &new_position.map(PositionWrapper),
                    checkpoint_id,
                )?;
            }
            self.db.checkpoints.put(rwtxn, checkpoint_id, &checkpoint)?;
            Ok(true)
        })
    }

    fn remove_checkpoint(
        &mut self,
        checkpoint_id: &Self::CheckpointId,
    ) -> Result<(), Self::Error> {
        self.write_with(|rwtxn| {
            let Some(checkpoint) =
                self.db.checkpoints.try_get(rwtxn, checkpoint_id)?
            else {
                return Ok(());
            };
            self.db.checkpoints.delete(rwtxn, checkpoint_id)?;
            let pos = checkpoint.0.position();
            self.db.position_to_checkpoint_id.delete_one(
                rwtxn,
                &pos.map(PositionWrapper),
                checkpoint_id,
            )?;
            Ok(())
        })
    }

    fn truncate_checkpoints_retaining(
        &mut self,
        checkpoint_id: &Self::CheckpointId,
    ) -> Result<(), Self::Error> {
        self.write_with(|rwtxn| {
            let checkpoints_to_delete: Vec<_> = self
                .db
                .checkpoints
                .rev_iter(rwtxn)?
                .take_while(|(cid, _)| Ok(cid >= checkpoint_id))
                .collect()?;
            for (checkpoint_id, checkpoint) in checkpoints_to_delete {
                let pos = checkpoint.0.position();
                self.db.checkpoints.delete(rwtxn, &checkpoint_id)?;
                self.db.position_to_checkpoint_id.delete_one(
                    rwtxn,
                    &pos.map(PositionWrapper),
                    &checkpoint_id,
                )?;
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
