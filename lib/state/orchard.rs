//! Orchard state

use heed::types::{SerdeBincode, Unit};
use sneed::{DatabaseUnique, RoDatabaseUnique, RwTxn, UnitKey};

use crate::{
    state::{self, error::Orchard as Error},
    types::{
        BlockHash, VERSION, Version,
        orchard::{Anchor, Frontier, MerkleHashOrchard, Nullifier},
    },
};

#[derive(Clone)]
pub struct Orchard {
    /// Maps block hashes to historical roots.
    /// A value for `None` MUST always exist.
    block_hash_to_root:
        DatabaseUnique<SerdeBincode<Option<BlockHash>>, SerdeBincode<Anchor>>,
    // Should always exist
    frontier: DatabaseUnique<UnitKey, SerdeBincode<Frontier>>,
    /// Maps historical roots to block hashes.
    /// At least one value must always exist, which maps to `None`.
    historical_roots:
        DatabaseUnique<SerdeBincode<Anchor>, SerdeBincode<Option<BlockHash>>>,
    nullifiers: DatabaseUnique<SerdeBincode<Nullifier>, Unit>,
    /// Version number for this DB
    version: DatabaseUnique<UnitKey, SerdeBincode<Version>>,
}

impl Orchard {
    pub const NUM_DBS: u32 = 5;

    pub fn new(
        env: &sneed::Env,
        rwtxn: &mut RwTxn,
    ) -> Result<Self, state::Error> {
        let block_hash_to_root =
            DatabaseUnique::create(env, rwtxn, "orchard_block_hash_to_root")?;
        let frontier = DatabaseUnique::create(env, rwtxn, "orchard_frontier")?;
        let historical_roots =
            DatabaseUnique::create(env, rwtxn, "orchard_historical_roots")?;
        let nullifiers =
            DatabaseUnique::create(env, rwtxn, "orchard_nullifiers")?;
        let version =
            DatabaseUnique::create(env, rwtxn, "state_orchard_version")?;
        let res = Self {
            block_hash_to_root,
            frontier,
            historical_roots,
            nullifiers,
            version,
        };
        if !res.frontier.contains_key(rwtxn, &())? {
            res.frontier.put(rwtxn, &(), &Frontier::empty())?;
        }
        if res.historical_roots.len(rwtxn)? == 0 {
            let empty_root = Frontier::empty().root().into();
            res.block_hash_to_root.put(rwtxn, &None, &empty_root)?;
            res.historical_roots.put(rwtxn, &empty_root, &None)?;
        }
        assert_ne!(res.block_hash_to_root.len(rwtxn)?, 0);
        if !res.version.contains_key(rwtxn, &())? {
            res.version.put(rwtxn, &(), &*VERSION)?;
        }
        Ok(res)
    }

    pub fn frontier(
        &self,
    ) -> &RoDatabaseUnique<UnitKey, SerdeBincode<Frontier>> {
        &self.frontier
    }

    pub fn historical_roots(
        &self,
    ) -> &RoDatabaseUnique<SerdeBincode<Anchor>, SerdeBincode<Option<BlockHash>>>
    {
        &self.historical_roots
    }

    pub fn nullifiers(
        &self,
    ) -> &RoDatabaseUnique<SerdeBincode<Nullifier>, Unit> {
        &self.nullifiers
    }

    pub(in crate::state) fn put_frontier(
        &self,
        rwtxn: &mut RwTxn,
        frontier: &Frontier,
    ) -> Result<(), Error> {
        self.frontier.put(rwtxn, &(), frontier).map_err(Error::from)
    }

    /// Store a historical root, if it is new.
    /// Returns `true` if the root is new, `false` otherwise
    pub(in crate::state) fn put_historical_root(
        &self,
        rwtxn: &mut RwTxn,
        block_hash: BlockHash,
        root: MerkleHashOrchard,
    ) -> Result<bool, Error> {
        let root = Anchor::from(root);
        if self.historical_roots.contains_key(rwtxn, &root)? {
            Ok(false)
        } else {
            self.block_hash_to_root
                .put(rwtxn, &Some(block_hash), &root)?;
            self.historical_roots.put(rwtxn, &root, &Some(block_hash))?;
            Ok(true)
        }
    }

    pub(in crate::state) fn put_nullifier(
        &self,
        rwtxn: &mut RwTxn,
        nullifier: &Nullifier,
    ) -> Result<(), Error> {
        self.nullifiers
            .put(rwtxn, nullifier, &())
            .map_err(Error::from)
    }

    /// Delete the historical root for the specified block hash
    pub(in crate::state) fn delete_historical_root(
        &self,
        rwtxn: &mut RwTxn,
        block_hash: BlockHash,
    ) -> Result<bool, Error> {
        let Some(root) =
            self.block_hash_to_root.try_get(rwtxn, &Some(block_hash))?
        else {
            return Ok(false);
        };
        self.block_hash_to_root.delete(rwtxn, &Some(block_hash))?;
        self.historical_roots.delete(rwtxn, &root)?;
        Ok(true)
    }

    pub(in crate::state) fn delete_nullifier(
        &self,
        rwtxn: &mut RwTxn,
        nullifier: &Nullifier,
    ) -> Result<bool, Error> {
        self.nullifiers
            .delete(rwtxn, nullifier)
            .map_err(Error::from)
    }
}
