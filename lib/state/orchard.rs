//! Orchard state

use heed::{
    byteorder::BE,
    types::{SerdeBincode, U32, Unit},
};
use sneed::{DatabaseUnique, RoDatabaseUnique, RoTxn, RwTxn, UnitKey};

use crate::{
    state::{self, error::Orchard as Error},
    types::{
        VERSION, Version,
        orchard::{Anchor, Frontier, MerkleHashOrchard, Nullifier},
    },
};

/// Sequential ID for historical roots
type HistoricalRootSeqId = u32;

#[derive(Clone)]
pub struct Orchard {
    // Should always exist
    frontier: DatabaseUnique<UnitKey, SerdeBincode<Frontier>>,
    /// Maps historical roots to sequential IDs.
    historical_roots: DatabaseUnique<SerdeBincode<Anchor>, U32<BE>>,
    /// Maps sequential IDs to historical roots
    historical_root_seq_ids: DatabaseUnique<U32<BE>, SerdeBincode<Anchor>>,
    nullifiers: DatabaseUnique<SerdeBincode<Nullifier>, Unit>,
    /// Version number for this DB
    version: DatabaseUnique<UnitKey, SerdeBincode<Version>>,
}

impl Orchard {
    pub const NUM_DBS: u32 = 5;

    /// Returns the sequential ID of the next historical root
    fn next_historical_root_seq_id(
        &self,
        rotxn: &RoTxn,
    ) -> Result<HistoricalRootSeqId, Error> {
        match self.historical_root_seq_ids.lazy_decode().last(rotxn)? {
            Some((last, _)) => Ok(last + 1),
            None => Ok(0),
        }
    }

    /// Store a historical root, if it is new.
    /// Returns `true` if the root is new, `false` otherwise
    pub(in crate::state) fn put_historical_root(
        &self,
        rwtxn: &mut RwTxn,
        root: MerkleHashOrchard,
    ) -> Result<bool, Error> {
        let root = Anchor::from(root);
        if self.historical_roots.contains_key(rwtxn, &root)? {
            Ok(false)
        } else {
            let seq_id = self.next_historical_root_seq_id(rwtxn)?;
            self.historical_roots.put(rwtxn, &root, &seq_id)?;
            self.historical_root_seq_ids.put(rwtxn, &seq_id, &root)?;
            Ok(true)
        }
    }

    pub fn new(
        env: &sneed::Env,
        rwtxn: &mut RwTxn,
    ) -> Result<Self, state::Error> {
        let frontier = DatabaseUnique::create(env, rwtxn, "orchard_frontier")?;
        let historical_roots =
            DatabaseUnique::create(env, rwtxn, "orchard_historical_roots")?;
        let historical_root_seq_ids = DatabaseUnique::create(
            env,
            rwtxn,
            "orchard_historical_root_seq_ids",
        )?;
        let nullifiers =
            DatabaseUnique::create(env, rwtxn, "orchard_nullifiers")?;
        let version =
            DatabaseUnique::create(env, rwtxn, "state_orchard_version")?;
        let res = Self {
            frontier,
            historical_roots,
            historical_root_seq_ids,
            nullifiers,
            version,
        };
        if !res.frontier.contains_key(rwtxn, &())? {
            res.frontier.put(rwtxn, &(), &Frontier::empty())?;
        }
        if res.historical_roots.len(rwtxn)? == 0 {
            res.put_historical_root(rwtxn, Frontier::empty().root())?;
        }
        assert_ne!(res.historical_root_seq_ids.len(rwtxn)?, 0);
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
    ) -> &RoDatabaseUnique<SerdeBincode<Anchor>, U32<BE>> {
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

    pub(in crate::state) fn put_nullifier(
        &self,
        rwtxn: &mut RwTxn,
        nullifier: &Nullifier,
    ) -> Result<(), Error> {
        self.nullifiers
            .put(rwtxn, nullifier, &())
            .map_err(Error::from)
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
