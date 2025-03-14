use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    path::PathBuf,
};

use bitcoin::{self, hashes::Hash as _};
use fallible_iterator::{FallibleIterator, IteratorExt};
use heed::types::SerdeBincode;
use sneed::{
    DatabaseUnique, EnvError, RoTxn, RwTxn, UnitKey,
    db::error::Error as DbError, rwtxn::Error as RwTxnError,
};

use crate::types::{
    Accumulator, BlockHash, BmmResult, Body, Header, Tip, VERSION, Version,
    proto::mainchain,
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Db(#[from] DbError),
    #[error("Database env error")]
    DbEnv(#[from] EnvError),
    #[error("Database write error")]
    DbWrite(#[from] RwTxnError),
    #[error(
        "Incompatible DB version ({}). Please clear the DB (`{}`) and re-sync",
        .version,
        .db_path.display()
    )]
    IncompatibleVersion { version: Version, db_path: PathBuf },
    #[error("invalid merkle root")]
    InvalidMerkleRoot,
    #[error("invalid previous side hash")]
    InvalidPrevSideHash,
    #[error("no accumulator for block {0}")]
    NoAccumulator(BlockHash),
    #[error("no ancestor with depth {depth} for block {block_hash}")]
    NoAncestor { block_hash: BlockHash, depth: u32 },
    #[error("no mainchain ancestor with depth {depth} for block {block_hash}")]
    NoMainAncestor {
        block_hash: bitcoin::BlockHash,
        depth: u32,
    },
    #[error("unknown block hash: {0}")]
    NoBlockHash(BlockHash),
    #[error("no BMM result with block {0}")]
    NoBmmResult(BlockHash),
    #[error("no block body with hash {0}")]
    NoBody(BlockHash),
    #[error("no deposits info for block {0}")]
    NoDepositsInfo(bitcoin::BlockHash),
    #[error("no header with hash {0}")]
    NoHeader(BlockHash),
    #[error("no height info for block hash {0}")]
    NoHeight(BlockHash),
    #[error("unknown mainchain block hash: {0}")]
    NoMainBlockHash(bitcoin::BlockHash),
    #[error("no mainchain block info for block hash {0}")]
    NoMainBlockInfo(bitcoin::BlockHash),
    #[error("no mainchain header info for block hash {0}")]
    NoMainHeaderInfo(bitcoin::BlockHash),
    #[error("no height info for mainchain block hash {0}")]
    NoMainHeight(bitcoin::BlockHash),
}

#[derive(Clone)]
pub struct Archive {
    accumulators:
        DatabaseUnique<SerdeBincode<BlockHash>, SerdeBincode<Accumulator>>,
    block_hash_to_height:
        DatabaseUnique<SerdeBincode<BlockHash>, SerdeBincode<u32>>,
    /// BMM results for each header.
    /// All ancestors of any block should always be present.
    /// All relevant mainchain headers should exist in `main_headers`.
    /// Note that it is possible for a block to have BMM commitments in several
    /// different mainchain blocks, if there are any mainchain forks.
    bmm_results: DatabaseUnique<
        SerdeBincode<BlockHash>,
        SerdeBincode<HashMap<bitcoin::BlockHash, BmmResult>>,
    >,
    bodies: DatabaseUnique<SerdeBincode<BlockHash>, SerdeBincode<Body>>,
    /// Ancestors, indexed exponentially such that the nth element in a vector
    /// corresponds to the ancestor 2^(i+1) blocks before.
    /// eg.
    /// * the 0th element is the grandparent (2nd ancestor)
    /// * the 1st element is the grandparent's grandparent (4th ancestor)
    /// * the 3rd element is the 16th ancestor
    exponential_ancestors:
        DatabaseUnique<SerdeBincode<BlockHash>, SerdeBincode<Vec<BlockHash>>>,
    /// Mainchain ancestors, indexed exponentially such that the nth element in a vector
    /// corresponds to the ancestor 2^(i+1) blocks before.
    /// eg.
    /// * the 0th element is the grandparent (2nd ancestor)
    /// * the 1st element is the grandparent's grandparent (4th ancestor)
    /// * the 3rd element is the 16th ancestor
    exponential_main_ancestors: DatabaseUnique<
        SerdeBincode<bitcoin::BlockHash>,
        SerdeBincode<Vec<bitcoin::BlockHash>>,
    >,
    /// Sidechain headers. All ancestors of any header should always be present.
    headers: DatabaseUnique<SerdeBincode<BlockHash>, SerdeBincode<Header>>,
    main_block_hash_to_height:
        DatabaseUnique<SerdeBincode<bitcoin::BlockHash>, SerdeBincode<u32>>,
    /// Mainchain block infos.
    /// All ancestors of any header should always be present.
    /// BMM commitments do not imply existence of a sidechain block header.
    /// BMM commitments do not imply BMM validity of a sidechain block,
    /// as BMM commitments for ancestors may not exist.
    main_block_infos: DatabaseUnique<
        SerdeBincode<bitcoin::BlockHash>,
        SerdeBincode<mainchain::BlockInfo>,
    >,
    /// Mainchain header infos.
    /// All ancestors of any header should always be present.
    main_header_infos: DatabaseUnique<
        SerdeBincode<bitcoin::BlockHash>,
        SerdeBincode<mainchain::BlockHeaderInfo>,
    >,
    /// Mainchain successor blocks. ALL known block hashes, INCLUDING the zero hash,
    /// MUST be present.
    main_successors: DatabaseUnique<
        SerdeBincode<bitcoin::BlockHash>,
        SerdeBincode<HashSet<bitcoin::BlockHash>>,
    >,
    /// Successor blocks. ALL known block hashes MUST be present.
    successors: DatabaseUnique<
        SerdeBincode<Option<BlockHash>>,
        SerdeBincode<HashSet<BlockHash>>,
    >,
    /// Total work for mainchain headers with BMM verifications.
    /// All ancestors of any block should always be present
    total_work: DatabaseUnique<
        SerdeBincode<bitcoin::BlockHash>,
        SerdeBincode<bitcoin::Work>,
    >,
    _version: DatabaseUnique<UnitKey, SerdeBincode<Version>>,
}

impl Archive {
    pub const NUM_DBS: u32 = 14;

    pub fn new(env: &sneed::Env) -> Result<Self, Error> {
        let mut rwtxn = env.write_txn().map_err(EnvError::from)?;
        let version =
            DatabaseUnique::create(env, &mut rwtxn, "archive_version")
                .map_err(EnvError::from)?;
        match version.try_get(&rwtxn, &()).map_err(DbError::from)? {
            Some(db_version)
                if db_version
                    < Version {
                        major: 0,
                        minor: 12,
                        patch: 0,
                    } =>
            {
                // `deposits` and `main_bmm_commitments` were removed in
                // 0.12.0, and `main_block_infos` was added
                return Err(Error::IncompatibleVersion {
                    version: db_version,
                    db_path: env.path().to_path_buf(),
                });
            }
            Some(_) => (),
            None => version
                .put(&mut rwtxn, &(), &*VERSION)
                .map_err(DbError::from)?,
        }
        let accumulators =
            DatabaseUnique::create(env, &mut rwtxn, "accumulators")
                .map_err(EnvError::from)?;
        let block_hash_to_height =
            DatabaseUnique::create(env, &mut rwtxn, "hash_to_height")
                .map_err(EnvError::from)?;
        let bmm_results =
            DatabaseUnique::create(env, &mut rwtxn, "bmm_results")
                .map_err(EnvError::from)?;
        let bodies = DatabaseUnique::create(env, &mut rwtxn, "bodies")
            .map_err(EnvError::from)?;
        let exponential_ancestors =
            DatabaseUnique::create(env, &mut rwtxn, "exponential_ancestors")
                .map_err(EnvError::from)?;
        let exponential_main_ancestors = DatabaseUnique::create(
            env,
            &mut rwtxn,
            "exponential_main_ancestors",
        )
        .map_err(EnvError::from)?;
        let headers = DatabaseUnique::create(env, &mut rwtxn, "headers")
            .map_err(EnvError::from)?;
        let main_block_hash_to_height =
            DatabaseUnique::create(env, &mut rwtxn, "main_hash_to_height")
                .map_err(EnvError::from)?;
        let main_block_infos =
            DatabaseUnique::create(env, &mut rwtxn, "main_block_infos")
                .map_err(EnvError::from)?;
        let main_header_infos =
            DatabaseUnique::create(env, &mut rwtxn, "main_header_infos")
                .map_err(EnvError::from)?;
        let main_successors =
            DatabaseUnique::create(env, &mut rwtxn, "main_successors")
                .map_err(EnvError::from)?;
        if main_successors
            .try_get(&rwtxn, &bitcoin::BlockHash::all_zeros())
            .map_err(DbError::from)?
            .is_none()
        {
            main_successors
                .put(
                    &mut rwtxn,
                    &bitcoin::BlockHash::all_zeros(),
                    &HashSet::new(),
                )
                .map_err(DbError::from)?;
        }
        let successors = DatabaseUnique::create(env, &mut rwtxn, "successors")
            .map_err(EnvError::from)?;
        if successors
            .try_get(&rwtxn, &None)
            .map_err(DbError::from)?
            .is_none()
        {
            successors
                .put(&mut rwtxn, &None, &HashSet::new())
                .map_err(DbError::from)?;
        }
        let total_work = DatabaseUnique::create(env, &mut rwtxn, "total_work")
            .map_err(EnvError::from)?;
        rwtxn.commit().map_err(RwTxnError::from)?;
        Ok(Self {
            accumulators,
            block_hash_to_height,
            bmm_results,
            bodies,
            exponential_ancestors,
            exponential_main_ancestors,
            headers,
            main_block_infos,
            main_block_hash_to_height,
            main_header_infos,
            main_successors,
            successors,
            total_work,
            _version: version,
        })
    }

    pub fn try_get_accumulator(
        &self,
        rotxn: &RoTxn,
        block_hash: BlockHash,
    ) -> Result<Option<Accumulator>, Error> {
        let accumulator = self
            .accumulators
            .try_get(rotxn, &block_hash)
            .map_err(DbError::from)?;
        Ok(accumulator)
    }

    pub fn get_accumulator(
        &self,
        rotxn: &RoTxn,
        block_hash: BlockHash,
    ) -> Result<Accumulator, Error> {
        self.try_get_accumulator(rotxn, block_hash)?
            .ok_or(Error::NoAccumulator(block_hash))
    }

    pub fn try_get_height(
        &self,
        rotxn: &RoTxn,
        block_hash: BlockHash,
    ) -> Result<Option<u32>, Error> {
        self.block_hash_to_height
            .try_get(rotxn, &block_hash)
            .map_err(|err| DbError::from(err).into())
    }

    pub fn get_height(
        &self,
        rotxn: &RoTxn,
        block_hash: BlockHash,
    ) -> Result<u32, Error> {
        self.try_get_height(rotxn, block_hash)?
            .ok_or(Error::NoHeight(block_hash))
    }

    pub fn get_bmm_results(
        &self,
        rotxn: &RoTxn,
        block_hash: BlockHash,
    ) -> Result<HashMap<bitcoin::BlockHash, BmmResult>, Error> {
        let results = self
            .bmm_results
            .try_get(rotxn, &block_hash)
            .map_err(DbError::from)?
            .unwrap_or_default();
        Ok(results)
    }

    pub fn try_get_bmm_result(
        &self,
        rotxn: &RoTxn,
        block_hash: BlockHash,
        main_hash: bitcoin::BlockHash,
    ) -> Result<Option<BmmResult>, Error> {
        let results = self.get_bmm_results(rotxn, block_hash)?;
        Ok(results.get(&main_hash).copied())
    }

    pub fn get_bmm_result(
        &self,
        rotxn: &RoTxn,
        block_hash: BlockHash,
        main_hash: bitcoin::BlockHash,
    ) -> Result<BmmResult, Error> {
        self.try_get_bmm_result(rotxn, block_hash, main_hash)?
            .ok_or(Error::NoBmmResult(block_hash))
    }

    pub fn try_get_body(
        &self,
        rotxn: &RoTxn,
        block_hash: BlockHash,
    ) -> Result<Option<Body>, Error> {
        let body = self
            .bodies
            .try_get(rotxn, &block_hash)
            .map_err(DbError::from)?;
        Ok(body)
    }

    pub fn get_body(
        &self,
        rotxn: &RoTxn,
        block_hash: BlockHash,
    ) -> Result<Body, Error> {
        self.try_get_body(rotxn, block_hash)?
            .ok_or(Error::NoBody(block_hash))
    }

    pub fn try_get_header(
        &self,
        rotxn: &RoTxn,
        block_hash: BlockHash,
    ) -> Result<Option<Header>, Error> {
        let header = self
            .headers
            .try_get(rotxn, &block_hash)
            .map_err(DbError::from)?;
        Ok(header)
    }

    pub fn get_header(
        &self,
        rotxn: &RoTxn,
        block_hash: BlockHash,
    ) -> Result<Header, Error> {
        self.try_get_header(rotxn, block_hash)?
            .ok_or(Error::NoHeader(block_hash))
    }

    pub fn try_get_main_block_info(
        &self,
        rotxn: &RoTxn,
        main_hash: &bitcoin::BlockHash,
    ) -> Result<Option<mainchain::BlockInfo>, Error> {
        let block_info = self
            .main_block_infos
            .try_get(rotxn, main_hash)
            .map_err(DbError::from)?;
        Ok(block_info)
    }

    pub fn get_main_block_info(
        &self,
        rotxn: &RoTxn,
        main_hash: &bitcoin::BlockHash,
    ) -> Result<mainchain::BlockInfo, Error> {
        self.try_get_main_block_info(rotxn, main_hash)?
            .ok_or_else(|| Error::NoMainBlockInfo(*main_hash))
    }

    pub fn try_get_main_height(
        &self,
        rotxn: &RoTxn,
        block_hash: bitcoin::BlockHash,
    ) -> Result<Option<u32>, Error> {
        if block_hash == bitcoin::BlockHash::all_zeros() {
            Ok(Some(0))
        } else {
            self.main_block_hash_to_height
                .try_get(rotxn, &block_hash)
                .map_err(|err| DbError::from(err).into())
        }
    }

    pub fn get_main_height(
        &self,
        rotxn: &RoTxn,
        block_hash: bitcoin::BlockHash,
    ) -> Result<u32, Error> {
        self.try_get_main_height(rotxn, block_hash)?
            .ok_or(Error::NoMainHeight(block_hash))
    }

    pub fn try_get_main_header_info(
        &self,
        rotxn: &RoTxn,
        block_hash: &bitcoin::BlockHash,
    ) -> Result<Option<mainchain::BlockHeaderInfo>, Error> {
        let header_info = self
            .main_header_infos
            .try_get(rotxn, block_hash)
            .map_err(DbError::from)?;
        Ok(header_info)
    }

    fn get_main_header_info(
        &self,
        rotxn: &RoTxn,
        block_hash: &bitcoin::BlockHash,
    ) -> Result<mainchain::BlockHeaderInfo, Error> {
        self.try_get_main_header_info(rotxn, block_hash)?
            .ok_or_else(|| Error::NoMainHeaderInfo(*block_hash))
    }

    pub fn try_get_main_successors(
        &self,
        rotxn: &RoTxn,
        block_hash: bitcoin::BlockHash,
    ) -> Result<Option<HashSet<bitcoin::BlockHash>>, Error> {
        let successors = self
            .main_successors
            .try_get(rotxn, &block_hash)
            .map_err(DbError::from)?;
        Ok(successors)
    }

    pub fn get_main_successors(
        &self,
        rotxn: &RoTxn,
        block_hash: bitcoin::BlockHash,
    ) -> Result<HashSet<bitcoin::BlockHash>, Error> {
        self.try_get_main_successors(rotxn, block_hash)?
            .ok_or(Error::NoMainBlockHash(block_hash))
    }

    /// If block_hash is None, get genesis blocks
    pub fn try_get_successors(
        &self,
        rotxn: &RoTxn,
        block_hash: Option<BlockHash>,
    ) -> Result<Option<HashSet<BlockHash>>, Error> {
        let successors = self
            .successors
            .try_get(rotxn, &block_hash)
            .map_err(DbError::from)?;
        Ok(successors)
    }

    /// If block_hash is None, get genesis blocks
    pub fn get_successors(
        &self,
        rotxn: &RoTxn,
        block_hash: Option<BlockHash>,
    ) -> Result<HashSet<BlockHash>, Error> {
        self.try_get_successors(rotxn, block_hash)?.ok_or_else(|| {
            Error::NoBlockHash(
                block_hash.expect("Successors to None should always be known"),
            )
        })
    }

    pub fn try_get_total_work(
        &self,
        rotxn: &RoTxn,
        block_hash: bitcoin::BlockHash,
    ) -> Result<Option<bitcoin::Work>, Error> {
        let total_work = self
            .total_work
            .try_get(rotxn, &block_hash)
            .map_err(DbError::from)?;
        Ok(total_work)
    }

    pub fn get_total_work(
        &self,
        rotxn: &RoTxn,
        block_hash: bitcoin::BlockHash,
    ) -> Result<bitcoin::Work, Error> {
        self.try_get_total_work(rotxn, block_hash)?
            .ok_or(Error::NoMainHeaderInfo(block_hash))
    }

    /// Try to get the best valid mainchain verification for the specified block.
    pub fn try_get_best_main_verification(
        &self,
        rotxn: &RoTxn,
        block_hash: BlockHash,
    ) -> Result<Option<bitcoin::BlockHash>, Error> {
        let verifications = self.get_bmm_results(rotxn, block_hash)?;
        verifications
            .into_iter()
            .filter_map(|(main_hash, bmm_result)| {
                if bmm_result == BmmResult::Verified {
                    Some(Ok(main_hash))
                } else {
                    None
                }
            })
            .transpose_into_fallible()
            .max_by_key(|main_hash| self.get_total_work(rotxn, *main_hash))
    }

    /// Try to get the best valid mainchain verification for the specified block.
    pub fn get_best_main_verification(
        &self,
        rotxn: &RoTxn,
        block_hash: BlockHash,
    ) -> Result<bitcoin::BlockHash, Error> {
        self.try_get_best_main_verification(rotxn, block_hash)?
            .ok_or(Error::NoBmmResult(block_hash))
    }

    pub fn get_nth_ancestor(
        &self,
        rotxn: &RoTxn,
        mut block_hash: BlockHash,
        mut n: u32,
    ) -> Result<BlockHash, Error> {
        let orig_block_hash = block_hash;
        let orig_n = n;
        while n > 0 {
            let height = self.get_height(rotxn, block_hash)?;
            if n > height {
                return Err(Error::NoAncestor {
                    block_hash: orig_block_hash,
                    depth: orig_n,
                });
            }
            if n == 1 {
                let parent = self
                    .get_header(rotxn, block_hash)?
                    .prev_side_hash
                    .expect("block with height >= 1 should have a parent");
                return Ok(parent);
            } else {
                let exp_ancestor_index = u32::ilog2(n) - 1;
                block_hash = self
                    .exponential_ancestors
                    .get(rotxn, &block_hash)
                    .map_err(DbError::from)?[exp_ancestor_index as usize];
                n -= 2 << exp_ancestor_index;
            }
        }
        Ok(block_hash)
    }

    pub fn get_nth_main_ancestor(
        &self,
        rotxn: &RoTxn,
        mut block_hash: bitcoin::BlockHash,
        mut n: u32,
    ) -> Result<bitcoin::BlockHash, Error> {
        let orig_block_hash = block_hash;
        let orig_n = n;
        while n > 0 {
            let height = self.get_main_height(rotxn, block_hash)?;
            if n > height {
                return Err(Error::NoMainAncestor {
                    block_hash: orig_block_hash,
                    depth: orig_n,
                });
            }
            if n == 1 {
                let parent = self
                    .get_main_header_info(rotxn, &block_hash)?
                    .prev_block_hash;
                return Ok(parent);
            } else {
                let exp_ancestor_index = u32::ilog2(n) - 1;
                block_hash = self
                    .exponential_main_ancestors
                    .get(rotxn, &block_hash)
                    .map_err(DbError::from)?[exp_ancestor_index as usize];
                n -= 2 << exp_ancestor_index;
            }
        }
        Ok(block_hash)
    }

    /// Get block locator for the specified block hash
    pub fn get_block_locator(
        &self,
        rotxn: &RoTxn,
        block_hash: BlockHash,
    ) -> Result<Vec<BlockHash>, Error> {
        let header = self.get_header(rotxn, block_hash)?;
        let mut res = self
            .exponential_ancestors
            .get(rotxn, &block_hash)
            .map_err(DbError::from)?;
        if let Some(parent) = header.prev_side_hash {
            res.reverse();
            res.push(parent);
            res.reverse();
        }
        Ok(res)
    }

    /// Returns true if the second specified block is a descendant of the first
    /// specified block
    /// Returns an error if either of the specified block headers do not exist
    /// in the archive.
    pub fn is_descendant(
        &self,
        rotxn: &RoTxn,
        ancestor: BlockHash,
        descendant: BlockHash,
    ) -> Result<bool, Error> {
        if ancestor == descendant {
            return Ok(true);
        }
        let ancestor_height = self.get_height(rotxn, ancestor)?;
        let descendant_height = self.get_height(rotxn, descendant)?;
        if ancestor_height > descendant_height {
            return Ok(false);
        }
        let res = ancestor
            == self.get_nth_ancestor(
                rotxn,
                descendant,
                descendant_height - ancestor_height,
            )?;
        Ok(res)
    }

    /// Returns true if the second specified mainchain block is a descendant of
    /// the first specified block.
    /// Returns an error if either of the specified block headers do not exist
    /// in the archive.
    pub fn is_main_descendant(
        &self,
        rotxn: &RoTxn,
        ancestor: bitcoin::BlockHash,
        descendant: bitcoin::BlockHash,
    ) -> Result<bool, Error> {
        if ancestor == descendant {
            return Ok(true);
        }
        let ancestor_height = self.get_main_height(rotxn, ancestor)?;
        let descendant_height = self.get_main_height(rotxn, descendant)?;
        if ancestor_height > descendant_height {
            return Ok(false);
        }
        let res = ancestor
            == self.get_nth_main_ancestor(
                rotxn,
                descendant,
                descendant_height - ancestor_height,
            )?;
        Ok(res)
    }

    /// Store a block body. The header must already exist.
    pub fn put_accumulator(
        &self,
        rwtxn: &mut RwTxn,
        block_hash: BlockHash,
        accumulator: &Accumulator,
    ) -> Result<(), Error> {
        self.accumulators
            .put(rwtxn, &block_hash, accumulator)
            .map_err(DbError::from)?;
        Ok(())
    }

    /// Store a block body. The header must already exist.
    pub fn put_body(
        &self,
        rwtxn: &mut RwTxn,
        block_hash: BlockHash,
        body: &Body,
    ) -> Result<(), Error> {
        let header = self.get_header(rwtxn, block_hash)?;
        if header.merkle_root != body.compute_merkle_root() {
            return Err(Error::InvalidMerkleRoot);
        }
        self.bodies
            .put(rwtxn, &block_hash, body)
            .map_err(DbError::from)?;
        Ok(())
    }

    /// Store a header.
    ///
    /// The following predicates MUST be met before calling this function:
    /// * Ancestor headers MUST be stored
    /// * BMM commitments MUST be stored for mainchain header where
    ///   `main_header.prev_blockhash == header.prev_main_hash`
    pub fn put_header(
        &self,
        rwtxn: &mut RwTxn,
        header: &Header,
    ) -> Result<(), Error> {
        let height = match header.prev_side_hash {
            None => 0,
            Some(parent) => {
                self.try_get_height(rwtxn, parent)?
                    .ok_or(Error::InvalidPrevSideHash)?
                    + 1
            }
        };
        let block_hash = header.hash();
        self.block_hash_to_height
            .put(rwtxn, &block_hash, &height)
            .map_err(DbError::from)?;
        self.headers
            .put(rwtxn, &block_hash, header)
            .map_err(DbError::from)?;
        // Add to successors for predecessor
        {
            let mut pred_successors =
                self.get_successors(rwtxn, header.prev_side_hash)?;
            pred_successors.insert(block_hash);
            self.successors
                .put(rwtxn, &header.prev_side_hash, &pred_successors)
                .map_err(DbError::from)?;
        }
        // Store successors
        {
            let successors = self
                .try_get_successors(rwtxn, Some(block_hash))?
                .unwrap_or_default();
            self.successors
                .put(rwtxn, &Some(block_hash), &successors)
                .map_err(DbError::from)?;
        }
        // populate exponential ancestors
        let mut exponential_ancestors = Vec::<BlockHash>::new();
        if height >= 2 {
            let grandparent = self.get_nth_ancestor(
                rwtxn,
                header.prev_side_hash.unwrap(),
                1,
            )?;
            exponential_ancestors.push(grandparent);
            let mut next_exponential_ancestor_depth = 4u64;
            while height as u64 >= next_exponential_ancestor_depth {
                let next_exponential_ancestor = self.get_nth_ancestor(
                    rwtxn,
                    *exponential_ancestors.last().unwrap(),
                    next_exponential_ancestor_depth as u32 / 2,
                )?;
                exponential_ancestors.push(next_exponential_ancestor);
                next_exponential_ancestor_depth *= 2;
            }
        }
        self.exponential_ancestors
            .put(rwtxn, &block_hash, &exponential_ancestors)
            .map_err(DbError::from)?;
        // Populate BMM verifications
        {
            let mut bmm_results = self.get_bmm_results(rwtxn, block_hash)?;
            let parent_bmm_results = if let Some(parent) = header.prev_side_hash
            {
                Some(self.get_bmm_results(rwtxn, parent)?)
            } else {
                None
            };
            let main_blocks =
                self.get_main_successors(rwtxn, header.prev_main_hash)?;
            for main_block in main_blocks {
                let Some(commitment) = self
                    .get_main_block_info(rwtxn, &main_block)?
                    .bmm_commitment
                else {
                    tracing::trace!(%block_hash, "Failed BMM @ {main_block}: missing commitment");
                    bmm_results.insert(main_block, BmmResult::Failed);
                    continue;
                };
                if commitment != block_hash {
                    tracing::trace!(%block_hash, "Failed BMM @ {main_block}: commitment to other block ({commitment})");
                    bmm_results.insert(main_block, BmmResult::Failed);
                    continue;
                }
                let main_header_info =
                    self.get_main_header_info(rwtxn, &main_block)?;
                if header.prev_main_hash != main_header_info.prev_block_hash {
                    tracing::trace!(%block_hash, "Failed BMM @ {main_block}: should be impossible?");
                    bmm_results.insert(main_block, BmmResult::Failed);
                    continue;
                }
                let Some(parent_bmm_results) = parent_bmm_results.as_ref()
                else {
                    tracing::trace!(%block_hash, "Verified BMM @ {main_block}: no parent");
                    bmm_results.insert(main_block, BmmResult::Verified);
                    continue;
                };
                // Check if there is a valid BMM commitment to the parent in the
                // main ancestry
                let main_ancestry_contains_valid_bmm_commitment_to_parent =
                    parent_bmm_results
                        .iter()
                        .map(Ok)
                        .transpose_into_fallible()
                        .any(|(bmm_block, bmm_result)| {
                            let parent_verified = *bmm_result
                                == BmmResult::Verified
                                && self.is_main_descendant(
                                    rwtxn, *bmm_block, main_block,
                                )?;
                            Result::<bool, Error>::Ok(parent_verified)
                        })?;
                if main_ancestry_contains_valid_bmm_commitment_to_parent {
                    tracing::trace!(%block_hash, "Verified BMM @ {main_block}: verified parent");
                    bmm_results.insert(main_block, BmmResult::Verified);
                    continue;
                } else {
                    tracing::trace!(%block_hash, "Failed BMM @ {main_block}: no valid BMM commitment to parent in main ancestry");
                    bmm_results.insert(main_block, BmmResult::Failed);
                    continue;
                }
            }
            self.bmm_results
                .put(rwtxn, &block_hash, &bmm_results)
                .map_err(DbError::from)?;
        }
        Ok(())
    }

    /// All ancestors MUST be present.
    /// Mainchain blocks MUST be present in `main_headers`.
    pub fn put_main_block_info(
        &self,
        rwtxn: &mut RwTxn,
        main_hash: bitcoin::BlockHash,
        block_info: &mainchain::BlockInfo,
    ) -> Result<(), Error> {
        let main_header_info = self.get_main_header_info(rwtxn, &main_hash)?;
        if main_header_info.prev_block_hash != bitcoin::BlockHash::all_zeros() {
            let _parent_info = self.get_main_block_info(
                rwtxn,
                &main_header_info.prev_block_hash,
            )?;
        }
        self.main_block_infos
            .put(rwtxn, &main_hash, block_info)
            .map_err(DbError::from)?;
        let Some(commitment) = block_info.bmm_commitment else {
            return Ok(());
        };
        let Some(header) = self.try_get_header(rwtxn, commitment)? else {
            return Ok(());
        };
        let bmm_result = if header.prev_main_hash
            != main_header_info.prev_block_hash
        {
            BmmResult::Failed
        } else if let Some(parent) = header.prev_side_hash {
            // Check if there is a valid BMM commitment to the parent in the
            // main ancestry
            let parent_bmm_results = self.get_bmm_results(rwtxn, parent)?;
            let main_ancestry_contains_valid_bmm_commitment_to_parent =
                parent_bmm_results
                    .into_iter()
                    .map(Ok)
                    .transpose_into_fallible()
                    .any(|(bmm_block, bmm_result)| {
                        let parent_verified = bmm_result == BmmResult::Verified
                            && self.is_main_descendant(
                                rwtxn, bmm_block, main_hash,
                            )?;
                        Result::<bool, Error>::Ok(parent_verified)
                    })?;
            if main_ancestry_contains_valid_bmm_commitment_to_parent {
                BmmResult::Verified
            } else {
                BmmResult::Failed
            }
        } else {
            BmmResult::Verified
        };
        let mut bmm_results = self.get_bmm_results(rwtxn, commitment)?;
        bmm_results.insert(main_hash, bmm_result);
        self.bmm_results
            .put(rwtxn, &commitment, &bmm_results)
            .map_err(DbError::from)?;
        Ok(())
    }

    pub fn put_main_header_info(
        &self,
        rwtxn: &mut RwTxn,
        header_info: &mainchain::BlockHeaderInfo,
    ) -> Result<(), Error> {
        if self
            .try_get_main_header_info(rwtxn, &header_info.prev_block_hash)?
            .is_none()
            && header_info.prev_block_hash != bitcoin::BlockHash::all_zeros()
        {
            return Err(Error::NoMainHeaderInfo(header_info.prev_block_hash));
        }
        let block_hash = header_info.block_hash;
        let prev_height =
            self.get_main_height(rwtxn, header_info.prev_block_hash)?;
        let height = prev_height + 1;
        let total_work =
            if header_info.prev_block_hash != bitcoin::BlockHash::all_zeros() {
                let prev_work =
                    self.get_total_work(rwtxn, header_info.prev_block_hash)?;
                prev_work + header_info.work
            } else {
                header_info.work
            };
        self.main_block_hash_to_height
            .put(rwtxn, &block_hash, &height)
            .map_err(DbError::from)?;
        self.main_header_infos
            .put(rwtxn, &block_hash, header_info)
            .map_err(DbError::from)?;
        self.total_work
            .put(rwtxn, &block_hash, &total_work)
            .map_err(DbError::from)?;
        // Add to successors for predecessor
        {
            let mut pred_successors =
                self.get_main_successors(rwtxn, header_info.prev_block_hash)?;
            pred_successors.insert(block_hash);
            self.main_successors
                .put(rwtxn, &header_info.prev_block_hash, &pred_successors)
                .map_err(DbError::from)?;
        }
        // Store successors
        {
            let successors = self
                .try_get_main_successors(rwtxn, block_hash)?
                .unwrap_or_default();
            self.main_successors
                .put(rwtxn, &block_hash, &successors)
                .map_err(DbError::from)?;
        }
        // populate exponential ancestors
        let mut exponential_ancestors = Vec::<bitcoin::BlockHash>::new();
        if height >= 2 {
            let grandparent = self.get_nth_main_ancestor(
                rwtxn,
                header_info.prev_block_hash,
                1,
            )?;
            exponential_ancestors.push(grandparent);
            let mut next_exponential_ancestor_depth = 4u64;
            while height as u64 >= next_exponential_ancestor_depth {
                let next_exponential_ancestor = self.get_nth_main_ancestor(
                    rwtxn,
                    *exponential_ancestors.last().unwrap(),
                    next_exponential_ancestor_depth as u32 / 2,
                )?;
                exponential_ancestors.push(next_exponential_ancestor);
                next_exponential_ancestor_depth *= 2;
            }
        }
        self.exponential_main_ancestors
            .put(rwtxn, &block_hash, &exponential_ancestors)
            .map_err(DbError::from)?;
        Ok(())
    }

    /// Return a fallible iterator over ancestors of a block,
    /// starting with the specified block's header
    pub fn ancestors<'a, 'rotxn>(
        &'a self,
        rotxn: &'a RoTxn<'rotxn>,
        block_hash: BlockHash,
    ) -> Ancestors<'a, 'rotxn> where {
        Ancestors {
            archive: self,
            rotxn,
            block_hash: Some(block_hash),
        }
    }

    /// Get missing bodies in the ancestry of the specified block, up to the
    /// specified ancestor.
    /// The specified ancestor must exist.
    /// Blocks for which bodies are missing are returned oldest-to-newest.
    pub fn get_missing_bodies(
        &self,
        rotxn: &RoTxn,
        block_hash: BlockHash,
        ancestor: Option<BlockHash>,
    ) -> Result<Vec<BlockHash>, Error> {
        // TODO: check that ancestor is nth ancestor of block_hash
        let mut res: Vec<BlockHash> = self
            .ancestors(rotxn, block_hash)
            .take_while(|block_hash| {
                Ok(ancestor.is_none_or(|ancestor| *block_hash != ancestor))
            })
            .filter_map(|block_hash| {
                match self.try_get_body(rotxn, block_hash)? {
                    Some(_) => Ok(None),
                    None => Ok(Some(block_hash)),
                }
            })
            .collect()?;
        res.reverse();
        Ok(res)
    }

    /// Return a fallible iterator over ancestors of a mainchain block,
    /// starting with the specified block's header
    pub fn main_ancestors<'a>(
        &'a self,
        rotxn: &'a RoTxn,
        mut block_hash: bitcoin::BlockHash,
    ) -> impl FallibleIterator<Item = bitcoin::BlockHash, Error = Error> + 'a
    {
        fallible_iterator::from_fn(move || {
            if block_hash == bitcoin::BlockHash::all_zeros() {
                Ok(None)
            } else {
                let res = Some(block_hash);
                let header_info =
                    self.get_main_header_info(rotxn, &block_hash)?;
                block_hash = header_info.prev_block_hash;
                Ok(res)
            }
        })
    }

    /// Find the last common ancestor of two blocks, if headers for both exist
    pub fn last_common_ancestor(
        &self,
        rotxn: &RoTxn,
        mut block_hash0: BlockHash,
        mut block_hash1: BlockHash,
    ) -> Result<Option<BlockHash>, Error> {
        let mut height0 = self.get_height(rotxn, block_hash0)?;
        let height1 = self.get_height(rotxn, block_hash1)?;
        // Equalize heights to min(height0, height1)
        match height0.cmp(&height1) {
            Ordering::Equal => (),
            Ordering::Less => {
                block_hash1 = self.get_nth_ancestor(
                    rotxn,
                    block_hash1,
                    height1 - height0,
                )?;
            }
            Ordering::Greater => {
                block_hash0 = self.get_nth_ancestor(
                    rotxn,
                    block_hash0,
                    height0 - height1,
                )?;
                height0 = height1;
            }
        }
        // if the block hashes are the same, return early
        if block_hash0 == block_hash1 {
            return Ok(Some(block_hash0));
        }
        // if there is no shared lineage, return None
        {
            let oldest_ancestor_0 =
                self.get_nth_ancestor(rotxn, block_hash0, height0)?;
            let oldest_ancestor_1 =
                self.get_nth_ancestor(rotxn, block_hash1, height0)?;
            if oldest_ancestor_0 != oldest_ancestor_1 {
                return Ok(None);
            }
        }
        // use a binary search to find the last common ancestor
        let mut lo_depth = 1;
        let mut hi_depth = height0;
        while lo_depth < hi_depth {
            let mid_depth = (lo_depth + hi_depth) / 2;
            let mid_ancestor0 =
                self.get_nth_ancestor(rotxn, block_hash0, mid_depth)?;
            let mid_ancestor1 =
                self.get_nth_ancestor(rotxn, block_hash1, mid_depth)?;
            if mid_ancestor0 == mid_ancestor1 {
                hi_depth = mid_depth;
            } else {
                lo_depth = mid_depth + 1;
            }
        }
        self.get_nth_ancestor(rotxn, block_hash0, hi_depth)
            .map(Some)
    }

    /// Find the last common mainchain ancestor of two blocks,
    /// if headers for both exist
    pub fn last_common_main_ancestor(
        &self,
        rotxn: &RoTxn,
        mut block_hash0: bitcoin::BlockHash,
        mut block_hash1: bitcoin::BlockHash,
    ) -> Result<bitcoin::BlockHash, Error> {
        let mut height0 = self.get_main_height(rotxn, block_hash0)?;
        let height1 = self.get_main_height(rotxn, block_hash1)?;
        // Equalize heights to min(height0, height1)
        match height0.cmp(&height1) {
            Ordering::Equal => (),
            Ordering::Less => {
                block_hash1 = self.get_nth_main_ancestor(
                    rotxn,
                    block_hash1,
                    height1 - height0,
                )?;
            }
            Ordering::Greater => {
                block_hash0 = self.get_nth_main_ancestor(
                    rotxn,
                    block_hash0,
                    height0 - height1,
                )?;
                height0 = height1;
            }
        }
        // if the block hashes are the same, return early
        if block_hash0 == block_hash1 {
            return Ok(block_hash0);
        }
        // use a binary search to find the last common ancestor
        let mut lo_depth = 1;
        let mut hi_depth = height0;
        while lo_depth < hi_depth {
            let mid_depth = (lo_depth + hi_depth) / 2;
            let mid_ancestor0 =
                self.get_nth_main_ancestor(rotxn, block_hash0, mid_depth)?;
            let mid_ancestor1 =
                self.get_nth_main_ancestor(rotxn, block_hash1, mid_depth)?;
            if mid_ancestor0 == mid_ancestor1 {
                hi_depth = mid_depth;
            } else {
                lo_depth = mid_depth + 1;
            }
        }
        self.get_nth_main_ancestor(rotxn, block_hash0, hi_depth)
    }

    /// Determine if two mainchain blocks are part of a shared lineage,
    /// ie. one block is a descendent of the other
    pub fn shared_mainchain_lineage(
        &self,
        rotxn: &RoTxn,
        mut block_hash0: bitcoin::BlockHash,
        mut block_hash1: bitcoin::BlockHash,
    ) -> Result<bool, Error> {
        let height0 = self.get_main_height(rotxn, block_hash0)?;
        let height1 = self.get_main_height(rotxn, block_hash1)?;
        match height0.cmp(&height1) {
            Ordering::Equal => (),
            Ordering::Less => {
                block_hash1 = self.get_nth_main_ancestor(
                    rotxn,
                    block_hash1,
                    height1 - height0,
                )?;
            }
            Ordering::Greater => {
                block_hash0 = self.get_nth_main_ancestor(
                    rotxn,
                    block_hash0,
                    height0 - height1,
                )?;
            }
        }
        Ok(block_hash0 == block_hash1)
    }

    /// Compares two potential tips and returns the better tip, if there is one.
    /// Headers for each tip MUST exist.
    /// It is possible that neither tip is better, eg. if the mainchain lineage
    /// is not shared and the tip with greater total work had lower height before
    /// the common mainchain ancestor.
    /// ie. the tip with either:
    /// * if the mainchain lineage is shared:
    ///   * greater height
    ///   * equal height and greater total work
    /// * if the mainchain lineage is not shared:
    ///   * greater height AND equal work
    ///   * greater or equal height AND greater total work
    ///   * greater total work AND greater or equal height before common
    ///     mainchain ancestor
    ///   * equal height AND equal total work AND greater height before common mainchain ancestor
    // TODO: Review this rule
    pub fn better_tip(
        &self,
        rotxn: &RoTxn,
        tip0: Tip,
        tip1: Tip,
    ) -> Result<Option<Tip>, Error> {
        if tip0 == tip1 {
            return Ok(None);
        }
        let block_hash0 = tip0.block_hash;
        let block_hash1 = tip1.block_hash;
        let height0 = self.get_height(rotxn, block_hash0)?;
        let height1 = self.get_height(rotxn, block_hash1)?;
        match (height0, height1) {
            (0, 0) => return Ok(None),
            (0, _) => return Ok(Some(tip1)),
            (_, 0) => return Ok(Some(tip0)),
            (_, _) => (),
        }
        let work0 = self.get_total_work(rotxn, tip0.main_block_hash)?;
        let work1 = self.get_total_work(rotxn, tip1.main_block_hash)?;
        match (work0.cmp(&work1), height0.cmp(&height1)) {
            (Ordering::Less | Ordering::Equal, Ordering::Less) => {
                // No ancestor of tip0 can have greater height,
                // so tip1 is better.
                Ok(Some(tip1))
            }
            (Ordering::Equal | Ordering::Greater, Ordering::Greater) => {
                // No ancestor of tip1 can have greater height,
                // so tip0 is better.
                Ok(Some(tip0))
            }
            (Ordering::Less, Ordering::Equal) => {
                // Within the same mainchain lineage, prefer lower work
                // Otherwise, prefer tip with greater work
                if self.shared_mainchain_lineage(
                    rotxn,
                    tip0.main_block_hash,
                    tip1.main_block_hash,
                )? {
                    Ok(Some(tip0))
                } else {
                    Ok(Some(tip1))
                }
            }
            (Ordering::Greater, Ordering::Equal) => {
                // Within the same mainchain lineage, prefer lower work
                // Otherwise, prefer tip with greater work
                if !self.shared_mainchain_lineage(
                    rotxn,
                    tip0.main_block_hash,
                    tip1.main_block_hash,
                )? {
                    Ok(Some(tip0))
                } else {
                    Ok(Some(tip1))
                }
            }
            (Ordering::Less, Ordering::Greater) => {
                // Need to check if tip0 ancestor before common
                // mainchain ancestor had greater or equal height
                let main_ancestor = self.last_common_main_ancestor(
                    rotxn,
                    tip0.main_block_hash,
                    tip1.main_block_hash,
                )?;
                let tip0_ancestor_height = self
                    .ancestors(rotxn, block_hash0)
                    .find_map(|tip0_ancestor| {
                    let header = self.get_header(rotxn, tip0_ancestor)?;
                    if !self.is_main_descendant(
                        rotxn,
                        header.prev_main_hash,
                        main_ancestor,
                    )? {
                        return Ok(None);
                    }
                    if header.prev_main_hash == main_ancestor {
                        return Ok(None);
                    }
                    self.get_height(rotxn, tip0_ancestor).map(Some)
                })?;
                if tip0_ancestor_height >= Some(height1) {
                    Ok(Some(tip0))
                } else {
                    Ok(Some(tip1))
                }
            }
            (Ordering::Greater, Ordering::Less) => {
                // Need to check if tip1 ancestor before common
                // mainchain ancestor had greater or equal height
                let main_ancestor = self.last_common_main_ancestor(
                    rotxn,
                    tip0.main_block_hash,
                    tip1.main_block_hash,
                )?;
                let tip1_ancestor_height = self
                    .ancestors(rotxn, block_hash1)
                    .find_map(|tip1_ancestor| {
                    let header = self.get_header(rotxn, tip1_ancestor)?;
                    if !self.is_main_descendant(
                        rotxn,
                        header.prev_main_hash,
                        main_ancestor,
                    )? {
                        return Ok(None);
                    }
                    if header.prev_main_hash == main_ancestor {
                        return Ok(None);
                    }
                    self.get_height(rotxn, tip1_ancestor).map(Some)
                })?;
                if tip1_ancestor_height < Some(height0) {
                    Ok(Some(tip0))
                } else {
                    Ok(Some(tip1))
                }
            }
            (Ordering::Equal, Ordering::Equal) => {
                // If tip0 is the same as tip1, return tip0
                if block_hash0 == block_hash1 {
                    return Ok(Some(tip0));
                }
                // Need to compare tip0 ancestor and tip1 ancestor
                // before common mainchain ancestor
                let main_ancestor = self.last_common_main_ancestor(
                    rotxn,
                    tip0.main_block_hash,
                    tip1.main_block_hash,
                )?;
                let main_ancestor_height =
                    self.get_main_height(rotxn, main_ancestor)?;
                let (tip0_ancestor_height, tip0_ancestor_work) = self
                    .ancestors(rotxn, block_hash0)
                    .find_map(|tip0_ancestor| {
                        let header = self.get_header(rotxn, tip0_ancestor)?;
                        if !self.is_main_descendant(
                            rotxn,
                            header.prev_main_hash,
                            main_ancestor,
                        )? {
                            return Ok(None);
                        }
                        if header.prev_main_hash == main_ancestor {
                            return Ok(None);
                        }
                        let height = self.get_height(rotxn, tip0_ancestor)?;
                        // Find mainchain block hash to get total work
                        let main_block = {
                            let prev_height = self.get_main_height(
                                rotxn,
                                header.prev_main_hash,
                            )?;
                            let height = prev_height + 1;
                            self.get_nth_main_ancestor(
                                rotxn,
                                main_ancestor,
                                main_ancestor_height - height,
                            )?
                        };
                        let work = self.get_total_work(rotxn, main_block)?;
                        Ok(Some((height, work)))
                    })?
                    .map_or((None, None), |(height, work)| {
                        (Some(height), Some(work))
                    });
                let (tip1_ancestor_height, tip1_ancestor_work) = self
                    .ancestors(rotxn, block_hash1)
                    .find_map(|tip1_ancestor| {
                        let header = self.get_header(rotxn, tip1_ancestor)?;
                        if !self.is_main_descendant(
                            rotxn,
                            header.prev_main_hash,
                            main_ancestor,
                        )? {
                            return Ok(None);
                        }
                        if header.prev_main_hash == main_ancestor {
                            return Ok(None);
                        }
                        let height = self.get_height(rotxn, tip1_ancestor)?;
                        // Find mainchain block hash to get total work
                        let main_block = {
                            let prev_height = self.get_main_height(
                                rotxn,
                                header.prev_main_hash,
                            )?;
                            let height = prev_height + 1;
                            self.get_nth_main_ancestor(
                                rotxn,
                                main_ancestor,
                                main_ancestor_height - height,
                            )?
                        };
                        let work = self.get_total_work(rotxn, main_block)?;
                        Ok(Some((height, work)))
                    })?
                    .map_or((None, None), |(height, work)| {
                        (Some(height), Some(work))
                    });
                match (
                    tip0_ancestor_work.cmp(&tip1_ancestor_work),
                    tip0_ancestor_height.cmp(&tip1_ancestor_height),
                ) {
                    (Ordering::Less | Ordering::Equal, Ordering::Equal)
                    | (_, Ordering::Greater) => {
                        // tip1 is not better
                        Ok(Some(tip0))
                    }
                    (Ordering::Greater, Ordering::Equal)
                    | (_, Ordering::Less) => {
                        // tip1 is better
                        Ok(Some(tip1))
                    }
                }
            }
        }
    }
}

/// Return a fallible iterator over ancestors of a block,
/// starting with the specified block.
/// created by [`Archive::ancestors`]
pub struct Ancestors<'a, 'rotxn> {
    archive: &'a Archive,
    rotxn: &'a RoTxn<'rotxn>,
    block_hash: Option<BlockHash>,
}

impl FallibleIterator for Ancestors<'_, '_> {
    type Item = BlockHash;
    type Error = Error;

    fn next(&mut self) -> Result<Option<Self::Item>, Self::Error> {
        match self.block_hash {
            None => Ok(None),
            Some(res) => {
                let header = self.archive.get_header(self.rotxn, res)?;
                self.block_hash = header.prev_side_hash;
                Ok(Some(res))
            }
        }
    }
}
