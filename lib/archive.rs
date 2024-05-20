use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
};

use bip300301::{
    bitcoin::{self, hashes::Hash},
    DepositInfo, Header as BitcoinHeader,
};
use fallible_iterator::{FallibleIterator, IteratorExt};
use heed::{types::SerdeBincode, Database, RoTxn, RwTxn};

use crate::types::{Accumulator, BlockHash, BmmResult, Body, Header, Tip};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid mainchain block hash for deposit")]
    DepositInvalidMainBlockHash,
    #[error("heed error")]
    Heed(#[from] heed::Error),
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
    #[error("no BMM commitments data for mainchain block {0}")]
    NoMainBmmCommitments(bitcoin::BlockHash),
    #[error("no mainchain header with hash {0}")]
    NoMainHeader(bitcoin::BlockHash),
    #[error("no height info for mainchain block hash {0}")]
    NoMainHeight(bitcoin::BlockHash),
}

#[derive(Clone)]
pub struct Archive {
    accumulators: Database<SerdeBincode<BlockHash>, SerdeBincode<Accumulator>>,
    block_hash_to_height: Database<SerdeBincode<BlockHash>, SerdeBincode<u32>>,
    /// BMM results for each header.
    /// All ancestors of any block should always be present.
    /// All relevant mainchain headers should exist in `main_headers`.
    /// Note that it is possible for a block to have BMM commitments in several
    /// different mainchain blocks, if there are any mainchain forks.
    bmm_results: Database<
        SerdeBincode<BlockHash>,
        SerdeBincode<HashMap<bitcoin::BlockHash, BmmResult>>,
    >,
    bodies: Database<SerdeBincode<BlockHash>, SerdeBincode<Body>>,
    /// Deposits by mainchain block, sorted first-to-last in each block
    deposits: Database<
        SerdeBincode<bitcoin::BlockHash>,
        SerdeBincode<Vec<DepositInfo>>,
    >,
    /// Ancestors, indexed exponentially such that the nth element in a vector
    /// corresponds to the ancestor 2^(i+1) blocks before.
    /// eg.
    /// * the 0th element is the grandparent (2nd ancestor)
    /// * the 1st element is the grandparent's grandparent (4th ancestor)
    /// * the 3rd element is the 16th ancestor
    exponential_ancestors:
        Database<SerdeBincode<BlockHash>, SerdeBincode<Vec<BlockHash>>>,
    /// Mainchain ancestors, indexed exponentially such that the nth element in a vector
    /// corresponds to the ancestor 2^(i+1) blocks before.
    /// eg.
    /// * the 0th element is the grandparent (2nd ancestor)
    /// * the 1st element is the grandparent's grandparent (4th ancestor)
    /// * the 3rd element is the 16th ancestor
    exponential_main_ancestors: Database<
        SerdeBincode<bitcoin::BlockHash>,
        SerdeBincode<Vec<bitcoin::BlockHash>>,
    >,
    /// Sidechain headers. All ancestors of any header should always be present.
    headers: Database<SerdeBincode<BlockHash>, SerdeBincode<Header>>,
    main_block_hash_to_height:
        Database<SerdeBincode<bitcoin::BlockHash>, SerdeBincode<u32>>,
    /// BMM commitments in each mainchain block.
    /// All ancestors must be present.
    /// Mainchain blocks MUST be present in `main_headers`, but not all
    /// mainchain headers will be present, if the blocks are not available.
    /// BMM commitments do not imply existence of a sidechain block header.
    /// BMM commitments do not imply BMM validity of a sidechain block,
    /// as BMM commitments for ancestors may not exist.
    main_bmm_commitments: Database<
        SerdeBincode<bitcoin::BlockHash>,
        SerdeBincode<Option<BlockHash>>,
    >,
    /// Mainchain headers. All ancestors of any header should always be present
    main_headers:
        Database<SerdeBincode<bitcoin::BlockHash>, SerdeBincode<BitcoinHeader>>,
    /// Mainchain successor blocks. ALL known block hashes, INCLUDING the zero hash,
    /// MUST be present.
    main_successors: Database<
        SerdeBincode<bitcoin::BlockHash>,
        SerdeBincode<HashSet<bitcoin::BlockHash>>,
    >,
    /// Successor blocks. ALL known block hashes, INCLUDING the zero hash,
    /// MUST be present.
    successors:
        Database<SerdeBincode<BlockHash>, SerdeBincode<HashSet<BlockHash>>>,
    /// Total work for mainchain headers with BMM verifications.
    /// All ancestors of any block should always be present
    total_work:
        Database<SerdeBincode<bitcoin::BlockHash>, SerdeBincode<bitcoin::Work>>,
}

impl Archive {
    pub const NUM_DBS: u32 = 14;

    pub fn new(env: &heed::Env) -> Result<Self, Error> {
        let mut rwtxn = env.write_txn()?;
        let accumulators =
            env.create_database(&mut rwtxn, Some("accumulators"))?;
        let block_hash_to_height =
            env.create_database(&mut rwtxn, Some("hash_to_height"))?;
        let bmm_results =
            env.create_database(&mut rwtxn, Some("bmm_results"))?;
        let bodies = env.create_database(&mut rwtxn, Some("bodies"))?;
        let deposits = env.create_database(&mut rwtxn, Some("deposits"))?;
        let exponential_ancestors =
            env.create_database(&mut rwtxn, Some("exponential_ancestors"))?;
        let exponential_main_ancestors = env
            .create_database(&mut rwtxn, Some("exponential_main_ancestors"))?;
        let headers = env.create_database(&mut rwtxn, Some("headers"))?;
        let main_block_hash_to_height =
            env.create_database(&mut rwtxn, Some("main_hash_to_height"))?;
        let main_bmm_commitments =
            env.create_database(&mut rwtxn, Some("main_bmm_commitments"))?;
        let main_headers =
            env.create_database(&mut rwtxn, Some("main_headers"))?;
        let main_successors =
            env.create_database(&mut rwtxn, Some("main_successors"))?;
        if main_successors
            .get(&rwtxn, &bitcoin::BlockHash::all_zeros())?
            .is_none()
        {
            main_successors.put(
                &mut rwtxn,
                &bitcoin::BlockHash::all_zeros(),
                &HashSet::new(),
            )?;
        }
        let successors = env.create_database(&mut rwtxn, Some("successors"))?;
        if successors.get(&rwtxn, &BlockHash::default())?.is_none() {
            successors.put(
                &mut rwtxn,
                &BlockHash::default(),
                &HashSet::new(),
            )?;
        }
        let total_work = env.create_database(&mut rwtxn, Some("total_work"))?;
        rwtxn.commit()?;
        Ok(Self {
            accumulators,
            block_hash_to_height,
            bmm_results,
            bodies,
            deposits,
            exponential_ancestors,
            exponential_main_ancestors,
            headers,
            main_bmm_commitments,
            main_block_hash_to_height,
            main_headers,
            main_successors,
            successors,
            total_work,
        })
    }

    pub fn try_get_accumulator(
        &self,
        rotxn: &RoTxn,
        block_hash: BlockHash,
    ) -> Result<Option<Accumulator>, Error> {
        if block_hash == BlockHash::default() {
            Ok(Some(Accumulator::default()))
        } else {
            let accumulator = self.accumulators.get(rotxn, &block_hash)?;
            Ok(accumulator)
        }
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
        if block_hash == BlockHash::default() {
            Ok(Some(0))
        } else {
            self.block_hash_to_height
                .get(rotxn, &block_hash)
                .map_err(Error::from)
        }
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
            .get(rotxn, &block_hash)
            .map_err(Error::from)?
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
        let body = self.bodies.get(rotxn, &block_hash)?;
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

    pub fn try_get_deposits(
        &self,
        rotxn: &RoTxn,
        block_hash: bitcoin::BlockHash,
    ) -> Result<Option<Vec<DepositInfo>>, Error> {
        let deposits = self.deposits.get(rotxn, &block_hash)?;
        Ok(deposits)
    }

    pub fn get_deposits(
        &self,
        rotxn: &RoTxn,
        block_hash: bitcoin::BlockHash,
    ) -> Result<Vec<DepositInfo>, Error> {
        self.try_get_deposits(rotxn, block_hash)?
            .ok_or(Error::NoDepositsInfo(block_hash))
    }

    pub fn try_get_header(
        &self,
        rotxn: &RoTxn,
        block_hash: BlockHash,
    ) -> Result<Option<Header>, Error> {
        let header = self.headers.get(rotxn, &block_hash)?;
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

    pub fn try_get_main_bmm_commitment(
        &self,
        rotxn: &RoTxn,
        main_hash: bitcoin::BlockHash,
    ) -> Result<Option<Option<BlockHash>>, Error> {
        let commitments = self.main_bmm_commitments.get(rotxn, &main_hash)?;
        Ok(commitments)
    }

    pub fn get_main_bmm_commitment(
        &self,
        rotxn: &RoTxn,
        main_hash: bitcoin::BlockHash,
    ) -> Result<Option<BlockHash>, Error> {
        self.try_get_main_bmm_commitment(rotxn, main_hash)?
            .ok_or(Error::NoMainBmmCommitments(main_hash))
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
                .get(rotxn, &block_hash)
                .map_err(Error::from)
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

    pub fn try_get_main_header(
        &self,
        rotxn: &RoTxn,
        block_hash: bitcoin::BlockHash,
    ) -> Result<Option<BitcoinHeader>, Error> {
        let header = self.main_headers.get(rotxn, &block_hash)?;
        Ok(header)
    }

    fn get_main_header(
        &self,
        rotxn: &RoTxn,
        block_hash: bitcoin::BlockHash,
    ) -> Result<BitcoinHeader, Error> {
        self.try_get_main_header(rotxn, block_hash)?
            .ok_or(Error::NoMainHeader(block_hash))
    }

    pub fn try_get_main_successors(
        &self,
        rotxn: &RoTxn,
        block_hash: bitcoin::BlockHash,
    ) -> Result<Option<HashSet<bitcoin::BlockHash>>, Error> {
        let successors = self.main_successors.get(rotxn, &block_hash)?;
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

    pub fn try_get_successors(
        &self,
        rotxn: &RoTxn,
        block_hash: BlockHash,
    ) -> Result<Option<HashSet<BlockHash>>, Error> {
        let successors = self.successors.get(rotxn, &block_hash)?;
        Ok(successors)
    }

    pub fn get_successors(
        &self,
        rotxn: &RoTxn,
        block_hash: BlockHash,
    ) -> Result<HashSet<BlockHash>, Error> {
        self.try_get_successors(rotxn, block_hash)?
            .ok_or(Error::NoBlockHash(block_hash))
    }

    pub fn try_get_total_work(
        &self,
        rotxn: &RoTxn,
        block_hash: bitcoin::BlockHash,
    ) -> Result<Option<bitcoin::Work>, Error> {
        let total_work = self.total_work.get(rotxn, &block_hash)?;
        Ok(total_work)
    }

    pub fn get_total_work(
        &self,
        rotxn: &RoTxn,
        block_hash: bitcoin::BlockHash,
    ) -> Result<bitcoin::Work, Error> {
        self.try_get_total_work(rotxn, block_hash)?
            .ok_or(Error::NoMainHeader(block_hash))
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
                let parent = self.get_header(rotxn, block_hash)?.prev_side_hash;
                return Ok(parent);
            } else {
                let exp_ancestor_index = u32::ilog2(n) - 1;
                block_hash = self
                    .exponential_ancestors
                    .get(rotxn, &block_hash)?
                    .unwrap()[exp_ancestor_index as usize];
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
                let parent =
                    self.get_main_header(rotxn, block_hash)?.prev_blockhash;
                return Ok(parent);
            } else {
                let exp_ancestor_index = u32::ilog2(n) - 1;
                block_hash = self
                    .exponential_main_ancestors
                    .get(rotxn, &block_hash)?
                    .unwrap()[exp_ancestor_index as usize];
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
        if block_hash == BlockHash::default() {
            return Ok(Vec::new());
        }
        let header = self.get_header(rotxn, block_hash)?;
        let mut res =
            self.exponential_ancestors.get(rotxn, &block_hash)?.unwrap();
        res.reverse();
        res.push(header.prev_side_hash);
        res.reverse();
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
        self.accumulators.put(rwtxn, &block_hash, accumulator)?;
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
        self.bodies.put(rwtxn, &block_hash, body)?;
        Ok(())
    }

    /// Store deposit info for a block
    pub fn put_deposits(
        &self,
        rwtxn: &mut RwTxn,
        block_hash: bitcoin::BlockHash,
        mut deposits: Vec<DepositInfo>,
    ) -> Result<(), Error> {
        deposits.sort_by_key(|deposit| deposit.tx_index);
        if !deposits
            .iter()
            .all(|deposit| deposit.block_hash == block_hash)
        {
            return Err(Error::DepositInvalidMainBlockHash);
        };
        self.deposits.put(rwtxn, &block_hash, &deposits)?;
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
        let Some(prev_height) =
            self.try_get_height(rwtxn, header.prev_side_hash)?
        else {
            return Err(Error::InvalidPrevSideHash);
        };
        let height = prev_height + 1;
        let block_hash = header.hash();
        self.block_hash_to_height.put(rwtxn, &block_hash, &height)?;
        self.headers.put(rwtxn, &block_hash, header)?;
        // Add to successors for predecessor
        {
            let mut pred_successors =
                self.get_successors(rwtxn, header.prev_side_hash)?;
            pred_successors.insert(block_hash);
            self.successors.put(
                rwtxn,
                &header.prev_side_hash,
                &pred_successors,
            )?;
        }
        // Store successors
        {
            let successors = self
                .try_get_successors(rwtxn, block_hash)?
                .unwrap_or_default();
            self.successors.put(rwtxn, &block_hash, &successors)?;
        }
        // populate exponential ancestors
        let mut exponential_ancestors = Vec::<BlockHash>::new();
        if height >= 2 {
            let grandparent =
                self.get_nth_ancestor(rwtxn, header.prev_side_hash, 1)?;
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
        self.exponential_ancestors.put(
            rwtxn,
            &block_hash,
            &exponential_ancestors,
        )?;
        // Populate BMM verifications
        {
            let mut bmm_results = self.get_bmm_results(rwtxn, block_hash)?;
            let parent_bmm_results =
                self.get_bmm_results(rwtxn, header.prev_side_hash)?;
            let main_blocks =
                self.get_main_successors(rwtxn, header.prev_main_hash)?;
            for main_block in main_blocks {
                let Some(commitment) =
                    self.get_main_bmm_commitment(rwtxn, main_block)?
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
                let main_header = self.get_main_header(rwtxn, main_block)?;
                if header.prev_main_hash != main_header.prev_blockhash {
                    tracing::trace!(%block_hash, "Failed BMM @ {main_block}: should be impossible?");
                    bmm_results.insert(main_block, BmmResult::Failed);
                    continue;
                }
                if header.prev_side_hash == BlockHash::default() {
                    tracing::trace!(%block_hash, "Verified BMM @ {main_block}: no parent");
                    bmm_results.insert(main_block, BmmResult::Verified);
                    continue;
                }
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
            self.bmm_results.put(rwtxn, &block_hash, &bmm_results)?;
        }
        Ok(())
    }

    /// All ancestors MUST be present.
    /// Mainchain blocks MUST be present in `main_headers`.
    pub fn put_main_bmm_commitment(
        &self,
        rwtxn: &mut RwTxn,
        main_hash: bitcoin::BlockHash,
        commitment: Option<BlockHash>,
    ) -> Result<(), Error> {
        let main_header = self.get_main_header(rwtxn, main_hash)?;
        if main_header.prev_blockhash != bitcoin::BlockHash::all_zeros() {
            let _ = self
                .get_main_bmm_commitment(rwtxn, main_header.prev_blockhash)?;
        }
        self.main_bmm_commitments
            .put(rwtxn, &main_hash, &commitment)?;
        let Some(commitment) = commitment else {
            return Ok(());
        };
        let Some(header) = self.try_get_header(rwtxn, commitment)? else {
            return Ok(());
        };
        let bmm_result = if header.prev_main_hash != main_header.prev_blockhash
        {
            BmmResult::Failed
        } else if header.prev_side_hash == BlockHash::default() {
            BmmResult::Verified
        } else {
            // Check if there is a valid BMM commitment to the parent in the
            // main ancestry
            let parent_bmm_results =
                self.get_bmm_results(rwtxn, header.prev_side_hash)?;
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
        };
        let mut bmm_results = self.get_bmm_results(rwtxn, commitment)?;
        bmm_results.insert(main_hash, bmm_result);
        self.bmm_results.put(rwtxn, &commitment, &bmm_results)?;
        Ok(())
    }

    pub fn put_main_header(
        &self,
        rwtxn: &mut RwTxn,
        header: &BitcoinHeader,
    ) -> Result<(), Error> {
        if self
            .try_get_main_header(rwtxn, header.prev_blockhash)?
            .is_none()
            && header.prev_blockhash != bitcoin::BlockHash::all_zeros()
        {
            return Err(Error::NoMainHeader(header.prev_blockhash));
        }
        let block_hash = header.hash;
        let prev_height = self.get_main_height(rwtxn, header.prev_blockhash)?;
        let height = prev_height + 1;
        let total_work =
            if header.prev_blockhash != bitcoin::BlockHash::all_zeros() {
                let prev_work =
                    self.get_total_work(rwtxn, header.prev_blockhash)?;
                prev_work + header.work()
            } else {
                header.work()
            };
        self.main_block_hash_to_height
            .put(rwtxn, &block_hash, &height)?;
        self.main_headers.put(rwtxn, &block_hash, header)?;
        self.total_work.put(rwtxn, &block_hash, &total_work)?;
        // Add to successors for predecessor
        {
            let mut pred_successors =
                self.get_main_successors(rwtxn, header.prev_blockhash)?;
            pred_successors.insert(block_hash);
            self.main_successors.put(
                rwtxn,
                &header.prev_blockhash,
                &pred_successors,
            )?;
        }
        // Store successors
        {
            let successors = self
                .try_get_main_successors(rwtxn, block_hash)?
                .unwrap_or_default();
            self.main_successors.put(rwtxn, &block_hash, &successors)?;
        }
        // populate exponential ancestors
        let mut exponential_ancestors = Vec::<bitcoin::BlockHash>::new();
        if height >= 2 {
            let grandparent =
                self.get_nth_main_ancestor(rwtxn, header.prev_blockhash, 1)?;
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
        self.exponential_main_ancestors.put(
            rwtxn,
            &block_hash,
            &exponential_ancestors,
        )?;
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
            block_hash,
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
        ancestor: BlockHash,
    ) -> Result<Vec<BlockHash>, Error> {
        // TODO: check that ancestor is nth ancestor of block_hash
        let mut res: Vec<BlockHash> = self
            .ancestors(rotxn, block_hash)
            .take_while(|block_hash| Ok(*block_hash != ancestor))
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
                let header = self.get_main_header(rotxn, block_hash)?;
                block_hash = header.prev_blockhash;
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
    ) -> Result<BlockHash, Error> {
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
            return Ok(block_hash0);
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
                        if tip0_ancestor == BlockHash::default() {
                            return Ok(Some(0));
                        }
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
                    })?
                    .unwrap();
                if tip0_ancestor_height >= height1 {
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
                        if tip1_ancestor == BlockHash::default() {
                            return Ok(Some(0));
                        }
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
                    })?
                    .unwrap();
                if tip1_ancestor_height < height0 {
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
                        if tip0_ancestor == BlockHash::default() {
                            return Ok(Some((0, None)));
                        }
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
                        Ok(Some((height, Some(work))))
                    })?
                    .unwrap();
                let (tip1_ancestor_height, tip1_ancestor_work) = self
                    .ancestors(rotxn, block_hash1)
                    .find_map(|tip1_ancestor| {
                        if tip1_ancestor == BlockHash::default() {
                            return Ok(Some((0, None)));
                        }
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
                        Ok(Some((height, Some(work))))
                    })?
                    .unwrap();
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
    block_hash: BlockHash,
}

impl<'a, 'rotxn> FallibleIterator for Ancestors<'a, 'rotxn> {
    type Item = BlockHash;
    type Error = Error;

    fn next(&mut self) -> Result<Option<Self::Item>, Self::Error> {
        if self.block_hash == BlockHash::default() {
            Ok(None)
        } else {
            let res = self.block_hash;
            let header =
                self.archive.get_header(self.rotxn, self.block_hash)?;
            self.block_hash = header.prev_side_hash;
            Ok(Some(res))
        }
    }
}
