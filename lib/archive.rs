use std::{cmp::Ordering, collections::HashSet};

use bip300301::{
    bitcoin::{self, hashes::Hash},
    DepositInfo, Header as BitcoinHeader,
};
use fallible_iterator::FallibleIterator;
use heed::{types::SerdeBincode, Database, RoTxn, RwTxn};

use crate::types::{Accumulator, BlockHash, Body, Header};

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
    #[error("no block body with hash {0}")]
    NoBody(BlockHash),
    #[error("no BMM verification result with for block {0}")]
    NoBmmVerification(BlockHash),
    #[error("no deposits info for block {0}")]
    NoDepositsInfo(bitcoin::BlockHash),
    #[error("no header with hash {0}")]
    NoHeader(BlockHash),
    #[error("no height info for block hash {0}")]
    NoHeight(BlockHash),
    #[error("no mainchain header with hash {0}")]
    NoMainHeader(bitcoin::BlockHash),
    #[error("no height info for mainchain block hash {0}")]
    NoMainHeight(bitcoin::BlockHash),
}

#[derive(Clone)]
pub struct Archive {
    accumulators: Database<SerdeBincode<BlockHash>, SerdeBincode<Accumulator>>,
    block_hash_to_height: Database<SerdeBincode<BlockHash>, SerdeBincode<u32>>,
    /// BMM verification status for each header.
    /// A status of false indicates that verification failed.
    /// All ancestors of any block should always be present.
    bmm_verifications: Database<SerdeBincode<BlockHash>, SerdeBincode<bool>>,
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
    /// Mainchain headers. All ancestors of any header should always be present
    main_headers:
        Database<SerdeBincode<bitcoin::BlockHash>, SerdeBincode<BitcoinHeader>>,
    /// Successor blocks. ALL known block hashes, INCLUDING the zero hash,
    /// MUST be present.
    successors:
        Database<SerdeBincode<BlockHash>, SerdeBincode<HashSet<BlockHash>>>,
    /// Total work for mainchain headers.
    /// All ancestors of any block should always be present
    total_work:
        Database<SerdeBincode<bitcoin::BlockHash>, SerdeBincode<bitcoin::Work>>,
}

impl Archive {
    pub const NUM_DBS: u32 = 12;

    pub fn new(env: &heed::Env) -> Result<Self, Error> {
        let mut rwtxn = env.write_txn()?;
        let accumulators =
            env.create_database(&mut rwtxn, Some("accumulators"))?;
        let block_hash_to_height =
            env.create_database(&mut rwtxn, Some("hash_to_height"))?;
        let bmm_verifications =
            env.create_database(&mut rwtxn, Some("bmm_verifications"))?;
        let bodies = env.create_database(&mut rwtxn, Some("bodies"))?;
        let deposits = env.create_database(&mut rwtxn, Some("deposits"))?;
        let exponential_ancestors =
            env.create_database(&mut rwtxn, Some("exponential_ancestors"))?;
        let exponential_main_ancestors = env
            .create_database(&mut rwtxn, Some("exponential_main_ancestors"))?;
        let headers = env.create_database(&mut rwtxn, Some("headers"))?;
        let main_block_hash_to_height =
            env.create_database(&mut rwtxn, Some("main_hash_to_height"))?;
        let main_headers =
            env.create_database(&mut rwtxn, Some("main_headers"))?;
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
            bmm_verifications,
            bodies,
            deposits,
            exponential_ancestors,
            exponential_main_ancestors,
            headers,
            main_block_hash_to_height,
            main_headers,
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

    pub fn try_get_bmm_verification(
        &self,
        rotxn: &RoTxn,
        block_hash: BlockHash,
    ) -> Result<Option<bool>, Error> {
        if block_hash == BlockHash::default() {
            Ok(Some(true))
        } else {
            self.bmm_verifications
                .get(rotxn, &block_hash)
                .map_err(Error::from)
        }
    }

    pub fn get_bmm_verification(
        &self,
        rotxn: &RoTxn,
        block_hash: BlockHash,
    ) -> Result<bool, Error> {
        self.try_get_bmm_verification(rotxn, block_hash)?
            .ok_or(Error::NoBmmVerification(block_hash))
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

    /// Store a BMM verification result
    pub fn put_bmm_verification(
        &self,
        rwtxn: &mut RwTxn,
        block_hash: BlockHash,
        verification_result: bool,
    ) -> Result<(), Error> {
        self.bmm_verifications
            .put(rwtxn, &block_hash, &verification_result)?;
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
        self.successors.put(rwtxn, &block_hash, &HashSet::new())?;
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
        block_hash0: BlockHash,
        block_hash1: BlockHash,
    ) -> Result<Option<BlockHash>, Error> {
        let height0 = self.get_height(rotxn, block_hash0)?;
        let height1 = self.get_height(rotxn, block_hash1)?;
        match (height0, height1) {
            (0, 0) => return Ok(None),
            (0, _) => return Ok(Some(block_hash1)),
            (_, 0) => return Ok(Some(block_hash0)),
            (_, _) => (),
        }
        let header0 = self.get_header(rotxn, block_hash0)?;
        let header1 = self.get_header(rotxn, block_hash1)?;
        if self.shared_mainchain_lineage(
            rotxn,
            header0.prev_main_hash,
            header1.prev_main_hash,
        )? {
            match height0.cmp(&height1) {
                Ordering::Less => Ok(Some(block_hash1)),
                Ordering::Greater => Ok(Some(block_hash0)),
                Ordering::Equal => {
                    let work0 =
                        self.get_total_work(rotxn, header0.prev_main_hash)?;
                    let work1 =
                        self.get_total_work(rotxn, header1.prev_main_hash)?;
                    match work0.cmp(&work1) {
                        Ordering::Less => Ok(Some(block_hash1)),
                        Ordering::Greater => Ok(Some(block_hash0)),
                        Ordering::Equal => Ok(None),
                    }
                }
            }
        } else {
            let work0 = self.get_total_work(rotxn, header0.prev_main_hash)?;
            let work1 = self.get_total_work(rotxn, header1.prev_main_hash)?;
            match (height0.cmp(&height1), work0.cmp(&work1)) {
                (Ordering::Less, Ordering::Equal) => Ok(Some(block_hash1)),
                (Ordering::Greater, Ordering::Equal) => Ok(Some(block_hash0)),
                (Ordering::Less | Ordering::Equal, Ordering::Less) => {
                    Ok(Some(block_hash1))
                }
                (Ordering::Greater | Ordering::Equal, Ordering::Greater) => {
                    Ok(Some(block_hash0))
                }
                (Ordering::Less, Ordering::Greater)
                | (Ordering::Greater, Ordering::Less)
                | (Ordering::Equal, Ordering::Equal) => {
                    let common_mainchain_ancestor = self
                        .last_common_main_ancestor(
                            rotxn,
                            header0.prev_main_hash,
                            header1.prev_main_hash,
                        )?;
                    let common_mainchain_ancestor_height =
                        self.get_main_height(rotxn, common_mainchain_ancestor)?;
                    let height_before_common_mainchain_ancestor0 = self
                        .ancestors(rotxn, block_hash0)
                        .find_map(|block_hash| {
                            if block_hash == BlockHash::default() {
                                return Ok(Some(0));
                            };
                            let header = self.get_header(rotxn, block_hash)?;
                            let main_height = self.get_main_height(
                                rotxn,
                                header.prev_main_hash,
                            )?;
                            if main_height > common_mainchain_ancestor_height {
                                return Ok(None);
                            };
                            let height = self.get_height(rotxn, block_hash)?;
                            Ok(Some(height))
                        })?
                        .unwrap();
                    let height_before_common_mainchain_ancestor1 = self
                        .ancestors(rotxn, block_hash1)
                        .find_map(|block_hash| {
                            if block_hash == BlockHash::default() {
                                return Ok(Some(0));
                            };
                            let header = self.get_header(rotxn, block_hash)?;
                            let main_height = self.get_main_height(
                                rotxn,
                                header.prev_main_hash,
                            )?;
                            if main_height > common_mainchain_ancestor_height {
                                return Ok(None);
                            };
                            let height = self.get_height(rotxn, block_hash)?;
                            Ok(Some(height))
                        })?
                        .unwrap();
                    match (
                        work0.cmp(&work1),
                        height_before_common_mainchain_ancestor0
                            .cmp(&height_before_common_mainchain_ancestor1),
                    ) {
                        (Ordering::Less, Ordering::Less | Ordering::Equal) => {
                            Ok(Some(block_hash1))
                        }
                        (Ordering::Less, Ordering::Greater) => Ok(None),
                        (
                            Ordering::Greater,
                            Ordering::Greater | Ordering::Equal,
                        ) => Ok(Some(block_hash0)),
                        (Ordering::Greater, Ordering::Less) => Ok(None),
                        (Ordering::Equal, Ordering::Less) => {
                            Ok(Some(block_hash1))
                        }
                        (Ordering::Equal, Ordering::Greater) => {
                            Ok(Some(block_hash0))
                        }
                        (Ordering::Equal, Ordering::Equal) => Ok(None),
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
