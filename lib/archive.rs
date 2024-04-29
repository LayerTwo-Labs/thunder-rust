use std::cmp::Ordering;

use fallible_iterator::FallibleIterator;
use heed::{types::SerdeBincode, Database, RoTxn, RwTxn};

use crate::types::{Accumulator, BlockHash, Body, Header};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("heed error")]
    Heed(#[from] heed::Error),
    #[error("invalid previous side hash")]
    InvalidPrevSideHash,
    #[error("invalid merkle root")]
    InvalidMerkleRoot,
    #[error("no accumulator for block {0}")]
    NoAccumulator(BlockHash),
    #[error("no block with hash {0}")]
    NoBlock(BlockHash),
    #[error("no header with hash {0}")]
    NoHeader(BlockHash),
    #[error("no height info hash {0}")]
    NoHeight(BlockHash),
}

#[derive(Clone)]
pub struct Archive {
    accumulators: Database<SerdeBincode<BlockHash>, SerdeBincode<Accumulator>>,
    headers: Database<SerdeBincode<BlockHash>, SerdeBincode<Header>>,
    bodies: Database<SerdeBincode<BlockHash>, SerdeBincode<Body>>,
    hash_to_height: Database<SerdeBincode<BlockHash>, SerdeBincode<u32>>,
}

impl Archive {
    pub const NUM_DBS: u32 = 4;

    pub fn new(env: &heed::Env) -> Result<Self, Error> {
        let accumulators = env.create_database(Some("accumulators"))?;
        let headers = env.create_database(Some("headers"))?;
        let bodies = env.create_database(Some("bodies"))?;
        let hash_to_height = env.create_database(Some("hash_to_height"))?;
        Ok(Self {
            accumulators,
            headers,
            bodies,
            hash_to_height,
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
            .ok_or(Error::NoBlock(block_hash))
    }

    pub fn try_get_height(
        &self,
        rotxn: &RoTxn,
        block_hash: BlockHash,
    ) -> Result<Option<u32>, Error> {
        if block_hash == BlockHash::default() {
            Ok(Some(0))
        } else {
            self.hash_to_height
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
        self.headers.put(rwtxn, &block_hash, header)?;
        self.hash_to_height.put(rwtxn, &block_hash, &height)?;
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

    /// Find the last common ancestor of two blocks, if headers for both exist
    pub fn last_common_ancestor(
        &self,
        rotxn: &RoTxn,
        mut block_hash0: BlockHash,
        mut block_hash1: BlockHash,
    ) -> Result<BlockHash, Error> {
        let mut height0 = self.get_height(rotxn, block_hash0)?;
        let mut height1 = self.get_height(rotxn, block_hash1)?;
        let mut header0 = self.try_get_header(rotxn, block_hash0)?;
        let mut header1 = self.try_get_header(rotxn, block_hash1)?;
        // Find respective ancestors of block_hash0 and block_hash1 with height
        // equal to min(height0, height1)
        loop {
            match height0.cmp(&height1) {
                Ordering::Less => {
                    block_hash1 = header1.unwrap().prev_side_hash;
                    header1 = self.try_get_header(rotxn, block_hash1)?;
                    height1 -= 1;
                }
                Ordering::Greater => {
                    block_hash0 = header0.unwrap().prev_side_hash;
                    header0 = self.try_get_header(rotxn, block_hash0)?;
                    height0 -= 1;
                }
                Ordering::Equal => {
                    if block_hash0 == block_hash1 {
                        return Ok(block_hash0);
                    } else {
                        block_hash0 = header0.unwrap().prev_side_hash;
                        block_hash1 = header1.unwrap().prev_side_hash;
                        header0 = self.try_get_header(rotxn, block_hash0)?;
                        header1 = self.try_get_header(rotxn, block_hash1)?;
                        height0 -= 1;
                        height1 -= 1;
                    }
                }
            };
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
