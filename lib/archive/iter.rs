use fallible_iterator::FallibleIterator;
use sneed::RoTxn;

use crate::{
    archive::{Archive, Error},
    types::{BlockHash, Header},
};

/// Return a fallible iterator over ancestor headers of a block,
/// starting with the specified block.
/// created by [`Archive::ancestor_headers`]
pub struct AncestorHeaders<'a, 'rotxn> {
    pub(in crate::archive) archive: &'a Archive,
    pub(in crate::archive) rotxn: &'a RoTxn<'rotxn>,
    pub(in crate::archive) block_hash: Option<BlockHash>,
}

impl FallibleIterator for AncestorHeaders<'_, '_> {
    type Item = (BlockHash, Header);
    type Error = Error;

    fn next(&mut self) -> Result<Option<Self::Item>, Self::Error> {
        match self.block_hash {
            None => Ok(None),
            Some(block_hash) => {
                let header = self.archive.get_header(self.rotxn, block_hash)?;
                self.block_hash = header.prev_side_hash;
                Ok(Some((block_hash, header)))
            }
        }
    }
}

/// Return a fallible iterator over ancestors of a block,
/// starting with the specified block.
/// created by [`Archive::ancestors`]
#[repr(transparent)]
pub struct Ancestors<'a, 'rotxn> {
    pub(in crate::archive) inner: AncestorHeaders<'a, 'rotxn>,
}

impl FallibleIterator for Ancestors<'_, '_> {
    type Item = BlockHash;
    type Error = Error;

    fn next(&mut self) -> Result<Option<Self::Item>, Self::Error> {
        self.inner
            .next()
            .map(|item| item.map(|(block_hash, _)| block_hash))
    }
}

struct AncestorsRevInner {
    end_height: u32,
    /// Buffer of ancestors, newer-to-older.
    buffer: Vec<(BlockHash, Header)>,
}

/// Return a Fallible iterator over ancestor headers of a block,
/// starting from the specified block height,
/// and ending with the specified block.
pub(in crate::archive) struct AncestorsRev<'a, 'rotxn> {
    archive: &'a Archive,
    rotxn: &'a RoTxn<'rotxn>,
    /// Inclusive.
    /// None indicates that the iterator is done.
    /// MUST be Some(_) on construction.
    batch_start_height: Option<u32>,
    block_hash: BlockHash,
    inner: Option<AncestorsRevInner>,
}

impl<'a, 'rotxn> AncestorsRev<'a, 'rotxn> {
    pub(in crate::archive) fn new(
        archive: &'a Archive,
        rotxn: &'a RoTxn<'rotxn>,
        block_hash: BlockHash,
        start_height: u32,
    ) -> Self {
        Self {
            archive,
            rotxn,
            batch_start_height: Some(start_height),
            block_hash,
            inner: None,
        }
    }
}

impl FallibleIterator for AncestorsRev<'_, '_> {
    type Item = (BlockHash, Header);
    type Error = Error;

    fn next(&mut self) -> Result<Option<Self::Item>, Self::Error> {
        // Amortize get_nth_ancestor lookups by batching
        const MAX_BATCH_SIZE: u32 = 32;
        let inner = match self.inner.as_mut() {
            Some(inner) => inner,
            None => self.inner.insert(AncestorsRevInner {
                end_height: self
                    .archive
                    .get_height(self.rotxn, self.block_hash)?,
                buffer: Vec::with_capacity(MAX_BATCH_SIZE as usize),
            }),
        };
        if let Some(item) = inner.buffer.pop() {
            Ok(Some(item))
        } else if let Some(batch_start_height) = self.batch_start_height {
            let Some(height_diff) =
                inner.end_height.checked_sub(batch_start_height)
            else {
                return Ok(None);
            };
            // Offset from batch start height
            let ancestor_offset = height_diff.min(MAX_BATCH_SIZE - 1);
            let batch_size = ancestor_offset + 1;
            let nth_ancestor = height_diff - ancestor_offset;
            let ancestor = self.archive.get_nth_ancestor(
                self.rotxn,
                self.block_hash,
                nth_ancestor,
            )?;
            let () = self
                .archive
                .ancestor_headers(self.rotxn, ancestor)
                .take(batch_size as usize)
                .for_each(|item| {
                    inner.buffer.push(item);
                    Ok(())
                })?;
            self.batch_start_height = if let Some(batch_start_height) =
                batch_start_height.checked_add(batch_size)
                && batch_start_height <= inner.end_height
            {
                Some(batch_start_height)
            } else {
                None
            };
            Ok(inner.buffer.pop())
        } else {
            Ok(None)
        }
    }
}

pub mod mainchain_ancestors_rev {
    use bitcoin::BlockHash;
    use fallible_iterator::FallibleIterator;
    use sneed::RoTxn;

    use crate::{
        archive::{Archive, Error},
        types::proto::mainchain::BlockHeaderInfo,
    };

    struct Inner {
        end_height: u32,
        /// Buffer of ancestors, newer-to-older.
        buffer: Vec<BlockHeaderInfo>,
    }

    /// A Fallible iterator over ancestor headers of a block,
    /// starting from the specified block height,
    /// and ending with the specified block.
    pub(in crate::archive) struct Iter<'a, 'rotxn> {
        archive: &'a Archive,
        rotxn: &'a RoTxn<'rotxn>,
        /// Inclusive.
        /// None indicates that the iterator is done.
        /// MUST be Some(_) on construction.
        batch_start_height: Option<u32>,
        end_block_hash: BlockHash,
        inner: Option<Inner>,
    }

    impl<'a, 'rotxn> Iter<'a, 'rotxn> {
        pub(in crate::archive) fn new(
            archive: &'a Archive,
            rotxn: &'a RoTxn<'rotxn>,
            end_block_hash: BlockHash,
            start_height: u32,
        ) -> Self {
            Self {
                archive,
                rotxn,
                batch_start_height: Some(start_height),
                end_block_hash,
                inner: None,
            }
        }
    }

    impl FallibleIterator for Iter<'_, '_> {
        type Item = BlockHeaderInfo;
        type Error = Error;

        fn next(&mut self) -> Result<Option<Self::Item>, Self::Error> {
            // Amortize get_nth_ancestor lookups by batching
            const MAX_BATCH_SIZE: u32 = 32;
            let inner = match self.inner.as_mut() {
                Some(inner) => inner,
                None => self.inner.insert(Inner {
                    end_height: self
                        .archive
                        .get_main_header_info(self.rotxn, &self.end_block_hash)?
                        .height,
                    buffer: Vec::with_capacity(MAX_BATCH_SIZE as usize),
                }),
            };
            if let Some(item) = inner.buffer.pop() {
                Ok(Some(item))
            } else if let Some(batch_start_height) = self.batch_start_height {
                let Some(height_diff) =
                    inner.end_height.checked_sub(batch_start_height)
                else {
                    return Ok(None);
                };
                // Offset from batch start height
                let ancestor_offset = height_diff.min(MAX_BATCH_SIZE - 1);
                let batch_size = ancestor_offset + 1;
                let nth_ancestor = height_diff - ancestor_offset;
                let ancestor = self.archive.get_nth_main_ancestor(
                    self.rotxn,
                    self.end_block_hash,
                    nth_ancestor,
                )?;
                let () = self
                    .archive
                    .main_ancestor_header_infos(self.rotxn, ancestor)
                    .take(batch_size as usize)
                    .for_each(|item| {
                        inner.buffer.push(item);
                        Ok(())
                    })?;
                self.batch_start_height = if let Some(batch_start_height) =
                    batch_start_height.checked_add(batch_size)
                    && batch_start_height <= inner.end_height
                {
                    Some(batch_start_height)
                } else {
                    None
                };
                Ok(inner.buffer.pop())
            } else {
                Ok(None)
            }
        }
    }
}
pub(in crate::archive) use mainchain_ancestors_rev::Iter as MainchainAncestorsRev;
