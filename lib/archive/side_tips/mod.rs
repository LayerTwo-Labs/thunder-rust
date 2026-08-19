//! State as of the current mainchain tip, excluding sidechain state

use bitcoin::hashes::Hash as _;
use heed::types::SerdeBincode;
use serde::{Deserialize, Serialize};
use serde_with::serde_as;
use sneed::{
    DatabaseUnique, RoDatabaseUnique, RoTxn, RwTxn, UnitKey,
    db::error as db_error,
};

use crate::types::proto::mainchain::BlockHeaderInfo;

pub mod error;
pub use error::Error;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct MainchainTip {
    cumulative_work: bitcoin::Work,
    pub(crate) tip_info: Option<BlockHeaderInfo>,
}

impl MainchainTip {
    pub fn block_hash(&self) -> bitcoin::BlockHash {
        match self.tip_info {
            Some(info) => info.block_hash,
            None => bitcoin::BlockHash::all_zeros(),
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[repr(transparent)]
#[serde(transparent)]
struct BigEndianU32([u8; 4]);

impl From<BigEndianU32> for u32 {
    #[inline(always)]
    fn from(value: BigEndianU32) -> Self {
        u32::from_be_bytes(value.0)
    }
}

impl From<u32> for BigEndianU32 {
    #[inline(always)]
    fn from(value: u32) -> Self {
        Self(value.to_be_bytes())
    }
}

#[serde_as]
#[derive(
    Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize,
)]
/// Ordered by sidechain height, and then work.
pub struct SidechainTipInfo {
    #[serde_as(as = "serde_with::FromInto<BigEndianU32>")]
    sidechain_height: u32,
    pub(crate) work: bitcoin::Work,
    /// Main block hash with the least amount of work
    pub(crate) main_block_hash: bitcoin::BlockHash,
}

pub struct SidechainTip {
    pub block_hash: crate::types::BlockHash,
    pub info: SidechainTipInfo,
}

#[derive(Clone, Copy, Debug)]
pub struct SidechainHeaderData {
    pub prev_side_hash: Option<crate::types::BlockHash>,
    pub prev_main_hash: bitcoin::BlockHash,
}

impl From<&crate::types::Header> for SidechainHeaderData {
    fn from(header: &crate::types::Header) -> Self {
        Self {
            prev_side_hash: header.prev_side_hash,
            prev_main_hash: header.prev_main_hash,
        }
    }
}

impl From<crate::types::Header> for SidechainHeaderData {
    fn from(header: crate::types::Header) -> Self {
        Self {
            prev_side_hash: header.prev_side_hash,
            prev_main_hash: header.prev_main_hash,
        }
    }
}

/// BMM commitment for a sidechain tip.
#[derive(Debug)]
pub(crate) struct BmmCommitment {
    pub sidechain_block_hash: crate::types::BlockHash,
    pub sidechain_header_data: SidechainHeaderData,
}

#[derive(Clone)]
pub struct SideTips {
    mainchain_tip: DatabaseUnique<UnitKey, SerdeBincode<MainchainTip>>,
    /// Tips for this sidechain.
    /// Sidechain headers may not be available.
    /// If sidechain headers are available, there must be a valid BMM in the
    /// active chain for the parent block.
    sidechain_tips: DatabaseUnique<
        SerdeBincode<crate::types::BlockHash>,
        SerdeBincode<SidechainTipInfo>,
    >,
    /// Tips for this sidechain, sorted by height, then work.
    ordered_sidechain_tips: DatabaseUnique<
        SerdeBincode<SidechainTipInfo>,
        SerdeBincode<crate::types::BlockHash>,
    >,
}

impl SideTips {
    pub const NUM_DBS: u32 = 3;

    pub(crate) fn create<Tls>(
        env: &sneed::Env<Tls>,
        rwtxn: &mut RwTxn,
    ) -> Result<Self, error::Create> {
        let mainchain_tip =
            DatabaseUnique::create(env, rwtxn, "mainchain_tip")?;
        if mainchain_tip.try_get(rwtxn, &())?.is_none() {
            mainchain_tip.put(
                rwtxn,
                &(),
                &MainchainTip {
                    cumulative_work: bitcoin::Work::from_le_bytes([0; 32]),
                    tip_info: None,
                },
            )?;
        }
        let sidechain_tips =
            DatabaseUnique::create(env, rwtxn, "sidechain_tips")?;
        let ordered_sidechain_tips =
            DatabaseUnique::create(env, rwtxn, "ordered_sidechain_tips")?;
        Ok(Self {
            mainchain_tip,
            sidechain_tips,
            ordered_sidechain_tips,
        })
    }

    pub fn sidechain_tips(
        &self,
    ) -> &RoDatabaseUnique<
        SerdeBincode<crate::types::BlockHash>,
        SerdeBincode<SidechainTipInfo>,
    > {
        &self.sidechain_tips
    }

    pub fn ordered_sidechain_tips(
        &self,
    ) -> &RoDatabaseUnique<
        SerdeBincode<SidechainTipInfo>,
        SerdeBincode<crate::types::BlockHash>,
    > {
        &self.ordered_sidechain_tips
    }

    pub fn get_mainchain_tip(
        &self,
        rotxn: &RoTxn,
    ) -> Result<MainchainTip, db_error::Get> {
        self.mainchain_tip.get(rotxn, &())
    }

    pub fn best_side_tip(
        &self,
        rotxn: &RoTxn,
    ) -> Result<Option<SidechainTip>, db_error::Last> {
        if let Some((info, block_hash)) =
            self.ordered_sidechain_tips.last(rotxn)?
        {
            Ok(Some(SidechainTip { block_hash, info }))
        } else {
            Ok(None)
        }
    }

    /// Connect a sidechain tip.
    /// The mainchain block hash MUST be either the current mainchain tip,
    /// or an ancestor of the current tip.
    /// The mainchain block hash MUST be the earliest mainchain block that
    /// includes a valid BMM commitment to the sidechain tip, ie the earliest
    /// mainchain block such that all ancestors of the sidechain tip also valid
    /// BMM commitments in the mainchain block's ancestry.
    /// The sidechain tip MUST be valid to re-org to.
    pub(in crate::archive) fn connect_sidechain_tip(
        &self,
        rwtxn: &mut RwTxn,
        main_block_hash: bitcoin::BlockHash,
        cumulative_work: bitcoin::Work,
        sidechain_block_hash: crate::types::BlockHash,
        sidechain_header_data: SidechainHeaderData,
    ) -> Result<(), error::ConnectSidechainTip> {
        let sidechain_height = match sidechain_header_data.prev_side_hash {
            Some(parent_hash) => {
                let Some(parent_info) =
                    self.sidechain_tips.try_get(rwtxn, &parent_hash)?
                else {
                    let err = error::MissingSidechainParent(parent_hash);
                    return Err(err.into());
                };
                parent_info.sidechain_height.saturating_add(1)
            }
            None => 0,
        };
        if !self
            .sidechain_tips
            .contains_key(rwtxn, &sidechain_block_hash)?
        {
            let side_tip_info = SidechainTipInfo {
                sidechain_height,
                work: cumulative_work,
                main_block_hash,
            };
            self.sidechain_tips.put(
                rwtxn,
                &sidechain_block_hash,
                &side_tip_info,
            )?;
            self.ordered_sidechain_tips.put(
                rwtxn,
                &side_tip_info,
                &sidechain_block_hash,
            )?;
        }
        Ok(())
    }

    /// Disconnect a sidechain tip, if it is present.
    /// The caller MUST also disconnect any descendant tips before calling any
    /// other method.
    pub(in crate::archive) unsafe fn disconnect_sidechain_tip(
        &self,
        rwtxn: &mut RwTxn,
        side_block_hash: &crate::types::BlockHash,
    ) -> Result<(), error::DisconnectSidechainTip> {
        let Some(side_tip_info) =
            self.sidechain_tips.try_get(rwtxn, side_block_hash)?
        else {
            return Ok(());
        };
        self.ordered_sidechain_tips.delete(rwtxn, &side_tip_info)?;
        self.sidechain_tips.delete(rwtxn, side_block_hash)?;
        Ok(())
    }

    /// Connect a mainchain tip.
    /// The BMM commitment MUST only be included if the sidechain tip is valid
    /// to re-org to.
    pub(crate) fn connect_mainchain_tip(
        &self,
        rwtxn: &mut RwTxn,
        header_info: BlockHeaderInfo,
        bmm_commitment: Option<BmmCommitment>,
    ) -> Result<(), error::ConnectMainchainTip> {
        let mut mainchain_tip = self.mainchain_tip.get(rwtxn, &())?;
        let expected_prev_main_hash = mainchain_tip.block_hash();
        if header_info.prev_block_hash != expected_prev_main_hash {
            return Err(error::ConnectMainchainTip::InvalidMainchainParent {
                expected: expected_prev_main_hash,
            });
        }
        if let Some(tip_info) = mainchain_tip.tip_info
            && header_info.height != tip_info.height + 1
        {
            let err = error::ConnectMainchainTip::InvalidTipHeight {
                expected: tip_info.height + 1,
            };
            return Err(err);
        }
        mainchain_tip.cumulative_work =
            mainchain_tip.cumulative_work + header_info.work;
        if let Some(bmm_commitment) = bmm_commitment {
            let () = self.connect_sidechain_tip(
                rwtxn,
                header_info.block_hash,
                mainchain_tip.cumulative_work,
                bmm_commitment.sidechain_block_hash,
                bmm_commitment.sidechain_header_data,
            )?;
        }
        mainchain_tip.tip_info = Some(header_info);
        self.mainchain_tip.put(rwtxn, &(), &mainchain_tip)?;
        Ok(())
    }

    pub(crate) fn disconnect_mainchain_tip(
        &self,
        rwtxn: &mut RwTxn,
        prev_tip_info: Option<BlockHeaderInfo>,
        bmm_commitment: Option<crate::types::BlockHash>,
    ) -> Result<(), error::DisconnectMainchainTip> {
        let MainchainTip {
            cumulative_work,
            tip_info: mainchain_tip_info,
        } = self.mainchain_tip.get(rwtxn, &())?;
        let Some(mainchain_tip_info) = mainchain_tip_info else {
            return Err(error::DisconnectMainchainTip::NoMainchainTip);
        };
        let new_mainchain_tip_info = if let Some(prev_tip_info) = prev_tip_info
        {
            if prev_tip_info.block_hash != mainchain_tip_info.prev_block_hash {
                return Err(
                    error::DisconnectMainchainTip::InvalidMainchainParent {
                        expected: mainchain_tip_info.prev_block_hash,
                    },
                );
            } else if prev_tip_info.height + 1 != mainchain_tip_info.height {
                let err = error::DisconnectMainchainTip::InvalidMainchainParentHeight {
                    tip_height: mainchain_tip_info.height,
                };
                return Err(err);
            } else if mainchain_tip_info.prev_block_hash
                != bitcoin::BlockHash::all_zeros()
            {
                Some(prev_tip_info)
            } else {
                None
            }
        } else if mainchain_tip_info.prev_block_hash
            == bitcoin::BlockHash::all_zeros()
        {
            None
        } else {
            return Err(
                error::DisconnectMainchainTip::InvalidMainchainParent {
                    expected: mainchain_tip_info.prev_block_hash,
                },
            );
        };
        if let Some(side_block_hash) = bmm_commitment {
            let Some(side_tip_info) =
                self.sidechain_tips.try_get(rwtxn, &side_block_hash)?
            else {
                let err = error::MissingSidechainTip(side_block_hash);
                return Err(err.into());
            };
            if side_tip_info.main_block_hash == mainchain_tip_info.block_hash
                && !self.ordered_sidechain_tips.delete(rwtxn, &side_tip_info)?
            {
                let err = error::MissingSidechainTip(side_block_hash);
                return Err(err.into());
            }
        }
        let new_mainchain_tip = MainchainTip {
            cumulative_work: cumulative_work - mainchain_tip_info.work,
            tip_info: new_mainchain_tip_info,
        };
        self.mainchain_tip.put(rwtxn, &(), &new_mainchain_tip)?;
        Ok(())
    }
}
