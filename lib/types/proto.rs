//! Protobuf types

use thiserror::Error;

/// Convenience alias to avoid writing out a lengthy trait bound
pub trait Transport = where
    Self: tonic::client::GrpcService<tonic::body::BoxBody>,
    Self::Error: Into<tonic::codegen::StdError>,
    Self::ResponseBody:
        tonic::codegen::Body<Data = tonic::codegen::Bytes> + Send + 'static,
    <Self::ResponseBody as tonic::codegen::Body>::Error:
        Into<tonic::codegen::StdError> + Send;

#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    Grpc(Box<tonic::Status>),
    #[error(
        "Invalid enum variant in field `{field_name}` of message `{message_name}`: `{variant_name}`"
    )]
    InvalidEnumVariant {
        field_name: String,
        message_name: String,
        variant_name: String,
    },
    #[error(
        "Invalid field value in field `{field_name}` of message `{message_name}`: `{value}`"
    )]
    InvalidFieldValue {
        field_name: String,
        message_name: String,
        value: String,
    },
    #[error(
        "Invalid value in repeated field `{field_name}` of message `{message_name}`: `{value}`"
    )]
    InvalidRepeatedValue {
        field_name: String,
        message_name: String,
        value: String,
    },
    #[error("Missing field in message `{message_name}`: `{field_name}`")]
    MissingField {
        field_name: String,
        message_name: String,
    },
    #[error(
        "Unknown enum tag in field `{field_name}` of message `{message_name}`: `{tag}`"
    )]
    UnknownEnumTag {
        field_name: String,
        message_name: String,
        tag: i32,
    },
}

impl Error {
    pub fn invalid_enum_variant<Message>(
        field_name: &str,
        variant_name: &str,
    ) -> Self
    where
        Message: prost::Name,
    {
        Self::InvalidEnumVariant {
            field_name: field_name.to_owned(),
            message_name: Message::full_name(),
            variant_name: variant_name.to_owned(),
        }
    }

    pub fn invalid_field_value<Message>(field_name: &str, value: &str) -> Self
    where
        Message: prost::Name,
    {
        Self::InvalidFieldValue {
            field_name: field_name.to_owned(),
            message_name: Message::full_name(),
            value: value.to_owned(),
        }
    }

    pub fn invalid_repeated_value<Message>(
        field_name: &str,
        value: &str,
    ) -> Self
    where
        Message: prost::Name,
    {
        Self::InvalidRepeatedValue {
            field_name: field_name.to_owned(),
            message_name: Message::full_name(),
            value: value.to_owned(),
        }
    }

    pub fn missing_field<Message>(field_name: &str) -> Self
    where
        Message: prost::Name,
    {
        Self::MissingField {
            field_name: field_name.to_owned(),
            message_name: Message::full_name(),
        }
    }
}

impl From<tonic::Status> for Error {
    fn from(err: tonic::Status) -> Self {
        Self::Grpc(Box::new(err))
    }
}

pub mod common {
    pub mod generated {
        tonic::include_proto!("cusf.common.v1");
    }

    pub use generated::{ConsensusHex, Hex, ReverseHex};

    impl ConsensusHex {
        pub fn decode<Message, T>(
            &self,
            field_name: &str,
        ) -> Result<T, super::Error>
        where
            Message: prost::Name,
            T: bitcoin::consensus::Decodable,
        {
            let Self { hex } = self;
            let hex = hex
                .as_ref()
                .ok_or_else(|| super::Error::missing_field::<Self>("hex"))?;
            bitcoin::consensus::encode::deserialize_hex(hex).map_err(|_err| {
                super::Error::invalid_field_value::<Message>(field_name, hex)
            })
        }

        /// Variant of [`Self::decode`] that returns a `tonic::Status` error
        pub fn decode_tonic<Message, T>(
            self,
            field_name: &str,
        ) -> Result<T, Box<tonic::Status>>
        where
            Message: prost::Name,
            T: bitcoin::consensus::Decodable,
        {
            self.decode::<Message, _>(field_name).map_err(|err| {
                Box::new(tonic::Status::from_error(Box::new(err)))
            })
        }

        pub fn encode<T>(value: &T) -> Self
        where
            T: bitcoin::consensus::Encodable,
        {
            let hex = bitcoin::consensus::encode::serialize_hex(value);
            Self { hex: Some(hex) }
        }
    }

    impl Hex {
        pub fn decode_bytes<Message>(
            self,
            field_name: &str,
        ) -> Result<Vec<u8>, super::Error>
        where
            Message: prost::Name,
        {
            let Self { hex } = self;
            let hex =
                hex.ok_or_else(|| super::Error::missing_field::<Self>("hex"))?;
            hex::decode(&hex).map_err(|_err| {
                super::Error::invalid_field_value::<Message>(field_name, &hex)
            })
        }

        pub fn decode<Message, T>(
            self,
            field_name: &str,
        ) -> Result<T, super::Error>
        where
            Message: prost::Name,
            T: borsh::BorshDeserialize,
        {
            let Self { hex } = self;
            let hex =
                hex.ok_or_else(|| super::Error::missing_field::<Self>("hex"))?;
            let bytes = hex::decode(&hex).map_err(|_err| {
                super::Error::invalid_field_value::<Message>(field_name, &hex)
            })?;
            T::try_from_slice(&bytes).map_err(|_err| {
                super::Error::invalid_field_value::<Message>(field_name, &hex)
            })
        }

        pub fn encode<T>(value: &T) -> Self
        where
            T: hex::ToHex,
        {
            let hex = value.encode_hex();
            Self { hex: Some(hex) }
        }
    }

    impl ReverseHex {
        pub fn decode<Message, T>(
            &self,
            field_name: &str,
        ) -> Result<T, super::Error>
        where
            Message: prost::Name,
            T: bitcoin::consensus::Decodable,
        {
            let Self { hex } = self;
            let hex = hex
                .as_ref()
                .ok_or_else(|| super::Error::missing_field::<Self>("hex"))?;
            let mut bytes = hex::decode(hex).map_err(|_| {
                super::Error::invalid_field_value::<Message>(field_name, hex)
            })?;
            bytes.reverse();
            bitcoin::consensus::deserialize(&bytes).map_err(|_err| {
                super::Error::invalid_field_value::<Message>(field_name, hex)
            })
        }

        /// Variant of [`Self::decode`] that returns a `tonic::Status` error
        pub fn decode_tonic<Message, T>(
            self,
            field_name: &str,
        ) -> Result<T, Box<tonic::Status>>
        where
            Message: prost::Name,
            T: bitcoin::consensus::Decodable,
        {
            self.decode::<Message, _>(field_name).map_err(|err| {
                Box::new(tonic::Status::from_error(Box::new(err)))
            })
        }

        pub fn encode<T>(value: &T) -> Self
        where
            T: bitcoin::consensus::Encodable,
        {
            let mut bytes = bitcoin::consensus::encode::serialize(value);
            bytes.reverse();
            Self {
                hex: Some(hex::encode(bytes)),
            }
        }
    }
}

pub mod mainchain {
    use std::str::FromStr;

    use bitcoin::{
        self, BlockHash, Network, OutPoint, Transaction, Txid, Work,
        hashes::Hash as _,
    };
    use futures::{StreamExt as _, TryStreamExt as _, stream::BoxStream};
    use hashlink::LinkedHashMap;
    use nonempty::NonEmpty;
    use serde::{Deserialize, Serialize};
    use thiserror::Error;

    use super::common::{ConsensusHex, ReverseHex};
    use crate::types::{M6id, Output, OutputContent, THIS_SIDECHAIN};

    pub mod generated {
        tonic::include_proto!("cusf.mainchain.v1");
    }

    impl generated::Network {
        fn decode<Message>(
            self,
            field_name: &str,
        ) -> Result<bitcoin::Network, super::Error>
        where
            Message: prost::Name,
        {
            match self {
                unknown @ Self::Unknown => {
                    Err(super::Error::invalid_enum_variant::<Message>(
                        field_name,
                        unknown.as_str_name(),
                    ))
                }
                unspecified @ Self::Unspecified => {
                    Err(super::Error::invalid_enum_variant::<Message>(
                        field_name,
                        unspecified.as_str_name(),
                    ))
                }
                Self::Mainnet => Ok(bitcoin::Network::Bitcoin),
                Self::Regtest => Ok(bitcoin::Network::Regtest),
                Self::Signet => Ok(bitcoin::Network::Signet),
                Self::Testnet => Ok(bitcoin::Network::Testnet),
            }
        }
    }

    #[derive(
        Copy, Clone, Debug, Eq, Error, Hash, Ord, PartialEq, PartialOrd,
    )]
    #[error("Block not found: {0}")]
    pub struct BlockNotFoundError(pub BlockHash);

    impl
        TryFrom<
            generated::get_bmm_h_star_commitment_response::BlockNotFoundError,
        > for BlockNotFoundError
    {
        type Error = super::Error;
        fn try_from(
            err: generated::get_bmm_h_star_commitment_response::BlockNotFoundError,
        ) -> Result<Self, Self::Error> {
            let generated::get_bmm_h_star_commitment_response::BlockNotFoundError { block_hash }
                = err;
            block_hash.ok_or_else(||
                super::Error::missing_field::<generated::get_bmm_h_star_commitment_response::BlockNotFoundError>("block_hash")
            )?
            .decode::<generated::get_bmm_h_star_commitment_response::BlockNotFoundError, _>("block_hash")
            .map(Self)
        }
    }

    impl
        TryFrom<
            generated::get_bmm_h_star_commitment_response::OptionalCommitment,
        > for Option<crate::types::BlockHash>
    {
        type Error = super::Error;

        fn try_from(
            commitment: generated::get_bmm_h_star_commitment_response::OptionalCommitment,
        ) -> Result<Self, Self::Error> {
            let generated::get_bmm_h_star_commitment_response::OptionalCommitment {
                commitment,
            } = commitment;
            let Some(commitment) = commitment else {
                return Ok(None);
            };
            commitment.decode::<generated::get_bmm_h_star_commitment_response::OptionalCommitment, _>("commitment")
                .map(|block_hash| Some(crate::types::BlockHash(block_hash)))
        }
    }

    impl TryFrom<generated::get_bmm_h_star_commitment_response::Commitment>
        for nonempty::NonEmpty<Option<crate::types::BlockHash>>
    {
        type Error = super::Error;

        fn try_from(
            commitment: generated::get_bmm_h_star_commitment_response::Commitment,
        ) -> Result<Self, Self::Error> {
            let generated::get_bmm_h_star_commitment_response::Commitment {
                commitment,
                ancestor_commitments,
            } = commitment;
            let commitment = commitment.map(|commitment|
                commitment.decode::<generated::get_bmm_h_star_commitment_response::Commitment, _>("commitment")
                .map(crate::types::BlockHash)
            ).transpose()?;
            let ancestor_commitments = ancestor_commitments
                .into_iter()
                .map(|ancestor_commitment| ancestor_commitment.try_into())
                .collect::<Result<_, _>>()?;
            Ok(nonempty::NonEmpty {
                head: commitment,
                tail: ancestor_commitments,
            })
        }
    }

    impl TryFrom<generated::get_bmm_h_star_commitment_response::Result>
        for Result<
            nonempty::NonEmpty<Option<crate::types::BlockHash>>,
            BlockNotFoundError,
        >
    {
        type Error = super::Error;

        fn try_from(
            res: generated::get_bmm_h_star_commitment_response::Result,
        ) -> Result<Self, Self::Error> {
            use generated::get_bmm_h_star_commitment_response as resp;
            match res {
                resp::Result::BlockNotFound(err) => Ok(Err(err.try_into()?)),
                resp::Result::Commitment(commitment) => {
                    commitment.try_into().map(Ok)
                }
            }
        }
    }

    impl TryFrom<generated::OutPoint> for OutPoint {
        type Error = super::Error;

        fn try_from(
            outpoint: generated::OutPoint,
        ) -> Result<Self, Self::Error> {
            let generated::OutPoint { txid, vout } = outpoint;
            let txid = txid
                .ok_or_else(|| {
                    super::Error::missing_field::<generated::OutPoint>("txid")
                })?
                .decode::<generated::OutPoint, _>("txid")?;
            let vout = vout.ok_or_else(|| {
                super::Error::missing_field::<generated::OutPoint>("vout")
            })?;
            Ok(Self { txid, vout })
        }
    }

    impl TryFrom<generated::deposit::Output> for Output {
        type Error = super::Error;

        fn try_from(
            output: generated::deposit::Output,
        ) -> Result<Self, Self::Error> {
            use crate::types::TransparentAddress;
            let generated::deposit::Output {
                address,
                value_sats,
            } = output;
            let address = 'address: {
                // It is wrong to assume that the address is valid UTF8.
                // In the case that it is not valid UTF8, the deposit should be
                // ignored.
                let address_bytes: Vec<u8> =
                    address
                        .ok_or_else(|| {
                            super::Error::missing_field::<
                                generated::deposit::Output,
                            >("address")
                        })?
                        .decode_bytes::<generated::deposit::Output>(
                            "address",
                        )?;
                let address_utf8: &str =
                    match std::str::from_utf8(&address_bytes) {
                        Ok(address_str) => address_str,
                        Err(_) => {
                            tracing::warn!(
                                address_bytes = hex::encode(address_bytes),
                                "Ignoring invalid deposit address"
                            );
                            break 'address TransparentAddress::ALL_ZEROS;
                        }
                    };
                match TransparentAddress::from_str(address_utf8) {
                    Ok(address) => address,
                    Err(_) => {
                        tracing::warn!(
                            address_utf8,
                            "Ignoring invalid deposit address"
                        );
                        TransparentAddress::ALL_ZEROS
                    }
                }
            };
            let value = value_sats
                .ok_or_else(|| {
                    super::Error::missing_field::<generated::deposit::Output>(
                        "value_sats",
                    )
                })
                .map(bitcoin::Amount::from_sat)?;
            Ok(Self {
                address,
                content: OutputContent::Value(value),
            })
        }
    }

    #[derive(Clone, Copy, Debug, Deserialize, Serialize)]
    pub struct BlockHeaderInfo {
        pub block_hash: BlockHash,
        pub prev_block_hash: BlockHash,
        pub height: u32,
        pub work: Work,
    }

    #[derive(Clone, Copy, Debug, Deserialize, Serialize)]
    pub struct ChainInfo {
        pub network: Network,
    }

    #[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
    pub struct Deposit {
        /// Position of this transaction within the block that included it
        pub tx_index: u64,
        pub outpoint: OutPoint,
        pub output: Output,
    }

    impl TryFrom<generated::Deposit> for Deposit {
        type Error = super::Error;

        fn try_from(deposit: generated::Deposit) -> Result<Self, Self::Error> {
            let generated::Deposit {
                sequence_number,
                outpoint,
                output,
            } = deposit;
            let sequence_number = sequence_number.ok_or_else(|| {
                super::Error::missing_field::<generated::Deposit>(
                    "sequence_number",
                )
            })?;
            let Some(outpoint) = outpoint else {
                return Err(super::Error::missing_field::<generated::Deposit>(
                    "outpoint",
                ));
            };
            let Some(output) = output else {
                return Err(super::Error::missing_field::<generated::Deposit>(
                    "output",
                ));
            };
            Ok(Self {
                tx_index: sequence_number,
                outpoint: outpoint.try_into()?,
                output: output.try_into()?,
            })
        }
    }

    impl From<generated::withdrawal_bundle_event::event::Event>
        for crate::types::WithdrawalBundleStatus
    {
        fn from(
            event: generated::withdrawal_bundle_event::event::Event,
        ) -> Self {
            use generated::withdrawal_bundle_event::event::{
                Event, Failed, Submitted, Succeeded,
            };
            match event {
                Event::Failed(Failed {}) => Self::Failed,
                Event::Submitted(Submitted {}) => Self::Submitted,
                Event::Succeeded(Succeeded {
                    sequence_number: _,
                    transaction: _,
                }) => Self::Confirmed,
            }
        }
    }

    impl TryFrom<generated::withdrawal_bundle_event::Event>
        for crate::types::WithdrawalBundleStatus
    {
        type Error = super::Error;

        fn try_from(
            event: generated::withdrawal_bundle_event::Event,
        ) -> Result<Self, Self::Error> {
            use generated::withdrawal_bundle_event::Event;
            let Event { event } = event;
            event
                .ok_or_else(|| Self::Error::missing_field::<Event>("event"))
                .map(|event| event.into())
        }
    }

    impl TryFrom<generated::WithdrawalBundleEvent>
        for crate::types::WithdrawalBundleEvent
    {
        type Error = super::Error;

        fn try_from(
            event: generated::WithdrawalBundleEvent,
        ) -> Result<Self, Self::Error> {
            use generated::WithdrawalBundleEvent;
            let WithdrawalBundleEvent { m6id, event } = event;
            let m6id = m6id
                .ok_or_else(|| {
                    Self::Error::missing_field::<WithdrawalBundleEvent>("m6id")
                })?
                .decode::<WithdrawalBundleEvent, _>("m6id")
                .map(M6id)?;
            let status = event
                .ok_or_else(|| {
                    Self::Error::missing_field::<WithdrawalBundleEvent>("event")
                })?
                .try_into()?;
            Ok(Self { m6id, status })
        }
    }

    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub enum BlockEvent {
        Deposit(Deposit),
        WithdrawalBundle(crate::types::WithdrawalBundleEvent),
    }

    impl From<Deposit> for BlockEvent {
        fn from(deposit: Deposit) -> Self {
            Self::Deposit(deposit)
        }
    }

    impl From<crate::types::WithdrawalBundleEvent> for BlockEvent {
        fn from(bundle_event: crate::types::WithdrawalBundleEvent) -> Self {
            Self::WithdrawalBundle(bundle_event)
        }
    }

    impl TryFrom<generated::block_info::event::Event> for BlockEvent {
        type Error = super::Error;

        fn try_from(
            event: generated::block_info::event::Event,
        ) -> Result<Self, Self::Error> {
            use generated::block_info::event::Event;
            match event {
                Event::Deposit(deposit) => {
                    Ok(BlockEvent::Deposit(deposit.try_into()?))
                }
                Event::WithdrawalBundle(bundle_event) => {
                    Ok(BlockEvent::WithdrawalBundle(bundle_event.try_into()?))
                }
            }
        }
    }

    impl TryFrom<generated::block_info::Event> for BlockEvent {
        type Error = super::Error;

        fn try_from(
            event: generated::block_info::Event,
        ) -> Result<Self, Self::Error> {
            use generated::block_info::Event;
            let Event { event } = event;
            event
                .ok_or_else(|| Self::Error::missing_field::<Event>("event"))?
                .try_into()
        }
    }

    #[derive(Clone, Debug, Default, Deserialize, Serialize)]
    pub struct BlockInfo {
        pub bmm_commitment: Option<crate::types::BlockHash>,
        pub events: Vec<BlockEvent>,
    }

    impl BlockInfo {
        pub fn deposits(&self) -> impl DoubleEndedIterator<Item = &Deposit> {
            self.events.iter().filter_map(|event| match event {
                BlockEvent::Deposit(deposit) => Some(deposit),
                BlockEvent::WithdrawalBundle(_) => None,
            })
        }

        pub fn into_deposits(self) -> impl DoubleEndedIterator<Item = Deposit> {
            self.events.into_iter().filter_map(|event| match event {
                BlockEvent::Deposit(deposit) => Some(deposit),
                BlockEvent::WithdrawalBundle(_) => None,
            })
        }

        pub fn withdrawal_bundle_events(
            &self,
        ) -> impl DoubleEndedIterator<Item = &crate::types::WithdrawalBundleEvent>
        {
            self.events.iter().filter_map(|event| match event {
                BlockEvent::WithdrawalBundle(bundle_event) => {
                    Some(bundle_event)
                }
                BlockEvent::Deposit(_) => None,
            })
        }
    }

    impl TryFrom<generated::BlockInfo> for BlockInfo {
        type Error = super::Error;

        fn try_from(
            block_info: generated::BlockInfo,
        ) -> Result<Self, Self::Error> {
            let generated::BlockInfo {
                bmm_commitment,
                events,
            } = block_info;
            let bmm_commitment = bmm_commitment
                .map(|bmm_commitment| {
                    bmm_commitment
                        .decode::<generated::BlockInfo, _>("bmm_commitment")
                        .map(crate::types::BlockHash)
                })
                .transpose()?;
            let events = events
                .into_iter()
                .map(BlockEvent::try_from)
                .collect::<Result<_, Self::Error>>()?;
            Ok(Self {
                bmm_commitment,
                events,
            })
        }
    }

    impl TryFrom<generated::get_block_info_response::Info>
        for (BlockHeaderInfo, BlockInfo)
    {
        type Error = super::Error;

        fn try_from(
            info: generated::get_block_info_response::Info,
        ) -> Result<Self, Self::Error> {
            use generated::get_block_info_response::Info;
            let Info {
                header_info,
                block_info,
            } = info;
            let header_info = header_info
                .as_ref()
                .ok_or_else(|| {
                    Self::Error::missing_field::<Info>("header_info")
                })?
                .try_into()?;
            let block_info = block_info
                .ok_or_else(|| {
                    Self::Error::missing_field::<Info>("block_info")
                })?
                .try_into()?;
            Ok((header_info, block_info))
        }
    }

    #[derive(Clone, Debug, Default, Deserialize, Serialize)]
    pub struct TwoWayPegData {
        pub block_info: LinkedHashMap<BlockHash, BlockInfo>,
    }

    impl TwoWayPegData {
        pub fn deposits(
            &self,
        ) -> impl DoubleEndedIterator<Item = (BlockHash, Vec<&Deposit>)>
        {
            self.block_info.iter().flat_map(|(block_hash, block_info)| {
                let deposits: Vec<_> = block_info.deposits().collect();
                if deposits.is_empty() {
                    None
                } else {
                    Some((*block_hash, deposits))
                }
            })
        }

        pub fn into_deposits(
            self,
        ) -> impl DoubleEndedIterator<Item = (BlockHash, Vec<Deposit>)>
        {
            self.block_info
                .into_iter()
                .flat_map(|(block_hash, block_info)| {
                    let deposits: Vec<_> = block_info.into_deposits().collect();
                    if deposits.is_empty() {
                        None
                    } else {
                        Some((block_hash, deposits))
                    }
                })
        }

        pub fn withdrawal_bundle_events(
            &self,
        ) -> impl DoubleEndedIterator<
            Item = (&'_ BlockHash, &'_ crate::types::WithdrawalBundleEvent),
        > + '_ {
            self.block_info.iter().flat_map(|(block_hash, block_info)| {
                block_info
                    .withdrawal_bundle_events()
                    .map(move |event| (block_hash, event))
            })
        }

        /// Latest deposit block hash
        pub fn latest_deposit_block_hash(&self) -> Option<BlockHash> {
            self.deposits()
                .next_back()
                .map(|(block_hash, _)| block_hash)
        }

        /// Latest withdrawal bundle event block hash
        pub fn latest_withdrawal_bundle_event_block_hash(
            &self,
        ) -> Option<&BlockHash> {
            self.withdrawal_bundle_events()
                .next_back()
                .map(|(block_hash, _)| block_hash)
        }
    }

    impl TryFrom<generated::GetTwoWayPegDataResponse> for TwoWayPegData {
        type Error = super::Error;

        fn try_from(
            two_way_peg_data: generated::GetTwoWayPegDataResponse,
        ) -> Result<Self, Self::Error> {
            let generated::GetTwoWayPegDataResponse { blocks } =
                two_way_peg_data;
            let block_info = blocks
                .into_iter()
                .map(|item| {
                    let generated::get_two_way_peg_data_response::ResponseItem {
                    block_header_info,
                    block_info
                } = item;
                    let Some(block_header_info) = block_header_info else {
                        return Err(super::Error::missing_field::<generated::get_two_way_peg_data_response::ResponseItem>("block_header_info"));
                    };
                    let BlockHeaderInfo { block_hash, .. } =
                        (&block_header_info).try_into()?;
                    let Some(block_info) = block_info else {
                        return Err(super::Error::missing_field::<generated::get_two_way_peg_data_response::ResponseItem>("block_info"));
                    };
                    Ok((block_hash, block_info.try_into()?))
                })
                .collect::<Result<LinkedHashMap<_, _>, _>>()?;
            Ok(TwoWayPegData { block_info })
        }
    }

    impl TryFrom<&generated::BlockHeaderInfo> for BlockHeaderInfo {
        type Error = super::Error;

        fn try_from(
            header_info: &generated::BlockHeaderInfo,
        ) -> Result<Self, Self::Error> {
            let generated::BlockHeaderInfo {
                block_hash,
                prev_block_hash,
                height,
                work,
            } = header_info;
            let block_hash = block_hash
                .as_ref()
                .ok_or_else(|| {
                    super::Error::missing_field::<generated::BlockHeaderInfo>(
                        "block_hash",
                    )
                })?
                .decode::<generated::BlockHeaderInfo, _>("block_hash")?;
            let prev_block_hash = prev_block_hash
                .as_ref()
                .ok_or_else(|| {
                    super::Error::missing_field::<generated::BlockHeaderInfo>(
                        "prev_block_hash",
                    )
                })?
                .decode::<generated::BlockHeaderInfo, _>("prev_block_hash")?;
            let work = work
                .as_ref()
                .ok_or_else(|| {
                    super::Error::missing_field::<generated::BlockHeaderInfo>(
                        "work",
                    )
                })?
                .decode::<generated::BlockHeaderInfo, _>("work")
                .map(bitcoin::Work::from_le_bytes)?;
            Ok(BlockHeaderInfo {
                block_hash,
                prev_block_hash,
                height: *height,
                work,
            })
        }
    }

    #[derive(Clone, Debug)]
    pub enum Event {
        ConnectBlock {
            header_info: BlockHeaderInfo,
            block_info: BlockInfo,
        },
        DisconnectBlock {
            block_hash: BlockHash,
        },
    }

    impl TryFrom<generated::subscribe_events_response::event::Event> for Event {
        type Error = super::Error;

        fn try_from(
            event: generated::subscribe_events_response::event::Event,
        ) -> Result<Self, Self::Error> {
            use generated::subscribe_events_response::event;
            match event {
                event::Event::ConnectBlock(connect_block) => {
                    let event::ConnectBlock {
                        header_info,
                        block_info,
                    } = connect_block;
                    let Some(header_info) = header_info else {
                        return Err(super::Error::missing_field::<
                            event::ConnectBlock,
                        >("header_info"));
                    };
                    let Some(block_info) = block_info else {
                        return Err(super::Error::missing_field::<
                            event::ConnectBlock,
                        >("block_info"));
                    };
                    Ok(Self::ConnectBlock {
                        header_info: (&header_info).try_into()?,
                        block_info: block_info.try_into()?,
                    })
                }
                event::Event::DisconnectBlock(disconnect_block) => {
                    let event::DisconnectBlock { block_hash } =
                        disconnect_block;
                    let block_hash =
                        block_hash
                            .ok_or_else(|| {
                                super::Error::missing_field::<
                                    event::DisconnectBlock,
                                >(
                                    "disconnect_block"
                                )
                            })?
                            .decode::<event::DisconnectBlock, _>(
                                "disconnect_block",
                            )?;
                    let block_hash = BlockHash::from_byte_array(block_hash);
                    Ok(Self::DisconnectBlock { block_hash })
                }
            }
        }
    }

    impl TryFrom<generated::subscribe_events_response::Event> for Event {
        type Error = super::Error;

        fn try_from(
            event: generated::subscribe_events_response::Event,
        ) -> Result<Self, Self::Error> {
            let generated::subscribe_events_response::Event { event } = event;
            let Some(event) = event else {
                return Err(super::Error::missing_field::<
                    generated::subscribe_events_response::Event,
                >("event"));
            };
            event.try_into()
        }
    }

    impl TryFrom<generated::SubscribeEventsResponse> for Event {
        type Error = super::Error;

        fn try_from(
            event: generated::SubscribeEventsResponse,
        ) -> Result<Self, Self::Error> {
            let generated::SubscribeEventsResponse { event } = event;
            let Some(event) = event else {
                return Err(super::Error::missing_field::<
                    generated::SubscribeEventsResponse,
                >("event"));
            };
            event.try_into()
        }
    }

    pub struct EventStream;

    #[derive(Clone, Debug)]
    #[repr(transparent)]
    pub struct ValidatorClient<T>(
        pub generated::validator_service_client::ValidatorServiceClient<T>,
    );

    impl<T> ValidatorClient<T>
    where
        T: super::Transport,
    {
        pub fn new(inner: T) -> Self {
            // 1GB
            const MAX_DECODE_MESSAGE_SIZE: usize = 1024 * 1024 * 1024;
            Self(generated::validator_service_client::ValidatorServiceClient::<T>::new(
                inner,
            ).max_decoding_message_size(MAX_DECODE_MESSAGE_SIZE)
            )
        }

        pub async fn get_block_header_info(
            &mut self,
            block_hash: BlockHash,
        ) -> Result<Option<BlockHeaderInfo>, super::Error> {
            let request = generated::GetBlockHeaderInfoRequest {
                block_hash: Some(ReverseHex::encode(&block_hash)),
                max_ancestors: Some(0),
            };
            let generated::GetBlockHeaderInfoResponse { header_infos } =
                self.0.get_block_header_info(request).await?.into_inner();
            let Some(header_info) = header_infos.first() else {
                return Ok(None);
            };
            let header_info = header_info.try_into()?;
            Ok(Some(header_info))
        }

        pub async fn get_block_header_infos(
            &mut self,
            block_hash: BlockHash,
            max_ancestors: u32,
        ) -> Result<Option<NonEmpty<BlockHeaderInfo>>, super::Error> {
            let request = generated::GetBlockHeaderInfoRequest {
                block_hash: Some(ReverseHex::encode(&block_hash)),
                max_ancestors: Some(max_ancestors),
            };
            let generated::GetBlockHeaderInfoResponse { header_infos } =
                self.0.get_block_header_info(request).await?.into_inner();
            let Some(header_infos) = NonEmpty::from_vec(header_infos) else {
                return Ok(None);
            };
            let header_infos: NonEmpty<BlockHeaderInfo> = header_infos
                .try_map(|header_info| (&header_info).try_into())?;
            let mut expected_block_hash = header_infos.head.prev_block_hash;
            // Check that ancestor infos are sequential
            for header_info in &header_infos.tail {
                if header_info.block_hash == expected_block_hash {
                    expected_block_hash = header_info.prev_block_hash;
                } else {
                    return Err(super::Error::invalid_repeated_value::<
                        generated::GetBlockHeaderInfoResponse,
                    >(
                        "header_infos",
                        &serde_json::to_string(&header_info).unwrap(),
                    ));
                }
            }
            Ok(Some(header_infos))
        }

        pub async fn get_block_infos(
            &mut self,
            block_hash: BlockHash,
            max_ancestors: u32,
        ) -> Result<Option<NonEmpty<(BlockHeaderInfo, BlockInfo)>>, super::Error>
        {
            let request = generated::GetBlockInfoRequest {
                block_hash: Some(ReverseHex::encode(&block_hash)),
                sidechain_id: Some(THIS_SIDECHAIN as u32),
                max_ancestors: Some(max_ancestors),
            };
            let generated::GetBlockInfoResponse { infos } =
                self.0.get_block_info(request).await?.into_inner();
            let Some(infos) = NonEmpty::from_vec(infos) else {
                return Ok(None);
            };
            let infos: NonEmpty<(BlockHeaderInfo, BlockInfo)> =
                infos.try_map(|info| info.try_into())?;
            let mut expected_block_hash = infos.head.0.prev_block_hash;
            // Check that ancestor infos are sequential
            for (header_info, block_info) in &infos.tail {
                if header_info.block_hash == expected_block_hash {
                    expected_block_hash = header_info.prev_block_hash;
                } else {
                    return Err(super::Error::invalid_repeated_value::<
                        generated::GetBlockInfoResponse,
                    >(
                        "infos",
                        &serde_json::to_string(&(header_info, block_info))
                            .unwrap(),
                    ));
                }
            }
            Ok(Some(infos))
        }

        pub async fn get_bmm_hstar_commitments(
            &mut self,
            block_hash: BlockHash,
            max_ancestors: u32,
        ) -> Result<
            Result<
                nonempty::NonEmpty<Option<crate::types::BlockHash>>,
                BlockNotFoundError,
            >,
            super::Error,
        > {
            let request = generated::GetBmmHStarCommitmentRequest {
                block_hash: Some(ReverseHex::encode(&block_hash)),
                sidechain_id: Some(THIS_SIDECHAIN as u32),
                max_ancestors: Some(max_ancestors),
            };
            let generated::GetBmmHStarCommitmentResponse { result } = self
                .0
                .get_bmm_h_star_commitment(request)
                .await?
                .into_inner();
            let Some(result) = result else {
                return Err(super::Error::missing_field::<
                    generated::GetBmmHStarCommitmentResponse,
                >("result"));
            };
            result.try_into()
        }

        pub async fn get_chain_info(
            &mut self,
        ) -> Result<ChainInfo, super::Error> {
            let request = generated::GetChainInfoRequest {};
            let generated::GetChainInfoResponse { network } =
                self.0.get_chain_info(request).await?.into_inner();
            let network = generated::Network::try_from(network)
                .map_err(|_| super::Error::UnknownEnumTag {
                    field_name: "network".to_owned(),
                    message_name:
                        <generated::GetChainInfoResponse as prost::Name>::NAME
                            .to_owned(),
                    tag: network,
                })?
                .decode::<generated::GetChainInfoResponse>("network")?;
            Ok(ChainInfo { network })
        }

        pub async fn get_chain_tip(
            &mut self,
        ) -> Result<BlockHeaderInfo, super::Error> {
            let request = generated::GetChainTipRequest {};
            let generated::GetChainTipResponse { block_header_info } =
                self.0.get_chain_tip(request).await?.into_inner();
            let Some(block_header_info) = block_header_info else {
                return Err(super::Error::missing_field::<
                    generated::GetChainTipResponse,
                >("block_header_info"));
            };
            (&block_header_info).try_into()
        }

        pub async fn get_two_way_peg_data(
            &mut self,
            start_block_hash: Option<BlockHash>,
            end_block_hash: BlockHash,
        ) -> Result<TwoWayPegData, super::Error> {
            let request = generated::GetTwoWayPegDataRequest {
                sidechain_id: Some(THIS_SIDECHAIN as u32),
                start_block_hash: start_block_hash.map(|start_block_hash| {
                    ReverseHex::encode(&start_block_hash)
                }),
                end_block_hash: Some(ReverseHex::encode(&end_block_hash)),
            };
            self.0
                .get_two_way_peg_data(request)
                .await?
                .into_inner()
                .try_into()
        }

        pub async fn subscribe_events(
            &mut self,
        ) -> Result<BoxStream<'_, Result<Event, super::Error>>, super::Error>
        {
            let request = generated::SubscribeEventsRequest {
                sidechain_id: Some(THIS_SIDECHAIN as u32),
            };
            let event_stream =
                self.0.subscribe_events(request).await?.into_inner();
            let event_stream = event_stream
                .map(|event_res| match event_res {
                    Ok(event) => event.try_into(),
                    Err(err) => Err(err.into()),
                })
                .boxed();
            Ok(event_stream)
        }
    }

    #[derive(Clone, Debug)]
    #[repr(transparent)]
    pub struct WalletClient<T>(
        pub generated::wallet_service_client::WalletServiceClient<T>,
    );

    impl<T> WalletClient<T>
    where
        T: super::Transport,
    {
        pub fn new(inner: T) -> Self {
            Self(
                generated::wallet_service_client::WalletServiceClient::<T>::new(
                    inner,
                ),
            )
        }

        pub async fn broadcast_withdrawal_bundle(
            &mut self,
            transaction: &Transaction,
        ) -> Result<(), super::Error> {
            let request = generated::BroadcastWithdrawalBundleRequest {
                sidechain_id: Some(THIS_SIDECHAIN as u32),
                transaction: Some(bitcoin::consensus::serialize(transaction)),
            };
            let generated::BroadcastWithdrawalBundleResponse {} = self
                .0
                .broadcast_withdrawal_bundle(request)
                .await?
                .into_inner();
            Ok(())
        }

        pub async fn create_bmm_critical_data_tx(
            &mut self,
            value_sats: u64,
            height: u32,
            critical_hash: [u8; 32],
            prev_bytes: BlockHash,
        ) -> Result<Txid, super::Error> {
            let request = generated::CreateBmmCriticalDataTransactionRequest {
                sidechain_id: Some(THIS_SIDECHAIN as u32),
                value_sats: Some(value_sats),
                height: Some(height),
                critical_hash: Some(ConsensusHex::encode(&critical_hash)),
                prev_bytes: Some(ReverseHex::encode(&prev_bytes)),
            };
            let generated::CreateBmmCriticalDataTransactionResponse { txid } =
                self.0
                    .create_bmm_critical_data_transaction(request)
                    .await?
                    .into_inner();
            let txid = txid.ok_or_else(||
                super::Error::missing_field::<generated::CreateBmmCriticalDataTransactionResponse>("txid"))?
                .decode::<generated::CreateBmmCriticalDataTransactionResponse, _>("txid")?;
            Ok(txid)
        }

        pub async fn create_deposit_tx(
            &mut self,
            address: crate::types::TransparentAddress,
            value_sats: u64,
            fee_sats: u64,
        ) -> Result<Txid, super::Error> {
            let request = generated::CreateDepositTransactionRequest {
                sidechain_id: Some(THIS_SIDECHAIN as u32),
                address: Some(address.to_string()),
                value_sats: Some(value_sats),
                fee_sats: Some(fee_sats),
            };
            let generated::CreateDepositTransactionResponse { txid } = self
                .0
                .create_deposit_transaction(request)
                .await?
                .into_inner();
            let txid = txid
                .ok_or_else(|| {
                    super::Error::missing_field::<
                        generated::CreateDepositTransactionResponse,
                    >("txid")
                })?
                .decode::<generated::CreateDepositTransactionResponse, _>(
                    "txid",
                )?;
            Ok(txid)
        }

        pub async fn create_new_address(
            &mut self,
        ) -> Result<
            bitcoin::Address<bitcoin::address::NetworkUnchecked>,
            super::Error,
        > {
            let request = generated::CreateNewAddressRequest {};
            let generated::CreateNewAddressResponse { address } =
                self.0.create_new_address(request).await?.into_inner();
            let address = address.parse().map_err(|_| {
                super::Error::invalid_field_value::<
                    generated::CreateNewAddressResponse,
                >("address", &address)
            })?;
            Ok(address)
        }

        pub async fn generate_blocks(
            &mut self,
            blocks: u32,
        ) -> Result<(), super::Error> {
            let request = generated::GenerateBlocksRequest {
                blocks: Some(blocks),
                ack_all_proposals: true,
            };
            let _resp: Vec<generated::GenerateBlocksResponse> = self
                .0
                .generate_blocks(request)
                .await?
                .into_inner()
                .try_collect()
                .await?;
            Ok(())
        }
    }
}
