/// Convenience alias to avoid writing out a lengthy trait bound
pub trait Transport = where
    Self: tonic::client::GrpcService<tonic::body::BoxBody>,
    Self::Error: Into<tonic::codegen::StdError>,
    Self::ResponseBody:
        tonic::codegen::Body<Data = tonic::codegen::Bytes> + Send + 'static,
    <Self::ResponseBody as tonic::codegen::Body>::Error:
        Into<tonic::codegen::StdError> + Send;

pub mod mainchain {
    use bitcoin::{
        self, hashes::Hash as _, BlockHash, Network, OutPoint, Transaction,
        Txid, Work,
    };
    use futures::{stream::BoxStream, StreamExt as _};
    use hashlink::LinkedHashMap;
    use serde::{Deserialize, Serialize};
    use thiserror::Error;

    use crate::types::{Output, OutputContent, THIS_SIDECHAIN};

    pub mod generated {
        tonic::include_proto!("cusf.mainchain.v1");
    }

    #[derive(Debug, Error)]
    pub enum Error {
        #[error(transparent)]
        Grpc(#[from] tonic::Status),
        #[error("Invalid enum variant in field `{field_name}` of message `{message_name}`: `{variant_name}`")]
        InvalidEnumVariant {
            field_name: String,
            message_name: String,
            variant_name: String,
        },
        #[error("Invalid field value in field `{field_name}` of message `{message_name}`: `{value}`")]
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
        #[error("Unknown enum tag in field `{field_name}` of message `{message_name}`: `{tag}`")]
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

        pub fn invalid_field_value<Message>(
            field_name: &str,
            value: &str,
        ) -> Self
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

    impl generated::ConsensusHex {
        pub fn consensus_decode<Message, T>(
            self,
            field_name: &str,
        ) -> Result<T, Error>
        where
            Message: prost::Name,
            T: bitcoin::consensus::Decodable,
        {
            let Self { hex } = self;
            let hex = hex.ok_or_else(|| Error::missing_field::<Self>("hex"))?;
            bitcoin::consensus::encode::deserialize_hex(&hex).map_err(|_err| {
                Error::invalid_field_value::<Message>(field_name, &hex)
            })
        }

        /// Variant of [`Self::decode`] that returns a `tonic::Status` error
        pub fn consensus_decode_tonic<Message, T>(
            self,
            field_name: &str,
        ) -> Result<T, tonic::Status>
        where
            Message: prost::Name,
            T: bitcoin::consensus::Decodable,
        {
            self.consensus_decode::<Message, _>(field_name)
                .map_err(|err| tonic::Status::from_error(Box::new(err)))
        }

        pub fn consensus_encode<T>(value: &T) -> Self
        where
            T: bitcoin::consensus::Encodable,
        {
            let hex = bitcoin::consensus::encode::serialize_hex(value);
            Self { hex: Some(hex) }
        }

        pub fn decode<Message, T>(self, field_name: &str) -> Result<T, Error>
        where
            Message: prost::Name,
            T: borsh::BorshDeserialize,
        {
            let Self { hex } = self;
            let hex = hex.ok_or_else(|| Error::missing_field::<Self>("hex"))?;
            let bytes = hex::decode(&hex).map_err(|_err| {
                Error::invalid_field_value::<Message>(field_name, &hex)
            })?;
            T::try_from_slice(&bytes).map_err(|_err| {
                Error::invalid_field_value::<Message>(field_name, &hex)
            })
        }
    }

    impl generated::ReverseHex {
        pub fn decode<Message, T>(self, field_name: &str) -> Result<T, Error>
        where
            Message: prost::Name,
            T: bitcoin::consensus::Decodable,
        {
            let Self { hex } = self;
            let hex = hex.ok_or_else(|| Error::missing_field::<Self>("hex"))?;
            let mut bytes = hex::decode(&hex).map_err(|_| {
                Error::invalid_field_value::<Message>(field_name, &hex)
            })?;
            bytes.reverse();
            bitcoin::consensus::deserialize(&bytes).map_err(|_err| {
                Error::invalid_field_value::<Message>(field_name, &hex)
            })
        }

        /// Variant of [`Self::decode`] that returns a `tonic::Status` error
        pub fn decode_tonic<Message, T>(
            self,
            field_name: &str,
        ) -> Result<T, tonic::Status>
        where
            Message: prost::Name,
            T: bitcoin::consensus::Decodable,
        {
            self.decode::<Message, _>(field_name)
                .map_err(|err| tonic::Status::from_error(Box::new(err)))
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

    #[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
    pub enum AddressType {
        Bech32,
        #[default]
        Default,
        Legacy,
        P2shSegwit,
    }

    impl From<AddressType> for generated::AddressType {
        fn from(address_type: AddressType) -> Self {
            match address_type {
                AddressType::Default => Self::Default,
                AddressType::Bech32 => Self::Bech32,
                AddressType::Legacy => Self::Legacy,
                AddressType::P2shSegwit => Self::P2shSegwit,
            }
        }
    }

    impl generated::Network {
        fn decode<Message>(
            self,
            field_name: &str,
        ) -> Result<bitcoin::Network, Error>
        where
            Message: prost::Name,
        {
            match self {
                unknown @ Self::Unknown => {
                    Err(Error::invalid_enum_variant::<Message>(
                        field_name,
                        unknown.as_str_name(),
                    ))
                }
                unspecified @ Self::Unspecified => {
                    Err(Error::invalid_enum_variant::<Message>(
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
        type Error = Error;
        fn try_from(
            err: generated::get_bmm_h_star_commitment_response::BlockNotFoundError,
        ) -> Result<Self, Self::Error> {
            let generated::get_bmm_h_star_commitment_response::BlockNotFoundError { block_hash }
                = err;
            block_hash.ok_or_else(||
                Error::missing_field::<generated::get_bmm_h_star_commitment_response::BlockNotFoundError>("block_hash")
            )?
            .decode::<generated::get_bmm_h_star_commitment_response::BlockNotFoundError, _>("block_hash")
            .map(Self)
        }
    }

    impl TryFrom<generated::get_bmm_h_star_commitment_response::Commitment>
        for Option<crate::types::BlockHash>
    {
        type Error = Error;

        fn try_from(
            commitment: generated::get_bmm_h_star_commitment_response::Commitment,
        ) -> Result<Self, Self::Error> {
            let generated::get_bmm_h_star_commitment_response::Commitment {
                commitment,
            } = commitment;
            let Some(commitment) = commitment else {
                return Ok(None);
            };
            commitment.decode::<generated::get_bmm_h_star_commitment_response::Commitment, _>("commitment")
                .map(Some)
        }
    }

    impl TryFrom<generated::get_bmm_h_star_commitment_response::Result>
        for Result<Option<crate::types::BlockHash>, BlockNotFoundError>
    {
        type Error = Error;

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
        type Error = Error;

        fn try_from(
            outpoint: generated::OutPoint,
        ) -> Result<Self, Self::Error> {
            let generated::OutPoint { txid, vout } = outpoint;
            let txid = txid
                .ok_or_else(|| {
                    Error::missing_field::<generated::OutPoint>("txid")
                })?
                .consensus_decode::<generated::OutPoint, _>("txid")?;
            Ok(Self { txid, vout })
        }
    }

    impl TryFrom<generated::Output> for Output {
        type Error = Error;

        fn try_from(output: generated::Output) -> Result<Self, Self::Error> {
            let generated::Output {
                address,
                value_sats,
            } = output;
            let address = address
                .ok_or_else(|| {
                    Error::missing_field::<generated::Output>("address")
                })?
                .decode::<generated::Output, _>("address")?;
            Ok(Self {
                address,
                content: OutputContent::Value(value_sats),
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
        type Error = Error;

        fn try_from(deposit: generated::Deposit) -> Result<Self, Self::Error> {
            let generated::Deposit {
                sequence_number: tx_index,
                outpoint,
                output,
            } = deposit;
            let Some(outpoint) = outpoint else {
                return Err(Error::missing_field::<generated::Deposit>(
                    "outpoint",
                ));
            };
            let Some(output) = output else {
                return Err(Error::missing_field::<generated::Deposit>(
                    "output",
                ));
            };
            Ok(Self {
                tx_index,
                outpoint: outpoint.try_into()?,
                output: output.try_into()?,
            })
        }
    }

    #[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
    pub enum WithdrawalBundleStatus {
        Confirmed,
        Failed,
        Submitted,
    }

    impl generated::WithdrawalBundleEventType {
        fn decode<Message>(
            self,
            field_name: &str,
        ) -> Result<WithdrawalBundleStatus, Error>
        where
            Message: prost::Name,
        {
            match self {
                unspecified @ Self::Unspecified => {
                    Err(Error::invalid_enum_variant::<Message>(
                        field_name,
                        unspecified.as_str_name(),
                    ))
                }
                generated::WithdrawalBundleEventType::Failed => {
                    Ok(WithdrawalBundleStatus::Failed)
                }
                generated::WithdrawalBundleEventType::Submitted => {
                    Ok(WithdrawalBundleStatus::Submitted)
                }
                generated::WithdrawalBundleEventType::Succeded => {
                    Ok(WithdrawalBundleStatus::Confirmed)
                }
            }
        }
    }

    impl TryFrom<generated::WithdrawalBundleEvent>
        for (Txid, WithdrawalBundleStatus)
    {
        type Error = Error;

        fn try_from(
            withdrawal_bundle_event: generated::WithdrawalBundleEvent,
        ) -> Result<Self, Self::Error> {
            let generated::WithdrawalBundleEvent {
                m6id,
                withdrawal_bundle_event_type,
            } = withdrawal_bundle_event;
            let txid = m6id
                .ok_or_else(|| {
                    Error::missing_field::<generated::WithdrawalBundleEvent>(
                        "m6id",
                    )
                })?
                .consensus_decode::<generated::WithdrawalBundleEvent, _>(
                    "m6id",
                )?;
            let status = generated::WithdrawalBundleEventType::try_from(
                withdrawal_bundle_event_type,
            )
            .map_err(|_| Error::UnknownEnumTag {
                field_name: "withdrawal_bundle_event_type".to_owned(),
                message_name:
                    <generated::WithdrawalBundleEvent as prost::Name>::NAME
                        .to_owned(),
                tag: withdrawal_bundle_event_type,
            })?
            .decode::<generated::WithdrawalBundleEvent>(
                "withdrawal_bundle_event_type",
            )?;
            Ok((txid, status))
        }
    }

    #[derive(Clone, Debug, Default, Deserialize, Serialize)]
    pub struct BlockInfo {
        pub deposits: Vec<Deposit>,
        pub withdrawal_bundle_events: Vec<(Txid, WithdrawalBundleStatus)>,
        pub bmm_commitment: Option<crate::types::BlockHash>,
    }

    impl TryFrom<generated::BlockInfo> for BlockInfo {
        type Error = Error;

        fn try_from(
            block_info: generated::BlockInfo,
        ) -> Result<Self, Self::Error> {
            let generated::BlockInfo {
                deposits,
                withdrawal_bundle_events,
                bmm_commitment,
            } = block_info;
            let deposits = deposits
                .into_iter()
                .map(|deposit| deposit.try_into())
                .collect::<Result<Vec<Deposit>, _>>()?;
            let withdrawal_bundle_events = withdrawal_bundle_events
                .into_iter()
                .map(|withdrawal_bundle_event| {
                    <(_, _)>::try_from(withdrawal_bundle_event)
                })
                .collect::<Result<_, _>>()?;
            let bmm_commitment = bmm_commitment
                .map(|bmm_commitment| {
                    bmm_commitment
                        .decode::<generated::BlockInfo, _>("bmm_commitment")
                })
                .transpose()?;
            Ok(Self {
                deposits,
                withdrawal_bundle_events,
                bmm_commitment,
            })
        }
    }

    #[derive(Clone, Debug, Default, Deserialize, Serialize)]
    pub struct TwoWayPegData {
        pub block_info: LinkedHashMap<BlockHash, BlockInfo>,
    }

    impl TwoWayPegData {
        pub fn deposits(
            &self,
        ) -> impl DoubleEndedIterator<Item = (BlockHash, &Vec<Deposit>)>
        {
            self.block_info
                .iter()
                .filter_map(|(block_hash, block_info)| {
                    if block_info.deposits.is_empty() {
                        None
                    } else {
                        Some((*block_hash, &block_info.deposits))
                    }
                })
        }

        pub fn withdrawal_bundle_events(
            &self,
        ) -> impl DoubleEndedIterator<
            Item = (BlockHash, Txid, WithdrawalBundleStatus),
        > + '_ {
            self.block_info.iter().flat_map(|(block_hash, block_info)| {
                block_info
                    .withdrawal_bundle_events
                    .iter()
                    .map(|(txid, status)| (*block_hash, *txid, *status))
            })
        }

        /// Last deposit block hash
        pub fn deposit_block_hash(&self) -> Option<BlockHash> {
            self.deposits()
                .next_back()
                .map(|(block_hash, _)| block_hash)
        }
    }

    impl TryFrom<generated::GetTwoWayPegDataResponse> for TwoWayPegData {
        type Error = Error;

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
                        return Err(Error::missing_field::<generated::get_two_way_peg_data_response::ResponseItem>("block_header_info"));
                    };
                    let BlockHeaderInfo { block_hash, .. } =
                        block_header_info.try_into()?;
                    let Some(block_info) = block_info else {
                        return Err(Error::missing_field::<generated::get_two_way_peg_data_response::ResponseItem>("block_info"));
                    };
                    Ok((block_hash, block_info.try_into()?))
                })
                .collect::<Result<LinkedHashMap<_, _>, _>>()?;
            Ok(TwoWayPegData { block_info })
        }
    }

    impl TryFrom<generated::BlockHeaderInfo> for BlockHeaderInfo {
        type Error = Error;

        fn try_from(
            header_info: generated::BlockHeaderInfo,
        ) -> Result<Self, Self::Error> {
            let generated::BlockHeaderInfo {
                block_hash,
                prev_block_hash,
                height,
                work,
            } = header_info;
            let block_hash = block_hash
                .ok_or_else(|| {
                    Error::missing_field::<generated::BlockHeaderInfo>(
                        "block_hash",
                    )
                })?
                .decode::<generated::BlockHeaderInfo, _>("block_hash")?;
            let prev_block_hash = prev_block_hash
                .ok_or_else(|| {
                    Error::missing_field::<generated::BlockHeaderInfo>(
                        "prev_block_hash",
                    )
                })?
                .decode::<generated::BlockHeaderInfo, _>("prev_block_hash")?;
            let work = work
                .ok_or_else(|| {
                    Error::missing_field::<generated::BlockHeaderInfo>("work")
                })?
                .consensus_decode::<generated::BlockHeaderInfo, _>("work")
                .map(bitcoin::Work::from_le_bytes)?;
            Ok(BlockHeaderInfo {
                block_hash,
                prev_block_hash,
                height,
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
        type Error = Error;

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
                        return Err(
                            Error::missing_field::<event::ConnectBlock>(
                                "header_info",
                            ),
                        );
                    };
                    let Some(block_info) = block_info else {
                        return Err(
                            Error::missing_field::<event::ConnectBlock>(
                                "block_info",
                            ),
                        );
                    };
                    Ok(Self::ConnectBlock {
                        header_info: header_info.try_into()?,
                        block_info: block_info.try_into()?,
                    })
                }
                event::Event::DisconnectBlock(disconnect_block) => {
                    let event::DisconnectBlock { block_hash } =
                        disconnect_block;
                    let block_hash = block_hash
                        .ok_or_else(|| {
                            Error::missing_field::<event::DisconnectBlock>(
                                "disconnect_block",
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
        type Error = Error;

        fn try_from(
            event: generated::subscribe_events_response::Event,
        ) -> Result<Self, Self::Error> {
            let generated::subscribe_events_response::Event { event } = event;
            let Some(event) = event else {
                return Err(Error::missing_field::<
                    generated::subscribe_events_response::Event,
                >("event"));
            };
            event.try_into()
        }
    }

    impl TryFrom<generated::SubscribeEventsResponse> for Event {
        type Error = Error;

        fn try_from(
            event: generated::SubscribeEventsResponse,
        ) -> Result<Self, Self::Error> {
            let generated::SubscribeEventsResponse { event } = event;
            let Some(event) = event else {
                return Err(Error::missing_field::<
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
            Self(generated::validator_service_client::ValidatorServiceClient::<T>::new(
                inner,
            ))
        }

        pub async fn get_block_header_info(
            &mut self,
            block_hash: BlockHash,
        ) -> Result<BlockHeaderInfo, Error> {
            let request = generated::GetBlockHeaderInfoRequest {
                block_hash: Some(generated::ReverseHex::encode(&block_hash)),
            };
            let generated::GetBlockHeaderInfoResponse { header_info } =
                self.0.get_block_header_info(request).await?.into_inner();
            let Some(header_info) = header_info else {
                return Err(Error::missing_field::<
                    generated::GetBlockHeaderInfoResponse,
                >("header_info"));
            };
            header_info.try_into()
        }

        pub async fn get_bmm_hstar_commitments(
            &mut self,
            block_hash: BlockHash,
        ) -> Result<
            Result<Option<crate::types::BlockHash>, BlockNotFoundError>,
            Error,
        > {
            let request = generated::GetBmmHStarCommitmentRequest {
                block_hash: Some(generated::ReverseHex::encode(&block_hash)),
                sidechain_id: Some(THIS_SIDECHAIN as u32),
            };
            let generated::GetBmmHStarCommitmentResponse { result } = self
                .0
                .get_bmm_h_star_commitment(request)
                .await?
                .into_inner();
            let Some(result) = result else {
                return Err(Error::missing_field::<
                    generated::GetBmmHStarCommitmentResponse,
                >("result"));
            };
            result.try_into()
        }

        pub async fn get_chain_info(&mut self) -> Result<ChainInfo, Error> {
            let request = generated::GetChainInfoRequest {};
            let generated::GetChainInfoResponse { network } =
                self.0.get_chain_info(request).await?.into_inner();
            let network = generated::Network::try_from(network)
                .map_err(|_| Error::UnknownEnumTag {
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
        ) -> Result<BlockHeaderInfo, Error> {
            let request = generated::GetChainTipRequest {};
            let generated::GetChainTipResponse { block_header_info } =
                self.0.get_chain_tip(request).await?.into_inner();
            let Some(block_header_info) = block_header_info else {
                return Err(Error::missing_field::<
                    generated::GetChainTipResponse,
                >("block_header_info"));
            };
            block_header_info.try_into()
        }

        pub async fn get_two_way_peg_data(
            &mut self,
            start_block_hash: Option<BlockHash>,
            end_block_hash: BlockHash,
        ) -> Result<TwoWayPegData, Error> {
            let request = generated::GetTwoWayPegDataRequest {
                sidechain_id: Some(THIS_SIDECHAIN as u32),
                start_block_hash: start_block_hash.map(|start_block_hash| {
                    generated::ReverseHex::encode(&start_block_hash)
                }),
                end_block_hash: Some(generated::ReverseHex::encode(
                    &end_block_hash,
                )),
            };
            self.0
                .get_two_way_peg_data(request)
                .await?
                .into_inner()
                .try_into()
        }

        pub async fn subscribe_events(
            &mut self,
        ) -> Result<BoxStream<'_, Result<Event, Error>>, Error> {
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
        ) -> Result<(), Error> {
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
            critical_hash: BlockHash,
            prev_bytes: [u8; 4],
        ) -> Result<Txid, Error> {
            let request = generated::CreateBmmCriticalDataTransactionRequest {
                sidechain_id: Some(THIS_SIDECHAIN as u32),
                value_sats: Some(value_sats),
                height: Some(height),
                critical_hash: Some(generated::ConsensusHex::consensus_encode(
                    &critical_hash,
                )),
                prev_bytes: Some(generated::ConsensusHex::consensus_encode(
                    &prev_bytes,
                )),
            };
            let generated::CreateBmmCriticalDataTransactionResponse { txid } =
                self.0
                    .create_bmm_critical_data_transaction(request)
                    .await?
                    .into_inner();
            let txid = txid.ok_or_else(||
                Error::missing_field::<generated::CreateBmmCriticalDataTransactionResponse>("txid"))?
                .consensus_decode::<generated::CreateBmmCriticalDataTransactionResponse, _>("txid")?;
            Ok(txid)
        }

        pub async fn create_deposit_tx(
            &mut self,
            address: String,
            value_sats: u64,
            fee_sats: u64,
        ) -> Result<Txid, Error> {
            let request = generated::CreateDepositTransactionRequest {
                sidechain_id: THIS_SIDECHAIN as u32,
                address,
                value_sats,
                fee_sats,
            };
            let generated::CreateDepositTransactionResponse { txid } = self
                .0
                .create_deposit_transaction(request)
                .await?
                .into_inner();
            let txid = txid.ok_or_else(||
                Error::missing_field::<generated::CreateDepositTransactionResponse>("txid"))?
                .consensus_decode::<generated::CreateDepositTransactionResponse, _>("txid")?;
            Ok(txid)
        }

        pub async fn create_new_address(
            &mut self,
            label: Option<String>,
            address_type: AddressType,
        ) -> Result<bitcoin::Address<bitcoin::address::NetworkUnchecked>, Error>
        {
            let request = generated::CreateNewAddressRequest {
                label,
                address_type: generated::AddressType::from(address_type) as i32,
            };
            let generated::CreateNewAddressResponse { address } =
                self.0.create_new_address(request).await?.into_inner();
            let address =
                address.parse().map_err(|_| {
                    Error::invalid_field_value::<
                        generated::CreateNewAddressResponse,
                    >("address", &address)
                })?;
            Ok(address)
        }

        pub async fn generate_blocks(
            &mut self,
            blocks: u32,
        ) -> Result<(), Error> {
            let request = generated::GenerateBlocksRequest {
                blocks: Some(blocks),
            };
            let generated::GenerateBlocksResponse {} =
                self.0.generate_blocks(request).await?.into_inner();
            Ok(())
        }
    }
}
