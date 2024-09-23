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
        tonic::include_proto!("cusf.mainchain");
    }

    #[derive(Debug, Error)]
    pub enum Error {
        #[error(transparent)]
        Grpc(#[from] tonic::Status),
        #[error(
            "Invalid field data in message `{message_name}`: `{field_name}`"
        )]
        InvalidFieldData {
            message_name: String,
            field_name: String,
        },
        #[error("Invalid variant in enum `{enum_name}`: `{variant_name}`")]
        InvalidVariant {
            enum_name: String,
            variant_name: String,
        },
        #[error("Missing field in message `{message_name}`: `{field_name}`")]
        MissingField {
            message_name: String,
            field_name: String,
        },
        #[error("Unknown tag for enum `{enum_name}`: `{tag}`")]
        UnknownTag { enum_name: String, tag: i32 },
    }

    impl Error {
        fn invalid_field_data<Message>(field_name: &str) -> Self
        where
            Message: prost::Name,
        {
            Self::InvalidFieldData {
                message_name: Message::full_name(),
                field_name: field_name.to_owned(),
            }
        }

        fn missing_field<Message>(field_name: &str) -> Self
        where
            Message: prost::Name,
        {
            Self::MissingField {
                message_name: Message::full_name(),
                field_name: field_name.to_owned(),
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

    impl TryFrom<generated::Network> for bitcoin::Network {
        type Error = Error;

        fn try_from(network: generated::Network) -> Result<Self, Self::Error> {
            match network {
                generated::Network::Unspecified => Err(Error::InvalidVariant {
                    enum_name: "Network".to_owned(),
                    variant_name: "Unspecified".to_owned(),
                }),
                generated::Network::Mainnet => Ok(bitcoin::Network::Bitcoin),
                generated::Network::Regtest => Ok(bitcoin::Network::Regtest),
                generated::Network::Signet => Ok(bitcoin::Network::Signet),
                generated::Network::Testnet => Ok(bitcoin::Network::Testnet),
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
            generated::get_bmm_h_star_commitments_response::BlockNotFoundError,
        > for BlockNotFoundError
    {
        type Error = Error;
        fn try_from(
            err: generated::get_bmm_h_star_commitments_response::BlockNotFoundError,
        ) -> Result<Self, Self::Error> {
            let generated::get_bmm_h_star_commitments_response::BlockNotFoundError { block_hash }
                = err;
            let block_hash = block_hash.try_into().map_err(|_|
                Error::invalid_field_data::<generated::get_bmm_h_star_commitments_response::BlockNotFoundError>("block_hash")
            )?;
            Ok(Self(BlockHash::from_byte_array(block_hash)))
        }
    }

    impl TryFrom<generated::get_bmm_h_star_commitments_response::Commitments>
        for Vec<crate::types::BlockHash>
    {
        type Error = Error;

        fn try_from(
            commitments: generated::get_bmm_h_star_commitments_response::Commitments,
        ) -> Result<Self, Self::Error> {
            let generated::get_bmm_h_star_commitments_response::Commitments {
                commitments,
            } = commitments;
            commitments
                .into_iter()
                .map(|commitment_bytes| {
                    let commitment: crate::types::Hash = commitment_bytes
                        .try_into()
                        .map_err(|_|
                            Error::invalid_field_data::<generated::get_bmm_h_star_commitments_response::Commitments>(
                                "commitments"
                            )
                        )?;
                    Ok(commitment.into())
                })
                .collect()
        }
    }

    impl TryFrom<generated::get_bmm_h_star_commitments_response::Result>
        for Result<Vec<crate::types::BlockHash>, BlockNotFoundError>
    {
        type Error = Error;

        fn try_from(
            res: generated::get_bmm_h_star_commitments_response::Result,
        ) -> Result<Self, Self::Error> {
            use generated::get_bmm_h_star_commitments_response as resp;
            match res {
                resp::Result::BlockNotFound(err) => Ok(Err(err.try_into()?)),
                resp::Result::Commitments(commitments) => {
                    commitments.try_into().map(Ok)
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
            let txid: <Txid as bitcoin::hashes::Hash>::Bytes =
                txid.try_into().map_err(|_| {
                    Error::invalid_field_data::<generated::OutPoint>("txid")
                })?;
            Ok(Self {
                txid: Txid::from_byte_array(txid),
                vout,
            })
        }
    }

    impl TryFrom<generated::Output> for Output {
        type Error = Error;

        fn try_from(output: generated::Output) -> Result<Self, Self::Error> {
            let generated::Output {
                address,
                value_sats,
            } = output;
            let address = match address.try_into() {
                Ok(address) => crate::types::Address(address),
                Err(_) => {
                    return Err(Error::invalid_field_data::<generated::Output>(
                        "address",
                    ))
                }
            };
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
        Failed,
        Confirmed,
    }

    impl TryFrom<generated::WithdrawalBundleEventType> for WithdrawalBundleStatus {
        type Error = Error;

        fn try_from(
            status: generated::WithdrawalBundleEventType,
        ) -> Result<Self, Self::Error> {
            match status {
                generated::WithdrawalBundleEventType::SubmittedUnspecified => {
                    Err(Error::InvalidVariant {
                        enum_name: "WithdrawalBundleEventType".to_owned(),
                        variant_name: "SubmittedUnspecified".to_string(),
                    })
                }
                generated::WithdrawalBundleEventType::Failed => {
                    Ok(Self::Failed)
                }
                generated::WithdrawalBundleEventType::Succeded => {
                    Ok(Self::Confirmed)
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
            let txid = m6id.try_into().map_err(|_| {
                Error::invalid_field_data::<generated::WithdrawalBundleEvent>(
                    "m6id",
                )
            })?;
            let status = generated::WithdrawalBundleEventType::try_from(
                withdrawal_bundle_event_type,
            )
            .map_err(|_| Error::UnknownTag {
                enum_name: "WithdrawalBundleEventType".to_owned(),
                tag: withdrawal_bundle_event_type,
            })?;
            Ok((Txid::from_byte_array(txid), status.try_into()?))
        }
    }

    #[derive(Clone, Debug, Default, Deserialize, Serialize)]
    pub struct BlockInfo {
        pub deposits: Vec<Deposit>,
        pub withdrawal_bundle_status: Option<(Txid, WithdrawalBundleStatus)>,
        pub bmm_commitment: Option<crate::types::BlockHash>,
    }

    impl TryFrom<generated::BlockInfo> for BlockInfo {
        type Error = Error;

        fn try_from(
            block_info: generated::BlockInfo,
        ) -> Result<Self, Self::Error> {
            let generated::BlockInfo {
                deposits,
                withdrawal_bundle_event,
                bmm_commitment,
            } = block_info;
            let deposits = deposits
                .into_iter()
                .map(|deposit| deposit.try_into())
                .collect::<Result<Vec<Deposit>, _>>()?;
            let withdrawal_bundle_status = withdrawal_bundle_event
                .map(|withdrawal_bundle_event| {
                    withdrawal_bundle_event.try_into()
                })
                .transpose()?;
            let bmm_commitment = bmm_commitment
                .map(|bmm_commitment| match bmm_commitment.try_into() {
                    Ok(bmm_commitment) => {
                        Ok(crate::types::BlockHash(bmm_commitment))
                    }
                    Err(_) => Err(Error::invalid_field_data::<
                        generated::BlockInfo,
                    >("bmm_commitment")),
                })
                .transpose()?;
            Ok(Self {
                deposits,
                withdrawal_bundle_status,
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
            self.block_info
                .iter()
                .filter_map(|(block_hash, block_info)| {
                    if let Some((txid, status)) =
                        block_info.withdrawal_bundle_status
                    {
                        Some((*block_hash, txid, status))
                    } else {
                        None
                    }
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
            let block_hash = block_hash.try_into().map_err(|_| {
                Error::invalid_field_data::<generated::BlockHeaderInfo>(
                    "block_hash",
                )
            })?;
            let prev_block_hash = prev_block_hash.try_into().map_err(|_| {
                Error::invalid_field_data::<generated::BlockHeaderInfo>(
                    "prev_block_hash",
                )
            })?;
            let work = work.try_into().map_err(|_| {
                Error::invalid_field_data::<generated::BlockHeaderInfo>("work")
            })?;
            Ok(BlockHeaderInfo {
                block_hash: BlockHash::from_byte_array(block_hash),
                prev_block_hash: BlockHash::from_byte_array(prev_block_hash),
                height,
                work: Work::from_le_bytes(work),
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

    impl TryFrom<generated::event_response::event::Event> for Event {
        type Error = Error;

        fn try_from(
            event: generated::event_response::event::Event,
        ) -> Result<Self, Self::Error> {
            use generated::event_response::event;
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
                    let block_hash = block_hash.try_into().map_err(|_| {
                        Error::invalid_field_data::<event::DisconnectBlock>(
                            "block_hash",
                        )
                    })?;
                    let block_hash = BlockHash::from_byte_array(block_hash);
                    Ok(Self::DisconnectBlock { block_hash })
                }
            }
        }
    }

    impl TryFrom<generated::event_response::Event> for Event {
        type Error = Error;

        fn try_from(
            event: generated::event_response::Event,
        ) -> Result<Self, Self::Error> {
            let generated::event_response::Event { event } = event;
            let Some(event) = event else {
                return Err(Error::missing_field::<
                    generated::event_response::Event,
                >("event"));
            };
            event.try_into()
        }
    }

    impl TryFrom<generated::EventResponse> for Event {
        type Error = Error;

        fn try_from(
            event: generated::EventResponse,
        ) -> Result<Self, Self::Error> {
            let generated::EventResponse { event } = event;
            let Some(event) = event else {
                return Err(Error::missing_field::<generated::EventResponse>(
                    "event",
                ));
            };
            event.try_into()
        }
    }

    pub struct EventStream;

    #[derive(Clone, Debug)]
    #[repr(transparent)]
    pub struct Client<T>(pub generated::mainchain_client::MainchainClient<T>);

    impl<T> Client<T>
    where
        T: super::Transport,
    {
        pub fn new(inner: T) -> Self {
            Self(generated::mainchain_client::MainchainClient::<T>::new(
                inner,
            ))
        }

        pub async fn broadcast_withdrawal_bundle(
            &mut self,
            transaction: &Transaction,
        ) -> Result<(), Error> {
            let request = generated::BroadcastWithdrawalBundleRequest {
                sidechain_id: THIS_SIDECHAIN as u32,
                transaction: bitcoin::consensus::serialize(transaction),
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
                sidechain_id: THIS_SIDECHAIN as u32,
                value_sats,
                height,
                critical_hash: critical_hash.to_byte_array().to_vec(),
                prev_bytes: prev_bytes.to_vec(),
            };
            let generated::CreateBmmCriticalDataTransactionResponse { txid } =
                self.0
                    .create_bmm_critical_data_transaction(request)
                    .await?
                    .into_inner();
            let txid = txid.try_into().map_err(|_| {
                Error::invalid_field_data::<
                    generated::CreateBmmCriticalDataTransactionResponse,
                >("txid")
            })?;
            Ok(Txid::from_byte_array(txid))
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
            let txid = txid.try_into().map_err(|_| {
                Error::invalid_field_data::<
                    generated::CreateDepositTransactionResponse,
                >("txid")
            })?;
            Ok(Txid::from_byte_array(txid))
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
                    Error::invalid_field_data::<
                        generated::CreateNewAddressResponse,
                    >("address")
                })?;
            Ok(address)
        }

        pub async fn generate_blocks(
            &mut self,
            blocks: u32,
        ) -> Result<(), Error> {
            let request = generated::GenerateBlocksRequest { blocks };
            let generated::GenerateBlocksResponse {} =
                self.0.generate_blocks(request).await?.into_inner();
            Ok(())
        }

        pub async fn get_block_header_info(
            &mut self,
            block_hash: BlockHash,
        ) -> Result<BlockHeaderInfo, Error> {
            let request = generated::GetBlockHeaderInfoRequest {
                block_hash: block_hash.to_raw_hash().to_byte_array().to_vec(),
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
            Result<Vec<crate::types::BlockHash>, BlockNotFoundError>,
            Error,
        > {
            let request = generated::GetBmmHStarCommitmentsRequest {
                block_hash: block_hash.to_raw_hash().to_byte_array().to_vec(),
                sidechain_id: THIS_SIDECHAIN as u32,
            };
            let generated::GetBmmHStarCommitmentsResponse { result } = self
                .0
                .get_bmm_h_star_commitments(request)
                .await?
                .into_inner();
            let Some(result) = result else {
                return Err(Error::missing_field::<
                    generated::GetBmmHStarCommitmentsResponse,
                >("result"));
            };
            result.try_into()
        }

        pub async fn get_chain_info(&mut self) -> Result<ChainInfo, Error> {
            let request = generated::GetChainInfoRequest {};
            let generated::GetChainInfoResponse { network } =
                self.0.get_chain_info(request).await?.into_inner();
            let network = generated::Network::try_from(network)
                .map_err(|_| Error::UnknownTag {
                    enum_name: "Network".to_owned(),
                    tag: network,
                })?
                .try_into()?;
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
                sidechain_id: THIS_SIDECHAIN as u32,
                start_block_hash: start_block_hash.map(|start_block_hash| {
                    start_block_hash.to_byte_array().to_vec()
                }),
                end_block_hash: end_block_hash.to_byte_array().to_vec(),
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
                sidechain_id: THIS_SIDECHAIN as u32,
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
}
