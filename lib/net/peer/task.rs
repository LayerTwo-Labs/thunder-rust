//! Peer connection task

use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    sync::{Arc, atomic::AtomicBool},
};

use fallible_iterator::FallibleIterator;
use futures::{StreamExt as _, channel::mpsc};
use quinn::SendStream;
use sneed::EnvError;

use crate::{
    net::peer::{
        BanReason, Connection, ConnectionContext, Info, PeerState, PeerStateId,
        Request, TipInfo,
        error::Error,
        mailbox::{self, InternalMessage, MailboxItem},
        message::{self, Heartbeat, RequestMessage, ResponseMessage},
        request_queue,
    },
    types::{
        AuthorizedTransaction, BlockHash, BmmResult, Header, Tip, VERSION,
    },
};

pub(in crate::net::peer) struct ConnectionTask {
    pub connection: Connection,
    pub ctxt: ConnectionContext,
    pub info_tx: mpsc::UnboundedSender<Info>,
    /// Sender for the task's mailbox
    pub mailbox_rx: mailbox::Receiver,
    /// Receiver for the task's mailbox
    pub mailbox_tx: mailbox::Sender,
    /// `True` if a valid message has been received successfully
    pub received_msg_successfully: Arc<AtomicBool>,
}

impl ConnectionTask {
    /// Check if peer tip is better, requesting headers if necessary.
    /// Returns `Some(true)` if the peer tip is better and headers are available,
    /// `Some(false)` if the peer tip is better and headers were requested,
    /// and `None` if the peer tip is not better.
    fn check_peer_tip_and_request_headers(
        ctxt: &ConnectionContext,
        request_queue: &mut request_queue::Sender,
        tip_info: Option<&TipInfo>,
        peer_tip_info: &TipInfo,
        peer_state_id: PeerStateId,
    ) -> Result<Option<bool>, Error> {
        // Check if the peer tip is better, requesting headers if necessary
        let Some(tip_info) = tip_info else {
            // No tip.
            // Request headers from peer if necessary
            let rotxn = ctxt.env.read_txn().map_err(EnvError::from)?;
            if ctxt
                .archive
                .try_get_header(&rotxn, peer_tip_info.tip.block_hash)?
                .is_none()
            {
                let request = message::GetHeadersRequest {
                    start: HashSet::new(),
                    end: peer_tip_info.tip.block_hash,
                    height: Some(peer_tip_info.block_height),
                    peer_state_id: Some(peer_state_id),
                };
                let _: bool = request_queue.send_request(request.into())?;
                return Ok(Some(false));
            } else {
                return Ok(Some(true));
            }
        };
        match (
            tip_info.total_work.cmp(&peer_tip_info.total_work),
            tip_info.block_height.cmp(&peer_tip_info.block_height),
        ) {
            (Ordering::Less | Ordering::Equal, Ordering::Less) => {
                // No tip ancestor can have greater height,
                // so peer tip is better.
                // Request headers if necessary
                let rotxn = ctxt.env.read_txn().map_err(EnvError::from)?;
                if ctxt
                    .archive
                    .try_get_header(&rotxn, peer_tip_info.tip.block_hash)?
                    .is_none()
                {
                    let start =
                        HashSet::from_iter(ctxt.archive.get_block_locator(
                            &rotxn,
                            tip_info.tip.block_hash,
                        )?);
                    let request = message::GetHeadersRequest {
                        start,
                        end: peer_tip_info.tip.block_hash,
                        height: Some(peer_tip_info.block_height),
                        peer_state_id: Some(peer_state_id),
                    };
                    let _: bool = request_queue.send_request(request.into())?;
                    Ok(Some(false))
                } else {
                    Ok(Some(true))
                }
            }
            (Ordering::Equal | Ordering::Greater, Ordering::Greater) => {
                // No peer tip ancestor can have greater height,
                // so tip is better.
                // Nothing to do in this case
                Ok(None)
            }
            (Ordering::Less, Ordering::Equal) => {
                // Within the same mainchain lineage, prefer lower work
                // Otherwise, prefer tip with greater work
                let rotxn = ctxt.env.read_txn().map_err(EnvError::from)?;
                if ctxt.archive.shared_mainchain_lineage(
                    &rotxn,
                    tip_info.tip.main_block_hash,
                    peer_tip_info.tip.main_block_hash,
                )? {
                    // Nothing to do in this case
                    return Ok(None);
                }
                // Request headers if necessary
                if ctxt
                    .archive
                    .try_get_header(&rotxn, peer_tip_info.tip.block_hash)?
                    .is_none()
                {
                    let start =
                        HashSet::from_iter(ctxt.archive.get_block_locator(
                            &rotxn,
                            tip_info.tip.block_hash,
                        )?);
                    let request = message::GetHeadersRequest {
                        start,
                        end: peer_tip_info.tip.block_hash,
                        height: Some(peer_tip_info.block_height),
                        peer_state_id: Some(peer_state_id),
                    };
                    let _: bool = request_queue.send_request(request.into())?;
                    Ok(Some(false))
                } else {
                    Ok(Some(true))
                }
            }
            (Ordering::Greater, Ordering::Equal) => {
                // Within the same mainchain lineage, prefer lower work
                // Otherwise, prefer tip with greater work
                let rotxn = ctxt.env.read_txn().map_err(EnvError::from)?;
                if !ctxt.archive.shared_mainchain_lineage(
                    &rotxn,
                    tip_info.tip.main_block_hash,
                    peer_tip_info.tip.main_block_hash,
                )? {
                    // Nothing to do in this case
                    return Ok(None);
                }
                // Request headers if necessary
                if ctxt
                    .archive
                    .try_get_header(&rotxn, peer_tip_info.tip.block_hash)?
                    .is_none()
                {
                    let start =
                        HashSet::from_iter(ctxt.archive.get_block_locator(
                            &rotxn,
                            tip_info.tip.block_hash,
                        )?);
                    let request = message::GetHeadersRequest {
                        start,
                        end: peer_tip_info.tip.block_hash,
                        height: Some(peer_tip_info.block_height),
                        peer_state_id: Some(peer_state_id),
                    };
                    let _: bool = request_queue.send_request(request.into())?;
                    Ok(Some(false))
                } else {
                    Ok(Some(true))
                }
            }
            (Ordering::Less, Ordering::Greater) => {
                // Need to check if tip ancestor before common
                // mainchain ancestor had greater or equal height
                let rotxn = ctxt.env.read_txn().map_err(EnvError::from)?;
                let main_ancestor = ctxt.archive.last_common_main_ancestor(
                    &rotxn,
                    tip_info.tip.main_block_hash,
                    peer_tip_info.tip.main_block_hash,
                )?;
                let tip_ancestor_height = ctxt
                    .archive
                    .ancestors(&rotxn, tip_info.tip.block_hash)
                    .find_map(|tip_ancestor| {
                        let header =
                            ctxt.archive.get_header(&rotxn, tip_ancestor)?;
                        if !ctxt.archive.is_main_descendant(
                            &rotxn,
                            header.prev_main_hash,
                            main_ancestor,
                        )? {
                            return Ok(None);
                        }
                        if header.prev_main_hash == main_ancestor {
                            return Ok(None);
                        }
                        ctxt.archive.get_height(&rotxn, tip_ancestor).map(Some)
                    })?;
                if tip_ancestor_height >= Some(peer_tip_info.block_height) {
                    // Nothing to do in this case
                    return Ok(None);
                }
                // Request headers if necessary
                if ctxt
                    .archive
                    .try_get_header(&rotxn, peer_tip_info.tip.block_hash)?
                    .is_none()
                {
                    let start =
                        HashSet::from_iter(ctxt.archive.get_block_locator(
                            &rotxn,
                            tip_info.tip.block_hash,
                        )?);
                    let request = message::GetHeadersRequest {
                        start,
                        end: peer_tip_info.tip.block_hash,
                        height: Some(peer_tip_info.block_height),
                        peer_state_id: Some(peer_state_id),
                    };
                    let _: bool = request_queue.send_request(request.into())?;
                    Ok(Some(false))
                } else {
                    Ok(Some(true))
                }
            }
            (Ordering::Greater, Ordering::Less) => {
                // Need to check if peer's tip ancestor before common
                // mainchain ancestor had greater or equal height
                let rotxn = ctxt.env.read_txn().map_err(EnvError::from)?;
                if ctxt
                    .archive
                    .try_get_header(&rotxn, peer_tip_info.tip.block_hash)?
                    .is_none()
                {
                    let start =
                        HashSet::from_iter(ctxt.archive.get_block_locator(
                            &rotxn,
                            tip_info.tip.block_hash,
                        )?);
                    let request = message::GetHeadersRequest {
                        start,
                        end: peer_tip_info.tip.block_hash,
                        height: Some(peer_tip_info.block_height),
                        peer_state_id: Some(peer_state_id),
                    };
                    let _: bool = request_queue.send_request(request.into())?;
                    return Ok(Some(false));
                }
                let main_ancestor = ctxt.archive.last_common_main_ancestor(
                    &rotxn,
                    tip_info.tip.main_block_hash,
                    peer_tip_info.tip.main_block_hash,
                )?;
                let peer_tip_ancestor_height = ctxt
                    .archive
                    .ancestors(&rotxn, peer_tip_info.tip.block_hash)
                    .find_map(|peer_tip_ancestor| {
                        let header = ctxt
                            .archive
                            .get_header(&rotxn, peer_tip_ancestor)?;
                        if !ctxt.archive.is_main_descendant(
                            &rotxn,
                            header.prev_main_hash,
                            main_ancestor,
                        )? {
                            return Ok(None);
                        }
                        if header.prev_main_hash == main_ancestor {
                            return Ok(None);
                        }
                        ctxt.archive
                            .get_height(&rotxn, peer_tip_ancestor)
                            .map(Some)
                    })?;
                if peer_tip_ancestor_height < Some(tip_info.block_height) {
                    // Nothing to do in this case
                    Ok(None)
                } else {
                    Ok(Some(true))
                }
            }
            (Ordering::Equal, Ordering::Equal) => {
                // If the peer tip is the same as the tip, nothing to do
                if peer_tip_info.tip.block_hash == tip_info.tip.block_hash {
                    return Ok(None);
                }
                // Need to compare tip ancestor and peer's tip ancestor
                // before common mainchain ancestor
                let rotxn = ctxt.env.read_txn().map_err(EnvError::from)?;
                if ctxt
                    .archive
                    .try_get_header(&rotxn, peer_tip_info.tip.block_hash)?
                    .is_none()
                {
                    let start =
                        HashSet::from_iter(ctxt.archive.get_block_locator(
                            &rotxn,
                            tip_info.tip.block_hash,
                        )?);
                    let request = message::GetHeadersRequest {
                        start,
                        end: peer_tip_info.tip.block_hash,
                        height: Some(peer_tip_info.block_height),
                        peer_state_id: Some(peer_state_id),
                    };
                    let _: bool = request_queue.send_request(request.into())?;
                    return Ok(Some(true));
                }
                let main_ancestor = ctxt.archive.last_common_main_ancestor(
                    &rotxn,
                    tip_info.tip.main_block_hash,
                    peer_tip_info.tip.main_block_hash,
                )?;
                let main_ancestor_height =
                    ctxt.archive.get_main_height(&rotxn, main_ancestor)?;
                let (tip_ancestor_height, tip_ancestor_work) = ctxt
                    .archive
                    .ancestors(&rotxn, tip_info.tip.block_hash)
                    .find_map(|tip_ancestor| {
                        let header =
                            ctxt.archive.get_header(&rotxn, tip_ancestor)?;
                        if !ctxt.archive.is_main_descendant(
                            &rotxn,
                            header.prev_main_hash,
                            main_ancestor,
                        )? {
                            return Ok(None);
                        }
                        if header.prev_main_hash == main_ancestor {
                            return Ok(None);
                        }
                        let height =
                            ctxt.archive.get_height(&rotxn, tip_ancestor)?;
                        // Find mainchain block hash to get total work
                        let main_block = {
                            let prev_height = ctxt.archive.get_main_height(
                                &rotxn,
                                header.prev_main_hash,
                            )?;
                            let height = prev_height + 1;
                            ctxt.archive.get_nth_main_ancestor(
                                &rotxn,
                                main_ancestor,
                                main_ancestor_height - height,
                            )?
                        };
                        let work =
                            ctxt.archive.get_total_work(&rotxn, main_block)?;
                        Ok(Some((height, work)))
                    })?
                    .map_or((None, None), |(height, work)| {
                        (Some(height), Some(work))
                    });
                let (peer_tip_ancestor_height, peer_tip_ancestor_work) = ctxt
                    .archive
                    .ancestors(&rotxn, peer_tip_info.tip.block_hash)
                    .find_map(|peer_tip_ancestor| {
                        let header = ctxt
                            .archive
                            .get_header(&rotxn, peer_tip_ancestor)?;
                        if !ctxt.archive.is_main_descendant(
                            &rotxn,
                            header.prev_main_hash,
                            main_ancestor,
                        )? {
                            return Ok(None);
                        }
                        if header.prev_main_hash == main_ancestor {
                            return Ok(None);
                        }
                        let height = ctxt
                            .archive
                            .get_height(&rotxn, peer_tip_ancestor)?;
                        // Find mainchain block hash to get total work
                        let main_block = {
                            let prev_height = ctxt.archive.get_main_height(
                                &rotxn,
                                header.prev_main_hash,
                            )?;
                            let height = prev_height + 1;
                            ctxt.archive.get_nth_main_ancestor(
                                &rotxn,
                                main_ancestor,
                                main_ancestor_height - height,
                            )?
                        };
                        let work =
                            ctxt.archive.get_total_work(&rotxn, main_block)?;
                        Ok(Some((height, work)))
                    })?
                    .map_or((None, None), |(height, work)| {
                        (Some(height), Some(work))
                    });
                match (
                    tip_ancestor_work.cmp(&peer_tip_ancestor_work),
                    tip_ancestor_height.cmp(&peer_tip_ancestor_height),
                ) {
                    (Ordering::Less | Ordering::Equal, Ordering::Equal)
                    | (_, Ordering::Greater) => {
                        // Peer tip is not better, nothing to do
                        Ok(None)
                    }
                    (Ordering::Greater, Ordering::Equal)
                    | (_, Ordering::Less) => {
                        // Peer tip is better
                        Ok(Some(true))
                    }
                }
            }
        }
    }

    /// * Request any missing mainchain headers
    /// * Check claimed work
    /// * Check that BMM commitment matches peer tip
    /// * Check if peer tip is better, requesting headers if necessary
    /// * If peer tip is better:
    ///   * request headers if missing
    ///   * verify BMM
    ///   * request missing bodies
    ///   * notify net task / node that new tip is ready
    async fn handle_peer_state(
        ctxt: &ConnectionContext,
        info_tx: &mpsc::UnboundedSender<Info>,
        request_queue: &mut request_queue::Sender,
        peer_state: &PeerState,
    ) -> Result<(), Error> {
        let Some(peer_tip_info) = peer_state.tip_info else {
            // Nothing to do in this case
            return Ok(());
        };
        let tip_info = 'tip_info: {
            let rotxn = ctxt.env.read_txn().map_err(EnvError::from)?;
            let Some(tip) = ctxt.state.try_get_tip(&rotxn)? else {
                break 'tip_info None;
            };
            let tip_height = ctxt
                .state
                .try_get_height(&rotxn)?
                .expect("Height should be known for tip");
            let bmm_verification =
                ctxt.archive.get_best_main_verification(&rotxn, tip)?;
            let total_work =
                ctxt.archive.get_total_work(&rotxn, bmm_verification)?;
            let tip = Tip {
                block_hash: tip,
                main_block_hash: bmm_verification,
            };
            Some(TipInfo {
                tip,
                block_height: tip_height,
                total_work,
            })
        };
        // Check claimed work, request mainchain headers if necessary, verify
        // BMM
        {
            let rotxn = ctxt.env.read_txn().map_err(EnvError::from)?;
            match ctxt.archive.try_get_main_header_info(
                &rotxn,
                &peer_tip_info.tip.main_block_hash,
            )? {
                None => {
                    let info = Info::NeedMainchainAncestors {
                        main_hash: peer_tip_info.tip.main_block_hash,
                        peer_state_id: peer_state.into(),
                    };
                    info_tx
                        .unbounded_send(info)
                        .map_err(|_| Error::SendInfo)?;
                    return Ok(());
                }
                Some(_main_header_info) => {
                    let computed_total_work = ctxt.archive.get_total_work(
                        &rotxn,
                        peer_tip_info.tip.main_block_hash,
                    )?;
                    if peer_tip_info.total_work != computed_total_work {
                        let ban_reason = BanReason::IncorrectTotalWork {
                            tip: peer_tip_info.tip,
                            total_work: peer_tip_info.total_work,
                        };
                        return Err(Error::PeerBan(ban_reason));
                    }
                    let bmm_commitment = ctxt
                        .archive
                        .get_main_block_info(
                            &rotxn,
                            &peer_tip_info.tip.main_block_hash,
                        )?
                        .bmm_commitment;
                    if bmm_commitment != Some(peer_tip_info.tip.block_hash) {
                        let ban_reason =
                            BanReason::BmmVerificationFailed(peer_tip_info.tip);
                        return Err(Error::PeerBan(ban_reason));
                    }
                }
            }
        }
        // Check if the peer tip is better, requesting headers if necessary
        match Self::check_peer_tip_and_request_headers(
            ctxt,
            request_queue,
            tip_info.as_ref(),
            &peer_tip_info,
            peer_state.into(),
        )? {
            Some(false) | None => return Ok(()),
            Some(true) => (),
        }
        // Check BMM now that headers are available
        {
            let rotxn = ctxt.env.read_txn().map_err(EnvError::from)?;
            let Some(BmmResult::Verified) = ctxt.archive.try_get_bmm_result(
                &rotxn,
                peer_tip_info.tip.block_hash,
                peer_tip_info.tip.main_block_hash,
            )?
            else {
                let ban_reason =
                    BanReason::BmmVerificationFailed(peer_tip_info.tip);
                return Err(Error::PeerBan(ban_reason));
            };
        }
        // Request missing bodies, or notify that a new tip is ready
        let (common_ancestor, missing_bodies): (
            Option<BlockHash>,
            Vec<BlockHash>,
        ) = {
            let rotxn = ctxt.env.read_txn().map_err(EnvError::from)?;
            let common_ancestor = if let Some(tip_info) = tip_info {
                ctxt.archive.last_common_ancestor(
                    &rotxn,
                    tip_info.tip.block_hash,
                    peer_tip_info.tip.block_hash,
                )?
            } else {
                None
            };
            let missing_bodies = ctxt.archive.get_missing_bodies(
                &rotxn,
                peer_tip_info.tip.block_hash,
                common_ancestor,
            )?;
            (common_ancestor, missing_bodies)
        };
        if missing_bodies.is_empty() {
            let info = Info::NewTipReady(peer_tip_info.tip);
            info_tx.unbounded_send(info).map_err(|_| Error::SendInfo)?;
        } else {
            const MAX_BLOCK_REQUESTS: usize = 100;
            // Request missing bodies
            missing_bodies
                .into_iter()
                .take(MAX_BLOCK_REQUESTS)
                .try_for_each(|block_hash| {
                    let request = message::GetBlockRequest {
                        block_hash,
                        descendant_tip: Some(peer_tip_info.tip),
                        peer_state_id: Some(peer_state.into()),
                        ancestor: common_ancestor,
                    };
                    let _: bool = request_queue.send_request(request.into())?;
                    Ok::<_, Error>(())
                })?;
        }
        Ok(())
    }

    async fn handle_get_block(
        ctxt: &ConnectionContext,
        response_tx: SendStream,
        block_hash: BlockHash,
    ) -> Result<(), Error> {
        let (header, body) = {
            let rotxn = ctxt.env.read_txn().map_err(EnvError::from)?;
            let header = ctxt.archive.try_get_header(&rotxn, block_hash)?;
            let body = ctxt.archive.try_get_body(&rotxn, block_hash)?;
            (header, body)
        };
        let resp = match (header, body) {
            (Some(header), Some(body)) => {
                ResponseMessage::Block { header, body }
            }
            (_, _) => ResponseMessage::NoBlock { block_hash },
        };
        let () = Connection::send_response(response_tx, resp).await?;
        Ok(())
    }

    async fn handle_get_headers(
        ctxt: &ConnectionContext,
        response_tx: SendStream,
        start: HashSet<BlockHash>,
        end: BlockHash,
    ) -> Result<(), Error> {
        let response = {
            let rotxn = ctxt.env.read_txn().map_err(EnvError::from)?;
            if ctxt.archive.try_get_header(&rotxn, end)?.is_some() {
                let mut headers: Vec<Header> = ctxt
                    .archive
                    .ancestors(&rotxn, end)
                    .take_while(|block_hash| Ok(!start.contains(block_hash)))
                    .map(|block_hash| {
                        ctxt.archive.get_header(&rotxn, block_hash)
                    })
                    .collect()?;
                headers.reverse();
                ResponseMessage::Headers(headers)
            } else {
                ResponseMessage::NoHeader { block_hash: end }
            }
        };
        let () = Connection::send_response(response_tx, response).await?;
        Ok(())
    }

    async fn handle_push_tx(
        ctxt: &ConnectionContext,
        info_tx: &mpsc::UnboundedSender<Info>,
        response_tx: SendStream,
        tx: AuthorizedTransaction,
    ) -> Result<(), Error> {
        let txid = tx.transaction.txid();
        let validate_tx_result = {
            let rotxn = ctxt.env.read_txn().map_err(EnvError::from)?;
            ctxt.state.validate_transaction(&rotxn, &tx)
        };
        match validate_tx_result {
            Err(err) => {
                Connection::send_response(
                    response_tx,
                    ResponseMessage::TransactionRejected(txid),
                )
                .await?;
                Err(Error::from(err))
            }
            Ok(_) => {
                Connection::send_response(
                    response_tx,
                    ResponseMessage::TransactionAccepted(txid),
                )
                .await?;
                info_tx
                    .unbounded_send(Info::NewTransaction(tx))
                    .map_err(|_| Error::SendInfo)?;
                Ok(())
            }
        }
    }

    async fn handle_peer_request(
        ctxt: &ConnectionContext,
        info_tx: &mpsc::UnboundedSender<Info>,
        request_queue: &mut request_queue::Sender,
        peer_state: &mut Option<PeerStateId>,
        // Map associating peer state hashes to peer state
        peer_states: &mut HashMap<PeerStateId, PeerState>,
        response_tx: SendStream,
        request_msg: RequestMessage,
    ) -> Result<(), Error> {
        match request_msg {
            RequestMessage::Heartbeat(heartbeat) => {
                let new_peer_state = heartbeat.0;
                let new_peer_state_id = (&new_peer_state).into();
                peer_states.insert(new_peer_state_id, new_peer_state);
                if *peer_state != Some(new_peer_state_id) {
                    let () = Self::handle_peer_state(
                        ctxt,
                        info_tx,
                        request_queue,
                        &new_peer_state,
                    )
                    .await?;
                    *peer_state = Some(new_peer_state_id);
                }
                Ok(())
            }
            RequestMessage::Request(Request::GetBlock(
                message::GetBlockRequest {
                    block_hash,
                    descendant_tip: _,
                    ancestor: _,
                    peer_state_id: _,
                },
            )) => Self::handle_get_block(ctxt, response_tx, block_hash).await,
            RequestMessage::Request(Request::GetHeaders(
                message::GetHeadersRequest {
                    start,
                    end,
                    height: _,
                    peer_state_id: _,
                },
            )) => Self::handle_get_headers(ctxt, response_tx, start, end).await,
            RequestMessage::Request(Request::PushTransaction(
                message::PushTransactionRequest { transaction },
            )) => {
                Self::handle_push_tx(ctxt, info_tx, response_tx, transaction)
                    .await
            }
        }
    }

    async fn handle_internal_message(
        ctxt: &ConnectionContext,
        info_tx: &mpsc::UnboundedSender<Info>,
        request_queue: &mut request_queue::Sender,
        // known peer states
        peer_states: &HashMap<PeerStateId, PeerState>,
        msg: InternalMessage,
    ) -> Result<(), Error> {
        match msg {
            InternalMessage::ForwardRequest(request) => {
                let _: bool = request_queue.send_request(request)?;
            }
            InternalMessage::BmmVerification { res, peer_state_id } => {
                if let Err(block_not_found) = res {
                    tracing::warn!("{block_not_found}");
                    return Ok(());
                }
                let Some(peer_state) = peer_states.get(&peer_state_id) else {
                    return Err(Error::MissingPeerState(peer_state_id));
                };
                let () = Self::handle_peer_state(
                    ctxt,
                    info_tx,
                    request_queue,
                    peer_state,
                )
                .await?;
            }
            InternalMessage::BmmVerificationError(err) => {
                let err: anyhow::Error = err;
                tracing::error!("Error attempting BMM verification: {err:#}");
            }
            InternalMessage::MainchainAncestorsError(err) => {
                let err: anyhow::Error = err;
                tracing::error!("Error fetching mainchain ancestors: {err:#}");
            }
            InternalMessage::MainchainAncestors(peer_state_id)
            | InternalMessage::Headers(peer_state_id)
            | InternalMessage::BodiesAvailable(peer_state_id) => {
                let Some(peer_state) = peer_states.get(&peer_state_id) else {
                    return Err(Error::MissingPeerState(peer_state_id));
                };
                let () = Self::handle_peer_state(
                    ctxt,
                    info_tx,
                    request_queue,
                    peer_state,
                )
                .await?;
            }
        }
        Ok(())
    }

    pub async fn run(mut self) -> Result<(), Error> {
        // current peer state
        let mut peer_state = Option::<PeerStateId>::None;
        // known peer states
        let mut peer_states = HashMap::<PeerStateId, PeerState>::new();
        let mut mailbox_stream = self
            .mailbox_rx
            .into_stream(self.connection, &self.received_msg_successfully);
        while let Some(mailbox_item) = mailbox_stream.next().await {
            match mailbox_item {
                MailboxItem::Error(err) => return Err(err.into()),
                MailboxItem::InternalMessage(msg) => {
                    let () = Self::handle_internal_message(
                        &self.ctxt,
                        &self.info_tx,
                        &mut self.mailbox_tx.request_tx,
                        &peer_states,
                        msg,
                    )
                    .await?;
                }
                MailboxItem::Heartbeat => {
                    let tip_info = 'tip_info: {
                        let rotxn =
                            self.ctxt.env.read_txn().map_err(EnvError::from)?;
                        let Some(tip) = self.ctxt.state.try_get_tip(&rotxn)?
                        else {
                            break 'tip_info None;
                        };
                        let tip_height = self
                            .ctxt
                            .state
                            .try_get_height(&rotxn)?
                            .expect("Height for tip should be known");
                        let bmm_verification = self
                            .ctxt
                            .archive
                            .get_best_main_verification(&rotxn, tip)?;
                        let total_work = self
                            .ctxt
                            .archive
                            .get_total_work(&rotxn, bmm_verification)?;
                        let tip = Tip {
                            block_hash: tip,
                            main_block_hash: bmm_verification,
                        };
                        Some(TipInfo {
                            tip,
                            block_height: tip_height,
                            total_work,
                        })
                    };
                    let heartbeat_msg = Heartbeat(PeerState {
                        tip_info,
                        version: *VERSION,
                    });
                    self.mailbox_tx.request_tx.send_heartbeat(heartbeat_msg)?;
                }
                MailboxItem::PeerRequest((request, response_tx)) => {
                    let () = Self::handle_peer_request(
                        &self.ctxt,
                        &self.info_tx,
                        &mut self.mailbox_tx.request_tx,
                        &mut peer_state,
                        &mut peer_states,
                        response_tx,
                        request,
                    )
                    .await?;
                }
                MailboxItem::PeerResponse(peer_response) => {
                    let info = peer_response
                        .response
                        .map(|resp| {
                            Info::Response(Box::new((
                                resp,
                                peer_response.request,
                            )))
                        })
                        .into();
                    if self.info_tx.unbounded_send(info).is_err() {
                        tracing::error!("Failed to send response info")
                    };
                }
            }
        }
        Ok(())
    }
}
