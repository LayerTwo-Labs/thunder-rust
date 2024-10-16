//! Task to manage peers and their responses

use std::{
    collections::{HashMap, HashSet},
    net::SocketAddr,
    sync::Arc,
};

use fallible_iterator::{FallibleIterator, IteratorExt};
use futures::{
    channel::{
        mpsc::{self, UnboundedReceiver, UnboundedSender},
        oneshot,
    },
    stream, StreamExt,
};
use heed::RwTxn;
use thiserror::Error;
use tokio::task::JoinHandle;
use tokio_stream::StreamNotifyClose;
use tokio_util::task::LocalPoolHandle;

use super::mainchain_task::{self, MainchainTaskHandle};
use crate::{
    archive::{self, Archive},
    mempool::{self, MemPool},
    net::{
        self, Net, PeerConnectionInfo, PeerConnectionMessage, PeerInfoRx,
        PeerRequest, PeerResponse, PeerStateId,
    },
    state::{self, State},
    types::{
        proto::{self, mainchain},
        BlockHash, BmmResult, Body, Header, Tip,
    },
    util::UnitKey,
};

#[derive(Debug, Error)]
pub enum Error {
    #[error("archive error")]
    Archive(#[from] archive::Error),
    #[error("CUSF mainchain proto error")]
    CusfMainchain(#[from] proto::Error),
    #[error("Forward mainchain task request failed")]
    ForwardMainchainTaskRequest,
    #[error("heed error")]
    Heed(#[from] heed::Error),
    #[error("mempool error")]
    MemPool(#[from] mempool::Error),
    #[error("Net error")]
    Net(#[from] net::Error),
    #[error("peer info stream closed")]
    PeerInfoRxClosed,
    #[error("Receive mainchain task response cancelled")]
    ReceiveMainchainTaskResponse,
    #[error("Receive reorg result cancelled (oneshot)")]
    ReceiveReorgResultOneshot,
    #[error("Send mainchain task request failed")]
    SendMainchainTaskRequest,
    #[error("Send new tip ready failed")]
    SendNewTipReady,
    #[error("Send reorg result error (oneshot)")]
    SendReorgResultOneshot,
    #[error("state error")]
    State(#[from] state::Error),
}

async fn connect_tip_<Transport>(
    rwtxn: &mut RwTxn<'_>,
    archive: &Archive,
    cusf_mainchain: &mut mainchain::ValidatorClient<Transport>,
    mempool: &MemPool,
    state: &State,
    header: &Header,
    body: &Body,
) -> Result<(), Error>
where
    Transport: proto::Transport,
{
    let last_deposit_block_hash = state.get_last_deposit_block_hash(rwtxn)?;
    let two_way_peg_data = cusf_mainchain
        .get_two_way_peg_data(last_deposit_block_hash, header.prev_main_hash)
        .await?;
    let block_hash = header.hash();
    let _fees: u64 = state.validate_block(rwtxn, header, body)?;
    if tracing::enabled!(tracing::Level::DEBUG) {
        let merkle_root = body.compute_merkle_root();
        let height = state.get_height(rwtxn)?;
        let () = state.connect_block(rwtxn, header, body)?;
        tracing::debug!(%height, %merkle_root, %block_hash,
                            "connected body")
    } else {
        let () = state.connect_block(rwtxn, header, body)?;
    }
    let () = state.connect_two_way_peg_data(rwtxn, &two_way_peg_data)?;
    let accumulator = state.get_accumulator(rwtxn)?;
    let () = archive.put_header(rwtxn, header)?;
    let () = archive.put_body(rwtxn, block_hash, body)?;
    let () = archive.put_accumulator(rwtxn, block_hash, &accumulator)?;
    for transaction in &body.transactions {
        let () = mempool.delete(rwtxn, transaction.txid())?;
    }
    let () = mempool.regenerate_proofs(rwtxn, &accumulator)?;
    Ok(())
}

async fn disconnect_tip_<Transport>(
    rwtxn: &mut RwTxn<'_>,
    archive: &Archive,
    cusf_mainchain: &mut mainchain::ValidatorClient<Transport>,
    mempool: &MemPool,
    state: &State,
) -> Result<(), Error>
where
    Transport: proto::Transport,
{
    let tip_block_hash = state.get_tip(rwtxn)?;
    let tip_header = archive.get_header(rwtxn, tip_block_hash)?;
    let tip_body = archive.get_body(rwtxn, tip_block_hash)?;
    let height = state.get_height(rwtxn)?;
    let two_way_peg_data = {
        let start_block_hash = state
            .deposit_blocks
            .rev_iter(rwtxn)?
            .transpose_into_fallible()
            .find_map(|(_, (block_hash, applied_height))| {
                if applied_height < height - 1 {
                    Ok(Some(block_hash))
                } else {
                    Ok(None)
                }
            })?;
        cusf_mainchain
            .get_two_way_peg_data(start_block_hash, tip_header.prev_main_hash)
            .await?
    };
    let () = state.disconnect_two_way_peg_data(rwtxn, &two_way_peg_data)?;
    let () = state.disconnect_tip(rwtxn, &tip_header, &tip_body)?;
    // TODO: revert accumulator only necessary because rustreexo does not
    // support undo yet
    {
        let new_tip = state.get_tip(rwtxn)?;
        let accumulator = archive.get_accumulator(rwtxn, new_tip)?;
        let () =
            state
                .utreexo_accumulator
                .put(rwtxn, &UnitKey, &accumulator)?;
    }
    for transaction in tip_body.authorized_transactions().iter().rev() {
        mempool.put(rwtxn, transaction)?;
    }
    let accumulator = state.get_accumulator(rwtxn)?;
    mempool.regenerate_proofs(rwtxn, &accumulator)?;
    Ok(())
}

/// Re-org to the specified tip, if it is better than the current tip.
/// The new tip block and all ancestor blocks must exist in the node's archive.
/// A result of `Ok(true)` indicates a successful re-org.
/// A result of `Ok(false)` indicates that no re-org was attempted.
async fn reorg_to_tip<Transport>(
    env: &heed::Env,
    archive: &Archive,
    cusf_mainchain: &mut mainchain::ValidatorClient<Transport>,
    mempool: &MemPool,
    state: &State,
    new_tip: Tip,
) -> Result<bool, Error>
where
    Transport: proto::Transport,
{
    let mut rwtxn = env.write_txn()?;
    let tip_hash = state.get_tip(&rwtxn)?;
    let tip_height = state.get_height(&rwtxn)?;
    if tip_hash != BlockHash::default() {
        let bmm_verification =
            archive.get_best_main_verification(&rwtxn, tip_hash)?;
        let tip = Tip {
            block_hash: tip_hash,
            main_block_hash: bmm_verification,
        };
        // check that new tip is better than current tip
        if archive.better_tip(&rwtxn, tip, new_tip)? != Some(new_tip) {
            return Ok(false);
        }
    }
    let common_ancestor =
        archive.last_common_ancestor(&rwtxn, tip_hash, new_tip.block_hash)?;
    // Check that all necessary bodies exist before disconnecting tip
    let blocks_to_apply: Vec<(Header, Body)> = archive
        .ancestors(&rwtxn, new_tip.block_hash)
        .take_while(|block_hash| Ok(*block_hash != common_ancestor))
        .map(|block_hash| {
            let header = archive.get_header(&rwtxn, block_hash)?;
            let body = archive.get_body(&rwtxn, block_hash)?;
            Ok((header, body))
        })
        .collect()?;
    // Disconnect tip until common ancestor is reached
    let common_ancestor_height = archive.get_height(&rwtxn, common_ancestor)?;
    for _ in 0..tip_height - common_ancestor_height {
        let () = disconnect_tip_(
            &mut rwtxn,
            archive,
            cusf_mainchain,
            mempool,
            state,
        )
        .await?;
    }
    let tip = state.get_tip(&rwtxn)?;
    assert_eq!(tip, common_ancestor);
    // Apply blocks until new tip is reached
    for (header, body) in blocks_to_apply.into_iter().rev() {
        let () = connect_tip_(
            &mut rwtxn,
            archive,
            cusf_mainchain,
            mempool,
            state,
            &header,
            &body,
        )
        .await?;
    }
    let tip = state.get_tip(&rwtxn)?;
    assert_eq!(tip, new_tip.block_hash);
    rwtxn.commit()?;
    tracing::info!("synced to tip: {}", new_tip.block_hash);
    Ok(true)
}

#[derive(Clone)]
struct NetTaskContext<MainchainTransport> {
    env: heed::Env,
    archive: Archive,
    cusf_mainchain: mainchain::ValidatorClient<MainchainTransport>,
    mainchain_task: MainchainTaskHandle,
    mempool: MemPool,
    net: Net,
    state: State,
}

/// Message indicating a tip that is ready to reorg to, with the address of the
/// peer connection that caused the request, if it originated from a peer.
/// If the request originates from this node, then the socket address is
/// None.
/// An optional oneshot sender can be used receive the result of attempting
/// to reorg to the new tip, on the corresponding oneshot receiver.
type NewTipReadyMessage =
    (Tip, Option<SocketAddr>, Option<oneshot::Sender<bool>>);

struct NetTask<MainchainTransport> {
    ctxt: NetTaskContext<MainchainTransport>,
    /// Receive a request to forward to the mainchain task, with the address of
    /// the peer connection that caused the request, and the peer state ID of
    /// the request
    forward_mainchain_task_request_rx:
        UnboundedReceiver<(mainchain_task::Request, SocketAddr, PeerStateId)>,
    /// Push a request to forward to the mainchain task, with the address of
    /// the peer connection that caused the request, and the peer state ID of
    /// the request
    forward_mainchain_task_request_tx:
        UnboundedSender<(mainchain_task::Request, SocketAddr, PeerStateId)>,
    mainchain_task_response_rx: UnboundedReceiver<mainchain_task::Response>,
    /// Receive a tip that is ready to reorg to, with the address of the peer
    /// connection that caused the request, if it originated from a peer.
    /// If the request originates from this node, then the socket address is
    /// None.
    /// An optional oneshot sender can be used receive the result of attempting
    /// to reorg to the new tip, on the corresponding oneshot receiver.
    new_tip_ready_rx: UnboundedReceiver<NewTipReadyMessage>,
    /// Push a tip that is ready to reorg to, with the address of the peer
    /// connection that caused the request, if it originated from a peer.
    /// If the request originates from this node, then the socket address is
    /// None.
    /// An optional oneshot sender can be used receive the result of attempting
    /// to reorg to the new tip, on the corresponding oneshot receiver.
    new_tip_ready_tx: UnboundedSender<NewTipReadyMessage>,
    peer_info_rx: PeerInfoRx,
}

impl<MainchainTransport> NetTask<MainchainTransport>
where
    MainchainTransport: proto::Transport,
{
    async fn handle_response(
        ctxt: &NetTaskContext<MainchainTransport>,
        // Attempt to switch to a descendant tip once a body has been
        // stored, if all other ancestor bodies are available.
        // Each descendant tip maps to the peers that sent that tip.
        descendant_tips: &mut HashMap<Tip, HashMap<Tip, HashSet<SocketAddr>>>,
        new_tip_ready_tx: &UnboundedSender<NewTipReadyMessage>,
        addr: SocketAddr,
        resp: PeerResponse,
        req: PeerRequest,
    ) -> Result<(), Error> {
        match (req, resp) {
            (
                req @ PeerRequest::GetBlock {
                    block_hash,
                    descendant_tip: Some(descendant_tip),
                    ancestor: Some(ancestor),
                    peer_state_id: Some(peer_state_id),
                },
                ref resp @ PeerResponse::Block {
                    ref header,
                    ref body,
                },
            ) => {
                if header.hash() != block_hash {
                    // Invalid response
                    tracing::warn!(%addr, ?req, ?resp,"Invalid response from peer; unexpected block hash");
                    let () = ctxt.net.remove_active_peer(addr);
                    return Ok(());
                }
                {
                    let mut rwtxn = ctxt.env.write_txn()?;
                    let () =
                        ctxt.archive.put_body(&mut rwtxn, block_hash, body)?;
                    rwtxn.commit()?;
                }
                // Notify the peer connection if all requested block bodies are
                // now available
                {
                    let rotxn = ctxt.env.read_txn()?;
                    let missing_bodies = ctxt
                        .archive
                        .get_missing_bodies(&rotxn, block_hash, ancestor)?;
                    if missing_bodies.is_empty() {
                        let message = PeerConnectionMessage::BodiesAvailable(
                            peer_state_id,
                        );
                        let () =
                            ctxt.net.push_internal_message(message, addr)?;
                    }
                }
                // Check if any new tips can be applied,
                // and send new tip ready if so
                {
                    let rotxn = ctxt.env.read_txn()?;
                    let tip_hash = ctxt.state.get_tip(&rotxn)?;
                    // Find the BMM verification that is an ancestor of
                    // `main_descendant_tip`
                    let main_block_hash = ctxt
                        .archive
                        .get_bmm_results(&rotxn, block_hash)?
                        .into_iter()
                        .map(Result::<_, Error>::Ok)
                        .transpose_into_fallible()
                        .find_map(|(main_block_hash, bmm_result)| {
                            match bmm_result {
                                BmmResult::Failed => Ok(None),
                                BmmResult::Verified => {
                                    if ctxt.archive.is_main_descendant(
                                        &rotxn,
                                        main_block_hash,
                                        descendant_tip.main_block_hash,
                                    )? {
                                        Ok(Some(main_block_hash))
                                    } else {
                                        Ok(None)
                                    }
                                }
                            }
                        })?
                        .unwrap();
                    let block_tip = Tip {
                        block_hash,
                        main_block_hash,
                    };
                    if header.prev_side_hash == tip_hash {
                        let () = new_tip_ready_tx
                            .unbounded_send((block_tip, Some(addr), None))
                            .map_err(|_| Error::SendNewTipReady)?;
                    }
                    let Some(descendant_tips) =
                        descendant_tips.remove(&block_tip)
                    else {
                        return Ok(());
                    };
                    for (descendant_tip, sources) in descendant_tips {
                        let common_ancestor =
                            ctxt.archive.last_common_ancestor(
                                &rotxn,
                                descendant_tip.block_hash,
                                tip_hash,
                            )?;
                        let missing_bodies = ctxt.archive.get_missing_bodies(
                            &rotxn,
                            descendant_tip.block_hash,
                            common_ancestor,
                        )?;
                        if missing_bodies.is_empty() {
                            for addr in sources {
                                let () = new_tip_ready_tx
                                    .unbounded_send((
                                        descendant_tip,
                                        Some(addr),
                                        None,
                                    ))
                                    .map_err(|_| Error::SendNewTipReady)?;
                            }
                        }
                    }
                }
                Ok(())
            }
            (
                PeerRequest::GetBlock {
                    block_hash: req_block_hash,
                    descendant_tip: Some(_),
                    ancestor: Some(_),
                    peer_state_id: Some(_),
                },
                PeerResponse::NoBlock {
                    block_hash: resp_block_hash,
                },
            ) if req_block_hash == resp_block_hash => Ok(()),
            (
                ref req @ PeerRequest::GetHeaders {
                    ref start,
                    end,
                    height: Some(height),
                    peer_state_id: Some(peer_state_id),
                },
                PeerResponse::Headers(headers),
            ) => {
                // check that the end header is as requested
                let Some(end_header) = headers.last() else {
                    tracing::warn!(%addr, ?req, "Invalid response from peer; missing end header");
                    let () = ctxt.net.remove_active_peer(addr);
                    return Ok(());
                };
                let end_header_hash = end_header.hash();
                if end_header_hash != end {
                    tracing::warn!(%addr, ?req, ?end_header,"Invalid response from peer; unexpected end header");
                    let () = ctxt.net.remove_active_peer(addr);
                    return Ok(());
                }
                // Must be at least one header due to previous check
                let start_hash = headers.first().unwrap().prev_side_hash;
                // check that the first header is after a start block
                if !(start.contains(&start_hash)
                    || start_hash == BlockHash::default())
                {
                    tracing::warn!(%addr, ?req, ?start_hash, "Invalid response from peer; invalid start hash");
                    let () = ctxt.net.remove_active_peer(addr);
                    return Ok(());
                }
                // check that the end header height is as expected
                {
                    let rotxn = ctxt.env.read_txn()?;
                    let start_height =
                        ctxt.archive.get_height(&rotxn, start_hash)?;
                    if start_height + headers.len() as u32 != height {
                        tracing::warn!(%addr, ?req, ?start_hash, "Invalid response from peer; invalid end height");
                        let () = ctxt.net.remove_active_peer(addr);
                        return Ok(());
                    }
                }
                // check that headers are sequential based on prev_side_hash
                let mut prev_side_hash = start_hash;
                for header in &headers {
                    if header.prev_side_hash != prev_side_hash {
                        tracing::warn!(%addr, ?req, ?headers,"Invalid response from peer; non-sequential headers");
                        let () = ctxt.net.remove_active_peer(addr);
                        return Ok(());
                    }
                    prev_side_hash = header.hash();
                }
                // Store new headers
                let mut rwtxn = ctxt.env.write_txn()?;
                for header in &headers {
                    let block_hash = header.hash();
                    if ctxt
                        .archive
                        .try_get_header(&rwtxn, block_hash)?
                        .is_none()
                    {
                        if header.prev_side_hash == BlockHash::default()
                            || ctxt
                                .archive
                                .try_get_header(&rwtxn, header.prev_side_hash)?
                                .is_some()
                        {
                            ctxt.archive.put_header(&mut rwtxn, header)?;
                        } else {
                            break;
                        }
                    }
                }
                rwtxn.commit()?;
                // Notify peer connection that headers are available
                let message = PeerConnectionMessage::Headers(peer_state_id);
                let () = ctxt.net.push_internal_message(message, addr)?;
                Ok(())
            }
            (
                PeerRequest::GetHeaders {
                    start: _,
                    end,
                    height: _,
                    peer_state_id: _,
                },
                PeerResponse::NoHeader { block_hash },
            ) if end == block_hash => Ok(()),
            (
                PeerRequest::PushTransaction { transaction: _ },
                PeerResponse::TransactionAccepted(_),
            ) => Ok(()),
            (
                PeerRequest::PushTransaction { transaction: _ },
                PeerResponse::TransactionRejected(_),
            ) => Ok(()),
            (
                req @ (PeerRequest::GetBlock { .. }
                | PeerRequest::GetHeaders { .. }
                | PeerRequest::Heartbeat(_)
                | PeerRequest::PushTransaction { .. }),
                resp,
            ) => {
                // Invalid response
                tracing::warn!(%addr, ?req, ?resp,"Invalid response from peer");
                let () = ctxt.net.remove_active_peer(addr);
                Ok(())
            }
        }
    }

    async fn run(mut self) -> Result<(), Error> {
        enum MailboxItem {
            AcceptConnection(Result<(), Error>),
            // Forward a mainchain task request, along with the peer that
            // caused the request, and the peer state ID of the request
            ForwardMainchainTaskRequest(
                mainchain_task::Request,
                SocketAddr,
                PeerStateId,
            ),
            MainchainTaskResponse(mainchain_task::Response),
            // Apply new tip from peer or self.
            // An optional oneshot sender can be used receive the result of
            // attempting to reorg to the new tip, on the corresponding oneshot
            // receiver.
            NewTipReady(Tip, Option<SocketAddr>, Option<oneshot::Sender<bool>>),
            PeerInfo(Option<(SocketAddr, Option<PeerConnectionInfo>)>),
        }
        let accept_connections = stream::try_unfold((), |()| {
            let env = self.ctxt.env.clone();
            let net = self.ctxt.net.clone();
            let fut = async move {
                let () = net.accept_incoming(env).await?;
                Result::<_, Error>::Ok(Some(((), ())))
            };
            Box::pin(fut)
        })
        .map(MailboxItem::AcceptConnection);
        let forward_request_stream = self
            .forward_mainchain_task_request_rx
            .map(|(request, addr, peer_state_id)| {
                MailboxItem::ForwardMainchainTaskRequest(
                    request,
                    addr,
                    peer_state_id,
                )
            });
        let mainchain_task_response_stream = self
            .mainchain_task_response_rx
            .map(MailboxItem::MainchainTaskResponse);
        let new_tip_ready_stream =
            self.new_tip_ready_rx.map(|(block_hash, addr, resp_tx)| {
                MailboxItem::NewTipReady(block_hash, addr, resp_tx)
            });
        let peer_info_stream = StreamNotifyClose::new(self.peer_info_rx)
            .map(MailboxItem::PeerInfo);
        let mut mailbox_stream = stream::select_all([
            accept_connections.boxed(),
            forward_request_stream.boxed(),
            mainchain_task_response_stream.boxed(),
            new_tip_ready_stream.boxed(),
            peer_info_stream.boxed(),
        ]);
        // Attempt to switch to a descendant tip once a body has been
        // stored, if all other ancestor bodies are available.
        // Each descendant tip maps to the peers that sent that tip.
        let mut descendant_tips =
            HashMap::<Tip, HashMap<Tip, HashSet<SocketAddr>>>::new();
        // Map associating mainchain task requests with the peer(s) that
        // caused the request, and the request peer state ID
        let mut mainchain_task_request_sources = HashMap::<
            mainchain_task::Request,
            HashSet<(SocketAddr, PeerStateId)>,
        >::new();
        while let Some(mailbox_item) = mailbox_stream.next().await {
            match mailbox_item {
                MailboxItem::AcceptConnection(res) => res?,
                MailboxItem::ForwardMainchainTaskRequest(
                    request,
                    peer,
                    peer_state_id,
                ) => {
                    mainchain_task_request_sources
                        .entry(request)
                        .or_default()
                        .insert((peer, peer_state_id));
                    let () = self
                        .ctxt
                        .mainchain_task
                        .request(request)
                        .map_err(|_| Error::SendMainchainTaskRequest)?;
                }
                MailboxItem::MainchainTaskResponse(response) => {
                    let request = (&response).into();
                    match response {
                        mainchain_task::Response::AncestorHeaders(
                            _block_hash,
                            res,
                        ) => {
                            let Some(sources) =
                                mainchain_task_request_sources.remove(&request)
                            else {
                                continue;
                            };
                            let res = res.map_err(Arc::new);
                            for (addr, peer_state_id) in sources {
                                let message = match res {
                                    Ok(()) => PeerConnectionMessage::MainchainAncestors(
                                        peer_state_id,
                                    ),
                                    Err(ref err) => PeerConnectionMessage::MainchainAncestorsError(
                                        anyhow::Error::from(err.clone())
                                    )
                                };
                                let () = self
                                    .ctxt
                                    .net
                                    .push_internal_message(message, addr)?;
                            }
                        }
                        mainchain_task::Response::VerifyBmm(
                            _block_hash,
                            res,
                        ) => {
                            let Some(sources) =
                                mainchain_task_request_sources.remove(&request)
                            else {
                                continue;
                            };
                            let res = res.map_err(Arc::new);
                            for (addr, peer_state_id) in sources {
                                let message = match res {
                                    Ok(bmm_verification_res) => PeerConnectionMessage::BmmVerification {
                                        res: bmm_verification_res,
                                        peer_state_id,
                                    },
                                    Err(ref err) => PeerConnectionMessage::BmmVerificationError(anyhow::Error::from(err.clone()))
                                };
                                let () = self
                                    .ctxt
                                    .net
                                    .push_internal_message(message, addr)?;
                            }
                        }
                    }
                }
                MailboxItem::NewTipReady(new_tip, _addr, resp_tx) => {
                    let reorg_applied = reorg_to_tip(
                        &self.ctxt.env,
                        &self.ctxt.archive,
                        &mut self.ctxt.cusf_mainchain,
                        &self.ctxt.mempool,
                        &self.ctxt.state,
                        new_tip,
                    )
                    .await?;
                    if let Some(resp_tx) = resp_tx {
                        let () = resp_tx
                            .send(reorg_applied)
                            .map_err(|_| Error::SendReorgResultOneshot)?;
                    }
                }
                MailboxItem::PeerInfo(None) => {
                    return Err(Error::PeerInfoRxClosed)
                }
                MailboxItem::PeerInfo(Some((addr, None))) => {
                    // peer connection is closed, remove it
                    tracing::warn!(%addr, "Connection to peer closed");
                    let () = self.ctxt.net.remove_active_peer(addr);
                    continue;
                }
                MailboxItem::PeerInfo(Some((addr, Some(peer_info)))) => {
                    match peer_info {
                        PeerConnectionInfo::Error(err) => {
                            let err = anyhow::anyhow!(err);
                            tracing::error!(%addr, err = format!("{err:#}"), "Peer connection error");
                            let () = self.ctxt.net.remove_active_peer(addr);
                        }
                        PeerConnectionInfo::NeedBmmVerification {
                            main_hash,
                            peer_state_id,
                        } => {
                            let request =
                                mainchain_task::Request::VerifyBmm(main_hash);
                            let () = self
                                .forward_mainchain_task_request_tx
                                .unbounded_send((request, addr, peer_state_id))
                                .map_err(|_| {
                                    Error::ForwardMainchainTaskRequest
                                })?;
                        }
                        PeerConnectionInfo::NeedMainchainAncestors {
                            main_hash,
                            peer_state_id,
                        } => {
                            let request =
                                mainchain_task::Request::AncestorHeaders(
                                    main_hash,
                                );
                            let () = self
                                .forward_mainchain_task_request_tx
                                .unbounded_send((request, addr, peer_state_id))
                                .map_err(|_| {
                                    Error::ForwardMainchainTaskRequest
                                })?;
                        }
                        PeerConnectionInfo::NewTipReady(new_tip) => {
                            self.new_tip_ready_tx
                                .unbounded_send((new_tip, Some(addr), None))
                                .map_err(|_| Error::SendNewTipReady)?;
                        }
                        PeerConnectionInfo::NewTransaction(mut new_tx) => {
                            let mut rwtxn = self.ctxt.env.write_txn()?;
                            let () = self.ctxt.state.regenerate_proof(
                                &rwtxn,
                                &mut new_tx.transaction,
                            )?;
                            self.ctxt.mempool.put(&mut rwtxn, &new_tx)?;
                            rwtxn.commit()?;
                            // broadcast
                            let () = self
                                .ctxt
                                .net
                                .push_tx(HashSet::from_iter([addr]), new_tx);
                        }
                        PeerConnectionInfo::Response(boxed) => {
                            let (resp, req) = *boxed;
                            let () = Self::handle_response(
                                &self.ctxt,
                                &mut descendant_tips,
                                &self.new_tip_ready_tx,
                                addr,
                                resp,
                                req,
                            )
                            .await?;
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Handle to the net task.
/// Task is aborted on drop.
#[derive(Clone)]
pub(super) struct NetTaskHandle {
    task: Arc<JoinHandle<()>>,
    /// Push a tip that is ready to reorg to, with the address of the peer
    /// connection that caused the request, if it originated from a peer.
    /// If the request originates from this node, then the socket address is
    /// None.
    /// An optional oneshot sender can be used receive the result of attempting
    /// to reorg to the new tip, on the corresponding oneshot receiver.
    new_tip_ready_tx: UnboundedSender<NewTipReadyMessage>,
}

impl NetTaskHandle {
    #[allow(clippy::too_many_arguments)]
    pub fn new<MainchainTransport>(
        local_pool: LocalPoolHandle,
        env: heed::Env,
        archive: Archive,
        cusf_mainchain: mainchain::ValidatorClient<MainchainTransport>,
        mainchain_task: MainchainTaskHandle,
        mainchain_task_response_rx: UnboundedReceiver<mainchain_task::Response>,
        mempool: MemPool,
        net: Net,
        peer_info_rx: PeerInfoRx,
        state: State,
    ) -> Self
    where
        MainchainTransport: proto::Transport + Send + 'static,
    {
        let ctxt = NetTaskContext {
            env,
            archive,
            cusf_mainchain,
            mainchain_task,
            mempool,
            net,
            state,
        };
        let (
            forward_mainchain_task_request_tx,
            forward_mainchain_task_request_rx,
        ) = mpsc::unbounded();
        let (new_tip_ready_tx, new_tip_ready_rx) = mpsc::unbounded();
        let task = NetTask {
            ctxt,
            forward_mainchain_task_request_tx,
            forward_mainchain_task_request_rx,
            mainchain_task_response_rx,
            new_tip_ready_tx: new_tip_ready_tx.clone(),
            new_tip_ready_rx,
            peer_info_rx,
        };
        let task = local_pool.spawn_pinned(|| async {
            if let Err(err) = task.run().await {
                let err = anyhow::Error::from(err);
                tracing::error!("Net task error: {err:#}");
            }
        });
        NetTaskHandle {
            task: Arc::new(task),
            new_tip_ready_tx,
        }
    }

    /// Push a tip that is ready to reorg to.
    #[allow(dead_code)]
    pub fn new_tip_ready(&self, new_tip: Tip) -> Result<(), Error> {
        self.new_tip_ready_tx
            .unbounded_send((new_tip, None, None))
            .map_err(|_| Error::SendNewTipReady)
    }

    /// Push a tip that is ready to reorg to, and await successful application.
    /// A result of Ok(true) indicates that the tip was applied and reorged
    /// to successfully.
    /// A result of Ok(false) indicates that the tip was not reorged to.
    pub async fn new_tip_ready_confirm(
        &self,
        new_tip: Tip,
    ) -> Result<bool, Error> {
        let (oneshot_tx, oneshot_rx) = oneshot::channel();
        let () = self
            .new_tip_ready_tx
            .unbounded_send((new_tip, None, Some(oneshot_tx)))
            .map_err(|_| Error::SendNewTipReady)?;
        oneshot_rx
            .await
            .map_err(|_| Error::ReceiveReorgResultOneshot)
    }
}

impl Drop for NetTaskHandle {
    // If only one reference exists (ie. within self), abort the net task.
    fn drop(&mut self) {
        // use `Arc::get_mut` since `Arc::into_inner` requires ownership of the
        // Arc, and cloning would increase the reference count
        if let Some(task) = Arc::get_mut(&mut self.task) {
            task.abort()
        }
    }
}
