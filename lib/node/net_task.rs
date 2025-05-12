//! Task to manage peers and their responses

use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    net::SocketAddr,
    sync::Arc,
    time::Duration,
};

use fallible_iterator::{FallibleIterator, IteratorExt};
use futures::{
    StreamExt,
    channel::{
        mpsc::{self, TrySendError, UnboundedReceiver, UnboundedSender},
        oneshot,
    },
    stream,
};
use nonempty::NonEmpty;
use sneed::{DbError, EnvError, RwTxn, RwTxnError, db};
use thiserror::Error;
use tokio::task::{self, JoinHandle};
use tokio_stream::StreamNotifyClose;

use super::mainchain_task::{self, MainchainTaskHandle};
use crate::{
    archive::{self, Archive},
    mempool::{self, MemPool},
    net::{
        self, Net, PeerConnectionError, PeerConnectionInfo,
        PeerConnectionMailboxError, PeerConnectionMessage, PeerInfoRx,
        PeerRequest, PeerResponse, PeerStateId, peer_message,
    },
    state::{self, State},
    types::{
        BmmResult, Body, Header, Tip,
        proto::{self, mainchain},
    },
    util::join_set,
};

#[allow(clippy::duplicated_attributes)]
#[derive(transitive::Transitive, Debug, Error)]
#[transitive(from(db::error::IterInit, DbError))]
#[transitive(from(db::error::IterItem, DbError))]
pub enum Error {
    #[error("archive error")]
    Archive(#[from] archive::Error),
    #[error("CUSF mainchain proto error")]
    CusfMainchain(#[from] proto::Error),
    #[error(transparent)]
    Db(#[from] DbError),
    #[error("Database env error")]
    DbEnv(#[from] EnvError),
    #[error("Database write error")]
    DbWrite(#[from] RwTxnError),
    #[error("Forward mainchain task request failed")]
    ForwardMainchainTaskRequest,
    #[error("mempool error")]
    MemPool(#[from] mempool::Error),
    #[error("Net error")]
    Net(#[from] net::Error),
    #[error("peer info stream closed")]
    PeerInfoRxClosed,
    #[error("Receive mainchain task response cancelled")]
    ReceiveMainchainTaskResponse,
    #[error("Receive reorg result cancelled (oneshot)")]
    ReceiveReorgResultOneshot(#[source] oneshot::Canceled),
    #[error("Send mainchain task request failed")]
    SendMainchainTaskRequest,
    #[error("Send new tip ready failed")]
    SendNewTipReady(#[source] TrySendError<NewTipReadyMessage>),
    #[error("Send reorg result error (oneshot)")]
    SendReorgResultOneshot,
    #[error("state error")]
    State(#[from] Box<state::Error>),
}

impl From<state::Error> for Error {
    fn from(err: state::Error) -> Self {
        Self::State(Box::new(err))
    }
}

fn connect_tip_(
    rwtxn: &mut RwTxn<'_>,
    archive: &Archive,
    mempool: &MemPool,
    state: &State,
    header: &Header,
    body: &Body,
    two_way_peg_data: &mainchain::TwoWayPegData,
) -> Result<(), Error> {
    let block_hash = header.hash();
    let _fees: bitcoin::Amount = state.validate_block(rwtxn, header, body)?;
    let orchard_frontier = if tracing::enabled!(tracing::Level::DEBUG) {
        let merkle_root = body.compute_merkle_root();
        let height = state.try_get_height(rwtxn)?;
        let orchard_frontier = state.connect_block(rwtxn, header, body)?;
        tracing::debug!(?height, %merkle_root, %block_hash,
                            "connected body");
        orchard_frontier
    } else {
        state.connect_block(rwtxn, header, body)?
    };
    if let Some(orchard_frontier) = orchard_frontier {
        let () = archive.put_orchard_frontier(
            rwtxn,
            block_hash,
            &orchard_frontier,
        )?;
    }
    let accumulator =
        state.connect_two_way_peg_data(rwtxn, two_way_peg_data)?;
    let () = archive.put_header(rwtxn, header)?;
    let () = archive.put_body(rwtxn, block_hash, body)?;
    let () = archive.put_accumulator(rwtxn, block_hash, &accumulator)?;
    for transaction in &body.transactions {
        let () = mempool.delete(rwtxn, transaction.txid())?;
    }
    let () = mempool.regenerate_proofs(rwtxn, &accumulator)?;
    Ok(())
}

fn disconnect_tip_(
    rwtxn: &mut RwTxn<'_>,
    archive: &Archive,
    mempool: &MemPool,
    state: &State,
) -> Result<(), Error> {
    let tip_block_hash =
        state.try_get_tip(rwtxn)?.ok_or(state::Error::NoTip)?;
    let tip_header = archive.get_header(rwtxn, tip_block_hash)?;
    let tip_body = archive.get_body(rwtxn, tip_block_hash)?;
    let height = state.try_get_height(rwtxn)?.ok_or(state::Error::NoTip)?;
    let two_way_peg_data = {
        let last_applied_deposit_block = state
            .deposit_blocks
            .rev_iter(rwtxn)
            .map_err(DbError::from)?
            .find_map(|(_, (block_hash, applied_height))| {
                if applied_height < height - 1 {
                    Ok(Some((block_hash, applied_height)))
                } else {
                    Ok(None)
                }
            })
            .map_err(DbError::from)?;
        let last_applied_withdrawal_bundle_event_block = state
            .withdrawal_bundle_event_blocks
            .rev_iter(rwtxn)
            .map_err(DbError::from)?
            .find_map(|(_, (block_hash, applied_height))| {
                if applied_height < height - 1 {
                    Ok(Some((block_hash, applied_height)))
                } else {
                    Ok(None)
                }
            })
            .map_err(DbError::from)?;
        let start_block_hash = match (
            last_applied_deposit_block,
            last_applied_withdrawal_bundle_event_block,
        ) {
            (None, None) => None,
            (Some((block_hash, _)), None) | (None, Some((block_hash, _))) => {
                Some(block_hash)
            }
            (
                Some((deposit_block, deposit_block_applied_height)),
                Some((
                    withdrawal_event_block,
                    withdrawal_event_block_applied_height,
                )),
            ) => {
                match deposit_block_applied_height
                    .cmp(&withdrawal_event_block_applied_height)
                {
                    Ordering::Less => Some(withdrawal_event_block),
                    Ordering::Greater => Some(deposit_block),
                    Ordering::Equal => {
                        if archive.is_main_descendant(
                            rwtxn,
                            withdrawal_event_block,
                            deposit_block,
                        )? {
                            Some(withdrawal_event_block)
                        } else {
                            assert!(archive.is_main_descendant(
                                rwtxn,
                                deposit_block,
                                withdrawal_event_block
                            )?);
                            Some(deposit_block)
                        }
                    }
                }
            }
        };
        let block_infos: Vec<_> = archive
            .main_ancestors(rwtxn, tip_header.prev_main_hash)
            .take_while(|ancestor| {
                Ok(Some(ancestor) != start_block_hash.as_ref())
            })
            .filter_map(|ancestor| {
                let block_info =
                    archive.get_main_block_info(rwtxn, &ancestor)?;
                if block_info.events.is_empty() {
                    Ok(None)
                } else {
                    Ok(Some((ancestor, block_info)))
                }
            })
            .collect()?;
        mainchain::TwoWayPegData {
            block_info: block_infos.into_iter().rev().collect(),
        }
    };
    let () = state.disconnect_two_way_peg_data(rwtxn, &two_way_peg_data)?;
    let prev_accumulator =
        archive.get_accumulator(rwtxn, &tip_header.prev_side_hash)?;
    let prev_frontier = archive
        .orchard_frontiers()
        .try_get(rwtxn, &tip_header.prev_side_hash)
        .map_err(archive::Error::from)?;
    let () = state.disconnect_tip(
        rwtxn,
        &tip_header,
        &tip_body,
        &prev_accumulator,
        prev_frontier.as_ref(),
    )?;
    for transaction in tip_body.authorized_transactions().iter().rev() {
        mempool.put(rwtxn, transaction)?;
    }
    mempool.regenerate_proofs(rwtxn, &prev_accumulator)?;
    Ok(())
}

/// Re-org to the specified tip, if it is better than the current tip.
/// The new tip block and all ancestor blocks must exist in the node's archive.
/// A result of `Ok(true)` indicates a successful re-org.
/// A result of `Ok(false)` indicates that no re-org was attempted.
fn reorg_to_tip(
    env: &sneed::Env,
    archive: &Archive,
    mempool: &MemPool,
    state: &State,
    new_tip: Tip,
) -> Result<bool, Error> {
    let mut rwtxn = env.write_txn().map_err(EnvError::from)?;
    let tip_height = state.try_get_height(&rwtxn)?;
    let tip = state
        .try_get_tip(&rwtxn)?
        .map(|tip_hash| {
            let bmm_verification =
                archive.get_best_main_verification(&rwtxn, tip_hash)?;
            Ok::<_, Error>(Tip {
                block_hash: tip_hash,
                main_block_hash: bmm_verification,
            })
        })
        .transpose()?;
    if let Some(tip) = tip {
        // check that new tip is better than current tip
        if archive.better_tip(&rwtxn, tip, new_tip)? != Some(new_tip) {
            tracing::debug!(
                ?tip,
                ?new_tip,
                "New tip is not better than current tip"
            );
            return Ok(false);
        }
    }
    let common_ancestor = if let Some(tip) = tip {
        archive.last_common_ancestor(
            &rwtxn,
            tip.block_hash,
            new_tip.block_hash,
        )?
    } else {
        None
    };
    // Check that all necessary bodies exist before disconnecting tip
    let blocks_to_apply: NonEmpty<(Header, Body)> = {
        let header = archive.get_header(&rwtxn, new_tip.block_hash)?;
        let body = archive.get_body(&rwtxn, new_tip.block_hash)?;
        let ancestors = if let Some(prev_side_hash) = header.prev_side_hash {
            archive
                .ancestors(&rwtxn, prev_side_hash)
                .take_while(|block_hash| {
                    Ok(common_ancestor.is_none_or(|common_ancestor| {
                        *block_hash != common_ancestor
                    }))
                })
                .map(|block_hash| {
                    let header = archive.get_header(&rwtxn, block_hash)?;
                    let body = archive.get_body(&rwtxn, block_hash)?;
                    Ok((header, body))
                })
                .collect()?
        } else {
            Vec::new()
        };
        NonEmpty {
            head: (header, body),
            tail: ancestors,
        }
    };
    // Disconnect tip until common ancestor is reached
    if let Some(tip_height) = tip_height {
        let common_ancestor_height =
            if let Some(common_ancestor) = common_ancestor {
                Some(archive.get_height(&rwtxn, common_ancestor)?)
            } else {
                None
            };
        tracing::debug!(
            ?tip,
            ?tip_height,
            ?common_ancestor,
            ?common_ancestor_height,
            "Disconnecting tip until common ancestor is reached"
        );
        let disconnects =
            if let Some(common_ancestor_height) = common_ancestor_height {
                tip_height - common_ancestor_height
            } else {
                tip_height + 1
            };
        for _ in 0..disconnects {
            let () = disconnect_tip_(&mut rwtxn, archive, mempool, state)?;
        }
    }
    {
        let tip_hash = state.try_get_tip(&rwtxn)?;
        assert_eq!(tip_hash, common_ancestor);
    }
    let mut two_way_peg_data_batch: Vec<_> = {
        let common_ancestor_header =
            if let Some(common_ancestor) = common_ancestor {
                Some(archive.get_header(&rwtxn, common_ancestor)?)
            } else {
                None
            };
        let common_ancestor_prev_main_hash =
            common_ancestor_header.map(|header| header.prev_main_hash);
        archive
            .main_ancestors(&rwtxn, blocks_to_apply.head.0.prev_main_hash)
            .take_while(|ancestor| {
                Ok(Some(ancestor) != common_ancestor_prev_main_hash.as_ref())
            })
            .map(|ancestor| {
                let block_info =
                    archive.get_main_block_info(&rwtxn, &ancestor)?;
                Ok((ancestor, block_info))
            })
            .collect()?
    };
    // Apply blocks until new tip is reached
    for (header, body) in blocks_to_apply.into_iter().rev() {
        let two_way_peg_data = {
            let mut two_way_peg_data = mainchain::TwoWayPegData::default();
            'fill_2wpd: while let Some((block_hash, block_info)) =
                two_way_peg_data_batch.pop()
            {
                two_way_peg_data.block_info.replace(block_hash, block_info);
                if block_hash == header.prev_main_hash {
                    break 'fill_2wpd;
                }
            }
            two_way_peg_data
        };
        let () = connect_tip_(
            &mut rwtxn,
            archive,
            mempool,
            state,
            &header,
            &body,
            &two_way_peg_data,
        )?;
        let new_tip_hash = state.try_get_tip(&rwtxn)?.unwrap();
        let bmm_verification =
            archive.get_best_main_verification(&rwtxn, new_tip_hash)?;
        let new_tip = Tip {
            block_hash: new_tip_hash,
            main_block_hash: bmm_verification,
        };
        if let Some(tip) = tip
            && archive.better_tip(&rwtxn, tip, new_tip)? != Some(new_tip)
        {
            continue;
        }
        rwtxn.commit().map_err(RwTxnError::from)?;
        tracing::info!("synced to tip: {}", new_tip.block_hash);
        rwtxn = env.write_txn().map_err(EnvError::from)?;
    }
    let tip = state.try_get_tip(&rwtxn)?;
    assert_eq!(tip, Some(new_tip.block_hash));
    rwtxn.commit().map_err(RwTxnError::from)?;
    tracing::info!("synced to tip: {}", new_tip.block_hash);
    Ok(true)
}

#[derive(Clone)]
struct NetTaskContext {
    env: sneed::Env,
    archive: Archive,
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

struct NetTask {
    ctxt: NetTaskContext,
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

impl NetTask {
    async fn handle_response(
        ctxt: &NetTaskContext,
        // Attempt to switch to a descendant tip once a body has been
        // stored, if all other ancestor bodies are available.
        // Each descendant tip maps to the peers that sent that tip.
        descendant_tips: &mut HashMap<
            crate::types::BlockHash,
            HashMap<Tip, HashSet<SocketAddr>>,
        >,
        new_tip_ready_tx: &UnboundedSender<NewTipReadyMessage>,
        addr: SocketAddr,
        resp: PeerResponse,
        req: PeerRequest,
    ) -> Result<(), Error> {
        tracing::debug!(?req, ?resp, "starting response handler");
        match (req, resp) {
            (
                PeerRequest::GetBlock(
                    req @ peer_message::GetBlockRequest {
                        block_hash,
                        descendant_tip: Some(descendant_tip),
                        ancestor,
                        peer_state_id: Some(peer_state_id),
                    },
                ),
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
                    let mut rwtxn =
                        ctxt.env.write_txn().map_err(EnvError::from)?;
                    let () =
                        ctxt.archive.put_body(&mut rwtxn, block_hash, body)?;
                    rwtxn.commit().map_err(RwTxnError::from)?;
                }
                // Notify the peer connection if all requested block bodies are
                // now available
                {
                    let rotxn = ctxt.env.read_txn().map_err(EnvError::from)?;
                    let missing_bodies = ctxt
                        .archive
                        .get_missing_bodies(&rotxn, block_hash, ancestor)?;
                    if let Some(earliest_missing_body) = missing_bodies.first()
                    {
                        descendant_tips
                            .entry(*earliest_missing_body)
                            .or_default()
                            .entry(descendant_tip)
                            .or_default()
                            .insert(addr);
                    } else {
                        let message = PeerConnectionMessage::BodiesAvailable(
                            peer_state_id,
                        );
                        let _: bool =
                            ctxt.net.push_internal_message(message, addr);
                    }
                }
                // Check if any new tips can be applied,
                // and send new tip ready if so
                {
                    let rotxn = ctxt.env.read_txn().map_err(EnvError::from)?;
                    let tip = ctxt
                        .state
                        .try_get_tip(&rotxn)?
                        .map(|tip_hash| {
                            let bmm_verification = ctxt
                                .archive
                                .get_best_main_verification(&rotxn, tip_hash)?;
                            Ok::<_, Error>(Tip {
                                block_hash: tip_hash,
                                main_block_hash: bmm_verification,
                            })
                        })
                        .transpose()?;
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

                    if header.prev_side_hash == tip.map(|tip| tip.block_hash) {
                        tracing::trace!(
                            ?block_tip,
                            %addr,
                            "sending new tip ready, originating from peer"
                        );

                        let () = new_tip_ready_tx
                            .unbounded_send((block_tip, Some(addr), None))
                            .map_err(Error::SendNewTipReady)?;
                    }
                    let Some(block_descendant_tips) =
                        descendant_tips.remove(&block_hash)
                    else {
                        return Ok(());
                    };
                    for (descendant_tip, sources) in block_descendant_tips {
                        let common_ancestor = if let Some(tip) = tip {
                            ctxt.archive.last_common_ancestor(
                                &rotxn,
                                descendant_tip.block_hash,
                                tip.block_hash,
                            )?
                        } else {
                            None
                        };
                        let missing_bodies = ctxt.archive.get_missing_bodies(
                            &rotxn,
                            descendant_tip.block_hash,
                            common_ancestor,
                        )?;
                        // If a better tip is ready, send a notification
                        'better_tip: {
                            let next_tip = if let Some(earliest_missing_body) =
                                missing_bodies.first()
                            {
                                descendant_tips
                                    .entry(*earliest_missing_body)
                                    .or_default()
                                    .entry(descendant_tip)
                                    .or_default()
                                    .extend(sources.iter().cloned());

                                // Parent of the earlist missing body
                                ctxt.archive
                                    .get_header(&rotxn, *earliest_missing_body)?
                                    .prev_side_hash
                                    .map(|tip_hash| {
                                        let bmm_verification = ctxt
                                            .archive
                                            .get_best_main_verification(
                                                &rotxn, tip_hash,
                                            )?;
                                        Ok::<_, Error>(Tip {
                                            block_hash: tip_hash,
                                            main_block_hash: bmm_verification,
                                        })
                                    })
                                    .transpose()?
                            } else {
                                Some(descendant_tip)
                            };
                            let Some(next_tip) = next_tip else {
                                break 'better_tip;
                            };
                            if let Some(tip) = tip
                                && ctxt
                                    .archive
                                    .better_tip(&rotxn, tip, next_tip)?
                                    != Some(next_tip)
                            {
                                break 'better_tip;
                            } else {
                                tracing::debug!(
                                    new_tip = ?next_tip,
                                    "sending new tip ready to sources"
                                );
                                for addr in sources {
                                    tracing::trace!(%addr, new_tip = ?next_tip, "sending new tip ready");
                                    let () = new_tip_ready_tx
                                        .unbounded_send((
                                            next_tip,
                                            Some(addr),
                                            None,
                                        ))
                                        .map_err(Error::SendNewTipReady)?;
                                }
                            }
                        }
                    }
                }
                Ok(())
            }
            (
                PeerRequest::GetBlock(peer_message::GetBlockRequest {
                    block_hash: req_block_hash,
                    descendant_tip: Some(_),
                    ancestor: _,
                    peer_state_id: Some(_),
                }),
                PeerResponse::NoBlock {
                    block_hash: resp_block_hash,
                },
            ) if req_block_hash == resp_block_hash => Ok(()),
            (
                PeerRequest::GetHeaders(
                    ref req @ peer_message::GetHeadersRequest {
                        ref start,
                        end,
                        height: Some(height),
                        peer_state_id: Some(peer_state_id),
                    },
                ),
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
                if let Some(start_hash) = start_hash
                    && !start.contains(&start_hash)
                {
                    tracing::warn!(%addr, ?req, %start_hash, "Invalid response from peer; invalid start hash");
                    let () = ctxt.net.remove_active_peer(addr);
                    return Ok(());
                }
                // check that the end header height is as expected
                {
                    let rotxn = ctxt.env.read_txn().map_err(EnvError::from)?;
                    let start_height = if let Some(start_hash) = start_hash {
                        Some(ctxt.archive.get_height(&rotxn, start_hash)?)
                    } else {
                        None
                    };
                    let end_height = match start_height {
                        Some(start_height) => {
                            start_height + headers.len() as u32
                        }
                        None => headers.len() as u32 - 1,
                    };
                    if end_height != height {
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
                    prev_side_hash = Some(header.hash());
                }
                // Store new headers
                let mut rwtxn = ctxt.env.write_txn().map_err(EnvError::from)?;
                for header in &headers {
                    let block_hash = header.hash();
                    if ctxt
                        .archive
                        .try_get_header(&rwtxn, block_hash)?
                        .is_none()
                    {
                        if let Some(parent) = header.prev_side_hash
                            && ctxt
                                .archive
                                .try_get_header(&rwtxn, parent)?
                                .is_none()
                        {
                            break;
                        } else {
                            ctxt.archive.put_header(&mut rwtxn, header)?;
                        }
                    }
                }
                rwtxn.commit().map_err(RwTxnError::from)?;
                // Notify peer connection that headers are available
                let message = PeerConnectionMessage::Headers(peer_state_id);
                let _: bool = ctxt.net.push_internal_message(message, addr);
                Ok(())
            }
            (
                PeerRequest::GetHeaders(peer_message::GetHeadersRequest {
                    start: _,
                    end,
                    height: _,
                    peer_state_id: _,
                }),
                PeerResponse::NoHeader { block_hash },
            ) if end == block_hash => Ok(()),
            (
                PeerRequest::PushTransaction(
                    peer_message::PushTransactionRequest { transaction: _ },
                ),
                PeerResponse::TransactionAccepted(_),
            ) => Ok(()),
            (
                PeerRequest::PushTransaction(
                    peer_message::PushTransactionRequest { transaction: _ },
                ),
                PeerResponse::TransactionRejected(_),
            ) => Ok(()),
            (
                req @ (PeerRequest::GetBlock { .. }
                | PeerRequest::GetHeaders { .. }
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

    async fn run(self) -> Result<(), Error> {
        tracing::debug!("starting net task");
        #[derive(Debug)]
        enum MailboxItem {
            AcceptConnection(Result<Option<SocketAddr>, Error>),
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
            // Signal to reconnect to a peer
            ReconnectPeer(SocketAddr),
        }
        let accept_connections = stream::try_unfold((), |()| {
            let env = self.ctxt.env.clone();
            let net = self.ctxt.net.clone();
            let fut = async move {
                let maybe_socket_addr = net.accept_incoming(env).await?;

                // / Return:
                // - The value to yield (maybe_socket_addr)
                // - The state for the next iteration (())
                // Wrapped in Result and Option
                Result::<Option<(Option<SocketAddr>, ())>, Error>::Ok(Some((
                    maybe_socket_addr,
                    (),
                )))
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
        let (reconnect_peer_spawner, reconnect_peer_rx) = join_set::new();
        let reconnect_peer_stream = reconnect_peer_rx
            .map(|addr| MailboxItem::ReconnectPeer(addr.unwrap()));
        let mut mailbox_stream = stream::select_all([
            accept_connections.boxed(),
            forward_request_stream.boxed(),
            mainchain_task_response_stream.boxed(),
            new_tip_ready_stream.boxed(),
            peer_info_stream.boxed(),
            reconnect_peer_stream.boxed(),
        ]);
        // Attempt to switch to a descendant tip once a body has been
        // stored, if all other ancestor bodies are available.
        // Each descendant tip maps to the peers that sent that tip.
        let mut descendant_tips = HashMap::<
            crate::types::BlockHash,
            HashMap<Tip, HashSet<SocketAddr>>,
        >::new();
        // Map associating mainchain task requests with the peer(s) that
        // caused the request, and the request peer state ID
        let mut mainchain_task_request_sources = HashMap::<
            mainchain_task::Request,
            HashSet<(SocketAddr, PeerStateId)>,
        >::new();
        while let Some(mailbox_item) = mailbox_stream.next().await {
            tracing::trace!(?mailbox_item, "received new mailbox item");
            match mailbox_item {
                MailboxItem::AcceptConnection(res) => match res {
                    // We received a connection new incoming network connection, but no peer
                    // was added
                    Ok(None) => {
                        continue;
                    }
                    Ok(Some(addr)) => {
                        tracing::trace!(%addr, "accepted new incoming connection");
                    }
                    Err(err) => {
                        tracing::error!(%err, "failed to accept connection");
                    }
                },
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
                        mainchain_task::Response::AncestorInfos(
                            block_hash,
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
                                    Ok(true) => PeerConnectionMessage::MainchainAncestors(
                                        peer_state_id,
                                    ),
                                    Ok(false) => PeerConnectionMessage::MainchainAncestorsError(
                                        anyhow::anyhow!("Requested block was not available: {block_hash}")
                                    ),
                                    Err(ref err) => PeerConnectionMessage::MainchainAncestorsError(
                                        anyhow::Error::from(err.clone())
                                    )
                                };
                                let _: bool = self
                                    .ctxt
                                    .net
                                    .push_internal_message(message, addr);
                            }
                        }
                    }
                }
                MailboxItem::NewTipReady(new_tip, _addr, resp_tx) => {
                    let reorg_applied = task::block_in_place(|| {
                        reorg_to_tip(
                            &self.ctxt.env,
                            &self.ctxt.archive,
                            &self.ctxt.mempool,
                            &self.ctxt.state,
                            new_tip,
                        )
                    })?;
                    if let Some(resp_tx) = resp_tx {
                        let () = resp_tx
                            .send(reorg_applied)
                            .map_err(|_| Error::SendReorgResultOneshot)?;
                    }
                }
                MailboxItem::PeerInfo(None) => {
                    return Err(Error::PeerInfoRxClosed);
                }
                MailboxItem::PeerInfo(Some((addr, None))) => {
                    // peer connection is closed, remove it
                    tracing::warn!(%addr, "Connection to peer closed");
                    let () = self.ctxt.net.remove_active_peer(addr);
                    continue;
                }
                MailboxItem::PeerInfo(Some((addr, Some(peer_info)))) => {
                    tracing::trace!(%addr, ?peer_info, "mailbox item: received PeerInfo");
                    match peer_info {
                        PeerConnectionInfo::Error(
                            PeerConnectionError::Mailbox(
                                PeerConnectionMailboxError::HeartbeatTimeout,
                            ),
                        ) => {
                            const RECONNECT_DELAY: Duration =
                                Duration::from_secs(10);
                            // Attempt to reconnect if a valid message was
                            // received successfully
                            let Some(received_msg_successfully) =
                                self.ctxt.net.try_with_active_peer_connection(
                                    addr,
                                    |conn_handle| {
                                        conn_handle.received_msg_successfully()
                                    },
                                )
                            else {
                                continue;
                            };
                            let () = self.ctxt.net.remove_active_peer(addr);
                            if !received_msg_successfully {
                                continue;
                            }
                            reconnect_peer_spawner.spawn(async move {
                                tokio::time::sleep(RECONNECT_DELAY).await;
                                addr
                            });
                        }
                        PeerConnectionInfo::Error(err) => {
                            let err = anyhow::anyhow!(err);
                            tracing::error!(%addr, err = format!("{err:#}"), "Peer connection error");
                            let () = self.ctxt.net.remove_active_peer(addr);
                        }
                        PeerConnectionInfo::NeedMainchainAncestors {
                            main_hash,
                            peer_state_id,
                        } => {
                            let request =
                                mainchain_task::Request::AncestorInfos(
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
                            tracing::debug!(
                                ?new_tip,
                                %addr,
                                "mailbox item: received NewTipReady from peer, sending on channel"
                            );
                            self.new_tip_ready_tx
                                .unbounded_send((new_tip, Some(addr), None))
                                .map_err(Error::SendNewTipReady)?;
                        }
                        PeerConnectionInfo::NewTransaction(mut new_tx) => {
                            let mut rwtxn = self
                                .ctxt
                                .env
                                .write_txn()
                                .map_err(EnvError::from)?;
                            let () = self.ctxt.state.regenerate_proof(
                                &rwtxn,
                                &mut new_tx.transaction,
                            )?;
                            self.ctxt.mempool.put(&mut rwtxn, &new_tx)?;
                            rwtxn.commit().map_err(RwTxnError::from)?;
                            // broadcast
                            let () = self
                                .ctxt
                                .net
                                .push_tx(HashSet::from_iter([addr]), *new_tx);
                        }
                        PeerConnectionInfo::Response(boxed) => {
                            let (resp, req) = *boxed;
                            tracing::trace!(
                                ?resp,
                                ?req,
                                "mail box: received PeerConnectionInfo::Response"
                            );
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
                MailboxItem::ReconnectPeer(peer_address) => {
                    match self
                        .ctxt
                        .net
                        .connect_peer(self.ctxt.env.clone(), peer_address)
                    {
                        Ok(()) => (),
                        Err(err) => {
                            let err = anyhow::Error::from(err);
                            tracing::error!(
                                %peer_address,
                                "Failed to connect to peer: {err:#}"
                            )
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
    pub fn new(
        runtime: &tokio::runtime::Runtime,
        env: sneed::Env,
        archive: Archive,
        mainchain_task: MainchainTaskHandle,
        mainchain_task_response_rx: UnboundedReceiver<mainchain_task::Response>,
        mempool: MemPool,
        net: Net,
        peer_info_rx: PeerInfoRx,
        state: State,
    ) -> Self {
        let ctxt = NetTaskContext {
            env,
            archive,
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
        let task = runtime.spawn(async {
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

    /// Push a tip that is ready to reorg to, and await successful application.
    /// A result of Ok(true) indicates that the tip was applied and reorged
    /// to successfully.
    /// A result of Ok(false) indicates that the tip was not reorged to.
    pub async fn new_tip_ready_confirm(
        &self,
        new_tip: Tip,
    ) -> Result<bool, Error> {
        tracing::debug!(?new_tip, "sending new tip ready confirm");

        let (oneshot_tx, oneshot_rx) = oneshot::channel();
        let () = self
            .new_tip_ready_tx
            .unbounded_send((new_tip, None, Some(oneshot_tx)))
            .map_err(Error::SendNewTipReady)?;
        oneshot_rx.await.map_err(Error::ReceiveReorgResultOneshot)
    }
}

impl Drop for NetTaskHandle {
    // If only one reference exists (ie. within self), abort the net task.
    fn drop(&mut self) {
        // use `Arc::get_mut` since `Arc::into_inner` requires ownership of the
        // Arc, and cloning would increase the reference count
        if let Some(task) = Arc::get_mut(&mut self.task) {
            tracing::debug!("dropping net task handle, aborting task");
            task.abort()
        }
    }
}
