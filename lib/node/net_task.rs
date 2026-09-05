//! Task to manage peers and their responses

use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    net::SocketAddr,
    sync::Arc,
    time::Duration,
};

use error_fatality::{Nested as _, Split};
use fallible_iterator::{FallibleIterator, IteratorExt};
use futures::{
    StreamExt,
    channel::{
        mpsc::{self, UnboundedReceiver, UnboundedSender},
        oneshot,
    },
    stream,
};
use nonempty::NonEmpty;
use sneed::{DbError, EnvError, RwTxn, RwTxnError};
use tokio::task::{self, JoinHandle};
use tokio_stream::StreamNotifyClose;

use crate::{
    archive::{self, Archive},
    mempool::MemPool,
    net::{
        self, Net, PeerConnectionInfo, PeerConnectionMessage, PeerInfoRx,
        PeerRequest, PeerResponse, PeerStateId, error::peer::Recoverable as _,
        peer_message,
    },
    node::{
        error::net_task::{self as error, Error},
        mainchain_task::{self, MainchainTaskHandle},
    },
    state::{self, State},
    types::{
        BmmResult, Body, Header, MerkleRoot, Tip,
        net::ResolvedPeerAddress,
        proto::mainchain::{self, Event as MainchainBlockEvent},
    },
    util::{ErrorChain, join_set},
};

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
    let prevalidated = state.prevalidate_block(rwtxn, header, body)?;
    if tracing::enabled!(tracing::Level::DEBUG) {
        let height = state.try_get_height(rwtxn)?;
        let merkle_root = state.connect_prevalidated_block(
            rwtxn,
            header,
            body,
            prevalidated,
        )?;
        tracing::debug!(?height, %merkle_root, %block_hash, "connected body")
    } else {
        let _: MerkleRoot = state.connect_prevalidated_block(
            rwtxn,
            header,
            body,
            prevalidated,
        )?;
    }
    let () = state.connect_two_way_peg_data(rwtxn, two_way_peg_data)?;
    let accumulator = state.get_accumulator(rwtxn)?;
    // TODO: are these needed?
    {
        let () = archive.put_header(rwtxn, header)?;
        let () = archive.put_body(rwtxn, block_hash, body)?;
    }
    let () = archive.put_accumulator(rwtxn, block_hash, &accumulator)?;
    for transaction in &body.transactions {
        let () = mempool.delete(rwtxn, transaction.txid())?;
    }
    let () = mempool.regenerate_proofs(rwtxn, &accumulator)?;
    Ok(())
}

pub(in crate::node) fn disconnect_tip_(
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
    let () = state.disconnect_tip(rwtxn, &tip_header, &tip_body)?;
    // TODO: revert accumulator only necessary because rustreexo does not
    // support undo yet
    {
        match state.try_get_tip(rwtxn)? {
            Some(new_tip) => {
                let accumulator = archive.get_accumulator(rwtxn, new_tip)?;
                let () = state
                    .utreexo_accumulator
                    .put(rwtxn, &(), &accumulator)
                    .map_err(DbError::from)?;
            }
            None => {
                state
                    .utreexo_accumulator
                    .delete(rwtxn, &())
                    .map_err(DbError::from)?;
            }
        };
    }
    for transaction in tip_body.authorized_transactions().iter().rev() {
        mempool.put(rwtxn, transaction)?;
    }
    let accumulator = state.get_accumulator(rwtxn)?;
    mempool.regenerate_proofs(rwtxn, &accumulator)?;
    Ok(())
}

fn is_fatal_reorg_error(err: &Error) -> bool {
    !matches!(err, Error::State(_))
}

/// Re-org to the specified tip, if it is better than the current tip.
/// The new tip block and all ancestor blocks must exist in the node's archive.
/// A result of `Ok(true)` indicates a successful re-org.
/// A result of `Ok(false)` indicates that no re-org was attempted.
// a state error means a peer sent an invalid block; it must not be fatal
fn reorg_to_tip<ThreadLocalStorage>(
    env: &sneed::Env<ThreadLocalStorage>,
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
        let () = match connect_tip_(
            &mut rwtxn,
            archive,
            mempool,
            state,
            &header,
            &body,
            &two_way_peg_data,
        ) {
            Ok(()) => (),
            Err(err) => {
                if !is_fatal_reorg_error(&err) {
                    // The stored body for this block failed validation (e.g. a peer
                    // supplied a body whose contents do not match the header's merkle
                    // root). Abort the reorg and discard the invalid body from the
                    // archive so that the block is reported missing again and the real
                    // body is re-requested, instead of the archive staying poisoned.
                    drop(rwtxn);
                    let mut rwtxn = env.write_txn()?;
                    let () = archive.delete_body(
                        &mut rwtxn,
                        header.hash(),
                        &body,
                    )?;
                    rwtxn.commit()?;
                }
                return Err(err);
            }
        };
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
    env: sneed::Env<heed::WithoutTls>,
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
    mainchain_task_event_rx: UnboundedReceiver<mainchain_task::Event>,
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
    fn handle_response(
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
                    tracing::warn!(
                        %addr,
                        ?req,
                        ?resp,
                        "Invalid response from peer; unexpected block hash"
                    );
                    let () = ctxt.net.remove_active_peer(addr);
                    return Ok::<_, Error>(());
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
                    let ancestor_height = if let Some(ancestor) = ancestor {
                        Some(ctxt.archive.get_height(&rotxn, ancestor)?)
                    } else {
                        None
                    };
                    let earliest_missing_body = ctxt
                        .archive
                        .iter_missing_bodies(
                            &rotxn,
                            block_hash,
                            ancestor_height.map_or(0, |height| height + 1),
                        )
                        .next()?;
                    if let Some(earliest_missing_body) = earliest_missing_body {
                        descendant_tips
                            .entry(earliest_missing_body)
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
                            .map_err(|err| {
                                Error::SendNewTipReady(err.into_send_error())
                            })?;
                    }
                    let Some(block_descendant_tips) =
                        descendant_tips.remove(&block_hash)
                    else {
                        return Ok(());
                    };
                    for (descendant_tip, sources) in block_descendant_tips {
                        let common_ancestor_height = if let Some(tip) = tip
                            && let Some(common_ancestor) =
                                ctxt.archive.last_common_ancestor(
                                    &rotxn,
                                    descendant_tip.block_hash,
                                    tip.block_hash,
                                )? {
                            Some(
                                ctxt.archive
                                    .get_height(&rotxn, common_ancestor)?,
                            )
                        } else {
                            None
                        };
                        let earliest_missing_body = ctxt
                            .archive
                            .iter_missing_bodies(
                                &rotxn,
                                descendant_tip.block_hash,
                                common_ancestor_height
                                    .map_or(0, |height| height + 1),
                            )
                            .next()?;
                        // If a better tip is ready, send a notification
                        'better_tip: {
                            let next_tip = if let Some(earliest_missing_body) =
                                earliest_missing_body
                            {
                                descendant_tips
                                    .entry(earliest_missing_body)
                                    .or_default()
                                    .entry(descendant_tip)
                                    .or_default()
                                    .extend(sources.iter().cloned());

                                // Parent of the earlist missing body
                                ctxt.archive
                                    .get_header(&rotxn, earliest_missing_body)?
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
                                        .map_err(|err| {
                                            Error::SendNewTipReady(
                                                err.into_send_error(),
                                            )
                                        })?;
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
                // check that headers are sequential based on prev_side_hash,
                // and no header builds on an invalidated block.
                {
                    let rotxn = ctxt.env.read_txn().map_err(EnvError::from)?;
                    let mut prev_side_hash = start_hash;
                    for header in &headers {
                        if header.prev_side_hash != prev_side_hash {
                            tracing::warn!(%addr, ?req, ?headers,"Invalid response from peer; non-sequential headers");
                            let () = ctxt.net.remove_active_peer(addr);
                            return Ok(());
                        }
                        if ctxt
                            .archive
                            .invalidated_block(&rotxn, &header.hash())?
                        {
                            tracing::warn!(%addr, ?req, ?headers,"Invalid response from peer; invalidated block header");
                            let () = ctxt.net.remove_active_peer(addr);
                            return Ok(());
                        }
                        prev_side_hash = Some(header.hash());
                    }
                }
                // Store new headers
                let () = tokio::task::block_in_place(|| {
                    let mut rwtxn =
                        ctxt.env.write_txn().map_err(EnvError::from)?;
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
                    Ok::<_, Error>(())
                })?;
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

    fn handle_mainchain_block_event(
        ctxt: &NetTaskContext,
        _event: MainchainBlockEvent,
    ) -> Result<(), Error> {
        let mut rwtxn = ctxt.env.write_txn().map_err(EnvError::from)?;
        while let Some(state_tip) = ctxt.state.try_get_tip(&rwtxn)?
            && !ctxt
                .archive
                .side_tips()
                .sidechain_tips()
                .contains_key(&rwtxn, &state_tip)
                .map_err(archive::Error::from)?
        {
            let header = ctxt.archive.get_header(&rwtxn, state_tip)?;
            let body = ctxt.archive.get_body(&rwtxn, state_tip)?;
            let () = ctxt.state.disconnect_tip(&mut rwtxn, &header, &body)?;
        }
        let best_side_tip = ctxt
            .archive
            .side_tips()
            .best_side_tip(&rwtxn)
            .map_err(archive::Error::from)?;
        rwtxn.commit()?;
        if let Some(best_side_tip) = best_side_tip {
            let best_side_tip = Tip {
                block_hash: best_side_tip.block_hash,
                main_block_hash: best_side_tip.info.main_block_hash,
            };
            let _: bool = reorg_to_tip(
                &ctxt.env,
                &ctxt.archive,
                &ctxt.mempool,
                &ctxt.state,
                best_side_tip,
            )?;
        }
        Ok(())
    }

    fn handle_mainchain_task_response(
        ctxt: &NetTaskContext,
        mainchain_task_request_sources: &mut HashMap<
            mainchain_task::Request,
            HashSet<(SocketAddr, PeerStateId)>,
        >,
        response: mainchain_task::Response,
    ) -> Result<(), Error> {
        let request = (&response).into();
        match response {
            mainchain_task::Response::AncestorInfos(block_hash, res) => {
                let Some(sources) =
                    mainchain_task_request_sources.remove(&request)
                else {
                    return Ok(());
                };
                let res = res.map_err(Arc::new);
                for (addr, peer_state_id) in sources {
                    let message = match res {
                        Ok(true) => PeerConnectionMessage::MainchainAncestors(
                            peer_state_id,
                        ),
                        Ok(false) => {
                            PeerConnectionMessage::MainchainAncestorsError(
                                error::MainchainAncestors::BlockNotAvailable {
                                    block_hash,
                                },
                            )
                        }
                        Err(ref err) => {
                            PeerConnectionMessage::MainchainAncestorsError(
                                err.clone().into(),
                            )
                        }
                    };
                    let _: bool = ctxt.net.push_internal_message(message, addr);
                }
                Ok(())
            }
        }
    }

    #[inline]
    fn handle_mainchain_task_event(
        ctxt: &NetTaskContext,
        mainchain_task_request_sources: &mut HashMap<
            mainchain_task::Request,
            HashSet<(SocketAddr, PeerStateId)>,
        >,
        event: mainchain_task::Event,
    ) -> Result<(), Error> {
        match event {
            mainchain_task::Event::Block(event) => {
                Self::handle_mainchain_block_event(ctxt, event)
            }
            mainchain_task::Event::Response(resp) => {
                Self::handle_mainchain_task_response(
                    ctxt,
                    mainchain_task_request_sources,
                    resp,
                )
            }
        }
    }

    async fn run(self) -> Result<(), Error> {
        tracing::debug!("starting net task");
        #[derive(Debug)]
        enum MailboxItem {
            AcceptConnection(
                Result<
                    Option<SocketAddr>,
                    <net::error::AcceptConnection as Split>::Fatal,
                >,
            ),
            // Forward a mainchain task request, along with the peer that
            // caused the request, and the peer state ID of the request
            ForwardMainchainTaskRequest(
                mainchain_task::Request,
                SocketAddr,
                PeerStateId,
            ),
            MainchainTaskEvent(mainchain_task::Event),
            // Apply new tip from peer or self.
            // An optional oneshot sender can be used receive the result of
            // attempting to reorg to the new tip, on the corresponding oneshot
            // receiver.
            NewTipReady(Tip, Option<SocketAddr>, Option<oneshot::Sender<bool>>),
            PeerInfo(Option<(SocketAddr, Option<PeerConnectionInfo>)>),
            // Signal to reconnect to a peer
            ReconnectPeer(ResolvedPeerAddress),
        }
        let accept_connections = stream::try_unfold((), |()| {
            let env = self.ctxt.env.clone();
            let net = self.ctxt.net.clone();
            let fut = async move {
                let maybe_socket_addr =
                    net.accept_incoming(env).await.into_nested()?;

                // / Return:
                // - The value to yield (maybe_socket_addr)
                // - The state for the next iteration (())
                // Wrapped in Result and Option
                Result::<_, _>::Ok(Some((maybe_socket_addr, ())))
            };
            Box::pin(fut)
        })
        .filter_map(async |item| match item {
            Ok(Ok(maybe_socket_addr)) => Some(Ok(maybe_socket_addr)),
            Ok(Err(non_fatal_err)) => {
                // type the error explicitly
                let non_fatal_err:
                    <net::error::AcceptConnection as Split>::Jfyi =
                    non_fatal_err;
                tracing::error!(
                    "Failed to accept connection: {:#}",
                    ErrorChain::new(&non_fatal_err)
                );
                None
            }
            Err(fatal_err) => Some(Err(fatal_err)),
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
        let mainchain_task_event_stream = self
            .mainchain_task_event_rx
            .map(MailboxItem::MainchainTaskEvent);
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
            mainchain_task_event_stream.boxed(),
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
                    Err(fatal_err) => {
                        // explicitly type error
                        let fatal_err: <net::error::AcceptConnection as Split>::Fatal =
                            fatal_err;
                        tracing::error!(
                            "failed to accept connection: {:#}",
                            ErrorChain::new(&fatal_err)
                        );
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
                MailboxItem::MainchainTaskEvent(event) => {
                    let () = Self::handle_mainchain_task_event(
                        &self.ctxt,
                        &mut mainchain_task_request_sources,
                        event,
                    )?;
                }
                MailboxItem::NewTipReady(new_tip, addr, resp_tx) => {
                    let reorg_result = task::block_in_place(|| {
                        {
                            let rotxn = self
                                .ctxt
                                .env
                                .read_txn()
                                .map_err(|err| Error::DbEnv(err.into()))?;
                            if !self
                                .ctxt
                                .archive
                                .side_tips()
                                .sidechain_tips()
                                .contains_key(&rotxn, &new_tip.block_hash)
                                .map_err(archive::Error::from)?
                            {
                                return Ok(false);
                            }
                            let side_tips_tip = self
                                .ctxt
                                .archive
                                .side_tips()
                                .get_mainchain_tip(&rotxn)
                                .map_err(archive::Error::from)?;
                            if !self.ctxt.archive.is_main_descendant(
                                &rotxn,
                                new_tip.main_block_hash,
                                side_tips_tip.block_hash(),
                            )? {
                                return Ok(false);
                            }
                        }
                        reorg_to_tip(
                            &self.ctxt.env,
                            &self.ctxt.archive,
                            &self.ctxt.mempool,
                            &self.ctxt.state,
                            new_tip,
                        )
                    });
                    let reorg_applied = match reorg_result {
                        Ok(applied) => applied,
                        Err(err) if is_fatal_reorg_error(&err) => {
                            return Err(err);
                        }
                        // an invalid block must not kill the net task; drop the
                        // peer and keep running
                        Err(err) => {
                            tracing::warn!(
                                ?new_tip,
                                ?addr,
                                err = format!("{:#}", ErrorChain::new(&err)),
                                "rejecting invalid tip from peer"
                            );
                            if let Some(addr) = addr {
                                let () = self.ctxt.net.remove_active_peer(addr);
                            }
                            false
                        }
                    };
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
                }
                MailboxItem::PeerInfo(Some((addr, Some(peer_info)))) => {
                    tracing::trace!(%addr, ?peer_info, "mailbox item: received PeerInfo");
                    match peer_info {
                        PeerConnectionInfo::Error {
                            err,
                            resolved_peer_addr,
                        } => {
                            const RECONNECT_DELAY: Duration =
                                Duration::from_secs(10);
                            let err_msg =
                                format!("{:#}", ErrorChain::new(&err));
                            tracing::error!(
                                %addr,
                                err = err_msg,
                                "Peer connection error",
                            );
                            // Attempt to reconnect if a valid message was
                            // received successfully
                            let received_msg_successfully =
                                self.ctxt.net.try_with_active_peer_connection(
                                    addr,
                                    |conn_handle| {
                                        conn_handle.received_msg_successfully()
                                    },
                                );
                            let () = self.ctxt.net.remove_active_peer(addr);
                            // A peer on another network never becomes useful,
                            // so it must not survive into the next start.
                            if err.is_bad_magic() {
                                let peer_address = resolved_peer_addr
                                    .as_peer_address()
                                    .to_owned();
                                let mut rwtxn = self
                                    .ctxt
                                    .env
                                    .write_txn()
                                    .map_err(EnvError::from)?;
                                let forgotten = self
                                    .ctxt
                                    .net
                                    .forget_peer(&mut rwtxn, &peer_address)?;
                                rwtxn.commit().map_err(RwTxnError::from)?;
                                if forgotten {
                                    tracing::warn!(
                                        %peer_address,
                                        "forgot peer: it runs another network"
                                    );
                                }
                                continue;
                            }
                            let Some(received_msg_successfully) =
                                received_msg_successfully
                            else {
                                continue;
                            };
                            if received_msg_successfully && err.may_reconnect()
                            {
                                reconnect_peer_spawner.spawn(async move {
                                    tokio::time::sleep(RECONNECT_DELAY).await;
                                    resolved_peer_addr
                                });
                            } else if let (_, Some(resolved_peer_addr)) =
                                resolved_peer_addr.pop_first_ip_addr()
                            {
                                reconnect_peer_spawner.spawn(async move {
                                    tokio::time::sleep(RECONNECT_DELAY).await;
                                    resolved_peer_addr
                                });
                            }
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
                                .map_err(|err| {
                                    Error::SendNewTipReady(
                                        err.into_send_error(),
                                    )
                                })?;
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
                                .push_tx(HashSet::from_iter([addr]), &new_tx);
                        }
                        PeerConnectionInfo::Response(boxed) => {
                            let (resp, req) = *boxed;
                            tracing::trace!(
                                resp = format!("{resp:#?}"),
                                req = format!("{req:#?}"),
                                "mail box: received PeerConnectionInfo::Response"
                            );
                            let () = tokio::task::block_in_place(|| {
                                Self::handle_response(
                                    &self.ctxt,
                                    &mut descendant_tips,
                                    &self.new_tip_ready_tx,
                                    addr,
                                    resp,
                                    req,
                                )
                            })?;
                        }
                    }
                }
                MailboxItem::ReconnectPeer(resolved_peer_address) => {
                    let peer_address =
                        resolved_peer_address.as_peer_address().to_owned();
                    match self.ctxt.net.connect_peer(
                        self.ctxt.env.clone(),
                        resolved_peer_address,
                    ) {
                        Ok(()) => (),
                        Err(err) => {
                            tracing::error!(
                                %peer_address,
                                "Failed to connect to peer: {:#}",
                                ErrorChain::new(&err)
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
        env: sneed::Env<heed::WithoutTls>,
        archive: Archive,
        mainchain_task: MainchainTaskHandle,
        mainchain_task_event_rx: UnboundedReceiver<mainchain_task::Event>,
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
            mainchain_task_event_rx,
            new_tip_ready_tx: new_tip_ready_tx.clone(),
            new_tip_ready_rx,
            peer_info_rx,
        };
        let task = runtime.spawn(async {
            if let Err(err) = task.run().await {
                tracing::error!("Net task error: {:#}", ErrorChain::new(&err));
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
            .map_err(|err| Error::SendNewTipReady(err.into_send_error()))?;
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

#[cfg(test)]
mod test {
    use crate::{
        node::net_task::{Error, is_fatal_reorg_error},
        state,
    };

    // a peer's invalid block (value out > value in) must not be fatal
    #[test]
    fn invalid_peer_block_is_not_fatal() {
        let err = Error::State(state::Error::NotEnoughValueIn);
        assert!(!is_fatal_reorg_error(&err));
    }

    // local infrastructure errors stay fatal
    #[test]
    fn infrastructure_error_is_fatal() {
        assert!(is_fatal_reorg_error(&Error::PeerInfoRxClosed));
    }
}
