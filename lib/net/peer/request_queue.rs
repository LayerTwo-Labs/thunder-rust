//! Request queue that handles rate limiting and deduplication

use std::{collections::HashSet, num::NonZeroU32, sync::Arc};

use futures::{Stream, StreamExt, channel::mpsc, stream};
use governor::{DefaultDirectRateLimiter, Quota};
use parking_lot::Mutex;

use crate::{
    net::peer::{
        Connection, PeerResponseItem,
        channel_pool::{self, ChannelPool},
        error,
        message::{Heartbeat, Request},
    },
    types::{Hash, hash},
};

const REQUEST_QUOTA: Quota =
    Quota::per_second(NonZeroU32::new(50_000).unwrap());

/// The rate limiter token cost for a request
const fn request_cost(req: &Request) -> NonZeroU32 {
    match req {
        Request::GetBlock { .. } => NonZeroU32::new(1000).unwrap(),
        Request::GetHeaders { .. } => NonZeroU32::new(10_000).unwrap(),
        Request::PushTransaction { .. } => NonZeroU32::new(10).unwrap(),
    }
}

/// Receiver for errors when sending messages.
/// If dropped, sending new messages will fail, and existing tasks to send
/// messages and receive responses will be aborted.
#[must_use]
pub struct ErrorRx {
    heartbeat_rx: mpsc::UnboundedReceiver<Heartbeat>,
    request_rx: mpsc::UnboundedReceiver<Request>,
    rate_limiter: Arc<DefaultDirectRateLimiter>,
}

impl ErrorRx {
    pub fn into_stream(
        self,
        connection: Connection,
        peer_response_tx: mpsc::UnboundedSender<PeerResponseItem>,
    ) -> impl Stream<Item = error::request_queue::Error> {
        // Items in the combined source stream
        enum SourceItem {
            Error(error::channel_pool::SendMessage),
            Heartbeat(Heartbeat, channel_pool::LimiterGuard<Heartbeat>),
            PeerResponse(PeerResponseItem),
            Request(Request, channel_pool::LimiterGuard<Request>),
        }
        let (channel_pool, channel_pool_rx) = ChannelPool::new(connection);
        let channel_pool_stream = channel_pool_rx
            .into_stream()
            .map(|item| match item {
                futures::future::Either::Left(error) => {
                    SourceItem::Error(error)
                }
                futures::future::Either::Right(peer_response) => {
                    SourceItem::PeerResponse(peer_response)
                }
            })
            .boxed();
        let heartbeat_stream = self
            .heartbeat_rx
            .then({
                let limiter: channel_pool::Limiter<_> =
                    channel_pool.outbound_heartbeat_limiter().clone();
                move |heartbeat| {
                    let limiter = limiter.clone();
                    async move {
                        let guard = limiter.acquire().await;
                        SourceItem::Heartbeat(heartbeat, guard)
                    }
                }
            })
            .boxed();
        let request_stream = self
            .request_rx
            .then({
                let limiter = channel_pool.outbound_request_limiter().clone();
                move |request| {
                    let limiter = limiter.clone();
                    let rate_limiter = self.rate_limiter.clone();
                    async move {
                        let guard = limiter.acquire().await;
                        rate_limiter
                            .until_n_ready(request_cost(&request))
                            .await
                            .unwrap();
                        SourceItem::Request(request, guard)
                    }
                }
            })
            .boxed();
        // TODO: Prioritize heartbeats
        stream::select_all([
            channel_pool_stream,
            heartbeat_stream,
            request_stream,
        ])
        .filter_map(move |source_item| {
            std::future::ready(match source_item {
                SourceItem::Error(err) => Some(err.into()),
                SourceItem::Heartbeat(heartbeat, guard) => {
                    match channel_pool.send_heartbeat(heartbeat, guard) {
                        Ok(()) => None,
                        Err(err) => Some(err.into()),
                    }
                }
                SourceItem::Request(request, guard) => {
                    match channel_pool.send_request(request, guard) {
                        Ok(()) => None,
                        Err(err) => Some(err.into()),
                    }
                }
                SourceItem::PeerResponse(peer_response) => {
                    match peer_response_tx.unbounded_send(peer_response) {
                        Ok(()) => None,
                        Err(_err) => {
                            Some(error::request_queue::Error::PushPeerResponse)
                        }
                    }
                }
            })
        })
    }
}

#[derive(Clone)]
pub struct Sender {
    heartbeat_tx: mpsc::UnboundedSender<Heartbeat>,
    request_tx: mpsc::UnboundedSender<Request>,
    /// Used to deduplicate requests
    request_hashes: Arc<Mutex<HashSet<Hash>>>,
}

impl Sender {
    pub fn send_heartbeat(
        &self,
        heartbeat: Heartbeat,
    ) -> Result<(), error::request_queue::SendHeartbeat> {
        self.heartbeat_tx
            .unbounded_send(heartbeat)
            .map_err(|_| error::request_queue::SendHeartbeat)
    }

    /// Returns `Ok(true)` if the request was sent. Requests may be ignored if they
    /// are duplicates of messages that have already been sent.
    /// Returns `Ok(false)` if the request was ignored.
    pub fn send_request(
        &self,
        request: Request,
    ) -> Result<bool, error::request_queue::SendRequest> {
        let request_hash = hash(&request);
        if self.request_hashes.lock().insert(request_hash) {
            let () = self
                .request_tx
                .unbounded_send(request)
                .map_err(|_| error::request_queue::SendRequest)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

pub fn new() -> (Sender, ErrorRx) {
    let (heartbeat_tx, heartbeat_rx) = mpsc::unbounded();
    let (request_tx, request_rx) = mpsc::unbounded();
    let sender = Sender {
        heartbeat_tx,
        request_tx,
        request_hashes: Arc::new(Mutex::new(HashSet::new())),
    };
    let rate_limiter = DefaultDirectRateLimiter::direct(REQUEST_QUOTA);
    let error_rx = ErrorRx {
        heartbeat_rx,
        request_rx,
        rate_limiter: Arc::new(rate_limiter),
    };
    (sender, error_rx)
}
