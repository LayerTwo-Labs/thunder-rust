//! Request queue that handles rate limiting and deduplication

use std::{collections::HashSet, num::NonZeroU32, sync::Arc};

use futures::{Stream, StreamExt, channel::mpsc, stream};
use governor::{DefaultDirectRateLimiter, Quota};

use crate::{
    net::peer::{Connection, Request},
    types::{Hash, hash},
};

const REQUEST_QUOTA: Quota =
    Quota::per_second(NonZeroU32::new(50_000).unwrap());

pub struct Receiver {
    /// Requests that cost zero quota can skip the regular line
    fast_lane_rx: mpsc::UnboundedReceiver<Request>,
    inner_rx: mpsc::UnboundedReceiver<Request>,
    rate_limiter: Arc<DefaultDirectRateLimiter>,
}

impl Receiver {
    pub fn into_stream(self) -> impl Stream<Item = Request> {
        let inner_stream = self.inner_rx.then(move |request| {
            let rate_limiter = self.rate_limiter.clone();
            async move {
                if let Some(request_cost) =
                    NonZeroU32::new(Connection::request_cost(&request))
                {
                    let () =
                        rate_limiter.until_n_ready(request_cost).await.unwrap();
                }
                request
            }
        });
        // TODO: Prioritize fast lane
        stream::select(self.fast_lane_rx, inner_stream)
    }
}

#[derive(thiserror::Error, Debug)]
#[error("Failed to add request to queue")]
pub struct SendError;

pub struct Sender {
    /// Requests that cost zero quota can skip the regular line
    fast_lane_tx: mpsc::UnboundedSender<Request>,
    inner_tx: mpsc::UnboundedSender<Request>,
    /// Used to deduplicate requests
    request_hashes: HashSet<Hash>,
}

impl Sender {
    /// Returns `Ok(true)` if the request was sent. Requests may be ignored if they
    /// are duplicates of messages that have already been sent.
    /// Returns `Ok(false)` if the request was ignored.
    pub fn send(&mut self, request: Request) -> Result<bool, SendError> {
        let request_hash = hash(&request);
        if self.request_hashes.insert(request_hash) {
            if Connection::request_cost(&request) == 0 {
                let () = self
                    .fast_lane_tx
                    .unbounded_send(request)
                    .map_err(|_| SendError)?;
            } else {
                let () = self
                    .inner_tx
                    .unbounded_send(request)
                    .map_err(|_| SendError)?;
            };
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

pub fn new() -> (Sender, Receiver) {
    let (fast_lane_tx, fast_lane_rx) = mpsc::unbounded();
    let (inner_tx, inner_rx) = mpsc::unbounded();
    let send = Sender {
        fast_lane_tx,
        inner_tx,
        request_hashes: HashSet::new(),
    };
    let rate_limiter = DefaultDirectRateLimiter::direct(REQUEST_QUOTA);
    let recv = Receiver {
        fast_lane_rx,
        inner_rx,
        rate_limiter: Arc::new(rate_limiter),
    };
    (send, recv)
}
