//! A channel pool to manage multiple simultaneous channels to a peer.
//!
//! In order to send an outbound message, one should first acquire a permit
//! via [`Limiter::acquire`].
//! Send errors, task errors, and responses are received via a channel.

use std::{marker::PhantomData, num::NonZeroUsize, sync::Arc};

use async_lock::{Semaphore, SemaphoreGuardArc};
use futures::{FutureExt as _, Stream, StreamExt as _, future::Either, stream};
use tokio::task::AbortHandle;

use crate::{
    net::peer::{
        Connection, PeerResponseItem, error,
        message::{Heartbeat, Request},
    },
    util::join_set,
};

/// Type tags for channel limiters
pub trait LimiterTag {
    const MAX_CHANNELS: NonZeroUsize;
}

/// Guard for a channel limiter.
/// MUST be dropped when the channel is closed.
#[repr(transparent)]
pub struct LimiterGuard<T> {
    inner: SemaphoreGuardArc,
    _tag: PhantomData<T>,
}

/// Channel limiter
#[derive(educe::Educe)]
#[educe(Clone(bound()))]
#[repr(transparent)]
pub struct Limiter<T> {
    inner: Arc<Semaphore>,
    _tag: PhantomData<T>,
}

impl<T> Limiter<T>
where
    T: LimiterTag,
{
    pub async fn acquire(&self) -> LimiterGuard<T> {
        self.inner
            .acquire_arc()
            .map(|guard| LimiterGuard {
                inner: guard,
                _tag: PhantomData,
            })
            .await
    }
}

impl<T> Default for Limiter<T>
where
    T: LimiterTag,
{
    fn default() -> Self {
        Self {
            inner: Arc::new(Semaphore::new(T::MAX_CHANNELS.get())),
            _tag: PhantomData,
        }
    }
}

impl LimiterTag for Heartbeat {
    const MAX_CHANNELS: NonZeroUsize = NonZeroUsize::new(1).unwrap();
}

/// Limits the number of outbound channels for heartbeats
type OutboundHeartbeatLimiter = Limiter<Heartbeat>;

impl LimiterTag for Request {
    const MAX_CHANNELS: NonZeroUsize = NonZeroUsize::new(10).unwrap();
}

/// Limits the number of outbound channels for requests
type OutboundRequestLimiter = Limiter<Request>;

type SendHeartbeatResult = Result<(), error::connection::SendHeartbeat>;

type SendRequestResult =
    Result<PeerResponseItem, error::connection::SendRequest>;

type ReceiverItem = Either<error::channel_pool::SendMessage, PeerResponseItem>;

/// Receiver for responses and errors when sending messages.
/// If dropped, sending new messages will fail, and existing tasks to send
/// messages will be aborted.
#[must_use]
pub struct Receiver {
    send_heartbeat_task_rx: join_set::Receiver<SendHeartbeatResult>,
    send_request_task_rx: join_set::Receiver<SendRequestResult>,
}

impl Receiver {
    pub fn into_stream(self) -> impl Stream<Item = ReceiverItem> {
        let heartbeat_stream =
            self.send_heartbeat_task_rx
                .filter_map(async |res| match res {
                    Ok(Ok(())) => None,
                    Ok(Err(err)) => Some(Either::Left(err.into())),
                    Err(err) => Some(Either::Left(
                        error::channel_pool::Task::Heartbeat(err).into(),
                    )),
                });
        let request_stream = self.send_request_task_rx.map(|res| match res {
            Ok(Ok(peer_response)) => Either::Right(peer_response),
            Ok(Err(err)) => Either::Left(err.into()),
            Err(err) => {
                Either::Left(error::channel_pool::Task::Request(err).into())
            }
        });
        stream::select(heartbeat_stream, request_stream)
    }
}

pub struct ChannelPool {
    connection: Connection,
    outbound_heartbeat_limiter: OutboundHeartbeatLimiter,
    outbound_request_limiter: OutboundRequestLimiter,
    send_heartbeat_task_spawner: join_set::Spawner<SendHeartbeatResult>,
    send_request_task_spawner: join_set::Spawner<SendRequestResult>,
}

impl ChannelPool {
    pub fn new(connection: Connection) -> (Self, Receiver) {
        let (send_heartbeat_task_spawner, send_heartbeat_task_rx) =
            join_set::new();
        let (send_request_task_spawner, send_request_task_rx) = join_set::new();
        let this = Self {
            connection,
            outbound_heartbeat_limiter: OutboundHeartbeatLimiter::default(),
            outbound_request_limiter: OutboundRequestLimiter::default(),
            send_heartbeat_task_spawner,
            send_request_task_spawner,
        };
        let receiver = Receiver {
            send_heartbeat_task_rx,
            send_request_task_rx,
        };
        (this, receiver)
    }

    pub fn outbound_heartbeat_limiter(&self) -> &Limiter<Heartbeat> {
        &self.outbound_heartbeat_limiter
    }

    /// Acquire a guard to send an outbound request message
    pub fn outbound_request_limiter(&self) -> &Limiter<Request> {
        &self.outbound_request_limiter
    }

    pub fn send_heartbeat(
        &self,
        heartbeat: Heartbeat,
        guard: LimiterGuard<Heartbeat>,
    ) -> Result<(), error::channel_pool::SpawnHeartbeatTask> {
        let conn = self.connection.clone();
        self.send_heartbeat_task_spawner
            .spawn(async move {
                conn.send_heartbeat(&heartbeat)
                    .inspect(|_| drop(guard))
                    .await
            })
            .map(|_: AbortHandle| ())
            .ok_or(error::channel_pool::SpawnHeartbeatTask)
    }

    pub fn send_request(
        &self,
        request: Request,
        guard: LimiterGuard<Request>,
    ) -> Result<(), error::channel_pool::SpawnRequestTask> {
        let conn = self.connection.clone();
        self.send_request_task_spawner
            .spawn(async move {
                let response = conn.send_request(&request).await;
                drop(guard);
                response.map(|response| PeerResponseItem { request, response })
            })
            .map(|_: AbortHandle| ())
            .ok_or(error::channel_pool::SpawnRequestTask)
    }
}
