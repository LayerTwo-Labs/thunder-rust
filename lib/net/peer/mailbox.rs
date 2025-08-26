//! Mailbox for a peer connection task

use std::{
    sync::{
        Arc,
        atomic::{self, AtomicBool},
    },
    task::Poll,
};

use futures::{
    Stream, StreamExt as _, TryFutureExt as _,
    channel::mpsc,
    stream::{self, BoxStream, Fuse, SelectAll},
};
use quinn::SendStream;
use tokio::time::{interval, timeout};
use tokio_stream::wrappers::IntervalStream;

use crate::{
    net::peer::{
        Connection, PeerResponseItem, PeerStateId,
        error::{self, mailbox::Error},
        message::{Request, RequestMessage},
        request_queue,
    },
    types::proto::mainchain,
    util::join_set,
};

/// Message received from the connection task / net task / node
#[derive(Debug)]
pub enum InternalMessage {
    /// Indicates if a BMM verification request completed.
    /// Does not indicate that BMM was verified successfully.
    BmmVerification {
        res: Result<(), mainchain::BlockNotFoundError>,
        peer_state_id: PeerStateId,
    },
    /// Indicates an error attempting BMM verification
    BmmVerificationError(anyhow::Error),
    /// Forward a request
    ForwardRequest(Request),
    /// Indicates that mainchain ancestors are now available
    MainchainAncestors(PeerStateId),
    /// Indicates an error fetching mainchain ancestors
    MainchainAncestorsError(anyhow::Error),
    /// Indicates that the requested headers are now available
    Headers(PeerStateId),
    /// Indicates that all requested missing block bodies are now available
    BodiesAvailable(PeerStateId),
}

impl From<Request> for InternalMessage {
    fn from(request: Request) -> Self {
        Self::ForwardRequest(request)
    }
}

pub struct ForwardResponseItem {
    pub serialized_response: Vec<u8>,
    pub response_tx: SendStream,
}

pub type ForwardResponseResult =
    Result<ForwardResponseItem, error::forward_response::TaskError>;

pub enum MailboxItem {
    Error(Error),
    /// Response computed on owned task set
    ForwardResponse(ForwardResponseItem),
    /// Internal messages from the connection task / net task / node
    InternalMessage(InternalMessage),
    /// Signals that a heartbeat message should be sent to the peer
    Heartbeat,
    /// Request received from peer
    PeerRequest((RequestMessage, SendStream)),
    /// Response received from peer
    PeerResponse(PeerResponseItem),
}

pub type BlockingTaskResult = Result<(), error::blocking_task::TaskError>;

/// Prioritize heartbeater above other streams
struct ReceiverStream<'a> {
    heartbeat: Fuse<IntervalStream>,
    blocking_task_queue: BoxStream<'a, error::blocking_task::Error>,
    others: SelectAll<BoxStream<'a, MailboxItem>>,
}

impl Stream for ReceiverStream<'_> {
    type Item = MailboxItem;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        match self.heartbeat.poll_next_unpin(cx) {
            Poll::Ready(Some(_heartbeat)) => {
                return Poll::Ready(Some(MailboxItem::Heartbeat));
            }
            Poll::Ready(None) | Poll::Pending => (),
        };
        match self.blocking_task_queue.poll_next_unpin(cx) {
            Poll::Ready(Some(err)) => {
                return Poll::Ready(Some(MailboxItem::Error(err.into())));
            }
            Poll::Ready(None) | Poll::Pending => (),
        };
        self.others.poll_next_unpin(cx)
    }
}

pub type BlockingTaskFn = Box<dyn FnOnce() -> BlockingTaskResult + Send>;

#[must_use]
pub struct Receiver {
    internal_message_rx: mpsc::UnboundedReceiver<InternalMessage>,
    blocking_task_queue_rx: mpsc::UnboundedReceiver<BlockingTaskFn>,
    send_response_rx:
        join_set::Receiver<Result<(), error::connection::SendResponse>>,
    forward_response_rx: join_set::Receiver<ForwardResponseResult>,
    request_queue_err_rx: request_queue::ErrorRx,
}

impl Receiver {
    /// `received_msg_successfully` is set to `True` if a valid message is
    /// received successfully.
    pub fn into_stream(
        self,
        connection: Connection,
        received_msg_successfully: &Arc<AtomicBool>,
    ) -> impl Stream<Item = MailboxItem> + Unpin {
        let Self {
            internal_message_rx,
            blocking_task_queue_rx,
            send_response_rx,
            forward_response_rx,
            request_queue_err_rx,
        } = self;
        let (peer_response_tx, peer_response_rx) = mpsc::unbounded();
        let internal_message_stream =
            internal_message_rx.map(MailboxItem::InternalMessage);
        let blocking_task_queue_stream =
            blocking_task_queue_rx.filter_map(|task| async {
                match tokio::task::spawn_blocking(task).await {
                    Ok(Ok(())) => None,
                    Ok(Err(err)) => Some(err.into()),
                    Err(err) => Some(err.into()),
                }
            });
        let send_response_stream = send_response_rx.filter_map(|res| async {
            match res {
                Ok(Ok(())) => None,
                Ok(Err(err)) => {
                    Some(MailboxItem::Error(Error::SendResponse(err)))
                }
                Err(err) => {
                    Some(MailboxItem::Error(Error::JoinSendResponse(err)))
                }
            }
        });
        let forward_response_stream =
            forward_response_rx.map(|item| match item {
                Ok(Ok(forward_response)) => {
                    MailboxItem::ForwardResponse(forward_response)
                }
                Ok(Err(err)) => {
                    MailboxItem::Error(Error::ForwardResponse(err.into()))
                }
                Err(err) => {
                    MailboxItem::Error(Error::ForwardResponse(err.into()))
                }
            });
        let heartbeat_stream =
            IntervalStream::new(interval(Connection::HEARTBEAT_SEND_INTERVAL));
        let request_queue_err_stream = request_queue_err_rx
            .into_stream(connection.clone(), peer_response_tx)
            .map(|err| MailboxItem::Error(err.into()));
        let peer_request_stream = stream::try_unfold((), move |()| {
            let conn = connection.clone();
            let received_msg_successfully = received_msg_successfully.clone();
            let fut = async move {
                let item = timeout(
                    Connection::HEARTBEAT_TIMEOUT_INTERVAL,
                    conn.receive_request().inspect_ok(|_| {
                        received_msg_successfully
                            .store(true, atomic::Ordering::SeqCst);
                    }),
                )
                .map_err(|_| Error::HeartbeatTimeout)
                .await??;
                Result::<_, Error>::Ok(Some((item, ())))
            };
            Box::pin(fut)
        })
        .map(|item| match item {
            Ok(peer_request) => MailboxItem::PeerRequest(peer_request),
            Err(err) => MailboxItem::Error(err),
        });
        let peer_response_stream = peer_response_rx.map(|resp| {
            if resp.response.is_ok() {
                received_msg_successfully.store(true, atomic::Ordering::SeqCst);
            }
            MailboxItem::PeerResponse(resp)
        });
        ReceiverStream {
            heartbeat: heartbeat_stream.fuse(),
            blocking_task_queue: blocking_task_queue_stream.boxed(),
            others: stream::select_all([
                internal_message_stream.boxed(),
                forward_response_stream.boxed(),
                send_response_stream.boxed(),
                peer_request_stream.boxed(),
                peer_response_stream.boxed(),
                request_queue_err_stream.boxed(),
            ]),
        }
    }
}

pub struct Sender {
    pub internal_message_tx: mpsc::UnboundedSender<InternalMessage>,
    pub blocking_task_queue_tx: mpsc::UnboundedSender<BlockingTaskFn>,
    pub send_response_spawner:
        join_set::Spawner<Result<(), error::connection::SendResponse>>,
    pub forward_response_spawner: join_set::Spawner<ForwardResponseResult>,
    pub request_tx: request_queue::Sender,
}

pub fn new() -> (Sender, Receiver) {
    let (internal_message_tx, internal_message_rx) = mpsc::unbounded();
    let (blocking_task_queue_tx, blocking_task_queue_rx) = mpsc::unbounded();
    let (send_response_spawner, send_response_rx) = join_set::new();
    let (forward_response_spawner, forward_response_rx) = join_set::new();
    let (request_tx, request_queue_err_rx) = request_queue::new();
    let receiver = Receiver {
        internal_message_rx,
        blocking_task_queue_rx,
        send_response_rx,
        forward_response_rx,
        request_queue_err_rx,
    };
    let sender = Sender {
        internal_message_tx,
        blocking_task_queue_tx,
        send_response_spawner,
        forward_response_spawner,
        request_tx,
    };
    (sender, receiver)
}
