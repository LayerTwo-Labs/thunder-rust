//! Mailbox for a peer connection task

use std::sync::{
    Arc,
    atomic::{self, AtomicBool},
};

use futures::{
    Stream, StreamExt as _, TryFutureExt as _, channel::mpsc, stream,
};
use quinn::SendStream;
use tokio::time::{interval, timeout};
use tokio_stream::wrappers::IntervalStream;

use crate::{
    net::peer::{
        Connection, PeerResponseItem, PeerStateId,
        error::mailbox::Error,
        message::{Request, RequestMessage},
        request_queue,
    },
    types::proto::mainchain,
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

pub enum MailboxItem {
    Error(Error),
    /// Internal messages from the connection task / net task / node
    InternalMessage(InternalMessage),
    /// Signals that a heartbeat message should be sent to the peer
    Heartbeat,
    /// Request received from peer
    PeerRequest((RequestMessage, SendStream)),
    /// Response received from peer
    PeerResponse(PeerResponseItem),
}

#[must_use]
pub struct Receiver {
    internal_message_rx: mpsc::UnboundedReceiver<InternalMessage>,
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
        let (peer_response_tx, peer_response_rx) = mpsc::unbounded();
        let internal_message_stream =
            self.internal_message_rx.map(MailboxItem::InternalMessage);
        let heartbeat_stream =
            IntervalStream::new(interval(Connection::HEARTBEAT_SEND_INTERVAL))
                .map(|_| MailboxItem::Heartbeat);
        let request_queue_err_stream = self
            .request_queue_err_rx
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
        stream::select_all([
            internal_message_stream.boxed(),
            heartbeat_stream.boxed(),
            peer_request_stream.boxed(),
            peer_response_stream.boxed(),
            request_queue_err_stream.boxed(),
        ])
    }
}

pub struct Sender {
    pub internal_message_tx: mpsc::UnboundedSender<InternalMessage>,
    pub request_tx: request_queue::Sender,
}

pub fn new() -> (Sender, Receiver) {
    let (internal_message_tx, internal_message_rx) = mpsc::unbounded();
    let (request_tx, request_queue_err_rx) = request_queue::new();
    let receiver = Receiver {
        internal_message_rx,
        request_queue_err_rx,
    };
    let sender = Sender {
        internal_message_tx,
        request_tx,
    };
    (sender, receiver)
}
