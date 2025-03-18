//! Mailbox for a peer connection task

use futures::{
    Stream, StreamExt as _, TryFutureExt as _, TryStreamExt as _,
    channel::mpsc, stream,
};
use quinn::SendStream;
use tokio::time::{interval, timeout};
use tokio_stream::wrappers::IntervalStream;

use crate::{
    net::peer::{
        Connection, ConnectionError, PeerStateId, Request, Response,
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

pub struct PeerResponseItem {
    pub request: Request,
    pub response: Result<Response, ConnectionError>,
}

pub enum MailboxItem {
    /// Internal messages from the connection task / net task / node
    InternalMessage(InternalMessage),
    /// Signals that a heartbeat message should be sent to the peer
    Heartbeat,
    /// Request received from peer
    PeerRequest((Request, SendStream)),
    /// Response received from peer
    PeerResponse(PeerResponseItem),
    /// Request that is ready to send to the peer
    SendRequest(Request),
}

pub struct Receiver {
    internal_message_rx: mpsc::UnboundedReceiver<InternalMessage>,
    peer_response_rx: mpsc::UnboundedReceiver<PeerResponseItem>,
    request_rx: request_queue::Receiver,
}

impl Receiver {
    pub fn into_stream(
        self,
        connection: &Connection,
    ) -> impl Stream<Item = Result<MailboxItem, ConnectionError>> + Unpin {
        let internal_message_stream = self
            .internal_message_rx
            .map(|msg| Ok(MailboxItem::InternalMessage(msg)));
        let heartbeat_stream =
            IntervalStream::new(interval(Connection::HEARTBEAT_SEND_INTERVAL))
                .map(|_| Ok(MailboxItem::Heartbeat));
        let peer_request_stream = stream::try_unfold((), move |()| {
            let conn = connection.clone();
            let fut = async move {
                let item = timeout(
                    Connection::HEARTBEAT_TIMEOUT_INTERVAL,
                    conn.receive_request(),
                )
                .map_err(|_| ConnectionError::HeartbeatTimeout)
                .await??;
                Result::<_, ConnectionError>::Ok(Some((item, ())))
            };
            Box::pin(fut)
        })
        .map_ok(MailboxItem::PeerRequest);
        let peer_response_stream = self
            .peer_response_rx
            .map(|peer_resp| Ok(MailboxItem::PeerResponse(peer_resp)));
        let send_request_stream = self
            .request_rx
            .into_stream()
            .map(|request| Ok(MailboxItem::SendRequest(request)));
        stream::select_all([
            internal_message_stream.boxed(),
            heartbeat_stream.boxed(),
            peer_request_stream.boxed(),
            peer_response_stream.boxed(),
            send_request_stream.boxed(),
        ])
    }
}

pub struct Sender {
    pub internal_message_tx: mpsc::UnboundedSender<InternalMessage>,
    pub peer_response_tx: mpsc::UnboundedSender<PeerResponseItem>,
    pub request_tx: request_queue::Sender,
}

pub fn new() -> (Sender, Receiver) {
    let (internal_message_tx, internal_message_rx) = mpsc::unbounded();
    let (peer_response_tx, peer_response_rx) = mpsc::unbounded();
    let (request_tx, request_rx) = request_queue::new();
    let receiver = Receiver {
        internal_message_rx,
        peer_response_rx,
        request_rx,
    };
    let sender = Sender {
        internal_message_tx,
        peer_response_tx,
        request_tx,
    };
    (sender, receiver)
}
