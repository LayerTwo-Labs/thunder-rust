use std::{
    collections::HashSet,
    net::SocketAddr,
    sync::{
        Arc,
        atomic::{self, AtomicBool},
    },
};

use bitcoin::Work;
use borsh::BorshSerialize;
use futures::channel::mpsc;
use nonempty::NonEmpty;
use quinn::SendStream;
use serde::{Deserialize, Serialize};
use sneed::DbError;
use thiserror::Error;
use tokio::{spawn, task::JoinHandle, time::Duration};

use crate::{
    archive::{self, Archive},
    state::{self, State},
    types::{
        AuthorizedTransaction, BlockHash, Body, Hash, Header, Tip, Txid,
        Version, hash, schema,
    },
};

mod mailbox;
mod request_queue;
mod task;

pub use mailbox::InternalMessage;
use task::ConnectionTask;

#[derive(Debug, Error)]
pub enum BanReason {
    #[error(
        "BMM verification failed for block hash {} at {}",
        .0.block_hash,
        .0.main_block_hash
    )]
    BmmVerificationFailed(Tip),
    #[error(
        "Incorrect total work for block {} at {}: {total_work}",
        tip.block_hash,
        tip.main_block_hash
    )]
    IncorrectTotalWork { tip: Tip, total_work: Work },
}

#[must_use]
#[derive(Debug, Error)]
pub enum ConnectionError {
    #[error("archive error")]
    Archive(#[from] archive::Error),
    #[error("bincode error")]
    Bincode(#[from] bincode::Error),
    #[error("connection already closed")]
    ClosedStream(#[from] quinn::ClosedStream),
    #[error("connection error")]
    Connection(#[from] quinn::ConnectionError),
    #[error(transparent)]
    Db(#[from] DbError),
    #[error("Database env error")]
    DbEnv(#[from] sneed::env::Error),
    #[error("Heartbeat timeout")]
    HeartbeatTimeout,
    #[error("missing peer state for id {0}")]
    MissingPeerState(PeerStateId),
    #[error("peer should be banned; {0}")]
    PeerBan(#[from] BanReason),
    #[error("read to end error")]
    ReadToEnd(#[from] quinn::ReadToEndError),
    #[error(transparent)]
    RequestQueue(#[from] request_queue::SendError),
    #[error("send datagram error")]
    SendDatagram(#[from] quinn::SendDatagramError),
    #[error("send internal message error")]
    SendInternalMessage,
    #[error("send info error")]
    SendInfo,
    #[error("state error")]
    State(#[from] state::Error),
    #[error("write error ({stream_id})")]
    Write {
        stream_id: quinn::StreamId,
        source: quinn::WriteError,
    },
}

impl From<mpsc::TrySendError<Info>> for ConnectionError {
    fn from(_: mpsc::TrySendError<Info>) -> Self {
        Self::SendInfo
    }
}

impl From<mpsc::TrySendError<InternalMessage>> for ConnectionError {
    fn from(_: mpsc::TrySendError<InternalMessage>) -> Self {
        Self::SendInternalMessage
    }
}

fn borsh_serialize_work<W>(work: &Work, writer: &mut W) -> borsh::io::Result<()>
where
    W: borsh::io::Write,
{
    borsh::BorshSerialize::serialize(&work.to_le_bytes(), writer)
}

#[derive(BorshSerialize, Clone, Copy, Debug, Deserialize, Serialize)]
pub struct TipInfo {
    block_height: u32,
    tip: Tip,
    #[borsh(serialize_with = "borsh_serialize_work")]
    total_work: Work,
}

#[derive(BorshSerialize, Clone, Copy, Debug, Deserialize, Serialize)]
pub struct PeerState {
    tip_info: Option<TipInfo>,
    version: Version,
}

/// Unique identifier for a peer state
#[derive(Clone, Copy, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct PeerStateId(Hash);

impl From<&PeerState> for PeerStateId {
    fn from(peer_state: &PeerState) -> Self {
        Self(hash(peer_state))
    }
}

impl std::fmt::Debug for PeerStateId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        hex::encode(self.0).fmt(f)
    }
}

impl std::fmt::Display for PeerStateId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        hex::encode(self.0).fmt(f)
    }
}

#[derive(educe::Educe, Serialize, Deserialize)]
#[educe(Debug)]
pub enum Response {
    Block {
        header: Header,
        body: Body,
    },
    /// Headers, from start to end
    Headers(#[educe(Debug(method(Response::fmt_headers)))] Vec<Header>),
    NoBlock {
        block_hash: BlockHash,
    },
    NoHeader {
        block_hash: BlockHash,
    },
    TransactionAccepted(Txid),
    TransactionRejected(Txid),
    /// Blocks, sorted newest-first
    Blocks(NonEmpty<(Header, Body)>),
}

impl Response {
    /// Format headers for `Debug` impl
    fn fmt_headers(
        headers: &Vec<Header>,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        if let [first, .., last] = headers.as_slice() {
            write!(f, "[{first:?}, .., {last:?}]")
        } else {
            std::fmt::Debug::fmt(headers, f)
        }
    }
}

#[derive(BorshSerialize, Clone, Debug, Deserialize, Serialize)]
pub enum Request {
    Heartbeat(PeerState),
    GetBlock {
        block_hash: BlockHash,
        /// Mainchain descendant tip that we are requesting the block to reach.
        /// Only relevant for the requester, so serialization is skipped
        #[borsh(skip)]
        #[serde(skip)]
        descendant_tip: Option<Tip>,
        /// Ancestor block. If no bodies are missing between `descendant_tip`
        /// and `ancestor`, then `descendant_tip` is ready to apply.
        /// Only relevant for the requester, so serialization is skipped
        #[borsh(skip)]
        #[serde(skip)]
        ancestor: Option<BlockHash>,
        /// Only relevant for the requester, so serialization is skipped
        #[borsh(skip)]
        #[serde(skip)]
        peer_state_id: Option<PeerStateId>,
    },
    /// Request headers up to [`end`]
    GetHeaders {
        /// Request headers AFTER (not including) the first ancestor found in
        /// the specified list, if such an ancestor exists.
        start: HashSet<BlockHash>,
        end: BlockHash,
        /// Height is only relevant for the requester,
        /// so serialization is skipped
        #[borsh(skip)]
        #[serde(skip)]
        height: Option<u32>,
        /// Only relevant for the requester, so serialization is skipped
        #[borsh(skip)]
        #[serde(skip)]
        peer_state_id: Option<PeerStateId>,
    },
    PushTransaction {
        transaction: AuthorizedTransaction,
    },
    GetBlocks {
        block_hash: BlockHash,
        /// Request up to max_ancestors ancestor blocks, if they exist
        max_ancestors: u8,
        /// Mainchain descendant tip that we are requesting the block to reach.
        /// Only relevant for the requester, so serialization is skipped
        #[borsh(skip)]
        #[serde(skip)]
        descendant_tip: Option<Tip>,
        /// Ancestor block. If no bodies are missing between `descendant_tip`
        /// and `ancestor`, then `descendant_tip` is ready to apply.
        /// Only relevant for the requester, so serialization is skipped
        #[borsh(skip)]
        #[serde(skip)]
        ancestor: Option<BlockHash>,
        /// Only relevant for the requester, so serialization is skipped
        #[borsh(skip)]
        #[serde(skip)]
        peer_state_id: Option<PeerStateId>,
    },
}

/// Info to send to the net task / node
#[must_use]
#[derive(Debug)]
pub enum Info {
    Error(ConnectionError),
    /// Need BMM verification for the specified tip
    NeedBmmVerification {
        main_hash: bitcoin::BlockHash,
        peer_state_id: PeerStateId,
    },
    /// Need Mainchain ancestors for the specified tip
    NeedMainchainAncestors {
        main_hash: bitcoin::BlockHash,
        peer_state_id: PeerStateId,
    },
    /// New tip ready (body and header exist in archive, BMM verified)
    NewTipReady(Tip),
    NewTransaction(AuthorizedTransaction),
    Response(Box<(Response, Request)>),
}

impl From<ConnectionError> for Info {
    fn from(err: ConnectionError) -> Self {
        Self::Error(err)
    }
}

impl<T> From<Result<T, ConnectionError>> for Info
where
    Info: From<T>,
{
    fn from(res: Result<T, ConnectionError>) -> Self {
        match res {
            Ok(value) => value.into(),
            Err(err) => Self::Error(err),
        }
    }
}

#[derive(Clone)]
pub struct Connection {
    pub(in crate::net) inner: quinn::Connection,
}

impl Connection {
    // 100KB limit for reading requests (tx size could be ~100KB)
    pub const READ_REQUEST_LIMIT: usize = 100 * 1024;

    pub const HEARTBEAT_SEND_INTERVAL: Duration = Duration::from_secs(1);

    pub const HEARTBEAT_TIMEOUT_INTERVAL: Duration = Duration::from_secs(5);

    // 10MB limit for blocks
    pub const READ_BLOCK_LIMIT: usize = 10 * 1024 * 1024;

    // 1KB limit per header
    pub const READ_HEADER_LIMIT: usize = 1024;

    // 256B limit per tx ack (response size is ~192)
    pub const READ_TX_ACK_LIMIT: usize = 256;

    pub const fn read_response_limit(req: &Request) -> usize {
        // TODO: Add constant for discriminant
        match req {
            Request::GetBlock { .. } => Self::READ_BLOCK_LIMIT,
            Request::GetHeaders {
                height: Some(height),
                ..
            } => (*height as usize + 1) * Self::READ_HEADER_LIMIT,
            // Should have no response, so limit zero
            Request::Heartbeat(_) => 0,
            Request::PushTransaction { .. } => Self::READ_TX_ACK_LIMIT,
            // Should never happen, so limit zero
            Request::GetHeaders { height: None, .. } => 0,
            Request::GetBlocks { max_ancestors, .. } => {
                (*max_ancestors as usize + 1) * Self::READ_BLOCK_LIMIT
            }
        }
    }

    /// The rate limiter token cost for a request
    pub const fn request_cost(req: &Request) -> u32 {
        match req {
            Request::GetBlock { .. } => 1000,
            Request::GetHeaders { .. } => 10_000,
            Request::Heartbeat(_) => 0,
            Request::PushTransaction { .. } => 10,
            Request::GetBlocks { max_ancestors, .. } => {
                (*max_ancestors as u32 + 1) * 1000
            }
        }
    }

    pub fn addr(&self) -> SocketAddr {
        self.inner.remote_address()
    }

    pub async fn new(
        connecting: quinn::Connecting,
    ) -> Result<Self, ConnectionError> {
        let addr = connecting.remote_address();
        tracing::trace!(%addr, "connecting to peer");
        let connection = connecting.await?;
        tracing::info!(%addr, "connected successfully to peer");
        Ok(Self { inner: connection })
    }

    async fn receive_request(
        &self,
    ) -> Result<(Request, SendStream), ConnectionError> {
        let (tx, mut rx) = self.inner.accept_bi().await?;
        tracing::trace!(recv_id = %rx.id(), "Receiving request");
        let request_bytes =
            rx.read_to_end(Connection::READ_REQUEST_LIMIT).await?;
        let request: Request = bincode::deserialize(&request_bytes)?;
        tracing::trace!(
            recv_id = %rx.id(),
            ?request,
            "Received request"
        );
        Ok((request, tx))
    }

    pub async fn request(
        &self,
        message: &Request,
    ) -> Result<Option<Response>, ConnectionError> {
        let read_response_limit = Self::read_response_limit(message);
        let (mut send, mut recv) = self.inner.open_bi().await?;
        tracing::trace!(
            request = ?message,
            send_id = %send.id(),
            "Sending request"
        );
        let message = bincode::serialize(message)?;
        send.write_all(&message).await.map_err(|err| {
            ConnectionError::Write {
                stream_id: send.id(),
                source: err,
            }
        })?;
        send.finish()?;
        if read_response_limit > 0 {
            tracing::trace!(recv_id = %recv.id(), "Receiving response");
            let response_bytes = recv.read_to_end(read_response_limit).await?;
            let response: Response = bincode::deserialize(&response_bytes)?;
            tracing::trace!(
                recv_id = %recv.id(),
                ?response,
                "Received response"
            );
            Ok(Some(response))
        } else {
            Ok(None)
        }
    }
}

impl From<quinn::Connection> for Connection {
    fn from(inner: quinn::Connection) -> Self {
        Self { inner }
    }
}

pub struct ConnectionContext {
    pub env: sneed::Env,
    pub archive: Archive,
    pub state: State,
}

#[derive(
    Clone,
    Copy,
    Eq,
    PartialEq,
    serde::Serialize,
    serde::Deserialize,
    strum::Display,
    utoipa::ToSchema,
)]
pub enum PeerConnectionStatus {
    /// We're still in the process of initializing the peer connection
    Connecting,
    /// The connection is successfully established
    Connected,
}

impl PeerConnectionStatus {
    /// Convert from boolean representation
    // Should remain private to this module
    fn from_repr(repr: bool) -> Self {
        match repr {
            false => Self::Connecting,
            true => Self::Connected,
        }
    }

    /// Convert to boolean representation
    // Should remain private to this module
    fn as_repr(self) -> bool {
        match self {
            Self::Connecting => false,
            Self::Connected => true,
        }
    }
}

/// Connection killed on drop
pub struct ConnectionHandle {
    task: JoinHandle<()>,
    /// Representation of [`PeerConnectionStatus`]
    pub(in crate::net) status_repr: Arc<AtomicBool>,
    /// Push messages from connection task / net task / node
    pub internal_message_tx: mpsc::UnboundedSender<InternalMessage>,
}

impl ConnectionHandle {
    pub fn connection_status(&self) -> PeerConnectionStatus {
        PeerConnectionStatus::from_repr(
            self.status_repr.load(atomic::Ordering::SeqCst),
        )
    }
}

impl Drop for ConnectionHandle {
    fn drop(&mut self) {
        self.task.abort()
    }
}

/// Handle an existing connection
pub fn handle(
    ctxt: ConnectionContext,
    connection: Connection,
) -> (ConnectionHandle, mpsc::UnboundedReceiver<Info>) {
    let addr = connection.addr();

    let (info_tx, info_rx) = mpsc::unbounded();
    let (mailbox_tx, mailbox_rx) = mailbox::new();
    let internal_message_tx = mailbox_tx.internal_message_tx.clone();
    let connection_task = {
        let info_tx = info_tx.clone();
        move || async move {
            let connection_task = ConnectionTask {
                connection,
                ctxt,
                info_tx,
                mailbox_rx,
                mailbox_tx,
            };
            connection_task.run().await
        }
    };
    let task = spawn(async move {
        if let Err(err) = connection_task().await {
            tracing::error!(%addr, "connection task error, sending on info_tx: {err:#}");

            if let Err(send_error) = info_tx.unbounded_send(err.into())
                && let Info::Error(err) = send_error.into_inner()
            {
                tracing::warn!("Failed to send error to receiver: {err}")
            }
        }
    });
    let status = PeerConnectionStatus::Connected;
    let connection_handle = ConnectionHandle {
        task,
        status_repr: Arc::new(AtomicBool::new(status.as_repr())),
        internal_message_tx,
    };
    (connection_handle, info_rx)
}

pub fn connect(
    connecting: quinn::Connecting,
    ctxt: ConnectionContext,
) -> (ConnectionHandle, mpsc::UnboundedReceiver<Info>) {
    let connection_status = PeerConnectionStatus::Connecting;
    let status_repr = Arc::new(AtomicBool::new(connection_status.as_repr()));
    let (info_tx, info_rx) = mpsc::unbounded();
    let (mailbox_tx, mailbox_rx) = mailbox::new();
    let internal_message_tx = mailbox_tx.internal_message_tx.clone();
    let connection_task = {
        let status_repr = status_repr.clone();
        let info_tx = info_tx.clone();
        move || async move {
            let connection = Connection::new(connecting).await?;
            status_repr.store(
                PeerConnectionStatus::Connected.as_repr(),
                atomic::Ordering::SeqCst,
            );

            let connection_task = ConnectionTask {
                connection,
                ctxt,
                info_tx,
                mailbox_rx,
                mailbox_tx,
            };
            connection_task.run().await
        }
    };
    let task = spawn(async move {
        if let Err(err) = connection_task().await {
            if let Err(send_error) = info_tx.unbounded_send(err.into())
                && let Info::Error(err) = send_error.into_inner()
            {
                tracing::warn!("Failed to send error to receiver: {err}")
            }
        }
    });
    let connection_handle = ConnectionHandle {
        task,
        status_repr,
        internal_message_tx,
    };
    (connection_handle, info_rx)
}

// RPC output representation for peer + state
#[derive(Clone, serde::Deserialize, serde::Serialize, utoipa::ToSchema)]
pub struct Peer {
    #[schema(value_type = schema::SocketAddr)]
    pub address: SocketAddr,
    pub status: PeerConnectionStatus,
}
