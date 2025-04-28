use std::{
    net::SocketAddr,
    num::NonZeroUsize,
    sync::{
        Arc,
        atomic::{self, AtomicBool},
    },
};

use bitcoin::Work;
use borsh::BorshSerialize;
use futures::channel::mpsc;
use quinn::{RecvStream, SendStream};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::{spawn, task::JoinHandle, time::Duration};

use crate::{
    archive::Archive,
    state::State,
    types::{AuthorizedTransaction, Hash, Tip, Version, hash, schema},
};

mod channel_pool;
pub(crate) mod error;
pub(crate) mod mailbox;
pub mod message;
mod request_queue;
mod task;

pub use error::Error as ConnectionError;
pub use mailbox::InternalMessage;
use message::{Heartbeat, RequestMessage, RequestMessageRef};
pub use message::{Request, ResponseMessage};
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

type ResponseResult =
    Result<ResponseMessage, error::connection::ReceiveResponse>;

pub struct PeerResponseItem {
    pub request: Request,
    pub response: ResponseResult,
}

/// Info to send to the net task / node
#[must_use]
#[derive(Debug)]
pub enum Info {
    Error(ConnectionError),
    /// Need Mainchain ancestors for the specified tip
    NeedMainchainAncestors {
        main_hash: bitcoin::BlockHash,
        peer_state_id: PeerStateId,
    },
    /// New tip ready (body and header exist in archive, BMM verified)
    NewTipReady(Tip),
    NewTransaction(AuthorizedTransaction),
    Response(Box<(ResponseMessage, Request)>),
}

impl From<ConnectionError> for Info {
    fn from(err: ConnectionError) -> Self {
        Self::Error(err)
    }
}

impl<E, T> From<Result<T, E>> for Info
where
    ConnectionError: From<E>,
    Info: From<T>,
{
    fn from(res: Result<T, E>) -> Self {
        match res {
            Ok(value) => value.into(),
            Err(err) => Self::Error(err.into()),
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

    pub fn addr(&self) -> SocketAddr {
        self.inner.remote_address()
    }

    pub async fn new(
        connecting: quinn::Connecting,
    ) -> Result<Self, quinn::ConnectionError> {
        let addr = connecting.remote_address();
        tracing::trace!(%addr, "connecting to peer");
        let connection = connecting.await?;
        tracing::info!(%addr, "connected successfully to peer");
        Ok(Self { inner: connection })
    }

    async fn receive_request(
        &self,
    ) -> Result<(RequestMessage, SendStream), error::connection::ReceiveRequest>
    {
        let (tx, mut rx) = self.inner.accept_bi().await?;
        tracing::trace!(recv_id = %rx.id(), "Receiving request");
        let msg_bytes = rx.read_to_end(Connection::READ_REQUEST_LIMIT).await?;
        let msg: RequestMessage = bincode::deserialize(&msg_bytes)?;
        tracing::trace!(
            recv_id = %rx.id(),
            ?msg,
            "Received request"
        );
        Ok((msg, tx))
    }

    async fn send_heartbeat(
        &self,
        heartbeat: &Heartbeat,
    ) -> Result<(), error::connection::SendHeartbeat> {
        let (mut send, _recv) = self.inner.open_bi().await?;
        tracing::trace!(
            heartbeat = ?heartbeat,
            send_id = %send.id(),
            "Sending heartbeat"
        );
        let message = RequestMessageRef::from(heartbeat);
        let message = bincode::serialize(&message)?;
        send.write_all(&message).await.map_err(|err| {
            error::connection::Send::Write {
                stream_id: send.id(),
                source: err,
            }
        })?;
        send.finish()?;
        Ok(())
    }

    async fn receive_response(
        mut recv: RecvStream,
        read_response_limit: NonZeroUsize,
    ) -> ResponseResult {
        tracing::trace!(recv_id = %recv.id(), "Receiving response");
        let response_bytes =
            recv.read_to_end(read_response_limit.get()).await?;
        let response: ResponseMessage = bincode::deserialize(&response_bytes)?;
        tracing::trace!(
            recv_id = %recv.id(),
            ?response,
            "Received response"
        );
        Ok(response)
    }

    async fn send_request(
        &self,
        request: &Request,
    ) -> Result<ResponseResult, error::connection::SendRequest> {
        let read_response_limit = request.read_response_limit();
        let (mut send, recv) = self.inner.open_bi().await?;
        tracing::trace!(
            request = ?request,
            send_id = %send.id(),
            "Sending request"
        );
        let message = RequestMessageRef::from(request);
        let message = bincode::serialize(&message)?;
        send.write_all(&message).await.map_err(|err| {
            error::connection::Send::Write {
                stream_id: send.id(),
                source: err,
            }
        })?;
        send.finish()?;
        Ok(Self::receive_response(recv, read_response_limit).await)
    }

    async fn send_response(
        mut response_tx: SendStream,
        response: ResponseMessage,
    ) -> Result<(), error::connection::SendResponse> {
        tracing::trace!(
            ?response,
            send_id = %response_tx.id(),
            "Sending response"
        );
        let response_bytes = bincode::serialize(&response)?;
        response_tx.write_all(&response_bytes).await.map_err(|err| {
            {
                error::connection::Send::Write {
                    stream_id: response_tx.id(),
                    source: err,
                }
            }
            .into()
        })
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
    /// Indicates that at least one message has been received successfully
    pub(in crate::net) received_msg_successfully: Arc<AtomicBool>,
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

    /// Indicates that at least one message has been received successfully
    pub fn received_msg_successfully(&self) -> bool {
        self.received_msg_successfully
            .load(atomic::Ordering::SeqCst)
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
    let received_msg_successfully = Arc::new(AtomicBool::new(false));
    let connection_task = {
        let info_tx = info_tx.clone();
        let received_msg_successfully = received_msg_successfully.clone();
        move || async move {
            let connection_task = ConnectionTask {
                connection,
                ctxt,
                info_tx,
                mailbox_rx,
                mailbox_tx,
                received_msg_successfully,
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
        received_msg_successfully,
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
    let received_msg_successfully = Arc::new(AtomicBool::new(false));
    let (info_tx, info_rx) = mpsc::unbounded();
    let (mailbox_tx, mailbox_rx) = mailbox::new();
    let internal_message_tx = mailbox_tx.internal_message_tx.clone();
    let connection_task = {
        let received_msg_successfully = received_msg_successfully.clone();
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
                received_msg_successfully,
            };
            connection_task.run().await
        }
    };
    let task = spawn(async move {
        if let Err(err) = connection_task().await
            && let Err(send_error) = info_tx.unbounded_send(err.into())
            && let Info::Error(err) = send_error.into_inner()
        {
            tracing::warn!("Failed to send error to receiver: {err}")
        }
    });
    let connection_handle = ConnectionHandle {
        task,
        received_msg_successfully,
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
