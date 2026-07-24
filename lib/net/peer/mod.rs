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
    types::{
        AuthorizedTransaction, Hash, Network, Tip, Version, hash,
        net::PeerConnectionStatus,
    },
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
        const_hex::encode(self.0).fmt(f)
    }
}

impl std::fmt::Display for PeerStateId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        const_hex::encode(self.0).fmt(f)
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
    pub network: Network,
}

impl Connection {
    // 100KB limit for reading requests (tx size could be ~100KB)
    pub const READ_REQUEST_LIMIT: usize = 100 * 1024;

    pub const HEARTBEAT_SEND_INTERVAL: Duration = Duration::from_secs(1);

    pub const HEARTBEAT_TIMEOUT_INTERVAL: Duration = Duration::from_secs(5);

    pub const MIN_READ_RESPONSE_TIMEOUT: Duration = Duration::from_secs(5);

    pub fn addr(&self) -> SocketAddr {
        self.inner.remote_address()
    }

    /// Timeout for reading a full response from a peer, scaled to the maximum
    /// permitted response size.
    /// Bounds the wait for an unresponsive peer while leaving ample time for a
    /// large but steadily-progressing response.
    const fn response_read_timeout(
        read_response_limit: NonZeroUsize,
    ) -> Duration {
        // Minimum sustained throughput, in bytes per second, that a peer
        // streaming a response body is expected to achieve.
        // Used to scale the response read timeout to `read_response_limit`, so
        // a large (up to 10MB) block response is given proportionally more
        // time. Deliberately conservative so a slow or congested link is not
        // aborted; a peer that stops making progress entirely still hits the
        // timeout.
        const MIN_READ_RESPONSE_THROUGHPUT: u64 = 64 * 1024;

        let body_allowance = Duration::from_secs(
            read_response_limit.get() as u64 / MIN_READ_RESPONSE_THROUGHPUT,
        );
        Self::MIN_READ_RESPONSE_TIMEOUT.saturating_add(body_allowance)
    }

    pub fn new(connection: quinn::Connection, network: Network) -> Self {
        Self {
            inner: connection,
            network,
        }
    }

    pub async fn from_connecting(
        connecting: quinn::Connecting,
        network: Network,
    ) -> Result<Self, quinn::ConnectionError> {
        let addr = connecting.remote_address();
        tracing::trace!(%addr, "connecting to peer");
        let connection = connecting.await?;
        tracing::info!(%addr, "connected successfully to peer");
        Ok(Self {
            inner: connection,
            network,
        })
    }

    async fn receive_request(
        &self,
    ) -> Result<(RequestMessage, SendStream), error::connection::ReceiveRequest>
    {
        let (tx, mut rx) = self.inner.accept_bi().await?;
        tracing::trace!(recv_id = %rx.id(), "Receiving request");
        let mut magic_bytes = [0u8; message::MAGIC_BYTES_LEN];
        rx.read_exact(&mut magic_bytes)
            .await
            .map_err(error::connection::Receive::ReadMagic)?;
        if magic_bytes != message::magic_bytes(self.network) {
            return Err(
                error::connection::Receive::BadMagic(magic_bytes).into()
            );
        }
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
        let mut message_buf = message::magic_bytes(self.network).to_vec();
        bincode::serialize_into::<&mut Vec<_>, _>(&mut message_buf, &message)?;
        send.write_all(&message_buf).await.map_err(|err| {
            error::connection::Send::Write {
                stream_id: send.id(),
                source: err,
            }
        })?;
        send.finish()?;
        Ok(())
    }

    async fn receive_response(
        network: Network,
        mut recv: RecvStream,
        read_response_limit: NonZeroUsize,
    ) -> ResponseResult {
        tracing::trace!(recv_id = %recv.id(), "Receiving response");
        let mut magic_bytes = [0u8; message::MAGIC_BYTES_LEN];
        recv.read_exact(&mut magic_bytes)
            .await
            .map_err(error::connection::Receive::ReadMagic)?;
        if magic_bytes != message::magic_bytes(network) {
            return Err(
                error::connection::Receive::BadMagic(magic_bytes).into()
            );
        }
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
        let mut message_buf = message::magic_bytes(self.network).to_vec();
        bincode::serialize_into::<&mut Vec<_>, _>(&mut message_buf, &message)?;
        send.write_all(&message_buf).await.map_err(|err| {
            error::connection::Send::Write {
                stream_id: send.id(),
                source: err,
            }
        })?;
        send.finish()?;
        // Bound the wait for a response so an unresponsive peer that holds the
        // bi-stream open cannot pin the outbound request channel indefinitely.
        // The timeout scales with the maximum response size, so a large (up to
        // 10MB) block response on a slow or congested link is not aborted.
        let response = match tokio::time::timeout(
            Self::response_read_timeout(read_response_limit),
            Self::receive_response(self.network, recv, read_response_limit),
        )
        .await
        {
            Ok(response) => response,
            Err(_elapsed) => Err(error::connection::Receive::Timeout.into()),
        };
        Ok(response)
    }

    // Send a pre-serialized response, where the response does not include
    // magic bytes
    async fn send_serialized_response(
        network: Network,
        mut response_tx: SendStream,
        serialized_response: &[u8],
    ) -> Result<(), error::connection::SendResponse> {
        tracing::trace!(
            send_id = %response_tx.id(),
            "Sending response"
        );
        async {
            response_tx
                .write_all(&message::magic_bytes(network))
                .await?;
            response_tx.write_all(serialized_response).await
        }
        .await
        .map_err(|err| {
            {
                error::connection::Send::Write {
                    stream_id: response_tx.id(),
                    source: err,
                }
            }
            .into()
        })
    }

    async fn send_response(
        network: Network,
        mut response_tx: SendStream,
        response: ResponseMessage,
    ) -> Result<(), error::connection::SendResponse> {
        tracing::trace!(
            ?response,
            send_id = %response_tx.id(),
            "Sending response"
        );
        let mut message_buf = message::magic_bytes(network).to_vec();
        bincode::serialize_into::<&mut Vec<_>, _>(&mut message_buf, &response)?;
        response_tx.write_all(&message_buf).await.map_err(|err| {
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

pub struct ConnectionContext {
    pub env: sneed::Env<heed::WithoutTls>,
    pub archive: Archive,
    pub network: Network,
    pub state: State,
}

/// Used to make `bool` representation explicit and unique
#[repr(transparent)]
struct StatusRepr(bool);

impl From<StatusRepr> for PeerConnectionStatus {
    fn from(repr: StatusRepr) -> Self {
        let StatusRepr(repr) = repr;
        match repr {
            false => Self::Connecting,
            true => Self::Connected,
        }
    }
}

impl From<PeerConnectionStatus> for StatusRepr {
    fn from(status: PeerConnectionStatus) -> Self {
        match status {
            PeerConnectionStatus::Connecting => Self(false),
            PeerConnectionStatus::Connected => Self(true),
        }
    }
}

/// Atomic representation of [`PeerConnectionStatus`]
#[repr(transparent)]
pub(in crate::net) struct AtomicStatus {
    atomic_repr: AtomicBool,
}

impl AtomicStatus {
    #[inline(always)]
    fn load(&self, ordering: atomic::Ordering) -> StatusRepr {
        StatusRepr(self.atomic_repr.load(ordering))
    }

    #[inline(always)]
    fn store(&self, repr: StatusRepr, ordering: atomic::Ordering) {
        self.atomic_repr.store(repr.0, ordering)
    }
}

impl<T> From<T> for AtomicStatus
where
    StatusRepr: From<T>,
{
    #[inline(always)]
    fn from(value: T) -> Self {
        let StatusRepr(repr) = value.into();
        Self {
            atomic_repr: AtomicBool::new(repr),
        }
    }
}

/// Connection killed on drop
pub struct ConnectionHandle {
    task: JoinHandle<()>,
    /// Indicates that at least one message has been received successfully
    pub(in crate::net) received_msg_successfully: Arc<AtomicBool>,
    /// Atomic representation of [`PeerConnectionStatus`]
    pub(in crate::net) status: Arc<AtomicStatus>,
    /// Push messages from connection task / net task / node
    pub internal_message_tx: mpsc::UnboundedSender<InternalMessage>,
}

impl ConnectionHandle {
    pub fn connection_status(&self) -> PeerConnectionStatus {
        self.status.load(atomic::Ordering::SeqCst).into()
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
        status: Arc::new(AtomicStatus::from(status)),
        internal_message_tx,
    };
    (connection_handle, info_rx)
}

pub fn connect(
    connecting: quinn::Connecting,
    ctxt: ConnectionContext,
) -> (ConnectionHandle, mpsc::UnboundedReceiver<Info>) {
    let connection_status = PeerConnectionStatus::Connecting;
    let status = Arc::new(AtomicStatus::from(connection_status));
    let received_msg_successfully = Arc::new(AtomicBool::new(false));
    let (info_tx, info_rx) = mpsc::unbounded();
    let (mailbox_tx, mailbox_rx) = mailbox::new();
    let internal_message_tx = mailbox_tx.internal_message_tx.clone();
    let connection_task = {
        let received_msg_successfully = received_msg_successfully.clone();
        let status = status.clone();
        let info_tx = info_tx.clone();
        move || async move {
            let connection =
                Connection::from_connecting(connecting, ctxt.network).await?;
            status.store(
                PeerConnectionStatus::Connected.into(),
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
        status,
        internal_message_tx,
    };
    (connection_handle, info_rx)
}

#[cfg(test)]
mod test {
    use std::num::NonZeroUsize;

    use crate::{
        net::peer::{Connection, message::GetBlockRequest},
        types::BlockHash,
    };

    /// A large (up to 10MB) block response must be granted substantially more
    /// time than the heartbeat timeout, so that a slow but steadily-progressing
    /// response over a congested link is not spuriously aborted (which would
    /// stall IBD).
    #[test]
    fn large_response_read_timeout_leaves_headroom() {
        let get_block = GetBlockRequest {
            block_hash: BlockHash([0u8; 32]),
            descendant_tip: None,
            ancestor: None,
            peer_state_id: None,
        };
        let timeout =
            Connection::response_read_timeout(get_block.read_response_limit());
        assert!(
            timeout > Connection::HEARTBEAT_TIMEOUT_INTERVAL,
            "10MB block response timeout {timeout:?} must exceed the heartbeat \
             timeout {:?}",
            Connection::HEARTBEAT_TIMEOUT_INTERVAL,
        );
    }

    /// A tiny response is still bounded, but never below the base allowance
    /// covering round-trip latency.
    #[test]
    fn small_response_read_timeout_at_least_base() {
        let limit = NonZeroUsize::new(256).unwrap();
        let timeout = Connection::response_read_timeout(limit);
        assert_eq!(timeout, Connection::MIN_READ_RESPONSE_TIMEOUT);
    }
}
