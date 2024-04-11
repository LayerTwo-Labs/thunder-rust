use std::net::SocketAddr;

use futures::{channel::mpsc, stream, StreamExt, TryFutureExt, TryStreamExt};
use quinn::{Endpoint, SendStream};
use serde::{Deserialize, Serialize};
use tokio::{
    spawn,
    task::{JoinHandle, JoinSet},
    time::{interval, timeout, Duration},
};
use tokio_stream::wrappers::IntervalStream;

use crate::{
    archive::{self, Archive},
    state::{self, State},
    types::{AuthorizedTransaction, Body, Header, Txid},
};

#[derive(Debug, thiserror::Error)]
pub enum ConnectionError {
    #[error("archive error")]
    Archive(#[from] archive::Error),
    #[error("bincode error")]
    Bincode(#[from] bincode::Error),
    #[error("connect error")]
    Connect(#[from] quinn::ConnectError),
    #[error("connection error")]
    Connection(#[from] quinn::ConnectionError),
    #[error("Heartbeat timeout")]
    HeartbeatTimeout,
    #[error("heed error")]
    Heed(#[from] heed::Error),
    #[error("read to end error")]
    ReadToEnd(#[from] quinn::ReadToEndError),
    #[error("send datagram error")]
    SendDatagram(#[from] quinn::SendDatagramError),
    #[error("send info error")]
    SendInfo,
    #[error("state error")]
    State(#[from] state::Error),
    #[error("write error")]
    Write(#[from] quinn::WriteError),
}

impl From<mpsc::TrySendError<Info>> for ConnectionError {
    fn from(_: mpsc::TrySendError<Info>) -> Self {
        Self::SendInfo
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct PeerState {
    block_height: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum Response {
    Block { header: Header, body: Body },
    NoBlock { height: u32 },
    TransactionAccepted(Txid),
    TransactionRejected(Txid),
}

#[derive(Debug, Deserialize, Serialize)]
pub enum Request {
    Heartbeat(PeerState),
    GetBlock { height: u32 },
    PushTransaction { transaction: AuthorizedTransaction },
}

#[must_use]
#[derive(Debug)]
pub enum Info {
    Error(ConnectionError),
    NewTransaction(AuthorizedTransaction),
    Response(Response, Request),
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
pub struct Connection(pub(super) quinn::Connection);

impl Connection {
    pub const READ_LIMIT: usize = 1024;

    pub const HEARTBEAT_SEND_INTERVAL: Duration = Duration::from_secs(1);

    pub const HEARTBEAT_TIMEOUT_INTERVAL: Duration = Duration::from_secs(5);

    pub fn addr(&self) -> SocketAddr {
        self.0.remote_address()
    }

    pub async fn new(
        endpoint: &Endpoint,
        addr: SocketAddr,
    ) -> Result<Self, ConnectionError> {
        let connection = endpoint.connect(addr, "localhost")?.await?;
        tracing::info!("Connected to peer at {addr}");
        Ok(Self(connection))
    }

    async fn receive_request(
        &self,
    ) -> Result<(Request, SendStream), ConnectionError> {
        let (tx, mut rx) = self.0.accept_bi().await?;
        let request_bytes = rx.read_to_end(Connection::READ_LIMIT).await?;
        let request: Request = bincode::deserialize(&request_bytes)?;
        Ok((request, tx))
    }

    pub fn heart_beat(&self, state: &PeerState) -> Result<(), ConnectionError> {
        let message = bincode::serialize(state)?;
        self.0.send_datagram(bytes::Bytes::from(message))?;
        Ok(())
    }

    pub async fn request(
        &self,
        message: &Request,
    ) -> Result<Response, ConnectionError> {
        let (mut send, mut recv) = self.0.open_bi().await?;
        let message = bincode::serialize(message)?;
        send.write_all(&message).await?;
        send.finish().await?;
        let response = recv.read_to_end(Self::READ_LIMIT).await?;
        let response: Response = bincode::deserialize(&response)?;
        Ok(response)
    }
}

pub struct ConnectionContext {
    pub env: heed::Env,
    pub archive: Archive,
    pub state: State,
}

struct ConnectionTask {
    connection: Connection,
    ctxt: ConnectionContext,
    info_tx: mpsc::UnboundedSender<Info>,
    peer_state: Option<PeerState>,
    /// Receive requests to forward to the peer
    forward_request_rx: mpsc::UnboundedReceiver<Request>,
}

impl ConnectionTask {
    async fn send_request(
        conn: &Connection,
        info_tx: &mpsc::UnboundedSender<Info>,
        request: Request,
    ) {
        let resp = conn.request(&request).await;
        let info = resp.map(|resp| Info::Response(resp, request)).into();
        if info_tx.unbounded_send(info).is_err() {
            let addr = conn.addr();
            tracing::error!(%addr, "Failed to send peer connection info")
        };
    }

    async fn send_response(
        mut response_tx: SendStream,
        response: Response,
    ) -> Result<(), ConnectionError> {
        let response_bytes = bincode::serialize(&response)?;
        response_tx
            .write_all(&response_bytes)
            .await
            .map_err(ConnectionError::from)
    }

    async fn handle_heartbeat(
        ctxt: &ConnectionContext,
        conn: &Connection,
        info_tx: &mpsc::UnboundedSender<Info>,
        peer_state: &PeerState,
    ) -> Result<(), ConnectionError> {
        let height = {
            let rotxn = ctxt.env.read_txn()?;
            ctxt.archive.get_height(&rotxn)
        }?;
        let peer_height = peer_state.block_height;
        if peer_height > height {
            let request = Request::GetBlock { height: height + 1 };
            Self::send_request(conn, info_tx, request).await
        }
        Ok(())
    }

    async fn handle_get_block(
        ctxt: &ConnectionContext,
        response_tx: SendStream,
        height: u32,
    ) -> Result<(), ConnectionError> {
        let (header, body) = {
            let rotxn = ctxt.env.read_txn()?;
            let header = ctxt.archive.get_header(&rotxn, height)?;
            let body = ctxt.archive.get_body(&rotxn, height)?;
            (header, body)
        };
        let resp = match (header, body) {
            (Some(header), Some(body)) => Response::Block { header, body },
            (_, _) => Response::NoBlock { height },
        };
        Self::send_response(response_tx, resp).await
    }

    async fn handle_push_tx(
        ctxt: &ConnectionContext,
        info_tx: &mpsc::UnboundedSender<Info>,
        response_tx: SendStream,
        tx: AuthorizedTransaction,
    ) -> Result<(), ConnectionError> {
        let txid = tx.transaction.txid();
        let validate_tx_result = {
            let rotxn = ctxt.env.read_txn()?;
            ctxt.state.validate_transaction(&rotxn, &tx)
        };
        match validate_tx_result {
            Err(err) => {
                Self::send_response(
                    response_tx,
                    Response::TransactionRejected(txid),
                )
                .await?;
                Err(ConnectionError::from(err))
            }
            Ok(_) => {
                Self::send_response(
                    response_tx,
                    Response::TransactionAccepted(txid),
                )
                .await?;
                info_tx.unbounded_send(Info::NewTransaction(tx))?;
                Ok(())
            }
        }
    }

    async fn handle_request(
        ctxt: &ConnectionContext,
        conn: &Connection,
        info_tx: &mpsc::UnboundedSender<Info>,
        peer_state: &mut Option<PeerState>,
        response_tx: SendStream,
        request: Request,
    ) -> Result<(), ConnectionError> {
        match request {
            Request::Heartbeat(new_peer_state) => {
                let () = Self::handle_heartbeat(
                    ctxt,
                    conn,
                    info_tx,
                    &new_peer_state,
                )
                .await?;
                *peer_state = Some(new_peer_state);
                Ok(())
            }
            Request::GetBlock { height } => {
                Self::handle_get_block(ctxt, response_tx, height).await
            }
            Request::PushTransaction { transaction } => {
                Self::handle_push_tx(ctxt, info_tx, response_tx, transaction)
                    .await
            }
        }
    }

    async fn run(mut self) -> Result<(), ConnectionError> {
        enum MailboxItem {
            ForwardRequest(Request),
            /// Signals that a heartbeat message should be sent to the peer
            Heartbeat,
            Request((Request, SendStream)),
        }
        let forward_request_stream = self
            .forward_request_rx
            .map(|request| Ok(MailboxItem::ForwardRequest(request)));
        let heartbeat_stream =
            IntervalStream::new(interval(Connection::HEARTBEAT_SEND_INTERVAL))
                .map(|_| Ok(MailboxItem::Heartbeat));
        let request_stream = stream::try_unfold((), {
            let conn = self.connection.clone();
            move |()| {
                let conn = conn.clone();
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
            }
        })
        .map_ok(MailboxItem::Request);
        let mut mailbox_stream = stream::select_all([
            forward_request_stream.boxed(),
            heartbeat_stream.boxed(),
            request_stream.boxed(),
        ]);
        // spawn child tasks on a JoinSet so that they are dropped alongside this task
        let mut task_set: JoinSet<()> = JoinSet::new();
        while let Some(mailbox_item) = mailbox_stream.try_next().await? {
            match mailbox_item {
                MailboxItem::ForwardRequest(request) => {
                    task_set.spawn({
                        let connection = self.connection.clone();
                        let info_tx = self.info_tx.clone();
                        async move {
                            Self::send_request(&connection, &info_tx, request)
                                .await
                        }
                    });
                }
                MailboxItem::Heartbeat => {
                    let tip_height = {
                        let rotxn = self.ctxt.env.read_txn()?;
                        self.ctxt.archive.get_height(&rotxn)?
                    };
                    let state_msg = PeerState {
                        block_height: tip_height,
                    };
                    let () = self.connection.heart_beat(&state_msg)?;
                }
                MailboxItem::Request((request, response_tx)) => {
                    let () = Self::handle_request(
                        &self.ctxt,
                        &self.connection,
                        &self.info_tx,
                        &mut self.peer_state,
                        response_tx,
                        request,
                    )
                    .await?;
                }
            }
        }
        Ok(())
    }
}

/// Connection killed on drop
pub struct ConnectionHandle {
    task: JoinHandle<()>,
    pub forward_request_tx: mpsc::UnboundedSender<Request>,
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
    let (forward_request_tx, forward_request_rx) = mpsc::unbounded();
    let (info_tx, info_rx) = mpsc::unbounded();
    let connection_task = {
        let info_tx = info_tx.clone();
        move || async move {
            let connection_task = ConnectionTask {
                connection,
                ctxt,
                info_tx,
                peer_state: None,
                forward_request_rx,
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
        forward_request_tx,
    };
    (connection_handle, info_rx)
}

pub fn connect(
    endpoint: Endpoint,
    addr: SocketAddr,
    ctxt: ConnectionContext,
) -> (ConnectionHandle, mpsc::UnboundedReceiver<Info>) {
    let (forward_request_tx, forward_request_rx) = mpsc::unbounded();
    let (info_tx, info_rx) = mpsc::unbounded();
    let connection_task = {
        let info_tx = info_tx.clone();
        move || async move {
            let connection = Connection::new(&endpoint, addr).await?;
            let connection_task = ConnectionTask {
                connection,
                ctxt,
                info_tx,
                peer_state: None,
                forward_request_rx,
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
        forward_request_tx,
    };
    (connection_handle, info_rx)
}
