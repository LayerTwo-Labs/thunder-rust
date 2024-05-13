use std::{collections::HashSet, net::SocketAddr};

use bip300301::bitcoin::{self, hashes::Hash as _};
use borsh::BorshSerialize;
use fallible_iterator::FallibleIterator;
use futures::{channel::mpsc, stream, StreamExt, TryFutureExt, TryStreamExt};
use quinn::{Endpoint, SendStream};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::{
    spawn,
    task::{JoinHandle, JoinSet},
    time::{interval, timeout, Duration},
};
use tokio_stream::wrappers::IntervalStream;

use crate::{
    archive::{self, Archive},
    state::{self, State},
    types::{hash, AuthorizedTransaction, BlockHash, Body, Hash, Header, Txid},
};

#[derive(Debug, Error)]
pub enum BanReason {
    #[error("BMM verification failed for block {0}")]
    BmmVerificationFailed(BlockHash),
    #[error("Incorrect total work for block {block_hash}: {total_work:?}")]
    IncorrectTotalWork {
        block_hash: BlockHash,
        total_work: Option<bitcoin::Work>,
    },
}

#[must_use]
#[derive(Debug, Error)]
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
    #[error("peer should be banned; {0}")]
    PeerBan(#[from] BanReason),
    #[error("read to end error")]
    ReadToEnd(#[from] quinn::ReadToEndError),
    #[error("send datagram error")]
    SendDatagram(#[from] quinn::SendDatagramError),
    #[error("send forward request error")]
    SendForwardRequest,
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

impl From<mpsc::TrySendError<Request>> for ConnectionError {
    fn from(_: mpsc::TrySendError<Request>) -> Self {
        Self::SendForwardRequest
    }
}

fn borsh_serialize_work<W>(
    work: &bitcoin::Work,
    writer: &mut W,
) -> borsh::io::Result<()>
where
    W: borsh::io::Write,
{
    borsh::BorshSerialize::serialize(&work.to_le_bytes(), writer)
}

fn borsh_serialize_option_work<W>(
    work: &Option<bitcoin::Work>,
    writer: &mut W,
) -> borsh::io::Result<()>
where
    W: borsh::io::Write,
{
    #[derive(BorshSerialize)]
    struct BorshWrapper(
        #[borsh(serialize_with = "borsh_serialize_work")] bitcoin::Work,
    );
    borsh::BorshSerialize::serialize(&work.map(BorshWrapper), writer)
}

#[derive(BorshSerialize, Clone, Debug, Default, Deserialize, Serialize)]
pub struct PeerState {
    block_height: u32,
    tip: BlockHash,
    #[borsh(serialize_with = "borsh_serialize_option_work")]
    total_work: Option<bitcoin::Work>,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum Response {
    Block {
        header: Header,
        body: Body,
    },
    /// Headers, from start to end
    Headers(Vec<Header>),
    NoBlock {
        block_hash: BlockHash,
    },
    NoHeader {
        block_hash: BlockHash,
    },
    TransactionAccepted(Txid),
    TransactionRejected(Txid),
}

#[derive(BorshSerialize, Clone, Debug, Deserialize, Serialize)]
pub enum Request {
    Heartbeat(PeerState),
    GetBlock {
        block_hash: BlockHash,
    },
    /// Request headers up to [`end`]
    GetHeaders {
        /// Request headers AFTER (not including) the first ancestor found in
        /// the specified list, if such an ancestor exists.
        start: HashSet<BlockHash>,
        end: BlockHash,
        /// Height is only relevant for the requester,
        /// so serialization is skipped
        #[serde(skip)]
        height: Option<u32>,
    },
    PushTransaction {
        transaction: AuthorizedTransaction,
    },
}

#[must_use]
#[derive(Debug)]
pub enum Info {
    Error(ConnectionError),
    /// Need BMM verification for the specified block
    NeedBmmVerification(BlockHash),
    /// Need Mainchain ancestors for the specified block hash
    NeedMainchainAncestors(BlockHash),
    /// New tip ready (body and header exist in archive, BMM verified)
    NewTipReady(BlockHash),
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
        match req {
            Request::GetBlock { .. } => Self::READ_BLOCK_LIMIT,
            Request::GetHeaders {
                height: Some(height),
                ..
            } => *height as usize * Self::READ_HEADER_LIMIT,
            // Should have no response, so limit zero
            Request::Heartbeat(_) => 0,
            Request::PushTransaction { .. } => Self::READ_TX_ACK_LIMIT,
            // Should never happen, so limit zero
            Request::GetHeaders { height: None, .. } => 0,
        }
    }

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
        let request_bytes =
            rx.read_to_end(Connection::READ_REQUEST_LIMIT).await?;
        let request: Request = bincode::deserialize(&request_bytes)?;
        Ok((request, tx))
    }

    pub async fn request(
        &self,
        message: &Request,
    ) -> Result<Option<Response>, ConnectionError> {
        let read_response_limit = Self::read_response_limit(message);
        let (mut send, mut recv) = self.0.open_bi().await?;
        let message = bincode::serialize(message)?;
        send.write_all(&message).await?;
        send.finish().await?;
        if read_response_limit > 0 {
            let response_bytes = recv.read_to_end(read_response_limit).await?;
            let response: Response = bincode::deserialize(&response_bytes)?;
            Ok(Some(response))
        } else {
            Ok(None)
        }
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
    /// Push a request to forward to the peer
    forward_request_tx: mpsc::UnboundedSender<Request>,
    /// Receive requests to forward to the peer
    forward_request_rx: mpsc::UnboundedReceiver<Request>,
}

impl ConnectionTask {
    async fn send_request(
        conn: &Connection,
        response_tx: &mpsc::UnboundedSender<(
            Result<Response, ConnectionError>,
            Request,
        )>,
        request: Request,
    ) {
        let resp = match conn.request(&request).await {
            Ok(Some(resp)) => Ok(resp),
            Err(err) => Err(err),
            Ok(None) => return,
        };
        if response_tx.unbounded_send((resp, request)).is_err() {
            let addr = conn.addr();
            tracing::error!(%addr, "Failed to send response")
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

    /// If a new tip is announced with greater height than the current tip:
    /// * If the header does not exist, request it
    /// * Verify height of the new tip.
    /// * If the previous mainchain header does not exist, request it
    /// * Verify PoW
    /// * Verify BMM
    /// * If ancestor bodies do not exist, request them
    /// * Attempt to apply the new tip
    async fn handle_heartbeat(
        ctxt: &ConnectionContext,
        info_tx: &mpsc::UnboundedSender<Info>,
        forward_request_tx: &mpsc::UnboundedSender<Request>,
        peer_state: &PeerState,
    ) -> Result<(), ConnectionError> {
        let (tip, tip_height, total_work) = {
            let rotxn = ctxt.env.read_txn()?;
            let tip = ctxt.state.get_tip(&rotxn)?;
            let tip_height = ctxt.state.get_height(&rotxn)?;
            let total_work = match ctxt.archive.try_get_header(&rotxn, tip)? {
                None => None,
                Some(header)
                    if header.prev_main_hash
                        == bitcoin::BlockHash::all_zeros() =>
                {
                    None
                }
                Some(header) => Some(
                    ctxt.archive
                        .get_total_work(&rotxn, header.prev_main_hash)?,
                ),
            };
            (tip, tip_height, total_work)
        };
        let peer_height = peer_state.block_height;
        if peer_height > tip_height
            || (peer_height == tip_height && peer_state.total_work > total_work)
        {
            let header = {
                let rotxn = ctxt.env.read_txn()?;
                ctxt.archive.try_get_header(&rotxn, peer_state.tip)?
            };
            let Some(header) = header else {
                // Request headers
                let request = Request::GetHeaders {
                    // TODO: provide alternative start points
                    start: HashSet::new(),
                    end: peer_state.tip,
                    height: Some(peer_state.block_height),
                };
                forward_request_tx.unbounded_send(request)?;
                return Ok(());
            };
            // Check mainchain headers
            let prev_main_header = {
                let rotxn = ctxt.env.read_txn()?;
                ctxt.archive
                    .try_get_main_header(&rotxn, header.prev_main_hash)?
            };
            let Some(_prev_main_header) = prev_main_header else {
                let info = Info::NeedMainchainAncestors(header.hash());
                info_tx.unbounded_send(info)?;
                return Ok(());
            };
            // Check PoW
            let prev_main_total_work = {
                let rotxn = ctxt.env.read_txn()?;
                ctxt.archive.get_total_work(&rotxn, header.prev_main_hash)?
            };
            if Some(prev_main_total_work) != peer_state.total_work {
                let ban_reason = BanReason::IncorrectTotalWork {
                    block_hash: peer_state.tip,
                    total_work: peer_state.total_work,
                };
                return Err(ConnectionError::PeerBan(ban_reason));
            }
            // Verify BMM
            {
                let rotxn = ctxt.env.read_txn()?;
                match ctxt
                    .archive
                    .try_get_bmm_verification(&rotxn, peer_state.tip)?
                {
                    Some(true) => (),
                    Some(false) => {
                        let ban_reason =
                            BanReason::BmmVerificationFailed(peer_state.tip);
                        return Err(ConnectionError::PeerBan(ban_reason));
                    }
                    None => {
                        let info = Info::NeedBmmVerification(peer_state.tip);
                        info_tx.unbounded_send(info)?;
                        return Ok(());
                    }
                }
            };
            let missing_bodies: Vec<BlockHash> = {
                let rotxn = ctxt.env.read_txn()?;
                let common_ancestor = ctxt.archive.last_common_ancestor(
                    &rotxn,
                    tip,
                    peer_state.tip,
                )?;
                ctxt.archive.get_missing_bodies(
                    &rotxn,
                    peer_state.tip,
                    common_ancestor,
                )?
            };
            if missing_bodies.is_empty() {
                let info = Info::NewTipReady(peer_state.tip);
                info_tx.unbounded_send(info)?;
            } else {
                // Request missing bodies
                missing_bodies.into_iter().try_for_each(|block_hash| {
                    let request = Request::GetBlock { block_hash };
                    forward_request_tx.unbounded_send(request)
                })?;
            }
        }
        Ok(())
    }

    async fn handle_get_block(
        ctxt: &ConnectionContext,
        response_tx: SendStream,
        block_hash: BlockHash,
    ) -> Result<(), ConnectionError> {
        let (header, body) = {
            let rotxn = ctxt.env.read_txn()?;
            let header = ctxt.archive.try_get_header(&rotxn, block_hash)?;
            let body = ctxt.archive.try_get_body(&rotxn, block_hash)?;
            (header, body)
        };
        let resp = match (header, body) {
            (Some(header), Some(body)) => Response::Block { header, body },
            (_, _) => Response::NoBlock { block_hash },
        };
        Self::send_response(response_tx, resp).await
    }

    async fn handle_get_headers(
        ctxt: &ConnectionContext,
        response_tx: SendStream,
        mut start: HashSet<BlockHash>,
        end: BlockHash,
    ) -> Result<(), ConnectionError> {
        start.insert(BlockHash::default());
        let response = {
            let rotxn = ctxt.env.read_txn()?;
            if ctxt.archive.try_get_header(&rotxn, end)?.is_some() {
                let mut headers: Vec<Header> = ctxt
                    .archive
                    .ancestors(&rotxn, end)
                    .take_while(|block_hash| Ok(!start.contains(block_hash)))
                    .map(|block_hash| {
                        ctxt.archive.get_header(&rotxn, block_hash)
                    })
                    .collect()?;
                headers.reverse();
                Response::Headers(headers)
            } else {
                Response::NoHeader { block_hash: end }
            }
        };
        Self::send_response(response_tx, response).await
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
        info_tx: &mpsc::UnboundedSender<Info>,
        forward_request_tx: &mpsc::UnboundedSender<Request>,
        peer_state: &mut Option<PeerState>,
        response_tx: SendStream,
        request: Request,
    ) -> Result<(), ConnectionError> {
        match request {
            Request::Heartbeat(new_peer_state) => {
                let () = Self::handle_heartbeat(
                    ctxt,
                    info_tx,
                    forward_request_tx,
                    &new_peer_state,
                )
                .await?;
                *peer_state = Some(new_peer_state);
                Ok(())
            }
            Request::GetBlock { block_hash } => {
                Self::handle_get_block(ctxt, response_tx, block_hash).await
            }
            Request::GetHeaders {
                start,
                end,
                height: _,
            } => Self::handle_get_headers(ctxt, response_tx, start, end).await,
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
            Response(Result<Response, ConnectionError>, Request),
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
        let (response_tx, response_rx) = mpsc::unbounded();
        let response_stream =
            response_rx.map(|(resp, req)| Ok(MailboxItem::Response(resp, req)));
        let mut mailbox_stream = stream::select_all([
            forward_request_stream.boxed(),
            heartbeat_stream.boxed(),
            request_stream.boxed(),
            response_stream.boxed(),
        ]);
        // spawn child tasks on a JoinSet so that they are dropped alongside this task
        let mut task_set: JoinSet<()> = JoinSet::new();
        // Do not repeat requests
        let mut pending_request_hashes = HashSet::<Hash>::new();
        while let Some(mailbox_item) = mailbox_stream.try_next().await? {
            match mailbox_item {
                MailboxItem::ForwardRequest(request) => {
                    let request_hash = hash(&request);
                    if !pending_request_hashes.insert(request_hash) {
                        continue;
                    }
                    task_set.spawn({
                        let connection = self.connection.clone();
                        let response_tx = response_tx.clone();
                        async move {
                            Self::send_request(
                                &connection,
                                &response_tx,
                                request,
                            )
                            .await
                        }
                    });
                }
                MailboxItem::Heartbeat => {
                    let (tip, tip_height, total_work) = {
                        let rotxn = self.ctxt.env.read_txn()?;
                        let tip = self.ctxt.state.get_tip(&rotxn)?;
                        let tip_height = self.ctxt.state.get_height(&rotxn)?;
                        let total_work = match self
                            .ctxt
                            .archive
                            .try_get_header(&rotxn, tip)?
                        {
                            None => None,
                            Some(header)
                                if header.prev_main_hash
                                    == bitcoin::BlockHash::all_zeros() =>
                            {
                                None
                            }
                            Some(header) => {
                                Some(self.ctxt.archive.get_total_work(
                                    &rotxn,
                                    header.prev_main_hash,
                                )?)
                            }
                        };
                        (tip, tip_height, total_work)
                    };
                    let heartbeat_msg = Request::Heartbeat(PeerState {
                        block_height: tip_height,
                        tip,
                        total_work,
                    });
                    task_set.spawn({
                        let connection = self.connection.clone();
                        let response_tx = response_tx.clone();
                        async move {
                            Self::send_request(
                                &connection,
                                &response_tx,
                                heartbeat_msg,
                            )
                            .await;
                        }
                    });
                }
                MailboxItem::Request((request, response_tx)) => {
                    let () = Self::handle_request(
                        &self.ctxt,
                        &self.info_tx,
                        &self.forward_request_tx,
                        &mut self.peer_state,
                        response_tx,
                        request,
                    )
                    .await?;
                }
                MailboxItem::Response(resp, req) => {
                    let request_hash = hash(&req);
                    pending_request_hashes.remove(&request_hash);
                    let info =
                        resp.map(|resp| Info::Response(resp, req)).into();
                    if self.info_tx.unbounded_send(info).is_err() {
                        tracing::error!("Failed to send response info")
                    };
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
        let forward_request_tx = forward_request_tx.clone();
        move || async move {
            let connection_task = ConnectionTask {
                connection,
                ctxt,
                info_tx,
                peer_state: None,
                forward_request_tx,
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
        let forward_request_tx = forward_request_tx.clone();
        move || async move {
            let connection = Connection::new(&endpoint, addr).await?;
            let connection_task = ConnectionTask {
                connection,
                ctxt,
                info_tx,
                peer_state: None,
                forward_request_tx,
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
