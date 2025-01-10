use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    fmt::Debug,
    net::SocketAddr,
};

use bitcoin::{self, hashes::Hash as _, Work};
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
    types::{
        hash, proto::mainchain, AuthorizedTransaction, BlockHash, BmmResult,
        Body, Hash, Header, Tip, Txid,
    },
};

#[derive(Debug, Error)]
pub enum BanReason {
    #[error(
        "BMM verification failed for block hash {} at {}",
        .0.block_hash,
        .0.main_block_hash
    )]
    BmmVerificationFailed(Tip),
    #[error(
        "Incorrect total work for block {} at {}: {total_work:?}",
        tip.block_hash,
        tip.main_block_hash
    )]
    IncorrectTotalWork { tip: Tip, total_work: Option<Work> },
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
    #[error("connect error")]
    Connect(#[from] quinn::ConnectError),
    #[error("connection error")]
    Connection(#[from] quinn::ConnectionError),
    #[error("Heartbeat timeout")]
    HeartbeatTimeout,
    #[error("heed error")]
    Heed(#[from] heed::Error),
    #[error("missing peer state for id {0}")]
    MissingPeerState(PeerStateId),
    #[error("peer should be banned; {0}")]
    PeerBan(#[from] BanReason),
    #[error("read to end error")]
    ReadToEnd(#[from] quinn::ReadToEndError),
    #[error("send datagram error")]
    SendDatagram(#[from] quinn::SendDatagramError),
    #[error("send internal message error")]
    SendInternalMessage,
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

fn borsh_serialize_option_work<W>(
    work: &Option<Work>,
    writer: &mut W,
) -> borsh::io::Result<()>
where
    W: borsh::io::Write,
{
    #[derive(BorshSerialize)]
    struct BorshWrapper(#[borsh(serialize_with = "borsh_serialize_work")] Work);
    borsh::BorshSerialize::serialize(&work.map(BorshWrapper), writer)
}

#[derive(
    BorshSerialize, Clone, Copy, Debug, Default, Deserialize, Serialize,
)]
pub struct PeerState {
    block_height: u32,
    tip: Tip,
    #[borsh(serialize_with = "borsh_serialize_option_work")]
    total_work: Option<Work>,
}

/// Unique identifier for a peer state
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct PeerStateId(Hash);

impl From<&PeerState> for PeerStateId {
    fn from(peer_state: &PeerState) -> Self {
        Self(hash(peer_state))
    }
}

impl std::fmt::Display for PeerStateId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
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
        /// Mainchain descendant tip that we are requesting the block to reach.
        /// Only relevant for the requester, so serialization is skipped
        #[borsh(skip)]
        #[serde(skip)]
        descendant_tip: Option<Tip>,
        /// Ancestor block. If no bodies are missing between `descendant_tip`
        /// and `ancestor`, then `descendant_tip` is ready to apply.
        /// Only relevant for the requester, so serialization is skipped
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
        send.finish()?;
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
    /// Push an internal message from connection task / net task / node
    internal_message_tx: mpsc::UnboundedSender<InternalMessage>,
    /// Receive an internal message from connection task / net task / node
    internal_message_rx: mpsc::UnboundedReceiver<InternalMessage>,
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

    /// * Request any missing mainchain headers
    /// * Check claimed work
    /// * Request BMM commitments if necessary
    /// * Check that BMM commitment matches peer tip
    /// * Check if peer tip is better, requesting headers if necessary
    /// * If peer tip is better:
    ///   * request headers if missing
    ///   * verify BMM
    ///   * request missing bodies
    ///   * notify net task / node that new tip is ready
    async fn handle_peer_state(
        ctxt: &ConnectionContext,
        info_tx: &mpsc::UnboundedSender<Info>,
        internal_message_tx: &mpsc::UnboundedSender<InternalMessage>,
        peer_state: &PeerState,
    ) -> Result<(), ConnectionError> {
        if peer_state.tip.main_block_hash == bitcoin::BlockHash::all_zeros() {
            // Nothing to do in this case
            return Ok(());
        }
        let Some(peer_total_work) = peer_state.total_work else {
            // Peer should send total work if they send a main tip
            let ban_reason = BanReason::IncorrectTotalWork {
                tip: peer_state.tip,
                total_work: None,
            };
            return Err(ConnectionError::PeerBan(ban_reason));
        };
        let (tip, tip_height, total_work) = {
            let rotxn = ctxt.env.read_txn()?;
            let tip = ctxt.state.get_tip(&rotxn)?;
            let tip_height = ctxt.state.get_height(&rotxn)?;
            let (bmm_verification, total_work) = if tip == BlockHash::default()
            {
                (bitcoin::BlockHash::all_zeros(), None)
            } else {
                let bmm_verification =
                    ctxt.archive.get_best_main_verification(&rotxn, tip)?;
                let work =
                    ctxt.archive.get_total_work(&rotxn, bmm_verification)?;
                (bmm_verification, Some(work))
            };
            let tip = Tip {
                block_hash: tip,
                main_block_hash: bmm_verification,
            };
            (tip, tip_height, total_work)
        };
        // Check claimed work and request mainchain headers and BMM commitments
        // if necessary
        {
            let rotxn = ctxt.env.read_txn()?;
            match ctxt.archive.try_get_main_header_info(
                &rotxn,
                peer_state.tip.main_block_hash,
            )? {
                None => {
                    let info = Info::NeedMainchainAncestors {
                        main_hash: peer_state.tip.main_block_hash,
                        peer_state_id: peer_state.into(),
                    };
                    info_tx.unbounded_send(info)?;
                    return Ok(());
                }
                Some(_main_header_info) => {
                    let computed_total_work = ctxt.archive.get_total_work(
                        &rotxn,
                        peer_state.tip.main_block_hash,
                    )?;
                    if peer_total_work != computed_total_work {
                        let ban_reason = BanReason::IncorrectTotalWork {
                            tip: peer_state.tip,
                            total_work: Some(peer_total_work),
                        };
                        return Err(ConnectionError::PeerBan(ban_reason));
                    }
                    if peer_state.tip.block_hash == BlockHash::default() {
                        // Nothing else to do in this case
                        return Ok(());
                    }
                    let Some(bmm_commitment) =
                        ctxt.archive.try_get_main_bmm_commitment(
                            &rotxn,
                            peer_state.tip.main_block_hash,
                        )?
                    else {
                        let info = Info::NeedBmmVerification {
                            main_hash: peer_state.tip.main_block_hash,
                            peer_state_id: peer_state.into(),
                        };
                        info_tx.unbounded_send(info)?;
                        return Ok(());
                    };
                    if bmm_commitment != Some(peer_state.tip.block_hash) {
                        let ban_reason =
                            BanReason::BmmVerificationFailed(peer_state.tip);
                        return Err(ConnectionError::PeerBan(ban_reason));
                    }
                }
            }
        }
        // Check if the peer tip is better, requesting headers if necessary
        match (
            total_work.cmp(&Some(peer_total_work)),
            tip_height.cmp(&peer_state.block_height),
        ) {
            (Ordering::Less | Ordering::Equal, Ordering::Less) => {
                // No tip ancestor can have greater height,
                // so peer tip is better.
                // Request headers if necessary
                let rotxn = ctxt.env.read_txn()?;
                if ctxt
                    .archive
                    .try_get_header(&rotxn, peer_state.tip.block_hash)?
                    .is_none()
                {
                    let start = HashSet::from_iter(
                        ctxt.archive
                            .get_block_locator(&rotxn, tip.block_hash)?,
                    );
                    let request = Request::GetHeaders {
                        start,
                        end: peer_state.tip.block_hash,
                        height: Some(peer_state.block_height),
                        peer_state_id: Some(peer_state.into()),
                    };
                    internal_message_tx.unbounded_send(request.into())?;
                    return Ok(());
                }
            }
            (Ordering::Equal | Ordering::Greater, Ordering::Greater) => {
                // No peer tip ancestor can have greater height,
                // so tip is better.
                // Nothing to do in this case
                return Ok(());
            }
            (Ordering::Less, Ordering::Equal) => {
                // Within the same mainchain lineage, prefer lower work
                // Otherwise, prefer tip with greater work
                let rotxn = ctxt.env.read_txn()?;
                if ctxt.archive.shared_mainchain_lineage(
                    &rotxn,
                    tip.main_block_hash,
                    peer_state.tip.main_block_hash,
                )? {
                    // Nothing to do in this case
                    return Ok(());
                }
                // Request headers if necessary
                if ctxt
                    .archive
                    .try_get_header(&rotxn, peer_state.tip.block_hash)?
                    .is_none()
                {
                    let start = HashSet::from_iter(
                        ctxt.archive
                            .get_block_locator(&rotxn, tip.block_hash)?,
                    );
                    let request = Request::GetHeaders {
                        start,
                        end: peer_state.tip.block_hash,
                        height: Some(peer_state.block_height),
                        peer_state_id: Some(peer_state.into()),
                    };
                    internal_message_tx.unbounded_send(request.into())?;
                    return Ok(());
                }
            }
            (Ordering::Greater, Ordering::Equal) => {
                // Within the same mainchain lineage, prefer lower work
                // Otherwise, prefer tip with greater work
                let rotxn = ctxt.env.read_txn()?;
                if !ctxt.archive.shared_mainchain_lineage(
                    &rotxn,
                    tip.main_block_hash,
                    peer_state.tip.main_block_hash,
                )? {
                    // Nothing to do in this case
                    return Ok(());
                }
                // Request headers if necessary
                if ctxt
                    .archive
                    .try_get_header(&rotxn, peer_state.tip.block_hash)?
                    .is_none()
                {
                    let start = HashSet::from_iter(
                        ctxt.archive
                            .get_block_locator(&rotxn, tip.block_hash)?,
                    );
                    let request = Request::GetHeaders {
                        start,
                        end: peer_state.tip.block_hash,
                        height: Some(peer_state.block_height),
                        peer_state_id: Some(peer_state.into()),
                    };
                    internal_message_tx.unbounded_send(request.into())?;
                    return Ok(());
                }
            }
            (Ordering::Less, Ordering::Greater) => {
                // Need to check if tip ancestor before common
                // mainchain ancestor had greater or equal height
                let rotxn = ctxt.env.read_txn()?;
                let main_ancestor = ctxt.archive.last_common_main_ancestor(
                    &rotxn,
                    tip.main_block_hash,
                    peer_state.tip.main_block_hash,
                )?;
                let tip_ancestor_height = ctxt
                    .archive
                    .ancestors(&rotxn, tip.block_hash)
                    .find_map(|tip_ancestor| {
                        if tip_ancestor == BlockHash::default() {
                            return Ok(Some(0));
                        }
                        let header =
                            ctxt.archive.get_header(&rotxn, tip_ancestor)?;
                        if !ctxt.archive.is_main_descendant(
                            &rotxn,
                            header.prev_main_hash,
                            main_ancestor,
                        )? {
                            return Ok(None);
                        }
                        if header.prev_main_hash == main_ancestor {
                            return Ok(None);
                        }
                        ctxt.archive.get_height(&rotxn, tip_ancestor).map(Some)
                    })?
                    .unwrap();
                if tip_ancestor_height >= peer_state.block_height {
                    // Nothing to do in this case
                    return Ok(());
                }
                // Request headers if necessary
                if ctxt
                    .archive
                    .try_get_header(&rotxn, peer_state.tip.block_hash)?
                    .is_none()
                {
                    let start = HashSet::from_iter(
                        ctxt.archive
                            .get_block_locator(&rotxn, tip.block_hash)?,
                    );
                    let request = Request::GetHeaders {
                        start,
                        end: peer_state.tip.block_hash,
                        height: Some(peer_state.block_height),
                        peer_state_id: Some(peer_state.into()),
                    };
                    internal_message_tx.unbounded_send(request.into())?;
                    return Ok(());
                }
            }
            (Ordering::Greater, Ordering::Less) => {
                // Need to check if peer's tip ancestor before common
                // mainchain ancestor had greater or equal height
                let rotxn = ctxt.env.read_txn()?;
                if ctxt
                    .archive
                    .try_get_header(&rotxn, peer_state.tip.block_hash)?
                    .is_none()
                {
                    let start = HashSet::from_iter(
                        ctxt.archive
                            .get_block_locator(&rotxn, tip.block_hash)?,
                    );
                    let request = Request::GetHeaders {
                        start,
                        end: peer_state.tip.block_hash,
                        height: Some(peer_state.block_height),
                        peer_state_id: Some(peer_state.into()),
                    };
                    internal_message_tx.unbounded_send(request.into())?;
                    return Ok(());
                }
                let main_ancestor = ctxt.archive.last_common_main_ancestor(
                    &rotxn,
                    tip.main_block_hash,
                    peer_state.tip.main_block_hash,
                )?;
                let peer_tip_ancestor_height = ctxt
                    .archive
                    .ancestors(&rotxn, peer_state.tip.block_hash)
                    .find_map(|peer_tip_ancestor| {
                        if peer_tip_ancestor == BlockHash::default() {
                            return Ok(Some(0));
                        }
                        let header = ctxt
                            .archive
                            .get_header(&rotxn, peer_tip_ancestor)?;
                        if !ctxt.archive.is_main_descendant(
                            &rotxn,
                            header.prev_main_hash,
                            main_ancestor,
                        )? {
                            return Ok(None);
                        }
                        if header.prev_main_hash == main_ancestor {
                            return Ok(None);
                        }
                        ctxt.archive
                            .get_height(&rotxn, peer_tip_ancestor)
                            .map(Some)
                    })?
                    .unwrap();
                if peer_tip_ancestor_height < tip_height {
                    // Nothing to do in this case
                    return Ok(());
                }
            }
            (Ordering::Equal, Ordering::Equal) => {
                // If the peer tip is the same as the tip, nothing to do
                if peer_state.tip.block_hash == tip.block_hash {
                    return Ok(());
                }
                // Need to compare tip ancestor and peer's tip ancestor
                // before common mainchain ancestor
                let rotxn = ctxt.env.read_txn()?;
                if ctxt
                    .archive
                    .try_get_header(&rotxn, peer_state.tip.block_hash)?
                    .is_none()
                {
                    let start = HashSet::from_iter(
                        ctxt.archive
                            .get_block_locator(&rotxn, tip.block_hash)?,
                    );
                    let request = Request::GetHeaders {
                        start,
                        end: peer_state.tip.block_hash,
                        height: Some(peer_state.block_height),
                        peer_state_id: Some(peer_state.into()),
                    };
                    internal_message_tx.unbounded_send(request.into())?;
                    return Ok(());
                }
                let main_ancestor = ctxt.archive.last_common_main_ancestor(
                    &rotxn,
                    tip.main_block_hash,
                    peer_state.tip.main_block_hash,
                )?;
                let main_ancestor_height =
                    ctxt.archive.get_main_height(&rotxn, main_ancestor)?;
                let (tip_ancestor_height, tip_ancestor_work) = ctxt
                    .archive
                    .ancestors(&rotxn, tip.block_hash)
                    .find_map(|tip_ancestor| {
                        if tip_ancestor == BlockHash::default() {
                            return Ok(Some((0, None)));
                        }
                        let header =
                            ctxt.archive.get_header(&rotxn, tip_ancestor)?;
                        if !ctxt.archive.is_main_descendant(
                            &rotxn,
                            header.prev_main_hash,
                            main_ancestor,
                        )? {
                            return Ok(None);
                        }
                        if header.prev_main_hash == main_ancestor {
                            return Ok(None);
                        }
                        let height =
                            ctxt.archive.get_height(&rotxn, tip_ancestor)?;
                        // Find mainchain block hash to get total work
                        let main_block = {
                            let prev_height = ctxt.archive.get_main_height(
                                &rotxn,
                                header.prev_main_hash,
                            )?;
                            let height = prev_height + 1;
                            ctxt.archive.get_nth_main_ancestor(
                                &rotxn,
                                main_ancestor,
                                main_ancestor_height - height,
                            )?
                        };
                        let work =
                            ctxt.archive.get_total_work(&rotxn, main_block)?;
                        Ok(Some((height, Some(work))))
                    })?
                    .unwrap();
                let (peer_tip_ancestor_height, peer_tip_ancestor_work) = ctxt
                    .archive
                    .ancestors(&rotxn, peer_state.tip.block_hash)
                    .find_map(|peer_tip_ancestor| {
                        if peer_tip_ancestor == BlockHash::default() {
                            return Ok(Some((0, None)));
                        }
                        let header = ctxt
                            .archive
                            .get_header(&rotxn, peer_tip_ancestor)?;
                        if !ctxt.archive.is_main_descendant(
                            &rotxn,
                            header.prev_main_hash,
                            main_ancestor,
                        )? {
                            return Ok(None);
                        }
                        if header.prev_main_hash == main_ancestor {
                            return Ok(None);
                        }
                        let height = ctxt
                            .archive
                            .get_height(&rotxn, peer_tip_ancestor)?;
                        // Find mainchain block hash to get total work
                        let main_block = {
                            let prev_height = ctxt.archive.get_main_height(
                                &rotxn,
                                header.prev_main_hash,
                            )?;
                            let height = prev_height + 1;
                            ctxt.archive.get_nth_main_ancestor(
                                &rotxn,
                                main_ancestor,
                                main_ancestor_height - height,
                            )?
                        };
                        let work =
                            ctxt.archive.get_total_work(&rotxn, main_block)?;
                        Ok(Some((height, Some(work))))
                    })?
                    .unwrap();
                match (
                    tip_ancestor_work.cmp(&peer_tip_ancestor_work),
                    tip_ancestor_height.cmp(&peer_tip_ancestor_height),
                ) {
                    (Ordering::Less | Ordering::Equal, Ordering::Equal)
                    | (_, Ordering::Greater) => {
                        // Peer tip is not better, nothing to do
                        return Ok(());
                    }
                    (Ordering::Greater, Ordering::Equal)
                    | (_, Ordering::Less) => {
                        // Peer tip is better
                    }
                }
            }
        }
        // Check BMM now that headers are available
        {
            let rotxn = ctxt.env.read_txn()?;
            let Some(BmmResult::Verified) = ctxt.archive.try_get_bmm_result(
                &rotxn,
                peer_state.tip.block_hash,
                peer_state.tip.main_block_hash,
            )?
            else {
                let ban_reason =
                    BanReason::BmmVerificationFailed(peer_state.tip);
                return Err(ConnectionError::PeerBan(ban_reason));
            };
        }
        // Request missing bodies, or notify that a new tip is ready
        let (common_ancestor, missing_bodies): (BlockHash, Vec<BlockHash>) = {
            let rotxn = ctxt.env.read_txn()?;
            let common_ancestor = ctxt.archive.last_common_ancestor(
                &rotxn,
                tip.block_hash,
                peer_state.tip.block_hash,
            )?;
            let missing_bodies = ctxt.archive.get_missing_bodies(
                &rotxn,
                peer_state.tip.block_hash,
                common_ancestor,
            )?;
            (common_ancestor, missing_bodies)
        };
        if missing_bodies.is_empty() {
            let info = Info::NewTipReady(peer_state.tip);
            info_tx.unbounded_send(info)?;
        } else {
            // Request missing bodies
            missing_bodies.into_iter().try_for_each(|block_hash| {
                let request = Request::GetBlock {
                    block_hash,
                    descendant_tip: Some(peer_state.tip),
                    peer_state_id: Some(peer_state.into()),
                    ancestor: Some(common_ancestor),
                };
                internal_message_tx.unbounded_send(request.into())
            })?;
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
        internal_message_tx: &mpsc::UnboundedSender<InternalMessage>,
        peer_state: &mut Option<PeerStateId>,
        // Map associating peer state hashes to peer state
        peer_states: &mut HashMap<PeerStateId, PeerState>,
        response_tx: SendStream,
        request: Request,
    ) -> Result<(), ConnectionError> {
        match request {
            Request::Heartbeat(new_peer_state) => {
                let new_peer_state_id = (&new_peer_state).into();
                peer_states.insert(new_peer_state_id, new_peer_state);
                if *peer_state != Some(new_peer_state_id) {
                    let () = Self::handle_peer_state(
                        ctxt,
                        info_tx,
                        internal_message_tx,
                        &new_peer_state,
                    )
                    .await?;
                    *peer_state = Some(new_peer_state_id);
                }
                Ok(())
            }
            Request::GetBlock {
                block_hash,
                descendant_tip: _,
                ancestor: _,
                peer_state_id: _,
            } => Self::handle_get_block(ctxt, response_tx, block_hash).await,
            Request::GetHeaders {
                start,
                end,
                height: _,
                peer_state_id: _,
            } => Self::handle_get_headers(ctxt, response_tx, start, end).await,
            Request::PushTransaction { transaction } => {
                Self::handle_push_tx(ctxt, info_tx, response_tx, transaction)
                    .await
            }
        }
    }

    async fn run(self) -> Result<(), ConnectionError> {
        enum MailboxItem {
            /// Internal messages from the connection task / net task / node
            InternalMessage(InternalMessage),
            /// Signals that a heartbeat message should be sent to the peer
            Heartbeat,
            Request((Request, SendStream)),
            Response(Result<Response, ConnectionError>, Request),
        }
        let internal_message_stream = self
            .internal_message_rx
            .map(|msg| Ok(MailboxItem::InternalMessage(msg)));
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
            internal_message_stream.boxed(),
            heartbeat_stream.boxed(),
            request_stream.boxed(),
            response_stream.boxed(),
        ]);
        // spawn child tasks on a JoinSet so that they are dropped alongside this task
        let mut task_set: JoinSet<()> = JoinSet::new();
        // current peer state
        let mut peer_state = Option::<PeerStateId>::None;
        // known peer states
        let mut peer_states = HashMap::<PeerStateId, PeerState>::new();
        // Do not repeat requests
        let mut pending_request_hashes = HashSet::<Hash>::new();
        while let Some(mailbox_item) = mailbox_stream.try_next().await? {
            match mailbox_item {
                MailboxItem::InternalMessage(
                    InternalMessage::ForwardRequest(request),
                ) => {
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
                MailboxItem::InternalMessage(
                    InternalMessage::BmmVerification { res, peer_state_id },
                ) => {
                    if let Err(block_not_found) = res {
                        tracing::warn!("{block_not_found}");
                        continue;
                    }
                    let Some(peer_state) = peer_states.get(&peer_state_id)
                    else {
                        return Err(ConnectionError::MissingPeerState(
                            peer_state_id,
                        ));
                    };
                    let () = Self::handle_peer_state(
                        &self.ctxt,
                        &self.info_tx,
                        &self.internal_message_tx,
                        peer_state,
                    )
                    .await?;
                }
                MailboxItem::InternalMessage(
                    InternalMessage::BmmVerificationError(err),
                ) => {
                    tracing::error!(
                        "Error attempting BMM verification: {err:#}"
                    );
                }
                MailboxItem::InternalMessage(
                    InternalMessage::MainchainAncestorsError(err),
                ) => {
                    tracing::error!(
                        "Error fetching mainchain ancestors: {err:#}"
                    );
                }
                MailboxItem::InternalMessage(
                    InternalMessage::MainchainAncestors(peer_state_id)
                    | InternalMessage::Headers(peer_state_id)
                    | InternalMessage::BodiesAvailable(peer_state_id),
                ) => {
                    let Some(peer_state) = peer_states.get(&peer_state_id)
                    else {
                        return Err(ConnectionError::MissingPeerState(
                            peer_state_id,
                        ));
                    };
                    let () = Self::handle_peer_state(
                        &self.ctxt,
                        &self.info_tx,
                        &self.internal_message_tx,
                        peer_state,
                    )
                    .await?;
                }
                MailboxItem::Heartbeat => {
                    let (tip, tip_height, total_work) = {
                        let rotxn = self.ctxt.env.read_txn()?;
                        let tip = self.ctxt.state.get_tip(&rotxn)?;
                        let tip_height = self.ctxt.state.get_height(&rotxn)?;
                        let (bmm_verification, total_work) =
                            if tip == BlockHash::default() {
                                (bitcoin::BlockHash::all_zeros(), None)
                            } else {
                                let bmm_verification = self
                                    .ctxt
                                    .archive
                                    .get_best_main_verification(&rotxn, tip)?;
                                let work = self
                                    .ctxt
                                    .archive
                                    .get_total_work(&rotxn, bmm_verification)?;
                                (bmm_verification, Some(work))
                            };
                        let tip = Tip {
                            block_hash: tip,
                            main_block_hash: bmm_verification,
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
                        &self.internal_message_tx,
                        &mut peer_state,
                        &mut peer_states,
                        response_tx,
                        request,
                    )
                    .await?;
                }
                MailboxItem::Response(resp, req) => {
                    let request_hash = hash(&req);
                    pending_request_hashes.remove(&request_hash);
                    let info = resp
                        .map(|resp| Info::Response(Box::new((resp, req))))
                        .into();
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
    /// Push messages from connection task / net task / node
    pub internal_message_tx: mpsc::UnboundedSender<InternalMessage>,
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
    let (internal_message_tx, internal_message_rx) = mpsc::unbounded();
    let (info_tx, info_rx) = mpsc::unbounded();
    let connection_task = {
        let info_tx = info_tx.clone();
        let internal_message_tx = internal_message_tx.clone();
        move || async move {
            let connection_task = ConnectionTask {
                connection,
                ctxt,
                info_tx,
                internal_message_tx,
                internal_message_rx,
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
        internal_message_tx,
    };
    (connection_handle, info_rx)
}

pub fn connect(
    endpoint: Endpoint,
    addr: SocketAddr,
    ctxt: ConnectionContext,
) -> (ConnectionHandle, mpsc::UnboundedReceiver<Info>) {
    let (internal_message_tx, internal_message_rx) = mpsc::unbounded();
    let (info_tx, info_rx) = mpsc::unbounded();
    let connection_task = {
        let info_tx = info_tx.clone();
        let internal_message_tx = internal_message_tx.clone();
        move || async move {
            let connection = Connection::new(&endpoint, addr).await?;
            let connection_task = ConnectionTask {
                connection,
                ctxt,
                info_tx,
                internal_message_tx,
                internal_message_rx,
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
        internal_message_tx,
    };
    (connection_handle, info_rx)
}
