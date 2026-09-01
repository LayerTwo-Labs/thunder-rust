use thiserror::Error;

use crate::net::peer::PeerStateId;

/// Errors that are potentially recoverable by reconnecting.
/// Does not imply anything regarding error fatality.
pub(crate) trait Recoverable {
    fn may_reconnect(&self) -> bool;
}

impl Recoverable for quinn::ClosedStream {
    fn may_reconnect(&self) -> bool {
        true
    }
}

impl Recoverable for quinn::ConnectionError {
    fn may_reconnect(&self) -> bool {
        match self {
            Self::ApplicationClosed(_)
            | Self::CidsExhausted
            | Self::ConnectionClosed(_)
            | Self::LocallyClosed
            | Self::Reset
            | Self::TimedOut => true,
            Self::TransportError(_) | Self::VersionMismatch => false,
        }
    }
}

impl Recoverable for quinn::ReadError {
    fn may_reconnect(&self) -> bool {
        match self {
            Self::ClosedStream
            | Self::IllegalOrderedRead
            | Self::Reset(_)
            | Self::ZeroRttRejected => true,
            Self::ConnectionLost(err) => err.may_reconnect(),
        }
    }
}

impl Recoverable for quinn::ReadExactError {
    fn may_reconnect(&self) -> bool {
        match self {
            Self::FinishedEarly(_) => true,
            Self::ReadError(err) => err.may_reconnect(),
        }
    }
}

impl Recoverable for quinn::ReadToEndError {
    fn may_reconnect(&self) -> bool {
        match self {
            Self::Read(err) => err.may_reconnect(),
            Self::TooLong => true,
        }
    }
}

impl Recoverable for quinn::WriteError {
    fn may_reconnect(&self) -> bool {
        match self {
            Self::ClosedStream | Self::Stopped(_) | Self::ZeroRttRejected => {
                true
            }
            Self::ConnectionLost(err) => err.may_reconnect(),
        }
    }
}

pub(in crate::net::peer) mod connection {
    use thiserror::Error;

    use crate::net::peer::error::Recoverable;

    #[derive(Debug, Error)]
    pub enum Send {
        #[error("connection already closed")]
        ClosedStream(#[from] quinn::ClosedStream),
        #[error("connection error")]
        Connection(#[from] quinn::ConnectionError),
        #[error("failed to serialize message")]
        SerializeMessage(#[source] bincode::Error),
        #[error("write error ({stream_id})")]
        Write {
            stream_id: quinn::StreamId,
            source: quinn::WriteError,
        },
    }

    impl Recoverable for Send {
        fn may_reconnect(&self) -> bool {
            match self {
                Self::SerializeMessage(_) => true,
                Self::ClosedStream(err) => err.may_reconnect(),
                Self::Connection(err) => err.may_reconnect(),
                Self::Write {
                    stream_id: _,
                    source,
                } => source.may_reconnect(),
            }
        }
    }

    #[derive(Debug, Error)]
    #[error("Failed to send heartbeat")]
    #[repr(transparent)]
    pub struct SendHeartbeat(#[source] Send);

    impl<E> From<E> for SendHeartbeat
    where
        Send: From<E>,
    {
        fn from(err: E) -> Self {
            Self(err.into())
        }
    }

    impl Recoverable for SendHeartbeat {
        fn may_reconnect(&self) -> bool {
            self.0.may_reconnect()
        }
    }

    #[derive(Debug, Error)]
    #[error("Failed to send request")]
    #[repr(transparent)]
    pub struct SendRequest(#[source] Send);

    impl<E> From<E> for SendRequest
    where
        Send: From<E>,
    {
        fn from(err: E) -> Self {
            Self(err.into())
        }
    }

    impl Recoverable for SendRequest {
        fn may_reconnect(&self) -> bool {
            self.0.may_reconnect()
        }
    }

    #[derive(Debug, Error)]
    #[error("Failed to send response")]
    pub struct SendResponse(#[source] Send);

    impl<E> From<E> for SendResponse
    where
        Send: From<E>,
    {
        fn from(err: E) -> Self {
            Self(err.into())
        }
    }

    impl Recoverable for SendResponse {
        fn may_reconnect(&self) -> bool {
            self.0.may_reconnect()
        }
    }

    #[derive(Debug, Error)]
    pub enum SendMessage {
        #[error(transparent)]
        Heartbeat(#[from] SendHeartbeat),
        #[error(transparent)]
        Request(#[from] SendRequest),
    }

    impl Recoverable for SendMessage {
        fn may_reconnect(&self) -> bool {
            match self {
                Self::Heartbeat(err) => err.may_reconnect(),
                Self::Request(err) => err.may_reconnect(),
            }
        }
    }

    #[derive(Debug, Error)]
    pub enum Receive {
        #[error("received incorrect magic: {}", const_hex::encode(.0))]
        BadMagic(crate::net::peer::message::MagicBytes),
        #[error("connection error")]
        Connection(#[from] quinn::ConnectionError),
        #[error("failed to deserialize message")]
        DeserializeMessage(#[source] bincode::Error),
        #[error("failed to read magic bytes")]
        ReadMagic(#[source] quinn::ReadExactError),
        #[error("read to end error")]
        ReadToEnd(#[from] quinn::ReadToEndError),
        #[error("timed out waiting for response")]
        Timeout,
    }

    impl Recoverable for Receive {
        fn may_reconnect(&self) -> bool {
            match self {
                Self::BadMagic(_) | Self::DeserializeMessage(_) => false,
                Self::Timeout => true,
                Self::Connection(err) => err.may_reconnect(),
                Self::ReadMagic(err) => err.may_reconnect(),
                Self::ReadToEnd(err) => err.may_reconnect(),
            }
        }
    }

    #[derive(Debug, Error)]
    #[error("Failed to receive request from peer")]
    #[repr(transparent)]
    pub struct ReceiveRequest(#[source] Receive);

    impl<E> From<E> for ReceiveRequest
    where
        Receive: From<E>,
    {
        fn from(err: E) -> Self {
            Self(err.into())
        }
    }

    impl Receive {
        /// True when the peer sent another network's magic bytes.
        pub fn is_bad_magic(&self) -> bool {
            matches!(self, Self::BadMagic(_))
        }
    }

    impl ReceiveRequest {
        pub fn is_bad_magic(&self) -> bool {
            self.0.is_bad_magic()
        }
    }

    impl Recoverable for ReceiveRequest {
        fn may_reconnect(&self) -> bool {
            self.0.may_reconnect()
        }
    }

    #[derive(Debug, Error)]
    #[error("Failed to receive response from peer")]
    #[repr(transparent)]
    pub struct ReceiveResponse(#[source] Receive);

    impl ReceiveResponse {
        pub fn is_bad_magic(&self) -> bool {
            self.0.is_bad_magic()
        }
    }

    impl<E> From<E> for ReceiveResponse
    where
        Receive: From<E>,
    {
        fn from(err: E) -> Self {
            Self(err.into())
        }
    }

    impl Recoverable for ReceiveResponse {
        fn may_reconnect(&self) -> bool {
            self.0.may_reconnect()
        }
    }
}

pub(in crate::net::peer) mod channel_pool {
    use thiserror::Error;

    use crate::net::peer::error::Recoverable;

    #[derive(Debug, Error)]
    pub enum Task {
        #[error("Send heartbeat task error")]
        Heartbeat(#[source] tokio::task::JoinError),
        #[error("Send request task error")]
        Request(#[source] tokio::task::JoinError),
    }

    impl Recoverable for Task {
        fn may_reconnect(&self) -> bool {
            true
        }
    }

    #[allow(clippy::duplicated_attributes)]
    #[derive(transitive::Transitive, Debug, Error)]
    #[transitive(
        from(super::connection::SendHeartbeat, super::connection::SendMessage),
        from(super::connection::SendRequest, super::connection::SendMessage)
    )]
    pub enum SendMessage {
        #[error(transparent)]
        Connection(#[from] super::connection::SendMessage),
        #[error(transparent)]
        Task(#[from] Task),
    }

    impl Recoverable for SendMessage {
        fn may_reconnect(&self) -> bool {
            match self {
                Self::Connection(err) => err.may_reconnect(),
                Self::Task(err) => err.may_reconnect(),
            }
        }
    }

    #[derive(Debug, Error)]
    #[error("Failed to spawn task to send heartbeat message: receiver dropped")]
    pub struct SpawnHeartbeatTask;

    impl Recoverable for SpawnHeartbeatTask {
        fn may_reconnect(&self) -> bool {
            true
        }
    }

    #[derive(Debug, Error)]
    #[error("Failed to spawn task to send request message: receiver dropped")]
    pub struct SpawnRequestTask;

    impl Recoverable for SpawnRequestTask {
        fn may_reconnect(&self) -> bool {
            true
        }
    }

    #[derive(Debug, Error)]
    pub enum SpawnTask {
        #[error(transparent)]
        Heartbeat(#[from] SpawnHeartbeatTask),
        #[error(transparent)]
        Request(#[from] SpawnRequestTask),
    }

    impl Recoverable for SpawnTask {
        fn may_reconnect(&self) -> bool {
            match self {
                Self::Heartbeat(err) => err.may_reconnect(),
                Self::Request(err) => err.may_reconnect(),
            }
        }
    }

    #[derive(Debug, Error)]
    pub enum Error {
        #[error(transparent)]
        SendMessage(#[from] SendMessage),
        #[error(transparent)]
        SpawnTask(#[from] SpawnTask),
    }

    impl Recoverable for Error {
        fn may_reconnect(&self) -> bool {
            match self {
                Self::SendMessage(err) => err.may_reconnect(),
                Self::SpawnTask(err) => err.may_reconnect(),
            }
        }
    }
}

pub(in crate::net::peer) mod request_queue {
    use thiserror::Error;

    use crate::net::peer::error::Recoverable;

    #[derive(Debug, Error)]
    #[error("Failed to add heartbeat to send queue")]
    pub struct SendHeartbeat;

    impl Recoverable for SendHeartbeat {
        fn may_reconnect(&self) -> bool {
            true
        }
    }

    #[derive(Debug, Error)]
    #[error("Failed to add request to send queue")]
    pub struct SendRequest;

    impl Recoverable for SendRequest {
        fn may_reconnect(&self) -> bool {
            true
        }
    }

    #[allow(clippy::duplicated_attributes)]
    #[derive(transitive::Transitive, Debug, Error)]
    #[transitive(
        from(super::channel_pool::SendMessage, super::channel_pool::Error),
        from(
            super::channel_pool::SpawnHeartbeatTask,
            super::channel_pool::SpawnTask
        ),
        from(
            super::channel_pool::SpawnRequestTask,
            super::channel_pool::SpawnTask
        ),
        from(super::channel_pool::SpawnTask, super::channel_pool::Error)
    )]
    pub enum Error {
        #[error(transparent)]
        ChannelPool(#[from] super::channel_pool::Error),
        #[error("Failed to push peer response")]
        PushPeerResponse,
    }

    impl Recoverable for Error {
        fn may_reconnect(&self) -> bool {
            match self {
                Self::ChannelPool(err) => err.may_reconnect(),
                Self::PushPeerResponse => true,
            }
        }
    }
}

pub(in crate::net::peer) mod blocking_task {
    use thiserror::Error;

    use crate::net::peer::error::Recoverable;

    #[derive(Debug, Error)]
    pub enum TaskError {
        #[error("archive error")]
        Archive(#[from] crate::archive::Error),
        #[error("peer should be banned; {0}")]
        PeerBan(#[from] crate::net::peer::BanReason),
        #[error(transparent)]
        ReadTxn(#[from] sneed::env::error::ReadTxn),
        #[error("Failed to push info message")]
        SendInfo,
        #[error(transparent)]
        SendRequest(#[from] super::request_queue::SendRequest),
        #[error("state error")]
        State(#[from] crate::state::Error),
    }

    impl Recoverable for TaskError {
        fn may_reconnect(&self) -> bool {
            match self {
                Self::Archive(_)
                | Self::ReadTxn(_)
                | Self::SendInfo
                | Self::State(_) => true,
                Self::PeerBan(_) => false,
                Self::SendRequest(err) => err.may_reconnect(),
            }
        }
    }

    #[derive(Debug, Error)]
    pub enum Error {
        #[error("Failed to execute blocking task to completion")]
        Join(#[from] tokio::task::JoinError),
        #[error(transparent)]
        Task(Box<TaskError>),
    }

    impl From<TaskError> for Error {
        fn from(err: TaskError) -> Self {
            Self::Task(Box::new(err))
        }
    }

    impl Recoverable for Error {
        fn may_reconnect(&self) -> bool {
            match self {
                Self::Join(_) => true,
                Self::Task(err) => err.may_reconnect(),
            }
        }
    }
}

pub(in crate::net::peer) mod forward_response {
    use thiserror::Error;

    use crate::net::peer::error::Recoverable;

    #[derive(Debug, Error)]
    pub enum TaskError {
        #[error("archive error")]
        Archive(#[source] Box<crate::archive::Error>),
        #[error("bincode error")]
        Bincode(#[from] bincode::Error),
        #[error(transparent)]
        ReadTxn(#[from] sneed::env::error::ReadTxn),
    }

    impl Recoverable for TaskError {
        fn may_reconnect(&self) -> bool {
            match self {
                Self::Archive(_) | Self::Bincode(_) | Self::ReadTxn(_) => true,
            }
        }
    }

    impl From<crate::archive::Error> for TaskError {
        fn from(err: crate::archive::Error) -> Self {
            Self::Archive(Box::new(err))
        }
    }

    #[derive(Debug, Error)]
    pub enum Error {
        #[error("Failed to execute task to completion")]
        Join(#[from] tokio::task::JoinError),
        #[error(transparent)]
        Task(#[from] TaskError),
    }

    impl Recoverable for Error {
        fn may_reconnect(&self) -> bool {
            match self {
                Self::Join(_) => true,
                Self::Task(err) => err.may_reconnect(),
            }
        }
    }
}

pub mod mailbox {
    use crate::net::peer::error::Recoverable;

    #[derive(thiserror::Error, Debug)]
    pub enum Error {
        #[error("Blocking task error")]
        BlockingTask(#[from] super::blocking_task::Error),
        #[error("Failed to generate response")]
        ForwardResponse(#[from] super::forward_response::Error),
        #[error("Heartbeat timeout")]
        HeartbeatTimeout,
        #[error("Failed to send response")]
        JoinSendResponse(#[source] tokio::task::JoinError),
        #[error(transparent)]
        ReceiveRequest(#[from] super::connection::ReceiveRequest),
        #[error(transparent)]
        RequestQueue(#[from] super::request_queue::Error),
        #[error(transparent)]
        SendResponse(#[from] super::connection::SendResponse),
    }

    impl Error {
        pub fn is_bad_magic(&self) -> bool {
            match self {
                Self::ReceiveRequest(err) => err.is_bad_magic(),
                _ => false,
            }
        }
    }

    impl Recoverable for Error {
        fn may_reconnect(&self) -> bool {
            match self {
                Self::HeartbeatTimeout | Self::JoinSendResponse(_) => true,
                Self::BlockingTask(err) => err.may_reconnect(),
                Self::ForwardResponse(err) => err.may_reconnect(),
                Self::ReceiveRequest(err) => err.may_reconnect(),
                Self::RequestQueue(err) => err.may_reconnect(),
                Self::SendResponse(err) => err.may_reconnect(),
            }
        }
    }
}

#[derive(Debug, Error)]
#[must_use]
pub enum Error {
    #[error("archive error")]
    Archive(#[from] crate::archive::Error),
    #[error("connection error")]
    Connection(#[from] quinn::ConnectionError),
    #[error("Database env error")]
    DbEnv(#[from] sneed::env::Error),
    #[error(transparent)]
    Mailbox(#[from] mailbox::Error),
    #[error("missing peer state for id {0}")]
    MissingPeerState(PeerStateId),
    #[error(transparent)]
    ReceiveResponse(#[from] connection::ReceiveResponse),
    #[error("Failed to push blocking task")]
    SendBlockingTask,
    #[error(transparent)]
    SendHeartbeat(#[from] request_queue::SendHeartbeat),
    #[error("Failed to push info message")]
    SendInfo,
    #[error(transparent)]
    SendRequest(#[from] request_queue::SendRequest),
    #[error(transparent)]
    SendResponse(#[from] connection::SendResponse),
    #[error("state error")]
    State(#[from] crate::state::Error),
}

impl Recoverable for Error {
    fn may_reconnect(&self) -> bool {
        match self {
            Self::Archive(_)
            | Self::DbEnv(_)
            | Self::MissingPeerState(_)
            | Self::SendBlockingTask
            | Self::SendInfo
            | Self::State(_) => true,
            Self::Connection(err) => err.may_reconnect(),
            Self::Mailbox(err) => err.may_reconnect(),
            Self::ReceiveResponse(err) => err.may_reconnect(),
            Self::SendHeartbeat(err) => err.may_reconnect(),
            Self::SendRequest(err) => err.may_reconnect(),
            Self::SendResponse(err) => err.may_reconnect(),
        }
    }
}

impl Error {
    /// True when the peer answered with another network's magic bytes. Such a
    /// peer runs a different chain, so it never becomes useful.
    pub fn is_bad_magic(&self) -> bool {
        match self {
            Self::Mailbox(err) => err.is_bad_magic(),
            Self::ReceiveResponse(err) => err.is_bad_magic(),
            _ => false,
        }
    }
}

#[cfg(test)]
mod test {
    use super::{Error, connection, mailbox};
    use crate::net::peer::message;

    const FOREIGN_MAGIC: message::MagicBytes = [0x85, 0x18, 0x95, 0x01];

    // A peer that answers a request with another network's magic must read as
    // bad magic through every wrapper the error passes.
    #[test]
    fn bad_magic_survives_the_request_path() {
        let inner = connection::Receive::BadMagic(FOREIGN_MAGIC);
        let err = Error::Mailbox(mailbox::Error::ReceiveRequest(
            connection::ReceiveRequest::from(inner),
        ));
        assert!(err.is_bad_magic());
    }

    #[test]
    fn bad_magic_survives_the_response_path() {
        let inner = connection::Receive::BadMagic(FOREIGN_MAGIC);
        let err =
            Error::ReceiveResponse(connection::ReceiveResponse::from(inner));
        assert!(err.is_bad_magic());
    }

    // A timeout says nothing about the peer's network, so the node keeps it.
    #[test]
    fn a_timeout_is_not_bad_magic() {
        let err = Error::ReceiveResponse(connection::ReceiveResponse::from(
            connection::Receive::Timeout,
        ));
        assert!(!err.is_bad_magic());
    }

    #[test]
    fn a_heartbeat_timeout_is_not_bad_magic() {
        let err = Error::Mailbox(mailbox::Error::HeartbeatTimeout);
        assert!(!err.is_bad_magic());
    }
}
