use thiserror::Error;

use crate::net::peer::PeerStateId;

pub(in crate::net::peer) mod connection {
    use thiserror::Error;

    #[derive(Debug, Error)]
    pub enum Send {
        #[error("bincode error")]
        Bincode(#[from] bincode::Error),
        #[error("connection already closed")]
        ClosedStream(#[from] quinn::ClosedStream),
        #[error("connection error")]
        Connection(#[from] quinn::ConnectionError),
        #[error("write error ({stream_id})")]
        Write {
            stream_id: quinn::StreamId,
            source: quinn::WriteError,
        },
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

    #[derive(Debug, Error)]
    pub enum SendMessage {
        #[error(transparent)]
        Heartbeat(#[from] SendHeartbeat),
        #[error(transparent)]
        Request(#[from] SendRequest),
    }

    #[derive(Debug, Error)]
    pub enum Receive {
        #[error("received incorrect magic: {}", hex::encode(.0))]
        BadMagic(crate::net::peer::message::MagicBytes),
        #[error("bincode error")]
        Bincode(#[from] bincode::Error),
        #[error("connection error")]
        Connection(#[from] quinn::ConnectionError),
        #[error("failed to read magic bytes")]
        ReadMagic(#[source] quinn::ReadExactError),
        #[error("read to end error")]
        ReadToEnd(#[from] quinn::ReadToEndError),
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

    #[derive(Debug, Error)]
    #[error("Failed to receive response from peer")]
    #[repr(transparent)]
    pub struct ReceiveResponse(#[source] Receive);

    impl<E> From<E> for ReceiveResponse
    where
        Receive: From<E>,
    {
        fn from(err: E) -> Self {
            Self(err.into())
        }
    }
}

pub(in crate::net::peer) mod channel_pool {
    use thiserror::Error;

    #[derive(Debug, Error)]
    pub enum Task {
        #[error("Send heartbeat task error")]
        Heartbeat(#[source] tokio::task::JoinError),
        #[error("Send request task error")]
        Request(#[source] tokio::task::JoinError),
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

    #[derive(Debug, Error)]
    #[error("Failed to spawn task to send heartbeat message: receiver dropped")]
    pub struct SpawnHeartbeatTask;

    #[derive(Debug, Error)]
    #[error("Failed to spawn task to send request message: receiver dropped")]
    pub struct SpawnRequestTask;

    #[derive(Debug, Error)]
    pub enum SpawnTask {
        #[error(transparent)]
        Heartbeat(#[from] SpawnHeartbeatTask),
        #[error(transparent)]
        Request(#[from] SpawnRequestTask),
    }

    #[derive(Debug, Error)]
    pub enum Error {
        #[error(transparent)]
        SendMessage(#[from] SendMessage),
        #[error(transparent)]
        SpawnTask(#[from] SpawnTask),
    }
}

pub(in crate::net::peer) mod request_queue {
    use thiserror::Error;

    #[derive(Debug, Error)]
    #[error("Failed to add heartbeat to send queue")]
    pub struct SendHeartbeat;

    #[derive(Debug, Error)]
    #[error("Failed to add request to send queue")]
    pub struct SendRequest;

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
}

pub(in crate::net::peer) mod blocking_task {
    use thiserror::Error;

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
}

pub(in crate::net::peer) mod forward_response {
    use thiserror::Error;

    #[derive(Debug, Error)]
    pub enum TaskError {
        #[error("archive error")]
        Archive(#[source] Box<crate::archive::Error>),
        #[error("bincode error")]
        Bincode(#[from] bincode::Error),
        #[error(transparent)]
        ReadTxn(#[from] sneed::env::error::ReadTxn),
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
}

pub mod mailbox {
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
