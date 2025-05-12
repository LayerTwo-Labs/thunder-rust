//! Task to communicate with mainchain node

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use bitcoin::{self, hashes::Hash as _};
use futures::{
    StreamExt,
    channel::{
        mpsc::{self, UnboundedReceiver, UnboundedSender},
        oneshot,
    },
};
use sneed::{EnvError, RwTxnError};
use thiserror::Error;
use tokio::{
    spawn,
    task::{self, JoinHandle},
};

use crate::{
    archive::{self, Archive},
    types::proto::{self, mainchain},
};

/// Request data from the mainchain node
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) enum Request {
    /// Request missing mainchain ancestor header/infos
    AncestorInfos(bitcoin::BlockHash),
}

/// Error included in a response
#[derive(Debug, Error)]
pub enum ResponseError {
    #[error("Archive error")]
    Archive(#[from] archive::Error),
    #[error("Database env error")]
    DbEnv(#[from] EnvError),
    #[error("Database write error")]
    DbWrite(#[from] sneed::rwtxn::Error),
    #[error("CUSF Mainchain proto error")]
    Mainchain(#[from] proto::Error),
}

/// Response indicating that a request has been fulfilled
#[derive(Debug)]
pub(super) enum Response {
    /// Response bool indicates if the requested header was available
    AncestorInfos(bitcoin::BlockHash, Result<bool, ResponseError>),
}

impl From<&Response> for Request {
    fn from(resp: &Response) -> Self {
        match resp {
            Response::AncestorInfos(block_hash, _) => {
                Request::AncestorInfos(*block_hash)
            }
        }
    }
}

#[derive(Debug, Error)]
enum Error {
    #[error("Send response error")]
    SendResponse(Response),
    #[error("Send response error (oneshot)")]
    SendResponseOneshot(Response),
}

struct MainchainTask<Transport = tonic::transport::Channel> {
    env: sneed::Env,
    archive: Archive,
    mainchain: proto::mainchain::ValidatorClient<Transport>,
    // receive a request, and optional oneshot sender to send the result to
    // instead of sending on `response_tx`
    request_rx: UnboundedReceiver<(Request, Option<oneshot::Sender<Response>>)>,
    response_tx: UnboundedSender<Response>,
}

impl<Transport> MainchainTask<Transport>
where
    Transport: proto::Transport,
{
    /// Request ancestor header info and block info from the mainchain node,
    /// including the specified header.
    /// Returns `false` if the specified block was not available.
    async fn request_ancestor_infos(
        env: &sneed::Env,
        archive: &Archive,
        cusf_mainchain: &mut proto::mainchain::ValidatorClient<Transport>,
        block_hash: bitcoin::BlockHash,
    ) -> Result<bool, ResponseError> {
        if block_hash == bitcoin::BlockHash::all_zeros() {
            return Ok(true);
        } else {
            let rotxn = env.read_txn().map_err(EnvError::from)?;
            if archive
                .try_get_main_header_info(&rotxn, &block_hash)?
                .is_some()
            {
                return Ok(true);
            }
        }
        let mut current_block_hash = block_hash;
        let mut current_height = None;
        let mut block_infos =
            Vec::<(mainchain::BlockHeaderInfo, mainchain::BlockInfo)>::new();
        tracing::debug!(%block_hash, "requesting ancestor headers/info");
        const LOG_PROGRESS_INTERVAL: Duration = Duration::from_secs(5);
        const BATCH_REQUEST_SIZE: u32 = 1000;
        let mut progress_logged = Instant::now();
        loop {
            if let Some(current_height) = current_height {
                let now = Instant::now();
                if now.duration_since(progress_logged) >= LOG_PROGRESS_INTERVAL
                {
                    progress_logged = now;
                    tracing::debug!(
                        %block_hash,
                        "requesting ancestor headers: {current_block_hash}({current_height} remaining)");
                }
                tracing::trace!(%block_hash, "requesting ancestor headers: {current_block_hash}({current_height})")
            }
            let Some(block_infos_resp) = cusf_mainchain
                .get_block_infos(current_block_hash, BATCH_REQUEST_SIZE - 1)
                .await?
            else {
                return Ok(false);
            };
            {
                let (current_header, _) = block_infos_resp.last();
                current_block_hash = current_header.prev_block_hash;
                current_height = current_header.height.checked_sub(1);
            }
            block_infos.extend(block_infos_resp);
            if current_block_hash == bitcoin::BlockHash::all_zeros() {
                break;
            } else {
                let rotxn = env.read_txn().map_err(EnvError::from)?;
                if archive
                    .try_get_main_header_info(&rotxn, &current_block_hash)?
                    .is_some()
                {
                    break;
                }
            }
        }
        block_infos.reverse();
        // Writing all headers during IBD can starve archive readers.
        tracing::trace!(%block_hash, "storing ancestor headers/info");
        task::block_in_place(|| {
            let mut rwtxn = env.write_txn().map_err(EnvError::from)?;
            for (header_info, block_info) in block_infos {
                let () =
                    archive.put_main_header_info(&mut rwtxn, &header_info)?;
                let () = archive.put_main_block_info(
                    &mut rwtxn,
                    header_info.block_hash,
                    &block_info,
                )?;
            }
            rwtxn.commit().map_err(RwTxnError::from)?;
            tracing::trace!(%block_hash, "stored ancestor headers/info");
            Ok(true)
        })
    }

    async fn run(mut self) -> Result<(), Error> {
        while let Some((request, response_tx)) = self.request_rx.next().await {
            match request {
                Request::AncestorInfos(main_block_hash) => {
                    let res = Self::request_ancestor_infos(
                        &self.env,
                        &self.archive,
                        &mut self.mainchain,
                        main_block_hash,
                    )
                    .await;
                    let response =
                        Response::AncestorInfos(main_block_hash, res);
                    if let Some(response_tx) = response_tx {
                        response_tx
                            .send(response)
                            .map_err(Error::SendResponseOneshot)?;
                    } else {
                        self.response_tx.unbounded_send(response).map_err(
                            |err| Error::SendResponse(err.into_inner()),
                        )?;
                    }
                }
            }
        }
        Ok(())
    }
}

/// Handle to the task to communicate with mainchain node.
/// Task is aborted on drop.
#[derive(Clone)]
pub(super) struct MainchainTaskHandle {
    task: Arc<JoinHandle<()>>,
    // send a request, and optional oneshot sender to receive the result on the
    // corresponding oneshot receiver
    request_tx:
        mpsc::UnboundedSender<(Request, Option<oneshot::Sender<Response>>)>,
}

impl MainchainTaskHandle {
    pub fn new<Transport>(
        env: sneed::Env,
        archive: Archive,
        mainchain: mainchain::ValidatorClient<Transport>,
    ) -> (Self, mpsc::UnboundedReceiver<Response>)
    where
        Transport: proto::Transport + Send + 'static,
        <Transport as tonic::client::GrpcService<tonic::body::BoxBody>>::Future:
            Send,
    {
        let (request_tx, request_rx) = mpsc::unbounded();
        let (response_tx, response_rx) = mpsc::unbounded();
        let task = MainchainTask {
            env,
            archive,
            mainchain,
            request_rx,
            response_tx,
        };
        let task = spawn(async move {
            if let Err(err) = task.run().await {
                let err = anyhow::Error::from(err);
                tracing::error!("Mainchain task error: {err:#}");
            }
        });
        let task_handle = MainchainTaskHandle {
            task: Arc::new(task),
            request_tx,
        };
        (task_handle, response_rx)
    }

    /// Send a request
    pub fn request(&self, request: Request) -> Result<(), Request> {
        self.request_tx
            .unbounded_send((request, None))
            .map_err(|err| {
                let (request, _) = err.into_inner();
                request
            })
    }

    /// Send a request, and receive the response on a oneshot receiver instead
    /// of the response stream
    pub fn request_oneshot(
        &self,
        request: Request,
    ) -> Result<oneshot::Receiver<Response>, Request> {
        let (oneshot_tx, oneshot_rx) = oneshot::channel();
        let () = self
            .request_tx
            .unbounded_send((request, Some(oneshot_tx)))
            .map_err(|err| {
                let (request, _) = err.into_inner();
                request
            })?;
        Ok(oneshot_rx)
    }
}

impl Drop for MainchainTaskHandle {
    // If only one reference exists (ie. within self), abort the net task.
    fn drop(&mut self) {
        // use `Arc::get_mut` since `Arc::into_inner` requires ownership of the
        // Arc, and cloning would increase the reference count
        if let Some(task) = Arc::get_mut(&mut self.task) {
            task.abort()
        }
    }
}
