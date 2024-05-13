//! Task to communicate with mainchain node

use std::sync::Arc;

use bip300301::{
    bitcoin::{self, hashes::Hash as _},
    Drivechain, Header as BitcoinHeader,
};
use fallible_iterator::FallibleIterator;
use futures::{
    channel::{
        mpsc::{self, UnboundedReceiver, UnboundedSender},
        oneshot,
    },
    StreamExt,
};
use thiserror::Error;
use tokio::{
    spawn,
    task::{self, JoinHandle},
    time::Duration,
};

use crate::{
    archive::{self, Archive},
    types::BlockHash,
};

/// Request data from the mainchain node
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) enum Request {
    /// Request missing mainchain ancestor headers
    AncestorHeaders(BlockHash),
    /// Request recursive BMM verification
    VerifyBmm(BlockHash),
}

/// Response indicating that a request has been fulfilled successfully
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Response(pub Request);

#[derive(Debug, Error)]
enum Error {
    #[error("Archive error")]
    Archive(#[from] archive::Error),
    #[error("Drivechain error")]
    Drivechain(#[from] bip300301::Error),
    #[error("Heed error")]
    Heed(#[from] heed::Error),
    #[error("Send response error")]
    SendResponse(Response),
    #[error("Send response error (oneshot)")]
    SendResponseOneshot(Response),
}

struct MainchainTask {
    env: heed::Env,
    archive: Archive,
    drivechain: Drivechain,
    // receive a request, and optional oneshot sender to send the result to
    // instead of sending on `response_tx`
    request_rx: UnboundedReceiver<(Request, Option<oneshot::Sender<Response>>)>,
    response_tx: UnboundedSender<Response>,
}

impl MainchainTask {
    /// Request ancestor headers from the mainchain node,
    /// including the specified header
    async fn request_ancestor_headers(
        env: &heed::Env,
        archive: &Archive,
        drivechain: &bip300301::Drivechain,
        mut block_hash: bitcoin::BlockHash,
    ) -> Result<(), Error> {
        tracing::debug!("requesting ancestor headers for {block_hash}");
        let mut headers: Vec<BitcoinHeader> = Vec::new();
        loop {
            if block_hash == bitcoin::BlockHash::all_zeros() {
                break;
            } else {
                let rotxn = env.read_txn()?;
                if archive.try_get_main_header(&rotxn, block_hash)?.is_some() {
                    break;
                }
            }
            let header = drivechain.get_header(block_hash).await?;
            block_hash = header.prev_blockhash;
            headers.push(header);
        }
        headers.reverse();
        if headers.is_empty() {
            Ok(())
        } else {
            // Writing all headers during IBD can starve archive readers.
            task::block_in_place(|| {
                let mut rwtxn = env.write_txn()?;
                headers.into_iter().try_for_each(|header| {
                    archive.put_main_header(&mut rwtxn, &header)
                })?;
                rwtxn.commit()?;
                Ok(())
            })
        }
    }

    /// Attempt to verify bmm for the specified block,
    /// and store the verification result
    async fn verify_bmm(
        env: &heed::Env,
        archive: &Archive,
        drivechain: &bip300301::Drivechain,
        block_hash: BlockHash,
    ) -> Result<bool, Error> {
        use jsonrpsee::types::error::ErrorCode as JsonrpseeErrorCode;
        const VERIFY_BMM_POLL_INTERVAL: Duration = Duration::from_secs(15);
        let header = {
            let rotxn = env.read_txn()?;
            archive.get_header(&rotxn, block_hash)?
        };
        let res = match drivechain
            .verify_bmm(
                &header.prev_main_hash,
                &block_hash.into(),
                VERIFY_BMM_POLL_INTERVAL,
            )
            .await
        {
            Ok(()) => true,
            Err(bip300301::Error::Jsonrpsee(jsonrpsee::core::Error::Call(
                err,
            ))) if JsonrpseeErrorCode::from(err.code())
                == JsonrpseeErrorCode::ServerError(-1) =>
            {
                false
            }
            Err(err) => return Err(Error::from(err)),
        };
        let mut rwtxn = env.write_txn()?;
        let () = archive.put_bmm_verification(&mut rwtxn, block_hash, res)?;
        rwtxn.commit()?;
        Ok(res)
    }

    /// Attempt to verify bmm recursively up to the specified block,
    /// and store the verification results
    async fn recursive_verify_bmm(
        env: &heed::Env,
        archive: &Archive,
        drivechain: &bip300301::Drivechain,
        block_hash: BlockHash,
    ) -> Result<(), Error> {
        tracing::debug!(
            "requesting recursive BMM verification for {block_hash}"
        );
        let blocks_to_verify: Vec<BlockHash> = {
            let rotxn = env.read_txn()?;
            archive
                .ancestors(&rotxn, block_hash)
                .take_while(|block_hash| {
                    archive
                        .try_get_bmm_verification(&rotxn, *block_hash)
                        .map(|bmm_verification| bmm_verification.is_none())
                })
                .collect()?
        };
        let mut blocks_to_verify_iter = blocks_to_verify.into_iter().rev();
        while let Some(block_hash) = blocks_to_verify_iter.next() {
            if !Self::verify_bmm(env, archive, drivechain, block_hash).await? {
                // mark descendent blocks as BMM failed,
                // no need to request from mainchain node
                let mut rwtxn = env.write_txn()?;
                for block_hash in blocks_to_verify_iter {
                    let () = archive
                        .put_bmm_verification(&mut rwtxn, block_hash, false)?;
                }
                rwtxn.commit()?;
                break;
            }
        }
        Ok(())
    }

    async fn run(mut self) -> Result<(), Error> {
        while let Some((request, response_tx)) = self.request_rx.next().await {
            match request {
                Request::AncestorHeaders(block_hash) => {
                    let header = {
                        let rotxn = self.env.read_txn()?;
                        self.archive.get_header(&rotxn, block_hash)?
                    };
                    let () = Self::request_ancestor_headers(
                        &self.env,
                        &self.archive,
                        &self.drivechain,
                        header.prev_main_hash,
                    )
                    .await?;
                    let response = Response(request);
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
                Request::VerifyBmm(block_hash) => {
                    let () = Self::recursive_verify_bmm(
                        &self.env,
                        &self.archive,
                        &self.drivechain,
                        block_hash,
                    )
                    .await?;
                    let response = Response(request);
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
    pub fn new(
        env: heed::Env,
        archive: Archive,
        drivechain: Drivechain,
    ) -> (Self, mpsc::UnboundedReceiver<Response>) {
        let (request_tx, request_rx) = mpsc::unbounded();
        let (response_tx, response_rx) = mpsc::unbounded();
        let task = MainchainTask {
            env,
            archive,
            drivechain,
            request_rx,
            response_tx,
        };
        let task = spawn(async {
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
