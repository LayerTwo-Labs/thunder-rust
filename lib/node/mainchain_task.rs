//! Task to communicate with mainchain node

use std::sync::Arc;

use bip300301::{
    bitcoin::{self, hashes::Hash as _},
    client::{BlockCommitment, SidechainId},
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
};

use crate::{
    archive::{self, Archive},
    node::THIS_SIDECHAIN,
    types::BlockHash,
};

/// Request data from the mainchain node
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) enum Request {
    /// Request missing mainchain ancestor headers
    AncestorHeaders(bitcoin::BlockHash),
    /// Request recursive BMM verification
    VerifyBmm(bitcoin::BlockHash),
}

/// Error included in a response
#[derive(Debug, Error)]
pub(super) enum ResponseError {
    #[error("Archive error")]
    Archive(#[from] archive::Error),
    #[error("Drivechain error")]
    Drivechain(#[from] bip300301::Error),
    #[error("Heed error")]
    Heed(#[from] heed::Error),
}

/// Response indicating that a request has been fulfilled
#[derive(Debug)]
pub(super) enum Response {
    AncestorHeaders(bitcoin::BlockHash, Result<(), ResponseError>),
    VerifyBmm(
        bitcoin::BlockHash,
        Result<Result<(), bip300301::BlockNotFoundError>, ResponseError>,
    ),
}

impl From<&Response> for Request {
    fn from(resp: &Response) -> Self {
        match resp {
            Response::AncestorHeaders(block_hash, _) => {
                Request::AncestorHeaders(*block_hash)
            }
            Response::VerifyBmm(block_hash, _) => {
                Request::VerifyBmm(*block_hash)
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
        block_hash: bitcoin::BlockHash,
    ) -> Result<(), ResponseError> {
        if block_hash == bitcoin::BlockHash::all_zeros() {
            return Ok(());
        } else {
            let rotxn = env.read_txn()?;
            if archive.try_get_main_header(&rotxn, block_hash)?.is_some() {
                return Ok(());
            }
        }
        let mut current_block_hash = block_hash;
        let mut current_height = None;
        let mut headers: Vec<BitcoinHeader> = Vec::new();
        tracing::debug!(%block_hash, "requesting ancestor headers");
        loop {
            if let Some(current_height) = current_height {
                tracing::trace!(%block_hash, "requesting ancestor headers: {current_block_hash}({current_height})")
            }
            let header = drivechain.get_header(current_block_hash).await?;
            current_block_hash = header.prev_blockhash;
            current_height = header.height.checked_sub(1);
            headers.push(header);
            if current_block_hash == bitcoin::BlockHash::all_zeros() {
                break;
            } else {
                let rotxn = env.read_txn()?;
                if archive
                    .try_get_main_header(&rotxn, current_block_hash)?
                    .is_some()
                {
                    break;
                }
            }
        }
        headers.reverse();
        // Writing all headers during IBD can starve archive readers.
        tracing::trace!(%block_hash, "storing ancestor headers");
        task::block_in_place(|| {
            let mut rwtxn = env.write_txn()?;
            headers.into_iter().try_for_each(|header| {
                archive.put_main_header(&mut rwtxn, &header)
            })?;
            rwtxn.commit()?;
            tracing::trace!(%block_hash, "stored ancestor headers");
            Ok(())
        })
    }

    /// Request ancestor BMM commitments from the mainchain node,
    /// up to and including the specified block.
    /// Mainchain headers for the specified block and all ancestors MUST exist
    /// in the archive.
    async fn request_bmm_commitments(
        env: &heed::Env,
        archive: &Archive,
        drivechain: &bip300301::Drivechain,
        main_hash: bitcoin::BlockHash,
    ) -> Result<Result<(), bip300301::BlockNotFoundError>, ResponseError> {
        if main_hash == bitcoin::BlockHash::all_zeros() {
            return Ok(Ok(()));
        } else {
            let rotxn = env.read_txn()?;
            if archive
                .try_get_main_bmm_commitment(&rotxn, main_hash)?
                .is_some()
            {
                return Ok(Ok(()));
            }
        }
        let mut missing_commitments: Vec<_> = {
            let rotxn = env.read_txn()?;
            archive
                .main_ancestors(&rotxn, main_hash)
                .take_while(|block_hash| {
                    Ok(*block_hash != bitcoin::BlockHash::all_zeros()
                        && archive
                            .try_get_main_bmm_commitment(&rotxn, *block_hash)?
                            .is_none())
                })
                .collect()?
        };
        missing_commitments.reverse();
        tracing::debug!(%main_hash, "requesting ancestor bmm commitments");
        for missing_commitment in missing_commitments {
            tracing::trace!(%main_hash,
                "requesting ancestor bmm commitment: {missing_commitment}"
            );
            let commitments: Vec<BlockHash> = match drivechain
                .get_block_commitments(missing_commitment)
                .await?
            {
                Ok(commitments) => commitments
                    .into_iter()
                    .filter_map(|(_, commitment)| match commitment {
                        BlockCommitment::BmmHStar {
                            commitment,
                            sidechain_id: SidechainId(THIS_SIDECHAIN),
                            prev_bytes: _,
                        } => Some(commitment.into()),
                        BlockCommitment::BmmHStar { .. }
                        | BlockCommitment::ScdbUpdateBytes { .. }
                        | BlockCommitment::SidechainActivationAck { .. }
                        | BlockCommitment::SidechainProposal
                        | BlockCommitment::WithdrawalBundleHash { .. }
                        | BlockCommitment::WitnessCommitment { .. } => None,
                    })
                    .collect(),
                Err(block_not_found) => return Ok(Err(block_not_found)),
            };
            // Should never be more than one commitment
            assert!(commitments.len() <= 1);
            let commitment = commitments.first().copied();
            tracing::trace!(%main_hash,
                "storing ancestor bmm commitment: {missing_commitment}"
            );
            {
                let mut rwtxn = env.write_txn()?;
                archive.put_main_bmm_commitment(
                    &mut rwtxn,
                    missing_commitment,
                    commitment,
                )?;
                rwtxn.commit()?;
            }
            tracing::trace!(%main_hash,
                "stored ancestor bmm commitment: {missing_commitment}"
            );
        }
        Ok(Ok(()))
    }

    async fn run(mut self) -> Result<(), Error> {
        while let Some((request, response_tx)) = self.request_rx.next().await {
            match request {
                Request::AncestorHeaders(main_block_hash) => {
                    let res = Self::request_ancestor_headers(
                        &self.env,
                        &self.archive,
                        &self.drivechain,
                        main_block_hash,
                    )
                    .await;
                    let response =
                        Response::AncestorHeaders(main_block_hash, res);
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
                    let response = Response::VerifyBmm(
                        block_hash,
                        Self::request_bmm_commitments(
                            &self.env,
                            &self.archive,
                            &self.drivechain,
                            block_hash,
                        )
                        .await,
                    );
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
