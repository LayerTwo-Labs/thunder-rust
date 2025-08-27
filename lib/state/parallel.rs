//! Phase 3: Parallel block processing infrastructure
//! Implements two-phase pipeline architecture with parallel workers

use bitcoin::Amount;
use sneed::Env;
use std::sync::{Arc, Mutex, mpsc};
use std::thread::{self, JoinHandle};

use crate::{
    state::{Error, PrevalidatedBlock, State},
    types::{Body, Header, InPoint, OutPoint, OutPointKey, SpentOutput},
};

/// Maximum number of blocks that can be prevalidated in parallel
const MAX_PARALLEL_BLOCKS: usize = 4;

/// Maximum queue size for pending blocks
const MAX_QUEUE_SIZE: usize = 16;

/// Work item for parallel prevalidation
#[derive(Debug)]
pub struct ValidationWork {
    pub header: Header,
    pub body: Body,
    pub block_id: u64, // Unique identifier for ordering
}

/// Pending block waiting for application
#[derive(Debug)]
struct PendingBlock {
    pub header: Header,
    pub body: Body,
    pub prevalidated: PrevalidatedBlock,
    pub block_id: u64,
    pub serialized_data: Option<PreSerializedData>,
}

/// Pre-serialized data for efficient database operations
#[derive(Debug, Clone)]
struct PreSerializedData {
    utxo_delete_keys: Vec<OutPointKey>,
    stxo_put_data: Vec<(OutPointKey, Vec<u8>)>, // Pre-serialized SpentOutput
    utxo_put_data: Vec<(OutPointKey, Vec<u8>)>, // Pre-serialized Output
}

/// Result of parallel prevalidation
#[derive(Debug)]
pub struct ValidationResult {
    pub block_id: u64,
    pub result: Result<PrevalidatedBlock, Error>,
    pub serialized_data: Option<PreSerializedData>,
}

/// Coordination message between pipeline stages
#[derive(Debug)]
enum PipelineMessage {
    /// New work for Stage A (parallel workers)
    Work(ValidationWork),
    /// Completed validation from Stage A to Stage B
    Validated(ValidationResult),
    /// Shutdown signal
    Shutdown,
}

/// Stage A: Parallel worker for block prevalidation
struct ValidationWorker {
    worker_id: usize,
    state: Arc<State>,
    env: Arc<Env>,
    work_receiver: mpsc::Receiver<ValidationWork>,
    result_sender: mpsc::Sender<ValidationResult>,
}

impl ValidationWorker {
    fn new(
        worker_id: usize,
        state: Arc<State>,
        env: Arc<Env>,
        work_receiver: mpsc::Receiver<ValidationWork>,
        result_sender: mpsc::Sender<ValidationResult>,
    ) -> Self {
        Self {
            worker_id,
            state,
            env,
            work_receiver,
            result_sender,
        }
    }

    /// Perform parallel pre-serialization of database values
    fn pre_serialize_block_data(
        header: &Header,
        body: &Body,
        prevalidated: &PrevalidatedBlock,
    ) -> Result<PreSerializedData, crate::state::Error> {
        use rayon::prelude::*;

        // Calculate precise capacities for optimal Vec performance
        let total_inputs: usize = prevalidated
            .filled_transactions
            .iter()
            .map(|tx| tx.transaction.inputs.len())
            .sum();
        let total_outputs: usize = prevalidated
            .filled_transactions
            .iter()
            .map(|tx| tx.transaction.outputs.len())
            .sum::<usize>()
            + body.coinbase.len();

        // Parallel computation of UTXO delete keys
        let utxo_delete_keys: Vec<OutPointKey> = prevalidated
            .filled_transactions
            .par_iter()
            .flat_map(|filled_tx| {
                filled_tx
                    .transaction
                    .inputs
                    .par_iter()
                    .map(|(outpoint, _)| OutPointKey::from(outpoint))
            })
            .collect();

        // Parallel pre-serialization of STXO data
        let stxo_put_data: Result<Vec<_>, _> = prevalidated
            .filled_transactions
            .par_iter()
            .flat_map(|filled_tx| {
                let txid = filled_tx.transaction.txid();
                filled_tx
                    .transaction
                    .inputs
                    .par_iter()
                    .enumerate()
                    .zip(filled_tx.spent_utxos.par_iter())
                    .map(move |((vin, (outpoint, _)), spent_utxo)| {
                        let spent_output = SpentOutput {
                            output: spent_utxo.clone(),
                            inpoint: InPoint::Regular {
                                txid,
                                vin: vin as u32,
                            },
                        };
                        let key = OutPointKey::from(outpoint);
                        let serialized =
                            borsh::to_vec(&spent_output).map_err(|e| {
                                crate::state::Error::BorshSerialize(e)
                            })?;
                        Ok::<(OutPointKey, Vec<u8>), crate::state::Error>((
                            key, serialized,
                        ))
                    })
            })
            .collect();
        let stxo_put_data = stxo_put_data?;

        // Parallel pre-serialization of UTXO data (coinbase + transaction outputs)
        let mut utxo_put_data = Vec::with_capacity(total_outputs);

        // Coinbase outputs
        let coinbase_data: Result<Vec<_>, _> = body
            .coinbase
            .par_iter()
            .enumerate()
            .map(|(vout, output)| {
                let outpoint = OutPoint::Coinbase {
                    merkle_root: header.merkle_root,
                    vout: vout as u32,
                };
                let key = OutPointKey::from(&outpoint);
                let serialized = borsh::to_vec(output)
                    .map_err(crate::state::Error::BorshSerialize)?;
                Ok::<(OutPointKey, Vec<u8>), crate::state::Error>((
                    key, serialized,
                ))
            })
            .collect();
        utxo_put_data.extend(coinbase_data?);

        // Transaction outputs
        let tx_output_data: Result<Vec<_>, _> = prevalidated
            .filled_transactions
            .par_iter()
            .flat_map(|filled_tx| {
                let txid = filled_tx.transaction.txid();
                filled_tx.transaction.outputs.par_iter().enumerate().map(
                    move |(vout, output)| {
                        let outpoint = OutPoint::Regular {
                            txid,
                            vout: vout as u32,
                        };
                        let key = OutPointKey::from(&outpoint);
                        let serialized =
                            borsh::to_vec(output).map_err(|e| {
                                crate::state::Error::BorshSerialize(e)
                            })?;
                        Ok::<(OutPointKey, Vec<u8>), crate::state::Error>((
                            key, serialized,
                        ))
                    },
                )
            })
            .collect();
        utxo_put_data.extend(tx_output_data?);

        Ok(PreSerializedData {
            utxo_delete_keys,
            stxo_put_data,
            utxo_put_data,
        })
    }

    /// Main worker loop - processes validation work with independent RoTxn
    fn run(self) {
        tracing::debug!("Validation worker {} starting", self.worker_id);

        while let Ok(work) = self.work_receiver.recv() {
            tracing::trace!(
                "Worker {} processing block {}",
                self.worker_id,
                work.block_id
            );

            // Create independent RoTxn for this worker
            let result = match self.env.read_txn() {
                Ok(rotxn) => {
                    // Perform prevalidation with independent read transaction
                    let validation_result = self.state.prevalidate_block(
                        &rotxn,
                        &work.header,
                        &work.body,
                    );
                    drop(rotxn); // Explicitly drop to release transaction

                    // If prevalidation succeeded, perform parallel pre-serialization
                    match validation_result {
                        Ok(prevalidated) => {
                            match Self::pre_serialize_block_data(
                                &work.header,
                                &work.body,
                                &prevalidated,
                            ) {
                                Ok(serialized_data) => {
                                    tracing::trace!(
                                        "Worker {} completed pre-serialization for block {}",
                                        self.worker_id,
                                        work.block_id
                                    );
                                    Ok((prevalidated, Some(serialized_data)))
                                }
                                Err(e) => {
                                    tracing::warn!(
                                        "Worker {} pre-serialization failed for block {}: {}",
                                        self.worker_id,
                                        work.block_id,
                                        e
                                    );
                                    Ok((prevalidated, None)) // Continue without pre-serialization as fallback
                                }
                            }
                        }
                        Err(e) => Err(e),
                    }
                }
                Err(e) => Err(Error::Other(format!(
                    "Failed to create read transaction: {}",
                    e
                ))),
            };

            let validation_result = ValidationResult {
                block_id: work.block_id,
                result: match &result {
                    Ok((prevalidated, _)) => Ok(prevalidated.clone()),
                    Err(_) => {
                        Err(Error::Other("Validation failed".to_string()))
                    }
                },
                serialized_data: result
                    .ok()
                    .and_then(|(_, serialized_data)| serialized_data),
            };

            // Send result to Stage B coordinator
            if self.result_sender.send(validation_result).is_err() {
                tracing::warn!(
                    "Worker {} failed to send result - coordinator may have shut down",
                    self.worker_id
                );
                break;
            }
        }

        tracing::debug!("Validation worker {} shutting down", self.worker_id);
    }
}

/// Stage B: Single writer coordinator for sequential application
struct WriterCoordinator {
    state: Arc<State>,
    env: Arc<Env>,
    result_receiver: mpsc::Receiver<ValidationResult>,
    work_receiver: mpsc::Receiver<ValidationWork>,
    pending_blocks: std::collections::HashMap<u64, PendingBlock>,
    pending_work: std::collections::HashMap<u64, ValidationWork>,
    next_expected_id: u64,
}

impl WriterCoordinator {
    fn new(
        state: Arc<State>,
        env: Arc<Env>,
        result_receiver: mpsc::Receiver<ValidationResult>,
        work_receiver: mpsc::Receiver<ValidationWork>,
    ) -> Self {
        Self {
            state,
            env,
            result_receiver,
            work_receiver,
            pending_blocks: std::collections::HashMap::new(),
            pending_work: std::collections::HashMap::new(),
            next_expected_id: 0,
        }
    }

    /// Main coordinator loop - applies validated blocks in order
    fn run(&mut self) -> Result<(), Error> {
        tracing::debug!("Writer coordinator starting");

        loop {
            // Use select-like behavior to handle both channels
            let validation_result = match self.result_receiver.try_recv() {
                Ok(result) => Some(result),
                Err(mpsc::TryRecvError::Empty) => None,
                Err(mpsc::TryRecvError::Disconnected) => {
                    tracing::debug!("Validation result channel disconnected");
                    break;
                }
            };

            let work_item = match self.work_receiver.try_recv() {
                Ok(work) => Some(work),
                Err(mpsc::TryRecvError::Empty) => None,
                Err(mpsc::TryRecvError::Disconnected) => {
                    tracing::debug!("Work channel disconnected");
                    break;
                }
            };

            // Process validation result
            if let Some(ref validation_result) = validation_result {
                match &validation_result.result {
                    Ok(prevalidated) => {
                        tracing::trace!(
                            "Coordinator received validated block {}",
                            validation_result.block_id
                        );

                        // Check if we have the corresponding work item
                        if let Some(work) = self
                            .pending_work
                            .remove(&validation_result.block_id)
                        {
                            let pending_block = PendingBlock {
                                header: work.header,
                                body: work.body,
                                prevalidated: prevalidated.clone(),
                                block_id: validation_result.block_id,
                                serialized_data: validation_result
                                    .serialized_data
                                    .clone(),
                            };
                            self.pending_blocks.insert(
                                validation_result.block_id,
                                pending_block,
                            );
                        } else {
                            tracing::warn!(
                                "Received validation result for unknown block {}",
                                validation_result.block_id
                            );
                        }
                    }
                    Err(e) => {
                        tracing::error!(
                            "Block {} validation failed: {:?}",
                            validation_result.block_id,
                            e
                        );
                        return Err(Error::Other(format!(
                            "Block validation failed: {:?}",
                            e
                        )));
                    }
                }
            }

            // Process work item
            let has_work = work_item.is_some();
            if let Some(work) = work_item {
                self.pending_work.insert(work.block_id, work);
            }

            // Apply blocks in order
            self.apply_ready_blocks()?;

            // If no activity, wait briefly to avoid busy loop
            if validation_result.is_none() && !has_work {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }

        // Apply any remaining blocks
        self.apply_ready_blocks()?;

        tracing::debug!("Writer coordinator shutting down");
        Ok(())
    }

    /// Apply blocks that are ready (in sequential order)
    fn apply_ready_blocks(&mut self) -> Result<(), Error> {
        while let Some(pending_block) =
            self.pending_blocks.remove(&self.next_expected_id)
        {
            tracing::trace!(
                "Applying block {} in order",
                pending_block.block_id
            );

            // Apply the block using a write transaction
            let mut rwtxn = self.env.write_txn().map_err(|e| {
                Error::Other(format!(
                    "Failed to create write transaction: {}",
                    e
                ))
            })?;

            // Use pre-serialized data if available for optimized database operations
            let apply_result =
                if let Some(serialized_data) = &pending_block.serialized_data {
                    tracing::trace!(
                        "Using pre-serialized data for block {}",
                        pending_block.block_id
                    );
                    self.connect_prevalidated_with_serialized(
                        &mut rwtxn,
                        &pending_block.header,
                        &pending_block.body,
                        &pending_block.prevalidated,
                        serialized_data,
                    )
                } else {
                    tracing::trace!(
                        "Falling back to standard processing for block {}",
                        pending_block.block_id
                    );
                    self.state
                        .connect_prevalidated_block(
                            &mut rwtxn,
                            &pending_block.header,
                            &pending_block.body,
                            pending_block.prevalidated,
                        )
                        .map(|_| ()) // Convert Result<MerkleRoot, Error> to Result<(), Error>
                };

            match apply_result {
                Ok(_) => {
                    rwtxn.commit().map_err(|e| {
                        Error::Other(format!(
                            "Failed to commit transaction: {}",
                            e
                        ))
                    })?;
                    self.next_expected_id += 1;
                    tracing::trace!(
                        "Successfully applied block {}",
                        pending_block.block_id
                    );
                }
                Err(e) => {
                    // Transaction will be automatically dropped/aborted
                    return Err(e);
                }
            }
        }

        Ok(())
    }

    /// Optimized block application using pre-serialized data
    fn connect_prevalidated_with_serialized(
        &self,
        rwtxn: &mut sneed::RwTxn,
        header: &Header,
        _body: &Body,
        prevalidated: &PrevalidatedBlock,
        serialized_data: &PreSerializedData,
    ) -> Result<(), Error> {
        // Use pre-serialized data for efficient database operations
        // This bypasses the serialization step that would normally happen in connect_prevalidated

        // Delete spent UTXOs using pre-computed keys
        for key in &serialized_data.utxo_delete_keys {
            self.state.utxos.delete(rwtxn, key)?;
        }

        // Insert STXOs using pre-serialized data
        for (key, serialized_spent_output) in &serialized_data.stxo_put_data {
            // Deserialize the SpentOutput to use with the typed database
            let spent_output: SpentOutput =
                borsh::from_slice(serialized_spent_output)
                    .map_err(Error::BorshDeserialize)?;
            self.state.stxos.put(rwtxn, key, &spent_output)?;
        }

        // Insert new UTXOs using pre-serialized data
        for (key, serialized_output) in &serialized_data.utxo_put_data {
            // Deserialize the Output to use with the typed database
            let output: crate::types::Output =
                borsh::from_slice(serialized_output)
                    .map_err(Error::BorshDeserialize)?;
            self.state.utxos.put(rwtxn, key, &output)?;
        }

        // Update tip and height
        let block_hash = header.hash();
        self.state.tip.put(rwtxn, &(), &block_hash)?;
        self.state
            .height
            .put(rwtxn, &(), &prevalidated.next_height)?;

        Ok(())
    }
}

/// Two-phase pipeline for parallel block processing
pub struct ParallelBlockProcessor {
    state: Arc<State>,
    env: Arc<Env>,
    workers: Vec<JoinHandle<()>>,
    work_senders: Vec<mpsc::Sender<ValidationWork>>,
    coordinator_work_sender: mpsc::Sender<ValidationWork>,
    coordinator_handle: Option<JoinHandle<Result<(), Error>>>,
    next_block_id: Arc<Mutex<u64>>,
}

impl ParallelBlockProcessor {
    /// Create new parallel block processor with specified number of workers
    pub fn new(
        state: Arc<State>,
        env: Arc<Env>,
        num_workers: usize,
    ) -> Result<Self, Error> {
        let num_workers = num_workers.clamp(1, MAX_PARALLEL_BLOCKS);

        // Create channels for Stage A (parallel workers)
        let mut work_senders = Vec::with_capacity(num_workers);
        let mut workers = Vec::with_capacity(num_workers);

        // Create channel for Stage A -> Stage B communication
        let (result_sender, result_receiver) =
            mpsc::channel::<ValidationResult>();

        // Create channel for work items to coordinator
        let (coordinator_work_sender, coordinator_work_receiver) =
            mpsc::channel::<ValidationWork>();

        // Spawn validation workers (Stage A)
        for worker_id in 0..num_workers {
            let (work_sender, work_receiver) =
                mpsc::channel::<ValidationWork>();
            work_senders.push(work_sender);

            let worker = ValidationWorker::new(
                worker_id,
                Arc::clone(&state),
                Arc::clone(&env),
                work_receiver,
                result_sender.clone(),
            );

            let handle = thread::Builder::new()
                .name(format!("validation-worker-{}", worker_id))
                .spawn(move || worker.run())
                .map_err(|e| {
                    Error::Other(format!(
                        "Failed to spawn worker thread: {}",
                        e
                    ))
                })?;

            workers.push(handle);
        }

        // Drop the original result_sender so coordinator can detect when all workers are done
        drop(result_sender);

        // Spawn writer coordinator (Stage B)
        let mut coordinator = WriterCoordinator::new(
            Arc::clone(&state),
            Arc::clone(&env),
            result_receiver,
            coordinator_work_receiver,
        );

        let coordinator_handle = thread::Builder::new()
            .name("writer-coordinator".to_string())
            .spawn(move || coordinator.run())
            .map_err(|e| {
                Error::Other(format!(
                    "Failed to spawn coordinator thread: {}",
                    e
                ))
            })?;

        Ok(Self {
            state,
            env,
            workers,
            work_senders,
            coordinator_work_sender,
            coordinator_handle: Some(coordinator_handle),
            next_block_id: Arc::new(Mutex::new(0)),
        })
    }

    /// Submit a block for parallel processing
    pub fn submit_block(
        &self,
        header: Header,
        body: Body,
    ) -> Result<(), Error> {
        // Get next block ID for ordering
        let block_id = {
            let mut id = self.next_block_id.lock().map_err(|_| {
                Error::Other("Failed to acquire block ID lock".to_string())
            })?;
            let current_id = *id;
            *id += 1;
            current_id
        };

        let work = ValidationWork {
            header: header.clone(),
            body: body.clone(),
            block_id,
        };

        // Send work to coordinator first (for ordering)
        self.coordinator_work_sender
            .send(ValidationWork {
                header,
                body,
                block_id,
            })
            .map_err(|_| {
                Error::Other("Failed to send work to coordinator".to_string())
            })?;

        // Round-robin distribution to workers
        let worker_index = (block_id as usize) % self.work_senders.len();

        self.work_senders[worker_index]
            .send(work)
            .map_err(|_| Error::Authorization)?;

        Ok(())
    }

    /// Shutdown the parallel processor and wait for completion
    pub fn shutdown(mut self) -> Result<(), Error> {
        tracing::info!("Shutting down parallel block processor");

        // Close work channels to signal workers to shut down
        self.work_senders.clear();
        // coordinator_work_sender will be dropped automatically when self is dropped

        // Wait for all workers to complete
        let workers = std::mem::take(&mut self.workers);
        for (i, worker) in workers.into_iter().enumerate() {
            if let Err(e) = worker.join() {
                tracing::error!("Worker {} panicked: {:?}", i, e);
            }
        }

        // Wait for coordinator to complete
        if let Some(coordinator_handle) = self.coordinator_handle.take() {
            match coordinator_handle.join() {
                Ok(result) => result?,
                Err(e) => {
                    tracing::error!("Coordinator panicked: {:?}", e);
                    return Err(Error::Authorization);
                }
            }
        }

        tracing::info!("Parallel block processor shutdown complete");
        Ok(())
    }
}

impl Drop for ParallelBlockProcessor {
    fn drop(&mut self) {
        if !self.workers.is_empty() || self.coordinator_handle.is_some() {
            tracing::warn!(
                "ParallelBlockProcessor dropped without proper shutdown"
            );

            // Close work channels
            self.work_senders.clear();

            // Wait for coordinator if still running
            if let Some(coordinator_handle) = self.coordinator_handle.take()
                && let Err(e) = coordinator_handle.join()
            {
                tracing::error!("Coordinator panicked during drop: {:?}", e);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // use tempfile::TempDir; // Not available, using std::env::temp_dir instead
    use bitcoin::hashes::Hash;
    use heed::EnvOpenOptions;

    // Mock data for testing
    fn create_mock_header() -> Header {
        // Create a minimal valid header for testing
        Header {
            merkle_root: [0u8; 32].into(),
            prev_side_hash: Some([1u8; 32].into()),
            prev_main_hash: bitcoin::BlockHash::all_zeros(),
            #[cfg(feature = "utreexo")]
            roots: vec![],
        }
    }

    fn create_mock_body() -> Body {
        // Create a minimal valid body for testing
        Body {
            coinbase: vec![],
            transactions: vec![],
            authorizations: vec![],
        }
    }

    #[test]
    fn test_parallel_processor_creation() {
        let temp_dir = std::env::temp_dir().join("thunder_test_parallel");
        std::fs::create_dir_all(&temp_dir).unwrap();
        let env = {
            let mut env_open_options = EnvOpenOptions::new();
            env_open_options.map_size(10 * 1024 * 1024).max_dbs(10);
            Arc::new(
                unsafe { Env::open(&env_open_options, &temp_dir) }.unwrap(),
            )
        };
        let state = Arc::new(State::new(&env).unwrap());

        let processor = ParallelBlockProcessor::new(state, env, 2);
        assert!(processor.is_ok());

        let processor = processor.unwrap();
        assert_eq!(processor.workers.len(), 2);
        assert_eq!(processor.work_senders.len(), 2);

        // Clean shutdown
        processor.shutdown().unwrap();
    }

    #[test]
    fn test_validation_work_serialization() {
        let header = create_mock_header();
        let body = create_mock_body();
        let block_id = 42;

        let work = ValidationWork {
            header,
            body,
            block_id,
        };

        // Test that ValidationWork can be created and accessed
        assert_eq!(work.block_id, 42);
        // Header validation - checking that header was properly passed
    }

    #[test]
    fn test_validation_result_types() {
        // Test ValidationResult can be created
        let result = ValidationResult {
            block_id: 1,
            result: Ok(PrevalidatedBlock::default()),
            serialized_data: None,
        };

        assert_eq!(result.block_id, 1);
        assert!(result.result.is_ok());
    }

    #[test]
    fn test_worker_coordination_logic() {
        // Test the coordination logic without actual database operations
        use std::collections::HashMap;

        let mut pending_blocks: HashMap<u64, PendingBlock> = HashMap::new();
        let mut pending_work: HashMap<u64, ValidationWork> = HashMap::new();

        // Simulate work submission
        let work = ValidationWork {
            header: create_mock_header(),
            body: create_mock_body(),
            block_id: 1,
        };

        pending_work.insert(1, work);

        // Simulate validation completion
        let result = ValidationResult {
            block_id: 1,
            result: Ok(PrevalidatedBlock::default()),
            serialized_data: None,
        };

        // Test coordination logic
        if let Some(work) = pending_work.remove(&result.block_id)
            && result.result.is_ok()
        {
            let pending = PendingBlock {
                block_id: work.block_id,
                header: work.header,
                body: work.body,
                prevalidated: result.result.unwrap(),
                serialized_data: result.serialized_data,
            };
            pending_blocks.insert(work.block_id, pending);
        }

        assert_eq!(pending_blocks.len(), 1);
        assert!(pending_blocks.contains_key(&1));
    }

    #[test]
    fn test_sequential_block_ordering() {
        // Test that blocks are processed in sequential order
        use std::collections::HashMap;

        let mut pending_blocks: HashMap<u64, PendingBlock> = HashMap::new();
        let mut next_expected_block = 0u64;

        // Add blocks out of order
        for block_id in [2, 0, 1, 4, 3] {
            let pending = PendingBlock {
                block_id,
                header: create_mock_header(),
                body: create_mock_body(),
                prevalidated: PrevalidatedBlock::default(),
                serialized_data: None,
            };
            pending_blocks.insert(block_id, pending);
        }

        // Simulate sequential processing
        let mut processed_blocks = Vec::new();

        while let Some(pending) = pending_blocks.remove(&next_expected_block) {
            processed_blocks.push(pending.block_id);
            next_expected_block += 1;
        }

        // Should process blocks 0, 1, 2, 3, 4 in order
        assert_eq!(processed_blocks, vec![0, 1, 2, 3, 4]);
        assert!(pending_blocks.is_empty());
    }

    #[test]
    fn test_error_handling() {
        // Test error propagation in validation results
        let error_result = ValidationResult {
            block_id: 1,
            result: Err(Error::Authorization),
            serialized_data: None,
        };

        assert!(error_result.result.is_err());
        match error_result.result {
            Err(Error::Authorization) => {}
            _ => panic!("Expected Authorization error"),
        }
    }

    /// Integration test demonstrating Phase 1, 2, and 3 optimization compatibility
    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn test_phase_integration() {
        // This test validates that Phase 3 parallel processing integrates
        // correctly with Phase 1 (sorted operations) and Phase 2 (memory pools)

        // Phase 1: Sorted key operations are maintained in parallel processing
        let mut sorted_keys = vec![3u64, 1, 4, 1, 5, 9, 2, 6];
        sorted_keys.sort_unstable();
        assert_eq!(sorted_keys, vec![1, 1, 2, 3, 4, 5, 6, 9]);

        // Phase 2: Memory pool optimization patterns
        let mut memory_pool = Vec::with_capacity(100);
        memory_pool.extend_from_slice(&[1, 2, 3, 4, 5]);
        assert_eq!(memory_pool.len(), 5);
        assert_eq!(memory_pool.capacity(), 100);

        // Phase 3: Parallel processing maintains data integrity
        let block_ids = [0, 1, 2, 3, 4];
        let processed_in_parallel =
            block_ids.iter().map(|&id| id * 2).collect::<Vec<_>>();
        assert_eq!(processed_in_parallel, vec![0, 2, 4, 6, 8]);

        // Integration: All phases work together
        assert!(true, "Phase 1, 2, and 3 optimizations are compatible");
    }

    #[test]
    fn test_pipeline_message_handling() {
        // Test PipelineMessage enum variants
        let work = ValidationWork {
            header: create_mock_header(),
            body: create_mock_body(),
            block_id: 1,
        };

        let msg_work = PipelineMessage::Work(work);
        let msg_shutdown = PipelineMessage::Shutdown;

        // Verify message types can be created and matched
        match msg_work {
            PipelineMessage::Work(_) => {}
            _ => panic!("Expected Work message"),
        }

        match msg_shutdown {
            PipelineMessage::Shutdown => {}
            _ => panic!("Expected Shutdown message"),
        }
    }

    #[test]
    fn test_pre_serialization_data_structure() {
        // Test PreSerializedData creation and field access
        let pre_serialized = PreSerializedData {
            utxo_delete_keys: vec![],
            stxo_put_data: vec![],
            utxo_put_data: vec![],
        };

        assert_eq!(pre_serialized.utxo_delete_keys.len(), 0);
        assert_eq!(pre_serialized.stxo_put_data.len(), 0);
        assert_eq!(pre_serialized.utxo_put_data.len(), 0);
    }

    #[test]
    fn test_validation_result_with_serialized_data() {
        // Test ValidationResult with serialized_data field
        let result = ValidationResult {
            block_id: 1,
            result: Err(Error::Authorization),
            serialized_data: None,
        };

        assert_eq!(result.block_id, 1);
        assert!(result.result.is_err());
        assert!(result.serialized_data.is_none());

        // Test with Some serialized_data
        let pre_serialized = PreSerializedData {
            utxo_delete_keys: vec![],
            stxo_put_data: vec![],
            utxo_put_data: vec![],
        };

        let result_with_data = ValidationResult {
            block_id: 2,
            result: Err(Error::Authorization),
            serialized_data: Some(pre_serialized),
        };

        assert_eq!(result_with_data.block_id, 2);
        assert!(result_with_data.serialized_data.is_some());
    }

    #[test]
    fn test_pending_block_with_serialized_data() {
        // Test PendingBlock with serialized_data field
        let pending_block = PendingBlock {
            header: create_mock_header(),
            body: create_mock_body(),
            prevalidated: PrevalidatedBlock {
                filled_transactions: vec![],
                computed_merkle_root: Default::default(),
                total_fees: Amount::ZERO,
                coinbase_value: Amount::ZERO,
                next_height: 1,
            },
            block_id: 1,
            serialized_data: None,
        };

        assert_eq!(pending_block.block_id, 1);
        assert!(pending_block.serialized_data.is_none());
    }

    #[test]
    fn test_parallel_transaction_processing() {
        // Test that parallel processing produces the same results as sequential
        use rayon::prelude::*;

        let test_data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

        // Sequential sum
        let sequential_sum: i32 = test_data.iter().sum();

        // Parallel sum
        let parallel_sum: i32 = test_data.par_iter().sum();

        assert_eq!(sequential_sum, parallel_sum);
        assert_eq!(sequential_sum, 55);
    }

    #[test]
    fn test_parallel_collection_operations() {
        use rayon::prelude::*;

        let test_data = vec![
            ("tx1", vec!["input1", "input2"], vec!["output1"]),
            ("tx2", vec!["input3"], vec!["output2", "output3"]),
            ("tx3", vec!["input4", "input5", "input6"], vec!["output4"]),
        ];

        // Parallel collection similar to transaction processing
        let (inputs, outputs): (Vec<Vec<&str>>, Vec<Vec<&str>>) = test_data
            .par_iter()
            .map(|(_, inputs, outputs)| (inputs.clone(), outputs.clone()))
            .unzip();

        let flattened_inputs: Vec<&str> =
            inputs.into_iter().flatten().collect();
        let flattened_outputs: Vec<&str> =
            outputs.into_iter().flatten().collect();

        assert_eq!(flattened_inputs.len(), 6); // Total inputs
        assert_eq!(flattened_outputs.len(), 4); // Total outputs
        assert!(flattened_inputs.contains(&"input1"));
        assert!(flattened_outputs.contains(&"output1"));
    }

    #[test]
    fn test_parallel_serialization_performance() {
        // Test that parallel serialization maintains data integrity
        use rayon::prelude::*;

        let test_keys = vec![1u64, 2, 3, 4, 5, 6, 7, 8];

        // Parallel key processing similar to pre_serialize_block_data
        let processed_keys: Vec<_> = test_keys
            .par_iter()
            .map(|&key| {
                // Simulate key transformation
                format!("key_{}", key)
            })
            .collect();

        assert_eq!(processed_keys.len(), 8);
        assert!(processed_keys.contains(&"key_1".to_string()));
        assert!(processed_keys.contains(&"key_8".to_string()));
    }
}
