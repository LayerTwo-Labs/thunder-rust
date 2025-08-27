use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Mutex;

use fallible_iterator::FallibleIterator;
use futures::Stream;
use heed::types::SerdeBincode;
#[cfg(feature = "utreexo")]
use rustreexo::accumulator::{node_hash::BitcoinNodeHash, proof::Proof};
use serde::{Deserialize, Serialize};
use sneed::{
    DatabaseUnique, RoTxn, RwTxn, UnitKey,
    db::error::{self as db_error, Error as DbError},
    env::Error as EnvError,
    rwtxn::Error as RwTxnError,
};

#[cfg(feature = "utreexo")]
use crate::types::Accumulator;
use crate::{
    types::{
        Address, AmountOverflowError, AmountUnderflowError, Authorization,
        Authorized, AuthorizedTransaction, BlockHash, Body, FilledTransaction,
        GetAddress, GetValue, Header, InPoint, M6id, MerkleRoot, OutPoint,
        OutPointKey, Output, SpentOutput, Transaction, VERSION, Verify,
        Version, WithdrawalBundle, WithdrawalBundleStatus,
        proto::mainchain::TwoWayPegData,
    },
    util::Watchable,
};

#[cfg(feature = "bench")]
pub mod bench;
mod block;
mod error;
mod parallel;
mod rollback;
mod two_way_peg_data;

pub use error::Error;
pub use parallel::ParallelBlockProcessor;
use rollback::RollBack;

pub const WITHDRAWAL_BUNDLE_FAILURE_GAP: u32 = 4;

/// Phase 2: Memory pools for frequently allocated data structures
/// Reduces allocation overhead during block processing
struct MemoryPools {
    /// Pool for OutPointKey vectors (UTXO delete keys)
    outpoint_key_pool: Mutex<Vec<Vec<OutPointKey>>>,
    /// Pool for STXO put data vectors
    stxo_data_pool: Mutex<Vec<Vec<(OutPointKey, SpentOutput)>>>,
    /// Pool for UTXO put data vectors
    utxo_data_pool: Mutex<Vec<Vec<(OutPointKey, Output)>>>,
}

impl Clone for MemoryPools {
    fn clone(&self) -> Self {
        // Create new empty pools for cloned instance
        Self::new()
    }
}

impl MemoryPools {
    fn new() -> Self {
        Self {
            outpoint_key_pool: Mutex::new(Vec::new()),
            stxo_data_pool: Mutex::new(Vec::new()),
            utxo_data_pool: Mutex::new(Vec::new()),
        }
    }

    /// Get a pre-allocated vector or create a new one
    fn get_outpoint_key_vec(&self, capacity: usize) -> Vec<OutPointKey> {
        if let Ok(mut pool) = self.outpoint_key_pool.lock() {
            if let Some(mut vec) = pool.pop() {
                vec.clear();
                vec.reserve(capacity);
                return vec;
            }
        }
        Vec::with_capacity(capacity)
    }

    /// Return a vector to the pool for reuse
    fn return_outpoint_key_vec(&self, mut vec: Vec<OutPointKey>) {
        if vec.capacity() > 1024 {
            // Prevent excessive memory usage
            vec.shrink_to(1024);
        }
        if let Ok(mut pool) = self.outpoint_key_pool.lock() {
            if pool.len() < 8 {
                // Limit pool size
                pool.push(vec);
            }
        }
    }
}

/// Prevalidated block data containing computed values from validation
/// to avoid redundant computation during connection
#[derive(Debug, Clone, Default)]
pub struct PrevalidatedBlock {
    pub filled_transactions: Vec<FilledTransaction>,
    pub computed_merkle_root: MerkleRoot,
    pub total_fees: bitcoin::Amount,
    pub coinbase_value: bitcoin::Amount,
    pub next_height: u32, // Precomputed next height to avoid DB read in write txn
    #[cfg(feature = "utreexo")]
    pub accumulator_diff: crate::types::AccumulatorDiff,
}

/// Information we have regarding a withdrawal bundle
#[derive(Debug, Deserialize, Serialize)]
enum WithdrawalBundleInfo {
    /// Withdrawal bundle is known
    Known(WithdrawalBundle),
    /// Withdrawal bundle is unknown but unconfirmed / failed
    Unknown,
    /// If an unknown withdrawal bundle is confirmed, ALL UTXOs are
    /// considered spent.
    UnknownConfirmed {
        spend_utxos: BTreeMap<OutPoint, Output>,
    },
}

impl WithdrawalBundleInfo {
    fn is_known(&self) -> bool {
        match self {
            Self::Known(_) => true,
            Self::Unknown | Self::UnknownConfirmed { .. } => false,
        }
    }
}

#[derive(Clone)]
pub struct State {
    /// Current tip
    tip: DatabaseUnique<UnitKey, SerdeBincode<BlockHash>>,
    /// Current height
    height: DatabaseUnique<UnitKey, SerdeBincode<u32>>,
    pub utxos: DatabaseUnique<OutPointKey, SerdeBincode<Output>>,
    pub stxos: DatabaseUnique<OutPointKey, SerdeBincode<SpentOutput>>,
    /// Pending withdrawal bundle and block height
    pub pending_withdrawal_bundle:
        DatabaseUnique<UnitKey, SerdeBincode<(WithdrawalBundle, u32)>>,
    /// Latest failed (known) withdrawal bundle
    latest_failed_withdrawal_bundle:
        DatabaseUnique<UnitKey, SerdeBincode<RollBack<M6id>>>,
    /// Withdrawal bundles and their status.
    /// Some withdrawal bundles may be unknown.
    /// in which case they are `None`.
    withdrawal_bundles: DatabaseUnique<
        SerdeBincode<M6id>,
        SerdeBincode<(WithdrawalBundleInfo, RollBack<WithdrawalBundleStatus>)>,
    >,
    /// deposit blocks and the height at which they were applied, keyed sequentially
    pub deposit_blocks: DatabaseUnique<
        SerdeBincode<u32>,
        SerdeBincode<(bitcoin::BlockHash, u32)>,
    >,
    /// withdrawal bundle event blocks and the height at which they were applied, keyed sequentially
    pub withdrawal_bundle_event_blocks: DatabaseUnique<
        SerdeBincode<u32>,
        SerdeBincode<(bitcoin::BlockHash, u32)>,
    >,
    #[cfg(feature = "utreexo")]
    pub utreexo_accumulator: DatabaseUnique<UnitKey, SerdeBincode<Accumulator>>,
    _version: DatabaseUnique<UnitKey, SerdeBincode<Version>>,
    /// Phase 2: Memory pools for frequently allocated data structures
    /// Reduces allocation overhead during block processing
    memory_pools: MemoryPools,
}

impl State {
    pub const NUM_DBS: u32 = {
        cfg_if::cfg_if! {
            if #[cfg(feature = "utreexo")] {
                11
            } else {
                10
            }
        }
    };

    pub fn new(env: &sneed::Env) -> Result<Self, Error> {
        let mut rwtxn = env.write_txn().map_err(EnvError::from)?;
        let tip = DatabaseUnique::create(env, &mut rwtxn, "tip")
            .map_err(EnvError::from)?;
        let height = DatabaseUnique::create(env, &mut rwtxn, "height")
            .map_err(EnvError::from)?;
        let utxos = DatabaseUnique::create(env, &mut rwtxn, "utxos")
            .map_err(EnvError::from)?;
        let stxos = DatabaseUnique::create(env, &mut rwtxn, "stxos")
            .map_err(EnvError::from)?;
        let pending_withdrawal_bundle = DatabaseUnique::create(
            env,
            &mut rwtxn,
            "pending_withdrawal_bundle",
        )
        .map_err(EnvError::from)?;
        let latest_failed_withdrawal_bundle = DatabaseUnique::create(
            env,
            &mut rwtxn,
            "latest_failed_withdrawal_bundle",
        )
        .map_err(EnvError::from)?;
        let withdrawal_bundles =
            DatabaseUnique::create(env, &mut rwtxn, "withdrawal_bundles")
                .map_err(EnvError::from)?;
        let deposit_blocks =
            DatabaseUnique::create(env, &mut rwtxn, "deposit_blocks")
                .map_err(EnvError::from)?;
        let withdrawal_bundle_event_blocks = DatabaseUnique::create(
            env,
            &mut rwtxn,
            "withdrawal_bundle_event_blocks",
        )
        .map_err(EnvError::from)?;
        #[cfg(feature = "utreexo")]
        let utreexo_accumulator =
            DatabaseUnique::create(env, &mut rwtxn, "utreexo_accumulator")
                .map_err(EnvError::from)?;
        let version = DatabaseUnique::create(env, &mut rwtxn, "state_version")
            .map_err(EnvError::from)?;
        if version
            .try_get(&rwtxn, &())
            .map_err(DbError::from)?
            .is_none()
        {
            version
                .put(&mut rwtxn, &(), &*VERSION)
                .map_err(DbError::from)?;
        }
        rwtxn.commit().map_err(RwTxnError::from)?;
        Ok(Self {
            tip,
            height,
            utxos,
            stxos,
            pending_withdrawal_bundle,
            latest_failed_withdrawal_bundle,
            withdrawal_bundles,
            deposit_blocks,
            withdrawal_bundle_event_blocks,
            #[cfg(feature = "utreexo")]
            utreexo_accumulator,
            _version: version,
            memory_pools: MemoryPools::new(),
        })
    }

    pub fn try_get_tip(
        &self,
        rotxn: &RoTxn,
    ) -> Result<Option<BlockHash>, Error> {
        let tip = self.tip.try_get(rotxn, &())?;
        Ok(tip)
    }

    pub fn try_get_height(&self, rotxn: &RoTxn) -> Result<Option<u32>, Error> {
        let height = self.height.try_get(rotxn, &())?;
        Ok(height)
    }

    pub fn get_utxos(
        &self,
        rotxn: &RoTxn,
    ) -> Result<HashMap<OutPoint, Output>, db_error::Iter> {
        let utxos: HashMap<OutPoint, Output> = self
            .utxos
            .iter(rotxn)?
            .map(|(key, output)| Ok((key.into(), output)))
            .collect()?;
        Ok(utxos)
    }

    pub fn get_utxos_by_addresses(
        &self,
        rotxn: &RoTxn,
        addresses: &HashSet<Address>,
    ) -> Result<HashMap<OutPoint, Output>, db_error::Iter> {
        let utxos: HashMap<OutPoint, Output> = self
            .utxos
            .iter(rotxn)?
            .filter(|(_, output)| Ok(addresses.contains(&output.address)))
            .map(|(key, output)| Ok((key.into(), output)))
            .collect()?;
        Ok(utxos)
    }

    /// Get the latest failed withdrawal bundle, and the height at which it failed
    pub fn get_latest_failed_withdrawal_bundle(
        &self,
        rotxn: &RoTxn,
    ) -> Result<Option<(u32, M6id)>, db_error::TryGet> {
        let Some(latest_failed_m6id) =
            self.latest_failed_withdrawal_bundle.try_get(rotxn, &())?
        else {
            return Ok(None);
        };
        let latest_failed_m6id = latest_failed_m6id.latest().value;
        let (_bundle, bundle_status) = self.withdrawal_bundles.try_get(rotxn, &latest_failed_m6id)?
            .expect("Inconsistent DBs: latest failed m6id should exist in withdrawal_bundles");
        let bundle_status = bundle_status.latest();
        assert_eq!(bundle_status.value, WithdrawalBundleStatus::Failed);
        Ok(Some((bundle_status.height, latest_failed_m6id)))
    }

    #[cfg(feature = "utreexo")]
    /// Get the current Utreexo accumulator
    pub fn get_accumulator(&self, rotxn: &RoTxn) -> Result<Accumulator, Error> {
        let accumulator = self
            .utreexo_accumulator
            .try_get(rotxn, &())
            .map_err(DbError::from)?
            .unwrap_or_default();
        Ok(accumulator)
    }

    #[cfg(feature = "utreexo")]
    /// Regenerate utreexo proof for a tx
    pub fn regenerate_proof(
        &self,
        rotxn: &RoTxn,
        tx: &mut Transaction,
    ) -> Result<(), Error> {
        let accumulator = self.get_accumulator(rotxn)?;
        let targets: Vec<_> = tx
            .inputs
            .iter()
            .map(|(_, utxo_hash)| utxo_hash.into())
            .collect();
        tx.proof = accumulator.prove(&targets)?;
        Ok(())
    }

    #[cfg(feature = "utreexo")]
    /// Get a Utreexo proof for the provided utxos
    pub fn get_utreexo_proof<'a, Utxos>(
        &self,
        rotxn: &RoTxn,
        utxos: Utxos,
    ) -> Result<Proof, Error>
    where
        Utxos: IntoIterator<Item = &'a crate::types::PointedOutput>,
    {
        let accumulator = self.get_accumulator(rotxn)?;
        let targets: Vec<BitcoinNodeHash> =
            utxos.into_iter().map(BitcoinNodeHash::from).collect();
        let proof = accumulator.prove(&targets)?;
        Ok(proof)
    }

    fn fill_transaction(
        &self,
        rotxn: &RoTxn,
        transaction: &Transaction,
    ) -> Result<FilledTransaction, Error> {
        let mut spent_utxos = Vec::with_capacity(transaction.inputs.len());
        for (outpoint, _) in &transaction.inputs {
            let key = OutPointKey::from(outpoint);
            let utxo =
                self.utxos.try_get(rotxn, &key)?.ok_or(Error::NoUtxo {
                    outpoint: *outpoint,
                })?;
            spent_utxos.push(utxo);
        }
        Ok(FilledTransaction {
            spent_utxos,
            transaction: transaction.clone(),
        })
    }

    pub fn fill_authorized_transaction(
        &self,
        rotxn: &RoTxn,
        transaction: AuthorizedTransaction,
    ) -> Result<Authorized<FilledTransaction>, Error> {
        let filled_tx =
            self.fill_transaction(rotxn, &transaction.transaction)?;
        let authorizations = transaction.authorizations;
        Ok(Authorized {
            transaction: filled_tx,
            authorizations,
        })
    }

    /// Get pending withdrawal bundle and block height
    pub fn get_pending_withdrawal_bundle(
        &self,
        txn: &RoTxn,
    ) -> Result<Option<(WithdrawalBundle, u32)>, Error> {
        Ok(self
            .pending_withdrawal_bundle
            .try_get(txn, &())
            .map_err(DbError::from)?)
    }

    pub fn validate_filled_transaction(
        &self,
        transaction: &FilledTransaction,
    ) -> Result<bitcoin::Amount, Error> {
        let mut value_in = bitcoin::Amount::ZERO;
        let mut value_out = bitcoin::Amount::ZERO;
        for utxo in &transaction.spent_utxos {
            value_in = value_in
                .checked_add(utxo.get_value())
                .ok_or(AmountOverflowError)?;
        }
        for output in &transaction.transaction.outputs {
            value_out = value_out
                .checked_add(output.get_value())
                .ok_or(AmountOverflowError)?;
        }
        if value_out > value_in {
            return Err(Error::NotEnoughValueIn);
        }
        value_in
            .checked_sub(value_out)
            .ok_or_else(|| AmountUnderflowError.into())
    }

    pub fn validate_transaction(
        &self,
        rotxn: &RoTxn,
        transaction: &AuthorizedTransaction,
    ) -> Result<bitcoin::Amount, Error> {
        let filled_transaction =
            self.fill_transaction(rotxn, &transaction.transaction)?;
        for (authorization, spent_utxo) in transaction
            .authorizations
            .iter()
            .zip(filled_transaction.spent_utxos.iter())
        {
            if authorization.get_address() != spent_utxo.address {
                return Err(Error::WrongPubKeyForAddress);
            }
        }
        if Authorization::verify_transaction(transaction).is_err() {
            return Err(Error::Authorization);
        }
        let fee = self.validate_filled_transaction(&filled_transaction)?;
        Ok(fee)
    }

    #[cfg(not(feature = "bench"))]
    const LIMIT_GROWTH_EXPONENT: f64 = 1.04;

    cfg_if::cfg_if! {
        if #[cfg(feature = "bench")] {
            pub fn body_sigops_limit(_height: u32) -> usize {
                usize::MAX
            }
        } else {
            pub fn body_sigops_limit(height: u32) -> usize {
                // Starting body size limit is 8MB = 8 * 1024 * 1024 B
                // 2 input 2 output transaction is 392 B
                // 2 * ceil(8 * 1024 * 1024 B / 392 B) = 42800
                const START: usize = 42800;
                let month = height / (6 * 24 * 30);
                if month < 120 {
                    (START as f64 * Self::LIMIT_GROWTH_EXPONENT.powi(month as i32))
                        .floor() as usize
                } else {
                    // 1.04 ** 120 = 110.6625
                    // So we are rounding up.
                    START * 111
                }
            }
        }
    }

    // in bytes
    cfg_if::cfg_if! {
        if #[cfg(feature = "bench")] {
            pub fn body_size_limit(_height: u32) -> usize {
                usize::MAX
            }
        } else {
            pub fn body_size_limit(height: u32) -> usize {
                // 8MB starting body size limit.
                const START: usize = 8 * 1024 * 1024;
                let month = height / (6 * 24 * 30);
                if month < 120 {
                    (START as f64 * Self::LIMIT_GROWTH_EXPONENT.powi(month as i32))
                    .floor() as usize
                } else {
                    // 1.04 ** 120 = 110.6625
                    // So we are rounding up.
                    START * 111
                }
            }
        }
    }

    pub fn get_last_deposit_block_hash(
        &self,
        rotxn: &RoTxn,
    ) -> Result<Option<bitcoin::BlockHash>, Error> {
        let block_hash = self
            .deposit_blocks
            .last(rotxn)
            .map_err(DbError::from)?
            .map(|(_, (block_hash, _))| block_hash);
        Ok(block_hash)
    }

    pub fn get_last_withdrawal_bundle_event_block_hash(
        &self,
        rotxn: &RoTxn,
    ) -> Result<Option<bitcoin::BlockHash>, Error> {
        let block_hash = self
            .withdrawal_bundle_event_blocks
            .last(rotxn)
            .map_err(DbError::from)?
            .map(|(_, (block_hash, _))| block_hash);
        Ok(block_hash)
    }

    /// Get total sidechain wealth in Bitcoin
    pub fn sidechain_wealth(
        &self,
        rotxn: &RoTxn,
    ) -> Result<bitcoin::Amount, Error> {
        let mut total_deposit_utxo_value = bitcoin::Amount::ZERO;
        self.utxos
            .iter(rotxn)
            .map_err(DbError::from)?
            .map_err(|err| DbError::from(err).into())
            .for_each(|(outpoint_key, output)| {
                let outpoint = outpoint_key.into();
                if let OutPoint::Deposit(_) = outpoint {
                    total_deposit_utxo_value = total_deposit_utxo_value
                        .checked_add(output.get_value())
                        .ok_or(AmountOverflowError)?;
                }
                Ok::<_, Error>(())
            })?;
        let mut total_deposit_stxo_value = bitcoin::Amount::ZERO;
        let mut total_withdrawal_stxo_value = bitcoin::Amount::ZERO;
        self.stxos
            .iter(rotxn)
            .map_err(DbError::from)?
            .map_err(|err| DbError::from(err).into())
            .for_each(|(outpoint_key, spent_output)| {
                let outpoint = outpoint_key.into();
                if let OutPoint::Deposit(_) = outpoint {
                    total_deposit_stxo_value = total_deposit_stxo_value
                        .checked_add(spent_output.output.get_value())
                        .ok_or(AmountOverflowError)?;
                }
                if let InPoint::Withdrawal { .. } = spent_output.inpoint {
                    total_withdrawal_stxo_value = total_deposit_stxo_value
                        .checked_add(spent_output.output.get_value())
                        .ok_or(AmountOverflowError)?;
                }
                Ok::<_, Error>(())
            })?;

        let total_wealth: bitcoin::Amount = total_deposit_utxo_value
            .checked_add(total_deposit_stxo_value)
            .ok_or(AmountOverflowError)?
            .checked_sub(total_withdrawal_stxo_value)
            .ok_or(AmountOverflowError)?;
        Ok(total_wealth)
    }

    pub fn validate_block(
        &self,
        rotxn: &RoTxn,
        header: &Header,
        body: &Body,
    ) -> Result<(bitcoin::Amount, MerkleRoot), Error> {
        block::validate(self, rotxn, header, body)
    }

    pub fn connect_block(
        &self,
        rwtxn: &mut RwTxn,
        header: &Header,
        body: &Body,
    ) -> Result<MerkleRoot, Error> {
        block::connect(self, rwtxn, header, body)
    }

    pub fn prevalidate_block(
        &self,
        rotxn: &RoTxn,
        header: &Header,
        body: &Body,
    ) -> Result<PrevalidatedBlock, Error> {
        block::prevalidate(self, rotxn, header, body)
    }

    pub fn connect_prevalidated_block(
        &self,
        rwtxn: &mut RwTxn,
        header: &Header,
        body: &Body,
        prevalidated: PrevalidatedBlock,
    ) -> Result<MerkleRoot, Error> {
        block::connect_prevalidated(self, rwtxn, header, body, prevalidated)
    }

    pub fn apply_block(
        &self,
        rwtxn: &mut RwTxn,
        header: &Header,
        body: &Body,
    ) -> Result<(), Error> {
        let prevalidated = self.prevalidate_block(rwtxn, header, body)?;
        self.connect_prevalidated_block(rwtxn, header, body, prevalidated)?;
        Ok(())
    }

    /// Apply block with separate prevalidation phase for maximum parallelism
    /// This allows multiple blocks to be prevalidated concurrently using independent RoTxn
    pub fn apply_block_parallel(
        &self,
        env: &sneed::Env,
        header: &Header,
        body: &Body,
    ) -> Result<(), Error> {
        // Phase 1: Prevalidate with independent RoTxn (can run in parallel)
        let rotxn = env
            .read_txn()
            .map_err(|e| sneed::Error::Env(sneed::EnvError::ReadTxn(e)))?;
        let prevalidated = self.prevalidate_block(&rotxn, header, body)?;
        drop(rotxn);

        // Phase 2: Apply with single RwTxn (serialized for LMDB)
        let mut rwtxn = env.write_txn()?;
        self.connect_prevalidated_block(
            &mut rwtxn,
            header,
            body,
            prevalidated,
        )?;
        rwtxn.commit()?;
        Ok(())
    }

    /// Create a new parallel block processor for high-throughput block processing
    /// Uses two-phase pipeline: Stage A (parallel prevalidation) -> Stage B (sequential application)
    pub fn create_parallel_processor(
        self: &std::sync::Arc<Self>,
        env: std::sync::Arc<sneed::Env>,
        num_workers: usize,
    ) -> Result<ParallelBlockProcessor, Error> {
        ParallelBlockProcessor::new(
            std::sync::Arc::clone(self),
            env,
            num_workers,
        )
    }

    /// Process multiple blocks in parallel using independent RoTxn per worker
    /// This enables cross-request parallelism by moving prevalidation out of write transactions
    pub fn process_blocks_parallel(
        self: &std::sync::Arc<Self>,
        env: std::sync::Arc<sneed::Env>,
        blocks: Vec<(Header, Body)>,
        num_workers: Option<usize>,
    ) -> Result<Vec<PrevalidatedBlock>, Error> {
        use std::sync::{Arc, Mutex};
        use std::thread;

        let num_workers = num_workers.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(4)
                .min(blocks.len())
                .max(1)
        });

        if blocks.is_empty() {
            return Ok(Vec::new());
        }

        if num_workers == 1 || blocks.len() == 1 {
            // Single-threaded fallback
            let mut results = Vec::with_capacity(blocks.len());
            let rotxn = match env.read_txn() {
                Ok(txn) => txn,
                Err(e) => {
                    return Err(Error::from(sneed::env::Error::from(e)));
                }
            };
            for (header, body) in blocks {
                results.push(self.prevalidate_block(&rotxn, &header, &body)?);
            }
            return Ok(results);
        }

        // Parallel processing with independent RoTxn per worker
        let blocks = Arc::new(blocks);
        let results = Arc::new(Mutex::new(vec![None; blocks.len()]));
        let error_occurred = Arc::new(Mutex::new(None));

        let chunk_size = (blocks.len() + num_workers - 1) / num_workers;
        let mut handles = Vec::new();

        for worker_id in 0..num_workers {
            let state = Arc::clone(self);
            let env = Arc::clone(&env);
            let blocks = Arc::clone(&blocks);
            let results = Arc::clone(&results);
            let error_occurred = Arc::clone(&error_occurred);

            let handle = thread::spawn(move || {
                // Create independent RoTxn for this worker
                let rotxn = match env.read_txn() {
                    Ok(txn) => txn,
                    Err(e) => {
                        let mut error = error_occurred.lock().unwrap();
                        if error.is_none() {
                            *error =
                                Some(Error::from(sneed::env::Error::from(e)));
                        }
                        return;
                    }
                };

                let start_idx = worker_id * chunk_size;
                let end_idx = ((worker_id + 1) * chunk_size).min(blocks.len());

                for i in start_idx..end_idx {
                    // Check if another worker encountered an error
                    if error_occurred.lock().unwrap().is_some() {
                        break;
                    }

                    let (header, body) = &blocks[i];
                    match state.prevalidate_block(&rotxn, header, body) {
                        Ok(prevalidated) => {
                            let mut results = results.lock().unwrap();
                            results[i] = Some(prevalidated);
                        }
                        Err(e) => {
                            let mut error = error_occurred.lock().unwrap();
                            if error.is_none() {
                                *error = Some(e);
                            }
                            break;
                        }
                    }
                }
            });

            handles.push(handle);
        }

        // Wait for all workers to complete
        for handle in handles {
            drop(handle.join());
        }

        // Check for errors
        if let Some(error) = Arc::try_unwrap(error_occurred)
            .unwrap()
            .into_inner()
            .unwrap()
        {
            return Err(error);
        }

        // Extract results
        let results = Arc::try_unwrap(results).unwrap().into_inner().unwrap();
        let results: Result<Vec<_>, _> = results
            .into_iter()
            .enumerate()
            .map(|(_i, opt)| opt.ok_or(Error::Authorization))
            .collect();

        results
    }

    pub fn disconnect_tip(
        &self,
        rwtxn: &mut RwTxn,
        header: &Header,
        body: &Body,
    ) -> Result<(), Error> {
        block::disconnect_tip(self, rwtxn, header, body)
    }

    pub fn connect_two_way_peg_data(
        &self,
        rwtxn: &mut RwTxn,
        two_way_peg_data: &TwoWayPegData,
    ) -> Result<(), Error> {
        two_way_peg_data::connect(self, rwtxn, two_way_peg_data)
    }

    pub fn disconnect_two_way_peg_data(
        &self,
        rwtxn: &mut RwTxn,
        two_way_peg_data: &TwoWayPegData,
    ) -> Result<(), Error> {
        two_way_peg_data::disconnect(self, rwtxn, two_way_peg_data)
    }
}

impl Watchable<()> for State {
    type WatchStream = impl Stream<Item = ()>;

    /// Get a signal that notifies whenever the tip changes
    fn watch(&self) -> Self::WatchStream {
        tokio_stream::wrappers::WatchStream::new(self.tip.watch().clone())
    }
}
