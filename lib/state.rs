use std::collections::{BTreeMap, HashMap, HashSet};

use futures::Stream;
use heed::{types::SerdeBincode, Database, RoTxn, RwTxn};
use nonempty::{nonempty, NonEmpty};
use rustreexo::accumulator::{node_hash::NodeHash, proof::Proof};
use serde::{Deserialize, Serialize};

use crate::{
    authorization::Authorization,
    types::{
        hash, proto::mainchain::TwoWayPegData, Accumulator, Address,
        AggregatedWithdrawal, AmountOverflowError, AmountUnderflowError,
        AuthorizedTransaction, BlockHash, Body, FilledTransaction, GetAddress,
        GetValue, Header, InPoint, M6id, MerkleRoot, OutPoint, Output,
        OutputContent, PointedOutput, SpentOutput, Transaction, Txid, Verify,
        WithdrawalBundle, WithdrawalBundleError, WithdrawalBundleStatus,
    },
    util::{EnvExt, UnitKey, Watchable, WatchableDb},
};

#[derive(Debug, thiserror::Error)]
pub enum InvalidHeaderError {
    #[error("expected block hash {expected}, but computed {computed}")]
    BlockHash {
        expected: BlockHash,
        computed: BlockHash,
    },
    #[error("expected previous sidechain block hash {expected}, but received {received}")]
    PrevSideHash {
        expected: BlockHash,
        received: BlockHash,
    },
}

/// Data of type `T` paired with block height at which it was last updated
#[derive(Clone, Debug, Deserialize, Serialize)]
struct HeightStamped<T> {
    value: T,
    height: u32,
}

/// Wrapper struct for fields that support rollbacks
#[derive(Clone, Debug, Deserialize, Serialize)]
#[repr(transparent)]
#[serde(transparent)]
struct RollBack<T>(NonEmpty<HeightStamped<T>>);

impl<T> RollBack<T> {
    fn new(value: T, height: u32) -> Self {
        let txid_stamped = HeightStamped { value, height };
        Self(nonempty![txid_stamped])
    }

    /// Pop the most recent value
    fn pop(mut self) -> (Option<Self>, HeightStamped<T>) {
        if let Some(value) = self.0.pop() {
            (Some(self), value)
        } else {
            (None, self.0.head)
        }
    }

    /// Attempt to push a value as the new most recent.
    /// Returns the value if the operation fails.
    fn push(&mut self, value: T, height: u32) -> Result<(), T> {
        if self.0.last().height >= height {
            return Err(value);
        }
        let height_stamped = HeightStamped { value, height };
        self.0.push(height_stamped);
        Ok(())
    }

    /// Returns the earliest value
    fn earliest(&self) -> &HeightStamped<T> {
        self.0.first()
    }

    /// Returns the most recent value
    fn latest(&self) -> &HeightStamped<T> {
        self.0.last()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("failed to verify authorization")]
    AuthorizationError,
    #[error(transparent)]
    AmountOverflow(#[from] AmountOverflowError),
    #[error(transparent)]
    AmountUnderflow(#[from] AmountUnderflowError),
    #[error("binvode error")]
    Bincode(#[from] bincode::Error),
    #[error("body too large")]
    BodyTooLarge,
    #[error("invalid body: expected merkle root {expected}, but computed {computed}")]
    InvalidBody {
        expected: MerkleRoot,
        computed: MerkleRoot,
    },
    #[error("invalid header: {0}")]
    InvalidHeader(InvalidHeaderError),
    #[error("heed error")]
    Heed(#[from] heed::Error),
    #[error("deposit block doesn't exist")]
    NoDepositBlock,
    #[error("total fees less than coinbase value")]
    NotEnoughFees,
    #[error("no tip")]
    NoTip,
    #[error("stxo {outpoint} doesn't exist")]
    NoStxo { outpoint: OutPoint },
    #[error("value in is less than value out")]
    NotEnoughValueIn,
    #[error("utxo {outpoint} doesn't exist")]
    NoUtxo { outpoint: OutPoint },
    #[error("Withdrawal bundle event block doesn't exist")]
    NoWithdrawalBundleEventBlock,
    #[error("utreexo error: {0}")]
    Utreexo(String),
    #[error("Utreexo proof verification failed for tx {txid}")]
    UtreexoProofFailed { txid: Txid },
    #[error("Computed Utreexo roots do not match the header roots")]
    UtreexoRootsMismatch,
    #[error("utxo double spent")]
    UtxoDoubleSpent,
    #[error("too many sigops")]
    TooManySigops,
    #[error("Unknown withdrawal bundle: {m6id}")]
    UnknownWithdrawalBundle { m6id: M6id },
    #[error("wrong public key for address")]
    WrongPubKeyForAddress,
    #[error(transparent)]
    WithdrawalBundle(#[from] WithdrawalBundleError),
}

#[derive(Clone)]
pub struct State {
    /// Current tip
    tip: WatchableDb<SerdeBincode<UnitKey>, SerdeBincode<BlockHash>>,
    /// Current height
    height: Database<SerdeBincode<UnitKey>, SerdeBincode<u32>>,
    pub utxos: Database<SerdeBincode<OutPoint>, SerdeBincode<Output>>,
    pub stxos: Database<SerdeBincode<OutPoint>, SerdeBincode<SpentOutput>>,
    /// Pending withdrawal bundle and block height
    pub pending_withdrawal_bundle:
        Database<SerdeBincode<UnitKey>, SerdeBincode<(WithdrawalBundle, u32)>>,
    latest_failed_withdrawal_bundle:
        Database<SerdeBincode<UnitKey>, SerdeBincode<RollBack<M6id>>>,
    withdrawal_bundles: Database<
        SerdeBincode<M6id>,
        SerdeBincode<(WithdrawalBundle, RollBack<WithdrawalBundleStatus>)>,
    >,
    /// deposit blocks and the height at which they were applied, keyed sequentially
    pub deposit_blocks:
        Database<SerdeBincode<u32>, SerdeBincode<(bitcoin::BlockHash, u32)>>,
    /// withdrawal bundle event blocks and the height at which they were applied, keyed sequentially
    pub withdrawal_bundle_event_blocks:
        Database<SerdeBincode<u32>, SerdeBincode<(bitcoin::BlockHash, u32)>>,
    pub utreexo_accumulator:
        Database<SerdeBincode<UnitKey>, SerdeBincode<Accumulator>>,
}

impl State {
    pub const NUM_DBS: u32 = 10;
    pub const WITHDRAWAL_BUNDLE_FAILURE_GAP: u32 = 4;

    pub fn new(env: &heed::Env) -> Result<Self, Error> {
        let mut rwtxn = env.write_txn()?;
        let tip = env.create_watchable_db(&mut rwtxn, "tip")?;
        let height = env.create_database(&mut rwtxn, Some("height"))?;
        let utxos = env.create_database(&mut rwtxn, Some("utxos"))?;
        let stxos = env.create_database(&mut rwtxn, Some("stxos"))?;
        let pending_withdrawal_bundle =
            env.create_database(&mut rwtxn, Some("pending_withdrawal_bundle"))?;
        let latest_failed_withdrawal_bundle = env.create_database(
            &mut rwtxn,
            Some("latest_failed_withdrawal_bundle"),
        )?;
        let withdrawal_bundles =
            env.create_database(&mut rwtxn, Some("withdrawal_bundles"))?;
        let deposit_blocks =
            env.create_database(&mut rwtxn, Some("deposit_blocks"))?;
        let withdrawal_bundle_event_blocks = env.create_database(
            &mut rwtxn,
            Some("withdrawal_bundle_event_blocks"),
        )?;
        let utreexo_accumulator =
            env.create_database(&mut rwtxn, Some("utreexo_accumulator"))?;
        rwtxn.commit()?;
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
            utreexo_accumulator,
        })
    }

    pub fn get_tip(&self, rotxn: &RoTxn) -> Result<BlockHash, Error> {
        let tip = self.tip.try_get(rotxn, &UnitKey)?.unwrap_or_default();
        Ok(tip)
    }

    pub fn get_height(&self, rotxn: &RoTxn) -> Result<u32, Error> {
        let height = self.height.get(rotxn, &UnitKey)?.unwrap_or_default();
        Ok(height)
    }

    pub fn get_utxos(
        &self,
        txn: &RoTxn,
    ) -> Result<HashMap<OutPoint, Output>, Error> {
        let mut utxos = HashMap::new();
        for item in self.utxos.iter(txn)? {
            let (outpoint, output) = item?;
            utxos.insert(outpoint, output);
        }
        Ok(utxos)
    }

    pub fn get_utxos_by_addresses(
        &self,
        txn: &RoTxn,
        addresses: &HashSet<Address>,
    ) -> Result<HashMap<OutPoint, Output>, Error> {
        let mut utxos = HashMap::new();
        for item in self.utxos.iter(txn)? {
            let (outpoint, output) = item?;
            if addresses.contains(&output.address) {
                utxos.insert(outpoint, output);
            }
        }
        Ok(utxos)
    }

    /// Get the latest failed withdrawal bundle, and the height at which it failed
    fn get_latest_failed_withdrawal_bundle(
        &self,
        rotxn: &RoTxn,
    ) -> Result<Option<(u32, WithdrawalBundle)>, Error> {
        let Some(latest_failed_m6id) =
            self.latest_failed_withdrawal_bundle.get(rotxn, &UnitKey)?
        else {
            return Ok(None);
        };
        let latest_failed_m6id = latest_failed_m6id.latest().value;
        let (bundle, bundle_status) = self.withdrawal_bundles.get(rotxn, &latest_failed_m6id)?
            .expect("Inconsistent DBs: latest failed m6id should exist in withdrawal_bundles");
        let bundle_status = bundle_status.latest();
        assert_eq!(bundle_status.value, WithdrawalBundleStatus::Failed);
        Ok(Some((bundle_status.height, bundle)))
    }

    /// Get the current Utreexo accumulator
    pub fn get_accumulator(&self, rotxn: &RoTxn) -> Result<Accumulator, Error> {
        let accumulator = self
            .utreexo_accumulator
            .get(rotxn, &UnitKey)?
            .unwrap_or_default();
        Ok(accumulator)
    }

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
        tx.proof = accumulator.0.prove(&targets).map_err(Error::Utreexo)?;
        Ok(())
    }

    /// Get a Utreexo proof for the provided utxos
    pub fn get_utreexo_proof<'a, Utxos>(
        &self,
        rotxn: &RoTxn,
        utxos: Utxos,
    ) -> Result<Proof, Error>
    where
        Utxos: IntoIterator<Item = &'a PointedOutput>,
    {
        let accumulator = self.get_accumulator(rotxn)?;
        let targets: Vec<NodeHash> =
            utxos.into_iter().map(NodeHash::from).collect();
        let proof = accumulator.0.prove(&targets).map_err(Error::Utreexo)?;
        Ok(proof)
    }

    pub fn fill_transaction(
        &self,
        txn: &RoTxn,
        transaction: &Transaction,
    ) -> Result<FilledTransaction, Error> {
        let mut spent_utxos = vec![];
        for (outpoint, _) in &transaction.inputs {
            let utxo = self.utxos.get(txn, outpoint)?.ok_or(Error::NoUtxo {
                outpoint: *outpoint,
            })?;
            spent_utxos.push(utxo);
        }
        Ok(FilledTransaction {
            spent_utxos,
            transaction: transaction.clone(),
        })
    }

    fn collect_withdrawal_bundle(
        &self,
        txn: &RoTxn,
        block_height: u32,
    ) -> Result<Option<WithdrawalBundle>, Error> {
        // Weight of a bundle with 0 outputs.
        const BUNDLE_0_WEIGHT: u64 = 504;
        // Weight of a single output.
        const OUTPUT_WEIGHT: u64 = 128;
        // Turns out to be 3121.
        const MAX_BUNDLE_OUTPUTS: usize =
            ((bitcoin::policy::MAX_STANDARD_TX_WEIGHT as u64 - BUNDLE_0_WEIGHT)
                / OUTPUT_WEIGHT) as usize;

        // Aggregate all outputs by destination.
        // destination -> (value, mainchain fee, spent_utxos)
        let mut address_to_aggregated_withdrawal = HashMap::<
            bitcoin::Address<bitcoin::address::NetworkUnchecked>,
            AggregatedWithdrawal,
        >::new();
        for item in self.utxos.iter(txn)? {
            let (outpoint, output) = item?;
            if let OutputContent::Withdrawal {
                value,
                ref main_address,
                main_fee,
            } = output.content
            {
                let aggregated = address_to_aggregated_withdrawal
                    .entry(main_address.clone())
                    .or_insert(AggregatedWithdrawal {
                        spend_utxos: HashMap::new(),
                        main_address: main_address.clone(),
                        value: bitcoin::Amount::ZERO,
                        main_fee: bitcoin::Amount::ZERO,
                    });
                // Add up all values.
                aggregated.value = aggregated
                    .value
                    .checked_add(value)
                    .ok_or(AmountOverflowError)?;
                // Set maximum mainchain fee.
                if main_fee > aggregated.main_fee {
                    aggregated.main_fee = main_fee;
                }
                aggregated.spend_utxos.insert(outpoint, output);
            }
        }
        if address_to_aggregated_withdrawal.is_empty() {
            return Ok(None);
        }
        let mut aggregated_withdrawals: Vec<_> =
            address_to_aggregated_withdrawal.into_values().collect();
        aggregated_withdrawals.sort_by_key(|a| std::cmp::Reverse(a.clone()));
        let mut fee = bitcoin::Amount::ZERO;
        let mut spend_utxos = BTreeMap::<OutPoint, Output>::new();
        let mut bundle_outputs = vec![];
        for aggregated in &aggregated_withdrawals {
            if bundle_outputs.len() > MAX_BUNDLE_OUTPUTS {
                break;
            }
            let bundle_output = bitcoin::TxOut {
                value: aggregated.value,
                script_pubkey: aggregated
                    .main_address
                    .assume_checked_ref()
                    .script_pubkey(),
            };
            spend_utxos.extend(aggregated.spend_utxos.clone());
            bundle_outputs.push(bundle_output);
            fee += aggregated.main_fee;
        }
        let bundle = WithdrawalBundle::new(
            block_height,
            fee,
            spend_utxos,
            bundle_outputs,
        )?;
        Ok(Some(bundle))
    }

    /// Get pending withdrawal bundle and block height
    pub fn get_pending_withdrawal_bundle(
        &self,
        txn: &RoTxn,
    ) -> Result<Option<(WithdrawalBundle, u32)>, Error> {
        Ok(self.pending_withdrawal_bundle.get(txn, &UnitKey)?)
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
            return Err(Error::AuthorizationError);
        }
        let fee = self.validate_filled_transaction(&filled_transaction)?;
        Ok(fee)
    }

    const LIMIT_GROWTH_EXPONENT: f64 = 1.04;

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

    // in bytes
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

    pub fn validate_block(
        &self,
        rotxn: &RoTxn,
        header: &Header,
        body: &Body,
    ) -> Result<bitcoin::Amount, Error> {
        let tip_hash = self.get_tip(rotxn)?;
        if header.prev_side_hash != tip_hash {
            let err = InvalidHeaderError::PrevSideHash {
                expected: tip_hash,
                received: header.prev_side_hash,
            };
            return Err(Error::InvalidHeader(err));
        };
        let height = self.get_height(rotxn)?;
        if body.authorizations.len() > Self::body_sigops_limit(height) {
            return Err(Error::TooManySigops);
        }
        if bincode::serialize(&body)?.len() > Self::body_size_limit(height) {
            return Err(Error::BodyTooLarge);
        }
        let mut accumulator = self
            .utreexo_accumulator
            .get(rotxn, &UnitKey)?
            .unwrap_or_default();
        // New leaves for the accumulator
        let mut accumulator_add = Vec::<NodeHash>::new();
        // Accumulator leaves to delete
        let mut accumulator_del = Vec::<NodeHash>::new();
        let mut coinbase_value = bitcoin::Amount::ZERO;
        let merkle_root = body.compute_merkle_root();
        if merkle_root != header.merkle_root {
            let err = Error::InvalidBody {
                expected: merkle_root,
                computed: header.merkle_root,
            };
            return Err(err);
        }
        for (vout, output) in body.coinbase.iter().enumerate() {
            coinbase_value = coinbase_value
                .checked_add(output.get_value())
                .ok_or(AmountOverflowError)?;
            let outpoint = OutPoint::Coinbase {
                merkle_root,
                vout: vout as u32,
            };
            let pointed_output = PointedOutput {
                outpoint,
                output: output.clone(),
            };
            accumulator_add.push((&pointed_output).into());
        }
        let mut total_fees = bitcoin::Amount::ZERO;
        let mut spent_utxos = HashSet::new();
        let filled_transactions: Vec<_> = body
            .transactions
            .iter()
            .map(|t| self.fill_transaction(rotxn, t))
            .collect::<Result<_, _>>()?;
        for filled_transaction in &filled_transactions {
            let txid = filled_transaction.transaction.txid();
            // hashes of spent utxos, used to verify the utreexo proof
            let mut spent_utxo_hashes = Vec::<NodeHash>::new();
            for (outpoint, utxo_hash) in &filled_transaction.transaction.inputs
            {
                if spent_utxos.contains(outpoint) {
                    return Err(Error::UtxoDoubleSpent);
                }
                spent_utxos.insert(*outpoint);
                spent_utxo_hashes.push(utxo_hash.into());
                accumulator_del.push(utxo_hash.into());
            }
            for (vout, output) in
                filled_transaction.transaction.outputs.iter().enumerate()
            {
                let outpoint = OutPoint::Regular {
                    txid,
                    vout: vout as u32,
                };
                let pointed_output = PointedOutput {
                    outpoint,
                    output: output.clone(),
                };
                accumulator_add.push((&pointed_output).into());
            }
            total_fees = total_fees
                .checked_add(
                    self.validate_filled_transaction(filled_transaction)?,
                )
                .ok_or(AmountOverflowError)?;
            // verify utreexo proof
            if !accumulator
                .0
                .verify(
                    &filled_transaction.transaction.proof,
                    &spent_utxo_hashes,
                )
                .map_err(Error::Utreexo)?
            {
                return Err(Error::UtreexoProofFailed { txid });
            }
        }
        if coinbase_value > total_fees {
            return Err(Error::NotEnoughFees);
        }
        let spent_utxos = filled_transactions
            .iter()
            .flat_map(|t| t.spent_utxos.iter());
        for (authorization, spent_utxo) in
            body.authorizations.iter().zip(spent_utxos)
        {
            if authorization.get_address() != spent_utxo.address {
                return Err(Error::WrongPubKeyForAddress);
            }
        }
        if Authorization::verify_body(body).is_err() {
            return Err(Error::AuthorizationError);
        }
        accumulator
            .0
            .modify(&accumulator_add, &accumulator_del)
            .map_err(Error::Utreexo)?;
        let roots: Vec<NodeHash> = accumulator
            .0
            .get_roots()
            .iter()
            .map(|node| node.get_data())
            .collect();
        if roots != header.roots {
            return Err(Error::UtreexoRootsMismatch);
        }
        Ok(total_fees)
    }

    pub fn get_last_deposit_block_hash(
        &self,
        rotxn: &RoTxn,
    ) -> Result<Option<bitcoin::BlockHash>, Error> {
        let block_hash = self
            .deposit_blocks
            .last(rotxn)?
            .map(|(_, (block_hash, _))| block_hash);
        Ok(block_hash)
    }

    pub fn get_last_withdrawal_bundle_event_block_hash(
        &self,
        rotxn: &RoTxn,
    ) -> Result<Option<bitcoin::BlockHash>, Error> {
        let block_hash = self
            .withdrawal_bundle_event_blocks
            .last(rotxn)?
            .map(|(_, (block_hash, _))| block_hash);
        Ok(block_hash)
    }

    pub fn connect_two_way_peg_data(
        &self,
        rwtxn: &mut RwTxn,
        two_way_peg_data: &TwoWayPegData,
    ) -> Result<(), Error> {
        let block_height = self.get_height(rwtxn)?;
        tracing::trace!(%block_height, "Connecting 2WPD...");
        let mut accumulator = self
            .utreexo_accumulator
            .get(rwtxn, &UnitKey)?
            .unwrap_or_default();
        // New leaves for the accumulator
        let mut accumulator_add = Vec::<NodeHash>::new();
        // Accumulator leaves to delete
        let mut accumulator_del = HashSet::<NodeHash>::new();
        // Handle deposits.
        if let Some(latest_deposit_block_hash) =
            two_way_peg_data.latest_deposit_block_hash()
        {
            let deposit_block_seq_idx = self
                .deposit_blocks
                .last(rwtxn)?
                .map_or(0, |(seq_idx, _)| seq_idx + 1);
            self.deposit_blocks.put(
                rwtxn,
                &deposit_block_seq_idx,
                &(latest_deposit_block_hash, block_height - 1),
            )?;
        }
        for deposit in two_way_peg_data
            .deposits()
            .flat_map(|(_, deposits)| deposits)
        {
            let outpoint = OutPoint::Deposit(deposit.outpoint);
            let output = deposit.output.clone();
            self.utxos.put(rwtxn, &outpoint, &output)?;
            let utxo_hash = hash(&PointedOutput { outpoint, output });
            accumulator_add.push(utxo_hash.into())
        }

        // Handle withdrawals
        if let Some(latest_withdrawal_bundle_event_block_hash) =
            two_way_peg_data.latest_withdrawal_bundle_event_block_hash()
        {
            let withdrawal_bundle_event_block_seq_idx = self
                .withdrawal_bundle_event_blocks
                .last(rwtxn)?
                .map_or(0, |(seq_idx, _)| seq_idx + 1);
            self.withdrawal_bundle_event_blocks.put(
                rwtxn,
                &withdrawal_bundle_event_block_seq_idx,
                &(*latest_withdrawal_bundle_event_block_hash, block_height - 1),
            )?;
        }
        let last_withdrawal_bundle_failure_height = self
            .get_latest_failed_withdrawal_bundle(rwtxn)?
            .map(|(height, _bundle)| height)
            .unwrap_or_default();
        if block_height - last_withdrawal_bundle_failure_height
            > Self::WITHDRAWAL_BUNDLE_FAILURE_GAP
            && self
                .pending_withdrawal_bundle
                .get(rwtxn, &UnitKey)?
                .is_none()
        {
            if let Some(bundle) =
                self.collect_withdrawal_bundle(rwtxn, block_height)?
            {
                for (outpoint, spend_output) in bundle.spend_utxos() {
                    let utxo_hash = hash(&PointedOutput {
                        outpoint: *outpoint,
                        output: spend_output.clone(),
                    });
                    accumulator_del.insert(utxo_hash.into());
                    self.utxos.delete(rwtxn, outpoint)?;
                    let m6id = bundle.compute_m6id();
                    let spent_output = SpentOutput {
                        output: spend_output.clone(),
                        inpoint: InPoint::Withdrawal { m6id },
                    };
                    self.stxos.put(rwtxn, outpoint, &spent_output)?;
                }
                let m6id = bundle.compute_m6id();
                self.pending_withdrawal_bundle.put(
                    rwtxn,
                    &UnitKey,
                    &(bundle, block_height),
                )?;
                tracing::trace!(
                    %block_height,
                    %m6id,
                    "Stored pending withdrawal bundle"
                );
            }
        }
        for (_, m6id, event) in two_way_peg_data.withdrawal_bundle_events() {
            match event.status {
                WithdrawalBundleStatus::Submitted => {
                    let Some((bundle, bundle_block_height)) =
                        self.pending_withdrawal_bundle.get(rwtxn, &UnitKey)?
                    else {
                        if let Some((_bundle, bundle_status)) =
                            self.withdrawal_bundles.get(rwtxn, m6id)?
                        {
                            // Already applied
                            assert_eq!(
                                bundle_status.earliest().value,
                                WithdrawalBundleStatus::Submitted
                            );
                            continue;
                        }
                        return Err(Error::UnknownWithdrawalBundle {
                            m6id: *m6id,
                        });
                    };
                    assert_eq!(bundle_block_height, block_height - 2);
                    if bundle.compute_m6id() != *m6id {
                        return Err(Error::UnknownWithdrawalBundle {
                            m6id: *m6id,
                        });
                    }
                    tracing::debug!(
                        %m6id,
                        "Withdrawal bundle successfully submitted"
                    );
                    self.withdrawal_bundles.put(
                        rwtxn,
                        m6id,
                        &(
                            bundle,
                            RollBack::new(
                                WithdrawalBundleStatus::Submitted,
                                block_height,
                            ),
                        ),
                    )?;
                    self.pending_withdrawal_bundle.delete(rwtxn, &UnitKey)?;
                }
                WithdrawalBundleStatus::Confirmed => {
                    let Some((bundle, mut bundle_status)) =
                        self.withdrawal_bundles.get(rwtxn, m6id)?
                    else {
                        return Err(Error::UnknownWithdrawalBundle {
                            m6id: *m6id,
                        });
                    };
                    if bundle_status.latest().value
                        == WithdrawalBundleStatus::Confirmed
                    {
                        // Already applied
                        continue;
                    } else {
                        assert_eq!(
                            bundle_status.latest().value,
                            WithdrawalBundleStatus::Submitted
                        );
                    }
                    bundle_status
                        .push(WithdrawalBundleStatus::Confirmed, block_height)
                        .expect("Push confirmed status should be valid");
                    self.withdrawal_bundles.put(
                        rwtxn,
                        m6id,
                        &(bundle, bundle_status),
                    )?;
                }
                WithdrawalBundleStatus::Failed => {
                    let Some((bundle, mut bundle_status)) =
                        self.withdrawal_bundles.get(rwtxn, m6id)?
                    else {
                        return Err(Error::UnknownWithdrawalBundle {
                            m6id: *m6id,
                        });
                    };
                    if bundle_status.latest().value
                        == WithdrawalBundleStatus::Failed
                    {
                        // Already applied
                        continue;
                    } else {
                        assert_eq!(
                            bundle_status.latest().value,
                            WithdrawalBundleStatus::Submitted
                        );
                    }
                    bundle_status
                        .push(WithdrawalBundleStatus::Failed, block_height)
                        .expect("Push failed status should be valid");
                    for (outpoint, output) in bundle.spend_utxos() {
                        self.stxos.delete(rwtxn, outpoint)?;
                        self.utxos.put(rwtxn, outpoint, output)?;
                        let utxo_hash = hash(&PointedOutput {
                            outpoint: *outpoint,
                            output: output.clone(),
                        });
                        accumulator_del.remove(&utxo_hash.into());
                    }
                    let latest_failed_m6id =
                        if let Some(mut latest_failed_m6id) = self
                            .latest_failed_withdrawal_bundle
                            .get(rwtxn, &UnitKey)?
                        {
                            latest_failed_m6id
                                .push(*m6id, block_height)
                                .expect(
                                    "Push latest failed m6id should be valid",
                                );
                            latest_failed_m6id
                        } else {
                            RollBack::new(*m6id, block_height)
                        };
                    self.latest_failed_withdrawal_bundle.put(
                        rwtxn,
                        &UnitKey,
                        &latest_failed_m6id,
                    )?;
                    self.withdrawal_bundles.put(
                        rwtxn,
                        m6id,
                        &(bundle, bundle_status),
                    )?;
                }
            }
        }
        let accumulator_del: Vec<_> = accumulator_del.into_iter().collect();
        accumulator
            .0
            .modify(&accumulator_add, &accumulator_del)
            .map_err(Error::Utreexo)?;
        self.utreexo_accumulator
            .put(rwtxn, &UnitKey, &accumulator)?;
        Ok(())
    }

    pub fn disconnect_two_way_peg_data(
        &self,
        rwtxn: &mut RwTxn,
        two_way_peg_data: &TwoWayPegData,
    ) -> Result<(), Error> {
        let block_height = self.get_height(rwtxn)?;
        let mut accumulator = self
            .utreexo_accumulator
            .get(rwtxn, &UnitKey)?
            .unwrap_or_default();
        // New leaves for the accumulator
        let mut accumulator_add = Vec::<NodeHash>::new();
        // Accumulator leaves to delete
        let mut accumulator_del = HashSet::<NodeHash>::new();

        // Restore pending withdrawal bundle
        for (_, m6id, event) in
            two_way_peg_data.withdrawal_bundle_events().rev()
        {
            match event.status {
                WithdrawalBundleStatus::Submitted => {
                    let Some((bundle, bundle_status)) =
                        self.withdrawal_bundles.get(rwtxn, m6id)?
                    else {
                        if let Some((bundle, _)) = self
                            .pending_withdrawal_bundle
                            .get(rwtxn, &UnitKey)?
                            && bundle.compute_m6id() == *m6id
                        {
                            // Already applied
                            continue;
                        }
                        return Err(Error::UnknownWithdrawalBundle {
                            m6id: *m6id,
                        });
                    };
                    let bundle_status = bundle_status.latest();
                    assert_eq!(
                        bundle_status.value,
                        WithdrawalBundleStatus::Submitted
                    );
                    assert_eq!(bundle_status.height, block_height);
                    self.pending_withdrawal_bundle.put(
                        rwtxn,
                        &UnitKey,
                        &(bundle, bundle_status.height - 2),
                    )?;
                    self.withdrawal_bundles.delete(rwtxn, m6id)?;
                }
                WithdrawalBundleStatus::Confirmed => {
                    let Some((bundle, bundle_status)) =
                        self.withdrawal_bundles.get(rwtxn, m6id)?
                    else {
                        return Err(Error::UnknownWithdrawalBundle {
                            m6id: *m6id,
                        });
                    };
                    let (prev_bundle_status, latest_bundle_status) =
                        bundle_status.pop();
                    if latest_bundle_status.value
                        == WithdrawalBundleStatus::Submitted
                    {
                        // Already applied
                        continue;
                    } else {
                        assert_eq!(
                            latest_bundle_status.value,
                            WithdrawalBundleStatus::Confirmed
                        );
                    }
                    assert_eq!(latest_bundle_status.height, block_height);
                    let prev_bundle_status = prev_bundle_status
                        .expect("Pop confirmed bundle status should be valid");
                    assert_eq!(
                        prev_bundle_status.latest().value,
                        WithdrawalBundleStatus::Submitted
                    );
                    self.withdrawal_bundles.put(
                        rwtxn,
                        m6id,
                        &(bundle, prev_bundle_status),
                    )?;
                }
                WithdrawalBundleStatus::Failed => {
                    let Some((bundle, bundle_status)) =
                        self.withdrawal_bundles.get(rwtxn, m6id)?
                    else {
                        return Err(Error::UnknownWithdrawalBundle {
                            m6id: *m6id,
                        });
                    };
                    let (prev_bundle_status, latest_bundle_status) =
                        bundle_status.pop();
                    if latest_bundle_status.value
                        == WithdrawalBundleStatus::Submitted
                    {
                        // Already applied
                        continue;
                    } else {
                        assert_eq!(
                            latest_bundle_status.value,
                            WithdrawalBundleStatus::Failed
                        );
                    }
                    assert_eq!(latest_bundle_status.height, block_height);
                    let prev_bundle_status = prev_bundle_status
                        .expect("Pop failed bundle status should be valid");
                    assert_eq!(
                        prev_bundle_status.latest().value,
                        WithdrawalBundleStatus::Submitted
                    );
                    for (outpoint, output) in bundle.spend_utxos().iter().rev()
                    {
                        let spent_output = SpentOutput {
                            output: output.clone(),
                            inpoint: InPoint::Withdrawal { m6id: *m6id },
                        };
                        self.stxos.put(rwtxn, outpoint, &spent_output)?;
                        if self.utxos.delete(rwtxn, outpoint)? {
                            return Err(Error::NoUtxo {
                                outpoint: *outpoint,
                            });
                        };
                        let utxo_hash = hash(&PointedOutput {
                            outpoint: *outpoint,
                            output: output.clone(),
                        });
                        accumulator_add.push(utxo_hash.into());
                    }
                    self.withdrawal_bundles.put(
                        rwtxn,
                        m6id,
                        &(bundle, prev_bundle_status),
                    )?;
                    let (prev_latest_failed_m6id, latest_failed_m6id) = self
                        .latest_failed_withdrawal_bundle
                        .get(rwtxn, &UnitKey)?
                        .expect("latest failed withdrawal bundle should exist")
                        .pop();
                    assert_eq!(latest_failed_m6id.value, *m6id);
                    assert_eq!(latest_failed_m6id.height, block_height);
                    if let Some(prev_latest_failed_m6id) =
                        prev_latest_failed_m6id
                    {
                        self.latest_failed_withdrawal_bundle.put(
                            rwtxn,
                            &UnitKey,
                            &prev_latest_failed_m6id,
                        )?;
                    } else {
                        self.latest_failed_withdrawal_bundle
                            .delete(rwtxn, &UnitKey)?;
                    }
                }
            }
        }
        // Handle withdrawals
        if let Some(latest_withdrawal_bundle_event_block_hash) =
            two_way_peg_data.latest_withdrawal_bundle_event_block_hash()
        {
            let (
                last_withdrawal_bundle_event_block_seq_idx,
                (
                    last_withdrawal_bundle_event_block_hash,
                    last_withdrawal_bundle_event_block_height,
                ),
            ) = self
                .withdrawal_bundle_event_blocks
                .last(rwtxn)?
                .ok_or(Error::NoWithdrawalBundleEventBlock)?;
            assert_eq!(
                *latest_withdrawal_bundle_event_block_hash,
                last_withdrawal_bundle_event_block_hash
            );
            assert_eq!(
                block_height - 1,
                last_withdrawal_bundle_event_block_height
            );
            if !self
                .deposit_blocks
                .delete(rwtxn, &last_withdrawal_bundle_event_block_seq_idx)?
            {
                return Err(Error::NoWithdrawalBundleEventBlock);
            };
        }
        let last_withdrawal_bundle_failure_height = self
            .get_latest_failed_withdrawal_bundle(rwtxn)?
            .map(|(height, _bundle)| height)
            .unwrap_or_default();
        if block_height - last_withdrawal_bundle_failure_height
            > Self::WITHDRAWAL_BUNDLE_FAILURE_GAP
            && let Some((bundle, bundle_height)) =
                self.pending_withdrawal_bundle.get(rwtxn, &UnitKey)?
            && bundle_height == block_height - 2
        {
            self.pending_withdrawal_bundle.delete(rwtxn, &UnitKey)?;
            for (outpoint, output) in bundle.spend_utxos().iter().rev() {
                let utxo_hash = hash(&PointedOutput {
                    outpoint: *outpoint,
                    output: output.clone(),
                });
                accumulator_add.push(utxo_hash.into());
                if !self.stxos.delete(rwtxn, outpoint)? {
                    return Err(Error::NoStxo {
                        outpoint: *outpoint,
                    });
                };
                self.utxos.put(rwtxn, outpoint, output)?;
            }
        }
        // Handle deposits.
        if let Some(latest_deposit_block_hash) =
            two_way_peg_data.latest_deposit_block_hash()
        {
            let (
                last_deposit_block_seq_idx,
                (last_deposit_block_hash, last_deposit_block_height),
            ) = self
                .deposit_blocks
                .last(rwtxn)?
                .ok_or(Error::NoDepositBlock)?;
            assert_eq!(latest_deposit_block_hash, last_deposit_block_hash);
            assert_eq!(block_height - 1, last_deposit_block_height);
            if !self
                .deposit_blocks
                .delete(rwtxn, &last_deposit_block_seq_idx)?
            {
                return Err(Error::NoDepositBlock);
            };
        }
        for deposit in two_way_peg_data
            .deposits()
            .flat_map(|(_, deposits)| deposits)
            .rev()
        {
            let outpoint = OutPoint::Deposit(deposit.outpoint);
            let output = deposit.output.clone();
            if !self.utxos.delete(rwtxn, &outpoint)? {
                return Err(Error::NoUtxo { outpoint });
            }
            let utxo_hash = hash(&PointedOutput { outpoint, output });
            accumulator_del.insert(utxo_hash.into());
        }
        let accumulator_del: Vec<_> = accumulator_del.into_iter().collect();
        accumulator
            .0
            .modify(&accumulator_add, &accumulator_del)
            .map_err(Error::Utreexo)?;
        self.utreexo_accumulator
            .put(rwtxn, &UnitKey, &accumulator)?;
        Ok(())
    }

    pub fn connect_block(
        &self,
        rwtxn: &mut RwTxn,
        header: &Header,
        body: &Body,
    ) -> Result<(), Error> {
        let tip_hash = self.get_tip(rwtxn)?;
        if tip_hash != header.prev_side_hash {
            let err = InvalidHeaderError::PrevSideHash {
                expected: tip_hash,
                received: header.prev_side_hash,
            };
            return Err(Error::InvalidHeader(err));
        }
        let merkle_root = body.compute_merkle_root();
        if merkle_root != header.merkle_root {
            let err = Error::InvalidBody {
                expected: merkle_root,
                computed: header.merkle_root,
            };
            return Err(err);
        }
        let mut accumulator = self
            .utreexo_accumulator
            .get(rwtxn, &UnitKey)?
            .unwrap_or_default();
        // New leaves for the accumulator
        let mut accumulator_add = Vec::<NodeHash>::new();
        // Accumulator leaves to delete
        let mut accumulator_del = Vec::<NodeHash>::new();
        for (vout, output) in body.coinbase.iter().enumerate() {
            let outpoint = OutPoint::Coinbase {
                merkle_root,
                vout: vout as u32,
            };
            let pointed_output = PointedOutput {
                outpoint,
                output: output.clone(),
            };
            accumulator_add.push((&pointed_output).into());
            self.utxos.put(rwtxn, &outpoint, output)?;
        }
        for transaction in &body.transactions {
            let txid = transaction.txid();
            for (vin, (outpoint, utxo_hash)) in
                transaction.inputs.iter().enumerate()
            {
                let spent_output =
                    self.utxos.get(rwtxn, outpoint)?.ok_or(Error::NoUtxo {
                        outpoint: *outpoint,
                    })?;
                accumulator_del.push(utxo_hash.into());
                self.utxos.delete(rwtxn, outpoint)?;
                let spent_output = SpentOutput {
                    output: spent_output,
                    inpoint: InPoint::Regular {
                        txid,
                        vin: vin as u32,
                    },
                };
                self.stxos.put(rwtxn, outpoint, &spent_output)?;
            }
            for (vout, output) in transaction.outputs.iter().enumerate() {
                let outpoint = OutPoint::Regular {
                    txid,
                    vout: vout as u32,
                };
                let pointed_output = PointedOutput {
                    outpoint,
                    output: output.clone(),
                };
                accumulator_add.push((&pointed_output).into());
                self.utxos.put(rwtxn, &outpoint, output)?;
            }
        }
        let block_hash = header.hash();
        let height = self.get_height(rwtxn)?;
        self.tip.put(rwtxn, &UnitKey, &block_hash)?;
        self.height.put(rwtxn, &UnitKey, &(height + 1))?;
        accumulator
            .0
            .modify(&accumulator_add, &accumulator_del)
            .map_err(Error::Utreexo)?;
        self.utreexo_accumulator
            .put(rwtxn, &UnitKey, &accumulator)?;
        Ok(())
    }

    pub fn disconnect_tip(
        &self,
        rwtxn: &mut RwTxn,
        header: &Header,
        body: &Body,
    ) -> Result<(), Error> {
        let tip_hash = self.tip.try_get(rwtxn, &UnitKey)?.unwrap_or_default();
        if tip_hash != header.hash() {
            let err = InvalidHeaderError::BlockHash {
                expected: tip_hash,
                computed: header.hash(),
            };
            return Err(Error::InvalidHeader(err));
        }
        let merkle_root = body.compute_merkle_root();
        if merkle_root != header.merkle_root {
            let err = Error::InvalidBody {
                expected: merkle_root,
                computed: header.merkle_root,
            };
            return Err(err);
        }
        let mut accumulator = self
            .utreexo_accumulator
            .get(rwtxn, &UnitKey)?
            .unwrap_or_default();
        tracing::debug!("Got acc");
        // New leaves for the accumulator
        let mut accumulator_add = Vec::<NodeHash>::new();
        // Accumulator leaves to delete
        let mut accumulator_del = Vec::<NodeHash>::new();
        // revert txs, last-to-first
        body.transactions.iter().rev().try_for_each(|tx| {
            let txid = tx.txid();
            // delete UTXOs, last-to-first
            tx.outputs.iter().enumerate().rev().try_for_each(
                |(vout, output)| {
                    let outpoint = OutPoint::Regular {
                        txid,
                        vout: vout as u32,
                    };
                    let pointed_output = PointedOutput {
                        outpoint,
                        output: output.clone(),
                    };
                    accumulator_del.push((&pointed_output).into());
                    if self.utxos.delete(rwtxn, &outpoint)? {
                        Ok(())
                    } else {
                        Err(Error::NoUtxo { outpoint })
                    }
                },
            )?;
            // unspend STXOs, last-to-first
            tx.inputs
                .iter()
                .rev()
                .try_for_each(|(outpoint, utxo_hash)| {
                    if let Some(spent_output) =
                        self.stxos.get(rwtxn, outpoint)?
                    {
                        accumulator_add.push(utxo_hash.into());
                        self.stxos.delete(rwtxn, outpoint)?;
                        self.utxos.put(
                            rwtxn,
                            outpoint,
                            &spent_output.output,
                        )?;
                        Ok(())
                    } else {
                        Err(Error::NoStxo {
                            outpoint: *outpoint,
                        })
                    }
                })
        })?;
        // delete coinbase UTXOs, last-to-first
        body.coinbase.iter().enumerate().rev().try_for_each(
            |(vout, output)| {
                let outpoint = OutPoint::Coinbase {
                    merkle_root,
                    vout: vout as u32,
                };
                let pointed_output = PointedOutput {
                    outpoint,
                    output: output.clone(),
                };
                accumulator_del.push((&pointed_output).into());
                if self.utxos.delete(rwtxn, &outpoint)? {
                    Ok(())
                } else {
                    Err(Error::NoUtxo { outpoint })
                }
            },
        )?;
        let height = self.get_height(rwtxn)?;
        self.tip.put(rwtxn, &UnitKey, &header.prev_side_hash)?;
        self.height.put(rwtxn, &UnitKey, &(height - 1))?;
        accumulator
            .0
            .modify(&accumulator_add, &accumulator_del)
            .map_err(Error::Utreexo)?;
        self.utreexo_accumulator
            .put(rwtxn, &UnitKey, &accumulator)?;
        Ok(())
    }

    /// Get total sidechain wealth in Bitcoin
    pub fn sidechain_wealth(
        &self,
        rotxn: &RoTxn,
    ) -> Result<bitcoin::Amount, Error> {
        let mut total_deposit_utxo_value = bitcoin::Amount::ZERO;
        self.utxos.iter(rotxn)?.try_for_each(|utxo| {
            let (outpoint, output) = utxo?;
            if let OutPoint::Deposit(_) = outpoint {
                total_deposit_utxo_value = total_deposit_utxo_value
                    .checked_add(output.get_value())
                    .ok_or(AmountOverflowError)?;
            }
            Ok::<_, Error>(())
        })?;
        let mut total_deposit_stxo_value = bitcoin::Amount::ZERO;
        let mut total_withdrawal_stxo_value = bitcoin::Amount::ZERO;
        self.stxos.iter(rotxn)?.try_for_each(|stxo| {
            let (outpoint, spent_output) = stxo?;
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
}

impl Watchable<()> for State {
    type WatchStream = impl Stream<Item = ()>;

    /// Get a signal that notifies whenever the tip changes
    fn watch(&self) -> Self::WatchStream {
        tokio_stream::wrappers::WatchStream::new(self.tip.watch())
    }
}
