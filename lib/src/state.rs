use crate::authorization::Authorization;
use crate::types::*;
use bip300301::TwoWayPegData;
use bip300301::{bitcoin, WithdrawalBundleStatus};
pub use heed;
use heed::types::*;
use heed::{Database, RoTxn, RwTxn};
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;

#[derive(Clone)]
pub struct State {
    pub utxos: Database<SerdeBincode<OutPoint>, SerdeBincode<Output>>,
    pub pending_withdrawal_bundle: Database<OwnedType<u32>, SerdeBincode<WithdrawalBundle>>,
    pub last_withdrawal_bundle_failure_height: Database<OwnedType<u32>, OwnedType<u32>>,
    pub last_deposit_block: Database<OwnedType<u32>, SerdeBincode<bitcoin::BlockHash>>,
}

impl State {
    pub const NUM_DBS: u32 = 4;
    pub const WITHDRAWAL_BUNDLE_FAILURE_GAP: u32 = 4;

    pub fn new(env: &heed::Env) -> Result<Self, Error> {
        let utxos = env.create_database(Some("utxos"))?;

        let pending_withdrawal_bundle = env.create_database(Some("pending_withdrawal_bundle"))?;
        let last_withdrawal_bundle_failure_height =
            env.create_database(Some("last_withdrawal_bundle_failure_height"))?;
        let last_deposit_block = env.create_database(Some("last_deposit_block"))?;
        Ok(Self {
            utxos,
            pending_withdrawal_bundle,
            last_withdrawal_bundle_failure_height,
            last_deposit_block,
        })
    }

    pub fn get_utxos(&self, txn: &RoTxn) -> Result<HashMap<OutPoint, Output>, Error> {
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

    pub fn fill_transaction(
        &self,
        txn: &RoTxn,
        transaction: &Transaction,
    ) -> Result<FilledTransaction, Error> {
        let mut spent_utxos = vec![];
        for input in &transaction.inputs {
            let utxo = self
                .utxos
                .get(txn, input)?
                .ok_or(Error::NoUtxo { outpoint: *input })?;
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
        use bitcoin::blockdata::{opcodes, script};
        // Weight of a bundle with 0 outputs.
        const BUNDLE_0_WEIGHT: u64 = 504;
        // Weight of a single output.
        const OUTPUT_WEIGHT: u64 = 128;
        // Turns out to be 3121.
        const MAX_BUNDLE_OUTPUTS: usize = ((bitcoin::policy::MAX_STANDARD_TX_WEIGHT as u64
            - BUNDLE_0_WEIGHT)
            / OUTPUT_WEIGHT) as usize;

        // Aggregate all outputs by destination.
        // destination -> (value, mainchain fee, spent_utxos)
        let mut address_to_aggregated_withdrawal = HashMap::<
            bitcoin::Address<bitcoin::address::NetworkUnchecked>,
            AggregatedWithdrawal,
        >::new();
        for item in self.utxos.iter(txn)? {
            let (outpoint, output) = item?;
            if let Content::Withdrawal {
                value,
                ref main_address,
                main_fee,
            } = output.content
            {
                let aggregated = address_to_aggregated_withdrawal
                    .entry(main_address.clone())
                    .or_insert(AggregatedWithdrawal {
                        spent_utxos: HashMap::new(),
                        main_address: main_address.clone(),
                        value: 0,
                        main_fee: 0,
                    });
                // Add up all values.
                aggregated.value += value;
                // Set maximum mainchain fee.
                if main_fee > aggregated.main_fee {
                    aggregated.main_fee = main_fee;
                }
                aggregated.spent_utxos.insert(outpoint, output);
            }
        }
        if address_to_aggregated_withdrawal.is_empty() {
            return Ok(None);
        }
        let mut aggregated_withdrawals: Vec<_> =
            address_to_aggregated_withdrawal.into_values().collect();
        aggregated_withdrawals.sort_by_key(|a| std::cmp::Reverse(a.clone()));
        let mut fee = 0;
        let mut spent_utxos = HashMap::<OutPoint, Output>::new();
        let mut bundle_outputs = vec![];
        for aggregated in &aggregated_withdrawals {
            if bundle_outputs.len() > MAX_BUNDLE_OUTPUTS {
                break;
            }
            let bundle_output = bitcoin::TxOut {
                value: aggregated.value,
                script_pubkey: aggregated.main_address.payload.script_pubkey(),
            };
            spent_utxos.extend(aggregated.spent_utxos.clone());
            bundle_outputs.push(bundle_output);
            fee += aggregated.main_fee;
        }
        let txin = bitcoin::TxIn {
            script_sig: script::Builder::new()
                // OP_FALSE == OP_0
                .push_opcode(opcodes::OP_FALSE)
                .into_script(),
            ..bitcoin::TxIn::default()
        };
        // Create return dest output.
        // The destination string for the change of a WT^
        let script = script::Builder::new()
            .push_opcode(opcodes::all::OP_RETURN)
            .push_slice([68; 1])
            .into_script();
        let return_dest_txout = bitcoin::TxOut {
            value: 0,
            script_pubkey: script,
        };
        // Create mainchain fee output.
        let script = script::Builder::new()
            .push_opcode(opcodes::all::OP_RETURN)
            .push_slice(fee.to_le_bytes())
            .into_script();
        let mainchain_fee_txout = bitcoin::TxOut {
            value: 0,
            script_pubkey: script,
        };
        // Create inputs commitment.
        let inputs: Vec<OutPoint> = [
            // Commit to inputs.
            spent_utxos.keys().copied().collect(),
            // Commit to block height.
            vec![OutPoint::Regular {
                txid: [0; 32].into(),
                vout: block_height,
            }],
        ]
        .concat();
        let commitment = hash(&inputs);
        let script = script::Builder::new()
            .push_opcode(opcodes::all::OP_RETURN)
            .push_slice(&commitment)
            .into_script();
        let inputs_commitment_txout = bitcoin::TxOut {
            value: 0,
            script_pubkey: script,
        };
        let transaction = bitcoin::Transaction {
            version: 2,
            lock_time: bitcoin::blockdata::locktime::absolute::LockTime::ZERO,
            input: vec![txin],
            output: [
                vec![
                    return_dest_txout,
                    mainchain_fee_txout,
                    inputs_commitment_txout,
                ],
                bundle_outputs,
            ]
            .concat(),
        };
        if transaction.weight().to_wu() > bitcoin::policy::MAX_STANDARD_TX_WEIGHT as u64 {
            Err(Error::BundleTooHeavy {
                weight: transaction.weight().to_wu(),
                max_weight: bitcoin::policy::MAX_STANDARD_TX_WEIGHT as u64,
            })?;
        }
        Ok(Some(WithdrawalBundle {
            spent_utxos,
            transaction,
        }))
    }

    pub fn get_pending_withdrawal_bundle(
        &self,
        txn: &RoTxn,
    ) -> Result<Option<WithdrawalBundle>, Error> {
        Ok(self.pending_withdrawal_bundle.get(txn, &0)?)
    }

    pub fn validate_filled_transaction(
        &self,
        transaction: &FilledTransaction,
    ) -> Result<u64, Error> {
        let mut value_in: u64 = 0;
        let mut value_out: u64 = 0;
        for utxo in &transaction.spent_utxos {
            value_in += utxo.get_value();
        }
        for output in &transaction.transaction.outputs {
            value_out += output.get_value();
        }
        if value_out > value_in {
            return Err(Error::NotEnoughValueIn);
        }
        Ok(value_in - value_out)
    }

    pub fn body_sigops_limit(height: u32) -> usize {
        // starting body size limit is 8 * 1024 * 1024 B
        // 2 input 2 output transaction is 392 B
        // floor(8 * 1024 * 1024 B / 392 B) = 21399
        const START: usize = 21399;
        let month = height / (6 * 24 * 30);
        if month < 120 {
            (START as f64 * (1.04 as f64).powi(month as i32)).floor() as usize
        } else {
            START * 111
        }
    }

    // in bytes
    pub fn body_size_limit(height: u32) -> usize {
        const START: usize = 8 * 1024 * 1024;
        let month = height / (6 * 24 * 30);
        if month < 120 {
            (START as f64 * (1.04 as f64).powi(month as i32)).floor() as usize
        } else {
            START * 111
        }
    }

    pub fn validate_body(&self, txn: &RoTxn, body: &Body, height: u32) -> Result<u64, Error> {
        if body.authorizations.len() > Self::body_sigops_limit(height) {
            return Err(Error::TooManySigops);
        }
        if bincode::serialize(&body)?.len() > Self::body_size_limit(height) {
            return Err(Error::BodyTooLarge);
        }
        let mut coinbase_value: u64 = 0;
        for output in &body.coinbase {
            coinbase_value += output.get_value();
        }
        let mut total_fees: u64 = 0;
        let mut spent_utxos = HashSet::new();
        let filled_transactions: Vec<_> = body
            .transactions
            .iter()
            .map(|t| self.fill_transaction(txn, t))
            .collect::<Result<_, _>>()?;
        for filled_transaction in &filled_transactions {
            for input in &filled_transaction.transaction.inputs {
                if spent_utxos.contains(input) {
                    return Err(Error::UtxoDoubleSpent);
                }
                spent_utxos.insert(*input);
            }
            total_fees += self.validate_filled_transaction(filled_transaction)?;
        }
        if coinbase_value > total_fees {
            return Err(Error::NotEnoughFees);
        }
        let spent_utxos = filled_transactions
            .iter()
            .flat_map(|t| t.spent_utxos.iter());
        for (authorization, spent_utxo) in body.authorizations.iter().zip(spent_utxos) {
            if authorization.get_address() != spent_utxo.address {
                return Err(Error::WrongPubKeyForAddress);
            }
        }
        if Authorization::verify_body(body).is_err() {
            return Err(Error::AuthorizationError);
        }
        Ok(total_fees)
    }

    pub fn get_last_deposit_block_hash(
        &self,
        txn: &RoTxn,
    ) -> Result<Option<bitcoin::BlockHash>, Error> {
        Ok(self.last_deposit_block.get(&txn, &0)?)
    }

    pub fn connect_two_way_peg_data(
        &self,
        txn: &mut RwTxn,
        two_way_peg_data: &TwoWayPegData,
        block_height: u32,
    ) -> Result<(), Error> {
        // Handle deposits.
        if let Some(deposit_block_hash) = two_way_peg_data.deposit_block_hash {
            self.last_deposit_block.put(txn, &0, &deposit_block_hash)?;
        }
        for (outpoint, deposit) in &two_way_peg_data.deposits {
            if let Ok(address) = deposit.address.parse() {
                let outpoint = OutPoint::Deposit(*outpoint);
                let output = Output {
                    address,
                    content: Content::Value(deposit.value),
                };
                self.utxos.put(txn, &outpoint, &output)?;
            }
        }

        // Handle withdrawals.
        let last_withdrawal_bundle_failure_height = self
            .last_withdrawal_bundle_failure_height
            .get(txn, &0)?
            .unwrap_or(0);
        if (block_height + 1) - last_withdrawal_bundle_failure_height
            > Self::WITHDRAWAL_BUNDLE_FAILURE_GAP
            && self.pending_withdrawal_bundle.get(txn, &0)?.is_none()
        {
            if let Some(bundle) = self.collect_withdrawal_bundle(txn, block_height + 1)? {
                for outpoint in bundle.spent_utxos.keys() {
                    self.utxos.delete(txn, outpoint)?;
                }
                self.pending_withdrawal_bundle.put(txn, &0, &bundle)?;
            }
        }
        for (txid, status) in &two_way_peg_data.bundle_statuses {
            if let Some(bundle) = self.pending_withdrawal_bundle.get(txn, &0)? {
                if bundle.transaction.txid() != *txid {
                    continue;
                }
                match status {
                    WithdrawalBundleStatus::Failed => {
                        self.last_withdrawal_bundle_failure_height.put(
                            txn,
                            &0,
                            &(block_height + 1),
                        )?;
                        self.pending_withdrawal_bundle.delete(txn, &0)?;
                        for (outpoint, output) in &bundle.spent_utxos {
                            self.utxos.put(txn, outpoint, output)?;
                        }
                    }
                    WithdrawalBundleStatus::Confirmed => {
                        self.pending_withdrawal_bundle.delete(txn, &0)?;
                    }
                }
            }
        }
        Ok(())
    }

    pub fn connect_body(&self, txn: &mut RwTxn, body: &Body) -> Result<(), Error> {
        let merkle_root = body.compute_merkle_root();
        for (vout, output) in body.coinbase.iter().enumerate() {
            let outpoint = OutPoint::Coinbase {
                merkle_root,
                vout: vout as u32,
            };
            self.utxos.put(txn, &outpoint, output)?;
        }
        for transaction in &body.transactions {
            let txid = transaction.txid();
            for input in &transaction.inputs {
                self.utxos.delete(txn, input)?;
            }
            for (vout, output) in transaction.outputs.iter().enumerate() {
                let outpoint = OutPoint::Regular {
                    txid,
                    vout: vout as u32,
                };
                self.utxos.put(txn, &outpoint, output)?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("failed to verify authorization")]
    AuthorizationError,
    #[error("heed error")]
    Heed(#[from] heed::Error),
    #[error("binvode error")]
    Bincode(#[from] bincode::Error),
    #[error("utxo {outpoint} doesn't exist")]
    NoUtxo { outpoint: OutPoint },
    #[error("value in is less than value out")]
    NotEnoughValueIn,
    #[error("total fees less than coinbase value")]
    NotEnoughFees,
    #[error("utxo double spent")]
    UtxoDoubleSpent,
    #[error("wrong public key for address")]
    WrongPubKeyForAddress,
    #[error("bundle too heavy {weight} > {max_weight}")]
    BundleTooHeavy { weight: u64, max_weight: u64 },
    #[error("too many sigops")]
    TooManySigops,
    #[error("body too large")]
    BodyTooLarge,
}
