use std::collections::{HashMap, HashSet};

use heed::{
    types::{OwnedType, SerdeBincode},
    Database, RoTxn, RwTxn,
};

use bip300301::{
    bitcoin::{
        self, transaction::Version as BitcoinTxVersion, Amount as BitcoinAmount,
    },
    TwoWayPegData, WithdrawalBundleStatus,
};
use rustreexo::accumulator::{
    node_hash::NodeHash, pollard::Pollard, proof::Proof,
};
use serde::{Deserialize, Serialize};

use crate::{
    authorization::Authorization,
    types::{
        hash, Address, AggregatedWithdrawal, Body, FilledTransaction,
        GetAddress, GetValue, InPoint, OutPoint, Output, OutputContent,
        PointedOutput, SpentOutput, Transaction, Txid, Verify,
        WithdrawalBundle,
    },
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("failed to verify authorization")]
    AuthorizationError,
    #[error("binvode error")]
    Bincode(#[from] bincode::Error),
    #[error("body too large")]
    BodyTooLarge,
    #[error("bundle too heavy {weight} > {max_weight}")]
    BundleTooHeavy { weight: u64, max_weight: u64 },
    #[error("heed error")]
    Heed(#[from] heed::Error),
    #[error("total fees less than coinbase value")]
    NotEnoughFees,
    #[error("value in is less than value out")]
    NotEnoughValueIn,
    #[error("utxo {outpoint} doesn't exist")]
    NoUtxo { outpoint: OutPoint },
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
    #[error("wrong public key for address")]
    WrongPubKeyForAddress,
}

/// Unit key. LMDB can't use zero-sized keys, so this encodes to a single byte
#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd, Ord)]
pub struct UnitKey;

impl<'de> Deserialize<'de> for UnitKey {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // Deserialize any byte (ignoring it) and return UnitKey
        let _ = u8::deserialize(deserializer)?;
        Ok(UnitKey)
    }
}

impl Serialize for UnitKey {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // Always serialize to the same arbitrary byte
        serializer.serialize_u8(0x69)
    }
}

#[derive(Debug, Default)]
#[repr(transparent)]
pub struct Accumulator(pub Pollard);

impl<'de> Deserialize<'de> for Accumulator {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let bytes: &[u8] = <&[u8] as Deserialize>::deserialize(deserializer)?;
        let pollard = Pollard::deserialize(bytes)
            .map_err(<D::Error as serde::de::Error>::custom)?;
        Ok(Self(pollard))
    }
}

impl Serialize for Accumulator {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut bytes = Vec::new();
        self.0
            .serialize(&mut bytes)
            .map_err(<S::Error as serde::ser::Error>::custom)?;
        bytes.serialize(serializer)
    }
}

#[derive(Clone)]
pub struct State {
    pub utxos: Database<SerdeBincode<OutPoint>, SerdeBincode<Output>>,
    pub stxos: Database<SerdeBincode<OutPoint>, SerdeBincode<SpentOutput>>,
    pub pending_withdrawal_bundle:
        Database<OwnedType<u32>, SerdeBincode<WithdrawalBundle>>,
    pub last_withdrawal_bundle_failure_height:
        Database<OwnedType<u32>, OwnedType<u32>>,
    pub last_deposit_block:
        Database<OwnedType<u32>, SerdeBincode<bitcoin::BlockHash>>,
    pub utreexo_accumulator:
        Database<SerdeBincode<UnitKey>, SerdeBincode<Accumulator>>,
}

impl State {
    pub const NUM_DBS: u32 = 6;
    pub const WITHDRAWAL_BUNDLE_FAILURE_GAP: u32 = 4;

    pub fn new(env: &heed::Env) -> Result<Self, Error> {
        let utxos = env.create_database(Some("utxos"))?;
        let stxos = env.create_database(Some("stxos"))?;
        let pending_withdrawal_bundle =
            env.create_database(Some("pending_withdrawal_bundle"))?;
        let last_withdrawal_bundle_failure_height =
            env.create_database(Some("last_withdrawal_bundle_failure_height"))?;
        let last_deposit_block =
            env.create_database(Some("last_deposit_block"))?;
        let utreexo_accumulator =
            env.create_database(Some("utreexo_accumulator"))?;
        Ok(Self {
            utxos,
            stxos,
            pending_withdrawal_bundle,
            last_withdrawal_bundle_failure_height,
            last_deposit_block,
            utreexo_accumulator,
        })
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

    /// Get the current Utreexo accumulator
    pub fn get_accumulator(&self, rotxn: &RoTxn) -> Result<Pollard, Error> {
        let accumulator = self
            .utreexo_accumulator
            .get(rotxn, &UnitKey)?
            .unwrap_or_default()
            .0;
        Ok(accumulator)
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
        let (proof, _) = accumulator.prove(&targets).map_err(Error::Utreexo)?;
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
        use bitcoin::blockdata::{opcodes, script};
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
                        value: 0,
                        main_fee: 0,
                    });
                // Add up all values.
                aggregated.value += value;
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
        let mut fee = 0;
        let mut spend_utxos = HashMap::<OutPoint, Output>::new();
        let mut bundle_outputs = vec![];
        for aggregated in &aggregated_withdrawals {
            if bundle_outputs.len() > MAX_BUNDLE_OUTPUTS {
                break;
            }
            let bundle_output = bitcoin::TxOut {
                value: BitcoinAmount::from_sat(aggregated.value),
                script_pubkey: aggregated
                    .main_address
                    .payload()
                    .script_pubkey(),
            };
            spend_utxos.extend(aggregated.spend_utxos.clone());
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
            value: BitcoinAmount::ZERO,
            script_pubkey: script,
        };
        // Create mainchain fee output.
        let script = script::Builder::new()
            .push_opcode(opcodes::all::OP_RETURN)
            .push_slice(fee.to_le_bytes())
            .into_script();
        let mainchain_fee_txout = bitcoin::TxOut {
            value: BitcoinAmount::ZERO,
            script_pubkey: script,
        };
        // Create inputs commitment.
        let inputs: Vec<OutPoint> = [
            // Commit to inputs.
            spend_utxos.keys().copied().collect(),
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
            .push_slice(commitment)
            .into_script();
        let inputs_commitment_txout = bitcoin::TxOut {
            value: BitcoinAmount::ZERO,
            script_pubkey: script,
        };
        let transaction = bitcoin::Transaction {
            version: BitcoinTxVersion::TWO,
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
        if transaction.weight().to_wu()
            > bitcoin::policy::MAX_STANDARD_TX_WEIGHT as u64
        {
            Err(Error::BundleTooHeavy {
                weight: transaction.weight().to_wu(),
                max_weight: bitcoin::policy::MAX_STANDARD_TX_WEIGHT as u64,
            })?;
        }
        Ok(Some(WithdrawalBundle {
            spend_utxos,
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

    pub fn validate_body(
        &self,
        txn: &RoTxn,
        // Utreexo roots from the block header
        header_utreexo_roots: &[NodeHash],
        body: &Body,
        height: u32,
    ) -> Result<u64, Error> {
        if body.authorizations.len() > Self::body_sigops_limit(height) {
            return Err(Error::TooManySigops);
        }
        if bincode::serialize(&body)?.len() > Self::body_size_limit(height) {
            return Err(Error::BodyTooLarge);
        }
        let mut accumulator = self
            .utreexo_accumulator
            .get(txn, &UnitKey)?
            .unwrap_or_default();
        // New leaves for the accumulator
        let mut accumulator_add = Vec::<NodeHash>::new();
        // Accumulator leaves to delete
        let mut accumulator_del = Vec::<NodeHash>::new();
        let mut coinbase_value: u64 = 0;
        let merkle_root = body.compute_merkle_root();
        for (vout, output) in body.coinbase.iter().enumerate() {
            coinbase_value += output.get_value();
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
        let mut total_fees: u64 = 0;
        let mut spent_utxos = HashSet::new();
        let filled_transactions: Vec<_> = body
            .transactions
            .iter()
            .map(|t| self.fill_transaction(txn, t))
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
            total_fees +=
                self.validate_filled_transaction(filled_transaction)?;
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
        if roots != header_utreexo_roots {
            return Err(Error::UtreexoRootsMismatch);
        }
        Ok(total_fees)
    }

    pub fn get_last_deposit_block_hash(
        &self,
        txn: &RoTxn,
    ) -> Result<Option<bitcoin::BlockHash>, Error> {
        Ok(self.last_deposit_block.get(txn, &0)?)
    }

    pub fn connect_two_way_peg_data(
        &self,
        txn: &mut RwTxn,
        two_way_peg_data: &TwoWayPegData,
        block_height: u32,
    ) -> Result<(), Error> {
        let mut accumulator = self
            .utreexo_accumulator
            .get(txn, &UnitKey)?
            .unwrap_or_default();
        // New leaves for the accumulator
        let mut accumulator_add = Vec::<NodeHash>::new();
        // Accumulator leaves to delete
        let mut accumulator_del = HashSet::<NodeHash>::new();
        // Handle deposits.
        if let Some(deposit_block_hash) = two_way_peg_data.deposit_block_hash {
            self.last_deposit_block.put(txn, &0, &deposit_block_hash)?;
        }
        for (outpoint, deposit) in &two_way_peg_data.deposits {
            if let Ok(address) = deposit.address.parse() {
                let outpoint = OutPoint::Deposit(*outpoint);
                let output = Output {
                    address,
                    content: OutputContent::Value(deposit.value),
                };
                self.utxos.put(txn, &outpoint, &output)?;
                let utxo_hash = hash(&PointedOutput { outpoint, output });
                accumulator_add.push(utxo_hash.into())
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
            if let Some(bundle) =
                self.collect_withdrawal_bundle(txn, block_height + 1)?
            {
                for (outpoint, spend_output) in &bundle.spend_utxos {
                    let utxo_hash = hash(&PointedOutput {
                        outpoint: *outpoint,
                        output: spend_output.clone(),
                    });
                    accumulator_del.insert(utxo_hash.into());
                    self.utxos.delete(txn, outpoint)?;
                    let txid = bundle.transaction.txid();
                    let spent_output = SpentOutput {
                        output: spend_output.clone(),
                        inpoint: InPoint::Withdrawal { txid },
                    };
                    self.stxos.put(txn, outpoint, &spent_output)?;
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
                        for (outpoint, output) in &bundle.spend_utxos {
                            self.stxos.delete(txn, outpoint)?;
                            self.utxos.put(txn, outpoint, output)?;
                            let utxo_hash = hash(&PointedOutput {
                                outpoint: *outpoint,
                                output: output.clone(),
                            });
                            accumulator_del.remove(&utxo_hash.into());
                        }
                    }
                    WithdrawalBundleStatus::Confirmed => {
                        self.pending_withdrawal_bundle.delete(txn, &0)?;
                    }
                }
            }
        }
        let accumulator_del: Vec<_> = accumulator_del.into_iter().collect();
        accumulator
            .0
            .modify(&accumulator_add, &accumulator_del)
            .map_err(Error::Utreexo)?;
        self.utreexo_accumulator.put(txn, &UnitKey, &accumulator)?;
        Ok(())
    }

    pub fn connect_body(
        &self,
        txn: &mut RwTxn,
        body: &Body,
    ) -> Result<(), Error> {
        let merkle_root = body.compute_merkle_root();
        let mut accumulator = self
            .utreexo_accumulator
            .get(txn, &UnitKey)?
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
            self.utxos.put(txn, &outpoint, output)?;
        }
        for transaction in &body.transactions {
            let txid = transaction.txid();
            for (vin, (outpoint, utxo_hash)) in
                transaction.inputs.iter().enumerate()
            {
                let spent_output =
                    self.utxos.get(txn, outpoint)?.ok_or(Error::NoUtxo {
                        outpoint: *outpoint,
                    })?;
                accumulator_del.push(utxo_hash.into());
                self.utxos.delete(txn, outpoint)?;
                let spent_output = SpentOutput {
                    output: spent_output,
                    inpoint: InPoint::Regular {
                        txid,
                        vin: vin as u32,
                    },
                };
                self.stxos.put(txn, outpoint, &spent_output)?;
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
                self.utxos.put(txn, &outpoint, output)?;
            }
        }
        accumulator
            .0
            .modify(&accumulator_add, &accumulator_del)
            .map_err(Error::Utreexo)?;
        self.utreexo_accumulator.put(txn, &UnitKey, &accumulator)?;
        Ok(())
    }

    /// Get total sidechain wealth in Bitcoin
    pub fn sidechain_wealth(
        &self,
        rotxn: &RoTxn,
    ) -> Result<BitcoinAmount, Error> {
        let mut total_deposit_utxo_value: u64 = 0;
        self.utxos.iter(rotxn)?.try_for_each(|utxo| {
            let (outpoint, output) = utxo?;
            if let OutPoint::Deposit(_) = outpoint {
                total_deposit_utxo_value += output.get_value();
            }
            Ok::<_, Error>(())
        })?;
        let mut total_deposit_stxo_value: u64 = 0;
        let mut total_withdrawal_stxo_value: u64 = 0;
        self.stxos.iter(rotxn)?.try_for_each(|stxo| {
            let (outpoint, spent_output) = stxo?;
            if let OutPoint::Deposit(_) = outpoint {
                total_deposit_stxo_value += spent_output.output.get_value();
            }
            if let InPoint::Withdrawal { .. } = spent_output.inpoint {
                total_withdrawal_stxo_value += spent_output.output.get_value();
            }
            Ok::<_, Error>(())
        })?;

        let total_wealth_sats: u64 = (total_deposit_utxo_value
            + total_deposit_stxo_value)
            - total_withdrawal_stxo_value;
        let total_wealth = BitcoinAmount::from_sat(total_wealth_sats);
        Ok(total_wealth)
    }
}
