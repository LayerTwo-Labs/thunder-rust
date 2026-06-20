//! Connect and disconnect blocks

use rustreexo::accumulator::node_hash::BitcoinNodeHash;
use sneed::{RoTxn, RwTxn, db::error::Error as DbError};

use crate::{
    authorization::Authorization,
    state::{Error, PrevalidatedBlock, State, error},
    types::{
        AccumulatorDiff, AmountOverflowError, Body, FilledTransaction,
        GetAddress as _, GetValue as _, Header, InPoint, MerkleRoot, OutPoint,
        OutPointKey, PointedOutput, SpentOutput, Verify as _,
    },
};

/// Prevalidate a block: compute and verify all read-only checks and
/// prepare data needed for fast connection.
pub fn prevalidate(
    state: &State,
    rotxn: &RoTxn,
    header: &Header,
    body: &Body,
) -> Result<PrevalidatedBlock, Error> {
    let tip_hash = state.try_get_tip(rotxn)?;
    if header.prev_side_hash != tip_hash {
        let err = error::InvalidHeader::PrevSideHash {
            expected: tip_hash,
            received: header.prev_side_hash,
        };
        return Err(Error::InvalidHeader(err));
    };
    let next_height = state.try_get_height(rotxn)?.map_or(0, |h| h + 1);
    if body.authorizations.len() > State::body_sigops_limit(next_height) {
        return Err(Error::TooManySigops);
    }
    let body_size =
        borsh::object_length(&body).map_err(Error::BorshSerialize)?;
    if body_size > State::body_size_limit(next_height) {
        return Err(Error::BodyTooLarge);
    }

    let mut accumulator = state
        .utreexo_accumulator
        .try_get(rotxn, &())
        .map_err(DbError::from)?
        .unwrap_or_default();

    // gather and verify transactions
    let mut total_inputs: usize = 0;
    let mut total_outputs: usize = 0;
    for tx in &body.transactions {
        total_inputs += tx.inputs.len();
        total_outputs += tx.outputs.len();
    }
    let mut all_input_keys: Vec<OutPointKey> = Vec::with_capacity(total_inputs);
    // Accumulator diff from txs. The bool value is true for insertions, false
    // for deletions.
    let mut accumulator_diff_txs =
        Vec::with_capacity(total_inputs + total_outputs);
    let mut filled_transactions: Vec<FilledTransaction> =
        Vec::with_capacity(body.transactions.len());
    let mut total_fees = bitcoin::Amount::ZERO;
    for transaction in &body.transactions {
        let txid = transaction.txid();
        let mut spent_utxos = Vec::with_capacity(transaction.inputs.len());
        let mut spent_utxo_hashes =
            Vec::<BitcoinNodeHash>::with_capacity(transaction.inputs.len());
        for (outpoint, utxo_hash) in &transaction.inputs {
            let key = OutPointKey::from(outpoint);
            let spent_output =
                state.utxos.try_get(rotxn, &key)?.ok_or(error::NoUtxo {
                    outpoint: *outpoint,
                })?;
            all_input_keys.push(OutPointKey::from(outpoint));
            spent_utxos.push(spent_output);
            spent_utxo_hashes.push(utxo_hash.into());
            accumulator_diff_txs.push((false, utxo_hash.into()));
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
            accumulator_diff_txs.push((true, (&pointed_output).into()));
        }
        if !accumulator.verify(&transaction.proof, &spent_utxo_hashes)? {
            return Err(Error::UtreexoProofFailed { txid });
        }
        let filled_tx = FilledTransaction {
            spent_utxos,
            transaction: transaction.clone(),
        };
        total_fees = total_fees
            .checked_add(state.validate_filled_transaction(&filled_tx)?)
            .ok_or(AmountOverflowError)?;
        filled_transactions.push(filled_tx);
    }
    let computed_merkle_root = Body::compute_merkle_root(
        body.coinbase.as_slice(),
        filled_transactions.as_slice(),
    )?;
    if computed_merkle_root != header.merkle_root {
        let err = Error::InvalidBody {
            expected: header.merkle_root,
            computed: computed_merkle_root,
        };
        return Err(err);
    }
    {
        use rayon::prelude::ParallelSliceMut;
        all_input_keys.par_sort_unstable();
        if all_input_keys.windows(2).any(|w| w[0] == w[1]) {
            return Err(Error::UtxoDoubleSpent);
        }
    }
    let mut coinbase_value = bitcoin::Amount::ZERO;
    let mut accumulator_diff = AccumulatorDiff::with_capacity(
        body.coinbase.len() + accumulator_diff_txs.len(),
    );
    for (vout, output) in body.coinbase.iter().enumerate() {
        coinbase_value = coinbase_value
            .checked_add(output.get_value())
            .ok_or(AmountOverflowError)?;
        let outpoint = OutPoint::Coinbase {
            merkle_root: computed_merkle_root,
            vout: vout as u32,
        };
        let pointed_output = PointedOutput {
            outpoint,
            output: output.clone(),
        };
        accumulator_diff.insert((&pointed_output).into());
    }
    for (insert, utxo_hash) in accumulator_diff_txs {
        if insert {
            accumulator_diff.insert(utxo_hash);
        } else {
            accumulator_diff.remove(utxo_hash);
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
    let () = Authorization::verify_body(body)?;
    // Check root consistency without committing to DB
    let () = accumulator.apply_diff(accumulator_diff.clone())?;
    let roots: Vec<BitcoinNodeHash> = accumulator.get_roots();
    if roots != header.roots {
        return Err(Error::UtreexoRootsMismatch);
    }

    Ok(PrevalidatedBlock {
        filled_transactions,
        computed_merkle_root,
        total_fees,
        coinbase_value,
        next_height,
        accumulator_diff,
    })
}

/// Connect a block using the provided prevalidated data.
pub fn connect_prevalidated(
    state: &State,
    rwtxn: &mut RwTxn,
    header: &Header,
    body: &Body,
    pre: PrevalidatedBlock,
) -> Result<crate::types::MerkleRoot, Error> {
    let tip_hash = state.try_get_tip(rwtxn)?;
    if tip_hash != header.prev_side_hash {
        let err = error::InvalidHeader::PrevSideHash {
            expected: tip_hash,
            received: header.prev_side_hash,
        };
        return Err(Error::InvalidHeader(err));
    }
    if pre.computed_merkle_root != header.merkle_root {
        let err = Error::InvalidBody {
            expected: pre.computed_merkle_root,
            computed: header.merkle_root,
        };
        return Err(err);
    }

    // Apply UTXO set changes
    for (vout, output) in body.coinbase.iter().enumerate() {
        let outpoint = OutPoint::Coinbase {
            merkle_root: pre.computed_merkle_root,
            vout: vout as u32,
        };
        state
            .utxos
            .put(rwtxn, &OutPointKey::from(&outpoint), output)
            .map_err(DbError::from)?;
    }

    for filled in &pre.filled_transactions {
        let txid = filled.transaction.txid();
        for (vin, (outpoint, _)) in filled.transaction.inputs.iter().enumerate()
        {
            let spent_output = state
                .utxos
                .try_get(rwtxn, &OutPointKey::from(outpoint))
                .map_err(DbError::from)?
                .ok_or(error::NoUtxo {
                    outpoint: *outpoint,
                })?;
            state
                .utxos
                .delete(rwtxn, &OutPointKey::from(outpoint))
                .map_err(DbError::from)?;
            let spent_output = SpentOutput {
                output: spent_output,
                inpoint: InPoint::Regular {
                    txid,
                    vin: vin as u32,
                },
            };
            state
                .stxos
                .put(rwtxn, &OutPointKey::from(outpoint), &spent_output)
                .map_err(DbError::from)?;
        }
        for (vout, output) in filled.transaction.outputs.iter().enumerate() {
            let outpoint = OutPoint::Regular {
                txid,
                vout: vout as u32,
            };
            state
                .utxos
                .put(rwtxn, &OutPointKey::from(&outpoint), output)
                .map_err(DbError::from)?;
        }
    }

    // Update tip/height
    let block_hash = header.hash();
    state
        .tip
        .put(rwtxn, &(), &block_hash)
        .map_err(DbError::from)?;
    state
        .height
        .put(rwtxn, &(), &pre.next_height)
        .map_err(DbError::from)?;

    // Apply accumulator diff
    let mut accumulator = state
        .utreexo_accumulator
        .try_get(rwtxn, &())
        .map_err(DbError::from)?
        .unwrap_or_default();
    let () = accumulator.apply_diff(pre.accumulator_diff)?;
    state
        .utreexo_accumulator
        .put(rwtxn, &(), &accumulator)
        .map_err(DbError::from)?;

    Ok(pre.computed_merkle_root)
}

pub fn validate(
    state: &State,
    rotxn: &RoTxn,
    header: &Header,
    body: &Body,
) -> Result<(bitcoin::Amount, MerkleRoot), Error> {
    let tip_hash = state.try_get_tip(rotxn)?;
    if header.prev_side_hash != tip_hash {
        let err = error::InvalidHeader::PrevSideHash {
            expected: tip_hash,
            received: header.prev_side_hash,
        };
        return Err(Error::InvalidHeader(err));
    };
    let height = state.try_get_height(rotxn)?.map_or(0, |height| height + 1);
    if body.authorizations.len() > State::body_sigops_limit(height) {
        return Err(Error::TooManySigops);
    }
    let body_size =
        borsh::object_length(&body).map_err(Error::BorshSerialize)?;
    if body_size > State::body_size_limit(height) {
        return Err(Error::BodyTooLarge);
    }
    let mut accumulator = state
        .utreexo_accumulator
        .try_get(rotxn, &())
        .map_err(DbError::from)?
        .unwrap_or_default();
    let filled_transactions: Vec<_> = body
        .transactions
        .iter()
        .map(|t| state.fill_transaction(rotxn, t))
        .collect::<Result<_, _>>()?;
    let merkle_root = Body::compute_merkle_root(
        body.coinbase.as_slice(),
        filled_transactions.as_slice(),
    )?;
    if merkle_root != header.merkle_root {
        let err = Error::InvalidBody {
            expected: header.merkle_root,
            computed: merkle_root,
        };
        return Err(err);
    }
    let mut accumulator_diff = AccumulatorDiff::default();
    let mut coinbase_value = bitcoin::Amount::ZERO;
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
        accumulator_diff.insert((&pointed_output).into());
    }
    let mut total_fees = bitcoin::Amount::ZERO;
    // Gather all input keys to check double-spends via sort-and-scan
    let total_inputs = body.inputs_len();
    let mut all_input_keys = Vec::with_capacity(total_inputs);
    for filled_transaction in &filled_transactions {
        let txid = filled_transaction.transaction.txid();
        // hashes of spent utxos, used to verify the utreexo proof
        let mut spent_utxo_hashes = Vec::<BitcoinNodeHash>::with_capacity(
            filled_transaction.transaction.inputs.len(),
        );
        for (outpoint, utxo_hash) in &filled_transaction.transaction.inputs {
            all_input_keys.push(OutPointKey::from(outpoint));
            spent_utxo_hashes.push(utxo_hash.into());
            accumulator_diff.remove(utxo_hash.into());
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
            accumulator_diff.insert((&pointed_output).into());
        }
        total_fees = total_fees
            .checked_add(state.validate_filled_transaction(filled_transaction)?)
            .ok_or(AmountOverflowError)?;
        // verify utreexo proof
        if !accumulator
            .verify(&filled_transaction.transaction.proof, &spent_utxo_hashes)?
        {
            return Err(Error::UtreexoProofFailed { txid });
        }
    }
    // Sort and check for duplicate outpoints (double-spend detection)
    {
        use rayon::prelude::ParallelSliceMut;
        all_input_keys.par_sort_unstable();
        if all_input_keys.windows(2).any(|w| w[0] == w[1]) {
            return Err(Error::UtxoDoubleSpent);
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
    let () = Authorization::verify_body(body)?;
    // Check root consistency without committing to DB
    let () = accumulator.apply_diff(accumulator_diff)?;
    let roots: Vec<BitcoinNodeHash> = accumulator.get_roots();
    if roots != header.roots {
        return Err(Error::UtreexoRootsMismatch);
    }
    Ok((total_fees, merkle_root))
}

pub fn connect(
    state: &State,
    rwtxn: &mut RwTxn,
    header: &Header,
    body: &Body,
) -> Result<MerkleRoot, Error> {
    let tip_hash = state.try_get_tip(rwtxn)?;
    if tip_hash != header.prev_side_hash {
        let err = error::InvalidHeader::PrevSideHash {
            expected: tip_hash,
            received: header.prev_side_hash,
        };
        return Err(Error::InvalidHeader(err));
    }
    let mut accumulator = state
        .utreexo_accumulator
        .try_get(rwtxn, &())?
        .unwrap_or_default();
    let mut accumulator_diff = AccumulatorDiff::default();
    for (vout, output) in body.coinbase.iter().enumerate() {
        let outpoint = OutPoint::Coinbase {
            merkle_root: header.merkle_root,
            vout: vout as u32,
        };
        let pointed_output = PointedOutput {
            outpoint,
            output: output.clone(),
        };
        accumulator_diff.insert((&pointed_output).into());
        let key = OutPointKey::from(&outpoint);
        state.utxos.put(rwtxn, &key, output)?;
    }
    let mut filled_txs: Vec<FilledTransaction> =
        Vec::with_capacity(body.transactions.len());
    for transaction in &body.transactions {
        let mut spent_utxos = Vec::with_capacity(transaction.inputs.len());
        let txid = transaction.txid();
        for (vin, (outpoint, utxo_hash)) in
            transaction.inputs.iter().enumerate()
        {
            let key = OutPointKey::from(outpoint);
            let spent_output =
                state.utxos.try_get(rwtxn, &key)?.ok_or(error::NoUtxo {
                    outpoint: *outpoint,
                })?;

            accumulator_diff.remove(utxo_hash.into());
            let key = OutPointKey::from(outpoint);
            state.utxos.delete(rwtxn, &key)?;
            let spent_output = SpentOutput {
                output: spent_output,
                inpoint: InPoint::Regular {
                    txid,
                    vin: vin as u32,
                },
            };
            state
                .stxos
                .put(rwtxn, &OutPointKey::from(outpoint), &spent_output)
                .map_err(DbError::from)?;
            spent_utxos.push(spent_output.output);
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
            accumulator_diff.insert((&pointed_output).into());
            let key = OutPointKey::from(&outpoint);
            state.utxos.put(rwtxn, &key, output)?;
        }
        let filled_tx = FilledTransaction {
            spent_utxos,
            transaction: transaction.clone(),
        };
        filled_txs.push(filled_tx);
    }
    let merkle_root = Body::compute_merkle_root(
        body.coinbase.as_slice(),
        filled_txs.as_slice(),
    )?;
    if merkle_root != header.merkle_root {
        let err = Error::InvalidBody {
            expected: header.merkle_root,
            computed: merkle_root,
        };
        return Err(err);
    }
    let block_hash = header.hash();
    let height = state.try_get_height(rwtxn)?.map_or(0, |height| height + 1);
    state.tip.put(rwtxn, &(), &block_hash)?;
    state.height.put(rwtxn, &(), &height)?;
    let () = accumulator.apply_diff(accumulator_diff)?;
    state.utreexo_accumulator.put(rwtxn, &(), &accumulator)?;
    Ok(merkle_root)
}

pub fn disconnect_tip(
    state: &State,
    rwtxn: &mut RwTxn,
    header: &Header,
    body: &Body,
) -> Result<(), Error> {
    let tip_hash = state
        .tip
        .try_get(rwtxn, &())
        .map_err(DbError::from)?
        .ok_or(Error::NoTip)?;
    if tip_hash != header.hash() {
        let err = error::InvalidHeader::BlockHash {
            expected: tip_hash,
            computed: header.hash(),
        };
        return Err(Error::InvalidHeader(err));
    }
    let mut accumulator = state
        .utreexo_accumulator
        .try_get(rwtxn, &())
        .map_err(DbError::from)?
        .unwrap_or_default();
    tracing::debug!("Got acc");
    let mut accumulator_diff = AccumulatorDiff::default();
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
                accumulator_diff.remove((&pointed_output).into());
                let key = OutPointKey::from(&outpoint);
                if state.utxos.delete(rwtxn, &key)? {
                    Ok::<_, Error>(())
                } else {
                    Err(error::NoUtxo { outpoint }.into())
                }
            },
        )?;
        // unspend STXOs, last-to-first
        tx.inputs
            .iter()
            .rev()
            .try_for_each(|(outpoint, utxo_hash)| {
                let key = OutPointKey::from(outpoint);
                if let Some(spent_output) = state.stxos.try_get(rwtxn, &key)? {
                    accumulator_diff.insert(utxo_hash.into());
                    state.stxos.delete(rwtxn, &key)?;
                    state.utxos.put(rwtxn, &key, &spent_output.output)?;
                    Ok(())
                } else {
                    Err(Error::NoStxo {
                        outpoint: *outpoint,
                    })
                }
            })
    })?;
    // delete coinbase UTXOs, last-to-first
    body.coinbase
        .iter()
        .enumerate()
        .rev()
        .try_for_each(|(vout, output)| {
            let outpoint = OutPoint::Coinbase {
                merkle_root: header.merkle_root,
                vout: vout as u32,
            };
            let pointed_output = PointedOutput {
                outpoint,
                output: output.clone(),
            };
            accumulator_diff.remove((&pointed_output).into());
            let key = OutPointKey::from(&outpoint);
            if state.utxos.delete(rwtxn, &key)? {
                Ok::<_, Error>(())
            } else {
                Err(error::NoUtxo { outpoint }.into())
            }
        })?;
    let height = state
        .try_get_height(rwtxn)?
        .expect("Height should not be None");
    match (header.prev_side_hash, height) {
        (None, 0) => {
            state.tip.delete(rwtxn, &()).map_err(DbError::from)?;
            state.height.delete(rwtxn, &()).map_err(DbError::from)?;
        }
        (None, _) | (_, 0) => return Err(Error::NoTip),
        (Some(prev_side_hash), height) => {
            state
                .tip
                .put(rwtxn, &(), &prev_side_hash)
                .map_err(DbError::from)?;
            state
                .height
                .put(rwtxn, &(), &(height - 1))
                .map_err(DbError::from)?;
        }
    }
    let () = accumulator.apply_diff(accumulator_diff)?;
    state
        .utreexo_accumulator
        .put(rwtxn, &(), &accumulator)
        .map_err(DbError::from)?;
    Ok(())
}

#[cfg(test)]
mod test {
    use crate::state::test::{fresh_state, value_output};

    #[test]
    fn validation_rejects_outpoint_utxo_hash_mismatch() -> anyhow::Result<()> {
        use bitcoin::hashes::Hash as _;
        use rustreexo::accumulator::node_hash::BitcoinNodeHash;

        use crate::{
            authorization::{SigningKey, authorize, get_address},
            types::{
                Accumulator, AccumulatorDiff, Body, Header, OutPoint,
                OutPointKey, PointedOutput, Transaction, hash,
            },
        };
        let (env, state) =
            fresh_state("validation_rejects_outpoint_utxo_hash_mismatch")?;

        // Attacker key (owns A). Victim key (owns B).
        let attacker = SigningKey::from_seeds(
            &[0x11; fips205::slh_dsa_shake_256s::N],
            &[0x11; fips205::slh_dsa_shake_256s::N],
            &[0x11; fips205::slh_dsa_shake_256s::N],
        );
        let attacker_addr = get_address(&attacker.verifying_key());
        let victim = SigningKey::from_seeds(
            &[0x22; fips205::slh_dsa_shake_256s::N],
            &[0x22; fips205::slh_dsa_shake_256s::N],
            &[0x22; fips205::slh_dsa_shake_256s::N],
        );
        let victim_addr = get_address(&victim.verifying_key());

        // UTXO A (attacker, 10_000) and victim UTXO B (20_000).
        let outpoint_a = OutPoint::Deposit(bitcoin::OutPoint {
            txid: bitcoin::Txid::from_byte_array([0xAA; 32]),
            vout: 0,
        });
        let output_a = value_output(attacker_addr, 10_000);
        let outpoint_b = OutPoint::Deposit(bitcoin::OutPoint {
            txid: bitcoin::Txid::from_byte_array([0xBB; 32]),
            vout: 0,
        });
        let output_b = value_output(victim_addr, 20_000);

        // Leaf hashes for A and B (the Utreexo commitments).
        let pointed_a = PointedOutput {
            outpoint: outpoint_a,
            output: output_a.clone(),
        };
        let pointed_b = PointedOutput {
            outpoint: outpoint_b,
            output: output_b.clone(),
        };
        let leaf_a: BitcoinNodeHash = (&pointed_a).into();
        let leaf_b: BitcoinNodeHash = (&pointed_b).into();
        let hash_b: crate::types::Hash = hash(&pointed_b); // input's utxo_hash

        // Helper: build a fresh accumulator seeded with leaves A and B
        // (Accumulator is not Clone, so re-seed when a fresh copy is needed).
        let seeded_accumulator = || -> anyhow::Result<_> {
            let mut acc = Accumulator::default();
            let mut diff = AccumulatorDiff::default();
            diff.insert(leaf_a);
            diff.insert(leaf_b);
            acc.apply_diff(diff)?;
            Ok(acc)
        };

        // Seed UTXO DB with A and B; seed the accumulator with both leaves.
        let pre_accumulator = seeded_accumulator()?;
        {
            let mut rwtxn = env.write_txn()?;
            state.utxos.put(
                &mut rwtxn,
                &OutPointKey::from(&outpoint_a),
                &output_a,
            )?;
            state.utxos.put(
                &mut rwtxn,
                &OutPointKey::from(&outpoint_b),
                &output_b,
            )?;
            state
                .utreexo_accumulator
                .put(&mut rwtxn, &(), &pre_accumulator)?;
            // tip stays unset (None) so validate's prev_side_hash check
            // expects header.prev_side_hash == None.
            rwtxn.commit()?;
        }

        // Build the malicious tx: input = (outpoint_A, hash_B) + proof for B;
        // output C = 9_000 to the attacker.
        let proof_for_b = pre_accumulator.prove(&[leaf_b])?;
        let output_c = value_output(attacker_addr, 9_000);
        let tx = Transaction {
            inputs: vec![(outpoint_a, hash_b)],
            proof: proof_for_b,
            outputs: vec![output_c.clone()],
        };
        // Sign with A's key (the spender of outpoint A authorizes the tx).
        let authorized = authorize(
            &mut rand::thread_rng(),
            &[(attacker_addr, &attacker)],
            tx,
        )?;

        // Assemble body.
        let body = Body::new(vec![authorized], Vec::new());

        // Compute the header the validator expects:
        //   merkle_root from the filled tx, roots = post-block accumulator
        //   (B's leaf removed, C's leaf inserted) -- exactly the diff validate
        //   builds from the SUPPLIED utxo_hash.
        let filled = {
            let rotxn = env.read_txn()?;
            state.fill_transaction(&rotxn, &body.transactions[0])?
        };

        // tx validation REJECTS the outpoint/utxo_hash mismatch.
        anyhow::ensure!(state.validate_filled_transaction(&filled).is_err());
        let merkle_root =
            Body::compute_merkle_root(body.coinbase.as_slice(), &[filled])?;
        let mut post_accumulator = seeded_accumulator()?;
        {
            let mut diff = AccumulatorDiff::default();
            // validate removes the SUPPLIED utxo_hash (B), inserts output C.
            diff.remove(leaf_b);
            let txid = body.transactions[0].txid();
            let pointed_c = PointedOutput {
                outpoint: OutPoint::Regular { txid, vout: 0 },
                output: output_c.clone(),
            };
            diff.insert((&pointed_c).into());
            post_accumulator.apply_diff(diff)?;
        }
        let header = Header {
            merkle_root,
            prev_side_hash: None,
            prev_main_hash: bitcoin::BlockHash::from_byte_array([0u8; 32]),
            roots: post_accumulator.get_roots(),
        };

        // block validation REJECTS the outpoint/utxo_hash mismatch.
        {
            let rotxn = env.read_txn()?;
            anyhow::ensure!(
                state.validate_block(&rotxn, &header, &body).is_err(),
                "BUG: real validate_block accepts an input whose outpoint (A) \
                and utxo_hash (B) refer to different UTXOs",
            );
        }
        Ok(())
    }
}
