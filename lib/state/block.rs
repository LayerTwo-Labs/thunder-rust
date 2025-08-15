//! Connect and disconnect blocks

use std::collections::{BTreeMap, HashSet};

use rayon::prelude::*;
#[cfg(feature = "utreexo")]
use rustreexo::accumulator::node_hash::BitcoinNodeHash;
use sneed::{RoTxn, RwTxn, db::error::Error as DbError};

#[cfg(feature = "utreexo")]
use crate::types::{AccumulatorDiff, PointedOutput};
use crate::{
    authorization::Authorization,
    state::{Error, MemoryPool, PrevalidatedBlock, State, error},
    types::{
        AmountOverflowError, Body, FilledTransaction, GetAddress as _,
        GetValue as _, Header, InPoint, MerkleRoot, OutPoint, SpentOutput,
        Verify as _,
    },
};

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
    #[cfg(feature = "utreexo")]
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
    #[cfg(feature = "utreexo")]
    let mut accumulator_diff = AccumulatorDiff::default();
    let mut coinbase_value = bitcoin::Amount::ZERO;
    for (vout, output) in body.coinbase.iter().enumerate() {
        #[cfg(not(feature = "utreexo"))]
        let _ = vout;
        coinbase_value = coinbase_value
            .checked_add(output.get_value())
            .ok_or(AmountOverflowError)?;
        #[cfg(feature = "utreexo")]
        {
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
    }
    if merkle_root != header.merkle_root {
        let err = Error::InvalidBody {
            expected: header.merkle_root,
            computed: merkle_root,
        };
        return Err(err);
    }
    let mut total_fees = bitcoin::Amount::ZERO;
    let mut spent_utxos = HashSet::new();
    for filled_transaction in &filled_transactions {
        #[cfg(feature = "utreexo")]
        let txid = filled_transaction.transaction.txid();
        #[cfg(feature = "utreexo")]
        // hashes of spent utxos, used to verify the utreexo proof
        let mut spent_utxo_hashes = Vec::<BitcoinNodeHash>::new();
        for (outpoint, utxo_hash) in &filled_transaction.transaction.inputs {
            #[cfg(not(feature = "utreexo"))]
            let _ = utxo_hash;
            if spent_utxos.contains(outpoint) {
                return Err(Error::UtxoDoubleSpent);
            }
            spent_utxos.insert(*outpoint);
            #[cfg(feature = "utreexo")]
            {
                spent_utxo_hashes.push(utxo_hash.into());
                accumulator_diff.remove(utxo_hash.into());
            }
        }
        #[cfg(feature = "utreexo")]
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
        #[cfg(feature = "utreexo")]
        {
            // verify utreexo proof
            if !accumulator.verify(
                &filled_transaction.transaction.proof,
                &spent_utxo_hashes,
            )? {
                return Err(Error::UtreexoProofFailed { txid });
            }
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
        return Err(Error::Authorization);
    }
    #[cfg(feature = "utreexo")]
    {
        let () = accumulator.apply_diff(accumulator_diff)?;
        let roots: Vec<BitcoinNodeHash> = accumulator.get_roots();
        if roots != header.roots {
            return Err(Error::UtreexoRootsMismatch);
        }
    }
    Ok((total_fees, merkle_root))
}

/// Parallel prevalidation function
/// Replaces the original sequential prevalidate with parallel transaction processing
pub fn prevalidate_parallel(
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
    let height = state.try_get_height(rotxn)?.map_or(0, |height| height + 1);
    if body.authorizations.len() > State::body_sigops_limit(height) {
        return Err(Error::TooManySigops);
    }
    let body_size =
        borsh::object_length(&body).map_err(Error::BorshSerialize)?;
    if body_size > State::body_size_limit(height) {
        return Err(Error::BodyTooLarge);
    }
    #[cfg(feature = "utreexo")]
    let mut accumulator = state
        .utreexo_accumulator
        .try_get(rotxn, &())
        .map_err(DbError::from)?
        .unwrap_or_default();

    // Sequential transaction filling (required due to RoTxn not being Sync)
    // But we can parallelize the validation afterwards
    let filled_transactions: Vec<_> = body
        .transactions
        .iter()
        .map(|t| state.fill_transaction(rotxn, t))
        .collect::<Result<_, _>>()?;

    let merkle_root = Body::compute_merkle_root(
        body.coinbase.as_slice(),
        filled_transactions.as_slice(),
    )?;

    #[cfg(feature = "utreexo")]
    let mut accumulator_diff = AccumulatorDiff::default();
    let mut coinbase_value = bitcoin::Amount::ZERO;
    for (vout, output) in body.coinbase.iter().enumerate() {
        #[cfg(not(feature = "utreexo"))]
        let _ = vout;
        coinbase_value = coinbase_value
            .checked_add(output.get_value())
            .ok_or(AmountOverflowError)?;
        #[cfg(feature = "utreexo")]
        {
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
    }
    if merkle_root != header.merkle_root {
        let err = Error::InvalidBody {
            expected: header.merkle_root,
            computed: merkle_root,
        };
        return Err(err);
    }

    // Parallel validation of individual transactions
    let validation_results: Vec<_> = filled_transactions
        .par_iter()
        .enumerate()
        .map(|(tx_index, filled_transaction)| -> Result<_, Error> {
            #[cfg(feature = "utreexo")]
            let txid = filled_transaction.transaction.txid();

            // Collect spent UTXOs for conflict detection later
            let spent_outpoints: Vec<OutPoint> = filled_transaction
                .transaction
                .inputs
                .iter()
                .map(|(outpoint, _)| *outpoint)
                .collect();

            #[cfg(feature = "utreexo")]
            let spent_utxo_hashes: Vec<BitcoinNodeHash> = filled_transaction
                .transaction
                .inputs
                .iter()
                .map(|(_, utxo_hash)| utxo_hash.into())
                .collect();

            #[cfg(feature = "utreexo")]
            let output_diffs: Vec<PointedOutput> = filled_transaction
                .transaction
                .outputs
                .iter()
                .enumerate()
                .map(|(vout, output)| {
                    let outpoint = OutPoint::Regular {
                        txid,
                        vout: vout as u32,
                    };
                    PointedOutput {
                        outpoint,
                        output: output.clone(),
                    }
                })
                .collect();

            // Validate transaction and calculate fee (this is thread-safe)
            let fee = state.validate_filled_transaction(filled_transaction)?;

            Ok((
                tx_index,
                fee,
                spent_outpoints,
                #[cfg(feature = "utreexo")]
                spent_utxo_hashes,
                #[cfg(feature = "utreexo")]
                output_diffs,
            ))
        })
        .collect::<Result<Vec<_>, Error>>()?;

    // Sequential conflict detection and accumulator updates
    let mut spent_utxos = HashSet::new();
    let mut total_fees = bitcoin::Amount::ZERO;

    for result_tuple in validation_results {
        #[cfg(feature = "utreexo")]
        let (tx_index, fee, spent_outpoints, spent_utxo_hashes, output_diffs) =
            result_tuple;
        #[cfg(not(feature = "utreexo"))]
        let (_tx_index, fee, spent_outpoints) = result_tuple;
        // Sequential double-spend check
        for outpoint in &spent_outpoints {
            if spent_utxos.contains(outpoint) {
                return Err(Error::UtxoDoubleSpent);
            }
            spent_utxos.insert(*outpoint);
        }

        total_fees = total_fees.checked_add(fee).ok_or(AmountOverflowError)?;

        #[cfg(feature = "utreexo")]
        {
            // Sequential utreexo proof verification using transaction index
            let tx_filled = &filled_transactions[tx_index];
            let txid = tx_filled.transaction.txid();
            if !accumulator
                .verify(&tx_filled.transaction.proof, &spent_utxo_hashes)?
            {
                return Err(Error::UtreexoProofFailed { txid });
            }

            // Sequential accumulator updates
            for utxo_hash in spent_utxo_hashes {
                accumulator_diff.remove(utxo_hash);
            }
            for output_diff in output_diffs {
                accumulator_diff.insert((&output_diff).into());
            }
        }
    }

    if coinbase_value > total_fees {
        return Err(Error::NotEnoughFees);
    }
    let spent_utxos_iter = filled_transactions
        .iter()
        .flat_map(|t| t.spent_utxos.iter());
    for (authorization, spent_utxo) in
        body.authorizations.iter().zip(spent_utxos_iter)
    {
        if authorization.get_address() != spent_utxo.address {
            return Err(Error::WrongPubKeyForAddress);
        }
    }

    if Authorization::verify_body(body).is_err() {
        return Err(Error::Authorization);
    }

    #[cfg(feature = "utreexo")]
    {
        let () = accumulator.apply_diff(accumulator_diff.clone())?;
        let roots: Vec<BitcoinNodeHash> = accumulator.get_roots();
        if roots != header.roots {
            return Err(Error::UtreexoRootsMismatch);
        }
    }
    Ok(PrevalidatedBlock {
        filled_transactions,
        computed_merkle_root: merkle_root,
        total_fees,
        coinbase_value,
        #[cfg(feature = "utreexo")]
        accumulator_diff,
    })
}

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
    let height = state.try_get_height(rotxn)?.map_or(0, |height| height + 1);
    if body.authorizations.len() > State::body_sigops_limit(height) {
        return Err(Error::TooManySigops);
    }
    let body_size =
        borsh::object_length(&body).map_err(Error::BorshSerialize)?;
    if body_size > State::body_size_limit(height) {
        return Err(Error::BodyTooLarge);
    }
    #[cfg(feature = "utreexo")]
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
    #[cfg(feature = "utreexo")]
    let mut accumulator_diff = AccumulatorDiff::default();
    let mut coinbase_value = bitcoin::Amount::ZERO;
    for (vout, output) in body.coinbase.iter().enumerate() {
        #[cfg(not(feature = "utreexo"))]
        let _ = vout;
        coinbase_value = coinbase_value
            .checked_add(output.get_value())
            .ok_or(AmountOverflowError)?;
        #[cfg(feature = "utreexo")]
        {
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
    }
    if merkle_root != header.merkle_root {
        let err = Error::InvalidBody {
            expected: header.merkle_root,
            computed: merkle_root,
        };
        return Err(err);
    }
    let mut total_fees = bitcoin::Amount::ZERO;
    let mut spent_utxos = HashSet::new();
    for filled_transaction in &filled_transactions {
        #[cfg(feature = "utreexo")]
        let txid = filled_transaction.transaction.txid();
        #[cfg(feature = "utreexo")]
        // hashes of spent utxos, used to verify the utreexo proof
        let mut spent_utxo_hashes = Vec::<BitcoinNodeHash>::new();
        for (outpoint, utxo_hash) in &filled_transaction.transaction.inputs {
            #[cfg(not(feature = "utreexo"))]
            let _ = utxo_hash;
            if spent_utxos.contains(outpoint) {
                return Err(Error::UtxoDoubleSpent);
            }
            spent_utxos.insert(*outpoint);
            #[cfg(feature = "utreexo")]
            {
                spent_utxo_hashes.push(utxo_hash.into());
                accumulator_diff.remove(utxo_hash.into());
            }
        }
        #[cfg(feature = "utreexo")]
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
        #[cfg(feature = "utreexo")]
        {
            // verify utreexo proof
            if !accumulator.verify(
                &filled_transaction.transaction.proof,
                &spent_utxo_hashes,
            )? {
                return Err(Error::UtreexoProofFailed { txid });
            }
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
        return Err(Error::Authorization);
    }
    #[cfg(feature = "utreexo")]
    {
        let () = accumulator.apply_diff(accumulator_diff.clone())?;
        let roots: Vec<BitcoinNodeHash> = accumulator.get_roots();
        if roots != header.roots {
            return Err(Error::UtreexoRootsMismatch);
        }
    }
    Ok(PrevalidatedBlock {
        filled_transactions,
        computed_merkle_root: merkle_root,
        total_fees,
        coinbase_value,
        #[cfg(feature = "utreexo")]
        accumulator_diff,
    })
}

pub fn connect_prevalidated(
    state: &State,
    rwtxn: &mut RwTxn,
    header: &Header,
    body: &Body,
    prevalidated: PrevalidatedBlock,
) -> Result<MerkleRoot, Error> {
    let tip_hash = state.try_get_tip(rwtxn)?;
    if tip_hash != header.prev_side_hash {
        let err = error::InvalidHeader::PrevSideHash {
            expected: tip_hash,
            received: header.prev_side_hash,
        };
        return Err(Error::InvalidHeader(err));
    }

    let merkle_root = prevalidated.computed_merkle_root;
    if merkle_root != header.merkle_root {
        let err = Error::InvalidBody {
            expected: header.merkle_root,
            computed: merkle_root,
        };
        return Err(err);
    }

    #[cfg(feature = "utreexo")]
    let mut accumulator = state
        .utreexo_accumulator
        .try_get(rwtxn, &())?
        .unwrap_or_default();

    let mut utxo_deletes = BTreeMap::new();
    let mut stxo_puts = BTreeMap::new();
    let mut utxo_puts = BTreeMap::new();

    for (vout, output) in body.coinbase.iter().enumerate() {
        let outpoint = OutPoint::Coinbase {
            merkle_root: header.merkle_root,
            vout: vout as u32,
        };
        utxo_puts.insert(outpoint, output.clone());
    }

    for filled_transaction in &prevalidated.filled_transactions {
        let txid = filled_transaction.transaction.txid();

        for (vin, (outpoint, _utxo_hash)) in
            filled_transaction.transaction.inputs.iter().enumerate()
        {
            let spent_utxo = &filled_transaction.spent_utxos[vin];
            let spent_output = SpentOutput {
                output: spent_utxo.clone(),
                inpoint: InPoint::Regular {
                    txid,
                    vin: vin as u32,
                },
            };

            utxo_deletes.insert(*outpoint, ());
            stxo_puts.insert(*outpoint, spent_output);
        }

        for (vout, output) in
            filled_transaction.transaction.outputs.iter().enumerate()
        {
            let outpoint = OutPoint::Regular {
                txid,
                vout: vout as u32,
            };
            utxo_puts.insert(outpoint, output.clone());
        }
    }

    for outpoint in utxo_deletes.keys() {
        state.utxos.delete(rwtxn, outpoint)?;
    }

    for (outpoint, spent_output) in &stxo_puts {
        state.stxos.put(rwtxn, outpoint, spent_output)?;
    }

    for (outpoint, output) in &utxo_puts {
        state.utxos.put(rwtxn, outpoint, output)?;
    }

    let block_hash = header.hash();
    let height = state.try_get_height(rwtxn)?.map_or(0, |height| height + 1);

    // Update tip and height using regular database operations
    state.tip.put(rwtxn, &(), &block_hash)?;
    state.height.put(rwtxn, &(), &height)?;

    #[cfg(feature = "utreexo")]
    // Apply utreexo accumulator diff
    {
        let () = accumulator.apply_diff(prevalidated.accumulator_diff)?;
        state.utreexo_accumulator.put(rwtxn, &(), &accumulator)?;
    }

    Ok(merkle_root)
}

pub fn connect_prevalidated_with_pool(
    state: &State,
    rwtxn: &mut RwTxn,
    header: &Header,
    body: &Body,
    prevalidated: PrevalidatedBlock,
    pool: &mut MemoryPool,
) -> Result<MerkleRoot, Error> {
    let tip_hash = state.try_get_tip(rwtxn)?;
    if tip_hash != header.prev_side_hash {
        let err = error::InvalidHeader::PrevSideHash {
            expected: tip_hash,
            received: header.prev_side_hash,
        };
        return Err(Error::InvalidHeader(err));
    }

    let merkle_root = prevalidated.computed_merkle_root;
    if merkle_root != header.merkle_root {
        let err = Error::InvalidBody {
            expected: header.merkle_root,
            computed: merkle_root,
        };
        return Err(err);
    }

    // Estimate operations for memory pool sizing
    let estimated_ops = body.coinbase.len()
        + prevalidated
            .filled_transactions
            .iter()
            .map(|tx| {
                tx.transaction.inputs.len() + tx.transaction.outputs.len()
            })
            .sum::<usize>();

    pool.prepare_for_batch(estimated_ops);
    let (utxo_deletes, utxo_puts, stxo_puts, _spent_utxos) =
        pool.get_operation_buffers();

    #[cfg(feature = "utreexo")]
    let mut accumulator = state
        .utreexo_accumulator
        .try_get(rwtxn, &())?
        .unwrap_or_default();

    // Build operations in memory pool buffers
    for (vout, output) in body.coinbase.iter().enumerate() {
        let outpoint = OutPoint::Coinbase {
            merkle_root: header.merkle_root,
            vout: vout as u32,
        };
        utxo_puts.push((outpoint, output.clone()));
    }

    for filled_transaction in &prevalidated.filled_transactions {
        let txid = filled_transaction.transaction.txid();

        for (vin, (outpoint, _utxo_hash)) in
            filled_transaction.transaction.inputs.iter().enumerate()
        {
            let spent_utxo = &filled_transaction.spent_utxos[vin];
            let spent_output = SpentOutput {
                output: spent_utxo.clone(),
                inpoint: InPoint::Regular {
                    txid,
                    vin: vin as u32,
                },
            };

            utxo_deletes.push(*outpoint);
            stxo_puts.push((*outpoint, spent_output));
        }

        for (vout, output) in
            filled_transaction.transaction.outputs.iter().enumerate()
        {
            let outpoint = OutPoint::Regular {
                txid,
                vout: vout as u32,
            };
            utxo_puts.push((outpoint, output.clone()));
        }
    }

    // Execute bulk database operations
    bulk_execute_database_operations(
        state,
        rwtxn,
        utxo_deletes,
        stxo_puts,
        utxo_puts,
    )?;

    let block_hash = header.hash();
    let height = state.try_get_height(rwtxn)?.map_or(0, |height| height + 1);

    // Update tip and height
    state.tip.put(rwtxn, &(), &block_hash)?;
    state.height.put(rwtxn, &(), &height)?;

    #[cfg(feature = "utreexo")]
    {
        let () = accumulator.apply_diff(prevalidated.accumulator_diff)?;
        state.utreexo_accumulator.put(rwtxn, &(), &accumulator)?;
    }

    Ok(merkle_root)
}

fn bulk_execute_database_operations(
    state: &State,
    rwtxn: &mut RwTxn,
    utxo_deletes: &[OutPoint],
    stxo_puts: &[(OutPoint, SpentOutput)],
    utxo_puts: &[(OutPoint, crate::types::Output)],
) -> Result<(), Error> {
    // Sort operations for better LMDB performance (cache locality)
    let mut sorted_deletes = utxo_deletes.to_vec();
    sorted_deletes
        .sort_by_key(|outpoint| borsh::to_vec(outpoint).unwrap_or_default());

    let mut sorted_stxo_puts = stxo_puts.to_vec();
    sorted_stxo_puts.sort_by_key(|(outpoint, _)| {
        borsh::to_vec(outpoint).unwrap_or_default()
    });

    let mut sorted_utxo_puts = utxo_puts.to_vec();
    sorted_utxo_puts.sort_by_key(|(outpoint, _)| {
        borsh::to_vec(outpoint).unwrap_or_default()
    });

    // Execute in optimal order: deletes first (frees space), then inserts
    for outpoint in sorted_deletes {
        state.utxos.delete(rwtxn, &outpoint)?;
    }

    for (outpoint, spent_output) in sorted_stxo_puts {
        state.stxos.put(rwtxn, &outpoint, &spent_output)?;
    }

    for (outpoint, output) in sorted_utxo_puts {
        state.utxos.put(rwtxn, &outpoint, &output)?;
    }

    Ok(())
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
    #[cfg(feature = "utreexo")]
    let mut accumulator = state
        .utreexo_accumulator
        .try_get(rwtxn, &())?
        .unwrap_or_default();
    #[cfg(feature = "utreexo")]
    let mut accumulator_diff = AccumulatorDiff::default();
    for (vout, output) in body.coinbase.iter().enumerate() {
        let outpoint = OutPoint::Coinbase {
            merkle_root: header.merkle_root,
            vout: vout as u32,
        };
        #[cfg(feature = "utreexo")]
        {
            let pointed_output = PointedOutput {
                outpoint,
                output: output.clone(),
            };
            accumulator_diff.insert((&pointed_output).into());
        }
        state.utxos.put(rwtxn, &outpoint, output)?;
    }
    let mut filled_txs: Vec<FilledTransaction> = Vec::new();
    for transaction in &body.transactions {
        let mut spent_utxos = Vec::new();
        let txid = transaction.txid();
        for (vin, (outpoint, utxo_hash)) in
            transaction.inputs.iter().enumerate()
        {
            #[cfg(not(feature = "utreexo"))]
            let _ = utxo_hash;
            let spent_output =
                state.utxos.try_get(rwtxn, outpoint)?.ok_or(Error::NoUtxo {
                    outpoint: *outpoint,
                })?;
            #[cfg(feature = "utreexo")]
            accumulator_diff.remove(utxo_hash.into());
            state.utxos.delete(rwtxn, outpoint)?;
            let spent_output = SpentOutput {
                output: spent_output,
                inpoint: InPoint::Regular {
                    txid,
                    vin: vin as u32,
                },
            };
            state.stxos.put(rwtxn, outpoint, &spent_output)?;
            spent_utxos.push(spent_output.output);
        }
        for (vout, output) in transaction.outputs.iter().enumerate() {
            let outpoint = OutPoint::Regular {
                txid,
                vout: vout as u32,
            };
            #[cfg(feature = "utreexo")]
            {
                let pointed_output = PointedOutput {
                    outpoint,
                    output: output.clone(),
                };
                accumulator_diff.insert((&pointed_output).into());
            }
            state.utxos.put(rwtxn, &outpoint, output)?;
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
    #[cfg(feature = "utreexo")]
    {
        let () = accumulator.apply_diff(accumulator_diff)?;
        state.utreexo_accumulator.put(rwtxn, &(), &accumulator)?;
    }
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
    #[cfg(feature = "utreexo")]
    let mut accumulator = state
        .utreexo_accumulator
        .try_get(rwtxn, &())
        .map_err(DbError::from)?
        .unwrap_or_default();
    #[cfg(feature = "utreexo")]
    tracing::debug!("Got acc");
    #[cfg(feature = "utreexo")]
    let mut accumulator_diff = AccumulatorDiff::default();
    // revert txs, last-to-first
    body.transactions.iter().rev().try_for_each(|tx| {
        let txid = tx.txid();
        // delete UTXOs, last-to-first
        tx.outputs.iter().enumerate().rev().try_for_each(
            |(vout, output)| {
                #[cfg(not(feature = "utreexo"))]
                let _ = output;
                let outpoint = OutPoint::Regular {
                    txid,
                    vout: vout as u32,
                };
                #[cfg(feature = "utreexo")]
                {
                    let pointed_output = PointedOutput {
                        outpoint,
                        output: output.clone(),
                    };
                    accumulator_diff.remove((&pointed_output).into());
                }
                if state
                    .utxos
                    .delete(rwtxn, &outpoint)
                    .map_err(DbError::from)?
                {
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
                #[cfg(not(feature = "utreexo"))]
                let _ = utxo_hash;
                if let Some(spent_output) = state
                    .stxos
                    .try_get(rwtxn, outpoint)
                    .map_err(DbError::from)?
                {
                    #[cfg(feature = "utreexo")]
                    accumulator_diff.insert(utxo_hash.into());
                    state
                        .stxos
                        .delete(rwtxn, outpoint)
                        .map_err(DbError::from)?;
                    state
                        .utxos
                        .put(rwtxn, outpoint, &spent_output.output)
                        .map_err(DbError::from)?;
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
            #[cfg(not(feature = "utreexo"))]
            let _ = output;
            let outpoint = OutPoint::Coinbase {
                merkle_root: header.merkle_root,
                vout: vout as u32,
            };
            #[cfg(feature = "utreexo")]
            {
                let pointed_output = PointedOutput {
                    outpoint,
                    output: output.clone(),
                };
                accumulator_diff.remove((&pointed_output).into());
            }
            if state
                .utxos
                .delete(rwtxn, &outpoint)
                .map_err(DbError::from)?
            {
                Ok(())
            } else {
                Err(Error::NoUtxo { outpoint })
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
    #[cfg(feature = "utreexo")]
    {
        let () = accumulator.apply_diff(accumulator_diff)?;
        state
            .utreexo_accumulator
            .put(rwtxn, &(), &accumulator)
            .map_err(DbError::from)?;
    }
    Ok(())
}
