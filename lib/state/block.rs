//! Connect and disconnect blocks

use rayon::prelude::*;
use std::collections::HashSet;
use typed_arena::Arena;

#[cfg(feature = "utreexo")]
use rustreexo::accumulator::node_hash::BitcoinNodeHash;
use sneed::{RoTxn, RwTxn, db::error::Error as DbError};

#[cfg(feature = "utreexo")]
use crate::types::{AccumulatorDiff, PointedOutput};
use crate::{
    authorization::Authorization,
    state::{Error, PrevalidatedBlock, State, error},
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

    // Arena allocator for grouping similar objects and reducing heap allocation overhead
    // This reduces memory fragmentation and improves cache locality for pointer-heavy structures
    let spent_output_arena = Arena::new();

    // Determine buffer sizes to avoid dynamic reallocations during processing
    let mut input_count = 0;
    let mut output_count = 0;
    for tx in &prevalidated.filled_transactions {
        input_count += tx.transaction.inputs.len();
        output_count += tx.transaction.outputs.len();
    }

    // Initialize collections with calculated sizes for efficient memory usage
    let mut deletion_queue = Vec::with_capacity(input_count);
    let mut spent_outputs = Vec::with_capacity(input_count);
    let mut new_utxos = Vec::with_capacity(output_count + body.coinbase.len());

    // Temporary working vectors for reuse across transactions to reduce allocations
    let mut temp_outpoints = Vec::with_capacity(input_count);
    let mut temp_spent_outputs = Vec::with_capacity(input_count);

    // Handle coinbase transaction outputs first
    for (index, output) in body.coinbase.iter().enumerate() {
        let coinbase_outpoint = OutPoint::Coinbase {
            merkle_root: header.merkle_root,
            vout: index as u32,
        };
        new_utxos.push((coinbase_outpoint, output.clone()));
    }

    // Iterate through all transactions for input/output processing
    for tx_data in &prevalidated.filled_transactions {
        let transaction_id = tx_data.transaction.txid();

        // Clear and reuse temporary vectors for this transaction
        temp_outpoints.clear();
        temp_spent_outputs.clear();

        // Handle transaction inputs (spending existing UTXOs)
        for (input_idx, (outpoint_ref, _hash)) in
            tx_data.transaction.inputs.iter().enumerate()
        {
            let consumed_utxo = &tx_data.spent_utxos[input_idx];

            temp_outpoints.push(*outpoint_ref);

            // Use arena allocation for spent output to reduce heap fragmentation
            let spent_output_data = SpentOutput {
                output: consumed_utxo.clone(),
                inpoint: InPoint::Regular {
                    txid: transaction_id,
                    vin: input_idx as u32,
                },
            };
            let spent_output = spent_output_arena.alloc(spent_output_data.clone());
            temp_spent_outputs.push((*outpoint_ref, spent_output));
        }

        // Batch append to main vectors for better memory access patterns
        deletion_queue.extend_from_slice(&temp_outpoints);
        for (outpoint, spent_output) in &temp_spent_outputs {
            spent_outputs.push((*outpoint, (*spent_output).clone()));
        }

        // Handle transaction outputs (creating new UTXOs)
        for (output_idx, output_data) in
            tx_data.transaction.outputs.iter().enumerate()
        {
            let new_outpoint = OutPoint::Regular {
                txid: transaction_id,
                vout: output_idx as u32,
            };
            new_utxos.push((new_outpoint, output_data.clone()));
        }
    }

    // Organize data structures for database efficiency
    // Sorting improves LMDB performance by enabling sequential access patterns
    // which reduces disk seeks and improves cache utilization
    deletion_queue.par_sort_unstable();
    spent_outputs.par_sort_unstable_by_key(|(key, _)| *key);
    new_utxos.par_sort_unstable_by_key(|(key, _)| *key);

    for outpoint in &deletion_queue {
        state.utxos.delete(rwtxn, outpoint)?;
    }

    for (outpoint, spent_output) in &spent_outputs {
        state.stxos.put(rwtxn, outpoint, spent_output)?;
    }

    for (outpoint, output) in &new_utxos {
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
        
    // Arena allocator will handle bulk deallocation automatically when dropped

    Ok(merkle_root)
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
