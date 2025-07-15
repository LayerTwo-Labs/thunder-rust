//! Connect and disconnect blocks

use std::collections::HashSet;

use rayon::prelude::*;
use rustreexo::accumulator::{node_hash::BitcoinNodeHash, stump::Stump};
use sneed::{RoTxn, RwTxn, db::error::Error as DbError};

use crate::{
    authorization::Authorization,
    state::{Error, State, error},
    types::{
        AccumulatorDiff, AmountOverflowError, Body, FilledTransaction,
        GetAddress as _, GetValue as _, Header, InPoint, OutPoint,
        PointedOutput, SpentOutput, Verify as _,
    },
};

/// Parallel proof verification using thread-safe Stump accumulator with fixed-size chunking
/// Returns true if all proofs are valid, false if any fail
fn verify_proofs_parallel(
    stump: &Stump<BitcoinNodeHash>,
    filled_transactions: &[FilledTransaction],
) -> Result<bool, Error> {
    // For small transaction counts, use simple parallel iteration
    if filled_transactions.len() <= 10 {
        return verify_proofs_simple_parallel(stump, filled_transactions);
    }

    // Create fixed-size chunks of 10 transactions each
    let chunks: Vec<&[FilledTransaction]> =
        filled_transactions.chunks(10).collect();

    tracing::debug!(
        "Using {} chunks for {} transactions (10 txs per chunk)",
        chunks.len(),
        filled_transactions.len()
    );

    // Process chunks in parallel - each chunk runs on a separate thread
    let results: Vec<_> = chunks
        .par_iter()
        .enumerate()
        .map(|(chunk_id, chunk)| {
            tracing::trace!(
                "Processing chunk {} ({} txs)", 
                chunk_id,
                chunk.len()
            );
            // Verify all transactions in this chunk sequentially
            // This provides better cache locality and reduces thread contention
            for filled_transaction in *chunk {
                let spent_utxo_hashes: Vec<BitcoinNodeHash> = filled_transaction
                    .transaction
                    .inputs
                    .iter()
                    .map(|(_, utxo_hash)| utxo_hash.into())
                    .collect();
                let result = stump.verify(&filled_transaction.transaction.proof, &spent_utxo_hashes);
                match result {
                    Ok(true) => continue,  // This proof verified successfully
                    Ok(false) => {
                        tracing::warn!(
                            "Proof verification failed for transaction {} in chunk {}", 
                            filled_transaction.transaction.txid(),
                            chunk_id
                        );
                        return Ok(false);
                    }
                    Err(e) => {
                        tracing::error!(
                            "Proof verification error for transaction {} in chunk {}: {}", 
                            filled_transaction.transaction.txid(),
                            chunk_id,
                            e
                        );
                        return Err(Error::UtreexoProofFailed {
                            txid: filled_transaction.transaction.txid()
                        });
                    }
                }
            }
            Ok(true)  // All transactions in this chunk verified successfully
        })
        .collect();

    // Check if all chunks succeeded
    for result in results {
        match result {
            Ok(true) => continue,
            Ok(false) => return Ok(false),
            Err(e) => return Err(e),
        }
    }

    Ok(true)
}

/// Simple parallel verification for small transaction counts
fn verify_proofs_simple_parallel(
    stump: &Stump<BitcoinNodeHash>,
    filled_transactions: &[FilledTransaction],
) -> Result<bool, Error> {
    let results: Vec<_> = filled_transactions
        .par_iter()
        .map(|filled_transaction| {
            let spent_utxo_hashes: Vec<BitcoinNodeHash> = filled_transaction
                .transaction
                .inputs
                .iter()
                .map(|(_, utxo_hash)| utxo_hash.into())
                .collect();

            let result = stump.verify(
                &filled_transaction.transaction.proof,
                &spent_utxo_hashes,
            );

            match result {
                Ok(is_valid) => {
                    if !is_valid {
                        tracing::warn!(
                            "Proof verification failed for transaction {}",
                            filled_transaction.transaction.txid()
                        );
                    }
                    Ok(is_valid)
                }
                Err(e) => {
                    tracing::error!(
                        "Proof verification error for transaction {}: {}",
                        filled_transaction.transaction.txid(),
                        e
                    );
                    Err(Error::UtreexoProofFailed {
                        txid: filled_transaction.transaction.txid(),
                    })
                }
            }
        })
        .collect();

    for result in results {
        match result {
            Ok(true) => continue,
            Ok(false) => return Ok(false),
            Err(e) => return Err(e),
        }
    }

    Ok(true)
}

pub fn validate(
    state: &State,
    rotxn: &RoTxn,
    header: &Header,
    body: &Body,
) -> Result<bitcoin::Amount, Error> {
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
    let mut accumulator_diff = AccumulatorDiff::default();
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
        accumulator_diff.insert((&pointed_output).into());
    }
    let mut total_fees = bitcoin::Amount::ZERO;
    let mut spent_utxos = HashSet::new();
    let filled_transactions: Vec<_> = body
        .transactions
        .iter()
        .map(|t| state.fill_transaction(rotxn, t))
        .collect::<Result<_, _>>()?;

    // Validate basic transaction constraints and accumulator diffs
    for filled_transaction in &filled_transactions {
        let txid = filled_transaction.transaction.txid();

        for (outpoint, utxo_hash) in &filled_transaction.transaction.inputs {
            if spent_utxos.contains(outpoint) {
                return Err(Error::UtxoDoubleSpent);
            }
            spent_utxos.insert(*outpoint);
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
    }

    // Parallel proof verification using thread-safe Stump
    if !filled_transactions.is_empty() {
        let stump = accumulator.to_stump();
        if !verify_proofs_parallel(&stump, &filled_transactions)? {
            return Err(Error::UtreexoProofFailed {
                txid: filled_transactions
                    .first()
                    .map(|tx| tx.transaction.txid())
                    .unwrap_or_default(),
            });
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
    let () = accumulator.apply_diff(accumulator_diff)?;
    let roots: Vec<BitcoinNodeHash> = accumulator.get_roots();
    if roots != header.roots {
        return Err(Error::UtreexoRootsMismatch);
    }
    Ok(total_fees)
}

pub fn connect(
    state: &State,
    rwtxn: &mut RwTxn,
    header: &Header,
    body: &Body,
) -> Result<(), Error> {
    let tip_hash = state.try_get_tip(rwtxn)?;
    if tip_hash != header.prev_side_hash {
        let err = error::InvalidHeader::PrevSideHash {
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
    let mut accumulator = state
        .utreexo_accumulator
        .try_get(rwtxn, &())
        .map_err(DbError::from)?
        .unwrap_or_default();
    let mut accumulator_diff = AccumulatorDiff::default();
    for (vout, output) in body.coinbase.iter().enumerate() {
        let outpoint = OutPoint::Coinbase {
            merkle_root,
            vout: vout as u32,
        };
        let pointed_output = PointedOutput {
            outpoint,
            output: output.clone(),
        };
        accumulator_diff.insert((&pointed_output).into());
        state
            .utxos
            .put(rwtxn, &outpoint, output)
            .map_err(DbError::from)?;
    }
    for transaction in &body.transactions {
        let txid = transaction.txid();
        for (vin, (outpoint, utxo_hash)) in
            transaction.inputs.iter().enumerate()
        {
            let spent_output = state
                .utxos
                .try_get(rwtxn, outpoint)
                .map_err(DbError::from)?
                .ok_or(Error::NoUtxo {
                    outpoint: *outpoint,
                })?;
            accumulator_diff.remove(utxo_hash.into());
            state.utxos.delete(rwtxn, outpoint).map_err(DbError::from)?;
            let spent_output = SpentOutput {
                output: spent_output,
                inpoint: InPoint::Regular {
                    txid,
                    vin: vin as u32,
                },
            };
            state
                .stxos
                .put(rwtxn, outpoint, &spent_output)
                .map_err(DbError::from)?;
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
            state
                .utxos
                .put(rwtxn, &outpoint, output)
                .map_err(DbError::from)?;
        }
    }
    let block_hash = header.hash();
    let height = state.try_get_height(rwtxn)?.map_or(0, |height| height + 1);
    state
        .tip
        .put(rwtxn, &(), &block_hash)
        .map_err(DbError::from)?;
    state
        .height
        .put(rwtxn, &(), &height)
        .map_err(DbError::from)?;
    let () = accumulator.apply_diff(accumulator_diff)?;
    state
        .utreexo_accumulator
        .put(rwtxn, &(), &accumulator)
        .map_err(DbError::from)?;
    Ok(())
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
    let merkle_root = body.compute_merkle_root();
    if merkle_root != header.merkle_root {
        let err = Error::InvalidBody {
            expected: merkle_root,
            computed: header.merkle_root,
        };
        return Err(err);
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
                if let Some(spent_output) = state
                    .stxos
                    .try_get(rwtxn, outpoint)
                    .map_err(DbError::from)?
                {
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
            let outpoint = OutPoint::Coinbase {
                merkle_root,
                vout: vout as u32,
            };
            let pointed_output = PointedOutput {
                outpoint,
                output: output.clone(),
            };
            accumulator_diff.remove((&pointed_output).into());
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
    let () = accumulator.apply_diff(accumulator_diff)?;
    state
        .utreexo_accumulator
        .put(rwtxn, &(), &accumulator)
        .map_err(DbError::from)?;
    Ok(())
}
