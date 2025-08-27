//! Connect and disconnect blocks

use heed::types::SerdeBincode;
#[cfg(feature = "utreexo")]
use rustreexo::accumulator::node_hash::BitcoinNodeHash;
use sneed::{DatabaseUnique, RoTxn, RwTxn, db::error::Error as DbError};

#[cfg(feature = "utreexo")]
use crate::types::{AccumulatorDiff, PointedOutput};
use crate::{
    state::{Error, PrevalidatedBlock, State, error},
    types::{
        AmountOverflowError, Authorization, Body, FilledTransaction,
        GetAddress as _, GetValue as _, Header, InPoint, MerkleRoot, OutPoint,
        OutPointKey, Output, SpentOutput, Verify as _,
    },
};

pub fn validate(
    state: &State,
    rotxn: &RoTxn,
    header: &Header,
    body: &Body,
) -> Result<(bitcoin::Amount, MerkleRoot), Error> {
    use rayon::prelude::ParallelSliceMut;
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
    // Fill transactions sequentially (RoTxn is not Send)
    let mut filled_transactions = Vec::with_capacity(body.transactions.len());
    for transaction in &body.transactions {
        filled_transactions.push(state.fill_transaction(rotxn, transaction)?);
    }
    let merkle_root = Body::compute_merkle_root(
        body.coinbase.as_slice(),
        filled_transactions.as_slice(),
    )?;
    #[cfg(feature = "utreexo")]
    let mut accumulator_diff = AccumulatorDiff::default();
    // Parallel coinbase value calculation
    let coinbase_value: bitcoin::Amount = body
        .coinbase
        .par_iter()
        .map(|output| output.get_value())
        .reduce(
            || bitcoin::Amount::ZERO,
            |acc, value| {
                acc.checked_add(value).unwrap_or(bitcoin::Amount::ZERO)
            },
        );

    // Sequential utreexo accumulator updates (due to accumulator mutations)
    #[cfg(feature = "utreexo")]
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
    }
    if merkle_root != header.merkle_root {
        let err = Error::InvalidBody {
            expected: header.merkle_root,
            computed: merkle_root,
        };
        return Err(err);
    }
    let mut total_fees = bitcoin::Amount::ZERO;
    let total_inputs = body.inputs_len();

    // Collect all inputs as fixed-width keys for efficient double-spend detection via sort-and-scan
    let mut all_input_keys = Vec::with_capacity(total_inputs);
    for filled_transaction in &filled_transactions {
        for (outpoint, _) in &filled_transaction.transaction.inputs {
            all_input_keys.push(OutPointKey::from(outpoint));
        }
    }

    // Sort and check for duplicate outpoints (double-spend detection)
    all_input_keys.par_sort_unstable();
    if all_input_keys.windows(2).any(|w| w[0] == w[1]) {
        return Err(Error::UtxoDoubleSpent);
    }

    // Parallel fee validation for all transactions
    use rayon::prelude::*;
    let transaction_fees: Result<Vec<bitcoin::Amount>, Error> =
        filled_transactions
            .par_iter()
            .map(|filled_transaction| {
                state.validate_filled_transaction(filled_transaction)
            })
            .collect();
    let transaction_fees = transaction_fees?;

    // Sum fees sequentially to avoid overflow issues
    for fee in transaction_fees {
        total_fees = total_fees.checked_add(fee).ok_or(AmountOverflowError)?;
    }

    // Process transactions for utreexo (sequential due to accumulator mutations)
    for filled_transaction in &filled_transactions {
        #[cfg(feature = "utreexo")]
        let txid = filled_transaction.transaction.txid();
        #[cfg(feature = "utreexo")]
        // hashes of spent utxos, used to verify the utreexo proof
        let mut spent_utxo_hashes = Vec::<BitcoinNodeHash>::with_capacity(
            filled_transaction.transaction.inputs.len(),
        );
        for (_outpoint, utxo_hash) in &filled_transaction.transaction.inputs {
            #[cfg(not(feature = "utreexo"))]
            let _ = utxo_hash;
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
    // Parallel authorization verification
    let spent_utxos: Vec<_> = filled_transactions
        .iter()
        .flat_map(|t| t.spent_utxos.iter())
        .collect();

    // Parallel address verification
    let address_verification_result: Result<(), Error> = body
        .authorizations
        .par_iter()
        .zip(spent_utxos.par_iter())
        .try_for_each(|(authorization, spent_utxo)| {
            if authorization.get_address() != spent_utxo.address {
                Err(Error::WrongPubKeyForAddress)
            } else {
                Ok(())
            }
        });
    address_verification_result?;

    // Batch signature verification (already parallelized in Authorization::verify_body)
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
    use rayon::prelude::ParallelSliceMut;
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
    // Fill transactions sequentially (RoTxn is not Send)
    let mut filled_transactions = Vec::with_capacity(body.transactions.len());
    for transaction in &body.transactions {
        filled_transactions.push(state.fill_transaction(rotxn, transaction)?);
    }
    let merkle_root = Body::compute_merkle_root(
        body.coinbase.as_slice(),
        filled_transactions.as_slice(),
    )?;
    #[cfg(feature = "utreexo")]
    let mut accumulator_diff = AccumulatorDiff::default();
    // Parallel coinbase value calculation
    let coinbase_value: bitcoin::Amount = body
        .coinbase
        .par_iter()
        .map(|output| output.get_value())
        .reduce(
            || bitcoin::Amount::ZERO,
            |acc, value| {
                acc.checked_add(value).unwrap_or(bitcoin::Amount::ZERO)
            },
        );

    // Sequential utreexo accumulator updates (due to accumulator mutations)
    #[cfg(feature = "utreexo")]
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
    }
    if merkle_root != header.merkle_root {
        let err = Error::InvalidBody {
            expected: header.merkle_root,
            computed: merkle_root,
        };
        return Err(err);
    }
    let mut total_fees = bitcoin::Amount::ZERO;
    let total_inputs = body.inputs_len();

    // Collect all inputs as fixed-width keys for efficient double-spend detection via sort-and-scan
    let mut all_input_keys = Vec::with_capacity(total_inputs);
    for filled_transaction in &filled_transactions {
        for (outpoint, _) in &filled_transaction.transaction.inputs {
            all_input_keys.push(OutPointKey::from(outpoint));
        }
    }

    // Sort and check for duplicate outpoints (double-spend detection)
    all_input_keys.par_sort_unstable();
    if all_input_keys.windows(2).any(|w| w[0] == w[1]) {
        return Err(Error::UtxoDoubleSpent);
    }

    // Parallel fee validation for all transactions
    use rayon::prelude::*;
    let transaction_fees: Result<Vec<bitcoin::Amount>, Error> =
        filled_transactions
            .par_iter()
            .map(|filled_transaction| {
                state.validate_filled_transaction(filled_transaction)
            })
            .collect();
    let transaction_fees = transaction_fees?;

    // Sum fees sequentially to avoid overflow issues
    for fee in transaction_fees {
        total_fees = total_fees.checked_add(fee).ok_or(AmountOverflowError)?;
    }

    // Process transactions for utreexo (sequential due to accumulator mutations)
    for filled_transaction in &filled_transactions {
        #[cfg(feature = "utreexo")]
        let txid = filled_transaction.transaction.txid();
        #[cfg(feature = "utreexo")]
        // hashes of spent utxos, used to verify the utreexo proof
        let mut spent_utxo_hashes = Vec::<BitcoinNodeHash>::with_capacity(
            filled_transaction.transaction.inputs.len(),
        );
        for (_outpoint, utxo_hash) in &filled_transaction.transaction.inputs {
            #[cfg(not(feature = "utreexo"))]
            let _ = utxo_hash;
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
        next_height: height,
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
    use rayon::prelude::{
        IntoParallelRefIterator, ParallelIterator, ParallelSliceMut,
    };
    let merkle_root = prevalidated.computed_merkle_root;

    #[cfg(feature = "utreexo")]
    let mut accumulator = state
        .utreexo_accumulator
        .try_get(rwtxn, &())?
        .unwrap_or_default();

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

    // Use Vec + sort_unstable instead of BTreeMap for better performance
    let mut utxo_deletes: Vec<OutPoint> = Vec::with_capacity(total_inputs);
    let mut stxo_puts: Vec<(OutPoint, SpentOutput)> =
        Vec::with_capacity(total_inputs);
    let mut utxo_puts: Vec<(OutPoint, Output)> =
        Vec::with_capacity(total_outputs);

    for (vout, output) in body.coinbase.iter().enumerate() {
        let outpoint = OutPoint::Coinbase {
            merkle_root: header.merkle_root,
            vout: vout as u32,
        };
        utxo_puts.push((outpoint, output.clone()));
    }

    // Parallel collection of transaction operations
    let tx_results: Vec<(
        Vec<OutPoint>,
        Vec<(OutPoint, SpentOutput)>,
        Vec<(OutPoint, Output)>,
    )> = prevalidated
        .filled_transactions
        .par_iter()
        .map(|filled_transaction| {
            let txid = filled_transaction.transaction.txid();
            let mut deletes =
                Vec::with_capacity(filled_transaction.transaction.inputs.len());
            let mut stxo_puts_local =
                Vec::with_capacity(filled_transaction.transaction.inputs.len());
            let mut utxo_puts_local = Vec::with_capacity(
                filled_transaction.transaction.outputs.len(),
            );

            // Process inputs
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

                deletes.push(*outpoint);
                stxo_puts_local.push((*outpoint, spent_output));
            }

            // Process outputs
            for (vout, output) in
                filled_transaction.transaction.outputs.iter().enumerate()
            {
                let outpoint = OutPoint::Regular {
                    txid,
                    vout: vout as u32,
                };
                utxo_puts_local.push((outpoint, output.clone()));
            }

            (deletes, stxo_puts_local, utxo_puts_local)
        })
        .collect();

    // Separate the three vectors
    let (tx_deletes, tx_stxo_puts, tx_utxo_puts): (
        Vec<Vec<OutPoint>>,
        Vec<Vec<(OutPoint, SpentOutput)>>,
        Vec<Vec<(OutPoint, Output)>>,
    ) = tx_results.into_iter().fold(
        (Vec::new(), Vec::new(), Vec::new()),
        |(mut deletes, mut stxo_puts, mut utxo_puts), (d, s, u)| {
            deletes.push(d);
            stxo_puts.push(s);
            utxo_puts.push(u);
            (deletes, stxo_puts, utxo_puts)
        },
    );

    // Flatten the parallel results
    utxo_deletes.extend(tx_deletes.into_iter().flatten());
    stxo_puts.extend(tx_stxo_puts.into_iter().flatten());
    utxo_puts.extend(tx_utxo_puts.into_iter().flatten());

    // Pre-encode all keys in parallel and serialize values for cursor operations
    let mut utxo_delete_keys: Vec<OutPointKey> =
        utxo_deletes.par_iter().map(OutPointKey::from).collect();

    // Phase 2: Parallel key encoding and data preparation
    let mut stxo_put_data: Vec<(OutPointKey, &SpentOutput)> = stxo_puts
        .par_iter()
        .map(|(op, spent)| {
            let key = OutPointKey::from(op);
            (key, spent)
        })
        .collect();

    let mut utxo_put_data: Vec<(OutPointKey, &Output)> = utxo_puts
        .par_iter()
        .map(|(op, output)| {
            let key = OutPointKey::from(op);
            (key, output)
        })
        .collect();

    // Sort all vectors in parallel for optimal cursor access
    utxo_delete_keys.par_sort_unstable();
    stxo_put_data.par_sort_unstable_by_key(|(key, _)| *key);
    utxo_put_data.par_sort_unstable_by_key(|(key, _)| *key);

    // Phase 2: Batch cursor operations for optimal B-tree traversal
    // Process deletions first to free up space before insertions
    batch_delete_utxos(rwtxn, &state.utxos, &utxo_delete_keys)?;

    // Batch insertions using sorted data for sequential B-tree access
    batch_put_stxos(rwtxn, &state.stxos, &stxo_put_data)?;
    batch_put_utxos(rwtxn, &state.utxos, &utxo_put_data)?;

    let block_hash = header.hash();

    // Update tip and height using precomputed values (no redundant DB reads)
    state.tip.put(rwtxn, &(), &block_hash)?;
    state.height.put(rwtxn, &(), &prevalidated.next_height)?;

    #[cfg(feature = "utreexo")]
    // Apply utreexo accumulator diff
    {
        let () = accumulator.apply_diff(prevalidated.accumulator_diff)?;
        state.utreexo_accumulator.put(rwtxn, &(), &accumulator)?;
    }

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
            #[cfg(not(feature = "utreexo"))]
            let _ = utxo_hash;
            let key = OutPointKey::from(outpoint);
            let spent_output =
                state.utxos.try_get(rwtxn, &key)?.ok_or(Error::NoUtxo {
                    outpoint: *outpoint,
                })?;
            #[cfg(feature = "utreexo")]
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
            let key = OutPointKey::from(outpoint);
            state.stxos.put(rwtxn, &key, &spent_output)?;
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
                let key = OutPointKey::from(&outpoint);
                if state.utxos.delete(rwtxn, &key).map_err(DbError::from)? {
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
                let key = OutPointKey::from(outpoint);
                if let Some(spent_output) =
                    state.stxos.try_get(rwtxn, &key).map_err(DbError::from)?
                {
                    #[cfg(feature = "utreexo")]
                    accumulator_diff.insert(utxo_hash.into());
                    state.stxos.delete(rwtxn, &key).map_err(DbError::from)?;
                    state
                        .utxos
                        .put(rwtxn, &key, &spent_output.output)
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
            let key = OutPointKey::from(&outpoint);
            if state.utxos.delete(rwtxn, &key).map_err(DbError::from)? {
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

// Phase 2: Batch database operations for optimal B-tree performance
// These functions leverage sorted data and cursor operations to minimize B-tree traversals

/// Batch delete UTXOs using sorted keys for optimal B-tree access
fn batch_delete_utxos(
    rwtxn: &mut RwTxn,
    utxos_db: &DatabaseUnique<OutPointKey, SerdeBincode<Output>>,
    sorted_keys: &[OutPointKey],
) -> Result<(), Error> {
    // Use sequential access pattern for better cache locality
    for key in sorted_keys {
        utxos_db.delete(rwtxn, key)?;
    }
    Ok(())
}

/// Batch insert STXOs using sorted data for optimal B-tree insertion
fn batch_put_stxos(
    rwtxn: &mut RwTxn,
    stxos_db: &DatabaseUnique<OutPointKey, SerdeBincode<SpentOutput>>,
    sorted_data: &[(OutPointKey, &SpentOutput)],
) -> Result<(), Error> {
    // Sequential insertion pattern reduces B-tree rebalancing overhead
    for (key, spent_output) in sorted_data {
        stxos_db.put(rwtxn, key, spent_output)?
    }
    Ok(())
}

/// Batch insert UTXOs using sorted data for optimal B-tree insertion
fn batch_put_utxos(
    rwtxn: &mut RwTxn,
    utxos_db: &DatabaseUnique<OutPointKey, SerdeBincode<Output>>,
    sorted_data: &[(OutPointKey, &Output)],
) -> Result<(), Error> {
    // Sequential insertion pattern reduces B-tree rebalancing overhead
    for (key, output) in sorted_data {
        utxos_db.put(rwtxn, key, output)?
    }
    Ok(())
}
