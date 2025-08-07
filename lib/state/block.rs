//! Connect and disconnect blocks

use std::collections::{BTreeMap, HashSet};

use rustreexo::accumulator::node_hash::BitcoinNodeHash;
use sneed::{RoTxn, RwTxn};

use crate::{
    authorization,
    state::{PrevalidatedBlock, State, error},
    types::{
        AccumulatorDiff, AmountOverflowError, Body, FilledTransaction,
        GetAddress as _, GetValue as _, Header, InPoint, MerkleRoot, OutPoint,
        PointedOutput, SpentOutput, Transaction, Txid,
    },
};

pub fn validate(
    state: &State,
    rotxn: &RoTxn,
    header: &Header,
    body: &Body,
) -> Result<(bitcoin::Amount, MerkleRoot), error::ValidateBlockInner> {
    let tip_hash = state.try_get_tip(rotxn)?;
    if header.prev_side_hash != tip_hash {
        let err = error::InvalidHeaderInner::PrevSideHash {
            expected: tip_hash,
            received: header.prev_side_hash,
        };
        return Err(error::ValidateBlockInner::InvalidHeader(err.into()));
    };
    let height = state.try_get_height(rotxn)?.map_or(0, |height| height + 1);
    if body.authorizations.len() > State::body_sigops_limit(height) {
        return Err(error::ValidateBlockInner::TooManySigops);
    }
    let body_size = borsh::object_length(&body)
        .map_err(error::ValidateBlockInner::BorshSerialize)?;
    if body_size > State::body_size_limit(height) {
        return Err(error::ValidateBlockInner::BodyTooLarge);
    }
    let mut accumulator = state
        .utreexo_accumulator
        .try_get(rotxn, &())?
        .unwrap_or_default();
    let filled_transactions: Vec<_> = body
        .transactions
        .iter()
        .map(|t| {
            state.fill_transaction(rotxn, t).map_err(|err| {
                error::ValidateTransaction {
                    txid: t.txid(),
                    source: err,
                }
            })
        })
        .collect::<Result<_, _>>()?;
    let merkle_root = Body::compute_merkle_root(
        body.coinbase.as_slice(),
        filled_transactions.as_slice(),
    )?;
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
    if merkle_root != header.merkle_root {
        let err = error::InvalidBody {
            expected: header.merkle_root,
            computed: merkle_root,
        };
        return Err(err.into());
    }
    let mut total_fees = bitcoin::Amount::ZERO;
    let mut spent_utxos = HashSet::new();
    for filled_transaction in &filled_transactions {
        let txid = filled_transaction.transaction.txid();
        // hashes of spent utxos, used to verify the utreexo proof
        let mut spent_utxo_hashes = Vec::<BitcoinNodeHash>::new();
        for (outpoint, utxo_hash) in &filled_transaction.transaction.inputs {
            if spent_utxos.contains(outpoint) {
                return Err(error::UtxoDoubleSpent.into());
            }
            spent_utxos.insert(*outpoint);
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
        let tx_fees =
            State::validate_filled_transaction_(filled_transaction).map_err(
                |err| error::ValidateTransaction { txid, source: err },
            )?;
        total_fees =
            total_fees.checked_add(tx_fees).ok_or(AmountOverflowError)?;
        // verify utreexo proof
        if !accumulator
            .verify(&filled_transaction.transaction.proof, &spent_utxo_hashes)?
        {
            return Err(error::ValidateBlockInner::UtreexoProofFailed { txid });
        }
    }
    if coinbase_value > total_fees {
        return Err(error::ValidateBlockInner::NotEnoughFees);
    }
    let spent_utxos = filled_transactions
        .iter()
        .flat_map(|t| t.spent_utxos.iter());
    for (index, (authorization, spent_utxo)) in
        body.authorizations.iter().zip(spent_utxos).enumerate()
    {
        if authorization.get_address() != spent_utxo.address {
            return Err(error::WrongPubKeyForAddress {
                pubkey: authorization.verifying_key,
                address: spent_utxo.address,
                index,
            }
            .into());
        }
    }
    let () = authorization::verify_body(body)?;
    let () = accumulator.apply_diff(accumulator_diff)?;
    let roots: Vec<BitcoinNodeHash> = accumulator.get_roots();
    if roots != header.roots {
        return Err(error::ValidateBlockInner::UtreexoRootsMismatch);
    }
    Ok((total_fees, merkle_root))
}

pub fn prevalidate(
    state: &State,
    rotxn: &RoTxn,
    header: &Header,
    body: &Body,
) -> Result<PrevalidatedBlock, error::ValidateBlockInner> {
    let tip_hash = state.try_get_tip(rotxn)?;
    if header.prev_side_hash != tip_hash {
        let err = error::InvalidHeaderInner::PrevSideHash {
            expected: tip_hash,
            received: header.prev_side_hash,
        };
        return Err(error::ValidateBlockInner::InvalidHeader(err.into()));
    };
    let height = state.try_get_height(rotxn)?.map_or(0, |height| height + 1);
    if body.authorizations.len() > State::body_sigops_limit(height) {
        return Err(error::ValidateBlockInner::TooManySigops);
    }
    let body_size = borsh::object_length(&body)
        .map_err(error::ValidateBlockInner::BorshSerialize)?;
    if body_size > State::body_size_limit(height) {
        return Err(error::ValidateBlockInner::BodyTooLarge);
    }
    let mut accumulator = state
        .utreexo_accumulator
        .try_get(rotxn, &())?
        .unwrap_or_default();
    let filled_transactions: Vec<_> = body
        .transactions
        .iter()
        .map(|t| {
            state.fill_transaction(rotxn, t).map_err(|err| {
                error::ValidateTransaction {
                    txid: t.txid(),
                    source: err,
                }
            })
        })
        .collect::<Result<_, _>>()?;
    let merkle_root = Body::compute_merkle_root(
        body.coinbase.as_slice(),
        filled_transactions.as_slice(),
    )?;
    let mut accumulator_diff = AccumulatorDiff::default();
    let mut coinbase_value = bitcoin::Amount::ZERO;
    for (vout, output) in body.coinbase.iter().enumerate() {
        coinbase_value = coinbase_value
            .checked_add(output.get_value())
            .ok_or(AmountOverflowError)?;
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
        let err = error::InvalidBody {
            expected: header.merkle_root,
            computed: merkle_root,
        };
        return Err(error::ValidateBlockInner::InvalidBody(err));
    }
    let mut total_fees = bitcoin::Amount::ZERO;
    let mut spent_utxos = HashSet::new();
    for filled_transaction in &filled_transactions {
        let txid = filled_transaction.transaction.txid();
        // hashes of spent utxos, used to verify the utreexo proof
        let mut spent_utxo_hashes = Vec::<BitcoinNodeHash>::new();
        for (outpoint, utxo_hash) in &filled_transaction.transaction.inputs {
            if spent_utxos.contains(outpoint) {
                return Err(error::UtxoDoubleSpent.into());
            }
            spent_utxos.insert(*outpoint);
            {
                spent_utxo_hashes.push(utxo_hash.into());
                accumulator_diff.remove(utxo_hash.into());
            }
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
        let tx_fees =
            State::validate_filled_transaction_(filled_transaction).map_err(
                |err| error::ValidateTransaction { txid, source: err },
            )?;
        total_fees =
            total_fees.checked_add(tx_fees).ok_or(AmountOverflowError)?;
        {
            // verify utreexo proof
            if !accumulator.verify(
                &filled_transaction.transaction.proof,
                &spent_utxo_hashes,
            )? {
                return Err(error::ValidateBlockInner::UtreexoProofFailed {
                    txid,
                });
            }
        }
    }
    if coinbase_value > total_fees {
        return Err(error::ValidateBlockInner::NotEnoughFees);
    }
    let spent_utxos = filled_transactions
        .iter()
        .flat_map(|t| t.spent_utxos.iter());
    for (index, (authorization, spent_utxo)) in
        body.authorizations.iter().zip(spent_utxos).enumerate()
    {
        if authorization.get_address() != spent_utxo.address {
            return Err(error::WrongPubKeyForAddress {
                pubkey: authorization.verifying_key,
                address: spent_utxo.address,
                index,
            }
            .into());
        }
    }
    let () = authorization::verify_body(body)?;
    {
        let () = accumulator.apply_diff(accumulator_diff.clone())?;
        let roots: Vec<BitcoinNodeHash> = accumulator.get_roots();
        if roots != header.roots {
            return Err(error::ValidateBlockInner::UtreexoRootsMismatch);
        }
    }
    Ok(PrevalidatedBlock {
        filled_transactions,
        computed_merkle_root: merkle_root,
        total_fees,
        coinbase_value,
        accumulator_diff,
    })
}

pub fn connect_prevalidated(
    state: &State,
    rwtxn: &mut RwTxn,
    header: &Header,
    body: &Body,
    prevalidated: PrevalidatedBlock,
) -> Result<MerkleRoot, error::ConnectBlockInner> {
    let tip_hash = state.tip.try_get(rwtxn, &())?;
    if tip_hash != header.prev_side_hash {
        let err = error::InvalidHeaderInner::PrevSideHash {
            expected: tip_hash,
            received: header.prev_side_hash,
        };
        return Err(error::ConnectBlockInner::InvalidHeader(err.into()));
    }

    let merkle_root = prevalidated.computed_merkle_root;
    if merkle_root != header.merkle_root {
        let err = error::InvalidBody {
            expected: header.merkle_root,
            computed: merkle_root,
        };
        return Err(error::ConnectBlockInner::InvalidBody(err));
    }

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
    let height = state
        .height
        .try_get(rwtxn, &())?
        .map_or(0, |height| height + 1);

    // Update tip and height using regular database operations
    state.tip.put(rwtxn, &(), &block_hash)?;
    state.height.put(rwtxn, &(), &height)?;

    // Apply utreexo accumulator diff
    {
        let () = accumulator.apply_diff(prevalidated.accumulator_diff)?;
        state.utreexo_accumulator.put(rwtxn, &(), &accumulator)?;
    }

    Ok(merkle_root)
}

fn connect_tx_(
    state: &State,
    rwtxn: &mut RwTxn,
    accumulator_diff: &mut AccumulatorDiff,
    filled_txs: &mut Vec<FilledTransaction>,
    transaction: &Transaction,
    txid: Txid,
) -> Result<(), error::ConnectTransactionInner> {
    let mut spent_utxos = Vec::new();
    for (vin, (outpoint, utxo_hash)) in transaction.inputs.iter().enumerate() {
        let spent_output =
            state.utxos.try_get(rwtxn, outpoint)?.ok_or(error::NoUtxo {
                outpoint: *outpoint,
            })?;

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
        let pointed_output = PointedOutput {
            outpoint,
            output: output.clone(),
        };
        accumulator_diff.insert((&pointed_output).into());
        state.utxos.put(rwtxn, &outpoint, output)?;
    }
    let filled_tx = FilledTransaction {
        spent_utxos,
        transaction: transaction.clone(),
    };
    filled_txs.push(filled_tx);
    Ok(())
}

fn connect_tx(
    state: &State,
    rwtxn: &mut RwTxn,
    accumulator_diff: &mut AccumulatorDiff,
    filled_txs: &mut Vec<FilledTransaction>,
    transaction: &Transaction,
) -> Result<(), error::ConnectTransaction> {
    let txid = transaction.txid();
    connect_tx_(
        state,
        rwtxn,
        accumulator_diff,
        filled_txs,
        transaction,
        txid,
    )
    .map_err(|err| error::ConnectTransaction::new(txid, err))
}

pub fn connect(
    state: &State,
    rwtxn: &mut RwTxn,
    header: &Header,
    body: &Body,
) -> Result<MerkleRoot, error::ConnectBlockInner> {
    let tip_hash = state.tip.try_get(rwtxn, &())?;
    if tip_hash != header.prev_side_hash {
        let err = error::InvalidHeaderInner::PrevSideHash {
            expected: tip_hash,
            received: header.prev_side_hash,
        };
        return Err(error::ConnectBlockInner::InvalidHeader(err.into()));
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
        state.utxos.put(rwtxn, &outpoint, output)?;
    }
    let mut filled_txs: Vec<FilledTransaction> =
        Vec::with_capacity(body.transactions.len());
    for transaction in &body.transactions {
        let () = connect_tx(
            state,
            rwtxn,
            &mut accumulator_diff,
            &mut filled_txs,
            transaction,
        )?;
    }
    let merkle_root = Body::compute_merkle_root(
        body.coinbase.as_slice(),
        filled_txs.as_slice(),
    )?;
    if merkle_root != header.merkle_root {
        let err = error::InvalidBody {
            expected: header.merkle_root,
            computed: merkle_root,
        };
        return Err(error::ConnectBlockInner::InvalidBody(err));
    }
    let block_hash = header.hash();
    let height = state
        .height
        .try_get(rwtxn, &())?
        .map_or(0, |height| height + 1);
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
) -> Result<(), error::DisconnectTipInner> {
    let tip_hash = state.try_get_tip(rwtxn)?.ok_or(error::NoTip)?;
    if tip_hash != header.hash() {
        let err = error::InvalidHeaderInner::BlockHash {
            expected: tip_hash,
            computed: header.hash(),
        };
        return Err(error::DisconnectTipInner::InvalidHeader(err.into()));
    }
    let mut accumulator = state
        .utreexo_accumulator
        .try_get(rwtxn, &())?
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
                if state.utxos.delete(rwtxn, &outpoint)? {
                    Ok::<_, error::DisconnectTipInner>(())
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
                if let Some(spent_output) =
                    state.stxos.try_get(rwtxn, outpoint)?
                {
                    accumulator_diff.insert(utxo_hash.into());
                    state.stxos.delete(rwtxn, outpoint)?;
                    state.utxos.put(rwtxn, outpoint, &spent_output.output)?;
                    Ok::<_, error::DisconnectTipInner>(())
                } else {
                    Err(error::NoStxo {
                        outpoint: *outpoint,
                    }
                    .into())
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
            if state.utxos.delete(rwtxn, &outpoint)? {
                Ok::<_, error::DisconnectTipInner>(())
            } else {
                Err(error::NoUtxo { outpoint }.into())
            }
        })?;
    let height = state
        .try_get_height(rwtxn)?
        .expect("Height should not be None");
    match (header.prev_side_hash, height) {
        (None, 0) => {
            state.tip.delete(rwtxn, &())?;
            state.height.delete(rwtxn, &())?;
        }
        (None, _) | (_, 0) => return Err(error::NoTip.into()),
        (Some(prev_side_hash), height) => {
            state.tip.put(rwtxn, &(), &prev_side_hash)?;
            state.height.put(rwtxn, &(), &(height - 1))?;
        }
    }
    let () = accumulator.apply_diff(accumulator_diff)?;
    state.utreexo_accumulator.put(rwtxn, &(), &accumulator)?;
    Ok(())
}
