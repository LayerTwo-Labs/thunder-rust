//! Connect and disconnect two-way peg data

use std::collections::{BTreeMap, HashMap};

use fallible_iterator::FallibleIterator;
use sneed::{RoTxn, RwTxn, db::error::Error as DbError};

use crate::{
    state::{
        Error, MAX_WITHDRAWAL_BUNDLE_PAYOUT_OUTPUTS, State,
        WITHDRAWAL_BUNDLE_FAILURE_GAP, WithdrawalBundleInfo,
        WithdrawalBundleScheme, error, rollback::RollBack,
    },
    types::{
        AccumulatorDiff, AggregatedWithdrawal, AmountOverflowError, Coinbase,
        CoinbaseTxid, FilledTransaction, InPoint, M6id, OutPoint, OutPointKey,
        Output, OutputContent, PointedOutput, PointedOutputRef, SpentOutput,
        WithdrawalBundle, WithdrawalBundleEvent, WithdrawalBundleEventStatus,
        WithdrawalBundleStatus, hash,
        proto::mainchain::{BlockEvent, TwoWayPegData},
    },
};

/// Check whether a new withdrawal bundle should be constructed at
/// `block_height`: the failure gap has passed since the latest
/// unsuccessful (failed or dropped) bundle, and no bundle is pending.
pub(crate) fn withdrawal_bundle_creation_due(
    state: &State,
    rotxn: &RoTxn,
    block_height: u32,
) -> Result<bool, Error> {
    let last_failure_height = state
        .get_latest_failed_withdrawal_bundle(rotxn)
        .map_err(DbError::from)?
        .map(|(height, _bundle)| height)
        .unwrap_or_default();
    let last_drop_height = state
        .dropped_withdrawal_bundles
        .try_get(rotxn, &())?
        .map(|dropped| dropped.latest().height)
        .unwrap_or_default();
    let last_unsuccessful_height = last_failure_height.max(last_drop_height);
    Ok(
        block_height - last_unsuccessful_height
            >= WITHDRAWAL_BUNDLE_FAILURE_GAP
            && state
                .pending_withdrawal_bundle
                .try_get(rotxn, &())
                .map_err(DbError::from)?
                .is_none(),
    )
}

fn min_mainchain_fee_rate(
    state: &State,
    rotxn: &RoTxn,
) -> Result<bitcoin::FeeRate, DbError> {
    Ok(state
        .withdrawal_bundle_min_mainchain_fee_rate
        .last(rotxn)?
        .map(|(_, (_height, fee_rate))| fee_rate)
        .unwrap_or(bitcoin::FeeRate::ZERO))
}

/// Record a changed adaptive fee-rate floor at `block_height`.
fn set_min_mainchain_fee_rate(
    state: &State,
    rwtxn: &mut RwTxn,
    block_height: u32,
    updated: bitcoin::FeeRate,
) -> Result<(), DbError> {
    let current = min_mainchain_fee_rate(state, rwtxn)?;
    if updated == current {
        return Ok(());
    }
    let sequence = state
        .withdrawal_bundle_min_mainchain_fee_rate
        .last(rwtxn)?
        .map_or(0, |(sequence, _)| {
            sequence.checked_add(1).expect("sequence overflow")
        });
    Ok(state.withdrawal_bundle_min_mainchain_fee_rate.put(
        rwtxn,
        &sequence,
        &(block_height, updated),
    )?)
}

/// Reset the floor to zero after a timely submission (`proposed`), or
/// double it after a timeout, raising zero to 1 sat/kwu. Construction
/// stores the applied floor so timeouts double the rate actually used.
fn update_min_mainchain_fee_rate(
    state: &State,
    rwtxn: &mut RwTxn,
    block_height: u32,
    proposed: bool,
) -> Result<(), DbError> {
    let current = min_mainchain_fee_rate(state, rwtxn)?;
    let updated = if proposed {
        bitcoin::FeeRate::ZERO
    } else if current == bitcoin::FeeRate::ZERO {
        bitcoin::FeeRate::from_sat_per_kwu(1)
    } else {
        current
            .to_sat_per_kwu()
            .checked_mul(2)
            .map_or(bitcoin::FeeRate::MAX, bitcoin::FeeRate::from_sat_per_kwu)
    };
    set_min_mainchain_fee_rate(state, rwtxn, block_height, updated)
}

/// Undo a fee-floor update made at `block_height`, if there was one.
fn rollback_min_mainchain_fee_rate(
    state: &State,
    rwtxn: &mut RwTxn,
    block_height: u32,
) -> Result<(), DbError> {
    let Some((sequence, (height, _))) =
        state.withdrawal_bundle_min_mainchain_fee_rate.last(rwtxn)?
    else {
        return Ok(());
    };
    if height == block_height {
        state
            .withdrawal_bundle_min_mainchain_fee_rate
            .delete(rwtxn, &sequence)?;
    }
    Ok(())
}

/// Aggregate one candidate withdrawal output by destination:
/// destination -> (value, mainchain fee, spent_utxos)
pub(crate) fn aggregate_withdrawal(
    address_to_aggregated_withdrawal: &mut HashMap<
        bitcoin::Address<bitcoin::address::NetworkUnchecked>,
        AggregatedWithdrawal,
    >,
    outpoint: OutPoint,
    output: Output,
) -> Result<(), Error> {
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
        aggregated.main_fee = aggregated
            .main_fee
            .checked_add(main_fee)
            .ok_or(AmountOverflowError)?;
        aggregated.spend_utxos.insert(outpoint, output);
    }
    Ok(())
}

/// Build a bundle from withdrawals aggregated by destination, using the
/// selected ordering and fee-rate floor. Returns the bundle and applied floor.
pub(crate) fn bundle_from_aggregated(
    address_to_aggregated_withdrawal: HashMap<
        bitcoin::Address<bitcoin::address::NetworkUnchecked>,
        AggregatedWithdrawal,
    >,
    block_height: u32,
    scheme: WithdrawalBundleScheme,
    min_mainchain_fee_rate: bitcoin::FeeRate,
) -> Result<Option<(WithdrawalBundle, bitcoin::FeeRate)>, Error> {
    if address_to_aggregated_withdrawal.is_empty() {
        return Ok(None);
    }
    let adaptive = scheme == WithdrawalBundleScheme::Adaptive;
    let mut aggregated_withdrawals: Vec<_> =
        address_to_aggregated_withdrawal.into_values().collect();
    match scheme {
        // Preserve pre-activation payout order exactly: it determines the M6ID.
        WithdrawalBundleScheme::Legacy => {
            aggregated_withdrawals.sort_by(|a, b| b.legacy_cmp(a));
        }
        WithdrawalBundleScheme::Adaptive => {
            aggregated_withdrawals.sort_by(|a, b| b.fee_rate_cmp(a));
        }
    }
    // Cap the floor at the best available payout rate so construction is
    // never suppressed.
    let min_mainchain_fee_rate = if adaptive {
        let max_payout_fee_rate = aggregated_withdrawals
            .first()
            .and_then(AggregatedWithdrawal::payout_fee_rate)
            .unwrap_or(bitcoin::FeeRate::ZERO);
        min_mainchain_fee_rate.min(max_payout_fee_rate)
    } else {
        min_mainchain_fee_rate
    };
    let mut fee = bitcoin::Amount::ZERO;
    let mut spend_utxos = BTreeMap::<OutPoint, Output>::new();
    let mut bundle_outputs = Vec::new();
    let mut bundle_txouts_size: u32 = 0;
    for aggregated in &aggregated_withdrawals {
        let script_pubkey =
            aggregated.main_address.assume_checked_ref().script_pubkey();
        let Ok(n_outputs) = u32::try_from(bundle_outputs.len() + 1) else {
            break;
        };
        if adaptive && n_outputs > MAX_WITHDRAWAL_BUNDLE_PAYOUT_OUTPUTS {
            break;
        }
        let Ok(spk_size) = u32::try_from(script_pubkey.len()) else {
            // This SPK is invalid, but others might be ok
            continue;
        };
        let Some(txout_size) = WithdrawalBundle::txout_size(spk_size) else {
            // This SPK is invalid, but others might be ok
            continue;
        };
        if adaptive {
            let txout_weight = bitcoin::Weight::from_wu(
                u64::from(txout_size) * bitcoin::Weight::WITNESS_SCALE_FACTOR,
            );
            let Some(min_mainchain_fee) =
                min_mainchain_fee_rate.fee_wu(txout_weight)
            else {
                // An overflowing fee requirement cannot be satisfied.
                continue;
            };
            if aggregated.main_fee < min_mainchain_fee {
                continue;
            }
        }
        if let Some(sum_txout_sizes) =
            bundle_txouts_size.checked_add(txout_size)
        {
            bundle_txouts_size = sum_txout_sizes;
        } else {
            break;
        };
        if WithdrawalBundle::predict_weight(n_outputs, bundle_txouts_size)
            .is_none()
        {
            break;
        }
        let bundle_output = bitcoin::TxOut {
            value: aggregated.value,
            script_pubkey,
        };
        spend_utxos.extend(aggregated.spend_utxos.clone());
        bundle_outputs.push(bundle_output);
        fee += aggregated.main_fee;
    }
    if adaptive && bundle_outputs.is_empty() {
        // No candidate has a valid payout output.
        return Ok(None);
    }
    let bundle =
        WithdrawalBundle::new(block_height, fee, spend_utxos, bundle_outputs)?;
    Ok(Some((bundle, min_mainchain_fee_rate)))
}

/// Collect a withdrawal bundle and its applied fee-rate floor.
fn collect_withdrawal_bundle(
    state: &State,
    rotxn: &RoTxn,
    block_height: u32,
    scheme: WithdrawalBundleScheme,
) -> Result<Option<(WithdrawalBundle, bitcoin::FeeRate)>, Error> {
    let mut aggregated = HashMap::new();
    let mut candidates = state.utxos.iter(rotxn).map_err(DbError::from)?;
    while let Some((outpoint_key, output)) =
        candidates.next().map_err(DbError::from)?
    {
        aggregate_withdrawal(&mut aggregated, outpoint_key.into(), output)?;
    }
    let min_mainchain_fee_rate = min_mainchain_fee_rate(state, rotxn)?;
    bundle_from_aggregated(
        aggregated,
        block_height,
        scheme,
        min_mainchain_fee_rate,
    )
}

/// Simulate withdrawal bundle construction at `block_height`: compute
/// the accumulator diff that constructing a new bundle would produce,
/// from the current UTXO set, the block's transactions, and coinbase.
/// Read-only; used to compute header roots, which commit to the
/// post-events + post-txs + post-construction accumulator.
pub(crate) fn simulate_withdrawal_bundle_construction(
    state: &State,
    rotxn: &RoTxn,
    block_height: u32,
    scheme: WithdrawalBundleScheme,
    filled_transactions: &[FilledTransaction],
    coinbase: &Coinbase,
    coinbase_txid: CoinbaseTxid,
) -> Result<AccumulatorDiff, Error> {
    let mut construction_diff = AccumulatorDiff::default();
    if !withdrawal_bundle_creation_due(state, rotxn, block_height)? {
        return Ok(construction_diff);
    }
    // Candidates: all withdrawal UTXOs, minus withdrawal outputs spent
    // by the block's txs, plus withdrawal outputs created by the
    // block's txs and coinbase.
    let mut spent_withdrawal_outpoints = std::collections::HashSet::new();
    for filled in filled_transactions {
        for ((outpoint, _), spent_utxo) in filled
            .transaction
            .inputs
            .iter()
            .zip(filled.spent_utxos.iter())
        {
            if spent_utxo.content.is_withdrawal() {
                spent_withdrawal_outpoints.insert(*outpoint);
            }
        }
    }
    let mut aggregated = HashMap::new();
    let mut db_candidates = state.utxos.iter(rotxn).map_err(DbError::from)?;
    while let Some((outpoint_key, output)) =
        db_candidates.next().map_err(DbError::from)?
    {
        let outpoint: OutPoint = outpoint_key.into();
        if spent_withdrawal_outpoints.contains(&outpoint) {
            continue;
        }
        aggregate_withdrawal(&mut aggregated, outpoint, output)?;
    }
    for filled in filled_transactions {
        let txid = filled.transaction.txid();
        for (vout, output) in filled.transaction.outputs.iter().enumerate() {
            aggregate_withdrawal(
                &mut aggregated,
                OutPoint::Regular {
                    txid,
                    vout: vout as u32,
                },
                output.clone(),
            )?;
        }
    }
    for (vout, output) in coinbase.outputs.iter().enumerate() {
        aggregate_withdrawal(
            &mut aggregated,
            OutPoint::Coinbase {
                txid: coinbase_txid,
                vout: vout as u32,
            },
            output.clone(),
        )?;
    }
    let min_mainchain_fee_rate = min_mainchain_fee_rate(state, rotxn)?;
    if let Some((bundle, _applied_fee_rate)) = bundle_from_aggregated(
        aggregated,
        block_height,
        scheme,
        min_mainchain_fee_rate,
    )? {
        for (outpoint, output) in bundle.spend_utxos() {
            let utxo_hash = hash(&PointedOutputRef {
                outpoint: *outpoint,
                output,
            });
            construction_diff.remove(utxo_hash.into());
        }
    }
    Ok(construction_diff)
}

/// Outpoint for an output refunded by an unsuccessful bundle.
fn unsuccessful_withdrawal_outpoint(
    m6id: M6id,
    block_height: u32,
    vout: usize,
) -> OutPoint {
    let vout = u32::try_from(vout).expect("bundle outputs fit in u32");
    OutPoint::UnsuccessfulWithdrawal {
        m6id,
        height: block_height,
        vout,
    }
}

/// Recreate the outputs spent by an unsuccessful (failed or dropped)
/// withdrawal bundle as new UTXOs at
/// `OutPoint::UnsuccessfulWithdrawal` outpoints. The original outputs
/// remain spent by the bundle; their STXO records are untouched.
fn recreate_bundle_outputs(
    state: &State,
    rwtxn: &mut RwTxn,
    accumulator_diff: &mut AccumulatorDiff,
    block_height: u32,
    m6id: M6id,
    bundle: &WithdrawalBundle,
) -> Result<(), Error> {
    for (vout, (_, output)) in bundle.spend_utxos().iter().enumerate() {
        let outpoint =
            unsuccessful_withdrawal_outpoint(m6id, block_height, vout);
        state
            .utxos
            .put(rwtxn, &OutPointKey::from(&outpoint), output)?;
        accumulator_diff
            .insert(hash(&PointedOutputRef { outpoint, output }).into());
    }
    Ok(())
}

/// Delete the outputs recreated from an unsuccessful (failed or
/// dropped) withdrawal bundle, last-to-first. The original outputs
/// remain spent by the bundle; their STXO records are untouched.
fn delete_recreated_outputs(
    state: &State,
    rwtxn: &mut RwTxn,
    accumulator_diff: &mut AccumulatorDiff,
    block_height: u32,
    m6id: M6id,
    bundle: &WithdrawalBundle,
) -> Result<(), Error> {
    for (vout, (_, output)) in bundle.spend_utxos().iter().enumerate().rev() {
        let outpoint =
            unsuccessful_withdrawal_outpoint(m6id, block_height, vout);
        if !state.utxos.delete(rwtxn, &OutPointKey::from(&outpoint))? {
            return Err(error::NoUtxo { outpoint }.into());
        }
        accumulator_diff
            .remove(hash(&PointedOutputRef { outpoint, output }).into());
    }
    Ok(())
}

/// Drop a stale pending withdrawal bundle: if it was not submitted
/// within `WITHDRAWAL_BUNDLE_FAILURE_GAP` blocks, mark it dropped and
/// recreate its outputs, so that the block's transactions can spend
/// them and, if unspent, they can be included in the fresh bundle
/// constructed after the block's transactions.
fn maybe_drop_stale_pending_bundle(
    state: &State,
    rwtxn: &mut RwTxn,
    accumulator_diff: &mut AccumulatorDiff,
    block_height: u32,
) -> Result<(), Error> {
    let Some(m6id) = state.pending_withdrawal_bundle.try_get(rwtxn, &())?
    else {
        return Ok(());
    };
    let (bundle, mut bundle_status) =
        state.withdrawal_bundles.get(rwtxn, &m6id)?;
    let pending_scheme =
        state.withdrawal_bundle_scheme(bundle_status.latest().height);
    debug_assert_eq!(
        bundle_status.latest().value,
        WithdrawalBundleStatus::Pending
    );
    let pending_height = bundle_status.latest().height;
    if block_height - pending_height < WITHDRAWAL_BUNDLE_FAILURE_GAP {
        return Ok(());
    }
    let bundle = match bundle {
        WithdrawalBundleInfo::Known(bundle) => bundle,
        WithdrawalBundleInfo::Unknown
        | WithdrawalBundleInfo::UnknownConfirmed { .. } => {
            return Err(error::PendingWithdrawalBundleUnknown(m6id).into());
        }
    };
    recreate_bundle_outputs(
        state,
        rwtxn,
        accumulator_diff,
        block_height,
        m6id,
        &bundle,
    )?;
    state.pending_withdrawal_bundle.delete(rwtxn, &())?;
    bundle_status
        .push(WithdrawalBundleStatus::Dropped, block_height)
        .expect("push dropped status should be valid");
    state.withdrawal_bundles.put(
        rwtxn,
        &m6id,
        &(WithdrawalBundleInfo::Known(bundle), bundle_status),
    )?;
    // Record the drop so that it can be found again on disconnect.
    let dropped = if let Some(mut dropped) =
        state.dropped_withdrawal_bundles.try_get(rwtxn, &())?
    {
        dropped
            .push(m6id, block_height)
            .expect("push dropped m6id should be valid");
        dropped
    } else {
        RollBack::new(m6id, block_height)
    };
    state.dropped_withdrawal_bundles.put(rwtxn, &(), &dropped)?;
    if pending_scheme == WithdrawalBundleScheme::Adaptive {
        update_min_mainchain_fee_rate(state, rwtxn, block_height, false)?;
    }
    tracing::warn!(
        %block_height,
        %m6id,
        "Pending withdrawal bundle dropped: not submitted in time"
    );
    Ok(())
}

/// Undo a stale-bundle drop performed at `block_height`, if any:
/// delete the recreated outputs and restore the pending marker.
/// Drops are connected after the block's events, so they are undone
/// before disconnecting the events.
fn disconnect_dropped_bundle(
    state: &State,
    rwtxn: &mut RwTxn,
    accumulator_diff: &mut AccumulatorDiff,
    block_height: u32,
) -> Result<(), Error> {
    let Some(dropped) = state.dropped_withdrawal_bundles.try_get(rwtxn, &())?
    else {
        return Ok(());
    };
    if dropped.latest().height != block_height {
        return Ok(());
    }
    let m6id = dropped.latest().value;
    let (prev_dropped, _latest_dropped) = dropped.pop();
    if let Some(prev_dropped) = prev_dropped {
        state
            .dropped_withdrawal_bundles
            .put(rwtxn, &(), &prev_dropped)?;
    } else {
        state.dropped_withdrawal_bundles.delete(rwtxn, &())?;
    }
    let (bundle, bundle_status) = state
        .withdrawal_bundles
        .try_get(rwtxn, &m6id)?
        .ok_or(Error::UnknownWithdrawalBundle { m6id })?;
    let (prev_bundle_status, latest_bundle_status) = bundle_status.pop();
    assert_eq!(latest_bundle_status.value, WithdrawalBundleStatus::Dropped);
    assert_eq!(latest_bundle_status.height, block_height);
    let prev_bundle_status =
        prev_bundle_status.expect("pop dropped bundle status should be valid");
    assert_eq!(
        prev_bundle_status.latest().value,
        WithdrawalBundleStatus::Pending
    );
    let bundle = match bundle {
        WithdrawalBundleInfo::Known(bundle) => bundle,
        WithdrawalBundleInfo::Unknown
        | WithdrawalBundleInfo::UnknownConfirmed { .. } => {
            return Err(error::PendingWithdrawalBundleUnknown(m6id).into());
        }
    };
    let pending_scheme =
        state.withdrawal_bundle_scheme(prev_bundle_status.latest().height);
    delete_recreated_outputs(
        state,
        rwtxn,
        accumulator_diff,
        block_height,
        m6id,
        &bundle,
    )?;
    state.pending_withdrawal_bundle.put(rwtxn, &(), &m6id)?;
    state.withdrawal_bundles.put(
        rwtxn,
        &m6id,
        &(WithdrawalBundleInfo::Known(bundle), prev_bundle_status),
    )?;
    if pending_scheme == WithdrawalBundleScheme::Adaptive {
        rollback_min_mainchain_fee_rate(state, rwtxn, block_height)?;
    }
    Ok(())
}

fn connect_withdrawal_bundle_submitted(
    state: &State,
    rwtxn: &mut RwTxn,
    block_height: u32,
    event_block_hash: &bitcoin::BlockHash,
    m6id: M6id,
) -> Result<(), error::ConnectWithdrawalBundleSubmitted> {
    if let Some(bundle_m6id) =
        state.pending_withdrawal_bundle.try_get(rwtxn, &())?
        && bundle_m6id == m6id
    {
        tracing::debug!(
            %block_height,
            %m6id,
            "Pending withdrawal bundle submission confirmed"
        );
        let (bundle, mut bundle_status) = state
            .withdrawal_bundles
            .try_get(rwtxn, &m6id)?
            .ok_or(error::PendingWithdrawalBundleUnknown(m6id))?;
        let bundle = match bundle {
            WithdrawalBundleInfo::Known(bundle) => bundle,
            WithdrawalBundleInfo::Unknown
            | WithdrawalBundleInfo::UnknownConfirmed { spend_utxos: _ } => {
                let err = error::PendingWithdrawalBundleUnknown(m6id);
                return Err(err.into());
            }
        };
        let pending_scheme =
            state.withdrawal_bundle_scheme(bundle_status.latest().height);
        // The bundle's spend_utxos were already spent when the bundle
        // was created; only the status changes here.
        for outpoint in bundle.spend_utxos().keys() {
            debug_assert!(
                state
                    .utxos
                    .try_get(rwtxn, &OutPointKey::from(outpoint))?
                    .is_none(),
                "pending bundle output {outpoint} should already be spent"
            );
        }
        assert_eq!(
            bundle_status.latest().value,
            WithdrawalBundleStatus::Pending
        );
        bundle_status
            .push(WithdrawalBundleStatus::Submitted, block_height)
            .expect("push submitted status should be valid");
        state.withdrawal_bundles.put(
            rwtxn,
            &m6id,
            &(WithdrawalBundleInfo::Known(bundle), bundle_status),
        )?;
        state.pending_withdrawal_bundle.delete(rwtxn, &())?;
        if pending_scheme == WithdrawalBundleScheme::Adaptive {
            update_min_mainchain_fee_rate(state, rwtxn, block_height, true)?;
        }
    } else if let Some((bundle, mut bundle_status)) =
        state.withdrawal_bundles.try_get(rwtxn, &m6id)?
    {
        match (&bundle, bundle_status.latest().value) {
            (_, WithdrawalBundleStatus::Confirmed) => {
                let err = error::ConnectWithdrawalBundleSubmitted::ConfirmedResubmitted {
                    event_block_hash: *event_block_hash,
                    m6id
                };
                return Err(err);
            }
            (
                _,
                WithdrawalBundleStatus::Submitted
                | WithdrawalBundleStatus::SubmittedUnexpected,
            ) => {
                let err =
                    error::ConnectWithdrawalBundleSubmitted::Resubmitted {
                        event_block_hash: *event_block_hash,
                        m6id,
                        submitted_block_height: bundle_status.latest().height,
                    };
                return Err(err);
            }
            (
                WithdrawalBundleInfo::Known(_),
                WithdrawalBundleStatus::Dropped,
            ) => {
                tracing::warn!(%event_block_hash, %m6id, "dropped bundle submitted");
            }
            (
                WithdrawalBundleInfo::Unknown
                | WithdrawalBundleInfo::UnknownConfirmed { spend_utxos: _ },
                WithdrawalBundleStatus::Dropped,
            ) => {
                let err =
                    error::ConnectWithdrawalBundleSubmitted::UnknownDropped {
                        m6id,
                        dropped_block_height: bundle_status.latest().height,
                    };
                return Err(err);
            }
            (
                WithdrawalBundleInfo::Known(_),
                WithdrawalBundleStatus::Pending,
            ) => {
                let err =
                    error::ConnectWithdrawalBundleSubmitted::DroppedPending(
                        m6id,
                    );
                return Err(err);
            }
            (
                WithdrawalBundleInfo::Unknown
                | WithdrawalBundleInfo::UnknownConfirmed { spend_utxos: _ },
                WithdrawalBundleStatus::Pending,
            ) => {
                let err =
                    error::ConnectWithdrawalBundleSubmitted::UnknownPending {
                        m6id,
                        pending_block_height: bundle_status.latest().height,
                    };
                return Err(err);
            }
            (
                WithdrawalBundleInfo::Known(_) | WithdrawalBundleInfo::Unknown,
                WithdrawalBundleStatus::Failed,
            ) => {
                tracing::warn!(%event_block_hash, %m6id, "failed bundle resubmitted");
            }
            (
                WithdrawalBundleInfo::UnknownConfirmed { spend_utxos: _ },
                WithdrawalBundleStatus::Failed,
            ) => {
                let err = error::ConnectWithdrawalBundleSubmitted::UnknownConfirmedFailed {
                    m6id,
                    failed_block_height: bundle_status.latest().height,
                };
                return Err(err);
            }
        }
        bundle_status
            .push(WithdrawalBundleStatus::SubmittedUnexpected, block_height)
            .expect("push submitted unexpected status should be valid");
        state
            .withdrawal_bundles
            .put(rwtxn, &m6id, &(bundle, bundle_status))?
    } else {
        tracing::warn!(
            %event_block_hash,
            %m6id,
            "Unknown withdrawal bundle submitted"
        );
        state.withdrawal_bundles.put(
            rwtxn,
            &m6id,
            &(
                WithdrawalBundleInfo::Unknown,
                RollBack::new(WithdrawalBundleStatus::Submitted, block_height),
            ),
        )?;
    };
    Ok(())
}

fn connect_withdrawal_bundle_confirmed(
    state: &State,
    rwtxn: &mut RwTxn,
    block_height: u32,
    accumulator_diff: &mut AccumulatorDiff,
    event_block_hash: &bitcoin::BlockHash,
    m6id: M6id,
) -> Result<(), Error> {
    let (mut bundle, mut bundle_status) = state
        .withdrawal_bundles
        .try_get(rwtxn, &m6id)
        .map_err(DbError::from)?
        .ok_or(Error::UnknownWithdrawalBundle { m6id })?;
    if bundle_status.latest().value == WithdrawalBundleStatus::Confirmed {
        // Already applied
        return Ok(());
    }
    assert!(matches!(
        bundle_status.latest().value,
        WithdrawalBundleStatus::Submitted
            | WithdrawalBundleStatus::SubmittedUnexpected
    ));
    match &bundle {
        WithdrawalBundleInfo::UnknownConfirmed { spend_utxos: _ } => {
            return Err(Error::UnknownWithdrawalBundleReconfirmed {
                event_block_hash: *event_block_hash,
                m6id,
            });
        }
        WithdrawalBundleInfo::Unknown => {
            // If an unknown bundle is confirmed, all UTXOs older than the
            // bundle submission are potentially spent.
            // This is only accepted in the case that block height is 0,
            // and so no UTXOs could possibly have been double-spent yet.
            // In this case, ALL UTXOs are considered spent.
            if block_height == 0 {
                tracing::warn!(
                    %event_block_hash,
                    %m6id,
                    "Unknown withdrawal bundle confirmed, marking all UTXOs as spent"
                );
                let utxos: BTreeMap<OutPoint, Output> = state
                    .utxos
                    .iter(rwtxn)
                    .map_err(DbError::from)?
                    .map(|(key, output)| Ok((key.into(), output)))
                    .collect()
                    .map_err(DbError::from)?;
                for (outpoint, output) in &utxos {
                    let spent_output = SpentOutput {
                        output: output.clone(),
                        inpoint: InPoint::Withdrawal { m6id },
                    };
                    state
                        .stxos
                        .put(rwtxn, &OutPointKey::from(outpoint), &spent_output)
                        .map_err(DbError::from)?;
                    let utxo_hash = hash(&PointedOutputRef {
                        outpoint: *outpoint,
                        output: &spent_output.output,
                    });
                    accumulator_diff.remove(utxo_hash.into());
                }
                state.utxos.clear(rwtxn).map_err(DbError::from)?;
                bundle = WithdrawalBundleInfo::UnknownConfirmed {
                    spend_utxos: utxos,
                };
            } else {
                return Err(Error::UnknownWithdrawalBundleConfirmed {
                    event_block_hash: *event_block_hash,
                    m6id,
                });
            }
        }
        WithdrawalBundleInfo::Known(_) => {
            if matches!(
                bundle_status.latest().value,
                WithdrawalBundleStatus::SubmittedUnexpected
            ) {
                // A previously failed or dropped bundle was confirmed.
                // Its outputs were already spent when it was created,
                // and recreated when it became unsuccessful; confirming
                // it would pay the mainchain without removing the
                // recreated claims.
                return Err(Error::UnexpectedWithdrawalBundleConfirmed {
                    event_block_hash: *event_block_hash,
                    m6id,
                });
            }
        }
    }
    bundle_status
        .push(WithdrawalBundleStatus::Confirmed, block_height)
        .expect("Push confirmed status should be valid");
    state
        .withdrawal_bundles
        .put(rwtxn, &m6id, &(bundle, bundle_status))
        .map_err(DbError::from)?;
    Ok(())
}

fn connect_withdrawal_bundle_failed(
    state: &State,
    rwtxn: &mut RwTxn,
    block_height: u32,
    accumulator_diff: &mut AccumulatorDiff,
    m6id: M6id,
) -> Result<(), Error> {
    tracing::debug!(
        %block_height,
        %m6id,
        "Handling failed withdrawal bundle");
    let (bundle, mut bundle_status) = state
        .withdrawal_bundles
        .try_get(rwtxn, &m6id)
        .map_err(DbError::from)?
        .ok_or_else(|| Error::UnknownWithdrawalBundle { m6id })?;
    if bundle_status.latest().value == WithdrawalBundleStatus::Failed {
        // Already applied
        return Ok(());
    }
    assert!(matches!(
        bundle_status.latest().value,
        WithdrawalBundleStatus::Submitted
            | WithdrawalBundleStatus::SubmittedUnexpected
    ));
    match &bundle {
        WithdrawalBundleInfo::Unknown
        | WithdrawalBundleInfo::UnknownConfirmed { .. } => (),
        WithdrawalBundleInfo::Known(bundle) => 'known: {
            if matches!(
                bundle_status.latest().value,
                WithdrawalBundleStatus::SubmittedUnexpected
            ) {
                break 'known;
            }
            // Recreate the spent outputs at outpoints derived from the
            // failed bundle's M6ID and the current block height, so
            // that they are spendable again (e.g. to cancel the
            // withdrawal). The original outputs remain spent by the
            // bundle.
            recreate_bundle_outputs(
                state,
                rwtxn,
                accumulator_diff,
                block_height,
                m6id,
                bundle,
            )?;
            let latest_failed_m6id = if let Some(mut latest_failed_m6id) = state
                .latest_failed_withdrawal_bundle
                .try_get(rwtxn, &())
                .map_err(DbError::from)?
            {
                latest_failed_m6id
                    .push(m6id, block_height)
                    .expect("Push latest failed m6id should be valid");
                latest_failed_m6id
            } else {
                RollBack::new(m6id, block_height)
            };
            state
                .latest_failed_withdrawal_bundle
                .put(rwtxn, &(), &latest_failed_m6id)
                .map_err(DbError::from)?;
        }
    }
    bundle_status
        .push(WithdrawalBundleStatus::Failed, block_height)
        .expect("Push failed status should be valid");
    state
        .withdrawal_bundles
        .put(rwtxn, &m6id, &(bundle, bundle_status))
        .map_err(DbError::from)?;
    Ok(())
}

fn connect_withdrawal_bundle_event(
    state: &State,
    rwtxn: &mut RwTxn,
    block_height: u32,
    accumulator_diff: &mut AccumulatorDiff,
    event_block_hash: &bitcoin::BlockHash,
    event: &WithdrawalBundleEvent,
) -> Result<(), Error> {
    match event.status {
        WithdrawalBundleEventStatus::Submitted => {
            connect_withdrawal_bundle_submitted(
                state,
                rwtxn,
                block_height,
                event_block_hash,
                event.m6id,
            )
            .map_err(Error::ConnectWithdrawalBundleSubmitted)
        }
        WithdrawalBundleEventStatus::Confirmed => {
            connect_withdrawal_bundle_confirmed(
                state,
                rwtxn,
                block_height,
                accumulator_diff,
                event_block_hash,
                event.m6id,
            )
        }
        WithdrawalBundleEventStatus::Failed => {
            connect_withdrawal_bundle_failed(
                state,
                rwtxn,
                block_height,
                accumulator_diff,
                event.m6id,
            )
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn connect_event(
    state: &State,
    rwtxn: &mut RwTxn,
    block_height: u32,
    accumulator_diff: &mut AccumulatorDiff,
    latest_deposit_block_hash: &mut Option<bitcoin::BlockHash>,
    latest_withdrawal_bundle_event_block_hash: &mut Option<bitcoin::BlockHash>,
    event_block_hash: bitcoin::BlockHash,
    event_block_height: u32,
    event: &BlockEvent,
) -> Result<(), Error> {
    match event {
        BlockEvent::Deposit(deposit) => {
            state.deposit_mainchain_height.put(
                rwtxn,
                &deposit.outpoint.txid,
                &event_block_height,
            )?;
            let outpoint = OutPoint::Deposit(deposit.outpoint);
            let output = &deposit.output;
            state
                .utxos
                .put(rwtxn, &OutPointKey::from(&outpoint), output)
                .map_err(DbError::from)?;
            let utxo_hash = hash(&PointedOutputRef { outpoint, output });
            accumulator_diff.insert(utxo_hash.into());
            *latest_deposit_block_hash = Some(event_block_hash);
        }
        BlockEvent::WithdrawalBundle(withdrawal_bundle_event) => {
            let () = connect_withdrawal_bundle_event(
                state,
                rwtxn,
                block_height,
                accumulator_diff,
                &event_block_hash,
                withdrawal_bundle_event,
            )?;
            *latest_withdrawal_bundle_event_block_hash = Some(event_block_hash);
        }
    }
    Ok(())
}

pub fn connect(
    state: &State,
    rwtxn: &mut RwTxn,
    two_way_peg_data: &TwoWayPegData,
) -> Result<(), Error> {
    // 2WPD is connected BEFORE the block's transactions; the block
    // being connected is one higher than the current tip (0 at
    // genesis).
    let block_height =
        state.try_get_height(rwtxn)?.map_or(0, |height| height + 1);
    tracing::trace!(%block_height, "Connecting 2WPD...");
    let mut accumulator = state
        .utreexo_accumulator
        .try_get(rwtxn, &())
        .map_err(DbError::from)?
        .unwrap_or_default();
    let mut accumulator_diff = AccumulatorDiff::default();
    let mut latest_deposit_block_hash = None;
    let mut latest_withdrawal_bundle_event_block_hash = None;
    for (event_block_hash, event_block_info) in &two_way_peg_data.block_info {
        for event in &event_block_info.inner.events {
            let () = connect_event(
                state,
                rwtxn,
                block_height,
                &mut accumulator_diff,
                &mut latest_deposit_block_hash,
                &mut latest_withdrawal_bundle_event_block_hash,
                *event_block_hash,
                event_block_info.height,
                event,
            )?;
        }
    }
    // Handle deposits.
    if let Some(latest_deposit_block_hash) = latest_deposit_block_hash {
        let deposit_block_seq_idx = state
            .deposit_blocks
            .last(rwtxn)
            .map_err(DbError::from)?
            .map_or(0, |(seq_idx, _)| seq_idx + 1);
        state
            .deposit_blocks
            .put(
                rwtxn,
                &deposit_block_seq_idx,
                &(latest_deposit_block_hash, block_height),
            )
            .map_err(DbError::from)?;
    }
    // Handle withdrawals
    if let Some(latest_withdrawal_bundle_event_block_hash) =
        latest_withdrawal_bundle_event_block_hash
    {
        let withdrawal_bundle_event_block_seq_idx = state
            .withdrawal_bundle_event_blocks
            .last(rwtxn)
            .map_err(DbError::from)?
            .map_or(0, |(seq_idx, _)| seq_idx + 1);
        state
            .withdrawal_bundle_event_blocks
            .put(
                rwtxn,
                &withdrawal_bundle_event_block_seq_idx,
                &(latest_withdrawal_bundle_event_block_hash, block_height),
            )
            .map_err(DbError::from)?;
    }
    // Drop a stale pending bundle: if it was not submitted within
    // WITHDRAWAL_BUNDLE_FAILURE_GAP blocks, recreate its outputs so
    // that this block's txs can spend them and, if unspent, they can
    // be included in the fresh bundle constructed after this block's
    // txs.
    let () = maybe_drop_stale_pending_bundle(
        state,
        rwtxn,
        &mut accumulator_diff,
        block_height,
    )?;
    let () = accumulator.apply_diff(accumulator_diff)?;
    state
        .utreexo_accumulator
        .put(rwtxn, &(), &accumulator)
        .map_err(DbError::from)?;
    Ok(())
}

/// Construct a new pending withdrawal bundle if due, spending the
/// selected withdrawal outputs.
/// MUST be applied AFTER the block's transactions have been connected.
pub fn maybe_create_withdrawal_bundle(
    state: &State,
    rwtxn: &mut RwTxn,
) -> Result<(), Error> {
    let block_height = state.try_get_height(rwtxn)?.ok_or(Error::NoTip)?;
    if !withdrawal_bundle_creation_due(state, rwtxn, block_height)? {
        return Ok(());
    }
    let scheme = state.withdrawal_bundle_scheme(block_height);
    let Some((bundle, applied_fee_rate)) =
        collect_withdrawal_bundle(state, rwtxn, block_height, scheme)?
    else {
        return Ok(());
    };
    if scheme == WithdrawalBundleScheme::Adaptive {
        // Persist the applied floor so timeouts double the rate actually used.
        set_min_mainchain_fee_rate(
            state,
            rwtxn,
            block_height,
            applied_fee_rate,
        )?;
    }
    let m6id = bundle.compute_m6id();
    let mut accumulator = state
        .utreexo_accumulator
        .try_get(rwtxn, &())?
        .unwrap_or_default();
    let mut accumulator_diff = AccumulatorDiff::default();
    // Spend the selected outputs.
    for (outpoint, output) in bundle.spend_utxos() {
        let key = OutPointKey::from(outpoint);
        if !state.utxos.delete(rwtxn, &key)? {
            return Err(error::NoUtxo {
                outpoint: *outpoint,
            }
            .into());
        }
        let spent_output = SpentOutput {
            output: output.clone(),
            inpoint: InPoint::Withdrawal { m6id },
        };
        state.stxos.put(rwtxn, &key, &spent_output)?;
        let utxo_hash = hash(&PointedOutputRef {
            outpoint: *outpoint,
            output,
        });
        accumulator_diff.remove(utxo_hash.into());
    }
    state.pending_withdrawal_bundle.put(rwtxn, &(), &m6id)?;
    let bundle_status = if let Some((_bundle, mut bundle_status)) =
        state.withdrawal_bundles.try_get(rwtxn, &m6id)?
    {
        bundle_status
            .push(WithdrawalBundleStatus::Pending, block_height)
            .expect("push pending status should be valid");
        bundle_status
    } else {
        RollBack::new(WithdrawalBundleStatus::Pending, block_height)
    };
    state.withdrawal_bundles.put(
        rwtxn,
        &m6id,
        &(WithdrawalBundleInfo::Known(bundle), bundle_status),
    )?;
    let () = accumulator.apply_diff(accumulator_diff)?;
    state
        .utreexo_accumulator
        .put(rwtxn, &(), &accumulator)
        .map_err(DbError::from)?;
    tracing::trace!(
        %block_height,
        %m6id,
        "Stored pending withdrawal bundle"
    );
    Ok(())
}

fn disconnect_withdrawal_bundle_submitted(
    state: &State,
    rwtxn: &mut RwTxn,
    block_height: u32,
    m6id: M6id,
) -> Result<(), Error> {
    let Some((bundle, bundle_status)) =
        state.withdrawal_bundles.try_get(rwtxn, &m6id)?
    else {
        if let Some(pending_bundle_m6id) =
            state.pending_withdrawal_bundle.try_get(rwtxn, &())?
            && pending_bundle_m6id == m6id
        {
            // Already applied
            return Ok(());
        } else {
            return Err(Error::UnknownWithdrawalBundle { m6id });
        }
    };
    let (bundle_status, latest_bundle_status) = bundle_status.pop();
    assert!(matches!(
        latest_bundle_status.value,
        WithdrawalBundleStatus::Submitted
            | WithdrawalBundleStatus::SubmittedUnexpected
    ));
    assert_eq!(latest_bundle_status.height, block_height);
    let mut submitted_scheme = None;
    match &bundle {
        WithdrawalBundleInfo::Unknown
        | WithdrawalBundleInfo::UnknownConfirmed { .. } => (),
        WithdrawalBundleInfo::Known(_bundle) => {
            if let Some(bundle_status) = &bundle_status
                && bundle_status.latest().value
                    == WithdrawalBundleStatus::Pending
            {
                submitted_scheme =
                    Some(state.withdrawal_bundle_scheme(
                        bundle_status.latest().height,
                    ));
                // The bundle's outputs remain spent; only the pending
                // marker is restored.
                state.pending_withdrawal_bundle.put(rwtxn, &(), &m6id)?;
            }
        }
    }
    if let Some(bundle_status) = bundle_status {
        state
            .withdrawal_bundles
            .put(rwtxn, &m6id, &(bundle, bundle_status))?;
    } else {
        state.withdrawal_bundles.delete(rwtxn, &m6id)?;
    }
    if submitted_scheme == Some(WithdrawalBundleScheme::Adaptive) {
        rollback_min_mainchain_fee_rate(state, rwtxn, block_height)?;
    }
    Ok(())
}

fn disconnect_withdrawal_bundle_confirmed(
    state: &State,
    rwtxn: &mut RwTxn,
    block_height: u32,
    accumulator_diff: &mut AccumulatorDiff,
    m6id: M6id,
) -> Result<(), Error> {
    let (mut bundle, bundle_status) = state
        .withdrawal_bundles
        .try_get(rwtxn, &m6id)
        .map_err(DbError::from)?
        .ok_or_else(|| Error::UnknownWithdrawalBundle { m6id })?;
    let (prev_bundle_status, latest_bundle_status) = bundle_status.pop();
    if matches!(
        latest_bundle_status.value,
        WithdrawalBundleStatus::Submitted
            | WithdrawalBundleStatus::SubmittedUnexpected
    ) {
        // Already applied
        return Ok(());
    }
    assert_eq!(
        latest_bundle_status.value,
        WithdrawalBundleStatus::Confirmed
    );
    assert_eq!(latest_bundle_status.height, block_height);
    let prev_bundle_status = prev_bundle_status
        .expect("Pop confirmed bundle status should be valid");
    assert!(matches!(
        prev_bundle_status.latest().value,
        WithdrawalBundleStatus::Submitted
            | WithdrawalBundleStatus::SubmittedUnexpected
    ));
    match &bundle {
        // Confirming a known bundle never changes the UTXO set: its
        // outputs were already spent when it was created. Confirming an
        // unexpected bundle is an error on connect, so it can never
        // need disconnecting.
        WithdrawalBundleInfo::Known(_) => (),
        WithdrawalBundleInfo::UnknownConfirmed { spend_utxos } => {
            for (outpoint, output) in spend_utxos {
                let outpoint_key = OutPointKey::from(outpoint);
                state
                    .utxos
                    .put(rwtxn, &outpoint_key, output)
                    .map_err(DbError::from)?;
                if !state
                    .stxos
                    .delete(rwtxn, &outpoint_key)
                    .map_err(DbError::from)?
                {
                    return Err(Error::NoStxo {
                        outpoint: *outpoint,
                    });
                };
                let utxo_hash = hash(&PointedOutputRef {
                    outpoint: *outpoint,
                    output,
                });
                accumulator_diff.insert(utxo_hash.into());
            }
            bundle = WithdrawalBundleInfo::Unknown;
        }
        WithdrawalBundleInfo::Unknown => (),
    }
    state
        .withdrawal_bundles
        .put(rwtxn, &m6id, &(bundle, prev_bundle_status))
        .map_err(DbError::from)?;
    Ok(())
}

fn disconnect_withdrawal_bundle_failed(
    state: &State,
    rwtxn: &mut RwTxn,
    block_height: u32,
    accumulator_diff: &mut AccumulatorDiff,
    m6id: M6id,
) -> Result<(), Error> {
    let (bundle, bundle_status) = state
        .withdrawal_bundles
        .try_get(rwtxn, &m6id)
        .map_err(DbError::from)?
        .ok_or_else(|| Error::UnknownWithdrawalBundle { m6id })?;
    let (prev_bundle_status, latest_bundle_status) = bundle_status.pop();
    if latest_bundle_status.value == WithdrawalBundleStatus::Submitted {
        // Already applied
        return Ok(());
    } else {
        assert_eq!(latest_bundle_status.value, WithdrawalBundleStatus::Failed);
    }
    assert_eq!(latest_bundle_status.height, block_height);
    let prev_bundle_status =
        prev_bundle_status.expect("Pop failed bundle status should be valid");
    assert!(matches!(
        prev_bundle_status.latest().value,
        WithdrawalBundleStatus::Submitted
            | WithdrawalBundleStatus::SubmittedUnexpected
    ));
    match &bundle {
        WithdrawalBundleInfo::Unknown
        | WithdrawalBundleInfo::UnknownConfirmed { .. } => (),
        WithdrawalBundleInfo::Known(bundle) => 'known: {
            if matches!(
                prev_bundle_status.latest().value,
                WithdrawalBundleStatus::SubmittedUnexpected
            ) {
                break 'known;
            }
            // Delete the outputs recreated from the failed bundle. The
            // original outputs remain spent by the bundle; their STXO
            // records are untouched.
            delete_recreated_outputs(
                state,
                rwtxn,
                accumulator_diff,
                block_height,
                m6id,
                bundle,
            )?;
            let (prev_latest_failed_m6id, latest_failed_m6id) = state
                .latest_failed_withdrawal_bundle
                .try_get(rwtxn, &())
                .map_err(DbError::from)?
                .expect("latest failed withdrawal bundle should exist")
                .pop();
            assert_eq!(latest_failed_m6id.value, m6id);
            assert_eq!(latest_failed_m6id.height, block_height);
            if let Some(prev_latest_failed_m6id) = prev_latest_failed_m6id {
                state
                    .latest_failed_withdrawal_bundle
                    .put(rwtxn, &(), &prev_latest_failed_m6id)
                    .map_err(DbError::from)?;
            } else {
                state
                    .latest_failed_withdrawal_bundle
                    .delete(rwtxn, &())
                    .map_err(DbError::from)?;
            }
        }
    }
    state
        .withdrawal_bundles
        .put(rwtxn, &m6id, &(bundle, prev_bundle_status))
        .map_err(DbError::from)?;
    Ok(())
}

fn disconnect_withdrawal_bundle_event(
    state: &State,
    rwtxn: &mut RwTxn,
    block_height: u32,
    accumulator_diff: &mut AccumulatorDiff,
    event: &WithdrawalBundleEvent,
) -> Result<(), Error> {
    match event.status {
        WithdrawalBundleEventStatus::Submitted => {
            disconnect_withdrawal_bundle_submitted(
                state,
                rwtxn,
                block_height,
                event.m6id,
            )
        }
        WithdrawalBundleEventStatus::Confirmed => {
            disconnect_withdrawal_bundle_confirmed(
                state,
                rwtxn,
                block_height,
                accumulator_diff,
                event.m6id,
            )
        }
        WithdrawalBundleEventStatus::Failed => {
            disconnect_withdrawal_bundle_failed(
                state,
                rwtxn,
                block_height,
                accumulator_diff,
                event.m6id,
            )
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn disconnect_event(
    state: &State,
    rwtxn: &mut RwTxn,
    block_height: u32,
    accumulator_diff: &mut AccumulatorDiff,
    latest_deposit_block_hash: &mut Option<bitcoin::BlockHash>,
    latest_withdrawal_bundle_event_block_hash: &mut Option<bitcoin::BlockHash>,
    event_block_hash: bitcoin::BlockHash,
    event: &BlockEvent,
) -> Result<(), Error> {
    match event {
        BlockEvent::Deposit(deposit) => {
            state
                .deposit_mainchain_height
                .delete(rwtxn, &deposit.outpoint.txid)?;
            let outpoint = OutPoint::Deposit(deposit.outpoint);
            let output = deposit.output.clone();
            if !state
                .utxos
                .delete(rwtxn, &OutPointKey::from(&outpoint))
                .map_err(DbError::from)?
            {
                return Err(error::NoUtxo { outpoint }.into());
            }
            let utxo_hash = hash(&PointedOutput { outpoint, output });
            accumulator_diff.remove(utxo_hash.into());
            // Blocks are iterated in reverse here, so the first event block
            // hash seen is the latest. Keep it to match what `connect` stored.
            if latest_deposit_block_hash.is_none() {
                *latest_deposit_block_hash = Some(event_block_hash);
            }
        }
        BlockEvent::WithdrawalBundle(withdrawal_bundle_event) => {
            let () = disconnect_withdrawal_bundle_event(
                state,
                rwtxn,
                block_height,
                accumulator_diff,
                withdrawal_bundle_event,
            )?;
            // Blocks are iterated in reverse here, so the first event block
            // hash seen is the latest. Keep it to match what `connect` stored.
            if latest_withdrawal_bundle_event_block_hash.is_none() {
                *latest_withdrawal_bundle_event_block_hash =
                    Some(event_block_hash);
            }
        }
    }
    Ok(())
}

pub fn disconnect(
    state: &State,
    rwtxn: &mut RwTxn,
    two_way_peg_data: &TwoWayPegData,
) -> Result<(), Error> {
    let block_height = state
        .try_get_height(rwtxn)?
        .expect("Height should not be None");
    let mut accumulator = state
        .utreexo_accumulator
        .try_get(rwtxn, &())
        .map_err(DbError::from)?
        .unwrap_or_default();
    let mut accumulator_diff = AccumulatorDiff::default();
    let mut latest_deposit_block_hash = None;
    let mut latest_withdrawal_bundle_event_block_hash = None;
    // Undo a stale-bundle drop performed at this height. Drops are
    // connected after the block's events, so they are undone before
    // disconnecting the events.
    let () = disconnect_dropped_bundle(
        state,
        rwtxn,
        &mut accumulator_diff,
        block_height,
    )?;
    for (event_block_hash, event_block_info) in
        two_way_peg_data.block_info.iter().rev()
    {
        for event in event_block_info.inner.events.iter().rev() {
            let () = disconnect_event(
                state,
                rwtxn,
                block_height,
                &mut accumulator_diff,
                &mut latest_deposit_block_hash,
                &mut latest_withdrawal_bundle_event_block_hash,
                *event_block_hash,
                event,
            )?;
        }
    }
    // Handle withdrawals
    if let Some(latest_withdrawal_bundle_event_block_hash) =
        latest_withdrawal_bundle_event_block_hash
    {
        let (
            last_withdrawal_bundle_event_block_seq_idx,
            (
                last_withdrawal_bundle_event_block_hash,
                last_withdrawal_bundle_event_block_height,
            ),
        ) = state
            .withdrawal_bundle_event_blocks
            .last(rwtxn)?
            .ok_or(Error::NoWithdrawalBundleEventBlock)?;
        assert_eq!(
            latest_withdrawal_bundle_event_block_hash,
            last_withdrawal_bundle_event_block_hash
        );
        assert_eq!(block_height, last_withdrawal_bundle_event_block_height);
        if !state
            .withdrawal_bundle_event_blocks
            .delete(rwtxn, &last_withdrawal_bundle_event_block_seq_idx)?
        {
            return Err(Error::NoWithdrawalBundleEventBlock);
        };
    }
    // Handle deposits.
    if let Some(latest_deposit_block_hash) = latest_deposit_block_hash {
        let (
            last_deposit_block_seq_idx,
            (last_deposit_block_hash, last_deposit_block_height),
        ) = state
            .deposit_blocks
            .last(rwtxn)?
            .ok_or(Error::NoDepositBlock)?;
        assert_eq!(latest_deposit_block_hash, last_deposit_block_hash);
        assert_eq!(block_height, last_deposit_block_height);
        if !state
            .deposit_blocks
            .delete(rwtxn, &last_deposit_block_seq_idx)?
        {
            return Err(Error::NoDepositBlock);
        };
    }
    let () = accumulator.apply_diff(accumulator_diff)?;
    state.utreexo_accumulator.put(rwtxn, &(), &accumulator)?;
    Ok(())
}

/// Undo a withdrawal bundle constructed at the current height,
/// unspending its selected outputs last-to-first.
/// MUST be applied BEFORE disconnecting the block's transactions.
pub fn disconnect_withdrawal_bundle_creation(
    state: &State,
    rwtxn: &mut RwTxn,
) -> Result<(), Error> {
    let block_height = state.try_get_height(rwtxn)?.ok_or(Error::NoTip)?;
    let Some(bundle_m6id) =
        state.pending_withdrawal_bundle.try_get(rwtxn, &())?
    else {
        return Ok(());
    };
    let (bundle, bundle_status) = state
        .withdrawal_bundles
        .try_get(rwtxn, &bundle_m6id)?
        .ok_or(error::PendingWithdrawalBundleUnknown(bundle_m6id))?;
    // Only undo a bundle created at this height: it is then still
    // Pending — a Submitted event at this height is disconnected later,
    // and would have cleared the pending marker anyway.
    if bundle_status.latest().value != WithdrawalBundleStatus::Pending
        || bundle_status.latest().height != block_height
    {
        return Ok(());
    }
    let bundle = match bundle {
        WithdrawalBundleInfo::Known(bundle) => bundle,
        WithdrawalBundleInfo::Unknown
        | WithdrawalBundleInfo::UnknownConfirmed { .. } => {
            return Err(
                error::PendingWithdrawalBundleUnknown(bundle_m6id).into()
            );
        }
    };
    let mut accumulator = state
        .utreexo_accumulator
        .try_get(rwtxn, &())?
        .unwrap_or_default();
    let mut accumulator_diff = AccumulatorDiff::default();
    for (outpoint, output) in bundle.spend_utxos().iter().rev() {
        let key = OutPointKey::from(outpoint);
        if !state.stxos.delete(rwtxn, &key)? {
            return Err(Error::NoStxo {
                outpoint: *outpoint,
            });
        }
        state.utxos.put(rwtxn, &key, output)?;
        let utxo_hash = hash(&PointedOutputRef {
            outpoint: *outpoint,
            output,
        });
        accumulator_diff.insert(utxo_hash.into());
    }
    state.pending_withdrawal_bundle.delete(rwtxn, &())?;
    // Each block connects at most one floor update, so this rolls back only
    // the construction's own update, if any.
    rollback_min_mainchain_fee_rate(state, rwtxn, block_height)?;
    if let (Some(bundle_status), _latest_bundle_status) = bundle_status.pop() {
        state.withdrawal_bundles.put(
            rwtxn,
            &bundle_m6id,
            &(WithdrawalBundleInfo::Known(bundle), bundle_status),
        )?;
    } else {
        state.withdrawal_bundles.delete(rwtxn, &bundle_m6id)?;
    }
    let () = accumulator.apply_diff(accumulator_diff)?;
    state.utreexo_accumulator.put(rwtxn, &(), &accumulator)?;
    Ok(())
}

#[cfg(test)]
mod test {
    use std::collections::{BTreeMap, HashMap};

    use bitcoin::{
        Network,
        hashes::Hash as _,
        secp256k1::{Secp256k1, SecretKey},
    };
    use hashlink::LinkedHashMap;

    use crate::{
        state::{
            MAX_WITHDRAWAL_BUNDLE_PAYOUT_OUTPUTS, State,
            WITHDRAWAL_BUNDLE_FAILURE_GAP, WithdrawalBundleInfo,
            WithdrawalBundleScheme,
            rollback::RollBack,
            test::{fresh_state, fresh_state_on_network, value_output},
            two_way_peg_data::{
                bundle_from_aggregated, collect_withdrawal_bundle, connect,
                connect_withdrawal_bundle_confirmed,
                connect_withdrawal_bundle_failed,
                connect_withdrawal_bundle_submitted, disconnect,
                disconnect_withdrawal_bundle_failed,
                maybe_create_withdrawal_bundle, min_mainchain_fee_rate,
                simulate_withdrawal_bundle_construction,
                unsuccessful_withdrawal_outpoint,
            },
        },
        types::{
            AccumulatorDiff, Address, AggregatedWithdrawal, Body, Coinbase,
            FilledTransaction, Header, InPoint, M6id, OutPoint, OutPointKey,
            Output, OutputContent, PointedOutputRef, Transaction, Txid,
            WithdrawalBundle, WithdrawalBundleEvent,
            WithdrawalBundleEventStatus, WithdrawalBundleStatus,
            authorization::BatchVerificationContext,
            hash,
            proto::mainchain::{
                BlockEvent, BlockInfo, Deposit, TwoWayPegData, two_way_peg_data,
            },
        },
    };

    // A pre-activation bundle failed after activation must refund under its
    // legacy M6ID; disconnecting the failure must undo those refunds.
    #[test]
    fn legacy_failure_refunds_under_legacy_m6id() -> anyhow::Result<()> {
        let (_temp_dir, env, state) = fresh_state_on_network(
            "legacy_failure_refunds_under_legacy_m6id",
            crate::types::Network::Mainnet,
        )?;
        let mut rwtxn = env.write_txn()?;
        let created_height = crate::types::Network::Mainnet
            .adaptive_bundle_construction_activation_height()
            - 1;
        state.height.put(&mut rwtxn, &(), &created_height)?;
        state.withdrawal_bundle_min_mainchain_fee_rate.put(
            &mut rwtxn,
            &0,
            &(0, bitcoin::FeeRate::BROADCAST_MIN),
        )?;
        for (idx, main_fee) in [(1_u8, 30), (2, 200)] {
            insert_withdrawal_utxo(&state, &mut rwtxn, idx, main_fee)?;
        }
        maybe_create_withdrawal_bundle(&state, &mut rwtxn)?;
        let legacy_m6id = pending_m6id(&state, &rwtxn)?;
        let (legacy_bundle, _) = state
            .try_get_pending_withdrawal_bundle(&rwtxn)?
            .expect("pending legacy bundle should exist");

        let submission_height = created_height + 1;
        submit_bundle(&state, &mut rwtxn, submission_height, legacy_m6id)?;
        let failed_height = submission_height + 1;
        connect_withdrawal_bundle_failed(
            &state,
            &mut rwtxn,
            failed_height,
            &mut AccumulatorDiff::default(),
            legacy_m6id,
        )?;
        for (vout, (outpoint, output)) in
            legacy_bundle.spend_utxos().iter().enumerate()
        {
            let refunded_outpoint = unsuccessful_withdrawal_outpoint(
                legacy_m6id,
                failed_height,
                vout,
            );
            let refunded = state
                .utxos
                .get(&rwtxn, &OutPointKey::from(&refunded_outpoint))?;
            anyhow::ensure!(&refunded == output);
            let stxo = state.stxos.get(&rwtxn, &OutPointKey::from(outpoint))?;
            anyhow::ensure!(
                stxo.inpoint == InPoint::Withdrawal { m6id: legacy_m6id }
            );
        }

        disconnect_withdrawal_bundle_failed(
            &state,
            &mut rwtxn,
            failed_height,
            &mut AccumulatorDiff::default(),
            legacy_m6id,
        )?;
        for (vout, outpoint) in legacy_bundle.spend_utxos().keys().enumerate() {
            let refunded_outpoint = unsuccessful_withdrawal_outpoint(
                legacy_m6id,
                failed_height,
                vout,
            );
            anyhow::ensure!(!utxo_exists(&state, &rwtxn, &refunded_outpoint)?);
            let stxo = state.stxos.get(&rwtxn, &OutPointKey::from(outpoint))?;
            anyhow::ensure!(
                stxo.inpoint == InPoint::Withdrawal { m6id: legacy_m6id }
            );
        }
        Ok(())
    }

    fn withdrawal_output_with_fee(
        main_address: bitcoin::Address<bitcoin::address::NetworkUnchecked>,
        main_fee: bitcoin::Amount,
    ) -> Output {
        Output {
            address: Address::ALL_ZEROS,
            content: OutputContent::Withdrawal {
                value: bitcoin::Amount::from_sat(1000),
                main_fee,
                main_address,
            },
        }
    }

    fn withdrawal_output(
        main_address: bitcoin::Address<bitcoin::address::NetworkUnchecked>,
    ) -> Output {
        withdrawal_output_with_fee(main_address, bitcoin::Amount::from_sat(300))
    }

    /// Put a UTXO and insert its leaf into the utreexo accumulator.
    fn insert_utxo(
        state: &State,
        rwtxn: &mut sneed::RwTxn,
        outpoint: OutPoint,
        output: &Output,
    ) -> anyhow::Result<()> {
        state
            .utxos
            .put(rwtxn, &OutPointKey::from(&outpoint), output)?;
        let utxo_hash = hash(&PointedOutputRef { outpoint, output });
        let mut accumulator_diff = AccumulatorDiff::default();
        accumulator_diff.insert(utxo_hash.into());
        let mut accumulator = state
            .utreexo_accumulator
            .try_get(rwtxn, &())?
            .unwrap_or_default();
        accumulator.apply_diff(accumulator_diff)?;
        state.utreexo_accumulator.put(rwtxn, &(), &accumulator)?;
        Ok(())
    }

    fn insert_withdrawal_utxo(
        state: &State,
        rwtxn: &mut sneed::RwTxn,
        idx: u8,
        main_fee: u64,
    ) -> anyhow::Result<()> {
        insert_utxo(
            state,
            rwtxn,
            regular_outpoint(idx, u32::from(idx)),
            &withdrawal_output_with_fee(
                regtest_p2wpkh_address(u32::from(idx)),
                bitcoin::Amount::from_sat(main_fee),
            ),
        )
    }

    fn regular_outpoint(idx: u8, vout: u32) -> OutPoint {
        OutPoint::Regular {
            txid: Txid::from([idx; 32]),
            vout,
        }
    }

    fn utxo_exists(
        state: &State,
        rotxn: &sneed::RoTxn,
        outpoint: &OutPoint,
    ) -> anyhow::Result<bool> {
        Ok(state
            .utxos
            .try_get(rotxn, &OutPointKey::from(outpoint))?
            .is_some())
    }

    fn try_pending_m6id(
        state: &State,
        rwtxn: &sneed::RoTxn,
    ) -> anyhow::Result<Option<M6id>> {
        Ok(state.pending_withdrawal_bundle.try_get(rwtxn, &())?)
    }

    fn pending_m6id(
        state: &State,
        rwtxn: &sneed::RoTxn,
    ) -> anyhow::Result<M6id> {
        Ok(try_pending_m6id(state, rwtxn)?
            .expect("pending bundle should exist"))
    }

    fn submit_bundle(
        state: &State,
        rwtxn: &mut sneed::RwTxn,
        block_height: u32,
        m6id: M6id,
    ) -> anyhow::Result<()> {
        Ok(connect_withdrawal_bundle_submitted(
            state,
            rwtxn,
            block_height,
            &bitcoin::BlockHash::from_byte_array([9; 32]),
            m6id,
        )?)
    }

    // constructing a withdrawal bundle spends the selected outputs
    #[test]
    fn maybe_create_withdrawal_bundle_spends_outputs() -> anyhow::Result<()> {
        let (_temp_dir, env, state) =
            fresh_state("maybe_create_withdrawal_bundle_spends_outputs")?;
        let mut rwtxn = env.write_txn()?;
        state
            .height
            .put(&mut rwtxn, &(), &WITHDRAWAL_BUNDLE_FAILURE_GAP)?;
        let withdrawal_outpoint = regular_outpoint(1, 0);
        let withdrawal_output = withdrawal_output(regtest_p2wpkh_address(1));
        let value_outpoint = regular_outpoint(2, 0);
        insert_utxo(
            &state,
            &mut rwtxn,
            withdrawal_outpoint,
            &withdrawal_output,
        )?;
        insert_utxo(
            &state,
            &mut rwtxn,
            value_outpoint,
            &value_output(Address::ALL_ZEROS, 50),
        )?;
        maybe_create_withdrawal_bundle(&state, &mut rwtxn)?;
        // the withdrawal output is spent by the bundle...
        anyhow::ensure!(!utxo_exists(&state, &rwtxn, &withdrawal_outpoint)?);
        let pending_m6id = pending_m6id(&state, &rwtxn)?;
        anyhow::ensure!(state.withdrawal_bundles.len(&rwtxn)? == 1);
        let stxo = state
            .stxos
            .get(&rwtxn, &OutPointKey::from(&withdrawal_outpoint))?;
        anyhow::ensure!(
            stxo.inpoint == InPoint::Withdrawal { m6id: pending_m6id }
        );
        // ...while the value output is untouched
        anyhow::ensure!(utxo_exists(&state, &rwtxn, &value_outpoint)?);
        // the pending bundle is well-formed and spends the output
        let (bundle, height) =
            state.try_get_pending_withdrawal_bundle(&rwtxn)?.unwrap();
        anyhow::ensure!(height == WITHDRAWAL_BUNDLE_FAILURE_GAP);
        anyhow::ensure!(bundle.spend_utxos().len() == 1);
        anyhow::ensure!(
            bundle.spend_utxos().contains_key(&withdrawal_outpoint)
        );
        rwtxn.commit()?;
        Ok(())
    }

    // failing a submitted bundle recreates its outputs at
    // `OutPoint::UnsuccessfulWithdrawal` outpoints; the recreations are
    // spendable
    #[test]
    fn failed_bundle_recreates_spendable_outputs() -> anyhow::Result<()> {
        let (_temp_dir, env, state) =
            fresh_state("failed_bundle_recreates_spendable_outputs")?;
        let mut rwtxn = env.write_txn()?;
        state
            .height
            .put(&mut rwtxn, &(), &WITHDRAWAL_BUNDLE_FAILURE_GAP)?;
        let withdrawal_outpoint = regular_outpoint(1, 0);
        let withdrawal_output = withdrawal_output(regtest_p2wpkh_address(1));
        insert_utxo(
            &state,
            &mut rwtxn,
            withdrawal_outpoint,
            &withdrawal_output,
        )?;
        maybe_create_withdrawal_bundle(&state, &mut rwtxn)?;
        let m6id = pending_m6id(&state, &rwtxn)?;
        // submit the bundle
        submit_bundle(&state, &mut rwtxn, WITHDRAWAL_BUNDLE_FAILURE_GAP, m6id)?;
        // fail the bundle at the next height
        let failed_height = WITHDRAWAL_BUNDLE_FAILURE_GAP + 1;
        let mut accumulator_diff = AccumulatorDiff::default();
        connect_withdrawal_bundle_failed(
            &state,
            &mut rwtxn,
            failed_height,
            &mut accumulator_diff,
            m6id,
        )?;
        // the recreated output exists, with identical content
        let recreated_outpoint =
            unsuccessful_withdrawal_outpoint(m6id, failed_height, 0);
        let recreated = state
            .utxos
            .get(&rwtxn, &OutPointKey::from(&recreated_outpoint))?;
        anyhow::ensure!(recreated == withdrawal_output);
        // the original STXO record is untouched
        let stxo = state
            .stxos
            .get(&rwtxn, &OutPointKey::from(&withdrawal_outpoint))?;
        anyhow::ensure!(stxo.inpoint == InPoint::Withdrawal { m6id });
        // the recreation is spendable
        let utxo_hash = hash(&PointedOutputRef {
            outpoint: recreated_outpoint,
            output: &recreated,
        });
        let tx = FilledTransaction {
            transaction: Transaction {
                inputs: vec![(recreated_outpoint, utxo_hash)].into(),
                outputs: vec![value_output(Address::ALL_ZEROS, 1300)].into(),
                ..Default::default()
            },
            spent_utxos: vec![recreated],
        };
        let fee = state.validate_filled_transaction(&rwtxn, 0, &tx)?;
        anyhow::ensure!(fee == bitcoin::Amount::ZERO);
        rwtxn.commit()?;
        Ok(())
    }

    // a stale pending bundle is dropped while connecting 2WPD; its
    // outputs are recreated and can be included in the fresh bundle
    // constructed after the block's txs
    #[test]
    fn stale_pending_bundle_dropped_and_rebundled() -> anyhow::Result<()> {
        let (_temp_dir, env, state) =
            fresh_state("stale_pending_bundle_dropped_and_rebundled")?;
        let mut rwtxn = env.write_txn()?;
        let created_height = WITHDRAWAL_BUNDLE_FAILURE_GAP;
        state.height.put(&mut rwtxn, &(), &created_height)?;
        let withdrawal_outpoint = regular_outpoint(1, 0);
        let withdrawal_output = withdrawal_output(regtest_p2wpkh_address(1));
        insert_utxo(
            &state,
            &mut rwtxn,
            withdrawal_outpoint,
            &withdrawal_output,
        )?;
        maybe_create_withdrawal_bundle(&state, &mut rwtxn)?;
        let stale_m6id = pending_m6id(&state, &rwtxn)?;
        // connect 2WPD for a block GAP heights later: the stale bundle
        // is dropped before the block's txs
        let drop_height = created_height + WITHDRAWAL_BUNDLE_FAILURE_GAP;
        state.height.put(&mut rwtxn, &(), &(drop_height - 1))?;
        connect(&state, &mut rwtxn, &TwoWayPegData::default())?;
        let (_bundle, status) = state
            .try_get_withdrawal_bundle(&rwtxn, &stale_m6id)?
            .unwrap();
        anyhow::ensure!(status == WithdrawalBundleStatus::Dropped);
        anyhow::ensure!(try_pending_m6id(&state, &rwtxn)?.is_none());
        let recreated_outpoint =
            unsuccessful_withdrawal_outpoint(stale_m6id, drop_height, 0);
        anyhow::ensure!(utxo_exists(&state, &rwtxn, &recreated_outpoint)?);
        // no fresh bundle yet: the failure gap applies after a drop
        state.height.put(&mut rwtxn, &(), &drop_height)?;
        maybe_create_withdrawal_bundle(&state, &mut rwtxn)?;
        anyhow::ensure!(try_pending_m6id(&state, &rwtxn)?.is_none());
        // the recreated output is spendable in the meantime (covered
        // by failed_bundle_recreates_spendable_outputs), and is
        // included in the fresh bundle once the gap passes
        state.height.put(
            &mut rwtxn,
            &(),
            &(drop_height + WITHDRAWAL_BUNDLE_FAILURE_GAP),
        )?;
        maybe_create_withdrawal_bundle(&state, &mut rwtxn)?;
        let (bundle, _) =
            state.try_get_pending_withdrawal_bundle(&rwtxn)?.unwrap();
        anyhow::ensure!(bundle.compute_m6id() != stale_m6id);
        anyhow::ensure!(bundle.spend_utxos().contains_key(&recreated_outpoint));
        rwtxn.commit()?;
        Ok(())
    }

    // confirming a previously failed bundle is an insolvency error: its
    // outputs were already spent at creation and recreated at failure
    #[test]
    fn unexpected_bundle_confirmed_is_insolvent_error() -> anyhow::Result<()> {
        let (_temp_dir, env, state) =
            fresh_state("unexpected_bundle_confirmed_is_insolvent_error")?;
        let mut rwtxn = env.write_txn()?;
        state
            .height
            .put(&mut rwtxn, &(), &WITHDRAWAL_BUNDLE_FAILURE_GAP)?;
        // seed a known bundle with status SubmittedUnexpected
        let outpoint = regular_outpoint(1, 0);
        let output = value_output(Address::ALL_ZEROS, 1000);
        let mut spend_utxos = BTreeMap::new();
        spend_utxos.insert(outpoint, output);
        let bundle = WithdrawalBundle::new(
            1,
            bitcoin::Amount::ZERO,
            spend_utxos,
            Vec::new(),
        )?;
        let m6id = bundle.compute_m6id();
        let mut bundle_status =
            RollBack::new(WithdrawalBundleStatus::Failed, 1);
        bundle_status
            .push(WithdrawalBundleStatus::SubmittedUnexpected, 2)
            .unwrap();
        state.withdrawal_bundles.put(
            &mut rwtxn,
            &m6id,
            &(WithdrawalBundleInfo::Known(bundle), bundle_status),
        )?;
        let mut accumulator_diff = AccumulatorDiff::default();
        let res = connect_withdrawal_bundle_confirmed(
            &state,
            &mut rwtxn,
            3,
            &mut accumulator_diff,
            &bitcoin::BlockHash::from_byte_array([9; 32]),
            m6id,
        );
        anyhow::ensure!(matches!(
            res,
            Err(
                crate::state::Error::UnexpectedWithdrawalBundleConfirmed { .. }
            )
        ));
        Ok(())
    }

    // disconnecting a withdrawal bundle event must remove its
    // withdrawal_bundle_event_blocks record, not a deposit_blocks record that
    // happens to share the same sequence index
    #[test]
    fn disconnect_withdrawal_event_block_uses_correct_db() -> anyhow::Result<()>
    {
        let (_temp_dir, env, state) =
            fresh_state("disconnect_withdrawal_event_block_uses_correct_db")?;

        let block_height = 5u32;
        let m6id = M6id(bitcoin::Txid::from_byte_array([7; 32]));
        let event_block_hash = bitcoin::BlockHash::from_byte_array([9; 32]);
        let deposit_block_hash = bitcoin::BlockHash::from_byte_array([3; 32]);

        let mut rwtxn = env.write_txn()?;
        state.height.put(&mut rwtxn, &(), &block_height)?;
        state.withdrawal_bundles.put(
            &mut rwtxn,
            &m6id,
            &(
                WithdrawalBundleInfo::Unknown,
                RollBack::new(WithdrawalBundleStatus::Submitted, block_height),
            ),
        )?;
        state.withdrawal_bundle_event_blocks.put(
            &mut rwtxn,
            &0,
            &(event_block_hash, block_height),
        )?;
        // a deposit record at the same sequence index that must survive
        state.deposit_blocks.put(
            &mut rwtxn,
            &0,
            &(deposit_block_hash, block_height),
        )?;
        rwtxn.commit()?;

        let two_way_peg_data = {
            let mut block_info = LinkedHashMap::new();
            block_info.insert(
                event_block_hash,
                two_way_peg_data::BlockInfo {
                    height: 999,
                    inner: BlockInfo {
                        bmm_commitment: None,
                        events: vec![BlockEvent::WithdrawalBundle(
                            WithdrawalBundleEvent {
                                m6id,
                                status: WithdrawalBundleEventStatus::Submitted,
                            },
                        )],
                    },
                },
            );
            TwoWayPegData { block_info }
        };

        let mut rwtxn = env.write_txn()?;
        disconnect(&state, &mut rwtxn, &two_way_peg_data)?;
        anyhow::ensure!(
            state
                .withdrawal_bundle_event_blocks
                .try_get(&rwtxn, &0)?
                .is_none()
        );
        anyhow::ensure!(state.deposit_blocks.try_get(&rwtxn, &0)?.is_some());
        rwtxn.commit()?;
        Ok(())
    }

    fn seeded_public_key(idx: u32) -> bitcoin::CompressedPublicKey {
        let secp = Secp256k1::new();
        let mut key_bytes = [0_u8; 32];
        key_bytes[28..].copy_from_slice(&idx.to_be_bytes());
        let secret_key = SecretKey::from_slice(&key_bytes)
            .expect("small non-zero integers are valid secret keys");
        let public_key =
            bitcoin::secp256k1::PublicKey::from_secret_key(&secp, &secret_key);
        bitcoin::CompressedPublicKey(public_key)
    }

    fn regtest_p2wpkh_address(
        idx: u32,
    ) -> bitcoin::Address<bitcoin::address::NetworkUnchecked> {
        let public_key = seeded_public_key(idx);
        bitcoin::Address::p2wpkh(&public_key, Network::Regtest).into_unchecked()
    }

    fn regtest_p2tr_address(
        idx: u32,
    ) -> bitcoin::Address<bitcoin::address::NetworkUnchecked> {
        let (x_only_public_key, _parity) =
            seeded_public_key(idx).0.x_only_public_key();
        bitcoin::Address::p2tr(
            &Secp256k1::new(),
            x_only_public_key,
            None,
            Network::Regtest,
        )
        .into_unchecked()
    }

    fn aggregated_withdrawal(
        idx: u32,
        main_address: bitcoin::Address<bitcoin::address::NetworkUnchecked>,
        main_fee: u64,
    ) -> AggregatedWithdrawal {
        let outpoint = regular_outpoint(idx as u8, idx);
        let output = withdrawal_output_with_fee(
            main_address.clone(),
            bitcoin::Amount::from_sat(main_fee),
        );
        AggregatedWithdrawal {
            spend_utxos: HashMap::from([(outpoint, output)]),
            main_address,
            value: bitcoin::Amount::from_sat(1_000),
            main_fee: bitcoin::Amount::from_sat(main_fee),
        }
    }

    fn header_with_bundle_scheme(
        state: &State,
        rotxn: &sneed::RoTxn,
        block_height: u32,
        body: &Body,
        scheme: WithdrawalBundleScheme,
    ) -> Result<Header, crate::state::Error> {
        let no_txs: &[FilledTransaction] = &[];
        let merkle_root = Body::compute_merkle_root(
            &body.coinbase,
            no_txs,
            &body.authorizations,
        )?;
        let mut header = Header {
            merkle_root,
            prev_side_hash: None,
            prev_main_hash: bitcoin::BlockHash::from_byte_array([99; 32]),
            roots: Vec::new(),
        };
        let construction_diff = simulate_withdrawal_bundle_construction(
            state,
            rotxn,
            block_height,
            scheme,
            no_txs,
            &body.coinbase,
            header.compute_coinbase_txid(),
        )?;
        let mut accumulator = state.get_accumulator(rotxn)?;
        accumulator.apply_diff(construction_diff)?;
        header.roots = accumulator.get_roots();
        Ok(header)
    }

    fn with_state_with_withdrawals<R>(
        test_name: &str,
        count: u32,
        main_address: fn(
            u32,
        ) -> bitcoin::Address<
            bitcoin::address::NetworkUnchecked,
        >,
        f: impl FnOnce(&State, &mut sneed::RwTxn<'_>) -> R,
    ) -> anyhow::Result<R> {
        let (_temp_dir, env, state) = fresh_state(test_name)?;
        let res = {
            let mut rwtxn = env.write_txn()?;
            state.height.put(
                &mut rwtxn,
                &(),
                &crate::state::WITHDRAWAL_BUNDLE_FAILURE_GAP,
            )?;

            for idx in 1..=count {
                let mut txid_bytes = [0_u8; 32];
                txid_bytes[28..].copy_from_slice(&idx.to_be_bytes());
                let outpoint = OutPoint::Regular {
                    txid: txid_bytes.into(),
                    vout: 0,
                };
                let output = Output {
                    address: {
                        let mut addr = [0u8; 20];
                        let idx = idx.to_be_bytes();
                        addr[..idx.len()].copy_from_slice(&idx);
                        Address::from(addr)
                    },
                    content: OutputContent::Withdrawal {
                        value: bitcoin::Amount::from_sat(1_000),
                        main_fee: bitcoin::Amount::ZERO,
                        main_address: main_address(idx),
                    },
                };
                state.utxos.put(
                    &mut rwtxn,
                    &OutPointKey::from(&outpoint),
                    &output,
                )?;
            }
            f(&state, &mut rwtxn)
        };
        Ok(res)
    }

    #[test]
    fn collect_withdrawal_bundle_p2wpkh_off_by_one_does_not_exceed_weight()
    -> anyhow::Result<()> {
        const CLAIMED_MAX_BUNDLE_OUTPUTS: u32 = 3_222;

        let bundle = with_state_with_withdrawals(
            "collect_withdrawal_bundle_p2wpkh_off_by_one",
            CLAIMED_MAX_BUNDLE_OUTPUTS + 1,
            regtest_p2wpkh_address,
            |state, rwtxn| {
                collect_withdrawal_bundle(
                    state,
                    rwtxn,
                    42,
                    WithdrawalBundleScheme::Legacy,
                )
            },
        )?;
        let bundle = match bundle {
            Ok(Some((bundle, _applied_fee_rate))) => bundle,
            Ok(None) => anyhow::bail!("expected a withdrawal bundle"),
            Err(err) => anyhow::bail!("unexpected collection error: {err:?}"),
        };
        let output_count = bundle.tx().output.len();
        let weight = bundle.tx().weight().to_wu();

        anyhow::ensure!(
            output_count == (CLAIMED_MAX_BUNDLE_OUTPUTS as usize + 2),
            "expected {} tx outputs including metadata, got {output_count}",
            CLAIMED_MAX_BUNDLE_OUTPUTS as usize + 2,
        );
        anyhow::ensure!(
            weight <= bitcoin::policy::MAX_STANDARD_TX_WEIGHT as u64,
            "unexpected overweight P2WPKH bundle: {weight} wu"
        );
        Ok(())
    }

    #[test]
    fn adaptive_bundle_enforces_payout_output_limit() -> anyhow::Result<()> {
        let (bundle, _applied_fee_rate) = with_state_with_withdrawals(
            "adaptive_bundle_enforces_payout_output_limit",
            MAX_WITHDRAWAL_BUNDLE_PAYOUT_OUTPUTS + 1,
            regtest_p2wpkh_address,
            |state, rwtxn| {
                collect_withdrawal_bundle(
                    state,
                    rwtxn,
                    42,
                    WithdrawalBundleScheme::Adaptive,
                )
            },
        )?
        .expect("adaptive bundle should be constructed")
        .expect("payout candidates should produce a bundle");
        anyhow::ensure!(
            bundle.tx().output.len()
                == MAX_WITHDRAWAL_BUNDLE_PAYOUT_OUTPUTS as usize + 2
        );
        anyhow::ensure!(
            bundle.spend_utxos().len()
                == MAX_WITHDRAWAL_BUNDLE_PAYOUT_OUTPUTS as usize
        );
        Ok(())
    }

    #[test]
    fn adaptive_bundle_filters_by_payout_output_weight() -> anyhow::Result<()> {
        let candidates = vec![
            // P2WPKH payout weight is 124 WU: 30 sats is below 1 sat/vB.
            aggregated_withdrawal(1, regtest_p2wpkh_address(1), 30),
            // 31 sats exactly covers 124 WU at 1 sat/vB.
            aggregated_withdrawal(2, regtest_p2wpkh_address(2), 31),
            // P2TR payout weight is 172 WU: 42 sats is below 1 sat/vB.
            aggregated_withdrawal(3, regtest_p2tr_address(3), 42),
            // 43 sats exactly covers 172 WU at 1 sat/vB.
            aggregated_withdrawal(4, regtest_p2tr_address(4), 43),
        ];
        let aggregated = candidates
            .into_iter()
            .map(|aggregated| (aggregated.main_address.clone(), aggregated))
            .collect::<HashMap<_, _>>();
        // Excessive floors are capped; MAX also exercises overflow avoidance.
        for min_mainchain_fee_rate in [
            bitcoin::FeeRate::BROADCAST_MIN,
            bitcoin::FeeRate::from_sat_per_vb_u32(2),
            bitcoin::FeeRate::MAX,
        ] {
            let (adaptive, applied_fee_rate) = bundle_from_aggregated(
                aggregated.clone(),
                42,
                WithdrawalBundleScheme::Adaptive,
                min_mainchain_fee_rate,
            )?
            .expect("two payout outputs should pass the fee-rate filter");
            anyhow::ensure!(
                applied_fee_rate == bitcoin::FeeRate::BROADCAST_MIN
            );
            let selected =
                adaptive.spend_utxos().keys().copied().collect::<Vec<_>>();
            anyhow::ensure!(selected.len() == 2);
            anyhow::ensure!(selected.contains(&regular_outpoint(2, 2)));
            anyhow::ensure!(selected.contains(&regular_outpoint(4, 4)));
        }

        // Legacy construction ignores the adaptive fee-rate floor.
        let (legacy, _) = bundle_from_aggregated(
            aggregated,
            42,
            WithdrawalBundleScheme::Legacy,
            bitcoin::FeeRate::MAX,
        )?
        .expect("legacy construction should retain every payout");
        anyhow::ensure!(legacy.spend_utxos().len() == 4);
        Ok(())
    }

    #[test]
    fn adaptive_fee_rate_doubles_resets_and_rolls_back() -> anyhow::Result<()> {
        let (_temp_dir, env, state) =
            fresh_state("adaptive_fee_rate_doubles_resets_and_rolls_back")?;
        let mut rwtxn = env.write_txn()?;
        let created_height = WITHDRAWAL_BUNDLE_FAILURE_GAP;
        state.height.put(&mut rwtxn, &(), &created_height)?;
        for (idx, main_fee) in [(1_u8, 30), (2, 200)] {
            insert_withdrawal_utxo(&state, &mut rwtxn, idx, main_fee)?;
        }
        maybe_create_withdrawal_bundle(&state, &mut rwtxn)?;
        let stale_m6id = pending_m6id(&state, &rwtxn)?;
        anyhow::ensure!(
            min_mainchain_fee_rate(&state, &rwtxn)? == bitcoin::FeeRate::ZERO
        );

        // Timing out raises the floor from zero to 1 sat/kwu.
        let drop_height = created_height + WITHDRAWAL_BUNDLE_FAILURE_GAP;
        state.height.put(&mut rwtxn, &(), &(drop_height - 1))?;
        connect(&state, &mut rwtxn, &TwoWayPegData::default())?;
        anyhow::ensure!(
            min_mainchain_fee_rate(&state, &rwtxn)?
                == bitcoin::FeeRate::from_sat_per_kwu(1)
        );
        anyhow::ensure!(try_pending_m6id(&state, &rwtxn)?.is_none());

        // Rebundle after the gap, then time out again: the floor doubles.
        let rebundle_height = drop_height + WITHDRAWAL_BUNDLE_FAILURE_GAP;
        state.height.put(&mut rwtxn, &(), &rebundle_height)?;
        maybe_create_withdrawal_bundle(&state, &mut rwtxn)?;
        let rebundled_m6id = pending_m6id(&state, &rwtxn)?;
        anyhow::ensure!(rebundled_m6id != stale_m6id);
        let second_drop_height =
            rebundle_height + WITHDRAWAL_BUNDLE_FAILURE_GAP;
        state
            .height
            .put(&mut rwtxn, &(), &(second_drop_height - 1))?;
        let second_drop_data = TwoWayPegData::default();
        connect(&state, &mut rwtxn, &second_drop_data)?;
        anyhow::ensure!(
            min_mainchain_fee_rate(&state, &rwtxn)?
                == bitcoin::FeeRate::from_sat_per_kwu(2)
        );

        // Rebundle and submit: the floor resets to zero.
        let final_bundle_height =
            second_drop_height + WITHDRAWAL_BUNDLE_FAILURE_GAP;
        state.height.put(&mut rwtxn, &(), &final_bundle_height)?;
        maybe_create_withdrawal_bundle(&state, &mut rwtxn)?;
        let final_m6id = pending_m6id(&state, &rwtxn)?;
        anyhow::ensure!(final_m6id != rebundled_m6id);
        let submission_height = final_bundle_height + 1;
        let submission_hash = bitcoin::BlockHash::from_byte_array([3; 32]);
        let submission_data = {
            let mut block_info = LinkedHashMap::new();
            block_info.insert(
                submission_hash,
                two_way_peg_data::BlockInfo {
                    height: 3,
                    inner: BlockInfo {
                        bmm_commitment: None,
                        events: vec![BlockEvent::WithdrawalBundle(
                            WithdrawalBundleEvent {
                                m6id: final_m6id,
                                status: WithdrawalBundleEventStatus::Submitted,
                            },
                        )],
                    },
                },
            );
            TwoWayPegData { block_info }
        };
        connect(&state, &mut rwtxn, &submission_data)?;
        anyhow::ensure!(
            min_mainchain_fee_rate(&state, &rwtxn)? == bitcoin::FeeRate::ZERO
        );
        anyhow::ensure!(try_pending_m6id(&state, &rwtxn)?.is_none());

        // Disconnecting the submission restores both the pending bundle and
        // the pre-submission fee-rate floor.
        state.height.put(&mut rwtxn, &(), &submission_height)?;
        disconnect(&state, &mut rwtxn, &submission_data)?;
        anyhow::ensure!(
            min_mainchain_fee_rate(&state, &rwtxn)?
                == bitcoin::FeeRate::from_sat_per_kwu(2)
        );
        anyhow::ensure!(try_pending_m6id(&state, &rwtxn)? == Some(final_m6id));

        // Unwinding the final bundle's creation and then the second timeout
        // restores the earlier floor and the rebundled pending bundle.
        state.height.put(&mut rwtxn, &(), &final_bundle_height)?;
        state.disconnect_withdrawal_bundle_creation(&mut rwtxn)?;
        state.height.put(&mut rwtxn, &(), &second_drop_height)?;
        disconnect(&state, &mut rwtxn, &second_drop_data)?;
        anyhow::ensure!(
            min_mainchain_fee_rate(&state, &rwtxn)?
                == bitcoin::FeeRate::from_sat_per_kwu(1)
        );
        anyhow::ensure!(
            try_pending_m6id(&state, &rwtxn)? == Some(rebundled_m6id)
        );
        Ok(())
    }

    #[test]
    fn adaptive_construction_activates_at_mainnet_height() -> anyhow::Result<()>
    {
        let (_temp_dir, env, state) = fresh_state_on_network(
            "adaptive_construction_activates_at_mainnet_height",
            crate::types::Network::Mainnet,
        )?;
        let activation_height = crate::types::Network::Mainnet
            .adaptive_bundle_construction_activation_height();
        let mut rwtxn = env.write_txn()?;
        state
            .height
            .put(&mut rwtxn, &(), &(activation_height - 2))?;
        state.withdrawal_bundle_min_mainchain_fee_rate.put(
            &mut rwtxn,
            &0,
            &(0, bitcoin::FeeRate::BROADCAST_MIN),
        )?;
        for (idx, main_fee) in [(1_u8, 30), (2, 200)] {
            insert_withdrawal_utxo(&state, &mut rwtxn, idx, main_fee)?;
        }
        let body = Body::new(vec![], Coinbase::default());
        let batch_verification_ctxt =
            BatchVerificationContext::new(&mut rand::rng());
        let prevalidate_with_scheme =
            |rotxn: &sneed::RoTxn,
             block_height: u32,
             scheme: WithdrawalBundleScheme|
             -> Result<(), crate::state::Error> {
                let header = header_with_bundle_scheme(
                    &state,
                    rotxn,
                    block_height,
                    &body,
                    scheme,
                )?;
                state.prevalidate_block(
                    rotxn,
                    &batch_verification_ctxt,
                    &header,
                    0,
                    &body,
                )?;
                Ok(())
            };
        prevalidate_with_scheme(
            &rwtxn,
            activation_height - 1,
            WithdrawalBundleScheme::Legacy,
        )?;
        anyhow::ensure!(matches!(
            prevalidate_with_scheme(
                &rwtxn,
                activation_height - 1,
                WithdrawalBundleScheme::Adaptive,
            ),
            Err(crate::state::Error::UtreexoRootsMismatch)
        ));

        state
            .height
            .put(&mut rwtxn, &(), &(activation_height - 1))?;
        maybe_create_withdrawal_bundle(&state, &mut rwtxn)?;
        let (legacy, legacy_height) = state
            .try_get_pending_withdrawal_bundle(&rwtxn)?
            .expect("legacy bundle should be constructed below activation");
        let legacy_m6id = legacy.compute_m6id();
        anyhow::ensure!(legacy_height == activation_height - 1);
        anyhow::ensure!(legacy.spend_utxos().len() == 2);
        anyhow::ensure!(state.withdrawal_bundles.len(&rwtxn)? == 1);
        submit_bundle(&state, &mut rwtxn, activation_height - 1, legacy_m6id)?;

        for (idx, main_fee) in [(3_u8, 30), (4, 200)] {
            insert_withdrawal_utxo(&state, &mut rwtxn, idx, main_fee)?;
        }
        anyhow::ensure!(matches!(
            prevalidate_with_scheme(
                &rwtxn,
                activation_height,
                WithdrawalBundleScheme::Legacy,
            ),
            Err(crate::state::Error::UtreexoRootsMismatch)
        ));
        prevalidate_with_scheme(
            &rwtxn,
            activation_height,
            WithdrawalBundleScheme::Adaptive,
        )?;

        state.height.put(&mut rwtxn, &(), &activation_height)?;
        maybe_create_withdrawal_bundle(&state, &mut rwtxn)?;
        let (adaptive, adaptive_height) = state
            .try_get_pending_withdrawal_bundle(&rwtxn)?
            .expect("adaptive bundle should be constructed at activation");
        let adaptive_m6id = adaptive.compute_m6id();
        anyhow::ensure!(adaptive_height == activation_height);
        anyhow::ensure!(adaptive_m6id != legacy_m6id);
        anyhow::ensure!(adaptive.spend_utxos().len() == 1);
        anyhow::ensure!(
            adaptive.spend_utxos().contains_key(&regular_outpoint(4, 4))
        );
        anyhow::ensure!(utxo_exists(&state, &rwtxn, &regular_outpoint(3, 3))?);
        let adaptive_stxo = state
            .stxos
            .get(&rwtxn, &OutPointKey::from(&regular_outpoint(4, 4)))?;
        anyhow::ensure!(
            adaptive_stxo.inpoint
                == InPoint::Withdrawal {
                    m6id: adaptive_m6id
                }
        );
        anyhow::ensure!(state.withdrawal_bundles.len(&rwtxn)? == 2);
        Ok(())
    }

    #[test]
    fn mainnet_timeout_restores_legacy_bundle() -> anyhow::Result<()> {
        let (_temp_dir, env, state) = fresh_state_on_network(
            "mainnet_timeout_restores_legacy_bundle",
            crate::types::Network::Mainnet,
        )?;
        let mut rwtxn = env.write_txn()?;
        let created_height = WITHDRAWAL_BUNDLE_FAILURE_GAP;
        state.height.put(&mut rwtxn, &(), &created_height)?;
        state.withdrawal_bundle_min_mainchain_fee_rate.put(
            &mut rwtxn,
            &0,
            &(0, bitcoin::FeeRate::BROADCAST_MIN),
        )?;
        for (idx, main_fee) in [(1_u8, 30), (2, 200)] {
            insert_withdrawal_utxo(&state, &mut rwtxn, idx, main_fee)?;
        }
        maybe_create_withdrawal_bundle(&state, &mut rwtxn)?;
        let legacy_m6id = pending_m6id(&state, &rwtxn)?;

        // Creation disconnect restores every output spent by the legacy bundle.
        state.disconnect_withdrawal_bundle_creation(&mut rwtxn)?;
        for idx in [1_u8, 2] {
            anyhow::ensure!(utxo_exists(
                &state,
                &rwtxn,
                &regular_outpoint(idx, u32::from(idx))
            )?);
        }
        maybe_create_withdrawal_bundle(&state, &mut rwtxn)?;

        let drop_height = created_height + WITHDRAWAL_BUNDLE_FAILURE_GAP;
        state.height.put(&mut rwtxn, &(), &(drop_height - 1))?;
        let drop_data = TwoWayPegData::default();
        connect(&state, &mut rwtxn, &drop_data)?;
        let (_, status) = state
            .try_get_withdrawal_bundle(&rwtxn, &legacy_m6id)?
            .unwrap();
        anyhow::ensure!(status == WithdrawalBundleStatus::Dropped);
        for vout in [0_usize, 1] {
            anyhow::ensure!(utxo_exists(
                &state,
                &rwtxn,
                &unsuccessful_withdrawal_outpoint(
                    legacy_m6id,
                    drop_height,
                    vout
                )
            )?);
        }
        // Legacy bundles do not affect the adaptive fee-rate floor.
        anyhow::ensure!(
            min_mainchain_fee_rate(&state, &rwtxn)?
                == bitcoin::FeeRate::BROADCAST_MIN
        );

        state.height.put(&mut rwtxn, &(), &drop_height)?;
        disconnect(&state, &mut rwtxn, &drop_data)?;
        anyhow::ensure!(try_pending_m6id(&state, &rwtxn)? == Some(legacy_m6id));
        for vout in [0_usize, 1] {
            anyhow::ensure!(!utxo_exists(
                &state,
                &rwtxn,
                &unsuccessful_withdrawal_outpoint(
                    legacy_m6id,
                    drop_height,
                    vout
                )
            )?);
        }
        anyhow::ensure!(
            min_mainchain_fee_rate(&state, &rwtxn)?
                == bitcoin::FeeRate::BROADCAST_MIN
        );
        Ok(())
    }

    // connecting a deposit then disconnecting it on a reorg must round-trip
    #[test]
    fn deposit_reorg_round_trips() -> anyhow::Result<()> {
        use crate::types::{
            Body, FilledTransaction, Header,
            authorization::BatchVerificationContext, proto::mainchain::Deposit,
        };

        let (_temp_dir, env, state) = fresh_state("deposit_reorg_round_trips")?;
        let batch_verification_ctxt =
            BatchVerificationContext::new(&mut rand::rng());
        let empty_body = Body {
            coinbase: Coinbase::default(),
            transactions: Vec::new(),
            authorizations: Vec::new(),
        };
        let no_txs: &[FilledTransaction] = &[];
        let merkle_root = Body::compute_merkle_root(
            &empty_body.coinbase,
            no_txs,
            &empty_body.authorizations,
        )?;
        let main0 = bitcoin::BlockHash::from_byte_array([10; 32]);
        let main_height0 = 99;
        let main1 = bitcoin::BlockHash::from_byte_array([11; 32]);
        let main_height1 = 9999;

        let genesis = Header {
            merkle_root,
            prev_side_hash: None,
            prev_main_hash: main0,
            roots: Vec::new(),
        };
        {
            let mut rwtxn = env.write_txn()?;
            // production order: 2WPD events before the block body
            state.connect_two_way_peg_data(
                &mut rwtxn,
                &TwoWayPegData::default(),
            )?;
            state.apply_block(
                &mut rwtxn,
                &batch_verification_ctxt,
                &genesis,
                main_height0,
                &empty_body,
            )?;
            rwtxn.commit()?;
        }

        let deposit_outpoint = bitcoin::OutPoint {
            txid: bitcoin::Txid::from_byte_array([2; 32]),
            vout: 0,
        };
        let deposit_key =
            OutPointKey::from(&OutPoint::Deposit(deposit_outpoint));
        let deposit_twpd = {
            let mut block_info = LinkedHashMap::new();
            block_info.insert(
                main1,
                two_way_peg_data::BlockInfo {
                    height: main_height1,
                    inner: BlockInfo {
                        bmm_commitment: None,
                        events: vec![BlockEvent::Deposit(Deposit {
                            tx_index: 0,
                            outpoint: deposit_outpoint,
                            output: value_output(Address::ALL_ZEROS, 1000),
                        })],
                    },
                },
            );
            TwoWayPegData { block_info }
        };
        let block1 = {
            let mut rwtxn = env.write_txn()?;
            // production order: 2WPD events before the block body
            state.connect_two_way_peg_data(&mut rwtxn, &deposit_twpd)?;
            // the header roots commit to the post-events accumulator
            let block1 = Header {
                merkle_root,
                prev_side_hash: Some(genesis.hash()),
                prev_main_hash: main1,
                roots: state.get_accumulator(&rwtxn)?.get_roots(),
            };
            state.apply_block(
                &mut rwtxn,
                &batch_verification_ctxt,
                &block1,
                main_height1,
                &empty_body,
            )?;
            anyhow::ensure!(
                state.utxos.try_get(&rwtxn, &deposit_key)?.is_some()
            );
            anyhow::ensure!(state.deposit_blocks.last(&rwtxn)?.is_some());
            rwtxn.commit()?;
            block1
        };

        {
            let mut rwtxn = env.write_txn()?;
            // production order: undo bundle construction, then the block
            // body, then 2WPD events, then the header
            state.disconnect_withdrawal_bundle_creation(&mut rwtxn)?;
            state.disconnect_tip_body(&mut rwtxn, &block1, &empty_body)?;
            state.disconnect_two_way_peg_data(&mut rwtxn, &deposit_twpd)?;
            state.disconnect_tip_header(&mut rwtxn, &block1)?;
            anyhow::ensure!(
                state.utxos.try_get(&rwtxn, &deposit_key)?.is_none()
            );
            anyhow::ensure!(state.deposit_blocks.last(&rwtxn)?.is_none());
            rwtxn.commit()?;
        }

        Ok(())
    }

    // A single two-way-peg batch can span multiple mainchain blocks. Connecting
    // deposits from two distinct blocks then disconnecting the batch must
    // restore the prior state. Before the fix, disconnect recomputed the latest
    // deposit block hash by reverse iteration (yielding the oldest block) and
    // panicked on the consistency assert against the newest hash connect stored.
    #[test]
    fn disconnect_two_deposit_blocks_restores_state() -> anyhow::Result<()> {
        fn deposit_block(salt: u8) -> (bitcoin::BlockHash, BlockInfo) {
            let dep = Deposit {
                tx_index: 0,
                outpoint: bitcoin::OutPoint {
                    txid: bitcoin::Txid::from_byte_array([salt; 32]),
                    vout: 0,
                },
                output: Output {
                    address: Address([salt; 20]),
                    content: OutputContent::Value(bitcoin::Amount::from_sat(
                        1000,
                    )),
                },
            };
            (
                bitcoin::BlockHash::from_byte_array([salt; 32]),
                BlockInfo {
                    bmm_commitment: None,
                    events: vec![BlockEvent::Deposit(dep)],
                },
            )
        }
        let (_temp_dir, env, state) =
            fresh_state("disconnect_two_deposit_blocks_restores_state")?;
        let mut rwtxn = env.write_txn()?;
        // events connect before the block body: connecting the events
        // of block 10 requires height 9
        state.height.put(&mut rwtxn, &(), &9)?;

        let mut block_info = LinkedHashMap::new();
        let (h1, b1) = deposit_block(1);
        let (h2, b2) = deposit_block(2);
        block_info.insert(
            h1,
            two_way_peg_data::BlockInfo {
                height: 1,
                inner: b1,
            },
        );
        block_info.insert(
            h2,
            two_way_peg_data::BlockInfo {
                height: 2,
                inner: b2,
            },
        );
        let tdp = TwoWayPegData { block_info };

        let () = connect(&state, &mut rwtxn, &tdp)?;
        anyhow::ensure!(state.utxos.len(&rwtxn)? == 2);
        // the block body connected at height 10; events disconnect
        // after the block body, with the height still at 10
        state.height.put(&mut rwtxn, &(), &10)?;
        disconnect(&state, &mut rwtxn, &tdp)?;

        anyhow::ensure!(state.utxos.len(&rwtxn)? == 0);
        anyhow::ensure!(state.deposit_blocks.len(&rwtxn)? == 0);
        Ok(())
    }
}
