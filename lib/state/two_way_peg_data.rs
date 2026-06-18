//! Connect and disconnect two-way peg data

use std::collections::{BTreeMap, HashMap};

use fallible_iterator::FallibleIterator;
use sneed::{RoTxn, RwTxn, db::error::Error as DbError};

use crate::{
    state::{
        Error, State, WITHDRAWAL_BUNDLE_FAILURE_GAP, WithdrawalBundleInfo,
        error, rollback::RollBack,
    },
    types::{
        AccumulatorDiff, AggregatedWithdrawal, AmountOverflowError, InPoint,
        M6id, OutPoint, OutPointKey, Output, OutputContent, PointedOutput,
        PointedOutputRef, SpentOutput, WithdrawalBundle, WithdrawalBundleEvent,
        WithdrawalBundleEventStatus, WithdrawalBundleStatus, hash,
        proto::mainchain::{BlockEvent, TwoWayPegData},
    },
};

fn collect_withdrawal_bundle(
    state: &State,
    rotxn: &RoTxn,
    block_height: u32,
) -> Result<Option<WithdrawalBundle>, Error> {
    // Aggregate all outputs by destination.
    // destination -> (value, mainchain fee, spent_utxos)
    let mut address_to_aggregated_withdrawal = HashMap::<
        bitcoin::Address<bitcoin::address::NetworkUnchecked>,
        AggregatedWithdrawal,
    >::new();
    let () = state
        .utxos
        .iter(rotxn)
        .map_err(DbError::from)?
        .map_err(|err| DbError::from(err).into())
        .for_each(|(outpoint_key, output)| {
            let outpoint: OutPoint = outpoint_key.into();
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
            Ok::<_, Error>(())
        })?;
    if address_to_aggregated_withdrawal.is_empty() {
        return Ok(None);
    }
    let mut aggregated_withdrawals: Vec<_> =
        address_to_aggregated_withdrawal.into_values().collect();
    aggregated_withdrawals.sort_by_key(|a| std::cmp::Reverse(a.clone()));
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
        let Ok(spk_size) = u32::try_from(script_pubkey.len()) else {
            // This SPK is invalid, but others might be ok
            continue;
        };
        let Some(txout_size) = WithdrawalBundle::txout_size(spk_size) else {
            // This SPK is invalid, but others might be ok
            continue;
        };
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
    let bundle =
        WithdrawalBundle::new(block_height, fee, spend_utxos, bundle_outputs)?;
    Ok(Some(bundle))
}

fn connect_withdrawal_bundle_submitted(
    state: &State,
    rwtxn: &mut RwTxn,
    block_height: u32,
    accumulator_diff: &mut AccumulatorDiff,
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
        for (outpoint, spend_output) in bundle.spend_utxos() {
            let utxo_hash = hash(&PointedOutputRef {
                outpoint: *outpoint,
                output: spend_output,
            });
            accumulator_diff.remove(utxo_hash.into());
            let key = OutPointKey::from(outpoint);
            if !state.utxos.delete(rwtxn, &key)? {
                return Err(error::NoUtxo {
                    outpoint: *outpoint,
                }
                .into());
            };
            let spent_output = SpentOutput {
                output: spend_output.clone(),
                inpoint: InPoint::Withdrawal { m6id },
            };
            state.stxos.put(rwtxn, &key, &spent_output)?;
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
        WithdrawalBundleInfo::Known(bundle) => {
            if matches!(
                bundle_status.latest().value,
                WithdrawalBundleStatus::SubmittedUnexpected
            ) {
                // If a previously dropped or failed bundle is confirmed,
                // then unless all of the bundle UTXOs can be spent,
                // the chain is insolvent, and cannot continue.
                tracing::warn!(
                    %event_block_hash,
                    %m6id,
                    "Unexpected withdrawal bundle confirmed, marking bundle UTXOs as spent"
                );
                for (outpoint, output) in bundle.spend_utxos() {
                    let outpoint_key = OutPointKey::from(outpoint);
                    if !state.utxos.delete(rwtxn, &outpoint_key)? {
                        return Err(
                            Error::UnexpectedWithdrawalBundleInsolvency {
                                event_block_hash: *event_block_hash,
                                m6id,
                                outpoint: *outpoint,
                            },
                        );
                    }
                    let spent_output = SpentOutput {
                        output: output.clone(),
                        inpoint: InPoint::Withdrawal { m6id },
                    };
                    state.stxos.put(rwtxn, &outpoint_key, &spent_output)?;
                    let utxo_hash = hash(&PointedOutputRef {
                        outpoint: *outpoint,
                        output: &spent_output.output,
                    });
                    accumulator_diff.remove(utxo_hash.into());
                }
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
            for (outpoint, output) in bundle.spend_utxos() {
                let key = OutPointKey::from(outpoint);
                state.stxos.delete(rwtxn, &key).map_err(DbError::from)?;
                state
                    .utxos
                    .put(rwtxn, &key, output)
                    .map_err(DbError::from)?;
                let utxo_hash = hash(&PointedOutput {
                    outpoint: *outpoint,
                    output: output.clone(),
                });
                accumulator_diff.insert(utxo_hash.into());
            }
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
                accumulator_diff,
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
    event: &BlockEvent,
) -> Result<(), Error> {
    match event {
        BlockEvent::Deposit(deposit) => {
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
    let block_height = state.try_get_height(rwtxn)?.ok_or(Error::NoTip)?;
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
        for event in &event_block_info.events {
            let () = connect_event(
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
    let last_withdrawal_bundle_failure_height = state
        .get_latest_failed_withdrawal_bundle(rwtxn)
        .map_err(DbError::from)?
        .map(|(height, _bundle)| height)
        .unwrap_or_default();
    if block_height - last_withdrawal_bundle_failure_height
        >= WITHDRAWAL_BUNDLE_FAILURE_GAP
        && state
            .pending_withdrawal_bundle
            .try_get(rwtxn, &())
            .map_err(DbError::from)?
            .is_none()
        && let Some(bundle) =
            collect_withdrawal_bundle(state, rwtxn, block_height)?
    {
        let m6id = bundle.compute_m6id();
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
        tracing::trace!(
            %block_height,
            %m6id,
            "Stored pending withdrawal bundle"
        );
    }
    let () = accumulator.apply_diff(accumulator_diff)?;
    state
        .utreexo_accumulator
        .put(rwtxn, &(), &accumulator)
        .map_err(DbError::from)?;
    Ok(())
}

fn disconnect_withdrawal_bundle_submitted(
    state: &State,
    rwtxn: &mut RwTxn,
    block_height: u32,
    accumulator_diff: &mut AccumulatorDiff,
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
    match &bundle {
        WithdrawalBundleInfo::Unknown
        | WithdrawalBundleInfo::UnknownConfirmed { .. } => (),
        WithdrawalBundleInfo::Known(bundle) => {
            if let Some(bundle_status) = &bundle_status
                && bundle_status.latest().value
                    == WithdrawalBundleStatus::Pending
            {
                for (outpoint, output) in bundle.spend_utxos().iter().rev() {
                    if !state
                        .stxos
                        .delete(rwtxn, &OutPointKey::from(outpoint))?
                    {
                        return Err(Error::NoStxo {
                            outpoint: *outpoint,
                        });
                    };
                    state.utxos.put(
                        rwtxn,
                        &OutPointKey::from(outpoint),
                        output,
                    )?;
                    let utxo_hash = hash(&PointedOutput {
                        outpoint: *outpoint,
                        output: output.clone(),
                    });
                    accumulator_diff.insert(utxo_hash.into());
                }
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
        WithdrawalBundleInfo::Known(bundle) => {
            if matches!(
                prev_bundle_status.latest().value,
                WithdrawalBundleStatus::SubmittedUnexpected
            ) {
                for (outpoint, output) in bundle.spend_utxos() {
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
            }
        }
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
            for (outpoint, output) in bundle.spend_utxos().iter().rev() {
                let spent_output = SpentOutput {
                    output: output.clone(),
                    inpoint: InPoint::Withdrawal { m6id },
                };
                state
                    .stxos
                    .put(rwtxn, &OutPointKey::from(outpoint), &spent_output)
                    .map_err(DbError::from)?;
                if !state
                    .utxos
                    .delete(rwtxn, &OutPointKey::from(outpoint))
                    .map_err(DbError::from)?
                {
                    return Err(error::NoUtxo {
                        outpoint: *outpoint,
                    }
                    .into());
                };
                let utxo_hash = hash(&PointedOutput {
                    outpoint: *outpoint,
                    output: output.clone(),
                });
                accumulator_diff.remove(utxo_hash.into());
            }
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
                accumulator_diff,
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
            *latest_deposit_block_hash = Some(event_block_hash);
        }
        BlockEvent::WithdrawalBundle(withdrawal_bundle_event) => {
            let () = disconnect_withdrawal_bundle_event(
                state,
                rwtxn,
                block_height,
                accumulator_diff,
                withdrawal_bundle_event,
            )?;
            *latest_withdrawal_bundle_event_block_hash = Some(event_block_hash);
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
    // Restore pending withdrawal bundle
    for (event_block_hash, event_block_info) in
        two_way_peg_data.block_info.iter().rev()
    {
        for event in event_block_info.events.iter().rev() {
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
    let last_withdrawal_bundle_failure_height = state
        .get_latest_failed_withdrawal_bundle(rwtxn)?
        .map(|(height, _bundle)| height)
        .unwrap_or_default();
    if block_height - last_withdrawal_bundle_failure_height
        >= WITHDRAWAL_BUNDLE_FAILURE_GAP
        && let Some(bundle_m6id) =
            state.pending_withdrawal_bundle.try_get(rwtxn, &())?
        && let (bundle, bundle_status) = state
            .withdrawal_bundles
            .try_get(rwtxn, &bundle_m6id)?
            .ok_or(error::PendingWithdrawalBundleUnknown(bundle_m6id))?
        && bundle_status.latest().height == block_height
    {
        state.pending_withdrawal_bundle.delete(rwtxn, &())?;
        if let (Some(bundle_status), _latest_bundle_status) =
            bundle_status.pop()
        {
            state.withdrawal_bundles.put(
                rwtxn,
                &bundle_m6id,
                &(bundle, bundle_status),
            )?;
        } else {
            state.withdrawal_bundles.delete(rwtxn, &bundle_m6id)?;
        }
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use bitcoin::{
        Network,
        hashes::Hash as _,
        secp256k1::{Secp256k1, SecretKey},
    };
    use hashlink::LinkedHashMap;

    use crate::{
        state::{
            State, WithdrawalBundleInfo,
            rollback::RollBack,
            two_way_peg_data::{
                collect_withdrawal_bundle, disconnect,
                disconnect_withdrawal_bundle_failed,
            },
        },
        types::{
            AccumulatorDiff, Address, InPoint, M6id, OutPoint, OutPointKey,
            Output, OutputContent, Txid, WithdrawalBundle,
            WithdrawalBundleEvent, WithdrawalBundleEventStatus,
            WithdrawalBundleStatus,
            proto::mainchain::{BlockEvent, BlockInfo, TwoWayPegData},
        },
    };

    fn temp_env_path(test_name: &str) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        path.push(format!("photon-{test_name}-{}-{nanos}", std::process::id()));
        path
    }

    // open a fresh state-backed env in a unique temp dir
    fn temp_env(test_name: &str) -> sneed::Env {
        let path = temp_env_path(test_name);
        std::fs::create_dir_all(&path).unwrap();
        let mut opts = heed::EnvOpenOptions::new();
        opts.map_size(16 * 1024 * 1024).max_dbs(State::NUM_DBS);
        unsafe { sneed::Env::open(&opts, &path) }.unwrap()
    }

    // a failed known bundle reinstates its utxos as spendable, so disconnecting
    // the failure must spend them again
    #[test]
    fn disconnect_failed_bundle_spends_reinstated_utxo() {
        let env = temp_env("disconnect_failed_bundle_spends_reinstated_utxo");
        let state = State::new(&env).unwrap();
        let outpoint = OutPoint::Regular {
            txid: Txid::from([1; 32]),
            vout: 0,
        };
        let output = Output {
            address: Address::ALL_ZEROS,
            content: OutputContent::Value(bitcoin::Amount::from_sat(1000)),
        };
        let key = OutPointKey::from(&outpoint);

        let m6id = {
            let mut spend_utxos = BTreeMap::new();
            spend_utxos.insert(outpoint, output.clone());
            let bundle = WithdrawalBundle::new(
                1,
                bitcoin::Amount::ZERO,
                spend_utxos,
                Vec::new(),
            )
            .unwrap();
            let m6id = bundle.compute_m6id();
            let mut bundle_status =
                RollBack::new(WithdrawalBundleStatus::Submitted, 0);
            bundle_status
                .push(WithdrawalBundleStatus::Failed, 1)
                .unwrap();
            let mut rwtxn = env.write_txn().unwrap();
            state
                .withdrawal_bundles
                .put(
                    &mut rwtxn,
                    &m6id,
                    &(WithdrawalBundleInfo::Known(bundle), bundle_status),
                )
                .unwrap();
            state
                .latest_failed_withdrawal_bundle
                .put(&mut rwtxn, &(), &RollBack::new(m6id, 1))
                .unwrap();
            // the failure reinstated the utxo
            state.utxos.put(&mut rwtxn, &key, &output).unwrap();
            rwtxn.commit().unwrap();
            m6id
        };

        let mut rwtxn = env.write_txn().unwrap();
        let mut accumulator_diff = AccumulatorDiff::default();
        disconnect_withdrawal_bundle_failed(
            &state,
            &mut rwtxn,
            1,
            &mut accumulator_diff,
            m6id,
        )
        .unwrap();
        assert!(state.utxos.try_get(&rwtxn, &key).unwrap().is_none());
        let stxo = state.stxos.try_get(&rwtxn, &key).unwrap().unwrap();
        assert_eq!(stxo.inpoint, InPoint::Withdrawal { m6id });
    }

    // disconnecting a withdrawal bundle event must remove its
    // withdrawal_bundle_event_blocks record, not a deposit_blocks record that
    // happens to share the same sequence index
    #[test]
    fn disconnect_withdrawal_event_block_uses_correct_db() {
        let env = temp_env("disconnect_withdrawal_event_block_uses_correct_db");
        let state = State::new(&env).unwrap();

        let block_height = 5u32;
        let m6id = M6id(bitcoin::Txid::from_byte_array([7; 32]));
        let event_block_hash = bitcoin::BlockHash::from_byte_array([9; 32]);
        let deposit_block_hash = bitcoin::BlockHash::from_byte_array([3; 32]);

        let mut rwtxn = env.write_txn().unwrap();
        state.height.put(&mut rwtxn, &(), &block_height).unwrap();
        state
            .withdrawal_bundles
            .put(
                &mut rwtxn,
                &m6id,
                &(
                    WithdrawalBundleInfo::Unknown,
                    RollBack::new(
                        WithdrawalBundleStatus::Submitted,
                        block_height,
                    ),
                ),
            )
            .unwrap();
        state
            .withdrawal_bundle_event_blocks
            .put(&mut rwtxn, &0, &(event_block_hash, block_height))
            .unwrap();
        // a deposit record at the same sequence index that must survive
        state
            .deposit_blocks
            .put(&mut rwtxn, &0, &(deposit_block_hash, block_height))
            .unwrap();
        rwtxn.commit().unwrap();

        let two_way_peg_data = {
            let mut block_info = LinkedHashMap::new();
            block_info.insert(
                event_block_hash,
                BlockInfo {
                    bmm_commitment: None,
                    events: vec![BlockEvent::WithdrawalBundle(
                        WithdrawalBundleEvent {
                            m6id,
                            status: WithdrawalBundleEventStatus::Submitted,
                        },
                    )],
                },
            );
            TwoWayPegData { block_info }
        };

        let mut rwtxn = env.write_txn().unwrap();
        disconnect(&state, &mut rwtxn, &two_way_peg_data).unwrap();
        assert!(
            state
                .withdrawal_bundle_event_blocks
                .try_get(&rwtxn, &0)
                .unwrap()
                .is_none()
        );
        assert!(state.deposit_blocks.try_get(&rwtxn, &0).unwrap().is_some());
        rwtxn.commit().unwrap();
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
        let env_path = temp_env_path(test_name);
        std::fs::create_dir_all(&env_path)?;
        let env = {
            let mut env_open_options = heed::EnvOpenOptions::new();
            env_open_options
                .map_size(16 * 1024 * 1024)
                .max_dbs(State::NUM_DBS);
            unsafe { sneed::Env::open(&env_open_options, &env_path) }?
        };
        let state = State::new(&env)?;
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
        let result = f(&state, &mut rwtxn);
        drop(rwtxn);
        drop(state);
        drop(env);
        drop(std::fs::remove_dir_all(&env_path));
        Ok(result)
    }

    #[test]
    fn collect_withdrawal_bundle_p2wpkh_off_by_one_does_not_exceed_weight()
    -> anyhow::Result<()> {
        const CLAIMED_MAX_BUNDLE_OUTPUTS: u32 = 3_222;

        let bundle = with_state_with_withdrawals(
            "collect_withdrawal_bundle_p2wpkh_off_by_one",
            CLAIMED_MAX_BUNDLE_OUTPUTS + 1,
            regtest_p2wpkh_address,
            |state, rwtxn| collect_withdrawal_bundle(state, rwtxn, 42),
        )?;
        let bundle = match bundle {
            Ok(Some(bundle)) => bundle,
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
}
