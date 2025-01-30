//! Connect and disconnect two-way peg data

use std::collections::{BTreeMap, HashMap};

use hashlink::LinkedHashSet;
use heed::{RoTxn, RwTxn};
use rustreexo::accumulator::node_hash::BitcoinNodeHash;

use crate::{
    state::{rollback::RollBack, Error, State, WITHDRAWAL_BUNDLE_FAILURE_GAP},
    types::{
        hash,
        proto::mainchain::{BlockEvent, TwoWayPegData},
        AggregatedWithdrawal, AmountOverflowError, InPoint, M6id, OutPoint,
        Output, OutputContent, PointedOutput, SpentOutput, WithdrawalBundle,
        WithdrawalBundleEvent, WithdrawalBundleStatus,
    },
    util::UnitKey,
};

fn collect_withdrawal_bundle(
    state: &State,
    rotxn: &RoTxn,
    block_height: u32,
) -> Result<Option<WithdrawalBundle>, Error> {
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
    for item in state.utxos.iter(rotxn)? {
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
    }
    if address_to_aggregated_withdrawal.is_empty() {
        return Ok(None);
    }
    let mut aggregated_withdrawals: Vec<_> =
        address_to_aggregated_withdrawal.into_values().collect();
    aggregated_withdrawals.sort_by_key(|a| std::cmp::Reverse(a.clone()));
    let mut fee = bitcoin::Amount::ZERO;
    let mut spend_utxos = BTreeMap::<OutPoint, Output>::new();
    let mut bundle_outputs = vec![];
    for aggregated in &aggregated_withdrawals {
        if bundle_outputs.len() > MAX_BUNDLE_OUTPUTS {
            break;
        }
        let bundle_output = bitcoin::TxOut {
            value: aggregated.value,
            script_pubkey: aggregated
                .main_address
                .assume_checked_ref()
                .script_pubkey(),
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
    accumulator_del: &mut LinkedHashSet<BitcoinNodeHash>,
    event_block_hash: &bitcoin::BlockHash,
    m6id: M6id,
) -> Result<(), Error> {
    if let Some((bundle, bundle_block_height)) =
        state.pending_withdrawal_bundle.get(rwtxn, &UnitKey)?
        && bundle.compute_m6id() == m6id
    {
        assert_eq!(bundle_block_height, block_height - 1);
        tracing::debug!(
            %block_height,
            %m6id,
            "Withdrawal bundle successfully submitted"
        );
        for (outpoint, spend_output) in bundle.spend_utxos() {
            let utxo_hash = hash(&PointedOutput {
                outpoint: *outpoint,
                output: spend_output.clone(),
            });
            accumulator_del.replace(utxo_hash.into());
            state.utxos.delete(rwtxn, outpoint)?;
            let spent_output = SpentOutput {
                output: spend_output.clone(),
                inpoint: InPoint::Withdrawal { m6id },
            };
            state.stxos.put(rwtxn, outpoint, &spent_output)?;
        }
        state.withdrawal_bundles.put(
            rwtxn,
            &m6id,
            &(
                Some(bundle),
                RollBack::new(WithdrawalBundleStatus::Submitted, block_height),
            ),
        )?;
        state.pending_withdrawal_bundle.delete(rwtxn, &UnitKey)?;
    } else if let Some((_bundle, bundle_status)) =
        state.withdrawal_bundles.get(rwtxn, &m6id)?
    {
        // Already applied
        assert_eq!(
            bundle_status.earliest().value,
            WithdrawalBundleStatus::Submitted
        );
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
                None,
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
    accumulator_del: &mut LinkedHashSet<BitcoinNodeHash>,
    event_block_hash: &bitcoin::BlockHash,
    m6id: M6id,
) -> Result<(), Error> {
    let (bundle, mut bundle_status) = state
        .withdrawal_bundles
        .get(rwtxn, &m6id)?
        .ok_or(Error::UnknownWithdrawalBundle { m6id })?;
    if bundle_status.latest().value == WithdrawalBundleStatus::Confirmed {
        // Already applied
        return Ok(());
    }
    assert_eq!(
        bundle_status.latest().value,
        WithdrawalBundleStatus::Submitted
    );
    // If an unknown bundle is confirmed, all UTXOs older than the
    // bundle submission are potentially spent.
    // This is only accepted in the case that block height is 0,
    // and so no UTXOs could possibly have been double-spent yet.
    // In this case, ALL UTXOs are considered spent.
    if bundle.is_none() {
        if block_height == 0 {
            tracing::warn!(
                %event_block_hash,
                %m6id,
                "Unknown withdrawal bundle confirmed, marking all UTXOs as spent"
            );
            let utxos: Vec<_> =
                state.utxos.iter(rwtxn)?.collect::<Result<_, _>>()?;
            for (outpoint, output) in utxos {
                let spent_output = SpentOutput {
                    output,
                    inpoint: InPoint::Withdrawal { m6id },
                };
                state.stxos.put(rwtxn, &outpoint, &spent_output)?;
                let utxo_hash = hash(&PointedOutput {
                    outpoint,
                    output: spent_output.output,
                });
                accumulator_del.replace(utxo_hash.into());
            }
            state.utxos.clear(rwtxn)?;
        } else {
            return Err(Error::UnknownWithdrawalBundleConfirmed {
                event_block_hash: *event_block_hash,
                m6id,
            });
        }
    }
    bundle_status
        .push(WithdrawalBundleStatus::Confirmed, block_height)
        .expect("Push confirmed status should be valid");
    state
        .withdrawal_bundles
        .put(rwtxn, &m6id, &(bundle, bundle_status))?;
    Ok(())
}

fn connect_withdrawal_bundle_failed(
    state: &State,
    rwtxn: &mut RwTxn,
    block_height: u32,
    accumulator_add: &mut Vec<BitcoinNodeHash>,
    m6id: M6id,
) -> Result<(), Error> {
    tracing::debug!(
        %block_height,
        %m6id,
        "Handling failed withdrawal bundle");
    let (bundle, mut bundle_status) = state
        .withdrawal_bundles
        .get(rwtxn, &m6id)?
        .ok_or_else(|| Error::UnknownWithdrawalBundle { m6id })?;
    if bundle_status.latest().value == WithdrawalBundleStatus::Failed {
        // Already applied
        return Ok(());
    }
    assert_eq!(
        bundle_status.latest().value,
        WithdrawalBundleStatus::Submitted
    );
    bundle_status
        .push(WithdrawalBundleStatus::Failed, block_height)
        .expect("Push failed status should be valid");
    if let Some(bundle) = &bundle {
        for (outpoint, output) in bundle.spend_utxos() {
            state.stxos.delete(rwtxn, outpoint)?;
            state.utxos.put(rwtxn, outpoint, output)?;
            let utxo_hash = hash(&PointedOutput {
                outpoint: *outpoint,
                output: output.clone(),
            });
            accumulator_add.push(utxo_hash.into());
        }
        let latest_failed_m6id = if let Some(mut latest_failed_m6id) =
            state.latest_failed_withdrawal_bundle.get(rwtxn, &UnitKey)?
        {
            latest_failed_m6id
                .push(m6id, block_height)
                .expect("Push latest failed m6id should be valid");
            latest_failed_m6id
        } else {
            RollBack::new(m6id, block_height)
        };
        state.latest_failed_withdrawal_bundle.put(
            rwtxn,
            &UnitKey,
            &latest_failed_m6id,
        )?;
    }
    state
        .withdrawal_bundles
        .put(rwtxn, &m6id, &(bundle, bundle_status))?;
    Ok(())
}

fn connect_withdrawal_bundle_event(
    state: &State,
    rwtxn: &mut RwTxn,
    block_height: u32,
    accumulator_add: &mut Vec<BitcoinNodeHash>,
    accumulator_del: &mut LinkedHashSet<BitcoinNodeHash>,
    event_block_hash: &bitcoin::BlockHash,
    event: &WithdrawalBundleEvent,
) -> Result<(), Error> {
    match event.status {
        WithdrawalBundleStatus::Submitted => {
            connect_withdrawal_bundle_submitted(
                state,
                rwtxn,
                block_height,
                accumulator_del,
                event_block_hash,
                event.m6id,
            )
        }
        WithdrawalBundleStatus::Confirmed => {
            connect_withdrawal_bundle_confirmed(
                state,
                rwtxn,
                block_height,
                accumulator_del,
                event_block_hash,
                event.m6id,
            )
        }
        WithdrawalBundleStatus::Failed => connect_withdrawal_bundle_failed(
            state,
            rwtxn,
            block_height,
            accumulator_add,
            event.m6id,
        ),
    }
}

fn connect_event(
    state: &State,
    rwtxn: &mut RwTxn,
    block_height: u32,
    accumulator_add: &mut Vec<BitcoinNodeHash>,
    accumulator_del: &mut LinkedHashSet<BitcoinNodeHash>,
    latest_deposit_block_hash: &mut Option<bitcoin::BlockHash>,
    latest_withdrawal_bundle_event_block_hash: &mut Option<bitcoin::BlockHash>,
    event_block_hash: bitcoin::BlockHash,
    event: &BlockEvent,
) -> Result<(), Error> {
    match event {
        BlockEvent::Deposit(deposit) => {
            let outpoint = OutPoint::Deposit(deposit.outpoint);
            let output = deposit.output.clone();
            state.utxos.put(rwtxn, &outpoint, &output)?;
            let utxo_hash = hash(&PointedOutput { outpoint, output });
            accumulator_add.push(utxo_hash.into());
            *latest_deposit_block_hash = Some(event_block_hash);
        }
        BlockEvent::WithdrawalBundle(withdrawal_bundle_event) => {
            let () = connect_withdrawal_bundle_event(
                state,
                rwtxn,
                block_height,
                accumulator_add,
                accumulator_del,
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
        .get(rwtxn, &UnitKey)?
        .unwrap_or_default();
    // New leaves for the accumulator
    let mut accumulator_add = Vec::<BitcoinNodeHash>::new();
    // Accumulator leaves to delete
    let mut accumulator_del = hashlink::LinkedHashSet::<BitcoinNodeHash>::new();
    let mut latest_deposit_block_hash = None;
    let mut latest_withdrawal_bundle_event_block_hash = None;
    for (event_block_hash, event_block_info) in &two_way_peg_data.block_info {
        for event in &event_block_info.events {
            let () = connect_event(
                state,
                rwtxn,
                block_height,
                &mut accumulator_add,
                &mut accumulator_del,
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
            .last(rwtxn)?
            .map_or(0, |(seq_idx, _)| seq_idx + 1);
        state.deposit_blocks.put(
            rwtxn,
            &deposit_block_seq_idx,
            &(latest_deposit_block_hash, block_height),
        )?;
    }
    // Handle withdrawals
    if let Some(latest_withdrawal_bundle_event_block_hash) =
        latest_withdrawal_bundle_event_block_hash
    {
        let withdrawal_bundle_event_block_seq_idx = state
            .withdrawal_bundle_event_blocks
            .last(rwtxn)?
            .map_or(0, |(seq_idx, _)| seq_idx + 1);
        state.withdrawal_bundle_event_blocks.put(
            rwtxn,
            &withdrawal_bundle_event_block_seq_idx,
            &(latest_withdrawal_bundle_event_block_hash, block_height),
        )?;
    }
    let last_withdrawal_bundle_failure_height = state
        .get_latest_failed_withdrawal_bundle(rwtxn)?
        .map(|(height, _bundle)| height)
        .unwrap_or_default();
    if block_height - last_withdrawal_bundle_failure_height
        >= WITHDRAWAL_BUNDLE_FAILURE_GAP
        && state
            .pending_withdrawal_bundle
            .get(rwtxn, &UnitKey)?
            .is_none()
    {
        if let Some(bundle) =
            collect_withdrawal_bundle(state, rwtxn, block_height)?
        {
            let m6id = bundle.compute_m6id();
            state.pending_withdrawal_bundle.put(
                rwtxn,
                &UnitKey,
                &(bundle, block_height),
            )?;
            tracing::trace!(
                %block_height,
                %m6id,
                "Stored pending withdrawal bundle"
            );
        }
    }
    let accumulator_del: Vec<_> = accumulator_del.into_iter().collect();
    tracing::debug!(
        accumulator = %accumulator.0,
        accumulator_add = ?accumulator_add,
        accumulator_del = ?accumulator_del,
        "Updating accumulator");
    accumulator
        .0
        .modify(&accumulator_add, &accumulator_del)
        .map_err(Error::Utreexo)?;
    tracing::debug!(accumulator = %accumulator.0, "Updated accumulator");
    state
        .utreexo_accumulator
        .put(rwtxn, &UnitKey, &accumulator)?;
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
        .get(rwtxn, &UnitKey)?
        .unwrap_or_default();
    // New leaves for the accumulator
    let mut accumulator_add = Vec::<BitcoinNodeHash>::new();
    // Accumulator leaves to delete
    let mut accumulator_del = hashlink::LinkedHashSet::<BitcoinNodeHash>::new();

    // Restore pending withdrawal bundle
    for (_, event) in two_way_peg_data.withdrawal_bundle_events().rev() {
        match event.status {
            WithdrawalBundleStatus::Submitted => {
                let Some((bundle, bundle_status)) =
                    state.withdrawal_bundles.get(rwtxn, &event.m6id)?
                else {
                    if let Some((bundle, _)) =
                        state.pending_withdrawal_bundle.get(rwtxn, &UnitKey)?
                        && bundle.compute_m6id() == event.m6id
                    {
                        // Already applied
                        continue;
                    }
                    return Err(Error::UnknownWithdrawalBundle {
                        m6id: event.m6id,
                    });
                };
                let bundle_status = bundle_status.latest();
                assert_eq!(
                    bundle_status.value,
                    WithdrawalBundleStatus::Submitted
                );
                assert_eq!(bundle_status.height, block_height);
                for (outpoint, output) in bundle.spend_utxos().iter().rev() {
                    if !state.stxos.delete(rwtxn, outpoint)? {
                        return Err(Error::NoStxo {
                            outpoint: *outpoint,
                        });
                    };
                    state.utxos.put(rwtxn, outpoint, output)?;
                    let utxo_hash = hash(&PointedOutput {
                        outpoint: *outpoint,
                        output: output.clone(),
                    });
                    accumulator_add.push(utxo_hash.into());
                }
                state.pending_withdrawal_bundle.put(
                    rwtxn,
                    &UnitKey,
                    &(bundle, bundle_status.height - 1),
                )?;
                state.withdrawal_bundles.delete(rwtxn, &event.m6id)?;
            }
            WithdrawalBundleStatus::Confirmed => {
                let Some((bundle, bundle_status)) =
                    state.withdrawal_bundles.get(rwtxn, &event.m6id)?
                else {
                    return Err(Error::UnknownWithdrawalBundle {
                        m6id: event.m6id,
                    });
                };
                let (prev_bundle_status, latest_bundle_status) =
                    bundle_status.pop();
                if latest_bundle_status.value
                    == WithdrawalBundleStatus::Submitted
                {
                    // Already applied
                    continue;
                } else {
                    assert_eq!(
                        latest_bundle_status.value,
                        WithdrawalBundleStatus::Confirmed
                    );
                }
                assert_eq!(latest_bundle_status.height, block_height);
                let prev_bundle_status = prev_bundle_status
                    .expect("Pop confirmed bundle status should be valid");
                assert_eq!(
                    prev_bundle_status.latest().value,
                    WithdrawalBundleStatus::Submitted
                );
                state.withdrawal_bundles.put(
                    rwtxn,
                    &event.m6id,
                    &(bundle, prev_bundle_status),
                )?;
            }
            WithdrawalBundleStatus::Failed => {
                let Some((bundle, bundle_status)) =
                    state.withdrawal_bundles.get(rwtxn, &event.m6id)?
                else {
                    return Err(Error::UnknownWithdrawalBundle {
                        m6id: event.m6id,
                    });
                };
                let (prev_bundle_status, latest_bundle_status) =
                    bundle_status.pop();
                if latest_bundle_status.value
                    == WithdrawalBundleStatus::Submitted
                {
                    // Already applied
                    continue;
                } else {
                    assert_eq!(
                        latest_bundle_status.value,
                        WithdrawalBundleStatus::Failed
                    );
                }
                assert_eq!(latest_bundle_status.height, block_height);
                let prev_bundle_status = prev_bundle_status
                    .expect("Pop failed bundle status should be valid");
                assert_eq!(
                    prev_bundle_status.latest().value,
                    WithdrawalBundleStatus::Submitted
                );
                for (outpoint, output) in bundle.spend_utxos().iter().rev() {
                    let spent_output = SpentOutput {
                        output: output.clone(),
                        inpoint: InPoint::Withdrawal { m6id: event.m6id },
                    };
                    state.stxos.put(rwtxn, outpoint, &spent_output)?;
                    if state.utxos.delete(rwtxn, outpoint)? {
                        return Err(Error::NoUtxo {
                            outpoint: *outpoint,
                        });
                    };
                    let utxo_hash = hash(&PointedOutput {
                        outpoint: *outpoint,
                        output: output.clone(),
                    });
                    accumulator_del.replace(utxo_hash.into());
                }
                state.withdrawal_bundles.put(
                    rwtxn,
                    &event.m6id,
                    &(bundle, prev_bundle_status),
                )?;
                let (prev_latest_failed_m6id, latest_failed_m6id) = state
                    .latest_failed_withdrawal_bundle
                    .get(rwtxn, &UnitKey)?
                    .expect("latest failed withdrawal bundle should exist")
                    .pop();
                assert_eq!(latest_failed_m6id.value, event.m6id);
                assert_eq!(latest_failed_m6id.height, block_height);
                if let Some(prev_latest_failed_m6id) = prev_latest_failed_m6id {
                    state.latest_failed_withdrawal_bundle.put(
                        rwtxn,
                        &UnitKey,
                        &prev_latest_failed_m6id,
                    )?;
                } else {
                    state
                        .latest_failed_withdrawal_bundle
                        .delete(rwtxn, &UnitKey)?;
                }
            }
        }
    }
    // Handle withdrawals
    if let Some(latest_withdrawal_bundle_event_block_hash) =
        two_way_peg_data.latest_withdrawal_bundle_event_block_hash()
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
            *latest_withdrawal_bundle_event_block_hash,
            last_withdrawal_bundle_event_block_hash
        );
        assert_eq!(block_height - 1, last_withdrawal_bundle_event_block_height);
        if !state
            .deposit_blocks
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
        > WITHDRAWAL_BUNDLE_FAILURE_GAP
        && let Some((bundle, bundle_height)) =
            state.pending_withdrawal_bundle.get(rwtxn, &UnitKey)?
        && bundle_height == block_height - 1
    {
        state.pending_withdrawal_bundle.delete(rwtxn, &UnitKey)?;
        for (outpoint, output) in bundle.spend_utxos().iter().rev() {
            let utxo_hash = hash(&PointedOutput {
                outpoint: *outpoint,
                output: output.clone(),
            });
            accumulator_add.push(utxo_hash.into());
            if !state.stxos.delete(rwtxn, outpoint)? {
                return Err(Error::NoStxo {
                    outpoint: *outpoint,
                });
            };
            state.utxos.put(rwtxn, outpoint, output)?;
        }
    }
    // Handle deposits.
    if let Some(latest_deposit_block_hash) =
        two_way_peg_data.latest_deposit_block_hash()
    {
        let (
            last_deposit_block_seq_idx,
            (last_deposit_block_hash, last_deposit_block_height),
        ) = state
            .deposit_blocks
            .last(rwtxn)?
            .ok_or(Error::NoDepositBlock)?;
        assert_eq!(latest_deposit_block_hash, last_deposit_block_hash);
        assert_eq!(block_height - 1, last_deposit_block_height);
        if !state
            .deposit_blocks
            .delete(rwtxn, &last_deposit_block_seq_idx)?
        {
            return Err(Error::NoDepositBlock);
        };
    }
    for deposit in two_way_peg_data
        .deposits()
        .flat_map(|(_, deposits)| deposits)
        .rev()
    {
        let outpoint = OutPoint::Deposit(deposit.outpoint);
        let output = deposit.output.clone();
        if !state.utxos.delete(rwtxn, &outpoint)? {
            return Err(Error::NoUtxo { outpoint });
        }
        let utxo_hash = hash(&PointedOutput { outpoint, output });
        accumulator_del.insert(utxo_hash.into());
    }
    let accumulator_del: Vec<_> = accumulator_del.into_iter().collect();
    accumulator
        .0
        .modify(&accumulator_add, &accumulator_del)
        .map_err(Error::Utreexo)?;
    state
        .utreexo_accumulator
        .put(rwtxn, &UnitKey, &accumulator)?;
    Ok(())
}
