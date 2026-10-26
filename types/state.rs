use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::{M6id, OutPoint, Output, WithdrawalBundle};

/// Information we have regarding a withdrawal bundle
#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub enum WithdrawalBundleInfo {
    /// Withdrawal bundle is known
    Known(WithdrawalBundle),
    /// Withdrawal bundle is unknown but unconfirmed / failed
    Unknown,
    /// If an unknown withdrawal bundle is confirmed, ALL UTXOs are
    /// considered spent.
    UnknownConfirmed {
        spend_utxos: BTreeMap<OutPoint, Output>,
    },
}

/// A coin movement that a block applied outside its body
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize, ToSchema)]
pub enum TwoWayPegEvent {
    /// A mainchain deposit created this output
    Deposit { outpoint: OutPoint, output: Output },
    /// A withdrawal bundle spent this output
    BundleSpend { outpoint: OutPoint, m6id: M6id },
    /// A failed withdrawal bundle returned this output to the UTXO set
    BundleReturn {
        outpoint: OutPoint,
        output: Output,
        m6id: M6id,
    },
}
