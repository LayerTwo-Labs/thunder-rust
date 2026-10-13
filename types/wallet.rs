use std::collections::BTreeMap;

use bitcoin::Amount;
use serde::{Deserialize, Serialize};
use serde_with::{MapPreventDuplicates, serde_as};
use utoipa::ToSchema;

use crate::Address;

/// Destinations of a transfer. Each address takes a value in sats.
/// A repeated address is an error.
#[serde_as]
#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize, ToSchema)]
#[schema(value_type = BTreeMap<String, u64>)]
pub struct TransferDests(
    #[serde_as(as = "MapPreventDuplicates<_, _>")] pub BTreeMap<Address, u64>,
);

#[derive(Clone, Debug, Default, Deserialize, Serialize, ToSchema)]
pub struct Balance {
    #[serde(rename = "total_sats", with = "bitcoin::amount::serde::as_sat")]
    #[schema(value_type = u64)]
    pub total: Amount,
    #[serde(
        rename = "available_sats",
        with = "bitcoin::amount::serde::as_sat"
    )]
    #[schema(value_type = u64)]
    pub available: Amount,
}
