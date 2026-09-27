use bitcoin::Amount;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

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
    /// Value in outputs that the mempool holds and no block carries yet. It
    /// always counts toward `total`. It counts toward `available` only when
    /// `--spend-zero-conf-change` lets the wallet take it.
    #[serde(
        rename = "unconfirmed_sats",
        with = "bitcoin::amount::serde::as_sat"
    )]
    #[schema(value_type = u64)]
    pub unconfirmed: Amount,
}
