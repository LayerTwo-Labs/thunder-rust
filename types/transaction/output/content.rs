use serde::{Deserialize, Serialize};
use utoipa::{PartialSchema, ToSchema};

use crate::{schema, transaction::GetValue, util};

/// Default representation for Serde
#[derive(Deserialize, Serialize)]
enum DefaultRepr {
    Value(bitcoin::Amount),
    Withdrawal {
        value: bitcoin::Amount,
        main_fee: bitcoin::Amount,
        main_address: bitcoin::Address<bitcoin::address::NetworkUnchecked>,
    },
}

/// Human-readable representation for Serde
#[derive(Deserialize, Serialize, ToSchema)]
#[schema(as = OutputContent, description = "")]
enum HumanReadableRepr {
    #[schema(value_type = u64)]
    Value(#[serde(with = "bitcoin::amount::serde::as_sat")] bitcoin::Amount),
    Withdrawal {
        #[serde(with = "bitcoin::amount::serde::as_sat")]
        #[serde(rename = "value_sats")]
        #[schema(value_type = u64)]
        value: bitcoin::Amount,
        #[serde(with = "bitcoin::amount::serde::as_sat")]
        #[serde(rename = "main_fee_sats")]
        #[schema(value_type = u64)]
        main_fee: bitcoin::Amount,
        #[schema(value_type = schema::BitcoinAddr)]
        main_address: bitcoin::Address<bitcoin::address::NetworkUnchecked>,
    },
}

type SerdeRepr = serde_with::IfIsHumanReadable<
    serde_with::FromInto<DefaultRepr>,
    serde_with::FromInto<HumanReadableRepr>,
>;

#[derive(borsh::BorshSerialize, Clone, Debug, Eq, PartialEq)]
pub enum Content {
    Value(
        #[borsh(serialize_with = "util::borsh::serialize::bitcoin_amount")]
        bitcoin::Amount,
    ),
    Withdrawal {
        #[borsh(serialize_with = "util::borsh::serialize::bitcoin_amount")]
        value: bitcoin::Amount,
        #[borsh(serialize_with = "util::borsh::serialize::bitcoin_amount")]
        main_fee: bitcoin::Amount,
        #[borsh(serialize_with = "util::borsh::serialize::bitcoin_address")]
        main_address: bitcoin::Address<bitcoin::address::NetworkUnchecked>,
    },
}

impl Content {
    pub fn is_value(&self) -> bool {
        matches!(self, Self::Value(_))
    }
    pub fn is_withdrawal(&self) -> bool {
        matches!(self, Self::Withdrawal { .. })
    }
}

impl GetValue for Content {
    #[inline(always)]
    fn get_value(&self) -> bitcoin::Amount {
        match self {
            Self::Value(value) => *value,
            // a withdrawal removes both the payout and the mainchain fee
            // from the sidechain, since the enforcer pays both out of the
            // treasury
            Self::Withdrawal {
                value, main_fee, ..
            } => value.checked_add(*main_fee).unwrap_or(bitcoin::Amount::MAX),
        }
    }
}

impl From<Content> for DefaultRepr {
    fn from(content: Content) -> Self {
        match content {
            Content::Value(value) => Self::Value(value),
            Content::Withdrawal {
                value,
                main_fee,
                main_address,
            } => Self::Withdrawal {
                value,
                main_fee,
                main_address,
            },
        }
    }
}

impl From<Content> for HumanReadableRepr {
    fn from(content: Content) -> Self {
        match content {
            Content::Value(value) => Self::Value(value),
            Content::Withdrawal {
                value,
                main_fee,
                main_address,
            } => Self::Withdrawal {
                value,
                main_fee,
                main_address,
            },
        }
    }
}

impl From<DefaultRepr> for Content {
    fn from(repr: DefaultRepr) -> Self {
        match repr {
            DefaultRepr::Value(value) => Self::Value(value),
            DefaultRepr::Withdrawal {
                value,
                main_fee,
                main_address,
            } => Self::Withdrawal {
                value,
                main_fee,
                main_address,
            },
        }
    }
}

impl From<HumanReadableRepr> for Content {
    fn from(repr: HumanReadableRepr) -> Self {
        match repr {
            HumanReadableRepr::Value(value) => Self::Value(value),
            HumanReadableRepr::Withdrawal {
                value,
                main_fee,
                main_address,
            } => Self::Withdrawal {
                value,
                main_fee,
                main_address,
            },
        }
    }
}

impl<'de> Deserialize<'de> for Content {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        <SerdeRepr as serde_with::DeserializeAs<'de, _>>::deserialize_as(
            deserializer,
        )
    }
}

impl Serialize for Content {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        <SerdeRepr as serde_with::SerializeAs<_>>::serialize_as(
            self, serializer,
        )
    }
}

impl PartialSchema for Content {
    fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
        <HumanReadableRepr as PartialSchema>::schema()
    }
}

impl ToSchema for Content {
    fn name() -> std::borrow::Cow<'static, str> {
        <HumanReadableRepr as ToSchema>::name()
    }
}
