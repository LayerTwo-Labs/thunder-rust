//! Schemas for OpenAPI

use serde::{Deserialize, Serialize};
use utoipa::{
    openapi::{self, RefOr, Schema},
    PartialSchema, ToSchema,
};

pub struct BitcoinTxid;

impl PartialSchema for BitcoinTxid {
    fn schema() -> RefOr<Schema> {
        let obj = utoipa::openapi::Object::with_type(openapi::Type::String);
        RefOr::T(Schema::Object(obj))
    }
}

impl ToSchema for BitcoinTxid {
    fn name() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("bitcoin.Txid")
    }
}

// RPC output representation for peer + state
// TODO: use better types here. Struggling with how to satisfy utoipa

#[derive(Clone, Deserialize, Serialize, ToSchema)]
pub struct Peer {
    pub address: String,
    pub state: String,
}

pub struct OpenApi;

impl PartialSchema for OpenApi {
    fn schema() -> RefOr<Schema> {
        let obj = utoipa::openapi::Object::new();
        RefOr::T(Schema::Object(obj))
    }
}

pub struct SocketAddr;

impl PartialSchema for SocketAddr {
    fn schema() -> RefOr<Schema> {
        let obj = utoipa::openapi::Object::with_type(openapi::Type::String);
        RefOr::T(Schema::Object(obj))
    }
}
