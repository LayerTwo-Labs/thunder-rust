//! Schemas for OpenAPI

use utoipa::{
    openapi::{self, RefOr, Schema},
    PartialSchema, ToSchema,
};

pub struct BitcoinAddr;

impl PartialSchema for BitcoinAddr {
    fn schema() -> RefOr<Schema> {
        let obj = utoipa::openapi::Object::with_type(openapi::Type::String);
        RefOr::T(Schema::Object(obj))
    }
}

impl ToSchema for BitcoinAddr {
    fn name() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("bitcoin.Address")
    }
}

pub struct BitcoinOutPoint;

impl PartialSchema for BitcoinOutPoint {
    fn schema() -> RefOr<Schema> {
        let obj = utoipa::openapi::Object::new();
        RefOr::T(Schema::Object(obj))
    }
}

impl ToSchema for BitcoinOutPoint {
    fn name() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("bitcoin.OutPoint")
    }
}
