//! Schemas for OpenAPI

use utoipa::{
    PartialSchema, ToSchema,
    openapi::{self, RefOr, Schema},
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

pub struct BitcoinBlockHash;

impl PartialSchema for BitcoinBlockHash {
    fn schema() -> RefOr<Schema> {
        let obj = utoipa::openapi::Object::with_type(openapi::Type::String);
        RefOr::T(Schema::Object(obj))
    }
}

impl ToSchema for BitcoinBlockHash {
    fn name() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("bitcoin.BlockHash")
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

pub struct BitcoinTransaction;

impl PartialSchema for BitcoinTransaction {
    fn schema() -> RefOr<Schema> {
        let obj = utoipa::openapi::Object::new();
        RefOr::T(Schema::Object(obj))
    }
}

impl ToSchema for BitcoinTransaction {
    fn name() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("bitcoin.Transaction")
    }
}

pub struct UtreexoNodeHash;

impl PartialSchema for UtreexoNodeHash {
    fn schema() -> RefOr<Schema> {
        let obj = utoipa::openapi::Object::with_type(openapi::Type::String);
        RefOr::T(Schema::Object(obj))
    }
}

impl ToSchema for UtreexoNodeHash {
    fn name() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("utreexo.NodeHash")
    }
}

pub struct UtreexoProof;

impl PartialSchema for UtreexoProof {
    fn schema() -> RefOr<Schema> {
        let obj = utoipa::openapi::Object::new();
        RefOr::T(Schema::Object(obj))
    }
}

impl ToSchema for UtreexoProof {
    fn name() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("utreexo.Proof")
    }
}

pub struct SocketAddr;

impl PartialSchema for SocketAddr {
    fn schema() -> RefOr<Schema> {
        let obj = utoipa::openapi::Object::with_type(openapi::Type::String);
        RefOr::T(Schema::Object(obj))
    }
}

impl ToSchema for SocketAddr {
    fn name() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("net.SocketAddr")
    }
}

pub struct PeerConnectionState;

impl PartialSchema for PeerConnectionState {
    fn schema() -> RefOr<Schema> {
        let obj = utoipa::openapi::Object::with_type(openapi::Type::String);
        RefOr::T(Schema::Object(obj))
    }
}

impl ToSchema for PeerConnectionState {
    fn name() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("net.PeerConnectionState")
    }
}
