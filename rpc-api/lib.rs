//! RPC API

use std::net::SocketAddr;

use jsonrpsee::{core::RpcResult, proc_macros::rpc};
use l2l_openapi::open_api;
use thunder::types::{
    Address, MerkleRoot, OutPoint, Output, OutputContent, PointedOutput, Txid,
};
use utoipa::{
    openapi::{RefOr, Schema, SchemaType},
    PartialSchema, ToSchema,
};

struct BitcoinAddrSchema;

impl PartialSchema for BitcoinAddrSchema {
    fn schema() -> RefOr<Schema> {
        let obj = utoipa::openapi::Object::with_type(SchemaType::String);
        RefOr::T(Schema::Object(obj))
    }
}

impl ToSchema<'static> for BitcoinAddrSchema {
    fn schema() -> (&'static str, RefOr<Schema>) {
        ("bitcoin.Address", <Self as PartialSchema>::schema())
    }
}

struct BitcoinAmountSchema;

impl PartialSchema for BitcoinAmountSchema {
    fn schema() -> RefOr<Schema> {
        let obj = utoipa::openapi::Object::with_type(SchemaType::String);
        RefOr::T(Schema::Object(obj))
    }
}

struct BitcoinOutPointSchema;

impl PartialSchema for BitcoinOutPointSchema {
    fn schema() -> RefOr<Schema> {
        let obj = utoipa::openapi::Object::new();
        RefOr::T(Schema::Object(obj))
    }
}

impl ToSchema<'static> for BitcoinOutPointSchema {
    fn schema() -> (&'static str, RefOr<Schema>) {
        ("bitcoin.OutPoint", <Self as PartialSchema>::schema())
    }
}

struct OpenApiSchema;

impl PartialSchema for OpenApiSchema {
    fn schema() -> RefOr<Schema> {
        let obj = utoipa::openapi::Object::new();
        RefOr::T(Schema::Object(obj))
    }
}

struct SocketAddrSchema;

impl PartialSchema for SocketAddrSchema {
    fn schema() -> RefOr<Schema> {
        let obj = utoipa::openapi::Object::with_type(SchemaType::String);
        RefOr::T(Schema::Object(obj))
    }
}

#[open_api(ref_schemas[
    Address, BitcoinAddrSchema, BitcoinOutPointSchema, MerkleRoot, OutPoint, Output,
    OutputContent, Txid
])]
#[rpc(client, server)]
pub trait Rpc {
    /// Get balance in sats
    #[method(name = "balance")]
    async fn balance(&self) -> RpcResult<u64>;

    /// Connect to a peer
    #[open_api_method(output_schema(ToSchema))]
    #[method(name = "connect_peer")]
    async fn connect_peer(
        &self,
        #[open_api_method_arg(schema(PartialSchema = "SocketAddrSchema"))]
        addr: SocketAddr,
    ) -> RpcResult<()>;

    /// Format a deposit address
    #[method(name = "format_deposit_address")]
    async fn format_deposit_address(
        &self,
        address: Address,
    ) -> RpcResult<String>;

    /// Generate a mnemonic seed phrase
    #[method(name = "generate_mnemonic")]
    async fn generate_mnemonic(&self) -> RpcResult<String>;

    /// Get a new address
    #[method(name = "get_new_address")]
    async fn get_new_address(&self) -> RpcResult<Address>;

    /// Get wallet addresses, sorted by base58 encoding
    #[method(name = "get_wallet_addresses")]
    async fn get_wallet_addresses(&self) -> RpcResult<Vec<Address>>;

    /// Get wallet UTXOs
    #[method(name = "get_wallet_utxos")]
    async fn get_wallet_utxos(&self) -> RpcResult<Vec<PointedOutput>>;

    /// Get the current block count
    #[method(name = "getblockcount")]
    async fn getblockcount(&self) -> RpcResult<u32>;

    /// List all UTXOs
    #[method(name = "list_utxos")]
    async fn list_utxos(&self) -> RpcResult<Vec<PointedOutput>>;

    /// Attempt to mine a sidechain block
    #[open_api_method(output_schema(ToSchema))]
    #[method(name = "mine")]
    async fn mine(&self, fee: Option<u64>) -> RpcResult<()>;

    /// Get OpenAPI schema
    #[open_api_method(output_schema(PartialSchema = "OpenApiSchema"))]
    #[method(name = "openapi_schema")]
    async fn openapi_schema(&self) -> RpcResult<utoipa::openapi::OpenApi>;

    /// Remove a tx from the mempool
    #[open_api_method(output_schema(ToSchema))]
    #[method(name = "remove_from_mempool")]
    async fn remove_from_mempool(&self, txid: Txid) -> RpcResult<()>;

    /// Set the wallet seed from a mnemonic seed phrase
    #[open_api_method(output_schema(ToSchema))]
    #[method(name = "set_seed_from_mnemonic")]
    async fn set_seed_from_mnemonic(&self, mnemonic: String) -> RpcResult<()>;

    /// Get total sidechain wealth
    #[open_api_method(output_schema(PartialSchema = "BitcoinAmountSchema"))]
    #[method(name = "sidechain_wealth")]
    async fn sidechain_wealth(&self) -> RpcResult<bitcoin::Amount>;

    /// Stop the node
    #[method(name = "stop")]
    async fn stop(&self);

    /// Transfer funds to the specified address
    #[method(name = "transfer")]
    async fn transfer(
        &self,
        dest: Address,
        value_sats: u64,
        fee_sats: u64,
    ) -> RpcResult<Txid>;

    /// Initiate a withdrawal to the specified mainchain address
    #[method(name = "withdraw")]
    async fn withdraw(
        &self,
        #[open_api_method_arg(schema(PartialSchema = "BitcoinAddrSchema"))]
        mainchain_address: bitcoin::Address<
            bitcoin::address::NetworkUnchecked,
        >,
        amount_sats: u64,
        fee_sats: u64,
        mainchain_fee_sats: u64,
    ) -> RpcResult<Txid>;
}
