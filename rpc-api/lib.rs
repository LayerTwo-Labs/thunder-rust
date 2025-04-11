//! RPC API

use std::net::SocketAddr;

use jsonrpsee::{core::RpcResult, proc_macros::rpc};
use l2l_openapi::open_api;
use thunder::{
    net::Peer,
    types::{
        MerkleRoot, OutPoint, Output, OutputContent, PointedOutput,
        ShieldedAddress, TransparentAddress, Txid, WithdrawalBundle,
        schema as thunder_schema,
    },
    wallet::Balance,
};

mod schema;

#[open_api(ref_schemas[
    MerkleRoot, OutPoint, Output, OutputContent, TransparentAddress, Txid,
    schema::BitcoinTxid, thunder_schema::BitcoinAddr,
    thunder_schema::BitcoinOutPoint,
])]
#[rpc(client, server)]
pub trait Rpc {
    /// Get balance in sats
    #[open_api_method(output_schema(ToSchema))]
    #[method(name = "balance")]
    async fn balance(&self) -> RpcResult<Balance>;

    /// Connect to a peer
    #[open_api_method(output_schema(ToSchema))]
    #[method(name = "connect_peer")]
    async fn connect_peer(
        &self,
        #[open_api_method_arg(schema(
            PartialSchema = "thunder_schema::SocketAddr"
        ))]
        addr: SocketAddr,
    ) -> RpcResult<()>;

    /// Deposit to address
    #[open_api_method(output_schema(PartialSchema = "schema::BitcoinTxid"))]
    #[method(name = "create_deposit")]
    async fn create_deposit(
        &self,
        address: TransparentAddress,
        value_sats: u64,
        fee_sats: u64,
    ) -> RpcResult<bitcoin::Txid>;

    /// Format a deposit address
    #[method(name = "format_deposit_address")]
    async fn format_deposit_address(
        &self,
        address: TransparentAddress,
    ) -> RpcResult<String>;

    /// Generate a mnemonic seed phrase
    #[method(name = "generate_mnemonic")]
    async fn generate_mnemonic(&self) -> RpcResult<String>;

    /// Get the block with specified block hash, if it exists
    #[method(name = "get_block")]
    async fn get_block(
        &self,
        block_hash: thunder::types::BlockHash,
    ) -> RpcResult<Option<thunder::types::Block>>;

    /// Get mainchain blocks that commit to a specified block hash
    #[open_api_method(output_schema(
        PartialSchema = "thunder_schema::BitcoinBlockHash"
    ))]
    #[method(name = "get_bmm_inclusions")]
    async fn get_bmm_inclusions(
        &self,
        block_hash: thunder::types::BlockHash,
    ) -> RpcResult<Vec<bitcoin::BlockHash>>;

    /// Get the best mainchain block hash known by Thunder
    #[open_api_method(output_schema(
        PartialSchema = "schema::Optional<thunder_schema::BitcoinBlockHash>"
    ))]
    #[method(name = "get_best_mainchain_block_hash")]
    async fn get_best_mainchain_block_hash(
        &self,
    ) -> RpcResult<Option<bitcoin::BlockHash>>;

    /// Get the best sidechain block hash known by Thunder
    #[open_api_method(output_schema(
        PartialSchema = "schema::Optional<thunder::types::BlockHash>"
    ))]
    #[method(name = "get_best_sidechain_block_hash")]
    async fn get_best_sidechain_block_hash(
        &self,
    ) -> RpcResult<Option<thunder::types::BlockHash>>;

    /// Get a new shielded address
    #[method(name = "get_new_shielded_address")]
    async fn get_new_shielded_address(&self) -> RpcResult<ShieldedAddress>;

    /// Get a new transparent address
    #[method(name = "get_new_transparent_address")]
    async fn get_new_transparent_address(
        &self,
    ) -> RpcResult<TransparentAddress>;

    /// Get shielded wallet addresses, sorted by bech32m encoding
    #[method(name = "get_shielded_wallet_addresses")]
    async fn get_shielded_wallet_addresses(
        &self,
    ) -> RpcResult<Vec<thunder::types::orchard::Address>>;

    /// Get transparent wallet addresses, sorted by base58 encoding
    #[method(name = "get_transparent_wallet_addresses")]
    async fn get_transparent_wallet_addresses(
        &self,
    ) -> RpcResult<Vec<TransparentAddress>>;

    /// Get wallet UTXOs
    #[method(name = "get_wallet_utxos")]
    async fn get_wallet_utxos(&self) -> RpcResult<Vec<PointedOutput>>;

    /// Get the current block count
    #[method(name = "getblockcount")]
    async fn getblockcount(&self) -> RpcResult<u32>;

    /// Get the height of the latest failed withdrawal bundle
    #[method(name = "latest_failed_withdrawal_bundle_height")]
    async fn latest_failed_withdrawal_bundle_height(
        &self,
    ) -> RpcResult<Option<u32>>;

    /// List peers
    #[method(name = "list_peers")]
    async fn list_peers(&self) -> RpcResult<Vec<Peer>>;

    /// List all UTXOs
    #[method(name = "list_utxos")]
    async fn list_utxos(&self) -> RpcResult<Vec<PointedOutput>>;

    /// Attempt to mine a sidechain block
    #[open_api_method(output_schema(ToSchema))]
    #[method(name = "mine")]
    async fn mine(&self, fee: Option<u64>) -> RpcResult<()>;

    /// Get OpenAPI schema
    #[open_api_method(output_schema(PartialSchema = "schema::OpenApi"))]
    #[method(name = "openapi_schema")]
    async fn openapi_schema(&self) -> RpcResult<utoipa::openapi::OpenApi>;

    /// Get pending withdrawal bundle
    #[open_api_method(output_schema(ToSchema))]
    #[method(name = "pending_withdrawal_bundle")]
    async fn pending_withdrawal_bundle(
        &self,
    ) -> RpcResult<Option<WithdrawalBundle>>;

    /// Remove a tx from the mempool
    #[open_api_method(output_schema(ToSchema))]
    #[method(name = "remove_from_mempool")]
    async fn remove_from_mempool(&self, txid: Txid) -> RpcResult<()>;

    /// Set the wallet seed from a mnemonic seed phrase
    #[open_api_method(output_schema(ToSchema))]
    #[method(name = "set_seed_from_mnemonic")]
    async fn set_seed_from_mnemonic(&self, mnemonic: String) -> RpcResult<()>;

    /// Shield transparent funds
    #[method(name = "shield")]
    async fn shield(&self, value_sats: u64, fee_sats: u64) -> RpcResult<Txid>;

    /// Transfer shielded funds to the specified address
    #[method(name = "shielded_transfer")]
    async fn shielded_transfer(
        &self,
        dest: ShieldedAddress,
        value_sats: u64,
        fee_sats: u64,
    ) -> RpcResult<Txid>;

    /// Get total sidechain wealth
    #[method(name = "sidechain_wealth")]
    async fn sidechain_wealth_sats(&self) -> RpcResult<u64>;

    /// Stop the node
    #[method(name = "stop")]
    async fn stop(&self);

    /// Transfer transparent funds to the specified address
    #[method(name = "transparent_transfer")]
    async fn transparent_transfer(
        &self,
        dest: TransparentAddress,
        value_sats: u64,
        fee_sats: u64,
    ) -> RpcResult<Txid>;

    /// Unshield shielded funds
    #[method(name = "unshield")]
    async fn unshield(&self, value_sats: u64, fee_sats: u64)
    -> RpcResult<Txid>;

    /// Initiate a withdrawal to the specified mainchain address
    #[method(name = "withdraw")]
    async fn withdraw(
        &self,
        #[open_api_method_arg(schema(
            PartialSchema = "thunder::types::schema::BitcoinAddr"
        ))]
        mainchain_address: bitcoin::Address<
            bitcoin::address::NetworkUnchecked,
        >,
        amount_sats: u64,
        fee_sats: u64,
        mainchain_fee_sats: u64,
    ) -> RpcResult<Txid>;
}
