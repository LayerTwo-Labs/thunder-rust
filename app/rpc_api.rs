//! RPC API

use jsonrpsee::{core::RpcResult, proc_macros::rpc};
use thunder::types::Address;

#[rpc(client, server)]
pub trait Rpc {
    /// Balance in sats
    #[method(name = "balance")]
    async fn balance(&self) -> RpcResult<u64>;

    #[method(name = "format_deposit_address")]
    async fn format_deposit_address(
        &self,
        address: Address,
    ) -> RpcResult<String>;

    #[method(name = "generate_mnemonic")]
    async fn generate_mnemonic(&self) -> RpcResult<String>;

    #[method(name = "get_new_address")]
    async fn get_new_address(&self) -> RpcResult<Address>;

    #[method(name = "getblockcount")]
    async fn getblockcount(&self) -> RpcResult<u32>;

    #[method(name = "mine")]
    async fn mine(&self, fee: Option<u64>) -> RpcResult<()>;

    #[method(name = "set_seed_from_mnemonic")]
    async fn set_seed_from_mnemonic(&self, mnemonic: String) -> RpcResult<()>;

    #[method(name = "stop")]
    async fn stop(&self);
}
