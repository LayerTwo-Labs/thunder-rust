//! RPC API

use std::net::SocketAddr;

use bip300301::bitcoin;
use jsonrpsee::{core::RpcResult, proc_macros::rpc};
use thunder::types::{Address, Txid};

#[rpc(client, server)]
pub trait Rpc {
    /// Balance in sats
    #[method(name = "balance")]
    async fn balance(&self) -> RpcResult<u64>;

    /// Connect to a peer
    #[method(name = "connect_peer")]
    async fn connect_peer(&self, addr: SocketAddr) -> RpcResult<()>;

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

    #[method(name = "sidechain_wealth")]
    async fn sidechain_wealth(&self) -> RpcResult<bitcoin::Amount>;

    #[method(name = "stop")]
    async fn stop(&self);

    #[method(name = "withdraw")]
    async fn withdraw(
        &self,
        mainchain_address: bitcoin::Address<bitcoin::address::NetworkUnchecked>,
        amount_sats: u64,
        fee_sats: u64,
        mainchain_fee_sats: u64,
    ) -> RpcResult<Txid>;
}
