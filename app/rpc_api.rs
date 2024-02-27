//! RPC API

use jsonrpsee::{core::RpcResult, proc_macros::rpc};

#[rpc(server)]
pub trait Rpc {
    #[method(name = "stop")]
    async fn stop(&self);
    #[method(name = "getblockcount")]
    async fn getblockcount(&self) -> RpcResult<u32>;
}
