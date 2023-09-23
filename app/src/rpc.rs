use std::net::SocketAddr;

use jsonrpsee::core::async_trait;
use jsonrpsee::proc_macros::rpc;
use jsonrpsee::server::Server;
use thunder::node::Node;

#[rpc(server)]
pub trait Rpc {
    #[method(name = "stop")]
    async fn stop(&self);
    #[method(name = "getblockcount")]
    async fn getblockcount(&self) -> u32;
}

pub struct RpcServerImpl {
    node: Node,
}

#[async_trait]
impl RpcServer for RpcServerImpl {
    async fn stop(&self) {
        std::process::exit(0);
    }
    async fn getblockcount(&self) -> u32 {
        self.node.get_height().unwrap_or(0)
    }
}

pub async fn run_server(node: Node, rpc_addr: SocketAddr) -> anyhow::Result<SocketAddr> {
    let server = Server::builder().build(rpc_addr).await?;

    let addr = server.local_addr()?;
    let handle = server.start(RpcServerImpl { node }.into_rpc());

    // In this example we don't care about doing shutdown so let's it run forever.
    // You may use the `ServerHandle` to shut it down or manage it yourself.
    tokio::spawn(handle.stopped());

    Ok(addr)
}
