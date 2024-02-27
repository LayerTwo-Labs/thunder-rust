use std::net::SocketAddr;

use jsonrpsee::{
    core::{async_trait, RpcResult},
    server::Server,
    types::ErrorObject,
};
use thunder::node;
use thunder_app_rpc_api::RpcServer;

use crate::app::App;

pub struct RpcServerImpl {
    app: App,
}

fn custom_err(err_msg: impl Into<String>) -> ErrorObject<'static> {
    ErrorObject::owned(-1, err_msg.into(), Option::<()>::None)
}

fn convert_node_err(err: node::Error) -> ErrorObject<'static> {
    custom_err(format!("{:#}", anyhow::Error::from(err)))
}

#[async_trait]
impl RpcServer for RpcServerImpl {
    async fn stop(&self) {
        std::process::exit(0);
    }
    async fn getblockcount(&self) -> RpcResult<u32> {
        self.app.node.get_height().map_err(convert_node_err)
    }
}

pub async fn run_server(
    app: App,
    rpc_addr: SocketAddr,
) -> anyhow::Result<SocketAddr> {
    let server = Server::builder().build(rpc_addr).await?;

    let addr = server.local_addr()?;
    let handle = server.start(RpcServerImpl { app }.into_rpc());

    // In this example we don't care about doing shutdown so let's it run forever.
    // You may use the `ServerHandle` to shut it down or manage it yourself.
    tokio::spawn(handle.stopped());

    Ok(addr)
}
