use std::net::SocketAddr;

use bip300301::bitcoin;
use jsonrpsee::{
    core::{async_trait, RpcResult},
    server::Server,
    types::ErrorObject,
};
use thunder::{
    node,
    types::{Address, Txid},
    wallet,
};
use thunder_app_rpc_api::RpcServer;

use crate::app::{self, App};

pub struct RpcServerImpl {
    app: App,
}

fn custom_err_msg(err_msg: impl Into<String>) -> ErrorObject<'static> {
    ErrorObject::owned(-1, err_msg.into(), Option::<()>::None)
}

fn custom_err<Error>(error: Error) -> ErrorObject<'static>
where
    anyhow::Error: From<Error>,
{
    let error = anyhow::Error::from(error);
    custom_err_msg(format!("{error:#}"))
}

fn convert_app_err(err: app::Error) -> ErrorObject<'static> {
    let err = anyhow::anyhow!(err);
    tracing::error!("{err:#}");
    custom_err(err)
}

fn convert_node_err(err: node::Error) -> ErrorObject<'static> {
    custom_err(err)
}

fn convert_wallet_err(err: wallet::Error) -> ErrorObject<'static> {
    custom_err(err)
}

#[async_trait]
impl RpcServer for RpcServerImpl {
    async fn balance(&self) -> RpcResult<u64> {
        self.app.wallet.get_balance().map_err(convert_wallet_err)
    }

    async fn connect_peer(&self, addr: SocketAddr) -> RpcResult<()> {
        self.app
            .node
            .connect_peer(addr)
            .await
            .map_err(convert_node_err)
    }

    async fn format_deposit_address(
        &self,
        address: Address,
    ) -> RpcResult<String> {
        let deposit_address = thunder::format_deposit_address(
            node::THIS_SIDECHAIN,
            &address.to_string(),
        );
        Ok(deposit_address)
    }

    async fn generate_mnemonic(&self) -> RpcResult<String> {
        let mnemonic = bip39::Mnemonic::new(
            bip39::MnemonicType::Words12,
            bip39::Language::English,
        );
        Ok(mnemonic.to_string())
    }

    async fn get_new_address(&self) -> RpcResult<Address> {
        self.app
            .wallet
            .get_new_address()
            .map_err(convert_wallet_err)
    }

    async fn getblockcount(&self) -> RpcResult<u32> {
        self.app.node.get_height().map_err(convert_node_err)
    }

    async fn mine(&self, fee: Option<u64>) -> RpcResult<()> {
        let fee = fee.map(bip300301::bitcoin::Amount::from_sat);
        self.app.mine(fee).await.map_err(convert_app_err)
    }

    async fn set_seed_from_mnemonic(&self, mnemonic: String) -> RpcResult<()> {
        let mnemonic =
            bip39::Mnemonic::from_phrase(&mnemonic, bip39::Language::English)
                .map_err(custom_err)?;
        let seed = bip39::Seed::new(&mnemonic, "");
        let seed_bytes: [u8; 64] = seed.as_bytes().try_into().map_err(
            |err: <[u8; 64] as TryFrom<&[u8]>>::Error| custom_err(err),
        )?;
        self.app
            .wallet
            .set_seed(&seed_bytes)
            .map_err(convert_wallet_err)
    }

    async fn sidechain_wealth(&self) -> RpcResult<bitcoin::Amount> {
        self.app
            .node
            .get_sidechain_wealth()
            .map_err(convert_node_err)
    }

    async fn stop(&self) {
        std::process::exit(0);
    }

    async fn withdraw(
        &self,
        mainchain_address: bitcoin::Address<bitcoin::address::NetworkUnchecked>,
        amount_sats: u64,
        fee_sats: u64,
        mainchain_fee_sats: u64,
    ) -> RpcResult<Txid> {
        let accumulator =
            self.app.node.get_accumulator().map_err(convert_node_err)?;
        let tx = self
            .app
            .wallet
            .create_withdrawal(
                &accumulator,
                mainchain_address,
                amount_sats,
                mainchain_fee_sats,
                fee_sats,
            )
            .map_err(convert_wallet_err)?;
        let txid = tx.txid();
        self.app.sign_and_send(tx).map_err(convert_app_err)?;
        Ok(txid)
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
