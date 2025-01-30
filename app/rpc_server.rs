use std::net::SocketAddr;

use bitcoin::Amount;
use jsonrpsee::{
    core::{async_trait, RpcResult},
    server::Server,
    types::ErrorObject,
};
use thunder::{
    types::{Address, PointedOutput, Txid, WithdrawalBundle},
    wallet::Balance,
};
use thunder_app_rpc_api::{schema::Peer as RpcPeer, RpcServer};

use crate::app::App;

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
#[async_trait]
impl RpcServer for RpcServerImpl {
    async fn balance(&self) -> RpcResult<Balance> {
        self.app.wallet.get_balance().map_err(custom_err)
    }

    async fn create_deposit(
        &self,
        address: Address,
        value_sats: u64,
        fee_sats: u64,
    ) -> RpcResult<bitcoin::Txid> {
        let app = self.app.clone();
        tokio::task::spawn_blocking(move || {
            app.deposit(
                address,
                bitcoin::Amount::from_sat(value_sats),
                bitcoin::Amount::from_sat(fee_sats),
            )
            .map_err(custom_err)
        })
        .await
        .unwrap()
    }

    async fn connect_peer(&self, addr: SocketAddr) -> RpcResult<()> {
        self.app.node.connect_peer(addr).map_err(custom_err)
    }

    async fn format_deposit_address(
        &self,
        address: Address,
    ) -> RpcResult<String> {
        let deposit_address = address.format_for_deposit();
        Ok(deposit_address)
    }

    async fn generate_mnemonic(&self) -> RpcResult<String> {
        let mnemonic = bip39::Mnemonic::new(
            bip39::MnemonicType::Words12,
            bip39::Language::English,
        );
        Ok(mnemonic.to_string())
    }

    async fn get_block(
        &self,
        block_hash: thunder::types::BlockHash,
    ) -> RpcResult<Option<thunder::types::Block>> {
        let Some(header) = self
            .app
            .node
            .try_get_header(block_hash)
            .map_err(custom_err)?
        else {
            return Ok(None);
        };
        let body = self.app.node.get_body(block_hash).map_err(custom_err)?;
        let block = thunder::types::Block { header, body };
        Ok(Some(block))
    }

    async fn get_bmm_inclusions(
        &self,
        block_hash: thunder::types::BlockHash,
    ) -> RpcResult<Vec<bitcoin::BlockHash>> {
        self.app
            .node
            .get_bmm_inclusions(block_hash)
            .map_err(custom_err)
    }

    async fn get_new_address(&self) -> RpcResult<Address> {
        self.app.wallet.get_new_address().map_err(custom_err)
    }

    async fn get_wallet_addresses(&self) -> RpcResult<Vec<Address>> {
        let addrs = self.app.wallet.get_addresses().map_err(custom_err)?;
        let mut res: Vec<_> = addrs.into_iter().collect();
        res.sort_by_key(|addr| addr.as_base58());
        Ok(res)
    }

    async fn get_wallet_utxos(&self) -> RpcResult<Vec<PointedOutput>> {
        let utxos = self.app.wallet.get_utxos().map_err(custom_err)?;
        let utxos = utxos
            .into_iter()
            .map(|(outpoint, output)| PointedOutput { outpoint, output })
            .collect();
        Ok(utxos)
    }

    async fn getblockcount(&self) -> RpcResult<u32> {
        let height = self.app.node.try_get_height().map_err(custom_err)?;
        let block_count = height.map_or(0, |height| height + 1);
        Ok(block_count)
    }

    async fn latest_failed_withdrawal_bundle_height(
        &self,
    ) -> RpcResult<Option<u32>> {
        let height = self
            .app
            .node
            .get_latest_failed_withdrawal_bundle_height()
            .map_err(custom_err)?;
        Ok(height)
    }

    async fn list_peers(&self) -> RpcResult<Vec<RpcPeer>> {
        let peers = self.app.node.get_active_peers();
        let res: Vec<_> = peers
            .into_iter()
            .map(|(address, state)| RpcPeer {
                address: address.to_string(),
                state: state.to_string(),
            })
            .collect();
        Ok(res)
    }

    async fn list_utxos(&self) -> RpcResult<Vec<PointedOutput>> {
        let utxos = self.app.node.get_all_utxos().map_err(custom_err)?;
        let res = utxos
            .into_iter()
            .map(|(outpoint, output)| PointedOutput { outpoint, output })
            .collect();
        Ok(res)
    }

    async fn mine(&self, fee: Option<u64>) -> RpcResult<()> {
        let fee = fee.map(bitcoin::Amount::from_sat);
        self.app
            .local_pool
            .spawn_pinned({
                let app = self.app.clone();
                move || async move { app.mine(fee).await.map_err(custom_err) }
            })
            .await
            .unwrap()
    }

    async fn pending_withdrawal_bundle(
        &self,
    ) -> RpcResult<Option<WithdrawalBundle>> {
        self.app
            .node
            .get_pending_withdrawal_bundle()
            .map_err(custom_err)
    }

    async fn openapi_schema(&self) -> RpcResult<utoipa::openapi::OpenApi> {
        let res = <thunder_app_rpc_api::RpcDoc as utoipa::OpenApi>::openapi();
        Ok(res)
    }

    async fn remove_from_mempool(&self, txid: Txid) -> RpcResult<()> {
        self.app.node.remove_from_mempool(txid).map_err(custom_err)
    }

    async fn set_seed_from_mnemonic(&self, mnemonic: String) -> RpcResult<()> {
        let mnemonic =
            bip39::Mnemonic::from_phrase(&mnemonic, bip39::Language::English)
                .map_err(custom_err)?;
        let seed = bip39::Seed::new(&mnemonic, "");
        let seed_bytes: [u8; 64] = seed.as_bytes().try_into().map_err(
            |err: <[u8; 64] as TryFrom<&[u8]>>::Error| custom_err(err),
        )?;
        self.app.wallet.set_seed(&seed_bytes).map_err(custom_err)
    }

    async fn sidechain_wealth_sats(&self) -> RpcResult<u64> {
        let sidechain_wealth =
            self.app.node.get_sidechain_wealth().map_err(custom_err)?;
        Ok(sidechain_wealth.to_sat())
    }

    async fn stop(&self) {
        std::process::exit(0);
    }

    async fn transfer(
        &self,
        dest: Address,
        value_sats: u64,
        fee_sats: u64,
    ) -> RpcResult<Txid> {
        let accumulator =
            self.app.node.get_tip_accumulator().map_err(custom_err)?;
        let tx = self
            .app
            .wallet
            .create_transaction(
                &accumulator,
                dest,
                Amount::from_sat(value_sats),
                Amount::from_sat(fee_sats),
            )
            .map_err(custom_err)?;
        let txid = tx.txid();
        self.app.sign_and_send(tx).map_err(custom_err)?;
        Ok(txid)
    }

    async fn withdraw(
        &self,
        mainchain_address: bitcoin::Address<bitcoin::address::NetworkUnchecked>,
        amount_sats: u64,
        fee_sats: u64,
        mainchain_fee_sats: u64,
    ) -> RpcResult<Txid> {
        let accumulator =
            self.app.node.get_tip_accumulator().map_err(custom_err)?;
        let tx = self
            .app
            .wallet
            .create_withdrawal(
                &accumulator,
                mainchain_address,
                Amount::from_sat(amount_sats),
                Amount::from_sat(mainchain_fee_sats),
                Amount::from_sat(fee_sats),
            )
            .map_err(custom_err)?;
        let txid = tx.txid();
        self.app.sign_and_send(tx).map_err(custom_err)?;
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
