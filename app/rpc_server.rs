use std::{collections::HashSet, net::SocketAddr};

use bitcoin::Amount;
use jsonrpsee::{
    core::{RpcResult, async_trait, middleware::RpcServiceBuilder},
    server::Server,
    types::ErrorObject,
};
use thunder::types::{
    Address, Block, Pointed, PointedOutput, SpentOutput, Txid,
    WithdrawalBundle, net::Peer, wallet::Balance,
};
use thunder_app_rpc_api as rpc_api;
use tower_http::{
    cors::CorsLayer,
    request_id::{
        MakeRequestId, PropagateRequestIdLayer, RequestId, SetRequestIdLayer,
    },
    trace::{DefaultOnFailure, DefaultOnResponse, TraceLayer},
};

use crate::app::App;

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

#[derive(Clone)]
#[repr(transparent)]
pub struct RpcServerImpl<const ENABLE_PRIVATE_API: bool> {
    app: App,
}

pub struct PrivateOnlyRpcServerImpl;

#[async_trait]
impl rpc_api::open_api::RpcServer for PrivateOnlyRpcServerImpl {
    async fn openapi_schema(&self) -> RpcResult<utoipa::openapi::OpenApi> {
        use utoipa::OpenApi as _;
        let mut res = rpc_api::open_api::RpcDoc::openapi();
        res.merge(rpc_api::node::PrivateRpcDoc::openapi());
        res.merge(rpc_api::wallet::RpcDoc::openapi());
        Ok(res)
    }
}

#[async_trait]
impl rpc_api::open_api::RpcServer for RpcServerImpl<false> {
    async fn openapi_schema(&self) -> RpcResult<utoipa::openapi::OpenApi> {
        use utoipa::OpenApi as _;
        let mut res = rpc_api::open_api::RpcDoc::openapi();
        res.merge(rpc_api::node::RpcDoc::openapi());
        Ok(res)
    }
}

#[async_trait]
impl rpc_api::open_api::RpcServer for RpcServerImpl<true> {
    async fn openapi_schema(&self) -> RpcResult<utoipa::openapi::OpenApi> {
        use utoipa::OpenApi as _;
        let mut res = rpc_api::open_api::RpcDoc::openapi();
        res.merge(rpc_api::node::PrivateRpcDoc::openapi());
        res.merge(rpc_api::node::RpcDoc::openapi());
        res.merge(rpc_api::wallet::RpcDoc::openapi());
        Ok(res)
    }
}

#[async_trait]
impl rpc_api::node::PrivateRpcServer for RpcServerImpl<true> {
    async fn connect_peer(&self, addr: SocketAddr) -> RpcResult<()> {
        self.app.node.connect_peer(addr).map_err(custom_err)
    }

    async fn forget_peer(&self, addr: SocketAddr) -> RpcResult<()> {
        match self.app.node.forget_peer(&addr) {
            Ok(_) => Ok(()),
            Err(err) => Err(custom_err(err)),
        }
    }

    async fn invalidate_block(
        &self,
        block_hash: thunder::types::BlockHash,
    ) -> RpcResult<()> {
        self.app
            .node
            .invalidate_block(block_hash)
            .map_err(custom_err)
    }

    async fn remove_from_mempool(&self, txid: Txid) -> RpcResult<()> {
        self.app.node.remove_from_mempool(txid).map_err(custom_err)
    }

    async fn stop(&self) {
        std::process::exit(0);
    }
}

#[async_trait]
impl<const ENABLE_PRIVATE_API: bool> rpc_api::node::RpcServer
    for RpcServerImpl<ENABLE_PRIVATE_API>
{
    async fn connect_block(
        &self,
        block: Block,
        main_block_hash: bitcoin::BlockHash,
    ) -> RpcResult<bool> {
        self.app
            .local_pool
            .spawn_pinned({
                let app = self.app.clone();
                move || async move {
                    app.connect_block(block, main_block_hash)
                        .await
                        .map_err(custom_err)
                }
            })
            .await
            .unwrap()
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

    async fn get_best_sidechain_block_hash(
        &self,
    ) -> RpcResult<Option<thunder::types::BlockHash>> {
        self.app.node.try_get_tip().map_err(custom_err)
    }

    async fn get_best_mainchain_block_hash(
        &self,
    ) -> RpcResult<Option<bitcoin::BlockHash>> {
        let Some(sidechain_hash) =
            self.app.node.try_get_tip().map_err(custom_err)?
        else {
            // No sidechain tip, so no best mainchain block hash.
            return Ok(None);
        };
        let block_hash = self
            .app
            .node
            .get_best_main_verification(sidechain_hash)
            .map_err(custom_err)?;
        Ok(Some(block_hash))
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

    async fn get_stxos(
        &self,
        addresses: HashSet<Address>,
    ) -> RpcResult<Vec<Pointed<SpentOutput>>> {
        let res = self
            .app
            .node
            .get_stxos_by_addresses(&addresses)
            .map_err(custom_err)?
            .into_iter()
            .map(|(outpoint, output)| Pointed { outpoint, output })
            .collect();
        Ok(res)
    }

    async fn get_transaction(
        &self,
        txid: Txid,
    ) -> RpcResult<Option<rpc_api::node::GetTransactionResponse>> {
        let res =
            self.app
                .node
                .try_get_transaction(txid)
                .map_err(custom_err)?
                .map(|(tx, block_hash)| {
                    rpc_api::node::GetTransactionResponse { tx, block_hash }
                });
        Ok(res)
    }

    async fn get_utxos(
        &self,
        addresses: HashSet<Address>,
    ) -> RpcResult<Vec<PointedOutput>> {
        let res = self
            .app
            .node
            .get_utxos_by_addresses(&addresses)
            .map_err(custom_err)?
            .into_iter()
            .map(|(outpoint, output)| PointedOutput { outpoint, output })
            .collect();
        Ok(res)
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

    async fn list_peers(&self) -> RpcResult<Vec<Peer>> {
        let peers = self.app.node.get_active_peers();
        Ok(peers)
    }

    async fn list_utxos(&self) -> RpcResult<Vec<PointedOutput>> {
        let utxos = self.app.node.get_all_utxos().map_err(custom_err)?;
        let res = utxos
            .into_iter()
            .map(|(outpoint, output)| PointedOutput { outpoint, output })
            .collect();
        Ok(res)
    }

    async fn pending_withdrawal_bundle(
        &self,
    ) -> RpcResult<Option<WithdrawalBundle>> {
        self.app
            .node
            .try_get_pending_withdrawal_bundle()
            .map_err(custom_err)
    }

    async fn sidechain_wealth_sats(&self) -> RpcResult<u64> {
        let sidechain_wealth =
            self.app.node.get_sidechain_wealth().map_err(custom_err)?;
        Ok(sidechain_wealth.to_sat())
    }

    async fn submit_transaction(
        &self,
        mut transaction: thunder::types::AuthorizedTransaction,
    ) -> RpcResult<Txid> {
        let () = self
            .app
            .submit_transaction(&mut transaction)
            .map_err(custom_err)?;
        Ok(transaction.transaction.txid())
    }
}

#[async_trait]
impl rpc_api::wallet::RpcServer for RpcServerImpl<true> {
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
            app.deposit_blocking(
                address,
                bitcoin::Amount::from_sat(value_sats),
                bitcoin::Amount::from_sat(fee_sats),
            )
            .map_err(custom_err)
        })
        .await
        .unwrap()
    }

    async fn create_transfer(
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
        let () = self.app.sign_and_send(tx).map_err(custom_err)?;
        Ok(txid)
    }

    async fn create_withdrawal(
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
        let () = self.app.sign_and_send(tx).map_err(custom_err)?;
        Ok(txid)
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

    async fn get_block_template(
        &self,
    ) -> RpcResult<rpc_api::wallet::GetBlockTemplateResponse> {
        let template = self
            .app
            .local_pool
            .spawn_pinned({
                let app = self.app.clone();
                move || async move {
                    app.get_block_template().await.map_err(custom_err)
                }
            })
            .await
            .unwrap()?;
        Ok(rpc_api::wallet::GetBlockTemplateResponse {
            critical_hash: template.header.hash(),
            block: Block {
                header: template.header,
                body: template.body,
            },
            fees_sats: template.fees.to_sat(),
        })
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

    async fn sign_transaction(
        &self,
        transaction: thunder::types::Transaction,
        broadcast: Option<bool>,
    ) -> RpcResult<thunder::types::AuthorizedTransaction> {
        let mut authorized =
            self.app.wallet.authorize(transaction).map_err(custom_err)?;
        if let Some(true) = broadcast {
            let () = self
                .app
                .submit_transaction(&mut authorized)
                .map_err(custom_err)?;
        }
        Ok(authorized)
    }
}

#[derive(Clone, Debug)]
struct RequestIdMaker;

impl MakeRequestId for RequestIdMaker {
    fn make_request_id<B>(
        &mut self,
        _: &http::Request<B>,
    ) -> Option<RequestId> {
        use uuid::Uuid;
        // the 'simple' format renders the UUID with no dashes, which
        // makes for easier copy/pasting.
        let id = Uuid::new_v4();
        let id = id.as_simple();
        let id = format!("req_{id}"); // prefix all IDs with "req_", to make them easier to identify

        let Ok(header_value) = http::HeaderValue::from_str(&id) else {
            return None;
        };

        Some(RequestId::new(header_value))
    }
}

pub struct ServerAddesses {
    pub _rpc_addr: SocketAddr,
    pub _private_rpc_addr: SocketAddr,
}

pub async fn run_server(
    app: App,
    private_rpc_addr: SocketAddr,
    rpc_addr: SocketAddr,
) -> anyhow::Result<ServerAddesses> {
    const REQUEST_ID_HEADER: &str = "x-request-id";

    // Ordering here matters! Order here is from official docs on request IDs tracings
    // https://docs.rs/tower-http/latest/tower_http/request_id/index.html#using-trace
    let tracer = || {
        tower::ServiceBuilder::new()
            .layer(SetRequestIdLayer::new(
                http::HeaderName::from_static(REQUEST_ID_HEADER),
                RequestIdMaker,
            ))
            .layer(
                TraceLayer::new_for_http()
                    .make_span_with(move |request: &http::Request<_>| {
                        let request_id = request
                            .headers()
                            .get(http::HeaderName::from_static(
                                REQUEST_ID_HEADER,
                            ))
                            .and_then(|h| h.to_str().ok())
                            .filter(|s| !s.is_empty());

                        tracing::span!(
                            tracing::Level::DEBUG,
                            "request",
                            method = %request.method(),
                            uri = %request.uri(),
                            request_id , // this is needed for the record call below to work
                        )
                    })
                    .on_request(())
                    .on_eos(())
                    .on_response(
                        DefaultOnResponse::new().level(tracing::Level::INFO),
                    )
                    .on_failure(
                        DefaultOnFailure::new().level(tracing::Level::ERROR),
                    ),
            )
            .layer(PropagateRequestIdLayer::new(http::HeaderName::from_static(
                REQUEST_ID_HEADER,
            )))
            .into_inner()
    };

    let http_middleware = || {
        tower::ServiceBuilder::new()
            .layer(tracer())
            .layer(CorsLayer::permissive())
    };
    let rpc_middleware = || RpcServiceBuilder::new().rpc_logger(1024);

    let server = Server::builder()
        .set_http_middleware(http_middleware())
        .set_rpc_middleware(rpc_middleware())
        .build(rpc_addr)
        .await?;
    let rpc_server_addr = server.local_addr()?;

    let (_task_handle, server_addrs) = if private_rpc_addr != rpc_addr {
        let private_rpc_server = Server::builder()
            .set_http_middleware(http_middleware())
            .set_rpc_middleware(rpc_middleware())
            .build(private_rpc_addr)
            .await?;
        let private_rpc_server_addr = private_rpc_server.local_addr()?;

        let rpc_server_handle = {
            let rpc_server_impl = RpcServerImpl::<false> { app: app.clone() };
            let mut rpc_module =
                rpc_api::open_api::RpcServer::into_rpc(rpc_server_impl.clone());
            rpc_module
                .merge(rpc_api::node::RpcServer::into_rpc(rpc_server_impl))?;
            server.start(rpc_module)
        };
        let private_only_rpc_server_handle = {
            let rpc_server_impl = RpcServerImpl::<true> { app };
            let mut rpc_module = rpc_api::open_api::RpcServer::into_rpc(
                PrivateOnlyRpcServerImpl,
            );
            rpc_module.merge(rpc_api::node::PrivateRpcServer::into_rpc(
                rpc_server_impl.clone(),
            ))?;
            rpc_module
                .merge(rpc_api::wallet::RpcServer::into_rpc(rpc_server_impl))?;
            private_rpc_server.start(rpc_module)
        };
        let server_addrs = ServerAddesses {
            _rpc_addr: rpc_server_addr,
            _private_rpc_addr: private_rpc_server_addr,
        };
        let task_handle = tokio::spawn(async {
            tokio::select! {
                () = rpc_server_handle.stopped() => (),
                () = private_only_rpc_server_handle.stopped() => (),
            }
        });
        (task_handle, server_addrs)
    } else {
        let rpc_server_impl = RpcServerImpl::<true> { app };
        let mut rpc_module =
            rpc_api::open_api::RpcServer::into_rpc(rpc_server_impl.clone());
        rpc_module.merge(rpc_api::node::PrivateRpcServer::into_rpc(
            rpc_server_impl.clone(),
        ))?;
        rpc_module.merge(rpc_api::node::RpcServer::into_rpc(
            rpc_server_impl.clone(),
        ))?;
        rpc_module
            .merge(rpc_api::wallet::RpcServer::into_rpc(rpc_server_impl))?;

        let server_addrs = ServerAddesses {
            _rpc_addr: rpc_server_addr,
            _private_rpc_addr: rpc_server_addr,
        };
        let handle = server.start(rpc_module);
        let task_handle = tokio::spawn(handle.stopped());
        (task_handle, server_addrs)
    };
    Ok(server_addrs)
}
