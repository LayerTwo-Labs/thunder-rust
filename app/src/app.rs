use std::collections::HashMap;

use crate::cli::Config;

use lib::{
    bip300301::{self, bitcoin, jsonrpsee, MainClient},
    format_deposit_address,
    miner::{self, Miner},
    node::{self, Node, THIS_SIDECHAIN},
    types::{self, OutPoint, Output, Transaction},
    wallet::{self, Wallet},
};
pub use thunder as lib;

pub struct App {
    pub node: Node,
    pub wallet: Wallet,
    pub miner: Miner,
    pub utxos: HashMap<OutPoint, Output>,
    pub transaction: Transaction,
    runtime: tokio::runtime::Runtime,
}

impl App {
    pub fn new(config: &Config) -> Result<Self, Error> {
        // Node launches some tokio tasks for p2p networking, that is why we need a tokio runtime
        // here.
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;
        let wallet = Wallet::new(&config.datadir.join("wallet.mdb"))?;
        let miner = Miner::new(
            THIS_SIDECHAIN,
            config.main_addr,
            &config.main_user,
            &config.main_password,
        )?;
        let node = runtime.block_on(async {
            let node = match Node::new(
                &config.datadir,
                config.net_addr,
                config.main_addr,
                &config.main_user,
                &config.main_password,
            ) {
                Ok(node) => node,
                Err(err) => return Err(err),
            };
            Ok(node)
        })?;
        runtime.block_on(async {
            crate::rpc::run_server(node.clone()).await.unwrap();
        });
        let utxos = {
            let mut utxos = wallet.get_utxos()?;
            let transactions = node.get_all_transactions()?;
            for transaction in &transactions {
                for input in &transaction.transaction.inputs {
                    utxos.remove(input);
                }
            }
            utxos
        };
        Ok(Self {
            node,
            wallet,
            miner,
            utxos,
            transaction: Transaction {
                inputs: vec![],
                outputs: vec![],
            },
            runtime,
        })
    }

    pub fn sign_and_send(&mut self) -> Result<(), Error> {
        let authorized_transaction = self.wallet.authorize(self.transaction.clone())?;
        self.runtime
            .block_on(self.node.submit_transaction(&authorized_transaction))?;
        self.transaction = Transaction {
            inputs: vec![],
            outputs: vec![],
        };
        self.update_utxos()?;
        Ok(())
    }

    pub fn get_new_main_address(
        &self,
    ) -> Result<bitcoin::Address<bitcoin::address::NetworkChecked>, Error> {
        let address = self
            .runtime
            .block_on(self.miner.drivechain.client.getnewaddress("", "legacy"))?;
        let address: bitcoin::Address<bitcoin::address::NetworkChecked> =
            address.require_network(bitcoin::Network::Regtest).unwrap();
        Ok(address)
    }

    const EMPTY_BLOCK_BMM_BRIBE: u64 = 1000;
    pub fn mine(&mut self) -> Result<(), Error> {
        self.runtime.block_on(async {
            const NUM_TRANSACTIONS: usize = 1000;
            let (transactions, fee) = self.node.get_transactions(NUM_TRANSACTIONS)?;
            let coinbase = match fee {
                0 => vec![],
                _ => vec![types::Output {
                    address: self.wallet.get_new_address()?,
                    content: types::Content::Value(fee),
                }],
            };
            let body = types::Body::new(transactions, coinbase);
            let prev_side_hash = self.node.get_best_hash()?;
            let prev_main_hash = self.miner.drivechain.get_mainchain_tip().await?;
            let header = types::Header {
                merkle_root: body.compute_merkle_root(),
                prev_side_hash,
                prev_main_hash,
            };
            let bribe = if fee > 0 {
                fee
            } else {
                Self::EMPTY_BLOCK_BMM_BRIBE
            };
            let bribe = bitcoin::Amount::from_sat(bribe);
            self.miner
                .attempt_bmm(bribe.to_sat(), 0, header, body)
                .await?;
            self.miner.generate().await?;
            if let Ok(Some((header, body))) = self.miner.confirm_bmm().await {
                self.node.submit_block(&header, &body).await?;
            }

            Ok::<(), Error>(())
        })?;
        self.update_wallet()?;
        self.update_utxos()?;
        Ok(())
    }

    fn update_wallet(&mut self) -> Result<(), Error> {
        let addresses = self.wallet.get_addresses()?;
        let utxos = self.node.get_utxos_by_addresses(&addresses)?;
        let outpoints: Vec<_> = self.wallet.get_utxos()?.into_keys().collect();
        let spent = self.node.get_spent_utxos(&outpoints)?;
        self.wallet.put_utxos(&utxos)?;
        self.wallet.delete_utxos(&spent)?;
        Ok(())
    }

    fn update_utxos(&mut self) -> Result<(), Error> {
        let mut utxos = self.wallet.get_utxos()?;
        let transactions = self.node.get_all_transactions()?;
        for transaction in &transactions {
            for input in &transaction.transaction.inputs {
                utxos.remove(input);
            }
        }
        self.utxos = utxos;
        Ok(())
    }

    pub fn deposit(&mut self, amount: bitcoin::Amount, fee: bitcoin::Amount) -> Result<(), Error> {
        self.runtime.block_on(async {
            let address = self.wallet.get_new_address()?;
            let address = format_deposit_address(THIS_SIDECHAIN, &format!("{address}"));
            self.miner
                .drivechain
                .client
                .createsidechaindeposit(THIS_SIDECHAIN, &address, amount.into(), fee.into())
                .await?;
            Ok(())
        })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("node error")]
    Node(#[from] node::Error),
    #[error("wallet error")]
    Wallet(#[from] wallet::Error),
    #[error("miner error")]
    Miner(#[from] miner::Error),
    #[error("drivechain error")]
    Drivechain(#[from] bip300301::Error),
    #[error("io error")]
    Io(#[from] std::io::Error),
    #[error("jsonrpsee error")]
    Jsonrpsee(#[from] jsonrpsee::core::Error),
}
