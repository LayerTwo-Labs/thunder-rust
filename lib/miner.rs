use std::{net::SocketAddr, str::FromStr as _, time::Duration};

use bip300301::{
    bitcoin::{self, hashes::Hash as _},
    Drivechain,
};

use crate::types::{Body, Header};

pub use bip300301::MainClient;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("drivechain error")]
    Drivechain(#[from] bip300301::Error),
    #[error("invalid json: {json}")]
    InvalidJson { json: serde_json::Value },
}

#[derive(Clone)]
pub struct Miner {
    pub drivechain: Drivechain,
    block: Option<(Header, Body)>,
    sidechain_number: u8,
}

impl Miner {
    pub fn new(
        sidechain_number: u8,
        main_addr: SocketAddr,
        user: &str,
        password: &str,
    ) -> Result<Self, Error> {
        let drivechain =
            Drivechain::new(sidechain_number, main_addr, user, password)?;
        Ok(Self {
            drivechain,
            sidechain_number,
            block: None,
        })
    }

    pub async fn generate(&self) -> Result<(), Error> {
        self.drivechain.client.generate(1).await.map_err(|source| {
            bip300301::Error::Jsonrpsee {
                source,
                main_addr: self.drivechain.main_addr,
            }
        })?;
        Ok(())
    }

    pub async fn attempt_bmm(
        &mut self,
        amount: u64,
        height: u32,
        header: Header,
        body: Body,
    ) -> Result<bitcoin::Txid, Error> {
        let str_hash_prev = header.prev_main_hash.to_string();
        let critical_hash: [u8; 32] = header.hash().into();
        let critical_hash = bitcoin::BlockHash::from_byte_array(critical_hash);
        let amount = bitcoin::Amount::from_sat(amount);
        let prev_bytes = &str_hash_prev[str_hash_prev.len() - 8..];
        let value = self
            .drivechain
            .client
            .createbmmcriticaldatatx(
                amount.into(),
                height,
                &critical_hash,
                self.sidechain_number,
                prev_bytes,
            )
            .await
            .map_err(|source| bip300301::Error::Jsonrpsee {
                source,
                main_addr: self.drivechain.main_addr,
            })?;
        let txid = value["txid"]["txid"]
            .as_str()
            .map(|s| s.to_owned())
            .ok_or(Error::InvalidJson { json: value })?;
        let txid =
            bitcoin::Txid::from_str(&txid).map_err(bip300301::Error::from)?;
        tracing::info!("created BMM tx: {txid}");
        assert_eq!(header.merkle_root, body.compute_merkle_root());
        self.block = Some((header, body));
        Ok(txid)
    }

    pub async fn confirm_bmm(
        &mut self,
    ) -> Result<Option<(bitcoin::BlockHash, Header, Body)>, Error> {
        const VERIFY_BMM_POLL_INTERVAL: Duration = Duration::from_secs(15);
        if let Some((header, body)) = self.block.clone() {
            let block_hash = header.hash();
            tracing::trace!(%block_hash, "verifying bmm...");
            let (bmm_verified, main_hash) = self
                .drivechain
                .verify_bmm_next_block(
                    header.prev_main_hash,
                    block_hash.into(),
                    VERIFY_BMM_POLL_INTERVAL,
                )
                .await?;
            if bmm_verified {
                tracing::trace!(%block_hash, "verified bmm");
                self.block = None;
                return Ok(Some((main_hash, header, body)));
            } else {
                tracing::trace!(%block_hash, "bmm verification failed");
                self.block = None;
                return Ok(None);
            }
        }
        Ok(None)
    }
}
