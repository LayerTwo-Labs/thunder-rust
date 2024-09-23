use bitcoin::hashes::Hash as _;
use futures::TryStreamExt;

use crate::types::{
    proto::{self, mainchain},
    Body, Header,
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("drivechain error")]
    CusfMainchain(#[from] mainchain::Error),
    #[error("invalid json: {json}")]
    InvalidJson { json: serde_json::Value },
}

#[derive(Clone)]
pub struct Miner<MainchainTransport = tonic::transport::Channel> {
    pub cusf_mainchain: mainchain::Client<MainchainTransport>,
    block: Option<(Header, Body)>,
}

impl<MainchainTransport> Miner<MainchainTransport> {
    pub fn new(
        cusf_mainchain: mainchain::Client<MainchainTransport>,
    ) -> Result<Self, Error> {
        Ok(Self {
            cusf_mainchain,
            block: None,
        })
    }
}

impl<MainchainTransport> Miner<MainchainTransport>
where
    MainchainTransport: proto::Transport,
{
    pub async fn generate(&mut self) -> Result<(), Error> {
        let () = self.cusf_mainchain.generate_blocks(1).await?;
        Ok(())
    }

    pub async fn attempt_bmm(
        &mut self,
        amount: u64,
        height: u32,
        header: Header,
        body: Body,
    ) -> Result<bitcoin::Txid, Error> {
        let critical_hash: [u8; 32] = header.hash().into();
        let critical_hash = bitcoin::BlockHash::from_byte_array(critical_hash);
        let prev_main_hash: [u8; 32] = header.hash().into();
        let prev_bytes: [u8; 4] = *prev_main_hash.last_chunk::<4>().unwrap();
        let txid = self
            .cusf_mainchain
            .create_bmm_critical_data_tx(
                amount,
                height,
                critical_hash,
                prev_bytes,
            )
            .await?;
        tracing::info!("created BMM tx: {txid}");
        assert_eq!(header.merkle_root, body.compute_merkle_root());
        self.block = Some((header, body));
        Ok(txid)
    }

    pub async fn confirm_bmm(
        &mut self,
    ) -> Result<Option<(bitcoin::BlockHash, Header, Body)>, Error> {
        use mainchain::Event;

        let Some((header, body)) = self.block.clone() else {
            return Ok(None);
        };
        let block_hash = header.hash();
        tracing::trace!(%block_hash, "verifying bmm...");
        let mut events_stream = self.cusf_mainchain.subscribe_events().await?;
        if let Some(event) = events_stream.try_next().await? {
            match event {
                Event::ConnectBlock {
                    header_info,
                    block_info,
                } => {
                    if block_info.bmm_commitment == Some(block_hash) {
                        tracing::trace!(%block_hash, "verified bmm");
                        self.block = None;
                        return Ok(Some((
                            header_info.block_hash,
                            header,
                            body,
                        )));
                    }
                }
                Event::DisconnectBlock { .. } => (),
            }
        };
        tracing::trace!(%block_hash, "bmm verification failed");
        self.block = None;
        Ok(None)
    }
}
