use futures::TryStreamExt;

use crate::types::{
    Body, Header,
    proto::{self, mainchain},
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("CUSF mainchain proto error")]
    CusfMainchain(#[from] proto::Error),
    #[error("invalid json: {json}")]
    InvalidJson { json: serde_json::Value },
}

#[derive(Clone)]
pub struct Miner<MainchainTransport = tonic::transport::Channel> {
    pub cusf_mainchain: mainchain::ValidatorClient<MainchainTransport>,
    pub cusf_mainchain_wallet: mainchain::WalletClient<MainchainTransport>,
    block: Option<(Header, Body)>,
}

impl<MainchainTransport> Miner<MainchainTransport> {
    pub fn new(
        cusf_mainchain: mainchain::ValidatorClient<MainchainTransport>,
        cusf_mainchain_wallet: mainchain::WalletClient<MainchainTransport>,
    ) -> Result<Self, Error> {
        Ok(Self {
            cusf_mainchain,
            cusf_mainchain_wallet,
            block: None,
        })
    }
}

impl<MainchainTransport> Miner<MainchainTransport>
where
    MainchainTransport: proto::Transport,
{
    pub async fn generate(&mut self) -> Result<(), Error> {
        let () = self.cusf_mainchain_wallet.generate_blocks(1).await?;
        Ok(())
    }

    pub async fn attempt_bmm(
        &mut self,
        amount: u64,
        height: u32,
        header: Header,
        body: Body,
    ) -> Result<bitcoin::Txid, Error> {
        let critical_hash = header.hash().0;
        let txid = self
            .cusf_mainchain_wallet
            .create_bmm_critical_data_tx(
                amount,
                height,
                critical_hash,
                header.prev_main_hash,
            )
            .await?;
        tracing::info!("attempt BMM: created TX: {txid}");
        assert_eq!(header.merkle_root, body.compute_merkle_root());
        self.block = Some((header, body));
        Ok(txid)
    }

    // Wait for a block to be connected that contains our BMM request.
    pub async fn confirm_bmm(
        &mut self,
    ) -> Result<Option<(bitcoin::BlockHash, Header, Body)>, Error> {
        use mainchain::Event;

        let Some((header, body)) = self.block.clone() else {
            return Ok(None);
        };
        let block_hash = header.hash();
        tracing::debug!(%block_hash, "confirm BMM: verifying...");

        let mut events_stream = self.cusf_mainchain.subscribe_events().await?;
        if let Some(event) = events_stream.try_next().await? {
            match event {
                // Our BMM request made it into a block!
                Event::ConnectBlock {
                    header_info,
                    block_info,
                } => {
                    if let Some(bmm_commitment) = block_info.bmm_commitment
                        && bmm_commitment == block_hash
                    {
                        tracing::debug!(
                                side_hash = %block_hash,
                                main_height = header_info.height,
                                main_hash = %header_info.block_hash,
                                bmm_commitment = %bmm_commitment,
                                "confirm BMM: verified!"
                        );
                        self.block = None;
                        return Ok(Some((
                            header_info.block_hash,
                            header,
                            body,
                        )));
                    } else {
                        tracing::warn!(
                            side_hash = %block_hash,
                            main_height = header_info.height,
                            main_hash = %header_info.block_hash,
                            bmm_commitment = %block_info
                                .bmm_commitment
                                .map(|h| h.to_string())
                                .unwrap_or("none".to_string()),

                            "confirm BMM: received new block without our BMM commitment"
                        );
                    }
                }
                // This will actually never happen - there's currently no logic in
                // the enforcer that ends up sending this message. This is left in
                // for exhaustiveness purposes.
                disconnect @ Event::DisconnectBlock { .. } => {
                    tracing::warn!(
                        %block_hash,
                        event = ?disconnect,
                        "confirm BMM: received 'disconnect block' event"
                    );
                }
            }
        };

        // BMM requests expire after one block, so if we we weren't able to
        // get it in, the request failed.
        tracing::debug!(%block_hash, "confirm BMM: verification failed");
        self.block = None;
        Ok(None)
    }
}
