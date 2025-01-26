use std::{
    net::{IpAddr, Ipv4Addr, SocketAddr},
    time::Duration,
};

use clap::{Parser, Subcommand};
use jsonrpsee::http_client::{HttpClient, HttpClientBuilder};
use thunder::types::{Address, Txid, THIS_SIDECHAIN};
use thunder_app_rpc_api::RpcClient;

#[derive(Clone, Debug, Subcommand)]
#[command(arg_required_else_help(true))]
pub enum Command {
    /// Get balance in sats
    Balance,
    /// Connect to a peer
    ConnectPeer { addr: SocketAddr },
    /// Deposit to address
    CreateDeposit {
        address: Address,
        #[arg(long)]
        value_sats: u64,
        #[arg(long)]
        fee_sats: u64,
    },
    /// Format a deposit address
    FormatDepositAddress { address: Address },
    /// Generate a mnemonic seed phrase
    GenerateMnemonic,
    /// Get the block with specified block hash, if it exists
    GetBlock {
        block_hash: thunder::types::BlockHash,
    },
    /// Get mainchain blocks that commit to a specified block hash
    GetBmmInclusions {
        block_hash: thunder::types::BlockHash,
    },
    /// Get a new address
    GetNewAddress,
    /// Get wallet addresses, sorted by base58 encoding
    GetWalletAddresses,
    /// Get wallet UTXOs
    GetWalletUtxos,
    /// Get the current block count
    GetBlockcount,
    /// Get the height of the latest failed withdrawal bundle
    LatestFailedWithdrawalBundleHeight,
    /// List peers
    ListPeers,
    /// List all UTXOs
    ListUtxos,
    /// Attempt to mine a sidechain block
    Mine {
        #[arg(long)]
        fee_sats: Option<u64>,
    },
    /// Get pending withdrawal bundle
    PendingWithdrawalBundle,
    /// Show OpenAPI schema
    #[command(name = "openapi-schema")]
    OpenApiSchema,
    /// Remove a tx from the mempool
    RemoveFromMempool { txid: Txid },
    /// Set the wallet seed from a mnemonic seed phrase
    SetSeedFromMnemonic { mnemonic: String },
    /// Get total sidechain wealth
    SidechainWealth,
    /// Stop the node
    Stop,
    /// Transfer funds to the specified address
    Transfer {
        dest: Address,
        #[arg(long)]
        value_sats: u64,
        #[arg(long)]
        fee_sats: u64,
    },
    /// Initiate a withdrawal to the specified mainchain address
    Withdraw {
        mainchain_address: bitcoin::Address<bitcoin::address::NetworkUnchecked>,
        #[arg(long)]
        amount_sats: u64,
        #[arg(long)]
        fee_sats: u64,
        #[arg(long)]
        mainchain_fee_sats: u64,
    },
}

const DEFAULT_RPC_ADDR: SocketAddr = SocketAddr::new(
    IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)),
    6000 + THIS_SIDECHAIN as u16,
);

#[derive(Clone, Debug, Parser)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    /// address for use by the RPC server
    #[arg(default_value_t = DEFAULT_RPC_ADDR, long)]
    pub rpc_addr: SocketAddr,

    #[arg(
        long,
        default_value_t = 60,
        help = "Timeout for RPC requests in seconds"
    )]
    pub timeout: u64,

    #[command(subcommand)]
    pub command: Command,
}

impl Cli {
    pub async fn run(self) -> anyhow::Result<String> {
        let rpc_client: HttpClient = HttpClientBuilder::default()
            .request_timeout(Duration::from_secs(self.timeout))
            .build(format!("http://{}", self.rpc_addr))?;
        let res = match self.command {
            Command::Balance => {
                let balance = rpc_client.balance().await?;
                serde_json::to_string_pretty(&balance)?
            }
            Command::ConnectPeer { addr } => {
                let () = rpc_client.connect_peer(addr).await?;
                String::default()
            }
            Command::CreateDeposit {
                address,
                value_sats,
                fee_sats,
            } => {
                let txid = rpc_client
                    .create_deposit(address, value_sats, fee_sats)
                    .await?;
                format!("{txid}")
            }
            Command::FormatDepositAddress { address } => {
                rpc_client.format_deposit_address(address).await?
            }
            Command::GetBlock { block_hash } => {
                let block = rpc_client.get_block(block_hash).await?;
                serde_json::to_string_pretty(&block)?
            }
            Command::GetBmmInclusions { block_hash } => {
                let bmm_inclusions =
                    rpc_client.get_bmm_inclusions(block_hash).await?;
                serde_json::to_string_pretty(&bmm_inclusions)?
            }
            Command::GenerateMnemonic => rpc_client.generate_mnemonic().await?,
            Command::GetNewAddress => {
                let address = rpc_client.get_new_address().await?;
                format!("{address}")
            }
            Command::GetWalletAddresses => {
                let addresses = rpc_client.get_wallet_addresses().await?;
                serde_json::to_string_pretty(&addresses)?
            }
            Command::GetWalletUtxos => {
                let utxos = rpc_client.get_wallet_utxos().await?;
                serde_json::to_string_pretty(&utxos)?
            }
            Command::GetBlockcount => {
                let blockcount = rpc_client.getblockcount().await?;
                format!("{blockcount}")
            }
            Command::LatestFailedWithdrawalBundleHeight => {
                let height =
                    rpc_client.latest_failed_withdrawal_bundle_height().await?;
                serde_json::to_string_pretty(&height)?
            }
            Command::ListPeers => {
                let peers = rpc_client.list_peers().await?;
                serde_json::to_string_pretty(&peers)?
            }
            Command::ListUtxos => {
                let utxos = rpc_client.list_utxos().await?;
                serde_json::to_string_pretty(&utxos)?
            }
            Command::Mine { fee_sats } => {
                let () = rpc_client.mine(fee_sats).await?;
                String::default()
            }
            Command::PendingWithdrawalBundle => {
                let withdrawal_bundle =
                    rpc_client.pending_withdrawal_bundle().await?;
                serde_json::to_string_pretty(&withdrawal_bundle)?
            }
            Command::OpenApiSchema => {
                let openapi =
                    <thunder_app_rpc_api::RpcDoc as utoipa::OpenApi>::openapi();
                openapi.to_pretty_json()?
            }
            Command::RemoveFromMempool { txid } => {
                let () = rpc_client.remove_from_mempool(txid).await?;
                String::default()
            }
            Command::SetSeedFromMnemonic { mnemonic } => {
                let () = rpc_client.set_seed_from_mnemonic(mnemonic).await?;
                String::default()
            }
            Command::SidechainWealth => {
                let sidechain_wealth =
                    rpc_client.sidechain_wealth_sats().await?;
                format!("{sidechain_wealth}")
            }
            Command::Stop => {
                let () = rpc_client.stop().await?;
                String::default()
            }
            Command::Transfer {
                dest,
                value_sats,
                fee_sats,
            } => {
                let txid =
                    rpc_client.transfer(dest, value_sats, fee_sats).await?;
                format!("{txid}")
            }
            Command::Withdraw {
                mainchain_address,
                amount_sats,
                fee_sats,
                mainchain_fee_sats,
            } => {
                let txid = rpc_client
                    .withdraw(
                        mainchain_address,
                        amount_sats,
                        fee_sats,
                        mainchain_fee_sats,
                    )
                    .await?;
                format!("{txid}")
            }
        };
        Ok(res)
    }
}
