use std::net::{IpAddr, Ipv4Addr, SocketAddr};

use bip300301::bitcoin;
use clap::{Parser, Subcommand};
use jsonrpsee::http_client::{HttpClient, HttpClientBuilder};
use thunder::types::{Address, Txid};
use thunder_app_rpc_api::RpcClient;

#[derive(Clone, Debug, Subcommand)]
#[command(arg_required_else_help(true))]
pub enum Command {
    /// Get balance in sats
    Balance,
    /// Connect to a peer
    ConnectPeer { addr: SocketAddr },
    /// Format a deposit address
    FormatDepositAddress { address: Address },
    /// Generate a mnemonic seed phrase
    GenerateMnemonic,
    /// Get a new address
    GetNewAddress,
    /// Get the current block count
    GetBlockcount,
    /// Attempt to mine a sidechain block
    Mine {
        #[arg(long)]
        fee_sats: Option<u64>,
    },
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

const DEFAULT_RPC_ADDR: SocketAddr =
    SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 2020);

#[derive(Clone, Debug, Parser)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    /// address for use by the RPC server
    #[arg(default_value_t = DEFAULT_RPC_ADDR, long)]
    pub rpc_addr: SocketAddr,

    #[command(subcommand)]
    pub command: Command,
}

impl Cli {
    pub async fn run(self) -> anyhow::Result<String> {
        let rpc_client: HttpClient = HttpClientBuilder::default()
            .build(format!("http://{}", self.rpc_addr))?;
        let res = match self.command {
            Command::Balance => {
                let balance = rpc_client.balance().await?;
                format!("{balance}")
            }
            Command::ConnectPeer { addr } => {
                let () = rpc_client.connect_peer(addr).await?;
                String::default()
            }
            Command::FormatDepositAddress { address } => {
                rpc_client.format_deposit_address(address).await?
            }
            Command::GenerateMnemonic => rpc_client.generate_mnemonic().await?,
            Command::GetNewAddress => {
                let address = rpc_client.get_new_address().await?;
                format!("{address}")
            }
            Command::GetBlockcount => {
                let blockcount = rpc_client.getblockcount().await?;
                format!("{blockcount}")
            }
            Command::Mine { fee_sats } => {
                let () = rpc_client.mine(fee_sats).await?;
                String::default()
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
                let sidechain_wealth = rpc_client.sidechain_wealth().await?;
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
