use std::{fmt, io, net::SocketAddr, time::Duration};

use clap::{Parser, Subcommand};
use http::HeaderMap;
use jsonrpsee::{core::client::ClientT, http_client::HttpClientBuilder};

use thunder::types::{Address, Txid};
use thunder_app_rpc_api::RpcClient;
use tracing_subscriber::layer::SubscriberExt as _;

/// Custom error type for CLI-specific errors
#[derive(Debug)]
pub enum CliError {
    /// Connection error with details
    ConnectionError {
        /// The URL that was being connected to
        url: url::Url,
        /// The underlying error
        source: anyhow::Error,
    },
    /// Other errors
    Other(anyhow::Error),
}

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ConnectionError { url, source } => {
                write!(f, "Failed to connect to Thunder node at {}", url)?;

                // Check for common connection errors and provide helpful messages
                if let Some(io_err) = source.downcast_ref::<io::Error>() {
                    match io_err.kind() {
                        io::ErrorKind::ConnectionRefused => {
                            write!(f, "\nConnection refused. Please check that:")?;
                            write!(f, "\n  1. The Thunder node is running")?;
                            write!(f, "\n  2. The RPC server is enabled")?;
                            write!(f, "\n  3. The RPC address is correct ({})", url)?;
                            write!(f, "\n  4. There are no firewall rules blocking the connection")?;
                        }
                        io::ErrorKind::ConnectionReset => {
                            write!(f, "\nConnection reset by peer. The server might be overloaded or restarting.")?;
                        }
                        io::ErrorKind::ConnectionAborted => {
                            write!(f, "\nConnection aborted. The server might have terminated the connection.")?;
                        }
                        io::ErrorKind::TimedOut => {
                            write!(f, "\nConnection timed out. The server might be unresponsive or behind a firewall.")?;
                        }
                        io::ErrorKind::AddrNotAvailable => {
                            write!(f, "\nAddress not available. The specified address cannot be assigned.")?;
                        }
                        io::ErrorKind::NotConnected => {
                            write!(f, "\nNot connected. No connection could be established.")?;
                        }
                        io::ErrorKind::BrokenPipe => {
                            write!(f, "\nBroken pipe. The connection was unexpectedly closed.")?;
                        }
                        _ => {
                            write!(f, "\nIO Error: {} (kind: {:?})", source, io_err.kind())?;
                        }
                    }
                } else {
                    // Check for common error patterns in the error message
                    let err_str = source.to_string().to_lowercase();

                    if err_str.contains("connection refused") ||
                       err_str.contains("tcp connect error") {
                        write!(f, "\nConnection refused. Please check that:")?;
                        write!(f, "\n  1. The Thunder node is running")?;
                        write!(f, "\n  2. The RPC server is enabled")?;
                        write!(f, "\n  3. The RPC address is correct ({})", url)?;
                        write!(f, "\n  4. There are no firewall rules blocking the connection")?;
                    } else if err_str.contains("dns error") ||
                              err_str.contains("lookup") ||
                              err_str.contains("nodename nor servname provided") ||
                              err_str.contains("not known") {
                        write!(f, "\nDNS resolution failed. Could not resolve the host name.")?;
                        write!(f, "\nPlease check that the host part of the URL is correct.")?;
                    } else if err_str.contains("timeout") ||
                              err_str.contains("timed out") {
                        write!(f, "\nConnection timed out. The server might be unresponsive or behind a firewall.")?;
                    } else {
                        // For other errors, provide the details but in a more user-friendly format
                        let mut source_err = source.source();
                        if let Some(err) = source_err {
                            write!(f, "\nError: {}", err)?;

                            // Check if there's a more specific cause
                            source_err = err.source();
                            if let Some(err) = source_err {
                                write!(f, "\nCause: {}", err)?;
                            }
                        } else {
                            write!(f, "\nError: {}", source)?;
                        }
                    }

                    // Add a helpful suggestion for all connection errors
                    write!(f, "\n\nMake sure the Thunder node is running and accessible at {}", url)?;
                }
            }
            Self::Other(err) => write!(f, "{}", err)?,
        }
        Ok(())
    }
}

impl std::error::Error for CliError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ConnectionError { source, .. } => Some(source.as_ref()),
            Self::Other(err) => err.source(),
        }
    }
}

impl From<anyhow::Error> for CliError {
    fn from(err: anyhow::Error) -> Self {
        // Try to determine if this is a connection error
        if let Some(client_err) = err.downcast_ref::<jsonrpsee::core::ClientError>() {
            if client_err.to_string().contains("Connect") {
                // This is likely a connection error, but we don't have the URL here
                // We'll handle this in the run method where we have the URL
                return Self::Other(err);
            }
        }

        Self::Other(err)
    }
}

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
    /// Get the best mainchain block hash
    GetBestMainchainBlockHash,
    /// Get the best sidechain block hash
    GetBestSidechainBlockHash,
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

#[derive(Clone, Debug, Parser)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    /// Base URL used for requests to the RPC server.
    /// If protocol is not specified, http:// will be used.
    #[arg(default_value = "http://localhost:6009", long, value_parser = parse_url)]
    pub rpc_url: url::Url,

    #[arg(long, help = "Timeout for RPC requests in seconds (default: 60)")]
    pub timeout: Option<u64>,

    #[arg(short, long, help = "Enable verbose HTTP output")]
    pub verbose: bool,

    #[command(subcommand)]
    pub command: Command,
}

/// Custom URL parser that adds http:// prefix if missing
fn parse_url(s: &str) -> Result<url::Url, String> {
    // Check if the URL already has a scheme
    if s.contains("://") {
        // Try to parse the URL as-is
        match url::Url::parse(s) {
            Ok(url) => Ok(url),
            Err(e) => Err(format!(
                "Invalid URL '{}': {}. Make sure the URL format is correct (e.g., http://hostname:port).",
                s, e
            )),
        }
    } else {
        // Add http:// prefix and try to parse
        let url_with_scheme = format!("http://{}", s);
        match url::Url::parse(&url_with_scheme) {
            Ok(url) => Ok(url),
            Err(e) => Err(format!(
                "Invalid URL '{}': {}. Make sure the URL format is correct (e.g., http://hostname:port).",
                s, e
            )),
        }
    }
}
/// Handle a command, returning CLI output
async fn handle_command<RpcClient>(
    rpc_client: &RpcClient,
    command: Command,
) -> anyhow::Result<String>
where
    RpcClient: ClientT + Sync,
{
    Ok(match command {
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
        Command::GetBestMainchainBlockHash => {
            let block_hash = rpc_client.get_best_mainchain_block_hash().await?;
            serde_json::to_string_pretty(&block_hash)?
        }
        Command::GetBestSidechainBlockHash => {
            let block_hash = rpc_client.get_best_sidechain_block_hash().await?;
            serde_json::to_string_pretty(&block_hash)?
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
            let sidechain_wealth = rpc_client.sidechain_wealth_sats().await?;
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
            let txid = rpc_client.transfer(dest, value_sats, fee_sats).await?;
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
    })
}

fn set_tracing_subscriber() -> anyhow::Result<()> {
    let stdout_layer = tracing_subscriber::fmt::layer()
        .with_ansi(std::io::IsTerminal::is_terminal(&std::io::stdout()))
        .with_file(true)
        .with_line_number(true);

    let subscriber = tracing_subscriber::registry().with(stdout_layer);
    tracing::subscriber::set_global_default(subscriber)?;
    Ok(())
}

impl Cli {
    pub async fn run(self) -> Result<String, CliError> {
        if self.verbose {
            set_tracing_subscriber().map_err(|e| CliError::Other(e))?;
        }

        const DEFAULT_TIMEOUT: u64 = 60;

        let request_id = uuid::Uuid::new_v4().to_string().replace("-", "");

        tracing::info!("request ID: {}", request_id);

        let builder = HttpClientBuilder::default()
            .request_timeout(Duration::from_secs(
                self.timeout.unwrap_or(DEFAULT_TIMEOUT),
            ))
            .set_max_logging_length(1024)
            .set_headers(HeaderMap::from_iter([(
                http::header::HeaderName::from_static("x-request-id"),
                http::header::HeaderValue::from_str(&request_id)
                    .map_err(|e| CliError::Other(e.into()))?,
            )]));

        // Store the URL for potential error messages
        let rpc_url = self.rpc_url.clone();

        // Build the client and handle connection errors
        let client = builder.build(self.rpc_url.clone()).map_err(|err| {
            // Check if this is a connection error
            let err_str = err.to_string().to_lowercase();
            if err_str.contains("connect") ||
               err_str.contains("connection") ||
               err_str.contains("network") ||
               err_str.contains("tcp") ||
               err_str.contains("dns") ||
               err_str.contains("lookup") ||
               err_str.contains("timeout") ||
               err_str.contains("timed out") {
                return CliError::ConnectionError {
                    url: rpc_url.clone(),
                    source: err.into(),
                };
            }

            CliError::Other(err.into())
        })?;

        // Execute the command and handle potential connection errors
        let result = handle_command(&client, self.command)
            .await
            .map_err(|err| {
                // Check if this is a connection error that happened during the request
                if let Some(client_err) = err.downcast_ref::<jsonrpsee::core::ClientError>() {
                    if client_err.to_string().contains("Connect") ||
                       client_err.to_string().contains("connection") ||
                       client_err.to_string().contains("Connection") ||
                       client_err.to_string().contains("network") ||
                       client_err.to_string().contains("Network") ||
                       client_err.to_string().contains("timeout") ||
                       client_err.to_string().contains("timed out") {
                        return CliError::ConnectionError {
                            url: rpc_url.clone(),
                            source: err,
                        };
                    }
                }

                // Check for transport errors
                let err_str = err.to_string().to_lowercase();
                if err_str.contains("tcp connect error") ||
                   err_str.contains("connection refused") ||
                   err_str.contains("connection reset") ||
                   err_str.contains("connection aborted") ||
                   err_str.contains("broken pipe") ||
                   err_str.contains("dns error") ||
                   err_str.contains("lookup") ||
                   err_str.contains("timeout") ||
                   err_str.contains("timed out") {
                    return CliError::ConnectionError {
                        url: rpc_url.clone(),
                        source: err,
                    };
                }

                CliError::Other(err)
            })?;

        Ok(result)
    }
}
