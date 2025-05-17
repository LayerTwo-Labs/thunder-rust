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
                // First, check if this is a DNS error by examining the full error chain
                let is_dns_error = {
                    let err_str = source.to_string().to_lowercase();
                    let is_dns = err_str.contains("dns error") ||
                                 err_str.contains("lookup") ||
                                 err_str.contains("nodename nor servname provided") ||
                                 err_str.contains("not known") ||
                                 err_str.contains("name resolution") ||
                                 err_str.contains("host not found") ||
                                 err_str.contains("no such host");

                    // Also check the error chain for DNS-related errors
                    if !is_dns {
                        let mut current_err = source.source();
                        let mut found_dns_error = false;
                        while let Some(err) = current_err {
                            let err_str = err.to_string().to_lowercase();
                            if err_str.contains("dns error") ||
                               err_str.contains("lookup") ||
                               err_str.contains("nodename nor servname provided") ||
                               err_str.contains("not known") ||
                               err_str.contains("name resolution") ||
                               err_str.contains("host not found") ||
                               err_str.contains("no such host") {
                                found_dns_error = true;
                                break;
                            }
                            current_err = err.source();
                        }
                        found_dns_error
                    } else {
                        true
                    }
                };

                // Handle DNS errors first, as they're a special case
                if is_dns_error {
                    write!(f, "Unable to connect: DNS resolution failed")?;
                    write!(f, "\n\nCould not resolve the host name: {}", url.host_str().unwrap_or("unknown"))?;
                    write!(f, "\n\nPlease check that:")?;
                    write!(f, "\n  1. The hostname part of the URL is correct")?;
                    write!(f, "\n  2. Your network connection is working")?;
                    write!(f, "\n  3. DNS resolution is functioning properly")?;

                    return Ok(());
                }

                // Check for HTTP status code errors
                let err_str = source.to_string().to_lowercase();
                if err_str.contains("404") || err_str.contains("rejected") {
                    write!(f, "Unable to connect: Server responded with an error")?;
                    write!(f, "\n\nThe server at {} responded, but it doesn't appear to be a Thunder node.", url)?;
                    write!(f, "\n\nPlease check that:")?;
                    write!(f, "\n  1. The URL is correct")?;
                    write!(f, "\n  2. The server at this address is running Thunder")?;
                    write!(f, "\n  3. You're using the correct protocol (http:// vs https://)")?;

                    return Ok(());
                }

                // Check for connection closed errors
                if err_str.contains("connection closed") ||
                   err_str.contains("broken pipe") ||
                   err_str.contains("reset by peer") {
                    write!(f, "Unable to connect: Connection was closed unexpectedly")?;
                    write!(f, "\n\nThe server at {} closed the connection.", url)?;
                    write!(f, "\n\nThis usually means:")?;
                    write!(f, "\n  1. The server is not a Thunder node")?;
                    write!(f, "\n  2. You're connecting to the wrong port")?;
                    write!(f, "\n  3. The server requires a different protocol")?;
                    write!(f, "\n\nPlease verify the address and port are correct.")?;

                    return Ok(());
                }

                // Check for common connection errors and provide helpful messages
                if let Some(io_err) = source.downcast_ref::<io::Error>() {
                    match io_err.kind() {
                        io::ErrorKind::ConnectionRefused => {
                            write!(f, "Unable to connect: Connection refused")?;
                            write!(f, "\n\nCould not connect to Thunder node at {}", url)?;
                            write!(f, "\n\nPlease check that:")?;
                            write!(f, "\n  1. The Thunder node is running")?;
                            write!(f, "\n  2. The RPC server is enabled")?;
                            write!(f, "\n  3. The RPC address is correct ({})", url)?;
                            write!(f, "\n  4. There are no firewall rules blocking the connection")?;
                        }
                        io::ErrorKind::ConnectionReset => {
                            write!(f, "Unable to connect: Connection reset")?;
                            write!(f, "\n\nThe connection was reset by the server at {}", url)?;
                            write!(f, "\n\nThis usually happens when:")?;
                            write!(f, "\n  1. The server is overloaded")?;
                            write!(f, "\n  2. The server is restarting")?;
                            write!(f, "\n  3. There's a network issue between you and the server")?;
                            write!(f, "\n\nPlease try again in a few moments.")?;
                        }
                        io::ErrorKind::ConnectionAborted => {
                            write!(f, "Unable to connect: Connection aborted")?;
                            write!(f, "\n\nThe connection was aborted while trying to reach {}", url)?;
                            write!(f, "\n\nThis usually happens when:")?;
                            write!(f, "\n  1. The server terminated the connection")?;
                            write!(f, "\n  2. A network device (like a firewall) interrupted the connection")?;
                            write!(f, "\n\nPlease check your network settings and try again.")?;
                        }
                        io::ErrorKind::TimedOut => {
                            write!(f, "Unable to connect: Connection timed out")?;
                            write!(f, "\n\nThe connection to {} timed out.", url)?;
                            write!(f, "\n\nThis usually means:")?;
                            write!(f, "\n  1. The server is not responding")?;
                            write!(f, "\n  2. A firewall is blocking the connection")?;
                            write!(f, "\n  3. The network path to the server is congested or down")?;
                            write!(f, "\n\nPlease check that the server is running and accessible.")?;
                        }
                        io::ErrorKind::AddrNotAvailable => {
                            write!(f, "Unable to connect: Address not available")?;
                            write!(f, "\n\nThe address {} cannot be used.", url)?;
                            write!(f, "\n\nThis usually happens when:")?;
                            write!(f, "\n  1. The IP address is not valid on this network")?;
                            write!(f, "\n  2. The port is already in use or reserved")?;
                            write!(f, "\n\nPlease try a different address or port.")?;
                        }
                        io::ErrorKind::NotConnected => {
                            write!(f, "Unable to connect: Not connected")?;
                            write!(f, "\n\nCould not establish a connection to {}", url)?;
                            write!(f, "\n\nPlease check your network connection and try again.")?;
                        }
                        io::ErrorKind::BrokenPipe => {
                            write!(f, "Unable to connect: Connection broken")?;
                            write!(f, "\n\nThe connection to {} was unexpectedly closed.", url)?;
                            write!(f, "\n\nThis usually happens when the server closes the connection.")?;
                            write!(f, "\n\nPlease check that the server is running and try again.")?;
                        }
                        _ => {
                            write!(f, "Unable to connect to Thunder node at {}", url)?;
                            write!(f, "\n\nAn unexpected network error occurred.")?;
                            write!(f, "\n\nPlease check that:")?;
                            write!(f, "\n  1. The Thunder node is running")?;
                            write!(f, "\n  2. The address is correct")?;
                            write!(f, "\n  3. Your network connection is working")?;
                        }
                    }
                } else {
                    // Check for common error patterns in the error message
                    let err_str = source.to_string().to_lowercase();

                    if err_str.contains("connection refused") ||
                       err_str.contains("tcp connect error") ||
                       err_str.contains("connect error") ||
                       err_str.contains("connect failed") ||
                       err_str.contains("os error 61") ||
                       err_str.contains("client error (connect)") {
                        write!(f, "Unable to connect: Connection refused")?;
                        write!(f, "\n\nCould not connect to Thunder node at {}", url)?;
                        write!(f, "\n\nPlease check that:")?;
                        write!(f, "\n  1. The Thunder node is running")?;
                        write!(f, "\n  2. The RPC server is enabled")?;
                        write!(f, "\n  3. The RPC address is correct ({})", url)?;
                        write!(f, "\n  4. There are no firewall rules blocking the connection")?;
                    } else if err_str.contains("timeout") ||
                              err_str.contains("timed out") {
                        write!(f, "Unable to connect: Connection timed out")?;
                        write!(f, "\n\nThe connection to {} timed out.", url)?;
                        write!(f, "\n\nThis usually means:")?;
                        write!(f, "\n  1. The server is not responding")?;
                        write!(f, "\n  2. A firewall is blocking the connection")?;
                        write!(f, "\n  3. The network path to the server is congested or down")?;
                        write!(f, "\n\nPlease check that the server is running and accessible.")?;
                    } else {
                        // For other errors, provide a generic but user-friendly message
                        write!(f, "Unable to connect to Thunder node at {}", url)?;
                        write!(f, "\n\nAn unexpected error occurred while trying to connect.")?;
                        write!(f, "\n\nPlease check that:")?;
                        write!(f, "\n  1. The Thunder node is running")?;
                        write!(f, "\n  2. The address is correct")?;
                        write!(f, "\n  3. Your network connection is working")?;

                        // If in verbose mode or for debugging, we can include the technical details
                        if std::env::var("THUNDER_VERBOSE").is_ok() {
                            write!(f, "\n\nTechnical details (for support):")?;
                            write!(f, "\n{}", source)?;
                        }
                    }
                }
            }
            Self::Other(err) => {
                // For other errors, we'll try to make them more user-friendly too
                let err_str = err.to_string().to_lowercase();

                if err_str.contains("404") || err_str.contains("rejected") {
                    write!(f, "Unable to connect: Server responded with an error")?;
                    write!(f, "\n\nThe server responded, but it doesn't appear to be a Thunder node.")?;
                    write!(f, "\n\nPlease check that:")?;
                    write!(f, "\n  1. The URL is correct")?;
                    write!(f, "\n  2. The server at this address is running Thunder")?;
                    write!(f, "\n  3. You're using the correct protocol (http:// vs https://)")?;
                } else {
                    // For truly unknown errors, just pass through but with a nicer format
                    write!(f, "Error: {}", err)?;

                    // If in verbose mode or for debugging, we can include more technical details
                    if std::env::var("THUNDER_VERBOSE").is_ok() {
                        if let Some(source) = err.source() {
                            write!(f, "\n\nCaused by: {}", source)?;
                        }
                    }
                }
            }
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
            Err(e) => {
                let error_msg = e.to_string().to_lowercase();

                if error_msg.contains("invalid ip") || error_msg.contains("invalid ipv") {
                    Err(format!(
                        "Invalid IP address in URL: '{}'.\n\nPlease provide a valid IP address format.",
                        s
                    ))
                } else if error_msg.contains("invalid port") {
                    Err(format!(
                        "Invalid port in URL: '{}'.\n\nPorts must be numbers between 1-65535.",
                        s
                    ))
                } else if error_msg.contains("relative url") || error_msg.contains("empty host") {
                    Err(format!(
                        "Invalid URL format: '{}'.\n\nPlease provide a complete URL in the format: http://hostname:port",
                        s
                    ))
                } else {
                    Err(format!(
                        "Invalid URL: '{}'.\n\nPlease provide a valid URL in the format: http://hostname:port",
                        s
                    ))
                }
            }
        }
    } else {
        // Add http:// prefix and try to parse
        let url_with_scheme = format!("http://{}", s);
        match url::Url::parse(&url_with_scheme) {
            Ok(url) => Ok(url),
            Err(e) => {
                let error_msg = e.to_string().to_lowercase();

                if error_msg.contains("invalid ip") || error_msg.contains("invalid ipv") {
                    Err(format!(
                        "Invalid IP address: '{}'.\n\nPlease provide a valid IP address format.",
                        s
                    ))
                } else if error_msg.contains("invalid port") {
                    Err(format!(
                        "Invalid port: '{}'.\n\nPorts must be numbers between 1-65535.",
                        s
                    ))
                } else if error_msg.contains("relative url") || error_msg.contains("empty host") {
                    Err(format!(
                        "Invalid URL format: '{}'.\n\nPlease provide a complete URL in the format: hostname:port",
                        s
                    ))
                } else {
                    Err(format!(
                        "Invalid URL: '{}'.\n\nPlease provide a valid URL in the format: hostname:port",
                        s
                    ))
                }
            }
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

        let request_id = uuid::Uuid::new_v4().simple().to_string();

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
