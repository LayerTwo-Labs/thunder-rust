use std::{
    net::{IpAddr, Ipv4Addr, SocketAddr},
    ops::Deref,
    path::PathBuf,
    sync::LazyLock,
};

use clap::{Arg, Parser};
use thunder::types::THIS_SIDECHAIN;

use crate::util::saturating_pred_level;

const fn ipv4_socket_addr(ipv4_octets: [u8; 4], port: u16) -> SocketAddr {
    let [a, b, c, d] = ipv4_octets;
    let ipv4 = Ipv4Addr::new(a, b, c, d);
    SocketAddr::new(IpAddr::V4(ipv4), port)
}

static DEFAULT_DATA_DIR: LazyLock<Option<PathBuf>> =
    LazyLock::new(|| match dirs::data_dir() {
        None => {
            tracing::warn!("Failed to resolve default data dir");
            None
        }
        Some(data_dir) => Some(data_dir.join("thunder")),
    });

const DEFAULT_NET_ADDR: SocketAddr =
    ipv4_socket_addr([0, 0, 0, 0], 4000 + THIS_SIDECHAIN as u16);

const DEFAULT_RPC_ADDR: SocketAddr =
    ipv4_socket_addr([127, 0, 0, 1], 6000 + THIS_SIDECHAIN as u16);

/// Implement arg manually so that there is only a default if we can resolve
/// the default data dir
#[derive(Clone, Debug)]
#[repr(transparent)]
struct DatadirArg(PathBuf);

impl clap::FromArgMatches for DatadirArg {
    fn from_arg_matches(
        matches: &clap::ArgMatches,
    ) -> Result<Self, clap::Error> {
        let mut matches = matches.clone();
        Self::from_arg_matches_mut(&mut matches)
    }

    fn from_arg_matches_mut(
        matches: &mut clap::ArgMatches,
    ) -> Result<Self, clap::Error> {
        let datadir = matches
            .remove_one::<PathBuf>("DATADIR")
            .expect("`datadir` is required");
        Ok(Self(datadir))
    }

    fn update_from_arg_matches(
        &mut self,
        matches: &clap::ArgMatches,
    ) -> Result<(), clap::Error> {
        let mut matches = matches.clone();
        self.update_from_arg_matches_mut(&mut matches)
    }

    fn update_from_arg_matches_mut(
        &mut self,
        matches: &mut clap::ArgMatches,
    ) -> Result<(), clap::Error> {
        if let Some(datadir) = matches.remove_one("DATADIR") {
            self.0 = datadir;
        }
        Ok(())
    }
}

impl clap::Args for DatadirArg {
    fn augment_args(cmd: clap::Command) -> clap::Command {
        cmd.arg({
            let arg = Arg::new("DATADIR")
                .value_parser(clap::builder::PathBufValueParser::new())
                .long("datadir")
                .short('d')
                .help("Data directory for storing blockchain and wallet data");
            match DEFAULT_DATA_DIR.deref() {
                None => arg.required(true),
                Some(datadir) => {
                    arg.required(false).default_value(datadir.as_os_str())
                }
            }
        })
    }

    fn augment_args_for_update(cmd: clap::Command) -> clap::Command {
        Self::augment_args(cmd)
    }
}

#[derive(Clone, Debug, Parser)]
#[command(author, version, about, long_about = None)]
pub(super) struct Cli {
    /// Data directory for storing blockchain and wallet data
    #[command(flatten)]
    datadir: DatadirArg,
    /// If specified, the gui will not launch.
    #[arg(long)]
    headless: bool,
    /// Directory in which to store log files.
    /// Defaults to `<DATADIR>/logs/v<VERSION>`, where `<DATADIR>` is thunder's data
    /// directory, and `<VERSION>` is the thunder app version.
    /// By default, only logs at the WARN level and above are logged to file.
    /// If set to the empty string, logging to file will be disabled.
    #[arg(long)]
    log_dir: Option<PathBuf>,

    /// Log level for logs that get written to file
    #[arg(default_value_t = tracing::Level::WARN, long)]
    log_level_file: tracing::Level,

    /// Log level
    #[arg(default_value_t = tracing::Level::DEBUG, long)]
    log_level: tracing::Level,

    /// Connect to mainchain node gRPC server running at this URL
    #[arg(default_value = "http://localhost:50051", long)]
    mainchain_grpc_url: url::Url,

    /// Path to a mnemonic seed phrase
    #[arg(long)]
    mnemonic_seed_phrase_path: Option<PathBuf>,
    /// Socket address to use for P2P networking
    #[arg(default_value_t = DEFAULT_NET_ADDR, long, short)]
    net_addr: SocketAddr,
    /// Socket address to host the RPC server
    #[arg(default_value_t = DEFAULT_RPC_ADDR, long, short)]
    rpc_addr: SocketAddr,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub datadir: PathBuf,
    pub headless: bool,
    /// If None, logging to file should be disabled.
    pub log_dir: Option<PathBuf>,
    pub log_level: tracing::Level,
    pub log_level_file: tracing::Level, // Level for logs that get written to file
    pub mainchain_grpc_url: url::Url,
    pub mnemonic_seed_phrase_path: Option<PathBuf>,
    pub net_addr: SocketAddr,
    pub rpc_addr: SocketAddr,
}

impl Cli {
    pub fn get_config(self) -> anyhow::Result<Config> {
        let log_dir = match self.log_dir {
            None => {
                let version_dir_name =
                    format!("v{}", env!("CARGO_PKG_VERSION"));
                let log_dir =
                    self.datadir.0.join("logs").join(version_dir_name);
                Some(log_dir)
            }
            Some(log_dir) => {
                if log_dir.as_os_str().is_empty() {
                    None
                } else {
                    Some(log_dir)
                }
            }
        };
        let log_level = if self.headless {
            self.log_level
        } else {
            saturating_pred_level(self.log_level)
        };
        Ok(Config {
            datadir: self.datadir.0,
            headless: self.headless,
            log_dir,
            log_level,
            log_level_file: self.log_level_file,
            mainchain_grpc_url: self.mainchain_grpc_url,
            mnemonic_seed_phrase_path: self.mnemonic_seed_phrase_path,
            net_addr: self.net_addr,
            rpc_addr: self.rpc_addr,
        })
    }
}
