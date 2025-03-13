use std::{
    ffi::{OsStr, OsString},
    path::PathBuf,
};

use bip300301_enforcer_integration_tests::util::{
    AbortOnDrop, BinPaths as EnforcerBinPaths, VarError, get_env_var,
    spawn_command_with_args,
};

fn load_env_var_from_string(s: &str) -> Result<(), VarError> {
    dotenvy::from_read_override(s.as_bytes())
        .map_err(|err| VarError::new(s, err))
}

#[derive(Clone, Debug)]
pub struct BinPaths {
    pub thunder: PathBuf,
    pub others: EnforcerBinPaths,
}

impl BinPaths {
    /// Read from environment variables
    pub fn from_env() -> Result<Self, VarError> {
        let () = load_env_var_from_string("BITCOIN_UTIL=''")?;
        let () = load_env_var_from_string("SIGNET_MINER=''")?;
        Ok(Self {
            thunder: get_env_var("THUNDER_APP")?.into(),
            others: EnforcerBinPaths::from_env()?,
        })
    }
}

#[derive(Clone, Debug)]
pub struct ThunderApp {
    pub path: PathBuf,
    pub data_dir: PathBuf,
    pub log_level: Option<tracing::Level>,
    pub mainchain_grpc_port: u16,
    /// Port to use for P2P networking
    pub net_port: u16,
    /// Port to use for the RPC server
    pub rpc_port: u16,
}

impl ThunderApp {
    pub fn spawn_command_with_args<Env, Arg, Envs, Args, F>(
        &self,
        envs: Envs,
        args: Args,
        err_handler: F,
    ) -> AbortOnDrop<()>
    where
        Arg: AsRef<OsStr>,
        Env: AsRef<OsStr>,
        Envs: IntoIterator<Item = (Env, Env)>,
        Args: IntoIterator<Item = Arg>,
        F: FnOnce(anyhow::Error) + Send + 'static,
    {
        let mut default_args = vec![
            "--datadir".to_owned(),
            self.data_dir.display().to_string(),
            "--headless".to_owned(),
            "--mainchain-grpc-url".to_owned(),
            format!("http://127.0.0.1:{}", self.mainchain_grpc_port),
            "--net-addr".to_owned(),
            format!("127.0.0.1:{}", self.net_port),
            "--rpc-addr".to_owned(),
            format!("127.0.0.1:{}", self.rpc_port),
        ];
        if let Some(log_level) = self.log_level {
            default_args.push("--log-level".to_owned());
            default_args.push(log_level.as_str().to_owned());
        }
        let args = default_args
            .into_iter()
            .map(OsString::from)
            .chain(args.into_iter().map(|arg| arg.as_ref().to_owned()));
        spawn_command_with_args(
            &self.data_dir,
            self.path.clone(),
            envs,
            args,
            err_handler,
        )
    }
}
