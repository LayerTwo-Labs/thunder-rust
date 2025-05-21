use clap::Parser;
use thunder_app_cli_lib::{Cli, CliError};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.run().await {
        Ok(res) => {
            #[allow(clippy::print_stdout)]
            {
                println!("{res}");
            }
            Ok(())
        }
        Err(err) => {
            // For all errors, we want to show a user-friendly message without the stack trace
            // unless THUNDER_DEBUG is set
            if std::env::var("THUNDER_DEBUG").is_ok() {
                match err {
                    CliError::ConnectionError { .. } => {
                        eprintln!("{}", err);
                        std::process::exit(1);
                    }
                    CliError::Other(err) => {
                        // For other errors, we'll let anyhow handle it with the stack trace
                        Err(err)
                    }
                }
            } else {
                eprintln!("{}", err);
                std::process::exit(1);
            }
        }
    }
}
