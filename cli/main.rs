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
            match err {
                CliError::ConnectionError { .. } => {
                    // For connection errors, we want to show a user-friendly message
                    // without the stack trace
                    eprintln!("{}", err);
                    std::process::exit(1);
                }
                CliError::Other(err) => {
                    // For other errors, we'll let anyhow handle it with the stack trace
                    Err(err)
                }
            }
        }
    }
}
