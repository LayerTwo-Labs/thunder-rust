use clap::Parser;
use thunder_app_cli_lib::Cli;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let res = cli.run().await?;
    println!("{res}");
    Ok(())
}
