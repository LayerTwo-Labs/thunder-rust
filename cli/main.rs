use clap::Parser;
use mimalloc::MiMalloc;
use thunder_app_cli_lib::Cli;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;
fn configure_mimalloc() {
    // Same tuning as the app binary
    unsafe {
        std::env::set_var("MIMALLOC_ABANDONED_PAGE_LIMIT", "4");
        std::env::set_var("MIMALLOC_ABANDONED_PAGE_RESET", "1");
        std::env::set_var("MIMALLOC_ARENA_LIMIT", "4");
        std::env::set_var("MIMALLOC_USE_NUMA_NODES", "all");
        std::env::set_var("MIMALLOC_EAGER_COMMIT", "1");
        std::env::set_var("MIMALLOC_EAGER_REGION_COMMIT", "1");
        std::env::set_var("MIMALLOC_SEGMENT_CACHE", "32");
        std::env::set_var("MIMALLOC_LARGE_OS_PAGES", "1");
        std::env::set_var("MIMALLOC_RESERVE_HUGE_OS_PAGES", "4");
        std::env::set_var("MIMALLOC_PAGE_RESET", "0");
        std::env::set_var("MIMALLOC_SEGMENT_RESET", "0");
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    configure_mimalloc();
    let cli = Cli::parse();
    let res = cli.run().await?;
    #[allow(clippy::print_stdout)]
    {
        println!("{res}");
    }
    Ok(())
}
