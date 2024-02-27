use std::sync::mpsc;

use clap::Parser as _;

use tracing_subscriber::{filter as tracing_filter, layer::SubscriberExt};

mod app;
mod cli;
mod gui;
mod rpc_server;

// Configure logger
fn set_tracing_subscriber(log_level: tracing::Level) {
    let targets_filter = tracing_filter::Targets::new().with_targets([
        ("jsonrpsee_core::tracing", log_level),
        ("thunder", log_level),
        ("thunder_app", log_level),
    ]);
    let fmt_layer = tracing_subscriber::fmt::layer()
        .compact()
        .with_line_number(true);
    let tracing_subscriber = tracing_subscriber::registry()
        .with(targets_filter)
        .with(fmt_layer);
    tracing::subscriber::set_global_default(tracing_subscriber)
        .expect("setting default subscriber failed");
}

fn main() -> anyhow::Result<()> {
    let cli = cli::Cli::parse();
    let config = cli.get_config()?;
    let () = set_tracing_subscriber(config.log_level);
    let app = app::App::new(&config)?;
    // spawn rpc server
    app.runtime.spawn({
        let app = app.clone();
        async move { rpc_server::run_server(app, config.rpc_addr).await.unwrap() }
    });

    if config.headless {
        // wait for ctrlc signal
        let (tx, rx) = mpsc::channel();
        ctrlc::set_handler(move || {
            tx.send(()).unwrap();
        })
        .expect("Error setting Ctrl-C handler");
        rx.recv().unwrap();
        println!("Received Ctrl-C signal, exiting...");
    } else {
        let native_options = eframe::NativeOptions::default();
        eframe::run_native(
            "Thunder",
            native_options,
            Box::new(|cc| Box::new(gui::EguiApp::new(app, cc))),
        )
        .map_err(|err| anyhow::anyhow!("failed to launch egui app: {err}"))?
    }
    Ok(())
}
