#![feature(let_chains)]

use std::sync::mpsc;

use clap::Parser as _;

use tracing_subscriber::{filter as tracing_filter, layer::SubscriberExt};

mod app;
mod cli;
mod gui;
mod line_buffer;
mod rpc_server;
mod util;

use line_buffer::{LineBuffer, LineBufferWriter};

// Configure logger
fn set_tracing_subscriber(log_level: tracing::Level) -> LineBuffer {
    let targets_filter = tracing_filter::Targets::new().with_targets([
        ("bip300301", log_level),
        ("jsonrpsee_core::tracing", log_level),
        ("thunder", log_level),
        ("thunder_app", log_level),
    ]);
    let line_buffer = LineBuffer::default();
    let stdout_layer = tracing_subscriber::fmt::layer()
        .compact()
        .with_line_number(true);
    let capture_layer = tracing_subscriber::fmt::layer()
        .compact()
        .with_line_number(true)
        .with_ansi(false)
        .with_writer(LineBufferWriter::from(&line_buffer));
    let tracing_subscriber = tracing_subscriber::registry()
        .with(targets_filter)
        .with(stdout_layer)
        .with(capture_layer);
    tracing::subscriber::set_global_default(tracing_subscriber)
        .expect("setting default subscriber failed");
    line_buffer
}

fn main() -> anyhow::Result<()> {
    let cli = cli::Cli::parse();
    let config = cli.get_config()?;
    let line_buffer = set_tracing_subscriber(config.log_level);
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
            Box::new(move |cc| {
                Box::new(gui::EguiApp::new(
                    app,
                    cc,
                    line_buffer,
                    config.rpc_addr,
                ))
            }),
        )
        .map_err(|err| anyhow::anyhow!("failed to launch egui app: {err}"))?
    }
    Ok(())
}
