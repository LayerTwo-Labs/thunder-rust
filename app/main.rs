#![feature(let_chains)]

use std::{path::Path, sync::mpsc};

use clap::Parser as _;

use tracing_subscriber::{
    filter as tracing_filter, layer::SubscriberExt, Layer,
};

mod app;
mod cli;
mod gui;
mod line_buffer;
mod rpc_server;
mod util;

use line_buffer::{LineBuffer, LineBufferWriter};

/// Must be held for the lifetime of the program in order to keep the file
/// logger alive.
type RollingLoggerGuard = tracing_appender::non_blocking::WorkerGuard;

/// Rolling file logger.
/// Returns a guard that must be held for the lifetime of the program in order
/// to keep the file logger alive.
fn rolling_logger<S>(
    log_dir: &Path,
) -> anyhow::Result<(impl Layer<S>, RollingLoggerGuard)>
where
    S: tracing::Subscriber
        + for<'s> tracing_subscriber::registry::LookupSpan<'s>,
{
    const DEFAULT_LEVEL: tracing::Level = tracing::Level::WARN;
    const LOG_FILE_SUFFIX: &str = "log";
    let rolling_log_appender = tracing_appender::rolling::Builder::new()
        .rotation(tracing_appender::rolling::Rotation::DAILY)
        .filename_suffix(LOG_FILE_SUFFIX)
        .build(log_dir)?;
    let (non_blocking_rolling_log_writer, rolling_log_guard) =
        tracing_appender::non_blocking(rolling_log_appender);
    let level_filter =
        tracing_filter::Targets::new().with_default(DEFAULT_LEVEL);
    let rolling_log_layer = tracing_subscriber::fmt::layer()
        .compact()
        .with_ansi(false)
        .with_writer(non_blocking_rolling_log_writer)
        .with_filter(level_filter);
    Ok((rolling_log_layer, rolling_log_guard))
}

// Configure loggers.
// If the file logger is set, returns a guard that must be held for the
// lifetime of the program in order to keep the file logger alive.
fn set_tracing_subscriber(
    log_dir: Option<&Path>,
    log_level: tracing::Level,
) -> anyhow::Result<(LineBuffer, Option<RollingLoggerGuard>)> {
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
    let (rolling_log_layer, rolling_log_guard) = match log_dir {
        None => (None, None),
        Some(log_dir) => {
            let (layer, guard) = rolling_logger(log_dir)?;
            (Some(layer), Some(guard))
        }
    };
    let capture_layer = tracing_subscriber::fmt::layer()
        .compact()
        .with_line_number(true)
        .with_ansi(false)
        .with_writer(LineBufferWriter::from(&line_buffer));
    let tracing_subscriber = tracing_subscriber::registry()
        .with(targets_filter)
        .with(stdout_layer)
        .with(capture_layer)
        .with(rolling_log_layer);
    tracing::subscriber::set_global_default(tracing_subscriber)
        .expect("setting default subscriber failed");
    Ok((line_buffer, rolling_log_guard))
}

fn main() -> anyhow::Result<()> {
    let cli = cli::Cli::parse();
    let config = cli.get_config()?;
    let (line_buffer, _rolling_log_guard) =
        set_tracing_subscriber(config.log_dir.as_deref(), config.log_level)?;
    let app: Result<app::App, app::Error> =
        app::App::new(&config).inspect(|app| {
            // spawn rpc server
            app.runtime.spawn({
                let app = app.clone();
                async move {
                    rpc_server::run_server(app, config.rpc_addr).await.unwrap()
                }
            });
        });
    if config.headless {
        drop(line_buffer);
        let _app = app?;
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
        let app: Option<_> = app.map_or_else(
            |err| {
                let err = anyhow::Error::from(err);
                tracing::error!("{err:#}");
                None
            },
            Some,
        );
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
