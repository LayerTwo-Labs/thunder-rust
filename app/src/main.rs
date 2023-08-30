use clap::Parser as _;

mod app;
mod cli;
mod gui;

fn main() -> anyhow::Result<()> {
    let cli = cli::Cli::parse();
    let config = cli.get_config()?;
    let app = app::App::new(&config)?;

    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "Thunder",
        native_options,
        Box::new(|cc| Box::new(gui::EguiApp::new(app, cc))),
    )
    .expect("failed to launch egui app");
    Ok(())
}
