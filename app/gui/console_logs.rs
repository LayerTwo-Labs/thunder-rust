use std::sync::{
    Arc,
    atomic::{self, AtomicBool},
};

use clap::Parser;
use eframe::egui::{
    self, Key, KeyboardShortcut, Modifiers, ScrollArea, TextEdit, TextStyle,
    TopBottomPanel, Widget as _,
};

use crate::{
    app::App,
    line_buffer::{LineBuffer, LineBufferWriter},
};

const SHIFT_ENTER: KeyboardShortcut = KeyboardShortcut {
    modifiers: Modifiers::SHIFT,
    logical_key: Key::Enter,
};

#[derive(Parser)]
#[command(name(""), no_binary_name(true))]
pub struct ConsoleCommand {
    #[command(subcommand)]
    command: thunder_app_cli_lib::Command,
}

pub struct ConsoleLogs {
    line_buffer: LineBuffer,
    command_input: String,
    rpc_addr: url::Url,
    running_command: Arc<AtomicBool>,
}

impl ConsoleLogs {
    pub fn new(line_buffer: LineBuffer, rpc_addr: url::Url) -> Self {
        Self {
            line_buffer,
            command_input: String::new(),
            rpc_addr,
            running_command: Arc::new(AtomicBool::new(false)),
        }
    }

    fn console_command(&mut self, app: &App) {
        use std::io::Write;
        let Some(args) = shlex::split(&self.command_input) else {
            return;
        };
        let mut line_buffer_writer = LineBufferWriter::from(&self.line_buffer);
        if let Err(err) =
            writeln!(line_buffer_writer, "> {}", self.command_input)
        {
            tracing::error!("{err}");
            self.command_input.clear();
            return;
        } else {
            self.command_input.clear();
        }
        let command = match ConsoleCommand::try_parse_from(args) {
            Ok(ConsoleCommand { command }) => command,
            Err(err) => {
                if let Err(err) = writeln!(line_buffer_writer, "{err}") {
                    tracing::error!("{err}")
                }
                return;
            }
        };
        let cli = thunder_app_cli_lib::Cli {
            rpc_url: self.rpc_addr.clone(),
            timeout: None,
            command,
            verbose: false,
        };
        app.runtime.spawn({
            let running_command = self.running_command.clone();
            running_command.store(true, atomic::Ordering::SeqCst);
            async move {
                if let Err(err) = match cli.run().await {
                    Ok(res) => writeln!(line_buffer_writer, "{res}"),
                    Err(err) => writeln!(line_buffer_writer, "{err:#}"),
                } {
                    tracing::error!("{err}")
                }
                running_command.store(false, atomic::Ordering::SeqCst);
            }
        });
    }

    pub fn show(&mut self, app: Option<&App>, ui: &mut egui::Ui) {
        TopBottomPanel::bottom("command_input").show_inside(ui, |ui| {
            let command_input = TextEdit::multiline(&mut self.command_input)
                .font(TextStyle::Monospace)
                .desired_width(f32::INFINITY)
                .desired_rows(1)
                .hint_text("help")
                .return_key(SHIFT_ENTER);
            let command_input_resp = ui.add_enabled(
                app.is_some()
                    && !self.running_command.load(atomic::Ordering::SeqCst),
                command_input,
            );
            if command_input_resp.ctx.input_mut(|input| {
                !input.consume_shortcut(&SHIFT_ENTER)
                    && input.consume_key(Modifiers::NONE, Key::Enter)
                    && !self.running_command.load(atomic::Ordering::SeqCst)
            }) {
                self.console_command(app.unwrap());
            }
        });
        ScrollArea::vertical().stick_to_bottom(true).show(ui, |ui| {
            let line_buffer_read = self.line_buffer.as_str();
            let mut logs: &str = &line_buffer_read;
            TextEdit::multiline(&mut logs)
                .font(TextStyle::Monospace)
                .desired_width(f32::INFINITY)
                .ui(ui);
        });
    }
}
