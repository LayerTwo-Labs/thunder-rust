use std::{
    fs::File,
    io::Write,
    sync::{
        Arc,
        atomic::{self, AtomicBool},
    },
};

use clap::Parser;
use eframe::egui::{
    self, Color32, Grid, Key, Label, Modifiers, RichText,
    ScrollArea, Sense, TextEdit, TextStyle, TopBottomPanel,
};
use rfd::FileDialog;

use crate::{
    app::App,
    line_buffer::{LineBuffer, LineBufferWriter},
};



#[derive(Parser)]
#[command(name(""), no_binary_name(true))]
pub struct ConsoleCommand {
    #[command(subcommand)]
    command: thunder_app_cli_lib::Command,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
    Unknown,
}

impl LogLevel {
    fn from_str(s: &str) -> Self {
        // Normalize the string by trimming and converting to uppercase
        let normalized = s.trim().to_uppercase();

        // Check for exact matches first
        match normalized.as_str() {
            "ERROR" => LogLevel::Error,
            "WARN" | "WARNING" => LogLevel::Warn,
            "INFO" | "INFORMATION" => LogLevel::Info,
            "DEBUG" => LogLevel::Debug,
            "TRACE" => LogLevel::Trace,
            _ => {
                // If no exact match, check if the string contains any of the log levels
                if normalized.contains("ERROR") {
                    LogLevel::Error
                } else if normalized.contains("WARN") {
                    LogLevel::Warn
                } else if normalized.contains("INFO") {
                    LogLevel::Info
                } else if normalized.contains("DEBUG") {
                    LogLevel::Debug
                } else if normalized.contains("TRACE") {
                    LogLevel::Trace
                } else {
                    LogLevel::Unknown
                }
            }
        }
    }

    fn color(&self) -> Color32 {
        match self {
            LogLevel::Error => Color32::from_rgb(255, 85, 85),      // Bright red
            LogLevel::Warn => Color32::from_rgb(255, 184, 108),     // Orange
            LogLevel::Info => Color32::from_rgb(76, 175, 80),       // Material green - much better contrast
            LogLevel::Debug => Color32::from_rgb(189, 147, 249),    // Purple
            LogLevel::Trace => Color32::from_rgb(139, 233, 253),    // Cyan
            LogLevel::Unknown => Color32::GRAY,                     // Gray
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            LogLevel::Error => "ERROR",
            LogLevel::Warn => "WARN",
            LogLevel::Info => "INFO",
            LogLevel::Debug => "DEBUG",
            LogLevel::Trace => "TRACE",
            LogLevel::Unknown => "UNKNOWN",
        }
    }
}

#[derive(Debug, Clone)]
pub struct LogEntry {
    timestamp: String,
    level: LogLevel,
    message: String,
    source: Option<String>,
    expanded: bool,
}

impl LogEntry {
    fn parse(line: &str) -> Option<Self> {
        // Example log format: 2025-05-17T15:22:13.215206Z  INFO thunder_app::app:194: Instantiating wallet with data directory: <dir path>

        // First, split the line into words and handle potential multiple spaces
        let words: Vec<&str> = line.split_whitespace().collect();
        if words.len() < 2 {
            return None;
        }

        // The timestamp is the first word
        let timestamp = words[0].trim().to_string();

        // The log level is the second word
        let level_str = words[1].trim();
        let level = LogLevel::from_str(level_str);

        // The rest of the line is the source and message
        // Reconstruct the remaining part of the line
        let remaining_text = if words.len() > 2 {
            // Find the position after the level word
            if let Some(level_pos) = line.find(level_str) {
                let start_pos = level_pos + level_str.len();
                if start_pos < line.len() {
                    line[start_pos..].trim_start().to_string()
                } else {
                    String::new()
                }
            } else {
                // Fallback if we can't find the level in the original string
                words[2..].join(" ")
            }
        } else {
            String::new()
        };

        // Split the remaining text into source and message
        let message_parts: Vec<&str> = remaining_text.splitn(2, ": ").collect();
        let (source, message) = if message_parts.len() > 1 {
            (Some(message_parts[0].trim().to_string()), message_parts[1].trim().to_string())
        } else {
            (None, remaining_text)
        };

        Some(LogEntry {
            timestamp,
            level,
            message,
            source,
            expanded: false,
        })
    }
}

pub struct ConsoleLogs {
    line_buffer: LineBuffer,
    command_input: String,
    rpc_addr: url::Url,
    running_command: Arc<AtomicBool>,
    // New fields for enhanced log viewer
    parsed_logs: Vec<LogEntry>,
    search_text: String,
    show_error: bool,
    show_warn: bool,
    show_info: bool,
    show_debug: bool,
    show_trace: bool,
    show_unknown: bool,
    auto_scroll: bool,
    show_details: bool,
}

impl ConsoleLogs {
    pub fn new(line_buffer: LineBuffer, rpc_addr: url::Url) -> Self {
        Self {
            line_buffer,
            command_input: String::new(),
            rpc_addr,
            running_command: Arc::new(AtomicBool::new(false)),
            parsed_logs: Vec::new(),
            search_text: String::new(),
            show_error: true,
            show_warn: true,
            show_info: true,
            show_debug: true,
            show_trace: true,
            show_unknown: true,
            auto_scroll: true,
            show_details: false,
        }
    }

    fn parse_logs(&mut self) {
        let logs_str = self.line_buffer.as_str();
        let lines: Vec<&str> = logs_str.lines().collect();

        // Only parse if we have new lines
        if lines.len() != self.parsed_logs.len() {
            self.parsed_logs.clear();

            for line in lines {
                if let Some(log_entry) = LogEntry::parse(line) {
                    self.parsed_logs.push(log_entry);
                } else if !line.trim().is_empty() {
                    // Handle lines that don't match the expected format
                    // (like command outputs or malformed logs)
                    self.parsed_logs.push(LogEntry {
                        timestamp: String::new(),
                        level: LogLevel::Unknown,
                        message: line.to_string(),
                        source: None,
                        expanded: false,
                    });
                }
            }
        }
    }

    fn filtered_logs(&self) -> Vec<&LogEntry> {
        self.parsed_logs
            .iter()
            .filter(|log| {
                // Filter by log level
                let level_match = match log.level {
                    LogLevel::Error => self.show_error,
                    LogLevel::Warn => self.show_warn,
                    LogLevel::Info => self.show_info,
                    LogLevel::Debug => self.show_debug,
                    LogLevel::Trace => self.show_trace,
                    LogLevel::Unknown => self.show_unknown,
                };

                // Filter by search text if any
                let search_match = self.search_text.is_empty() ||
                    log.message.to_lowercase().contains(&self.search_text.to_lowercase()) ||
                    log.source.as_ref().map_or(false, |s| s.to_lowercase().contains(&self.search_text.to_lowercase()));

                // Both filters must match
                level_match && search_match
            })
            .collect()
    }

    fn export_logs(&self) {
        if let Some(path) = FileDialog::new()
            .add_filter("Text files", &["txt"])
            .add_filter("Log files", &["log"])
            .set_file_name("thunder_logs.log")
            .save_file()
        {
            let filtered_logs = self.filtered_logs();
            if let Ok(mut file) = File::create(path) {
                for log in filtered_logs {
                    let log_line = format!(
                        "{} {} {}{}\n",
                        log.timestamp,
                        log.level.as_str(),
                        log.source.as_ref().map_or("".to_string(), |s| format!("{}: ", s)),
                        log.message
                    );
                    if let Err(e) = file.write_all(log_line.as_bytes()) {
                        tracing::error!("Failed to write to log file: {}", e);
                    }
                }
            }
        }
    }

    fn clear_all_filters(&mut self) {
        self.show_error = true;
        self.show_warn = true;
        self.show_info = true;
        self.show_debug = true;
        self.show_trace = true;
        self.show_unknown = true;
        self.search_text.clear();
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
        // Parse logs from the line buffer
        self.parse_logs();

        // Top toolbar with filters and actions
        ui.horizontal(|ui| {
            ui.heading("Logs");
            ui.add_space(10.0);

            // Log level filters
            ui.label("Filters:");
            ui.checkbox(&mut self.show_error, "Error");
            ui.checkbox(&mut self.show_warn, "Warning");
            ui.checkbox(&mut self.show_info, "Info");
            ui.checkbox(&mut self.show_debug, "Debug");
            ui.checkbox(&mut self.show_trace, "Trace");

            ui.add_space(10.0);

            // Search box
            ui.label("Search:");
            let search_response = ui.add(
                TextEdit::singleline(&mut self.search_text)
                    .desired_width(150.0)
                    .hint_text("Search logs...")
            );

            if search_response.changed() {
                // Search text changed, no need to do anything special here
                // as we'll filter logs based on the current search_text
            }

            ui.add_space(10.0);

            // Action buttons
            if ui.button("Clear Filters").clicked() {
                self.clear_all_filters();
            }

            if ui.button("Export Logs").clicked() {
                self.export_logs();
            }

            ui.checkbox(&mut self.auto_scroll, "Auto-scroll");
            ui.checkbox(&mut self.show_details, "Show Details");
        });

        ui.separator();

        // Main log view
        // Create a clone of the filtered logs to avoid borrow issues
        let filtered_logs: Vec<LogEntry> = self.filtered_logs().iter().map(|&log| log.clone()).collect();

        let scroll_area = ScrollArea::vertical()
            .stick_to_bottom(self.auto_scroll)
            .auto_shrink([false; 2]);

        scroll_area.show(ui, |ui| {
            // Create a table-like view for logs
            Grid::new("logs_grid")
                .num_columns(4)
                .striped(true)
                .spacing([4.0, 4.0])
                .show(ui, |ui| {
                    // Table header
                    ui.label(RichText::new("Time").strong());
                    ui.label(RichText::new("Level").strong());
                    ui.label(RichText::new("Source").strong());
                    ui.label(RichText::new("Message").strong());
                    ui.end_row();

                    // Table rows
                    for (_i, log) in filtered_logs.iter().enumerate() {
                        // Time column
                        ui.label(&log.timestamp);

                        // Level column with color
                        ui.label(RichText::new(log.level.as_str()).color(log.level.color()));

                        // Source column
                        if let Some(source) = &log.source {
                            let source_text = if source.len() > 30 {
                                format!("{}...", &source[0..27])
                            } else {
                                source.clone()
                            };
                            ui.label(source_text);
                        } else {
                            ui.label("-");
                        }

                        // Message column
                        let message_text = if log.message.len() > 100 && !log.expanded {
                            format!("{}... (click to expand)", &log.message[0..97])
                        } else {
                            log.message.clone()
                        };

                        let message_response = ui.add(
                            Label::new(message_text)
                                .sense(Sense::click())
                        );

                        if message_response.clicked() {
                            // Find the original log entry and toggle its expanded state
                            if let Some(original_log) = self.parsed_logs.iter_mut().find(|l|
                                l.timestamp == log.timestamp &&
                                l.level == log.level &&
                                l.message == log.message) {
                                original_log.expanded = !original_log.expanded;
                            }
                        }

                        ui.end_row();

                        // Show detailed view if expanded and details are enabled
                        if log.expanded && self.show_details {
                            ui.label(""); // Empty time cell
                            ui.label(""); // Empty level cell
                            ui.label(""); // Empty source cell

                            // Full message with monospace font
                            let mut message_clone = log.message.clone();
                            ui.add(
                                TextEdit::multiline(&mut message_clone)
                                    .font(TextStyle::Monospace)
                                    .desired_width(ui.available_width())
                                    .desired_rows(3)
                                    .interactive(false)
                            );

                            ui.end_row();
                        }
                    }
                });
        });

        // Command input at the bottom
        TopBottomPanel::bottom("command_input").show_inside(ui, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label(RichText::new("Command:").strong());

                let command_input = TextEdit::singleline(&mut self.command_input)
                    .font(TextStyle::Monospace)
                    .desired_width(f32::INFINITY)
                    .hint_text("Type command here (e.g., 'help')");

                let command_input_resp = ui.add_enabled(
                    app.is_some()
                        && !self.running_command.load(atomic::Ordering::SeqCst),
                    command_input,
                );

                if command_input_resp.ctx.input_mut(|input| {
                    input.consume_key(Modifiers::NONE, Key::Enter)
                        && !self.running_command.load(atomic::Ordering::SeqCst)
                }) {
                    if let Some(app) = app {
                        self.console_command(app);
                    }
                }

                if ui.button("Run").clicked() && app.is_some() && !self.running_command.load(atomic::Ordering::SeqCst) {
                    self.console_command(app.unwrap());
                }
            });
            ui.add_space(4.0);
        });
    }
}
