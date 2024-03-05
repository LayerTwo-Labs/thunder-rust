use eframe::egui::{self, ScrollArea, TextEdit, TextStyle, Widget as _};

use crate::logs::LogsCapture;

pub struct Logs(LogsCapture);

impl Logs {
    pub fn new(capture: LogsCapture) -> Self {
        Self(capture)
    }

    pub fn show(&self, ui: &mut egui::Ui) {
        let text_read = self.0.as_str();
        let mut text: &str = &text_read;
        ScrollArea::vertical().stick_to_bottom(true).show(ui, |ui| {
            TextEdit::multiline(&mut text)
                .font(TextStyle::Monospace)
                .desired_width(f32::INFINITY)
                .ui(ui);
        });
    }
}
