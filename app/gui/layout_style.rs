//! Layout styling for the Thunder UI

use eframe::egui::{self, Color32, Response, Ui};

/// Layout dimensions
pub struct LayoutDimensions;

impl LayoutDimensions {
    pub const CONTAINER_PADDING: f32 = 16.0;
    pub const SECTION_SPACING: f32 = 24.0;
    pub const ELEMENT_SPACING: f32 = 8.0;
    pub const SMALL_SPACING: f32 = 4.0;
}

/// Layout colors
pub struct LayoutColors;

impl LayoutColors {
    pub const BACKGROUND: Color32 = Color32::from_rgb(0xF5, 0xF5, 0xF7);
    pub const CARD_BACKGROUND: Color32 = Color32::from_rgb(0xFF, 0xFF, 0xFF);
    pub const BORDER: Color32 = Color32::from_rgb(0xD2, 0xD2, 0xD7);
}

/// Layout UI extensions
pub trait LayoutUiExt {
    /// Create a card container with proper spacing and borders
    fn layout_card<R>(&mut self, add_contents: impl FnOnce(&mut Ui) -> R) -> R;

    /// Add a section heading
    fn layout_heading(&mut self, text: &str) -> Response;
}

impl LayoutUiExt for Ui {
    fn layout_card<R>(&mut self, add_contents: impl FnOnce(&mut Ui) -> R) -> R {
        let frame = egui::Frame::none()
            .fill(LayoutColors::CARD_BACKGROUND)
            .stroke(egui::Stroke::new(1.0, LayoutColors::BORDER))
            .inner_margin(LayoutDimensions::CONTAINER_PADDING);

        frame.show(self, add_contents).inner
    }

    fn layout_heading(&mut self, text: &str) -> Response {
        self.add_space(LayoutDimensions::ELEMENT_SPACING);
        let response = self.heading(text);
        self.add_space(LayoutDimensions::ELEMENT_SPACING);
        response
    }
}

/// Helper functions for layout
pub struct LayoutHelpers;

impl LayoutHelpers {
    /// Create a BTC input field with label
    pub fn btc_input_field(ui: &mut Ui, value: &mut String, label: &str, hint: &str) -> Response {
        ui.add_space(LayoutDimensions::SMALL_SPACING);
        ui.label(label);
        ui.add_space(LayoutDimensions::SMALL_SPACING);

        ui.horizontal(|ui| {
            let text_edit = egui::TextEdit::singleline(value)
                .hint_text(hint)
                .desired_width(ui.available_width() - 40.0);

            let response = ui.add(text_edit);
            ui.add_space(LayoutDimensions::SMALL_SPACING);
            ui.label("BTC");
            response
        }).inner
    }
}
