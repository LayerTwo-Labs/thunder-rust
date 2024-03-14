use std::borrow::Borrow;

use eframe::egui::{self, Response, Ui};

/// extension trait for egui::Ui
pub trait UiExt {
    fn monospace_selectable_singleline<Text>(
        &mut self,
        clip_text: bool,
        text: Text,
    ) -> Response
    where
        Text: Borrow<str>;

    fn monospace_selectable_multiline<Text>(&mut self, text: Text) -> Response
    where
        Text: Borrow<str>;
}

impl UiExt for Ui {
    fn monospace_selectable_singleline<Text>(
        &mut self,
        clip_text: bool,
        text: Text,
    ) -> Response
    where
        Text: Borrow<str>,
    {
        use egui::{TextEdit, TextStyle, Widget};
        let mut text: &str = text.borrow();
        TextEdit::singleline(&mut text)
            .font(TextStyle::Monospace)
            .clip_text(clip_text)
            .ui(self)
    }

    fn monospace_selectable_multiline<Text>(&mut self, text: Text) -> Response
    where
        Text: Borrow<str>,
    {
        use egui::{TextEdit, TextStyle, Widget};
        let mut text: &str = text.borrow();
        TextEdit::multiline(&mut text)
            .font(TextStyle::Monospace)
            .ui(self)
    }
}
