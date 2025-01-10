//! Load fonts for egui

use std::sync::{Arc, LazyLock};

use eframe::egui::{FontData, FontDefinitions, FontFamily};

static FIRA_MONO_NERD_REGULAR: &[u8] = include_path::include_path_bytes!(
    "../../res/nerd-fonts/patched-fonts/FiraMono/Regular/FiraMonoNerdFont-Regular.otf"
);

static NOTO_SANS_MONO_NERD_REGULAR: &[u8] = include_path::include_path_bytes!(
    "../../res/nerd-fonts/patched-fonts/Noto/Sans-Mono/NotoSansMNerdFont-Regular.ttf"
);

pub static FONT_DEFINITIONS: LazyLock<FontDefinitions> = LazyLock::new(|| {
    let mut fonts = FontDefinitions::default();
    // Install fonts
    fonts.font_data.insert(
        "Fira Mono Nerd Regular".to_owned(),
        Arc::new(FontData::from_static(FIRA_MONO_NERD_REGULAR)),
    );
    fonts.font_data.insert(
        "Noto Sans Mono Nerd Regular".to_owned(),
        Arc::new(FontData::from_static(NOTO_SANS_MONO_NERD_REGULAR)),
    );
    // Set Fira Mono Nerd Regular as first monospace font
    fonts
        .families
        .get_mut(&FontFamily::Monospace)
        .unwrap()
        .insert(0, "Fira Mono Nerd Regular".to_owned());
    // Set Noto Sans Mono Nerd Regular as second monospace font
    fonts
        .families
        .get_mut(&FontFamily::Monospace)
        .unwrap()
        .insert(1, "Noto Sans Mono Nerd Regular".to_owned());
    fonts
});
