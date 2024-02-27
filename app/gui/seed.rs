use crate::app::App;
use eframe::egui;

pub struct SetSeed {
    seed: String,
    passphrase: String,
}

impl Default for SetSeed {
    fn default() -> Self {
        Self {
            seed: "".into(),
            passphrase: "".into(),
        }
    }
}

impl SetSeed {
    pub fn show(&mut self, app: &App, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let seed_edit = egui::TextEdit::singleline(&mut self.seed)
                .hint_text("seed")
                .clip_text(false);
            ui.add(seed_edit);
            if ui.button("generate").clicked() {
                let mnemonic = bip39::Mnemonic::new(
                    bip39::MnemonicType::Words12,
                    bip39::Language::English,
                );
                self.seed = mnemonic.phrase().into();
            }
        });
        let passphrase_edit = egui::TextEdit::singleline(&mut self.passphrase)
            .hint_text("passphrase")
            .password(true)
            .clip_text(false);
        ui.add(passphrase_edit);
        let mnemonic =
            bip39::Mnemonic::from_phrase(&self.seed, bip39::Language::English);
        if ui
            .add_enabled(mnemonic.is_ok(), egui::Button::new("set"))
            .clicked()
        {
            let mnemonic = mnemonic.expect("should never happen");
            let seed = bip39::Seed::new(&mnemonic, &self.passphrase);
            app.wallet
                .set_seed(
                    seed.as_bytes().try_into().expect("seed it not 64 bytes"),
                )
                .expect("failed to set HD wallet seed");
        }
    }
}
