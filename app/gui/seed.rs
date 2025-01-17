use crate::app::App;
use eframe::egui;
use serde::Deserialize;
use thunder::types::THIS_SIDECHAIN;

pub const CIRCLE_ARROW: char = '\u{27f3}';

#[derive(Debug, Deserialize)]
struct StarterFile {
    mnemonic: String,
}

impl StarterFile {
    fn validate(&self) -> bool {
        bip39::Mnemonic::from_phrase(&self.mnemonic, bip39::Language::English)
            .is_ok()
    }
}

pub struct SetSeed {
    seed: String,
    passphrase: String,
    has_starter: bool,
    initial_check_done: bool,
}

impl Default for SetSeed {
    fn default() -> Self {
        Self {
            seed: "".into(),
            passphrase: "".into(),
            has_starter: false,
            initial_check_done: false,
        }
    }
}

impl SetSeed {
    pub fn new() -> Self {
        Self {
            seed: "".into(),
            passphrase: "".into(),
            has_starter: false,
            initial_check_done: false,
        }
    }

    fn check_starter_file(&mut self) {
        let app_dir = dirs::data_dir()
            .map(|dir| dir.join("cusf_launcher").join("wallet_starters"));

        self.has_starter = if let Some(dir) = app_dir {
            if !dir.exists() {
                if !self.initial_check_done {
                    tracing::debug!("No starter file found: directory does not exist at {:?}", dir);
                }
                false
            } else {
                let starter_file = dir
                    .join(format!("sidechain_{}_starter.json", THIS_SIDECHAIN));
                let exists = starter_file.exists();
                if !exists && !self.initial_check_done {
                    tracing::debug!(
                        "No starter file found at {:?}",
                        starter_file
                    );
                }
                exists
            }
        } else {
            if !self.initial_check_done {
                tracing::debug!("No starter file found: could not determine app data directory");
            }
            false
        };
        self.initial_check_done = true;
    }

    fn load_starter_file(&self) -> Option<StarterFile> {
        let app_dir = dirs::data_dir()?
            .join("cusf_launcher")
            .join("wallet_starters");

        let starter_file =
            app_dir.join(format!("sidechain_{}_starter.json", THIS_SIDECHAIN));

        let content = match std::fs::read_to_string(&starter_file) {
            Ok(content) => content,
            Err(err) => {
                tracing::error!("Failed to read starter file: {}", err);
                return None;
            }
        };

        match serde_json::from_str::<StarterFile>(&content) {
            Ok(starter) if starter.validate() => Some(starter),
            Ok(_) => {
                tracing::error!("Invalid mnemonic in starter file");
                None
            }
            Err(err) => {
                tracing::error!("Failed to parse starter file JSON: {}", err);
                None
            }
        }
    }

    pub fn show(&mut self, app: &App, ui: &mut egui::Ui) {
        if !self.initial_check_done {
            self.check_starter_file();
        }

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

            ui.horizontal(|ui| {
                if ui
                    .add_enabled(
                        self.has_starter,
                        egui::Button::new("use starter"),
                    )
                    .clicked()
                {
                    if let Some(starter) = self.load_starter_file() {
                        self.seed = starter.mnemonic;
                    }
                }
                if ui.small_button(CIRCLE_ARROW.to_string()).clicked() {
                    self.initial_check_done = false;
                    self.check_starter_file();
                }
            });
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
            app.wallet
                .set_seed_from_mnemonic(self.seed.as_str())
                .expect("failed to set HD wallet seed");
        }
    }
}
