use eframe::egui::{self, Button};
use thunder::types::{self, Output, OutputContent, Transaction};

use crate::app::App;

#[derive(Debug, Eq, PartialEq)]
enum UtxoType {
    Regular,
    Withdrawal,
}

impl std::fmt::Display for UtxoType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Regular => write!(f, "regular"),
            Self::Withdrawal => write!(f, "withdrawal"),
        }
    }
}

#[derive(Debug)]
pub struct UtxoCreator {
    utxo_type: UtxoType,
    value: String,
    address: String,
    main_address: String,
    main_fee: String,
}

impl Default for UtxoCreator {
    fn default() -> Self {
        Self {
            value: "".into(),
            address: "".into(),
            main_address: "".into(),
            main_fee: "".into(),
            utxo_type: UtxoType::Regular,
        }
    }
}

impl UtxoCreator {
    pub fn show(
        &mut self,
        app: Option<&App>,
        ui: &mut egui::Ui,
        tx: &mut Transaction,
    ) {
        ui.horizontal(|ui| {
            ui.heading("Create");
            egui::ComboBox::from_id_salt("utxo_type")
                .selected_text(format!("{}", self.utxo_type))
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut self.utxo_type,
                        UtxoType::Regular,
                        "regular",
                    );
                    ui.selectable_value(
                        &mut self.utxo_type,
                        UtxoType::Withdrawal,
                        "withdrawal",
                    );
                });
            ui.heading("UTXO");
        });
        ui.separator();
        ui.horizontal(|ui| {
            ui.monospace("Value:       ");
            ui.add(egui::TextEdit::singleline(&mut self.value));
            ui.monospace("BTC");
        });
        ui.horizontal(|ui| {
            ui.monospace("Address:     ");
            ui.add(egui::TextEdit::singleline(&mut self.address));
            if ui
                .add_enabled(app.is_some(), Button::new("generate"))
                .clicked()
            {
                self.address = app
                    .unwrap()
                    .wallet
                    .get_new_address()
                    .map(|address| format!("{address}"))
                    .unwrap_or("".into());
            }
        });
        if self.utxo_type == UtxoType::Withdrawal {
            ui.horizontal(|ui| {
                ui.monospace("Main Address:");
                ui.add(egui::TextEdit::singleline(&mut self.main_address));
                if ui
                    .add_enabled(app.is_some(), Button::new("generate"))
                    .clicked()
                {
                    match app.unwrap().get_new_main_address() {
                        Ok(main_address) => {
                            self.main_address = format!("{main_address}");
                        }
                        Err(err) => {
                            let err = anyhow::Error::new(err);
                            tracing::error!("{err:#}")
                        }
                    };
                }
            });
            ui.horizontal(|ui| {
                ui.monospace("Main Fee:    ");
                ui.add(egui::TextEdit::singleline(&mut self.main_fee));
                ui.monospace("BTC");
            });
        }
        ui.horizontal(|ui| {
            match self.utxo_type {
                UtxoType::Regular => {
                    let address: Option<types::Address> =
                        self.address.parse().ok();
                    let value: Option<bitcoin::Amount> =
                        bitcoin::Amount::from_str_in(
                            &self.value,
                            bitcoin::Denomination::Bitcoin,
                        )
                        .ok();
                    if ui
                        .add_enabled(
                            address.is_some() && value.is_some(),
                            egui::Button::new("create"),
                        )
                        .clicked()
                    {
                        let utxo = Output {
                            address: address.expect("should not happen"),
                            content: OutputContent::Value(
                                value.expect("should not happen"),
                            ),
                        };
                        tx.outputs.push(utxo);
                    }
                }
                UtxoType::Withdrawal => {
                    let value: Option<bitcoin::Amount> =
                        bitcoin::Amount::from_str_in(
                            &self.value,
                            bitcoin::Denomination::Bitcoin,
                        )
                        .ok();
                    let address: Option<types::Address> =
                        self.address.parse().ok();
                    let main_address: Option<
                        bitcoin::Address<bitcoin::address::NetworkUnchecked>,
                    > = self.main_address.parse().ok();
                    let main_fee: Option<bitcoin::Amount> =
                        bitcoin::Amount::from_str_in(
                            &self.main_fee,
                            bitcoin::Denomination::Bitcoin,
                        )
                        .ok();
                    if ui
                        .add_enabled(
                            value.is_some()
                                && address.is_some()
                                && main_address.is_some()
                                && main_fee.is_some(),
                            egui::Button::new("create"),
                        )
                        .clicked()
                    {
                        let utxo = Output {
                            address: address.expect("invalid address"),
                            content: OutputContent::Withdrawal {
                                value: value.expect("invalid value"),
                                main_address: main_address
                                    .expect("invalid main_address"),
                                main_fee: main_fee.expect("invalid main_fee"),
                            },
                        };
                        tx.outputs.push(utxo);
                    }
                }
            }
            if let Some(app) = app {
                let num_addresses = app.wallet.get_num_addresses().unwrap();
                ui.label(format!("{num_addresses} addresses generated"));
            }
        });
    }
}
