use eframe::egui;
use human_size::{Byte, Kibibyte, Mebibyte, SpecificSize};
use thunder::types::{GetValue, OutPoint};

use crate::app::App;

#[derive(Default)]
pub struct MemPoolExplorer {
    current: usize,
}

impl MemPoolExplorer {
    pub fn show(&mut self, app: Option<&App>, ui: &mut egui::Ui) {
        let transactions = app
            .and_then(|app| app.node.get_all_transactions().ok())
            .unwrap_or_default();
        let utxos = app
            .and_then(|app| app.wallet.get_utxos().ok())
            .unwrap_or_default();
        egui::SidePanel::left("transaction_picker")
            .resizable(false)
            .show_inside(ui, |ui| {
                ui.heading("Transactions");
                ui.separator();
                egui::Grid::new("transactions")
                    .striped(true)
                    .show(ui, |ui| {
                        ui.monospace("txid");
                        ui.monospace("value out");
                        ui.monospace("fee");
                        ui.end_row();
                        for (index, transaction) in
                            transactions.iter().enumerate()
                        {
                            let value_out: bitcoin::Amount = transaction
                                .transaction
                                .outputs
                                .iter()
                                .map(GetValue::get_value)
                                .sum();
                            let value_in: bitcoin::Amount = transaction
                                .transaction
                                .inputs
                                .iter()
                                .map(|(outpoint, _)| {
                                    utxos.get(outpoint).map(GetValue::get_value)
                                })
                                .sum::<Option<bitcoin::Amount>>()
                                .unwrap_or(bitcoin::Amount::ZERO);
                            let txid =
                                &format!("{}", transaction.transaction.txid())
                                    [0..8];
                            if value_in >= value_out {
                                let fee = value_in - value_out;
                                ui.selectable_value(
                                    &mut self.current,
                                    index,
                                    txid.to_string(),
                                );
                                ui.with_layout(
                                    egui::Layout::right_to_left(
                                        egui::Align::Max,
                                    ),
                                    |ui| {
                                        ui.monospace(format!("{value_out}"));
                                    },
                                );
                                ui.with_layout(
                                    egui::Layout::right_to_left(
                                        egui::Align::Max,
                                    ),
                                    |ui| {
                                        ui.monospace(format!("{fee}"));
                                    },
                                );
                                ui.end_row();
                            } else {
                                ui.selectable_value(
                                    &mut self.current,
                                    index,
                                    txid.to_string(),
                                );
                                ui.monospace("invalid");
                                ui.end_row();
                            }
                        }
                    });
            });
        if let Some(transaction) = transactions.get(self.current) {
            egui::SidePanel::left("inputs")
                .resizable(false)
                .show_inside(ui, |ui| {
                    ui.heading("Inputs");
                    ui.separator();
                    egui::Grid::new("inputs").striped(true).show(ui, |ui| {
                        ui.monospace("kind");
                        ui.monospace("outpoint");
                        ui.monospace("value");
                        ui.end_row();
                        for (outpoint, _) in &transaction.transaction.inputs {
                            let (kind, hash, vout) = match outpoint {
                                OutPoint::Regular { txid, vout } => {
                                    ("regular", format!("{txid}"), *vout)
                                }
                                OutPoint::Deposit(outpoint) => (
                                    "deposit",
                                    format!("{}", outpoint.txid),
                                    outpoint.vout,
                                ),
                                OutPoint::Coinbase { merkle_root, vout } => (
                                    "coinbase",
                                    format!("{merkle_root}"),
                                    *vout,
                                ),
                            };
                            let output = &utxos[outpoint];
                            let hash = &hash[0..8];
                            let value = output.get_value();
                            ui.monospace(kind.to_string());
                            ui.monospace(format!("{hash}:{vout}",));
                            ui.monospace(format!("{value}",));
                            ui.end_row();
                        }
                    });
                });
            egui::SidePanel::left("outputs")
                .resizable(false)
                .show_inside(ui, |ui| {
                    ui.heading("Outputs");
                    ui.separator();
                    egui::Grid::new("inputs").striped(true).show(ui, |ui| {
                        ui.monospace("vout");
                        ui.monospace("address");
                        ui.monospace("value");
                        ui.end_row();
                        for (vout, output) in
                            transaction.transaction.outputs.iter().enumerate()
                        {
                            let address = &format!("{}", output.address)[0..8];
                            let value = output.get_value();
                            ui.monospace(format!("{vout}"));
                            ui.monospace(address.to_string());
                            ui.monospace(format!("{value}"));
                            ui.end_row();
                        }
                    });
                });
            egui::CentralPanel::default().show_inside(ui, |ui| {
                ui.heading("Viewing");
                ui.separator();
                let txid = transaction.transaction.txid();
                ui.monospace(format!("Txid:             {txid}"));
                let transaction_size =
                    bincode::serialize(&transaction).unwrap_or(vec![]).len();
                let transaction_size = if let Ok(transaction_size) =
                    SpecificSize::new(transaction_size as f64, Byte)
                {
                    let bytes = transaction_size.to_bytes();
                    if bytes < 1024 {
                        format!("{transaction_size}")
                    } else if bytes < 1024 * 1024 {
                        let transaction_size: SpecificSize<Kibibyte> =
                            transaction_size.into();
                        format!("{transaction_size}")
                    } else {
                        let transaction_size: SpecificSize<Mebibyte> =
                            transaction_size.into();
                        format!("{transaction_size}")
                    }
                } else {
                    "".into()
                };
                ui.monospace(format!("Transaction size: {transaction_size}"));
            });
        } else {
            egui::CentralPanel::default().show_inside(ui, |ui| {
                ui.heading("No transactions in mempool");
            });
        }
    }
}
