use std::collections::HashSet;

use eframe::egui;

use photon::types::{GetValue, Transaction};

use super::{
    tx_creator::TxCreator,
    utxo_creator::UtxoCreator,
    utxo_selector::{UtxoSelector, show_utxo},
};
use crate::app::App;

#[derive(Debug, Default)]
pub struct TxBuilder {
    // regular tx without extra data or special inputs/outputs
    base_tx: Transaction,
    tx_creator: TxCreator,
    utxo_creator: UtxoCreator,
    utxo_selector: UtxoSelector,
}

impl TxBuilder {
    pub fn show_value_in(&mut self, app: Option<&App>, ui: &mut egui::Ui) {
        ui.heading("Value In");
        let Some(app) = app else {
            return;
        };
        let selected: HashSet<_> = self
            .base_tx
            .inputs
            .iter()
            .map(|(outpoint, _)| *outpoint)
            .collect();
        let utxos_read = app.utxos.read();
        let mut spent_utxos: Vec<_> = utxos_read
            .iter()
            .filter(|(outpoint, _)| selected.contains(outpoint))
            .collect();
        let value_in: bitcoin::Amount = spent_utxos
            .iter()
            .map(|(_, output)| output.get_value())
            .sum();
        self.tx_creator.value_in = value_in;
        spent_utxos.sort_by_key(|(outpoint, _)| {
            photon::types::OutPointKey::from(*outpoint)
        });
        ui.separator();
        ui.monospace(format!("Total: {value_in}"));
        ui.separator();
        egui::Grid::new("spent_utxos").striped(true).show(ui, |ui| {
            ui.monospace("kind");
            ui.monospace("outpoint");
            ui.monospace("value");
            ui.end_row();
            let mut remove = None;
            for (vout, (outpoint, _)) in self.base_tx.inputs.iter().enumerate()
            {
                let output = &utxos_read[outpoint];
                show_utxo(ui, outpoint, output);
                if ui.button("remove").clicked() {
                    remove = Some(vout);
                }
                ui.end_row();
            }
            if let Some(vout) = remove {
                self.base_tx.inputs.remove(vout);
            }
        });
    }

    pub fn show_value_out(&mut self, ui: &mut egui::Ui) {
        ui.heading("Value Out");
        ui.separator();
        let value_out: bitcoin::Amount =
            self.base_tx.outputs.iter().map(GetValue::get_value).sum();
        self.tx_creator.value_out = value_out;
        ui.monospace(format!("Total: {value_out}"));
        ui.separator();
        egui::Grid::new("outputs").striped(true).show(ui, |ui| {
            let mut remove = None;
            ui.monospace("vout");
            ui.monospace("address");
            ui.monospace("value");
            ui.end_row();
            for (vout, output) in self.base_tx.outputs.iter().enumerate() {
                let address = &format!("{}", output.address)[0..8];
                let value = output.get_value();
                ui.monospace(format!("{vout}"));
                ui.monospace(address.to_string());
                ui.with_layout(
                    egui::Layout::right_to_left(egui::Align::Max),
                    |ui| {
                        ui.monospace(format!("{value}"));
                    },
                );
                if ui.button("remove").clicked() {
                    remove = Some(vout);
                }
                ui.end_row();
            }
            if let Some(vout) = remove {
                self.base_tx.outputs.remove(vout);
            }
        });
    }

    pub fn show(
        &mut self,
        app: Option<&App>,
        ui: &mut egui::Ui,
    ) -> anyhow::Result<()> {
        egui::ScrollArea::horizontal().show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.set_width(250.0);
                    self.utxo_selector.show(app, ui, &mut self.base_tx);
                });
                ui.separator();
                ui.vertical(|ui| {
                    ui.set_width(250.0);
                    let () = self.show_value_in(app, ui);
                });
                ui.separator();
                ui.vertical(|ui| {
                    ui.set_width(250.0);
                    let () = self.show_value_out(ui);
                });
                ui.separator();
                ui.vertical(|ui| {
                    ui.set_width(450.0);
                    self.utxo_creator.show(app, ui, &mut self.base_tx);
                    ui.separator();
                    self.tx_creator.show(app, ui, &mut self.base_tx).unwrap();
                });
            });
        });
        Ok(())
    }
}
