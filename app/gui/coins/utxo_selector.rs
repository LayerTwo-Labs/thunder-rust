use std::collections::HashSet;

use eframe::egui;
use thunder::types::{
    hash, GetValue, OutPoint, Output, PointedOutput, Transaction,
};

use crate::app::App;

#[derive(Debug, Default)]
pub struct UtxoSelector;

impl UtxoSelector {
    pub fn show(
        &mut self,
        app: &mut App,
        ui: &mut egui::Ui,
        tx: &mut Transaction,
    ) {
        ui.heading("Spend UTXO");
        let selected: HashSet<_> =
            tx.inputs.iter().map(|(outpoint, _)| *outpoint).collect();
        let utxos_read = app.utxos.read();
        let total: u64 = utxos_read
            .iter()
            .filter(|(outpoint, _)| !selected.contains(outpoint))
            .map(|(_, output)| output.get_value())
            .sum();
        let mut utxos: Vec<_> = (*utxos_read).clone().into_iter().collect();
        drop(utxos_read);
        utxos.sort_by_key(|(outpoint, _)| format!("{outpoint}"));
        ui.separator();
        ui.monospace(format!("Total: {}", bitcoin::Amount::from_sat(total)));
        ui.separator();
        egui::Grid::new("utxos").striped(true).show(ui, |ui| {
            ui.monospace("kind");
            ui.monospace("outpoint");
            ui.monospace("value");
            ui.end_row();
            for (outpoint, output) in utxos {
                if selected.contains(&outpoint) {
                    continue;
                }
                //ui.horizontal(|ui| {});
                show_utxo(ui, &outpoint, &output);

                if ui
                    .add_enabled(
                        !selected.contains(&outpoint),
                        egui::Button::new("spend"),
                    )
                    .clicked()
                {
                    let utxo_hash = hash(&PointedOutput {
                        outpoint,
                        output: output.clone(),
                    });
                    tx.inputs.push((outpoint, utxo_hash));
                }
                ui.end_row();
            }
        });
    }
}

pub fn show_utxo(ui: &mut egui::Ui, outpoint: &OutPoint, output: &Output) {
    let (kind, hash, vout) = match outpoint {
        OutPoint::Regular { txid, vout } => {
            ("regular", format!("{txid}"), *vout)
        }
        OutPoint::Deposit(outpoint) => {
            ("deposit", format!("{}", outpoint.txid), outpoint.vout)
        }
        OutPoint::Coinbase { merkle_root, vout } => {
            ("coinbase", format!("{merkle_root}"), *vout)
        }
    };
    let hash = &hash[0..8];
    let value = bitcoin::Amount::from_sat(output.get_value());
    ui.monospace(kind.to_string());
    ui.monospace(format!("{hash}:{vout}",));
    ui.with_layout(egui::Layout::right_to_left(egui::Align::Max), |ui| {
        ui.monospace(format!("{value}"));
    });
}
