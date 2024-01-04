use crate::app::lib;
use crate::app::App;
use eframe::egui;
use lib::bip300301::bitcoin;
use lib::types::GetValue;

#[derive(Default)]
pub struct Withdrawals {}

impl Withdrawals {
    pub fn show(&mut self, app: &mut App, ui: &mut egui::Ui) {
        ui.heading("Pending withdrawals");
        let bundle = app.node.get_pending_withdrawal_bundle().ok().flatten();
        if let Some(bundle) = bundle {
            let mut spent_utxos: Vec<_> = bundle.spent_utxos.iter().collect();
            spent_utxos.sort_by_key(|(outpoint, _)| format!("{outpoint}"));
            egui::Grid::new("bundle_utxos")
                .striped(true)
                .show(ui, |ui| {
                    for (outpoint, output) in &spent_utxos {
                        ui.monospace(format!("{outpoint}"));
                        ui.monospace(format!("{}", bitcoin::Amount::from_sat(output.get_value())));
                        ui.end_row();
                    }
                });
        } else {
            ui.label("No pending bundle");
        }
    }
}
