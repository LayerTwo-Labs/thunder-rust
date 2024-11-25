use eframe::egui;
use thunder::types::GetValue;

use crate::app::App;

#[derive(Default)]
pub struct Withdrawals {}

impl Withdrawals {
    pub fn show(&mut self, app: Option<&App>, ui: &mut egui::Ui) {
        ui.heading("Pending withdrawals");
        let bundle = app.and_then(|app| {
            app.node.get_pending_withdrawal_bundle().ok().flatten()
        });
        if let Some(bundle) = bundle {
            let mut spent_utxos: Vec<_> = bundle.spend_utxos().iter().collect();
            spent_utxos.sort_by_key(|(outpoint, _)| format!("{outpoint}"));
            egui::Grid::new("bundle_utxos")
                .striped(true)
                .show(ui, |ui| {
                    for (outpoint, output) in &spent_utxos {
                        ui.monospace(format!("{outpoint}"));
                        ui.monospace(format!("{}", output.get_value()));
                        ui.end_row();
                    }
                });
        } else {
            ui.label("No pending bundle");
        }
    }
}
