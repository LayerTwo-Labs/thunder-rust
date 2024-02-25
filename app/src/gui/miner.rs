use crate::app::App;
use eframe::egui;

pub struct Miner;

impl Default for Miner {
    fn default() -> Self {
        Self
    }
}

impl Miner {
    pub fn show(&mut self, app: &mut App, ui: &mut egui::Ui) {
        let block_height = app.node.get_height().unwrap_or(0);
        let best_hash = app.node.get_best_hash().unwrap_or([0; 32].into());
        ui.label("Block height: ");
        ui.monospace(format!("{block_height}"));
        ui.label("Best hash: ");
        let best_hash = &format!("{best_hash}")[0..8];
        ui.monospace(format!("{best_hash}..."));
        if ui.button("mine").clicked() {
            let _result = app.mine();
        }
    }
}
