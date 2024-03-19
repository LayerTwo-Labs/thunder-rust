use std::sync::{
    atomic::{self, AtomicBool},
    Arc,
};

use eframe::egui::{self, Button};

use crate::app::App;

#[derive(Debug)]
pub struct Miner {
    running: Arc<AtomicBool>,
}

impl Default for Miner {
    fn default() -> Self {
        Self {
            running: Arc::new(AtomicBool::new(false)),
        }
    }
}

impl Miner {
    pub fn show(&mut self, app: &App, ui: &mut egui::Ui) {
        let block_height = app.node.get_height().unwrap_or(0);
        let best_hash = app.node.get_best_hash().unwrap_or([0; 32].into());
        ui.label("Block height: ");
        ui.monospace(format!("{block_height}"));
        ui.label("Best hash: ");
        let best_hash = &format!("{best_hash}")[0..8];
        ui.monospace(format!("{best_hash}..."));
        let running = self.running.load(atomic::Ordering::SeqCst);
        if ui.add_enabled(!running, Button::new("Mine")).clicked() {
            self.running.store(true, atomic::Ordering::SeqCst);
            app.runtime.spawn({
                let app = app.clone();
                let running = self.running.clone();
                async move {
                    tracing::debug!("Mining...");
                    let mining_result = app.mine(None).await;
                    running.store(false, atomic::Ordering::SeqCst);
                    if let Err(err) = mining_result {
                        tracing::error!("{:#}", anyhow::Error::new(err))
                    }
                }
            });
        }
    }
}
