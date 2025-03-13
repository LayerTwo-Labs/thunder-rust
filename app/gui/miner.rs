use std::sync::{
    Arc,
    atomic::{self, AtomicBool},
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
    pub fn show(&mut self, app: Option<&App>, ui: &mut egui::Ui) {
        let block_height = app
            .and_then(|app| app.node.try_get_height().ok().flatten())
            .unwrap_or(0);
        let best_hash = app
            .and_then(|app| app.node.try_get_best_hash().ok().flatten())
            .unwrap_or([0; 32].into());
        ui.label("Block height: ");
        ui.monospace(format!("{block_height}"));
        ui.label("Best hash: ");
        let best_hash = &format!("{best_hash}")[0..8];
        ui.monospace(format!("{best_hash}..."));
        let running = self.running.load(atomic::Ordering::SeqCst);
        if let Some(app) = app
            && ui
                .add_enabled(!running, Button::new("Mine / Refresh Block"))
                .clicked()
        {
            self.running.store(true, atomic::Ordering::SeqCst);
            app.local_pool.spawn_pinned({
                let app = app.clone();
                let running = self.running.clone();
                || async move {
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
