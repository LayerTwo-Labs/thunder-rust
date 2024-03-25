use eframe::egui;
use strum::{EnumIter, IntoEnumIterator};

use crate::{app::App, logs::LogsCapture};

mod block_explorer;
mod coins;
mod logs;
mod mempool_explorer;
mod miner;
mod parent_chain;
mod seed;
mod util;
mod withdrawals;

use block_explorer::BlockExplorer;
use coins::Coins;
use logs::Logs;
use mempool_explorer::MemPoolExplorer;
use miner::Miner;
use parent_chain::ParentChain;
use seed::SetSeed;
use withdrawals::Withdrawals;

pub struct EguiApp {
    app: App,
    block_explorer: BlockExplorer,
    coins: Coins,
    logs: Logs,
    mempool_explorer: MemPoolExplorer,
    miner: Miner,
    parent_chain: ParentChain,
    set_seed: SetSeed,
    tab: Tab,
    withdrawals: Withdrawals,
}

#[derive(Default, EnumIter, Eq, PartialEq, strum::Display)]
enum Tab {
    #[default]
    #[strum(to_string = "Parent Chain")]
    ParentChain,
    #[strum(to_string = "Coins")]
    Coins,
    #[strum(to_string = "Mempool Explorer")]
    MemPoolExplorer,
    #[strum(to_string = "Block Explorer")]
    BlockExplorer,
    #[strum(to_string = "Withdrawals")]
    Withdrawals,
    #[strum(to_string = "Logs")]
    Logs,
}

impl EguiApp {
    pub fn new(
        app: App,
        _cc: &eframe::CreationContext<'_>,
        logs_capture: LogsCapture,
    ) -> Self {
        // Customize egui here with cc.egui_ctx.set_fonts and cc.egui_ctx.set_visuals.
        // Restore app state using cc.storage (requires the "persistence" feature).
        // Use the cc.gl (a glow::Context) to create graphics shaders and buffers that you can use
        // for e.g. egui::PaintCallback.
        let coins = Coins::new(&app);
        let height = app.node.get_height().unwrap_or(0);
        let parent_chain = ParentChain::new(&app);
        Self {
            app,
            block_explorer: BlockExplorer::new(height),
            coins,
            logs: Logs::new(logs_capture),
            mempool_explorer: MemPoolExplorer::default(),
            miner: Miner::default(),
            parent_chain,
            set_seed: SetSeed::default(),
            tab: Tab::default(),
            withdrawals: Withdrawals::default(),
        }
    }

    fn bottom_panel_content(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            // Fill center space,
            // see https://github.com/emilk/egui/discussions/3908#discussioncomment-8270353

            // this frame target width
            // == this frame initial max rect width - last frame others width
            let id_cal_target_size = egui::Id::new("cal_target_size");
            let this_init_max_width = ui.max_rect().width();
            let last_others_width = ui.data(|data| {
                data.get_temp(id_cal_target_size)
                    .unwrap_or(this_init_max_width)
            });
            // this is the total available space for expandable widgets, you can divide
            // it up if you have multiple widgets to expand, even with different ratios.
            let this_target_width = this_init_max_width - last_others_width;

            ui.add_space(this_target_width);
            ui.separator();
            self.miner.show(&self.app, ui);
            // this frame others width
            // == this frame final min rect width - this frame target width
            ui.data_mut(|data| {
                data.insert_temp(
                    id_cal_target_size,
                    ui.min_rect().width() - this_target_width,
                )
            });
        });
    }
}

impl eframe::App for EguiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.app.wallet.has_seed().unwrap_or(false) {
            egui::TopBottomPanel::top("tabs").show(ctx, |ui| {
                ui.horizontal(|ui| {
                    Tab::iter().for_each(|tab_variant| {
                        let tab_name = tab_variant.to_string();
                        ui.selectable_value(
                            &mut self.tab,
                            tab_variant,
                            tab_name,
                        );
                    })
                });
            });
            egui::TopBottomPanel::bottom("util")
                .show(ctx, |ui| self.bottom_panel_content(ui));
            egui::CentralPanel::default().show(ctx, |ui| match self.tab {
                Tab::ParentChain => self.parent_chain.show(&mut self.app, ui),
                Tab::Coins => {
                    self.coins.show(&mut self.app, ui);
                }
                Tab::MemPoolExplorer => {
                    self.mempool_explorer.show(&mut self.app, ui);
                }
                Tab::BlockExplorer => {
                    self.block_explorer.show(&mut self.app, ui);
                }
                Tab::Withdrawals => {
                    self.withdrawals.show(&mut self.app, ui);
                }
                Tab::Logs => {
                    self.logs.show(ui);
                }
            });
        } else {
            egui::CentralPanel::default().show(ctx, |_ui| {
                egui::Window::new("Set Seed").show(ctx, |ui| {
                    self.set_seed.show(&self.app, ui);
                });
            });
        }
    }
}
