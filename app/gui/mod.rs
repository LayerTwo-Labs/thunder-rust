use std::{net::SocketAddr, task::Poll};

use eframe::egui;
use strum::{EnumIter, IntoEnumIterator};
use thunder::{util::Watchable, wallet::Wallet};

use crate::{app::App, line_buffer::LineBuffer, util::PromiseStream};

mod block_explorer;
mod coins;
mod console_logs;
mod mempool_explorer;
mod miner;
mod parent_chain;
mod seed;
mod util;
mod withdrawals;

use block_explorer::BlockExplorer;
use coins::Coins;
use console_logs::ConsoleLogs;
use mempool_explorer::MemPoolExplorer;
use miner::Miner;
use parent_chain::ParentChain;
use seed::SetSeed;
use withdrawals::Withdrawals;

use self::util::UiExt;

pub struct EguiApp {
    app: App,
    block_explorer: BlockExplorer,
    bottom_panel: BottomPanel,
    coins: Coins,
    console_logs: ConsoleLogs,
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
    #[strum(to_string = "Console / Logs")]
    ConsoleLogs,
}

struct BottomPanel {
    wallet_updated: PromiseStream<<Wallet as Watchable<()>>::WatchStream>,
    /// None if uninitialized
    /// Some(None) if failed to initialize
    balance: Option<Option<u64>>,
}

impl BottomPanel {
    /// MUST be run from within a tokio runtime
    fn new(wallet: &Wallet) -> Self {
        let wallet_updated = PromiseStream::from(wallet.watch());
        Self {
            wallet_updated,
            balance: None,
        }
    }

    /// Updates values
    fn update(&mut self, app: &App) {
        self.balance = match app.wallet.get_balance() {
            Ok(balance) => Some(Some(balance)),
            Err(err) => {
                let err = anyhow::Error::from(err);
                tracing::error!("Failed to update balance: {err:#}");
                Some(None)
            }
        }
    }

    fn show_balance(&self, ui: &mut egui::Ui) {
        match self.balance {
            Some(Some(balance)) => {
                ui.monospace_selectable_singleline(
                    false,
                    format!("Balance: {balance}"),
                );
            }
            Some(None) => {
                ui.monospace_selectable_singleline(
                    false,
                    "Balance error, check logs",
                );
            }
            None => {
                ui.monospace_selectable_singleline(false, "Loading balance");
            }
        }
    }

    fn show(&mut self, app: &App, miner: &mut Miner, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let rt_guard = app.runtime.enter();
            match self.wallet_updated.poll_next() {
                Some(Poll::Ready(())) => {
                    self.update(app);
                }
                Some(Poll::Pending) | None => (),
            }
            drop(rt_guard);
            self.show_balance(ui);
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
            miner.show(app, ui);
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

impl EguiApp {
    pub fn new(
        app: App,
        _cc: &eframe::CreationContext<'_>,
        logs_capture: LineBuffer,
        rpc_addr: SocketAddr,
    ) -> Self {
        // Customize egui here with cc.egui_ctx.set_fonts and cc.egui_ctx.set_visuals.
        // Restore app state using cc.storage (requires the "persistence" feature).
        // Use the cc.gl (a glow::Context) to create graphics shaders and buffers that you can use
        // for e.g. egui::PaintCallback.
        let rt_guard = app.runtime.enter();
        let bottom_panel = BottomPanel::new(&app.wallet);
        drop(rt_guard);
        let coins = Coins::new(&app);
        let console_logs = ConsoleLogs::new(logs_capture, rpc_addr);
        let height = app.node.get_height().unwrap_or(0);
        let parent_chain = ParentChain::new(&app);
        Self {
            app,
            block_explorer: BlockExplorer::new(height),
            bottom_panel,
            coins,
            console_logs,
            mempool_explorer: MemPoolExplorer::default(),
            miner: Miner::default(),
            parent_chain,
            set_seed: SetSeed::default(),
            tab: Tab::default(),
            withdrawals: Withdrawals::default(),
        }
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
            egui::TopBottomPanel::bottom("bottom_panel").show(ctx, |ui| {
                self.bottom_panel.show(&self.app, &mut self.miner, ui)
            });
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
                Tab::ConsoleLogs => {
                    self.console_logs.show(&self.app, ui);
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
