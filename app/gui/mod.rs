use std::task::Poll;

use eframe::egui::{self, RichText};
use strum::{EnumIter, IntoEnumIterator};
use thunder::{util::Watchable, wallet::Wallet};
use util::{BITCOIN_LOGO_FA, BITCOIN_ORANGE, show_btc_amount};

use crate::{app::App, line_buffer::LineBuffer, util::PromiseStream};

mod block_explorer;
mod coins;
mod console_logs;
mod fonts;
mod mempool_explorer;
mod miner;
mod parent_chain;
mod seed;
mod util;
mod withdrawals;

use block_explorer::BlockExplorer;
use coins::Coins;
use console_logs::ConsoleLogs;
use fonts::FONT_DEFINITIONS;
use mempool_explorer::MemPoolExplorer;
use miner::Miner;
use parent_chain::ParentChain;
use seed::SetSeed;
use withdrawals::Withdrawals;

use self::util::UiExt;

pub struct EguiApp {
    app: Option<App>,
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

/// Bottom panel, if initialized
struct BottomPanelInitialized {
    app: App,
    wallet_updated: PromiseStream<<Wallet as Watchable<()>>::WatchStream>,
}

impl BottomPanelInitialized {
    fn new(app: App) -> Self {
        let wallet_updated = {
            let rt_guard = app.runtime.enter();
            let wallet_updated = PromiseStream::from(app.wallet.watch());
            drop(rt_guard);
            wallet_updated
        };
        Self {
            app,
            wallet_updated,
        }
    }
}

struct BottomPanel {
    initialized: Option<BottomPanelInitialized>,
    /// None if uninitialized
    /// Some(None) if failed to initialize
    balance: Option<Option<bitcoin::Amount>>,
}

impl BottomPanel {
    /// MUST be run from within a tokio runtime
    fn new(app: Option<App>) -> Self {
        let initialized = app.map(BottomPanelInitialized::new);
        Self {
            initialized,
            balance: None,
        }
    }

    /// Updates values if the wallet has been updated
    fn update(&mut self) {
        let Some(initialized) = &mut self.initialized else {
            return;
        };
        let rt_guard = initialized.app.runtime.enter();
        match initialized.wallet_updated.poll_next() {
            Some(Poll::Ready(())) => {
                self.balance = match initialized.app.wallet.get_balance() {
                    Ok(balance) => Some(Some(balance.total)),
                    Err(err) => {
                        let err = anyhow::Error::from(err);
                        tracing::error!("Failed to update balance: {err:#}");
                        Some(None)
                    }
                }
            }
            Some(Poll::Pending) | None => (),
        }
        drop(rt_guard)
    }

    fn show_balance(&self, ui: &mut egui::Ui) {
        match self.balance {
            Some(Some(balance)) => {
                ui.monospace(
                    RichText::new(BITCOIN_LOGO_FA.to_string())
                        .color(BITCOIN_ORANGE),
                );
                ui.monospace_selectable_singleline(
                    false,
                    format!("Balance: {}", show_btc_amount(balance)),
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

    fn show(&mut self, miner: &mut Miner, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            self.update();
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
            miner.show(
                self.initialized
                    .as_ref()
                    .map(|initialized| &initialized.app),
                ui,
            );
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
        app: Option<App>,
        cc: &eframe::CreationContext<'_>,
        logs_capture: LineBuffer,
        rpc_addr: url::Url,
    ) -> Self {
        // Customize egui here with cc.egui_ctx.set_fonts and cc.egui_ctx.set_visuals.
        // Restore app state using cc.storage (requires the "persistence" feature).
        // Use the cc.gl (a glow::Context) to create graphics shaders and buffers that you can use
        // for e.g. egui::PaintCallback.
        cc.egui_ctx.set_fonts(FONT_DEFINITIONS.clone());
        let bottom_panel = BottomPanel::new(app.clone());
        let coins = Coins::new(app.as_ref());
        let console_logs = ConsoleLogs::new(logs_capture, rpc_addr);
        let height = app
            .as_ref()
            .and_then(|app| app.node.try_get_height().ok().flatten())
            .unwrap_or(0);
        let parent_chain = ParentChain::new(app.as_ref());
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
        ctx.request_repaint();
        if let Some(app) = self.app.as_ref()
            && !app.wallet.has_seed().unwrap_or(false)
        {
            egui::CentralPanel::default().show(ctx, |_ui| {
                egui::Window::new("Set Seed").show(ctx, |ui| {
                    self.set_seed.show(app, ui);
                });
            });
        } else {
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
            egui::TopBottomPanel::bottom("bottom_panel")
                .show(ctx, |ui| self.bottom_panel.show(&mut self.miner, ui));
            egui::CentralPanel::default().show(ctx, |ui| match self.tab {
                Tab::ParentChain => {
                    self.parent_chain.show(self.app.as_ref(), ui)
                }
                Tab::Coins => {
                    self.coins.show(self.app.as_ref(), ui);
                }
                Tab::MemPoolExplorer => {
                    self.mempool_explorer.show(self.app.as_ref(), ui);
                }
                Tab::BlockExplorer => {
                    self.block_explorer.show(self.app.as_ref(), ui);
                }
                Tab::Withdrawals => {
                    self.withdrawals.show(self.app.as_ref(), ui);
                }
                Tab::ConsoleLogs => {
                    self.console_logs.show(self.app.as_ref(), ui);
                }
            });
        }
    }
}
