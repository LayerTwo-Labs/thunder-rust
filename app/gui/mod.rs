use std::collections::HashSet;

use eframe::egui;
use thunder::{bip300301::bitcoin, types::GetValue};

use crate::app::App;

mod block_explorer;
mod deposit;
mod mempool_explorer;
mod miner;
mod seed;
mod utxo_creator;
mod utxo_selector;
mod withdrawals;

use block_explorer::BlockExplorer;
use deposit::Deposit;
use mempool_explorer::MemPoolExplorer;
use miner::Miner;
use seed::SetSeed;
use utxo_selector::{show_utxo, UtxoSelector};

use self::{utxo_creator::UtxoCreator, withdrawals::Withdrawals};

pub struct EguiApp {
    app: App,
    set_seed: SetSeed,
    miner: Miner,
    deposit: Deposit,
    tab: Tab,
    utxo_selector: UtxoSelector,
    utxo_creator: UtxoCreator,
    mempool_explorer: MemPoolExplorer,
    block_explorer: BlockExplorer,
    withdrawals: Withdrawals,
}

#[derive(Eq, PartialEq)]
enum Tab {
    TransactionBuilder,
    MemPoolExplorer,
    BlockExplorer,
    Withdrawals,
}

impl EguiApp {
    pub fn new(app: App, _cc: &eframe::CreationContext<'_>) -> Self {
        // Customize egui here with cc.egui_ctx.set_fonts and cc.egui_ctx.set_visuals.
        // Restore app state using cc.storage (requires the "persistence" feature).
        // Use the cc.gl (a glow::Context) to create graphics shaders and buffers that you can use
        // for e.g. egui::PaintCallback.
        let height = app.node.get_height().unwrap_or(0);
        Self {
            app,
            set_seed: SetSeed::default(),
            miner: Miner::default(),
            deposit: Deposit::default(),
            utxo_selector: UtxoSelector,
            utxo_creator: UtxoCreator::default(),
            mempool_explorer: MemPoolExplorer::default(),
            block_explorer: BlockExplorer::new(height),
            tab: Tab::TransactionBuilder,
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

            self.deposit.show(&mut self.app, ui);
            ui.separator();
            ui.add_space(this_target_width);
            ui.separator();
            self.miner.show(&mut self.app, ui);
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
                    ui.selectable_value(
                        &mut self.tab,
                        Tab::TransactionBuilder,
                        "transaction builder",
                    );
                    ui.selectable_value(
                        &mut self.tab,
                        Tab::MemPoolExplorer,
                        "mempool explorer",
                    );
                    ui.selectable_value(
                        &mut self.tab,
                        Tab::BlockExplorer,
                        "block explorer",
                    );
                    ui.selectable_value(
                        &mut self.tab,
                        Tab::Withdrawals,
                        "withdrawals",
                    );
                });
            });
            egui::TopBottomPanel::bottom("util")
                .show(ctx, |ui| self.bottom_panel_content(ui));
            egui::CentralPanel::default().show(ctx, |ui| match self.tab {
                Tab::TransactionBuilder => {
                    let selected: HashSet<_> =
                        self.app.transaction.inputs.iter().cloned().collect();
                    let utxos = self.app.utxos.clone();
                    let value_in: u64 = utxos
                        .read()
                        .iter()
                        .filter(|(outpoint, _)| selected.contains(outpoint))
                        .map(|(_, output)| output.get_value())
                        .sum();
                    let value_out: u64 = self
                        .app
                        .transaction
                        .outputs
                        .iter()
                        .map(GetValue::get_value)
                        .sum();
                    egui::SidePanel::left("spend_utxo")
                        .exact_width(250.)
                        .resizable(false)
                        .show_inside(ui, |ui| {
                            self.utxo_selector.show(&mut self.app.clone(), ui);
                        });
                    egui::SidePanel::left("value_in")
                        .exact_width(250.)
                        .resizable(false)
                        .show_inside(ui, |ui| {
                            ui.heading("Value In");
                            let utxos_read = utxos.read();
                            let mut utxos: Vec<_> = utxos_read
                                .iter()
                                .filter(|(outpoint, _)| {
                                    selected.contains(outpoint)
                                })
                                .collect();
                            utxos.sort_by_key(|(outpoint, _)| {
                                format!("{outpoint}")
                            });
                            ui.separator();
                            ui.monospace(format!(
                                "Total: {}",
                                bitcoin::Amount::from_sat(value_in)
                            ));
                            ui.separator();
                            egui::Grid::new("utxos").striped(true).show(
                                ui,
                                |ui| {
                                    ui.monospace("kind");
                                    ui.monospace("outpoint");
                                    ui.monospace("value");
                                    ui.end_row();
                                    let mut remove = None;
                                    for (vout, outpoint) in self
                                        .app
                                        .transaction
                                        .inputs
                                        .iter()
                                        .enumerate()
                                    {
                                        let output = &utxos_read[outpoint];
                                        show_utxo(ui, outpoint, output);
                                        if ui.button("remove").clicked() {
                                            remove = Some(vout);
                                        }
                                        ui.end_row();
                                    }
                                    if let Some(vout) = remove {
                                        self.app
                                            .transaction
                                            .inputs
                                            .remove(vout);
                                    }
                                },
                            );
                        });
                    egui::SidePanel::left("value_out")
                        .exact_width(250.)
                        .resizable(false)
                        .show_inside(ui, |ui| {
                            ui.heading("Value Out");
                            ui.separator();
                            ui.monospace(format!(
                                "Total: {}",
                                bitcoin::Amount::from_sat(value_out)
                            ));
                            ui.separator();
                            egui::Grid::new("outputs").striped(true).show(
                                ui,
                                |ui| {
                                    let mut remove = None;
                                    ui.monospace("vout");
                                    ui.monospace("address");
                                    ui.monospace("value");
                                    ui.end_row();
                                    for (vout, output) in self
                                        .app
                                        .transaction
                                        .outputs
                                        .iter()
                                        .enumerate()
                                    {
                                        let address =
                                            &format!("{}", output.address)
                                                [0..8];
                                        let value = bitcoin::Amount::from_sat(
                                            output.get_value(),
                                        );
                                        ui.monospace(format!("{vout}"));
                                        ui.monospace(address.to_string());
                                        ui.with_layout(
                                            egui::Layout::right_to_left(
                                                egui::Align::Max,
                                            ),
                                            |ui| {
                                                ui.monospace(format!(
                                                    "{value}"
                                                ));
                                            },
                                        );
                                        if ui.button("remove").clicked() {
                                            remove = Some(vout);
                                        }
                                        ui.end_row();
                                    }
                                    if let Some(vout) = remove {
                                        self.app
                                            .transaction
                                            .outputs
                                            .remove(vout);
                                    }
                                },
                            );
                        });
                    egui::SidePanel::left("create_utxo")
                        .exact_width(450.)
                        .resizable(false)
                        .show_separator_line(false)
                        .show_inside(ui, |ui| {
                            self.utxo_creator.show(&mut self.app.clone(), ui);
                            ui.separator();
                            ui.heading("Transaction");
                            let txid =
                                &format!("{}", self.app.transaction.txid())
                                    [0..8];
                            ui.monospace(format!("txid: {txid}"));
                            if value_in >= value_out {
                                let fee = value_in - value_out;
                                let fee = bitcoin::Amount::from_sat(fee);
                                ui.monospace(format!("fee:  {fee}"));
                                if ui.button("sign and send").clicked() {
                                    self.app.sign_and_send().unwrap_or(());
                                }
                            } else {
                                ui.label("Not Enough Value In");
                            }
                        });
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
