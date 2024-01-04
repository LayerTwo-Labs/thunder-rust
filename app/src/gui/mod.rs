use std::collections::HashSet;

use crate::app::{lib, App};
use eframe::egui;
use lib::{bip300301::bitcoin, types::GetValue};

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
            miner: Miner,
            deposit: Deposit::default(),
            utxo_selector: UtxoSelector,
            utxo_creator: UtxoCreator::default(),
            mempool_explorer: MemPoolExplorer::default(),
            block_explorer: BlockExplorer::new(height),
            tab: Tab::TransactionBuilder,
            withdrawals: Withdrawals::default(),
        }
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
                    ui.selectable_value(&mut self.tab, Tab::MemPoolExplorer, "mempool explorer");
                    ui.selectable_value(&mut self.tab, Tab::BlockExplorer, "block explorer");
                    ui.selectable_value(&mut self.tab, Tab::Withdrawals, "withdrawals");
                });
            });
            egui::TopBottomPanel::bottom("util").show(ctx, |ui| {
                ui.horizontal(|ui| {
                    self.miner.show(&mut self.app, ui);
                    ui.separator();
                    self.deposit.show(&mut self.app, ui);
                });
            });
            egui::CentralPanel::default().show(ctx, |ui| match self.tab {
                Tab::TransactionBuilder => {
                    let selected: HashSet<_> =
                        self.app.transaction.inputs.iter().cloned().collect();
                    let value_in: u64 = self
                        .app
                        .utxos
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
                            self.utxo_selector.show(&mut self.app, ui);
                        });
                    egui::SidePanel::left("value_in")
                        .exact_width(250.)
                        .resizable(false)
                        .show_inside(ui, |ui| {
                            ui.heading("Value In");
                            let mut utxos: Vec<_> = self
                                .app
                                .utxos
                                .iter()
                                .filter(|(outpoint, _)| selected.contains(outpoint))
                                .collect();
                            utxos.sort_by_key(|(outpoint, _)| format!("{outpoint}"));
                            ui.separator();
                            ui.monospace(format!("Total: {}", bitcoin::Amount::from_sat(value_in)));
                            ui.separator();
                            egui::Grid::new("utxos").striped(true).show(ui, |ui| {
                                ui.monospace("kind");
                                ui.monospace("outpoint");
                                ui.monospace("value");
                                ui.end_row();
                                let mut remove = None;
                                for (vout, outpoint) in
                                    self.app.transaction.inputs.iter().enumerate()
                                {
                                    let output = &self.app.utxos[&outpoint];
                                    show_utxo(ui, outpoint, output);
                                    if ui.button("remove").clicked() {
                                        remove = Some(vout);
                                    }
                                    ui.end_row();
                                }
                                if let Some(vout) = remove {
                                    self.app.transaction.inputs.remove(vout);
                                }
                            });
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
                            egui::Grid::new("outputs").striped(true).show(ui, |ui| {
                                let mut remove = None;
                                ui.monospace("vout");
                                ui.monospace("address");
                                ui.monospace("value");
                                ui.end_row();
                                for (vout, output) in
                                    self.app.transaction.outputs.iter().enumerate()
                                {
                                    let address = &format!("{}", output.address)[0..8];
                                    let value = bitcoin::Amount::from_sat(output.get_value());
                                    ui.monospace(format!("{vout}"));
                                    ui.monospace(address.to_string());
                                    ui.with_layout(
                                        egui::Layout::right_to_left(egui::Align::Max),
                                        |ui| {
                                            ui.monospace(format!("{value}"));
                                        },
                                    );
                                    if ui.button("remove").clicked() {
                                        remove = Some(vout);
                                    }
                                    ui.end_row();
                                }
                                if let Some(vout) = remove {
                                    self.app.transaction.outputs.remove(vout);
                                }
                            });
                        });
                    egui::SidePanel::left("create_utxo")
                        .exact_width(450.)
                        .resizable(false)
                        .show_separator_line(false)
                        .show_inside(ui, |ui| {
                            self.utxo_creator.show(&mut self.app, ui);
                            ui.separator();
                            ui.heading("Transaction");
                            let txid = &format!("{}", self.app.transaction.txid())[0..8];
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
