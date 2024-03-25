use eframe::egui;
use strum::{EnumIter, IntoEnumIterator};

use crate::app::App;

mod tx_builder;
mod tx_creator;
mod utxo_creator;
mod utxo_selector;

use tx_builder::TxBuilder;

#[derive(Default, EnumIter, Eq, PartialEq, strum::Display)]
enum Tab {
    #[default]
    #[strum(to_string = "Transaction Builder")]
    TransactionBuilder,
}

#[derive(Default)]
pub struct Coins {
    tab: Tab,
    tx_builder: TxBuilder,
}

impl Coins {
    pub fn show(&mut self, app: &mut App, ui: &mut egui::Ui) {
        egui::TopBottomPanel::top("coins_tabs").show(ui.ctx(), |ui| {
            ui.horizontal(|ui| {
                Tab::iter().for_each(|tab_variant| {
                    let tab_name = tab_variant.to_string();
                    ui.selectable_value(&mut self.tab, tab_variant, tab_name);
                })
            });
        });
        egui::CentralPanel::default().show(ui.ctx(), |ui| match self.tab {
            Tab::TransactionBuilder => {
                let () = self.tx_builder.show(app, ui).unwrap();
            }
        });
    }
}
