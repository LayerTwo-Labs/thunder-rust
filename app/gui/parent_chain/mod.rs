use eframe::egui;
use strum::{EnumIter, IntoEnumIterator};

use crate::app::App;

mod info;
mod transfer;

use info::Info;
use transfer::Transfer;

#[derive(Default, EnumIter, Eq, PartialEq, strum::Display)]
enum Tab {
    #[default]
    #[strum(to_string = "Transfer")]
    Transfer,
    #[strum(to_string = "Info")]
    Info,
}

pub struct ParentChain {
    info: Info,
    tab: Tab,
    transfer: Transfer,
}

impl ParentChain {
    pub fn new(app: Option<&App>) -> Self {
        let info = Info::new(app);
        Self {
            info,
            tab: Tab::default(),
            transfer: Transfer::default(),
        }
    }

    pub fn show(&mut self, app: Option<&App>, ui: &mut egui::Ui) {
        egui::TopBottomPanel::top("parent_chain_tabs").show(ui.ctx(), |ui| {
            ui.horizontal(|ui| {
                Tab::iter().for_each(|tab_variant| {
                    let tab_name = tab_variant.to_string();
                    ui.selectable_value(&mut self.tab, tab_variant, tab_name);
                })
            });
        });
        egui::CentralPanel::default().show(ui.ctx(), |ui| match self.tab {
            Tab::Transfer => {
                self.transfer.show(app, ui);
            }
            Tab::Info => {
                self.info.show(app, ui);
            }
        });
    }
}
