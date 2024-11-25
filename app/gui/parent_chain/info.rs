use eframe::egui::{self, Button};
use futures::FutureExt;
use thunder::types::proto::mainchain;

use crate::{app::App, gui::util::UiExt};

#[derive(Clone, Debug)]
struct Inner {
    mainchain_tip_info: mainchain::BlockHeaderInfo,
    sidechain_wealth: bitcoin::Amount,
}

pub(super) struct Info(Option<anyhow::Result<Inner>>);

impl Info {
    fn get_parent_chain_info(app: &App) -> anyhow::Result<Inner> {
        let mainchain_tip_info =
            app.runtime.block_on(app.node.with_cusf_mainchain(
                |cusf_mainchain| cusf_mainchain.get_chain_tip().boxed(),
            ))?;
        let sidechain_wealth = app.node.get_sidechain_wealth()?;
        Ok(Inner {
            mainchain_tip_info,
            sidechain_wealth,
        })
    }

    pub fn new(app: Option<&App>) -> Self {
        let inner = app.map(|app| {
            Self::get_parent_chain_info(app)
                .inspect_err(|err| tracing::error!("{err:#}"))
        });
        Self(inner)
    }

    fn refresh_parent_chain_info(&mut self, app: &App) {
        self.0 = Some(
            Self::get_parent_chain_info(app)
                .inspect_err(|err| tracing::error!("{err:#}")),
        );
    }

    pub fn show(&mut self, app: Option<&App>, ui: &mut egui::Ui) {
        if ui
            .add_enabled(app.is_some(), Button::new("Refresh"))
            .clicked()
        {
            let () = self.refresh_parent_chain_info(app.unwrap());
        }
        let parent_chain_info = match self.0.as_ref() {
            Some(Ok(parent_chain_info)) => parent_chain_info,
            Some(Err(err)) => {
                ui.monospace_selectable_multiline(format!("{err:#}"));
                return;
            }
            None => return,
        };
        ui.horizontal(|ui| {
            ui.monospace("Mainchain tip hash: ");
            ui.monospace_selectable_singleline(
                true,
                parent_chain_info.mainchain_tip_info.block_hash.to_string(),
            )
        });
        ui.horizontal(|ui| {
            ui.monospace("Mainchain tip height: ");
            ui.monospace_selectable_singleline(
                true,
                parent_chain_info.mainchain_tip_info.height.to_string(),
            )
        });
        ui.horizontal(|ui| {
            ui.monospace("Sidechain wealth: ");
            ui.monospace_selectable_singleline(
                false,
                parent_chain_info.sidechain_wealth.to_string(),
            )
        });
    }
}
