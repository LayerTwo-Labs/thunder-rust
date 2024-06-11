use bip300301::{bitcoin, MainClient};
use eframe::egui;
use futures::TryFutureExt;

use crate::{app::App, gui::util::UiExt};

#[derive(Clone, Debug)]
struct Inner {
    mainchain_tip: bip300301::client::Block,
    sidechain_wealth: bitcoin::Amount,
}

pub(super) struct Info(anyhow::Result<Inner>);

impl Info {
    fn get_parent_chain_info(app: &App) -> anyhow::Result<Inner> {
        let dc = app.node.drivechain();
        let mainchain_tip = app.runtime.block_on(async {
            let mainchain_tip_blockhash = dc.get_mainchain_tip().await?;
            dc.client
                .getblock(mainchain_tip_blockhash, None)
                .map_err(|source| bip300301::Error::Jsonrpsee {
                    source,
                    main_addr: dc.main_addr,
                })
                .await
        })?;
        let sidechain_wealth = app.node.get_sidechain_wealth()?;
        Ok(Inner {
            mainchain_tip,
            sidechain_wealth,
        })
    }

    pub fn new(app: &App) -> Self {
        let inner = Self::get_parent_chain_info(app)
            .inspect_err(|err| tracing::error!("{err:#}"));
        Self(inner)
    }

    fn refresh_parent_chain_info(&mut self, app: &App) {
        self.0 = Self::get_parent_chain_info(app)
            .inspect_err(|err| tracing::error!("{err:#}"));
    }

    pub fn show(&mut self, app: &mut App, ui: &mut egui::Ui) {
        if ui.button("Refresh").clicked() {
            let () = self.refresh_parent_chain_info(app);
        }
        let parent_chain_info = match self.0.as_ref() {
            Ok(parent_chain_info) => parent_chain_info,
            Err(err) => {
                ui.monospace_selectable_multiline(format!("{err:#}"));
                return;
            }
        };
        ui.horizontal(|ui| {
            ui.monospace("Mainchain tip hash: ");
            ui.monospace_selectable_singleline(
                true,
                parent_chain_info.mainchain_tip.hash.to_string(),
            )
        });
        ui.horizontal(|ui| {
            ui.monospace("Mainchain tip height: ");
            ui.monospace_selectable_singleline(
                true,
                parent_chain_info.mainchain_tip.height.to_string(),
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
