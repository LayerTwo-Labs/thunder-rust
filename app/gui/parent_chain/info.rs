use eframe::egui::{self, Button};
use futures::FutureExt;
use photon::types::proto::mainchain;

use crate::{app::App, gui::util::UiExt};

#[derive(Clone, Debug)]
struct Inner {
    mainchain_tip_info: mainchain::BlockHeaderInfo,
    sidechain_wealth: bitcoin::Amount,
}

pub(super) struct Info {
    promise: Option<poll_promise::Promise<anyhow::Result<Inner>>>,
    last_value: Option<anyhow::Result<Inner>>,
}

impl Info {
    pub fn new(app: Option<&App>) -> Self {
        let mut this = Self {
            promise: None,
            last_value: None,
        };
        if let Some(app) = app {
            this.refresh_parent_chain_info(app);
        }
        this
    }

    fn refresh_parent_chain_info(&mut self, app: &App) {
        let app = app.clone();
        self.promise = Some(poll_promise::Promise::spawn_async(async move {
            let mainchain_tip_info = app
                .node
                .with_cusf_mainchain(|cusf_mainchain| {
                    cusf_mainchain.get_chain_tip().boxed()
                })
                .await?;
            let sidechain_wealth = app.node.get_sidechain_wealth()?;
            Ok(Inner {
                mainchain_tip_info,
                sidechain_wealth,
            })
        }));
    }

    pub fn show(&mut self, app: Option<&App>, ui: &mut egui::Ui) {
        if let Some(result) = self.promise.as_ref().and_then(|p| p.ready()) {
            self.last_value = Some(match result {
                Ok(inner) => Ok(inner.clone()),
                Err(err) => Err(anyhow::anyhow!("{err:#}")),
            });
            self.promise = None;
        }

        ui.horizontal(|ui| {
            let is_refreshing = self.promise.is_some();
            if ui
                .add_enabled(
                    app.is_some() && !is_refreshing,
                    Button::new("Refresh"),
                )
                .clicked()
            {
                self.refresh_parent_chain_info(app.unwrap());
            }
            if is_refreshing {
                ui.spinner();
                ui.label("Refreshing parent chain info...");
            }
        });

        let parent_chain_info = match self.last_value.as_ref() {
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
