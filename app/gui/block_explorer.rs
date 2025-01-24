use bitcoin::Amount;
use eframe::egui;
use human_size::{Byte, Kibibyte, Mebibyte, SpecificSize};
use thunder::{
    state::State,
    types::{Body, GetValue, Header},
};

use crate::app::App;

use super::util::UiExt;

pub struct BlockExplorer {
    height: u32,
}

impl BlockExplorer {
    pub fn new(height: u32) -> Self {
        Self { height }
    }

    pub fn show(&mut self, app: Option<&App>, ui: &mut egui::Ui) {
        let max_height = app
            .and_then(|app| app.node.try_get_height().ok().flatten())
            .unwrap_or(0);
        let block: Option<(Header, Body)> = {
            if let Some(app) = app
                && let Ok(Some(block_hash)) =
                    app.node.try_get_block_hash(self.height)
                && let Ok(header) = app.node.get_header(block_hash)
                && let Ok(body) = app.node.get_body(block_hash)
            {
                Some((header, body))
            } else {
                None
            }
        };
        egui::CentralPanel::default().show_inside(ui, |ui| {
            ui.heading("Block");
            ui.horizontal(|ui| {
                if ui.button("<").clicked() && self.height > 0 {
                    self.height -= 1;
                }
                ui.monospace(format!("{}", self.height));
                if ui.button(">").clicked() && self.height < max_height {
                    self.height += 1;
                }
                if ui.button("latest").clicked() {
                    self.height = max_height;
                }
            });
            if let Some((header, body)) = block {
                let hash = &format!("{}", header.hash());
                let merkle_root = &format!("{}", header.merkle_root);
                let prev_side_hash = &format!("{:?}", header.prev_side_hash);
                let prev_main_hash = &format!("{}", header.prev_main_hash);
                let body_size =
                    bincode::serialize(&body).unwrap_or(vec![]).len();
                let coinbase_value: Amount =
                    body.coinbase.iter().map(GetValue::get_value).sum();
                let num_transactions = body.transactions.len();
                let body_size = if let Ok(body_size) =
                    SpecificSize::new(body_size as f64, Byte)
                {
                    let bytes = body_size.to_bytes();
                    if bytes < 1024 {
                        format!("{body_size}")
                    } else if bytes < 1024 * 1024 {
                        let body_size: SpecificSize<Kibibyte> =
                            body_size.into();
                        format!("{body_size}")
                    } else {
                        let body_size: SpecificSize<Mebibyte> =
                            body_size.into();
                        format!("{body_size}")
                    }
                } else {
                    "".into()
                };
                let num_sigops = body.authorizations.len();
                ui.monospace(format!("Block hash:       {hash}"));
                ui.monospace(format!("Merkle root:      {merkle_root}"));
                ui.monospace(format!("Prev side:        {prev_side_hash}"));
                ui.monospace(format!("Prev main:        {prev_main_hash}"));
                ui.monospace(format!("Num transactions: {num_transactions}"));
                ui.monospace(format!("Coinbase value:   {coinbase_value}"));
                ui.monospace(format!("Body size:        {body_size}"));
                ui.monospace(format!("Num sigops:       {num_sigops}"));

                let body_size_limit = State::body_size_limit(self.height);
                if let Ok(body_size_limit) =
                    SpecificSize::new(body_size_limit as f64, Byte)
                {
                    let body_size_limit: SpecificSize<Mebibyte> =
                        body_size_limit.into();
                    ui.monospace(format!(
                        "Body size limit:   {body_size_limit}"
                    ));
                }
                let body_sigops_limit = State::body_sigops_limit(self.height);
                ui.monospace(format!("Body sigops limit: {body_sigops_limit}"));

                if let Some(app) = app
                    && let Ok(Some(acc)) =
                        app.node.try_get_accumulator(header.hash())
                {
                    ui.monospace_selectable_multiline(format!(
                        "Utreexo accumulator: \n{}",
                        acc.0
                    ));
                }
            }
        });
    }
}
