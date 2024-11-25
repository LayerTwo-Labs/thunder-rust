use eframe::egui::{self, Button};
use thunder::types::Address;

use crate::{app::App, gui::util::UiExt};

#[derive(Debug, Default)]
struct Transfer {
    dest: String,
    amount: String,
    fee: String,
}

fn create_transfer(
    app: &App,
    dest: Address,
    amount: bitcoin::Amount,
    fee: bitcoin::Amount,
) -> anyhow::Result<()> {
    let accumulator = app.node.get_tip_accumulator()?;
    let tx = app
        .wallet
        .create_transaction(&accumulator, dest, amount, fee)?;
    app.sign_and_send(tx)?;
    Ok(())
}

impl Transfer {
    fn show(&mut self, app: Option<&App>, ui: &mut egui::Ui) {
        ui.add_sized((250., 10.), |ui: &mut egui::Ui| {
            ui.horizontal(|ui| {
                let dest_edit = egui::TextEdit::singleline(&mut self.dest)
                    .hint_text("destination address")
                    .desired_width(150.);
                ui.add(dest_edit);
            })
            .response
        });
        ui.add_sized((110., 10.), |ui: &mut egui::Ui| {
            ui.horizontal(|ui| {
                let amount_edit = egui::TextEdit::singleline(&mut self.amount)
                    .hint_text("amount")
                    .desired_width(80.);
                ui.add(amount_edit);
                ui.label("BTC");
            })
            .response
        });
        ui.add_sized((110., 10.), |ui: &mut egui::Ui| {
            ui.horizontal(|ui| {
                let fee_edit = egui::TextEdit::singleline(&mut self.fee)
                    .hint_text("fee")
                    .desired_width(80.);
                ui.add(fee_edit);
                ui.label("BTC");
            })
            .response
        });
        let dest: Option<Address> = self.dest.parse().ok();
        let amount = bitcoin::Amount::from_str_in(
            &self.amount,
            bitcoin::Denomination::Bitcoin,
        );
        let fee = bitcoin::Amount::from_str_in(
            &self.fee,
            bitcoin::Denomination::Bitcoin,
        );
        if ui
            .add_enabled(
                app.is_some()
                    && dest.is_some()
                    && amount.is_ok()
                    && fee.is_ok(),
                egui::Button::new("transfer"),
            )
            .clicked()
        {
            if let Err(err) = create_transfer(
                app.unwrap(),
                dest.expect("should not happen"),
                amount.expect("should not happen"),
                fee.expect("should not happen"),
            ) {
                tracing::error!("{err:#}");
            } else {
                *self = Self::default();
            }
        }
    }
}

#[derive(Debug)]
struct Receive {
    address: Option<anyhow::Result<Address>>,
}

impl Receive {
    fn new(app: Option<&App>) -> Self {
        let Some(app) = app else {
            return Self { address: None };
        };
        let address = app
            .wallet
            .get_new_address()
            .map_err(anyhow::Error::from)
            .inspect_err(|err| tracing::error!("{err:#}"));
        Self {
            address: Some(address),
        }
    }

    fn show(&mut self, app: Option<&App>, ui: &mut egui::Ui) {
        match &self.address {
            Some(Ok(address)) => {
                ui.monospace_selectable_singleline(false, address.to_string());
            }
            Some(Err(err)) => {
                ui.monospace_selectable_multiline(format!("{err:#}"));
            }
            None => (),
        }
        if ui
            .add_enabled(app.is_some(), Button::new("generate"))
            .clicked()
        {
            *self = Self::new(app)
        }
    }
}

#[derive(Debug)]
pub(super) struct TransferReceive {
    transfer: Transfer,
    receive: Receive,
}

impl TransferReceive {
    pub fn new(app: Option<&App>) -> Self {
        Self {
            transfer: Transfer::default(),
            receive: Receive::new(app),
        }
    }

    pub fn show(&mut self, app: Option<&App>, ui: &mut egui::Ui) {
        egui::SidePanel::left("transfer")
            .exact_width(ui.available_width() / 2.)
            .resizable(false)
            .show_inside(ui, |ui| {
                ui.vertical_centered(|ui| {
                    ui.heading("Transfer");
                    self.transfer.show(app, ui);
                })
            });
        egui::CentralPanel::default().show_inside(ui, |ui| {
            ui.vertical_centered(|ui| {
                ui.heading("Receive");
                self.receive.show(app, ui);
            })
        });
    }
}
