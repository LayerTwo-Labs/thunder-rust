use eframe::egui::{self, Button};
use thunder_orchard::types::{Address, ShieldedAddress, TransparentAddress};

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
    let tx = match dest {
        Address::Shielded(dest) => app.wallet.create_shielded_transaction(
            &accumulator,
            dest,
            amount,
            fee,
            [0u8; 512],
        )?,
        Address::Transparent(dest) => {
            app.wallet
                .create_transaction(&accumulator, dest, amount, fee)?
        }
    };
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
struct Addresses {
    shielded: anyhow::Result<ShieldedAddress>,
    transparent: anyhow::Result<TransparentAddress>,
}

impl Addresses {
    fn show(&mut self, app: Option<&App>, ui: &mut egui::Ui) {
        ui.label("Shielded address");
        match &self.shielded {
            Ok(address) => {
                ui.monospace_selectable_singleline(false, address.to_string());
            }
            Err(err) => {
                ui.monospace_selectable_multiline(format!("{err:#}"));
            }
        }
        if ui
            .add_enabled(app.is_some(), Button::new("generate"))
            .clicked()
        {
            self.shielded = (|| {
                let app = app.unwrap();
                let mut rwtxn = app.wallet.env().write_txn()?;
                let res = app.wallet.get_new_orchard_address(&mut rwtxn)?;
                rwtxn.commit()?;
                Ok::<_, thunder_orchard::wallet::Error>(res)
            })()
            .map_err(anyhow::Error::from)
        }
        ui.label("Transparent address");
        match &self.transparent {
            Ok(address) => {
                ui.monospace_selectable_singleline(false, address.to_string());
            }
            Err(err) => {
                ui.monospace_selectable_multiline(format!("{err:#}"));
            }
        }
        if ui
            .add_enabled(app.is_some(), Button::new("generate"))
            .clicked()
        {
            self.transparent = (|| {
                let app = app.unwrap();
                let mut rwtxn = app.wallet.env().write_txn()?;
                let res = app.wallet.get_new_transparent_address(&mut rwtxn)?;
                rwtxn.commit()?;
                Ok::<_, thunder_orchard::wallet::Error>(res)
            })()
            .map_err(anyhow::Error::from)
        }
    }
}

#[derive(Debug)]
struct Receive {
    addresses: Option<anyhow::Result<Addresses>>,
}

impl Receive {
    fn new(app: Option<&App>) -> Self {
        let Some(app) = app else {
            return Self { addresses: None };
        };
        let addresses = (|| {
            let mut rwtxn = app.wallet.env().write_txn()?;
            let shielded = app.wallet.get_new_orchard_address(&mut rwtxn);
            let transparent =
                app.wallet.get_new_transparent_address(&mut rwtxn);
            let addresses = Addresses {
                shielded: shielded.map_err(anyhow::Error::from),
                transparent: transparent.map_err(anyhow::Error::from),
            };
            rwtxn.commit()?;
            Ok::<_, thunder_orchard::wallet::Error>(addresses)
        })()
        .map_err(anyhow::Error::from)
        .inspect_err(|err| tracing::error!("{err:#}"));
        Self {
            addresses: Some(addresses),
        }
    }

    fn show(&mut self, app: Option<&App>, ui: &mut egui::Ui) {
        match &mut self.addresses {
            Some(Ok(addresses)) => {
                addresses.show(app, ui);
            }
            Some(Err(err)) => {
                ui.monospace_selectable_multiline(format!("{err:#}"));
            }
            None => (),
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
