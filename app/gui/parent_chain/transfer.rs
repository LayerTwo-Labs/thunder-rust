use eframe::egui::{self, Button};

use crate::app::App;

#[derive(Debug, Default)]
pub struct Deposit {
    amount: String,
    fee: String,
}

impl Deposit {
    pub fn show(&mut self, app: Option<&App>, ui: &mut egui::Ui) {
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
                app.is_some() && amount.is_ok() && fee.is_ok(),
                egui::Button::new("deposit"),
            )
            .clicked()
        {
            let app = app.unwrap();
            if let Err(err) = app.deposit(
                app.wallet.get_new_address().expect("should not happen"),
                amount.expect("should not happen"),
                fee.expect("should not happen"),
            ) {
                tracing::error!("{err}");
            } else {
                *self = Self::default();
            }
        }
    }
}

#[derive(Debug, Default)]
pub struct Withdrawal {
    mainchain_address: String,
    amount: String,
    fee: String,
    mainchain_fee: String,
}

fn create_withdrawal(
    app: &App,
    mainchain_address: bitcoin::Address<bitcoin::address::NetworkUnchecked>,
    amount: bitcoin::Amount,
    fee: bitcoin::Amount,
    mainchain_fee: bitcoin::Amount,
) -> anyhow::Result<()> {
    let accumulator = app.node.get_tip_accumulator()?;
    let tx = app.wallet.create_withdrawal(
        &accumulator,
        mainchain_address,
        amount,
        mainchain_fee,
        fee,
    )?;
    app.sign_and_send(tx)?;
    Ok(())
}

impl Withdrawal {
    pub fn show(&mut self, app: Option<&App>, ui: &mut egui::Ui) {
        ui.add_sized((250., 10.), |ui: &mut egui::Ui| {
            ui.horizontal(|ui| {
                let mainchain_address_edit =
                    egui::TextEdit::singleline(&mut self.mainchain_address)
                        .hint_text("mainchain address")
                        .desired_width(150.);
                ui.add(mainchain_address_edit);
                if ui
                    .add_enabled(app.is_some(), Button::new("generate"))
                    .clicked()
                {
                    match app.unwrap().get_new_main_address() {
                        Ok(main_address) => {
                            self.mainchain_address = main_address.to_string();
                        }
                        Err(err) => {
                            let err = anyhow::Error::new(err);
                            tracing::error!("{err:#}")
                        }
                    };
                }
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
        ui.add_sized((110., 10.), |ui: &mut egui::Ui| {
            ui.horizontal(|ui| {
                let fee_edit =
                    egui::TextEdit::singleline(&mut self.mainchain_fee)
                        .hint_text("mainchain fee")
                        .desired_width(80.);
                ui.add(fee_edit);
                ui.label("BTC");
            })
            .response
        });
        let mainchain_address: Option<
            bitcoin::Address<bitcoin::address::NetworkUnchecked>,
        > = self.mainchain_address.parse().ok();
        let amount = bitcoin::Amount::from_str_in(
            &self.amount,
            bitcoin::Denomination::Bitcoin,
        );
        let fee = bitcoin::Amount::from_str_in(
            &self.fee,
            bitcoin::Denomination::Bitcoin,
        );
        let mainchain_fee = bitcoin::Amount::from_str_in(
            &self.mainchain_fee,
            bitcoin::Denomination::Bitcoin,
        );

        if ui
            .add_enabled(
                app.is_some()
                    && mainchain_address.is_some()
                    && amount.is_ok()
                    && fee.is_ok()
                    && mainchain_fee.is_ok(),
                egui::Button::new("withdraw"),
            )
            .clicked()
        {
            if let Err(err) = create_withdrawal(
                app.unwrap(),
                mainchain_address.expect("should not happen"),
                amount.expect("should not happen"),
                fee.expect("should not happen"),
                mainchain_fee.expect("should not happen"),
            ) {
                tracing::error!("{err:#}");
            } else {
                *self = Self::default();
            }
        }
    }
}

#[derive(Default)]
pub(super) struct Transfer {
    deposit: Deposit,
    withdrawal: Withdrawal,
}

impl Transfer {
    pub fn show(&mut self, app: Option<&App>, ui: &mut egui::Ui) {
        egui::SidePanel::left("deposit")
            .exact_width(ui.available_width() / 2.)
            .resizable(false)
            .show_inside(ui, |ui| {
                ui.vertical_centered(|ui| {
                    ui.heading("Deposit");
                    self.deposit.show(app, ui);
                })
            });
        egui::CentralPanel::default().show_inside(ui, |ui| {
            ui.vertical_centered(|ui| {
                ui.heading("Withdrawal");
                self.withdrawal.show(app, ui);
            })
        });
    }
}
