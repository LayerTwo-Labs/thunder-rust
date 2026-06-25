use eframe::egui::{self, Button};

use crate::app::App;

#[derive(Default)]
pub struct Deposit {
    amount: String,
    fee: String,
    promise: Option<poll_promise::Promise<Result<bitcoin::Txid, String>>>,
}

impl std::fmt::Debug for Deposit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Deposit")
            .field("amount", &self.amount)
            .field("fee", &self.fee)
            .field("promise_active", &self.promise.is_some())
            .finish()
    }
}

impl Deposit {
    pub fn show(&mut self, app: Option<&App>, ui: &mut egui::Ui) {
        if let Some(promise) = &self.promise {
            match promise.ready() {
                None => {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label(
                            "Creating deposit transaction on parent chain...",
                        );
                    });
                    return;
                }
                Some(Ok(txid)) => {
                    tracing::info!("Deposit transaction created: {}", txid);
                    self.promise = None;
                    *self = Self::default();
                    return;
                }
                Some(Err(err)) => {
                    ui.colored_label(
                        egui::Color32::RED,
                        format!("Error: {err}"),
                    );
                    if ui.button("Dismiss").clicked() {
                        self.promise = None;
                    }
                    return;
                }
            }
        }

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
            let app = app.unwrap().clone();
            let amount = amount.expect("should not happen");
            let fee = fee.expect("should not happen");

            match app.wallet.get_new_address() {
                Ok(address) => {
                    self.promise =
                        Some(poll_promise::Promise::spawn_async(async move {
                            app.deposit(address, amount, fee)
                                .await
                                .map_err(|e| format!("{e:#}"))
                        }));
                }
                Err(err) => {
                    tracing::error!("Failed to get new address: {err}");
                }
            }
        }
    }
}

#[derive(Default)]
pub struct Withdrawal {
    mainchain_address: String,
    amount: String,
    fee: String,
    mainchain_fee: String,
    generate_promise: Option<
        poll_promise::Promise<
            Result<bitcoin::Address<bitcoin::address::NetworkChecked>, String>,
        >,
    >,
}

impl std::fmt::Debug for Withdrawal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Withdrawal")
            .field("mainchain_address", &self.mainchain_address)
            .field("amount", &self.amount)
            .field("fee", &self.fee)
            .field("mainchain_fee", &self.mainchain_fee)
            .field("generate_active", &self.generate_promise.is_some())
            .finish()
    }
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
        if let Some(promise) = &self.generate_promise {
            match promise.ready() {
                None => {}
                Some(Ok(address)) => {
                    self.mainchain_address = address.to_string();
                    self.generate_promise = None;
                }
                Some(Err(err)) => {
                    tracing::error!(
                        "Failed to generate mainchain address: {err}"
                    );
                    self.generate_promise = None;
                }
            }
        }

        ui.add_sized((250., 10.), |ui: &mut egui::Ui| {
            ui.horizontal(|ui| {
                let mainchain_address_edit =
                    egui::TextEdit::singleline(&mut self.mainchain_address)
                        .hint_text("mainchain address")
                        .desired_width(150.);
                ui.add(mainchain_address_edit);

                let is_generating = self.generate_promise.is_some();
                let generate_btn = if is_generating {
                    Button::new("generating...")
                } else {
                    Button::new("generate")
                };
                if ui
                    .add_enabled(app.is_some() && !is_generating, generate_btn)
                    .clicked()
                {
                    let app = app.unwrap().clone();
                    self.generate_promise =
                        Some(poll_promise::Promise::spawn_async(async move {
                            app.get_new_main_address()
                                .await
                                .map_err(|e| format!("{e:#}"))
                        }));
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
                let is_generating = self.generate_promise.is_some();
                *self = Self::default();
                if is_generating {
                    // Keep the generation promise active if it was running
                    // (though unlikely to happen during a withdrawal)
                    self.generate_promise =
                        Some(poll_promise::Promise::spawn_async(async move {
                            unreachable!()
                        }));
                }
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
        egui::Panel::left("deposit")
            .exact_size(ui.available_width() / 2.)
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
