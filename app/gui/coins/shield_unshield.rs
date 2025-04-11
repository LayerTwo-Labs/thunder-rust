use eframe::egui;

use crate::app::App;

#[derive(Debug, Default)]
struct Shield {
    amount: String,
    fee: String,
}

fn create_shield_tx(
    app: &App,
    amount: bitcoin::Amount,
    fee: bitcoin::Amount,
) -> anyhow::Result<()> {
    let accumulator = app.node.get_tip_accumulator()?;
    let tx = app
        .wallet
        .create_shield_transaction(&accumulator, amount, fee)?;
    app.sign_and_send(tx)?;
    Ok(())
}

impl Shield {
    fn show(&mut self, app: Option<&App>, ui: &mut egui::Ui) {
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
                egui::Button::new("shield"),
            )
            .clicked()
        {
            if let Err(err) = create_shield_tx(
                app.unwrap(),
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

#[derive(Debug, Default)]
struct Unshield {
    amount: String,
    fee: String,
}

fn create_unshield_tx(
    app: &App,
    amount: bitcoin::Amount,
    fee: bitcoin::Amount,
) -> anyhow::Result<()> {
    let accumulator = app.node.get_tip_accumulator()?;
    let tx =
        app.wallet
            .create_unshield_transaction(&accumulator, amount, fee)?;
    app.sign_and_send(tx)?;
    Ok(())
}

impl Unshield {
    fn show(&mut self, app: Option<&App>, ui: &mut egui::Ui) {
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
                egui::Button::new("unshield"),
            )
            .clicked()
        {
            if let Err(err) = create_unshield_tx(
                app.unwrap(),
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

#[derive(Debug, Default)]
pub(super) struct ShieldUnshield {
    shield: Shield,
    unshield: Unshield,
}

impl ShieldUnshield {
    pub fn show(&mut self, app: Option<&App>, ui: &mut egui::Ui) {
        egui::SidePanel::left("Shield")
            .exact_width(ui.available_width() / 2.)
            .resizable(false)
            .show_inside(ui, |ui| {
                ui.vertical_centered(|ui| {
                    ui.heading("Shield");
                    self.shield.show(app, ui);
                })
            });
        egui::CentralPanel::default().show_inside(ui, |ui| {
            ui.vertical_centered(|ui| {
                ui.heading("Unshield");
                self.unshield.show(app, ui);
            })
        });
    }
}
