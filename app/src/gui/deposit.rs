use crate::app::lib;
use crate::app::App;
use eframe::egui;
use lib::bip300301::bitcoin;

pub struct Deposit {
    amount: String,
    fee: String,
}

impl Default for Deposit {
    fn default() -> Self {
        Self {
            amount: "".into(),
            fee: "".into(),
        }
    }
}

impl Deposit {
    pub fn show(&mut self, app: &mut App, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let amount_edit = egui::TextEdit::singleline(&mut self.amount)
                .hint_text("amount")
                .desired_width(80.);
            ui.add(amount_edit);
            ui.label("BTC");
        });
        ui.horizontal(|ui| {
            let fee_edit = egui::TextEdit::singleline(&mut self.fee)
                .hint_text("fee")
                .desired_width(80.);
            ui.add(fee_edit);
            ui.label("BTC");
        });

        let amount = bitcoin::Amount::from_str_in(&self.amount, bitcoin::Denomination::Bitcoin);
        let fee = bitcoin::Amount::from_str_in(&self.fee, bitcoin::Denomination::Bitcoin);

        if ui
            .add_enabled(amount.is_ok() && fee.is_ok(), egui::Button::new("deposit"))
            .clicked()
        {
            app.deposit(
                amount.expect("should not happen"),
                fee.expect("should not happen"),
            );
        }
    }
}
