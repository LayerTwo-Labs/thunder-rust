use eframe::egui;

use crate::{
    app::App,
    gui::layout_style::{LayoutColors, LayoutDimensions, LayoutHelpers, LayoutUiExt},
};

#[derive(Debug, Default)]
pub struct Deposit {
    amount: String,
    fee: String,
}

impl Deposit {
    pub fn show(&mut self, app: Option<&App>, ui: &mut egui::Ui) {
        ui.layout_card(|ui| {
            // Amount field
            LayoutHelpers::btc_input_field(ui, &mut self.amount, "Amount", "Enter amount");

            ui.add_space(LayoutDimensions::ELEMENT_SPACING);

            // Fee field
            LayoutHelpers::btc_input_field(ui, &mut self.fee, "Fee", "Enter fee");

            // Parse values
            let amount = bitcoin::Amount::from_str_in(
                &self.amount,
                bitcoin::Denomination::Bitcoin,
            );

            // Show validation feedback for amount
            if !self.amount.is_empty() && amount.is_err() {
                ui.add_space(LayoutDimensions::SMALL_SPACING);
                ui.colored_label(
                    egui::Color32::from_rgb(255, 0, 0),
                    "Invalid amount format"
                );
            }

            let fee = bitcoin::Amount::from_str_in(
                &self.fee,
                bitcoin::Denomination::Bitcoin,
            );

            // Show validation feedback for fee
            if !self.fee.is_empty() && fee.is_err() {
                ui.add_space(LayoutDimensions::SMALL_SPACING);
                ui.colored_label(
                    egui::Color32::from_rgb(255, 0, 0),
                    "Invalid fee format"
                );
            }

            // Deposit button
            ui.add_space(LayoutDimensions::ELEMENT_SPACING);
            let enabled = app.is_some() && amount.is_ok() && fee.is_ok();
            if ui.add_enabled(enabled, egui::Button::new("Deposit")).clicked() {
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
        });
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
        ui.layout_card(|ui| {
            // Mainchain address field with generate button
            ui.add_space(LayoutDimensions::SMALL_SPACING);
            ui.label("Mainchain Address");
            ui.add_space(LayoutDimensions::SMALL_SPACING);

            ui.horizontal(|ui| {
                // Set a maximum length of 62 characters for Bitcoin addresses
                // This covers all Bitcoin address formats (P2PKH, P2SH, Bech32, etc.)
                if self.mainchain_address.len() > 62 {
                    self.mainchain_address.truncate(62);
                }

                let text_edit = egui::TextEdit::singleline(&mut self.mainchain_address)
                    .hint_text("Enter mainchain address")
                    .desired_width(ui.available_width() - 100.0)
                    .char_limit(62); // Enforce the 62 character limit

                ui.add(text_edit);
                ui.add_space(LayoutDimensions::SMALL_SPACING);

                if ui.add_enabled(app.is_some(), egui::Button::new("Generate")).clicked() {
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
            });

            ui.add_space(LayoutDimensions::ELEMENT_SPACING);

            // Amount field
            LayoutHelpers::btc_input_field(ui, &mut self.amount, "Amount", "Enter amount");

            ui.add_space(LayoutDimensions::ELEMENT_SPACING);

            // Fee field
            LayoutHelpers::btc_input_field(ui, &mut self.fee, "Fee", "Enter fee");

            ui.add_space(LayoutDimensions::ELEMENT_SPACING);

            // Mainchain fee field
            LayoutHelpers::btc_input_field(ui, &mut self.mainchain_fee, "Mainchain Fee", "Enter mainchain fee");

            // Parse values
            let mainchain_address: Option<
                bitcoin::Address<bitcoin::address::NetworkUnchecked>,
            > = self.mainchain_address.parse().ok();

            // Show validation feedback for the mainchain address
            if !self.mainchain_address.is_empty() && mainchain_address.is_none() {
                ui.add_space(LayoutDimensions::SMALL_SPACING);
                ui.colored_label(
                    egui::Color32::from_rgb(255, 0, 0),
                    "Invalid Bitcoin address format"
                );
            }

            let amount = bitcoin::Amount::from_str_in(
                &self.amount,
                bitcoin::Denomination::Bitcoin,
            );

            // Show validation feedback for amount
            if !self.amount.is_empty() && amount.is_err() {
                ui.add_space(LayoutDimensions::SMALL_SPACING);
                ui.colored_label(
                    egui::Color32::from_rgb(255, 0, 0),
                    "Invalid amount format"
                );
            }

            let fee = bitcoin::Amount::from_str_in(
                &self.fee,
                bitcoin::Denomination::Bitcoin,
            );

            // Show validation feedback for fee
            if !self.fee.is_empty() && fee.is_err() {
                ui.add_space(LayoutDimensions::SMALL_SPACING);
                ui.colored_label(
                    egui::Color32::from_rgb(255, 0, 0),
                    "Invalid fee format"
                );
            }

            let mainchain_fee = bitcoin::Amount::from_str_in(
                &self.mainchain_fee,
                bitcoin::Denomination::Bitcoin,
            );

            // Show validation feedback for mainchain fee
            if !self.mainchain_fee.is_empty() && mainchain_fee.is_err() {
                ui.add_space(LayoutDimensions::SMALL_SPACING);
                ui.colored_label(
                    egui::Color32::from_rgb(255, 0, 0),
                    "Invalid mainchain fee format"
                );
            }

            // Withdraw button
            ui.add_space(LayoutDimensions::ELEMENT_SPACING);
            let enabled = app.is_some()
                && mainchain_address.is_some()
                && amount.is_ok()
                && fee.is_ok()
                && mainchain_fee.is_ok();

            if ui.add_enabled(enabled, egui::Button::new("Withdraw")).clicked() {
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
        });
    }
}

#[derive(Default)]
pub(super) struct Transfer {
    deposit: Deposit,
    withdrawal: Withdrawal,
}

impl Transfer {
    pub fn show(&mut self, app: Option<&App>, ui: &mut egui::Ui) {
        // Set background color for the entire panel
        let frame = egui::Frame::none()
            .fill(LayoutColors::BACKGROUND)
            .inner_margin(LayoutDimensions::CONTAINER_PADDING);

        frame.show(ui, |ui| {
            // Create a horizontal layout with spacing between panels
            ui.horizontal(|ui| {
                // Left panel (Deposit)
                ui.vertical(|ui| {
                    let available_width = ui.available_width() / 2.0 - LayoutDimensions::ELEMENT_SPACING;
                    ui.set_width(available_width);

                    ui.layout_heading("Deposit");
                    self.deposit.show(app, ui);
                });

                // Add spacing between panels
                ui.add_space(LayoutDimensions::SECTION_SPACING);

                // Right panel (Withdrawal)
                ui.vertical(|ui| {
                    let available_width = ui.available_width();
                    ui.set_width(available_width);

                    ui.layout_heading("Withdrawal");
                    self.withdrawal.show(app, ui);
                });
            });
        });
    }
}
