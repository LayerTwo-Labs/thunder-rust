use eframe::egui;

use thunder::types::{Transaction, Txid};

use crate::app::App;

#[derive(Debug, Default)]
pub struct TxCreator {
    pub value_in: u64,
    pub value_out: u64,
    // if the base tx has changed, need to recompute final tx
    base_txid: Txid,
    final_tx: Option<Transaction>,
}

fn send_tx(app: &App, tx: &mut Transaction) -> anyhow::Result<()> {
    app.node.regenerate_proof(tx)?;
    let () = app.sign_and_send(tx.clone())?;
    Ok(())
}

impl TxCreator {
    pub fn show(
        &mut self,
        app: &mut App,
        ui: &mut egui::Ui,
        base_tx: &mut Transaction,
    ) -> anyhow::Result<()> {
        // if base txid has changed, store the new txid
        let base_txid = base_tx.txid();
        let base_txid_changed = base_txid != self.base_txid;
        if base_txid_changed {
            self.base_txid = base_txid;
        }
        // (re)compute final tx if:
        // * the tx type, tx data, or base txid has changed
        // * final tx not yet set
        let refresh_final_tx = base_txid_changed || self.final_tx.is_none();
        if refresh_final_tx {
            self.final_tx = Some(base_tx.clone());
        }
        let final_tx = match &mut self.final_tx {
            None => panic!("impossible! final tx should have been set"),
            Some(final_tx) => final_tx,
        };
        let txid = &format!("{}", final_tx.txid())[0..8];
        ui.monospace(format!("txid: {txid}"));
        if self.value_in >= self.value_out {
            let fee = self.value_in - self.value_out;
            let fee = bitcoin::Amount::from_sat(fee);
            ui.monospace(format!("fee:  {fee}"));
            if ui.button("sign and send").clicked() {
                if let Err(err) = send_tx(app, final_tx) {
                    tracing::error!("{err:#}");
                } else {
                    *base_tx = Transaction::default();
                    self.final_tx = None;
                }
            }
        } else {
            ui.label("Not Enough Value In");
        }
        Ok(())
    }
}
