use std::collections::VecDeque;

use heed::{types::SerdeBincode, Database, RoTxn, RwTxn};

use crate::types::{Accumulator, AuthorizedTransaction, OutPoint, Txid};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("heed error")]
    Heed(#[from] heed::Error),
    #[error("Utreexo error: {0}")]
    Utreexo(String),
    #[error("can't add transaction, utxo double spent")]
    UtxoDoubleSpent,
}

#[derive(Clone)]
pub struct MemPool {
    pub transactions:
        Database<SerdeBincode<Txid>, SerdeBincode<AuthorizedTransaction>>,
    pub spent_utxos: Database<SerdeBincode<OutPoint>, SerdeBincode<Txid>>,
}

impl MemPool {
    pub const NUM_DBS: u32 = 2;

    pub fn new(env: &heed::Env) -> Result<Self, Error> {
        let transactions = env.create_database(Some("transactions"))?;
        let spent_utxos = env.create_database(Some("spent_utxos"))?;
        Ok(Self {
            transactions,
            spent_utxos,
        })
    }

    pub fn put(
        &self,
        txn: &mut RwTxn,
        transaction: &AuthorizedTransaction,
    ) -> Result<(), Error> {
        let txid = transaction.transaction.txid();
        tracing::debug!("adding transaction {txid} to mempool");
        for (outpoint, _) in &transaction.transaction.inputs {
            if self.spent_utxos.get(txn, outpoint)?.is_some() {
                return Err(Error::UtxoDoubleSpent);
            }
            self.spent_utxos.put(txn, outpoint, &txid)?;
        }
        self.transactions.put(txn, &txid, transaction)?;
        Ok(())
    }

    pub fn delete(&self, rwtxn: &mut RwTxn, txid: Txid) -> Result<(), Error> {
        let mut pending_deletes = VecDeque::from([txid]);
        while let Some(txid) = pending_deletes.pop_front() {
            if let Some(tx) = self.transactions.get(rwtxn, &txid)? {
                for (outpoint, _) in &tx.transaction.inputs {
                    self.spent_utxos.delete(rwtxn, outpoint)?;
                }
                self.transactions.delete(rwtxn, &txid)?;
                for vout in 0..tx.transaction.outputs.len() {
                    let outpoint = OutPoint::Regular {
                        txid,
                        vout: vout as u32,
                    };
                    if let Some(child_txid) =
                        self.spent_utxos.get(rwtxn, &outpoint)?
                    {
                        pending_deletes.push_back(child_txid);
                    }
                }
            }
        }
        Ok(())
    }

    pub fn take(
        &self,
        txn: &RoTxn,
        number: usize,
    ) -> Result<Vec<AuthorizedTransaction>, Error> {
        let mut transactions = vec![];
        for item in self.transactions.iter(txn)?.take(number) {
            let (_, transaction) = item?;
            transactions.push(transaction);
        }
        Ok(transactions)
    }

    pub fn take_all(
        &self,
        txn: &RoTxn,
    ) -> Result<Vec<AuthorizedTransaction>, Error> {
        let mut transactions = vec![];
        for item in self.transactions.iter(txn)? {
            let (_, transaction) = item?;
            transactions.push(transaction);
        }
        Ok(transactions)
    }

    /// regenerate utreexo proofs for all txs in the mempool
    pub fn regenerate_proofs(
        &self,
        rwtxn: &mut RwTxn,
        accumulator: &Accumulator,
    ) -> Result<(), Error> {
        let mut iter = self.transactions.iter_mut(rwtxn)?;
        while let Some(tx) = iter.next() {
            let (txid, mut tx) = tx?;
            let targets: Vec<_> = tx
                .transaction
                .inputs
                .iter()
                .map(|(_, utxo_hash)| utxo_hash.into())
                .collect();
            tx.transaction.proof =
                accumulator.0.prove(&targets).map_err(Error::Utreexo)?;
            unsafe { iter.put_current(&txid, &tx) }?;
        }
        Ok(())
    }
}
