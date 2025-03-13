use std::collections::VecDeque;

use fallible_iterator::FallibleIterator as _;
use heed::types::SerdeBincode;
use sneed::{
    DatabaseUnique, EnvError, RoTxn, RwTxn, RwTxnError, UnitKey,
    db::error::Error as DbError,
};

use crate::types::{
    Accumulator, AuthorizedTransaction, OutPoint, Txid, UtreexoError, VERSION,
    Version,
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Db(#[from] DbError),
    #[error("Database env error")]
    DbEnv(#[from] EnvError),
    #[error("Database write error")]
    DbWrite(#[from] RwTxnError),
    #[error(transparent)]
    Utreexo(#[from] UtreexoError),
    #[error("can't add transaction, utxo double spent")]
    UtxoDoubleSpent,
}

#[derive(Clone)]
pub struct MemPool {
    pub transactions:
        DatabaseUnique<SerdeBincode<Txid>, SerdeBincode<AuthorizedTransaction>>,
    pub spent_utxos: DatabaseUnique<SerdeBincode<OutPoint>, SerdeBincode<Txid>>,
    _version: DatabaseUnique<UnitKey, SerdeBincode<Version>>,
}

impl MemPool {
    pub const NUM_DBS: u32 = 3;

    pub fn new(env: &sneed::Env) -> Result<Self, Error> {
        let mut rwtxn = env.write_txn().map_err(EnvError::from)?;
        let transactions =
            DatabaseUnique::create(env, &mut rwtxn, "transactions")
                .map_err(EnvError::from)?;
        let spent_utxos =
            DatabaseUnique::create(env, &mut rwtxn, "spent_utxos")
                .map_err(EnvError::from)?;
        let version =
            DatabaseUnique::create(env, &mut rwtxn, "mempool_version")
                .map_err(EnvError::from)?;
        if version
            .try_get(&rwtxn, &())
            .map_err(DbError::from)?
            .is_none()
        {
            version
                .put(&mut rwtxn, &(), &*VERSION)
                .map_err(DbError::from)?;
        }
        rwtxn.commit().map_err(RwTxnError::from)?;
        Ok(Self {
            transactions,
            spent_utxos,
            _version: version,
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
            if self
                .spent_utxos
                .try_get(txn, outpoint)
                .map_err(DbError::from)?
                .is_some()
            {
                return Err(Error::UtxoDoubleSpent);
            }
            self.spent_utxos
                .put(txn, outpoint, &txid)
                .map_err(DbError::from)?;
        }
        self.transactions
            .put(txn, &txid, transaction)
            .map_err(DbError::from)?;
        Ok(())
    }

    pub fn delete(&self, rwtxn: &mut RwTxn, txid: Txid) -> Result<(), Error> {
        let mut pending_deletes = VecDeque::from([txid]);
        while let Some(txid) = pending_deletes.pop_front() {
            if let Some(tx) = self
                .transactions
                .try_get(rwtxn, &txid)
                .map_err(DbError::from)?
            {
                for (outpoint, _) in &tx.transaction.inputs {
                    self.spent_utxos
                        .delete(rwtxn, outpoint)
                        .map_err(DbError::from)?;
                }
                self.transactions
                    .delete(rwtxn, &txid)
                    .map_err(DbError::from)?;
                for vout in 0..tx.transaction.outputs.len() {
                    let outpoint = OutPoint::Regular {
                        txid,
                        vout: vout as u32,
                    };
                    if let Some(child_txid) = self
                        .spent_utxos
                        .try_get(rwtxn, &outpoint)
                        .map_err(DbError::from)?
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
        rotxn: &RoTxn,
        number: usize,
    ) -> Result<Vec<AuthorizedTransaction>, Error> {
        self.transactions
            .iter(rotxn)
            .map_err(DbError::from)?
            .take(number)
            .map(|(_, transaction)| Ok(transaction))
            .collect()
            .map_err(|err| DbError::from(err).into())
    }

    pub fn take_all(
        &self,
        rotxn: &RoTxn,
    ) -> Result<Vec<AuthorizedTransaction>, Error> {
        self.transactions
            .iter(rotxn)
            .map_err(DbError::from)?
            .map(|(_, transaction)| Ok(transaction))
            .collect()
            .map_err(|err| DbError::from(err).into())
    }

    /// regenerate utreexo proofs for all txs in the mempool
    pub fn regenerate_proofs(
        &self,
        rwtxn: &mut RwTxn,
        accumulator: &Accumulator,
    ) -> Result<(), Error> {
        let txids: Vec<_> = self
            .transactions
            .iter_keys(rwtxn)
            .map_err(DbError::from)?
            .collect()
            .map_err(DbError::from)?;
        for txid in txids {
            let mut tx =
                self.transactions.get(rwtxn, &txid).map_err(DbError::from)?;
            let targets: Vec<_> = tx
                .transaction
                .inputs
                .iter()
                .map(|(_, utxo_hash)| utxo_hash.into())
                .collect();
            tx.transaction.proof = accumulator.prove(&targets)?;
            self.transactions
                .put(rwtxn, &txid, &tx)
                .map_err(DbError::from)?;
        }
        Ok(())
    }
}
