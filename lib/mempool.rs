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

#[cfg(test)]
mod p2p_validation_bypass_tests {
    //! Regression tests for the P2P transaction-validation gap.
    //! MemPool::put tracks conflicts, but it is not a transaction validator.
    //! Peer-supplied transactions must pass State::validate_transaction before
    //! the net task regenerates proofs or inserts them into the mempool.

    use bitcoin::Amount;
    use heed::EnvOpenOptions;

    use super::MemPool;
    use crate::authorization::{Authorization, get_address, sign};
    use crate::state::State;
    use crate::types::{
        Accumulator, AccumulatorDiff, Address, OutPoint, OutPointKey, Output,
        OutputContent, PointedOutput, Transaction, Txid, hash,
    };

    // ed25519 signing keys from fixed seeds (deterministic; no RNG).
    fn signing_key(seed: u8) -> crate::authorization::SigningKey {
        crate::authorization::SigningKey::from_bytes(&[seed; 32])
    }

    fn temp_env() -> sneed::Env {
        let dir = std::env::temp_dir()
            .join(format!("thunder-m001-{}", std::process::id()));
        drop(std::fs::remove_dir_all(&dir));
        std::fs::create_dir_all(&dir).expect("create temp env dir");
        let mut opts = EnvOpenOptions::new();
        opts.map_size(16 * 1024 * 1024)
            .max_dbs(State::NUM_DBS + MemPool::NUM_DBS + 4);
        unsafe { sneed::Env::open(&opts, &dir) }.expect("open env")
    }

    #[test]
    fn mempool_put_is_not_transaction_validation() {
        let env = temp_env();
        let state = State::new(&env).expect("State::new");
        let mempool = MemPool::new(&env).expect("MemPool::new");

        let victim = signing_key(1);
        let victim_addr: Address = get_address(&victim.verifying_key());
        let funding_outpoint = OutPoint::Regular {
            txid: Txid([7u8; 32]),
            vout: 0,
        };
        let funded_output = Output {
            address: victim_addr,
            content: OutputContent::Value(Amount::from_sat(100_000)),
        };
        let utxo_hash = hash(&PointedOutput {
            outpoint: funding_outpoint,
            output: funded_output.clone(),
        });
        {
            let mut rwtxn = env.write_txn().expect("write txn");
            state
                .utxos
                .put(
                    &mut rwtxn,
                    &OutPointKey::from(funding_outpoint),
                    &funded_output,
                )
                .expect("put utxo");
            // Keep the accumulator consistent so proof regeneration succeeds.
            let mut acc: Accumulator =
                state.get_accumulator(&rwtxn).expect("get accumulator");
            let mut diff = AccumulatorDiff::default();
            diff.insert(utxo_hash.into());
            acc.apply_diff(diff).expect("apply diff");
            state
                .utreexo_accumulator
                .put(&mut rwtxn, &(), &acc)
                .expect("put accumulator");
            rwtxn.commit().expect("commit funding");
        }

        let tx = Transaction {
            inputs: vec![(funding_outpoint, utxo_hash)],
            outputs: vec![Output {
                address: get_address(&signing_key(2).verifying_key()),
                content: OutputContent::Value(Amount::from_sat(90_000)),
            }],
            ..Default::default()
        };

        {
            let mut wrong_hash_tx = tx.clone();
            wrong_hash_tx.inputs[0].1 = [9u8; 32];
            let wrong_hash_authd_tx = crate::types::AuthorizedTransaction {
                transaction: wrong_hash_tx.clone(),
                authorizations: vec![Authorization {
                    verifying_key: victim.verifying_key(),
                    signature: sign(&victim, &wrong_hash_tx).expect("sign"),
                }],
            };
            let rotxn = env.read_txn().expect("read txn");
            let result =
                state.validate_transaction(&rotxn, &wrong_hash_authd_tx);
            assert!(
                matches!(
                    result,
                    Err(crate::state::ValidateTransaction::WrongUtxoHash)
                ),
                "validate_transaction must reject a mismatched UTXO hash, got {result:?}"
            );
        }

        // Present the victim verifying key, but sign with a different key.
        let attacker = signing_key(2);
        let forged = Authorization {
            verifying_key: victim.verifying_key(),
            signature: sign(&attacker, &tx).expect("sign"),
        };
        let mut authd_tx = crate::types::AuthorizedTransaction {
            transaction: tx,
            authorizations: vec![forged],
        };

        // The validator rejects the forged signature.
        {
            let rotxn = env.read_txn().expect("read txn");
            let result = state.validate_transaction(&rotxn, &authd_tx);
            assert!(
                matches!(
                    result,
                    Err(crate::state::ValidateTransaction::Authorization)
                ),
                "validate_transaction must reject the forged-signature tx with an Authorization error, got {result:?}"
            );
        }

        // The old lower-level sequence accepted it because proof regeneration
        // and mempool insertion are not signature-validation gates.
        {
            let mut rwtxn = env.write_txn().expect("write txn");
            state
                .regenerate_proof(&rwtxn, &mut authd_tx.transaction)
                .expect("regenerate_proof accepted the forged tx");
            mempool
                .put(&mut rwtxn, &authd_tx)
                .expect("mempool.put accepted the invalid tx (the bug)");
            rwtxn.commit().expect("commit mempool");
        }

        // Confirm the invalid tx is now resident in the mempool.
        {
            let rotxn = env.read_txn().expect("read txn");
            let in_mempool = mempool.take_all(&rotxn).expect("take_all");
            assert_eq!(
                in_mempool.len(),
                1,
                "the forged-signature tx must be sitting in the mempool"
            );
            assert_eq!(
                in_mempool[0].transaction.txid(),
                authd_tx.transaction.txid(),
            );
        }
    }
}
