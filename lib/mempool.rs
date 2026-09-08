use std::collections::{HashMap, HashSet, VecDeque};

use fallible_iterator::FallibleIterator as _;
use heed::types::SerdeBincode;
use sneed::{
    DatabaseUnique, DbError, EnvError, RoTxn, RwTxn, RwTxnError, UnitKey, db,
};

use crate::{
    types::{
        Accumulator, Address, AuthorizedTransaction, OutPoint, Output,
        Transaction, Txid, UtreexoError, VERSION, Version,
    },
    util::Watchable,
};

/// Longest chain of unconfirmed transactions the mempool accepts. Bitcoin
/// Core holds the same number in `DEFAULT_ANCESTOR_LIMIT`.
pub const MAX_UNCONFIRMED_ANCESTORS: usize = 25;

#[allow(clippy::duplicated_attributes)]
#[derive(Debug, thiserror::Error, transitive::Transitive)]
#[transitive(from(db::error::TryGet, DbError))]
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
    #[error(
        "can't add transaction, it has {count} unconfirmed ancestors and the \
         limit is {MAX_UNCONFIRMED_ANCESTORS}"
    )]
    TooManyAncestors { count: usize },
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

    pub fn new<Tls>(env: &sneed::Env<Tls>) -> Result<Self, Error> {
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
        let ancestors = self.ancestors(txn, &transaction.transaction)?;
        if ancestors.len() >= MAX_UNCONFIRMED_ANCESTORS {
            return Err(Error::TooManyAncestors {
                count: ancestors.len(),
            });
        }
        self.insert(txn, transaction)
    }

    /// Take back a transaction that a disconnected block carried. The chain
    /// accepted it once, so the ancestor limit does not apply.
    pub fn put_disconnected(
        &self,
        txn: &mut RwTxn,
        transaction: &AuthorizedTransaction,
    ) -> Result<(), Error> {
        self.insert(txn, transaction)
    }

    fn insert(
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

    /// Remove a transaction that a block confirms, and keep its children. A
    /// child of a confirmed parent spends a confirmed output, so it stays
    /// valid.
    pub fn delete_confirmed(
        &self,
        rwtxn: &mut RwTxn,
        txid: Txid,
    ) -> Result<(), Error> {
        let Some(tx) = self
            .transactions
            .try_get(rwtxn, &txid)
            .map_err(DbError::from)?
        else {
            return Ok(());
        };
        for (outpoint, _) in &tx.transaction.inputs {
            self.spent_utxos
                .delete(rwtxn, outpoint)
                .map_err(DbError::from)?;
        }
        self.transactions
            .delete(rwtxn, &txid)
            .map_err(DbError::from)?;
        Ok(())
    }

    /// The mempool transactions that `transaction` spends from, directly or
    /// through another mempool transaction.
    pub fn ancestors(
        &self,
        rotxn: &RoTxn,
        transaction: &Transaction,
    ) -> Result<HashSet<Txid>, Error> {
        let mut found = HashSet::new();
        let mut pending: VecDeque<Txid> = transaction
            .inputs
            .iter()
            .filter_map(|(outpoint, _)| parent_txid(outpoint))
            .collect();
        while let Some(txid) = pending.pop_front() {
            if found.contains(&txid) {
                continue;
            }
            let Some(tx) = self
                .transactions
                .try_get(rotxn, &txid)
                .map_err(DbError::from)?
            else {
                continue;
            };
            found.insert(txid);
            pending.extend(
                tx.transaction
                    .inputs
                    .iter()
                    .filter_map(|(outpoint, _)| parent_txid(outpoint)),
            );
        }
        Ok(found)
    }

    /// The outputs this mempool holds that `transaction` spends. The confirmed
    /// UTXO set holds none of them.
    pub fn unconfirmed_outputs(
        &self,
        rotxn: &RoTxn,
        transaction: &Transaction,
    ) -> Result<HashMap<OutPoint, Output>, Error> {
        let mut res = HashMap::new();
        for (outpoint, _) in &transaction.inputs {
            let OutPoint::Regular { txid, vout } = outpoint else {
                continue;
            };
            let Some(parent) = self
                .transactions
                .try_get(rotxn, txid)
                .map_err(DbError::from)?
            else {
                continue;
            };
            let Some(output) = parent.transaction.outputs.get(*vout as usize)
            else {
                continue;
            };
            res.insert(*outpoint, output.clone());
        }
        Ok(res)
    }

    /// Transactions with a parent before its child. `limit` caps how many the
    /// walk returns, and the result stays closed under parents, so a shorter
    /// walk never gives a child whose parent it left out. The read still
    /// covers the whole mempool; the limit bounds the walk and the clones.
    pub fn topological(
        &self,
        rotxn: &RoTxn,
        limit: Option<usize>,
    ) -> Result<Vec<AuthorizedTransaction>, Error> {
        let txs: Vec<(Txid, AuthorizedTransaction)> = self
            .transactions
            .iter(rotxn)
            .map_err(DbError::from)?
            .collect()
            .map_err(DbError::from)?;
        let by_txid: HashMap<Txid, &AuthorizedTransaction> =
            txs.iter().map(|(txid, tx)| (*txid, tx)).collect();
        let mut order = Vec::with_capacity(txs.len());
        let mut placed = HashSet::with_capacity(txs.len());
        for (txid, _) in &txs {
            if limit.is_some_and(|limit| order.len() >= limit) {
                break;
            }
            let mut stack = vec![*txid];
            while let Some(txid) = stack.last().copied() {
                if limit.is_some_and(|limit| order.len() >= limit) {
                    break;
                }
                if placed.contains(&txid) {
                    stack.pop();
                    continue;
                }
                let Some(tx) = by_txid.get(&txid) else {
                    stack.pop();
                    continue;
                };
                let parent =
                    tx.transaction.inputs.iter().find_map(|(outpoint, _)| {
                        parent_txid(outpoint).filter(|parent| {
                            by_txid.contains_key(parent)
                                && !placed.contains(parent)
                        })
                    });
                match parent {
                    Some(parent) => stack.push(parent),
                    None => {
                        placed.insert(txid);
                        order.push((*tx).clone());
                        stack.pop();
                    }
                }
            }
        }
        Ok(order)
    }

    /// The unconfirmed outputs that pay one of `addresses` and that this
    /// wallet made on its own.
    ///
    /// `confirmed` names the outputs the wallet already holds from the chain.
    /// Bitcoin Core takes an unconfirmed output only when the wallet funded
    /// every input of the transaction that made it, and it walks the parents
    /// to the last confirmed one. This copies that rule, so an unconfirmed
    /// payment from a stranger never appears here.
    ///
    /// The wallet counts these outputs in its balance whatever the
    /// `--spend-zero-conf-change` option says. The option decides only whether
    /// the wallet may put them in a new transaction.
    pub fn own_unconfirmed_utxos(
        &self,
        rotxn: &RoTxn,
        addresses: &HashSet<Address>,
        confirmed: &HashSet<OutPoint>,
    ) -> Result<HashMap<OutPoint, Output>, Error> {
        let mut res = HashMap::new();
        // A parent comes first, so its trust and its ancestors are known by the
        // time the walk reaches the child.
        let mut trusted: HashSet<OutPoint> = HashSet::new();
        let mut ancestors: HashMap<Txid, HashSet<Txid>> = HashMap::new();
        for tx in self.topological(rotxn, None)? {
            let txid = tx.transaction.txid();
            let mut is_trusted = true;
            let mut tx_ancestors = HashSet::new();
            for (outpoint, _) in &tx.transaction.inputs {
                if let Some(parent) = parent_txid(outpoint)
                    && let Some(parent_ancestors) = ancestors.get(&parent)
                {
                    tx_ancestors.extend(parent_ancestors.iter().copied());
                    tx_ancestors.insert(parent);
                }
                if confirmed.contains(outpoint) || trusted.contains(outpoint) {
                    continue;
                }
                is_trusted = false;
            }
            ancestors.insert(txid, tx_ancestors);
            if !is_trusted {
                continue;
            }
            for (vout, output) in tx.transaction.outputs.iter().enumerate() {
                if !addresses.contains(&output.address) {
                    continue;
                }
                let outpoint = OutPoint::Regular {
                    txid,
                    vout: vout as u32,
                };
                trusted.insert(outpoint);
                if self
                    .spent_utxos
                    .try_get(rotxn, &outpoint)
                    .map_err(DbError::from)?
                    .is_none()
                {
                    res.insert(outpoint, output.clone());
                }
            }
        }
        Ok(res)
    }

    /// The transaction in this mempool that spends `outpoint`, if there is
    /// one.
    pub fn spender(
        &self,
        rotxn: &RoTxn,
        outpoint: &OutPoint,
    ) -> Result<Option<Txid>, Error> {
        let txid = self
            .spent_utxos
            .try_get(rotxn, outpoint)
            .map_err(DbError::from)?;
        Ok(txid)
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
    ///
    /// A transaction whose inputs can no longer be proven against the
    /// accumulator (eg. because they were spent by a just-connected block via
    /// a conflicting transaction) is no longer valid. Such a transaction is
    /// evicted from the mempool, along with its descendants, rather than
    /// propagating an error that would abort block connect/disconnect.
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
            // The tx may already have been evicted as a descendant of an
            // earlier invalidated tx.
            let Some(mut tx) = self
                .transactions
                .try_get(rwtxn, &txid)
                .map_err(DbError::from)?
            else {
                continue;
            };
            let unconfirmed =
                self.unconfirmed_outputs(rwtxn, &tx.transaction)?;
            let targets: Vec<_> = tx
                .transaction
                .inputs
                .iter()
                .filter(|(outpoint, _)| !unconfirmed.contains_key(outpoint))
                .map(|(_, utxo_hash)| utxo_hash.into())
                .collect();
            match accumulator.prove(&targets) {
                Ok(proof) => {
                    tx.transaction.proof = proof;
                    self.transactions
                        .put(rwtxn, &txid, &tx)
                        .map_err(DbError::from)?;
                }
                Err(_) => {
                    tracing::debug!(
                        "evicting mempool transaction {txid}: inputs no \
                         longer in accumulator"
                    );
                    let () = self.delete(rwtxn, txid)?;
                }
            }
        }
        Ok(())
    }
}

impl Watchable<()> for MemPool {
    type WatchStream = tokio_stream::wrappers::WatchStream<()>;

    /// Get a signal that notifies whenever the mempool changes
    fn watch(&self) -> Self::WatchStream {
        tokio_stream::wrappers::WatchStream::new(
            self.transactions.watch().clone(),
        )
    }
}

/// The mempool transaction that could have made this outpoint. A coinbase or
/// a deposit outpoint names no transaction.
fn parent_txid(outpoint: &OutPoint) -> Option<Txid> {
    match outpoint {
        OutPoint::Regular { txid, .. } => Some(*txid),
        OutPoint::Coinbase { .. } | OutPoint::Deposit(_) => None,
    }
}

#[cfg(test)]
mod test {
    use bitcoin::hashes::Hash as _;

    use super::*;
    use crate::types::{
        Address, OutputContent, PointedOutput,
        authorization::{SigningKey, get_address},
        hash,
    };

    fn temp_env(
        test_name: &str,
    ) -> anyhow::Result<(temp_dir::TempDir, sneed::Env)> {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let temp_dir = temp_dir::TempDir::with_prefix(format!(
            "{test_name}-{nanos}-{}",
            std::process::id()
        ))?;
        let mut opts = heed::EnvOpenOptions::new();
        opts.map_size(16 * 1024 * 1024).max_dbs(MemPool::NUM_DBS);
        let env = unsafe { sneed::Env::open(&opts, temp_dir.path()) }?;
        Ok((temp_dir, env))
    }

    fn value_output(address: Address, sats: u64) -> Output {
        Output {
            address,
            content: OutputContent::Value(bitcoin::Amount::from_sat(sats)),
        }
    }

    fn deposit_outpoint(seed: u8) -> OutPoint {
        OutPoint::Deposit(bitcoin::OutPoint {
            txid: bitcoin::Txid::from_byte_array([seed; 32]),
            vout: 0,
        })
    }

    /// Build a transaction that spends `outpoint`, worth `output` after it.
    /// The mempool never checks a signature, so the authorization is empty.
    fn spend(
        outpoint: OutPoint,
        spent: &Output,
        output: Output,
    ) -> AuthorizedTransaction {
        let utxo_hash = hash(&PointedOutput {
            outpoint,
            output: spent.clone(),
        });
        AuthorizedTransaction {
            authorizations: Vec::new(),
            transaction: Transaction {
                inputs: vec![(outpoint, utxo_hash)],
                proof: Default::default(),
                outputs: vec![output],
            },
        }
    }

    /// A chain of `len` transactions, each spending the one before it.
    fn chain(
        address: Address,
        start: OutPoint,
        start_output: Output,
        len: usize,
    ) -> Vec<AuthorizedTransaction> {
        let mut txs = Vec::with_capacity(len);
        let mut outpoint = start;
        let mut spent = start_output;
        for i in 0..len {
            let output = value_output(address, 10_000 - i as u64 - 1);
            let tx = spend(outpoint, &spent, output.clone());
            outpoint = OutPoint::Regular {
                txid: tx.transaction.txid(),
                vout: 0,
            };
            spent = output;
            txs.push(tx);
        }
        txs
    }

    fn owner() -> (SigningKey, Address) {
        let key = SigningKey::from_bytes(&[0x44; 32]);
        let address = get_address(&key.verifying_key());
        (key, address)
    }

    #[test]
    fn topological_puts_a_parent_before_its_child() -> anyhow::Result<()> {
        let (_temp_dir, env) = temp_env("topological")?;
        let mempool = MemPool::new(&env)?;
        let (_key, address) = owner();
        let start = deposit_outpoint(0x01);
        let start_output = value_output(address, 10_000);
        let txs = chain(address, start, start_output, 4);

        let mut rwtxn = env.write_txn()?;
        // Insert the children first, so key order cannot pass the test by
        // accident.
        for tx in txs.iter().rev() {
            mempool.put(&mut rwtxn, tx)?;
        }
        rwtxn.commit()?;

        let rotxn = env.read_txn()?;
        let order: Vec<_> = mempool
            .topological(&rotxn, None)?
            .into_iter()
            .map(|tx| tx.transaction.txid())
            .collect();
        let expected: Vec<_> =
            txs.iter().map(|tx| tx.transaction.txid()).collect();
        anyhow::ensure!(
            order == expected,
            "expected {expected:?}, got {order:?}"
        );
        Ok(())
    }

    #[test]
    fn the_mempool_refuses_a_chain_past_the_limit() -> anyhow::Result<()> {
        let (_temp_dir, env) = temp_env("ancestor_limit")?;
        let mempool = MemPool::new(&env)?;
        let (_key, address) = owner();
        let start = deposit_outpoint(0x02);
        let start_output = value_output(address, 10_000);
        let txs =
            chain(address, start, start_output, MAX_UNCONFIRMED_ANCESTORS + 1);

        let mut rwtxn = env.write_txn()?;
        for tx in txs.iter().take(MAX_UNCONFIRMED_ANCESTORS) {
            mempool.put(&mut rwtxn, tx)?;
        }
        let last = mempool.put(&mut rwtxn, &txs[MAX_UNCONFIRMED_ANCESTORS]);
        anyhow::ensure!(
            matches!(last, Err(Error::TooManyAncestors { count }) if count
                == MAX_UNCONFIRMED_ANCESTORS),
            "expected TooManyAncestors, got {last:?}",
        );
        Ok(())
    }

    #[test]
    fn a_confirmed_parent_leaves_its_child_behind() -> anyhow::Result<()> {
        let (_temp_dir, env) = temp_env("delete_confirmed")?;
        let mempool = MemPool::new(&env)?;
        let (_key, address) = owner();
        let start = deposit_outpoint(0x03);
        let start_output = value_output(address, 10_000);
        let txs = chain(address, start, start_output, 2);

        let mut rwtxn = env.write_txn()?;
        for tx in &txs {
            mempool.put(&mut rwtxn, tx)?;
        }
        mempool.delete_confirmed(&mut rwtxn, txs[0].transaction.txid())?;
        rwtxn.commit()?;

        let rotxn = env.read_txn()?;
        let left: Vec<_> = mempool
            .take_all(&rotxn)?
            .into_iter()
            .map(|tx| tx.transaction.txid())
            .collect();
        anyhow::ensure!(
            left == vec![txs[1].transaction.txid()],
            "the child must stay, got {left:?}",
        );
        Ok(())
    }

    #[test]
    fn a_double_spending_parent_takes_its_child() -> anyhow::Result<()> {
        let (_temp_dir, env) = temp_env("delete_cascades")?;
        let mempool = MemPool::new(&env)?;
        let (_key, address) = owner();
        let start = deposit_outpoint(0x04);
        let start_output = value_output(address, 10_000);
        let txs = chain(address, start, start_output, 2);

        let mut rwtxn = env.write_txn()?;
        for tx in &txs {
            mempool.put(&mut rwtxn, tx)?;
        }
        mempool.delete(&mut rwtxn, txs[0].transaction.txid())?;
        rwtxn.commit()?;

        let rotxn = env.read_txn()?;
        anyhow::ensure!(mempool.take_all(&rotxn)?.is_empty());
        Ok(())
    }

    #[test]
    fn own_change_is_spendable_and_a_stranger_output_is_not()
    -> anyhow::Result<()> {
        let (_temp_dir, env) = temp_env("trust")?;
        let mempool = MemPool::new(&env)?;
        let (_key, address) = owner();
        let stranger =
            get_address(&SigningKey::from_bytes(&[0x55; 32]).verifying_key());

        // The wallet holds one confirmed coin and spends it. The change is
        // its own, so it is trusted.
        let mine = deposit_outpoint(0x05);
        let mine_output = value_output(address, 10_000);
        let own = spend(mine, &mine_output, value_output(address, 9_000));
        // A stranger spends a coin the wallet never held, and pays the wallet.
        let theirs = deposit_outpoint(0x06);
        let theirs_output = value_output(stranger, 5_000);
        let gift = spend(theirs, &theirs_output, value_output(address, 4_000));

        let mut rwtxn = env.write_txn()?;
        mempool.put(&mut rwtxn, &own)?;
        mempool.put(&mut rwtxn, &gift)?;
        rwtxn.commit()?;

        let addresses = HashSet::from([address]);
        let confirmed = HashSet::from([mine]);
        let own_outpoint = OutPoint::Regular {
            txid: own.transaction.txid(),
            vout: 0,
        };
        let gift_outpoint = OutPoint::Regular {
            txid: gift.transaction.txid(),
            vout: 0,
        };

        let rotxn = env.read_txn()?;
        let spendable =
            mempool.own_unconfirmed_utxos(&rotxn, &addresses, &confirmed)?;
        anyhow::ensure!(
            spendable.keys().collect::<Vec<_>>() == vec![&own_outpoint],
            "only own change is trusted, got {spendable:?}",
        );
        anyhow::ensure!(
            !spendable.contains_key(&gift_outpoint),
            "an unconfirmed payment from someone else waits for a block",
        );
        Ok(())
    }

    #[test]
    fn an_untrusted_ancestor_stops_the_whole_chain() -> anyhow::Result<()> {
        let (_temp_dir, env) = temp_env("trust_chain")?;
        let mempool = MemPool::new(&env)?;
        let (_key, address) = owner();
        let stranger =
            get_address(&SigningKey::from_bytes(&[0x66; 32]).verifying_key());

        let theirs = deposit_outpoint(0x07);
        let theirs_output = value_output(stranger, 5_000);
        let gift_output = value_output(address, 4_000);
        let gift = spend(theirs, &theirs_output, gift_output.clone());
        let gift_outpoint = OutPoint::Regular {
            txid: gift.transaction.txid(),
            vout: 0,
        };
        let child =
            spend(gift_outpoint, &gift_output, value_output(address, 3_000));

        let mut rwtxn = env.write_txn()?;
        mempool.put(&mut rwtxn, &gift)?;
        mempool.put(&mut rwtxn, &child)?;
        rwtxn.commit()?;

        let rotxn = env.read_txn()?;
        let spendable = mempool.own_unconfirmed_utxos(
            &rotxn,
            &HashSet::from([address]),
            &HashSet::new(),
        )?;
        anyhow::ensure!(
            spendable.is_empty(),
            "a child of a stranger's transaction is not trusted, got \
             {spendable:?}",
        );
        Ok(())
    }

    /// A chain at the limit stays visible, so the balance shows the money.
    /// The mempool refuses the next link, and the user reads a clear error.
    #[test]
    fn a_chain_at_the_limit_stays_visible() -> anyhow::Result<()> {
        let (_temp_dir, env) = temp_env("ancestor_boundary")?;
        let mempool = MemPool::new(&env)?;
        let (_key, address) = owner();
        let start = deposit_outpoint(0x09);
        let start_output = value_output(address, 10_000);
        let txs =
            chain(address, start, start_output, MAX_UNCONFIRMED_ANCESTORS);

        let mut rwtxn = env.write_txn()?;
        for tx in &txs {
            mempool.put(&mut rwtxn, tx)?;
        }
        rwtxn.commit()?;

        let last = &txs[MAX_UNCONFIRMED_ANCESTORS - 1];
        let tip = OutPoint::Regular {
            txid: last.transaction.txid(),
            vout: 0,
        };
        let rotxn = env.read_txn()?;
        anyhow::ensure!(
            mempool.ancestors(&rotxn, &last.transaction)?.len()
                == MAX_UNCONFIRMED_ANCESTORS - 1,
        );
        let visible = mempool.own_unconfirmed_utxos(
            &rotxn,
            &HashSet::from([address]),
            &HashSet::from([start]),
        )?;
        anyhow::ensure!(
            visible.keys().collect::<Vec<_>>() == vec![&tip],
            "the money the chain holds must stay visible, got {visible:?}",
        );
        drop(rotxn);

        // A child of the last link would carry 25 ancestors.
        let child = spend(
            tip,
            &value_output(address, 10_000 - MAX_UNCONFIRMED_ANCESTORS as u64),
            value_output(address, 1),
        );
        let mut rwtxn = env.write_txn()?;
        let refused = mempool.put(&mut rwtxn, &child);
        anyhow::ensure!(
            matches!(refused, Err(Error::TooManyAncestors { count }) if count
                == MAX_UNCONFIRMED_ANCESTORS),
            "the mempool must refuse the next link, got {refused:?}",
        );
        Ok(())
    }

    #[test]
    fn a_spent_unconfirmed_output_is_not_offered() -> anyhow::Result<()> {
        let (_temp_dir, env) = temp_env("trust_spent")?;
        let mempool = MemPool::new(&env)?;
        let (_key, address) = owner();
        let start = deposit_outpoint(0x08);
        let start_output = value_output(address, 10_000);
        let txs = chain(address, start, start_output, 2);

        let mut rwtxn = env.write_txn()?;
        for tx in &txs {
            mempool.put(&mut rwtxn, tx)?;
        }
        rwtxn.commit()?;

        let rotxn = env.read_txn()?;
        let spendable = mempool.own_unconfirmed_utxos(
            &rotxn,
            &HashSet::from([address]),
            &HashSet::from([start]),
        )?;
        let tip = OutPoint::Regular {
            txid: txs[1].transaction.txid(),
            vout: 0,
        };
        anyhow::ensure!(
            spendable.keys().collect::<Vec<_>>() == vec![&tip],
            "only the last output of the chain is unspent, got {spendable:?}",
        );
        Ok(())
    }
}
