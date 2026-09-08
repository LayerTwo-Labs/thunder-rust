use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use byteorder::{BigEndian, ByteOrder};
use ed25519_dalek_bip32::{ChildIndex, DerivationPath, ExtendedSigningKey};
use fallible_iterator::FallibleIterator as _;
use futures::{Stream, StreamExt};
use heed::types::{Bytes, SerdeBincode, U8, Unit};
use sneed::{Env, EnvError, RwTxnError, UnitKey, db::error::Error as DbError};
use tokio_stream::{StreamMap, wrappers::WatchStream};

use crate::{
    types::{
        Accumulator, Address, AmountOverflowError, AmountUnderflowError,
        AuthorizedTransaction, GetValue, InPoint, OutPoint, OutPointKey,
        Output, OutputContent, PointedOutput, SpentOutput, Transaction,
        UtreexoError, UtreexoNodeHash, VERSION, Version,
        authorization::{Authorization, get_address},
        hash,
        wallet::Balance,
    },
    util::Watchable,
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("address {address} does not exist")]
    AddressDoesNotExist { address: crate::types::Address },
    #[error(transparent)]
    AmountOverflow(#[from] AmountOverflowError),
    #[error(transparent)]
    AmountUnderflow(#[from] AmountUnderflowError),
    #[error("authorization error")]
    Authorization(#[from] crate::types::error::Authorization),
    #[error("bip32 error")]
    Bip32(#[from] ed25519_dalek_bip32::Error),
    #[error(transparent)]
    Db(#[from] DbError),
    #[error("Database env error")]
    DbEnv(#[from] EnvError),
    #[error("Database write error")]
    DbWrite(#[from] RwTxnError),
    #[error("io error")]
    Io(#[from] std::io::Error),
    #[error("no index for address {address}")]
    NoIndex { address: Address },
    #[error(
        "wallet does not have a seed (set with RPC `set-seed-from-mnemonic`)"
    )]
    NoSeed,
    #[error("not enough funds")]
    NotEnoughFunds,
    #[error("utxo does not exist")]
    NoUtxo,
    #[error("failed to parse mnemonic seed phrase")]
    ParseMnemonic(#[source] bip39::ErrorKind),
    #[error("seed has already been set")]
    SeedAlreadyExists,
    #[error(transparent)]
    Utreexo(#[from] UtreexoError),
}

/// Marker type for Wallet Env
pub struct WalletEnv;

type DatabaseUnique<KC, DC> = sneed::DatabaseUnique<KC, DC, WalletEnv>;
type RoTxn<'a> = sneed::RoTxn<'a, heed::AnyTls, WalletEnv>;

/// The coins a wallet picked for a transaction.
pub struct SelectedCoins {
    pub total: bitcoin::Amount,
    pub coins: HashMap<OutPoint, Output>,
    /// The picked coins that no block carries yet. The accumulator holds no
    /// leaf for them, so a proof leaves them out.
    pub unconfirmed: HashSet<OutPoint>,
}

#[derive(Clone)]
pub struct Wallet {
    env: sneed::Env<heed::WithoutTls, WalletEnv>,
    // Seed is always [u8; 64], but due to serde not implementing serialize
    // for [T; 64], use heed's `Bytes`
    // TODO: Don't store the seed in plaintext.
    seed: DatabaseUnique<U8, Bytes>,
    /// Map each address to it's index
    address_to_index:
        DatabaseUnique<SerdeBincode<Address>, SerdeBincode<[u8; 4]>>,
    /// Map each address index to an address
    index_to_address:
        DatabaseUnique<SerdeBincode<[u8; 4]>, SerdeBincode<Address>>,
    utxos: DatabaseUnique<OutPointKey, SerdeBincode<Output>>,
    stxos: DatabaseUnique<OutPointKey, SerdeBincode<SpentOutput>>,
    /// Unconfirmed outputs that the wallet may spend. The node fills it from
    /// the mempool on every sync.
    unconfirmed_utxos: DatabaseUnique<OutPointKey, SerdeBincode<Output>>,
    /// Confirmed outputs that a mempool transaction already spends. Picking
    /// one again would make a double spend that the mempool refuses.
    mempool_spent_utxos: DatabaseUnique<OutPointKey, Unit>,
    _version: DatabaseUnique<UnitKey, SerdeBincode<Version>>,
}

impl Wallet {
    pub const NUM_DBS: u32 = 8;

    pub fn new(path: &Path) -> Result<Self, Error> {
        std::fs::create_dir_all(path)?;
        let env = {
            use heed::EnvFlags;
            let mut env_open_options =
                heed::EnvOpenOptions::new().read_txn_without_tls();
            env_open_options
                .map_size(10 * 1024 * 1024) // 10MB
                .max_dbs(Self::NUM_DBS);
            // Apply LMDB "fast" flags consistent with our benchmark setup:
            // - WRITE_MAP lets us write directly into the memory map instead of
            //   copying into LMDB's page buffer, reducing syscall overhead for
            //   write-heavy workloads.
            // - MAP_ASYNC hands dirty-page flushing to the kernel so commits do
            //   not block waiting for msync, keeping latencies tight.
            // - NO_SYNC and NO_META_SYNC skip fsync calls for data and
            //   metadata; this trades durability for throughput, which is
            //   acceptable here because the state can be reconstructed from the
            //   canonical chain if a crash occurs.
            // - NO_READ_AHEAD disables kernel readahead that would otherwise
            //   touch cold pages we immediately overwrite, improving random
            //   access behaviour on SSDs used in testing.
            // - NO_TLS stops LMDB from relying on thread-local storage for
            //   reader slots so transactions can be moved across Tokio tasks.
            let fast_flags = EnvFlags::WRITE_MAP
                | EnvFlags::MAP_ASYNC
                | EnvFlags::NO_SYNC
                | EnvFlags::NO_META_SYNC
                | EnvFlags::NO_READ_AHEAD;
            unsafe { env_open_options.flags(fast_flags) };
            unsafe { Env::open(&env_open_options, path) }
                .map_err(EnvError::from)?
        };
        let mut rwtxn = env.write_txn().map_err(EnvError::from)?;
        let seed_db = DatabaseUnique::create(&env, &mut rwtxn, "seed")
            .map_err(EnvError::from)?;
        let address_to_index =
            DatabaseUnique::create(&env, &mut rwtxn, "address_to_index")
                .map_err(EnvError::from)?;
        let index_to_address =
            DatabaseUnique::create(&env, &mut rwtxn, "index_to_address")
                .map_err(EnvError::from)?;
        let utxos = DatabaseUnique::create(&env, &mut rwtxn, "utxos")
            .map_err(EnvError::from)?;
        let stxos = DatabaseUnique::create(&env, &mut rwtxn, "stxos")
            .map_err(EnvError::from)?;
        let unconfirmed_utxos =
            DatabaseUnique::create(&env, &mut rwtxn, "unconfirmed_utxos")
                .map_err(EnvError::from)?;
        let mempool_spent_utxos =
            DatabaseUnique::create(&env, &mut rwtxn, "mempool_spent_utxos")
                .map_err(EnvError::from)?;
        let version = DatabaseUnique::create(&env, &mut rwtxn, "version")
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
            env,
            seed: seed_db,
            address_to_index,
            index_to_address,
            utxos,
            stxos,
            unconfirmed_utxos,
            mempool_spent_utxos,
            _version: version,
        })
    }

    /// Overwrite the seed, or set it if it does not already exist.
    pub fn overwrite_seed(&self, seed: &[u8; 64]) -> Result<(), Error> {
        let mut rwtxn = self.env.write_txn().map_err(EnvError::from)?;
        self.seed.put(&mut rwtxn, &0, seed).map_err(DbError::from)?;
        self.address_to_index
            .clear(&mut rwtxn)
            .map_err(DbError::from)?;
        self.index_to_address
            .clear(&mut rwtxn)
            .map_err(DbError::from)?;
        self.utxos.clear(&mut rwtxn).map_err(DbError::from)?;
        self.stxos.clear(&mut rwtxn).map_err(DbError::from)?;
        self.unconfirmed_utxos
            .clear(&mut rwtxn)
            .map_err(DbError::from)?;
        self.mempool_spent_utxos
            .clear(&mut rwtxn)
            .map_err(DbError::from)?;
        rwtxn.commit().map_err(RwTxnError::from)?;
        Ok(())
    }

    pub fn has_seed(&self) -> Result<bool, Error> {
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        Ok(self
            .seed
            .try_get(&rotxn, &0)
            .map_err(DbError::from)?
            .is_some())
    }

    /// Set the seed, if it does not already exist
    pub fn set_seed(&self, seed: &[u8; 64]) -> Result<(), Error> {
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        match self.seed.try_get(&rotxn, &0).map_err(DbError::from)? {
            Some(current_seed) => {
                if current_seed == seed {
                    Ok(())
                } else {
                    Err(Error::SeedAlreadyExists)
                }
            }
            None => {
                drop(rotxn);
                self.overwrite_seed(seed)
            }
        }
    }

    /// Set the seed from a mnemonic seed phrase,
    /// if the seed does not already exist
    pub fn set_seed_from_mnemonic(&self, mnemonic: &str) -> Result<(), Error> {
        let mnemonic =
            bip39::Mnemonic::from_phrase(mnemonic, bip39::Language::English)
                .map_err(Error::ParseMnemonic)?;
        let seed = bip39::Seed::new(&mnemonic, "");
        let seed_bytes: [u8; 64] = seed.as_bytes().try_into().unwrap();
        self.set_seed(&seed_bytes)
    }

    pub fn create_withdrawal(
        &self,
        accumulator: &Accumulator,
        spend_zero_conf_change: bool,
        main_address: bitcoin::Address<bitcoin::address::NetworkUnchecked>,
        value: bitcoin::Amount,
        main_fee: bitcoin::Amount,
        fee: bitcoin::Amount,
    ) -> Result<Transaction, Error> {
        tracing::trace!(
            accumulator = %accumulator.0,
            fee = %fee.display_dynamic(),
            ?main_address,
            main_fee = %main_fee.display_dynamic(),
            value = %value.display_dynamic(),
            "Creating withdrawal"
        );
        let selected = self.select_coins(
            value
                .checked_add(fee)
                .ok_or(AmountOverflowError)?
                .checked_add(main_fee)
                .ok_or(AmountOverflowError)?,
            spend_zero_conf_change,
        )?;
        let change = selected.total - value - fee - main_fee;

        let inputs: Vec<_> = selected
            .coins
            .into_iter()
            .map(|(outpoint, output)| {
                let utxo_hash = hash(&PointedOutput { outpoint, output });
                (outpoint, utxo_hash)
            })
            .collect();
        let input_utxo_hashes: Vec<UtreexoNodeHash> = inputs
            .iter()
            .filter(|(outpoint, _)| !selected.unconfirmed.contains(outpoint))
            .map(|(_, hash)| hash.into())
            .collect();
        let proof = accumulator.prove(&input_utxo_hashes)?;
        let outputs = vec![
            Output {
                address: self.get_new_address()?,
                content: OutputContent::Withdrawal {
                    value,
                    main_fee,
                    main_address,
                },
            },
            Output {
                address: self.get_new_address()?,
                content: OutputContent::Value(change),
            },
        ];
        Ok(Transaction {
            inputs,
            proof,
            outputs,
        })
    }

    pub fn create_transaction(
        &self,
        accumulator: &Accumulator,
        spend_zero_conf_change: bool,
        address: Address,
        value: bitcoin::Amount,
        fee: bitcoin::Amount,
    ) -> Result<Transaction, Error> {
        let selected = self.select_coins(
            value.checked_add(fee).ok_or(AmountOverflowError)?,
            spend_zero_conf_change,
        )?;
        let change = selected.total - value - fee;
        let inputs: Vec<_> = selected
            .coins
            .into_iter()
            .map(|(outpoint, output)| {
                let utxo_hash = hash(&PointedOutput { outpoint, output });
                (outpoint, utxo_hash)
            })
            .collect();
        let input_utxo_hashes: Vec<UtreexoNodeHash> = inputs
            .iter()
            .filter(|(outpoint, _)| !selected.unconfirmed.contains(outpoint))
            .map(|(_, hash)| hash.into())
            .collect();
        let proof = accumulator.prove(&input_utxo_hashes)?;
        let outputs = vec![
            Output {
                address,
                content: OutputContent::Value(value),
            },
            Output {
                address: self.get_new_address()?,
                content: OutputContent::Value(change),
            },
        ];
        Ok(Transaction {
            inputs,
            proof,
            outputs,
        })
    }

    /// Pick coins worth at least `value`. A confirmed coin comes first, so a
    /// chain of unconfirmed transactions only forms when the confirmed coins
    /// fall short. Bitcoin Core orders its coin selection the same way.
    ///
    /// `spend_zero_conf_change` decides whether the wallet's own unconfirmed
    /// change joins the pick at all.
    pub fn select_coins(
        &self,
        value: bitcoin::Amount,
        spend_zero_conf_change: bool,
    ) -> Result<SelectedCoins, Error> {
        use rayon::prelude::ParallelSliceMut;
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        let mempool_spent: HashSet<OutPointKey> = self
            .mempool_spent_utxos
            .iter_keys(&rotxn)
            .map_err(DbError::from)?
            .collect()
            .map_err(DbError::from)?;
        let mut utxos: Vec<_> = self
            .utxos
            .iter(&rotxn)
            .map_err(DbError::from)?
            .collect::<Vec<_>>()
            .map_err(DbError::from)?;
        utxos.retain(|(outpoint_key, _)| !mempool_spent.contains(outpoint_key));
        utxos.par_sort_unstable_by_key(|(_, output)| output.get_value());
        let mut unconfirmed_utxos: Vec<_> = if spend_zero_conf_change {
            self.unconfirmed_utxos
                .iter(&rotxn)
                .map_err(DbError::from)?
                .collect()
                .map_err(DbError::from)?
        } else {
            Vec::new()
        };
        unconfirmed_utxos
            .par_sort_unstable_by_key(|(_, output)| output.get_value());
        let confirmed_count = utxos.len();

        let mut selected = HashMap::new();
        let mut unconfirmed = HashSet::new();
        let mut total = bitcoin::Amount::ZERO;
        for (index, (outpoint_key, output)) in
            utxos.iter().chain(&unconfirmed_utxos).enumerate()
        {
            if output.content.is_withdrawal() {
                continue;
            }
            if total > value {
                break;
            }
            total = total
                .checked_add(output.get_value())
                .ok_or(AmountOverflowError)?;
            let outpoint: OutPoint = outpoint_key.into();
            selected.insert(outpoint, output.clone());
            if index >= confirmed_count {
                unconfirmed.insert(outpoint);
            }
        }
        if total < value {
            return Err(Error::NotEnoughFunds);
        }
        Ok(SelectedCoins {
            total,
            coins: selected,
            unconfirmed,
        })
    }

    pub fn delete_utxos(&self, outpoints: &[OutPoint]) -> Result<(), Error> {
        let mut txn = self.env.write_txn().map_err(EnvError::from)?;
        for outpoint in outpoints {
            let key = OutPointKey::from(outpoint);
            self.utxos.delete(&mut txn, &key).map_err(DbError::from)?;
        }
        txn.commit().map_err(RwTxnError::from)?;
        Ok(())
    }

    pub fn spend_utxos(
        &self,
        spent: &[(OutPoint, InPoint)],
    ) -> Result<(), Error> {
        let mut txn = self.env.write_txn().map_err(EnvError::from)?;
        for (outpoint, inpoint) in spent {
            let key = OutPointKey::from(outpoint);
            let output =
                self.utxos.try_get(&txn, &key).map_err(DbError::from)?;
            if let Some(output) = output {
                self.utxos.delete(&mut txn, &key).map_err(DbError::from)?;
                let spent_output = SpentOutput {
                    output,
                    inpoint: *inpoint,
                };
                self.stxos
                    .put(&mut txn, &key, &spent_output)
                    .map_err(DbError::from)?;
            }
        }
        txn.commit().map_err(RwTxnError::from)?;
        Ok(())
    }

    /// Make the confirmed table say what the chain says. `utxos` is every
    /// output the chain holds for this wallet, and `spent` is what a block
    /// spent since the last call.
    ///
    /// A block that disconnects takes an output off the chain without a spend,
    /// and it takes the utreexo leaf with it. A row that stays behind reads as
    /// confirmed, so `create_transaction` makes it a proof target and every
    /// send fails. Delete such a row here, where the chain's answer is known.
    pub fn sync_confirmed(
        &self,
        utxos: &HashMap<OutPoint, Output>,
        spent: &[(OutPoint, InPoint)],
    ) -> Result<(), Error> {
        let mut rwtxn = self.env.write_txn().map_err(EnvError::from)?;
        for (outpoint, output) in utxos {
            self.utxos
                .put(&mut rwtxn, &OutPointKey::from(outpoint), output)
                .map_err(DbError::from)?;
        }
        for (outpoint, inpoint) in spent {
            let key = OutPointKey::from(outpoint);
            let Some(output) =
                self.utxos.try_get(&rwtxn, &key).map_err(DbError::from)?
            else {
                continue;
            };
            self.utxos.delete(&mut rwtxn, &key).map_err(DbError::from)?;
            let spent_output = SpentOutput {
                output,
                inpoint: *inpoint,
            };
            self.stxos
                .put(&mut rwtxn, &key, &spent_output)
                .map_err(DbError::from)?;
        }
        let stale: Vec<OutPointKey> = self
            .utxos
            .iter_keys(&rwtxn)
            .map_err(DbError::from)?
            .filter(|key| Ok(!utxos.contains_key(&key.into())))
            .collect()
            .map_err(DbError::from)?;
        for key in &stale {
            self.utxos.delete(&mut rwtxn, key).map_err(DbError::from)?;
        }
        rwtxn.commit().map_err(RwTxnError::from)?;
        Ok(())
    }

    pub fn put_utxos(
        &self,
        utxos: &HashMap<OutPoint, Output>,
    ) -> Result<(), Error> {
        let mut txn = self.env.write_txn().map_err(EnvError::from)?;
        for (outpoint, output) in utxos {
            let key = OutPointKey::from(outpoint);
            self.utxos
                .put(&mut txn, &key, output)
                .map_err(DbError::from)?;
        }
        txn.commit().map_err(RwTxnError::from)?;
        Ok(())
    }

    /// The value the wallet holds. A confirmed output that a mempool
    /// transaction already spends counts for nothing, because the money left.
    ///
    /// An unconfirmed output always counts toward `total` and `unconfirmed`,
    /// the way Bitcoin Core always reports such value. It counts toward
    /// `available` only when `spend_zero_conf_change` lets the wallet take it.
    pub fn get_balance(
        &self,
        spend_zero_conf_change: bool,
    ) -> Result<Balance, Error> {
        let mut balance = Balance::default();
        let txn = self.env.read_txn().map_err(EnvError::from)?;
        let () = self
            .utxos
            .iter(&txn)
            .map_err(DbError::from)?
            .map_err(|err| DbError::from(err).into())
            .for_each(|(key, utxo)| {
                if self
                    .mempool_spent_utxos
                    .try_get(&txn, &key)
                    .map_err(DbError::from)?
                    .is_some()
                {
                    return Ok(());
                }
                let value = utxo.get_value();
                balance.total = balance
                    .total
                    .checked_add(value)
                    .ok_or(AmountOverflowError)?;
                if !utxo.content.is_withdrawal() {
                    balance.available = balance
                        .available
                        .checked_add(value)
                        .ok_or(AmountOverflowError)?;
                }
                Ok::<_, Error>(())
            })?;
        let () = self
            .unconfirmed_utxos
            .iter(&txn)
            .map_err(DbError::from)?
            .map_err(|err| DbError::from(err).into())
            .for_each(|(_, utxo)| {
                let value = utxo.get_value();
                balance.total = balance
                    .total
                    .checked_add(value)
                    .ok_or(AmountOverflowError)?;
                balance.unconfirmed = balance
                    .unconfirmed
                    .checked_add(value)
                    .ok_or(AmountOverflowError)?;
                if spend_zero_conf_change && !utxo.content.is_withdrawal() {
                    balance.available = balance
                        .available
                        .checked_add(value)
                        .ok_or(AmountOverflowError)?;
                }
                Ok::<_, Error>(())
            })?;
        Ok(balance)
    }

    /// Replace what the wallet knows about the mempool: the unconfirmed
    /// outputs it may spend, and the confirmed outputs a mempool transaction
    /// already spends. The node states both on every sync, so a wholesale
    /// replacement leaves no stale row behind when a transaction drops out.
    pub fn set_mempool_view(
        &self,
        unconfirmed: &HashMap<OutPoint, Output>,
        spent: &HashSet<OutPoint>,
    ) -> Result<(), Error> {
        let mut rwtxn = self.env.write_txn().map_err(EnvError::from)?;
        self.unconfirmed_utxos
            .clear(&mut rwtxn)
            .map_err(DbError::from)?;
        self.mempool_spent_utxos
            .clear(&mut rwtxn)
            .map_err(DbError::from)?;
        for (outpoint, output) in unconfirmed {
            self.unconfirmed_utxos
                .put(&mut rwtxn, &OutPointKey::from(outpoint), output)
                .map_err(DbError::from)?;
        }
        for outpoint in spent {
            self.mempool_spent_utxos
                .put(&mut rwtxn, &OutPointKey::from(outpoint), &())
                .map_err(DbError::from)?;
        }
        rwtxn.commit().map_err(RwTxnError::from)?;
        Ok(())
    }

    pub fn get_mempool_spent_utxos(&self) -> Result<HashSet<OutPoint>, Error> {
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        let outpoints: HashSet<OutPoint> = self
            .mempool_spent_utxos
            .iter_keys(&rotxn)
            .map_err(DbError::from)?
            .map(|key| Ok((&key).into()))
            .collect()
            .map_err(DbError::from)?;
        Ok(outpoints)
    }

    pub fn get_unconfirmed_utxos(
        &self,
    ) -> Result<HashMap<OutPoint, Output>, Error> {
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        let utxos: HashMap<OutPoint, Output> = self
            .unconfirmed_utxos
            .iter(&rotxn)
            .map_err(DbError::from)?
            .map(|(key, output)| Ok((key.into(), output)))
            .collect()
            .map_err(DbError::from)?;
        Ok(utxos)
    }

    pub fn get_utxos(&self) -> Result<HashMap<OutPoint, Output>, Error> {
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        let utxos: HashMap<OutPoint, Output> = self
            .utxos
            .iter(&rotxn)
            .map_err(DbError::from)?
            .map(|(key, output)| Ok((key.into(), output)))
            .collect()
            .map_err(DbError::from)?;
        Ok(utxos)
    }

    pub fn get_addresses(&self) -> Result<HashSet<Address>, Error> {
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        let addresses: HashSet<_> = self
            .index_to_address
            .iter(&rotxn)
            .map_err(DbError::from)?
            .map(|(_, address)| Ok(address))
            .collect()
            .map_err(DbError::from)?;
        Ok(addresses)
    }

    pub fn authorize(
        &self,
        transaction: Transaction,
    ) -> Result<AuthorizedTransaction, Error> {
        let txn = self.env.read_txn().map_err(EnvError::from)?;
        let mut authorizations = Vec::with_capacity(transaction.inputs.len());
        for (outpoint, _) in &transaction.inputs {
            let key = OutPointKey::from(outpoint);
            let spent_utxo =
                match self.utxos.try_get(&txn, &key).map_err(DbError::from)? {
                    Some(spent_utxo) => spent_utxo,
                    None => self
                        .unconfirmed_utxos
                        .try_get(&txn, &key)
                        .map_err(DbError::from)?
                        .ok_or(Error::NoUtxo)?,
                };
            let index = self
                .address_to_index
                .try_get(&txn, &spent_utxo.address)
                .map_err(DbError::from)?
                .ok_or(Error::NoIndex {
                    address: spent_utxo.address,
                })?;
            let index = BigEndian::read_u32(&index);
            let signing_key = self.get_signing_key(&txn, index)?;
            let signature =
                crate::types::authorization::sign(&signing_key, &transaction)?;
            authorizations.push(Authorization {
                verifying_key: signing_key.verifying_key(),
                signature,
            });
        }
        Ok(AuthorizedTransaction {
            authorizations,
            transaction,
        })
    }

    pub fn get_new_address(&self) -> Result<Address, Error> {
        let mut txn = self.env.write_txn().map_err(EnvError::from)?;
        let (last_index, _) = self
            .index_to_address
            .last(&txn)
            .map_err(DbError::from)?
            .unwrap_or(([0; 4], [0; 20].into()));
        let last_index = BigEndian::read_u32(&last_index);
        let index = last_index + 1;
        let signing_key = self.get_signing_key(&txn, index)?;
        let address = get_address(&signing_key.verifying_key());
        let index = index.to_be_bytes();
        self.index_to_address
            .put(&mut txn, &index, &address)
            .map_err(DbError::from)?;
        self.address_to_index
            .put(&mut txn, &address, &index)
            .map_err(DbError::from)?;
        txn.commit().map_err(RwTxnError::from)?;
        Ok(address)
    }

    /// Gets the latest generated address.
    pub fn try_get_last_address(&self) -> Result<Option<Address>, Error> {
        let txn = self.env.read_txn().map_err(EnvError::from)?;
        let last = self.index_to_address.last(&txn).map_err(DbError::from)?;
        Ok(last.map(|(_, address)| address))
    }

    /// Gets the latest generated address, or generates a new one if no
    /// addresses have already been generated.
    pub fn get_or_generate_last_address(&self) -> Result<Address, Error> {
        if let Some(address) = self.try_get_last_address()? {
            Ok(address)
        } else {
            self.get_new_address()
        }
    }

    pub fn get_num_addresses(&self) -> Result<u32, Error> {
        let txn = self.env.read_txn().map_err(EnvError::from)?;
        let (last_index, _) = self
            .index_to_address
            .last(&txn)
            .map_err(DbError::from)?
            .unwrap_or(([0; 4], [0; 20].into()));
        let last_index = BigEndian::read_u32(&last_index);
        Ok(last_index)
    }

    fn get_signing_key(
        &self,
        rotxn: &RoTxn,
        index: u32,
    ) -> Result<ed25519_dalek::SigningKey, Error> {
        let seed = self
            .seed
            .try_get(rotxn, &0)
            .map_err(DbError::from)?
            .ok_or(Error::NoSeed)?;
        let xpriv = ExtendedSigningKey::from_seed(seed)?;
        let derivation_path = DerivationPath::new([
            ChildIndex::Hardened(1),
            ChildIndex::Hardened(0),
            ChildIndex::Hardened(0),
            ChildIndex::Hardened(index),
        ]);
        let xsigning_key = xpriv.derive(&derivation_path)?;
        Ok(xsigning_key.signing_key)
    }
}

impl Watchable<()> for Wallet {
    type WatchStream = std::pin::Pin<Box<dyn Stream<Item = ()> + Send>>;

    /// Get a signal that notifies whenever the wallet changes
    fn watch(&self) -> Self::WatchStream {
        let Self {
            env: _,
            seed,
            address_to_index,
            index_to_address,
            utxos,
            stxos,
            unconfirmed_utxos,
            mempool_spent_utxos,
            _version: _,
        } = self;
        let watchables = [
            seed.watch().clone(),
            address_to_index.watch().clone(),
            index_to_address.watch().clone(),
            utxos.watch().clone(),
            stxos.watch().clone(),
            unconfirmed_utxos.watch().clone(),
            mempool_spent_utxos.watch().clone(),
        ];
        let streams = StreamMap::from_iter(
            watchables.into_iter().map(WatchStream::new).enumerate(),
        );
        let streams_len = streams.len();
        Box::pin(streams.ready_chunks(streams_len).map(|signals| {
            assert_ne!(signals.len(), 0);
            #[allow(clippy::unused_unit)]
            ()
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A block that disconnects takes an output off the chain without a
    /// spend, and it takes the utreexo leaf with it. The confirmed row must go
    /// too, so the output moves to the unconfirmed side, where the proof code
    /// leaves it out.
    #[test]
    fn a_disconnected_output_leaves_the_confirmed_table() -> anyhow::Result<()>
    {
        use crate::types::OutputContent;

        let temp_dir = temp_dir::TempDir::with_prefix(format!(
            "wallet-disconnect-{}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos(),
            std::process::id()
        ))?;
        let wallet = Wallet::new(temp_dir.path())?;
        wallet.set_seed(&[0x99; 64])?;
        let address = wallet.get_new_address()?;
        let outpoint = OutPoint::Regular {
            txid: crate::types::hash(&[0u8; 32]).into(),
            vout: 0,
        };
        let output = Output {
            address,
            content: OutputContent::Value(bitcoin::Amount::from_sat(10_000)),
        };
        let held = HashMap::from([(outpoint, output.clone())]);

        wallet.sync_confirmed(&held, &[])?;
        anyhow::ensure!(wallet.get_utxos()?.len() == 1);

        // The chain drops the output, and no block spends it.
        wallet.sync_confirmed(&HashMap::new(), &[])?;
        anyhow::ensure!(
            wallet.get_utxos()?.is_empty(),
            "the confirmed row must go when the chain drops the output",
        );

        // The transaction that made it sits in the mempool again.
        wallet.set_mempool_view(&held, &HashSet::new())?;
        let balance = wallet.get_balance(true)?;
        anyhow::ensure!(
            balance.unconfirmed == bitcoin::Amount::from_sat(10_000),
            "the output reads as unconfirmed, got {balance:?}",
        );
        anyhow::ensure!(
            balance.total == bitcoin::Amount::from_sat(10_000),
            "the wallet counts the output one time, got {balance:?}",
        );
        anyhow::ensure!(
            balance.available == bitcoin::Amount::from_sat(10_000),
            "the wallet may take it, got {balance:?}",
        );

        // With the option off the value still shows, and the wallet may not
        // take it. Bitcoin Core reports such value the same way.
        let balance = wallet.get_balance(false)?;
        anyhow::ensure!(
            balance.unconfirmed == bitcoin::Amount::from_sat(10_000)
                && balance.total == bitcoin::Amount::from_sat(10_000),
            "the value stays visible, got {balance:?}",
        );
        anyhow::ensure!(
            balance.available == bitcoin::Amount::ZERO,
            "the wallet may not take it, got {balance:?}",
        );
        anyhow::ensure!(
            wallet
                .select_coins(bitcoin::Amount::from_sat(1_000), false)
                .is_err(),
            "coin selection must refuse the unconfirmed coin",
        );

        // With the option on the wallet takes it, and marks it unconfirmed so
        // the proof leaves it out.
        let selected =
            wallet.select_coins(bitcoin::Amount::from_sat(1_000), true)?;
        anyhow::ensure!(
            selected.unconfirmed.contains(&outpoint),
            "the wallet takes the coin and marks it unconfirmed",
        );
        Ok(())
    }

    #[test]
    fn test_get_or_generate_last_address() -> anyhow::Result<()> {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let test_dir =
            std::env::temp_dir().join(format!("thunder_test_wallet_{nanos}"));

        // Ensure clean state
        if test_dir.exists() {
            let _unused = std::fs::remove_dir_all(&test_dir);
        }

        let wallet = Wallet::new(&test_dir)?;

        // Seed must be set before we can generate addresses
        assert!(!wallet.has_seed()?);
        let seed = [1u8; 64];
        wallet.set_seed(&seed)?;
        assert!(wallet.has_seed()?);

        // Get last address when none have been generated
        let last = wallet.try_get_last_address()?;
        assert!(last.is_none());

        // The first call should generate the first address.
        let addr1 = wallet.get_or_generate_last_address()?;

        let last = wallet.try_get_last_address()?;
        assert_eq!(last, Some(addr1));

        let addr2 = wallet.get_or_generate_last_address()?;
        assert_eq!(addr1, addr2);

        let addr3 = wallet.get_new_address()?;
        assert_ne!(addr1, addr3);

        let last = wallet.try_get_last_address()?;
        assert_eq!(last, Some(addr3));

        let addr4 = wallet.get_or_generate_last_address()?;
        assert_eq!(addr3, addr4);

        // Clean up
        let _unused = std::fs::remove_dir_all(&test_dir);
        Ok(())
    }
}
