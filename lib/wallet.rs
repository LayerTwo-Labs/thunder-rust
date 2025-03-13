use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use bitcoin::Amount;
use byteorder::{BigEndian, ByteOrder};
use ed25519_dalek_bip32::{ChildIndex, DerivationPath, ExtendedSigningKey};
use fallible_iterator::FallibleIterator as _;
use futures::{Stream, StreamExt};
use heed::types::{Bytes, SerdeBincode, U8};
use rustreexo::accumulator::node_hash::BitcoinNodeHash;
use serde::{Deserialize, Serialize};
use sneed::{
    DatabaseUnique, Env, EnvError, RoTxn, RwTxnError, UnitKey,
    db::error::Error as DbError,
};
use tokio_stream::{StreamMap, wrappers::WatchStream};

pub use crate::{
    authorization::{Authorization, get_address},
    types::{
        Address, AuthorizedTransaction, GetValue, InPoint, OutPoint, Output,
        OutputContent, SpentOutput, Transaction,
    },
};
use crate::{
    types::{
        Accumulator, AmountOverflowError, AmountUnderflowError, PointedOutput,
        UtreexoError, VERSION, Version, hash,
    },
    util::Watchable,
};

#[derive(Clone, Debug, Default, Deserialize, Serialize, utoipa::ToSchema)]
pub struct Balance {
    #[serde(rename = "total_sats", with = "bitcoin::amount::serde::as_sat")]
    #[schema(value_type = u64)]
    pub total: Amount,
    #[serde(
        rename = "available_sats",
        with = "bitcoin::amount::serde::as_sat"
    )]
    #[schema(value_type = u64)]
    pub available: Amount,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("address {address} does not exist")]
    AddressDoesNotExist { address: crate::types::Address },
    #[error(transparent)]
    AmountOverflow(#[from] AmountOverflowError),
    #[error(transparent)]
    AmountUnderflow(#[from] AmountUnderflowError),
    #[error("authorization error")]
    Authorization(#[from] crate::authorization::Error),
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

#[derive(Clone)]
pub struct Wallet {
    env: sneed::Env,
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
    utxos: DatabaseUnique<SerdeBincode<OutPoint>, SerdeBincode<Output>>,
    stxos: DatabaseUnique<SerdeBincode<OutPoint>, SerdeBincode<SpentOutput>>,
    _version: DatabaseUnique<UnitKey, SerdeBincode<Version>>,
}

impl Wallet {
    pub const NUM_DBS: u32 = 6;

    pub fn new(path: &Path) -> Result<Self, Error> {
        std::fs::create_dir_all(path)?;
        let env = {
            let mut env_open_options = heed::EnvOpenOptions::new();
            env_open_options
                .map_size(10 * 1024 * 1024) // 10MB
                .max_dbs(Self::NUM_DBS);
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
        let (total, coins) = self.select_coins(
            value
                .checked_add(fee)
                .ok_or(AmountOverflowError)?
                .checked_add(main_fee)
                .ok_or(AmountOverflowError)?,
        )?;
        let change = total - value - fee;

        let inputs: Vec<_> = coins
            .into_iter()
            .map(|(outpoint, output)| {
                let utxo_hash = hash(&PointedOutput { outpoint, output });
                (outpoint, utxo_hash)
            })
            .collect();
        let input_utxo_hashes: Vec<BitcoinNodeHash> =
            inputs.iter().map(|(_, hash)| hash.into()).collect();
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
        address: Address,
        value: bitcoin::Amount,
        fee: bitcoin::Amount,
    ) -> Result<Transaction, Error> {
        let (total, coins) = self
            .select_coins(value.checked_add(fee).ok_or(AmountOverflowError)?)?;
        let change = total - value - fee;
        let inputs: Vec<_> = coins
            .into_iter()
            .map(|(outpoint, output)| {
                let utxo_hash = hash(&PointedOutput { outpoint, output });
                (outpoint, utxo_hash)
            })
            .collect();
        let input_utxo_hashes: Vec<BitcoinNodeHash> =
            inputs.iter().map(|(_, hash)| hash.into()).collect();
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

    pub fn select_coins(
        &self,
        value: bitcoin::Amount,
    ) -> Result<(bitcoin::Amount, HashMap<OutPoint, Output>), Error> {
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        let mut utxos: Vec<_> = self
            .utxos
            .iter(&rotxn)
            .map_err(DbError::from)?
            .collect()
            .map_err(DbError::from)?;
        utxos.sort_unstable_by_key(|(_, output)| output.get_value());

        let mut selected = HashMap::new();
        let mut total = bitcoin::Amount::ZERO;
        for (outpoint, output) in &utxos {
            if output.content.is_withdrawal() {
                continue;
            }
            if total > value {
                break;
            }
            total = total
                .checked_add(output.get_value())
                .ok_or(AmountOverflowError)?;
            selected.insert(*outpoint, output.clone());
        }
        if total < value {
            return Err(Error::NotEnoughFunds);
        }
        Ok((total, selected))
    }

    pub fn delete_utxos(&self, outpoints: &[OutPoint]) -> Result<(), Error> {
        let mut txn = self.env.write_txn().map_err(EnvError::from)?;
        for outpoint in outpoints {
            self.utxos
                .delete(&mut txn, outpoint)
                .map_err(DbError::from)?;
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
            let output =
                self.utxos.try_get(&txn, outpoint).map_err(DbError::from)?;
            if let Some(output) = output {
                self.utxos
                    .delete(&mut txn, outpoint)
                    .map_err(DbError::from)?;
                let spent_output = SpentOutput {
                    output,
                    inpoint: *inpoint,
                };
                self.stxos
                    .put(&mut txn, outpoint, &spent_output)
                    .map_err(DbError::from)?;
            }
        }
        txn.commit().map_err(RwTxnError::from)?;
        Ok(())
    }

    pub fn put_utxos(
        &self,
        utxos: &HashMap<OutPoint, Output>,
    ) -> Result<(), Error> {
        let mut txn = self.env.write_txn().map_err(EnvError::from)?;
        for (outpoint, output) in utxos {
            self.utxos
                .put(&mut txn, outpoint, output)
                .map_err(DbError::from)?;
        }
        txn.commit().map_err(RwTxnError::from)?;
        Ok(())
    }

    pub fn get_balance(&self) -> Result<Balance, Error> {
        let mut balance = Balance::default();
        let txn = self.env.read_txn().map_err(EnvError::from)?;
        let () = self
            .utxos
            .iter(&txn)
            .map_err(DbError::from)?
            .map_err(|err| DbError::from(err).into())
            .for_each(|(_, utxo)| {
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
        Ok(balance)
    }

    pub fn get_utxos(&self) -> Result<HashMap<OutPoint, Output>, Error> {
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        let utxos: HashMap<_, _> = self
            .utxos
            .iter(&rotxn)
            .map_err(DbError::from)?
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
        let mut authorizations = vec![];
        for (outpoint, _) in &transaction.inputs {
            let spent_utxo = self
                .utxos
                .try_get(&txn, outpoint)
                .map_err(DbError::from)?
                .ok_or(Error::NoUtxo)?;
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
                crate::authorization::sign(&signing_key, &transaction)?;
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
    type WatchStream = impl Stream<Item = ()>;

    /// Get a signal that notifies whenever the wallet changes
    fn watch(&self) -> Self::WatchStream {
        let Self {
            env: _,
            seed,
            address_to_index,
            index_to_address,
            utxos,
            stxos,
            _version: _,
        } = self;
        let watchables = [
            seed.watch().clone(),
            address_to_index.watch().clone(),
            index_to_address.watch().clone(),
            utxos.watch().clone(),
            stxos.watch().clone(),
        ];
        let streams = StreamMap::from_iter(
            watchables.into_iter().map(WatchStream::new).enumerate(),
        );
        let streams_len = streams.len();
        streams.ready_chunks(streams_len).map(|signals| {
            assert_ne!(signals.len(), 0);
            #[allow(clippy::unused_unit)]
            ()
        })
    }
}
