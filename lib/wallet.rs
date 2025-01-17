use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use bitcoin::Amount;
use byteorder::{BigEndian, ByteOrder};
use ed25519_dalek_bip32::{ChildIndex, DerivationPath, ExtendedSigningKey};
use fallible_iterator::{FallibleIterator as _, IteratorExt as _};
use futures::{Stream, StreamExt};
use heed::{
    types::{Bytes, SerdeBincode, U8},
    RoTxn,
};
use rustreexo::accumulator::node_hash::NodeHash;
use serde::{Deserialize, Serialize};
use tokio_stream::{wrappers::WatchStream, StreamMap};

pub use crate::{
    authorization::{get_address, Authorization},
    types::{
        Address, AuthorizedTransaction, GetValue, InPoint, OutPoint, Output,
        OutputContent, SpentOutput, Transaction,
    },
};
use crate::{
    types::{
        hash, Accumulator, AmountOverflowError, AmountUnderflowError,
        PointedOutput,
    },
    util::{EnvExt, Watchable, WatchableDb},
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
    #[error("heed error")]
    Heed(#[from] heed::Error),
    #[error("io error")]
    Io(#[from] std::io::Error),
    #[error("no index for address {address}")]
    NoIndex { address: Address },
    #[error("wallet doesn't have a seed")]
    NoSeed,
    #[error("not enough funds")]
    NotEnoughFunds,
    #[error("utxo doesn't exist")]
    NoUtxo,
    #[error("failed to parse mnemonic seed phrase")]
    ParseMnemonic(#[source] bip39::ErrorKind),
    #[error("seed has already been set")]
    SeedAlreadyExists,
    #[error("utreexo error: {0}")]
    Utreexo(String),
}

#[derive(Clone)]
pub struct Wallet {
    env: heed::Env,
    // Seed is always [u8; 64], but due to serde not implementing serialize
    // for [T; 64], use heed's `Bytes`
    // TODO: Don't store the seed in plaintext.
    seed: WatchableDb<U8, Bytes>,
    /// Map each address to it's index
    address_to_index: WatchableDb<SerdeBincode<Address>, SerdeBincode<[u8; 4]>>,
    /// Map each address index to an address
    index_to_address: WatchableDb<SerdeBincode<[u8; 4]>, SerdeBincode<Address>>,
    utxos: WatchableDb<SerdeBincode<OutPoint>, SerdeBincode<Output>>,
    stxos: WatchableDb<SerdeBincode<OutPoint>, SerdeBincode<SpentOutput>>,
}

impl Wallet {
    pub const NUM_DBS: u32 = 5;

    pub fn new(path: &Path) -> Result<Self, Error> {
        std::fs::create_dir_all(path)?;
        let env = unsafe {
            heed::EnvOpenOptions::new()
                .map_size(10 * 1024 * 1024) // 10MB
                .max_dbs(Self::NUM_DBS)
                .open(path)?
        };
        let mut rwtxn = env.write_txn()?;
        let seed_db = env.create_watchable_db(&mut rwtxn, "seed")?;
        let address_to_index =
            env.create_watchable_db(&mut rwtxn, "address_to_index")?;
        let index_to_address =
            env.create_watchable_db(&mut rwtxn, "index_to_address")?;
        let utxos = env.create_watchable_db(&mut rwtxn, "utxos")?;
        let stxos = env.create_watchable_db(&mut rwtxn, "stxos")?;
        rwtxn.commit()?;
        Ok(Self {
            env,
            seed: seed_db,
            address_to_index,
            index_to_address,
            utxos,
            stxos,
        })
    }

    /// Overwrite the seed, or set it if it does not already exist.
    pub fn overwrite_seed(&self, seed: &[u8; 64]) -> Result<(), Error> {
        let mut rwtxn = self.env.write_txn()?;
        self.seed.put(&mut rwtxn, &0, seed)?;
        self.address_to_index.clear(&mut rwtxn)?;
        self.index_to_address.clear(&mut rwtxn)?;
        self.utxos.clear(&mut rwtxn)?;
        self.stxos.clear(&mut rwtxn)?;
        rwtxn.commit()?;
        Ok(())
    }

    pub fn has_seed(&self) -> Result<bool, Error> {
        let rotxn = self.env.read_txn()?;
        Ok(self.seed.try_get(&rotxn, &0)?.is_some())
    }

    /// Set the seed, if it does not already exist
    pub fn set_seed(&self, seed: &[u8; 64]) -> Result<(), Error> {
        if self.has_seed()? {
            Err(Error::SeedAlreadyExists)
        } else {
            self.overwrite_seed(seed)
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
        let input_utxo_hashes: Vec<NodeHash> =
            inputs.iter().map(|(_, hash)| hash.into()).collect();
        let proof = accumulator
            .0
            .prove(&input_utxo_hashes)
            .map_err(Error::Utreexo)?;
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
        let input_utxo_hashes: Vec<NodeHash> =
            inputs.iter().map(|(_, hash)| hash.into()).collect();
        let proof = accumulator
            .0
            .prove(&input_utxo_hashes)
            .map_err(Error::Utreexo)?;
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
        let txn = self.env.read_txn()?;
        let mut utxos = vec![];
        for item in self.utxos.iter(&txn)? {
            utxos.push(item?);
        }
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
        let mut txn = self.env.write_txn()?;
        for outpoint in outpoints {
            self.utxos.delete(&mut txn, outpoint)?;
        }
        txn.commit()?;
        Ok(())
    }

    pub fn spend_utxos(
        &self,
        spent: &[(OutPoint, InPoint)],
    ) -> Result<(), Error> {
        let mut txn = self.env.write_txn()?;
        for (outpoint, inpoint) in spent {
            let output = self.utxos.try_get(&txn, outpoint)?;
            if let Some(output) = output {
                self.utxos.delete(&mut txn, outpoint)?;
                let spent_output = SpentOutput {
                    output,
                    inpoint: *inpoint,
                };
                self.stxos.put(&mut txn, outpoint, &spent_output)?;
            }
        }
        txn.commit()?;
        Ok(())
    }

    pub fn put_utxos(
        &self,
        utxos: &HashMap<OutPoint, Output>,
    ) -> Result<(), Error> {
        let mut txn = self.env.write_txn()?;
        for (outpoint, output) in utxos {
            self.utxos.put(&mut txn, outpoint, output)?;
        }
        txn.commit()?;
        Ok(())
    }

    pub fn get_balance(&self) -> Result<Balance, Error> {
        let mut balance = Balance::default();
        let txn = self.env.read_txn()?;
        let () = self
            .utxos
            .iter(&txn)?
            .transpose_into_fallible()
            .map_err(Error::from)
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
                Ok(())
            })?;
        Ok(balance)
    }

    pub fn get_utxos(&self) -> Result<HashMap<OutPoint, Output>, Error> {
        let txn = self.env.read_txn()?;
        let mut utxos = HashMap::new();
        for item in self.utxos.iter(&txn)? {
            let (outpoint, output) = item?;
            utxos.insert(outpoint, output);
        }
        Ok(utxos)
    }

    pub fn get_addresses(&self) -> Result<HashSet<Address>, Error> {
        let txn = self.env.read_txn()?;
        let mut addresses = HashSet::new();
        for item in self.index_to_address.iter(&txn)? {
            let (_, address) = item?;
            addresses.insert(address);
        }
        Ok(addresses)
    }

    pub fn authorize(
        &self,
        transaction: Transaction,
    ) -> Result<AuthorizedTransaction, Error> {
        let txn = self.env.read_txn()?;
        let mut authorizations = vec![];
        for (outpoint, _) in &transaction.inputs {
            let spent_utxo =
                self.utxos.try_get(&txn, outpoint)?.ok_or(Error::NoUtxo)?;
            let index = self
                .address_to_index
                .try_get(&txn, &spent_utxo.address)?
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
        let mut txn = self.env.write_txn()?;
        let (last_index, _) = self
            .index_to_address
            .last(&txn)?
            .unwrap_or(([0; 4], [0; 20].into()));
        let last_index = BigEndian::read_u32(&last_index);
        let index = last_index + 1;
        let signing_key = self.get_signing_key(&txn, index)?;
        let address = get_address(&signing_key.verifying_key());
        let index = index.to_be_bytes();
        self.index_to_address.put(&mut txn, &index, &address)?;
        self.address_to_index.put(&mut txn, &address, &index)?;
        txn.commit()?;
        Ok(address)
    }

    pub fn get_num_addresses(&self) -> Result<u32, Error> {
        let txn = self.env.read_txn()?;
        let (last_index, _) = self
            .index_to_address
            .last(&txn)?
            .unwrap_or(([0; 4], [0; 20].into()));
        let last_index = BigEndian::read_u32(&last_index);
        Ok(last_index)
    }

    fn get_signing_key(
        &self,
        rotxn: &RoTxn,
        index: u32,
    ) -> Result<ed25519_dalek::SigningKey, Error> {
        let seed = self.seed.try_get(rotxn, &0)?.ok_or(Error::NoSeed)?;
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

    pub fn clear_seed(&self) -> Result<(), Error> {
        let mut rwtxn = self.env.write_txn()?;
        self.seed.clear(&mut rwtxn)?;
        self.address_to_index.clear(&mut rwtxn)?;
        self.index_to_address.clear(&mut rwtxn)?;
        self.utxos.clear(&mut rwtxn)?;
        self.stxos.clear(&mut rwtxn)?;
        rwtxn.commit()?;
        Ok(())
    }

    /// Reset the wallet by removing all data including seed and transaction history.
    /// This ensures the wallet can only be restored by re-inputting the correct seed phrase.
    pub fn reset_wallet(&self) -> Result<(), Error> {
        let mut txn = self.env.write_txn()?;

        // Clear all databases
        self.seed.clear(&mut txn)?;
        self.address_to_index.clear(&mut txn)?;
        self.index_to_address.clear(&mut txn)?;
        self.utxos.clear(&mut txn)?;
        self.stxos.clear(&mut txn)?;

        txn.commit()?;
        Ok(())
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
        } = self;
        let watchables = [
            seed.watch(),
            address_to_index.watch(),
            index_to_address.watch(),
            utxos.watch(),
            stxos.watch(),
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
