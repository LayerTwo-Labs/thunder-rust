pub use crate::authorization::{get_address, Authorization};
use crate::types::{
    Address, AuthorizedTransaction, Content, GetValue, OutPoint, Output, Transaction,
};
use bip300301::bitcoin;
use byteorder::{BigEndian, ByteOrder};
use ed25519_dalek_bip32::*;
use heed::types::*;
use heed::{Database, RoTxn};
use std::collections::{HashMap, HashSet};
use std::path::Path;

#[derive(Clone)]
pub struct Wallet {
    env: heed::Env,
    // FIXME: Don't store the seed in plaintext.
    seed: Database<OwnedType<u8>, OwnedType<[u8; 64]>>,
    pub address_to_index: Database<SerdeBincode<Address>, OwnedType<[u8; 4]>>,
    pub index_to_address: Database<OwnedType<[u8; 4]>, SerdeBincode<Address>>,
    pub utxos: Database<SerdeBincode<OutPoint>, SerdeBincode<Output>>,
}

impl Wallet {
    pub const NUM_DBS: u32 = 4;

    pub fn new(path: &Path) -> Result<Self, Error> {
        std::fs::create_dir_all(path)?;
        let env = heed::EnvOpenOptions::new()
            .map_size(10 * 1024 * 1024) // 10MB
            .max_dbs(Self::NUM_DBS)
            .open(path)?;
        let seed_db = env.create_database(Some("seed"))?;
        let address_to_index = env.create_database(Some("address_to_index"))?;
        let index_to_address = env.create_database(Some("index_to_address"))?;
        let utxos = env.create_database(Some("utxos"))?;
        Ok(Self {
            env,
            seed: seed_db,
            address_to_index,
            index_to_address,
            utxos,
        })
    }

    pub fn set_seed(&self, seed: &[u8; 64]) -> Result<(), Error> {
        let mut txn = self.env.write_txn()?;
        self.seed.put(&mut txn, &0, seed)?;
        self.address_to_index.clear(&mut txn)?;
        self.index_to_address.clear(&mut txn)?;
        self.utxos.clear(&mut txn)?;
        txn.commit()?;
        Ok(())
    }

    pub fn has_seed(&self) -> Result<bool, Error> {
        let txn = self.env.read_txn()?;
        Ok(self.seed.get(&txn, &0)?.is_some())
    }

    pub fn create_withdrawal(
        &self,
        main_address: bitcoin::Address<bitcoin::address::NetworkUnchecked>,
        value: u64,
        main_fee: u64,
        fee: u64,
    ) -> Result<Transaction, Error> {
        let (total, coins) = self.select_coins(value + fee + main_fee)?;
        let change = total - value - fee;
        let inputs = coins.into_keys().collect();
        let outputs = vec![
            Output {
                address: self.get_new_address()?,
                content: Content::Withdrawal {
                    value,
                    main_fee,
                    main_address,
                },
            },
            Output {
                address: self.get_new_address()?,
                content: Content::Value(change),
            },
        ];
        Ok(Transaction { inputs, outputs })
    }

    pub fn create_transaction(
        &self,
        address: Address,
        value: u64,
        fee: u64,
    ) -> Result<Transaction, Error> {
        let (total, coins) = self.select_coins(value + fee)?;
        let change = total - value - fee;
        let inputs = coins.into_keys().collect();
        let outputs = vec![
            Output {
                address,
                content: Content::Value(value),
            },
            Output {
                address: self.get_new_address()?,
                content: Content::Value(change),
            },
        ];
        Ok(Transaction { inputs, outputs })
    }

    pub fn select_coins(&self, value: u64) -> Result<(u64, HashMap<OutPoint, Output>), Error> {
        let txn = self.env.read_txn()?;
        let mut utxos = vec![];
        for item in self.utxos.iter(&txn)? {
            utxos.push(item?);
        }
        utxos.sort_unstable_by_key(|(_, output)| output.get_value());

        let mut selected = HashMap::new();
        let mut total: u64 = 0;
        for (outpoint, output) in &utxos {
            if output.content.is_withdrawal() {
                continue;
            }
            if total > value {
                break;
            }
            total += output.get_value();
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

    pub fn put_utxos(&self, utxos: &HashMap<OutPoint, Output>) -> Result<(), Error> {
        let mut txn = self.env.write_txn()?;
        for (outpoint, output) in utxos {
            self.utxos.put(&mut txn, outpoint, output)?;
        }
        txn.commit()?;
        Ok(())
    }

    pub fn get_balance(&self) -> Result<u64, Error> {
        let mut balance: u64 = 0;
        let txn = self.env.read_txn()?;
        for item in self.utxos.iter(&txn)? {
            let (_, utxo) = item?;
            balance += utxo.get_value();
        }
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

    pub fn authorize(&self, transaction: Transaction) -> Result<AuthorizedTransaction, Error> {
        let txn = self.env.read_txn()?;
        let mut authorizations = vec![];
        for input in &transaction.inputs {
            let spent_utxo = self.utxos.get(&txn, input)?.ok_or(Error::NoUtxo)?;
            let index = self
                .address_to_index
                .get(&txn, &spent_utxo.address)?
                .ok_or(Error::NoIndex {
                    address: spent_utxo.address,
                })?;
            let index = BigEndian::read_u32(&index);
            let keypair = self.get_keypair(&txn, index)?;
            let signature = crate::authorization::sign(&keypair, &transaction)?;
            authorizations.push(Authorization {
                public_key: keypair.public,
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
        let keypair = self.get_keypair(&txn, index)?;
        let address = get_address(&keypair.public);
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

    fn get_keypair(&self, txn: &RoTxn, index: u32) -> Result<ed25519_dalek::Keypair, Error> {
        let seed = self.seed.get(txn, &0)?.ok_or(Error::NoSeed)?;
        let xpriv = ExtendedSecretKey::from_seed(&seed)?;
        let derivation_path = DerivationPath::new([
            ChildIndex::Hardened(1),
            ChildIndex::Hardened(0),
            ChildIndex::Hardened(0),
            ChildIndex::Hardened(index),
        ]);
        let child = xpriv.derive(&derivation_path)?;
        let public = child.public_key();
        let secret = child.secret_key;
        Ok(ed25519_dalek::Keypair { secret, public })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("heed error")]
    Heed(#[from] heed::Error),
    #[error("bip32 error")]
    Bip32(#[from] ed25519_dalek_bip32::Error),
    #[error("address {address} does not exist")]
    AddressDoesNotExist { address: crate::types::Address },
    #[error("utxo doesn't exist")]
    NoUtxo,
    #[error("wallet doesn't have a seed")]
    NoSeed,
    #[error("no index for address {address}")]
    NoIndex { address: Address },
    #[error("authorization error")]
    Authorization(#[from] crate::authorization::Error),
    #[error("io error")]
    Io(#[from] std::io::Error),
    #[error("not enough funds")]
    NotEnoughFunds,
}
