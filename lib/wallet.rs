use std::{
    collections::{BTreeMap, HashMap, HashSet},
    path::Path,
    rc::Rc,
};

use bitcoin::{
    Amount,
    bip32::{ChildNumber, DerivationPath, Xpriv},
};
use fallible_iterator::FallibleIterator as _;
use futures::{Stream, StreamExt};
use heed::{
    byteorder::BigEndian,
    types::{Bytes, SerdeBincode, U8, U32},
};
use parking_lot::RwLock;
use rustreexo::accumulator::node_hash::BitcoinNodeHash;
use serde::{Deserialize, Serialize};
use sneed::{
    DbError, EnvError, RoTxnError, RwTxnError, UnitKey, db, env, rotxn, rwtxn,
};
use tokio_stream::{StreamMap, wrappers::WatchStream};

use crate::{
    authorization,
    types::{
        Accumulator, AmountOverflowError, AmountUnderflowError, BlockHash,
        Body, Header, PointedOutput, Txid, UtreexoError, VERSION, Version,
        hash,
    },
    util::Watchable,
};
pub use crate::{
    authorization::{Authorization, get_address},
    types::{
        AuthorizedTransaction, GetValue, InPoint, OutPoint, Output,
        OutputContent, SpentOutput, Transaction, TransparentAddress,
        orchard::{self, ShardTree, ShardTreeDb},
    },
};

use self::orchard::ShardTreeDbTxn;

#[derive(Clone, Debug, Default, Deserialize, Serialize, utoipa::ToSchema)]
pub struct Balance {
    #[serde(
        rename = "total_shielded_sats",
        with = "bitcoin::amount::serde::as_sat"
    )]
    #[schema(value_type = u64)]
    pub total_shielded: Amount,
    #[serde(
        rename = "total_transparent_sats",
        with = "bitcoin::amount::serde::as_sat"
    )]
    #[schema(value_type = u64)]
    pub total_transparent: Amount,
    #[serde(
        rename = "available_shielded_sats",
        with = "bitcoin::amount::serde::as_sat"
    )]
    #[schema(value_type = u64)]
    pub available_shielded: Amount,
    #[serde(
        rename = "available_transparent_sats",
        with = "bitcoin::amount::serde::as_sat"
    )]
    #[schema(value_type = u64)]
    pub available_transparent: Amount,
}

impl Balance {
    /// Get the total balance
    pub fn total(&self) -> Amount {
        self.total_shielded + self.total_transparent
    }

    /// Get the total available amount
    pub fn available(&self) -> Amount {
        self.available_shielded + self.available_transparent
    }
}

#[allow(clippy::duplicated_attributes)]
#[derive(Debug, thiserror::Error, transitive::Transitive)]
#[transitive(
    from(db::error::Delete, DbError),
    from(db::error::Get, DbError),
    from(db::error::IterInit, DbError),
    from(db::error::IterItem, DbError),
    from(db::error::Last, DbError),
    from(db::error::Put, DbError),
    from(db::error::TryGet, DbError),
    from(env::error::CreateDb, EnvError),
    from(env::error::OpenEnv, EnvError),
    from(env::error::ReadTxn, EnvError),
    from(env::error::WriteTxn, EnvError),
    from(rotxn::error::Commit, RoTxnError),
    from(rwtxn::error::Commit, RwTxnError)
)]
pub enum Error {
    #[error("address {address} does not exist")]
    AddressDoesNotExist {
        address: crate::types::TransparentAddress,
    },
    #[error(transparent)]
    AmountOverflow(#[from] AmountOverflowError),
    #[error(transparent)]
    AmountUnderflow(#[from] AmountUnderflowError),
    #[error("authorization error")]
    Authorization(#[from] crate::authorization::Error),
    #[error("bip32 error")]
    Bip32(#[from] bitcoin::bip32::Error),
    #[error("Error creating orchard note commitments DBs")]
    CreateOrchardNoteCommitmentsDb(#[from] orchard::CreateShardTreeDbError),
    #[error(transparent)]
    Db(#[from] DbError),
    #[error("Database env error")]
    DbEnv(#[from] EnvError),
    #[error("Database read error")]
    DbRead(#[from] RoTxnError),
    #[error("Database write error")]
    DbWrite(#[from] RwTxnError),
    #[error("io error")]
    Io(#[from] std::io::Error),
    #[error("no index for address {address}")]
    NoIndex { address: TransparentAddress },
    #[error(
        "wallet does not have a seed (set with RPC `set-seed-from-mnemonic`)"
    )]
    NoSeed,
    #[error("not enough funds")]
    NotEnoughFunds,
    #[error("utxo does not exist")]
    NoUtxo,
    #[error("Orchard bundle builder error")]
    OrchardBuilder(#[from] orchard::BuildError),
    #[error("Orchard output error")]
    OrchardOutput(#[from] orchard::OutputError),
    #[error("Orchard value overflow error")]
    OrchardOverflow(#[from] orchard::OverflowError),
    #[error("Orchard ShardTree error")]
    OrchardShardTree(#[from] orchard::ShardTreeError),
    #[error(
        "Failed to truncate orchard shard tree to checkpoint `{checkpoint:?}`"
    )]
    OrchardShardTreeTruncate { checkpoint: Option<BlockHash> },
    #[error("Orchard ShardTreeStore error")]
    OrchardShardTreeStore(#[from] orchard::ShardTreeStoreError),
    #[error("Orchard spend error")]
    OrchardSpend(#[from] orchard::SpendError),
    #[error("failed to parse mnemonic seed phrase")]
    ParseMnemonic(#[source] bip39::ErrorKind),
    #[error("seed has already been set")]
    SeedAlreadyExists,
    #[error(transparent)]
    Utreexo(#[from] UtreexoError),
    #[error("zip32 error ({0})")]
    Zip32(::orchard::zip32::Error),
}

impl From<orchard::shardtree_db::db_txn::CommitError> for Error {
    fn from(err: orchard::shardtree_db::db_txn::CommitError) -> Self {
        match err {
            orchard::shardtree_db::db_txn::CommitError::Ro(err) => err.into(),
            orchard::shardtree_db::db_txn::CommitError::Rw(err) => err.into(),
        }
    }
}

/// Marker type for Wallet Env
pub struct WalletEnv;

type DatabaseUnique<KC, DC> = sneed::DatabaseUnique<KC, DC, WalletEnv>;
type Env = sneed::Env<WalletEnv>;
type RoTxn<'a> = sneed::RoTxn<'a, WalletEnv>;
pub type RwTxn<'a> = sneed::RwTxn<'a, WalletEnv>;

/// Note with position
type NotePosition = (orchard::Note, orchard::PositionWrapper);

#[derive(Clone)]
pub struct Wallet {
    env: sneed::Env<WalletEnv>,
    // Seed is always [u8; 64], but due to serde not implementing serialize
    // for [T; 64], use heed's `Bytes`
    // TODO: Don't store the seed in plaintext.
    seed: DatabaseUnique<U8, Bytes>,
    /// Map each address to it's index
    address_to_index:
        DatabaseUnique<SerdeBincode<TransparentAddress>, U32<BigEndian>>,
    /// Map each address index to an address
    index_to_address:
        DatabaseUnique<U32<BigEndian>, SerdeBincode<TransparentAddress>>,
    /// Map each orchard address to it's index
    orchard_address_to_index:
        DatabaseUnique<SerdeBincode<orchard::Address>, U32<BigEndian>>,
    /// Map each orchard address index to an orchard address
    orchard_index_to_address:
        DatabaseUnique<U32<BigEndian>, SerdeBincode<orchard::Address>>,
    /// Map tx and action index to plaintext memos.
    /// Memos are always [u8; 512], but due to serde not implementing serialize
    // for [T; 512], use heed's `Bytes`
    orchard_memos: DatabaseUnique<SerdeBincode<(Txid, u32)>, Bytes>,
    orchard_note_commitments: ShardTreeDb<WalletEnv>,
    orchard_notes: DatabaseUnique<
        SerdeBincode<orchard::Nullifier>,
        SerdeBincode<NotePosition>,
    >,
    orchard_spent_notes:
        DatabaseUnique<SerdeBincode<(Txid, u32)>, SerdeBincode<NotePosition>>,
    utxos: DatabaseUnique<SerdeBincode<OutPoint>, SerdeBincode<Output>>,
    stxos: DatabaseUnique<SerdeBincode<OutPoint>, SerdeBincode<SpentOutput>>,
    /// Block that the wallet was last synced to.
    /// May be empty, if there is no tip yet
    tip: DatabaseUnique<UnitKey, SerdeBincode<BlockHash>>,
    _version: DatabaseUnique<UnitKey, SerdeBincode<Version>>,
}

impl Wallet {
    pub const NUM_DBS: u32 = ShardTreeDb::<WalletEnv>::NUM_DBS + 12;

    pub fn new(path: &Path) -> Result<Self, Error> {
        std::fs::create_dir_all(path)?;
        let env = {
            let mut env_open_options = heed::EnvOpenOptions::new();
            env_open_options
                .map_size(100 * 1024 * 1024) // 100MB
                .max_dbs(Self::NUM_DBS);
            unsafe { Env::open(&env_open_options, path) }?
        };
        let mut rwtxn = env.write_txn()?;
        let seed_db = DatabaseUnique::create(&env, &mut rwtxn, "seed")?;
        let address_to_index =
            DatabaseUnique::create(&env, &mut rwtxn, "address_to_index")?;
        let index_to_address =
            DatabaseUnique::create(&env, &mut rwtxn, "index_to_address")?;
        let orchard_address_to_index = DatabaseUnique::create(
            &env,
            &mut rwtxn,
            "orchard_address_to_index",
        )?;
        let orchard_index_to_address = DatabaseUnique::create(
            &env,
            &mut rwtxn,
            "orchard_index_to_address",
        )?;
        let orchard_memos =
            DatabaseUnique::create(&env, &mut rwtxn, "orchard_memos")?;
        let orchard_note_commitments = ShardTreeDb::new(
            &env,
            &mut rwtxn,
            Some("orchard_note_commitments"),
        )?;
        let orchard_notes =
            DatabaseUnique::create(&env, &mut rwtxn, "orchard_notes")?;
        let orchard_spent_notes =
            DatabaseUnique::create(&env, &mut rwtxn, "orchard_spent_notes")?;
        let utxos = DatabaseUnique::create(&env, &mut rwtxn, "utxos")?;
        let stxos = DatabaseUnique::create(&env, &mut rwtxn, "stxos")?;
        let tip = DatabaseUnique::create(&env, &mut rwtxn, "tip")?;
        let version = DatabaseUnique::create(&env, &mut rwtxn, "version")?;
        if version.try_get(&rwtxn, &())?.is_none() {
            version.put(&mut rwtxn, &(), &*VERSION)?;
        }
        rwtxn.commit()?;
        Ok(Self {
            env,
            seed: seed_db,
            address_to_index,
            index_to_address,
            orchard_address_to_index,
            orchard_index_to_address,
            orchard_memos,
            orchard_note_commitments,
            orchard_notes,
            orchard_spent_notes,
            utxos,
            stxos,
            tip,
            _version: version,
        })
    }

    pub fn env(&self) -> &Env {
        &self.env
    }

    /// Returns `txn` after a successful load
    #[allow(clippy::type_complexity)]
    fn get_shard_tree<'a>(
        &self,
        txn: ShardTreeDbTxn<'a, WalletEnv>,
    ) -> Result<
        (
            ShardTree<'a, WalletEnv>,
            Rc<RwLock<Option<ShardTreeDbTxn<'a, WalletEnv>>>>,
            ShardTreeDbTxn<'a, WalletEnv>,
        ),
        Error,
    > {
        let db_txn = Rc::new(RwLock::new(Some(txn)));
        let store = orchard::ShardTreeStore {
            txn: Rc::downgrade(&db_txn),
            db: self.orchard_note_commitments.clone(),
        };
        let tree = orchard::shardtree_db::load_shard_tree(store)?;
        let txn = db_txn.write().take().unwrap();
        Ok((tree, db_txn, txn))
    }

    /// Returns the rwtxn, if `db_txn` is unique
    fn put_shard_tree<'a>(
        &self,
        rwtxn: RwTxn<'a>,
        db_txn: Rc<RwLock<Option<ShardTreeDbTxn<'a, WalletEnv>>>>,
        tree: ShardTree<'a, WalletEnv>,
    ) -> Result<Option<RwTxn<'a>>, Error> {
        *db_txn.write() = Some(ShardTreeDbTxn::Rw(rwtxn));
        let store = orchard::shardtree_db::store_shard_tree(tree)?;
        drop(store);
        let rwtxn = Rc::into_inner(db_txn).and_then(|lock| {
            match lock.into_inner().unwrap() {
                ShardTreeDbTxn::Ro(_) => None,
                ShardTreeDbTxn::Rw(rwtxn) => Some(rwtxn),
            }
        });
        Ok(rwtxn)
    }

    fn get_master_xpriv(&self, rotxn: &RoTxn) -> Result<Xpriv, Error> {
        let seed_bytes = self.seed.try_get(rotxn, &0)?.ok_or(Error::NoSeed)?;
        let res = Xpriv::new_master(bitcoin::NetworkKind::Test, seed_bytes)?;
        Ok(res)
    }

    fn get_orchard_spending_key(
        &self,
        rotxn: &RoTxn,
    ) -> Result<orchard::SpendingKey, Error> {
        let master_xpriv = self.get_master_xpriv(rotxn)?;
        let derivation_path = DerivationPath::master()
            .child(ChildNumber::Hardened { index: 2 })
            .child(ChildNumber::Hardened { index: 0 })
            .child(ChildNumber::Normal { index: 0 });
        let xpriv = master_xpriv
            .derive_priv(&bitcoin::key::Secp256k1::new(), &derivation_path)?;
        orchard::SpendingKey::from_zip32_seed(
            &xpriv.private_key.secret_bytes(),
            0,
            zip32::AccountId::ZERO,
        )
        .map_err(Error::Zip32)
    }

    fn get_orchard_full_viewing_key(
        &self,
        rotxn: &RoTxn,
    ) -> Result<orchard::FullViewingKey, Error> {
        self.get_orchard_spending_key(rotxn)
            .map(|spending_key| orchard::FullViewingKey::from(&spending_key))
    }

    /// Returns the external and internal incoming viewing keys
    pub fn get_orchard_incoming_viewing_keys(
        &self,
        rotxn: &RoTxn,
    ) -> Result<[orchard::IncomingViewingKey; 2], Error> {
        let fvk = self.get_orchard_full_viewing_key(rotxn)?;
        let external = fvk.to_ivk(orchard::Scope::External);
        let internal = fvk.to_ivk(orchard::Scope::Internal);
        Ok([external, internal])
    }

    /// Returns the external and internal outgoing viewing keys
    pub fn get_orchard_outgoing_viewing_keys(
        &self,
        rotxn: &RoTxn,
    ) -> Result<[orchard::OutgoingViewingKey; 2], Error> {
        let fvk = self.get_orchard_full_viewing_key(rotxn)?;
        let external = fvk.to_ovk(orchard::Scope::External);
        let internal = fvk.to_ovk(orchard::Scope::Internal);
        Ok([external, internal])
    }

    pub fn get_new_orchard_address(
        &self,
        rwtxn: &mut RwTxn,
    ) -> Result<orchard::Address, Error> {
        let next_index = self
            .orchard_index_to_address
            .last(rwtxn)?
            .map(|(idx, _)| idx + 1)
            .unwrap_or(0);
        let full_viewing_key = self.get_orchard_full_viewing_key(rwtxn)?;
        let address = orchard::Address(
            full_viewing_key.address_at(next_index, zip32::Scope::External),
        );
        self.orchard_index_to_address
            .put(rwtxn, &next_index, &address)?;
        self.orchard_address_to_index
            .put(rwtxn, &address, &next_index)?;
        Ok(address)
    }

    fn get_tx_signing_key(
        &self,
        rotxn: &RoTxn,
        index: u32,
    ) -> Result<ed25519_dalek::SigningKey, Error> {
        let master_xpriv = self.get_master_xpriv(rotxn)?;
        let derivation_path = DerivationPath::master()
            .child(ChildNumber::Hardened { index: 0 })
            .child(ChildNumber::Normal { index });
        let xpriv = master_xpriv
            .derive_priv(&bitcoin::key::Secp256k1::new(), &derivation_path)?;
        let signing_key = xpriv.private_key.secret_bytes().into();
        Ok(signing_key)
    }

    pub fn get_new_transparent_address(
        &self,
        rwtxn: &mut RwTxn,
    ) -> Result<TransparentAddress, Error> {
        let next_index = self
            .index_to_address
            .last(rwtxn)?
            .map(|(idx, _)| idx + 1)
            .unwrap_or(0);
        let tx_signing_key = self.get_tx_signing_key(rwtxn, next_index)?;
        let address = get_address(&tx_signing_key.verifying_key());
        self.index_to_address.put(rwtxn, &next_index, &address)?;
        self.address_to_index.put(rwtxn, &address, &next_index)?;
        Ok(address)
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

    #[allow(clippy::type_complexity)]
    pub fn select_shielded_coins<'a>(
        &self,
        txn: ShardTreeDbTxn<'a, WalletEnv>,
        value: bitcoin::Amount,
    ) -> Result<
        (
            ShardTreeDbTxn<'a, WalletEnv>,
            bitcoin::Amount,
            orchard::Anchor,
            BTreeMap<orchard::Nullifier, (orchard::Note, orchard::MerklePath)>,
        ),
        Error,
    > {
        let mut nullifiers: Vec<_> =
            self.orchard_notes.iter_keys(txn.as_ref())?.collect()?;
        rand::seq::SliceRandom::shuffle(
            nullifiers.as_mut_slice(),
            &mut rand::rngs::OsRng,
        );
        let (commitments_tree, _db_txn, txn) = self.get_shard_tree(txn)?;
        let anchor = match commitments_tree.root_at_checkpoint_depth(Some(0))? {
            Some(anchor) => anchor.into(),
            None => {
                assert!(nullifiers.is_empty());
                orchard::Anchor::empty_tree()
            }
        };
        let mut selected = BTreeMap::new();
        let mut total = bitcoin::Amount::ZERO;
        for nullifier in nullifiers {
            if total >= value {
                break;
            }
            let (note, position) =
                self.orchard_notes.get(txn.as_ref(), &nullifier)?;
            let path = commitments_tree
                .witness_at_checkpoint_depth(position.0, 0)?
                .expect("Should be able to compute merkle path at depth 0")
                .into();
            total =
                total.checked_add(note.value()).ok_or(AmountOverflowError)?;
            selected.insert(nullifier, (note, path));
        }
        if total < value {
            return Err(Error::NotEnoughFunds);
        }
        Ok((txn, total, anchor, selected))
    }

    pub fn select_transparent_coins(
        &self,
        rotxn: &RoTxn,
        value: bitcoin::Amount,
    ) -> Result<(bitcoin::Amount, HashMap<OutPoint, Output>), Error> {
        let mut utxos: Vec<_> = self
            .utxos
            .iter(rotxn)
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
        let mut rwtxn = self.env.write_txn()?;
        let (total, coins) = self.select_transparent_coins(
            &rwtxn,
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
                address: self.get_new_transparent_address(&mut rwtxn)?,
                content: OutputContent::Withdrawal {
                    value,
                    main_fee,
                    main_address,
                },
            },
            Output {
                address: self.get_new_transparent_address(&mut rwtxn)?,
                content: OutputContent::Value(change),
            },
        ];
        rwtxn.commit()?;
        Ok(Transaction {
            inputs,
            proof,
            outputs,
            orchard_bundle: None,
        })
    }

    pub fn create_transaction(
        &self,
        accumulator: &Accumulator,
        address: TransparentAddress,
        value: bitcoin::Amount,
        fee: bitcoin::Amount,
    ) -> Result<Transaction, Error> {
        let mut rwtxn = self.env.write_txn()?;
        let (total, coins) = self.select_transparent_coins(
            &rwtxn,
            value.checked_add(fee).ok_or(AmountOverflowError)?,
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
                address,
                content: OutputContent::Value(value),
            },
            Output {
                address: self.get_new_transparent_address(&mut rwtxn)?,
                content: OutputContent::Value(change),
            },
        ];
        rwtxn.commit()?;
        Ok(Transaction {
            inputs,
            proof,
            outputs,
            orchard_bundle: None,
        })
    }

    /// Create a fully shielded transaction.
    /// Fees are paid from shielded notes.
    pub fn create_shielded_transaction(
        &self,
        accumulator: &Accumulator,
        address: orchard::Address,
        value: bitcoin::Amount,
        fee: bitcoin::Amount,
        memo: [u8; 512],
    ) -> Result<Transaction, Error> {
        let mut rwtxn = self.env.write_txn()?;
        let change_addr = self.get_new_orchard_address(&mut rwtxn)?;
        let orchard_spending_key = self.get_orchard_spending_key(&rwtxn)?;
        let rwtxn = ShardTreeDbTxn::Rw(rwtxn);
        let (rwtxn, value_in, anchor, coins) = self.select_shielded_coins(
            rwtxn,
            value.checked_add(fee).ok_or(AmountOverflowError)?,
        )?;
        let change = value_in - value - fee;
        let utreexo_proof = accumulator.prove(&Vec::new())?;
        let orchard_bundle = 'orchard_bundle: {
            let fvk = orchard::FullViewingKey::from(&orchard_spending_key);
            let flags = orchard::BundleFlags::ENABLED;
            let mut builder = orchard::Builder::new(flags, false, anchor);
            let ovk = fvk.to_ovk(orchard::Scope::Internal);
            // Add recipient output
            builder.add_output(
                Some(ovk.clone()),
                address,
                orchard::NoteValue::from_raw(value.to_sat()),
                memo,
            )?;
            // Add change output
            builder.add_output(
                Some(ovk.clone()),
                change_addr,
                orchard::NoteValue::from_raw(change.to_sat()),
                [0u8; 512],
            )?;
            for (note, path) in coins.into_values() {
                builder.add_spend(fvk.clone(), note, path)?;
            }
            let Some((bundle, _metadata)) =
                builder.build(rand::rngs::OsRng, Some(ovk))?
            else {
                break 'orchard_bundle None;
            };
            let bundle = bundle.create_proof(rand::rngs::OsRng)?;
            Some(bundle)
        };
        let transaction = Transaction {
            inputs: Vec::new(),
            proof: utreexo_proof,
            outputs: Vec::new(),
            orchard_bundle,
        };
        let spend_auth_key =
            orchard::SpendAuthorizingKey::from(&orchard_spending_key);
        let res = authorization::sign_orchard(&[spend_auth_key], transaction)?;
        rwtxn.commit()?;
        Ok(res)
    }

    /// Create a transaction that shields the specified amount.
    ///
    /// If at least one note is available to spend, spends a note and creates
    /// a new note worth `value` more than the spent note.
    pub fn create_shield_transaction(
        &self,
        accumulator: &Accumulator,
        value: bitcoin::Amount,
        fee: bitcoin::Amount,
    ) -> Result<Transaction, Error> {
        let mut rwtxn = self.env.write_txn()?;
        let (value_in, coins) = self.select_transparent_coins(
            &rwtxn,
            value.checked_add(fee).ok_or(AmountOverflowError)?,
        )?;
        let change = value_in - value - fee;
        let inputs: Vec<_> = coins
            .into_iter()
            .map(|(outpoint, output)| {
                let utxo_hash = hash(&PointedOutput { outpoint, output });
                (outpoint, utxo_hash)
            })
            .collect();
        let input_utxo_hashes: Vec<BitcoinNodeHash> =
            inputs.iter().map(|(_, hash)| hash.into()).collect();
        let utreexo_proof = accumulator.prove(&input_utxo_hashes)?;
        let outputs = vec![Output {
            address: self.get_new_transparent_address(&mut rwtxn)?,
            content: OutputContent::Value(change),
        }];
        let shielded_addr = self.get_new_orchard_address(&mut rwtxn)?;
        let orchard_spending_key = self.get_orchard_spending_key(&rwtxn)?;
        let mut rwtxn = ShardTreeDbTxn::Rw(rwtxn);
        let orchard_bundle = 'orchard_bundle: {
            let fvk = orchard::FullViewingKey::from(&orchard_spending_key);
            let ovk = fvk.to_ovk(orchard::Scope::Internal);
            let nullifiers: Vec<_> =
                self.orchard_notes.iter_keys(rwtxn.as_ref())?.collect()?;
            let nullifier = rand::seq::SliceRandom::choose(
                nullifiers.as_slice(),
                &mut rand::rngs::OsRng,
            );
            let mut builder = if let Some(nullifier) = nullifier {
                let (spend_note, position) =
                    self.orchard_notes.get(rwtxn.as_ref(), nullifier)?;
                let flags = orchard::BundleFlags::ENABLED;
                let (shard_tree, _db_txn, rwtxn_) =
                    self.get_shard_tree(rwtxn)?;
                rwtxn = rwtxn_;
                let anchor = shard_tree
                    .root_at_checkpoint_depth(Some(0))?
                    .expect("Anchor should exist if notes exist")
                    .into();
                let merkle_path = shard_tree
                    .witness_at_checkpoint_depth(position.0, 0)?
                    .expect("Merkle path should exist for wallet notes")
                    .into();
                let mut builder = orchard::Builder::new(flags, false, anchor);
                builder.add_spend(fvk, spend_note, merkle_path)?;
                builder
            } else {
                let flags = orchard::BundleFlags::SPENDS_DISABLED;
                let anchor = orchard::Anchor::empty_tree();
                orchard::Builder::new(flags, true, anchor)
            };
            let output_note_value =
                value + builder.value_balance()?.to_unsigned().unwrap();
            builder.add_output(
                Some(ovk.clone()),
                shielded_addr,
                orchard::NoteValue::from_raw(output_note_value.to_sat()),
                [0u8; 512],
            )?;
            let Some((bundle, _metadata)) =
                builder.build(rand::rngs::OsRng, Some(ovk))?
            else {
                break 'orchard_bundle None;
            };
            let bundle = bundle.create_proof(rand::rngs::OsRng)?;
            Some(bundle)
        };
        let transaction = Transaction {
            inputs,
            proof: utreexo_proof,
            outputs,
            orchard_bundle,
        };
        let spend_auth_key =
            orchard::SpendAuthorizingKey::from(&orchard_spending_key);
        let res = authorization::sign_orchard(&[spend_auth_key], transaction)?;
        rwtxn.commit()?;
        Ok(res)
    }

    /// Create a transaction that unshields the specified amount.
    /// Fees are paid from shielded notes.
    pub fn create_unshield_transaction(
        &self,
        accumulator: &Accumulator,
        value: bitcoin::Amount,
        fee: bitcoin::Amount,
    ) -> Result<Transaction, Error> {
        let mut rwtxn = self.env.write_txn()?;
        let inputs = Vec::new();
        let input_utxo_hashes = Vec::<BitcoinNodeHash>::new();
        let utreexo_proof = accumulator.prove(&input_utxo_hashes)?;
        let outputs = vec![Output {
            address: self.get_new_transparent_address(&mut rwtxn)?,
            content: OutputContent::Value(value),
        }];
        let shielded_addr = self.get_new_orchard_address(&mut rwtxn)?;
        let orchard_spending_key = self.get_orchard_spending_key(&rwtxn)?;
        let (rwtxn, value_in, anchor, coins) = self.select_shielded_coins(
            ShardTreeDbTxn::Rw(rwtxn),
            value.checked_add(fee).ok_or(AmountOverflowError)?,
        )?;
        let change = value_in - value - fee;
        let orchard_bundle = 'orchard_bundle: {
            let fvk = orchard::FullViewingKey::from(&orchard_spending_key);
            let flags = orchard::BundleFlags::ENABLED;
            let mut builder = orchard::Builder::new(flags, true, anchor);
            let ovk = fvk.to_ovk(orchard::Scope::Internal);
            // Add change output
            builder.add_output(
                Some(ovk.clone()),
                shielded_addr,
                orchard::NoteValue::from_raw(change.to_sat()),
                [0u8; 512],
            )?;
            for (note, path) in coins.into_values() {
                builder.add_spend(fvk.clone(), note, path)?;
            }
            let Some((bundle, _metadata)) =
                builder.build(rand::rngs::OsRng, Some(ovk))?
            else {
                break 'orchard_bundle None;
            };
            let bundle = bundle.create_proof(rand::rngs::OsRng)?;
            Some(bundle)
        };
        let transaction = Transaction {
            inputs,
            proof: utreexo_proof,
            outputs,
            orchard_bundle,
        };
        let spend_auth_key =
            orchard::SpendAuthorizingKey::from(&orchard_spending_key);
        let res = authorization::sign_orchard(&[spend_auth_key], transaction)?;
        rwtxn.commit()?;
        Ok(res)
    }

    pub fn delete_utxos(&self, outpoints: &[OutPoint]) -> Result<(), Error> {
        let mut rwtxn = self.env.write_txn()?;
        for outpoint in outpoints {
            self.utxos.delete(&mut rwtxn, outpoint)?;
        }
        rwtxn.commit()?;
        Ok(())
    }

    pub fn spend_utxos(
        &self,
        rwtxn: &mut RwTxn,
        spent: &[(OutPoint, InPoint)],
    ) -> Result<(), Error> {
        for (outpoint, inpoint) in spent {
            let output = self.utxos.try_get(rwtxn, outpoint)?;
            if let Some(output) = output {
                self.utxos.delete(rwtxn, outpoint)?;
                let spent_output = SpentOutput {
                    output,
                    inpoint: *inpoint,
                };
                self.stxos.put(rwtxn, outpoint, &spent_output)?;
            }
        }
        Ok(())
    }

    /// Set the wallet tip
    pub fn put_tip(
        &self,
        rwtxn: &mut RwTxn,
        tip: &BlockHash,
    ) -> Result<(), Error> {
        self.tip.put(rwtxn, &(), tip)?;
        Ok(())
    }

    /// Connects ONLY the orchard effects from a block.
    /// Updates the wallet tip.
    pub fn connect_orchard_block<'a>(
        &self,
        mut rwtxn: RwTxn<'a>,
        header: &Header,
        body: &Body,
    ) -> Result<RwTxn<'a>, Error> {
        assert_eq!(self.try_get_tip(&rwtxn)?, header.prev_side_hash);
        assert_eq!(body.compute_merkle_root(), header.merkle_root);
        let fvk = self.get_orchard_full_viewing_key(&rwtxn)?;
        let ivks = self.get_orchard_incoming_viewing_keys(&rwtxn)?;
        let ovks = self.get_orchard_outgoing_viewing_keys(&rwtxn)?;
        let (mut shard_tree, db_txn, txn) =
            self.get_shard_tree(ShardTreeDbTxn::Rw(rwtxn))?;
        rwtxn = match txn {
            ShardTreeDbTxn::Ro(_) => panic!("impossible"),
            ShardTreeDbTxn::Rw(rw) => rw,
        };
        for tx in &body.transactions {
            let Some(orchard_bundle) = tx.orchard_bundle.as_ref() else {
                continue;
            };
            let next_leaf_position = shard_tree
                .max_leaf_position(None)?
                .map_or_else(|| 0.into(), |pos| pos + 1);
            let txid = tx.txid();
            let mut decrypted_incoming_note_idxs = HashSet::new();
            let decrypted_incoming_notes =
                orchard_bundle.decrypt_outputs_with_keys(&ivks);
            for (idx, _, note, _, memo) in decrypted_incoming_notes {
                decrypted_incoming_note_idxs.insert(idx);
                if memo != [0; 512] {
                    self.orchard_memos.put(
                        &mut rwtxn,
                        &(txid, idx as u32),
                        &memo,
                    )?;
                }
                if note.value() != Amount::ZERO {
                    let nullifier = note.nullifier(&fvk);
                    let position = next_leaf_position + idx as u64;
                    self.orchard_notes.put(
                        &mut rwtxn,
                        &nullifier,
                        &(note, orchard::PositionWrapper(position)),
                    )?;
                }
            }
            let mut decrypted_outgoing_note_idxs = HashSet::new();
            let decrypted_outgoing_notes =
                orchard_bundle.recover_outputs_with_ovks(&ovks);
            for (idx, _, _note, _, _) in decrypted_outgoing_notes {
                decrypted_outgoing_note_idxs.insert(idx);
                let nf = *orchard_bundle.actions()[idx].nullifier();
                let Some((spent_note, position)) =
                    self.orchard_notes.try_get(&rwtxn, &nf)?
                else {
                    tracing::warn!(nullifier = ?nf, "Missing spent note");
                    continue;
                };
                self.orchard_notes.delete(&mut rwtxn, &nf)?;
                self.orchard_spent_notes.put(
                    &mut rwtxn,
                    &(txid, idx as u32),
                    &(spent_note, position),
                )?;
            }
            for (idx, action) in orchard_bundle.actions().iter().enumerate() {
                let retention = if decrypted_incoming_note_idxs.contains(&idx) {
                    incrementalmerkletree::Retention::Marked
                } else {
                    // TODO: is this correct?
                    incrementalmerkletree::Retention::Ephemeral
                };
                let () = shard_tree.append(
                    orchard::MerkleHashOrchard::from_cmx(&action.cmx().0),
                    retention,
                )?;
                if decrypted_outgoing_note_idxs.contains(&idx) {
                    // Already decrypted
                    continue;
                }
                let nf = action.nullifier();
                // If the spent note still exists, then the action could not be
                // decrypted.
                if let Some((spent_note, position)) =
                    self.orchard_notes.try_get(&rwtxn, nf)?
                {
                    tracing::warn!(nullifier = ?nf, "Failed to decrypt action spending note");
                    self.orchard_notes.delete(&mut rwtxn, nf)?;
                    self.orchard_spent_notes.put(
                        &mut rwtxn,
                        &(txid, idx as u32),
                        &(spent_note, position),
                    )?;
                }
            }
        }
        let block_hash = header.hash();
        let () = self.put_tip(&mut rwtxn, &block_hash)?;
        let checkpoint_id = orchard::shardtree_db::CheckpointId {
            pos: shard_tree.max_leaf_position(None)?,
            seq: self.orchard_note_commitments.next_checkpoint_seq(&rwtxn)?,
            tip: Some(block_hash),
        };
        let _: bool = shard_tree.checkpoint(checkpoint_id)?;
        rwtxn = self.put_shard_tree(rwtxn, db_txn, shard_tree)?.unwrap();
        Ok(rwtxn)
    }

    /// Disconnects ONLY the orchard effects from a block.
    /// Does not delete memos.
    /// Updates the wallet tip.
    pub fn disconnect_orchard_block<'a>(
        &self,
        mut rwtxn: RwTxn<'a>,
        header: &Header,
        body: &Body,
    ) -> Result<RwTxn<'a>, Error> {
        assert_eq!(self.try_get_tip(&rwtxn)?, Some(header.hash()));
        assert_eq!(body.compute_merkle_root(), header.merkle_root);
        let fvk = self.get_orchard_full_viewing_key(&rwtxn)?;
        let ivks = self.get_orchard_incoming_viewing_keys(&rwtxn)?;
        let ovks = self.get_orchard_outgoing_viewing_keys(&rwtxn)?;
        for tx in body.transactions.iter().rev() {
            let Some(orchard_bundle) = tx.orchard_bundle.as_ref() else {
                continue;
            };
            let txid = tx.txid();
            let decrypted_incoming_notes =
                orchard_bundle.decrypt_outputs_with_keys(&ivks);
            for (_, _, note, _, _) in decrypted_incoming_notes.into_iter().rev()
            {
                let nullifier = note.nullifier(&fvk);
                self.orchard_notes.delete(&mut rwtxn, &nullifier)?;
            }
            let mut decrypted_outgoing_note_idxs = HashSet::new();
            let decrypted_outgoing_notes =
                orchard_bundle.recover_outputs_with_ovks(&ovks);
            for (idx, _, note, _, _memo) in
                decrypted_outgoing_notes.into_iter().rev()
            {
                decrypted_outgoing_note_idxs.insert(idx);
                let nf = *orchard_bundle.actions()[idx].nullifier();
                let Some((_, position)) = self
                    .orchard_spent_notes
                    .try_get(&rwtxn, &(txid, idx as u32))?
                else {
                    tracing::warn!(
                        %txid,
                        %idx,
                        value = %note.value(),
                        "Missing spent note"
                    );
                    continue;
                };
                let _: bool = self
                    .orchard_spent_notes
                    .delete(&mut rwtxn, &(txid, idx as u32))?;
                self.orchard_notes.put(&mut rwtxn, &nf, &(note, position))?;
            }
            for (idx, action) in
                orchard_bundle.actions().iter().enumerate().rev()
            {
                if decrypted_outgoing_note_idxs.contains(&idx) {
                    continue;
                }
                let nf = action.nullifier();
                if let Some((spent_note, position)) =
                    self.orchard_spent_notes
                        .try_get(&rwtxn, &(txid, idx as u32))?
                {
                    self.orchard_spent_notes
                        .delete(&mut rwtxn, &(txid, idx as u32))?;
                    self.orchard_notes.put(
                        &mut rwtxn,
                        nf,
                        &(spent_note, position),
                    )?;
                }
            }
        }
        let prev_tip = header.prev_side_hash;
        if let Some(prev_tip) = prev_tip {
            self.tip.put(&mut rwtxn, &(), &prev_tip)?;
        } else {
            self.tip.delete(&mut rwtxn, &())?;
        };
        let (mut shard_tree, db_txn, rwtxn) =
            self.get_shard_tree(ShardTreeDbTxn::Rw(rwtxn))?;
        let prev_checkpoint_id = self
            .orchard_note_commitments
            .get_checkpoint_id(rwtxn.as_ref(), prev_tip)?;
        if !shard_tree.truncate_to_checkpoint(&prev_checkpoint_id)? {
            return Err(Error::OrchardShardTreeTruncate {
                checkpoint: prev_tip,
            });
        }
        let rwtxn = match rwtxn {
            ShardTreeDbTxn::Ro(_) => panic!("impossible"),
            ShardTreeDbTxn::Rw(rw) => rw,
        };
        let rwtxn = self.put_shard_tree(rwtxn, db_txn, shard_tree)?.unwrap();
        Ok(rwtxn)
    }

    pub fn put_utxos(
        &self,
        rwtxn: &mut RwTxn,
        utxos: &HashMap<OutPoint, Output>,
    ) -> Result<(), Error> {
        for (outpoint, output) in utxos {
            self.utxos.put(rwtxn, outpoint, output)?;
        }
        Ok(())
    }

    pub fn get_balance(&self) -> Result<Balance, Error> {
        let mut balance = Balance::default();
        let rotxn = self.env.read_txn().map_err(EnvError::from)?;
        let () = self
            .utxos
            .iter(&rotxn)?
            .map_err(|err| DbError::from(err).into())
            .for_each(|(_, utxo)| {
                let value = utxo.get_value();
                balance.total_transparent = balance
                    .total_transparent
                    .checked_add(value)
                    .ok_or(AmountOverflowError)?;
                if !utxo.content.is_withdrawal() {
                    balance.available_transparent = balance
                        .available_transparent
                        .checked_add(value)
                        .ok_or(AmountOverflowError)?;
                }
                Ok::<_, Error>(())
            })?;
        let () = self
            .orchard_notes
            .iter(&rotxn)?
            .map_err(|err| DbError::from(err).into())
            .for_each(|(_, (note, _))| {
                let value = note.value();
                balance.total_shielded = balance
                    .total_shielded
                    .checked_add(value)
                    .ok_or(AmountOverflowError)?;
                balance.available_shielded = balance
                    .available_shielded
                    .checked_add(value)
                    .ok_or(AmountOverflowError)?;
                Ok::<_, Error>(())
            })?;
        Ok(balance)
    }

    pub fn get_utxos(
        &self,
        rotxn: &RoTxn,
    ) -> Result<HashMap<OutPoint, Output>, Error> {
        let utxos: HashMap<_, _> = self.utxos.iter(rotxn)?.collect()?;
        Ok(utxos)
    }

    pub fn get_shielded_addresses(
        &self,
        rotxn: &RoTxn,
    ) -> Result<HashSet<orchard::Address>, Error> {
        let addresses: HashSet<_> = self
            .orchard_index_to_address
            .iter(rotxn)?
            .map(|(_, address)| Ok(address))
            .collect()?;
        Ok(addresses)
    }

    pub fn get_transparent_addresses(
        &self,
        rotxn: &RoTxn,
    ) -> Result<HashSet<TransparentAddress>, Error> {
        let addresses: HashSet<_> = self
            .index_to_address
            .iter(rotxn)?
            .map(|(_, address)| Ok(address))
            .collect()?;
        Ok(addresses)
    }

    pub fn try_get_tip(
        &self,
        rotxn: &RoTxn,
    ) -> Result<Option<BlockHash>, Error> {
        let tip = self.tip.try_get(rotxn, &())?;
        Ok(tip)
    }

    pub fn authorize_orchard_bundle(
        &self,
        rotxn: &RoTxn,
        transaction: Transaction<
            orchard::InProgress<orchard::BundleProof, orchard::Unauthorized>,
        >,
    ) -> Result<Transaction, Error> {
        let spending_key = self.get_orchard_spending_key(rotxn)?;
        let spend_auth_key = orchard::SpendAuthorizingKey::from(&spending_key);
        let res = authorization::sign_orchard(&[spend_auth_key], transaction)?;
        Ok(res)
    }

    pub fn authorize(
        &self,
        transaction: Transaction,
    ) -> Result<AuthorizedTransaction, Error> {
        let txn = self.env.read_txn().map_err(EnvError::from)?;
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
            let signing_key = self.get_tx_signing_key(&txn, index)?;
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

    pub fn get_num_addresses(&self) -> Result<u32, Error> {
        let txn = self.env.read_txn().map_err(EnvError::from)?;
        let (last_index, _) = self
            .index_to_address
            .last(&txn)
            .map_err(DbError::from)?
            .unwrap_or((0, [0; 20].into()));
        Ok(last_index)
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
            orchard_address_to_index,
            orchard_index_to_address,
            orchard_memos,
            orchard_note_commitments: _,
            orchard_notes,
            orchard_spent_notes,
            utxos,
            stxos,
            tip,
            _version: _,
        } = self;
        let watchables = [
            seed.watch().clone(),
            address_to_index.watch().clone(),
            index_to_address.watch().clone(),
            orchard_address_to_index.watch().clone(),
            orchard_index_to_address.watch().clone(),
            orchard_memos.watch().clone(),
            orchard_notes.watch().clone(),
            orchard_spent_notes.watch().clone(),
            utxos.watch().clone(),
            stxos.watch().clone(),
            tip.watch().clone(),
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
