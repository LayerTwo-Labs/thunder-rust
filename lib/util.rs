//! Utility and convenience types and functions

use futures::Stream;
use heed::{Database, DefaultComparator, RoTxn, RwTxn};
use tokio::sync::watch;
use tokio_stream::wrappers::WatchStream;

/// Watchable types
pub mod watchable {
    use futures::Stream;

    /// Trait for types that can be watched.
    /// Streams are used instead of `watch::Receiver<T>` directly,
    /// so that it is possible to combine watch streams to produce a new stream
    pub trait Watchable<T> {
        type WatchStream: Stream<Item = T>;

        fn watch(&self) -> Self::WatchStream;
    }
}

pub use watchable::Watchable;

/// Heed DB augmented with watchable signals
#[derive(Debug)]
pub(crate) struct WatchableDb<KC, DC, C = DefaultComparator> {
    db: Database<KC, DC, C>,
    // TODO: Make this work like futures_signals::signal_map::MutableBTreeMap
    rx: watch::Receiver<()>,
    // TODO: Make this work like futures_signals::signal_map::MutableBTreeMap
    tx: watch::Sender<()>,
}

impl<KC, DC, C> WatchableDb<KC, DC, C> {
    /// Broadcast that the underlying db was modified
    #[inline(always)]
    fn broadcast_modified(&self) {
        self.tx.send_modify(|()| ())
    }

    /// Get a signal that notifies whenever the db changes
    pub fn watch(&self) -> watch::Receiver<()> {
        self.rx.clone()
    }

    pub fn clear(&self, rwtxn: &mut RwTxn<'_>) -> heed::Result<()> {
        self.db.clear(rwtxn).inspect(|()| self.broadcast_modified())
    }

    pub fn delete<'a>(
        &self,
        rwtxn: &mut RwTxn<'_>,
        key: &'a KC::EItem,
    ) -> heed::Result<bool>
    where
        KC: heed::BytesEncode<'a>,
    {
        self.db.delete(rwtxn, key).inspect(|deleted| {
            if *deleted {
                self.broadcast_modified()
            }
        })
    }

    pub fn iter<'rotxn>(
        &self,
        rotxn: &'rotxn RoTxn<'_>,
    ) -> heed::Result<heed::RoIter<'rotxn, KC, DC>> {
        self.db.iter(rotxn)
    }

    pub fn last<'rotxn>(
        &self,
        rotxn: &'rotxn RoTxn<'_>,
    ) -> heed::Result<Option<(KC::DItem, DC::DItem)>>
    where
        KC: heed::BytesDecode<'rotxn>,
        DC: heed::BytesDecode<'rotxn>,
    {
        self.db.last(rotxn)
    }

    /// Retrieves the value associated with a key.
    /// If the key does not exist, then None is returned.
    pub fn try_get<'a, 'rotxn>(
        &self,
        rotxn: &'rotxn RoTxn,
        key: &'a KC::EItem,
    ) -> heed::Result<Option<DC::DItem>>
    where
        KC: heed::BytesEncode<'a>,
        DC: heed::BytesDecode<'rotxn>,
    {
        self.db.get(rotxn, key)
    }

    /// Attempt to insert a key-value pair in this database,
    /// or if a value already exists for the key, returns the previous value.
    #[allow(dead_code)]
    fn get_or_put<'a, 'rwtxn>(
        &'rwtxn self,
        rwtxn: &mut RwTxn<'_>,
        key: &'a KC::EItem,
        data: &'a DC::EItem,
    ) -> heed::Result<Option<DC::DItem>>
    where
        KC: heed::BytesEncode<'a>,
        DC: heed::BytesEncode<'a> + heed::BytesDecode<'a>,
    {
        self.db.get_or_put(rwtxn, key, data).inspect(|res| {
            if res.is_none() {
                self.broadcast_modified()
            }
        })
    }

    /// Insert a key-value pair in this database, replacing any previous value.
    /// The entry is written with no specific flag.
    pub fn put<'a>(
        &self,
        rwtxn: &mut RwTxn<'_>,
        key: &'a KC::EItem,
        data: &'a DC::EItem,
    ) -> heed::Result<()>
    where
        KC: heed::BytesEncode<'a>,
        DC: heed::BytesEncode<'a>,
    {
        self.db
            .put(rwtxn, key, data)
            .inspect(|()| self.broadcast_modified())
    }
}

impl<KC, DC, C> Clone for WatchableDb<KC, DC, C> {
    fn clone(&self) -> Self {
        Self {
            db: self.db,
            rx: self.rx.clone(),
            tx: self.tx.clone(),
        }
    }
}

impl<KC, DC, C> From<Database<KC, DC, C>> for WatchableDb<KC, DC, C> {
    fn from(db: Database<KC, DC, C>) -> Self {
        let (tx, rx) = watch::channel(());
        Self { db, rx, tx }
    }
}

impl<KC, DC, C> Watchable<()> for WatchableDb<KC, DC, C> {
    type WatchStream = impl Stream<Item = ()>;

    fn watch(&self) -> Self::WatchStream {
        WatchStream::new(WatchableDb::watch(self))
    }
}

/// Extension methods for [`Env`]
pub(crate) trait EnvExt {
    /// Create/open a [`MutableDb`]
    fn create_watchable_db<KC, DC>(
        &self,
        rwtxn: &mut RwTxn<'_>,
        name: &str,
    ) -> heed::Result<WatchableDb<KC, DC>>
    where
        KC: 'static,
        DC: 'static;
}

impl EnvExt for heed::Env {
    fn create_watchable_db<KC, DC>(
        &self,
        rwtxn: &mut RwTxn<'_>,
        name: &str,
    ) -> heed::Result<WatchableDb<KC, DC>>
    where
        KC: 'static,
        DC: 'static,
    {
        self.create_database(rwtxn, Some(name))
            .map(WatchableDb::from)
    }
}
