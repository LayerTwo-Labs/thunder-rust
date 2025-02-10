//! Utility and convenience types and functions

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
