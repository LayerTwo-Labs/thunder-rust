//! Utility and convenience types and functions

pub mod join_set {
    //! A set of tasks that can spawn new tasks while streaming task results.
    //! Like a [`tokio::task::JoinSet`], that provides seperate channels for
    //! spawning tasks and receiving their results.

    use std::{
        future::Future,
        pin::Pin,
        sync::{Arc, Weak},
        task::{Poll, Waker},
    };

    use parking_lot::Mutex;
    use tokio::task::{AbortHandle, JoinError, JoinSet};

    /// Inner component of [`Spawner`] and [`Receiver`]
    struct Inner<T> {
        join_set: JoinSet<T>,
        /// If the underlying [`JoinSet`] is empty, a waker can be registered
        /// such that the underlying [`JoinSet`] will be polled once a new task
        /// is spawned.
        waker: Option<Waker>,
    }

    /// Used to spawn tasks on a join set.
    /// Tasks will be aborted if the corresponding [`Receiver`] is dropped.
    #[derive(Clone)]
    pub struct Spawner<T> {
        inner: Weak<Mutex<Inner<T>>>,
    }

    impl<T> Spawner<T> {
        /// Spawn a task on the underlying join set.
        /// Returns an [`AbortHandle`] if the task was spawned successfully,
        /// `None` if the receiver has been dropped and the task could not be
        /// spawned.
        pub fn spawn<Fut>(&self, task: Fut) -> Option<AbortHandle>
        where
            Fut: Future<Output = T> + Send + 'static,
            T: Send + 'static,
        {
            self.inner.upgrade().map(|inner| {
                let mut inner_lock = inner.lock();
                let abort_handle = inner_lock.join_set.spawn(task);
                if let Some(waker) = inner_lock.waker.take() {
                    waker.wake();
                }
                abort_handle
            })
        }
    }

    /// Used to receive task results. Must be polled in order to clear
    /// completed tasks from the underlying JoinSet.
    /// Tasks will be aborted if dropped.
    #[must_use]
    pub struct Receiver<T> {
        inner: Arc<Mutex<Inner<T>>>,
    }

    impl<T> futures::Stream for Receiver<T>
    where
        T: 'static,
    {
        type Item = Result<T, JoinError>;

        fn poll_next(
            self: Pin<&mut Self>,
            cx: &mut std::task::Context<'_>,
        ) -> Poll<Option<Self::Item>> {
            let mut inner_lock = self.inner.lock();
            let poll = inner_lock.join_set.poll_join_next(cx);
            match poll {
                Poll::Pending => Poll::Pending,
                Poll::Ready(None) => {
                    inner_lock.waker = Some(cx.waker().clone());
                    Poll::Pending
                }
                Poll::Ready(Some(item)) => Poll::Ready(Some(item)),
            }
        }
    }

    /// Create a new join set.
    pub fn new<T>() -> (Spawner<T>, Receiver<T>) {
        let inner = Arc::new(Mutex::new(Inner {
            join_set: JoinSet::new(),
            waker: None,
        }));
        let spawner = Spawner {
            inner: Arc::downgrade(&inner),
        };
        let receiver = Receiver { inner };
        (spawner, receiver)
    }
}

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

/// Display an error with causes.
/// This is useful for displaying errors without converting to
/// `miette::Report` or `anyhow::Error` first
pub struct ErrorChain<'a>(&'a dyn std::error::Error);

impl<'a> ErrorChain<'a> {
    pub fn new<E>(err: &'a E) -> Self
    where
        E: std::error::Error,
    {
        Self(err)
    }
}

impl std::fmt::Display for ErrorChain<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)?;
        let mut source: Option<&dyn std::error::Error> = self.0.source();
        while let Some(cause) = source {
            std::fmt::Display::fmt(": ", f)?;
            std::fmt::Display::fmt(cause, f)?;
            source = cause.source();
        }
        Ok(())
    }
}
