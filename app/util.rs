//! Convenience and utility types and functions

use std::{marker::Unpin, task::Poll};

use futures::{Stream, StreamExt};
use poll_promise::Promise;

type PromiseStreamInner<S> = Promise<Option<(S, <S as Stream>::Item)>>;

pub struct PromiseStream<S>(Option<PromiseStreamInner<S>>)
where
    S: Send + Stream + 'static,
    S::Item: Send;

impl<S> PromiseStream<S>
where
    S: Send + Stream + Unpin + 'static,
    S::Item: Send,
{
    fn promise(mut stream: S) -> Promise<Option<(S, S::Item)>>
    where
        S:,
    {
        Promise::spawn_async(async {
            stream.next().await.map(|item| (stream, item))
        })
    }

    /// Get the next item in the stream if available,
    /// or return None if the stream is complete
    pub fn poll_next(&mut self) -> Option<Poll<S::Item>> {
        let res;
        let inner = std::mem::take(&mut self.0);
        self.0 = match inner {
            Some(promise) => match promise.try_take() {
                Ok(Some((stream, item))) => {
                    res = Some(Poll::Ready(item));
                    Some(Self::promise(stream))
                }
                Ok(None) => {
                    res = None;
                    None
                }
                Err(promise) => {
                    res = Some(Poll::Pending);
                    Some(promise)
                }
            },
            None => {
                res = None;
                None
            }
        };
        res
    }
}

impl<S> From<S> for PromiseStream<S>
where
    S: Send + Stream + Unpin + 'static,
    S::Item: Send,
{
    fn from(stream: S) -> Self {
        Self(Some(Self::promise(stream)))
    }
}

/// Saturating predecessor of a log level
pub fn saturating_pred_level(log_level: tracing::Level) -> tracing::Level {
    match log_level {
        tracing::Level::TRACE => tracing::Level::DEBUG,
        tracing::Level::DEBUG => tracing::Level::INFO,
        tracing::Level::INFO => tracing::Level::WARN,
        tracing::Level::WARN => tracing::Level::ERROR,
        tracing::Level::ERROR => tracing::Level::ERROR,
    }
}
