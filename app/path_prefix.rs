//! gRPC transport that keeps the path of the mainchain URL

use std::{
    sync::Arc,
    task::{Context, Poll},
};

/// Prepends a path to every request, so the mainchain node can be served
/// under one, eg. `https://example.com/enforcer`. tonic itself only keeps
/// the URL's scheme and authority.
#[derive(Clone, Debug)]
pub struct PathPrefix<S> {
    inner: S,
    /// Without a trailing slash; `None` for the root
    prefix: Option<Arc<str>>,
}

impl<S> PathPrefix<S> {
    pub fn new(inner: S, prefix: &str) -> Self {
        let prefix = prefix.trim_end_matches('/');
        Self {
            inner,
            prefix: (!prefix.is_empty()).then(|| prefix.into()),
        }
    }
}

impl<S, B> tower::Service<http::Request<B>> for PathPrefix<S>
where
    S: tower::Service<http::Request<B>>,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = S::Future;

    fn poll_ready(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Result<(), S::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, mut request: http::Request<B>) -> Self::Future {
        if let Some(prefix) = &self.prefix {
            let mut parts = request.uri().clone().into_parts();
            let path_and_query =
                parts.path_and_query.as_ref().map_or("/", |pq| pq.as_str());
            parts.path_and_query =
                Some(format!("{prefix}{path_and_query}").parse().expect(
                    "a URL path followed by a gRPC path is a valid path",
                ));
            *request.uri_mut() =
                http::Uri::from_parts(parts).expect("only the path changed");
        }
        self.inner.call(request)
    }
}
