use std::net::SocketAddr;

use base64::Engine as _;
use http::HeaderValue;
use jsonrpsee::http_client::{HeaderMap, HttpClient, HttpClientBuilder};

pub use bitcoin;
pub use client::MainClient;
pub use jsonrpsee;

pub mod client;

pub use client::Header;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("jsonrpsee error ({})", .main_addr)]
    Jsonrpsee {
        #[source]
        source: jsonrpsee::core::ClientError,
        main_addr: SocketAddr,
    },
    #[error("header error")]
    InvalidHeaderValue(#[from] http::header::InvalidHeaderValue),
    #[error("bitcoin consensus encode error")]
    BitcoinConsensusEncode(#[from] bitcoin::consensus::encode::Error),
    #[error("hex error")]
    Hex(#[from] hex::FromHexError),
    #[error("no next block for prev_main_hash = {prev_main_hash}")]
    NoNextBlock { prev_main_hash: bitcoin::BlockHash },
    #[error("io error")]
    Io(#[from] bitcoin::io::Error),
}

/// Use the `builder` argument to manually set client options
pub fn client(
    main_addr: SocketAddr,
    builder: Option<HttpClientBuilder>,
    password: &str,
    user: &str,
) -> Result<HttpClient, Error> {
    let mut headers = HeaderMap::new();
    let auth = format!("{user}:{password}");
    let header_value = format!(
        "Basic {}",
        base64::engine::general_purpose::STANDARD.encode(auth)
    );
    let header_value = HeaderValue::from_str(&header_value)?;
    headers.insert("authorization", header_value);
    builder
        .unwrap_or_default()
        .set_headers(headers)
        .build(format!("http://{main_addr}"))
        .map_err(|source| Error::Jsonrpsee { source, main_addr })
}

#[cfg(test)]
mod tests;
