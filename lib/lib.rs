#![feature(impl_trait_in_assoc_type)]
#![feature(let_chains)]
#![feature(trait_alias)]

pub mod archive;
pub mod authorization;
pub mod mempool;
pub mod miner;
pub mod net;
pub mod node;
pub mod state;
pub mod types;
pub mod util;
pub mod wallet;

pub use heed;

/// Format `str_dest` with the proper `s{sidechain_number}_` prefix and a
/// checksum postfix for calling createsidechaindeposit on mainchain.
pub fn format_deposit_address(this_sidechain: u8, str_dest: &str) -> String {
    let deposit_address: String = format!("s{}_{}_", this_sidechain, str_dest);
    let hash = sha256::digest(deposit_address.as_bytes()).to_string();
    let hash: String = hash[..6].into();
    format!("{}{}", deposit_address, hash)
}

// TODO: Add error log.
