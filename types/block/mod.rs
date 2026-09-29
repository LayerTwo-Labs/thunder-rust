use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

pub mod body;
pub use body::Body;
pub mod coinbase;
pub use coinbase::Coinbase;
pub mod header;
pub use header::Header;

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct Block {
    pub header: Header,
    pub body: Body,
}
