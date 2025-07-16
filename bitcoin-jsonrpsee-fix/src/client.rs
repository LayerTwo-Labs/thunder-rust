use std::{
    collections::{HashMap, HashSet},
    fmt::Debug,
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

use bitcoin::{block, hashes::Hash as _, BlockHash, Txid, Weight, Wtxid};
use educe::Educe;
use hashlink::LinkedHashMap;
use jsonrpsee::proc_macros::rpc;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value as JsonValue;
use serde_with::{serde_as, DeserializeAs, DeserializeFromStr, FromInto, Map, SerializeAs};

/// Wrapper for consensus (de)serializing from hex
#[derive(Debug, Deserialize, Serialize)]
#[repr(transparent)]
#[serde(
    bound(
        deserialize = "T: bitcoin::consensus::Decodable, Case: bitcoin::consensus::serde::hex::Case",
        serialize = "T: bitcoin::consensus::Encodable, Case: bitcoin::consensus::serde::hex::Case",
    ),
    transparent
)]
pub struct ConsensusEncoded<T, Case = bitcoin::consensus::serde::hex::Lower>(
    #[serde(with = "bitcoin::consensus::serde::With::<bitcoin::consensus::serde::Hex<Case>>")] pub T,
    pub PhantomData<Case>,
);

#[derive(DeserializeFromStr)]
#[repr(transparent)]
struct CompactTargetRepr(bitcoin::CompactTarget);

impl std::str::FromStr for CompactTargetRepr {
    type Err = bitcoin::error::UnprefixedHexError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        bitcoin::CompactTarget::from_unprefixed_hex(s).map(Self)
    }
}

impl Serialize for CompactTargetRepr {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        hex::serde::serialize(self.0.to_consensus().to_be_bytes(), serializer)
    }
}

impl From<CompactTargetRepr> for bitcoin::CompactTarget {
    fn from(repr: CompactTargetRepr) -> Self {
        repr.0
    }
}

impl From<bitcoin::CompactTarget> for CompactTargetRepr {
    fn from(target: bitcoin::CompactTarget) -> Self {
        Self(target)
    }
}

#[serde_as]
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Header {
    pub hash: BlockHash,
    pub height: u32,
    pub version: bitcoin::block::Version,
    #[serde(rename = "previousblockhash", default = "BlockHash::all_zeros")]
    pub prev_blockhash: BlockHash,
    #[serde(rename = "merkleroot")]
    pub merkle_root: bitcoin::TxMerkleNode,
    pub time: u32,
    #[serde_as(as = "FromInto<CompactTargetRepr>")]
    pub bits: bitcoin::CompactTarget,
    pub nonce: u32,
}

impl Header {
    /// Computes the target (range [0, T] inclusive) that a blockhash must land in to be valid.
    pub fn target(&self) -> bitcoin::Target {
        self.bits.into()
    }

    /// Returns the total work of the block.
    pub fn work(&self) -> bitcoin::Work {
        self.target().to_work()
    }
}

impl From<Header> for bitcoin::block::Header {
    fn from(header: Header) -> Self {
        Self {
            version: header.version,
            prev_blockhash: header.prev_blockhash,
            merkle_root: header.merkle_root,
            time: header.time,
            bits: header.bits,
            nonce: header.nonce,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize)]
pub struct RawMempoolTxFees {
    pub base: u64,
    pub modified: u64,
    pub ancestor: u64,
    pub descendant: u64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct RawMempoolTxInfo {
    pub vsize: u64,
    pub weight: u64,
    #[serde(rename = "descendantcount")]
    pub descendant_count: u64,
    #[serde(rename = "descendantsize")]
    pub descendant_size: u64,
    #[serde(rename = "ancestorcount")]
    pub ancestor_count: u64,
    #[serde(rename = "ancestorsize")]
    pub ancestor_size: u64,
    pub wtxid: Wtxid,
    pub fees: RawMempoolTxFees,
    pub depends: Vec<Txid>,
    #[serde(rename = "spentby")]
    pub spent_by: Vec<Txid>,
    #[serde(rename = "bip125replaceable")]
    pub bip125_replaceable: bool,
    pub unbroadcast: bool,
}

#[derive(Clone, Debug, Deserialize)]
pub struct RawMempoolWithSequence {
    pub txids: Vec<Txid>,
    pub mempool_sequence: u64,
}

#[serde_as]
#[derive(Clone, Debug, Deserialize)]
pub struct RawMempoolVerbose {
    #[serde_as(as = "Map<_, _>")]
    pub entries: Vec<(Txid, RawMempoolTxInfo)>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TxOutSetInfo {
    pub height: u32,
    #[serde(rename = "bestblock")]
    pub best_block: BlockHash,
    #[serde(rename = "transactions")]
    pub n_txs: u64,
    #[serde(rename = "txouts")]
    pub n_txouts: u64,
    #[serde(with = "hex::serde")]
    pub hash_serialized_3: [u8; 32],
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Vote {
    Upvote,
    Abstain,
    Downvote,
}

#[derive(Clone, Debug, Deserialize)]
pub struct NetworkInfo {
    // Time offset in seconds
    #[serde(rename = "timeoffset")]
    pub time_offset_s: i64,
}

/// Output from `getrawtransaction` where `verbosity = 1`
#[derive(Clone, Debug, Deserialize)]
pub struct TxInfo {
    #[serde(deserialize_with = "hex::serde::deserialize")]
    pub hex: Vec<u8>,
    pub txid: Txid,
    // TODO: add more fields
}

mod private {
    pub trait Sealed {}
}

impl<const BOOL: bool> private::Sealed for BoolWitness<BOOL> {}

pub trait ShowTxDetails: private::Sealed {
    type Output;
}

impl ShowTxDetails for BoolWitness<false> {
    type Output = Txid;
}

impl ShowTxDetails for BoolWitness<true> {
    type Output = TxInfo;
}

#[serde_as]
#[derive(Educe)]
#[educe(
    Clone(bound(<BoolWitness<SHOW_TX_DETAILS> as ShowTxDetails>::Output: Clone)),
    Debug(bound(<BoolWitness<SHOW_TX_DETAILS> as ShowTxDetails>::Output: Debug)),
)]
#[derive(Deserialize, Serialize)]
#[serde(
    bound(
        deserialize = "for<'des> <BoolWitness<SHOW_TX_DETAILS> as ShowTxDetails>::Output: Deserialize<'des>",
        serialize = "<BoolWitness<SHOW_TX_DETAILS> as ShowTxDetails>::Output: Serialize"
    ),
    rename_all = "camelCase"
)]
pub struct Block<const SHOW_TX_DETAILS: bool>
where
    BoolWitness<SHOW_TX_DETAILS>: ShowTxDetails,
{
    pub hash: bitcoin::BlockHash,
    pub confirmations: isize, // Confirmations can be negative if block are reorged/invalidated
    pub strippedsize: usize,
    pub size: usize,
    pub weight: usize,
    pub height: u32,
    pub version: bitcoin::block::Version,
    pub version_hex: String,
    pub merkleroot: bitcoin::hash_types::TxMerkleNode,
    pub tx: Vec<<BoolWitness<SHOW_TX_DETAILS> as ShowTxDetails>::Output>,
    pub time: u32,
    pub mediantime: u32,
    pub nonce: u32,
    #[serde(rename = "bits")]
    #[serde_as(as = "FromInto<CompactTargetRepr>")]
    pub compact_target: bitcoin::CompactTarget,
    pub difficulty: f64,
    pub chainwork: String,
    pub previousblockhash: Option<bitcoin::BlockHash>,
    pub nextblockhash: Option<bitcoin::BlockHash>,
}

impl TryFrom<&Block<true>> for bitcoin::Block {
    type Error = bitcoin::consensus::encode::Error;

    fn try_from(block: &Block<true>) -> Result<Self, Self::Error> {
        let header = bitcoin::block::Header {
            version: block.version,
            prev_blockhash: block.previousblockhash.unwrap_or_else(BlockHash::all_zeros),
            merkle_root: block.merkleroot,
            time: block.time,
            bits: block.compact_target,
            nonce: block.nonce,
        };
        let txdata = block
            .tx
            .iter()
            .map(|tx_info| bitcoin::consensus::deserialize(&tx_info.hex))
            .collect::<Result<_, _>>()?;
        Ok(Self { header, txdata })
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct BlockTemplateRequest {
    #[serde(default)]
    pub rules: Vec<String>,
    #[serde(default)]
    pub capabilities: HashSet<String>,
}

impl Default for BlockTemplateRequest {
    fn default() -> Self {
        Self {
            rules: vec!["segwit".into()],
            capabilities: HashSet::new(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct BlockTemplateTransaction {
    #[serde(with = "hex::serde")]
    pub data: Vec<u8>,
    pub txid: Txid,
    // TODO: check that this is the wtxid
    pub hash: Wtxid,
    pub depends: Vec<u32>,
    #[serde(with = "bitcoin::amount::serde::as_sat")]
    pub fee: bitcoin::SignedAmount,
    pub sigops: Option<u64>,
    pub weight: u64,
}

/// Representation used with serde_with
#[derive(Clone, Copy, Debug, Default)]
struct LinkedHashMapRepr<K, V>(PhantomData<(K, V)>);

impl<'de, K0, K1, V0, V1> DeserializeAs<'de, LinkedHashMap<K1, V1>> for LinkedHashMapRepr<K0, V0>
where
    K0: DeserializeAs<'de, K1>,
    K1: Eq + std::hash::Hash,
    V0: DeserializeAs<'de, V1>,
{
    fn deserialize_as<D>(deserializer: D) -> Result<LinkedHashMap<K1, V1>, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        <serde_with::Map<K0, V0> as DeserializeAs<'de, Vec<(K1, V1)>>>::deserialize_as(deserializer)
            .map(LinkedHashMap::from_iter)
    }
}

impl<K0, K1, V0, V1> SerializeAs<LinkedHashMap<K1, V1>> for LinkedHashMapRepr<K0, V0>
where
    K0: SerializeAs<K1>,
    V0: SerializeAs<V1>,
{
    fn serialize_as<S>(source: &LinkedHashMap<K1, V1>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        <serde_with::Map<&K0, &V0> as SerializeAs<Vec<(&K1, &V1)>>>::serialize_as(
            &Vec::from_iter(source),
            serializer,
        )
    }
}

/// `coinbasetxn` or `coinbasevalue` field
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum CoinbaseTxnOrValue {
    #[serde(rename = "coinbasetxn")]
    Txn(BlockTemplateTransaction),
    #[serde(rename = "coinbasevalue")]
    ValueSats(u64),
}

#[serde_as]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct BlockTemplate {
    pub version: block::Version,
    pub rules: Vec<String>,
    #[serde(rename = "vbavailable")]
    pub version_bits_available: LinkedHashMap<String, JsonValue>,
    #[serde(rename = "vbrequired")]
    pub version_bits_required: block::Version,
    #[serde(rename = "previousblockhash")]
    pub prev_blockhash: bitcoin::BlockHash,
    pub transactions: Vec<BlockTemplateTransaction>,
    #[serde(rename = "coinbaseaux")]
    #[serde_as(as = "LinkedHashMapRepr<_, serde_with::hex::Hex>")]
    pub coinbase_aux: LinkedHashMap<String, Vec<u8>>,
    #[serde(flatten)]
    pub coinbase_txn_or_value: CoinbaseTxnOrValue,
    /// MUST be omitted if the server does not support long polling
    #[serde(rename = "longpollid")]
    pub long_poll_id: Option<String>,
    #[serde_as(as = "serde_with::hex::Hex")]
    pub target: [u8; 32],
    pub mintime: u64,
    pub mutable: Vec<String>,
    #[serde(rename = "noncerange")]
    #[serde_as(as = "serde_with::hex::Hex")]
    pub nonce_range: [u8; 8],
    #[serde(rename = "sigoplimit")]
    pub sigop_limit: u64,
    #[serde(rename = "sizelimit")]
    pub size_limit: u64,
    #[serde(rename = "weightlimit")]
    pub weight_limit: Weight,
    #[serde(rename = "curtime")]
    pub current_time: u64,
    #[serde(rename = "bits")]
    #[serde_as(as = "FromInto<CompactTargetRepr>")]
    pub compact_target: bitcoin::CompactTarget,
    pub height: u32,
    pub signet_challenge: Option<bitcoin::ScriptBuf>,
    #[serde_as(as = "Option<serde_with::hex::Hex>")]
    pub default_witness_commitment: Option<Vec<u8>>,
}

#[derive(Debug, Deserialize)]
pub struct AddressInfo {
    pub address: bitcoin::Address<bitcoin::address::NetworkUnchecked>,
    #[serde(rename = "scriptPubKey")]
    pub script_pub_key: String,
    #[serde(rename = "ismine")]
    pub is_mine: bool,
    #[serde(rename = "iswatchonly")]
    pub is_watch_only: bool,
    #[serde(rename = "isscript")]
    pub is_script: bool,
    #[serde(rename = "iswitness")]
    pub is_witness: bool,
    #[serde(rename = "hdkeypath")]
    pub hd_key_path: Option<String>,
    #[serde(rename = "hdseedid")]
    pub hd_seed_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct BlockchainInfo {
    #[serde(with = "bitcoin::network::as_core_arg")]
    pub chain: bitcoin::Network,
    pub blocks: u32,
    #[serde(rename = "bestblockhash")]
    pub best_blockhash: bitcoin::BlockHash,
    pub difficulty: f64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct IndexInfo {
    pub synced: bool,
    pub best_block_height: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ZMQNotification {
    #[serde(rename = "type")]
    pub notification_type: String,
    pub address: String,
    #[serde(rename = "hwm")]
    pub high_water_mark: u32,
}

#[rpc(client)]
pub trait Main {
    #[method(name = "generate")]
    async fn generate(&self, num: u32) -> Result<serde_json::Value, jsonrpsee::core::Error>;

    #[method(name = "generatetoaddress")]
    async fn generate_to_address(
        &self,
        n_blocks: u32,
        address: &bitcoin::Address<bitcoin::address::NetworkUnchecked>,
    ) -> Result<Vec<BlockHash>, jsonrpsee::core::Error>;

    #[method(name = "getblocktemplate")]
    async fn get_block_template(
        &self,
        block_template_request: BlockTemplateRequest,
    ) -> Result<BlockTemplate, jsonrpsee::core::Error>;

    #[method(name = "getblockchaininfo")]
    async fn get_blockchain_info(&self) -> Result<BlockchainInfo, jsonrpsee::core::Error>;

    #[method(name = "getmempoolentry")]
    async fn get_mempool_entry(
        &self,
        txid: Txid,
    ) -> Result<RawMempoolTxInfo, jsonrpsee::core::Error>;

    #[method(name = "getnetworkinfo")]
    async fn get_network_info(&self) -> jsonrpsee::core::RpcResult<NetworkInfo>;

    #[method(name = "getbestblockhash")]
    async fn getbestblockhash(&self) -> Result<bitcoin::BlockHash, jsonrpsee::core::Error>;

    #[method(name = "getblockhash")]
    async fn getblockhash(
        &self,
        height: usize,
    ) -> Result<bitcoin::BlockHash, jsonrpsee::core::Error>;

    #[method(name = "getblockcount")]
    async fn getblockcount(&self) -> Result<usize, jsonrpsee::core::Error>;

    #[method(name = "getblockheader")]
    async fn getblockheader(
        &self,
        block_hash: bitcoin::BlockHash,
    ) -> Result<Header, jsonrpsee::core::Error>;

    #[method(name = "getaddressinfo")]
    async fn get_address_info(
        &self,
        address: &bitcoin::Address<bitcoin::address::NetworkUnchecked>,
    ) -> Result<AddressInfo, jsonrpsee::core::Error>;

    #[method(name = "getnewaddress")]
    async fn getnewaddress(
        &self,
        account: &str,
        address_type: &str,
    ) -> Result<bitcoin::Address<bitcoin::address::NetworkUnchecked>, jsonrpsee::core::Error>;

    #[method(name = "getindexinfo")]
    async fn get_index_info(&self) -> Result<HashMap<String, IndexInfo>, jsonrpsee::core::Error>;

    #[method(name = "gettxoutsetinfo")]
    async fn gettxoutsetinfo(&self) -> Result<TxOutSetInfo, jsonrpsee::core::Error>;

    #[method(name = "invalidateblock")]
    async fn invalidate_block(
        &self,
        block_hash: bitcoin::BlockHash,
    ) -> Result<(), jsonrpsee::core::Error>;

    #[method(name = "prioritisetransaction", param_kind = map)]
    async fn prioritize_transaction(
        &self,
        txid: Txid,
        fee_delta: i64,
    ) -> Result<bool, jsonrpsee::core::Error>;

    // Max fee rate: BTC/kvB value
    // Max burn amount: BTC value
    #[method(name = "sendrawtransaction")]
    async fn send_raw_transaction(
        &self,
        tx_hex: String,
        max_fee_rate: Option<f64>,
        max_burn_amount: Option<f64>,
    ) -> Result<bitcoin::Txid, jsonrpsee::core::Error>;

    #[method(name = "stop")]
    async fn stop(&self) -> Result<String, jsonrpsee::core::Error>;

    #[method(name = "submitblock")]
    async fn submit_block(&self, block_hex: String) -> Result<(), jsonrpsee::core::Error>;

    #[method(name = "getzmqnotifications")]
    async fn get_zmq_notifications(&self) -> Result<Vec<ZMQNotification>, jsonrpsee::core::error>;
}

pub struct U8Witness<const U8: u8>;

impl<const U8: u8> Serialize for U8Witness<{ U8 }> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        U8.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for U8Witness<0> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Debug, Deserialize)]
        struct Repr(monostate::MustBe!(0));
        let _ = Repr::deserialize(deserializer)?;
        Ok(Self)
    }
}

impl<'de> Deserialize<'de> for U8Witness<1> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Debug, Deserialize)]
        struct Repr(monostate::MustBe!(1));
        let _ = Repr::deserialize(deserializer)?;
        Ok(Self)
    }
}

impl<'de> Deserialize<'de> for U8Witness<2> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Debug, Deserialize)]
        struct Repr(monostate::MustBe!(2));
        let _ = Repr::deserialize(deserializer)?;
        Ok(Self)
    }
}

pub trait GetBlockVerbosity {
    type Response: DeserializeOwned;
}

impl GetBlockVerbosity for U8Witness<0> {
    type Response = ConsensusEncoded<bitcoin::Block>;
}

impl GetBlockVerbosity for U8Witness<1> {
    type Response = Block<false>;
}

impl GetBlockVerbosity for U8Witness<2> {
    type Response = Block<true>;
}

#[rpc(
    client,
    client_bounds(Verbosity: Serialize + Send + Sync + 'static)
)]
pub trait GetBlock<Verbosity>
where
    Verbosity: GetBlockVerbosity,
{
    #[method(name = "getblock")]
    async fn get_block(
        &self,
        block_hash: BlockHash,
        verbosity: Verbosity,
    ) -> Result<<Verbosity as GetBlockVerbosity>::Response, jsonrpsee::core::Error>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoolWitness<const BOOL: bool>;

impl<const BOOL: bool> Serialize for BoolWitness<{ BOOL }> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        BOOL.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for BoolWitness<false> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Debug, Deserialize)]
        struct Repr(monostate::MustBe!(false));
        let _ = Repr::deserialize(deserializer)?;
        Ok(Self)
    }
}

impl<'de> Deserialize<'de> for BoolWitness<true> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Debug, Deserialize)]
        struct Repr(monostate::MustBe!(true));
        let _ = Repr::deserialize(deserializer)?;
        Ok(Self)
    }
}

pub struct GetRawMempoolParams<Verbose, MempoolSequence>(PhantomData<(Verbose, MempoolSequence)>);

pub trait GetRawMempoolResponse {
    type Response: DeserializeOwned;
}

impl GetRawMempoolResponse for GetRawMempoolParams<BoolWitness<false>, BoolWitness<false>> {
    type Response = Vec<Txid>;
}

impl GetRawMempoolResponse for GetRawMempoolParams<BoolWitness<false>, BoolWitness<true>> {
    type Response = RawMempoolWithSequence;
}

impl GetRawMempoolResponse for GetRawMempoolParams<BoolWitness<true>, BoolWitness<false>> {
    type Response = RawMempoolVerbose;
}

#[rpc(
    client,
    client_bounds(
        Verbose: Serialize + Send + Sync + 'static,
        MempoolSequence: Serialize + Send + Sync + 'static,
        GetRawMempoolParams<Verbose, MempoolSequence>: GetRawMempoolResponse
    )
)]
pub trait GetRawMempool<Verbose, MempoolSequence>
where
    GetRawMempoolParams<Verbose, MempoolSequence>: GetRawMempoolResponse,
{
    #[method(name = "getrawmempool")]
    async fn get_raw_mempool(
        &self,
        verbose: Verbose,
        mempool_sequence: MempoolSequence,
    ) -> Result<
        <GetRawMempoolParams<Verbose, MempoolSequence> as GetRawMempoolResponse>::Response,
        jsonrpsee::core::Error,
    >;
}

pub trait GetRawTransactionVerbosity {
    type Response: DeserializeOwned;
}

#[derive(Debug)]
pub struct GetRawTransactionVerbose<const VERBOSE: bool>;

impl<const VERBOSE: bool> Serialize for GetRawTransactionVerbose<{ VERBOSE }> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        VERBOSE.serialize(serializer)
    }
}

impl GetRawTransactionVerbosity for GetRawTransactionVerbose<false> {
    type Response = String;
}

impl<'de> Deserialize<'de> for GetRawTransactionVerbose<false> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Debug, Deserialize)]
        struct Repr(monostate::MustBe!(false));
        let _ = Repr::deserialize(deserializer)?;
        Ok(Self)
    }
}

impl GetRawTransactionVerbosity for GetRawTransactionVerbose<true> {
    type Response = serde_json::Value;
}

impl<'de> Deserialize<'de> for GetRawTransactionVerbose<true> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Debug, Deserialize)]
        struct Repr(monostate::MustBe!(true));
        let _ = Repr::deserialize(deserializer)?;
        Ok(Self)
    }
}

#[rpc(client)]
pub trait GetRawTransaction<T>
where
    T: GetRawTransactionVerbosity,
{
    #[method(name = "getrawtransaction")]
    async fn get_raw_transaction(
        &self,
        txid: Txid,
        verbose: T,
        block_hash: Option<bitcoin::BlockHash>,
    ) -> Result<<T as GetRawTransactionVerbosity>::Response, jsonrpsee::core::Error>;
}

// FIXME: Make mainchain API machine friendly. Parsing human readable amounts
// here is stupid -- just take and return values in satoshi.
#[derive(Clone, Copy, Deserialize, Serialize)]
pub struct AmountBtc(#[serde(with = "bitcoin::amount::serde::as_btc")] pub bitcoin::Amount);

impl From<bitcoin::Amount> for AmountBtc {
    fn from(other: bitcoin::Amount) -> AmountBtc {
        AmountBtc(other)
    }
}

impl From<AmountBtc> for bitcoin::Amount {
    fn from(other: AmountBtc) -> bitcoin::Amount {
        other.0
    }
}

impl Deref for AmountBtc {
    type Target = bitcoin::Amount;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for AmountBtc {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
