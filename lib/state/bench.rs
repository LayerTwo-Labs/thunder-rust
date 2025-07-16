use std::{collections::HashMap, path::Path};

use bitcoin::Amount;
use criterion::Criterion;
use ed25519_dalek::SigningKey;
use hashlink::{LinkedHashMap, LinkedHashSet};
use rand_chacha::ChaCha20Rng;
use rustreexo::accumulator::node_hash::BitcoinNodeHash;
use sneed::{Env, EnvError};

use crate::{
    authorization::Authorization,
    state::State,
    types::{
        Accumulator, AccumulatorDiff, Address, Block, Body, GetValue as _,
        Header, OutPoint, Output, OutputContent, PointedOutputRef, Transaction,
        hash,
        proto::mainchain::{self, TwoWayPegData},
    },
};

/// Generate an empty genesis block
fn genesis_block<Rng>(rng: &mut Rng) -> Block
where
    Rng: rand::Rng,
{
    use bitcoin::hashes::Hash as _;
    let body = Body {
        coinbase: Vec::new(),
        transactions: Vec::new(),
        authorizations: Vec::new(),
    };
    let header = Header {
        merkle_root: body.compute_merkle_root(),
        prev_side_hash: None,
        prev_main_hash: bitcoin::BlockHash::from_byte_array(rng.r#gen()),
        roots: Vec::new(),
    };
    Block { header, body }
}

/// Generate a signer
fn gen_signer<Rng>(rng: &mut Rng) -> (Address, SigningKey)
where
    Rng: rand::CryptoRng + rand::Rng,
{
    let sk = SigningKey::generate(rng);
    let addr = crate::authorization::get_address(&sk.verifying_key());
    (addr, sk)
}

/// Signer pool that pre-generates a fixed number of signers, and generates
/// more in batches when needed
struct SignerPool {
    rng: ChaCha20Rng,
    batch_size: u32,
    signers: LinkedHashMap<Address, SigningKey>,
    pre_generated: Vec<(Address, SigningKey)>,
}

impl SignerPool {
    /// Generate a batch of keys.
    /// The vector used to store the batch can be re-used to save allocations.
    fn generate_batch(
        rng: &mut ChaCha20Rng,
        batch_size: u32,
        output: &mut Vec<(Address, SigningKey)>,
    ) {
        use rayon::iter::{
            IndexedParallelIterator as _, IntoParallelIterator as _,
            ParallelIterator as _,
        };
        (0..batch_size)
            .into_par_iter()
            .map(|idx| {
                let mut rng = rng.clone();
                rng.set_stream(idx as u64);
                gen_signer(&mut rng)
            })
            .collect_into_vec(output);
        *rng = <ChaCha20Rng as rand::SeedableRng>::from_rng(&mut *rng).unwrap();
    }

    fn new(rng: &mut ChaCha20Rng, batch_size: u32) -> anyhow::Result<Self> {
        let mut rng = <ChaCha20Rng as rand::SeedableRng>::from_rng(rng)?;
        let mut pre_generated = Vec::with_capacity(batch_size as usize);
        Self::generate_batch(&mut rng, batch_size, &mut pre_generated);
        Ok(Self {
            rng,
            batch_size,
            signers: LinkedHashMap::new(),
            pre_generated,
        })
    }

    /// Adds a new signer to the pool, and returns a reference to it
    fn new_signer(&mut self) -> (Address, &SigningKey) {
        let (addr, sk) = if let Some(signer) = self.pre_generated.pop() {
            signer
        } else {
            Self::generate_batch(
                &mut self.rng,
                self.batch_size + 1,
                &mut self.pre_generated,
            );
            self.pre_generated.pop().unwrap()
        };
        let sk = self.signers.entry(addr).or_insert(sk);
        (addr, sk)
    }

    /// Removes a signer from the pool
    fn remove(&mut self, addr: &Address) -> Option<SigningKey> {
        self.signers.remove(addr)
    }
}

/// Generate n initial signers
fn gen_initial_signers(
    rng: &mut ChaCha20Rng,
    n: u32,
) -> anyhow::Result<SignerPool> {
    let mut res = SignerPool::new(rng, n)?;
    for _ in 0..n {
        let _ = res.new_signer();
    }
    Ok(res)
}

/// Generate initial deposits to addresses.
/// Each deposit will have a max amount of `21M / addrs.len()`.
/// If `addrs.len()` exceeds the maximum possible amount of sats, returns an
/// empty hashmap.
fn gen_initial_deposits<Rng>(
    rng: &mut Rng,
    addrs: &LinkedHashSet<Address>,
) -> LinkedHashMap<Address, Amount>
where
    Rng: rand::Rng,
{
    let n_addrs = addrs.len() as u64;
    if n_addrs == 0 || n_addrs > Amount::MAX_MONEY.to_sat() {
        return LinkedHashMap::new();
    }
    let max_sats: u64 = Amount::MAX_MONEY.to_sat() / n_addrs;
    addrs
        .iter()
        .copied()
        .map(|addr| (addr, Amount::from_sat(rng.gen_range(1..=max_sats))))
        .collect()
}

/// Generate 2wpd for initial deposits
fn initial_deposits_2wpd<Rng>(
    rng: &mut Rng,
    deposits: &LinkedHashMap<Address, Amount>,
) -> TwoWayPegData
where
    Rng: rand::Rng,
{
    use bitcoin::hashes::Hash as _;
    let block_hash = bitcoin::BlockHash::from_byte_array(rng.r#gen());
    let txid = bitcoin::Txid::from_byte_array(rng.r#gen());
    let deposits = deposits.iter().enumerate().map(|(idx, (addr, amount))| {
        mainchain::Deposit {
            tx_index: 1,
            outpoint: bitcoin::OutPoint {
                txid,
                vout: idx as u32,
            },
            output: Output {
                address: *addr,
                content: OutputContent::Value(*amount),
            },
        }
    });
    let block_events = deposits.map(mainchain::BlockEvent::Deposit).collect();
    let block_info = mainchain::BlockInfo {
        bmm_commitment: None,
        events: block_events,
    };
    TwoWayPegData {
        block_info: std::iter::once((block_hash, block_info)).collect(),
    }
}

/// Generate outputs such that the total sum value of outputs is equal to the
/// specified value.
/// Coins will be sent to new addresses, which will be added to `signers`.
fn gen_outputs<Rng>(
    rng: &mut Rng,
    signers: &mut SignerPool,
    mut value: Amount,
) -> Vec<Output>
where
    Rng: rand::CryptoRng + rand::Rng,
{
    let mut res = Vec::<Output>::new();
    while value > Amount::ZERO {
        let (receiver_addr, _receiver_sk) = signers.new_signer();
        let amount = if !res.is_empty() && rng.r#gen() {
            // At least 50% chance to add the remaining amount
            value
        } else {
            let max_amount = std::cmp::max(1, value.to_sat() - 1);
            Amount::from_sat(rng.r#gen_range(1..=max_amount))
        };
        value -= amount;
        res.push(Output {
            address: receiver_addr,
            content: OutputContent::Value(amount),
        });
    }
    res
}

/// Utxo set from which UTXOs can be deterministically randomly selected
#[derive(Debug, Default)]
struct UtxoSet<Rng> {
    rng: Rng,
    // Shuffled on every insert
    outpoints: Vec<OutPoint>,
    /// Must contain the same keys as `outpoints`
    utxos: HashMap<OutPoint, Output>,
}

impl<Rng> UtxoSet<Rng>
where
    Rng: rand::Rng,
{
    fn new(rng: &mut Rng) -> anyhow::Result<Self>
    where
        Rng: rand::SeedableRng,
    {
        Ok(Self {
            rng: <Rng as rand::SeedableRng>::from_rng(rng)?,
            outpoints: Vec::new(),
            utxos: HashMap::new(),
        })
    }

    /// Randomly sample a UTXO, removing it from the set
    fn sample_remove(&mut self) -> Option<(OutPoint, Output)> {
        let outpoint = self.outpoints.pop()?;
        let output = self.utxos.remove(&outpoint).unwrap();
        Some((outpoint, output))
    }
}

impl<Rng> std::iter::Extend<(OutPoint, Output)> for UtxoSet<Rng>
where
    Rng: rand::Rng,
{
    fn extend<T>(&mut self, iter: T)
    where
        T: IntoIterator<Item = (OutPoint, Output)>,
    {
        use rand::prelude::SliceRandom as _;
        let orig_len = self.outpoints.len();
        for (outpoint, output) in iter.into_iter() {
            self.utxos.insert(outpoint, output);
            self.outpoints.push(outpoint);
        }
        if self.outpoints.len() != orig_len {
            self.outpoints.shuffle(&mut self.rng);
        }
    }
}

/// Generate a transaction, if utxos are available.
/// If possible, the tx will include at least two inputs, and have at least
/// two outputs.
/// If possible, the tx will have a fee of 1 sat.
/// Spent outputs are removed from the UTXO set.
/// Used signers are removed from the signer pool, and appended to the
/// `used_signers` vec.
/// The newly created outputs are not added to the utxo set or accumulator.
/// Returns an unauthorized tx, the fee, and the used signers.
fn gen_tx<Rng>(
    rng: &mut Rng,
    signers: &mut SignerPool,
    utxo_set: &mut UtxoSet<Rng>,
    accumulator: &Accumulator,
    used_signers: &mut Vec<SigningKey>,
) -> anyhow::Result<Option<(Transaction, bitcoin::Amount)>>
where
    Rng: rand::CryptoRng + rand::Rng,
{
    if utxo_set.utxos.is_empty() {
        return Ok(None);
    }
    let mut res_tx = Transaction::default();
    let mut value_in = Amount::ZERO;
    while let Some((outpoint, output)) = utxo_set.sample_remove() {
        let pointed_output = PointedOutputRef {
            outpoint,
            output: &output,
        };
        let utxo_hash = hash(&pointed_output);
        res_tx.inputs.push((outpoint, utxo_hash));
        value_in += output.get_value();
        used_signers.push(signers.remove(&output.address).unwrap());
        if res_tx.inputs.len() >= 2 && rng.r#gen() {
            break;
        }
    }
    let input_utxo_hashes: Vec<BitcoinNodeHash> =
        res_tx.inputs.iter().map(|(_, hash)| hash.into()).collect();
    res_tx.proof = accumulator.prove(&input_utxo_hashes)?;
    let outputs_value = if value_in > Amount::from_sat(2) {
        value_in - Amount::from_sat(1)
    } else {
        value_in
    };
    res_tx.outputs = gen_outputs(rng, signers, outputs_value);
    Ok(Some((res_tx, value_in - outputs_value)))
}

/// Sign txs in parallel.
/// There must exist as many signers as tx inputs.
fn batch_sign_txs(
    txs: &[Transaction],
    signers: Vec<SigningKey>,
) -> anyhow::Result<Vec<Authorization>> {
    use rayon::iter::{
        IndexedParallelIterator as _, IntoParallelIterator as _,
        ParallelIterator as _,
    };
    let to_sign: Vec<(&Transaction, SigningKey)> = {
        let mut to_sign = Vec::with_capacity(signers.len());
        let mut signers_iter = signers.into_iter();
        for tx in txs {
            for _ in 0..tx.inputs.len() {
                let signer = signers_iter
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("Too few signers"))?;
                to_sign.push((tx, signer))
            }
        }
        to_sign
    };
    let mut res = Vec::with_capacity(to_sign.len());
    to_sign
        .into_par_iter()
        .map(|(tx, sk)| Authorization {
            verifying_key: sk.verifying_key(),
            signature: crate::authorization::sign(&sk, tx).unwrap(),
        })
        .collect_into_vec(&mut res);
    Ok(res)
}

/// Generate a block.
/// If there are no utxos in the utxo set, then the block will contain no
/// txs.
/// If there is only one utxo in the utxo set, then the block will begin with
/// a tx, spending to one address if the utxo is worth exactly 1 sat,
/// at least two addresses otherwise, with a fee of 1 sat if possible.
/// Otherwise, each tx will have at least two inputs, at least two outputs,
/// and a fee of one sat, if the inputs are worth more than the number of
/// outputs, in sats.
/// If there are any fees for the block, a coinbase will be included.
/// The `signers` hashmap must contain keys for every utxo.
fn gen_block<Rng>(
    rng: &mut Rng,
    signers: &mut SignerPool,
    utxo_set: &mut UtxoSet<Rng>,
    accumulator: &mut Accumulator,
    prev_main_hash: bitcoin::BlockHash,
    prev_side_hash: crate::types::BlockHash,
    txs_per_block: u32,
) -> anyhow::Result<Block>
where
    Rng: rand::CryptoRng + rand::Rng,
{
    let mut txs = Vec::new();
    let mut used_signers = Vec::new();
    let mut fees = Amount::ZERO;
    // Generate txs
    {
        for _ in 0..txs_per_block {
            let Some((tx, tx_fee)) =
                gen_tx(rng, signers, utxo_set, accumulator, &mut used_signers)?
            else {
                break;
            };
            txs.push(tx);
            fees += tx_fee;
        }
    }
    let coinbase = gen_outputs(rng, signers, fees);
    let body = Body {
        coinbase,
        authorizations: batch_sign_txs(&txs, used_signers)?,
        transactions: txs,
    };
    let merkle_root = body.compute_merkle_root();
    // update UTXO set and accumulator
    {
        let mut accumulator_diff = AccumulatorDiff::default();
        let mut new_utxos = Vec::new();
        for (vout, output) in body.coinbase.iter().cloned().enumerate() {
            let outpoint = OutPoint::Coinbase {
                merkle_root,
                vout: vout as u32,
            };
            let utxo_hash = PointedOutputRef {
                outpoint,
                output: &output,
            }
            .into();
            accumulator_diff.insert(utxo_hash);
            new_utxos.push((outpoint, output));
        }
        for tx in &body.transactions {
            let txid = tx.txid();
            for (_, utxo_hash) in &tx.inputs {
                accumulator_diff.remove(utxo_hash.into());
            }
            for (vout, output) in tx.outputs.iter().cloned().enumerate() {
                let outpoint = OutPoint::Regular {
                    txid,
                    vout: vout as u32,
                };
                let utxo_hash = PointedOutputRef {
                    outpoint,
                    output: &output,
                }
                .into();
                accumulator_diff.insert(utxo_hash);
                new_utxos.push((outpoint, output));
            }
        }
        let () = accumulator.apply_diff(accumulator_diff)?;
        utxo_set.extend(new_utxos);
    }
    let header = Header {
        merkle_root,
        prev_main_hash,
        prev_side_hash: Some(prev_side_hash),
        roots: accumulator.get_roots(),
    };
    Ok(Block { header, body })
}

struct Setup {
    env: Env,
    state: State,
    rng: ChaCha20Rng,
    signers: SignerPool,
    utxo_set: UtxoSet<ChaCha20Rng>,
    accumulator: Accumulator,
    tip_hash: crate::types::BlockHash,
}

impl Setup {
    fn setup_env(data_dir: &Path) -> anyhow::Result<Env> {
        let env_path = data_dir.join("data.mdb");
        std::fs::create_dir(&env_path)?;
        let env = {
            let mut env_open_opts = heed::EnvOpenOptions::new();
            env_open_opts
                .map_size(128 * 1024 * 1024 * 1024) // 128 GB
                .max_dbs(State::NUM_DBS);
            unsafe { Env::open(&env_open_opts, &env_path) }
                .map_err(EnvError::from)?
        };
        Ok(env)
    }

    fn new(
        rng: &mut ChaCha20Rng,
        n_initial_deposits: u32,
    ) -> anyhow::Result<Self> {
        let mut rng = <ChaCha20Rng as rand::SeedableRng>::from_rng(rng)?;
        let data_dir = temp_dir::TempDir::new()?;
        let env = Self::setup_env(data_dir.as_ref())?;
        let state = State::new(&env)?;
        let initial_signers =
            gen_initial_signers(&mut rng, n_initial_deposits)?;
        let initial_deposits = gen_initial_deposits(
            &mut rng,
            &initial_signers.signers.keys().copied().collect(),
        );
        let initial_deposits_2wpd =
            initial_deposits_2wpd(&mut rng, &initial_deposits);
        let utxo_set = {
            let mut utxo_set = UtxoSet::new(&mut rng)?;
            utxo_set.extend(initial_deposits_2wpd.deposits().flat_map(
                |(_block_hash, deposits)| {
                    deposits.into_iter().map(|deposit| {
                        let outpoint = OutPoint::Deposit(deposit.outpoint);
                        (outpoint, deposit.output.clone())
                    })
                },
            ));
            utxo_set
        };
        let (tip_hash, accumulator) = {
            let genesis_block = genesis_block(&mut rng);
            let mut rwtxn = env.write_txn()?;
            let () = state.connect_block(
                &mut rwtxn,
                &genesis_block.header,
                &genesis_block.body,
            )?;
            let () = state
                .connect_two_way_peg_data(&mut rwtxn, &initial_deposits_2wpd)?;
            let accumulator = state.get_accumulator(&rwtxn)?;
            rwtxn.commit()?;
            (genesis_block.header.hash(), accumulator)
        };
        Ok(Self {
            env,
            state,
            rng,
            signers: initial_signers,
            utxo_set,
            accumulator,
            tip_hash,
        })
    }
}

fn connect_blocks(
    setup: &mut Setup,
    txs_per_block: u32,
    n_blocks: u32,
) -> anyhow::Result<std::time::Duration> {
    let mut res = std::time::Duration::ZERO;
    for _ in 0..n_blocks {
        use bitcoin::hashes::Hash as _;
        use rand::Rng as _;
        let prev_main_hash =
            bitcoin::BlockHash::from_byte_array(setup.rng.r#gen());
        let block = gen_block(
            &mut setup.rng,
            &mut setup.signers,
            &mut setup.utxo_set,
            &mut setup.accumulator,
            prev_main_hash,
            setup.tip_hash,
            txs_per_block,
        )?;
        let start = std::time::Instant::now();
        let mut rwtxn = setup.env.write_txn()?;
        setup
            .state
            .validate_block(&rwtxn, &block.header, &block.body)?;
        setup
            .state
            .connect_block(&mut rwtxn, &block.header, &block.body)?;
        rwtxn.commit()?;
        res += start.elapsed();
        setup.tip_hash = block.header.hash();
    }
    Ok(res)
}

static SEED_PREIMAGE: std::sync::LazyLock<[u8; 32]> =
    std::sync::LazyLock::new(|| {
        use std::str::FromStr;
        const BITCOIN_BLOCK_905787_HASH_HEX: &str =
            "000000000000000000010f7ae70a7ffaf286770e9e70fef83f3f2ce8aea55c00";
        *bitcoin::BlockHash::from_str(BITCOIN_BLOCK_905787_HASH_HEX)
            .unwrap()
            .as_ref()
    });

pub fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("connect_blocks", |b| {
        use rand::SeedableRng as _;
        let mut rng =
            rand_chacha::ChaCha20Rng::from_seed(hash(&*SEED_PREIMAGE));
        b.iter_custom(|iters| {
            let mut res = std::time::Duration::ZERO;
            for _ in 0..iters {
                let mut setup = Setup::new(&mut rng, 400_000).unwrap();
                res += connect_blocks(&mut setup, 100_000, 5).unwrap();
            }
            res
        })
    });
}
