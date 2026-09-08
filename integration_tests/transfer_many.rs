//! Test that a node pays several addresses with one transfer

use std::collections::BTreeMap;

use bip300301_enforcer_integration_tests::{
    integration_test::{
        activate_sidechain, deposit, fund_enforcer, propose_sidechain,
    },
    setup::{
        Mode, Network, PostSetup as EnforcerPostSetup,
        PreSetup as EnforcerPreSetup, SetupOpts as EnforcerSetupOpts,
        Sidechain as _,
    },
    util::{
        AbortOnDrop, AsyncTrial, BinPaths as EnforcerBinPaths,
        TestFailureCollector, TestFileRegistry,
    },
};
use bitcoin::Amount;
use futures::{
    FutureExt as _, StreamExt as _, channel::mpsc, future::BoxFuture,
};
use thunder::types::{GetValue as _, OutPoint, wallet::TransferDests};
use thunder_app_rpc_api::{node::RpcClient as _, wallet::RpcClient as _};
use tokio::time::sleep;
use tracing::Instrument as _;

use crate::{
    setup::{Init, PostSetup},
    util::BinPaths,
};

const DEPOSIT_AMOUNT: Amount = Amount::from_sat(21_000_000);
const DEPOSIT_FEE: Amount = Amount::from_sat(1_000_000);
const TRANSFER_VALUES: [u64; 3] = [1_000_000, 2_000_000, 3_000_000];
const TRANSFER_FEE: u64 = 1_000;

/// Initial setup for the test
async fn setup(
    enforcer_bin_paths: &EnforcerBinPaths,
    res_tx: mpsc::UnboundedSender<anyhow::Result<()>>,
) -> anyhow::Result<EnforcerPostSetup> {
    let enforcer_pre_setup =
        EnforcerPreSetup::new(enforcer_bin_paths, Network::Regtest)?;
    let mut enforcer_post_setup = {
        let setup_opts: EnforcerSetupOpts = Default::default();
        enforcer_pre_setup
            .setup(Mode::Mempool, setup_opts, res_tx.clone())
            .await?
    };
    let () = propose_sidechain::<PostSetup>(&mut enforcer_post_setup).await?;
    let () = activate_sidechain::<PostSetup>(&mut enforcer_post_setup).await?;
    let () = fund_enforcer::<PostSetup>(&mut enforcer_post_setup).await?;
    Ok(enforcer_post_setup)
}

async fn transfer_many_task(
    bin_paths: BinPaths,
    res_tx: mpsc::UnboundedSender<anyhow::Result<()>>,
) -> anyhow::Result<()> {
    let mut enforcer_post_setup =
        setup(&bin_paths.others, res_tx.clone()).await?;
    let mut sidechain = PostSetup::setup(
        Init {
            thunder_app: bin_paths.thunder()?.clone(),
            data_dir_suffix: None,
        },
        &enforcer_post_setup,
        res_tx,
    )
    .await?;
    tracing::info!("Setup thunder node successfully");

    let deposit_address = sidechain.get_deposit_address().await?;
    let () = deposit(
        &mut enforcer_post_setup,
        &mut sidechain,
        &deposit_address,
        DEPOSIT_AMOUNT,
        DEPOSIT_FEE,
    )
    .await?;
    tracing::info!("Deposited to sidechain successfully");

    let mut dests = BTreeMap::new();
    for value_sats in TRANSFER_VALUES {
        let address = sidechain.rpc_client.get_new_address().await?;
        anyhow::ensure!(dests.insert(address, value_sats).is_none());
    }
    let txid = sidechain
        .rpc_client
        .create_transfer_many(TransferDests(dests.clone()), TRANSFER_FEE)
        .await?;
    tracing::info!(%txid, "Created a transfer to {} addresses", dests.len());

    tracing::debug!("Checking that a block accepts the transfer");
    let () = sidechain.bmm_single(&mut enforcer_post_setup).await?;

    tracing::debug!("Checking that one transaction pays each address");
    let utxos = sidechain.rpc_client.get_wallet_utxos().await?;
    let transfer_txid = |outpoint: OutPoint| match outpoint {
        OutPoint::Regular { txid, .. } => Some(txid),
        _ => None,
    };
    for (address, value_sats) in &dests {
        let utxo = utxos
            .iter()
            .find(|utxo| utxo.output.address == *address)
            .ok_or_else(|| {
                anyhow::anyhow!("no output paid the address `{address}`")
            })?;
        anyhow::ensure!(
            utxo.output.get_value() == Amount::from_sat(*value_sats)
        );
        anyhow::ensure!(transfer_txid(utxo.outpoint) == Some(txid));
    }
    // One output per address, and one more for the change.
    let transfer_outputs = utxos
        .iter()
        .filter(|utxo| transfer_txid(utxo.outpoint) == Some(txid))
        .count();
    anyhow::ensure!(transfer_outputs == dests.len() + 1);

    drop(sidechain);
    tracing::info!(
        "Removing {}",
        enforcer_post_setup.directories.base_dir.path().display()
    );
    drop(enforcer_post_setup.tasks);
    // Wait for tasks to die
    sleep(std::time::Duration::from_secs(1)).await;
    enforcer_post_setup.directories.base_dir.cleanup()?;
    Ok(())
}

async fn transfer_many(bin_paths: BinPaths) -> anyhow::Result<()> {
    let (res_tx, mut res_rx) = mpsc::unbounded();
    let _test_task: AbortOnDrop<()> = tokio::task::spawn({
        let res_tx = res_tx.clone();
        async move {
            let res = transfer_many_task(bin_paths, res_tx.clone()).await;
            let _send_err: Result<(), _> = res_tx.unbounded_send(res);
        }
        .in_current_span()
    })
    .into();
    res_rx.next().await.ok_or_else(|| {
        anyhow::anyhow!("Unexpected end of test task result stream")
    })?
}

pub fn transfer_many_trial(
    bin_paths: BinPaths,
    file_registry: TestFileRegistry,
    failure_collector: TestFailureCollector,
) -> AsyncTrial<BoxFuture<'static, anyhow::Result<()>>> {
    AsyncTrial::new(
        "transfer_many",
        transfer_many(bin_paths).boxed(),
        file_registry,
        failure_collector,
    )
}
