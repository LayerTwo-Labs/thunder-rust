//! Test transfers, both transparent and unshielded

use bip300301_enforcer_integration_tests::{
    integration_test::{
        activate_sidechain, deposit, fund_enforcer, propose_sidechain,
    },
    setup::{
        Mode, Network, PostSetup as EnforcerPostSetup, Sidechain as _,
        setup as setup_enforcer,
    },
    util::{AbortOnDrop, AsyncTrial},
};
use bitcoin::Amount;
use futures::{
    FutureExt as _, StreamExt as _, channel::mpsc, future::BoxFuture,
};
use thunder_orchard_app_rpc_api::RpcClient as _;
use tokio::time::sleep;
use tracing::Instrument as _;

use crate::{
    setup::{Init, PostSetup},
    util::BinPaths,
};

#[derive(Debug)]
struct SidechainNodes {
    /// Sidechain process that will be receiving/sending coins
    alice: PostSetup,
    /// Sidechain process that will be receiving/sending coins
    bob: PostSetup,
}

impl SidechainNodes {
    async fn setup(
        bin_paths: &BinPaths,
        res_tx: mpsc::UnboundedSender<anyhow::Result<()>>,
        enforcer_post_setup: &EnforcerPostSetup,
    ) -> anyhow::Result<Self> {
        // Initialize a single node
        let setup_single = |suffix: &str| {
            PostSetup::setup(
                Init {
                    thunder_orchard_app: bin_paths.thunder_orchard.clone(),
                    data_dir_suffix: Some(suffix.to_owned()),
                    // Orchard transfers can take a while to prove
                    rpc_client_request_timeout: Some(
                        std::time::Duration::from_secs(120),
                    ),
                },
                enforcer_post_setup,
                res_tx.clone(),
            )
        };
        let res = Self {
            alice: setup_single("alice").await?,
            bob: setup_single("bob").await?,
        };
        tracing::debug!(
            alice_addr = %res.alice.net_addr(),
            bob_addr = %res.bob.net_addr(),
            "Connecting alice to bob");
        let () = res
            .alice
            .rpc_client
            .connect_peer(res.bob.net_addr().into())
            .await?;
        Ok(res)
    }
}

const DEPOSIT_AMOUNT_SATS: u64 = 21_000_000;
const DEPOSIT_AMOUNT: Amount = Amount::from_sat(DEPOSIT_AMOUNT_SATS);
const DEPOSIT_FEE: Amount = Amount::from_sat(1_000_000);

/// Initial setup for the test
async fn setup(
    bin_paths: &BinPaths,
    res_tx: mpsc::UnboundedSender<anyhow::Result<()>>,
) -> anyhow::Result<(EnforcerPostSetup, SidechainNodes)> {
    let mut enforcer_post_setup = setup_enforcer(
        &bin_paths.others,
        Network::Regtest,
        Mode::Mempool,
        res_tx.clone(),
    )
    .await?;
    let () = propose_sidechain::<PostSetup>(&mut enforcer_post_setup).await?;
    tracing::info!("Proposed sidechain successfully");
    let () = activate_sidechain::<PostSetup>(&mut enforcer_post_setup).await?;
    tracing::info!("Activated sidechain successfully");
    let () = fund_enforcer::<PostSetup>(&mut enforcer_post_setup).await?;
    let mut sidechain_nodes =
        SidechainNodes::setup(bin_paths, res_tx, &enforcer_post_setup).await?;
    let alice_deposit_address =
        sidechain_nodes.alice.get_deposit_address().await?;
    tracing::info!("Creating a deposit to alice's sidechain address");
    let () = deposit(
        &mut enforcer_post_setup,
        &mut sidechain_nodes.alice,
        &alice_deposit_address,
        DEPOSIT_AMOUNT,
        DEPOSIT_FEE,
    )
    .await?;
    tracing::info!("Deposited to sidechain successfully");
    Ok((enforcer_post_setup, sidechain_nodes))
}

const TRANSPARENT_TRANSFER_AMOUNT_SATS: u64 = DEPOSIT_AMOUNT_SATS / 2;
const TRANSPARENT_TRANSFER_AMOUNT: Amount =
    Amount::from_sat(TRANSPARENT_TRANSFER_AMOUNT_SATS);
const SHIELD_AMOUNT_SATS: u64 = TRANSPARENT_TRANSFER_AMOUNT_SATS / 2;
const SHIELD_AMOUNT: Amount = Amount::from_sat(SHIELD_AMOUNT_SATS);
const SHIELDED_TRANSFER_AMOUNT_SATS: u64 = SHIELD_AMOUNT_SATS / 2;
const SHIELDED_TRANSFER_AMOUNT: Amount =
    Amount::from_sat(SHIELDED_TRANSFER_AMOUNT_SATS);
const SHIELDED_TRANSFER_FEE: Amount =
    Amount::from_sat(SHIELDED_TRANSFER_AMOUNT_SATS / 8);
const UNSHIELD_AMOUNT: Amount =
    Amount::from_sat(SHIELDED_TRANSFER_AMOUNT_SATS / 2);

async fn transfers_task(
    bin_paths: BinPaths,
    res_tx: mpsc::UnboundedSender<anyhow::Result<()>>,
) -> anyhow::Result<()> {
    let (mut enforcer_post_setup, sidechain_nodes) =
        setup(&bin_paths, res_tx.clone()).await?;
    // Check initial balances
    {
        let alice_balance = sidechain_nodes.alice.rpc_client.balance().await?;
        let bob_balance = sidechain_nodes.bob.rpc_client.balance().await?;
        assert_eq!(alice_balance.total_shielded, Amount::ZERO);
        assert_eq!(alice_balance.total_transparent, DEPOSIT_AMOUNT);
        assert_eq!(bob_balance.total(), Amount::ZERO);
    }
    tracing::info!("Transparent transfer (alice -> bob)");
    let _txid = sidechain_nodes
        .alice
        .rpc_client
        .transparent_transfer(
            sidechain_nodes
                .bob
                .rpc_client
                .get_new_transparent_address()
                .await?,
            TRANSPARENT_TRANSFER_AMOUNT.to_sat(),
            0,
        )
        .await?;
    sidechain_nodes
        .alice
        .bmm_single(&mut enforcer_post_setup)
        .await?;
    // Wait for Bob to sync
    sleep(std::time::Duration::from_secs(5)).await;
    anyhow::ensure!(
        sidechain_nodes.alice.rpc_client.getblockcount().await?
            == sidechain_nodes.bob.rpc_client.getblockcount().await?,
    );
    // Check balances
    {
        let alice_balance = sidechain_nodes.alice.rpc_client.balance().await?;
        let bob_balance = sidechain_nodes.bob.rpc_client.balance().await?;
        anyhow::ensure!(
            alice_balance.total_transparent
                == DEPOSIT_AMOUNT - TRANSPARENT_TRANSFER_AMOUNT
        );
        anyhow::ensure!(bob_balance.total_shielded == Amount::ZERO);
        anyhow::ensure!(
            bob_balance.total_transparent == TRANSPARENT_TRANSFER_AMOUNT
        );
    }
    tracing::info!("Shield coins (bob -> bob)");
    let _txid = sidechain_nodes
        .bob
        .rpc_client
        .shield(SHIELD_AMOUNT.to_sat(), 0)
        .await?;
    // Wait for Alice to receive tx
    sleep(std::time::Duration::from_secs(10)).await;
    sidechain_nodes
        .alice
        .bmm_single(&mut enforcer_post_setup)
        .await?;
    // Wait for Bob to sync
    sleep(std::time::Duration::from_secs(5)).await;
    // Check balances
    {
        let bob_balance = sidechain_nodes.bob.rpc_client.balance().await?;
        anyhow::ensure!(bob_balance.total_shielded == SHIELD_AMOUNT);
        anyhow::ensure!(
            bob_balance.total_transparent
                == TRANSPARENT_TRANSFER_AMOUNT - SHIELD_AMOUNT
        );
    }
    tracing::info!("Shielded transfer (bob -> alice)");
    let _txid = sidechain_nodes
        .bob
        .rpc_client
        .shielded_transfer(
            sidechain_nodes
                .alice
                .rpc_client
                .get_new_shielded_address()
                .await?,
            SHIELDED_TRANSFER_AMOUNT.to_sat(),
            SHIELDED_TRANSFER_FEE.to_sat(),
        )
        .await?;
    // Wait for Alice to receive tx
    sleep(std::time::Duration::from_secs(10)).await;
    sidechain_nodes
        .alice
        .bmm_single(&mut enforcer_post_setup)
        .await?;
    // Wait for Bob to sync
    sleep(std::time::Duration::from_secs(5)).await;
    // Check balances
    {
        let alice_balance = sidechain_nodes.alice.rpc_client.balance().await?;
        let bob_balance = sidechain_nodes.bob.rpc_client.balance().await?;
        anyhow::ensure!(
            alice_balance.total_shielded == SHIELDED_TRANSFER_AMOUNT
        );
        anyhow::ensure!(
            alice_balance.total_transparent
                == (DEPOSIT_AMOUNT - TRANSPARENT_TRANSFER_AMOUNT)
                    + SHIELDED_TRANSFER_FEE,
        );
        anyhow::ensure!(
            bob_balance.total_shielded
                == SHIELD_AMOUNT
                    - (SHIELDED_TRANSFER_AMOUNT + SHIELDED_TRANSFER_FEE),
        );
    }
    tracing::info!("Unshield (alice -> alice)");
    let _txid = sidechain_nodes
        .alice
        .rpc_client
        .unshield(UNSHIELD_AMOUNT.to_sat(), 0)
        .await?;
    sidechain_nodes
        .alice
        .bmm_single(&mut enforcer_post_setup)
        .await?;
    // Wait for Alice's wallet to sync
    sleep(std::time::Duration::from_secs(5)).await;
    // Check balances
    {
        let alice_balance = sidechain_nodes.alice.rpc_client.balance().await?;
        anyhow::ensure!(
            alice_balance.total_shielded
                == SHIELDED_TRANSFER_AMOUNT - UNSHIELD_AMOUNT,
        );
        anyhow::ensure!(
            alice_balance.total_transparent
                == (DEPOSIT_AMOUNT - TRANSPARENT_TRANSFER_AMOUNT)
                    + SHIELDED_TRANSFER_FEE
                    + UNSHIELD_AMOUNT,
        );
    }
    // Cleanup
    {
        drop(sidechain_nodes);
        tracing::info!(
            "Removing {}",
            enforcer_post_setup.out_dir.path().display()
        );
        drop(enforcer_post_setup.tasks);
        // Wait for tasks to die
        sleep(std::time::Duration::from_secs(5)).await;
        enforcer_post_setup.out_dir.cleanup()?;
    }
    Ok(())
}

async fn transfers(bin_paths: BinPaths) -> anyhow::Result<()> {
    let (res_tx, mut res_rx) = mpsc::unbounded();
    let _test_task: AbortOnDrop<()> = tokio::task::spawn({
        let res_tx = res_tx.clone();
        async move {
            let res = transfers_task(bin_paths, res_tx.clone()).await;
            let _send_err: Result<(), _> = res_tx.unbounded_send(res);
        }
        .in_current_span()
    })
    .into();
    res_rx.next().await.ok_or_else(|| {
        anyhow::anyhow!("Unexpected end of test task result stream")
    })?
}

pub fn transfers_trial(
    bin_paths: BinPaths,
) -> AsyncTrial<BoxFuture<'static, anyhow::Result<()>>> {
    AsyncTrial::new("transfers", transfers(bin_paths).boxed())
}
