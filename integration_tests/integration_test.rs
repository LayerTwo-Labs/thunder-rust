use bip300301_enforcer_integration_tests::{
    setup::{Mode, Network},
    util::AsyncTrial,
};
use futures::{future::BoxFuture, FutureExt};

use crate::{ibd::ibd_trial, setup::PostSetup, util::BinPaths};

fn deposit_withdraw_roundtrip(
    bin_paths: BinPaths,
) -> AsyncTrial<BoxFuture<'static, anyhow::Result<()>>> {
    AsyncTrial::new("deposit_withdraw_roundtrip", async move {
        bip300301_enforcer_integration_tests::integration_test::deposit_withdraw_roundtrip::<PostSetup>(
                bin_paths.others, Network::Regtest, Mode::Mempool, bin_paths.thunder
            ).await
    }.boxed())
}

pub fn tests(
    bin_paths: BinPaths,
) -> Vec<AsyncTrial<BoxFuture<'static, anyhow::Result<()>>>> {
    vec![
        deposit_withdraw_roundtrip(bin_paths.clone()),
        ibd_trial(bin_paths),
    ]
}
