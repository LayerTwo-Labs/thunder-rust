use bip300301_enforcer_integration_tests::{
    setup::{Mode, Network},
    util::{AsyncTrial, TestFailureCollector, TestFileRegistry},
};
use futures::{FutureExt, future::BoxFuture};

use crate::{
    ibd::ibd_trial,
    setup::{Init, PostSetup},
    unknown_withdrawal::unknown_withdrawal_trial,
    util::BinPaths,
};

fn deposit_withdraw_roundtrip(
    bin_paths: BinPaths,
    file_registry: TestFileRegistry,
    failure_collector: TestFailureCollector,
) -> AsyncTrial<BoxFuture<'static, anyhow::Result<()>>> {
    AsyncTrial::new(
        "deposit_withdraw_roundtrip",
        async move {
            let (res_tx, _) = futures::channel::mpsc::unbounded();
            let post_setup = bip300301_enforcer_integration_tests::setup::setup(
                &bin_paths.others,
                Network::Regtest,
                Mode::Mempool,
                res_tx
            ).await?;
            bip300301_enforcer_integration_tests::integration_test::deposit_withdraw_roundtrip::<PostSetup>(
                    post_setup,
                    Init {
                        thunder_app: bin_paths.thunder,
                        data_dir_suffix: None,
                    },
                ).await
        }.boxed(),
        file_registry,
        failure_collector,
    )
}

pub fn tests(
    bin_paths: BinPaths,
    file_registry: TestFileRegistry,
    failure_collector: TestFailureCollector,
) -> Vec<AsyncTrial<BoxFuture<'static, anyhow::Result<()>>>> {
    vec![
        deposit_withdraw_roundtrip(
            bin_paths.clone(),
            file_registry.clone(),
            failure_collector.clone(),
        ),
        ibd_trial(
            bin_paths.clone(),
            file_registry.clone(),
            failure_collector.clone(),
        ),
        unknown_withdrawal_trial(bin_paths, file_registry, failure_collector),
    ]
}
