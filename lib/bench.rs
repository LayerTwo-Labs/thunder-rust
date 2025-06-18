#![feature(impl_trait_in_assoc_type)]
#![feature(trait_alias)]

use criterion::{Criterion, criterion_group, criterion_main};

#[allow(dead_code, unused_imports)]
mod authorization;
#[allow(dead_code)]
mod state;
#[allow(dead_code)]
mod types;
#[allow(dead_code)]
mod util;

criterion_group! {
    name=benches;
    config = Criterion::default().sample_size(10);
    targets = state::bench::criterion_benchmark
}
criterion_main!(benches);
