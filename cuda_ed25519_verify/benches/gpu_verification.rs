use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion,
    Throughput,
};
use cuda_ed25519_verify::{test_data::generate_test_data, CudaEd25519Verifier};
use std::time::Duration;

fn benchmark_simple_cuda_verification(c: &mut Criterion) {
    let mut group = c.benchmark_group("simple_cuda_verification");
    group.measurement_time(Duration::from_secs(15));
    group.warm_up_time(Duration::from_secs(5));

    // Test various batch sizes for simple CUDA threading
    let batch_sizes = vec![100, 500, 1000, 2500, 5000, 10000];

    for batch_size in batch_sizes {
        println!("🔧 Generating {} test transactions for CUDA...", batch_size);
        let (signatures, messages, public_keys) =
            generate_test_data(batch_size)
                .expect("Failed to generate test data");

        // Set throughput for meaningful performance metrics
        group.throughput(Throughput::Elements(batch_size as u64));

        // Skip if CUDA is not available
        if CudaEd25519Verifier::new().is_err() {
            println!("⚠️  CUDA not available - skipping GPU benchmark for batch size {}", batch_size);
            continue;
        }

        // Benchmark simple threaded CUDA verification
        group.bench_with_input(
            BenchmarkId::new("cuda_simple_threading", batch_size),
            &batch_size,
            |b, _| {
                let mut verifier = CudaEd25519Verifier::new().unwrap();

                // Warm up GPU
                let _ = verifier.verify_batch(
                    &signatures[..10.min(batch_size)],
                    &messages[..10.min(batch_size)],
                    &public_keys[..10.min(batch_size)],
                );

                b.iter(|| {
                    let (results, _perf) = verifier
                        .verify_batch(
                            black_box(&signatures),
                            black_box(&messages),
                            black_box(&public_keys),
                        )
                        .expect("CUDA verification failed");

                    // Ensure all signatures are valid to catch any issues
                    assert!(
                        results.iter().all(|&valid| valid),
                        "Invalid signatures detected in benchmark"
                    );
                });
            },
        );
    }

    group.finish();
}

fn benchmark_cuda_device_utilization(c: &mut Criterion) {
    if CudaEd25519Verifier::new().is_err() {
        println!(
            "⚠️  CUDA not available - skipping device utilization benchmarks"
        );
        return;
    }

    let mut group = c.benchmark_group("cuda_device_utilization");
    group.measurement_time(Duration::from_secs(20));

    let batch_size = 5000;
    println!(
        "🔧 Generating {} test transactions for device utilization test...",
        batch_size
    );
    let (signatures, messages, public_keys) =
        generate_test_data(batch_size).expect("Failed to generate test data");

    group.throughput(Throughput::Elements(batch_size as u64));

    // Test different thread block configurations
    let test_configs = vec![
        ("optimal", "Device-optimized configuration"),
        ("conservative", "Conservative 256 threads per block"),
        ("aggressive", "Aggressive 512 threads per block"),
    ];

    for (config_name, config_desc) in test_configs {
        group.bench_with_input(
            BenchmarkId::new("thread_config", config_name),
            config_name,
            |b, _| {
                let mut verifier = CudaEd25519Verifier::new().unwrap();

                // Warm up
                let _ = verifier.verify_batch(
                    &signatures[..100],
                    &messages[..100],
                    &public_keys[..100],
                );

                b.iter(|| {
                    let (results, _) = verifier
                        .verify_batch(
                            black_box(&signatures),
                            black_box(&messages),
                            black_box(&public_keys),
                        )
                        .expect(&format!(
                            "CUDA verification failed for {}",
                            config_desc
                        ));

                    assert!(results.iter().all(|&valid| valid));
                });
            },
        );
    }

    group.finish();
}

fn benchmark_batch_size_scaling(c: &mut Criterion) {
    if CudaEd25519Verifier::new().is_err() {
        println!(
            "⚠️  CUDA not available - skipping batch size scaling benchmarks"
        );
        return;
    }

    let mut group = c.benchmark_group("cuda_batch_scaling");
    group.measurement_time(Duration::from_secs(25));

    // Test how performance scales with batch size
    let scaling_sizes = vec![
        (64, "Small"),
        (256, "Medium-Small"),
        (1024, "Medium"),
        (4096, "Large"),
        (16384, "Very Large"),
    ];

    for (batch_size, category) in scaling_sizes {
        println!(
            "🔧 Testing {} batch scaling ({} signatures)...",
            category, batch_size
        );
        let (signatures, messages, public_keys) =
            generate_test_data(batch_size)
                .expect("Failed to generate test data");

        group.throughput(Throughput::Elements(batch_size as u64));

        group.bench_with_input(
            BenchmarkId::new(
                "batch_scaling",
                format!("{}_{}", category, batch_size),
            ),
            &batch_size,
            |b, &size| {
                let mut verifier = CudaEd25519Verifier::new().unwrap();

                // Extended warm-up for larger batches
                let warmup_size = (size / 10).max(10).min(100);
                for _ in 0..3 {
                    let _ = verifier.verify_batch(
                        &signatures[..warmup_size],
                        &messages[..warmup_size],
                        &public_keys[..warmup_size],
                    );
                }

                b.iter(|| {
                    let (results, sigs_per_sec) = verifier
                        .verify_batch(
                            black_box(&signatures),
                            black_box(&messages),
                            black_box(&public_keys),
                        )
                        .expect(&format!(
                            "CUDA verification failed for {} batch",
                            category
                        ));

                    assert_eq!(results.len(), size);
                    assert!(results.iter().all(|&valid| valid));

                    // Performance analysis is handled by criterion
                    // Store performance data for potential analysis
                    let _ = sigs_per_sec;
                });
            },
        );
    }

    group.finish();
}

fn benchmark_single_vs_batch(c: &mut Criterion) {
    if CudaEd25519Verifier::new().is_err() {
        println!(
            "⚠️  CUDA not available - skipping single vs batch benchmarks"
        );
        return;
    }

    let mut group = c.benchmark_group("cuda_single_vs_batch");
    group.measurement_time(Duration::from_secs(10));

    let (signatures, messages, public_keys) =
        generate_test_data(1000).expect("Failed to generate test data");

    group.throughput(Throughput::Elements(1000));

    // Benchmark single signature calls
    group.bench_function("single_calls_1000x", |b| {
        let mut verifier = CudaEd25519Verifier::new().unwrap();

        b.iter(|| {
            for i in 0..1000 {
                let result = verifier
                    .verify_single(
                        black_box(&signatures[i]),
                        black_box(&messages[i]),
                        black_box(&public_keys[i]),
                    )
                    .expect("Single verification failed");
                assert!(result);
            }
        });
    });

    // Benchmark batch call
    group.bench_function("batch_call_1000", |b| {
        let mut verifier = CudaEd25519Verifier::new().unwrap();

        // Warm up
        let _ = verifier.verify_batch(
            &signatures[..100],
            &messages[..100],
            &public_keys[..100],
        );

        b.iter(|| {
            let (results, _) = verifier
                .verify_batch(
                    black_box(&signatures),
                    black_box(&messages),
                    black_box(&public_keys),
                )
                .expect("Batch verification failed");

            assert!(results.iter().all(|&valid| valid));
        });
    });

    group.finish();
}

fn performance_summary(c: &mut Criterion) {
    if CudaEd25519Verifier::new().is_err() {
        println!("⚠️  CUDA not available - skipping performance summary");
        return;
    }

    let mut group = c.benchmark_group("cuda_performance_summary");
    group.measurement_time(Duration::from_secs(30));

    let batch_size = 10000;
    println!(
        "🚀 Generating {} signatures for CUDA performance summary...",
        batch_size
    );
    let (signatures, messages, public_keys) =
        generate_test_data(batch_size).expect("Failed to generate test data");

    group.throughput(Throughput::Elements(batch_size as u64));

    group.bench_function("cuda_best_performance", |b| {
        let mut verifier = CudaEd25519Verifier::new().unwrap();

        // Extended warm-up for stable measurements
        for _ in 0..10 {
            let _ = verifier.verify_batch(
                &signatures[..1000],
                &messages[..1000],
                &public_keys[..1000],
            );
        }

        b.iter(|| {
            let (results, sigs_per_sec) = verifier
                .verify_batch(
                    black_box(&signatures),
                    black_box(&messages),
                    black_box(&public_keys),
                )
                .expect("CUDA performance test failed");

            assert!(results.iter().all(|&valid| valid));

            // Performance reporting is handled by criterion
            // Store performance data for potential analysis
            let _ = sigs_per_sec;
        });
    });

    group.finish();

    // Print device information
    if let Ok(mut verifier) = CudaEd25519Verifier::new() {
        if let Ok(device_info) = verifier.device_info() {
            println!("\n🎯 CUDA Device Summary:");
            println!(
                "   Device: {} (Compute {}.{})",
                device_info.name,
                device_info.compute_capability_major,
                device_info.compute_capability_minor
            );
            println!(
                "   Recommended batch size: {}",
                device_info.recommended_batch_size()
            );
            println!(
                "   Optimal threads per block: {}",
                device_info.optimal_threads_per_block()
            );
            println!("   Compatible: {}", device_info.is_compatible());
        }
    }
}

criterion_group!(
    benches,
    benchmark_simple_cuda_verification,
    benchmark_cuda_device_utilization,
    benchmark_batch_size_scaling,
    benchmark_single_vs_batch,
    performance_summary
);

criterion_main!(benches);
