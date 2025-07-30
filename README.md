# Improve the Benchmark performance

I updated the following (with reasoning) :

• **Thread-Local Hash Caching**: Implements `MERKLE_HASH_CACHE` using `RefCell<HashMap<Txid, Hash>>` to cache computed hashes and avoid redundant cryptographic operations during Merkle tree construction

• **Adaptive Parallel Processing**: Introduces `compute_merkle_leaves_parallel()` with intelligent chunking that adapts to system capabilities - uses smaller chunks (16-32 items) on 2-core systems and larger chunks on multi-core systems

• **Optimized Rayon Thread Pool**: Configures Rayon with reduced stack size (1MB vs 8MB default) and optimized thread count specifically for 2-core GitHub Actions runners to reduce memory overhead

• **Smart Workload Distribution**: Implements `get_adaptive_chunk_size()` that dynamically adjusts chunk sizes based on available CPU cores and total workload to minimize context switching on constrained systems

• **Conservative Chunking Strategy**: Uses smaller base chunk sizes (32 vs previous 50) to avoid SIMD regression and improve cache locality, especially beneficial for systems with limited CPU cores

• **Fallback Sequential Processing**: Automatically falls back to `compute_merkle_leaves_simple_parallel()` for small datasets (<1000 transactions on 2-core systems) to avoid parallelization overhead

• **Memory-Efficient Pre-allocation**: Pre-allocates result vectors with known capacity to reduce memory allocations during parallel Merkle tree computation
        