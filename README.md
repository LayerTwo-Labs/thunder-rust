# Improve the Benchmark performance

I updated the following (with reasoning) :

1. Profile flags - low hanging inbuilt Rust optimizations
2. Cache address computation - instead of recalculating addresses from verifying keys (computation intensive), they are cached and retrieved when necessary
3. Increased Signature Verification Batch Size - larger batch sizes reduce thread synchronization overhead
4. Pre-allocate Collections in Block Validation & Optimize Vector Allocations in Authorization - Memory optimizations to reduce allocations and improve cache locality