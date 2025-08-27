# Improve the Benchmark performance

I updated the following (with reasoning) :

## Database Optimization
- Sequential insertion patterns for UTXO/STXO database writes to minimize B-tree rebalancing overhead and reduce database traversal costs
- Key conversion using `MaybeUninit` and direct pointer operations, reducing memory copies and improving cache locality for the 37-byte fixed-width keys
- `fast_cmp_same_tag()` method to skip tag comparison when OutPoint types are known, optimizing lexicographic sorting performance
- `tag()`, `vout()`, and `id_bytes()` methods for efficient data extraction without full deserialization overhead
- Structured batch operations (`batch_delete_utxos`, `batch_put_stxos`, `batch_put_utxos`) to leverage LMDB's sequential insertion benefits

## Memory Pool Optimization
- `MemoryPools` struct with mutex-protected vector pools for `OutPointKey`, STXO, and UTXO data to eliminate repeated allocations during block processing
- Vector pooling strategy prevents allocation/deallocation cycles during high-frequency block processing operations

## Parallel Processing Optimization
- **Two-Phase Pipeline Architecture**: Parallel worker pool (Stage A) with independent RoTxn per worker for block prevalidation, coordinated by single writer (Stage B) for optimal LMDB performance
- **Cross-Request Parallelism**: Moved block prevalidation out of write transactions, enabling parallel processing across multiple blocks/requests and eliminating writer contention during validation
- **Independent RoTxn Workers**: Each validation worker uses independent read transactions, enabling true parallelism while respecting LMDB's single-writer constraint
- **Pipeline Coordination**: Message passing between parallel workers and sequential writer with proper ordering and error handling mechanisms
- **Parallel Transaction Processing**: Rayon patterns across authorization verification, merkle tree operations, and transaction processing within blocks for maximum CPU utilization
- **Pre-Serialization Optimization**: Off-thread value pre-serialization before entering writer phase, reducing database operation overhead through parallel data preparation
- **Parallel Merkle Tree Computation**: Threshold-based parallel merkle root calculation for larger transaction sets, for leaf node computation