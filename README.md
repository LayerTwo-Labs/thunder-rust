# Improve the Benchmark performance

I updated the following (with reasoning) :

### Memory Allocation

• Implements `typed-arena::Arena` for `SpentOutput` objects to reduce heap fragmentation and improve cache locality by grouping similar allocations together

• Replaces `BTreeMap` collections with `Vec` structures to eliminate balanced tree overhead and enable contiguous memory access patterns

• Pre-calculates `input_count` and `output_count` to initialize vectors with exact capacity, avoiding dynamic reallocations during transaction processing

### Parallel Processing & Database Optimizations

• Implements `par_sort_unstable()` and `par_sort_unstable_by_key()` to leverage multi-core processing

• Sorts data structures before DB writes to enable sequential LMDB access patterns, reducing disk seeks and improving cache utilization

• Uses both arena-allocated (`spent_outputs_arena`) and regular vectors (`spent_outputs`) to optimize memory layout

• Removes BTreeMap insertion/lookup overhead by directly appending to vectors during transaction processing, reducing computational complexity from O(log n) to O(1)


