## Allocator & Vec pre-allocs

*(follow‑up to the LMDB hot path and flags improvements – focuses on the **collections and iterations, still on the ingest** layer)*

### What I changed🔧

| Area | Before | After |
|------|--------|-------|
| **Allocator** | Standard allocator, pretty slow | **MiMalloc v3**: high performance allocator from Microsoft, especially fine tuned for our large set of Vecs and other collections we rapidly generate, iterate, and destroy |
| **Collections** | We had some BTree usage and expensive Ord derives for comparisons | Removed BTree usage from the hot path so we avoid expensive rebalancing. Rewrote `Ord` to work with a bytes-first `OutpointKey` to avoid expensive Struct comparisons |
| **Rayon Par** | Used extensively already | Extended the use to our new Vec collections to shave some extra time off |
| **HashMaps** | Fast, but not fast enough, and used for double spend detection | Switched to Vecs with parallel iterators
| **Vector allocs** | Generally left to fend for themselves in terms of allocation | Most hot-path Vecs now get pre-allocated to sensible defaults. Avoids strain on growth and reallocation |
| **Validation steps** | Double tip and merkle root verification | Verify tip once in prevalidate and carry - same with merkle

### Why💡

1. **Cuts B‑tree searches** – Vecs are faster than BTree and we don't leverage any special properties - its a drop-in change
2. **Faster allocator** – While the largest amount of our work is still the db ops, a large amount of time is spent allocating and deallocating internals during validation. MiMalloc has extremely high performance when dealing with large number of small alloc cycles.
3. **More parallelisation and better memory patterns** - removing expensive collections, pre-allocating and switching to more optimised options helps us gain some cycle here and there - with no large change needed.
4. **Cut useless validation** - we avoid an extra tip read and extra merkle comparison.

### Results📈

| Metric                     | Before | After      | Δ            |
| -------------------------- | ------ | ---------- | ------------ |
| Bench (**local**)    | 35.19s | **25.31s** | **‑28%**  |

* My local bench is on a Ryzen5900HX and 64GB of RAM. Hugepages are set to `[always]`, and MiMalloc does allocate anon pages.
---

*Submission for **thunder‑rust optimisation contest** – 2025‑08‑12*
