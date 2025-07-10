## Arena-Forest Refactor

*(from classic **MemForest** to compact **Arena + Dirty-Levels**)*

### What I changed 🔧

| Area                 | Before – `MemForest`                                                        | After – `ArenaForest`                                                                                                                                             |
| -------------------- | --------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Node storage**     | `Rc<RefCell<Node>>` graph (≈ 48 B + pointer / node, heavy pointer-chasing). | **SoA-friendly Arena**: `Vec<ArenaNode>` (48 B flat) with `parent` and `level` fields ⇒ 4-5 × better cache locality, zero `Rc` overhead.                          |
| **Hash propagation** | Recursive “bubble-up” per edit (`O(height)` per leaf).                      | **Dirty-Levels**: mark each touched node & its ancestors into `dirty_levels[level]`; one bottom-up pass hashes every dirty node exactly once (`O(#dirty)` total). |
| **Parallelism**      | None.                                                                       | Rayon level-wise hashing; adaptive switch (`threads * 4` dirty nodes) chosen via benchmark.                                                                       |
| **Root maintenance** | Placeholder roots allocated on every row change.                            | Placeholder only when the row is still populated; otherwise the root slot is removed, matching header layout.                                                     |
| **Bench & Tests**    | Ad-hoc unit tests.                                                          | Mixed add/del parity tests.                                                       |

### Why 💡

1. **Speed** – Arena eliminates pointer chasing; Dirty-Levels turns `k` leaf edits into at most `k log n` hash ops (vs. `k log n` *recursive* calls), and we can use Rayon for efficient threading. Net result: **\~10 % faster** than baseline in CI block bench; multicore machines may benefit even more - my local tests (running a Ryzen 5900HX) showed this implementation to be **~20 % faster** on the block bench.

3. **Memory footprint** – Single `Vec` for nodes drops heap usage and improves L1-hit rate.

4. **Maintainability** – No `Rc`, no global side effects.  Easy per level debugging with `dirty_levels[level]`.

### How to read the code

* `arena_forest/compatibility.rs`   – public API (`modify`, `prove`, `verify`).
* `arena_forest/types.rs` – compact node definition.
* `arena_forest/dirty_tracking.rs`     – Dirty-Levels + parallel hash pass.
* `arena_forest/forest.rs`            – forest implementation.
* `arena_forest/construction.rs & deletion.rs`            – add/delete.
* `arena_forest/navigation.rs`- efficiently navigate.
* `arena_forest/proofs.rs`- equivalent to `proof.rs`
* `arena_forest/serialization.rs`- simple serialize/deserialize.
* `arena_forest/tests.rs`            – regression tests.

---

*Submission for the thunder-rust optimisation project – 2025-07-09*
