## Contest notes
Currently, there're 2 operations that take the most processing time in connect_block: **Accumulator::apply_diff** and **Accumulator::verify**. Solution to improve *verify* has been submitted last week. 
**compute_merkle_root** only use a small amout of processing power compared to other operations. I don't think it's worth too much time on optimizing it. (see attached flamegraph_withutreexo_nomod.svg file for more details)

**apply_diff** eventually calls MemForest::del_single which removes a node and trigger hash recalculation (recompute_hashes) of the whole tree up to the root. This is inefficient since nodes can share ancestral paths (same root) and hence the root is unncessarily re-calculated multiple times. See examples below. The solution is to simply hold off hash recalculation when all specified nodes are removed. The newly added function **modify_optimized** to mem_forest.rs does exactly this. Instead of calling *del_single*, it uses a new function called *batch_del* to remove nodes without re-calculating hashes every time. A bunch of unit tests are added to the modified rustreexo library to make sure the added function behaves correctly.

Improvement to memforest's *modify* alone improves the sync time by 25%. Combined with solution last week (contest1_1, not included in this submission) will reduce sync time almost by 50% on a 4-core, 8-thread CPU without any significant changes to the current architecture.

Please run `git submodule update --init --recursive` to get the modified version of rustreexo.

### Example of redundancy when recalculating hashes: Sequential Deletion of Nodes [0, 2, 4]

**Tree Setup:**
```
        14 
       /    \
     12      13
    /  \    /  \
   8    9  10   11
  /|   /| /|   /|
 0 1  2 3 4 5 6 7
```

**Step 1: Delete Node 0**
```
Tree restructure: 1 moves up to replace parent 8
Recomputation path: 1 → 12 → 14
Hash calculations:
- hash(12) = hash(1, 9)     ← NEW
- hash(14) = hash(12', 13)  ← NEW (uses updated hash(12))
Total: 2 computations
```

**Step 2: Delete Node 2**
```
Tree restructure: 3 moves up to replace parent 9
Recomputation path: 3 → 12 → 14
Hash calculations:
- hash(12) = hash(1, 3)     ← REDUNDANT (12 computed again!)
- hash(14) = hash(12'', 13) ← REDUNDANT (14 computed again!)
Total: 2 more computations (4 cumulative)
```

**Step 3: Delete Node 4**
```
Tree restructure: 5 moves up to replace parent 10
Recomputation path: 5 → 13 → 14
Hash calculations:
- hash(13) = hash(5, 11)    ← NEW
- hash(14) = hash(12'', 13') ← REDUNDANT (14 computed third time!)
Total: 2 more computations (6 cumulative)
```
