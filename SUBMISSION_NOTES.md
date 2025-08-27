## CUDA-assisted verification (CPU+GPU hybrid)

*(single-focus follow-up: **signature verification offload**, nothing else)*

### What changed 🔧

* **Backend:** CPU-only → **Hybrid** (CUDA kernel splits batches in half).
* **Safety:** hard **CPU fallback** on any CUDA error.

### Why it helps 💡

* Signature checks are embarrassingly parallel → GPU matches the batched CPU speeds, effectively pushing throughput from ~800k sig/s to 1.3M sig/s (5900HX + RTX 3080).
* Why is it "only" ~5%? The current block size pushed the bottleneck back to the DB, so cutting down the DB will give better performance than signature optimisations.

### Flags & knobs ⚙️

* Enable with `--features gpu-verification`.

### Results 📈

| Metric (contest harness) | Before |     After |    Δ time |        Δ % |
| ------------------------ | -----: | --------: | --------: | ---------: |
| **Score (s)**            |  97.91 | **92.53** | **-5.38** | **-5.49%** |

*Same dataset & flags as baseline; improvement attributable solely to CUDA verification.*

---

*Submission for **thunder-rust optimisation contest** — **2025-09-09***
