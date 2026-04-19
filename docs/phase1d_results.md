# Phase 1D Results: Root-Causing the "Super-Linear Backward Scaling"

## TL;DR

**The super-linear backward scaling documented in Phase 1 was a benchmarking artifact, not an algorithmic bug.** DataLoader worker processes re-fork every time the iterator cycles over the (tiny) demo dataset. On a 40+ GB parent process, each fork stalls for 20+ seconds. The previous `batch=4` measurement of `bwd=9.58s` was actually `bwd=2.51s` plus a 22s stall during the iterator restart, averaged out.

**With `num_workers=0`, backward scales sublinearly with batch size** — exactly as expected for a compute-heavy kernel that amortizes per-layer overhead over more tokens.

| Batch | Tokens | Backward | ms/token |
|---|---|---|---|
| 2 | 1024 | 2.52s | 2.46 |
| 4 | 2048 | 3.15s | 1.54 |
| 8 | 4096 | 3.39s | 0.83 |

Throughput per token **triples** between batch=2 and batch=8, which is the signature of a compute-efficient kernel.

## How I Found It

The clue was in the step-level log from the earlier batch=4 run:

```
[step 1] loss=1.2623 | total=4.52s  | bwd=3.36s
[step 2] loss=1.0768 | total=3.51s  | bwd=2.52s
[step 3] loss=1.4752 | total=23.88s | bwd=22.74s   <-- spike
[step 4] loss=1.2623 | total=4.50s  | bwd=3.36s
```

Step 3 has `loss=1.4752`, which exactly matches the warmup step (step 0) loss. That's the smoking gun: the DataLoader iterator had exhausted the 10-sample demo dataset and restarted. That restart is what consumes 20 seconds.

The underlying backward was never slow — the *wall clock* was slow because the Python interpreter was blocked re-forking worker processes from a large parent image. Since the benchmark measures `time.perf_counter()` bracketed around `model.forward_and_backward(...)`, the stall gets attributed to backward time.

## What's Actually Happening

`torch.utils.data.DataLoader(..., num_workers=2)` spawns 2 worker processes that each get `fork()`ed from the main Python process. After the main process loads the model (~14 GB BF16 weights + 14 GB of layer templates + pinned buffers + autograd state ≈ 40 GB resident), a `fork()` has to:

1. Duplicate the parent's page tables (Linux uses copy-on-write, but the initial table copy is still O(process size))
2. Spin up the Python interpreter in the child, import modules, set up the dataset
3. Do the first `__getitem__`, which triggers tokenizer caching

On a 40 GB process, steps 1 and 2 together take 15-25 seconds. When the iterator exhausts and we call `iter(dataloader)` again, this whole dance repeats.

## The Fix

Two options, pick based on context:

**Benchmark context** (tiny datasets that cycle often): `num_workers=0`. No forks, no stalls. Applied to `scripts/benchmark.py`.

**Production context** (real datasets that don't cycle within a run): `num_workers > 0` is still fine, but either
- use `persistent_workers=True` (workers stay alive across iterations, no re-fork at cycle)
- use `multiprocessing_context='spawn'` (no fork, no page-table duplication — but slower startup)
- ensure workers are spawned *before* the big model is loaded (not always possible given MegaTrain's init flow)

For MegaTrain specifically, `persistent_workers=True` is the cleanest answer in production because it gives you parallel tokenization without the per-cycle cost.

## Corrected Phase 1 Numbers

With the benchmark artifact removed, I re-ran the Phase 1A A/B test cleanly:

| Batch | Backward (no prefetch) | Backward (Phase 1A prefetch) | Improvement |
|---|---|---|---|
| 2 | 2.575s | 2.521s | **-2.1%** |
| 4 | 3.233s | 3.146s | **-2.7%** |

Phase 1A is a real, consistent ~2% improvement. It isn't the ~15-25% I first hypothesized because at these batch sizes, GPU compute per backward layer (~45 ms) already exceeds the PCIe transfer per layer (~22 ms), so prefetch saves primarily sync overhead rather than hiding actual transfer time.

## Batch-Size Scaling (Corrected)

Re-running at multiple batches with `num_workers=0`:

| Batch | Forward | Backward | BWD/FWD | Tokens/sec |
|---|---|---|---|---|
| 2 | 0.79s | 2.52s | 3.19x | 291 |
| 4 | 0.89s | 3.15s | 3.51x | 421 |
| 8 | 0.96s | 3.39s | 3.54x | 449 |

Tokens/sec climbs monotonically with batch (291 → 421 → 449). Batch-size scaling is healthy; the earlier "10x blowup" was entirely the fork artifact.

## Implications for the Overall Roadmap

1. **Phase 1A is net-positive and ships as default-on** (it was already the default). Real 2% gain, zero numerical change, zero risk.
2. **Phase 1B (triple buffering)** still doesn't pay off at current scales because GPU compute dominates PCIe. It's correct, costs one extra layer buffer (~444 MB on 7B), and may help in configurations where PCIe is relatively more expensive (older PCIe, much larger layers). Keep it at the default `num_buffers=3` — the cost is tiny and it's insurance for future regimes.
3. **Phase 2 (FP8 weight transfer) stays opt-in**, with the caveat from its own writeup: the CPU quantize cost on commodity hardware exceeds the PCIe savings.
4. **Phase 1D's biggest concrete deliverable is the num_workers fix in the benchmark**, which makes all future comparisons trustworthy. Without it, every speedup measurement would be masked by ~20s fork stalls randomly sprinkled through the run.
5. **Bigger levers to focus on next** are not PCIe-related. Per-token GPU compute cost is the current ceiling. Directions worth exploring:
   - Use SDPA/Flash Attention 3 kernels if available
   - Reduce per-layer Python overhead (`requires_grad_` toggles, `p.grad = g` assignments in a loop)
   - Fuse `_unflatten_to_layer`'s 24 per-param memcpys into one grouped memcpy kernel

## Code Changes

- `scripts/benchmark.py`: `num_workers=0` in the DataLoader; added `--block-timing` flag.
- `infinity/model/cpu_master.py`: optional per-block CUDA event timing gated on `config.diagnostic_block_timing` (off by default, zero cost when off).
- `infinity/config/training.py`: new `diagnostic_block_timing: bool = False` field.

## Lesson

"Unexpected super-linear scaling" was the single most surprising result from Phase 1. It turned out to have nothing to do with the training engine. Always inspect step-level traces before attributing a slowdown to the algorithm under test.
