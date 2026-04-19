# MegaTrain-Plus: Progress Summary

Running tally of what's landed, what worked, and what didn't. Each phase has a dedicated deep-dive doc.

## Headline Result

Qwen2.5-7B, batch=2, seq=512, no optimizer step (RAM-safe), single GPU:

| Metric | Baseline | MegaTrain-Plus | Δ |
|---|---|---|---|
| Step time | 3.604s | **2.539s** | **-29.6%** |
| Forward | 0.794s | **0.678s** | **-14.6%** |
| Backward | 2.600s | **1.653s** | **-36.4%** |
| Throughput (tok/s) | 284.1 | **403.3** | **+42.0%** |
| Peak GPU mem | 7.15 GB | **6.86 GB** | **-290 MB** |
| Loss trajectory | baseline | **bit-exact identical** | — |

**Both faster AND less GPU memory.**

Baseline = original MegaTrain (double buffer, no backward prefetch, no store-all-activations, memcpy unflatten).
Plus = Phase 1A (backward prefetch) + Phase 1B (triple buffer) + Phase 3 (skip recompute) + Phase 5 (zero-copy unflatten).

## Shipped

| Phase | Change | Status | Gain | Ref |
|---|---|---|---|---|
| **0** | Reproducible benchmark harness (`scripts/benchmark.py`), baseline established | ✅ | enables everything else | [phase1_results.md](./phase1_results.md) |
| **1A** | Backward-pass prefetching (previously zero-prefetch) | ✅ | **-2% to -3% step time**, consistent across batch sizes | [phase1_results.md](./phase1_results.md) |
| **1B** | Configurable `num_buffers` (2/3), triple buffering default | ✅ | No measurable gain at tested scales; correctness preserved; small GPU-mem cost | [phase1_results.md](./phase1_results.md) |
| **1D** | Root-caused "super-linear backward" as a DataLoader fork artifact; fix applied | ✅ | **fixes the benchmark**, unblocks trustworthy measurements for all future phases; + `persistent_workers=True` in SFT train | [phase1d_results.md](./phase1d_results.md) |
| **3** | Store all activations; skip backward recompute loop (opt-in) | ✅ | **-30% backward, +25-28% throughput**, bit-exact loss, +150-570 MB GPU | [phase3_results.md](./phase3_results.md) |
| **5** | Zero-copy unflatten (pointer-swap template params to flat buffer views) | ✅ default-ON | **-8% step, +9% throughput, -440 MB GPU**, bit-exact loss, auto-disabled with FP8 | [phase5_results.md](./phase5_results.md) |

## Shipped but opt-in (not a net win on current hardware)

| Phase | Change | Status | Notes |
|---|---|---|---|
| **2** | FP8 E4M3 weight transfer quantization | ✅ opt-in via `weight_transfer_dtype="float8_e4m3"` | Correct, but CPU packing cost (~35 ms/layer) exceeds PCIe savings (~12 ms/layer). Net wall-clock regression on current hardware. See [phase2_results.md](./phase2_results.md) for when it could become net-positive. |

## Attempted but Wall-Clock Neutral (kept for cleaner code / future leverage)

| Phase | Change | Result |
|---|---|---|
| **4** | `torch._foreach_copy_` fusion in `_unflatten_to_layer` | No measurable wall-clock gain. Kernel-launch overhead is small vs 444MB memcpy work at tested scales. |
| **4b** | Cache `gpu_template_params` per template (avoids re-traversing module tree) | No measurable wall-clock gain. Python overhead for `parameters()` is not a bottleneck in the current regime. |

## Queued

| Phase | Idea | Priority | Why | 
|---|---|---|---|
| **1C** | Multi-threaded gradient worker (`ThreadPoolExecutor`) | low | Low risk, likely small benefit since compute-bound regime doesn't starve the grad worker |
| **Amortized quantization** | Quantize once per optimizer step, reuse for fwd+bwd (2x amortization with Phase 3) | medium | Makes Phase 2 closer to net-positive at the cost of extra CPU memory for pre-quantized shadow weights |
| **`persistent_workers` validation** | Verify recommendation holds under real (non-demo) dataset load | low | Applied in `examples/sft/train.py`; benchmark already uses `num_workers=0` |

## Key Numbers (Qwen2.5-7B, seq=512, no optimizer, `num_workers=0`)

Phase 3 + Phase 5 (default-ON) vs Phase 3 alone:

| Batch | P3 only step | P3+P5 step | Δ step | P3 only GPU | P3+P5 GPU | Δ GPU |
|---|---|---|---|---|---|---|
| 2 | 2.744s | **2.529s** | -7.8% | 7.30 GB | **6.86 GB** | -440 MB |
| 4 | 3.304s | **3.039s** | -8.0% | 7.77 GB | **7.33 GB** | -440 MB |
| 8 | 3.624s | **3.341s** | -7.8% | 8.70 GB | **8.27 GB** | -430 MB |

Tokens/sec monotonically climbs with batch. Phase 5 delivers a consistent ~8% step-time speedup AND saves ~440 MB of GPU memory across all tested configurations. Phase 3 contributes ~25-30% throughput improvement on top. Both preserve bit-exact identical loss.

## Experiment Reproducibility

All results can be reproduced with:

```bash
# Baseline (no Phase 1A prefetch, double buffering)
python scripts/benchmark.py --model Qwen/Qwen2.5-7B-Instruct \
    --batch-size 2 --seq-len 512 --steps 5 \
    --num-buffers 2 --no-backward-prefetch --no-optimizer

# Phase 1A (backward prefetch on, double buffering)
python scripts/benchmark.py --model Qwen/Qwen2.5-7B-Instruct \
    --batch-size 2 --seq-len 512 --steps 5 \
    --num-buffers 2 --no-optimizer

# Phase 1A+1B (triple buffering)
python scripts/benchmark.py --model Qwen/Qwen2.5-7B-Instruct \
    --batch-size 2 --seq-len 512 --steps 5 \
    --num-buffers 3 --no-optimizer

# Phase 2 FP8 weight transfer (opt-in; slower on this hardware)
python scripts/benchmark.py --model Qwen/Qwen2.5-0.5B-Instruct \
    --batch-size 4 --seq-len 512 --steps 5 \
    --weight-transfer-dtype float8_e4m3 --no-optimizer
```

All JSON outputs are committed under `docs/phase*_*.json`.

## What's Different From the v2 Plan

The [revised plan](./megatrain-plus-plan.md) correctly tempered expectations vs
the original v1 plan, but even it overcounted the expected gains from its
PCIe/transfer-focused improvements:

- **Claimed ~15-30% from backward prefetch (Phase 1A)**: actual is ~2-3%. At
  batch sizes that fit in practical GPU memory, GPU compute per backward layer
  already exceeds PCIe transfer time, so prefetch saves sync overhead, not
  transfer time.
- **Claimed 1.3-1.6x from FP8 weight transfer (Phase 2)**: actual is a
  wall-clock regression at default settings. CPU quantize overhead (~35ms/layer)
  dominates PCIe savings (~12ms/layer) on commodity hardware.

However, the plan **missed** the single biggest lever: **redundant recompute in
the backward pass.** Phase 3 wasn't in the plan at all — it came from re-reading
the code and noticing that every layer gets forward-computed twice during the
backward (once in recompute, once to build autograd graph). Eliminating that
redundancy delivered the expected ~25-30% speedup the plan had hoped to get
from PCIe tuning. Lesson: when a plan makes confident predictions about where
the bottleneck is, re-read the code before trusting them.

Remaining levers (in priority order):

1. **Further reduce per-layer GPU compute**. After Phase 3 the backward is
   essentially "N forwards + N backwards". Any work that makes the per-layer
   backward cheaper directly scales. Options: custom fused grad kernels, FP8
   compute (not just transfer).
2. **Per-layer Python/kernel-launch overhead reduction** (fuse the 24-param
   unflatten; fewer `requires_grad_` toggles per layer).
3. **Amortized FP8 weight quantization** — now more attractive because Phase 3
   shifted the compute/PCIe balance: with recompute skipped, PCIe is relatively
   more prominent. Amortizing the CPU quant cost across 2 reads (fwd + bwd)
   instead of 3 (fwd + recompute + bwd) still helps.
4. **Multi-GPU** — the single biggest untapped lever, requires a real design
   pass.
