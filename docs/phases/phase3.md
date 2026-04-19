# Phase 3 Results: Skip Redundant Recompute (the first big algorithmic win)

## TL;DR

**~30% backward pass speedup, ~25% throughput increase, bit-exact identical loss.**
Trades a small amount of GPU memory for a meaningful reduction in GPU compute.
This is the first optimization in this series that isn't a wash.

| Scale (7B, seq=512) | Baseline bwd | Phase 3 bwd | Δ bwd | Tokens/s gain | GPU mem Δ |
|---|---|---|---|---|---|
| batch=2 | 2.50s | 1.75s | **-30%** | +28% | +150 MB |
| batch=4 | 3.15s | 2.20s | **-30%** | +23% | +290 MB |
| batch=8 | 3.39s | 2.44s | **-28%** | +26% | +570 MB |
| batch=2, seq=1024 | 3.49s | 2.56s | **-27%** | +25% | +290 MB |

## The Insight

The current MegaTrain backward does **two forward passes per layer per step**:

1. **Recompute loop** (per block of `checkpoint_interval` layers): replay the forward with `torch.no_grad()` to reconstruct intermediate activations. Stores output of each layer in `recompute_cache`.
2. **Backward loop** (same block, reversed): for each layer, run forward *again* (this time with grad enabled so autograd can build a graph), then `torch.autograd.grad(...)`.

The second forward is unavoidable — `autograd.grad` needs an autograd graph. But the first forward is only there to reconstruct inputs to layers that came AFTER the checkpoint. If we simply **store every layer's input during the initial forward pass**, we can skip the recompute loop entirely.

Cost: one extra BF16 `hidden` tensor per layer instead of one per `checkpoint_interval` layers. For a 28-layer model with `interval=4`, that's 21 additional saved tensors. Each is `B × T × H × 2 bytes`.

## Implementation

New config flag (opt-in, off by default):

```python
store_all_activations: bool = False
```

When enabled:

- `_forward_hidden` saves `checkpoints[i] = hidden.detach()` for EVERY layer i, not just every `checkpoint_interval`.
- `forward_and_backward` skips the recompute loop entirely (the whole `with torch.no_grad():` block that replays forward).
- Inside the backward loop, `layer_input = checkpoints[i].detach().requires_grad_(True)` replaces the `recompute_cache[i-1]` lookup.

Both `forward_and_backward` and `forward_and_backward_custom_loss` were updated.

## Correctness

Loss values are **bit-exact identical** across baseline and Phase 3, step by step:

```
 step   baseline loss     phase3 loss       diff
    0        1.506000        1.506000   0.00e+00
    1        1.454100        1.454100   0.00e+00
    2        1.454500        1.454500   0.00e+00
    3        1.079900        1.079900   0.00e+00
    4        1.076800        1.076800   0.00e+00
```

There's no numerical approximation here — the recompute and store-all paths compute exactly the same gradients because the forward pass is deterministic in `torch.no_grad()` mode and the stored activation is identical to what the recompute would produce.

## Why It's Consistently ~30%

The backward time budget at `checkpoint_interval=4` on Qwen2.5-7B breaks down roughly as:

- **Recompute block**: 4 forward passes per block × 7 blocks = 28 forward passes.
- **Backward block**: 4 `(forward + autograd.grad)` per block × 7 blocks = 28 forwards + 28 backwards.

So baseline backward has **56 forwards + 28 backwards** of layer compute. Phase 3 eliminates the first 28: **28 forwards + 28 backwards**. In forward-equivalent units, baseline is 56+56=112, Phase 3 is 28+56=84. Savings: 28/112 = 25%. Observed: 27-30%. The extra 2-5% comes from the Python overhead saved (the entire recompute loop: prefetches, event records, `_unflatten_to_layer` calls × 28).

## Memory Cost Analysis

Extra GPU memory for stored activations:

```
extra_mem = (N_layers - N_checkpoints) * B * T * H * 2  (BF16)
```

For Qwen2.5-7B (N_layers=28, N_checkpoints=7 at interval=4, H=3584):

| B | T | Extra mem |
|---|---|---|
| 2 | 512 | 154 MB |
| 4 | 512 | 308 MB |
| 8 | 512 | 617 MB |
| 2 | 1024 | 309 MB |
| 8 | 4096 | **9.85 GB** |
| 32 | 2048 | **9.85 GB** |

**Recommendation**: Safe to enable when `batch * seq_len < 16384`. For aggressive configurations (very long sequences or very large batches), the memory cost may exceed the available slack on a 80 GB GPU and could force a smaller `batch_size`.

The config is opt-in (default `False`) specifically because the memory cost scales with `B × T`. Users who have headroom should flip it on for ~25% speedup.

## What This Changes About the Plan

This result is important because it's the first optimization in the MegaTrain-Plus roadmap that delivers a clean double-digit speedup. It came from re-reading the code rather than from the original plan's categories (PCIe, multi-GPU, quantization). That suggests:

1. **More wins may be hiding in the training loop itself** — not in the transfer pipeline. Places to look next:
   - `_unflatten_to_layer` launches 24 separate GPU memcpy kernels per call; a grouped copy could reduce launch overhead.
   - The per-layer Python loops (`requires_grad_(True)`, assigning `p.grad = g`, `requires_grad_(False)`) could be consolidated.
2. **The PCIe/compute balance shifts after Phase 3.** With backward compute reduced 28%, per-layer PCIe time becomes a relatively larger fraction. This makes triple buffering (Phase 1B) and FP8 weight transfer (Phase 2) slightly more attractive — though still not decisive at the tested batch sizes.

## Reproducibility

```bash
# Baseline (recompute)
python scripts/benchmark.py --model Qwen/Qwen2.5-7B-Instruct \
    --batch-size 2 --seq-len 512 --steps 5 --num-buffers 2 --no-optimizer

# Phase 3 (skip recompute)
python scripts/benchmark.py --model Qwen/Qwen2.5-7B-Instruct \
    --batch-size 2 --seq-len 512 --steps 5 --num-buffers 2 --no-optimizer \
    --store-all-activations
```

JSON outputs committed at `docs/benchmarks/phase3/*.json`.

## Code Changes

- `infinity/config/training.py`: new `store_all_activations: bool = False` field.
- `infinity/model/cpu_master.py`:
  - `_forward_hidden` / `forward_and_backward` forward loop: stores every activation when flag is on.
  - `forward_and_backward` backward loop: skips recompute block; uses `checkpoints[i]` directly as layer input.
  - `forward_and_backward_custom_loss` backward loop: same change applied.
- `scripts/benchmark.py`: new `--store-all-activations` flag; config recorded in result JSON.
