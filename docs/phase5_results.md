# Phase 5 Results: Zero-Copy Unflatten (the second big algorithmic win)

## TL;DR

**Template params are rebound at init to VIEWS of the flat GPU buffer.** The ~6ms/layer memcpy in `_unflatten_to_layer` becomes a no-op. Net effect: **-8% step time AND -440 MB GPU memory**. Combined with Phase 3: **-29.6% step / +42% throughput / -290 MB GPU** vs pre-MegaTrain-Plus baseline. Loss bit-exact identical.

Default ON (safe; auto-disabled when FP8 weight transfer is configured).

## The Insight

Each GPU layer template had its own storage. `_unflatten_to_layer` did a memcpy from the flat buffer to the template's parameter storage before every layer compute. For Qwen2.5-7B:

- 444 MB flat buffer × 2 passes per step (with Phase 3) = **888 MB of GPU memcpy per step**, wasted.
- Measured cost per layer: **6.2 ms** (46% of layer forward compute time).
- Per step: 6.2 ms × 28 layers × 2 passes = **349 ms wasted on memcpy**.

The templates exist because different layers share the same structure — we can reuse one template per `(structure_group, buffer_idx)` rather than allocating a full layer per layer index. But the templates don't need their own backing storage. They can just POINT INTO the flat buffer.

## Implementation

At init time, for each `(group_id, buffer_idx)` combo, rebind every template parameter's `.data` to a view of the corresponding flat GPU buffer:

```python
for buffer_idx in range(self.num_buffers):
    template = copy.deepcopy(self.cpu_layers[representative_idx]).to(self.device)
    if zero_copy_unflatten:
        flat = self.gpu_flat_buffers[buffer_idx]
        offset = 0
        for p, n, shape in zip(template.parameters(), group_numels, group_shapes):
            p.data = flat[offset:offset + n].view(shape)
            offset += n
```

After this, `gpu_layer(x)` reads weights directly from the flat buffer (via the view). `_unflatten_to_layer` becomes a no-op.

## Aliasing Safety

The flat buffer is being read during layer compute, so we can't free it until compute ends. In the memcpy path, `buffer_free_events[buffer_idx]` was recorded at the end of `_unflatten_to_layer`. In zero-copy mode, we record it AFTER the layer compute, via `_signal_buffer_free_after_compute(buffer_idx)` calls inserted into every forward/backward loop.

This slightly delays when the next H2D can start, but the saved 6 ms/layer memcpy more than offsets the lost overlap. Measured net effect is consistently +8% throughput.

## Correctness

```
 step    baseline        phase5          diff
    0    1.506000        1.506000        0.00e+00
    1    1.454100        1.454100        0.00e+00
    2    1.454500        1.454500        0.00e+00
    3    1.079900        1.079900        0.00e+00
    4    1.076800        1.076800        0.00e+00
```

Bit-exact loss. The physical bytes read by the layer forward are the same bytes (just accessed via view instead of being memcpy'd to a separate location first).

## Measured Impact

Qwen2.5-7B, seq=512, `num_buffers=3`, `store_all_activations=True`:

| Batch | Phase 3 only step | Phase 3+5 step | Δ step | Phase 3 only GPU | Phase 3+5 GPU | Δ GPU |
|---|---|---|---|---|---|---|
| 2 | 2.744s | **2.529s** | **-7.8%** | 7.30 GB | **6.86 GB** | **-440 MB** |
| 4 | 3.304s | **3.039s** | **-8.0%** | 7.77 GB | **7.33 GB** | **-440 MB** |
| 8 | 3.624s | **3.341s** | **-7.8%** | 8.70 GB | **8.27 GB** | **-430 MB** |

Phase 5 alone (without Phase 3) on batch=2, seq=512:

| Metric | Baseline | P5 alone | Δ |
|---|---|---|---|
| Step time | 3.604s | **3.185s** | **-11.6%** |
| Tokens/sec | 284 | **322** | **+13.4%** |
| GPU mem | 7.15 GB | **6.72 GB** | **-430 MB** |

## Why The Memory Went DOWN

With memcpy-based unflatten, each buffer had:
- One flat buffer (444 MB) used as H2D destination
- `N_buffers` GPU layer templates, each with their own 444 MB of parameter storage

Total: `(1 + N_buffers) * 444 MB` per structure group. At `num_buffers=3`, that's 4 × 444 MB = 1.78 GB.

With zero-copy:
- `N_buffers` flat buffers (444 MB each, used both as H2D destination AND as template storage via aliasing)

Total: `N_buffers * 444 MB`. At `num_buffers=3`: 1.33 GB. **Saves 444 MB**.

(Measured: -440 MB. Small delta due to alignment.)

## Why Phase 5 Works And FP8 Weight Transfer Doesn't

Both aim to reduce per-layer transfer/copy overhead. Why is Phase 5 a net win and Phase 2 a net loss at current scales?

- **Phase 2 (FP8 transfer)**: CPU-side quantize costs ~35 ms/layer. PCIe savings are ~12 ms/layer. CPU quant > PCIe savings ⇒ net loss.
- **Phase 5 (zero-copy)**: Zero CPU cost (one-time init). GPU savings are ~6 ms/layer (eliminated memcpy). Any nonzero savings at zero cost is a win.

## What It Cannot Do

- **Cannot combine with FP8 weight transfer** (`weight_transfer_dtype="float8_e4m3"`): FP8 needs a dequantize step during unflatten, which writes a different representation into the template. Pointer-swap only works when the flat buffer already contains the target dtype. The init-time guard `self._zero_copy_unflatten = enabled and self._weight_quantizer is None` handles this.
- **Does not help with CPU flatten**: The CPU-side flatten-into-pinned-buffer is unaffected. Phase 5 targets only the GPU-side memcpy.

## Code Changes

- `infinity/config/training.py`: new `zero_copy_unflatten: bool = True` (default ON).
- `infinity/model/cpu_master.py`:
  - `__init__`: rebind template params to flat buffer views when enabled.
  - `_unflatten_to_layer`: early return (no-op) when enabled, still waits on `template_free_events`.
  - `_signal_buffer_free_after_compute`: new helper; called after every compute that previously relied on `_unflatten_to_layer` to free the buffer.
  - Forward and backward loops: `_signal_buffer_free_after_compute(buffer_idx)` added after each layer compute.
  - `rebuild_gpu_buffers`: re-applies pointer-swap when rebuilding (VERL compat).
- `infinity/config/yaml_loader.py`: plumb `zero_copy_unflatten` through from YAML.
- `scripts/benchmark.py`: `--zero-copy-unflatten` / `--no-zero-copy-unflatten` A/B flags.

## Reproducibility

```bash
# Baseline (all off)
python scripts/benchmark.py --model Qwen/Qwen2.5-7B-Instruct \
    --batch-size 2 --seq-len 512 --steps 5 \
    --num-buffers 2 --no-backward-prefetch --no-zero-copy-unflatten --no-optimizer

# Phase 5 alone
python scripts/benchmark.py --model Qwen/Qwen2.5-7B-Instruct \
    --batch-size 2 --seq-len 512 --steps 5 \
    --num-buffers 2 --no-backward-prefetch --no-optimizer

# All wins
python scripts/benchmark.py --model Qwen/Qwen2.5-7B-Instruct \
    --batch-size 2 --seq-len 512 --steps 5 \
    --store-all-activations --no-optimizer
```
