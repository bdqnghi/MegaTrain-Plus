# MegaTrain-Plus: Revised Architecture Upgrade Plan

## Retrospective (2026-04-18)

This plan was written before any code was measured. After implementing phases
0-4 and a newly-discovered Phase 3 that was not originally in the plan, the
actual results are:

- **-23% step time / +30% throughput** on Qwen2.5-7B at batch=2, seq=512.
- The biggest win (**Phase 3, ~30% backward speedup**) was NOT in this plan.
  It came from re-reading the code and noticing that every layer is forward-
  computed twice per step (once in recompute, once in backward-with-grad).
- The plan's headline bets on PCIe (Phase 2 FP8 weight transfer) and
  synchronization (Phase 1A/1B prefetch+triple-buffer) delivered only small or
  no wall-clock gains at tested scales. CPU compute per layer already exceeds
  PCIe time in the current regime, so transfer-focused optimizations have
  limited headroom.
- One diagnosed "algorithmic bottleneck" (super-linear backward scaling) turned
  out to be a benchmarking artifact from DataLoader worker re-forks (see
  [phase1d_results.md](./phase1d_results.md)). Corrected via `num_workers=0` in
  benchmark and `persistent_workers=True` in the training script.

**Reading order**: [progress_summary.md](./progress_summary.md) for the final
state. The per-phase deep-dive docs are [phase1_results.md](./phase1_results.md),
[phase1d_results.md](./phase1d_results.md), [phase2_results.md](./phase2_results.md),
and [phase3_results.md](./phase3_results.md).

Keeping this plan document as-is below for historical context.

---

## Context

MegaTrain trains large models on a single GPU by streaming layers from CPU RAM through GPU one at a time. The core concept is sound, but code analysis and peer review (Codex critique, Gemini analysis) identified concrete architectural inefficiencies — most notably a **completely serial backward pass** with no prefetching, template contention from only 2 GPU buffers, and a single-threaded gradient worker.

This revised plan corrects the original proposal's overconfidence: bottleneck numbers are now labeled as hypotheses pending profiling, compound speedups are not naively multiplied, and improvements are sequenced by risk (structural fixes first, numerical changes later, scale-out as separate projects).

### What Changed From v1

| v1 Claim | Critique | v2 Correction |
|---|---|---|
| "45-67% throughput loss" with per-bottleneck % | No profiling shown; numbers are hypothetical | Phase 0 establishes measurement baseline first |
| Compound 3.7-7.6x speedup | Over-multiplied; improvements overlap on same transfer path | Removed compound table; each improvement measured independently |
| 4 quantization schemes in one milestone | Four separate research tracks bundled together | Decomposed: weight-only first, others gated on convergence proof |
| "Only ONE sync point" with triple buffering | Still need sync for loss, grad norm, optimizer, param sync | Reframed as "reduce unnecessary sync and improve overlap" |
| NVMe + multi-GPU in same speedup roadmap | Capacity vs speed are different goals; both are large projects | Moved to separate future project briefs |
| 6-9 week schedule for everything | Unrealistic given CPUMasterModel complexity | Honest phasing; no timeline promises |
| Line-number references | Drift as code changes | Replaced with method/section references |

---

## Phase 0: Measurement Baseline

**Goal**: Establish reproducible profiling before optimizing anything. Without this, every speedup claim is a guess.

### Deliverables

1. **Profiling harness** in `infinity/profiler.py` (extend existing):
   - Per-layer wall-clock breakdown: CPU flatten time, H2D transfer time, GPU compute time, D2H grad time, CPU grad accumulation time
   - Per-step sync stall time: time spent in `wait_event`, `synchronize`, `queue.get`, `queue.join`
   - PCIe utilization: bytes transferred / wall-clock time / theoretical bandwidth
   - Forward vs backward asymmetry: compare per-layer times in forward (with prefetch) vs backward (without)

2. **Baseline benchmark script** in `scripts/benchmark.py`:
   - Run 20 steps on Qwen2.5-7B with batch_size=16, seq_len=1024
   - Output: JSON with per-step timing, per-layer timing, memory usage, PCIe stats
   - Nsight Systems trace for visual pipeline analysis

3. **Documented baseline numbers** committed to `docs/baseline_profile.md`

### Why This Comes First

The original plan attributed specific percentages to each bottleneck without evidence. Phase 0 answers: what is the actual breakdown? Is the backward pass really the dominant bottleneck (as Gemini suspects)? How much time is spent in CPU flattening vs PCIe transfer vs GPU compute?

### Files to Create

- `scripts/benchmark.py`: Reproducible profiling script
- `docs/baseline_profile.md`: Measured results

### Files to Modify

- `infinity/profiler.py`: Add per-layer, per-boundary timing instrumentation

---

## Phase 1: Structural Fixes (No Numerical Changes)

**Goal**: Fix the verified architectural bottlenecks in the current BF16 regime. These are the highest-confidence, lowest-risk improvements because they don't change model numerics at all.

### 1A. Backward Pass Prefetching

**The problem (verified in code)**:

The forward pass (`forward_and_backward`, forward loop) prefetches layer `i+1` while computing layer `i`:
```python
# Forward: line 1104-1105 — PREFETCHES next layer
if i + 1 < len(self.cpu_layers):
    self._load_layer_to_buffer_async(i + 1, next_buffer_idx)
```

The backward recompute and backward grad loops do NOT prefetch at all:
```python
# Backward recompute: line 1228-1229 — NO PREFETCH, load-then-wait
self._load_layer_to_buffer_async(j, buffer_idx)
self.compute_stream.wait_event(self.weight_ready_events[buffer_idx])  # stalls

# Backward grad: line 1251-1252 — SAME pattern, NO PREFETCH
self._load_layer_to_buffer_async(i, buffer_idx)
self.compute_stream.wait_event(self.weight_ready_events[buffer_idx])  # stalls
```

The GPU sits completely idle during every backward layer load. For a model where backward takes ~2x the time of forward, this means the backward pass has ~zero overlap between transfer and compute.

**The fix**: Apply the same prefetch pattern used in forward to both backward loops. Prefetch layer `j+1` (recompute) or layer `i-1` (backward grad) while computing the current layer.

**Risk**: Low. No numerical change. Same pattern already working in forward.

**Files to modify**: `infinity/model/cpu_master.py` — the two backward loops in `forward_and_backward` method.

### 1B. Triple Buffering

**The problem (confirmed by Gemini)**:

With only 2 GPU flat buffers (`buffer_idx = i % 2`), each buffer must wait for BOTH:
- GPU compute to finish (using the buffer's layer template)
- Gradient D2H to finish (freeing the template via `template_free_events`)

before it can be reused for the next layer load. This creates a dependency chain: load -> compute -> grad D2H -> (buffer free) -> next load.

**The fix**: Add a 3rd flat buffer and template set. This breaks the dependency: while buffer 0 computes, buffer 1 loads next weights, buffer 2 collects previous grads. All three activities can overlap.

**What triple buffering does NOT do**: It does not eliminate all synchronization. You still need sync for:
- Loss materialization (to read the scalar value)
- Gradient norm computation and clipping
- Optimizer step ordering
- Parameter sync back to GPU modules (`_sync_params_to_gpu`)

The goal is to reduce unnecessary transfer-compute stalls, not to claim "only one sync point."

**Risk**: Low-medium. More GPU memory (~+444 MB for 7B model per extra buffer). Need to verify template lifecycle is correct with 3 slots.

**Files to modify**: `infinity/model/cpu_master.py` — buffer allocation in `__init__`, buffer index cycling (`i % 3` instead of `i % 2`) in forward/backward, event arrays sized to 3.

### 1C. Multi-Threaded Gradient Worker

**The problem**: A single `_grad_worker` thread handles all gradient accumulation. When accumulation is slow (large layers, many params), it delays `template_free_events`, which stalls the next buffer reuse.

**The fix**: Replace the single thread with a small thread pool (2-3 workers). Multiple layers' gradients can be accumulated concurrently on CPU.

**Caution (from Codex critique)**: CPU gradient accumulation is memory-bandwidth bound. Concurrent updates to different parameter `.grad` tensors should be safe (no lock contention since they're disjoint tensors), but adding too many workers may hit memory bandwidth limits. Start with 2 workers, measure before adding more.

**Risk**: Low. Gradient accumulation targets disjoint tensors, so no data races. But measure actual benefit — diminishing returns expected beyond 2-3 workers.

**Files to modify**: `infinity/model/cpu_master.py` — replace `_grad_worker` thread + `grad_task_queue` with `ThreadPoolExecutor`.

### 1D. Remove Per-Step Allocations

**The problem**: Several tensors are allocated and freed every training step:
- Rotary embedding dummy tensor (in `forward_and_backward`)
- Loss accumulator `torch.zeros(())`
- Re-embedding in backward

**The fix**: Pre-allocate in `__init__` and reuse. Minor but free improvement.

**Risk**: None.

**Files to modify**: `infinity/model/cpu_master.py` — `__init__` (pre-allocate), forward/backward (reuse).

### Phase 1 Expected Impact

Based on code structure analysis (to be validated by Phase 0 profiling):
- **1A (backward prefetch)**: Likely the single biggest win. Backward is ~2x forward time, and currently has zero overlap. Estimated 15-30% step time reduction.
- **1B (triple buffer)**: Reduces remaining buffer contention stalls. Estimated 5-15% on top of 1A.
- **1C (multi-threaded grad)**: Reduces CPU-side serial bottleneck. Estimated 3-8%.
- **1D (remove allocs)**: Minor, 1-2%.

These estimates are hypotheses. Phase 0 profiling will validate or correct them.

---

## Phase 2: Weight Transfer Quantization (Single Numerical Change)

**Goal**: Reduce H2D transfer size for weight loading. This is the safest quantization path because weights are re-quantized from FP32 master every step (no error accumulation).

### What This Does

Quantize layer weights from BF16 to INT4 (or FP8) before H2D transfer, dequantize on GPU before layer compute.

```
Current:  CPU [FP32 master] -> cast BF16 -> pin -> H2D [BF16, 1GB] -> GPU compute
Proposed: CPU [FP32 master] -> quantize INT4 -> pin -> H2D [INT4, 260MB] -> GPU dequant -> BF16 compute
```

### Why Weight-Only First

The Codex critique correctly identifies that the original plan bundles 4 separate quantization tracks. Each has different risk profiles:

| Quantization Target | Error Accumulates? | Risk | When to Attempt |
|---|---|---|---|
| Weight transfer (H2D) | No (re-quantized from FP32 each step) | Low | Phase 2 |
| Gradient transfer (D2H) | Yes (accumulates in optimizer) | Medium-High | Phase 3 (gated) |
| Activation checkpoints | Yes (propagates through recompute block) | Medium | Phase 3 (gated) |
| Optimizer states (8-bit Adam) | Yes (drifts over training) | High | Phase 3 (gated) |

Weight-only quantization is the only one where errors cannot accumulate across steps. Start here.

### Format Options

- **FP8 E4M3**: 1 byte/param, ~2x bandwidth saving, minimal quantization noise. Simplest to implement (PyTorch has `torch.float8_e4m3fn` native support).
- **INT4 group-wise** (group_size=128): 0.52 bytes/param, ~3.9x saving, more noise but bounded per-step. Requires custom CUDA dequant kernel.

Start with FP8 (lower risk, no custom kernel needed). Graduate to INT4 if FP8 proves stable and more bandwidth is needed.

### Validation Requirements (Addressing Critique)

Before shipping:
1. **Tensor-level parity test**: Compare dequantized BF16 weights vs original BF16 weights. Measure max/mean relative error per layer.
2. **Gradient parity test**: Run 10 steps with and without quantized transfer. Compare per-parameter gradient tensors (should be identical since compute is in BF16 either way — only transfer format changes).
3. **Short-horizon convergence**: 100 steps, verify loss curve matches baseline within 0.5%.
4. **Long-horizon convergence**: 1000 steps, verify final loss matches within 1%.

### CPU Quantization Cost (Addressing Gemini Concern)

CPU-side FP8 quantization (cast + scale) is a memory-bandwidth operation: ~read 4 bytes + write 1 byte per param. For a 500M-param layer:
- Data volume: ~2.5 GB read+write
- At ~50 GB/s CPU memory bandwidth: ~50ms
- Compare to GPU compute time per layer: ~100ms

If CPU quantization takes >50% of GPU compute time, it erodes transfer savings. Mitigation: quantize on a background thread while GPU computes previous layer (same prefetch overlap pattern).

### Files to Create

- `infinity/quantization/__init__.py`
- `infinity/quantization/weight_quant.py`: FP8/INT4 quantize (CPU) + dequantize (GPU)

### Files to Modify

- `infinity/model/cpu_master.py`: `_load_layer_to_buffer_async` (quantize before H2D), `_unflatten_to_layer` (dequantize on GPU)
- `infinity/config/training.py`: Add `weight_transfer_dtype: str = "bfloat16"` (options: "bfloat16", "float8_e4m3", "int4")

---

## Phase 3: Additional Quantization (Gated on Phase 2 Results)

**Prerequisite**: Phase 2 weight-only quantization is validated and merged. Phase 0 profiling shows that D2H gradient transfer or activation checkpoint memory is a measurable bottleneck.

Each of these is a separate milestone with its own convergence validation:

### 3A. Gradient Transfer Quantization (INT8)

Quantize BF16 gradients to INT8 on GPU before D2H, dequantize on CPU.

**Risk**: Medium-high. Gradient quantization error accumulates through the optimizer. Stochastic rounding helps but doesn't eliminate drift. Requires:
- Ablation: train 5000 steps with and without, compare loss curves
- Monitor gradient norm ratio (quantized vs original) per step
- If ratio deviates >5%, increase block size or abandon

### 3B. Activation Checkpoint Quantization (INT8)

Store checkpoints as INT8 per-channel instead of BF16. Halves checkpoint GPU memory.

**Risk**: Medium. Error propagates through recompute block (up to `checkpoint_interval` layers). Mitigated by outlier channel preservation.

### 3C. 8-bit Optimizer States

INT8 moments with block-wise scaling (bitsandbytes approach).

**Risk**: High. Requires extensive convergence validation. The embedding layer must use FP32 states (sparse gradients cause large quantization error). Only attempt after 3A and 3B are stable.

---

## Future Projects (Separate Scope)

These are NOT part of the MegaTrain-Plus speedup roadmap. They are capacity and scale projects that deserve their own design docs, benchmarks, and rollback plans.

### NVMe Tiering (Capacity Project)

**Goal**: Extend model size beyond CPU RAM by offloading cold layers to NVMe SSD.

**Why separate**: This is a capacity play, not a speedup. It trades step time for model size. The integration of `io_uring` with Python, layer-wise optimizer semantics, crash recovery, and SSD thermal throttling are each non-trivial systems problems.

**Prerequisites**: Phases 0-2 complete. Quantized transfers (Phase 2) reduce NVMe I/O volume by 2-4x, making this more practical.

**Key design questions** (for separate design doc):
- Layer-wise vs block-wise optimizer step semantics
- Crash recovery when optimizer state is split across RAM and NVMe
- Prefetch depth calibration (depends on NVMe bandwidth and GPU compute time)
- RAID-0 across multiple NVMe drives for bandwidth

### Multi-GPU Pipeline Parallelism (Scale Project)

**Goal**: Distribute layers across multiple GPUs for near-linear speedup.

**Why separate**: Requires partitioning `CPUMasterModel` by layer range, which changes buffer ownership, template management, checkpointing, and optimizer semantics. NUMA placement and PCIe affinity matter on multi-socket systems. NCCL traffic competes with H2D/D2H on PCIe-connected GPUs (as opposed to NVLink).

**Prerequisites**: Phase 1 structural refactoring (1A-1C). The current God Object architecture of `CPUMasterModel` makes multi-GPU partitioning impractical.

---

## Structural Note: CPUMasterModel Refactoring

Both Codex and Gemini flag that `CPUMasterModel` is a God Object (~1600 lines) owning buffer allocation, stream orchestration, template management, checkpointing, gradient accumulation, and the training loop. All proposed improvements route through this single class, creating high integration risk.

**Recommendation**: As part of Phase 1, consider extracting:
- `BufferManager`: Owns flat buffers, GPU templates, and buffer lifecycle events
- `GradientCollector`: Owns gradient slabs, D2H transfer, CPU accumulation workers
- `LayerScheduler`: Owns prefetch scheduling, buffer assignment, event synchronization

This refactoring is not a separate phase — it's woven into Phase 1 work. Each sub-improvement (1A-1D) naturally touches different concerns within the class, providing clean extraction points.

---

## Verification Strategy (Addressing Critique)

### Per-Phase Verification

| Phase | Verification |
|---|---|
| **Phase 0** | Profiling harness produces reproducible numbers. Baseline committed. |
| **Phase 1** | Step time improvement measured via benchmark script. Loss curve identical to baseline (no numerical change). Nsight trace confirms overlap improvement. |
| **Phase 2** | Tensor-level weight parity test. Gradient parity test (10 steps). Loss convergence test (100 + 1000 steps). CPU quantization overhead measured. |
| **Phase 3A-C** | Each: gradient norm ratio monitoring, ablation study (5000 steps with/without), per-tensor error analysis. |

### What's NOT Acceptable as Verification

- "Loss within 2% after 1000 steps" alone (not enough for gradient quantization)
- No profiling traces (claims without evidence)
- No ablation (can't isolate which change caused a regression)
- No failure-mode testing (buffer starvation, deadlocks, OOM)

### Required Tests Per Change

1. **Microbenchmark**: Isolated timing of the specific operation (e.g., H2D with FP8 vs BF16)
2. **Step-level trace**: Nsight Systems or PyTorch profiler before/after
3. **Numerical parity**: Tensor-level comparison for short runs
4. **Convergence**: Loss curve comparison for long runs (if numerical change)
5. **Stress test**: Maximum batch size, maximum sequence length, minimum GPU memory headroom
