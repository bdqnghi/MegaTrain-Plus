# MegaTrain-Plus: Status

> Single source of truth for what's landed and what's next. For depth on any phase, follow the links.

## Headline Result

Qwen2.5-7B, batch=2, seq=512, single GPU, RAM-safe benchmark (no optimizer step):

| Metric | Baseline | MegaTrain-Plus | Δ |
|---|---|---|---|
| Step time | 3.604s | **2.539s** | **-29.6%** |
| Forward | 0.794s | **0.678s** | **-14.6%** |
| Backward | 2.600s | **1.653s** | **-36.4%** |
| Throughput (tokens/s) | 284.1 | **403.3** | **+42.0%** |
| Peak GPU memory | 7.15 GB | **6.86 GB** | **-290 MB** |
| BWD/FWD ratio | 3.28x | **2.44x** | approaches theoretical ~2x |
| Loss trajectory | baseline | **bit-exact identical** | — |

**The MegaTrain-Plus build is both faster AND uses less GPU memory.**

Baseline = stock MegaTrain behavior: `num_buffers=2`, no backward prefetch, `store_all_activations=False`, memcpy-based unflatten.

Plus = defaults shipped in this branch: Phase 1A (backward prefetch), Phase 1B (`num_buffers=3`), Phase 3 (`store_all_activations=True`), Phase 5 (zero-copy unflatten, default ON).

## Shipped

### Default ON (no action needed by users)

| Phase | Change | Gain | Risk | Deep-dive |
|---|---|---|---|---|
| **1A** | Backward pass prefetching (recompute + grad loops) | -2 to -3% step time, consistent | None — numerical behavior unchanged | [phase1.md](phases/phase1.md) |
| **1B** | Triple GPU flat buffers (`num_buffers=3`) | Not measurable at tested scales; insurance for regimes where PCIe dominates | ~+440 MB GPU mem per extra buffer | [phase1.md](phases/phase1.md) |
| **5** | Zero-copy unflatten (template params alias flat GPU buffer) | **-8% step time, -440 MB GPU memory** | None — bit-exact loss, auto-disabled with FP8 | [phase5.md](phases/phase5.md) |
| **1C** | Multi-threaded gradient accumulation (`num_grad_workers=2`) | Within noise at tested scales; insurance for grad-worker-bound regimes | None | — |

### Opt-in (flip via config)

| Phase | Config flag | Gain | Cost | Deep-dive |
|---|---|---|---|---|
| **3** | `store_all_activations: true` | **-25 to -30% backward, +25-30% throughput** | +150-600 MB GPU per batch×seq increase | [phase3.md](phases/phase3.md) |
| **2** | `weight_transfer_dtype: float8_e4m3` | Correct but currently wall-clock NEGATIVE (CPU packing cost exceeds PCIe savings on commodity hw) | CPU overhead ~3.4x baseline copy | [phase2.md](phases/phase2.md) |

### Implementation-neutral (kept for clean code; no measurable wall-clock effect)

| Phase | Change | Why kept |
|---|---|---|
| **4** | `torch._foreach_copy_` fusion in `_unflatten_to_layer` | Cleaner; zero downside |
| **4b** | Cache `gpu_template_params` per template | Avoids repeated module-tree traversal; clearer code |

### Benchmark / infrastructure

| Phase | Change | Impact |
|---|---|---|
| **0** | `scripts/benchmark.py` with `--no-optimizer`, `--num-buffers`, `--no-backward-prefetch`, `--store-all-activations`, `--weight-transfer-dtype`, `--block-timing` flags | All results reproducible; JSONs committed |
| **1D** | DataLoader `num_workers=0` in benchmark; `persistent_workers=True` in `examples/sft/train.py` | Eliminates 20-second worker-refork stalls on small datasets; made all prior measurements trustworthy | [phase1d.md](phases/phase1d.md) |

## Config

New fields in `infinity/config/training.py`:

```python
num_buffers: int = 3                      # 2=double, 3=triple
backward_prefetch: bool = True            # Phase 1A
store_all_activations: bool = False       # Phase 3 (opt-in, +25-30% throughput)
zero_copy_unflatten: bool = True          # Phase 5 (default ON, -8% step + -440MB)
num_grad_workers: int = 2                 # Phase 1C
weight_transfer_dtype: str = "bfloat16"   # "bfloat16" | "float8_e4m3" (opt-in, currently wall-clock neg)
diagnostic_block_timing: bool = False     # Per-block CUDA event timing (diagnostic)
```

All are wired through `infinity/config/yaml_loader.py` so YAML configs can set them.

## Recommended Config For 7B-class Models On A Single 80 GB GPU

```yaml
memory:
  checkpoint_interval: 4
  num_grad_slabs: 12
  num_buffers: 3
  backward_prefetch: true
  store_all_activations: true   # +25-30% throughput if memory allows
```

Disable `store_all_activations` when:
- GPU has little headroom (peak mem > 90% after a trial run), or
- `batch_size * seq_len * hidden_dim * 2 * (N_layers - N_checkpoints)` > available GPU slack.

## Quantified Cost Of Each Optimization

Cost curves (Qwen2.5-7B, seq=512):

| Phase | Cost source | Magnitude |
|---|---|---|
| 1A | Extra prefetch CPU flatten calls (4 per block instead of 1) | None — CPU flatten already hidden behind GPU compute |
| 1B | One additional GPU flat buffer (`max_layer_numel * dtype.itemsize`) | +444 MB on 7B |
| 1B | One additional layer template per structure group | +444 MB on 7B |
| 3 | Saving activation at every layer during forward | `(N_layers - N_checkpoints) * B * T * H * 2 bytes`; 154 MB at b=2 seq=512; 617 MB at b=8 seq=512 |
| 2 | CPU quantize workload at every layer load | ~35 ms/layer on commodity CPU; dominates PCIe savings |

## Queued / Future Work

### Near-term (scoped, may or may not win)

| Item | Idea | Priority | Why |
|---|---|---|---|
| **1C** | Multi-threaded gradient worker (`ThreadPoolExecutor`) | low | Low risk, likely small benefit — compute-bound regime doesn't starve the grad worker |
| **Amortized FP8 quant** | Quantize weights once per optimizer step, reuse across fwd+bwd | medium | With Phase 3, each layer is loaded 2x per step (was 3x). Amortization halves per-load CPU cost. May still not beat baseline at small batch but could win at large batch. |
| **Auto-enable `store_all_activations`** | Probe GPU memory at init; enable if estimated extra mem fits with margin | low | User convenience; avoids accidental OOM |

### Long-term (separate design docs needed)

| Item | Why separate |
|---|---|
| **NVMe tier** | Capacity play, not speedup. Large systems project. |
| **Multi-GPU pipeline parallelism** | Requires partitioning `CPUMasterModel` by layer range; NUMA, NCCL, and micro-batch scheduling considerations. |
| **Faster loss + lm_head path** | Cross-entropy over vocab=152k is a meaningful chunk of forward time. Chunked CE with Flash CrossEntropy is already on; further gains would need fused lm_head+CE kernel. |

## Honest Retrospective

The original v1 plan predicted PCIe-focused improvements (weight transfer quantization, triple buffering, async pipeline redesign) would deliver the big wins. They did not.

**The two actual biggest wins were NOT in the plan:**
1. **Phase 3 (~30% backward speedup)** — noticed that every layer is forward-computed twice per backward pass (once in recompute, once to build the autograd graph). Store all activations during forward, skip recompute.
2. **Phase 5 (~8% step speedup AND -440 MB GPU)** — noticed that `_unflatten_to_layer` memcpys 444 MB per layer when the template could just ALIAS the flat buffer via pointer-swap. Zero-copy.

Both came from re-reading the code and asking "what work are we doing that isn't actually necessary?" rather than from the plan's engineering categories.

Lessons for future iterations:
1. **Algorithmic wins (re-examining what work is necessary) dominate engineering wins** (parallelization, pipelining) when the pipeline is already reasonably tuned. Two separate ~8-30% wins came from this angle; the plan's PCIe-focused items gave single-digit results.
2. **Measure the current code path carefully before trusting plan hypotheses.** One of our "bottlenecks" (super-linear backward scaling) turned out to be a DataLoader worker fork artifact, not algorithmic.
3. **CPU/GPU compute dominates PCIe** at the tested batch sizes. PCIe-focused optimizations have limited headroom until we move the compute/transfer balance.
4. **Cost measurement unlocks intuition.** Discovering that `_unflatten_to_layer` costs 6ms/layer (46% of layer forward compute) directly motivated Phase 5. Microbenchmarks beat guesses.
