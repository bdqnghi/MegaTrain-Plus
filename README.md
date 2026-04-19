<div align="center">

# MegaTrain-Plus

### A faster, leaner fork of MegaTrain, tuned and validated on NVIDIA DGX Spark.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)

[Improvements](#improvements) | [Quick Start](#quick-start) | [Results](#measured-results) | [Config Reference](#config-reference) | [Docs](#documentation)

</div>

---

## About

MegaTrain-Plus provides **full-precision training of large language models on a single NVIDIA DGX Spark**. The largest model validated so far is **Qwen2.5-32B-Instruct** (32.76B parameters); across 12 public models and three architectures (Qwen2.5, Qwen3, Llama via SmolLM2) from 0.36B to 32.76B, every run matches upstream MegaTrain's loss bit-exact.

It is a fork of [MegaTrain](https://arxiv.org/abs/2604.05091) (Yuan et al. 2026), the RAM-centric training framework that stores all parameters on CPU and streams them through the GPU layer-by-layer. This fork adds algorithmic and engineering improvements that deliver a clean wall-clock speedup while actually *reducing* GPU memory footprint. Every improvement is behind a config flag (some default-on, some opt-in) and every wall-clock claim is backed by a reproducible benchmark JSON committed to [`docs/`](docs/).

**Hardware scope.** All measurements were taken on a single NVIDIA DGX Spark (GB10 Superchip, 128 GB unified LPDDR5X). The ~32B ceiling is set by the 128 GB unified-memory envelope rather than by the method: 70B and larger do not fit in 128 GB at BF16 master precision on Spark. Upstream MegaTrain demonstrates 100B+ training on discrete-GPU servers with ~1 TB of host DRAM; that capability is inherited but not re-validated here. See the [Test Environment](#test-environment) section for how the unified-memory architecture affects interpretation of each improvement.

**Both the user-facing API and the YAML config format are backward compatible.** Existing MegaTrain configs and scripts work unchanged.

## How It Works on DGX Spark

MegaTrain's value on DGX Spark is a **memory-allocator trick**, not a bandwidth trick. The 128 GB unified pool is organized so the CUDA allocator holds only a transient working set, while the bulk of persistent training state sits in the lean CPU-side allocator. Three diagrams:

### The capacity trick: where the bytes sit

```
DGX Spark: 128 GB unified LPDDR5X  (one physical pool)
══════════════════════════════════════════════════════════

NAIVE PyTorch (7B)                  MegaTrain (7B)
─────────────────                   ──────────────

┌─── GPU allocator ────┐           ┌── GPU allocator ──┐
│                      │           │  [1 layer, ~7 GB] │  ← transient
│  BF16 weight   14 GB │           │   (streams in,    │
│  BF16 grad     14 GB │           │    evicts out)    │
│  FP32 master   28 GB │           └──────▲──────▼─────┘
│  Adam m        28 GB │                  │      │
│  Adam v        28 GB │           weights│      │grads
│  ─────────────────   │                  │      ▼
│    112 GB wanted     │           ┌── CPU allocator ──┐
│    × ~1.4 allocator  │           │  BF16 weight 14GB │  ← persistent
│    fragmentation     │           │  BF16 grad   14GB │
│    = ~157 GB needed  │           │  FP32 master 28GB │
│                      │           │  Adam m      28GB │
│  ✗ OOM on 128 GB     │           │  Adam v      28GB │
└──────────────────────┘           │  ─────────────── │
                                   │   ~112 GB total   │
Ceiling: ~2–3B params              └───────────────────┘

                                   Ceiling: ~8B with full
                                   Adam (or ~32B in the
                                   --no-optimizer benchmark
                                   mode this suite uses)
```

Same physical memory, different accounting. Moving the 12 B/param of optimizer state (FP32 master + Adam m + Adam v) off the CUDA allocator is what lifts the ceiling. The CUDA allocator is greedy and fragments, while the CPU allocator packs lean.

### The streaming mechanism: how a forward pass runs

```
Layer-by-layer streaming with ping-pong buffers
════════════════════════════════════════════════

CPU side                              GPU side (transient)
────────                              ────────────────────

┌─ layer 0 weights ─┐    DMA          ┌── buffer A ──┐
│ BF16 flat tensor  │ ────────────▶   │  layer i     │
└───────────────────┘   (weight       └──────┬───────┘
┌─ layer 1 weights ─┐    stream)             │
│ BF16 flat tensor  │                        │ compute
└───────────────────┘                        │
┌─ layer 2 weights ─┐                        ▼
│ BF16 flat tensor  │                 ┌── buffer B ──┐
└───────────────────┘                 │  layer i+1   │  ← prefetched
        ⋮                             │  (ready when │    while A
┌─ layer N weights ─┐                 │   A finishes)│    computed
│ BF16 flat tensor  │                 └──────────────┘
└───────────────────┘

Three concurrent CUDA streams orchestrate this:

  weight_stream   ──▶▶▶──  H2D DMA  (next layer's weights)
  compute_stream  ──▶▶▶──  matmul/attention on current layer
  grad_stream     ──▶▶▶──  D2H DMA  (backward: grads → CPU slab)

Timeline (forward):
  time →
  weight:  [load L0][load L1][load L2][load L3] ...
  compute:         [run  L0 ][run  L1 ][run  L2 ] ...
                    └── buffer A ┘└── buffer B ┘└── A again ┘
                       (ping)       (pong)        (ping)
```

While compute works on buffer A, the DMA engine silently fills buffer B with the next layer. GPU compute never stalls waiting for weights.

### One-screen summary

```
        ┌──────────────────────────────────────────────┐
        │    DGX SPARK (128 GB unified, 1 bucket)      │
        │                                              │
        │   CPU allocator view      GPU allocator view │
        │   ───────────────────     ─────────────────  │
        │                                              │
        │   persistent state        transient work     │
        │                                              │
        │   • FP32 master           • 1 layer's BF16   │
        │   • Adam m (FP32)           weights (flat)   │
        │   • Adam v (FP32)         • activations      │
        │   • BF16 weight copies    • autograd (1 L)   │
        │   • BF16 gradient slabs   • 2–3 flat buffers │
        │                             (ping-pong)      │
        │   ~12–16 B/param          ~1 B/param peak    │
        │        │   ▲                  ▲    │         │
        │        │   │  ── grads D2H ── │    │         │
        │        │   │                  │    │         │
        │        ▼   │  ── weights H2D ─▶    ▼         │
        │       ┌────┴─────────────────────┐           │
        │       │ NVLink-C2C (~600 GB/s,   │           │
        │       │  coherent, same-package) │           │
        │       └──────────────────────────┘           │
        └──────────────────────────────────────────────┘
```

On a discrete H100 + host server, these two views map to **two different physical tiers**: a 1 TB+ DDR host pool and 80 GB HBM connected by ~50 GB/s PCIe. That is the hardware context that makes upstream MegaTrain's "100B on a single GPU" demonstration possible. The large cheap tier holds what the small expensive tier can't. On DGX Spark both views land in the same 128 GB LPDDR5X, so the method still works as an allocator-organization tool, but the size ceiling is set by total package memory rather than by the size of a separate host tier.

## Improvements

| Area | Change | Effect | Default |
|---|---|---|---|
| Backward pipeline | Prefetch the next layer during the backward recompute and grad loops (upstream only did this in forward) | Removes a small per-layer sync stall; largest gain on configurations where PCIe is the binding constraint | ON |
| Buffer layout | Configurable GPU flat buffer count, default 3 (triple buffering) | No-op at most scales today; insurance for regimes where the transfer pipeline becomes the bottleneck | ON |
| Grad accumulation | CPU worker pool for gradient accumulation (replaces the single thread) | No-op at most scales today; insurance for configurations where grad accumulation is on the critical path | ON |
| **Activation memory** | **Store every layer's input during forward; skip the recompute block in backward.** Eliminates the redundant "forward pass for autograd graph" that the backward pass was otherwise doing | **Largest wall-clock win**: typically the biggest single contributor to the speedup. Trade: extra GPU memory for stored activations | opt-in |
| **Unflatten** | **Pointer-swap: template params alias the flat GPU buffer directly instead of holding their own copies.** Eliminates the per-layer GPU memcpy that previously populated template storage | Makes forward noticeably faster AND reduces peak GPU memory (one fewer copy of per-layer weights). Rare case where an optimization saves time and memory at once | ON |
| Hot-path cleanup | `torch._foreach_copy_` fusion plus cached parameter lists | Cleaner, wall-clock neutral | ON |
| Weight transfer | FP8 E4M3 per-tensor quantization for CPU→GPU weight transfer | Designed for PCIe-bound discrete-GPU systems. On unified-memory hosts (DGX Spark) the CPU pack cost currently exceeds the non-existent PCIe savings, so it's wall-clock negative. Kept opt-in for future SIMD / slower-PCIe systems | opt-in |

The two biggest wins come from re-reading the hot path and asking *"what work are we doing that isn't actually necessary?"* rather than from PCIe, quantization, or multi-GPU categories:

1. **Skip recompute.** The backward pass was forward-computing each layer twice per step: once in the recompute loop and once again to build an autograd graph. Storing every layer's input during the initial forward lets the backward use those directly.
2. **Zero-copy unflatten.** Each layer compute was preceded by a GPU-to-GPU memcpy from the flat buffer into a separate template. Pointing the template parameters' `.data` at views of the flat buffer removes the copy entirely and drops peak GPU memory.

Concrete numbers for each improvement vary by model, batch size, and sequence length. See the [Measured Results](#measured-results) section below for the full multi-model table.

Also included:
- **Reproducible benchmark harness** (`scripts/benchmark.py`) with A/B toggles for every improvement.
- **Profiling scripts** that isolate specific costs (unflatten memcpy, FP8 pack, backward scaling, quant overhead).
- **Retrospective** comparing plan predictions to measured results in [`docs/status.md`](docs/status.md).

## Measured Results

### Test Environment

All measurements in this README and under [`docs/`](docs/) were taken on a single **NVIDIA DGX Spark** workstation:

| Component | Spec |
|---|---|
| Platform | NVIDIA DGX Spark |
| Superchip | NVIDIA **GB10** (Grace CPU + Blackwell GPU on one package) |
| CPU | 20 ARM cores (10x Cortex-X925 + 10x Cortex-A725) |
| GPU | Blackwell, 48 SMs, compute capability 12.1 (sm_121) |
| Memory | 128 GB LPDDR5X **unified** (shared between CPU and GPU) |
| Driver / CUDA | 580.126.09 / CUDA 13.0 |
| Software | PyTorch 2.11.0 + cu130, Flash Attention 2, HuggingFace Transformers 5.x |

DGX Spark is architecturally different from a typical discrete-GPU server: **CPU and GPU share the same physical memory pool**, so what the existing MegaTrain code calls "CPU→GPU transfer" is really a coherent memory copy, not a PCIe transaction. This affects how each improvement behaves:

- **Skip recompute** and **zero-copy unflatten** are pure GPU-compute / GPU-memory wins. Results should translate to any GPU system with similar relative compute-to-memcpy ratios.
- **FP8 weight transfer quantization** shows up as wall-clock negative here because there is no PCIe to save. On a discrete GPU with slow PCIe (e.g. PCIe 3.0), the trade-off flips.
- **Triple buffering** and the **grad worker pool** fall into the "insurance" bucket on DGX Spark because compute dominates over what little transfer cost exists. On a PCIe-bound discrete system they could become load-bearing.

The numbers below are therefore best read as a lower bound on discrete-GPU systems for the compute-focused improvements, and as a specific data point for DGX Spark overall.

### Headline: Qwen2.5-7B

Qwen2.5-7B, batch=2, seq=512, single GPU, RAM-safe benchmark (`--no-optimizer`):

| Metric | Upstream MegaTrain | MegaTrain-Plus | Δ |
|---|---|---|---|
| Step time | 3.604s | **2.539s** | **-29.6%** |
| Forward | 0.794s | **0.678s** | **-14.6%** |
| Backward | 2.600s | **1.653s** | **-36.4%** |
| Throughput (tok/s) | 284.1 | **403.3** | **+42.0%** |
| Peak GPU memory | 7.15 GB | **6.86 GB** | **-290 MB** |
| BWD/FWD ratio | 3.28x | **2.44x** | approaches theoretical ~2x |
| Loss trajectory | baseline | **bit-exact identical** | same |

Both **faster** AND **less GPU memory**. The reason memory goes down: the zero-copy unflatten optimization lets the layer templates alias the flat GPU buffer instead of holding their own copies, so `(1 + N_buffers) × layer_size` becomes `N_buffers × layer_size` per structure group.

### Multi-model suite

The same A/B run across **12 public models** covering three architectures (Qwen2.5, Qwen3, Llama via SmolLM2) from 360M to **32.76B** parameters. Full table in [`docs/suite_summary.md`](docs/suite_summary.md):

| Model | Params | Layers | Δ step time | Δ throughput | Δ backward | Loss match |
|---|---|---|---|---|---|---|
| SmolLM2-360M-Instruct | 0.36B | 32 | -22.8% | +29.5% | -36.8% | yes |
| Qwen2.5-0.5B-Instruct | 0.49B | 24 | -26.0% | +35.0% | -38.3% | yes |
| Qwen3-0.6B | 0.60B | 28 | -23.9% | +31.4% | -35.8% | yes |
| Qwen2.5-1.5B-Instruct | 1.54B | 28 | -25.2% | +33.6% | -36.7% | yes |
| SmolLM2-1.7B-Instruct | 1.71B | 24 | -28.9% | +40.6% | -38.5% | yes |
| Qwen3-1.7B | 1.72B | 28 | -26.9% | +36.7% | -39.7% | yes |
| Qwen2.5-3B-Instruct | 3.09B | 36 | -27.6% | +38.0% | -31.5% | yes |
| Qwen3-4B | 4.02B | 36 | -28.1% | +39.1% | -36.4% | yes |
| Qwen2.5-7B-Instruct | 7.62B | 28 | -29.6% | +42.0% | -36.3% | yes |
| Qwen3-8B | 8.19B | 36 | -29.2% | +41.2% | -35.9% | yes |
| **Qwen2.5-14B-Instruct** | **14.77B** | **48** | **-34.1%** | **+51.9%** | **-42.2%** | **yes** |
| **Qwen2.5-32B-Instruct** | **32.76B** | **64** | **-12.3%** | **+14.8%** | **-15.5%** | **yes** |

**Across all 12 models:**
- Step time: mean **-26.2%**, range -34.1% to -12.3%
- Throughput: mean **+36.2%**, range +14.8% to +51.9%
- Backward: mean **-35.3%**, range -42.2% to -15.5%
- **12/12 pass loss bit-exact match.**

The gain scales up through 14B (the sweet spot, where `-34.1%` step and `+51.9%` throughput). It tapers at 32B (`-12.3%`) because at that scale the baseline BWD/FWD ratio drops to ~1.3x - forward itself is so heavy that the eliminated recompute represents a smaller fraction of the total step. It is still a real positive win with bit-exact identical loss.

### Reproduce

Single-model run:
```bash
# Upstream-equivalent baseline
python scripts/benchmark.py --model Qwen/Qwen2.5-7B-Instruct \
    --batch-size 2 --seq-len 512 --steps 5 --no-optimizer \
    --num-buffers 2 --no-backward-prefetch --no-zero-copy-unflatten

# MegaTrain-Plus (defaults + `store_all_activations` opt-in)
python scripts/benchmark.py --model Qwen/Qwen2.5-7B-Instruct \
    --batch-size 2 --seq-len 512 --steps 5 --no-optimizer \
    --store-all-activations
```

Full multi-model suite (12 models, ~30 min total not counting downloads):
```bash
python scripts/benchmark_suite.py \
    --models "Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-1.5B-Instruct,Qwen/Qwen2.5-3B-Instruct,Qwen/Qwen2.5-7B-Instruct,Qwen/Qwen2.5-14B-Instruct,Qwen/Qwen2.5-32B-Instruct,Qwen/Qwen3-0.6B,Qwen/Qwen3-1.7B,Qwen/Qwen3-4B,Qwen/Qwen3-8B,HuggingFaceTB/SmolLM2-360M-Instruct,HuggingFaceTB/SmolLM2-1.7B-Instruct" \
    --batch-size 2 --seq-len 512 --steps 5

# Regenerate the summary table from existing per-model JSONs
python scripts/merge_suite_results.py
```

> **Note on 32B**: Needs ~100 GB of CPU RAM for weights + slabs + pinned buffers even with `--no-optimizer`. DGX Spark's 128 GB unified memory fits this; smaller systems may need a lower `num_grad_slabs` or a smaller model. 70B and larger do not fit in 128 GB at BF16 master precision.

Additional scaling data across batches and sequence lengths: [`docs/phases/phase3.md`](docs/phases/phase3.md), [`docs/phases/phase5.md`](docs/phases/phase5.md).

## Quick Start

```bash
# Install
git clone https://github.com/bdqnghi/MegaTrain-Plus.git
cd MegaTrain-Plus
pip install -e .

# Optional but recommended
pip install flash-attn                             # memory-efficient attention
pip install deepspeed                              # SIMD CPUAdam (needs `ninja`)
pip install ninja                                  # C++ extension compiler

# SFT: Qwen2.5-7B with MetaMathQA (downloads on first run)
python examples/sft/train.py --config examples/sft/configs/qwen_7b.yaml

# SFT: any supported model
python examples/sft/train.py --config examples/sft/configs/llama3_8b.yaml

# RL (GRPO): single-GPU via VERL + SGLang
CUDA_VISIBLE_DEVICES=0 bash examples/rl/run_qwen2_5_7b_megatrain.sh
```

Before increasing batch size, estimate what fits:
```bash
python scripts/calc_resource.py
```

## Config Reference

MegaTrain-Plus adds five config fields. Defaults reflect the measured results above:

```yaml
memory:
  checkpoint_interval: 4        # layers per gradient-checkpoint block
  num_grad_slabs: 12
  num_buffers: 3                # GPU flat buffer count (2=double, 3=triple)
  backward_prefetch: true       # prefetch next layer in the backward loops
  store_all_activations: true   # store every layer's input, skip recompute in backward (+25-30% throughput)
  zero_copy_unflatten: true     # template params alias the flat buffer; no per-layer memcpy

quantization:
  weight_transfer_dtype: "bfloat16"   # "float8_e4m3" opts into FP8 weight transfer (currently slower on commodity CPUs)
```

Unknown keys are ignored with defaults. Older MegaTrain configs work unchanged.

**Memory note**: `store_all_activations: true` adds `(N_layers - N_checkpoints) * B * T * H * 2` bytes of GPU memory for activation storage. On Qwen2.5-7B with `checkpoint_interval=4`, that's ~150 MB at batch=2/seq=512 and ~600 MB at batch=8/seq=512. Disable if your GPU is tight on headroom.

## Supported Models

MegaTrain-Plus inherits universal HuggingFace support from upstream MegaTrain. Any decoder-only LLM or vision-language model works through `AutoModelForCausalLM` / `AutoModelForImageTextToText` with automatic structure discovery.

Families that upstream MegaTrain supports: Qwen2/2.5/3/3.5, Qwen3-Next, Llama 2/3/4, Mistral, Mixtral, DeepSeek, Phi-3/4, Gemma 2/3, GLM-4/4.5, InternLM, Yi, Baichuan, GPT-OSS, plus VLMs (Qwen2-VL, Qwen2.5-VL, Qwen3-VL, LLaVA, InternVL, MiniCPM-V, Gemma 3 VL). See [`examples/sft/configs/`](examples/sft/configs/) for ready-made configs.

The MegaTrain-Plus improvements are **validated** on 12 models across three architectures (Qwen2.5, Qwen3, Llama via SmolLM2) from 0.36B to 32.76B. See [`docs/suite_summary.md`](docs/suite_summary.md) for the full table. 12/12 pass bit-exact loss match. Speedup varies with scale:
- Small to mid-size (0.36B to 14B): -22 to -34% step time, +29 to +52% throughput
- Large (32B, 64 layers): -12% step time, +15% throughput (tapers because baseline BWD/FWD ratio drops)

The wins are algorithmic and do not depend on the specific model family.

> **Note:** Phi-3 currently fails on the benchmark because transformers 5.x does not support Flash Attention 2 for that architecture yet. This is an upstream transformers limitation, not a MegaTrain-Plus issue. Phi-3 should work once FA2 support lands upstream, or today with `attn_implementation: "sdpa"`.

## RL Training (GRPO)

MegaTrain-Plus retains the full upstream VERL + SGLang RL integration. Single-GPU GRPO on Qwen2.5-7B or Qwen3.5-27B uses MegaTrain as the actor/reference training backend and SGLang (FP8) as the rollout engine, so all three models coexist on one GPU without weight reloading.

```bash
CUDA_VISIBLE_DEVICES=0 bash examples/rl/run_qwen2_5_7b_megatrain.sh
CUDA_VISIBLE_DEVICES=0 bash examples/rl/run_qwen3_5_27b_megatrain.sh
```

The skip-recompute and zero-copy-unflatten optimizations apply transparently to the actor training phase. The VERL engine code lives at [`verl/verl/workers/engine/megatrain/`](verl/verl/workers/engine/megatrain/).

## Data

LlamaFactory-compatible `dataset_info.json` registry. See [`data/README.md`](data/README.md) for the full list.

```yaml
dataset:
  name: "metamath"           # name from data/dataset_info.json
  dataset_dir: "data"
  max_seq_len: 1024
```

Supports alpaca format, sharegpt format, local JSON/JSONL, and HuggingFace Hub datasets.

## Documentation

Start at [`docs/README.md`](docs/README.md) for the full index. At a glance:

- Entry points: [`status.md`](docs/status.md), [`suite_summary.md`](docs/suite_summary.md), [`progress_summary.md`](docs/progress_summary.md)
- Deep-dives on each improvement, under [`docs/phases/`](docs/phases/):
  - [`phase1.md`](docs/phases/phase1.md): backward prefetching + triple buffering
  - [`phase1d.md`](docs/phases/phase1d.md): DataLoader fork artifact investigation
  - [`phase2.md`](docs/phases/phase2.md): FP8 weight transfer (why it isn't a win yet)
  - [`phase3.md`](docs/phases/phase3.md): **skip recompute** (the biggest win)
  - [`phase5.md`](docs/phases/phase5.md): **zero-copy unflatten** (second biggest)
- Historical context, under [`docs/plan/`](docs/plan/):
  - [`plan.md`](docs/plan/plan.md): original plan with retrospective
  - [`codex-critique.md`](docs/plan/codex-critique.md): Codex critique of the plan
  - [`gemini-analysis.md`](docs/plan/gemini-analysis.md): Gemini technical analysis
- Raw benchmark outputs under [`docs/benchmarks/`](docs/benchmarks/): JSONs grouped by phase, plus a `suite/` subdir with 24 per-model runs from the 12-model validation

## Profiling Tools

```bash
# End-to-end step timing benchmark with A/B flags
scripts/benchmark.py

# Measure _unflatten_to_layer cost vs layer forward compute (motivated the zero-copy unflatten work)
scripts/profile_unflatten.py

# Isolate per-layer backward compute vs pipeline overhead
scripts/profile_backward_scaling.py

# Measure CPU-side FP8 pack cost vs PCIe savings (motivated keeping FP8 weight transfer opt-in)
scripts/profile_quant_cost.py

# Correctness test for FP8 weight quantizer
scripts/test_weight_quant.py
```

## Troubleshooting

<details><summary><b>OOM when enabling <code>store_all_activations</code>?</b></summary>

At large batch or long sequence, the extra activation storage from `store_all_activations: true` can push past your GPU budget. Either:
- Disable: `store_all_activations: false` in your YAML
- Reduce `batch_size` or `max_seq_len`
- Increase `checkpoint_interval` (more layers per block → fewer stored activations)

</details>

<details><summary><b>First-step latency or random 20-second stalls?</b></summary>

Likely DataLoader worker re-fork. See [`docs/phases/phase1d.md`](docs/phases/phase1d.md). Fix already applied in `examples/sft/train.py` (`persistent_workers=True`). For custom training scripts, either match that setting or use `num_workers=0`.

</details>

<details><summary><b>HuggingFace Transformers 5.x compatibility?</b></summary>

Already handled: `torch_dtype=` → `dtype=` and `device_map="cpu"` → `.to("cpu")`. If you hit a transformers-version error, make sure you're on the MegaTrain-Plus fork, not the original upstream.

</details>

## Installation

```bash
git clone https://github.com/bdqnghi/MegaTrain-Plus.git
cd MegaTrain-Plus
pip install -e .

# Recommended
pip install flash-attn                             # memory-efficient attention
pip install deepspeed                              # SIMD CPUAdam
pip install ninja                                  # required by DeepSpeed JIT

# For RL (GRPO) training
pip install verl sglang[all]
```

## Citation

If you use MegaTrain-Plus in your work, please cite both the upstream paper and this fork:

```bibtex
@misc{yuan2026megatrainprecisiontraining100b,
      title={MegaTrain: Full Precision Training of 100B+ Parameter Large Language Models on a Single GPU},
      author={Zhengqing Yuan and Hanchi Sun and Lichao Sun and Yanfang Ye},
      year={2026},
      eprint={2604.05091},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.05091},
}

@misc{megatrainplus2026,
      title={MegaTrain-Plus: algorithmic improvements for single-GPU CPU-offloaded training},
      author={Bui, Nghi D. Q.},
      year={2026},
      url={https://github.com/bdqnghi/MegaTrain-Plus},
      note={Fork of MegaTrain with skip-recompute and zero-copy unflatten optimizations.
            Validated on NVIDIA DGX Spark across 12 models from 0.36B to 32.76B parameters
            with bit-exact identical loss.},
}
```

## Acknowledgements

- **[MegaTrain](https://github.com/DLYuanGod/MegaTrain)** (Yuan et al.): the underlying architecture this fork builds on. All credit for the RAM-centric CPU-offload design belongs to the original authors.
- **[VERL](https://github.com/verl-project/verl)**: RL post-training framework integrated for single-GPU GRPO.
- **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)**: dataset registry design that the data pipeline borrows from.
- **[HuggingFace Transformers](https://github.com/huggingface/transformers)**: universal model loading.
- **[Flash Attention](https://github.com/Dao-AILab/flash-attention)** / **[Flash Linear Attention](https://github.com/fla-org/flash-linear-attention)**: attention kernels.
- **[DeepSpeed](https://github.com/microsoft/DeepSpeed)**: CPUAdam.
- **[SGLang](https://github.com/sgl-project/sglang)**: FP8 rollout inference.

## License

Apache-2.0 (inherited from upstream MegaTrain). See [LICENSE](LICENSE).
