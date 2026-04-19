<div align="center">

# MegaTrain-Plus

### A faster, leaner fork of MegaTrain — full-precision 100B+ training on a single GPU

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)

[Improvements](#improvements) | [Quick Start](#quick-start) | [Results](#measured-results) | [Config Reference](#config-reference) | [Docs](#documentation)

</div>

---

## About

MegaTrain-Plus is a fork of [MegaTrain](https://arxiv.org/abs/2604.05091) (Yuan et al. 2026) — the RAM-centric single-GPU training framework that stores all parameters on CPU and streams them through GPU layer-by-layer. The original MegaTrain made it possible to train 100B+ models on a single GPU for SFT and RL post-training.

This fork keeps the architecture intact and adds a series of **algorithmic and engineering improvements** that collectively deliver a clean wall-clock speedup while actually *reducing* GPU memory footprint. Every improvement is behind a config flag (some default-on, some opt-in) and every wall-clock claim is backed by a reproducible benchmark JSON committed to [`docs/`](docs/).

**Both the user-facing API and the YAML config format are backward compatible** — existing MegaTrain configs and scripts work unchanged.

## Improvements

| Area | Change | Effect | Default |
|---|---|---|---|
| Backward pipeline | Prefetch next layer in the backward recompute and grad loops | -2 to -3% step time | ON |
| Buffer layout | Configurable GPU flat buffer count, default 3 (triple buffering) | Insurance for PCIe-bound regimes | ON |
| Grad accumulation | CPU worker pool for gradient accumulation (replaces single thread) | Insurance for grad-worker-bound regimes | ON |
| Activation memory | Store every layer's input during forward; skip the recompute block in backward | **-25 to -30% backward time** | opt-in |
| Unflatten | Pointer-swap: template params alias the flat GPU buffer directly, eliminating a ~6 ms/layer memcpy | **-8% step time and -440 MB GPU memory** | ON |
| Hot-path cleanup | `torch._foreach_copy_` fusion plus cached parameter lists | Cleaner, wall-clock neutral | ON |
| Weight transfer | FP8 E4M3 per-tensor quantization for CPU→GPU weight transfer | Correct but currently wall-clock NEGATIVE on commodity CPUs; kept for slower PCIe / future SIMD paths | opt-in |

The two biggest wins come from re-reading the hot path and asking *"what work are we doing that isn't actually necessary?"* rather than from PCIe, quantization, or multi-GPU categories:

1. **Skip recompute.** The backward pass was forward-computing each layer twice per step — once in the recompute loop and once again to build an autograd graph. Storing every layer's input during the initial forward lets the backward use those directly.
2. **Zero-copy unflatten.** Each layer compute was preceded by a 444 MB GPU→GPU memcpy from the flat buffer into a separate template. Pointing the template parameters' `.data` at views of the flat buffer removes the copy entirely and drops peak GPU memory.

Also included:
- **Reproducible benchmark harness** (`scripts/benchmark.py`) with A/B toggles for every improvement.
- **Profiling scripts** that isolate specific costs (unflatten memcpy, FP8 pack, backward scaling, quant overhead).
- **Retrospective** comparing plan predictions to measured results in [`docs/status.md`](docs/status.md).

## Measured Results

Qwen2.5-7B, batch=2, seq=512, single GPU, RAM-safe benchmark (no optimizer step):

| Metric | Upstream MegaTrain | MegaTrain-Plus | Δ |
|---|---|---|---|
| Step time | 3.604s | **2.539s** | **-29.6%** |
| Forward | 0.794s | **0.678s** | **-14.6%** |
| Backward | 2.600s | **1.653s** | **-36.4%** |
| Throughput (tok/s) | 284.1 | **403.3** | **+42.0%** |
| Peak GPU memory | 7.15 GB | **6.86 GB** | **-290 MB** |
| BWD/FWD ratio | 3.28x | **2.44x** | approaches theoretical ~2x |
| Loss trajectory | baseline | **bit-exact identical** | — |

Both **faster** AND **less GPU memory**. The reason memory goes down: Phase 5 lets the layer templates alias the flat GPU buffer instead of holding their own copies, so `(1 + N_buffers) × layer_size` becomes `N_buffers × layer_size` per structure group.

Reproduce:
```bash
# Upstream-equivalent baseline
python scripts/benchmark.py --model Qwen/Qwen2.5-7B-Instruct \
    --batch-size 2 --seq-len 512 --steps 5 --no-optimizer \
    --num-buffers 2 --no-backward-prefetch --no-zero-copy-unflatten

# MegaTrain-Plus (defaults + Phase 3 opt-in)
python scripts/benchmark.py --model Qwen/Qwen2.5-7B-Instruct \
    --batch-size 2 --seq-len 512 --steps 5 --no-optimizer \
    --store-all-activations
```

Additional scaling data across batches and sequence lengths: [`docs/phase3_results.md`](docs/phase3_results.md), [`docs/phase5_results.md`](docs/phase5_results.md).

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

# SFT — Qwen2.5-7B with MetaMathQA (downloads on first run)
python examples/sft/train.py --config examples/sft/configs/qwen_7b.yaml

# SFT — any supported model
python examples/sft/train.py --config examples/sft/configs/llama3_8b.yaml

# RL (GRPO) — single-GPU via VERL + SGLang
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
  num_buffers: 3                # Phase 1B: GPU flat buffer count
  backward_prefetch: true       # Phase 1A: prefetch next layer in backward
  store_all_activations: true   # Phase 3: skip recompute (+25-30% throughput)
  zero_copy_unflatten: true     # Phase 5: template params alias flat buffer

quantization:
  weight_transfer_dtype: "bfloat16"   # "float8_e4m3" = Phase 2 (opt-in, currently slower)
```

Unknown keys are ignored with defaults — older MegaTrain configs work unchanged.

**Memory note**: `store_all_activations: true` adds `(N_layers - N_checkpoints) * B * T * H * 2` bytes of GPU memory for activation storage. On Qwen2.5-7B with `checkpoint_interval=4`, that's ~150 MB at batch=2/seq=512 and ~600 MB at batch=8/seq=512. Disable if your GPU is tight on headroom.

## Supported Models

MegaTrain-Plus inherits universal HuggingFace support from upstream MegaTrain — any decoder-only LLM or vision-language model works through `AutoModelForCausalLM` / `AutoModelForImageTextToText` with automatic structure discovery.

Tested families: Qwen2/2.5/3/3.5, Qwen3-Next, Llama 2/3/4, Mistral, Mixtral, DeepSeek, Phi-3/4, Gemma 2/3, GLM-4/4.5, InternLM, Yi, Baichuan, GPT-OSS, plus VLMs (Qwen2-VL, Qwen2.5-VL, Qwen3-VL, LLaVA, InternVL, MiniCPM-V, Gemma 3 VL). See [`examples/sft/configs/`](examples/sft/configs/) for ready-made configs.

## RL Training (GRPO)

MegaTrain-Plus retains the full upstream VERL + SGLang RL integration. Single-GPU GRPO on Qwen2.5-7B or Qwen3.5-27B uses MegaTrain as the actor/reference training backend and SGLang (FP8) as the rollout engine — all three models coexist on one GPU without weight reloading.

```bash
CUDA_VISIBLE_DEVICES=0 bash examples/rl/run_qwen2_5_7b_megatrain.sh
CUDA_VISIBLE_DEVICES=0 bash examples/rl/run_qwen3_5_27b_megatrain.sh
```

MegaTrain-Plus's Phase 3/5 improvements apply transparently to the actor training phase. The VERL engine code lives at [`verl/verl/workers/engine/megatrain/`](verl/verl/workers/engine/megatrain/).

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

The `docs/` directory is organized as follows:

- **[status.md](docs/status.md)** — single source of truth, recommended entry point
- **[progress_summary.md](docs/progress_summary.md)** — running tally of what's landed
- **Per-phase deep-dives**:
  - [phase1_results.md](docs/phase1_results.md) — Phase 1A/1B
  - [phase1d_results.md](docs/phase1d_results.md) — DataLoader fork artifact investigation
  - [phase2_results.md](docs/phase2_results.md) — FP8 weight transfer (why it isn't a win yet)
  - [phase3_results.md](docs/phase3_results.md) — **skip recompute (-30% backward)**
  - [phase5_results.md](docs/phase5_results.md) — **zero-copy unflatten (-8% step, -440 MB)**
- **Historical context**:
  - [megatrain-plus-plan.md](docs/megatrain-plus-plan.md) — original plan with retrospective
  - [megatrain-plus-critique.md](docs/megatrain-plus-critique.md) — Codex critique of the plan
  - [gemini_comment.md](docs/gemini_comment.md) — Gemini technical analysis
- **Benchmark JSONs**: ~25 files covering every wall-clock claim made in the docs

## Profiling Tools

```bash
# End-to-end step timing benchmark with A/B flags
scripts/benchmark.py

# Measure _unflatten_to_layer cost vs layer forward compute (motivated Phase 5)
scripts/profile_unflatten.py

# Isolate per-layer backward compute vs pipeline overhead
scripts/profile_backward_scaling.py

# Measure CPU-side FP8 pack cost vs PCIe savings (motivated Phase 2's "opt-in" status)
scripts/profile_quant_cost.py

# Correctness test for FP8 weight quantizer
scripts/test_weight_quant.py
```

## Troubleshooting

<details><summary><b>OOM when enabling <code>store_all_activations</code>?</b></summary>

At large batch / long sequence, Phase 3's extra activation storage can push past your GPU budget. Either:
- Disable: `store_all_activations: false` in your YAML
- Reduce `batch_size` or `max_seq_len`
- Increase `checkpoint_interval` (more layers per block → fewer stored activations)

</details>

<details><summary><b>First-step latency or random 20-second stalls?</b></summary>

Likely DataLoader worker re-fork. See [phase1d_results.md](docs/phase1d_results.md). Fix already applied in `examples/sft/train.py` (`persistent_workers=True`). For custom training scripts, either match that setting or use `num_workers=0`.

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
      title={MegaTrain-Plus: algorithmic improvements for single-GPU 100B+ training},
      author={Bui, Nghi D. Q.},
      year={2026},
      url={https://github.com/bdqnghi/MegaTrain-Plus},
      note={Fork of MegaTrain with -30\% step time and -290 MB GPU memory at bit-exact identical loss.},
}
```

## Acknowledgements

- **[MegaTrain](https://github.com/DLYuanGod/MegaTrain)** (Yuan et al.) — the underlying architecture this fork builds on. All credit for the RAM-centric CPU-offload design belongs to the original authors.
- **[VERL](https://github.com/verl-project/verl)** — RL post-training framework integrated for single-GPU GRPO.
- **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)** — dataset registry design that the data pipeline borrows from.
- **[HuggingFace Transformers](https://github.com/huggingface/transformers)** — universal model loading.
- **[Flash Attention](https://github.com/Dao-AILab/flash-attention)** / **[Flash Linear Attention](https://github.com/fla-org/flash-linear-attention)** — attention kernels.
- **[DeepSpeed](https://github.com/microsoft/DeepSpeed)** — CPUAdam.
- **[SGLang](https://github.com/sgl-project/sglang)** — FP8 rollout inference.

## License

Apache-2.0 (inherited from upstream MegaTrain). See [LICENSE](LICENSE).
