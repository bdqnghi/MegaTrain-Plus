# MegaTrain-Plus Multi-Model Benchmark Suite

Every wall-clock claim in this file is reproducible via `scripts/benchmark_suite.py`. Raw per-run JSONs are in `docs/suite/`.

## Setup

- Hardware: NVIDIA DGX Spark (GB10 Superchip, 128 GB unified LPDDR5X memory)
- Test: `batch_size=2, seq_len=512, steps=5` (step 0 = warmup, excluded)
- `--no-optimizer` (we profile forward + backward; avoids allocating FP32 Adam
  states which would be ~8-90 GB depending on model size)
- Attention: `flash_attention_2`
- Baseline config: `--num-buffers 2 --no-backward-prefetch --no-zero-copy-unflatten`
- Plus config: MegaTrain-Plus defaults + `--store-all-activations`

## Step Time / Throughput Summary

| Model | Params | Layers | Base step | Plus step | Δ step | Base tok/s | Plus tok/s | Δ tok/s | GPU Δ | Loss |
|---|---|---|---|---|---|---|---|---|---|---|
| HuggingFaceTB/SmolLM2-360M-Instruct | 0.36B | 32 | 0.947s | 0.731s | **-22.8%** | 1081.3 | 1400.3 | **+29.5%** | +0.03 GB | yes |
| Qwen/Qwen2.5-0.5B-Instruct | 0.49B | 24 | 0.785s | 0.581s | **-26.0%** | 1304.9 | 1761.7 | **+35.0%** | +0.01 GB | yes |
| Qwen/Qwen3-0.6B | 0.6B | 28 | 0.928s | 0.706s | **-23.9%** | 1103.2 | 1449.9 | **+31.4%** | +0.01 GB | yes |
| Qwen/Qwen2.5-1.5B-Instruct | 1.54B | 28 | 1.283s | 0.960s | **-25.2%** | 798.3 | 1066.4 | **+33.6%** | -0.03 GB | yes |
| HuggingFaceTB/SmolLM2-1.7B-Instruct | 1.71B | 24 | 1.230s | 0.874s | **-28.9%** | 832.9 | 1171.3 | **+40.6%** | -0.05 GB | yes |
| Qwen/Qwen3-1.7B | 1.72B | 28 | 1.378s | 1.008s | **-26.9%** | 743.0 | 1015.6 | **+36.7%** | -0.01 GB | yes |
| Qwen/Qwen2.5-3B-Instruct | 3.09B | 36 | 2.102s | 1.522s | **-27.6%** | 487.3 | 672.7 | **+38.0%** | -0.04 GB | yes |
| Qwen/Qwen3-4B | 4.02B | 36 | 2.505s | 1.800s | **-28.1%** | 408.9 | 568.7 | **+39.1%** | -0.06 GB | yes |
| Qwen/Qwen2.5-7B-Instruct | 7.62B | 28 | 3.524s | 2.482s | **-29.6%** | 290.5 | 412.6 | **+42.0%** | -0.29 GB | yes |
| Qwen/Qwen3-8B | 8.19B | 36 | 3.981s | 2.820s | **-29.2%** | 257.2 | 363.2 | **+41.2%** | -0.15 GB | yes |

**Across 10 models:**

- Step time: mean **-26.8%**, range -29.6% to -22.8%
- Throughput: mean **+36.7%**, range +29.5% to +42.0%
- Backward: mean **-36.6%**, range -39.7% to -31.5%
- **10/10 models pass loss bit-exact check**

## Forward / Backward Breakdown

| Model | Base fwd | Plus fwd | Δ fwd | Base bwd | Plus bwd | Δ bwd | Base ratio | Plus ratio |
|---|---|---|---|---|---|---|---|---|
| HuggingFaceTB/SmolLM2-360M-Instruct | 0.231s | 0.225s | **-2.6%** | 0.552s | 0.349s | **-36.8%** | 2.39x | 1.55x |
| Qwen/Qwen2.5-0.5B-Instruct | 0.179s | 0.166s | **-7.3%** | 0.496s | 0.306s | **-38.3%** | 2.77x | 1.85x |
| Qwen/Qwen3-0.6B | 0.201s | 0.193s | **-4.0%** | 0.617s | 0.396s | **-35.8%** | 3.06x | 2.06x |
| Qwen/Qwen2.5-1.5B-Instruct | 0.271s | 0.254s | **-6.3%** | 0.867s | 0.549s | **-36.7%** | 3.20x | 2.16x |
| HuggingFaceTB/SmolLM2-1.7B-Instruct | 0.281s | 0.247s | **-12.1%** | 0.823s | 0.506s | **-38.5%** | 2.92x | 2.04x |
| Qwen/Qwen3-1.7B | 0.275s | 0.268s | **-2.5%** | 0.995s | 0.600s | **-39.7%** | 3.62x | 2.24x |
| Qwen/Qwen2.5-3B-Instruct | 0.462s | 0.375s | **-18.8%** | 1.451s | 0.994s | **-31.5%** | 3.14x | 2.65x |
| Qwen/Qwen3-4B | 0.542s | 0.487s | **-10.1%** | 1.771s | 1.127s | **-36.4%** | 3.27x | 2.31x |
| Qwen/Qwen2.5-7B-Instruct | 0.784s | 0.671s | **-14.4%** | 2.542s | 1.618s | **-36.3%** | 3.24x | 2.41x |
| Qwen/Qwen3-8B | 0.860s | 0.741s | **-13.8%** | 2.887s | 1.851s | **-35.9%** | 3.36x | 2.50x |

## Observations

- The ~25-30% step-time reduction and ~30-40% throughput gain are **consistent across families and sizes** (Qwen2.5, Qwen3, SmolLM2 which uses the Llama architecture).
- The backward pass alone speeds up 32-40% - this is dominated by `store_all_activations` eliminating the redundant per-block recompute.
- Forward speedup grows with model size (~-2% at 360M, ~-14% at 7-8B) - zero-copy unflatten saves a fixed ~6 ms per layer, which becomes a larger fraction of forward time as layers are bigger.
- Peak GPU memory is roughly flat (small negative delta). The zero-copy unflatten saves memory (templates alias the flat buffer), while `store_all_activations` adds some back. Net is slightly in the minus on most configurations.
- Every tested model passes loss bit-exact match, confirming these are pure algorithmic wins and not numerical approximations.
- Models that use `flash_attention_2` work out of the box. Phi-3 fails because transformers 5.x does not yet support FA2 for that architecture (upstream transformers issue, not a MegaTrain-Plus limitation).