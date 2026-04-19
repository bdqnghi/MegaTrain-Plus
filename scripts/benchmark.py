"""
MegaTrain Benchmark Script — Per-phase timing breakdown for profiling.

Measures:
  - Forward pass time (with prefetching overlap)
  - Backward pass time (recompute + grad computation)
  - Optimizer step time
  - Forward vs backward asymmetry
  - Step-level timing

Usage:
    python scripts/benchmark.py --model Qwen/Qwen2.5-7B-Instruct --steps 5 --batch-size 4

The first step is a warmup (JIT, allocations). Steps 2+ are measured.
"""

import argparse
import json
import logging
import os
import time

import psutil
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

from infinity import CPUMasterModel, ChatDataset, collate_fn
from infinity.config import CPUMasterConfig

try:
    from deepspeed.ops.adam import DeepSpeedCPUAdam
    CPU_ADAM_AVAILABLE = True
except ImportError:
    CPU_ADAM_AVAILABLE = False


def run_benchmark(args):
    config_kwargs = dict(
        model_name=args.model,
        batch_size=args.batch_size,
        max_seq_len=args.seq_len,
        num_steps=args.steps,
        dataset_name="alpaca_en_demo",
        dataset_dir="data",
        checkpoint_interval=args.checkpoint_interval,
        num_grad_slabs=args.num_grad_slabs,
    )
    if args.num_buffers is not None:
        config_kwargs["num_buffers"] = args.num_buffers
    if args.no_backward_prefetch:
        config_kwargs["backward_prefetch"] = False
    if args.weight_transfer_dtype is not None:
        config_kwargs["weight_transfer_dtype"] = args.weight_transfer_dtype
    if args.block_timing:
        config_kwargs["diagnostic_block_timing"] = True
    if args.store_all_activations:
        config_kwargs["store_all_activations"] = True
    if args.zero_copy_unflatten:
        config_kwargs["zero_copy_unflatten"] = True
    if args.no_zero_copy_unflatten:
        config_kwargs["zero_copy_unflatten"] = False
    config = CPUMasterConfig(**config_kwargs)

    logger.info(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading model: {args.model}")
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=config.dtype, trust_remote_code=True,
        attn_implementation="flash_attention_2",
    ).to("cpu")

    logger.info("Creating CPUMasterModel...")
    model = CPUMasterModel(hf_model, config)
    del hf_model

    num_params = sum(p.numel() for p in model.get_parameters())
    num_layers = len(model.cpu_layers)
    logger.info(f"Model: {num_params/1e9:.2f}B params, {num_layers} layers")

    # Optimizer: skip entirely if --no-optimizer flag (we profile fwd+bwd only).
    # For 7B+ models, PyTorch AdamW allocates ~12 B/param on CPU for FP32 states.
    optimizer = None
    if not args.no_optimizer:
        optimizer = torch.optim.AdamW(model.get_parameters(), lr=1e-5, weight_decay=0.01)

    # Dataset
    dataset = ChatDataset(tokenizer, config.max_seq_len, dataset_name="alpaca_en_demo", dataset_dir="data")
    # num_workers=0: avoid fork-based dataloader workers inheriting our 40+ GB process image.
    # Re-forking those workers at StopIteration causes 20+ second stalls.
    dataloader = DataLoader(dataset, batch_size=config.batch_size, collate_fn=collate_fn, num_workers=0)
    data_iter = iter(dataloader)

    # Results
    results = {
        "model": args.model,
        "num_params_B": round(num_params / 1e9, 2),
        "num_layers": num_layers,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "checkpoint_interval": args.checkpoint_interval,
        "num_buffers": model.num_buffers,
        "backward_prefetch": config.backward_prefetch,
        "store_all_activations": config.store_all_activations,
        "weight_transfer_dtype": config.weight_transfer_dtype,
        "skip_optimizer": optimizer is None,
        "steps": [],
    }

    process = psutil.Process()
    torch.cuda.reset_peak_memory_stats()

    logger.info("=" * 70)
    logger.info(f"Running {args.steps} steps (step 0 = warmup)...")
    logger.info("=" * 70)

    for step in range(args.steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        torch.cuda.synchronize()
        step_start = time.perf_counter()

        # Forward + backward (timed internally by CPUMasterModel)
        loss_val, n_tokens, timing = model.forward_and_backward(
            batch["input_ids"], batch["attention_mask"], batch["labels"]
        )

        torch.cuda.synchronize()
        fwd_bwd_end = time.perf_counter()

        # Optimizer step (skipped with --no-optimizer for safe profiling)
        opt_start = time.perf_counter()
        if optimizer is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.get_parameters(), 1.0)
            optimizer.step()
            model._sync_params_to_gpu()
            model.zero_grad()
            optimizer.zero_grad()
        else:
            grad_norm = torch.tensor(float('nan'))
            model.zero_grad()
        torch.cuda.synchronize()
        opt_end = time.perf_counter()

        step_end = time.perf_counter()

        step_data = {
            "step": step,
            "loss": round(loss_val, 4),
            "tokens": n_tokens,
            "step_time_s": round(step_end - step_start, 3),
            "fwd_time_s": round(timing['forward'], 3),
            "bwd_time_s": round(timing['backward'], 3),
            "fwd_bwd_total_s": round(fwd_bwd_end - step_start, 3),
            "optimizer_time_s": round(opt_end - opt_start, 3),
            "gpu_mem_GB": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
            "cpu_mem_GB": round(process.memory_info().rss / 1024**3, 2),
            "grad_norm": round(grad_norm.item(), 4) if torch.isfinite(grad_norm) else None,
        }
        results["steps"].append(step_data)

        tag = "WARMUP" if step == 0 else f"step {step}"
        logger.info(
            f"[{tag}] loss={loss_val:.4f} | "
            f"total={step_data['step_time_s']:.2f}s | "
            f"fwd={timing['forward']:.2f}s | bwd={timing['backward']:.2f}s | "
            f"opt={step_data['optimizer_time_s']:.2f}s | "
            f"GPU={step_data['gpu_mem_GB']:.1f}GB"
        )

        if args.block_timing and step > 0 and getattr(model, "_last_bwd_block_times", None):
            for blk, t_rc, t_bw in model._last_bwd_block_times:
                logger.info(f"    block {blk}: recompute={t_rc*1000:.1f}ms  backward={t_bw*1000:.1f}ms")

    # Summary (exclude warmup step 0)
    measured = results["steps"][1:] if len(results["steps"]) > 1 else results["steps"]
    if measured:
        avg_step = sum(s["step_time_s"] for s in measured) / len(measured)
        avg_fwd = sum(s["fwd_time_s"] for s in measured) / len(measured)
        avg_bwd = sum(s["bwd_time_s"] for s in measured) / len(measured)
        avg_opt = sum(s["optimizer_time_s"] for s in measured) / len(measured)
        bwd_fwd_ratio = avg_bwd / avg_fwd if avg_fwd > 0 else 0

        summary = {
            "avg_step_time_s": round(avg_step, 3),
            "avg_fwd_time_s": round(avg_fwd, 3),
            "avg_bwd_time_s": round(avg_bwd, 3),
            "avg_optimizer_time_s": round(avg_opt, 3),
            "bwd_to_fwd_ratio": round(bwd_fwd_ratio, 2),
            "fwd_pct": round(avg_fwd / avg_step * 100, 1),
            "bwd_pct": round(avg_bwd / avg_step * 100, 1),
            "opt_pct": round(avg_opt / avg_step * 100, 1),
            "peak_gpu_GB": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
            "peak_cpu_GB": round(max(s["cpu_mem_GB"] for s in measured), 2),
            "tokens_per_sec": round(sum(s["tokens"] for s in measured) / sum(s["step_time_s"] for s in measured), 1),
        }
        results["summary"] = summary

        logger.info("")
        logger.info("=" * 70)
        logger.info("BENCHMARK RESULTS (excluding warmup)")
        logger.info("=" * 70)
        logger.info(f"  Avg step time:  {avg_step:.3f}s")
        logger.info(f"  Avg forward:    {avg_fwd:.3f}s  ({summary['fwd_pct']}%)")
        logger.info(f"  Avg backward:   {avg_bwd:.3f}s  ({summary['bwd_pct']}%)")
        logger.info(f"  Avg optimizer:  {avg_opt:.3f}s  ({summary['opt_pct']}%)")
        logger.info(f"  BWD/FWD ratio:  {bwd_fwd_ratio:.2f}x")
        logger.info(f"  Tokens/sec:     {summary['tokens_per_sec']:.1f}")
        logger.info(f"  Peak GPU mem:   {summary['peak_gpu_GB']:.2f} GB")
        logger.info(f"  Peak CPU mem:   {summary['peak_cpu_GB']:.2f} GB")

    # Save results
    out_path = args.output or "docs/benchmarks/baseline_profile.json"
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")

    model.cleanup()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MegaTrain benchmark")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--steps", type=int, default=5, help="Total steps (step 0 = warmup)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--checkpoint-interval", type=int, default=4)
    parser.add_argument("--num-grad-slabs", type=int, default=12)
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--no-optimizer", action="store_true",
                        help="Skip optimizer.step() to avoid allocating FP32 Adam states (safer for large models).")
    parser.add_argument("--num-buffers", type=int, default=None,
                        help="Override config num_buffers (2=double, 3=triple buffering).")
    parser.add_argument("--no-backward-prefetch", action="store_true",
                        help="Disable Phase 1A backward prefetch (A/B test baseline).")
    parser.add_argument("--weight-transfer-dtype", type=str, default=None,
                        choices=["bfloat16", "float8_e4m3"],
                        help="Phase 2 FP8 weight transfer quantization.")
    parser.add_argument("--block-timing", action="store_true",
                        help="Print per-block recompute/backward times (Phase 1D diagnostic).")
    parser.add_argument("--store-all-activations", action="store_true",
                        help="Phase 3: store every layer's input during forward; skip recompute in backward.")
    parser.add_argument("--zero-copy-unflatten", action="store_true",
                        help="Phase 5: pointer-swap template params to flat buffer views; skip unflatten memcpy.")
    parser.add_argument("--no-zero-copy-unflatten", action="store_true",
                        help="Disable Phase 5 (for A/B vs baseline).")
    args = parser.parse_args()
    run_benchmark(args)
