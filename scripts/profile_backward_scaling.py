"""Investigate why backward blows up super-linearly with batch size.

Hypotheses:
  H1. Flash Attention backward has non-linear cost for some shapes.
  H2. GPU allocator churn / fragmentation at larger batches.
  H3. torch.autograd.grad with many inputs has overhead that scales weirdly.
  H4. template_free_events / buffer contention worsens at longer compute.

Approach: measure per-block recompute vs backward times, plus the single-layer
breakdown (unflatten, forward, autograd.grad). Run at batch=2 and batch=4 and
compare the per-layer profile.
"""

import argparse
import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from infinity import CPUMasterModel, ChatDataset, collate_fn
from infinity.config import CPUMasterConfig


def _cuda_sync():
    torch.cuda.synchronize()


def measure_one_layer_backward(model, gpu_layer, layer_input, grad_hidden, layer_kwargs, iters=3):
    """Run only the autograd.grad path on a single layer; isolate its cost."""
    # Warmup
    for _ in range(2):
        for p in gpu_layer.parameters():
            p.requires_grad_(True)
        out = gpu_layer(layer_input, **layer_kwargs)
        layer_output = out[0] if isinstance(out, tuple) else out
        grads = torch.autograd.grad(
            outputs=layer_output,
            inputs=(layer_input,) + tuple(gpu_layer.parameters()),
            grad_outputs=grad_hidden,
            retain_graph=False, create_graph=False, allow_unused=True,
        )
        del out, layer_output, grads
        for p in gpu_layer.parameters():
            p.requires_grad_(False)
    _cuda_sync()

    # Measure forward alone
    t_fwd = 0.0
    for _ in range(iters):
        _cuda_sync()
        t0 = time.perf_counter()
        for p in gpu_layer.parameters():
            p.requires_grad_(True)
        out = gpu_layer(layer_input, **layer_kwargs)
        layer_output = out[0] if isinstance(out, tuple) else out
        _cuda_sync()
        t_fwd += time.perf_counter() - t0
        # Run grad to consume the graph and free memory
        grads = torch.autograd.grad(
            outputs=layer_output,
            inputs=(layer_input,) + tuple(gpu_layer.parameters()),
            grad_outputs=grad_hidden,
            retain_graph=False, create_graph=False, allow_unused=True,
        )
        del out, layer_output, grads
        for p in gpu_layer.parameters():
            p.requires_grad_(False)

    # Measure forward + autograd.grad together
    t_full = 0.0
    for _ in range(iters):
        _cuda_sync()
        t0 = time.perf_counter()
        for p in gpu_layer.parameters():
            p.requires_grad_(True)
        out = gpu_layer(layer_input, **layer_kwargs)
        layer_output = out[0] if isinstance(out, tuple) else out
        grads = torch.autograd.grad(
            outputs=layer_output,
            inputs=(layer_input,) + tuple(gpu_layer.parameters()),
            grad_outputs=grad_hidden,
            retain_graph=False, create_graph=False, allow_unused=True,
        )
        _cuda_sync()
        t_full += time.perf_counter() - t0
        del out, layer_output, grads
        for p in gpu_layer.parameters():
            p.requires_grad_(False)

    # Measure forward with no_grad (pure recompute cost, no graph build)
    t_nograd = 0.0
    for _ in range(iters):
        _cuda_sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = gpu_layer(layer_input, **layer_kwargs)
            _ = out[0] if isinstance(out, tuple) else out
        _cuda_sync()
        t_nograd += time.perf_counter() - t0
        del out

    return t_fwd / iters, t_full / iters, t_nograd / iters


def run(args):
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation=args.attn,
    ).to("cpu")

    config = CPUMasterConfig(
        model_name=args.model,
        batch_size=args.batch_size,
        max_seq_len=args.seq_len,
        num_steps=2,
        dataset_name="alpaca_en_demo",
        dataset_dir="data",
        num_buffers=2,
    )
    model = CPUMasterModel(hf_model, config)
    del hf_model

    dataset = ChatDataset(tokenizer, config.max_seq_len, dataset_name="alpaca_en_demo", dataset_dir="data")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=0)
    batch = next(iter(dataloader))

    # Prime one forward pass to set up all buffers and get a valid layer_input
    model.forward_and_backward(batch["input_ids"], batch["attention_mask"], batch["labels"])
    _cuda_sync()

    # For the single-layer benchmark we just need a plausible hidden state of the right shape
    B, T = batch["input_ids"].shape
    H = model.hidden_size
    layer_input = torch.randn(B, T, H, dtype=torch.bfloat16, device=model.device, requires_grad=True)
    grad_hidden = torch.randn_like(layer_input).detach()

    # Grab a representative layer (load into buffer 0)
    model._load_layer_to_buffer_async(0, 0)
    model.weight_stream.synchronize()
    model._unflatten_to_layer(0, 0)
    _cuda_sync()
    gpu_layer = model._get_gpu_layer(0, 0)

    # Build layer_kwargs for this model
    mask = batch["attention_mask"].to(model.device)
    cache_position = torch.arange(T, device=model.device)
    position_ids = torch.arange(T, device=model.device).unsqueeze(0).expand(B, -1)
    position_embeddings = None
    if model.rotary_gpu and model.layer_accepts_position_embeddings:
        dummy = torch.empty((1, 1, T, model.head_dim), device=model.device, dtype=torch.float32)
        cos, sin = model.rotary_gpu(dummy, position_ids[:1])
        position_embeddings = (cos.to(torch.bfloat16), sin.to(torch.bfloat16))
    layer_kwargs = model._build_layer_kwargs(mask, cache_position, position_ids, position_embeddings)

    t_fwd, t_full, t_nograd = measure_one_layer_backward(
        model, gpu_layer, layer_input, grad_hidden, layer_kwargs,
        iters=args.iters,
    )
    t_grad_only = t_full - t_fwd

    print(f"\nModel={args.model}  batch={args.batch_size}  seq={args.seq_len}  attn={args.attn}")
    print(f"  {'forward (grad on, no grad call)':<40} {t_fwd*1000:>8.2f} ms")
    print(f"  {'forward (no_grad, pure compute)':<40} {t_nograd*1000:>8.2f} ms")
    print(f"  {'forward + autograd.grad (full)':<40} {t_full*1000:>8.2f} ms")
    print(f"  {'autograd.grad (full - forward)':<40} {t_grad_only*1000:>8.2f} ms")
    print(f"  {'per-token fwd (pure compute)':<40} {t_nograd*1000 / (B*T):>8.3f} ms")
    print(f"  {'per-token grad (full - forward)':<40} {t_grad_only*1000 / (B*T):>8.3f} ms")

    model.cleanup()
    return t_fwd, t_full, t_nograd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--attn", type=str, default="flash_attention_2",
                        choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--iters", type=int, default=3)
    args = parser.parse_args()
    run(args)
