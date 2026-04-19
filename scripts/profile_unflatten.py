"""Measure the cost of _unflatten_to_layer vs the rest of the forward pass.

Isolates the per-layer GPU memcpy cost (24 params x memcpy each) to see if
replacing it with a pointer-swap (zero-copy) would materially speed things up.
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from infinity import CPUMasterModel
from infinity.config import CPUMasterConfig


def _cuda_sync():
    torch.cuda.synchronize()


def main():
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="flash_attention_2",
    ).to("cpu")

    config = CPUMasterConfig(
        model_name=model_name, batch_size=2, max_seq_len=512, num_steps=1,
        dataset_name="alpaca_en_demo", dataset_dir="data",
    )
    model = CPUMasterModel(hf_model, config)
    del hf_model

    # Load layer 0 into buffer 0
    model._load_layer_to_buffer_async(0, 0)
    model.weight_stream.synchronize()
    _cuda_sync()

    # Time just the unflatten
    ITERS = 50
    for _ in range(5):  # warmup
        with torch.cuda.stream(model.compute_stream):
            model._unflatten_to_layer(0, 0)
    _cuda_sync()

    t0 = time.perf_counter()
    for _ in range(ITERS):
        with torch.cuda.stream(model.compute_stream):
            model._unflatten_to_layer(0, 0)
    _cuda_sync()
    t_unflatten = (time.perf_counter() - t0) / ITERS

    # Time just the layer forward
    B, T = 2, 512
    H = model.hidden_size
    hidden = torch.randn(B, T, H, dtype=torch.bfloat16, device=model.device)

    # Set up layer kwargs
    mask = torch.ones(B, T, dtype=torch.long, device=model.device)
    cache_position = torch.arange(T, device=model.device)
    position_ids = torch.arange(T, device=model.device).unsqueeze(0).expand(B, -1)
    position_embeddings = None
    if model.rotary_gpu and model.layer_accepts_position_embeddings:
        dummy = torch.empty((1, 1, T, model.head_dim), device=model.device, dtype=torch.float32)
        cos, sin = model.rotary_gpu(dummy, position_ids[:1])
        position_embeddings = (cos.to(torch.bfloat16), sin.to(torch.bfloat16))
    layer_kwargs = model._build_layer_kwargs(mask, cache_position, position_ids, position_embeddings)

    gpu_layer = model._get_gpu_layer(0, 0)
    for _ in range(3):  # warmup
        with torch.cuda.stream(model.compute_stream):
            with torch.no_grad():
                _ = gpu_layer(hidden, **layer_kwargs)
    _cuda_sync()

    t0 = time.perf_counter()
    for _ in range(ITERS):
        with torch.cuda.stream(model.compute_stream):
            with torch.no_grad():
                _ = gpu_layer(hidden, **layer_kwargs)
    _cuda_sync()
    t_forward = (time.perf_counter() - t0) / ITERS

    print(f"Qwen2.5-7B, batch={B}, seq={T}")
    print(f"  _unflatten_to_layer  : {t_unflatten*1000:>6.3f} ms/iter")
    print(f"  gpu_layer(...)  fwd  : {t_forward*1000:>6.3f} ms/iter")
    print(f"  ratio                : {t_unflatten/t_forward:>6.2%}")
    print()
    print(f"For 28 layers × 3 passes/step (fwd + recompute + bwd):")
    print(f"  baseline unflatten total : {t_unflatten*1000*28*3:.0f} ms/step")
    print(f"  with Phase 3 (2 passes)  : {t_unflatten*1000*28*2:.0f} ms/step")

    model.cleanup()


if __name__ == "__main__":
    main()
