"""Measure where time is spent in CPU-side FP8 quantization.

Compares:
  - baseline: just flatten + copy into pinned buffer (what original MegaTrain does)
  - fp8 quant: current WeightTransferQuantizer
  - optimized: lower-overhead alternatives

Uses fake data sized like a Qwen2.5-7B layer (444 MB worth of params).
"""

import time
import torch

from infinity.quantization import WeightTransferQuantizer, parse_transfer_dtype


def make_qwen7b_layer():
    # Roughly matches Qwen2.5-7B decoder layer (per-param numel adds to ~222M).
    shapes = [
        (3584,),             # input_layernorm
        (3584, 3584),        # q_proj
        (512, 3584),         # k_proj
        (512, 3584),         # v_proj
        (3584, 3584),        # o_proj
        (3584,),             # post_attn_ln
        (18944, 3584),       # gate_proj
        (18944, 3584),       # up_proj
        (3584, 18944),       # down_proj
    ]
    return [torch.randn(s, dtype=torch.bfloat16) for s in shapes]


def bench(name: str, fn, warmup: int = 2, runs: int = 5) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    elapsed = (time.perf_counter() - t0) / runs
    print(f"  {name:<40} {elapsed*1000:>8.2f} ms/iter")
    return elapsed


def main():
    torch.manual_seed(0)
    params = make_qwen7b_layer()
    numels = [p.numel() for p in params]
    total = sum(numels)
    print(f"Layer: {len(params)} tensors, {total/1e6:.1f}M params ({total*2/1e6:.1f} MB BF16)")
    print()

    # Baseline: flatten + copy into pinned BF16 buffer (what existing MegaTrain does)
    cpu_flat_bf16 = torch.empty(total, dtype=torch.bfloat16).pin_memory()

    def baseline_copy():
        offset = 0
        for p, n in zip(params, numels):
            cpu_flat_bf16[offset:offset + n].copy_(p.data.reshape(-1))
            offset += n

    # FP8 quantize (current implementation)
    transfer_dtype = parse_transfer_dtype("float8_e4m3")
    quantizer = WeightTransferQuantizer(transfer_dtype=transfer_dtype, master_dtype=torch.bfloat16)
    cpu_flat_fp8 = torch.empty(total, dtype=transfer_dtype).pin_memory()
    cpu_scales = torch.empty(len(params), dtype=torch.float32).pin_memory()

    def fp8_current():
        quantizer.quantize_layer_cpu(params, numels, cpu_flat_fp8, cpu_scales)

    # Optimized: no float32 intermediate, stay in BF16 until the cast.
    fp8_max = torch.finfo(transfer_dtype).max
    scale_floor = 1.0 / fp8_max

    def fp8_bf16_scale():
        offset = 0
        for i, (p, n) in enumerate(zip(params, numels)):
            flat = p.data.reshape(-1)
            amax = flat.abs().max()
            scale = (amax.to(torch.float32) / fp8_max).clamp(min=scale_floor)
            cpu_scales[i] = scale
            inv_scale = (1.0 / scale).to(p.dtype)
            # In-BF16 multiply-then-cast; still allocates one intermediate.
            cpu_flat_fp8[offset:offset + n].copy_((flat * inv_scale).to(transfer_dtype))
            offset += n

    # Even leaner: compute amax, then a single chain of (flat * inv).to(fp8)
    def fp8_minimal():
        offset = 0
        for i, (p, n) in enumerate(zip(params, numels)):
            flat = p.data.reshape(-1)
            amax = flat.abs().max().item()  # grab scalar up front
            scale = max(amax / fp8_max, scale_floor)
            cpu_scales[i] = scale
            # Fused-ish: multiply by precomputed reciprocal, cast directly.
            cpu_flat_fp8[offset:offset + n].copy_(
                (flat * (1.0 / scale)).to(transfer_dtype)
            )
            offset += n

    # Threaded FP8 quantize (PyTorch releases GIL during tensor ops).
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(max_workers=4)
    param_offsets = []
    off = 0
    for n in numels:
        param_offsets.append(off)
        off += n

    def quant_one(i: int):
        p = params[i]
        n = numels[i]
        offset = param_offsets[i]
        flat = p.data.reshape(-1)
        amax = flat.abs().max().item()
        scale = max(amax / fp8_max, scale_floor)
        cpu_scales[i] = scale
        cpu_flat_fp8[offset:offset + n].copy_(
            (flat * (1.0 / scale)).to(transfer_dtype)
        )

    def fp8_threaded():
        futures = [pool.submit(quant_one, i) for i in range(len(params))]
        for f in futures:
            f.result()

    # Just the amax cost (to isolate reduction overhead)
    def just_amax():
        for p in params:
            _ = p.data.abs().max().item()

    # Just the cast cost (no scaling, no scale metadata)
    def just_cast():
        offset = 0
        for p, n in zip(params, numels):
            cpu_flat_fp8[offset:offset + n].copy_(p.data.reshape(-1).to(transfer_dtype))
            offset += n

    print("CPU-side timings (per layer):")
    t_base = bench("baseline: flatten + bf16 copy", baseline_copy)
    t_amax = bench("just amax (per-tensor reduction)", just_amax)
    t_cast = bench("just cast (bf16 -> fp8, no scale)", just_cast)
    t_current = bench("fp8 current (with float32 intermediate)", fp8_current)
    t_opt = bench("fp8 bf16 scale (skip float32 promote)", fp8_bf16_scale)
    t_min = bench("fp8 minimal (scalar scale, one temp)", fp8_minimal)
    t_thr = bench("fp8 threaded (4 workers)", fp8_threaded)

    print()
    print(f"Baseline (no quant): {t_base*1000:.2f} ms")
    best = min(t_current, t_opt, t_min, t_thr)
    print(f"Best FP8 option:     {best*1000:.2f} ms")
    print(f"CPU overhead ratio:  {best / t_base:.2f}x")
    print()
    # Expected PCIe savings
    bf16_bytes = total * 2
    fp8_bytes = total + len(params) * 4  # payload + scales
    pcie_gb_s = 20.0  # PCIe 4.0 effective
    pcie_saved_ms = (bf16_bytes - fp8_bytes) / (pcie_gb_s * 1e9) * 1000
    print(f"PCIe savings estimate @ 20 GB/s: {pcie_saved_ms:.2f} ms/layer")
    print(f"Break-even: CPU overhead must be < PCIe savings")


if __name__ == "__main__":
    main()
