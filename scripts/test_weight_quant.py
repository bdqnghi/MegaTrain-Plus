"""Standalone correctness test for FP8 weight transfer quantizer.

Builds a small fake layer, quantizes on CPU, ships to GPU, dequantizes, and
compares to the original master weights. Prints per-tensor relative error.
"""

import torch
from infinity.quantization import WeightTransferQuantizer, parse_transfer_dtype


def make_fake_layer(device: str = "cpu"):
    """Mimic the per-param structure of a small transformer layer."""
    params = [
        torch.randn(3584, dtype=torch.bfloat16, device=device),               # input_layernorm
        torch.randn(3584, 3584, dtype=torch.bfloat16, device=device),         # q_proj
        torch.randn(512, 3584, dtype=torch.bfloat16, device=device),          # k_proj (gqa)
        torch.randn(512, 3584, dtype=torch.bfloat16, device=device),          # v_proj
        torch.randn(3584, 3584, dtype=torch.bfloat16, device=device),         # o_proj
        torch.randn(3584, dtype=torch.bfloat16, device=device),               # post_attn_ln
        torch.randn(18944, 3584, dtype=torch.bfloat16, device=device),        # gate_proj
        torch.randn(18944, 3584, dtype=torch.bfloat16, device=device),        # up_proj
        torch.randn(3584, 18944, dtype=torch.bfloat16, device=device),        # down_proj
    ]
    return params


def main():
    torch.manual_seed(0)
    params = make_fake_layer(device="cpu")
    numels = [p.numel() for p in params]
    total = sum(numels)
    print(f"Fake layer: {len(params)} tensors, {total/1e6:.1f}M params")

    transfer_dtype = parse_transfer_dtype("float8_e4m3")
    quantizer = WeightTransferQuantizer(transfer_dtype=transfer_dtype, master_dtype=torch.bfloat16)

    # Allocate buffers (mimic what CPUMasterModel does)
    cpu_flat = torch.empty(total, dtype=transfer_dtype)
    cpu_scales = torch.empty(len(params), dtype=torch.float32)

    quantizer.quantize_layer_cpu(params, numels, cpu_flat, cpu_scales)

    # Move to GPU
    gpu_flat = cpu_flat.to("cuda")
    gpu_scales = cpu_scales.to("cuda")

    # Allocate GPU template params (BF16, empty)
    gpu_params = [torch.empty_like(p, device="cuda") for p in params]
    quantizer.dequantize_layer_gpu(gpu_params, numels, gpu_flat, gpu_scales)

    # Compare
    print(f"{'idx':>3} | {'shape':>20} | {'rel_err':>10} | {'max_abs_err':>12}")
    print("-" * 60)
    worst_rel = 0.0
    for i, (orig, quantized) in enumerate(zip(params, gpu_params)):
        orig_gpu = orig.to("cuda")
        diff = (quantized - orig_gpu).abs()
        rel = (diff / orig_gpu.abs().clamp(min=1e-6)).mean().item()
        max_abs = diff.max().item()
        worst_rel = max(worst_rel, rel)
        print(f"{i:>3} | {str(tuple(orig.shape)):>20} | {rel:>10.4%} | {max_abs:>12.4e}")

    print()
    print(f"Worst per-tensor mean relative error: {worst_rel:.4%}")
    # FP8 E4M3 has ~0.25% precision in the middle of its range; per-tensor scaling
    # preserves the largest magnitudes but truncates small values, so ~1-2% mean
    # relative error on random tensors is expected and fine for weight transfer.
    assert worst_rel < 0.05, f"Quantization error too large: {worst_rel:.4%}"
    print("PASS: FP8 weight transfer roundtrip within 5%.")


if __name__ == "__main__":
    main()
