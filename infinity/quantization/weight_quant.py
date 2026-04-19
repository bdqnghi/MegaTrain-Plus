"""FP8 E4M3 per-tensor weight transfer quantization.

Design
------
Halves the PCIe H2D payload for layer weights (2 bytes/param -> 1 byte/param +
~1 scale per parameter tensor).

Safe by construction: we re-quantize from the FP32/BF16 CPU master copy on every
layer load, so quantization error does NOT accumulate across training steps. The
only numerical effect is per-step noise, bounded by FP8 E4M3 precision.

Flow
----
1. CPU (in ``_load_layer_to_buffer_async``):
   - Flatten layer params into a pinned BF16 staging slice (one per param).
   - Compute per-tensor scale = amax / fp8_max.
   - Cast to FP8 E4M3 into the pinned FP8 flat buffer.
   - Record scales in a pinned FP32 scale buffer (one float per parameter tensor).

2. Stream (weight_stream):
   - Async H2D copy of the FP8 flat buffer + the scales buffer. Both are small:
     total bytes = 0.5 x bf16_bytes + 4 * num_params (scales).

3. GPU (in ``_unflatten_to_layer``):
   - For each parameter tensor: ``p.data.copy_(fp8_slice.to(bf16) * scale)``.
   - The template parameter is written in its native dtype (BF16). Flash
     Attention and any downstream kernels are untouched.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch

FP8_E4M3_DTYPE = torch.float8_e4m3fn


def parse_transfer_dtype(name: str) -> torch.dtype | None:
    """Return a torch dtype for a transfer-quantization identifier, or None for no quantization."""
    if name is None or name == "" or name.lower() in ("bfloat16", "bf16", "none"):
        return None
    if name.lower() in ("float8_e4m3", "float8_e4m3fn", "fp8", "fp8_e4m3"):
        return FP8_E4M3_DTYPE
    raise ValueError(f"Unsupported weight_transfer_dtype: {name}")


class WeightTransferQuantizer:
    """Per-tensor FP8 quantization for the weight H2D path.

    Stateless with respect to training; owns no CPU worker threads. Callers
    (``_load_layer_to_buffer_async`` / ``_unflatten_to_layer``) pass in the pinned
    buffers they already own.
    """

    def __init__(self, transfer_dtype: torch.dtype, master_dtype: torch.dtype):
        if transfer_dtype != FP8_E4M3_DTYPE:
            raise NotImplementedError(
                f"WeightTransferQuantizer only supports FP8 E4M3 for now, got {transfer_dtype}"
            )
        self.transfer_dtype = transfer_dtype
        self.master_dtype = master_dtype
        self.fp8_max = torch.finfo(transfer_dtype).max
        # Avoid dividing by zero on all-zero parameters.
        self._scale_floor = 1.0 / self.fp8_max

    def quantize_layer_cpu(
        self,
        params: Sequence[torch.Tensor],
        numels: Sequence[int],
        cpu_flat_fp8: torch.Tensor,
        cpu_scales: torch.Tensor,
    ) -> None:
        """Quantize all params of a layer into pre-allocated pinned buffers.

        Args:
            params: Ordered list of the layer's CPU parameter tensors (``.data``).
            numels: Precomputed ``p.numel()`` per param.
            cpu_flat_fp8: Pinned FP8 tensor, size >= sum(numels).
            cpu_scales: Pinned FP32 tensor, size >= len(params).
        """
        offset = 0
        for i, (p, n) in enumerate(zip(params, numels)):
            flat_view = p.data.reshape(-1)
            amax = flat_view.abs().max()
            # Compute scale; floor prevents div-by-zero on zeroed params.
            scale = (amax.to(torch.float32) / self.fp8_max).clamp(min=self._scale_floor)
            cpu_scales[i] = scale
            # Quantize into the destination slice. Division happens in bf16/fp32 then cast.
            # We use float32 for the division to avoid BF16 precision loss on the scale.
            scaled = (flat_view.to(torch.float32) / scale).to(self.transfer_dtype)
            cpu_flat_fp8[offset:offset + n].copy_(scaled)
            offset += n

    @staticmethod
    def dequantize_layer_gpu(
        gpu_params: Sequence[torch.Tensor],
        numels: Sequence[int],
        gpu_flat_fp8: torch.Tensor,
        gpu_scales: torch.Tensor,
    ) -> None:
        """Dequantize FP8 GPU flat buffer into template parameters.

        Args:
            gpu_params: Ordered list of the GPU template parameter tensors.
            numels: Precomputed ``p.numel()`` per param (same ordering as gpu_params).
            gpu_flat_fp8: GPU FP8 tensor, size >= sum(numels).
            gpu_scales: GPU FP32 tensor, size >= len(gpu_params).
        """
        offset = 0
        target_dtype = gpu_params[0].dtype
        for i, (p, n) in enumerate(zip(gpu_params, numels)):
            # Cast FP8 -> target dtype, then multiply by scalar scale.
            # scale is FP32; promoting to FP32 for the multiply keeps precision.
            bf16_view = gpu_flat_fp8[offset:offset + n].to(target_dtype)
            scale = gpu_scales[i].to(target_dtype)
            p.data.copy_((bf16_view * scale).view(p.shape))
            offset += n
