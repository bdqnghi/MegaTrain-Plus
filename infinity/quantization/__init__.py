"""Quantization schemes for MegaTrain.

Currently supports:
    - FP8 E4M3 per-tensor weight transfer quantization (weight H2D path only)

Future:
    - INT4 group-wise weight quantization
    - INT8 gradient D2H quantization
    - INT8 activation checkpoint quantization
    - 8-bit Adam optimizer states
"""

from .weight_quant import WeightTransferQuantizer, parse_transfer_dtype

__all__ = ["WeightTransferQuantizer", "parse_transfer_dtype"]
