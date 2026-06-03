# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8-MFMA KV cache format: FP4 E2M1 codepoints + E8M0 per-group-of-32
scales + FP8 E4M3 query at compute time.

Sibling of `fp4_g32`. The on-disk encoding is byte-identical for codes
(FP4 E2M1, same OCP grid, same packing) but scales are stored as 1-byte
E8M0 instead of 2-byte fp16. The decode kernel feeds K/V into a real
FP8 MFMA path with Q kept in FP8 E4M3 rather than reconstructing into
bf16.

Modules:
- fp8_levels: FP4 constants + E8M0 helpers, slot layout.
- reference:  PyTorch reference (fakequant + dequant) for testing.
- triton_store:  Triton kernel that writes K/V into the fp8_g32 cache.
- triton_decode: Triton kernel that reads fp8_g32 cache and computes
  attention via FP8 MFMA.

Per-slot layout (head_dim D = 128, group_size = 32):
    bytes [  0 .. 64): K codes (FP4 nibbles, 2/byte)
    bytes [ 64 .. 68): K scales (E8M0 × 4 groups)
    bytes [ 68 ..132): V codes
    bytes [132 ..136): V scales
    Total: 136 B/slot (vs 144 B in fp4_g32).
"""

from vllm.v1.attention.ops.fp8_g32.fp8_levels import (
    DEFAULT_CONSTANT_C,
    FP4_BITS_TO_VALUE,
    FP4_LEVELS_SORTED,
    FP4_MAX,
    GROUP_SIZE,
    MIDPOINTS_SORTED,
    SORTED_TO_BITS,
    UE8M0_BIAS,
    get_constant_c,
    get_group_size,
    get_token_norm,
    k_codes_bytes,
    k_codes_offset,
    k_norm_offset,
    k_scales_bytes,
    k_scales_offset,
    n_groups,
    slot_size,
    soa_block_content_bytes,
    soa_head_stride,
    soa_k_codes_region,
    soa_k_scales_region,
    soa_v_codes_region,
    soa_v_scales_region,
    ue8m0_decode,
    ue8m0_encode,
    v_codes_bytes,
    v_codes_offset,
    v_norm_offset,
    v_scales_bytes,
    v_scales_offset,
)

__all__ = [
    "DEFAULT_CONSTANT_C",
    "FP4_BITS_TO_VALUE",
    "FP4_LEVELS_SORTED",
    "FP4_MAX",
    "GROUP_SIZE",
    "MIDPOINTS_SORTED",
    "SORTED_TO_BITS",
    "UE8M0_BIAS",
    "fp8_g32_decode_attention",
    "fp8_g32_store",
    "get_constant_c",
    "get_group_size",
    "get_token_norm",
    "k_codes_bytes",
    "k_codes_offset",
    "k_norm_offset",
    "k_scales_bytes",
    "k_scales_offset",
    "n_groups",
    "slot_size",
    "soa_block_content_bytes",
    "soa_head_stride",
    "soa_k_codes_region",
    "soa_k_scales_region",
    "soa_v_codes_region",
    "soa_v_scales_region",
    "ue8m0_decode",
    "ue8m0_encode",
    "v_codes_bytes",
    "v_codes_offset",
    "v_norm_offset",
    "v_scales_bytes",
    "v_scales_offset",
]


def __getattr__(name: str):
    # Lazy imports for the Triton-dependent launchers.
    if name == "fp8_g32_store":
        from vllm.v1.attention.ops.fp8_g32.triton_store import fp8_g32_store
        return fp8_g32_store
    if name == "fp8_g32_decode_attention":
        from vllm.v1.attention.ops.fp8_g32.triton_decode import (
            fp8_g32_decode_attention,
        )
        return fp8_g32_decode_attention
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
