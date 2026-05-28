# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP4-g32 KV cache format: FP4 E2M1 codes + fp16 per-group-of-32 scales.

Modules:
- fp4_levels: FP4 E2M1 constants, midpoints, sorted↔bits remaps, slot layout.
- reference:  PyTorch reference (fakequant + dequant) for testing.
- triton_store:  Triton kernel that writes K/V into the FP4-g32 cache.
- triton_decode: Triton kernel that reads FP4-g32 cache and computes attention.

Per-slot layout (head_dim D = 128):
    bytes [  0 ..  64): K codes (2 nibbles/byte, 32-element groups)
    bytes [ 64 ..  72): K scales (fp16 × 4 groups)
    bytes [ 72 .. 136): V codes
    bytes [136 .. 144): V scales
"""

from vllm.v1.attention.ops.fp4_g32.fp4_levels import (
    DEFAULT_CONSTANT_C,
    FP4_BITS_TO_VALUE,
    FP4_LEVELS_SORTED,
    FP4_MAX,
    GROUP_SIZE,
    MIDPOINTS_SORTED,
    SORTED_TO_BITS,
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
    "fp4_g32_decode_attention",
    "fp4_g32_store",
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
    "v_codes_bytes",
    "v_codes_offset",
    "v_norm_offset",
    "v_scales_bytes",
    "v_scales_offset",
]


def __getattr__(name: str):
    # Lazy imports for the Triton-dependent launchers — keeps the module
    # importable on machines without Triton (useful for tests of
    # fp4_levels / reference that don't need a GPU).
    if name == "fp4_g32_store":
        from vllm.v1.attention.ops.fp4_g32.triton_store import (
            fp4_g32_store,
        )
        return fp4_g32_store
    if name == "fp4_g32_decode_attention":
        from vllm.v1.attention.ops.fp4_g32.triton_decode import (
            fp4_g32_decode_attention,
        )
        return fp4_g32_decode_attention
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
