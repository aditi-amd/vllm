# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP4 codepoint + E8M0 (UE8M0) scale constants for the fp8_g32 KV cache
format.

This module mirrors `fp4_g32.fp4_levels` but with three key changes to
enable real scaled F8F6F4 MFMA on AMD CDNA4:

1. **Scale format = UE8M0 (1 byte per group)**, not fp16 (2 bytes).
   The per-block scale `s = c * absmax` is snapped to a pure power of
   two `s = 2^round(log2(s))` and stored as a single uint8 (E8M0 = 8
   exponent bits, 0 mantissa bits). This is the format that the AMD
   `MFMA_SCALE_F32_*_F8F6F4` instructions consume natively and matches
   the OCP MX-FP4 spec exactly. It halves scale storage and (when wired
   through scaled MFMA) eliminates the post-MFMA scale multiply.

2. **K and V codepoints stay FP4 E2M1** — identical grid and packing to
   `fp4_g32`. The only difference at the codepoint level is the scale
   was UE8M0-snapped before encoding (slightly different rounding of
   `s`, same FP4 grid).

3. **Q is consumed as FP8 E4M3** in the decode kernel (not fp16/bf16).
   All 15 FP4 E2M1 levels are exactly representable in FP8 E4M3, so the
   K/V dequant path can route FP4 codes → FP8 lossless and feed
   F8F6F4 MFMA directly without an intermediate bf16 carrier.

Slot layout (D=128, group_size=32):
    bytes [  0 .. 64): K codes  (FP4 nibbles, 2/byte)
    bytes [ 64 .. 68): K scales (E8M0, 1 byte × 4 groups)        [was 8 bytes in fp4_g32]
    bytes [ 68 ..132): V codes
    bytes [132 ..136): V scales (E8M0, 1 byte × 4 groups)        [was 8 bytes in fp4_g32]
    Total: 136 bytes/slot (vs 144 in fp4_g32 — 8 bytes saved per slot per head).

Same FP4 grid, same `c = 0.156` constant, no per-token norm-fold, no V
rotation — matches the fp4_g32 V3 algorithm except for the scale format.
"""

from __future__ import annotations

import math
import os

# ── 15 unique FP4 E2M1 levels, sorted ascending (identical to fp4_g32) ────
FP4_LEVELS_SORTED: tuple[float, ...] = (
    -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5,
     0.0,
     0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
)
assert len(FP4_LEVELS_SORTED) == 15

# Midpoints between consecutive sorted levels (14 boundaries).
MIDPOINTS_SORTED: tuple[float, ...] = tuple(
    (FP4_LEVELS_SORTED[i] + FP4_LEVELS_SORTED[i + 1]) / 2.0
    for i in range(len(FP4_LEVELS_SORTED) - 1)
)
assert len(MIDPOINTS_SORTED) == 14

# FP4 E2M1 bit-pattern decode (16 entries indexed by 4-bit code).
# Sign bit (MSB) | 2 exp bits | 1 mantissa bit (LSB).
FP4_BITS_TO_VALUE: tuple[float, ...] = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)
assert len(FP4_BITS_TO_VALUE) == 16

# Sorted index → FP4 E2M1 bit pattern (uint4).
SORTED_TO_BITS: tuple[int, ...] = (
    0b1111, 0b1110, 0b1101, 0b1100, 0b1011, 0b1010, 0b1001,
    0b0000,
    0b0001, 0b0010, 0b0011, 0b0100, 0b0101, 0b0110, 0b0111,
)
assert len(SORTED_TO_BITS) == 15
for _i, _bits in enumerate(SORTED_TO_BITS):
    _v = FP4_BITS_TO_VALUE[_bits]
    _v = 0.0 if _v == 0.0 else _v
    assert _v == FP4_LEVELS_SORTED[_i]

# ── Format constants ──────────────────────────────────────────────────────
GROUP_SIZE: int = 32
"""Elements per group; one E8M0 scale per group. Fixed at 32 because the
AMD scaled F8F6F4 MFMA instruction also consumes one E8M0 scale per 32
elements along K. Other group sizes would force a fallback to plain MFMA
+ accumulator-side scale multiply (no hardware fast-path)."""

FP4_MAX: float = 6.0
"""Largest representable FP4 magnitude."""

DEFAULT_CONSTANT_C: float = 0.156
"""MSE-optimal scale constant; `s_raw = c * absmax`, then UE8M0-snapped.
Identical to fp4_g32's DEFAULT_CONSTANT_C — only the post-snap storage
differs."""


def get_constant_c() -> float:
    """Read the active constant c (lazily — env may change between runs)."""
    raw = os.environ.get("FP4FP16_CONSTANT_C")
    if raw is None:
        return DEFAULT_CONSTANT_C
    try:
        return float(raw)
    except ValueError:
        return DEFAULT_CONSTANT_C


def get_group_size() -> int:
    """Read the active group size (default 32). Only 32 is supported in
    fp8_g32 because the hardware scaled MFMA path requires it; 16 and
    other values fall through to 32 for now."""
    raw = os.environ.get("FP4_KV_GROUP_SIZE")
    if raw is None:
        return GROUP_SIZE
    try:
        gs = int(raw)
        if gs == 32:
            return gs
    except ValueError:
        pass
    return GROUP_SIZE


def get_token_norm() -> bool:
    """fp8_g32 follows fp4_g32 V3's `do_normfold=False` design — no
    per-token L2 norm. The env hook is kept so existing harnesses don't
    explode if `FP4_KV_TOKEN_NORM=1` is set, but it's a no-op here.

    Returning False unconditionally keeps the slot layout fixed (no
    optional 4-byte norm trailer) and the kernels free of the per-token
    branch.
    """
    return False


# ── UE8M0 helpers ──────────────────────────────────────────────────────────
# E8M0 = 8 exponent bits, 0 mantissa bits, no sign. The byte value `e`
# represents `2^(e - 127)` for e in [1, 254]. e=0 and e=255 are reserved
# (zero and NaN respectively in the OCP spec). We use e=0 as our zero
# sentinel for sink-zero groups.

UE8M0_BIAS: int = 127

# fp32 minimum positive normal = 2^-126 ≈ 1.175e-38. Anything smaller
# clamps to the zero sentinel.
_UE8M0_MIN_EXP: int = -126
_UE8M0_MAX_EXP: int = 127


def ue8m0_encode(s: float) -> int:
    """Snap a positive fp32 scale `s` to the nearest power of 2 and
    encode as a UE8M0 byte. `s <= 0` encodes as 0 (zero sentinel)."""
    if s <= 0.0 or not math.isfinite(s):
        return 0
    exp = int(round(math.log2(s)))
    if exp < _UE8M0_MIN_EXP:
        return 0
    if exp > _UE8M0_MAX_EXP:
        exp = _UE8M0_MAX_EXP
    return exp + UE8M0_BIAS


def ue8m0_decode(byte: int) -> float:
    """Decode a UE8M0 byte back to fp32. 0 → 0.0 (zero sentinel)."""
    if byte == 0:
        return 0.0
    return 2.0 ** (byte - UE8M0_BIAS)


# ── Per-element / per-slot byte layout ────────────────────────────────────
# Per slot, per head (group_size=GS=32):
#   bytes [0 .. D/2)               : K codes (2 nibbles/byte)
#   bytes [D/2 .. D/2 + Gk)        : K scales (E8M0, 1 byte each, Gk = D/GS)
#   bytes [D/2 + Gk .. D + Gk)     : V codes
#   bytes [D + Gk .. D + 2*Gk)     : V scales (E8M0, 1 byte each)
# For D=128, GS=32: K codes 64B + K scales 4B + V codes 64B + V scales 4B = 136 B.
# (vs 144 B for fp4_g32 — saves 8 B per slot per head from the E8M0 swap.)


def k_codes_bytes(head_dim: int) -> int:
    """Bytes for one head's packed K codes."""
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    return head_dim // 2


def n_groups(head_dim: int, group_size: int | None = None) -> int:
    gs = group_size if group_size is not None else get_group_size()
    if head_dim % gs != 0:
        raise ValueError(
            f"head_dim={head_dim} must be a multiple of group_size={gs}"
        )
    return head_dim // gs


def k_scales_bytes(head_dim: int, group_size: int | None = None) -> int:
    """Bytes for one head's K scales (one E8M0 byte per group)."""
    return n_groups(head_dim, group_size)


def v_codes_bytes(head_dim: int) -> int:
    return k_codes_bytes(head_dim)


def v_scales_bytes(head_dim: int, group_size: int | None = None) -> int:
    return k_scales_bytes(head_dim, group_size)


def slot_size(
    head_dim: int,
    group_size: int | None = None,
    token_norm: bool | None = None,  # kept for API parity; always False
) -> int:
    """Bytes per (token, head) slot. fp8_g32 has no token-norm trailer."""
    return (
        k_codes_bytes(head_dim)
        + k_scales_bytes(head_dim, group_size)
        + v_codes_bytes(head_dim)
        + v_scales_bytes(head_dim, group_size)
    )


def k_codes_offset(head_dim: int, group_size: int | None = None) -> int:
    return 0


def k_scales_offset(head_dim: int, group_size: int | None = None) -> int:
    return k_codes_bytes(head_dim)


def v_codes_offset(head_dim: int, group_size: int | None = None) -> int:
    return k_codes_bytes(head_dim) + k_scales_bytes(head_dim, group_size)


def v_scales_offset(head_dim: int, group_size: int | None = None) -> int:
    return v_codes_offset(head_dim, group_size) + v_codes_bytes(head_dim)


# fp8_g32 has no per-token norm trailer; these offsets are kept as stubs
# so the store/decode kernels can pass through the same constexpr list
# without being rewritten. Returning 0 keeps any accidental dereference
# inside the slot (a benign value, since USE_TOKEN_NORM is always False).
def k_norm_offset(head_dim: int, group_size: int | None = None) -> int:
    return 0


def v_norm_offset(head_dim: int, group_size: int | None = None) -> int:
    return 0
