# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP4 E2M1 constants for the FP4-g32 (FP4 + fp16 per-group-of-32 scale) KV
cache format.

This module is dependency-free (no torch, no triton) so it can be imported
from kernels, the reference, configs, and tests without pulling heavy deps.

Format summary (per `fp4.md` G4):
- Element: FP4 E2M1 (sign + 2 exp + 1 mantissa = 4 bits per element).
- Per group of 32 consecutive elements along the head_dim:
    M  = max(|x_i|)
    s  = c * M    where c is the MSE-optimal constant 0.156 by default
                  (overridable via FP4FP16_CONSTANT_C).
    s  is stored as fp16 (2 bytes per group).
    code_i = nearest_FP4_E2M1_bit_pattern(x_i / s)  (4 bits per element).
- Sink-zero: when M==0, scale is forced to 1.0 and codes are all 0 (which
  decodes to 0.0).

Two parallel encodings show up in the code:

1. **sorted index** (0..14): the index of x in the *sorted-ascending* list
   of 15 FP4 levels. This is what `torch.bucketize(x, MIDPOINTS_SORTED)`
   produces and what the snap step naturally computes. Useful as an
   intermediate inside the store kernel.

2. **FP4 E2M1 bit pattern** (0..15): the actual hardware-compatible 4-bit
   encoding (sign bit | 2 exp bits | 1 mantissa bit). This is what we
   store in the cache so that the decode path can use either software
   inline-bits-to-bf16 decode or a future native FP4 MFMA path.

`SORTED_TO_BITS` maps (1) -> (2) so the store kernel can convert after
snapping. We choose +0 (bit pattern 0b0000) for the zero codepoint; the
redundant -0 (0b1000) is never emitted at store time.
"""

from __future__ import annotations

import os

# ── 15 unique FP4 E2M1 levels, sorted ascending ────────────────────────────
# All values exactly representable in bf16 and fp16.
FP4_LEVELS_SORTED: tuple[float, ...] = (
    -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5,
     0.0,
     0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
)
assert len(FP4_LEVELS_SORTED) == 15

# Midpoints between consecutive sorted levels (14 boundaries).
# bucketize(x, MIDPOINTS_SORTED) returns the sorted index in 0..14 — exactly
# the snap result for nearest-neighbor on the FP4 grid. All midpoints are
# exactly representable in bf16/fp16 (see `_FP4_MIDPOINTS_EXACT_NOTE` below).
MIDPOINTS_SORTED: tuple[float, ...] = tuple(
    (FP4_LEVELS_SORTED[i] + FP4_LEVELS_SORTED[i + 1]) / 2.0
    for i in range(len(FP4_LEVELS_SORTED) - 1)
)
assert len(MIDPOINTS_SORTED) == 14
# _FP4_MIDPOINTS_EXACT_NOTE: midpoints are {-5, -3.5, -2.5, -1.75, -1.25,
# -0.75, -0.25, +0.25, +0.75, +1.25, +1.75, +2.5, +3.5, +5}. All exact in
# bf16 (round to representable powers of 2 + halves).

# ── FP4 E2M1 bit-pattern decode (16 entries indexed by 4-bit code) ─────────
# Sign bit (MSB) | 2 exp bits | 1 mantissa bit (LSB).
# 0b0000 = +0.0,  0b1000 = -0.0  (redundant; treated as +0 in attention).
#
# Computed from the E2M1 layout:
#   sign     = (code >> 3) & 1
#   exp_bits = (code >> 1) & 0b11
#   mant_bit = code & 1
#   value = (-1)^sign * (1 + mant_bit*0.5) * 2^(exp_bits - 1)  if exp_bits>0
#         = (-1)^sign * mant_bit * 2^(-1)                       if exp_bits=0
#
# Concretely (positive half):
#   0000 -> +0.0    0001 -> +0.5    0010 -> +1.0    0011 -> +1.5
#   0100 -> +2.0    0101 -> +3.0    0110 -> +4.0    0111 -> +6.0
# Negative half is the same magnitudes with sign bit set.
FP4_BITS_TO_VALUE: tuple[float, ...] = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)
assert len(FP4_BITS_TO_VALUE) == 16

# ── Sorted index (0..14) -> FP4 E2M1 bit pattern (uint4) ──────────────────
# The sorted ascending order doesn't match the bit-pattern numeric order,
# so we need an explicit remap. Index 7 (the zero level) maps to +0 (0b0000);
# -0 (0b1000) is never emitted at store time but the decode LUT honors it.
SORTED_TO_BITS: tuple[int, ...] = (
    0b1111,  # 0 → -6.0
    0b1110,  # 1 → -4.0
    0b1101,  # 2 → -3.0
    0b1100,  # 3 → -2.0
    0b1011,  # 4 → -1.5
    0b1010,  # 5 → -1.0
    0b1001,  # 6 → -0.5
    0b0000,  # 7 →  0.0  (canonical +0)
    0b0001,  # 8 →  0.5
    0b0010,  # 9 →  1.0
    0b0011,  # 10 → 1.5
    0b0100,  # 11 → 2.0
    0b0101,  # 12 → 3.0
    0b0110,  # 13 → 4.0
    0b0111,  # 14 → 6.0
)
assert len(SORTED_TO_BITS) == 15
# Self-check: decoding the bit pattern at sorted index i must yield the
# corresponding sorted level (with -0 coerced to +0).
for _i, _bits in enumerate(SORTED_TO_BITS):
    _v = FP4_BITS_TO_VALUE[_bits]
    _v = 0.0 if _v == 0.0 else _v  # collapse -0/+0
    assert _v == FP4_LEVELS_SORTED[_i], (
        f"SORTED_TO_BITS[{_i}] = {_bits:#06b} -> {_v}, expected "
        f"{FP4_LEVELS_SORTED[_i]}"
    )

# ── Format constants ──────────────────────────────────────────────────────
GROUP_SIZE: int = 32
"""Default elements per group; one fp16 scale per group.
Overridable at runtime via FP4_KV_GROUP_SIZE (supported: 16, 32)."""

FP4_MAX: float = 6.0
"""Largest representable FP4 magnitude."""

DEFAULT_CONSTANT_C: float = 0.156
"""MSE-optimal scale constant per fp4.md (G4 cell, LCB-128K +3.3pp vs
INT4-Lloyd). The scale is `s = c * absmax`; equivalently `s = absmax / 6.41`.
Overridable via FP4FP16_CONSTANT_C env var to match the reference fakequant's
convention (`fp4_fp16_perblock_fakequant.py:_CONSTANT_OPT_C_VALUE`)."""


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
    """Read the active group size (lazily — env may change between runs).

    Supported values: 16, 32 (default).  Any other value silently falls back
    to 32 to preserve backward compatibility.
    """
    raw = os.environ.get("FP4_KV_GROUP_SIZE")
    if raw is None:
        return GROUP_SIZE
    try:
        gs = int(raw)
        if gs in (16, 32):
            return gs
    except ValueError:
        pass
    return GROUP_SIZE


def get_token_norm() -> bool:
    """Return True when per-token L2 normalization is requested.

    Enabled by setting FP4_KV_TOKEN_NORM=1 in the environment.  When active,
    the store kernel normalises each K/V vector by its L2 norm before
    per-group quantisation, and stores the norm as a fp16 scalar appended to
    the slot.  The decode kernel loads the norm and rescales accordingly.
    """
    return os.environ.get("FP4_KV_TOKEN_NORM", "0") == "1"


# ── Per-element / per-slot byte layout ────────────────────────────────────
# Per slot, per head (group_size=GS, token_norm=False):
#   bytes [0 .. D/2)                 : K codes (2 nibbles/byte)
#   bytes [D/2 .. D/2 + 2*Gk)        : K scales (fp16 each, Gk = D/GS)
#   bytes [D/2 + 2*Gk .. D + 2*Gk)   : V codes
#   bytes [D + 2*Gk .. D + 4*Gk)     : V scales (fp16 each)
# For D=128, GS=32: K codes 64B + K scales 8B + V codes 64B + V scales 8B = 144 B.
# For D=128, GS=16: K codes 64B + K scales 16B + V codes 64B + V scales 16B = 160 B.
#
# When token_norm=True, two extra fp16 scalars are appended:
#   bytes [D + 4*Gk     .. D + 4*Gk + 2): K_norm (fp16, per token)
#   bytes [D + 4*Gk + 2 .. D + 4*Gk + 4): V_norm (fp16, per token)
# For D=128, GS=32: 144 + 4 = 148 B.
# For D=128, GS=16: 160 + 4 = 164 B.

def k_codes_bytes(head_dim: int) -> int:
    """Bytes for one head's packed K codes."""
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    return head_dim // 2


def n_groups(head_dim: int, group_size: int | None = None) -> int:
    """Number of groups in one head."""
    gs = group_size if group_size is not None else get_group_size()
    if head_dim % gs != 0:
        raise ValueError(
            f"head_dim={head_dim} must be a multiple of group_size={gs}"
        )
    return head_dim // gs


def k_scales_bytes(head_dim: int, group_size: int | None = None) -> int:
    """Bytes for one head's K scales (one fp16 per group)."""
    return 2 * n_groups(head_dim, group_size)


def v_codes_bytes(head_dim: int) -> int:
    return k_codes_bytes(head_dim)


def v_scales_bytes(head_dim: int, group_size: int | None = None) -> int:
    return k_scales_bytes(head_dim, group_size)


def slot_size(
    head_dim: int,
    group_size: int | None = None,
    token_norm: bool | None = None,
) -> int:
    """Bytes per (token, head) slot.

    Args:
        head_dim: head dimension (must be even and divisible by group_size).
        group_size: override group size (default: read from FP4_KV_GROUP_SIZE env).
        token_norm: override token-norm flag (default: read from FP4_KV_TOKEN_NORM env).
    """
    tn = token_norm if token_norm is not None else get_token_norm()
    base = (
        k_codes_bytes(head_dim)
        + k_scales_bytes(head_dim, group_size)
        + v_codes_bytes(head_dim)
        + v_scales_bytes(head_dim, group_size)
    )
    return base + (4 if tn else 0)  # 2 bytes K_norm + 2 bytes V_norm


# Offsets within a slot (all take optional group_size for non-default GS)
def k_codes_offset(head_dim: int, group_size: int | None = None) -> int:
    return 0


def k_scales_offset(head_dim: int, group_size: int | None = None) -> int:
    return k_codes_bytes(head_dim)


def v_codes_offset(head_dim: int, group_size: int | None = None) -> int:
    return k_codes_bytes(head_dim) + k_scales_bytes(head_dim, group_size)


def v_scales_offset(head_dim: int, group_size: int | None = None) -> int:
    return v_codes_offset(head_dim, group_size) + v_codes_bytes(head_dim)


def k_norm_offset(head_dim: int, group_size: int | None = None) -> int:
    """Byte offset of the per-token K L2 norm (fp16).  Only valid when token_norm is active."""
    return (
        k_codes_bytes(head_dim)
        + k_scales_bytes(head_dim, group_size)
        + v_codes_bytes(head_dim)
        + v_scales_bytes(head_dim, group_size)
    )


def v_norm_offset(head_dim: int, group_size: int | None = None) -> int:
    """Byte offset of the per-token V L2 norm (fp16).  Only valid when token_norm is active."""
    return k_norm_offset(head_dim, group_size) + 2
