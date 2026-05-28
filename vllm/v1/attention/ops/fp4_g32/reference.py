# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PyTorch reference for the FP4-g32 KV cache format.

Mirrors the `use_constant_optimal_c=True` path from the upstream fakequant
(`fp4_fp16_perblock_fakequant.py`) with the defaults pinned to the G4
production configuration described in `fp4.md`:

- constant c = 0.156 (overridable via FP4FP16_CONSTANT_C)
- per-group-of-32 absmax scale, cast through fp16
- Sylvester Hadamard rotation on K (left-multiply by H.T)
- **no per-token L2 normfold** (`do_normfold=False`) — per fp4.md design
  intent ("removes the per-token reduction direction")
- **no inverse rotation on K** — our Triton kernel keeps K in the rotated
  basis and the decode path consumes a pre-rotated Q. The upstream
  reference does `snapped @ H` to return K in the original basis because
  it then writes back into a plain fp16 cache; our kernel writes
  rotated-FP4 codes directly so we test against the pre-inverse-rotation
  intermediate.
- V: same FP4-g32 quantization, but NO rotation on either store or read.

This module is the ground-truth for store/decode kernel tests. Three
entry points:

- `fp4_g32_encode_decode(x, ...)`: full encode→decode round trip in fp32.
  Returns the reconstructed tensor in the *rotated* basis (no inverse H).
  Used to compare against the Triton decode output.

- `fp4_g32_encode(x, ...)`: returns (codes, scales) for store-kernel tests.
  `codes` is uint8 packed (2 nibbles/byte), `scales` is fp16 per group.
  Each is in the order produced by the snap step (after rotation).

- `fp4_g32_dequant(codes, scales)`: decode side without the store-side
  rotation/snap, for tests of the decode kernel in isolation.

All operations are written in float32 host-side; the only dtype rounds
that affect the algorithmic result are the fp16 scale cast (line marked
SCALE_FP16_CAST) and the bf16 cast of decoded values.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.v1.attention.ops.fp4_g32.fp4_levels import (
    FP4_LEVELS_SORTED,
    GROUP_SIZE,
    MIDPOINTS_SORTED,
    SORTED_TO_BITS,
    get_constant_c,
    n_groups,
)


# ── Tensor caches (device, dtype) → constant tensors ───────────────────────
_LEVELS_CACHE: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}
_MIDPOINTS_CACHE: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}
_SORTED_TO_BITS_CACHE: dict[torch.device, torch.Tensor] = {}


def _get_levels(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (device, dtype)
    t = _LEVELS_CACHE.get(key)
    if t is None:
        t = torch.tensor(FP4_LEVELS_SORTED, device=device, dtype=dtype)
        _LEVELS_CACHE[key] = t
    return t


def _get_midpoints(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (device, dtype)
    t = _MIDPOINTS_CACHE.get(key)
    if t is None:
        t = torch.tensor(MIDPOINTS_SORTED, device=device, dtype=dtype)
        _MIDPOINTS_CACHE[key] = t
    return t


def _get_sorted_to_bits(device: torch.device) -> torch.Tensor:
    t = _SORTED_TO_BITS_CACHE.get(device)
    if t is None:
        t = torch.tensor(SORTED_TO_BITS, device=device, dtype=torch.uint8)
        _SORTED_TO_BITS_CACHE[device] = t
    return t


# ── Sylvester Hadamard (cached, normalised) ────────────────────────────────
_HADAMARD_CACHE: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}


def hadamard_matrix(
    dim: int, device: torch.device, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Sylvester Hadamard, normalised so H @ H.T = I."""
    if dim <= 0 or (dim & (dim - 1)) != 0:
        raise ValueError(f"hadamard_matrix requires power-of-two dim, got {dim}")
    key = (dim, device, dtype)
    cached = _HADAMARD_CACHE.get(key)
    if cached is not None:
        return cached
    H = torch.tensor([[1.0]], dtype=torch.float64)
    while H.shape[0] < dim:
        H = torch.cat(
            [torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0
        )
    H = (H / (dim**0.5)).to(device=device, dtype=dtype).contiguous()
    _HADAMARD_CACHE[key] = H
    return H


# ── Core quantization primitives ───────────────────────────────────────────
def _snap_to_sorted_idx(
    x_norm: torch.Tensor, *, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Snap normalised values to nearest FP4 level, returning sorted index
    in [0, 14]. Uses torch.bucketize against the 14 midpoints, which is the
    semantic the Triton kernel reproduces with 14 `tl.where` comparisons.
    """
    boundaries = _get_midpoints(x_norm.device, dtype)
    return torch.bucketize(x_norm.to(dtype), boundaries)


def _pack_nibbles_last_dim(codes: torch.Tensor) -> torch.Tensor:
    """Pack a [..., D] uint8 tensor of 4-bit values into [..., D//2] uint8
    bytes. Within a byte: low nibble = code[2k], high nibble = code[2k+1].
    """
    if codes.shape[-1] % 2 != 0:
        raise ValueError(
            f"pack_nibbles last dim must be even, got {codes.shape[-1]}"
        )
    lo = codes[..., 0::2] & 0xF
    hi = codes[..., 1::2] & 0xF
    return (lo | (hi << 4)).to(torch.uint8)


def _unpack_nibbles_last_dim(packed: torch.Tensor, out_dim: int) -> torch.Tensor:
    """Inverse of _pack_nibbles_last_dim. `packed` is [..., out_dim//2]."""
    if out_dim != 2 * packed.shape[-1]:
        raise ValueError(
            f"unpack_nibbles: out_dim={out_dim} must be 2*packed.last={2*packed.shape[-1]}"
        )
    p = packed.to(torch.int32)
    lo = (p & 0xF).to(torch.uint8)
    hi = ((p >> 4) & 0xF).to(torch.uint8)
    out_shape = packed.shape[:-1] + (out_dim,)
    out = torch.empty(out_shape, dtype=torch.uint8, device=packed.device)
    out[..., 0::2] = lo
    out[..., 1::2] = hi
    return out


@dataclass
class FP4G32Encoded:
    """Encoded FP4-g32 representation of one tensor along its last axis.

    Shapes assume input shape `[..., D]`:
    - codes_packed: `[..., D // 2]` uint8 — 2 FP4 nibbles per byte
    - scales:       `[..., D // GROUP_SIZE]` fp16
    """

    codes_packed: torch.Tensor  # uint8, [..., D//2]
    scales: torch.Tensor  # fp16, [..., D//GROUP_SIZE]


def fp4_g32_encode(
    x: torch.Tensor,
    *,
    rotate: bool = True,
    constant_c: float | None = None,
) -> FP4G32Encoded:
    """Encode `x` (shape [..., D]) to FP4 codes + fp16 per-group-of-32 scales.

    Steps (matches `fp4_fp16_perblock_fakequant.py` `use_constant_optimal_c=True`
    with `do_normfold=False`):
    1. Optional Hadamard rotation: `x_rot = x @ H.T`
    2. Reshape to `[..., G, GROUP_SIZE]` where `G = D // GROUP_SIZE`.
    3. `absmax = max(|x_rot|)` per group.
    4. `s_fp32 = c * absmax`; cast `s` THROUGH fp16, then back to fp32 for
       the divide. (SCALE_FP16_CAST — this round-trip is part of the
       algorithm: it shifts the bin boundaries and affects which FP4 code
       each element snaps to.)
    5. For zero-amax groups: scale = 1.0 sentinel (output codes will be all 0).
    6. `sorted_idx = bucketize(x_rot / s, midpoints)` ∈ [0, 14]
    7. Remap sorted_idx → FP4 E2M1 bit pattern via SORTED_TO_BITS.
    8. Pack pairs of 4-bit codes into bytes (low nibble = even index).

    Returns (codes_packed [..., D//2] uint8, scales [..., G] fp16).
    """
    if x.shape[-1] % GROUP_SIZE != 0:
        raise ValueError(
            f"fp4_g32_encode: last dim {x.shape[-1]} must be a multiple of "
            f"GROUP_SIZE={GROUP_SIZE}"
        )
    D = x.shape[-1]
    G = D // GROUP_SIZE
    c = constant_c if constant_c is not None else get_constant_c()

    x_f32 = x.to(torch.float32)
    if rotate:
        H = hadamard_matrix(D, x.device, torch.float32)
        x_rot = x_f32 @ H.T
    else:
        x_rot = x_f32

    # Per-group absmax and scale.
    x_g = x_rot.reshape(*x.shape[:-1], G, GROUP_SIZE)
    absmax = x_g.abs().amax(dim=-1, keepdim=True)  # [..., G, 1]
    is_zero = absmax == 0
    safe_absmax = torch.where(is_zero, torch.ones_like(absmax), absmax)
    # SCALE_FP16_CAST: round-trip through fp16 to match storage precision.
    scale_fp16 = (safe_absmax * c).to(torch.float16)
    scale_for_div = scale_fp16.to(torch.float32)
    # Zero-amax groups: divide by 1.0 sentinel; resulting codes will be all
    # at sorted_idx=7 (which maps to +0.0 bit pattern 0b0000).
    scale_for_div = torch.where(
        is_zero, torch.ones_like(scale_for_div), scale_for_div
    )

    # Snap to nearest sorted FP4 level.
    sorted_idx = _snap_to_sorted_idx(x_g / scale_for_div)  # [..., G, GROUP_SIZE]

    # Remap sorted index → FP4 E2M1 bit pattern.
    sorted_to_bits = _get_sorted_to_bits(x.device)
    fp4_bits = sorted_to_bits[sorted_idx]  # uint8, [..., G, GROUP_SIZE]

    # Flatten back to [..., D] before packing along D.
    fp4_bits_flat = fp4_bits.reshape(*x.shape[:-1], D)
    codes_packed = _pack_nibbles_last_dim(fp4_bits_flat)  # [..., D//2] uint8

    # Squeeze scale's trailing 1 so we have [..., G] fp16.
    scales_out = scale_fp16.squeeze(-1).to(torch.float16)

    # For zero-amax groups, force scale to 0 in storage (per algorithm — the
    # decode multiplies by stored scale, so 0 scale × 0 code = 0).
    scales_out = torch.where(
        is_zero.squeeze(-1), torch.zeros_like(scales_out), scales_out
    )

    return FP4G32Encoded(codes_packed=codes_packed, scales=scales_out)


def fp4_g32_dequant(encoded: FP4G32Encoded, head_dim: int) -> torch.Tensor:
    """Decode `encoded` to fp32, NO inverse rotation. The result is in the
    rotated basis (if rotation was applied at encode).
    """
    G = head_dim // GROUP_SIZE
    codes = _unpack_nibbles_last_dim(encoded.codes_packed, head_dim)
    # Decode bit pattern → FP4 value (signed). Use bit-pattern LUT.
    # We use a small fp32 tensor cached per-device.
    bits_to_val = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        device=encoded.codes_packed.device,
        dtype=torch.float32,
    )
    # codes is uint8 in [0..15]
    decoded = bits_to_val[codes.to(torch.long)]  # [..., D] fp32
    # Reshape D → (G, GROUP_SIZE) so we can broadcast scales.
    decoded_g = decoded.reshape(*decoded.shape[:-1], G, GROUP_SIZE)
    scales_g = encoded.scales.to(torch.float32).unsqueeze(-1)  # [..., G, 1]
    out_g = decoded_g * scales_g
    return out_g.reshape(*decoded.shape[:-1], head_dim)


def fp4_g32_encode_decode(
    x: torch.Tensor,
    *,
    rotate: bool = True,
    constant_c: float | None = None,
) -> torch.Tensor:
    """Full encode→decode round-trip in the rotated basis. Returns fp32."""
    enc = fp4_g32_encode(x, rotate=rotate, constant_c=constant_c)
    return fp4_g32_dequant(enc, x.shape[-1])


# ── Reference attention against the encoded cache ──────────────────────────
def reference_fp4_g32_attention(
    query: torch.Tensor,  # [B, Hq, D] bf16 — RAW (will be rotated here)
    key: torch.Tensor,    # [B, N, Hk, D] bf16
    value: torch.Tensor,  # [B, N, Hk, D] bf16
    *,
    scale: float,
    sinks: torch.Tensor | None = None,   # [Hq] fp32, optional
    constant_c: float | None = None,
) -> torch.Tensor:
    """End-to-end reference attention with FP4-g32 K/V encode→decode round
    trip in the rotated basis. Q is rotated by the SAME Hadamard inside this
    function (mirrors what the Triton decode launcher will do).

    Output shape: [B, Hq, D] in query.dtype.
    """
    B, Hq, D = query.shape
    B2, N, Hk, D2 = key.shape
    assert B == B2 and D == D2, (B, B2, D, D2)
    assert value.shape == key.shape
    assert Hq % Hk == 0, f"Hq={Hq} must be a multiple of Hk={Hk}"
    g = Hq // Hk

    H = hadamard_matrix(D, query.device, torch.float32)
    q_rot = (query.to(torch.float32) @ H.T)  # [B, Hq, D] rotated

    # Encode K and V via fp4-g32 (K with rotation, V without), then decode
    # back to fp32 in the same basis we'll attend in.
    k_recon_rot = fp4_g32_encode_decode(
        key.transpose(1, 2).contiguous(),  # [B, Hk, N, D]
        rotate=True,
        constant_c=constant_c,
    )  # rotated basis
    v_recon = fp4_g32_encode_decode(
        value.transpose(1, 2).contiguous(),  # [B, Hk, N, D]
        rotate=False,
        constant_c=constant_c,
    )

    # Broadcast K/V over GQA group.
    k_recon_rot = k_recon_rot.unsqueeze(2).expand(B, Hk, g, N, D).reshape(B, Hq, N, D)
    v_recon = v_recon.unsqueeze(2).expand(B, Hk, g, N, D).reshape(B, Hq, N, D)

    # QK^T in rotated basis (Q is also rotated → dot is invariant).
    qk = torch.einsum("bhd,bhnd->bhn", q_rot, k_recon_rot) * scale  # [B, Hq, N]

    if sinks is not None:
        # Augment softmax with a per-head sink logit. Matches the TQ-v3 sink
        # semantic: include the sink in the normaliser but with zero value
        # contribution.
        sink_log = sinks.to(torch.float32).reshape(1, Hq, 1).expand(B, Hq, 1)
        qk_full = torch.cat([qk, sink_log], dim=-1)              # [B, Hq, N+1]
        p_full = torch.softmax(qk_full, dim=-1)
        p = p_full[..., :N]
    else:
        p = torch.softmax(qk, dim=-1)

    out = torch.einsum("bhn,bhnd->bhd", p, v_recon)  # [B, Hq, D]
    return out.to(query.dtype)
