# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton store kernel for the fp8_g32 KV cache format.

Per-token, per-head, per-group-of-32 the kernel computes:

    s_raw = c * absmax(group)            # c = 0.156 by default
    s     = 2^round(log2(s_raw))         # UE8M0 snap (power of 2)
    code  = quantize_to_fp4(group / s)   # OCP E2M1, 4 bits/elt

and stores `code` as packed nibbles + `s` as a single E8M0 byte per group.
The output layout matches `fp8_levels.py` (136 B/slot for D=128, GS=32).

Differences vs `fp4_g32.triton_store`:
- Scale is UE8M0 (1 byte/group), not fp16 (2 bytes/group).
- No per-token L2 norm path (`USE_TOKEN_NORM` removed — fp8_g32 follows
  fp4_g32 V3's `do_normfold=False` convention).
- K is still Hadamard-rotated (in-register WHT butterfly), V is not.
- The FP4 grid, the c=0.156 constant, the `>` midpoint snap, and the
  sorted→bits remap are byte-identical to fp4_g32.

The single algorithmic delta vs fp4_g32 V3 is the UE8M0 snap on `s`
before it is used as the divisor for the FP4 grid encode and before it
is stored. This swaps the post-snap rounding (was: nearest fp16; now:
nearest power of two) so the stored codepoints can differ by at most one
grid step from fp4_g32 at the same input — the price of having `s` be
exactly consumable by AMD's scaled F8F6F4 MFMA on the read side.
"""

from __future__ import annotations

import torch


def _kv_cache_flat(kv_cache: torch.Tensor) -> torch.Tensor:
    """1-D uint8 view of the raw KV allocation (handles padded layouts)."""
    if kv_cache.is_contiguous():
        return kv_cache.reshape(-1)
    n = kv_cache.untyped_storage().nbytes() // kv_cache.element_size()
    return torch.empty(0, dtype=kv_cache.dtype, device=kv_cache.device).set_(
        kv_cache.untyped_storage(), 0, (n,)
    )


from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.fp8_g32.fp8_levels import (
    GROUP_SIZE,
    MIDPOINTS_SORTED,
    SORTED_TO_BITS,
    UE8M0_BIAS,
    fp8_g32_soa_scales_enabled,
    get_constant_c,
    get_group_size,
    is_arch_b,
    k_codes_bytes,
    k_scales_offset,
    n_groups,
    slot_size,
    soa_head_stride,
    soa_k_codes_region,
    soa_k_scales_region,
    soa_v_codes_region,
    soa_v_scales_region,
    v_codes_offset,
    v_scales_offset,
)


_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16}


# ── Cached per-device constant tensors ─────────────────────────────────────
_MIDPOINTS_CACHE: dict[torch.device, torch.Tensor] = {}
_SORTED_TO_BITS_CACHE: dict[torch.device, torch.Tensor] = {}
_PIT_CACHE: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}


def _get_midpoints_tensor(device: torch.device) -> torch.Tensor:
    t = _MIDPOINTS_CACHE.get(device)
    if t is None:
        t = torch.tensor(MIDPOINTS_SORTED, device=device, dtype=torch.float32)
        _MIDPOINTS_CACHE[device] = t
    return t


def _get_sorted_to_bits_tensor(device: torch.device) -> torch.Tensor:
    t = _SORTED_TO_BITS_CACHE.get(device)
    if t is None:
        t = torch.tensor(SORTED_TO_BITS, device=device, dtype=torch.int32)
        _SORTED_TO_BITS_CACHE[device] = t
    return t


def _get_hadamard(
    dim: int, device: torch.device, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Sylvester Hadamard in fp32 (cached, normalised)."""
    key = (dim, device, dtype)
    cached = _PIT_CACHE.get(key)
    if cached is not None:
        return cached
    if dim <= 0 or (dim & (dim - 1)) != 0:
        raise ValueError(f"fp8_g32 store requires power-of-two dim, got {dim}")
    H = torch.tensor([[1.0]], dtype=torch.float64)
    while H.shape[0] < dim:
        H = torch.cat(
            [torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0
        )
    H = (H / (dim**0.5)).to(device=device, dtype=dtype).contiguous()
    _PIT_CACHE[key] = H
    return H


# ═══════════════════════════════════════════════════════════════════════════
# Triton kernel
# ═══════════════════════════════════════════════════════════════════════════


@triton.jit
def _fp8_g32_store_kernel(
    Key_ptr,                # [N*H, D] in raw dtype (bf16 or fp16)
    Value_ptr,              # [N*H, D] in raw dtype
    KV_cache_ptr,           # raw uint8 view, flat
    Slot_mapping_ptr,       # [N] int64
    Midpoints_ptr,          # [14] fp32
    SortedToBits_ptr,       # [15] int32
    # Cache strides (in bytes)
    stride_cache_block: tl.constexpr,
    stride_cache_pos: tl.constexpr,
    stride_cache_head: tl.constexpr,
    # Dimensions
    HEAD_DIM: tl.constexpr,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    LOG2_D: tl.constexpr,
    # Layout
    GROUP_SIZE_C: tl.constexpr,
    N_GROUPS_C: tl.constexpr,
    K_SCALES_OFFSET: tl.constexpr,
    V_CODES_OFFSET: tl.constexpr,
    V_SCALES_OFFSET: tl.constexpr,
    FP4_C: tl.constexpr,
    UE8M0_BIAS_C: tl.constexpr,
    ARCH_B_C: tl.constexpr,  # 0 = Arch A (c folded into byte), 1 = Arch B (c baked in codebook)
    # SoA layout (Path 4) — when SOA_SCALES=1 the per-group scales live in a
    # contiguous per-block region instead of inside each slot. These constexpr
    # are only consulted on the SoA branch; ignored (pass 0) for AoS.
    SOA_SCALES: tl.constexpr = 0,
    SOA_HEAD_STRIDE: tl.constexpr = 0,   # bytes per head within a block (block_size * slot_size)
    SOA_K_CODES_REGION: tl.constexpr = 0,
    SOA_K_SCALES_REGION: tl.constexpr = 0,
    SOA_V_CODES_REGION: tl.constexpr = 0,
    SOA_V_SCALES_REGION: tl.constexpr = 0,
    KC_BYTES: tl.constexpr = 0,          # head_dim // 2 (K/V code bytes per token)
):
    pid = tl.program_id(0)
    token_idx = pid // H
    head_idx = pid % H

    slot = tl.load(Slot_mapping_ptr + token_idx)
    if slot < 0:
        return
    blk = slot // BLOCK_SIZE
    off = slot % BLOCK_SIZE
    if SOA_SCALES:
        # Head-major, field-major within the block. Codes stay token-strided
        # (AoS within their region); scales land in the contiguous SoA region.
        per_head_base = (
            blk * stride_cache_block + head_idx * SOA_HEAD_STRIDE
        ).to(tl.int64)
        k_code_base = per_head_base + SOA_K_CODES_REGION + off * KC_BYTES
        k_scale_base = per_head_base + SOA_K_SCALES_REGION + off * N_GROUPS_C
        v_code_base = per_head_base + SOA_V_CODES_REGION + off * KC_BYTES
        v_scale_base = per_head_base + SOA_V_SCALES_REGION + off * N_GROUPS_C
    else:
        slot_base = (
            blk * stride_cache_block
            + off * stride_cache_pos
            + head_idx * stride_cache_head
        ).to(tl.int64)
        k_code_base = slot_base
        k_scale_base = slot_base + K_SCALES_OFFSET
        v_code_base = slot_base + V_CODES_OFFSET
        v_scale_base = slot_base + V_SCALES_OFFSET

    base = pid * HEAD_DIM
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    shifts_4 = tl.arange(0, 2) * 4   # 0, 4 for nibble packing
    g_offs = tl.arange(0, N_GROUPS_C)

    # ── K side ─────────────────────────────────────────────────────────
    # 1. Load + Hadamard-rotate K via WHT butterfly (identical to fp4_g32).
    k_raw = tl.load(Key_ptr + base + d_offs, mask=d_mask, other=0.0).to(tl.float32)
    k_rot = tl.where(d_mask, k_raw, 0.0)

    INV_SQRT2: tl.constexpr = 0.7071067811865476
    for _stage in tl.static_range(LOG2_D):
        _bmask = 1 << _stage
        _pidx = (d_offs ^ _bmask).to(tl.int32)
        _pval = tl.gather(k_rot, _pidx, 0)
        _sign = tl.where((d_offs & _bmask) == 0, 1.0, -1.0).to(tl.float32)
        k_rot = (_sign * k_rot + _pval) * INV_SQRT2
    k_rot = tl.where(d_mask, k_rot, 0.0)

    # 2. Per-group absmax.
    k_g = tl.reshape(k_rot, [N_GROUPS_C, GROUP_SIZE_C])
    k_absmax = tl.max(tl.abs(k_g), axis=1)            # [N_GROUPS]
    k_is_zero = k_absmax == 0.0

    # 3. UE8M0 snap. Two recipes share the byte format (`byte = exp + 127`)
    #    but differ in what `exp` is and what the encode divisor is:
    #
    #    Arch A (ARCH_B_C == 0): s_raw = c * absmax (fp32)
    #                            exp = round(log2(s_raw))    (round-half-up)
    #                            divisor = 2^exp             (c folded in)
    #
    #    Arch B (ARCH_B_C == 1): exp = ceil(log2(absmax))
    #                            divisor = 2^exp * c         (c applied here so
    #                                                         the bucketize is
    #                                                         still against the
    #                                                         raw FP4 midpoints)
    #
    #    For zero groups byte=0 (zero sentinel), snapped value is 0.0,
    #    and we substitute 1.0 into the divisor below to avoid NaN.
    if ARCH_B_C == 1:
        k_s_safe = tl.where(k_is_zero, 1.0, k_absmax)
        k_log2 = tl.log2(k_s_safe)
        k_exp = tl.cast(tl.ceil(k_log2), tl.int32)
        k_pow2 = tl.exp2(tl.cast(k_exp, tl.float32))
        k_s_snapped = k_pow2
        k_s_div = tl.where(k_is_zero, 1.0, k_pow2 * tl.cast(FP4_C, tl.float32))
    else:
        k_s_raw = k_absmax * tl.cast(FP4_C, tl.float32)
        k_s_safe = tl.where(k_is_zero, 1.0, k_s_raw)
        k_log2 = tl.log2(k_s_safe)
        # `round` here matches Python/torch's banker's rounding at .5; for
        # well-behaved log2 values the difference vs round-half-away-from-zero
        # is irrelevant. tl.extra.libdevice.rint() is the float-truncating
        # round-to-nearest-even; we replicate it via add-half + floor for
        # portability.
        k_exp = tl.cast(tl.floor(k_log2 + 0.5), tl.int32)  # round-half-up
        k_s_snapped = tl.exp2(tl.cast(k_exp, tl.float32))
        k_s_div = tl.where(k_is_zero, 1.0, k_s_snapped)

    # 4. Normalize + snap to sorted FP4 idx via 14 sequential tl.where (`>` for
    #    bucketize(right=False) parity with the reference).
    k_norm = k_g / k_s_div[:, None]
    k_sorted = tl.zeros([N_GROUPS_C, GROUP_SIZE_C], dtype=tl.int32)
    for i in tl.static_range(14):
        mid = tl.load(Midpoints_ptr + i)
        k_sorted += tl.where(k_norm > mid, 1, 0)
    k_sorted = tl.where(k_is_zero[:, None], 7, k_sorted)

    # 5. Remap sorted idx → FP4 E2M1 bit pattern.
    k_bits = tl.load(SortedToBits_ptr + k_sorted)

    # 6. Pack pairs of 4-bit codes into uint8 bytes.
    k_bits_flat = tl.reshape(k_bits, [HEAD_DIM])
    k_pairs = tl.reshape(k_bits_flat, [HEAD_DIM // 2, 2])
    k_packed = tl.sum((k_pairs & 0xF) << shifts_4[None, :], axis=1).to(tl.uint8)
    k_code_addrs = k_code_base + tl.arange(0, HEAD_DIM // 2)
    tl.store(KV_cache_ptr + k_code_addrs, k_packed)

    # 7. Store K scales as E8M0 bytes (one per group). Zero-amax groups
    #    write byte=0 (zero sentinel).
    k_byte = tl.where(k_is_zero, 0, k_exp + UE8M0_BIAS_C)
    k_byte = tl.where((k_byte < 0) | (k_byte > 255), 0, k_byte).to(tl.uint8)
    k_scale_addrs = k_scale_base + g_offs
    tl.store(KV_cache_ptr + k_scale_addrs, k_byte)

    # ── V side (mirror K but skip rotation) ────────────────────────────
    v_raw = tl.load(Value_ptr + base + d_offs, mask=d_mask, other=0.0).to(tl.float32)
    v_raw = tl.where(d_mask, v_raw, 0.0)

    v_g = tl.reshape(v_raw, [N_GROUPS_C, GROUP_SIZE_C])
    v_absmax = tl.max(tl.abs(v_g), axis=1)
    v_is_zero = v_absmax == 0.0

    if ARCH_B_C == 1:
        v_s_safe = tl.where(v_is_zero, 1.0, v_absmax)
        v_log2 = tl.log2(v_s_safe)
        v_exp = tl.cast(tl.ceil(v_log2), tl.int32)
        v_pow2 = tl.exp2(tl.cast(v_exp, tl.float32))
        v_s_snapped = v_pow2
        v_s_div = tl.where(v_is_zero, 1.0, v_pow2 * tl.cast(FP4_C, tl.float32))
    else:
        v_s_raw = v_absmax * tl.cast(FP4_C, tl.float32)
        v_s_safe = tl.where(v_is_zero, 1.0, v_s_raw)
        v_log2 = tl.log2(v_s_safe)
        v_exp = tl.cast(tl.floor(v_log2 + 0.5), tl.int32)
        v_s_snapped = tl.exp2(tl.cast(v_exp, tl.float32))
        v_s_div = tl.where(v_is_zero, 1.0, v_s_snapped)

    v_norm = v_g / v_s_div[:, None]
    v_sorted = tl.zeros([N_GROUPS_C, GROUP_SIZE_C], dtype=tl.int32)
    for i in tl.static_range(14):
        mid = tl.load(Midpoints_ptr + i)
        v_sorted += tl.where(v_norm > mid, 1, 0)
    v_sorted = tl.where(v_is_zero[:, None], 7, v_sorted)

    v_bits = tl.load(SortedToBits_ptr + v_sorted)
    v_bits_flat = tl.reshape(v_bits, [HEAD_DIM])
    v_pairs = tl.reshape(v_bits_flat, [HEAD_DIM // 2, 2])
    v_packed = tl.sum((v_pairs & 0xF) << shifts_4[None, :], axis=1).to(tl.uint8)
    v_code_addrs = v_code_base + tl.arange(0, HEAD_DIM // 2)
    tl.store(KV_cache_ptr + v_code_addrs, v_packed)

    v_byte = tl.where(v_is_zero, 0, v_exp + UE8M0_BIAS_C)
    v_byte = tl.where((v_byte < 0) | (v_byte > 255), 0, v_byte).to(tl.uint8)
    v_scale_addrs = v_scale_base + g_offs
    tl.store(KV_cache_ptr + v_scale_addrs, v_byte)


# ═══════════════════════════════════════════════════════════════════════════
# Launcher
# ═══════════════════════════════════════════════════════════════════════════


def fp8_g32_store(
    key: torch.Tensor,           # [N, H, D] bf16 or fp16
    value: torch.Tensor,         # [N, H, D] bf16 or fp16
    kv_cache: torch.Tensor,      # [num_blocks, block_size, Hk, slot_size_aligned] uint8
    slot_mapping: torch.Tensor,  # [N] int64
    *,
    PiT: torch.Tensor | None = None,    # unused — kept for API parity with fp4_g32
    constant_c: float | None = None,
) -> None:
    """Launch the fp8_g32 store kernel. Writes K/V into `kv_cache`
    in-place at the slots specified by `slot_mapping`."""
    if key.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"fp8_g32_store: key.dtype must be one of {_SUPPORTED_DTYPES}, got {key.dtype}"
        )
    if value.dtype != key.dtype:
        raise ValueError(
            f"fp8_g32_store: key/value dtype mismatch ({key.dtype} vs {value.dtype})"
        )
    if slot_mapping.dtype != torch.int64:
        slot_mapping = slot_mapping.to(torch.int64)

    N, H, D = key.shape
    NH = N * H
    block_size = kv_cache.shape[1]
    num_kv_heads = kv_cache.shape[2]
    padded_slot = kv_cache.shape[3]

    expected_slot = slot_size(D)
    if padded_slot < expected_slot:
        raise ValueError(
            f"fp8_g32_store: kv_cache slot {padded_slot} < expected {expected_slot} for head_dim={D}"
        )
    if num_kv_heads != H:
        raise ValueError(
            f"fp8_g32_store: kv_cache num_kv_heads {num_kv_heads} != key heads {H}"
        )

    midpoints = _get_midpoints_tensor(key.device)
    sorted_to_bits = _get_sorted_to_bits_tensor(key.device)

    k_flat = key.reshape(NH, D).contiguous()
    v_flat = value.reshape(NH, D).contiguous()

    gs = get_group_size()
    BLOCK_D = triton.next_power_of_2(D)
    N_GROUPS_C = n_groups(D, gs)
    K_SCALES_OFFSET = k_scales_offset(D, gs)
    V_CODES_OFFSET = v_codes_offset(D, gs)
    V_SCALES_OFFSET = v_scales_offset(D, gs)

    stride_block = kv_cache.stride(0)
    stride_pos = kv_cache.stride(1)
    stride_head = kv_cache.stride(2)

    c = constant_c if constant_c is not None else get_constant_c()
    arch_b_c = 1 if is_arch_b() else 0

    soa = fp8_g32_soa_scales_enabled()
    soa_scales = 1 if soa else 0
    SOA_HEAD_STRIDE = soa_head_stride(D, block_size, gs) if soa else 0
    SOA_K_CODES_REGION = soa_k_codes_region(D, block_size, gs) if soa else 0
    SOA_K_SCALES_REGION = soa_k_scales_region(D, block_size, gs) if soa else 0
    SOA_V_CODES_REGION = soa_v_codes_region(D, block_size, gs) if soa else 0
    SOA_V_SCALES_REGION = soa_v_scales_region(D, block_size, gs) if soa else 0
    KC_BYTES = k_codes_bytes(D)

    grid = (NH,)
    _fp8_g32_store_kernel[grid](
        k_flat,
        v_flat,
        _kv_cache_flat(kv_cache),
        slot_mapping,
        midpoints,
        sorted_to_bits,
        stride_cache_block=stride_block,
        stride_cache_pos=stride_pos,
        stride_cache_head=stride_head,
        HEAD_DIM=D,
        H=H,
        BLOCK_SIZE=block_size,
        BLOCK_D=BLOCK_D,
        LOG2_D=int(D).bit_length() - 1,
        GROUP_SIZE_C=gs,
        N_GROUPS_C=N_GROUPS_C,
        K_SCALES_OFFSET=K_SCALES_OFFSET,
        V_CODES_OFFSET=V_CODES_OFFSET,
        V_SCALES_OFFSET=V_SCALES_OFFSET,
        FP4_C=c,
        UE8M0_BIAS_C=UE8M0_BIAS,
        ARCH_B_C=arch_b_c,
        SOA_SCALES=soa_scales,
        SOA_HEAD_STRIDE=SOA_HEAD_STRIDE,
        SOA_K_CODES_REGION=SOA_K_CODES_REGION,
        SOA_K_SCALES_REGION=SOA_K_SCALES_REGION,
        SOA_V_CODES_REGION=SOA_V_CODES_REGION,
        SOA_V_SCALES_REGION=SOA_V_SCALES_REGION,
        KC_BYTES=KC_BYTES,
        num_warps=4,
        num_stages=1,
    )
