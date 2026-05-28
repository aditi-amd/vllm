# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton store kernel for the FP4-g32 KV cache format.

One program per (token, kv_head). Each program:
 1. Loads the raw K and V vectors for its slot, shape [HEAD_DIM].
 2. Hadamard-rotates K (matrix multiply against PiT — first version uses
    the O(D^2) GEMV loop matching `_tq_fully_fused_store_mse`; we can
    swap in the in-register WHT butterfly later as a perf opt).
 3. Reshapes to [N_GROUPS, GROUP_SIZE] and computes per-group absmax.
 4. scale = c * absmax, cast THROUGH fp16 (matches the reference fakequant
    SCALE_FP16_CAST step).
 5. Normalizes by scale (1.0 sentinel for zero-amax groups), snaps each
    element to the nearest sorted FP4 level via 14 sequential `tl.where`
    comparisons (= torch.bucketize against midpoints).
 6. Remaps sorted index → FP4 E2M1 bit pattern via a 16-entry gather.
 7. Packs pairs of 4-bit codes into bytes (low nibble = even index) and
    writes K codes + K scales (fp16, byte-split stores) to the slot.
 8. Repeats steps 3–7 for V, with rotation skipped.

Slot layout (matches fp4_levels.py / fp4.md G4):
    bytes [  0 ..  64): K codes (2 nibbles/byte)
    bytes [ 64 ..  72): K scales (fp16 × 4 groups)
    bytes [ 72 .. 136): V codes
    bytes [136 .. 144): V scales
"""

from __future__ import annotations

import torch


def _kv_cache_flat(kv_cache: torch.Tensor) -> torch.Tensor:
    """1-D view of the raw KV allocation.

    Padded page layouts use ``as_strided`` views that are not contiguous;
    kernels index via byte strides into the underlying allocation.
    """
    if kv_cache.is_contiguous():
        return kv_cache.reshape(-1)
    n = kv_cache.untyped_storage().nbytes() // kv_cache.element_size()
    return torch.empty(0, dtype=kv_cache.dtype, device=kv_cache.device).set_(
        kv_cache.untyped_storage(), 0, (n,)
    )

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.fp4_g32.fp4_levels import (
    GROUP_SIZE,
    MIDPOINTS_SORTED,
    SORTED_TO_BITS,
    get_constant_c,
    get_group_size,
    get_token_norm,
    k_norm_offset,
    k_scales_offset,
    n_groups,
    slot_size,
    v_codes_offset,
    v_norm_offset,
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
        # int32 keeps the in-kernel gather output type clean.
        t = torch.tensor(SORTED_TO_BITS, device=device, dtype=torch.int32)
        _SORTED_TO_BITS_CACHE[device] = t
    return t


def _get_hadamard(
    dim: int, device: torch.device, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Sylvester Hadamard in fp32, contiguous as `H.T` (i.e. PiT for the
    same convention as the rest of TQ — see triton_turboquant_store.py).
    Returned matrix is symmetric so `H == H.T` either way; the name `PiT`
    is kept for parity with existing code.
    """
    key = (dim, device, dtype)
    cached = _PIT_CACHE.get(key)
    if cached is not None:
        return cached
    if dim <= 0 or (dim & (dim - 1)) != 0:
        raise ValueError(f"FP4-g32 store requires power-of-two dim, got {dim}")
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
def _fp4_g32_store_kernel(
    Key_ptr,                # [N*H, D] in raw dtype (bf16 or fp16)
    Value_ptr,              # [N*H, D] in raw dtype
    PiT_ptr,                # [D, D] fp32 — Sylvester Hadamard (transpose)
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
    H: tl.constexpr,                # num_kv_heads
    BLOCK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,          # next_pow2(HEAD_DIM) — for masking
    LOG2_D: tl.constexpr,           # log2(HEAD_DIM) — # of butterfly stages
    # FP4-g32 layout constants
    GROUP_SIZE_C: tl.constexpr,     # group size (32 or 16)
    N_GROUPS_C: tl.constexpr,       # HEAD_DIM // GROUP_SIZE_C
    K_SCALES_OFFSET: tl.constexpr,
    V_CODES_OFFSET: tl.constexpr,
    V_SCALES_OFFSET: tl.constexpr,
    FP4_C: tl.constexpr,
    # Per-token L2 norm
    USE_TOKEN_NORM: tl.constexpr,   # 1 = normalise each K/V by its L2 norm before quant
    K_NORM_OFFSET: tl.constexpr,    # byte offset of per-token K norm in slot
    V_NORM_OFFSET: tl.constexpr,    # byte offset of per-token V norm in slot
):
    pid = tl.program_id(0)
    token_idx = pid // H
    head_idx = pid % H

    slot = tl.load(Slot_mapping_ptr + token_idx)
    if slot < 0:
        return
    blk = slot // BLOCK_SIZE
    off = slot % BLOCK_SIZE
    slot_base = (
        blk * stride_cache_block
        + off * stride_cache_pos
        + head_idx * stride_cache_head
    ).to(tl.int64)

    base = pid * HEAD_DIM
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM

    # ── K side ─────────────────────────────────────────────────────────
    # 1. Load raw K (bf16/fp16) → fp32.
    k_raw = tl.load(Key_ptr + base + d_offs, mask=d_mask, other=0.0).to(tl.float32)
    k_rot = tl.where(d_mask, k_raw, 0.0)

    # 2. WHT butterfly Hadamard rotation: O(D log D) = 896 ops for D=128,
    #    fully in-register (no HBM PiT loads). Mathematically identical to
    #    k_raw @ H_sylvester (= k_raw @ PiT, since H is symmetric and our
    #    PiT is the normalized Sylvester Hadamard). Replaces the O(D²)
    #    GEMV loop — measured 5.4× speedup at N=8192 on MI300X. See
    #    `_tq_butterfly_store_mse` in triton_turboquant_store.py for the
    #    precedent; the only difference here is we don't pre-normalize
    #    k (FP4-g32 does per-group absmax scaling, not vector-norm).
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    for _stage in tl.static_range(LOG2_D):
        _bmask = 1 << _stage
        _pidx = (d_offs ^ _bmask).to(tl.int32)
        _pval = tl.gather(k_rot, _pidx, 0)
        _sign = tl.where((d_offs & _bmask) == 0, 1.0, -1.0).to(tl.float32)
        k_rot = (_sign * k_rot + _pval) * INV_SQRT2
    k_rot = tl.where(d_mask, k_rot, 0.0)

    # 2b. Per-token L2 normalisation (optional).  Divides k_rot by its L2
    #     norm before per-group quantisation so that the stored codes capture
    #     the *shape* of the activation and the scalar norm captures its
    #     *magnitude*.  The norm is stored as fp16 in the slot and recovered
    #     at decode time by multiplying scores.
    if USE_TOKEN_NORM:
        k_l2_sq = tl.sum(k_rot * k_rot)
        k_l2 = tl.sqrt(k_l2_sq)
        k_l2_safe = tl.where(k_l2 == 0.0, 1.0, k_l2)
        k_rot = k_rot / k_l2_safe
        k_norm_fp16 = k_l2.to(tl.float16)
        k_norm_u16 = k_norm_fp16.to(tl.uint16, bitcast=True)
        tl.store(KV_cache_ptr + slot_base + K_NORM_OFFSET,
                 (k_norm_u16 & 0xFF).to(tl.uint8))
        tl.store(KV_cache_ptr + slot_base + K_NORM_OFFSET + 1,
                 ((k_norm_u16 >> 8) & 0xFF).to(tl.uint8))

    # 3. Reshape to [N_GROUPS, GROUP_SIZE] and per-group absmax.
    k_g = tl.reshape(k_rot, [N_GROUPS_C, GROUP_SIZE_C])
    k_absmax = tl.max(tl.abs(k_g), axis=1)             # [N_GROUPS]
    k_is_zero = k_absmax == 0.0

    # 4. scale = c * absmax, cast through fp16 (SCALE_FP16_CAST).
    #    The constant `c` must be applied in fp32 precision to match the
    #    PyTorch reference (which does `safe_absmax_fp32 * c_python_float`,
    #    promoted to fp32). Without this explicit cast, Triton may evaluate
    #    `k_absmax * FP4_C` at fp64 (constexpr), drifting from the
    #    reference's snap result on boundary samples.
    k_scale_fp32 = k_absmax * tl.cast(FP4_C, tl.float32)
    k_scale_fp16 = k_scale_fp32.to(tl.float16)
    k_scale_div = k_scale_fp16.to(tl.float32)
    k_safe_div = tl.where(k_is_zero, 1.0, k_scale_div)

    # 5. Normalize & snap to sorted FP4 idx via 14 sequential tl.where.
    #    Uses strict `>` (not `>=`) to match torch.bucketize(..., right=False)
    #    semantics — `bucketize(x, b) == count(x > b)`. Without this, values
    #    that land exactly on a midpoint (e.g. x_norm = ±3.5 between FP4
    #    levels 3 and 4) snap to the wrong side.
    k_norm = k_g / k_safe_div[:, None]
    k_sorted = tl.zeros([N_GROUPS_C, GROUP_SIZE_C], dtype=tl.int32)
    for i in tl.static_range(14):
        mid = tl.load(Midpoints_ptr + i)
        k_sorted += tl.where(k_norm > mid, 1, 0)
    k_sorted = tl.where(k_is_zero[:, None], 7, k_sorted)

    # 6. Remap sorted idx → FP4 E2M1 bit pattern.
    k_bits = tl.load(SortedToBits_ptr + k_sorted)

    # 7. Pack pairs of 4-bit codes into uint8 bytes.
    #    Uses tl.sum over the pair-axis with broadcast shifts (matches the
    #    pattern in triton_turboquant_store.py — Triton doesn't support
    #    tensor slicing like `pairs[:, 0]`).
    k_bits_flat = tl.reshape(k_bits, [HEAD_DIM])
    k_pairs = tl.reshape(k_bits_flat, [HEAD_DIM // 2, 2])
    shifts_4 = tl.arange(0, 2) * 4   # 0, 4
    k_packed = tl.sum((k_pairs & 0xF) << shifts_4[None, :], axis=1).to(tl.uint8)
    k_code_addrs = slot_base + tl.arange(0, HEAD_DIM // 2)
    tl.store(KV_cache_ptr + k_code_addrs, k_packed)

    # 8. Store K scales as fp16 → 2 uint8 byte-stores per group.
    k_scales_u16 = k_scale_fp16.to(tl.uint16, bitcast=True)
    scale_offs = tl.arange(0, N_GROUPS_C) * 2
    k_scale_base = slot_base + K_SCALES_OFFSET
    tl.store(
        KV_cache_ptr + k_scale_base + scale_offs,
        (k_scales_u16 & 0xFF).to(tl.uint8),
    )
    tl.store(
        KV_cache_ptr + k_scale_base + scale_offs + 1,
        ((k_scales_u16 >> 8) & 0xFF).to(tl.uint8),
    )

    # ── V side (mirror K but skip rotation) ────────────────────────────
    v_raw = tl.load(Value_ptr + base + d_offs, mask=d_mask, other=0.0).to(tl.float32)
    v_raw = tl.where(d_mask, v_raw, 0.0)

    if USE_TOKEN_NORM:
        v_l2_sq = tl.sum(v_raw * v_raw)
        v_l2 = tl.sqrt(v_l2_sq)
        v_l2_safe = tl.where(v_l2 == 0.0, 1.0, v_l2)
        v_raw = v_raw / v_l2_safe
        v_norm_fp16 = v_l2.to(tl.float16)
        v_norm_u16 = v_norm_fp16.to(tl.uint16, bitcast=True)
        tl.store(KV_cache_ptr + slot_base + V_NORM_OFFSET,
                 (v_norm_u16 & 0xFF).to(tl.uint8))
        tl.store(KV_cache_ptr + slot_base + V_NORM_OFFSET + 1,
                 ((v_norm_u16 >> 8) & 0xFF).to(tl.uint8))

    v_g = tl.reshape(v_raw, [N_GROUPS_C, GROUP_SIZE_C])
    v_absmax = tl.max(tl.abs(v_g), axis=1)
    v_is_zero = v_absmax == 0.0

    v_scale_fp32 = v_absmax * tl.cast(FP4_C, tl.float32)
    v_scale_fp16 = v_scale_fp32.to(tl.float16)
    v_scale_div = v_scale_fp16.to(tl.float32)
    v_safe_div = tl.where(v_is_zero, 1.0, v_scale_div)

    v_norm = v_g / v_safe_div[:, None]
    v_sorted = tl.zeros([N_GROUPS_C, GROUP_SIZE_C], dtype=tl.int32)
    for i in tl.static_range(14):
        mid = tl.load(Midpoints_ptr + i)
        v_sorted += tl.where(v_norm > mid, 1, 0)
    v_sorted = tl.where(v_is_zero[:, None], 7, v_sorted)

    v_bits = tl.load(SortedToBits_ptr + v_sorted)
    v_bits_flat = tl.reshape(v_bits, [HEAD_DIM])
    v_pairs = tl.reshape(v_bits_flat, [HEAD_DIM // 2, 2])
    v_packed = tl.sum((v_pairs & 0xF) << shifts_4[None, :], axis=1).to(tl.uint8)
    v_code_addrs = slot_base + V_CODES_OFFSET + tl.arange(0, HEAD_DIM // 2)
    tl.store(KV_cache_ptr + v_code_addrs, v_packed)

    v_scales_u16 = v_scale_fp16.to(tl.uint16, bitcast=True)
    v_scale_base = slot_base + V_SCALES_OFFSET
    tl.store(
        KV_cache_ptr + v_scale_base + scale_offs,
        (v_scales_u16 & 0xFF).to(tl.uint8),
    )
    tl.store(
        KV_cache_ptr + v_scale_base + scale_offs + 1,
        ((v_scales_u16 >> 8) & 0xFF).to(tl.uint8),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Launcher
# ═══════════════════════════════════════════════════════════════════════════


def fp4_g32_store(
    key: torch.Tensor,           # [N, H, D] bf16 or fp16
    value: torch.Tensor,         # [N, H, D] bf16 or fp16
    kv_cache: torch.Tensor,      # [num_blocks, block_size, Hk, slot_size_aligned] uint8
    slot_mapping: torch.Tensor,  # [N] int64
    *,
    PiT: torch.Tensor | None = None,    # [D, D] fp32; auto-built (Sylvester Hadamard) if None
    constant_c: float | None = None,
) -> None:
    """Launch the FP4-g32 store kernel.

    Writes K/V into `kv_cache` in-place at the slots specified by
    `slot_mapping`. Negative slot indices are skipped.
    """
    if key.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"fp4_g32_store: key.dtype must be one of {_SUPPORTED_DTYPES}, got {key.dtype}"
        )
    if value.dtype != key.dtype:
        raise ValueError(
            f"fp4_g32_store: key/value dtype mismatch ({key.dtype} vs {value.dtype})"
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
            f"fp4_g32_store: kv_cache slot {padded_slot} < expected {expected_slot} for head_dim={D}"
        )
    if num_kv_heads != H:
        raise ValueError(
            f"fp4_g32_store: kv_cache num_kv_heads {num_kv_heads} != key heads {H}"
        )

    if PiT is None:
        PiT = _get_hadamard(D, key.device)
    if PiT.dtype != torch.float32:
        PiT = PiT.to(torch.float32)
    if not PiT.is_contiguous():
        PiT = PiT.contiguous()

    midpoints = _get_midpoints_tensor(key.device)
    sorted_to_bits = _get_sorted_to_bits_tensor(key.device)

    k_flat = key.reshape(NH, D).contiguous()
    v_flat = value.reshape(NH, D).contiguous()

    gs = get_group_size()
    tn = get_token_norm()
    BLOCK_D = triton.next_power_of_2(D)
    N_GROUPS_C = n_groups(D, gs)
    K_SCALES_OFFSET = k_scales_offset(D, gs)
    V_CODES_OFFSET = v_codes_offset(D, gs)
    V_SCALES_OFFSET = v_scales_offset(D, gs)
    K_NORM_OFFSET = k_norm_offset(D, gs)
    V_NORM_OFFSET = v_norm_offset(D, gs)

    # In-bytes strides into the raw uint8 view of the cache.
    # Use tensor strides — padded page layout (mixed skip+fp4 layers) spaces
    # blocks farther apart than block_size * heads * slot.
    stride_block = kv_cache.stride(0)
    stride_pos = kv_cache.stride(1)
    stride_head = kv_cache.stride(2)

    c = constant_c if constant_c is not None else get_constant_c()

    grid = (NH,)
    _fp4_g32_store_kernel[grid](
        k_flat,
        v_flat,
        PiT,
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
        USE_TOKEN_NORM=1 if tn else 0,
        K_NORM_OFFSET=K_NORM_OFFSET,
        V_NORM_OFFSET=V_NORM_OFFSET,
        num_warps=4,
        num_stages=1,
    )
