# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified (prefill + decode) Triton attention kernel for FP4-g32.

Structure mirrors ``triton_turboquant_unified_attention.py`` (the v3 unified
TQ kernel). Only the K/V load helpers diverge — everything else (GQA
stacking into ``BLOCK_M``, MFMA ``tl.dot`` QK and PV ops, 2D/3D split-KV
dispatch, fused Q rotation, main/tail split, online softmax, ``reduce_segments``
reuse) is inherited verbatim.

Why v3 instead of v1 (per-head + scalar broadcast-mul-sum)?
- v3 uses real tensor-core MFMA via ``tl.dot``. On MI300X this is ~3.4-5.6×
  faster than v1 for FP16 TQ.
- The fp4_kv_g32 cache layout (FP4 E2M1 codes + fp16 per-group-32 scales)
  works inside this kernel with two clean substitutions:
    * K tile: FP4 LUT gather + per-group fp16 scale multiply, written in
      Hadamard-rotated space (no inverse rotation — FP4's advantage over
      v3's MSE path, which materializes K via a Pi_half matmul).
    * V tile: FP4 LUT gather + per-group fp16 scale multiply, no zero-point.

Opt-in via ``VLLM_FP4_G32_V3=1``. The legacy v1-based path in
``triton_decode.py`` remains as a fallback while v3 burns in.
"""

from __future__ import annotations

import math
import os
from typing import Any

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

# Reuse the v3 helpers — same code path, only the K/V loaders differ.
from vllm.v1.attention.ops.triton_turboquant_unified_attention import (
    _find_seq_idx,
    _tq_fuse_q_rotation,
)
from vllm.v1.attention.ops.triton_unified_attention import reduce_segments

from vllm.v1.attention.ops.fp4_g32.fp4_levels import (
    FP4_BITS_TO_VALUE,
    GROUP_SIZE,
    get_group_size,
    k_codes_bytes,
    k_scales_bytes,
    k_scales_offset,
    n_groups,
    slot_size,
    v_codes_offset,
    v_scales_offset,
)
from vllm.v1.attention.ops.fp4_g32.triton_store import _kv_cache_flat

_is_hip = current_platform.is_rocm()


# ---------------------------------------------------------------------------
# FP4 decode table caches: 16-entry single LUT and 256-entry pair-LUT.
# ---------------------------------------------------------------------------

_FP4_DECODE_BF16_CACHE: dict[torch.device, torch.Tensor] = {}
_FP4_PAIR_LUT_BF16_CACHE: dict[torch.device, torch.Tensor] = {}


def _get_fp4_decode_table_bf16(device: torch.device) -> torch.Tensor:
    """16-entry FP4 E2M1 bit-pattern → bf16 LUT (per-nibble decode path)."""
    t = _FP4_DECODE_BF16_CACHE.get(device)
    if t is None:
        t = torch.tensor(FP4_BITS_TO_VALUE, device=device, dtype=torch.bfloat16)
        _FP4_DECODE_BF16_CACHE[device] = t
    return t


def _get_fp4_pair_lut_bf16(device: torch.device) -> torch.Tensor:
    """256-entry pair-LUT: row b = [FP4_LUT[b & 0xF], FP4_LUT[(b >> 4) & 0xF]] bf16.

    One byte load + one 3D gather returns both decoded nibbles per packed byte
    (FLUTE-style). Tiny (256 * 2 * 2 = 1024 bytes) → resident in L1/registers.
    """
    t = _FP4_PAIR_LUT_BF16_CACHE.get(device)
    if t is None:
        single = torch.tensor(FP4_BITS_TO_VALUE, dtype=torch.float32)  # [16]
        lut = torch.zeros(256, 2, dtype=torch.bfloat16)
        for b in range(256):
            lut[b, 0] = single[b & 0xF].item()
            lut[b, 1] = single[(b >> 4) & 0xF].item()
        t = lut.to(device).contiguous()
        _FP4_PAIR_LUT_BF16_CACHE[device] = t
    return t


# ---------------------------------------------------------------------------
# FP4-g32 K-tile dequant: returns K_T : [HEAD_SIZE_PADDED, TILE_SIZE] in Q.dtype
# ready for tl.dot(Q, K_T).
#
# Per-slot layout (per (block, pos, kv_head)) for FP4-g32:
#     [k_codes (D/2) | k_scales (2*Gk) | v_codes (D/2) | v_scales (2*Gk)]
# where Gk = D / GROUP_SIZE (e.g. 4 for D=128). All offsets baked as constexpr
# by the launcher.
# ---------------------------------------------------------------------------


@triton.jit
def _fp4_g32_load_k_tile(
    KV_cache_ptr,
    KV_cache_u16_ptr,
    data_bases,            # [TILE_SIZE] int64 — byte offset to K codes (slot_base)
    k_scales_u16_addrs,    # [TILE_SIZE] int64 — u16 element index for first K scale
    Fp4_decode_ptr,        # [16] bf16
    Pair_lut_ptr,          # [256, 2] bf16
    d_offs,                # [HEAD_SIZE_PADDED] int32
    d_mask,                # [HEAD_SIZE_PADDED] int1
    tile_mask,             # [TILE_SIZE] int1 — ignored when UNMASKED=True
    OUT_DTYPE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr,
    N_GROUPS_C: tl.constexpr,
    USE_PAIR_LUT: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    UNMASKED: tl.constexpr = False,
):
    """Load + dequant a [TILE_SIZE, BLOCK_D] block of K (Hadamard-rotated)."""
    if USE_PAIR_LUT:
        HALF_D: tl.constexpr = BLOCK_D // 2
        half_offs = tl.arange(0, HALF_D)
        byte_mask = (half_offs * 2) < HEAD_DIM
        byte_addrs = data_bases[:, None] + half_offs[None, :]
        if UNMASKED:
            byte_raw = tl.load(
                KV_cache_ptr + byte_addrs, mask=byte_mask[None, :], other=0
            ).to(tl.int32)
        else:
            byte_raw = tl.load(
                KV_cache_ptr + byte_addrs,
                mask=tile_mask[:, None] & byte_mask[None, :],
                other=0,
            ).to(tl.int32)
        pair_slot = tl.arange(0, 2)
        if UNMASKED:
            c_pair = tl.load(
                Pair_lut_ptr + byte_raw[:, :, None] * 2 + pair_slot[None, None, :],
                mask=byte_mask[None, :, None],
                other=0.0,
            )
        else:
            c_pair = tl.load(
                Pair_lut_ptr + byte_raw[:, :, None] * 2 + pair_slot[None, None, :],
                mask=tile_mask[:, None, None] & byte_mask[None, :, None],
                other=0.0,
            )
        fp4_vals = tl.reshape(c_pair, [TILE_SIZE, BLOCK_D]).to(tl.float32)
    else:
        half_idx = d_offs // 2
        nibble_shift = (d_offs % 2) * 4
        addrs = data_bases[:, None] + half_idx[None, :]
        if UNMASKED:
            byte_raw = tl.load(
                KV_cache_ptr + addrs, mask=d_mask[None, :], other=0
            ).to(tl.int32)
        else:
            byte_raw = tl.load(
                KV_cache_ptr + addrs,
                mask=tile_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
        codes = (byte_raw >> nibble_shift[None, :]) & 0xF
        fp4_vals = tl.load(Fp4_decode_ptr + codes).to(tl.float32)

    # Per-group fp16 scales: N_GROUPS scales per token. Coalesced 2-byte loads
    # via the uint16 cache view.
    grp = tl.arange(0, N_GROUPS_C)
    scale_addrs = k_scales_u16_addrs[:, None] + grp[None, :]
    if UNMASKED:
        scale_raw = tl.load(KV_cache_u16_ptr + scale_addrs)
    else:
        scale_raw = tl.load(KV_cache_u16_ptr + scale_addrs, mask=tile_mask[:, None], other=0)
    scales = scale_raw.to(tl.float16, bitcast=True).to(tl.float32)

    # Per-group broadcast multiply: reshape [TILE, BLOCK_D] → [TILE, G, GS],
    # multiply by scales[:, :, None], reshape back.
    K_g = tl.reshape(fp4_vals, [TILE_SIZE, N_GROUPS_C, GROUP_SIZE_C])
    K = tl.reshape(K_g * scales[:, :, None], [TILE_SIZE, BLOCK_D])

    K_T = tl.trans(K.to(OUT_DTYPE))  # [BLOCK_D, TILE_SIZE] ready for tl.dot(Q, K_T)
    _ = HEAD_DIM
    return K_T


# ---------------------------------------------------------------------------
# FP4-g32 V-tile dequant: returns V : [TILE_SIZE, HEAD_SIZE_PADDED] in Q.dtype
# ready for tl.dot(P, V). Identical structure to K-tile loader (no zero-point).
# ---------------------------------------------------------------------------


@triton.jit
def _fp4_g32_load_v_tile(
    KV_cache_ptr,
    KV_cache_u16_ptr,
    val_bases,             # [TILE_SIZE] int64 — byte offset to V codes
    v_scales_u16_addrs,    # [TILE_SIZE] int64 — u16 element index for first V scale
    Fp4_decode_ptr,
    Pair_lut_ptr,
    d_offs,
    d_mask,
    tile_mask,
    OUT_DTYPE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr,
    N_GROUPS_C: tl.constexpr,
    USE_PAIR_LUT: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    UNMASKED: tl.constexpr = False,
):
    """Load + dequant a [TILE_SIZE, BLOCK_D] block of V (raw space)."""
    if USE_PAIR_LUT:
        HALF_D: tl.constexpr = BLOCK_D // 2
        half_offs = tl.arange(0, HALF_D)
        byte_mask = (half_offs * 2) < HEAD_DIM
        byte_addrs = val_bases[:, None] + half_offs[None, :]
        if UNMASKED:
            byte_raw = tl.load(
                KV_cache_ptr + byte_addrs, mask=byte_mask[None, :], other=0
            ).to(tl.int32)
        else:
            byte_raw = tl.load(
                KV_cache_ptr + byte_addrs,
                mask=tile_mask[:, None] & byte_mask[None, :],
                other=0,
            ).to(tl.int32)
        pair_slot = tl.arange(0, 2)
        if UNMASKED:
            c_pair = tl.load(
                Pair_lut_ptr + byte_raw[:, :, None] * 2 + pair_slot[None, None, :],
                mask=byte_mask[None, :, None],
                other=0.0,
            )
        else:
            c_pair = tl.load(
                Pair_lut_ptr + byte_raw[:, :, None] * 2 + pair_slot[None, None, :],
                mask=tile_mask[:, None, None] & byte_mask[None, :, None],
                other=0.0,
            )
        fp4_vals = tl.reshape(c_pair, [TILE_SIZE, BLOCK_D]).to(tl.float32)
    else:
        half_idx = d_offs // 2
        nibble_shift = (d_offs % 2) * 4
        addrs = val_bases[:, None] + half_idx[None, :]
        if UNMASKED:
            byte_raw = tl.load(
                KV_cache_ptr + addrs, mask=d_mask[None, :], other=0
            ).to(tl.int32)
        else:
            byte_raw = tl.load(
                KV_cache_ptr + addrs,
                mask=tile_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
        codes = (byte_raw >> nibble_shift[None, :]) & 0xF
        fp4_vals = tl.load(Fp4_decode_ptr + codes).to(tl.float32)

    grp = tl.arange(0, N_GROUPS_C)
    scale_addrs = v_scales_u16_addrs[:, None] + grp[None, :]
    if UNMASKED:
        scale_raw = tl.load(KV_cache_u16_ptr + scale_addrs)
    else:
        scale_raw = tl.load(KV_cache_u16_ptr + scale_addrs, mask=tile_mask[:, None], other=0)
    scales = scale_raw.to(tl.float16, bitcast=True).to(tl.float32)

    V_g = tl.reshape(fp4_vals, [TILE_SIZE, N_GROUPS_C, GROUP_SIZE_C])
    V = tl.reshape(V_g * scales[:, :, None], [TILE_SIZE, BLOCK_D])

    _ = HEAD_DIM
    return V.to(OUT_DTYPE)


# ---------------------------------------------------------------------------
# Unified 2D attention kernel (prefill / short-context decode).
# Mirrors `kernel_tq_unified_attention_2d` line-for-line; only K/V loaders
# and a few constexprs differ.
# ---------------------------------------------------------------------------


@triton.jit
def kernel_fp4_g32_unified_attention_2d(
    output_ptr,
    query_ptr,
    KV_cache_ptr,
    KV_cache_u16_ptr,
    Fp4_decode_ptr,
    Pair_lut_ptr,
    PiT_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    query_start_len_ptr,
    sinks_ptr,
    scale,
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    block_table_stride: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,
    stride_cache_block: tl.int64,
    pit_stride_0: tl.int64,
    pit_stride_1: tl.int64,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
    num_seqs: tl.int32,
    # FP4-g32 layout constants
    K_SCALES_OFFSET: tl.constexpr,   # byte offset to K scales within slot
    V_CODES_OFFSET: tl.constexpr,    # byte offset to V codes within slot
    V_SCALES_OFFSET: tl.constexpr,   # byte offset to V scales within slot
    GROUP_SIZE_C: tl.constexpr,
    N_GROUPS_C: tl.constexpr,
    USE_PAIR_LUT: tl.constexpr,
    stride_cache_pos: tl.int64,
    stride_cache_head: tl.int64,
    FUSE_Q_ROT: tl.constexpr = 0,
    USE_SINKS: tl.constexpr = 0,
    USE_BF16_DOT: tl.constexpr = 0,
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    seq_idx = _find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )
    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx
    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    if FUSE_Q_ROT:
        Q = _tq_fuse_q_rotation(
            Q,
            PiT_ptr,
            pit_stride_0,
            pit_stride_1,
            dim_mask,
            HEAD_SIZE_PADDED,
        )

    block_table_offset = seq_idx * block_table_stride

    if USE_SINKS:
        M = tl.load(
            sinks_ptr + query_offset_1, mask=query_mask_1, other=float("-inf")
        ).to(tl.float32)
    else:
        M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    context_len = seq_len - cur_batch_query_len

    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // num_queries_per_kv
        + 1
    )
    max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)
    num_tiles = tl.cdiv(max_seq_prefix_len, TILE_SIZE)

    query_abs_pos = context_len + query_pos[:, None]
    dummy_tile_mask = tl.full([TILE_SIZE], 1, tl.int1)

    for j in range(0, num_tiles - 1):
        seq_offset = j * TILE_SIZE + offs_t
        if TILE_SIZE == BLOCK_SIZE:
            physical_block_idx = tl.load(
                block_tables_ptr + block_table_offset + j
            ).to(tl.int64)
            slot_within_block = offs_t.to(tl.int64)
        else:
            physical_block_idx = tl.load(
                block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
            ).to(tl.int64)
            slot_within_block = (seq_offset % BLOCK_SIZE).to(tl.int64)
        block_base = physical_block_idx * stride_cache_block
        data_bases = (
            block_base
            + slot_within_block * stride_cache_pos
            + tl.cast(kv_head_idx, tl.int64) * stride_cache_head
        )
        val_bases = data_bases + V_CODES_OFFSET
        k_scales_u16_addrs = (data_bases + K_SCALES_OFFSET) // 2
        v_scales_u16_addrs = (data_bases + V_SCALES_OFFSET) // 2

        K_T = _fp4_g32_load_k_tile(
            KV_cache_ptr,
            KV_cache_u16_ptr,
            data_bases,
            k_scales_u16_addrs,
            Fp4_decode_ptr,
            Pair_lut_ptr,
            offs_d,
            dim_mask,
            dummy_tile_mask,
            OUT_DTYPE=Q.dtype,
            HEAD_DIM=HEAD_SIZE,
            BLOCK_D=HEAD_SIZE_PADDED,
            GROUP_SIZE_C=GROUP_SIZE_C,
            N_GROUPS_C=N_GROUPS_C,
            USE_PAIR_LUT=USE_PAIR_LUT,
            TILE_SIZE=TILE_SIZE,
            UNMASKED=True,
        )
        V = _fp4_g32_load_v_tile(
            KV_cache_ptr,
            KV_cache_u16_ptr,
            val_bases,
            v_scales_u16_addrs,
            Fp4_decode_ptr,
            Pair_lut_ptr,
            offs_d,
            dim_mask,
            dummy_tile_mask,
            OUT_DTYPE=Q.dtype,
            HEAD_DIM=HEAD_SIZE,
            BLOCK_D=HEAD_SIZE_PADDED,
            GROUP_SIZE_C=GROUP_SIZE_C,
            N_GROUPS_C=N_GROUPS_C,
            USE_PAIR_LUT=USE_PAIR_LUT,
            TILE_SIZE=TILE_SIZE,
            UNMASKED=True,
        )
        if USE_BF16_DOT:
            S = scale * tl.dot(Q.to(tl.bfloat16), K_T.to(tl.bfloat16))
        else:
            S = scale * tl.dot(Q, K_T)
        seq_mask = seq_offset[None, :] <= query_abs_pos
        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask,
            S,
            float("-inf"),
        )
        m_j = tl.maximum(M, tl.max(S, axis=1))
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
        P = tl.exp(S - m_j[:, None])
        l_j = tl.sum(P, axis=1)
        alpha = tl.exp(M - m_j)
        acc = acc * alpha[:, None]
        L = L * alpha + l_j
        M = m_j
        if USE_BF16_DOT:
            acc += tl.dot(P.to(tl.bfloat16), V.to(tl.bfloat16))
        else:
            acc += tl.dot(P.to(V.dtype), V)

    # Tail tile
    if num_tiles > 0:
        j = num_tiles - 1
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len
        if TILE_SIZE == BLOCK_SIZE:
            physical_block_idx = tl.load(
                block_tables_ptr + block_table_offset + j
            ).to(tl.int64)
            slot_within_block = offs_t.to(tl.int64)
        else:
            physical_block_idx = tl.load(
                block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
            ).to(tl.int64)
            slot_within_block = (seq_offset % BLOCK_SIZE).to(tl.int64)
        block_base = physical_block_idx * stride_cache_block
        data_bases = (
            block_base
            + slot_within_block * stride_cache_pos
            + tl.cast(kv_head_idx, tl.int64) * stride_cache_head
        )
        val_bases = data_bases + V_CODES_OFFSET
        k_scales_u16_addrs = (data_bases + K_SCALES_OFFSET) // 2
        v_scales_u16_addrs = (data_bases + V_SCALES_OFFSET) // 2

        K_T = _fp4_g32_load_k_tile(
            KV_cache_ptr,
            KV_cache_u16_ptr,
            data_bases,
            k_scales_u16_addrs,
            Fp4_decode_ptr,
            Pair_lut_ptr,
            offs_d,
            dim_mask,
            tile_mask,
            OUT_DTYPE=Q.dtype,
            HEAD_DIM=HEAD_SIZE,
            BLOCK_D=HEAD_SIZE_PADDED,
            GROUP_SIZE_C=GROUP_SIZE_C,
            N_GROUPS_C=N_GROUPS_C,
            USE_PAIR_LUT=USE_PAIR_LUT,
            TILE_SIZE=TILE_SIZE,
            UNMASKED=False,
        )
        V = _fp4_g32_load_v_tile(
            KV_cache_ptr,
            KV_cache_u16_ptr,
            val_bases,
            v_scales_u16_addrs,
            Fp4_decode_ptr,
            Pair_lut_ptr,
            offs_d,
            dim_mask,
            tile_mask,
            OUT_DTYPE=Q.dtype,
            HEAD_DIM=HEAD_SIZE,
            BLOCK_D=HEAD_SIZE_PADDED,
            GROUP_SIZE_C=GROUP_SIZE_C,
            N_GROUPS_C=N_GROUPS_C,
            USE_PAIR_LUT=USE_PAIR_LUT,
            TILE_SIZE=TILE_SIZE,
            UNMASKED=False,
        )
        if USE_BF16_DOT:
            S = scale * tl.dot(Q.to(tl.bfloat16), K_T.to(tl.bfloat16))
        else:
            S = scale * tl.dot(Q, K_T)
        seq_mask = seq_offset[None, :] <= query_abs_pos
        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask,
            S,
            float("-inf"),
        )
        m_j = tl.maximum(M, tl.max(S, axis=1))
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
        P = tl.exp(S - m_j[:, None])
        l_j = tl.sum(P, axis=1)
        alpha = tl.exp(M - m_j)
        acc = acc * alpha[:, None]
        L = L * alpha + l_j
        M = m_j
        if USE_BF16_DOT:
            acc += tl.dot(P.to(tl.bfloat16), V.to(tl.bfloat16))
        else:
            acc += tl.dot(P.to(V.dtype), V)

    # Epilogue
    acc = acc / L[:, None]

    output_offset = (
        query_offset_0[:, None] * output_stride_0
        + query_offset_1[:, None] * output_stride_1
        + offs_d[None, :]
    )
    tl.store(
        output_ptr + output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )


# ---------------------------------------------------------------------------
# Unified 3D (split-KV) attention kernel for long-context decode.
# Mirrors `kernel_tq_unified_attention_3d` with the same K/V loader swap.
# ---------------------------------------------------------------------------


@triton.jit
def kernel_fp4_g32_unified_attention_3d(
    segm_output_ptr,
    segm_max_ptr,
    segm_expsum_ptr,
    query_ptr,
    KV_cache_ptr,
    KV_cache_u16_ptr,
    Fp4_decode_ptr,
    Pair_lut_ptr,
    PiT_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    query_start_len_ptr,
    sinks_ptr,
    scale,
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    block_table_stride: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    stride_cache_block: tl.int64,
    pit_stride_0: tl.int64,
    pit_stride_1: tl.int64,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
    num_seqs: tl.int32,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
    K_SCALES_OFFSET: tl.constexpr,
    V_CODES_OFFSET: tl.constexpr,
    V_SCALES_OFFSET: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr,
    N_GROUPS_C: tl.constexpr,
    USE_PAIR_LUT: tl.constexpr,
    stride_cache_pos: tl.int64,
    stride_cache_head: tl.int64,
    FUSE_Q_ROT: tl.constexpr = 0,
    USE_SINKS: tl.constexpr = 0,
    USE_BF16_DOT: tl.constexpr = 0,
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2)

    seq_idx = _find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )
    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx
    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    tiles_per_segment = tl.cdiv(seq_len, NUM_SEGMENTS_PER_SEQ * TILE_SIZE)
    if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    if FUSE_Q_ROT:
        Q = _tq_fuse_q_rotation(
            Q,
            PiT_ptr,
            pit_stride_0,
            pit_stride_1,
            dim_mask,
            HEAD_SIZE_PADDED,
        )

    block_table_offset = seq_idx * block_table_stride

    if USE_SINKS and segm_idx == 0:
        M = tl.load(
            sinks_ptr + query_offset_1, mask=query_mask_1, other=float("-inf")
        ).to(tl.float32)
    else:
        M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    context_len = seq_len - cur_batch_query_len
    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // num_queries_per_kv
        + 1
    )
    max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)
    num_tiles = tl.cdiv(max_seq_prefix_len, TILE_SIZE)

    tile_lo = segm_idx * tiles_per_segment
    tile_hi = tl.minimum((segm_idx + 1) * tiles_per_segment, num_tiles)

    query_abs_pos = context_len + query_pos[:, None]
    dummy_tile_mask = tl.full([TILE_SIZE], 1, tl.int1)
    tail = tile_hi - 1

    for j in range(tile_lo, tile_hi - 1):
        seq_offset = j * TILE_SIZE + offs_t
        if TILE_SIZE == BLOCK_SIZE:
            physical_block_idx = tl.load(
                block_tables_ptr + block_table_offset + j
            ).to(tl.int64)
            slot_within_block = offs_t.to(tl.int64)
        else:
            physical_block_idx = tl.load(
                block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
            ).to(tl.int64)
            slot_within_block = (seq_offset % BLOCK_SIZE).to(tl.int64)
        block_base = physical_block_idx * stride_cache_block
        data_bases = (
            block_base
            + slot_within_block * stride_cache_pos
            + tl.cast(kv_head_idx, tl.int64) * stride_cache_head
        )
        val_bases = data_bases + V_CODES_OFFSET
        k_scales_u16_addrs = (data_bases + K_SCALES_OFFSET) // 2
        v_scales_u16_addrs = (data_bases + V_SCALES_OFFSET) // 2

        K_T = _fp4_g32_load_k_tile(
            KV_cache_ptr, KV_cache_u16_ptr, data_bases, k_scales_u16_addrs,
            Fp4_decode_ptr, Pair_lut_ptr, offs_d, dim_mask, dummy_tile_mask,
            OUT_DTYPE=Q.dtype, HEAD_DIM=HEAD_SIZE, BLOCK_D=HEAD_SIZE_PADDED,
            GROUP_SIZE_C=GROUP_SIZE_C, N_GROUPS_C=N_GROUPS_C,
            USE_PAIR_LUT=USE_PAIR_LUT, TILE_SIZE=TILE_SIZE, UNMASKED=True,
        )
        V = _fp4_g32_load_v_tile(
            KV_cache_ptr, KV_cache_u16_ptr, val_bases, v_scales_u16_addrs,
            Fp4_decode_ptr, Pair_lut_ptr, offs_d, dim_mask, dummy_tile_mask,
            OUT_DTYPE=Q.dtype, HEAD_DIM=HEAD_SIZE, BLOCK_D=HEAD_SIZE_PADDED,
            GROUP_SIZE_C=GROUP_SIZE_C, N_GROUPS_C=N_GROUPS_C,
            USE_PAIR_LUT=USE_PAIR_LUT, TILE_SIZE=TILE_SIZE, UNMASKED=True,
        )
        if USE_BF16_DOT:
            S = scale * tl.dot(Q.to(tl.bfloat16), K_T.to(tl.bfloat16))
        else:
            S = scale * tl.dot(Q, K_T)
        seq_mask = seq_offset[None, :] <= query_abs_pos
        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask,
            S,
            float("-inf"),
        )
        m_j = tl.maximum(M, tl.max(S, axis=1))
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
        P = tl.exp(S - m_j[:, None])
        l_j = tl.sum(P, axis=1)
        alpha = tl.exp(M - m_j)
        acc = acc * alpha[:, None]
        L = L * alpha + l_j
        M = m_j
        if USE_BF16_DOT:
            acc += tl.dot(P.to(tl.bfloat16), V.to(tl.bfloat16))
        else:
            acc += tl.dot(P.to(V.dtype), V)

    # Tail
    if tile_lo < tile_hi:
        j = tail
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len
        if TILE_SIZE == BLOCK_SIZE:
            physical_block_idx = tl.load(
                block_tables_ptr + block_table_offset + j
            ).to(tl.int64)
            slot_within_block = offs_t.to(tl.int64)
        else:
            physical_block_idx = tl.load(
                block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
            ).to(tl.int64)
            slot_within_block = (seq_offset % BLOCK_SIZE).to(tl.int64)
        block_base = physical_block_idx * stride_cache_block
        data_bases = (
            block_base
            + slot_within_block * stride_cache_pos
            + tl.cast(kv_head_idx, tl.int64) * stride_cache_head
        )
        val_bases = data_bases + V_CODES_OFFSET
        k_scales_u16_addrs = (data_bases + K_SCALES_OFFSET) // 2
        v_scales_u16_addrs = (data_bases + V_SCALES_OFFSET) // 2

        K_T = _fp4_g32_load_k_tile(
            KV_cache_ptr, KV_cache_u16_ptr, data_bases, k_scales_u16_addrs,
            Fp4_decode_ptr, Pair_lut_ptr, offs_d, dim_mask, tile_mask,
            OUT_DTYPE=Q.dtype, HEAD_DIM=HEAD_SIZE, BLOCK_D=HEAD_SIZE_PADDED,
            GROUP_SIZE_C=GROUP_SIZE_C, N_GROUPS_C=N_GROUPS_C,
            USE_PAIR_LUT=USE_PAIR_LUT, TILE_SIZE=TILE_SIZE, UNMASKED=False,
        )
        V = _fp4_g32_load_v_tile(
            KV_cache_ptr, KV_cache_u16_ptr, val_bases, v_scales_u16_addrs,
            Fp4_decode_ptr, Pair_lut_ptr, offs_d, dim_mask, tile_mask,
            OUT_DTYPE=Q.dtype, HEAD_DIM=HEAD_SIZE, BLOCK_D=HEAD_SIZE_PADDED,
            GROUP_SIZE_C=GROUP_SIZE_C, N_GROUPS_C=N_GROUPS_C,
            USE_PAIR_LUT=USE_PAIR_LUT, TILE_SIZE=TILE_SIZE, UNMASKED=False,
        )
        if USE_BF16_DOT:
            S = scale * tl.dot(Q.to(tl.bfloat16), K_T.to(tl.bfloat16))
        else:
            S = scale * tl.dot(Q, K_T)
        seq_mask = seq_offset[None, :] <= query_abs_pos
        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask,
            S,
            float("-inf"),
        )
        m_j = tl.maximum(M, tl.max(S, axis=1))
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
        P = tl.exp(S - m_j[:, None])
        l_j = tl.sum(P, axis=1)
        alpha = tl.exp(M - m_j)
        acc = acc * alpha[:, None]
        L = L * alpha + l_j
        M = m_j
        if USE_BF16_DOT:
            acc += tl.dot(P.to(tl.bfloat16), V.to(tl.bfloat16))
        else:
            acc += tl.dot(P.to(V.dtype), V)

    # Write partials for reduce_segments
    segm_output_offset = (
        query_offset_0[:, None].to(tl.int64)
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + segm_idx * HEAD_SIZE_PADDED
        + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
    )
    tl.store(
        segm_output_ptr + segm_output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )
    segm_offset = (
        query_offset_0.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_offset_1 * NUM_SEGMENTS_PER_SEQ
        + segm_idx
    )
    tl.store(segm_max_ptr + segm_offset, M, mask=query_mask_0 & query_mask_1)
    tl.store(segm_expsum_ptr + segm_offset, L, mask=query_mask_0 & query_mask_1)


# ---------------------------------------------------------------------------
# Launcher
# ---------------------------------------------------------------------------


_HADAMARD_CACHE: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}


def _get_pit(dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Sylvester Hadamard projection / sqrt(dim). PiT = PiT.T (symmetric)."""
    key = (dim, device, dtype)
    H = _HADAMARD_CACHE.get(key)
    if H is None:
        assert (dim & (dim - 1)) == 0, f"dim={dim} must be power of 2"
        H = torch.tensor([[1.0]], dtype=torch.float32)
        while H.shape[0] < dim:
            H = torch.cat(
                [torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0
            )
        H = (H / (dim**0.5)).to(device=device, dtype=dtype).contiguous()
        _HADAMARD_CACHE[key] = H
    return H


def fp4_g32_unified_attention(
    query: torch.Tensor,             # [num_tokens, Hq, D] - fp16/bf16 (raw, kernel rotates)
    kv_cache: torch.Tensor,          # [num_blocks, block_size, Hk, padded_slot] uint8
    block_table: torch.Tensor,       # [num_seqs, max_num_blocks] int32
    seq_lens: torch.Tensor,          # [num_seqs] int32
    query_start_loc: torch.Tensor,   # [num_seqs+1] int32
    scale: float,
    PiT: torch.Tensor | None = None,  # [D, D] fp32 — auto-built (Sylvester) if None
    output: torch.Tensor | None = None,
    tile_size: int | None = None,
    max_query_len: int | None = None,
    max_seq_len: int | None = None,
    num_kv_splits: int | None = None,
    force_2d: bool = False,
    fuse_q_rot: bool | None = None,
    sinks: torch.Tensor | None = None,
) -> torch.Tensor:
    """Launch unified FP4-g32 attention (v3-style).

    Single entry point for prefill (BLOCK_Q > 1) and decode (BLOCK_Q = 1).
    Replaces both the legacy ``fp4_g32_decode_attention`` and
    ``_fp4_g32_continuation_prefill`` paths. Returns ``output`` of shape
    ``[num_tokens, Hq, D]`` in ``query.dtype``.
    """
    assert query.dim() == 3, f"query must be [N, Hq, D], got {query.shape}"
    num_tokens, Hq, D = query.shape
    Hk = kv_cache.shape[2]
    block_size = kv_cache.shape[1]
    padded_slot = kv_cache.shape[3]
    kv_group_size = Hq // Hk
    num_seqs = int(query_start_loc.shape[0] - 1)
    device = query.device

    gs = get_group_size()
    if padded_slot < slot_size(D, gs):
        raise ValueError(
            f"fp4_g32_unified: cache slot {padded_slot} < expected {slot_size(D, gs)} "
            f"for D={D} group_size={gs}"
        )
    if Hq % Hk != 0:
        raise ValueError(f"Hq={Hq} must be a multiple of Hk={Hk}")

    # Q rotation strategy (mirrors v3's MSE-key path; FP4 always rotates since
    # K is stored in Hadamard-rotated space).
    if PiT is None:
        PiT = _get_pit(D, device, torch.float32)
    elif PiT.dtype != torch.float32 or not PiT.is_contiguous():
        PiT = PiT.to(torch.float32).contiguous()

    # Default: fp32 launcher-side rotation (matches the v1 fp4_g32 path's
    # precision; the kernel's fused rotation uses bf16 PiT, which empirically
    # caused 3/11 borderline answer flips at 128K context in lcb_128k smoke).
    # Override via VLLM_FP4_G32_FUSE_Q_ROT=1 if you want to A/B for perf.
    if fuse_q_rot is None:
        fuse_q_rot = os.environ.get("VLLM_FP4_G32_FUSE_Q_ROT", "0") == "1"
    apply_fuse_q_rot = bool(fuse_q_rot)
    if apply_fuse_q_rot:
        q_rot = query.contiguous()
    else:
        q_rot = (query.float() @ PiT).to(query.dtype).contiguous()

    pit_stride_0 = PiT.stride(0)
    pit_stride_1 = PiT.stride(1)

    if sinks is not None:
        sinks_f32 = sinks if sinks.dtype == torch.float32 else sinks.to(torch.float32)
        if not sinks_f32.is_contiguous():
            sinks_f32 = sinks_f32.contiguous()
        assert sinks_f32.numel() == Hq, (
            f"sinks must have shape [Hq={Hq}], got numel={sinks_f32.numel()}"
        )
        use_sinks = True
    else:
        sinks_f32 = PiT  # harmless dummy; never dereferenced when USE_SINKS=0
        use_sinks = False

    if output is None:
        output = torch.empty_like(query)

    # BLOCK_M heuristic (mirrors v3's TQ-specific choice).
    if max_query_len is not None:
        is_prefill_like = max_query_len > 1
    else:
        is_prefill_like = num_tokens > num_seqs

    if is_prefill_like:
        BLOCK_M = max(128, triton.next_power_of_2(kv_group_size))
    else:
        BLOCK_M = 16 if kv_group_size <= 16 else triton.next_power_of_2(kv_group_size)
    BLOCK_Q = BLOCK_M // kv_group_size

    total_num_q_blocks = num_tokens // BLOCK_Q + num_seqs

    # Pair-LUT fast path for FP4 (always 4-bit, single 16-entry LUT, so pair-LUT
    # of 256 entries is essentially free).
    use_pair_lut = True
    fp4_dec = _get_fp4_decode_table_bf16(device)
    pair_lut = _get_fp4_pair_lut_bf16(device) if use_pair_lut else fp4_dec

    # TILE_SIZE heuristic mirrors v3: 32 for prefill, 16 for decode.
    if tile_size is None:
        tile_size = 32 if is_prefill_like else 16
    _tile_override = os.environ.get("VLLM_FP4_G32_TILE_SIZE_DECODE")
    if (not is_prefill_like) and _tile_override is not None:
        tile_size = int(_tile_override)

    num_stages = int(os.environ.get("VLLM_FP4_G32_NUM_STAGES", "1" if _is_hip else "2"))
    use_bf16_dot = 1 if (_is_hip and query.dtype == torch.bfloat16) else 0

    kv_cache_u16 = kv_cache.view(torch.uint16)

    BLOCK_D = triton.next_power_of_2(D)
    N_GROUPS_C = n_groups(D, gs)

    # Layout constants — baked as constexpr.
    K_SCALES_OFFSET = k_scales_offset(D, gs)
    V_CODES_OFFSET = v_codes_offset(D, gs)
    V_SCALES_OFFSET = v_scales_offset(D, gs)

    # Dispatch: 2D for prefill / chunked; 3D for pure decode with long KV.
    if max_seq_len is None:
        max_seq_len_hint = int(block_table.shape[1]) * int(block_size)
    else:
        max_seq_len_hint = int(max_seq_len)
    use_3d = (not force_2d) and (not is_prefill_like) and max_seq_len_hint >= 1024

    if not use_3d:
        kernel_fp4_g32_unified_attention_2d[(total_num_q_blocks, Hk)](
            output_ptr=output,
            query_ptr=q_rot,
            KV_cache_ptr=kv_cache,
            KV_cache_u16_ptr=kv_cache_u16,
            Fp4_decode_ptr=fp4_dec,
            Pair_lut_ptr=pair_lut,
            PiT_ptr=PiT,
            block_tables_ptr=block_table,
            seq_lens_ptr=seq_lens,
            query_start_len_ptr=query_start_loc,
            sinks_ptr=sinks_f32,
            scale=scale,
            num_query_heads=Hq,
            num_queries_per_kv=kv_group_size,
            block_table_stride=block_table.stride(0),
            query_stride_0=q_rot.stride(0),
            query_stride_1=q_rot.stride(1),
            output_stride_0=output.stride(0),
            output_stride_1=output.stride(1),
            stride_cache_block=kv_cache.stride(0),
            pit_stride_0=pit_stride_0,
            pit_stride_1=pit_stride_1,
            BLOCK_SIZE=block_size,
            TILE_SIZE=tile_size,
            HEAD_SIZE=D,
            HEAD_SIZE_PADDED=BLOCK_D,
            BLOCK_Q=BLOCK_Q,
            BLOCK_M=BLOCK_M,
            num_seqs=num_seqs,
            K_SCALES_OFFSET=K_SCALES_OFFSET,
            V_CODES_OFFSET=V_CODES_OFFSET,
            V_SCALES_OFFSET=V_SCALES_OFFSET,
            GROUP_SIZE_C=gs,
            N_GROUPS_C=N_GROUPS_C,
            USE_PAIR_LUT=1 if use_pair_lut else 0,
            stride_cache_pos=kv_cache.stride(1),
            stride_cache_head=kv_cache.stride(2),
            FUSE_Q_ROT=1 if apply_fuse_q_rot else 0,
            USE_SINKS=1 if use_sinks else 0,
            USE_BF16_DOT=use_bf16_dot,
            num_warps=4,
            num_stages=num_stages,
        )
        return output

    # 3D split-KV path
    if num_kv_splits is None:
        num_kv_splits = 16
    max_possible_splits = max(1, (max_seq_len_hint + tile_size - 1) // tile_size)
    num_segments = max(1, min(num_kv_splits, max_possible_splits))

    segm_output = torch.empty(
        (num_tokens, Hq, num_segments, BLOCK_D),
        dtype=torch.float32, device=device,
    )
    segm_max = torch.empty(
        (num_tokens, Hq, num_segments), dtype=torch.float32, device=device
    )
    segm_expsum = torch.empty(
        (num_tokens, Hq, num_segments), dtype=torch.float32, device=device
    )

    kernel_fp4_g32_unified_attention_3d[(total_num_q_blocks, Hk, num_segments)](
        segm_output_ptr=segm_output,
        segm_max_ptr=segm_max,
        segm_expsum_ptr=segm_expsum,
        query_ptr=q_rot,
        KV_cache_ptr=kv_cache,
        KV_cache_u16_ptr=kv_cache_u16,
        Fp4_decode_ptr=fp4_dec,
        Pair_lut_ptr=pair_lut,
        PiT_ptr=PiT,
        block_tables_ptr=block_table,
        seq_lens_ptr=seq_lens,
        query_start_len_ptr=query_start_loc,
        sinks_ptr=sinks_f32,
        scale=scale,
        num_query_heads=Hq,
        num_queries_per_kv=kv_group_size,
        block_table_stride=block_table.stride(0),
        query_stride_0=q_rot.stride(0),
        query_stride_1=q_rot.stride(1),
        stride_cache_block=kv_cache.stride(0),
        pit_stride_0=pit_stride_0,
        pit_stride_1=pit_stride_1,
        BLOCK_SIZE=block_size,
        TILE_SIZE=tile_size,
        HEAD_SIZE=D,
        HEAD_SIZE_PADDED=BLOCK_D,
        BLOCK_Q=BLOCK_Q,
        BLOCK_M=BLOCK_M,
        num_seqs=num_seqs,
        NUM_SEGMENTS_PER_SEQ=num_segments,
        K_SCALES_OFFSET=K_SCALES_OFFSET,
        V_CODES_OFFSET=V_CODES_OFFSET,
        V_SCALES_OFFSET=V_SCALES_OFFSET,
        GROUP_SIZE_C=gs,
        N_GROUPS_C=N_GROUPS_C,
        USE_PAIR_LUT=1 if use_pair_lut else 0,
        stride_cache_pos=kv_cache.stride(1),
        stride_cache_head=kv_cache.stride(2),
        FUSE_Q_ROT=1 if apply_fuse_q_rot else 0,
        USE_SINKS=1 if use_sinks else 0,
        USE_BF16_DOT=use_bf16_dot,
        num_warps=int(os.environ.get("VLLM_FP4_G32_NUM_WARPS_3D", "2")),
        num_stages=num_stages,
    )

    # Reduce segment partials → final output (reuse v3's reducer; layout matches).
    reduce_segments[(num_tokens, Hq)](
        output_ptr=output,
        segm_output_ptr=segm_output,
        segm_max_ptr=segm_max,
        segm_expsum_ptr=segm_expsum,
        seq_lens_ptr=seq_lens,
        num_seqs=num_seqs,
        num_query_heads=Hq,
        out_scale_inv=1.0,
        output_stride_0=output.stride(0),
        output_stride_1=output.stride(1),
        block_table_stride=block_table.stride(0),
        TILE_SIZE=tile_size,
        HEAD_SIZE=D,
        HEAD_SIZE_PADDED=BLOCK_D,
        query_start_len_ptr=query_start_loc,
        BLOCK_Q=BLOCK_Q,
        NUM_SEGMENTS_PER_SEQ=num_segments,
        USE_FP8=False,
    )

    return output
