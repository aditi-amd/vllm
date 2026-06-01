# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton decode attention kernel for the FP4-g32 KV cache format.

Architecture mirrors `triton_turboquant_decode._tq_decode_stage1`:
- Stage 1 (this kernel): per (batch, q_head, kv_split) program does
  FA-2-style online softmax over its KV tile range, writing partial
  outputs to `mid_o [B, Hq, NUM_KV_SPLITS, HEAD_DIM + 1]` (LSE at slot
  HEAD_DIM).
- Stage 2: reuses `_fwd_kernel_stage2` from triton_decode_attention.

FP4-g32 specifics inside stage 1:
- K codes (4-bit, 32-per-byte) are loaded as uint8, decoded to bf16
  via a small 16-entry register-resident lookup (the "FP4 decode
  table"). This table is NOT a learned codebook — it's the universal
  FP4 E2M1 level grid `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}` — and
  disappears when native FP4 MFMA hardware lands.
- K's per-group fp16 scales are loaded as uint16, bitcast to fp16,
  promoted to fp32. The QK reduction is done per group of 32 and the
  scale is fused on the fp32 accumulator (one mul per (token, group)
  instead of per (token, dim)). This mirrors the math the MXFP4 MFMA
  hardware does internally.
- V is decoded the same way, but its scale is broadcast per element
  before the PV multiply (no algebraic shortcut since PV reduces over
  the token axis, not head_dim).

Sink support is included from v1 via `USE_SINKS` constexpr — only the
first KV split applies `m_prev = sink_logit`, matching TQ v3's split-aware
sink semantics (`triton_turboquant_decode.py:140-149`).
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton

from vllm.v1.attention.ops.fp4_g32.fp4_levels import (
    FP4_BITS_TO_VALUE,
    GROUP_SIZE,
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
from vllm.v1.attention.ops.fp4_g32.triton_store import _kv_cache_flat
from vllm.v1.attention.ops.triton_decode_attention import _fwd_kernel_stage2


_FP4_DECODE_CACHE: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}


# ═══════════════════════════════════════════════════════════════════════════
# Full dequant kernel — used by continuation prefill
# Mirrors `triton_turboquant_decode._tq_full_dequant_kv`:
#   grid = (max_seq, B * NUM_KV_HEADS); each program dequants one (token, head)
#   slot and writes the [HEAD_DIM] fp16/bf16 vector to the caller's output
#   buffer. K is preserved in Hadamard-rotated space (as stored); V is
#   reconstructed in original space. No inverse-rotation matmul needed — the
#   subsequent flash_attn runs Q in rotated space against K in rotated space.
# ═══════════════════════════════════════════════════════════════════════════


@triton.jit
def _fp4_g32_full_dequant_kv(
    KV_cache_ptr,         # uint8 view, flat
    KV_cache_u16_ptr,     # uint16 view (for fp16 scale loads)
    Block_table_ptr,      # [B, max_num_blocks] int32
    Fp4_decode_ptr,       # [16] bf16 — FP4 bit-pattern → bf16 value table
    K_out_ptr,            # [B, Hk, max_seq, D] in out_dtype
    V_out_ptr,            # [B, Hk, max_seq, D] in out_dtype
    stride_ko_b: tl.int64,
    stride_ko_h: tl.int64,
    stride_ko_s: tl.int64,
    stride_vo_b: tl.int64,
    stride_vo_h: tl.int64,
    stride_vo_s: tl.int64,
    stride_cache_block: tl.int64,
    stride_cache_pos: tl.int64,
    stride_cache_head: tl.int64,
    stride_bt_b: tl.int64,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    K_SCALES_OFFSET: tl.constexpr,
    V_CODES_OFFSET: tl.constexpr,
    V_SCALES_OFFSET: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr,
    N_GROUPS_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
    OUT_BF16: tl.constexpr,
    USE_TOKEN_NORM: tl.constexpr,
    K_NORM_OFFSET: tl.constexpr,
    V_NORM_OFFSET: tl.constexpr,
):
    pos = tl.program_id(0)
    bh = tl.program_id(1)
    bid = bh // NUM_KV_HEADS
    hid = bh % NUM_KV_HEADS

    page_idx = pos // BLOCK_SIZE
    page_off = pos % BLOCK_SIZE
    block_num = tl.load(Block_table_ptr + bid * stride_bt_b + page_idx).to(tl.int64)
    slot_base = (
        block_num * stride_cache_block
        + page_off * stride_cache_pos
        + tl.cast(hid, tl.int64) * stride_cache_head
    )

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    byte_idx = (d_offs // 2).to(tl.int32)
    nibble_shift = ((d_offs % 2) * 4).to(tl.int32)
    g_offs = tl.arange(0, N_GROUPS_C)

    out_dtype = tl.bfloat16 if OUT_BF16 else tl.float16

    # ── K dequant: codes → FP4 LUT → group-scale multiply (rotated space) ──
    k_byte_raw = tl.load(
        KV_cache_ptr + slot_base + byte_idx, mask=d_mask, other=0
    ).to(tl.int32)
    k_codes = (k_byte_raw >> nibble_shift) & 0xF
    k_dec = tl.load(Fp4_decode_ptr + k_codes).to(tl.float32)
    k_dec = tl.where(d_mask, k_dec, 0.0)

    k_scale_u16_addrs = (slot_base + K_SCALES_OFFSET) // 2 + g_offs
    k_scales_u16 = tl.load(KV_cache_u16_ptr + k_scale_u16_addrs)
    k_scales = k_scales_u16.to(tl.float16, bitcast=True).to(tl.float32)

    k_g = tl.reshape(k_dec, [N_GROUPS_C, GROUP_SIZE_C])
    k_recon = tl.reshape(k_g * k_scales[:, None], [BLOCK_D])
    k_recon = tl.where(d_mask, k_recon, 0.0)

    if USE_TOKEN_NORM:
        k_norm_u16 = tl.load(KV_cache_u16_ptr + (slot_base + K_NORM_OFFSET) // 2)
        k_norm_f32 = k_norm_u16.to(tl.float16, bitcast=True).to(tl.float32)
        k_recon = k_recon * k_norm_f32

    ko_base = bid * stride_ko_b + hid * stride_ko_h + pos * stride_ko_s
    tl.store(K_out_ptr + ko_base + d_offs, k_recon.to(out_dtype), mask=d_mask)

    # ── V dequant: codes → FP4 LUT → group-scale multiply (raw space) ──────
    v_byte_raw = tl.load(
        KV_cache_ptr + slot_base + V_CODES_OFFSET + byte_idx, mask=d_mask, other=0
    ).to(tl.int32)
    v_codes = (v_byte_raw >> nibble_shift) & 0xF
    v_dec = tl.load(Fp4_decode_ptr + v_codes).to(tl.float32)
    v_dec = tl.where(d_mask, v_dec, 0.0)

    v_scale_u16_addrs = (slot_base + V_SCALES_OFFSET) // 2 + g_offs
    v_scales_u16 = tl.load(KV_cache_u16_ptr + v_scale_u16_addrs)
    v_scales = v_scales_u16.to(tl.float16, bitcast=True).to(tl.float32)

    v_g = tl.reshape(v_dec, [N_GROUPS_C, GROUP_SIZE_C])
    v_recon = tl.reshape(v_g * v_scales[:, None], [BLOCK_D])
    v_recon = tl.where(d_mask, v_recon, 0.0)

    if USE_TOKEN_NORM:
        v_norm_u16 = tl.load(KV_cache_u16_ptr + (slot_base + V_NORM_OFFSET) // 2)
        v_norm_f32 = v_norm_u16.to(tl.float16, bitcast=True).to(tl.float32)
        v_recon = v_recon * v_norm_f32

    vo_base = bid * stride_vo_b + hid * stride_vo_h + pos * stride_vo_s
    tl.store(V_out_ptr + vo_base + d_offs, v_recon.to(out_dtype), mask=d_mask)


def fp4_g32_full_dequant_kv(
    kv_cache: torch.Tensor,    # [num_blocks, block_size, Hk, slot_size_aligned] uint8
    block_table: torch.Tensor, # [B, max_num_blocks] int32
    k_out: torch.Tensor,       # [B, Hk, max_seq, D] — pre-allocated, in out_dtype
    v_out: torch.Tensor,       # [B, Hk, max_seq, D] — pre-allocated, in out_dtype
    alloc_len: int,            # number of positions to dequant (≤ max_seq)
) -> None:
    """Triton launcher for full FP4-g32 KV dequant.

    Writes K (Hadamard-rotated) and V (raw) into pre-allocated output buffers.
    Mirrors `_tq_full_dequant_kv` launcher in v1's continuation prefill.
    """
    device = kv_cache.device
    B = block_table.shape[0]
    Hk = kv_cache.shape[2]
    D = k_out.shape[3]
    block_size = kv_cache.shape[1]
    BLOCK_D = triton.next_power_of_2(D)

    fp4_dec = _get_fp4_decode_table(device)  # [16] bf16
    kv_flat = _kv_cache_flat(kv_cache)
    kv_flat_u16 = kv_flat.view(torch.uint16)

    gs = get_group_size()
    tn = get_token_norm()
    out_bf16 = 1 if k_out.dtype == torch.bfloat16 else 0
    grid = (alloc_len, B * Hk)
    _fp4_g32_full_dequant_kv[grid](
        kv_flat,
        kv_flat_u16,
        block_table,
        fp4_dec,
        k_out,
        v_out,
        k_out.stride(0), k_out.stride(1), k_out.stride(2),
        v_out.stride(0), v_out.stride(1), v_out.stride(2),
        kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
        block_table.stride(0),
        HEAD_DIM=D,
        BLOCK_SIZE=block_size,
        NUM_KV_HEADS=Hk,
        K_SCALES_OFFSET=k_scales_offset(D, gs),
        V_CODES_OFFSET=v_codes_offset(D, gs),
        V_SCALES_OFFSET=v_scales_offset(D, gs),
        GROUP_SIZE_C=gs,
        N_GROUPS_C=n_groups(D, gs),
        BLOCK_D=BLOCK_D,
        OUT_BF16=out_bf16,
        USE_TOKEN_NORM=1 if tn else 0,
        K_NORM_OFFSET=k_norm_offset(D, gs),
        V_NORM_OFFSET=v_norm_offset(D, gs),
        num_warps=4,
    )


def fp4_g32_dequant_cached_kv(
    kv_cache: torch.Tensor,    # [num_blocks, block_size, Hk, slot_size_aligned] uint8
    block_table: torch.Tensor, # [1, max_num_blocks] int32
    cached_len: int,
    head_dim: int,
    out_dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequantize cached K and V from the fp4 paged KV cache (auto-allocate).

    Thin wrapper around `fp4_g32_full_dequant_kv` for callers that don't
    manage output buffers. Performance-sensitive callers (e.g. continuation
    prefill) should call `fp4_g32_full_dequant_kv` directly with reused
    buffers to avoid per-call allocation.

    Returns:
        K_rot: [cached_len, Hk, head_dim] in out_dtype
        V_raw: [cached_len, Hk, head_dim] in out_dtype
    """
    import math
    device = kv_cache.device
    block_size = kv_cache.shape[1]
    Hk = kv_cache.shape[2]

    alloc_len = math.ceil(cached_len / block_size) * block_size
    k_buf = torch.empty(1, Hk, alloc_len, head_dim, dtype=out_dtype, device=device)
    v_buf = torch.empty(1, Hk, alloc_len, head_dim, dtype=out_dtype, device=device)
    fp4_g32_full_dequant_kv(kv_cache, block_table, k_buf, v_buf, alloc_len)
    # Transpose [1, Hk, cached_len, D] → [cached_len, Hk, D]
    k_rot = k_buf[0, :, :cached_len, :].transpose(0, 1).contiguous()
    v_raw = v_buf[0, :, :cached_len, :].transpose(0, 1).contiguous()
    return k_rot, v_raw


def _get_fp4_decode_table(device: torch.device) -> torch.Tensor:
    """16-entry FP4 E2M1 bit-pattern → bf16 value table.

    Not a learned codebook: this is the OCP FP4 E2M1 universal value grid
    `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}` (with the redundant -0 entry
    pointing to 0.0). Decoded by the Triton kernel via a single small
    indexed load that stays in registers / L1. Replaced by native FP4
    MFMA hardware decoding once we wire up `batched_gemm_a16wfp4`-style
    paths.
    """
    key = (device, torch.bfloat16)
    t = _FP4_DECODE_CACHE.get(key)
    if t is None:
        t = torch.tensor(FP4_BITS_TO_VALUE, device=device, dtype=torch.bfloat16)
        _FP4_DECODE_CACHE[key] = t
    return t


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1
# ═══════════════════════════════════════════════════════════════════════════


@triton.jit
def _fp4_g32_decode_stage1(
    Q_rot_ptr,            # [B, Hq, HEAD_DIM] fp32 (Q pre-rotated by launcher)
    KV_cache_ptr,         # uint8 view of cache, flat
    KV_cache_u16_ptr,     # uint16 view of cache, flat (for scale loads)
    Block_table_ptr,      # [B, max_blocks] int32
    Seq_lens_ptr,         # [B] int32
    Fp4_decode_ptr,       # [16] bf16 — FP4 E2M1 bit → bf16 value
    Mid_o_ptr,            # [B, Hq, NUM_KV_SPLITS, HEAD_DIM+1] fp32
    Sink_ptr,             # [Hq] fp32 — sink logits (may be NULL)
    # Q stride
    stride_qb: tl.int64, stride_qh: tl.int64,
    # Cache strides (in bytes / uint8 elements)
    stride_cache_block: tl.int64,
    stride_cache_pos: tl.int64,
    stride_cache_head: tl.int64,
    # Block table stride
    stride_bt_b: tl.int64,
    # mid_o strides
    stride_mid_b: tl.int64,
    stride_mid_h: tl.int64,
    stride_mid_s: tl.int64,
    # Constexpr dims
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    # FP4-g32 layout
    GROUP_SIZE_C: tl.constexpr,
    N_GROUPS_C: tl.constexpr,
    K_SCALES_OFFSET: tl.constexpr,
    V_CODES_OFFSET: tl.constexpr,
    V_SCALES_OFFSET: tl.constexpr,
    # Attention
    ATTN_SCALE: tl.constexpr,
    # Tile sizes
    BLOCK_D: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    USE_SINKS: tl.constexpr,
    # Per-token L2 norm
    USE_TOKEN_NORM: tl.constexpr,
    K_NORM_OFFSET: tl.constexpr,
    V_NORM_OFFSET: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    sid = tl.program_id(2)

    kv_head = hid // KV_GROUP_SIZE

    seq_len = tl.load(Seq_lens_ptr + bid)
    split_len = tl.cdiv(seq_len, NUM_KV_SPLITS)
    split_start = split_len * sid
    split_end = tl.minimum(split_start + split_len, seq_len)
    if split_start >= split_end:
        return

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    kv_range = tl.arange(0, BLOCK_KV)

    # Load Q (rotated): [BLOCK_D] fp32 → keep in fp32 for the per-group dot
    # accumulator math; the actual partial sums happen in fp32 anyway.
    q_base = bid * stride_qb + hid * stride_qh
    q_rot = tl.load(Q_rot_ptr + q_base + d_offs, mask=d_mask, other=0.0).to(
        tl.float32
    )

    # Online softmax accumulators (FA-2 style). Sink only fires for sid==0
    # so reduce-stage averages it in exactly once across splits.
    if USE_SINKS:
        if sid == 0:
            m_prev = tl.load(Sink_ptr + hid).to(tl.float32)
            l_prev = 1.0
        else:
            m_prev = -float("inf")
            l_prev = 0.0
    else:
        m_prev = -float("inf")
        l_prev = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    bt_base = bid * stride_bt_b

    # Precompute per-element addressing patterns (loop-invariant).
    # Bytes for K codes: each token has HEAD_DIM // 2 bytes laid out
    # contiguously starting at slot_base. Each byte holds two nibbles.
    byte_idx = (d_offs // 2).to(tl.int32)
    nibble_shift = ((d_offs % 2) * 4).to(tl.int32)

    # q_rot is loop-invariant — reshape once outside the loop.
    # On AMD MLIR/Triton the compiler may not hoist this automatically.
    q_g = tl.reshape(q_rot, [N_GROUPS_C, GROUP_SIZE_C])       # [G, GS] fp32

    for start_n in range(split_start, split_end, BLOCK_KV):
        kv_offs = start_n + kv_range
        kv_mask = kv_offs < split_end

        page_idx = kv_offs // BLOCK_SIZE
        page_off = kv_offs % BLOCK_SIZE
        block_nums = tl.load(
            Block_table_ptr + bt_base + page_idx,
            mask=kv_mask,
            other=0,
        ).to(tl.int64)

        slot_within_block = page_off.to(tl.int64)
        slot_bases = (
            block_nums * stride_cache_block
            + slot_within_block * stride_cache_pos
            + tl.cast(kv_head, tl.int64) * stride_cache_head
        )  # [BLOCK_KV] int64

        # ── K codes load + decode ──────────────────────────────────────
        # Address: slot_bases[:, None] + byte_idx[None, :]
        k_byte_addrs = slot_bases[:, None] + byte_idx[None, :]
        k_byte_raw = tl.load(
            KV_cache_ptr + k_byte_addrs,
            mask=kv_mask[:, None] & d_mask[None, :],
            other=0,
        ).to(tl.int32)
        k_codes = (k_byte_raw >> nibble_shift[None, :]) & 0xF       # [BLOCK_KV, BLOCK_D] int32
        # Decode FP4 nibble → bf16 via small register table.
        k_dec_bf16 = tl.load(Fp4_decode_ptr + k_codes)              # bf16
        k_dec = k_dec_bf16.to(tl.float32)
        k_dec = tl.where(d_mask[None, :] & kv_mask[:, None], k_dec, 0.0)

        # ── K scales load ──────────────────────────────────────────────
        # 4 fp16 scales per token at slot_base + K_SCALES_OFFSET.
        # Use the uint16 view of the cache for clean 2-byte loads.
        k_scale_u16_addrs = (
            (slot_bases[:, None] + K_SCALES_OFFSET) // 2
            + tl.arange(0, N_GROUPS_C)[None, :]
        )
        k_scales_u16 = tl.load(
            KV_cache_u16_ptr + k_scale_u16_addrs,
            mask=kv_mask[:, None],
            other=0,
        )
        k_scales = k_scales_u16.to(tl.float16, bitcast=True).to(tl.float32)
        # [BLOCK_KV, N_GROUPS]

        # ── QK with scale fuse on accumulator ──────────────────────────
        # Per-group partial dot:
        #   partial[t, g] = sum_{d in group g} q_rot[d] * k_dec[t, d]
        # Then:
        #   qk[t] = sum_g k_scales[t, g] * partial[t, g]
        # q_g is hoisted outside the loop (loop-invariant).
        k_g = tl.reshape(k_dec, [BLOCK_KV, N_GROUPS_C, GROUP_SIZE_C])
        partial_g = tl.sum(q_g[None, :, :] * k_g, axis=2)            # [BLOCK_KV, G]
        scores = tl.sum(partial_g * k_scales, axis=1) * ATTN_SCALE   # [BLOCK_KV]

        if USE_TOKEN_NORM:
            # Load K L2 norms for each token in the tile and scale scores.
            k_norm_addrs = (slot_bases + K_NORM_OFFSET) // 2
            k_norms_u16 = tl.load(KV_cache_u16_ptr + k_norm_addrs,
                                  mask=kv_mask, other=0x3C00)  # 0x3C00 = fp16(1.0)
            k_norms = k_norms_u16.to(tl.float16, bitcast=True).to(tl.float32)
            scores = scores * k_norms

        scores = tl.where(kv_mask, scores, -float("inf"))

        # ── Online softmax update ─────────────────────────────────────
        n_e_max = tl.maximum(tl.max(scores, 0), m_prev)
        re_scale = tl.exp(m_prev - n_e_max)
        p = tl.exp(scores - n_e_max)                                 # [BLOCK_KV]

        # ── V codes load + decode ─────────────────────────────────────
        v_byte_addrs = slot_bases[:, None] + V_CODES_OFFSET + byte_idx[None, :]
        v_byte_raw = tl.load(
            KV_cache_ptr + v_byte_addrs,
            mask=kv_mask[:, None] & d_mask[None, :],
            other=0,
        ).to(tl.int32)
        v_codes = (v_byte_raw >> nibble_shift[None, :]) & 0xF
        v_dec_bf16 = tl.load(Fp4_decode_ptr + v_codes)
        v_dec = v_dec_bf16.to(tl.float32)

        # ── V scales load ─────────────────────────────────────────────
        v_scale_u16_addrs = (
            (slot_bases[:, None] + V_SCALES_OFFSET) // 2
            + tl.arange(0, N_GROUPS_C)[None, :]
        )
        v_scales_u16 = tl.load(
            KV_cache_u16_ptr + v_scale_u16_addrs,
            mask=kv_mask[:, None],
            other=0,
        )
        v_scales = v_scales_u16.to(tl.float16, bitcast=True).to(tl.float32)

        # ── V multiply by per-group scale (per-element broadcast) ─────
        v_dec_g = tl.reshape(v_dec, [BLOCK_KV, N_GROUPS_C, GROUP_SIZE_C])
        v_scaled_g = v_dec_g * v_scales[:, :, None]
        values = tl.reshape(v_scaled_g, [BLOCK_KV, BLOCK_D])
        values = tl.where(d_mask[None, :] & kv_mask[:, None], values, 0.0)

        if USE_TOKEN_NORM:
            # Load V L2 norms and broadcast per-token scalar onto [BLOCK_KV, BLOCK_D].
            v_norm_addrs = (slot_bases + V_NORM_OFFSET) // 2
            v_norms_u16 = tl.load(KV_cache_u16_ptr + v_norm_addrs,
                                  mask=kv_mask, other=0x3C00)
            v_norms = v_norms_u16.to(tl.float16, bitcast=True).to(tl.float32)
            values = values * v_norms[:, None]

        # ── PV accumulate ─────────────────────────────────────────────
        acc = acc * re_scale + tl.sum(p[:, None] * values, axis=0)
        l_prev = l_prev * re_scale + tl.sum(p, 0)
        m_prev = n_e_max

    # Write partial output for this (batch, q_head, kv_split).
    out_base = bid * stride_mid_b + hid * stride_mid_h + sid * stride_mid_s
    safe_l = tl.where(l_prev > 0.0, l_prev, 1.0)
    tl.store(Mid_o_ptr + out_base + d_offs, acc / safe_l, mask=d_mask)
    lse = m_prev + tl.log(safe_l)
    tl.store(Mid_o_ptr + out_base + HEAD_DIM, lse)


# ═══════════════════════════════════════════════════════════════════════════
# Launcher
# ═══════════════════════════════════════════════════════════════════════════


_HADAMARD_CACHE: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}


def _get_pit(
    dim: int, device: torch.device, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    key = (dim, device, dtype)
    cached = _HADAMARD_CACHE.get(key)
    if cached is not None:
        return cached
    if dim <= 0 or (dim & (dim - 1)) != 0:
        raise ValueError(f"FP4-g32 decode requires power-of-two dim, got {dim}")
    H = torch.tensor([[1.0]], dtype=torch.float64)
    while H.shape[0] < dim:
        H = torch.cat(
            [torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0
        )
    H = (H / (dim**0.5)).to(device=device, dtype=dtype).contiguous()
    _HADAMARD_CACHE[key] = H
    return H


def fp4_g32_decode_attention(
    query: torch.Tensor,         # [B, Hq, D] bf16 or fp16 — raw (will be rotated here)
    kv_cache: torch.Tensor,      # [num_blocks, block_size, Hk, slot_size_aligned] uint8
    block_table: torch.Tensor,   # [B, max_num_blocks] int32
    seq_lens: torch.Tensor,      # [B] int32
    *,
    scale: float,
    max_num_kv_splits: int = 32,
    PiT: torch.Tensor | None = None,        # [D, D] fp32; auto-built if None
    sinks: torch.Tensor | None = None,       # [Hq] fp32
    mid_o_buf: torch.Tensor | None = None,
    output_buf: torch.Tensor | None = None,
    lse_buf: torch.Tensor | None = None,
) -> torch.Tensor:
    """Launch FP4-g32 decode attention (stage 1 + reused stage 2).

    Returns: output tensor [B, Hq, D] in query.dtype.
    """
    B, Hq, D = query.shape
    Hk = kv_cache.shape[2]
    block_size = kv_cache.shape[1]
    padded_slot = kv_cache.shape[3]
    kv_group_size = Hq // Hk
    device = query.device

    gs = get_group_size()
    tn = get_token_norm()

    if padded_slot < slot_size(D, gs, tn):
        raise ValueError(
            f"fp4_g32_decode: kv_cache slot {padded_slot} < expected "
            f"{slot_size(D, gs, tn)} for head_dim={D} group_size={gs} token_norm={tn}"
        )
    if Hq % Hk != 0:
        raise ValueError(f"Hq={Hq} must be a multiple of Hk={Hk}")

    if PiT is None:
        PiT = _get_pit(D, device, torch.float32)
    elif PiT.dtype != torch.float32:
        PiT = PiT.to(torch.float32)
    if not PiT.is_contiguous():
        PiT = PiT.contiguous()

    # Pre-rotate Q (external rocBLAS GEMM — matches TQ v1 default decode path).
    q_rot = (query.float() @ PiT).contiguous()    # [B, Hq, D] fp32

    BLOCK_D = triton.next_power_of_2(D)
    N_GROUPS_C = n_groups(D, gs)
    # BLOCK_KV=4 matches TQ v1. Larger values (e.g. 16) cause severe VGPR
    # spilling on AMD MI3xx at long context (128K+) due to the [BLOCK_KV,
    # HEAD_DIM] intermediate tensors exceeding available registers, making
    # each decode step 10-100x slower and triggering EngineCore RPC timeouts.
    BLOCK_KV = 4

    # mid_o partial buffer: [B, Hq, NUM_KV_SPLITS, D+1] fp32.
    if (
        mid_o_buf is not None
        and mid_o_buf.shape[0] >= B
        and mid_o_buf.shape[2] >= max_num_kv_splits
    ):
        mid_o = mid_o_buf[:B, :Hq, :max_num_kv_splits, :]
    else:
        mid_o = torch.empty(
            B, Hq, max_num_kv_splits, D + 1, dtype=torch.float32, device=device
        )

    fp4_dec = _get_fp4_decode_table(device)

    # In-bytes strides into the raw uint8 view.
    stride_block = kv_cache.stride(0)
    stride_pos = kv_cache.stride(1)
    stride_head = kv_cache.stride(2)

    use_sinks = sinks is not None
    # If sinks is None, we need a non-NULL pointer for the kernel anyway —
    # the kernel branch guards on USE_SINKS so it never derefs in that case,
    # but Triton's tl.load on a NULL ptr is undefined. Pass a dummy tensor.
    sink_arg = sinks if use_sinks else torch.empty(Hq, dtype=torch.float32, device=device)

    kv_flat = _kv_cache_flat(kv_cache)
    grid = (B, Hq, max_num_kv_splits)
    _fp4_g32_decode_stage1[grid](
        q_rot,
        kv_flat,
        kv_flat.view(torch.uint16),
        block_table,
        seq_lens,
        fp4_dec,
        mid_o,
        sink_arg,
        q_rot.stride(0),
        q_rot.stride(1),
        stride_block,
        stride_pos,
        stride_head,
        block_table.stride(0),
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        NUM_KV_HEADS=Hk,
        HEAD_DIM=D,
        BLOCK_SIZE=block_size,
        NUM_KV_SPLITS=max_num_kv_splits,
        KV_GROUP_SIZE=kv_group_size,
        GROUP_SIZE_C=gs,
        N_GROUPS_C=N_GROUPS_C,
        K_SCALES_OFFSET=k_scales_offset(D, gs),
        V_CODES_OFFSET=v_codes_offset(D, gs),
        V_SCALES_OFFSET=v_scales_offset(D, gs),
        ATTN_SCALE=scale,
        BLOCK_D=BLOCK_D,
        BLOCK_KV=BLOCK_KV,
        USE_SINKS=1 if use_sinks else 0,
        USE_TOKEN_NORM=1 if tn else 0,
        K_NORM_OFFSET=k_norm_offset(D, gs),
        V_NORM_OFFSET=v_norm_offset(D, gs),
        num_warps=1,   # matches TQ v1 — lower register pressure at long context
        num_stages=1,  # no SW pipelining — avoids VGPR spill at BLOCK_KV*HEAD_DIM scale
    )

    # Stage 2: reduce across KV splits — reuses TQ's reduce kernel.
    out_dtype = query.dtype
    if (
        output_buf is not None
        and output_buf.shape[0] >= B
        and output_buf.dtype == out_dtype
    ):
        output = output_buf[:B, :Hq, :D]
    else:
        output = torch.empty(B, Hq, D, dtype=out_dtype, device=device)

    if lse_buf is not None and lse_buf.shape[0] >= B:
        lse = lse_buf[:B, :Hq]
    else:
        lse = torch.empty(B, Hq, dtype=torch.float32, device=device)

    grid2 = (B, Hq)
    _fwd_kernel_stage2[grid2](
        mid_o,
        output,
        lse,
        seq_lens,
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        output.stride(0),
        output.stride(1),
        lse.stride(0),
        NUM_KV_SPLITS=max_num_kv_splits,
        BLOCK_DV=BLOCK_D,
        Lv=D,
        OUTPUT_FP16=1 if out_dtype == torch.float16 else 0,
        num_warps=4,
        num_stages=2,
    )

    return output
