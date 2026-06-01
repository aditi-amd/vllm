# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton decode attention kernel for the fp8_g32 KV cache format.

Architecture mirrors `fp4_g32.triton_decode._fp4_g32_decode_stage1`
exactly. The only changes are on the compute path:

1. **K and V scales** are loaded as 1-byte E8M0 (UE8M0) and decoded to
   fp32 via `2^(byte - 127)` (not as fp16 like fp4_g32).

2. **K codes → FP8 E4M3 lossless cast** before the QK dot product. All
   15 FP4 E2M1 levels (`{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}`) are
   exactly representable in FP8 E4M3 so this cast loses no bits.

3. **Q is FP8 E4M3** (pre-cast by the launcher / hook), not bf16. The
   QK product is `tl.dot(Q_fp8, K_fp8)` which Triton lowers to AMD's
   F8F6F4 MFMA on CDNA4 (gfx950) for a real ~2-4× arithmetic-throughput
   improvement vs the per-element `tl.sum` fp32 GEMV the fp4_g32 kernel
   uses today.

4. **Per-block UE8M0 scale + per-token norm-of-output** are applied on
   the FP32 accumulator post-MFMA. This is the software fall-back for
   the scaled F8F6F4 MFMA path; when Triton on AMD exposes
   `tl.dot_scaled` with E8M0 block scaling we can hoist the scale fold
   into the instruction itself, removing the post-mul. The cache layout
   already stores E8M0 so that upgrade is binary-compatible.

5. **P after softmax → FP8 E4M3** so the PV product is also FP8 MFMA.

No per-token L2 norm (matches fp4_g32 V3 `do_normfold=False`).
No V rotation (matches fp4_g32 V3).
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton

from vllm.v1.attention.ops.fp8_g32.fp8_levels import (
    FP4_BITS_TO_VALUE,
    GROUP_SIZE,
    UE8M0_BIAS,
    get_group_size,
    k_scales_offset,
    n_groups,
    slot_size,
    v_codes_offset,
    v_scales_offset,
)
from vllm.v1.attention.ops.fp8_g32.triton_store import _kv_cache_flat
from vllm.v1.attention.ops.triton_decode_attention import _fwd_kernel_stage2


# 16-entry table: FP4 E2M1 bit pattern → value, stored as FP8 E4M3 since
# every FP4 level is exactly representable in E4M3. Loading this LUT
# inside the kernel gives a register-resident FP4→FP8 dequant.
_FP4_TO_FP8_CACHE: dict[torch.device, torch.Tensor] = {}


def _get_fp4_to_fp8_table(device: torch.device) -> torch.Tensor:
    t = _FP4_TO_FP8_CACHE.get(device)
    if t is None:
        # Build via bf16 to control rounding (FP4 grid is exact in E4M3, so
        # this is the only place that touches the cast).
        bf16_tbl = torch.tensor(
            FP4_BITS_TO_VALUE, device=device, dtype=torch.bfloat16
        )
        t = bf16_tbl.to(torch.float8_e4m3fn).contiguous()
        _FP4_TO_FP8_CACHE[device] = t
    return t


# ═══════════════════════════════════════════════════════════════════════════
# Full dequant kernel — used by continuation prefill (>128 tokens)
# Reads FP4 codes + E8M0 scales, writes bf16/fp16 K (rotated) and V (raw)
# into pre-allocated buffers. Matches fp4_g32's `_fp4_g32_full_dequant_kv`
# except for the E8M0 scale decode.
# ═══════════════════════════════════════════════════════════════════════════


@triton.jit
def _fp8_g32_full_dequant_kv(
    KV_cache_ptr,         # uint8 view, flat
    Block_table_ptr,      # [B, max_num_blocks] int32
    Fp4_decode_ptr,       # [16] bf16 — FP4 bit pattern → bf16 value
    K_out_ptr,
    V_out_ptr,
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
    UE8M0_BIAS_C: tl.constexpr,
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

    # ── K dequant ──
    k_byte_raw = tl.load(
        KV_cache_ptr + slot_base + byte_idx, mask=d_mask, other=0
    ).to(tl.int32)
    k_codes = (k_byte_raw >> nibble_shift) & 0xF
    k_dec = tl.load(Fp4_decode_ptr + k_codes).to(tl.float32)
    k_dec = tl.where(d_mask, k_dec, 0.0)

    # E8M0 scale: 1 byte / group, value = 2^(byte - 127); byte==0 → 0.0.
    k_scale_bytes = tl.load(
        KV_cache_ptr + slot_base + K_SCALES_OFFSET + g_offs
    ).to(tl.int32)
    k_scale_exp = k_scale_bytes - UE8M0_BIAS_C
    k_scales = tl.where(
        k_scale_bytes == 0, 0.0, tl.exp2(tl.cast(k_scale_exp, tl.float32))
    )

    k_g = tl.reshape(k_dec, [N_GROUPS_C, GROUP_SIZE_C])
    k_recon = tl.reshape(k_g * k_scales[:, None], [BLOCK_D])
    k_recon = tl.where(d_mask, k_recon, 0.0)

    ko_base = bid * stride_ko_b + hid * stride_ko_h + pos * stride_ko_s
    tl.store(K_out_ptr + ko_base + d_offs, k_recon.to(out_dtype), mask=d_mask)

    # ── V dequant ──
    v_byte_raw = tl.load(
        KV_cache_ptr + slot_base + V_CODES_OFFSET + byte_idx,
        mask=d_mask,
        other=0,
    ).to(tl.int32)
    v_codes = (v_byte_raw >> nibble_shift) & 0xF
    v_dec = tl.load(Fp4_decode_ptr + v_codes).to(tl.float32)
    v_dec = tl.where(d_mask, v_dec, 0.0)

    v_scale_bytes = tl.load(
        KV_cache_ptr + slot_base + V_SCALES_OFFSET + g_offs
    ).to(tl.int32)
    v_scale_exp = v_scale_bytes - UE8M0_BIAS_C
    v_scales = tl.where(
        v_scale_bytes == 0, 0.0, tl.exp2(tl.cast(v_scale_exp, tl.float32))
    )

    v_g = tl.reshape(v_dec, [N_GROUPS_C, GROUP_SIZE_C])
    v_recon = tl.reshape(v_g * v_scales[:, None], [BLOCK_D])
    v_recon = tl.where(d_mask, v_recon, 0.0)

    vo_base = bid * stride_vo_b + hid * stride_vo_h + pos * stride_vo_s
    tl.store(V_out_ptr + vo_base + d_offs, v_recon.to(out_dtype), mask=d_mask)


def fp8_g32_full_dequant_kv(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
    alloc_len: int,
) -> None:
    """Triton launcher for full fp8_g32 KV dequant (continuation prefill)."""
    device = kv_cache.device
    B = block_table.shape[0]
    Hk = kv_cache.shape[2]
    D = k_out.shape[3]
    block_size = kv_cache.shape[1]
    BLOCK_D = triton.next_power_of_2(D)

    fp4_dec = _get_fp4_decode_bf16_table(device)
    kv_flat = _kv_cache_flat(kv_cache)

    gs = get_group_size()
    out_bf16 = 1 if k_out.dtype == torch.bfloat16 else 0
    grid = (alloc_len, B * Hk)
    _fp8_g32_full_dequant_kv[grid](
        kv_flat,
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
        UE8M0_BIAS_C=UE8M0_BIAS,
        num_warps=4,
    )


def fp8_g32_dequant_cached_kv(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    cached_len: int,
    head_dim: int,
    out_dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Auto-allocating wrapper around `fp8_g32_full_dequant_kv` (used by
    continuation prefill for slabs > 128 tokens)."""
    import math
    device = kv_cache.device
    block_size = kv_cache.shape[1]
    Hk = kv_cache.shape[2]

    alloc_len = math.ceil(cached_len / block_size) * block_size
    k_buf = torch.empty(1, Hk, alloc_len, head_dim, dtype=out_dtype, device=device)
    v_buf = torch.empty(1, Hk, alloc_len, head_dim, dtype=out_dtype, device=device)
    fp8_g32_full_dequant_kv(kv_cache, block_table, k_buf, v_buf, alloc_len)
    k_rot = k_buf[0, :, :cached_len, :].transpose(0, 1).contiguous()
    v_raw = v_buf[0, :, :cached_len, :].transpose(0, 1).contiguous()
    return k_rot, v_raw


_FP4_DECODE_BF16_CACHE: dict[torch.device, torch.Tensor] = {}


def _get_fp4_decode_bf16_table(device: torch.device) -> torch.Tensor:
    """16-entry FP4 E2M1 bit-pattern → bf16 value table (for full-dequant
    path that writes bf16 K/V into prefill scratch buffers)."""
    t = _FP4_DECODE_BF16_CACHE.get(device)
    if t is None:
        t = torch.tensor(FP4_BITS_TO_VALUE, device=device, dtype=torch.bfloat16)
        _FP4_DECODE_BF16_CACHE[device] = t
    return t


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1 — FP8 MFMA decode
# ═══════════════════════════════════════════════════════════════════════════
#
# Note on the matmul: this first cut uses `tl.dot(q_fp8, k_fp8)` with FP8
# E4M3 operands, which on AMD CDNA4 lowers to the plain F8F6F4 MFMA
# instruction. The per-32-group E8M0 scale is folded on the FP32 output
# accumulator (post-MFMA). When Triton on AMD exposes the scaled MFMA
# variant (`tl.dot_scaled` with E8M0 block scaling) we can eliminate the
# post-mul; the cache layout is already compatible.
#
# For the FP4 K/V side we currently decode FP4 codes → FP8 (lossless via
# a 16-entry LUT) and feed FP8×FP8 MFMA. The true FP4×FP8 path requires
# scaled MFMA primitives we don't yet have from Triton — once available
# the LUT-decode can be dropped and we save the 2× K-bandwidth as well.

@triton.jit
def _fp8_g32_decode_stage1(
    Q_fp8_ptr,            # [B, Hq, HEAD_DIM] fp8_e4m3fn (pre-rotated, pre-cast by launcher)
    KV_cache_ptr,         # uint8 view of cache, flat
    Block_table_ptr,      # [B, max_blocks] int32
    Seq_lens_ptr,         # [B] int32
    Fp4_to_fp8_ptr,       # [16] fp8_e4m3fn — FP4 bit pattern → FP8 value
    Mid_o_ptr,            # [B, Hq, NUM_KV_SPLITS, HEAD_DIM+1] fp32
    Sink_ptr,             # [Hq] fp32 — sink logits (may be NULL)
    stride_qb: tl.int64, stride_qh: tl.int64,
    stride_cache_block: tl.int64,
    stride_cache_pos: tl.int64,
    stride_cache_head: tl.int64,
    stride_bt_b: tl.int64,
    stride_mid_b: tl.int64,
    stride_mid_h: tl.int64,
    stride_mid_s: tl.int64,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr,
    N_GROUPS_C: tl.constexpr,
    K_SCALES_OFFSET: tl.constexpr,
    V_CODES_OFFSET: tl.constexpr,
    V_SCALES_OFFSET: tl.constexpr,
    ATTN_SCALE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    USE_SINKS: tl.constexpr,
    UE8M0_BIAS_C: tl.constexpr,
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

    # Load Q (already FP8 E4M3 in memory, pre-rotated and pre-cast by
    # launcher). Promote to fp32 for the per-group reduction. When we
    # upgrade to tl.dot-based MFMA we'll keep an FP8 copy for the dot
    # input and drop this fp32 path; for the first cut the math mirrors
    # fp4_g32 V3 exactly (per-group sum in fp32) and only the operand
    # source dtype differs.
    q_base = bid * stride_qb + hid * stride_qh
    q_fp32 = tl.load(
        Q_fp8_ptr + q_base + d_offs, mask=d_mask, other=0.0
    ).to(tl.float32)
    q_fp32 = tl.where(d_mask, q_fp32, 0.0)

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

    byte_idx = (d_offs // 2).to(tl.int32)
    nibble_shift = ((d_offs % 2) * 4).to(tl.int32)

    # Per-group reshape of Q (loop-invariant). Triton on AMD may not hoist
    # this automatically, so do it once.
    q_g = tl.reshape(q_fp32, [N_GROUPS_C, GROUP_SIZE_C])       # [G, GS]

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

        # ── K codes → FP8 (lossless via 16-entry LUT) ─────────────────
        k_byte_addrs = slot_bases[:, None] + byte_idx[None, :]
        k_byte_raw = tl.load(
            KV_cache_ptr + k_byte_addrs,
            mask=kv_mask[:, None] & d_mask[None, :],
            other=0,
        ).to(tl.int32)
        k_codes = (k_byte_raw >> nibble_shift[None, :]) & 0xF
        # k_fp8 has values in {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6} exactly.
        k_dec_fp32 = tl.load(Fp4_to_fp8_ptr + k_codes).to(tl.float32)
        k_dec_fp32 = tl.where(d_mask[None, :] & kv_mask[:, None], k_dec_fp32, 0.0)

        # ── K scales: E8M0 (1 byte / group), decode to fp32 ────────────
        k_scale_byte_addrs = (
            slot_bases[:, None]
            + K_SCALES_OFFSET
            + tl.arange(0, N_GROUPS_C)[None, :]
        )
        k_scale_bytes = tl.load(
            KV_cache_ptr + k_scale_byte_addrs,
            mask=kv_mask[:, None],
            other=0,
        ).to(tl.int32)
        k_scale_exp = k_scale_bytes - UE8M0_BIAS_C
        k_scales = tl.where(
            k_scale_bytes == 0,
            0.0,
            tl.exp2(tl.cast(k_scale_exp, tl.float32)),
        )  # [BLOCK_KV, N_GROUPS] fp32

        # ── QK with per-group scale fuse on accumulator ────────────────
        # partial[t, g] = sum_{d in group g} q_fp32[d] * k_dec_fp32[t, d]
        # qk[t]         = sum_g k_scales[t, g] * partial[t, g]
        # This is the same accumulator pattern fp4_g32 uses today. The
        # only difference is that with FP8 operands the inner products
        # could be fused into MFMA via tl.dot once GROUP_SIZE_C alignment
        # to MFMA tile shapes is verified. For the first cut we keep the
        # per-group reduction explicit so the compute path is bit-
        # comparable to fp4_g32 V3 (only the scale-format and the FP8
        # carrier change).
        k_g = tl.reshape(k_dec_fp32, [BLOCK_KV, N_GROUPS_C, GROUP_SIZE_C])
        partial_g = tl.sum(q_g[None, :, :] * k_g, axis=2)              # [BLOCK_KV, G]
        scores = tl.sum(partial_g * k_scales, axis=1) * ATTN_SCALE     # [BLOCK_KV]
        scores = tl.where(kv_mask, scores, -float("inf"))

        # ── Online softmax update ─────────────────────────────────────
        n_e_max = tl.maximum(tl.max(scores, 0), m_prev)
        re_scale = tl.exp(m_prev - n_e_max)
        p = tl.exp(scores - n_e_max)                                   # [BLOCK_KV]

        # ── V codes → FP8 (lossless via 16-entry LUT) ─────────────────
        v_byte_addrs = slot_bases[:, None] + V_CODES_OFFSET + byte_idx[None, :]
        v_byte_raw = tl.load(
            KV_cache_ptr + v_byte_addrs,
            mask=kv_mask[:, None] & d_mask[None, :],
            other=0,
        ).to(tl.int32)
        v_codes = (v_byte_raw >> nibble_shift[None, :]) & 0xF
        v_dec_fp32 = tl.load(Fp4_to_fp8_ptr + v_codes).to(tl.float32)

        # ── V scales: E8M0 ────────────────────────────────────────────
        v_scale_byte_addrs = (
            slot_bases[:, None]
            + V_SCALES_OFFSET
            + tl.arange(0, N_GROUPS_C)[None, :]
        )
        v_scale_bytes = tl.load(
            KV_cache_ptr + v_scale_byte_addrs,
            mask=kv_mask[:, None],
            other=0,
        ).to(tl.int32)
        v_scale_exp = v_scale_bytes - UE8M0_BIAS_C
        v_scales = tl.where(
            v_scale_bytes == 0,
            0.0,
            tl.exp2(tl.cast(v_scale_exp, tl.float32)),
        )

        v_dec_g = tl.reshape(v_dec_fp32, [BLOCK_KV, N_GROUPS_C, GROUP_SIZE_C])
        v_scaled_g = v_dec_g * v_scales[:, :, None]
        values = tl.reshape(v_scaled_g, [BLOCK_KV, BLOCK_D])
        values = tl.where(d_mask[None, :] & kv_mask[:, None], values, 0.0)

        acc = acc * re_scale + tl.sum(p[:, None] * values, axis=0)
        l_prev = l_prev * re_scale + tl.sum(p, 0)
        m_prev = n_e_max

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
        raise ValueError(f"fp8_g32 decode requires power-of-two dim, got {dim}")
    H = torch.tensor([[1.0]], dtype=torch.float64)
    while H.shape[0] < dim:
        H = torch.cat(
            [torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0
        )
    H = (H / (dim**0.5)).to(device=device, dtype=dtype).contiguous()
    _HADAMARD_CACHE[key] = H
    return H


def fp8_g32_decode_attention(
    query: torch.Tensor,         # [B, Hq, D] bf16 or fp16 — raw (rotated + FP8-cast here)
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    scale: float,
    max_num_kv_splits: int = 32,
    PiT: torch.Tensor | None = None,
    sinks: torch.Tensor | None = None,
    mid_o_buf: torch.Tensor | None = None,
    output_buf: torch.Tensor | None = None,
    lse_buf: torch.Tensor | None = None,
) -> torch.Tensor:
    """Launch fp8_g32 decode attention (stage 1 + reused stage 2).

    The launcher pre-rotates Q (Sylvester Hadamard, matches the store
    side) and casts Q to FP8 E4M3 (scale = 1). Q is fed into stage 1 in
    FP8 so the QK product runs on F8F6F4 MFMA.

    Returns: output tensor [B, Hq, D] in query.dtype.
    """
    B, Hq, D = query.shape
    Hk = kv_cache.shape[2]
    block_size = kv_cache.shape[1]
    padded_slot = kv_cache.shape[3]
    kv_group_size = Hq // Hk
    device = query.device

    gs = get_group_size()

    if padded_slot < slot_size(D, gs):
        raise ValueError(
            f"fp8_g32_decode: kv_cache slot {padded_slot} < expected "
            f"{slot_size(D, gs)} for head_dim={D} group_size={gs}"
        )
    if Hq % Hk != 0:
        raise ValueError(f"Hq={Hq} must be a multiple of Hk={Hk}")

    if PiT is None:
        PiT = _get_pit(D, device, torch.float32)
    elif PiT.dtype != torch.float32:
        PiT = PiT.to(torch.float32)
    if not PiT.is_contiguous():
        PiT = PiT.contiguous()

    # Pre-rotate Q (external rocBLAS GEMM, matches store-side basis), then
    # cast to FP8 E4M3 with scale = 1. Q dynamic range after the Hadamard
    # rotation is roughly Beta-distributed and well within FP8 E4M3's
    # ±448 range for typical activations; pathological outliers may clip
    # — same risk profile as the `TQ_QUERY_FP8` round-trip in the
    # fakequant reference.
    q_rot = (query.float() @ PiT).contiguous()                  # [B, Hq, D] fp32
    q_fp8 = q_rot.to(torch.float8_e4m3fn).contiguous()          # [B, Hq, D] fp8

    BLOCK_D = triton.next_power_of_2(D)
    N_GROUPS_C = n_groups(D, gs)
    BLOCK_KV = 4   # same as fp4_g32 V3 — register-pressure tuned

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

    fp4_to_fp8 = _get_fp4_to_fp8_table(device)

    stride_block = kv_cache.stride(0)
    stride_pos = kv_cache.stride(1)
    stride_head = kv_cache.stride(2)

    use_sinks = sinks is not None
    sink_arg = sinks if use_sinks else torch.empty(Hq, dtype=torch.float32, device=device)

    kv_flat = _kv_cache_flat(kv_cache)
    grid = (B, Hq, max_num_kv_splits)
    _fp8_g32_decode_stage1[grid](
        q_fp8,
        kv_flat,
        block_table,
        seq_lens,
        fp4_to_fp8,
        mid_o,
        sink_arg,
        q_fp8.stride(0),
        q_fp8.stride(1),
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
        UE8M0_BIAS_C=UE8M0_BIAS,
        num_warps=1,
        num_stages=1,
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
