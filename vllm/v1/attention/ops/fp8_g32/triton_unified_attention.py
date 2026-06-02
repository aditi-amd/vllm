# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified (prefill + decode) Triton attention kernel for FP8-g32.

This file ships two QK compute paths; both produce numerically equivalent
output (within bf16 ULPs) and BOTH hit native CDNA4 hardware MFMA. They
differ only in WHICH MFMA opcode the Triton backend selects:

  Path "dot_scaled" (DEFAULT, VLLM_FP8_G32_V3_DOT_KIND=dot_scaled)
    - QK uses ``tl.dot_scaled(Q_fp8, K_T_packed, "e4m3", "e2m1",
      rhs_scale=K_scales_e8m0)`` directly — the K loader returns raw
      FP4 codes + raw E8M0 scale bytes, no software dequant.
    - In Triton 3.6 on gfx950 the full kernel context lowers this to
      the **native scaled F8F6F4 MFMA**:
      ``v_mfma_scale_f32_16x16x128_f8f6f4`` (K=128 per MFMA instruction,
      E8M0 scales consumed by the MFMA itself).

  Path "fp8mfma" (opt-in, VLLM_FP8_G32_V3_DOT_KIND=fp8mfma)
    - K loader dequants FP4 codes + folds E8M0 scales in software into
      FP8 E4M3 (lossless within typical decode range — FP4 values are
      exactly representable in FP8 E4M3, and E8M0 scales are powers of
      2 that only shift the exponent).
    - QK uses plain ``tl.dot(Q_fp8, K_T_fp8)`` which lowers to the
      non-scaled native MFMA ``v_mfma_f32_16x16x128_f8f6f4``.
    - Same K=128 throughput as the scaled variant but pays for software
      dequant. Kept for A/B and as a fallback.

Common to both paths:
  Q  : FP8 E4M3, pre-rotated by launcher with Hadamard PiT then cast.
  K  : stored as FP4 E2M1 (uint8, D-dim packed) + per-group E8M0 scales.
  V  : FP4 E2M1, LUT-decoded into bf16 + E8M0 scales decoded to bf16 and
       applied pre-MFMA (same path as fp4_g32 V3 V-tile — our V scales
       are grouped along the *output* axis, which microscaling can't
       consume for the rhs side anyway).
  PV : bf16 ``tl.dot(P, V)`` → ``v_mfma_f32_16x16x32_bf16``.

Other architectural choices (mirroring fp4_g32 V3):
  - 2D unified prefill kernel + 3D split-KV decode kernel.
  - GQA stacking into BLOCK_M (16 for decode, 128 for prefill).
  - Online softmax inside stage 1; vectorized ``reduce_segments`` stage 2.
  - Sinks support, sliding-window tile pruning, AMD MFMA hints.

Perf optimization env knobs (all opt-in, all preserve correctness):
  - VLLM_FP8_G32_NUM_KV_SPLITS=N : 3D decode split count. Default 16
    (matches TQ44 V3 default; gives ~36 tile iters per CTA at 8K context).
    Lower (8) = fewer CTAs / more per-CTA work; higher (64) = more CTAs /
    smaller per-CTA work. Best value is workload-dependent.
  - VLLM_FP8_G32_NUM_STAGES_3D=N : software pipeline depth for stage 1.
    Default 3 on HIP (deeper pipelining hides V-load latency in the inner
    loop; TQ44 regressed at stages=3 because its centroid LUT has higher
    register pressure, but dot_scaled K has lower register usage and
    benefits from the extra slot). =1 matches TQ44 but regressed -3% at
    C=64 on Qwen-72B; =2 was the previous default; =3 is +2.2% OutTPS,
    -2.2% TPOT vs the previous default at Qwen-72B 8K C=64.
  - VLLM_FP8_G32_FUSE_Q_ROT=1 : fuses the launcher-side Q@PiT rotation
    into the kernel prologue (bf16 MFMA). Bit-exact at D=64; ~3.9e-3
    max-abs at D=128 (within bf16 quant noise). Wins on low-batch /
    single-stream (saves Python overhead) but regressed at high
    concurrency on Qwen-72B 8K C=64 (PiT broadcast load × 8K+ programs
    per launch exceeds the launcher's one-shot rocBLAS cost).
  - VLLM_FP8_G32_V_BF16=1 : keep V dequant in bf16 (skip fp32 mul +
    fp32->bf16 cast). UE8M0 scales are exact pow2 → bit-exact at
    smoke-test scale. Wash on Qwen-72B 8K C=64 serving (within noise).
  - VLLM_FP8_G32_V_DOT_SCALED=1 : use fp8×fp8 dot_scaled for PV (instead
    of bf16 dot). Lowers to the native scaled FP8 MFMA on CDNA4
    (``v_mfma_scale_f32_16x16x128_f8f6f4`` at K=128 per issue → 2× the
    bf16 MFMA throughput). V loader bakes D-axis scales into fp8 codes;
    P is row-wise quantized to fp8 with one E8M0 scale per row.
    Constraints discovered during validation (gfx950, MI355X, Triton 3.6):
      * Forces TILE_SIZE=128 (NOT 32). The native scaled FP8 MFMA needs
        K=128 to fill its native shape; at K=32 dot_scaled is actually
        slower than bf16 dot. Isolated GEMM microbench:
            K=32  : bf16 3.1 TFLOPS, fp8 3.0 TFLOPS → 0.96×  (LOSS)
            K=64  : bf16 4.5 TFLOPS, fp8 5.7 TFLOPS → 1.26×
            K=128 : bf16 5.8 TFLOPS, fp8 11.7 TFLOPS → 2.03×  ✓
            K=256 : bf16 6.6 TFLOPS, fp8 23.0 TFLOPS → 3.51×
      * Forces NUM_STAGES_3D=1 (instead of default 3) to compensate for
        the 4× larger per-iter V working set at TILE_SIZE=128. Higher
        stages cause VGPR spills.
      * Net win is shape-dependent: small-to-medium batch (B≤16) sees
        modest wins (up to ~1.15× kernel time at Qwen-72B GQA shape);
        very large batch (B=64 at 8K) regresses ~20% because per-CTA
        inner-loop iterations drop to ~4, leaving launch overhead
        unamortized. Override VLLM_FP8_G32_NUM_KV_SPLITS=2 at very high
        concurrency to recover (sacrifices small-batch occupancy).
    Numerical cost: P loses ~3 bits of precision (per-row fp8 cast).
    Smoke tests pass at 5 representative shapes; FP4 V quant noise still
    dominates total error vs fp32 reference.
    Verdict: opt-in only — defaults preserve the current bf16 V path.
  - VLLM_FP8_G32_AMD_HINTS=1 : pass waves_per_eu=2 /
    matrix_instr_nonkdim=16 / kpack=2 to the AMD Triton backend. Opt-in
    for experimental tuning per workload.

How to run (canonical defaults, MI355X / Qwen-72B)
--------------------------------------------------

The defaults baked into THIS branch are the current production-ready set
(no env var overrides needed). The two BAU benches we regression-test
against are:

(a) Accuracy A/B (LCB-128K, TQ44 V3 vs fp8_g32 V3, 2-way parallel):
    bash benchmarks/lcb_qwen72b_128k_tq44_vs_fp8.sh
    # Per-leg env (set inside the script):
    #   TQ44 V3:  VLLM_TQ_DECODE_V3=1 VLLM_TQ_DECODE_V2=0
    #             VLLM_TQ_SOA_FUSION=0 VLLM_TQ_SOA_FUSION_STORE=0
    #   fp8_g32:  VLLM_FP8_G32_V3=1
    # Shared:   VLLM_ROCM_USE_AITER=1 HSA_NO_SCRATCH_RECLAIM=1
    #           --kv-cache-dtype {turboquant_4bit_nc | fp8_kv_g32}
    #           --tensor-parallel-size 2 --block-size 32
    #           --attention-backend ROCM_AITER_UNIFIED_ATTN
    #           --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'

(b) Throughput A/B (ISL=8192, OSL=1024, C=64, N=128 prompts, TP=2):
    python tmp_perf_qwen72b_8k1k_c64.py
    # Same backend / cudagraphs as (a). Default fp8_g32 leg env is just
    # VLLM_FP8_G32_V3=1 — the kernel's *internal* defaults are:
    #   VLLM_FP8_G32_V3_DOT_KIND=dot_scaled  (native scaled F8F6F4 MFMA on QK)
    #   VLLM_FP8_G32_NUM_KV_SPLITS=16        (= TQ44 V3 default)
    #   VLLM_FP8_G32_NUM_STAGES_3D=3 on HIP  (2 on non-HIP)
    #   VLLM_FP8_G32_FUSE_Q_ROT=0            (opt-in; regressed at C=64)
    #   VLLM_FP8_G32_V_BF16=0                (opt-in; wash at C=64)
    #   VLLM_FP8_G32_V_DOT_SCALED=0          (opt-in; forces TILE=128, regresses at C=64)
    #   VLLM_FP8_G32_AMD_HINTS=0             (opt-in tuning knob)

Quick fp8_g32-only smoke serve (for ad-hoc poking, NOT a regression):
    HIP_VISIBLE_DEVICES=0,1 VLLM_ROCM_USE_AITER=1 \\
    HSA_NO_SCRATCH_RECLAIM=1 VLLM_FP8_G32_V3=1 \\
        python -m vllm.entrypoints.openai.api_server \\
            --model /shareddata/Qwen/Qwen2.5-72B-Instruct \\
            --port 9421 --tensor-parallel-size 2 \\
            --gpu-memory-utilization 0.85 --max-model-len 9472 \\
            --kv-cache-dtype fp8_kv_g32 --block-size 32 \\
            --trust-remote-code --no-enable-prefix-caching \\
            --attention-backend ROCM_AITER_UNIFIED_ATTN \\
            --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
"""

from __future__ import annotations

import math
import os
from typing import Any

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

# Reuse the V3 helpers — sequence-idx search and the vectorized stage-2 reducer.
from vllm.v1.attention.ops.triton_turboquant_unified_attention import (
    _find_seq_idx,
)
from vllm.v1.attention.ops.triton_unified_attention import (
    find_seq_idx,
    reduce_segments,
)

from vllm.v1.attention.ops.fp8_g32.fp8_levels import (
    FP4_BITS_TO_VALUE,
    UE8M0_BIAS,
    get_constant_c,
    get_group_size,
    is_arch_b,
    k_scales_offset,
    n_groups,
    slot_size,
    v_codes_offset,
    v_scales_offset,
)
from vllm.v1.attention.ops.fp8_g32.triton_store import _kv_cache_flat

_is_hip = current_platform.is_rocm()


# ---------------------------------------------------------------------------
# FP4 decode LUT for V tile (16-entry, bf16).
# ---------------------------------------------------------------------------

_FP4_DECODE_BF16_CACHE: dict[torch.device, torch.Tensor] = {}


def _get_fp4_decode_table_bf16(device: torch.device) -> torch.Tensor:
    t = _FP4_DECODE_BF16_CACHE.get(device)
    if t is None:
        t = torch.tensor(FP4_BITS_TO_VALUE, device=device, dtype=torch.bfloat16)
        _FP4_DECODE_BF16_CACHE[device] = t
    return t


# ---------------------------------------------------------------------------
# Fused Q-rotation prologue (Opt FUSE_Q_ROT).
#
# When VLLM_FP8_G32_FUSE_Q_ROT=1 the launcher passes the raw bf16 query
# tensor directly and skips the Python-side
#
#     q_rot = (query.float() @ PiT).contiguous()
#     q_fp8 = q_rot.to(torch.float8_e4m3fn).contiguous()
#
# chain, which costs four allocations + four memory-roundtrips per decode
# step (bf16->fp32, fp32 GEMM via rocBLAS, fp32->fp8, fp32->fp8 contiguous).
# The kernel then loads raw bf16 Q, applies a bf16 Hadamard rotation, and
# casts the result to FP8 E4M3 just before the QK dot — one register-resident
# MFMA prologue per program, vs four tensor mallocs across all programs.
#
# Mirrors the design at `_tq_fuse_q_rotation` in
# `triton_turboquant_unified_attention.py`. The bf16 rotation matches TQ44
# "[Opt E]" — KV is 4-bit quantized so bf16 rounding sits well inside
# quantization noise. Validated by parity tests at unit-test scale.
# ---------------------------------------------------------------------------


@triton.jit
def _fp8_g32_fuse_q_rotation(
    Q,                     # [BLOCK_M, HEAD_SIZE_PADDED] — raw query (bf16/fp16)
    PiT_ptr,
    PiT_stride_0: tl.int64,
    PiT_stride_1: tl.int64,
    dim_mask,              # [HEAD_SIZE_PADDED] int1
    HEAD_SIZE_PADDED: tl.constexpr,
):
    """Fused Q @ PiT prologue. Returns the rotated Q in FP8 E4M3 ready for
    the QK dot. PiT is fp32 in storage but loaded as bf16 here so the
    rotation uses ``v_mfma_f32_16x16x32_bf16_1k`` (8 MFMAs at D=128) instead
    of fp32 MFMAs (64 instructions at D=128) — 8× fewer issued instructions
    and lower register pressure.
    """
    d_offs = tl.arange(0, HEAD_SIZE_PADDED)
    pit_offsets = (
        d_offs[:, None] * PiT_stride_0 + d_offs[None, :] * PiT_stride_1
    )
    pit_mask = dim_mask[:, None] & dim_mask[None, :]
    PiT_tile = tl.load(PiT_ptr + pit_offsets, mask=pit_mask, other=0.0).to(
        tl.bfloat16
    )
    Q_rot = tl.dot(Q.to(tl.bfloat16), PiT_tile)  # fp32 accumulator
    return Q_rot.to(tl.float8e4nv)


# ---------------------------------------------------------------------------
# K loader for the scaled-MFMA path: returns raw packed FP4 codes + E8M0
# scales, both as uint8. No software dequant — the hardware MFMA consumes
# both directly via ``tl.dot_scaled``.
# ---------------------------------------------------------------------------


@triton.jit
def _fp8_g32_load_k_fp8(
    KV_cache_ptr,          # uint8 view of cache
    data_bases,            # [TILE_SIZE] int64 — slot base byte offset
    Fp4_decode_ptr,        # [16] bf16 — FP4 LUT (re-cast to FP8 in-kernel)
    d_offs,                # [BLOCK_D] int32
    d_mask,                # [BLOCK_D] int1
    tile_mask,             # [TILE_SIZE] int1 (ignored when UNMASKED=True)
    BLOCK_D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr,
    N_GROUPS_C: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    K_SCALES_OFFSET: tl.constexpr,
    UE8M0_BIAS_C: tl.constexpr,
    UNMASKED: tl.constexpr,
    ARCH_B_C: tl.constexpr,  # 0 = Arch A, 1 = Arch B (multiply scales by SCALE_C)
    SCALE_C: tl.constexpr,   # c factor used in Arch B's c-baked codebook
):
    """Dequant K to FP8 E4M3 with E8M0 scales folded in (lossless within
    typical decode dynamic range).

    Returns ``K_T_fp8 : [BLOCK_D, TILE_SIZE]`` in fp8_e4m3fn, ready for
    ``tl.dot(Q_fp8, K_T_fp8)`` which lowers to ``v_mfma_f32_16x16x128_f8f6f4``
    on gfx950 (4× the K density per MFMA instruction vs the bf16 path).

    Math: FP4 LUT decode is exact in bf16. E8M0 scale `s = 2^e` multiplies
    only the exponent of the bf16 product, leaving the (FP4-shaped) mantissa
    unchanged — so the round-trip into FP8 E4M3 is lossless as long as the
    final exponent fits FP8's range [-9, +8].
    """
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
    fp4_vals = tl.load(Fp4_decode_ptr + codes).to(tl.float32)  # [TILE, BLOCK_D]

    # E8M0 scales -> fp32, then broadcast-multiply per group.
    grp = tl.arange(0, N_GROUPS_C)
    scale_addrs = data_bases[:, None] + K_SCALES_OFFSET + grp[None, :]
    if UNMASKED:
        scale_bytes = tl.load(KV_cache_ptr + scale_addrs).to(tl.int32)
    else:
        scale_bytes = tl.load(
            KV_cache_ptr + scale_addrs, mask=tile_mask[:, None], other=0
        ).to(tl.int32)
    scale_exp = scale_bytes - UE8M0_BIAS_C
    scales = tl.where(
        scale_bytes == 0, 0.0, tl.exp2(tl.cast(scale_exp, tl.float32))
    )  # [TILE, N_GROUPS] fp32
    if ARCH_B_C == 1:
        # Arch B: stored byte is pure pow2 of absmax — fold `c` in here so
        # the K_T_fp8 tensor lands on the same dimensional grid as Arch A.
        scales = scales * tl.cast(SCALE_C, tl.float32)

    K_g = tl.reshape(fp4_vals, [TILE_SIZE, N_GROUPS_C, GROUP_SIZE_C])
    K = tl.reshape(K_g * scales[:, :, None], [TILE_SIZE, BLOCK_D])
    # Cast to FP8 E4M3 and transpose to [BLOCK_D, TILE_SIZE] for tl.dot rhs.
    K_T_fp8 = tl.trans(K.to(tl.float8e4nv))
    _ = HEAD_DIM
    _ = GROUP_SIZE_C
    return K_T_fp8


@triton.jit
def _fp8_g32_load_k_packed(
    KV_cache_ptr,          # uint8 view of cache
    data_bases,            # [TILE_SIZE] int64 — slot base byte offset
    d_half_offs,           # [BLOCK_D // 2] int32
    half_mask,             # [BLOCK_D // 2] int1
    tile_mask,             # [TILE_SIZE] int1 (ignored when UNMASKED=True)
    BLOCK_D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr,
    N_GROUPS_C: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    K_SCALES_OFFSET: tl.constexpr,
    UNMASKED: tl.constexpr,
):
    """Load packed FP4 K codes (no decode) + E8M0 K scales as uint8 tensors.

    Returns:
        K_T_packed : [BLOCK_D // 2, TILE_SIZE] uint8 — FP4 codes, K-dim
                     packed (each byte = (codes[2h] | codes[2h+1] << 4)),
                     transposed to ready-for-tl.dot_scaled rhs layout.
        K_scales   : [TILE_SIZE, N_GROUPS_C] uint8 — E8M0 scale bytes,
                     ready-for-tl.dot_scaled rhs_scale layout (N rows, K//gs cols).
    """
    addrs = data_bases[:, None] + d_half_offs[None, :]
    if UNMASKED:
        K_codes = tl.load(
            KV_cache_ptr + addrs, mask=half_mask[None, :], other=0
        )
    else:
        K_codes = tl.load(
            KV_cache_ptr + addrs,
            mask=tile_mask[:, None] & half_mask[None, :],
            other=0,
        )
    K_T_packed = tl.trans(K_codes)  # [D // 2, TILE]

    grp = tl.arange(0, N_GROUPS_C)
    scale_addrs = data_bases[:, None] + K_SCALES_OFFSET + grp[None, :]
    if UNMASKED:
        K_scales = tl.load(KV_cache_ptr + scale_addrs)
    else:
        K_scales = tl.load(
            KV_cache_ptr + scale_addrs, mask=tile_mask[:, None], other=0
        )
    _ = HEAD_DIM
    _ = GROUP_SIZE_C
    _ = BLOCK_D
    return K_T_packed, K_scales


# ---------------------------------------------------------------------------
# V loader: LUT-decode FP4 → bf16 then multiply by E8M0-decoded fp32 scales.
#
# Our V scales are grouped along the *output* dim (HEAD_DIM), not the
# reduction dim (TILE_SIZE). MX-FP scaled dot_scaled wants scales on the
# reduction axis for rhs, so we cannot pass V scales directly as
# rhs_scale. We have two options:
#   (a) Bake D-axis scales into the dequantized values during the load,
#       then cast to fp8 and use dot_scaled with an identity rhs_scale.
#       This is what VLLM_FP8_G32_V_DOT_SCALED=1 does (Path 2). It gets
#       us native fp8×fp8 MFMA (2× bf16 ceiling) without changing the V
#       layout.
#   (b) Re-quantize V along the K-axis at store time (Path 1, not yet
#       implemented). This would let dot_scaled consume V's scales
#       directly, plus drop V down to fp4 inputs for 4× bf16 ceiling.
# Default (V_DOT_KIND=0) keeps the original bf16 GEMM path for
# correctness parity with fp4_g32 V3.
# ---------------------------------------------------------------------------


@triton.jit
def _fp8_g32_load_v_tile(
    KV_cache_ptr,
    val_bases,             # [TILE_SIZE] int64 — V codes base
    v_scales_addrs,        # [TILE_SIZE] int64 — V scales base (byte offset)
    Fp4_decode_ptr,        # [16] bf16
    d_offs,
    d_mask,
    tile_mask,
    OUT_DTYPE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr,
    N_GROUPS_C: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    UNMASKED: tl.constexpr,
    UE8M0_BIAS_C: tl.constexpr,
    ARCH_B_C: tl.constexpr,  # 0 = Arch A, 1 = Arch B (multiply scales by SCALE_C)
    SCALE_C: tl.constexpr,   # c factor used in Arch B's c-baked codebook
    V_BF16_MUL: tl.constexpr = 0,  # 1 = keep mul in bf16 (faster on MFMA), 0 = fp32 mul + cast
):
    """Load + dequant a [TILE_SIZE, BLOCK_D] block of V (raw, not rotated).

    When V_BF16_MUL=1 (opt-in via VLLM_FP8_G32_V_BF16=1) the entire
    decode+scale chain stays in bf16:
      - FP4 LUT lookup already returns bf16 (skip the .to(fp32))
      - UE8M0 scales are powers of 2 — exactly representable in bf16
        (within the bf16 exponent range, which covers all practical KV).
      - The dot(P, V) that consumes this tile uses a bf16 MFMA anyway,
        so producing V in bf16 saves 2k fp32 ALU ops + 2k fp32->bf16 casts
        per tile vs the fp32-mul path.
    """
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

    grp = tl.arange(0, N_GROUPS_C)
    scale_addrs = v_scales_addrs[:, None] + grp[None, :]
    if UNMASKED:
        scale_bytes = tl.load(KV_cache_ptr + scale_addrs).to(tl.int32)
    else:
        scale_bytes = tl.load(
            KV_cache_ptr + scale_addrs, mask=tile_mask[:, None], other=0
        ).to(tl.int32)
    scale_exp = scale_bytes - UE8M0_BIAS_C

    if V_BF16_MUL:
        fp4_vals = tl.load(Fp4_decode_ptr + codes)  # bf16 already
        scales = tl.where(
            scale_bytes == 0,
            tl.zeros([], tl.float32),
            tl.exp2(tl.cast(scale_exp, tl.float32)),
        ).to(tl.bfloat16)
        if ARCH_B_C == 1:
            scales = scales * tl.cast(SCALE_C, tl.bfloat16)
        V_g = tl.reshape(fp4_vals, [TILE_SIZE, N_GROUPS_C, GROUP_SIZE_C])
        V = tl.reshape(V_g * scales[:, :, None], [TILE_SIZE, BLOCK_D])
        return V.to(OUT_DTYPE)
    else:
        fp4_vals = tl.load(Fp4_decode_ptr + codes).to(tl.float32)
        scales = tl.where(
            scale_bytes == 0, 0.0, tl.exp2(tl.cast(scale_exp, tl.float32))
        )  # [TILE, N_GROUPS] fp32
        if ARCH_B_C == 1:
            scales = scales * tl.cast(SCALE_C, tl.float32)

        V_g = tl.reshape(fp4_vals, [TILE_SIZE, N_GROUPS_C, GROUP_SIZE_C])
        V = tl.reshape(V_g * scales[:, :, None], [TILE_SIZE, BLOCK_D]).to(OUT_DTYPE)
        return V


# ---------------------------------------------------------------------------
# PV dispatch helper — branches on V_DOT_KIND constexpr to pick between the
# bf16 dot path (V_DOT_KIND=0, default) and the fp8×fp8 scaled-MFMA path
# (V_DOT_KIND=1, opt-in via VLLM_FP8_G32_V_DOT_SCALED=1).
#
# Path V_DOT_KIND=0 (default, matches TQ44 V3 V-path):
#   - V loader returns bf16 with D-axis scales applied in software.
#   - PV uses ``tl.dot(P_bf16, V_bf16)`` → ``v_mfma_f32_16x16x32_bf16``.
#
# Path V_DOT_KIND=1 (opt-in, "Path 2" of the V-MFMA optimization sprint):
#   - V loader returns FP8 E4M3 with D-axis scales pre-multiplied into the
#     codes (bf16 multiply then cast to fp8, saturating).
#   - P quantized row-wise to FP8 E4M3 with a single E8M0 scale per row
#     (one scale group along TILE_SIZE, which is forced to 128).
#   - PV uses ``tl.dot_scaled(P_fp8, V_fp8, lhs_scale=P_row_e8m0,
#     rhs_scale=identity_e8m0)`` → native scaled FP8 MFMA on gfx950
#     (``v_mfma_scale_f32_16x16x128_f8f6f4`` at K=128 per issue). 2× the
#     bf16 MFMA throughput per issue at the right K.
#   - Numerical cost: P loses ~3 bits of precision (fp32 → fp8 with row
#     scaling); the per-row max + ceil(log2) scaling minimizes the loss.
#   - Hardware constraint: native scaled FP8 MFMA is K=128 per instruction.
#     At smaller K (e.g. TILE_SIZE=32 = 1 scale group) only 1/4 of the
#     MFMA throughput is used → net slower than bf16. The launcher
#     enforces TILE_SIZE=128 minimum when V_DOT_KIND=1.
#
# Why this is a win over the bf16 path (at the right K):
#   - Native fp8 MFMA on CDNA4: 512 TFLOPS vs bf16 256 TFLOPS = 2× ceiling.
#   - V codes carry only 4 bits of payload anyway (FP4 E2M1), so the
#     fp8-E4M3 cast at the end of the V loader is lossless for the value
#     itself (E4M3 has 3 mantissa bits, FP4 E2M1 has 1 mantissa bit, so
#     all FP4 values are exactly representable in FP8 E4M3).
#   - The D-axis scale multiply moves out of the inner MFMA path and into
#     a single bf16 mul during V load; the MFMA itself sees pre-scaled V.
#
# Why this isn't the default:
#   - The 2× MFMA win at K=128 is real but only materializes when per-CTA
#     work is large enough to amortize the increased per-iter overhead
#     (P quant, fp8 cast, scale build) AND the launch overhead per CTA.
#   - At our serving target (B=64, 8K context, NUM_KV_SPLITS=16), each
#     CTA does only ~4 inner-loop iters at TILE_SIZE=128, leaving the
#     launch overhead dominant — net ~20% slower than the bf16 path.
#   - At smaller batch sizes the path can win up to ~1.15×.
#   - For a real >2× win on V we'd need Path 1: re-layout V scales on the
#     K-axis so we can keep V as FP4 codes and use the 4×-bf16 scaled
#     FP4 MFMA (``v_mfma_scale_f32_16x16x128_f8f6f4`` at f4 inputs).
#     That's a store-side change and a future sprint.
# ---------------------------------------------------------------------------


@triton.jit
def _fp8_g32_pv(
    P,                       # [BLOCK_M, TILE_SIZE] fp32 (softmax probs)
    KV_cache_ptr,
    val_bases,               # [TILE_SIZE] int64
    v_scales_addrs,          # [TILE_SIZE] int64
    Fp4_decode_ptr,          # [16] bf16
    d_offs, d_mask, tile_mask,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr,
    N_GROUPS_C: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    UNMASKED: tl.constexpr,
    UE8M0_BIAS_C: tl.constexpr,
    ARCH_B_C: tl.constexpr,
    SCALE_C: tl.constexpr,
    V_BF16_MUL: tl.constexpr,
    V_DOT_KIND: tl.constexpr,   # 0=bf16 dot (default), 1=fp8×fp8 dot_scaled
):
    """Compute the (P @ V_tile) contribution. Returns [BLOCK_M, BLOCK_D] fp32.

    The caller is responsible for accumulating the returned tile into the
    output. This helper handles V loading + the dot kind dispatch.
    """
    if V_DOT_KIND == 0:
        # Default bf16 path — exactly the old code, unchanged behavior.
        V = _fp8_g32_load_v_tile(
            KV_cache_ptr, val_bases, v_scales_addrs, Fp4_decode_ptr,
            d_offs, d_mask, tile_mask,
            OUT_DTYPE=tl.bfloat16,
            HEAD_DIM=HEAD_DIM, BLOCK_D=BLOCK_D,
            GROUP_SIZE_C=GROUP_SIZE_C, N_GROUPS_C=N_GROUPS_C,
            TILE_SIZE=TILE_SIZE,
            UNMASKED=UNMASKED, UE8M0_BIAS_C=UE8M0_BIAS_C,
            ARCH_B_C=ARCH_B_C, SCALE_C=SCALE_C,
            V_BF16_MUL=V_BF16_MUL,
        )
        return tl.dot(P.to(tl.bfloat16), V, out_dtype=tl.float32)
    else:
        # New fp8×fp8 dot_scaled path (Path 2). TILE_SIZE must be 32.
        # 1) Load V with D-axis scales applied and cast to fp8 E4M3.
        V_fp8 = _fp8_g32_load_v_tile(
            KV_cache_ptr, val_bases, v_scales_addrs, Fp4_decode_ptr,
            d_offs, d_mask, tile_mask,
            OUT_DTYPE=tl.float8e4nv,
            HEAD_DIM=HEAD_DIM, BLOCK_D=BLOCK_D,
            GROUP_SIZE_C=GROUP_SIZE_C, N_GROUPS_C=N_GROUPS_C,
            TILE_SIZE=TILE_SIZE,
            UNMASKED=UNMASKED, UE8M0_BIAS_C=UE8M0_BIAS_C,
            ARCH_B_C=ARCH_B_C, SCALE_C=SCALE_C,
            V_BF16_MUL=1,  # bf16 multiply + fp8 cast (skip fp32 round-trip)
        )

        # 2) Row-wise P fp8 quantization with one E8M0 scale per row.
        #    P is post-softmax so it's non-negative in [0, 1].
        FP8_E4M3_MAX: tl.constexpr = 448.0
        P_row_max = tl.max(P, axis=1)  # [BLOCK_M]
        # Guard against all-zero rows (fully masked tiles).
        P_row_max_safe = tl.maximum(P_row_max, 1e-30)
        # e_row = ceil(log2(P_row_max / FP8_MAX)) so 2^e_row >= P_row_max / FP8_MAX.
        # Then P_scaled = P / 2^e_row fits entirely in fp8 E4M3 range.
        e_row = tl.ceil(tl.log2(P_row_max_safe / FP8_E4M3_MAX))
        # Encode as E8M0 byte: byte = e + UE8M0_BIAS, clamp to [0, 255].
        e_byte = tl.cast(e_row + UE8M0_BIAS_C, tl.int32)
        e_byte = tl.maximum(0, tl.minimum(255, e_byte))
        scale_factor = tl.exp2(
            tl.cast(e_byte - UE8M0_BIAS_C, tl.float32)
        )  # [BLOCK_M]
        P_scaled = P / scale_factor[:, None]  # [BLOCK_M, TILE_SIZE]
        P_fp8 = P_scaled.to(tl.float8e4nv)

        # 3) Build scale tensors for dot_scaled.
        #    Shape contract: lhs_scale=[M, K//32], rhs_scale=[N, K//32].
        #    M=BLOCK_M, K=TILE_SIZE, N=BLOCK_D, group=32.
        NUM_K_GROUPS_PV: tl.constexpr = TILE_SIZE // 32
        # lhs_scale: per-row P scale, broadcast across K-groups.
        lhs_scale = tl.broadcast_to(
            tl.cast(e_byte, tl.uint8)[:, None],
            [BLOCK_M, NUM_K_GROUPS_PV],
        )
        # rhs_scale: identity for V (D-axis scales already baked into V_fp8).
        #            0x7F = 127 = UE8M0_BIAS = exponent 0 → multiplier 1.0.
        rhs_scale = tl.full([BLOCK_D, NUM_K_GROUPS_PV], 0x7F, dtype=tl.uint8)

        return tl.dot_scaled(
            P_fp8, lhs_scale, "e4m3",
            V_fp8, rhs_scale, "e4m3",
            out_dtype=tl.float32,
        )


# ---------------------------------------------------------------------------
# QK dispatch helper — branches on DOT_KIND constexpr to pick between the
# native F8F6F4 MFMA path (DOT_KIND=0) and the dot_scaled emulation path
# (DOT_KIND=1). Inlined into both 2D and 3D kernels.
# ---------------------------------------------------------------------------


@triton.jit
def _fp8_g32_qk(
    Q,                       # [BLOCK_M, BLOCK_D] fp8_e4m3fn
    KV_cache_ptr,
    data_bases,              # [TILE_SIZE] int64
    Fp4_decode_ptr,          # [16] bf16
    offs_d, dim_mask,
    offs_d_half, half_mask,
    tile_mask,
    BLOCK_D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr,
    N_GROUPS_C: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    K_SCALES_OFFSET: tl.constexpr,
    UE8M0_BIAS_C: tl.constexpr,
    UNMASKED: tl.constexpr,
    DOT_KIND: tl.constexpr,  # 0=fp8mfma, 1=dot_scaled. BOTH emit native F8F6F4
                             # MFMA at K=128/instr; see lowering notes below.
    ARCH_B_C: tl.constexpr,  # 0 = Arch A, 1 = Arch B
    SCALE_C: tl.constexpr,   # Arch B's c (folded into K_T_fp8 for the fp8mfma
                             # path; pre-folded into softmax scale by the
                             # launcher for the dot_scaled path)
):
    if DOT_KIND == 0:
        # Path "fp8mfma": dequant K to FP8 in software (lossless within
        # typical Hadamard-rotated K range), then plain tl.dot → native
        # v_mfma_f32_16x16x128_f8f6f4 (K=128 per MFMA instruction,
        # non-scaled variant — scales already absorbed in software).
        K_T_fp8 = _fp8_g32_load_k_fp8(
            KV_cache_ptr,
            data_bases,
            Fp4_decode_ptr,
            offs_d, dim_mask,
            tile_mask,
            BLOCK_D=BLOCK_D, HEAD_DIM=HEAD_DIM,
            GROUP_SIZE_C=GROUP_SIZE_C, N_GROUPS_C=N_GROUPS_C,
            TILE_SIZE=TILE_SIZE,
            K_SCALES_OFFSET=K_SCALES_OFFSET,
            UE8M0_BIAS_C=UE8M0_BIAS_C,
            UNMASKED=UNMASKED,
            ARCH_B_C=ARCH_B_C,
            SCALE_C=SCALE_C,
        )
        S = tl.dot(Q, K_T_fp8, out_dtype=tl.float32)
    else:
        # Path "dot_scaled" (DEFAULT): Triton 3.6 on gfx950 lowers this
        # to v_mfma_scale_f32_16x16x128_f8f6f4 — the *scaled* native MFMA,
        # which consumes the E8M0 K-scales inside the MFMA itself (zero
        # software dequant). Verified by AMDGCN disassembly of the full
        # compiled kernel.
        #
        # Arch B note: the hardware MFMA cannot fold the extra `c` factor
        # in its rhs_scale path, so we pre-fold `c` into the softmax
        # `scale` parameter at the launcher (cheaper than an extra
        # per-tile multiply here).
        K_T_packed, K_scales = _fp8_g32_load_k_packed(
            KV_cache_ptr,
            data_bases,
            offs_d_half, half_mask,
            tile_mask,
            BLOCK_D=BLOCK_D, HEAD_DIM=HEAD_DIM,
            GROUP_SIZE_C=GROUP_SIZE_C, N_GROUPS_C=N_GROUPS_C,
            TILE_SIZE=TILE_SIZE,
            K_SCALES_OFFSET=K_SCALES_OFFSET,
            UNMASKED=UNMASKED,
        )
        S = tl.dot_scaled(
            Q, None, "e4m3", K_T_packed, K_scales, "e2m1",
            out_dtype=tl.float32,
        )
    return S


# ===========================================================================
# Unified 2D attention kernel (prefill / short-context decode)
# ===========================================================================


@triton.jit
def kernel_fp8_g32_unified_attention_2d(
    output_ptr,
    query_ptr,               # FP8 E4M3 (FUSE_Q_ROT=0) or raw bf16 (FUSE_Q_ROT=1)
    KV_cache_ptr,            # uint8 view
    Fp4_decode_ptr,          # [16] bf16
    PiT_ptr,                 # fp32 [D,D]; only used when FUSE_Q_ROT=1
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
    stride_cache_pos: tl.int64,
    stride_cache_head: tl.int64,
    pit_stride_0: tl.int64,
    pit_stride_1: tl.int64,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
    num_seqs: tl.int32,
    K_SCALES_OFFSET: tl.constexpr,
    V_CODES_OFFSET: tl.constexpr,
    V_SCALES_OFFSET: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr,
    N_GROUPS_C: tl.constexpr,
    UE8M0_BIAS_C: tl.constexpr,
    DOT_KIND: tl.constexpr = 0,  # 0=fp8mfma (default), 1=dot_scaled
    USE_SINKS: tl.constexpr = 0,
    SLIDING_WINDOW: tl.constexpr = 0,
    ARCH_B_C: tl.constexpr = 0,  # 0 = Arch A, 1 = Arch B (c-baked codebook)
    SCALE_C: tl.constexpr = 1.0, # Arch B's c (folded into K/V scale at dequant)
    FUSE_Q_ROT: tl.constexpr = 0,  # 1 = load raw bf16 Q + fused rotate, 0 = pre-rotated fp8
    V_BF16_MUL: tl.constexpr = 0,  # 1 = V dequant in bf16 (skip fp32 intermediate)
    V_DOT_KIND: tl.constexpr = 0,  # 0=bf16 dot (default), 1=fp8×fp8 dot_scaled (TILE_SIZE must be 32)
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    seq_idx = _find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )
    q_block_start_idx = (
        tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx
    )
    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
    cur_batch_query_len = (
        cur_batch_in_all_stop_index - cur_batch_in_all_start_index
    )

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_d_half = tl.arange(0, HEAD_SIZE_PADDED // 2)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = (
        kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    )
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    half_mask = tl.where(offs_d_half * 2 < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    if FUSE_Q_ROT:
        Q_raw = tl.load(
            query_ptr + query_offset,
            mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
            other=tl.zeros([], tl.bfloat16),
        )
        Q = _fp8_g32_fuse_q_rotation(
            Q_raw, PiT_ptr, pit_stride_0, pit_stride_1, dim_mask,
            HEAD_SIZE_PADDED=HEAD_SIZE_PADDED,
        )
    else:
        Q = tl.load(
            query_ptr + query_offset,
            mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
            other=tl.zeros([], tl.float8e4nv),
        )  # FP8 E4M3 already (caller pre-rotated + pre-cast).

    block_table_offset = seq_idx * block_table_stride

    if USE_SINKS:
        M = tl.load(
            sinks_ptr + query_offset_1, mask=query_mask_1, other=float("-inf"),
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

    tile_start = 0
    tile_end = num_tiles
    if SLIDING_WINDOW > 0:
        qpos_lo = q_block_local_idx * BLOCK_Q
        qpos_hi = tl.minimum(
            qpos_lo + (BLOCK_M - 1) // num_queries_per_kv,
            cur_batch_query_len - 1,
        )
        first_allowed_key = context_len + qpos_lo - SLIDING_WINDOW + 1
        last_allowed_key = context_len + qpos_hi
        tile_start = tl.maximum(0, first_allowed_key // TILE_SIZE)
        tile_end = tl.minimum((last_allowed_key // TILE_SIZE) + 1, num_tiles)

    query_abs_pos = context_len + query_pos[:, None]
    dummy_tile_mask = tl.full([TILE_SIZE], 1, tl.int1)

    for j in range(tile_start, tile_end - 1):
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
        v_scales_addrs = data_bases + V_SCALES_OFFSET

        S = scale * _fp8_g32_qk(
            Q, KV_cache_ptr, data_bases, Fp4_decode_ptr,
            offs_d, dim_mask, offs_d_half, half_mask, dummy_tile_mask,
            BLOCK_D=HEAD_SIZE_PADDED, HEAD_DIM=HEAD_SIZE,
            GROUP_SIZE_C=GROUP_SIZE_C, N_GROUPS_C=N_GROUPS_C,
            TILE_SIZE=TILE_SIZE,
            K_SCALES_OFFSET=K_SCALES_OFFSET,
            UE8M0_BIAS_C=UE8M0_BIAS_C,
            UNMASKED=True, DOT_KIND=DOT_KIND,
            ARCH_B_C=ARCH_B_C, SCALE_C=SCALE_C,
        )
        seq_mask = seq_offset[None, :] <= query_abs_pos
        if SLIDING_WINDOW > 0:
            seq_mask = seq_mask & (
                (query_abs_pos - seq_offset[None, :]) < SLIDING_WINDOW
            )
        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask,
            S, float("-inf"),
        )
        m_j = tl.maximum(M, tl.max(S, axis=1))
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
        P = tl.exp(S - m_j[:, None])
        l_j = tl.sum(P, axis=1)
        alpha = tl.exp(M - m_j)
        acc = acc * alpha[:, None]
        L = L * alpha + l_j
        M = m_j
        acc += _fp8_g32_pv(
            P, KV_cache_ptr, val_bases, v_scales_addrs, Fp4_decode_ptr,
            offs_d, dim_mask, dummy_tile_mask,
            HEAD_DIM=HEAD_SIZE, BLOCK_D=HEAD_SIZE_PADDED, BLOCK_M=BLOCK_M,
            GROUP_SIZE_C=GROUP_SIZE_C, N_GROUPS_C=N_GROUPS_C,
            TILE_SIZE=TILE_SIZE,
            UNMASKED=True, UE8M0_BIAS_C=UE8M0_BIAS_C,
            ARCH_B_C=ARCH_B_C, SCALE_C=SCALE_C,
            V_BF16_MUL=V_BF16_MUL,
            V_DOT_KIND=V_DOT_KIND,
        )

    # Tail tile (masked)
    if tile_end > tile_start:
        j = tile_end - 1
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
        v_scales_addrs = data_bases + V_SCALES_OFFSET

        S = scale * _fp8_g32_qk(
            Q, KV_cache_ptr, data_bases, Fp4_decode_ptr,
            offs_d, dim_mask, offs_d_half, half_mask, tile_mask,
            BLOCK_D=HEAD_SIZE_PADDED, HEAD_DIM=HEAD_SIZE,
            GROUP_SIZE_C=GROUP_SIZE_C, N_GROUPS_C=N_GROUPS_C,
            TILE_SIZE=TILE_SIZE,
            K_SCALES_OFFSET=K_SCALES_OFFSET,
            UE8M0_BIAS_C=UE8M0_BIAS_C,
            UNMASKED=False, DOT_KIND=DOT_KIND,
            ARCH_B_C=ARCH_B_C, SCALE_C=SCALE_C,
        )
        seq_mask = seq_offset[None, :] <= query_abs_pos
        if SLIDING_WINDOW > 0:
            seq_mask = seq_mask & (
                (query_abs_pos - seq_offset[None, :]) < SLIDING_WINDOW
            )
        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask,
            S, float("-inf"),
        )
        m_j = tl.maximum(M, tl.max(S, axis=1))
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
        P = tl.exp(S - m_j[:, None])
        l_j = tl.sum(P, axis=1)
        alpha = tl.exp(M - m_j)
        acc = acc * alpha[:, None]
        L = L * alpha + l_j
        M = m_j
        acc += _fp8_g32_pv(
            P, KV_cache_ptr, val_bases, v_scales_addrs, Fp4_decode_ptr,
            offs_d, dim_mask, tile_mask,
            HEAD_DIM=HEAD_SIZE, BLOCK_D=HEAD_SIZE_PADDED, BLOCK_M=BLOCK_M,
            GROUP_SIZE_C=GROUP_SIZE_C, N_GROUPS_C=N_GROUPS_C,
            TILE_SIZE=TILE_SIZE,
            UNMASKED=False, UE8M0_BIAS_C=UE8M0_BIAS_C,
            ARCH_B_C=ARCH_B_C, SCALE_C=SCALE_C,
            V_BF16_MUL=V_BF16_MUL,
            V_DOT_KIND=V_DOT_KIND,
        )

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


# ===========================================================================
# Unified 3D (split-KV) attention kernel
# ===========================================================================


@triton.jit
def kernel_fp8_g32_unified_attention_3d(
    segm_output_ptr,
    segm_max_ptr,
    segm_expsum_ptr,
    query_ptr,             # FP8 E4M3 (FUSE_Q_ROT=0) or raw bf16 (FUSE_Q_ROT=1)
    KV_cache_ptr,
    Fp4_decode_ptr,
    PiT_ptr,               # fp32 [D,D] Hadamard rotation matrix; only used when FUSE_Q_ROT=1
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
    stride_cache_pos: tl.int64,
    stride_cache_head: tl.int64,
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
    UE8M0_BIAS_C: tl.constexpr,
    DOT_KIND: tl.constexpr = 0,  # 0=fp8mfma (default), 1=dot_scaled
    USE_SINKS: tl.constexpr = 0,
    SLIDING_WINDOW: tl.constexpr = 0,
    ARCH_B_C: tl.constexpr = 0,  # 0 = Arch A, 1 = Arch B (c-baked codebook)
    SCALE_C: tl.constexpr = 1.0, # Arch B's c (folded into K/V scale at dequant)
    FUSE_Q_ROT: tl.constexpr = 0,  # 1 = load raw bf16 Q + fused rotate, 0 = pre-rotated fp8
    V_BF16_MUL: tl.constexpr = 0,  # 1 = V dequant in bf16 (skip fp32 intermediate)
    V_DOT_KIND: tl.constexpr = 0,  # 0=bf16 dot (default), 1=fp8×fp8 dot_scaled (TILE_SIZE must be 32)
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2)

    seq_idx = _find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )
    q_block_start_idx = (
        tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx
    )
    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
    cur_batch_query_len = (
        cur_batch_in_all_stop_index - cur_batch_in_all_start_index
    )

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    tiles_per_segment = tl.cdiv(seq_len, NUM_SEGMENTS_PER_SEQ * TILE_SIZE)
    if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_d_half = tl.arange(0, HEAD_SIZE_PADDED // 2)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = (
        kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    )
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    half_mask = tl.where(offs_d_half * 2 < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    if FUSE_Q_ROT:
        Q_raw = tl.load(
            query_ptr + query_offset,
            mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
            other=tl.zeros([], tl.bfloat16),
        )
        Q = _fp8_g32_fuse_q_rotation(
            Q_raw, PiT_ptr, pit_stride_0, pit_stride_1, dim_mask,
            HEAD_SIZE_PADDED=HEAD_SIZE_PADDED,
        )
    else:
        Q = tl.load(
            query_ptr + query_offset,
            mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
            other=tl.zeros([], tl.float8e4nv),
        )

    block_table_offset = seq_idx * block_table_stride

    if USE_SINKS and segm_idx == 0:
        M = tl.load(
            sinks_ptr + query_offset_1, mask=query_mask_1, other=float("-inf"),
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

    if SLIDING_WINDOW > 0:
        qpos_lo = q_block_local_idx * BLOCK_Q
        qpos_hi = tl.minimum(
            qpos_lo + (BLOCK_M - 1) // num_queries_per_kv,
            cur_batch_query_len - 1,
        )
        first_allowed_key = context_len + qpos_lo - SLIDING_WINDOW + 1
        last_allowed_key = context_len + qpos_hi
        swa_tile_start = tl.maximum(0, first_allowed_key // TILE_SIZE)
        swa_tile_end = tl.minimum((last_allowed_key // TILE_SIZE) + 1, num_tiles)
        tile_lo = tl.maximum(tile_lo, swa_tile_start)
        tile_hi = tl.minimum(tile_hi, swa_tile_end)

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
        v_scales_addrs = data_bases + V_SCALES_OFFSET

        S = scale * _fp8_g32_qk(
            Q, KV_cache_ptr, data_bases, Fp4_decode_ptr,
            offs_d, dim_mask, offs_d_half, half_mask, dummy_tile_mask,
            BLOCK_D=HEAD_SIZE_PADDED, HEAD_DIM=HEAD_SIZE,
            GROUP_SIZE_C=GROUP_SIZE_C, N_GROUPS_C=N_GROUPS_C,
            TILE_SIZE=TILE_SIZE,
            K_SCALES_OFFSET=K_SCALES_OFFSET,
            UE8M0_BIAS_C=UE8M0_BIAS_C,
            UNMASKED=True, DOT_KIND=DOT_KIND,
            ARCH_B_C=ARCH_B_C, SCALE_C=SCALE_C,
        )
        seq_mask = seq_offset[None, :] <= query_abs_pos
        if SLIDING_WINDOW > 0:
            seq_mask = seq_mask & (
                (query_abs_pos - seq_offset[None, :]) < SLIDING_WINDOW
            )
        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask,
            S, float("-inf"),
        )
        m_j = tl.maximum(M, tl.max(S, axis=1))
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
        P = tl.exp(S - m_j[:, None])
        l_j = tl.sum(P, axis=1)
        alpha = tl.exp(M - m_j)
        acc = acc * alpha[:, None]
        L = L * alpha + l_j
        M = m_j
        acc += _fp8_g32_pv(
            P, KV_cache_ptr, val_bases, v_scales_addrs, Fp4_decode_ptr,
            offs_d, dim_mask, dummy_tile_mask,
            HEAD_DIM=HEAD_SIZE, BLOCK_D=HEAD_SIZE_PADDED, BLOCK_M=BLOCK_M,
            GROUP_SIZE_C=GROUP_SIZE_C, N_GROUPS_C=N_GROUPS_C,
            TILE_SIZE=TILE_SIZE,
            UNMASKED=True, UE8M0_BIAS_C=UE8M0_BIAS_C,
            ARCH_B_C=ARCH_B_C, SCALE_C=SCALE_C,
            V_BF16_MUL=V_BF16_MUL,
            V_DOT_KIND=V_DOT_KIND,
        )

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
        v_scales_addrs = data_bases + V_SCALES_OFFSET

        S = scale * _fp8_g32_qk(
            Q, KV_cache_ptr, data_bases, Fp4_decode_ptr,
            offs_d, dim_mask, offs_d_half, half_mask, tile_mask,
            BLOCK_D=HEAD_SIZE_PADDED, HEAD_DIM=HEAD_SIZE,
            GROUP_SIZE_C=GROUP_SIZE_C, N_GROUPS_C=N_GROUPS_C,
            TILE_SIZE=TILE_SIZE,
            K_SCALES_OFFSET=K_SCALES_OFFSET,
            UE8M0_BIAS_C=UE8M0_BIAS_C,
            UNMASKED=False, DOT_KIND=DOT_KIND,
            ARCH_B_C=ARCH_B_C, SCALE_C=SCALE_C,
        )
        seq_mask = seq_offset[None, :] <= query_abs_pos
        if SLIDING_WINDOW > 0:
            seq_mask = seq_mask & (
                (query_abs_pos - seq_offset[None, :]) < SLIDING_WINDOW
            )
        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask,
            S, float("-inf"),
        )
        m_j = tl.maximum(M, tl.max(S, axis=1))
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
        P = tl.exp(S - m_j[:, None])
        l_j = tl.sum(P, axis=1)
        alpha = tl.exp(M - m_j)
        acc = acc * alpha[:, None]
        L = L * alpha + l_j
        M = m_j
        acc += _fp8_g32_pv(
            P, KV_cache_ptr, val_bases, v_scales_addrs, Fp4_decode_ptr,
            offs_d, dim_mask, tile_mask,
            HEAD_DIM=HEAD_SIZE, BLOCK_D=HEAD_SIZE_PADDED, BLOCK_M=BLOCK_M,
            GROUP_SIZE_C=GROUP_SIZE_C, N_GROUPS_C=N_GROUPS_C,
            TILE_SIZE=TILE_SIZE,
            UNMASKED=False, UE8M0_BIAS_C=UE8M0_BIAS_C,
            ARCH_B_C=ARCH_B_C, SCALE_C=SCALE_C,
            V_BF16_MUL=V_BF16_MUL,
            V_DOT_KIND=V_DOT_KIND,
        )

    # Write segment partials for stage-2 reduce.
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


# ===========================================================================
# Launcher
# ===========================================================================


def _amd_stage1_hints() -> dict[str, int]:
    """AMD Triton hints (waves_per_eu / matrix_instr_nonkdim / kpack)."""
    return {
        "waves_per_eu": int(os.environ.get("VLLM_FP8_G32_WAVES_PER_EU", "2")),
        "matrix_instr_nonkdim": int(os.environ.get(
            "VLLM_FP8_G32_MFMA_NONKDIM", "16")),
        "kpack": int(os.environ.get("VLLM_FP8_G32_KPACK", "2")),
    }


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


def fp8_g32_unified_attention(
    query: torch.Tensor,             # [num_tokens, Hq, D] fp16/bf16 (raw)
    kv_cache: torch.Tensor,          # [num_blocks, block_size, Hk, padded_slot] uint8
    block_table: torch.Tensor,       # [num_seqs, max_num_blocks] int32
    seq_lens: torch.Tensor,          # [num_seqs] int32
    query_start_loc: torch.Tensor,   # [num_seqs+1] int32
    scale: float,
    PiT: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
    tile_size: int | None = None,
    max_query_len: int | None = None,
    max_seq_len: int | None = None,
    num_kv_splits: int | None = None,
    force_2d: bool = False,
    sinks: torch.Tensor | None = None,
    sliding_window: int | None = None,
) -> torch.Tensor:
    """Launch unified FP8-g32 attention (V3 + scaled F8F6F4 MFMA)."""
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
            f"fp8_g32_unified: cache slot {padded_slot} < expected "
            f"{slot_size(D, gs)} for D={D} group_size={gs}"
        )
    if Hq % Hk != 0:
        raise ValueError(f"Hq={Hq} must be a multiple of Hk={Hk}")

    # Q rotation: either fused inside the kernel (FUSE_Q_ROT=1, opt-in) or
    # done here in Python via rocBLAS (FUSE_Q_ROT=0, default for now).
    #
    # [Perf OPT FUSE_Q_ROT] When VLLM_FP8_G32_FUSE_Q_ROT=1 we skip the
    # `(query.float() @ PiT).to(fp8)` chain (4 tensor allocs + 4 mem
    # roundtrips on [N, Hq, D]) and instead pass raw bf16 Q to the kernel
    # which does the Hadamard rotate in a tiny per-program bf16 MFMA
    # prologue and casts to fp8 inline. Mirrors TQ44's `FUSE_Q_ROT` (Opt B
    # in TQ).
    if PiT is None:
        PiT = _get_pit(D, device, torch.float32)
    elif PiT.dtype != torch.float32 or not PiT.is_contiguous():
        PiT = PiT.to(torch.float32).contiguous()

    fuse_q_rot = (
        os.environ.get("VLLM_FP8_G32_FUSE_Q_ROT", "0") == "1"
    )
    # [Perf OPT V_BF16_MUL] Compute V dequant in bf16 instead of fp32. The
    # final PV uses bf16 MFMA anyway, so the fp32 mul + fp32->bf16 cast in
    # the V loader is pure overhead. UE8M0 scales are pure pow2 → exactly
    # representable in bf16 within the bf16 exponent range (covers all
    # practical KV). Opt-in for safety; flip the default once benchmarked.
    v_bf16_mul = (
        os.environ.get("VLLM_FP8_G32_V_BF16", "0") == "1"
    )
    # [Perf OPT V_DOT_SCALED] Use fp8×fp8 dot_scaled for PV instead of bf16
    # dot. Lowers to native v_mfma_scale_f32_16x16x32_f8f8 on CDNA4 (2× the
    # bf16 MFMA throughput per issue). The V loader bakes D-axis scales into
    # the fp8 codes; P is row-wise quantized to fp8 with one E8M0 scale per
    # row. Requires TILE_SIZE % 32 == 0 (MX-FP scale-group standard); we
    # force tile_size=32 when this is enabled. Opt-in for first land.
    v_dot_scaled = (
        os.environ.get("VLLM_FP8_G32_V_DOT_SCALED", "0") == "1"
    )
    V_DOT_KIND = 1 if v_dot_scaled else 0
    if fuse_q_rot:
        q_for_kernel = query if query.is_contiguous() else query.contiguous()
        # PiT stays fp32 in storage; kernel loads as bf16 for an 8x smaller
        # MFMA tile count (D=128 → 8 bf16 MFMAs vs 64 fp32 MFMAs).
        pit_stride_0 = PiT.stride(0)
        pit_stride_1 = PiT.stride(1)
    else:
        q_rot = (query.float() @ PiT).contiguous()                  # [N, Hq, D] fp32
        q_fp8 = q_rot.to(torch.float8_e4m3fn).contiguous()          # [N, Hq, D] fp8
        q_for_kernel = q_fp8
        pit_stride_0 = 0  # unused under FUSE_Q_ROT=0
        pit_stride_1 = 0

    if sinks is not None:
        sinks_f32 = sinks if sinks.dtype == torch.float32 else sinks.to(torch.float32)
        if not sinks_f32.is_contiguous():
            sinks_f32 = sinks_f32.contiguous()
        assert sinks_f32.numel() == Hq, (
            f"sinks must have shape [Hq={Hq}], got numel={sinks_f32.numel()}"
        )
        use_sinks = True
    else:
        sinks_f32 = q_for_kernel  # harmless dummy; never dereferenced when USE_SINKS=0
        use_sinks = False

    if output is None:
        output = torch.empty_like(query)

    # BLOCK_M heuristic (mirrors fp4_g32 V3 / TQ V3).
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

    fp4_dec = _get_fp4_decode_table_bf16(device)

    if tile_size is None:
        tile_size = 32 if is_prefill_like else 16
    _tile_override = os.environ.get("VLLM_FP8_G32_TILE_SIZE_DECODE")
    if (not is_prefill_like) and _tile_override is not None:
        tile_size = int(_tile_override)
    # V_DOT_SCALED requires TILE_SIZE >= 128 to actually win:
    #   - The native scaled FP8 MFMA on CDNA4 is v_mfma_scale_f32_16x16x128_f8f6f4
    #     with K=128 per instruction. At K=32 (TILE_SIZE=32) we emit one MFMA
    #     per scale group but only fill 1/4 of its K throughput — net result
    #     is a regression vs bf16 dot.
    #   - Isolated microbench (gfx950, MI355X, Triton 3.6):
    #       K=32  : bf16 3.14 TFLOPS, fp8 3.02 TFLOPS → 0.96x  (LOSS)
    #       K=64  : bf16 4.52 TFLOPS, fp8 5.68 TFLOPS → 1.26x
    #       K=128 : bf16 5.77 TFLOPS, fp8 11.71 TFLOPS → 2.03x  ✓
    #       K=256 : bf16 6.55 TFLOPS, fp8 22.97 TFLOPS → 3.51x
    #   - Force TILE_SIZE to the smallest size that wins: 128.
    if V_DOT_KIND == 1 and tile_size < 128:
        tile_size = 128
    elif V_DOT_KIND == 1 and tile_size % 32 != 0:
        # Sanity guard if user manually overrides above 128.
        tile_size = (tile_size // 32) * 32 if tile_size >= 32 else 128

    _global_stages = os.environ.get("VLLM_FP8_G32_NUM_STAGES")
    if _global_stages is not None:
        num_stages_2d = int(_global_stages)
        num_stages_3d = int(_global_stages)
    else:
        num_stages_2d = int(os.environ.get(
            "VLLM_FP8_G32_NUM_STAGES_2D", "1" if _is_hip else "2"))
        # [Perf default] 3D decode benefits from deeper software pipelining
        # on HIP: at Qwen-72B 8K C=64 serving, num_stages=3 gave +2.2%
        # OutTPS and -2.2% TPOT vs num_stages=2 (closing the TQ44 V3 gap
        # from -3.4% to -1.6%). Unlike TQ44 — which regressed at stages=3
        # due to extra VGPR pressure from its centroid LUT — fp8_g32's
        # dot_scaled K path has lower register usage and can afford the
        # extra pipeline depth. Set =2 to revert.
        # [Perf default] V_DOT_KIND=1 uses TILE_SIZE=128 which roughly
        # quadruples per-iter V working set vs TILE_SIZE=32. Drop stages
        # from 3 to 1 to keep register pressure in check; otherwise we
        # see large-batch regressions from VGPR spills.
        _default_stages_3d = (
            "1" if (_is_hip and V_DOT_KIND == 1)
            else ("3" if _is_hip else "2")
        )
        num_stages_3d = int(os.environ.get(
            "VLLM_FP8_G32_NUM_STAGES_3D", _default_stages_3d))

    BLOCK_D = triton.next_power_of_2(D)
    N_GROUPS_C = n_groups(D, gs)
    K_SCALES_OFFSET = k_scales_offset(D, gs)
    V_CODES_OFFSET = v_codes_offset(D, gs)
    V_SCALES_OFFSET = v_scales_offset(D, gs)

    # QK dispatch path selector.
    #
    # DEFAULT = "dot_scaled" → ``tl.dot_scaled(Q_fp8, K_packed, "e4m3",
    # "e2m1", rhs_scale=K_scales_e8m0)``. In the full unified-kernel
    # context on Triton 3.6 / gfx950 this lowers to the **native scaled
    # F8F6F4 MFMA**: ``v_mfma_scale_f32_16x16x128_f8f6f4`` (K=128 per
    # MFMA instruction, E8M0 scales consumed by the MFMA itself — zero
    # software dequant). Microbench at MI355X B=1 Hq=8 Hk=2:
    #
    #   N      V1     V3-fp8mfma  V3-dot_scaled   fp4_g32-V3
    #   512    70 us  102 us       73 us           69 us
    #   2K     92 us   92 us       92 us          100 us
    #   8K    315 us   90 us       91 us           97 us
    #   32K  1488 us   96 us       90 us           97 us   ← 16.5x vs V1
    #
    # Alt = "fp8mfma" → software FP4→FP8 dequant in the K loader + plain
    # ``tl.dot(Q_fp8, K_fp8)`` which lowers to the non-scaled native MFMA
    # ``v_mfma_f32_16x16x128_f8f6f4``. Same K=128 throughput as the
    # scaled variant but pays for software dequant. Kept for A/B and for
    # the (unlikely) case the scaled variant regresses on a future
    # Triton drop.
    _dot_kind_env = os.environ.get("VLLM_FP8_G32_V3_DOT_KIND", "dot_scaled").lower()
    if _dot_kind_env == "fp8mfma":
        DOT_KIND = 0
    elif _dot_kind_env == "dot_scaled":
        DOT_KIND = 1
    else:
        raise ValueError(
            f"VLLM_FP8_G32_V3_DOT_KIND must be 'fp8mfma' or 'dot_scaled', "
            f"got {_dot_kind_env!r}"
        )

    # Arch B (c-baked codebook) opt-in. The two paths handle the `c`
    # factor differently:
    #   - fp8mfma path: the K loader multiplies decoded scales by `c` in
    #     software (constexpr `SCALE_C`); softmax `scale` is untouched.
    #   - dot_scaled path: the hardware MFMA cannot apply the extra `c`
    #     in its rhs_scale path, so we pre-fold `c` into the softmax
    #     `scale` and pass it as the kernel argument. (Equivalent: one
    #     extra fp32 mul per QK accumulation, scalar.)
    # In both paths the V loader applies `c` in software (`SCALE_C` in
    # `_fp8_g32_load_v_tile`).
    ARCH_B_C = 1 if is_arch_b() else 0
    SCALE_C = get_constant_c() if ARCH_B_C == 1 else 1.0
    if ARCH_B_C == 1 and DOT_KIND == 1:
        scale_for_kernel = float(scale) * float(SCALE_C)
    else:
        scale_for_kernel = float(scale)

    kv_flat = _kv_cache_flat(kv_cache)

    # Dispatch: 2D for prefill / chunked; 3D for pure decode with long KV.
    if max_seq_len is None:
        max_seq_len_hint = int(block_table.shape[1]) * int(block_size)
    else:
        max_seq_len_hint = int(max_seq_len)
    use_3d = (not force_2d) and (not is_prefill_like) and max_seq_len_hint >= 1024

    if not use_3d:
        kernel_fp8_g32_unified_attention_2d[(total_num_q_blocks, Hk)](
            output_ptr=output,
            query_ptr=q_for_kernel,
            KV_cache_ptr=kv_flat,
            Fp4_decode_ptr=fp4_dec,
            PiT_ptr=PiT,
            block_tables_ptr=block_table,
            seq_lens_ptr=seq_lens,
            query_start_len_ptr=query_start_loc,
            sinks_ptr=sinks_f32,
            scale=scale_for_kernel,
            num_query_heads=Hq,
            num_queries_per_kv=kv_group_size,
            block_table_stride=block_table.stride(0),
            query_stride_0=q_for_kernel.stride(0),
            query_stride_1=q_for_kernel.stride(1),
            output_stride_0=output.stride(0),
            output_stride_1=output.stride(1),
            stride_cache_block=kv_cache.stride(0),
            stride_cache_pos=kv_cache.stride(1),
            stride_cache_head=kv_cache.stride(2),
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
            UE8M0_BIAS_C=UE8M0_BIAS,
            DOT_KIND=DOT_KIND,
            USE_SINKS=1 if use_sinks else 0,
            SLIDING_WINDOW=int(sliding_window) if sliding_window and sliding_window > 0 else 0,
            ARCH_B_C=ARCH_B_C,
            SCALE_C=SCALE_C,
            FUSE_Q_ROT=1 if fuse_q_rot else 0,
            V_BF16_MUL=1 if v_bf16_mul else 0,
            V_DOT_KIND=V_DOT_KIND,
            num_warps=4,
            num_stages=num_stages_2d,
        )
        return output

    # 3D split-KV path
    if num_kv_splits is None:
        # [Perf default] num_kv_splits=16 on HIP matches TQ44 V3's default
        # and gives ~36 tile iters per CTA at 8K context — a much better
        # balance between launch overhead and per-CTA arithmetic intensity
        # than the previous 64 (which over-subscribed MI355X's 256 CUs by
        # ~128x at C=64 with only ~9 tile iters per CTA). Set the env var
        # to override; e.g. VLLM_FP8_G32_NUM_KV_SPLITS=64 to restore the
        # previous behavior, or =8 for even fewer launches at the cost of
        # less SM saturation.
        #
        # Note: V_DOT_KIND=1 with TILE_SIZE=128 means per-CTA iters at 8K
        # context drop from ~32 (TILE_SIZE=16, splits=16) to ~4. Scaling
        # splits down to 2 saturates B=64 better but under-utilizes the
        # GPU at small batches. Keeping the default at 16 trades off:
        # wins at small-medium batch (B≤16), regresses at very large
        # batch (B=64 8K = 0.7x — launch overhead per-CTA is not
        # amortized over only 4 inner tile iterations). Users with high
        # concurrency should override to splits=2.
        _default_splits = 16 if _is_hip else 16
        num_kv_splits = int(os.environ.get(
            "VLLM_FP8_G32_NUM_KV_SPLITS", str(_default_splits)))
    if num_kv_splits < 1:
        num_kv_splits = 1
    if num_kv_splits & (num_kv_splits - 1) != 0:
        num_kv_splits = 1 << (num_kv_splits.bit_length() - 1)
    max_possible_splits = max(1, (max_seq_len_hint + tile_size - 1) // tile_size)
    num_segments = max(1, min(num_kv_splits, max_possible_splits))

    segm_output = torch.empty(
        (num_tokens, Hq, num_segments, BLOCK_D),
        dtype=torch.float32, device=device,
    )
    segm_max = torch.empty(
        (num_tokens, Hq, num_segments), dtype=torch.float32, device=device,
    )
    segm_expsum = torch.empty(
        (num_tokens, Hq, num_segments), dtype=torch.float32, device=device,
    )

    kernel_fp8_g32_unified_attention_3d[(total_num_q_blocks, Hk, num_segments)](
        segm_output_ptr=segm_output,
        segm_max_ptr=segm_max,
        segm_expsum_ptr=segm_expsum,
        query_ptr=q_for_kernel,
        KV_cache_ptr=kv_flat,
        Fp4_decode_ptr=fp4_dec,
        PiT_ptr=PiT,
        block_tables_ptr=block_table,
        seq_lens_ptr=seq_lens,
        query_start_len_ptr=query_start_loc,
        sinks_ptr=sinks_f32,
        scale=scale_for_kernel,
        num_query_heads=Hq,
        num_queries_per_kv=kv_group_size,
        block_table_stride=block_table.stride(0),
        query_stride_0=q_for_kernel.stride(0),
        query_stride_1=q_for_kernel.stride(1),
        stride_cache_block=kv_cache.stride(0),
        stride_cache_pos=kv_cache.stride(1),
        stride_cache_head=kv_cache.stride(2),
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
        UE8M0_BIAS_C=UE8M0_BIAS,
        DOT_KIND=DOT_KIND,
        USE_SINKS=1 if use_sinks else 0,
        SLIDING_WINDOW=int(sliding_window) if sliding_window and sliding_window > 0 else 0,
        ARCH_B_C=ARCH_B_C,
        SCALE_C=SCALE_C,
        FUSE_Q_ROT=1 if fuse_q_rot else 0,
        V_BF16_MUL=1 if v_bf16_mul else 0,
        V_DOT_KIND=V_DOT_KIND,
        num_warps=int(os.environ.get("VLLM_FP8_G32_NUM_WARPS_3D", "2")),
        num_stages=num_stages_3d,
        **(_amd_stage1_hints() if _is_hip and
           os.environ.get("VLLM_FP8_G32_AMD_HINTS", "0") == "1" else {}),
    )

    # Stage-2 reduce (default to vectorized reduce_segments; matches fp4_g32 V3).
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
