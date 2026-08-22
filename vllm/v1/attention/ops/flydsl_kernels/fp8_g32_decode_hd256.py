# SPDX-License-Identifier: Apache-2.0
# fp8_g32 decode (FlyDSL) — MI355X / gfx950 / CDNA4 — HEAD_SIZE=256 variant
# ============================================================================
# HEAD_SIZE=256 sibling of ``fp8_g32_decode_v4.py`` (which handles 128-wide
# heads). Structurally this is the fp8_g32 group-quant dequant grafted onto the
# proven HEAD_SIZE=256 framework of ``tq_decode_hd256.py``: every 128-specific
# lane/byte constant is generalized to HEAD_SIZE via the per-lane data-movement
# geometry constants below, and each token's now-128-byte K/V code payload is
# loaded in ``HALVES``=2 x 16-byte buffer_load issues (AMD dwordx4 max).
#
# Dispatched only when D == 256. Paths (selected via build() flags):
#   * QK: qk_scaled native scaled FP4xE4M3 MFMA (2x mfma_scale_f32_16x16x128_
#         f8f6f4, K=128/issue) when QK_SCALED — the scale is applied by the
#         instruction, no post-MFMA fold; else qk_fp8 native fp8 MFMA (8x
#         mfma_f32_16x16x32_fp8_fp8) when QK_FP8; else bf16 wide-K MFMA
#         (mfma_f32_16x16x32_bf16); QK_K_CHUNKS=8.
#   * V : native cvt_scalef32_pk_bf16_fp4 dequant (V_CVT) into row-major V LDS
#         read back via ds_read_tr16_b64 HW transpose (USE_HW_TR); else software
#         FP4-centroid LUT dequant.
#   * Q : pre-rotated q_rot load (STEP B).
# qk_scaled (ported from the GQA-6 sibling) is the fewest-issue QK path: 2 K=128
# scaled MFMAs vs qk_fp8's 8 K=32 issues + 32 post-MFMA scale FMAs. The KV LDS
# tile is bank-conflict padded (see KV_ROW_*/KFP4_ROW_I32 below) so those native
# MFMA reads run unstalled. Correctness is bit-identical to the bf16-QK reference.
#
# fp8_g32 AoS slot layout (per (slot, head), D=256, group_size=32 => 8 groups):
#   [0        : 128)  K FP4 codes    (KEY_CODE_BYTES = 128, 2 nibbles/byte)
#   [128      : 136)  K UE8M0 scales (N_GROUPS = 8 bytes)
#   [136      : 264)  V FP4 codes    (VAL_CODE_BYTES = 128)
#   [264      : 272)  V UE8M0 scales (N_GROUPS = 8 bytes)
#
# MFMA operand layouts (unchanged from the 128 kernel):
#   QK:  mfma(A=K, B=Q, C=qk_acc)    ->  C[m=token, n=query],  4 fp32/lane
#   PV:  mfma(A=V_T, B=P, C=acc_pv)  ->  C[m=head_dim, n=query], 4 fp32/lane

from __future__ import annotations

import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl, vector
from flydsl.expr.typing import T, Int32
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf as _scf


# === Constants (256-wide-head fp8_g32 decode profile) =======================
HEAD_SIZE = 256
KV_BLOCK_SIZE = 16          # default; overridable via build(kv_block_size=...)
TILE_SIZE = 16              # MFMA tile = 16 tokens
N_CENTROIDS = 16            # FP4 E2M1 value table (16 entries indexed by nibble)
QUERY_GROUP_SIZE = 16       # default for tests; overridable via build(...)
WARP_SIZE = 64
NUM_WARPS = 1
BLOCK_THREADS = NUM_WARPS * WARP_SIZE
KV_COMPUTE_BLOCK = 256                      # 16 K-tiles × 16 tokens

# fp8_g32 AoS slot layout (per (slot, head), D=256, group_size=32 => 8 groups)
FP8_GROUP_SIZE = 32
N_GROUPS = HEAD_SIZE // FP8_GROUP_SIZE       # 8 UE8M0 scale bytes per head
UE8M0_BIAS = 127
KEY_CODE_BYTES = HEAD_SIZE // 2              # 128
VAL_CODE_BYTES = HEAD_SIZE // 2              # 128
K_SCALES_OFFSET = KEY_CODE_BYTES                         # 128
V_CODES_OFFSET = KEY_CODE_BYTES + N_GROUPS               # 136
V_SCALES_OFFSET = V_CODES_OFFSET + VAL_CODE_BYTES        # 264
SLOT_CONTENT_BYTES = V_SCALES_OFFSET + N_GROUPS          # 272

# MFMA
MFMA_M = MFMA_N = 16
MFMA_K_BF16_QK = 32                         # CDNA4 wide-K (mfma_f32_16x16x32_bf16)
MFMA_K_BF16_PV = 16                         # PV's K = token tile = 16
QK_K_CHUNKS = HEAD_SIZE // MFMA_K_BF16_QK   # 8 for HEAD_SIZE=256
PV_N_CHUNKS = HEAD_SIZE // MFMA_N           # 16 for HEAD_SIZE=256

# --- Per-lane data-movement geometry (generalized from the 128 kernel) ------
# The K/V dequant lane assignment splits each token's packed code bytes across
# the 4 ``chunk_in_tok`` sub-lanes (lane % 4). Each lane processes
# ``KEY_CODE_BYTES // 4`` code bytes = 32 B at 256. buffer_load maxes at 16 B
# (dwordx4) so a lane issues ``HALVES``=2 loads; each 16-B half covers exactly
# one UE8M0 group of 32 head-dims.
#   HEAD_SIZE=128: LANE_CODE_BYTES=16, HALVES=1, SUBCHUNK_HDIMS=32
#   HEAD_SIZE=256: LANE_CODE_BYTES=32, HALVES=2, SUBCHUNK_HDIMS=64
LANE_CODE_BYTES = KEY_CODE_BYTES // 4       # 32 code bytes per lane per K-tile
HALVES = LANE_CODE_BYTES // 16              # 2 for HEAD_SIZE=256
HALF_I64 = 8                                # i64 slots written per 16-B half
HALF_HDIMS = 32                             # head-dims per half (= one group)
SUBCHUNK_HDIMS = HEAD_SIZE // 4             # head-dims per chunk_in_tok sub-lane (64)
I64_PER_TOKEN = HEAD_SIZE // 4              # i64 slots per token row in KV LDS (64)
SUBCHUNK_I64 = I64_PER_TOKEN // 4           # i64 offset stride per chunk_in_tok (16)
# Q row-major LDS load geometry (STEP B): each lane loads 8 bf16 (=4 i32),
# so a row spans ``HEAD_SIZE // 8`` column-groups.
Q_GROUPS_PER_ROW = HEAD_SIZE // 8           # 32 for 256
Q_ROW_SHIFT = Q_GROUPS_PER_ROW.bit_length() - 1   # 5 for 256
Q_COL_MASK = Q_GROUPS_PER_ROW - 1           # 31 for 256

# LDS regions
CENTROID_LDS_BYTES = N_CENTROIDS * 4                       # 64
Q_LDS_BYTES = QUERY_GROUP_SIZE * HEAD_SIZE * 2             # 8192 for 256
KV_TILE_LDS_BYTES = TILE_SIZE * HEAD_SIZE * 2             # 8192 for 256 (unpadded ref)

# --- KV-tile LDS bank-conflict padding (mirrors tq_decode_hd256) ------------
# The KV tile is laid out [token][head_dim] in LDS and time-multiplexed: K for
# QK, then V for PV. With an unpadded row stride EVERY token-row starts on the
# same LDS banks, so the MFMA operand reads (lane t -> row=token) all hit the
# same banks -> a 16-way bank conflict (rocprof measured ~64% of the decode
# kernel's LDS cycles on the bf16 layout in tq_decode_hd256). Padding each
# token row shifts consecutive rows onto different banks. Two strides are
# needed because qk_fp8 stores K as E4M3 (1 B/elem) while V is bf16 (2 B/elem):
#   * V (and the bf16-K fallback): pad 8 bf16 (16 B) -> 264 elems / 528 B
#     (row bank offset (264/2)%32 = 4 -> 16-way becomes 2-way).
#   * qk_fp8 E4M3-K: HEAD_SIZE bytes = 256 B = exact banks; pad 2 i64 (16 B)
#     -> 34 i64 / 272 B (dword stride 68, %32 = 4 -> 16-way becomes 2-way).
# Both pads keep ds_read/ds_write_b64 (and b128) alignment. This is a pure
# addressing change: within-token head-dim offsets and every dequant/MFMA
# operand are bit-identical to the unpadded kernel.
KV_ROW_PAD_ELEMS = 8                                       # bf16 padding / token row
KV_ROW_ELEMS = HEAD_SIZE + KV_ROW_PAD_ELEMS               # 264 (V + bf16-K rows)
KV_ROW_I64 = KV_ROW_ELEMS // 4                            # 66
KV_ROW_BYTES = KV_ROW_ELEMS * 2                           # 528
KFP8_ROW_I64 = HEAD_SIZE // 8 + 2                         # 34 (qk_fp8 E4M3 K row)
#   * qk_scaled FP4-K: HEAD_SIZE//8 = 32 i32 = 128 B = exact banks; pad 4 i32
#     (16 B) -> 36 i32 / 144 B (row bank offset 36 % 32 = 4 -> 16-way -> 2-way).
KFP4_ROW_I32 = HEAD_SIZE // 8 + 4                          # 36 (qk_scaled FP4 K row)
KV_TILE_LDS_BYTES_PADDED = TILE_SIZE * KV_ROW_BYTES       # 8448 (V dominates)
# qk_fp8 only: per-(token, group) K UE8M0 scale staged for the post-MFMA fold.
SCALE_LDS_BYTES = TILE_SIZE * N_GROUPS * 4                 # 16*8*4 = 512 for 256

LOG2E = 1.4426950408889634
NEG_INF_VAL = float("-inf")


def _vsplat_mul(vec, scalar):
    s = scalar.ir_value() if hasattr(scalar, 'ir_value') else scalar
    return vec * vector.broadcast(T.f32x4, s)


allocator = None


def build_fp8_g32_decode_hd256_module(
    num_seqs: int,
    num_kv_heads: int,
    num_partitions: int,
    padded_slot: int,
    max_blocks_per_seq: int = 512,
    softmax_scale: float | None = None,
    query_group_size: int = QUERY_GROUP_SIZE,
    kv_block_size: int = KV_BLOCK_SIZE,
    use_hw_v_transpose: bool = False,
    tile_groups_per_partition: int = 1,
    use_wht_butterfly: bool = False,
    arch_b: bool = False,
    scale_c: float = 0.156,
    qk_fp8: bool = False,
    qk_scaled: bool = False,
    v_cvt: bool = False,
    q_hoist: bool = False,
    num_warps: int = 1,
):
    """Build an fp8_g32 HEAD_SIZE=256 decode kernel module (base path only).

    ``padded_slot`` is the per-(token, head) slot stride in bytes of the
    fp8_g32 KV cache (= ``kv_cache.shape[3]``, >= 272 for D=256/group=32).
    The cache is ``[num_blocks, block_size, num_kv_heads, padded_slot]`` uint8
    with the AoS byte layout documented at the top of this file.

    ``arch_b`` (default False): when True the dequant multiplies the UE8M0
    scale by the extra constant ``scale_c`` (the c-baked-codebook recipe).

    QK path selection (mutually exclusive): ``qk_fp8`` uses 8 K=32
    ``mfma_f32_16x16x32_fp8_fp8`` issues with the UE8M0 scale folded in
    afterwards; ``qk_scaled`` uses 2 K=128 ``mfma_scale_f32_16x16x128_f8f6f4``
    issues with the scale applied by the instruction; with neither set the
    canonical bf16 wide-K MFMA path runs. ``v_cvt`` + ``use_hw_v_transpose``
    (native FP4->bf16 V convert via HW transpose) are also implemented and used
    by the production D=256 config. ``q_hoist`` only affects ``qk_scaled``
    (loop-invariant Q operands); ``use_wht_butterfly`` is ignored at 256.
    """
    assert query_group_size in (8, 16), (
        f"query_group_size must be 8 or 16; got {query_group_size}"
    )
    # head_dim=256 hybrid models (Qwen3.6/3.8) force a larger mamba-aligned KV
    # block; the tiled inner loop (_TILES_PER_BLOCK = kv_block_size // TILE_SIZE)
    # handles any multiple of TILE_SIZE, so accept 16/32/64/128/256 like TQ v4.
    assert kv_block_size in (16, 32, 64, 128, 256), (
        f"kv_block_size must be 16/32/64/128/256; got {kv_block_size}"
    )
    assert kv_block_size % TILE_SIZE == 0
    assert int(tile_groups_per_partition) >= 1

    TGPP = int(tile_groups_per_partition)
    PARTITION_EXTENT_TOKENS = TGPP * KV_COMPUTE_BLOCK
    ARCH_B = bool(arch_b)
    SCALE_C = float(scale_c)
    # --- Fast paths (ported from the 128 kernel to 256-wide heads) ----------
    # USE_HW_TR : row-major V LDS + ds_read_tr16_b64 HW transpose (the primary
    #             bf16-beating win; identical mechanism to tq_decode_hd256).
    # QK_FP8    : native fp8 QK MFMA (8× mfma_f32_16x16x32_fp8_fp8, one per
    #             UE8M0 group of 32 head-dims) + per-token post-MFMA scale fold.
    # QK_SCALED : native scaled FP4xE4M3 MFMA. mfma_scale_f32_16x16x128_f8f6f4
    #             contracts a fixed K=128, so a 256-wide head is covered by
    #             MFMA_ISSUES=2 back-to-back issues chained through the f32x4
    #             accumulator, each consuming a 128-dim slice (4 UE8M0 groups)
    #             with the UE8M0 scale applied by the instruction. Replaces
    #             qk_fp8's 8 issues + 32 post-MFMA scale FMAs (ported from the
    #             GQA-6 sibling). q_hoist builds the loop-invariant Q operands.
    # V_CVT     : native cvt_scalef32_pk_bf16_fp4 V dequant (requires USE_HW_TR).
    USE_HW_TR = bool(use_hw_v_transpose)
    QK_FP8 = bool(qk_fp8)
    V_CVT = bool(v_cvt)
    QK_SCALED = bool(qk_scaled)
    Q_HOIST = bool(q_hoist)
    MFMA_SCALED_K = 128
    MFMA_ISSUES = HEAD_SIZE // MFMA_SCALED_K          # 2 for HEAD_SIZE=256
    GRPS_PER_ISSUE = MFMA_SCALED_K // FP8_GROUP_SIZE  # 4
    assert not (QK_FP8 and QK_SCALED), (
        "qk_fp8 and qk_scaled are mutually exclusive QK MFMA paths"
    )
    assert not (QK_SCALED and HEAD_SIZE % MFMA_SCALED_K != 0), (
        f"qk_scaled needs HEAD_SIZE % {MFMA_SCALED_K} == 0; got {HEAD_SIZE}"
    )
    assert not (V_CVT and not USE_HW_TR), (
        "v_cvt requires use_hw_v_transpose at HEAD_SIZE=256"
    )

    QG = int(query_group_size)
    QG_LOAD_ITERS = (QG * Q_GROUPS_PER_ROW) // WARP_SIZE   # 8 for QG=16, 4 for QG=8
    OOB_OFFSET = 0x7FFFFFF0

    # --- Occupancy: optional multi-warp (HIOCC) --------------------------------
    # num_warps=1 is the canonical single-wavefront path (bit-identical to the
    # historical kernel). num_warps>1 launches BLOCK_THREADS = num_warps*64 and
    # splits the partition's K-tile loop across warps with a cross-warp
    # online-softmax reduction in LDS (raises waves/CU from ~3). The body guards
    # every multi-warp-only code path on ``const_expr(_NUM_WARPS > 1)`` so the
    # nw=1 emission is unchanged.
    _NUM_WARPS = int(num_warps)
    assert _NUM_WARPS in (1, 2, 4), f"num_warps must be 1/2/4; got {_NUM_WARPS}"
    _BLOCK_THREADS = _NUM_WARPS * WARP_SIZE

    # --- Occupancy: size the Q LDS to the ACTUAL query group -------------------
    # The module-level Q_LDS_BYTES is sized for the QG=16 test default (8192 B),
    # but STEP B only writes rows [0, QG) and STEP C/F only read those rows. For
    # QG=8 the upper 4 KB is allocated-but-never-touched, so shrinking it to
    # QG*HEAD_SIZE*2 is bitwise identical and frees LDS/workgroup (17 KB -> 13 KB
    # at QG=8), lifting waves/CU (~3 -> ~5) since the kernel is LDS-occupancy
    # bound. The launcher pads segm pools to QG=16 regardless, so this is purely
    # an in-kernel LDS footprint reduction.
    _Q_LDS_BYTES = QG * HEAD_SIZE * 2

    _BS = int(kv_block_size)
    _TILES_PER_BLOCK = _BS // TILE_SIZE
    _PADDED_SLOT = int(padded_slot)
    assert _PADDED_SLOT >= SLOT_CONTENT_BYTES, (
        f"padded_slot={_PADDED_SLOT} < {SLOT_CONTENT_BYTES} (fp8_g32 D=256 content)"
    )
    assert _PADDED_SLOT % 4 == 0, (
        f"padded_slot={_PADDED_SLOT} must be 4-byte aligned"
    )

    global allocator
    arch = get_hip_arch()

    if softmax_scale is None:
        softmax_scale = 1.0 / (HEAD_SIZE ** 0.5)
    _qk_scale = float(softmax_scale)

    # --- Strides ---
    _Hq = num_kv_heads * QG
    _stride_q_seq = _Hq * HEAD_SIZE
    _stride_q_head = HEAD_SIZE
    _stride_bt_seq = max_blocks_per_seq

    # fp8_g32 AoS cache: [num_blocks, block_size, num_kv_heads, padded_slot].
    _stride_cache_head = _PADDED_SLOT
    _stride_cache_pos = num_kv_heads * _PADDED_SLOT
    _stride_cache_block = _BS * _stride_cache_pos

    _stride_out_part = QG * HEAD_SIZE
    _stride_out_head = num_partitions * QG * HEAD_SIZE
    _stride_out_seq = num_kv_heads * num_partitions * QG * HEAD_SIZE
    _stride_es_seq = num_kv_heads * num_partitions * QG
    _stride_ml_seq = _stride_es_seq

    # --- LDS layout ---
    allocator = SmemAllocator(None, arch=arch, global_sym_name="fp8_g32_hd256_smem")
    centroid_off = 0
    allocator.ptr = CENTROID_LDS_BYTES
    q_off = allocator.ptr
    allocator.ptr += _Q_LDS_BYTES
    kv_off = allocator.ptr
    allocator.ptr += KV_TILE_LDS_BYTES_PADDED
    scale_off = allocator.ptr
    if QK_FP8 or QK_SCALED:
        allocator.ptr += SCALE_LDS_BYTES

    @flyc.kernel
    def fp8_g32_decode_hd256_kernel(
        out_ptr: fx.Tensor,
        exp_sums_ptr: fx.Tensor,
        max_logits_ptr: fx.Tensor,
        query_ptr: fx.Tensor,
        kv_cache_ptr: fx.Tensor,
        centroids_ptr: fx.Tensor,
        block_tables_ptr: fx.Tensor,
        seq_lens_ptr: fx.Tensor,
    ):
        # ---- IDs ---------------------------------------------------------
        tid = gpu.thread_idx.x
        seq = gpu.block_idx.x
        kv_h = gpu.block_idx.y
        part = gpu.block_idx.z
        # For nw=1, warp_id is a compile-time 0 and lane == tid (bit-identical
        # to the historical single-warp kernel). For nw>1 each warp owns a
        # disjoint slice of the K-tile loop and reduces across warps in LDS.
        if const_expr(_NUM_WARPS > 1):
            warp_id = tid >> fx.Int32(6)                # 0.._NUM_WARPS-1
            lane = tid & fx.Int32(63)                   # 0..63 within warp
        else:
            warp_id = fx.Int32(0)
            lane = tid                                  # 0..63
        mfma_row = lane & fx.Int32(15)
        mfma_col_grp = lane >> fx.Int32(4)              # 0..3, K-group dim

        # ---- Buffer resources -------------------------------------------
        q_rsrc = buffer_ops.create_buffer_resource(query_ptr, max_size=True)
        bt_rsrc = buffer_ops.create_buffer_resource(block_tables_ptr, max_size=True)
        sl_rsrc = buffer_ops.create_buffer_resource(seq_lens_ptr, max_size=True)
        cent_rsrc = buffer_ops.create_buffer_resource(centroids_ptr, max_size=True)
        out_rsrc = buffer_ops.create_buffer_resource(out_ptr, max_size=True)
        es_rsrc = buffer_ops.create_buffer_resource(exp_sums_ptr, max_size=True)
        ml_rsrc = buffer_ops.create_buffer_resource(max_logits_ptr, max_size=True)

        # ---- LDS pointers -----------------------------------------------
        base = allocator.get_base()
        cent_lds = SmemPtr(base, centroid_off, T.f32, shape=(N_CENTROIDS,))
        q_lds_i32 = SmemPtr(base, q_off, T.i32, shape=(_Q_LDS_BYTES // 4,)).get()
        q_lds_i64 = SmemPtr(base, q_off, T.i64, shape=(_Q_LDS_BYTES // 8,)).get()
        kv_lds_i32 = SmemPtr(base, kv_off, T.i32, shape=(KV_TILE_LDS_BYTES_PADDED // 4,)).get()
        kv_lds_i64 = SmemPtr(base, kv_off, T.i64, shape=(KV_TILE_LDS_BYTES_PADDED // 8,)).get()
        kv_lds_i16 = SmemPtr(base, kv_off, T.i16, shape=(KV_TILE_LDS_BYTES_PADDED // 2,)).get()
        if const_expr(QK_FP8):
            scale_lds = SmemPtr(
                base, scale_off, T.f32, shape=(TILE_SIZE * N_GROUPS,)
            )
        if const_expr(QK_SCALED):
            # qk_scaled stages raw UE8M0 bytes (0..255) for the scaleA word.
            scale_lds_i32 = SmemPtr(
                base, scale_off, T.i32, shape=(TILE_SIZE * N_GROUPS,)
            )

        # ---- Constants ---------------------------------------------------
        c_sq = fx.Int32(_stride_q_seq)
        c_qh = fx.Int32(_stride_q_head)
        c_qg = fx.Int32(QG)
        c_bt = fx.Int32(_stride_bt_seq)
        c_block = fx.Int32(_stride_cache_block)
        c_stride_pos = fx.Int32(_stride_cache_pos)
        c_stride_head = fx.Int32(_stride_cache_head)
        c_kscale_off = fx.Int32(K_SCALES_OFFSET)       # 128
        c_vcode_off = fx.Int32(V_CODES_OFFSET)         # 136
        c_vscale_off = fx.Int32(V_SCALES_OFFSET)       # 264
        c_w = fx.Int32(WARP_SIZE)

        NEG_INF = arith.constant(NEG_INF_VAL, type=T.f32)
        ZERO_F = fx.Float32(0.0)
        ONE_F = fx.Float32(1.0)
        LOG2E_C = arith.constant(LOG2E, type=T.f32)
        QK_SCALE = arith.constant(_qk_scale, type=T.f32)

        def _ival(v):
            return v.ir_value() if hasattr(v, 'ir_value') else v

        if const_expr(QK_FP8 or QK_SCALED):
            c_zero_i32 = arith.constant(0, type=T.i32)

            def _f32x8_to_fp8_i64(f):
                # 8 f32 -> 8 E4M3 bytes packed as i64 (MFMA fp8 operand).
                w0 = rocdl.cvt_pk_fp8_f32(T.i32, f[0], f[1], c_zero_i32, 0)
                w0 = rocdl.cvt_pk_fp8_f32(T.i32, f[2], f[3], w0, 1)
                w1 = rocdl.cvt_pk_fp8_f32(T.i32, f[4], f[5], c_zero_i32, 0)
                w1 = rocdl.cvt_pk_fp8_f32(T.i32, f[6], f[7], w1, 1)
                pv = vector.from_elements(T.vec(2, T.i32), [w0, w1])
                return vector.extract(
                    vector.bitcast(T.vec(1, T.i64), pv), static_position=[0]
                )

            def _bf16x8_to_fp8_i64(v_bf16):
                f = [
                    arith.extf(T.f32, vector.extract(v_bf16, static_position=[i]))
                    for i in range(8)
                ]
                return _f32x8_to_fp8_i64(f)

        # ===== STEP A: Load centroids → LDS (cooperative, race-safe) =====
        c_idx_safe = lane & fx.Int32(N_CENTROIDS - 1)
        c_val = buffer_ops.buffer_load(
            cent_rsrc, c_idx_safe, vec_width=1, dtype=T.f32
        )
        cent_lds.store(c_val, [arith.index_cast(T.index, c_idx_safe)])
        gpu.barrier()

        # ===== STEP B: Load pre-rotated q_rot → row-major Q LDS ==========
        for c in range_constexpr(QG_LOAD_ITERS):
            row_chunk = lane + fx.Int32(c * WARP_SIZE)
            row = row_chunk >> fx.Int32(Q_ROW_SHIFT)       # 0..QG-1
            col_b = row_chunk & fx.Int32(Q_COL_MASK)       # 0..Q_GROUPS_PER_ROW-1
            col_elem = col_b * fx.Int32(8)                 # bf16 elem
            q_off_elem = (
                seq * c_sq
                + (kv_h * c_qg + row) * c_qh
                + col_elem
            )
            q_v = buffer_ops.buffer_load(
                q_rsrc, q_off_elem // fx.Int32(2),
                vec_width=4, dtype=T.i32,
            )
            q_lds_byte = row * fx.Int32(HEAD_SIZE * 2) + col_elem * fx.Int32(2)
            vector.store(
                q_v, q_lds_i32,
                [arith.index_cast(T.index, q_lds_byte // fx.Int32(4))],
            )
        gpu.barrier()

        # ===== STEP C: Pre-load Q operands for QK_K_CHUNKS K-chunks ======
        q_chunks = []
        for chk in range_constexpr(QK_K_CHUNKS):
            q_idx_i64 = (
                mfma_row * fx.Int32(HEAD_SIZE * 2 // 8)
                + fx.Int32(chk * 8)
                + mfma_col_grp * fx.Int32(2)
            )
            qv = vector.load_op(
                T.vec(2, T.i64), q_lds_i64,
                [arith.index_cast(T.index, q_idx_i64)],
            )
            q_chunks.append(vector.bitcast(T.vec(8, T.bf16), qv))
        # qk_fp8: pre-convert each Q chunk (8 bf16) to an E4M3 i64 MFMA operand.
        # Launcher already E4M3-rounded Q, so bf16 -> E4M3 is exact.
        q_fp8_chunks = []
        if const_expr(QK_FP8):
            for chk in range_constexpr(QK_K_CHUNKS):
                q_fp8_chunks.append(_bf16x8_to_fp8_i64(q_chunks[chk]))

        # qk_scaled + Q_HOIST: build the loop-invariant scaled-MFMA Q operands
        # once (one per MFMA issue). For issue h, lane t holds all 32 head-dims
        # of group h*GRPS_PER_ISSUE + mfma_col_grp at query row mfma_row: 4
        # sub-chunks of 8 bf16, each repacked to E4M3 i64, assembled to vec8 i32.
        q_op_hoisted = []
        if const_expr(QK_SCALED and Q_HOIST):
            for h in range_constexpr(MFMA_ISSUES):
                _q_hoist_words = []
                for j in range_constexpr(4):
                    _q_hoist_idx = (
                        mfma_row * fx.Int32(HEAD_SIZE * 2 // 8)
                        + (fx.Int32(h * GRPS_PER_ISSUE) + mfma_col_grp)
                        * fx.Int32(8)
                        + fx.Int32(j * 2)
                    )
                    _q_hoist_v = vector.load_op(
                        T.vec(2, T.i64), q_lds_i64,
                        [arith.index_cast(T.index, _q_hoist_idx)],
                    )
                    _q_hoist_words.append(
                        _bf16x8_to_fp8_i64(
                            vector.bitcast(T.vec(8, T.bf16), _q_hoist_v))
                    )
                q_op_hoisted.append(
                    vector.bitcast(
                        T.vec(8, T.i32),
                        vector.from_elements(T.vec(4, T.i64), _q_hoist_words),
                    )
                )

        # ===== STEP D: Online softmax + PV state =========================
        running_max = NEG_INF
        running_sum = ZERO_F
        zero_v4 = arith.constant_vector(0.0, T.f32x4)
        acc_pv = [zero_v4 for _ in range(PV_N_CHUNKS)]

        # ===== STEP E: Sequence-len + partition base ====================
        seq_len = buffer_ops.buffer_load(sl_rsrc, seq, vec_width=1, dtype=T.i32)
        partition_base = part * fx.Int32(PARTITION_EXTENT_TOKENS)

        # Per-K-tile dequant lane assignment: lane t -> token = t/4,
        # chunk_in_tok = t%4 (each chunk = SUBCHUNK_HDIMS=64 head-dims = 2 groups).
        tok_in_tile = lane >> fx.Int32(2)
        chunk_in_tok = lane & fx.Int32(3)
        # Both UE8M0 group bytes for this lane's chunk (groups 2c, 2c+1) live in
        # the SAME 4-byte scale word at offset (chunk_in_tok>>1)*4 within the
        # 8-byte scale region. Load once; extract per-half below.
        c_scale_word_off = (chunk_in_tok >> fx.Int32(1)) * fx.Int32(4)

        # ===== STEP F: K-tile loop =======================================
        c_kcb = fx.Int32(KV_COMPUTE_BLOCK)
        c_tgpp = fx.Int32(TGPP)
        c_zero_i32 = fx.Int32(0)
        c_one_i32 = fx.Int32(1)
        remaining = seq_len - partition_base
        in_range = remaining > c_zero_i32
        trip_raw = (remaining + c_kcb - c_one_i32) // c_kcb
        trip_clamped = (trip_raw > c_tgpp).select(c_tgpp, trip_raw)
        trip_or_zero = in_range.select(trip_clamped, c_zero_i32)
        c_zero_idx = arith.constant(0, index=True)
        c_one_idx = arith.constant(1, index=True)
        trip_idx = arith.index_cast(
            T.index,
            trip_or_zero.ir_value()
            if hasattr(trip_or_zero, 'ir_value') else trip_or_zero,
        )
        _init_iter = [
            _ival(running_max),
            _ival(running_sum),
            *[_ival(p) for p in acc_pv],
        ]
        _for_op = _scf.ForOp(
            c_zero_idx, trip_idx, c_one_idx, _init_iter,
        )
        _for_ip = ir.InsertionPoint(_for_op.body)
        _for_ip.__enter__()
        try:
            tg_idx = _for_op.induction_variable
            tg_i32 = fx.Int32(arith.index_cast(T.i32, tg_idx))
            partition_start = partition_base + tg_i32 * c_kcb
            bt_seq_base = (
                seq * c_bt + (partition_start // fx.Int32(_BS))
            )
            running_max = _for_op.inner_iter_args[0]
            running_sum = _for_op.inner_iter_args[1]
            acc_pv = list(_for_op.inner_iter_args[2:])
            for n_tile in range_constexpr(16):
                block_in_part = n_tile // _TILES_PER_BLOCK
                tile_in_block = n_tile % _TILES_PER_BLOCK
                tile_start_tok = (
                    partition_start + fx.Int32(n_tile * TILE_SIZE)
                )
                tile_in_seq = tile_start_tok < seq_len
                bt_off = bt_seq_base + fx.Int32(block_in_part)
                bt_off_safe = tile_in_seq.select(bt_off, seq * c_bt)
                phys_block = buffer_ops.buffer_load(
                    bt_rsrc, bt_off_safe,
                    vec_width=1, dtype=T.i32,
                )
                # A buffer descriptor addresses with a 32-bit voffset, so
                # ``phys_block * c_block`` wraps once the cache view exceeds
                # 4 GiB. Fold the block base into the descriptor's 64-bit
                # base pointer and keep only in-block offsets below.
                blk_off_i64 = (
                    arith.extui(T.i64, phys_block) * fx.Int64(_stride_cache_block)
                )
                blk_rsrc = buffer_ops.create_block_buffer_resource(
                    kv_cache_ptr, blk_off_i64
                )
                block_base = fx.Int32(0)

                slot = fx.Int32(tile_in_block * TILE_SIZE) + tok_in_tile
                slot_base_byte = (
                    block_base
                    + slot * c_stride_pos
                    + kv_h * c_stride_head
                )
                # K codes: LANE_CODE_BYTES (=32) at slot_base + chunk*32, in
                # HALVES x 16-byte buffer_loads.
                k_byte0 = slot_base_byte + chunk_in_tok * fx.Int32(LANE_CODE_BYTES)
                k_packed_list = []
                for hf in range_constexpr(HALVES):
                    k_packed_list.append(
                        buffer_ops.buffer_load(
                            blk_rsrc, (k_byte0 + fx.Int32(hf * 16)) // fx.Int32(4),
                            vec_width=4, dtype=T.i32,
                        )
                    )

                # ---- HOISTED: V codes + K/V scale HBM loads early ---------
                v_byte0 = (
                    slot_base_byte + c_vcode_off
                    + chunk_in_tok * fx.Int32(LANE_CODE_BYTES)
                )
                v_packed_list = []
                for hf in range_constexpr(HALVES):
                    v_packed_list.append(
                        buffer_ops.buffer_load(
                            blk_rsrc, (v_byte0 + fx.Int32(hf * 16)) // fx.Int32(4),
                            vec_width=4, dtype=T.i32,
                        )
                    )
                # UE8M0 group scale words (one 4-byte word covers this lane's
                # two groups). scale = 2^(byte-127) = bitcast_f32(byte<<23);
                # byte==0 -> +0.0 (zero sentinel).
                kscale_word = buffer_ops.buffer_load(
                    blk_rsrc,
                    (slot_base_byte + c_kscale_off + c_scale_word_off) // fx.Int32(4),
                    vec_width=1, dtype=T.i32,
                )
                vscale_word = buffer_ops.buffer_load(
                    blk_rsrc,
                    (slot_base_byte + c_vscale_off + c_scale_word_off) // fx.Int32(4),
                    vec_width=1, dtype=T.i32,
                )
                if ARCH_B:
                    c_arch_b = arith.constant(SCALE_C, type=T.f32)

                if const_expr(QK_FP8):
                    # qk_fp8 K dequant → LDS [token, head_dim] as E4M3 BYTES
                    # (1 byte/elem), UNSCALED (raw FP4 grid value). The UE8M0
                    # group scale is staged to scale_lds and folded post-MFMA.
                    # Lane owns head_dims chunk_in_tok*64..+63 (2 groups). Each
                    # half hf (32 head-dims) = 4× i64 at i64 index
                    # tok*(HEAD_SIZE//8) + chunk_in_tok*8 + hf*4 + w.
                    for hf in range_constexpr(HALVES):
                        k_packed = k_packed_list[hf]
                        grp_shift = (
                            (chunk_in_tok * fx.Int32(2) + fx.Int32(hf)) & fx.Int32(3)
                        ) * fx.Int32(8)
                        kscale_byte = (kscale_word >> grp_shift) & fx.Int32(0xFF)
                        kscale_f32 = arith.bitcast(
                            T.f32, _ival(kscale_byte << fx.Int32(23)))
                        if ARCH_B:
                            c_arch_b = arith.constant(SCALE_C, type=T.f32)
                            kscale_f32 = kscale_f32 * c_arch_b
                        for w in range_constexpr(4):
                            word_i32 = vector.extract(k_packed, static_position=[w])
                            cents = []
                            for n in range_constexpr(8):
                                nibble = (word_i32 >> fx.Int32(n * 4)) & fx.Int32(0xF)
                                nibble_idx = arith.index_cast(T.index, nibble)
                                cents.append(cent_lds.load([nibble_idx]))
                            k_i64 = _f32x8_to_fp8_i64(cents)
                            k_i64_idx = (
                                tok_in_tile * fx.Int32(KFP8_ROW_I64)
                                + chunk_in_tok * fx.Int32(8)
                                + fx.Int32(hf * 4 + w)
                            )
                            vector.store(
                                vector.from_elements(T.vec(1, T.i64), [k_i64]),
                                kv_lds_i64,
                                [arith.index_cast(T.index, k_i64_idx)],
                            )
                        # Stage this lane's (token, group) UE8M0 K scale for the
                        # post-MFMA per-token fold. group = 2*chunk_in_tok + hf.
                        scale_lds.store(
                            kscale_f32,
                            [arith.index_cast(
                                T.index,
                                tok_in_tile * fx.Int32(N_GROUPS)
                                + chunk_in_tok * fx.Int32(2) + fx.Int32(hf),
                            )],
                        )
                elif const_expr(QK_SCALED):
                    # qk_scaled: store raw FP4 codes (natural contiguous order,
                    # 16 B = 32 nibbles per UE8M0 group) straight to KV LDS as
                    # the scaled-MFMA A operand — no dequant here. ARCH_B's
                    # SCALE_C cannot ride on the raw UE8M0 byte, so it is
                    # applied once post-MFMA instead.
                    for hf in range_constexpr(HALVES):
                        k_packed = k_packed_list[hf]
                        grp = chunk_in_tok * fx.Int32(2) + fx.Int32(hf)
                        grp_shift = (grp & fx.Int32(3)) * fx.Int32(8)
                        kscale_byte = (kscale_word >> grp_shift) & fx.Int32(0xFF)
                        vector.store(
                            k_packed, kv_lds_i32,
                            [arith.index_cast(
                                T.index,
                                tok_in_tile * fx.Int32(KFP4_ROW_I32)
                                + grp * fx.Int32(4),
                            )],
                        )
                        scale_lds_i32.store(
                            kscale_byte,
                            [arith.index_cast(
                                T.index,
                                tok_in_tile * fx.Int32(N_GROUPS) + grp,
                            )],
                        )
                else:
                    # K dequant → LDS [token, head_dim] (natural). HALVES halves
                    # of 32 head-dims each; half hf is UE8M0 group (2*chunk+hf).
                    tok_kreg = tok_in_tile * fx.Int32(I64_PER_TOKEN)
                    chunk_kreg = tok_kreg + chunk_in_tok * fx.Int32(SUBCHUNK_I64)
                    for hf in range_constexpr(HALVES):
                        k_packed = k_packed_list[hf]
                        half_i64 = fx.Int32(hf * HALF_I64)
                        grp_shift = (
                            (chunk_in_tok * fx.Int32(2) + fx.Int32(hf)) & fx.Int32(3)
                        ) * fx.Int32(8)
                        kscale_byte = (kscale_word >> grp_shift) & fx.Int32(0xFF)
                        kscale_f32 = arith.bitcast(
                            T.f32, _ival(kscale_byte << fx.Int32(23)))
                        if ARCH_B:
                            c_arch_b = arith.constant(SCALE_C, type=T.f32)
                            kscale_f32 = kscale_f32 * c_arch_b
                        for w in range_constexpr(4):
                            word_i32 = vector.extract(k_packed, static_position=[w])
                            bf16_elems = []
                            for n in range_constexpr(8):
                                nibble = (word_i32 >> fx.Int32(n * 4)) & fx.Int32(0xF)
                                nibble_idx = arith.index_cast(T.index, nibble)
                                cent_f32 = cent_lds.load([nibble_idx])
                                elem_bf16 = arith.trunc_f(T.bf16, cent_f32 * kscale_f32)
                                bf16_elems.append(elem_bf16)
                            v_bf16 = vector.from_elements(T.vec(8, T.bf16), bf16_elems)
                            v_i64 = vector.bitcast(T.vec(2, T.i64), v_bf16)
                            vector.store(
                                v_i64, kv_lds_i64,
                                [arith.index_cast(
                                    T.index, chunk_kreg + half_i64 + fx.Int32(w * 2))],
                            )
                gpu.barrier()

                if const_expr(QK_FP8):
                    # qk_fp8 QK: 8× mfma_f32_16x16x32_fp8_fp8 (K=32 == one
                    # UE8M0 group). A=K[token, head_dim] E4M3 (8 fp8/lane = 1
                    # i64 at i64 idx mfma_row*(HEAD_SIZE//8) + chk*4 + col_grp),
                    # B=Q E4M3. Each group's fp32 accumulator is multiplied by
                    # the per-token UE8M0 scale vec4 (lane holds 4 tokens
                    # col_grp*4..+3) then summed: scores = Σ_g scale_g·partial_g.
                    qk_acc = zero_v4
                    for chk in range_constexpr(QK_K_CHUNKS):
                        k_idx_i64 = (
                            mfma_row * fx.Int32(KFP8_ROW_I64)
                            + fx.Int32(chk * 4)
                            + mfma_col_grp
                        )
                        k_op = vector.extract(
                            vector.load_op(
                                T.vec(1, T.i64), kv_lds_i64,
                                [arith.index_cast(T.index, k_idx_i64)],
                            ),
                            static_position=[0],
                        )
                        grp_acc = rocdl.mfma_f32_16x16x32_fp8_fp8(
                            T.f32x4, [k_op, q_fp8_chunks[chk], zero_v4, 0, 0, 0]
                        )
                        for elem in range_constexpr(4):
                            s_idx = (
                                (mfma_col_grp * fx.Int32(4) + fx.Int32(elem))
                                * fx.Int32(N_GROUPS)
                                + fx.Int32(chk)
                            )
                            se = scale_lds.load([arith.index_cast(T.index, s_idx)])
                            pe = vector.extract(grp_acc, static_position=[elem])
                            cur = vector.extract(qk_acc, static_position=[elem])
                            qk_acc = vector.insert(
                                cur + pe * se, qk_acc,
                                static_position=[elem], dynamic_position=[],
                            )
                elif const_expr(QK_SCALED):
                    # qk_scaled QK: MFMA_ISSUES native scaled MFMAs over K=128,
                    # chained through the accumulator.
                    #   A = K raw FP4 codes (vec4 i32 = 16 B) per (token, group)
                    #   B = Q E4M3 (vec8 i32 = 32 B) for the same group
                    #   scaleA = per-(token,group) UE8M0 byte at op_sel byte 0
                    #   scaleB = 0x7F identity (Q carries no scale)
                    BLGP_E2M1 = 4   # A operand type code = fp4 e2m1
                    CBSZ_E4M3 = 0   # B operand type code = fp8 e4m3
                    IDENT = fx.Int32(0x7F)
                    qk_acc = zero_v4
                    for h in range_constexpr(MFMA_ISSUES):
                        grp = fx.Int32(h * GRPS_PER_ISSUE) + mfma_col_grp
                        k_op = vector.load_op(
                            T.vec(4, T.i32), kv_lds_i32,
                            [arith.index_cast(
                                T.index,
                                mfma_row * fx.Int32(KFP4_ROW_I32)
                                + grp * fx.Int32(4),
                            )],
                        )
                        if const_expr(Q_HOIST):
                            q_op = q_op_hoisted[h]
                        else:
                            q_fp8_words = []
                            for j in range_constexpr(4):
                                qv = vector.load_op(
                                    T.vec(2, T.i64), q_lds_i64,
                                    [arith.index_cast(
                                        T.index,
                                        mfma_row * fx.Int32(HEAD_SIZE * 2 // 8)
                                        + grp * fx.Int32(8)
                                        + fx.Int32(j * 2),
                                    )],
                                )
                                q_fp8_words.append(
                                    _bf16x8_to_fp8_i64(
                                        vector.bitcast(T.vec(8, T.bf16), qv))
                                )
                            q_op = vector.bitcast(
                                T.vec(8, T.i32),
                                vector.from_elements(
                                    T.vec(4, T.i64), q_fp8_words),
                            )
                        scbyte = fx.Int32(scale_lds_i32.load(
                            [arith.index_cast(
                                T.index,
                                mfma_row * fx.Int32(N_GROUPS) + grp)]
                        ))
                        kscale = fx.Int32(0x7F7F7F00) | scbyte
                        qk_acc = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                            T.vec(4, T.f32),
                            [k_op, q_op, qk_acc,
                             BLGP_E2M1, CBSZ_E4M3, 0, kscale, 0, IDENT],
                        )
                    if ARCH_B:
                        qk_acc = _vsplat_mul(
                            qk_acc, arith.constant(SCALE_C, type=T.f32))
                else:
                    # ---- QK MFMA (CDNA4 wide-K): A=K[token, head_dim], B=Q ----
                    qk_acc = zero_v4
                    for chk in range_constexpr(QK_K_CHUNKS):
                        k_idx_i64 = (
                            mfma_row * fx.Int32(HEAD_SIZE * 2 // 8)
                            + fx.Int32(chk * 8)
                            + mfma_col_grp * fx.Int32(2)
                        )
                        kv_load = vector.load_op(
                            T.vec(2, T.i64), kv_lds_i64,
                            [arith.index_cast(T.index, k_idx_i64)],
                        )
                        k_op = vector.bitcast(T.vec(8, T.bf16), kv_load)
                        qk_acc = rocdl.mfma_f32_16x16x32_bf16(
                            T.f32x4, [k_op, q_chunks[chk], qk_acc, 0, 0, 0]
                        )

                # Scale + mask out-of-context tokens.
                qk_acc = _vsplat_mul(qk_acc, QK_SCALE)
                for elem in range_constexpr(4):
                    kv_tok = (
                        partition_start
                        + fx.Int32(n_tile * TILE_SIZE)
                        + mfma_col_grp * fx.Int32(4)
                        + fx.Int32(elem)
                    )
                    in_b = kv_tok < seq_len
                    v = vector.extract(qk_acc, static_position=[elem])
                    qk_acc = vector.insert(
                        in_b.select(v, NEG_INF), qk_acc,
                        static_position=[elem], dynamic_position=[],
                    )

                # FA2 online softmax: per-query-row reduce.
                local_max = vector.reduction(T.f32, "maxnumf", qk_acc)
                r1 = local_max.shuffle_xor(fx.Int32(16), c_w)
                local_max = local_max.maximumf(r1)
                r2 = local_max.shuffle_xor(fx.Int32(32), c_w)
                tile_max = local_max.maximumf(r2)

                new_max = running_max.maximumf(tile_max)
                max_diff = running_max - new_max
                safe_diff = (running_max > NEG_INF).select(max_diff, ZERO_F)
                scale = (safe_diff * LOG2E_C).exp2(fastmath=arith.FastMathFlags.fast)
                running_sum = running_sum * scale
                for h in range_constexpr(PV_N_CHUNKS):
                    acc_pv[h] = _vsplat_mul(acc_pv[h], scale)
                running_max = new_max

                tile_sum = ZERO_F
                for elem in range_constexpr(4):
                    s = vector.extract(qk_acc, static_position=[elem])
                    d = s - new_max
                    d = (new_max > NEG_INF).select(d, NEG_INF)
                    p = (d * LOG2E_C).exp2(fastmath=arith.FastMathFlags.fast)
                    tile_sum = tile_sum + p
                    qk_acc = vector.insert(p, qk_acc,
                                           static_position=[elem], dynamic_position=[])

                ts1 = tile_sum.shuffle_xor(fx.Int32(16), c_w)
                tile_sum = tile_sum + ts1
                ts2 = tile_sum.shuffle_xor(fx.Int32(32), c_w)
                tile_sum = tile_sum + ts2
                running_sum = running_sum + tile_sum

                # ---- V dequant → LDS. Each half hf uses UE8M0 group
                # (2*chunk_in_tok+hf) V scale; fp8_g32 V = cent[nibble]*vscale.
                if const_expr(USE_HW_TR):
                    # V dequant → LDS [token][head_dim] (ROW-MAJOR, no transpose)
                    # written as ds_write_b128 (vec(2,i64)); read back via the
                    # ds_read_tr16_b64 HW transpose in the PV block below.
                    v_lds_elem_base = (
                        tok_in_tile * fx.Int32(KV_ROW_ELEMS)
                        + chunk_in_tok * fx.Int32(SUBCHUNK_HDIMS)
                    )
                    for hf in range_constexpr(HALVES):
                        v_packed = v_packed_list[hf]
                        half_hd = fx.Int32(hf * HALF_HDIMS)
                        grp_shift = (
                            (chunk_in_tok * fx.Int32(2) + fx.Int32(hf)) & fx.Int32(3)
                        ) * fx.Int32(8)
                        vscale_byte = (vscale_word >> grp_shift) & fx.Int32(0xFF)
                        vscale_f32 = arith.bitcast(
                            T.f32, _ival(vscale_byte << fx.Int32(23)))
                        if ARCH_B:
                            c_arch_b = arith.constant(SCALE_C, type=T.f32)
                            vscale_f32 = vscale_f32 * c_arch_b
                        for w in range_constexpr(4):
                            word_i32 = vector.extract(v_packed, static_position=[w])
                            if const_expr(V_CVT):
                                # Native CDNA4 scaled convert: 2-wide
                                # cvt_scalef32_pk_bf16_fp4. srcSel s picks byte s
                                # of the word (nibbles 2s, 2s+1) → 2 bf16 with the
                                # UE8M0 group scale fused. 4 cvt calls/word.
                                bf16_elems = []
                                for s in range_constexpr(4):
                                    v2 = rocdl.cvt_scalef32_pk_bf16_fp4(
                                        T.vec(2, T.bf16),
                                        _ival(word_i32), _ival(vscale_f32), int(s),
                                    )
                                    bf16_elems.append(
                                        vector.extract(v2, static_position=[0]))
                                    bf16_elems.append(
                                        vector.extract(v2, static_position=[1]))
                                v_bf16 = vector.from_elements(
                                    T.vec(8, T.bf16), bf16_elems)
                            else:
                                bf16_elems = []
                                for n in range_constexpr(8):
                                    nibble = (word_i32 >> fx.Int32(n * 4)) & fx.Int32(0xF)
                                    nibble_idx = arith.index_cast(T.index, nibble)
                                    cent_f32 = cent_lds.load([nibble_idx])
                                    elem_bf16 = arith.trunc_f(
                                        T.bf16, cent_f32 * vscale_f32)
                                    bf16_elems.append(elem_bf16)
                                v_bf16 = vector.from_elements(
                                    T.vec(8, T.bf16), bf16_elems)
                            v_i64 = vector.bitcast(T.vec(2, T.i64), v_bf16)
                            v_lds_i64_idx = (
                                v_lds_elem_base + half_hd + fx.Int32(w * 8)
                            ) // fx.Int32(4)
                            vector.store(
                                v_i64, kv_lds_i64,
                                [arith.index_cast(T.index, v_lds_i64_idx)],
                            )
                    # HW V transpose cross-lane LDS fence (see 128 kernel).
                    rocdl.sched_barrier(0)
                    rocdl.s_waitcnt(0xC07F)
                else:
                    # Legacy transposed V_LDS path: ds_write_b16 per element.
                    for hf in range_constexpr(HALVES):
                        v_packed = v_packed_list[hf]
                        half_hd = hf * HALF_HDIMS
                        grp_shift = (
                            (chunk_in_tok * fx.Int32(2) + fx.Int32(hf)) & fx.Int32(3)
                        ) * fx.Int32(8)
                        vscale_byte = (vscale_word >> grp_shift) & fx.Int32(0xFF)
                        vscale_f32 = arith.bitcast(
                            T.f32, _ival(vscale_byte << fx.Int32(23)))
                        if ARCH_B:
                            c_arch_b = arith.constant(SCALE_C, type=T.f32)
                            vscale_f32 = vscale_f32 * c_arch_b
                        for w in range_constexpr(4):
                            word_i32 = vector.extract(v_packed, static_position=[w])
                            for n in range_constexpr(8):
                                nibble = (word_i32 >> fx.Int32(n * 4)) & fx.Int32(0xF)
                                nibble_idx = arith.index_cast(T.index, nibble)
                                cent_f32 = cent_lds.load([nibble_idx])
                                elem_bf16 = arith.trunc_f(T.bf16, cent_f32 * vscale_f32)
                                elem_i16 = arith.bitcast(T.i16, elem_bf16)
                                head_dim = chunk_in_tok * fx.Int32(SUBCHUNK_HDIMS) + fx.Int32(
                                    half_hd + w * 8 + n)
                                v_idx_i16 = head_dim * fx.Int32(TILE_SIZE) + tok_in_tile
                                v_vec = vector.from_elements(T.vec(1, T.i16), [elem_i16])
                                vector.store(v_vec, kv_lds_i16,
                                             [arith.index_cast(T.index, v_idx_i16)])
                gpu.barrier()

                # ---- PV MFMA: A=V[head_dim, token], B=P (=qk_acc bf16) -----
                p_bf16 = arith.trunc_f(T.vec(4, T.bf16), qk_acc)
                p_op = vector.bitcast(T.vec(4, T.i16), p_bf16)

                if const_expr(USE_HW_TR):
                    # HW-transpose PV: V_lds row-major V[token][head_dim].
                    token_idx = lane >> fx.Int32(2)
                    hd_sub = (lane & fx.Int32(3)) * fx.Int32(4)
                    v_lane_byte = (
                        fx.Int32(kv_off)
                        + token_idx * fx.Int32(KV_ROW_BYTES)
                        + hd_sub * fx.Int32(2)
                    )
                    for h in range_constexpr(PV_N_CHUNKS):
                        v_byte_off = v_lane_byte + fx.Int32(h * 32)
                        v_byte_i64 = fx.Int64(v_byte_off)
                        v_ptr = buffer_ops.create_llvm_ptr(
                            v_byte_i64, address_space=3,
                        )
                        v_op_raw = rocdl.ds_read_tr16_b64(
                            T.vec(4, T.i16), v_ptr,
                        ).result
                        acc_pv[h] = rocdl.mfma_f32_16x16x16bf16_1k(
                            T.f32x4, [v_op_raw, p_op, acc_pv[h], 0, 0, 0]
                        )
                else:
                    for h in range_constexpr(PV_N_CHUNKS):
                        v_idx_i64 = (
                            (mfma_row + fx.Int32(h * 16)) * fx.Int32(4)
                            + mfma_col_grp
                        )
                        kv_load = vector.load_op(
                            T.vec(1, T.i64), kv_lds_i64,
                            [arith.index_cast(T.index, v_idx_i64)],
                        )
                        v_op = vector.bitcast(T.vec(4, T.i16), kv_load)
                        acc_pv[h] = rocdl.mfma_f32_16x16x16bf16_1k(
                            T.f32x4, [v_op, p_op, acc_pv[h], 0, 0, 0]
                        )

            _scf.YieldOp([
                _ival(running_max),
                _ival(running_sum),
                *[_ival(p) for p in acc_pv],
            ])
        finally:
            _for_ip.__exit__(None, None, None)
        running_max = _for_op.results[0]
        running_sum = _for_op.results[1]
        acc_pv = list(_for_op.results[2:])

        # ===== STEP G: Output ===========================================
        safe_sum = (running_sum > ZERO_F).select(running_sum, ONE_F)
        rcp = ONE_F / safe_sum

        c_os = fx.Int32(_stride_out_seq)
        c_oh = fx.Int32(_stride_out_head)
        c_op_ = fx.Int32(_stride_out_part)
        out_base = seq * c_os + kv_h * c_oh + part * c_op_

        valid_row_pred = arith.cmpi(
            arith.CmpIPredicate.ult,
            mfma_row.ir_value() if hasattr(mfma_row, 'ir_value') else mfma_row,
            arith.constant(QG, type=T.i32),
        )
        _if = _scf.IfOp(valid_row_pred)
        with ir.InsertionPoint(_if.then_block):
            for h in range_constexpr(PV_N_CHUNKS):
                pv_norm = _vsplat_mul(acc_pv[h], rcp)
                pv_bf16 = arith.trunc_f(T.vec(4, T.bf16), pv_norm)
                pv_i32x2 = vector.bitcast(T.vec(2, T.i32), pv_bf16)
                head_dim_start = fx.Int32(h * 16) + mfma_col_grp * fx.Int32(4)
                out_off_elem = (
                    out_base
                    + mfma_row * fx.Int32(HEAD_SIZE)
                    + head_dim_start
                )
                buffer_ops.buffer_store(
                    pv_i32x2, out_rsrc,
                    out_off_elem * fx.Int32(2),
                    offset_is_bytes=True,
                )

            c_npq = fx.Int32(num_partitions * QG)
            ml_off = (
                seq * fx.Int32(_stride_ml_seq)
                + kv_h * c_npq + part * c_qg + mfma_row
            )
            es_off = (
                seq * fx.Int32(_stride_es_seq)
                + kv_h * c_npq + part * c_qg + mfma_row
            )
            buffer_ops.buffer_store(running_max, ml_rsrc, ml_off)
            buffer_ops.buffer_store(running_sum, es_rsrc, es_off)
            _scf.YieldOp([])

    return fp8_g32_decode_hd256_kernel
