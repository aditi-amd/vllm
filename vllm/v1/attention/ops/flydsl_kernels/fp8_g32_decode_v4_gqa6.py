# SPDX-License-Identifier: Apache-2.0
# fp8_g32 decode v4 GQA-6 sibling (FlyDSL) — MI355X / gfx950 / CDNA4
#
# Decode-only paged attention over the fp8_g32 (FP4 E2M1 + UE8M0 group scale)
# AoS KV cache. Targets MiniMax-M2.5 class models: HEAD_SIZE=128, GQA group=6,
# group_size=32 (=> N_GROUPS=4), BLOCK_SIZE in {16,32}, partitioned mode.
#
# === Sibling-file rationale ===
# Structurally identical to fp8_g32_decode_v4.py except:
#   * QUERY_GROUP_SIZE default = 6 (was 16)
#   * Build-time assert allows query_group_size = 6 only (Qwen sizes {8,16} go
#     to the canonical fp8_g32_decode_v4 kernel — NOT this file)
#   * QG_LOAD_ITERS = (QG + 3) // 4 (round-up; 2 iters for QG=6 load 8 LDS rows
#     of which only 6 carry real Q; rows 6..7 are address-redirected to the
#     in-bounds Q[0,0,col] slot and then gated out by the same mfma_row < QG
#     output predicate the canonical kernel already uses for QG=8)
#   * Q_LDS region kept at the full 16-row (4096 B) footprint
#   * Unique smem symbol + kernel/build function names so both modules can be
#     co-resident (Qwen GQA-{8,16} and MiniMax GQA-6) in one process.
# Mirrors the proven kernels/tq_decode_v4_gqa6.py delta. The GQA-6 sibling for
# fp8_g32 follows the same convention to preserve the canonical kernel's
# invariants for its production callers (Qwen2.5-72B / Qwen3-32B).
#
# This kernel is a direct port of the (bug-free) TurboQuant decode v4 kernel
# (kernels/tq_decode_v4.py). The Flash-Attention-2 online-softmax structure,
# the QK/PV MFMA layouts, the scf.ForOp split-K loop, the HW/SW V-transpose,
# the output store and the partition reducer are reused VERBATIM. Only the
# dequant + cache-addressing differ, per the fp8_g32 algorithm:
#
# === fp8_g32 vs TurboQuant differences ===
# * Cache layout is AoS: kv_cache[num_blocks, block_size, Hk, padded_slot]
#   uint8, with per-(slot,head) byte layout (D=128, group_size=32):
#       [  0 ..  64) K codes   (FP4 nibbles, 2/byte)
#       [ 64 ..  68) K scales  (UE8M0, 1 byte × 4 groups)
#       [ 68 .. 132) V codes
#       [132 .. 136) V scales  (UE8M0, 1 byte × 4 groups)
#   (vs TQ's separate data + SoA-meta regions with per-token k-norm/v-scale/
#   v-zero metadata.)
# * "Centroids" = the fixed 16-entry FP4 E2M1 value table (FP4_BITS_TO_VALUE),
#   loaded LDS-resident exactly like TQ's learned centroids. The stored nibble
#   is the E2M1 bit pattern, so cent_lds[nibble] is the dequant value directly.
# * K dequant:  value = cent_lds[nibble] * 2^(k_scale_byte - 127)
#   V dequant:  value = cent_lds[nibble] * 2^(v_scale_byte - 127)
#   The UE8M0 scale is one byte per group-of-32. Each of the 64 lanes owns
#   exactly one group (chunk_in_tok in 0..3 == head_dims chunk*32..+31), so a
#   lane loads exactly ONE k-scale + one v-scale byte. 2^(byte-127) needs NO
#   exp2: a normal fp32 with exponent==byte and zero mantissa equals it, so
#   scale_f32 = bitcast_f32(byte << 23) (byte==0 -> 0.0 zero sentinel).
# * No per-token k-norm, no v-zero, no affine. Arch B (opt-in) folds an extra
#   constant `c` into the scale.
# * Q is Hadamard-rotated by the launcher (PiT GEMM), round-tripped through
#   FP8 E4M3 (precision haircut matching the reference + Triton v3), then fed
#   as bf16 to the same bf16 QK MFMA.
#
# MFMA layouts used (unchanged from TQ v4):
#   QK:  mfma(A=K, B=Q, C=qk_acc)    →  C[m=token, n=query],  4 fp32/lane
#   PV:  mfma(A=V_T, B=P, C=acc_pv)  →  C[m=head_dim, n=query], 4 fp32/lane

from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl, vector
from flydsl.expr.typing import T, Int32
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf as _scf


# === Constants (MiniMax-M2.5-class fp8_g32 decode profile) ==================
HEAD_SIZE = 128
KV_BLOCK_SIZE = 16          # default; overridable via build_fp8_g32_decode_v4_gqa6_module(kv_block_size=...)
TILE_SIZE = 16              # MFMA tile = 16 tokens (do not change without re-deriving MFMA shapes)
N_CENTROIDS = 16            # FP4 E2M1 value table (16 entries indexed by nibble)
QUERY_GROUP_SIZE = 6         # GQA-6 default for MiniMax-M2.5
WARP_SIZE = 64
NUM_WARPS = 1
BLOCK_THREADS = NUM_WARPS * WARP_SIZE
KV_COMPUTE_BLOCK = 256                      # 16 K-tiles × 16 tokens

# fp8_g32 AoS slot layout (per (slot, head), D=128, group_size=32 => 4 groups)
FP8_GROUP_SIZE = 32
N_GROUPS = HEAD_SIZE // FP8_GROUP_SIZE       # 4 UE8M0 scale bytes per head
UE8M0_BIAS = 127
KEY_CODE_BYTES = HEAD_SIZE // 2              # 64
VAL_CODE_BYTES = HEAD_SIZE // 2              # 64
K_SCALES_OFFSET = KEY_CODE_BYTES                         # 64
V_CODES_OFFSET = KEY_CODE_BYTES + N_GROUPS               # 68
V_SCALES_OFFSET = V_CODES_OFFSET + VAL_CODE_BYTES        # 132
SLOT_CONTENT_BYTES = V_SCALES_OFFSET + N_GROUPS          # 136

# MFMA
MFMA_M = MFMA_N = 16
MFMA_K_BF16_QK = 32                         # CDNA4 wide-K (mfma_f32_16x16x32_bf16)
MFMA_K_BF16_PV = 16                         # PV's K = token tile = 16
QK_K_CHUNKS = HEAD_SIZE // MFMA_K_BF16_QK   # 4 (down from 8)
PV_N_CHUNKS = HEAD_SIZE // MFMA_N           # 8

# LDS regions
CENTROID_LDS_BYTES = N_CENTROIDS * 4                        # 64
Q_LDS_BYTES = 16 * HEAD_SIZE * 2                           # 4096 (full 16-row footprint, GQA-agnostic)
KV_TILE_LDS_BYTES = TILE_SIZE * HEAD_SIZE * 2               # 4096
# qk_fp8 only: per-(token, group) K UE8M0 scale staged for the post-MFMA fold.
SCALE_LDS_BYTES = TILE_SIZE * N_GROUPS * 4                  # 16*4*4 = 256

LOG2E = 1.4426950408889634
NEG_INF_VAL = float("-inf")


def _vsplat_mul(vec, scalar):
    s = scalar.ir_value() if hasattr(scalar, 'ir_value') else scalar
    return vec * vector.broadcast(T.f32x4, s)


allocator = None


def build_fp8_g32_decode_v4_gqa6_module(
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
):
    """Build an fp8_g32 decode v4 kernel module.

    ``padded_slot`` is the per-(token, head) slot stride in bytes of the
    fp8_g32 KV cache (= ``kv_cache.shape[3]``, >= 136 for D=128/group=32).
    The cache is ``[num_blocks, block_size, num_kv_heads, padded_slot]``
    uint8 with the AoS byte layout documented at the top of this file.

    ``arch_b`` (default False): when True the dequant multiplies the UE8M0
    scale by the extra constant ``scale_c`` (the c-baked-codebook recipe);
    Arch A (default) folds ``c`` into the stored scale byte so no extra
    multiply is needed.

    Below this point the docstring describes the framework reused verbatim
    from the TurboQuant v4 kernel.

    ``query_group_size`` (= num_query_heads // num_kv_heads) is fixed at 6
    for this sibling. The MFMA's 16-row capacity is 6/16 = 37.5% used;
    lanes 6..15 compute garbage and are gated out of all global-memory
    writes via the same ``mfma_row < QG`` predicate the canonical kernel
    uses for QG=8. Q load iterates ``(QG + 3) // 4`` = 2 chunks (loads
    8 LDS rows of which rows 6..7 are address-redirected to the in-bounds
    Q[0,0,col] slot — never consumed because the gating lanes don't write).

    For GQA-8 or GQA-16 (Qwen models), use the canonical
    ``build_fp8_g32_decode_v4_module`` from ``fp8_g32_decode_v4.py`` instead.

    ``kv_block_size`` is the number of tokens per vLLM cache block. Must
    be a multiple of TILE_SIZE (=16). Supported values: 16 and 32. With
    block_size=32, two MFMA K-tiles fit in each block; the kernel walks
    block-table entries every ``kv_block_size // TILE_SIZE`` tiles and
    re-uses the entry for sub-tiles within the same block. SoA metadata
    region grows linearly with block_size.

    ``use_hw_v_transpose`` (CDNA4 / gfx950 only) replaces the 32-element
    scattered ``ds_write_b16`` V-dequant pattern with a row-major
    ``V[token][head_dim]`` LDS layout written as 4 wide ``ds_write_b128``
    per lane, then read back into the MFMA A operand via the hardware
    ``ds_read_tr16_b64`` transpose. This eliminates ~28 ds_write
    instructions per lane per K-tile (32 -> 4) and 8 strided ds_read_b64
    per PV chunk (replaces them with 1 hw-transpose read each). Pattern
    derived from ``flash_attn_func.py`` (USE_HW_TR=True path).

    ``tile_groups_per_partition`` (default 1) controls the FA-2 split-K
    granularity. With value G each partition processes ``G * 16`` K-tiles
    = ``G * KV_COMPUTE_BLOCK`` tokens, with the existing 16-tile loop
    body iterated G times. The launcher uses this to bound
    ``num_partitions`` (e.g. cap at 32) for long context: at 32K /
    block_size=32 / num_partitions=32, G=5 covers 32*5*256 = 40 960
    tokens of worst-case context with grid.z=32 instead of 256.
    Behavior at G=1 is bit-identical to the pre-Option-A kernel.

    ``use_wht_butterfly`` (default False) replaces the STEP B HBM load of
    the externally-rotated ``q_rot`` tensor with an in-register 7-stage
    Walsh-Hadamard butterfly that computes ``H @ q`` directly (H = the
    normalised Hadamard matrix = PiT for TurboQuant).  The launcher must
    pass the raw ``query`` tensor instead of ``q_rot`` when this is True.
    Gate: ``VLLM_TQ_DECODE_V4_WHT_BUTTERFLY=1``.
    """
    assert query_group_size == 6, (
        f"build_fp8_g32_decode_v4_gqa6_module is GQA-6 only; got "
        f"query_group_size={query_group_size}. Use build_fp8_g32_decode_v4_module "
        f"from fp8_g32_decode_v4.py for GQA-8 or GQA-16."
    )
    assert kv_block_size in (16, 32), (
        f"kv_block_size must be 16 or 32; got {kv_block_size}"
    )
    assert kv_block_size % TILE_SIZE == 0
    assert int(tile_groups_per_partition) >= 1, (
        f"tile_groups_per_partition must be >= 1; got {tile_groups_per_partition}"
    )
    USE_HW_TR = bool(use_hw_v_transpose)
    TGPP = int(tile_groups_per_partition)
    PARTITION_EXTENT_TOKENS = TGPP * KV_COMPUTE_BLOCK
    ARCH_B = bool(arch_b)
    SCALE_C = float(scale_c)
    # QK fp8 MFMA path (Step A): replace the bf16 wide-K QK MFMA with 4x
    # mfma_f32_16x16x32_fp8_fp8 (one per UE8M0 group of 32 head-dims, since
    # GROUP_SIZE==MFMA_K==32). Q and K codes are cast to E4M3 (lossless: the
    # 15 FP4 E2M1 levels and the launcher's E4M3-rounded Q are exact in E4M3)
    # and the per-(token,group) UE8M0 scale is folded on each group's fp32
    # accumulator as a per-token vec4 (scores[t]=sum_g k_scale[t,g]*partial_g).
    # PV stays bf16 MFMA (its scale is not separable on the output).
    QK_FP8 = bool(qk_fp8)
    # QK scaled-MFMA path: SINGLE native scaled mfma_scale_f32_16x16x128_f8f6f4
    # (K=128 in one issue). A=K raw FP4 codes + per-(token,group) UE8M0 scaleA;
    # B=Q E4M3. Output layout identical to bf16 path. Mutually exclusive with
    # qk_fp8 (both replace the bf16 QK MFMA via different mechanisms).
    QK_SCALED = bool(qk_scaled)
    assert not (QK_FP8 and QK_SCALED), (
        "qk_fp8 and qk_scaled are mutually exclusive QK MFMA paths"
    )
    # V cvt: replace per-nibble LUT dequant with native cvt_scalef32_pk_bf16_fp4.
    # Only active when use_hw_v_transpose=True (requires HW-TR LDS layout).
    V_CVT = bool(v_cvt)
    # Q-operand hoist (qk_scaled only): build the loop-invariant fp8 Q operand
    # once in STEP C instead of rebuilding it per K-tile.
    Q_HOIST = bool(q_hoist)

    QG = int(query_group_size)
    # Round-up: 2 iters for QG=6 → loads 8 LDS rows (rows 6..7 are
    # address-redirected to the in-bounds Q[0,0,col] slot in STEP B; never
    # consumed because mfma_row >= QG lanes are gated out of all global stores).
    QG_LOAD_ITERS = (QG + 3) // 4
    OOB_OFFSET = 0x7FFFFFF0         # ~2GB byte offset; > any plausible buffer

    _BS = int(kv_block_size)
    _TILES_PER_BLOCK = _BS // TILE_SIZE     # 1 for BS=16, 2 for BS=32
    _PADDED_SLOT = int(padded_slot)
    assert _PADDED_SLOT >= SLOT_CONTENT_BYTES, (
        f"padded_slot={_PADDED_SLOT} < {SLOT_CONTENT_BYTES} (fp8_g32 content)"
    )
    assert _PADDED_SLOT % 4 == 0, (
        f"padded_slot={_PADDED_SLOT} must be 4-byte aligned for vec(4,i32) "
        "code loads + i32 scale-word loads"
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
    # All strides in BYTES (cache is uint8).
    _stride_cache_head = _PADDED_SLOT                       # per head slot
    _stride_cache_pos = num_kv_heads * _PADDED_SLOT         # per token in block
    _stride_cache_block = _BS * _stride_cache_pos           # per cache block

    _stride_out_part = QG * HEAD_SIZE
    _stride_out_head = num_partitions * QG * HEAD_SIZE
    _stride_out_seq = num_kv_heads * num_partitions * QG * HEAD_SIZE
    _stride_es_seq = num_kv_heads * num_partitions * QG
    _stride_ml_seq = _stride_es_seq

    # --- LDS layout ---
    allocator = SmemAllocator(None, arch=arch, global_sym_name="fp8_g32_v4_gqa6_smem")
    centroid_off = 0
    allocator.ptr = CENTROID_LDS_BYTES
    q_off = allocator.ptr
    allocator.ptr += Q_LDS_BYTES
    kv_off = allocator.ptr
    allocator.ptr += KV_TILE_LDS_BYTES
    scale_off = allocator.ptr
    if QK_FP8 or QK_SCALED:
        allocator.ptr += SCALE_LDS_BYTES

    @flyc.kernel
    def fp8_g32_decode_v4_gqa6_kernel(
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
        lane = tid                                      # 0..63
        mfma_row = lane & fx.Int32(15)                  # = query_row (B's N) /
                                                        #   token (A's M for QK)
        mfma_col_grp = lane >> fx.Int32(4)              # 0..3, K-group dim

        # ---- Buffer resources -------------------------------------------
        q_rsrc = buffer_ops.create_buffer_resource(query_ptr, max_size=True)
        kv_rsrc = buffer_ops.create_buffer_resource(kv_cache_ptr, max_size=True)
        bt_rsrc = buffer_ops.create_buffer_resource(block_tables_ptr, max_size=True)
        sl_rsrc = buffer_ops.create_buffer_resource(seq_lens_ptr, max_size=True)
        cent_rsrc = buffer_ops.create_buffer_resource(centroids_ptr, max_size=True)
        out_rsrc = buffer_ops.create_buffer_resource(out_ptr, max_size=True)
        es_rsrc = buffer_ops.create_buffer_resource(exp_sums_ptr, max_size=True)
        ml_rsrc = buffer_ops.create_buffer_resource(max_logits_ptr, max_size=True)

        # ---- LDS pointers -----------------------------------------------
        base = allocator.get_base()
        cent_lds = SmemPtr(base, centroid_off, T.f32, shape=(N_CENTROIDS,))
        q_lds_i32 = SmemPtr(base, q_off, T.i32, shape=(Q_LDS_BYTES // 4,)).get()
        q_lds_i64 = SmemPtr(base, q_off, T.i64, shape=(Q_LDS_BYTES // 8,)).get()
        kv_lds_i32 = SmemPtr(base, kv_off, T.i32, shape=(KV_TILE_LDS_BYTES // 4,)).get()
        kv_lds_i64 = SmemPtr(base, kv_off, T.i64, shape=(KV_TILE_LDS_BYTES // 8,)).get()
        kv_lds_i16 = SmemPtr(base, kv_off, T.i16, shape=(KV_TILE_LDS_BYTES // 2,)).get()
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
        c_stride_pos = fx.Int32(_stride_cache_pos)     # bytes per (token) slot row
        c_stride_head = fx.Int32(_stride_cache_head)   # bytes per (head) slot
        c_kscale_off = fx.Int32(K_SCALES_OFFSET)       # 64
        c_vcode_off = fx.Int32(V_CODES_OFFSET)         # 68
        c_vscale_off = fx.Int32(V_SCALES_OFFSET)       # 132
        c_w = fx.Int32(WARP_SIZE)

        NEG_INF = arith.constant(NEG_INF_VAL, type=T.f32)
        ZERO_F = fx.Float32(0.0)
        ONE_F = fx.Float32(1.0)
        LOG2E_C = arith.constant(LOG2E, type=T.f32)
        QK_SCALE = arith.constant(_qk_scale, type=T.f32)

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

        # Helper: unwrap fx wrapper → raw ir.Value (used in STEP B' and STEP F)
        def _ival(v):
            return v.ir_value() if hasattr(v, 'ir_value') else v

        # ===== STEP A: Load centroids → LDS (cooperative, race-safe) =====
        c_idx_safe = lane & fx.Int32(N_CENTROIDS - 1)
        c_val = buffer_ops.buffer_load(
            cent_rsrc, c_idx_safe, vec_width=1, dtype=T.f32
        )
        cent_lds.store(c_val, [arith.index_cast(T.index, c_idx_safe)])
        gpu.barrier()

        # ===== STEP B / B': Load Q → row-major LDS [query_row, head_dim] ===
        # Two paths controlled by the use_wht_butterfly compile-time flag:
        #
        # STEP B  (use_wht_butterfly=False, default):
        #   q_rsrc points to the externally-rotated q_rot tensor.  Load 8 bf16
        #   per lane per iter (vec_width=4 i32) in QG_LOAD_ITERS iterations.
        #
        # STEP B' (use_wht_butterfly=True):
        #   q_rsrc points to the RAW query tensor.  For each GQA head h, lane i
        #   loads elements [2*i, 2*i+1], applies a 7-stage Hadamard butterfly
        #   (1 intra-lane + 6 cross-lane shuffle_xor) to compute H @ q in
        #   register, scales by 1/sqrt(D), and writes the rotated pair to Q_LDS.
        #   No external GEMM or HBM q_rot tensor needed.
        if not use_wht_butterfly:
            # --- STEP B: load pre-rotated q_rot → Q_LDS ---
            # QG=6 → QG_LOAD_ITERS=2 walks 8 LDS rows (row ∈ 0..7) but only
            # rows 0..5 are real Q heads (Hq = num_kv_heads * 6). Rows 6..7
            # would address past the last (seq, kv_h)'s Q heads and, with the
            # buffer descriptor's max_size advertising 4 GB, the HW does no OOB
            # clamping → the padding read touches unmapped pages and the launch
            # dies with "Memory access fault by GPU node" (allocator-pattern-
            # dependent → reproduces as a heisenbug).
            #
            # Fix (mirrors tq_decode_v4_gqa6.py): redirect OOB lanes
            # (``row >= QG``) to load Q[0,0,col_elem] — a byte offset always
            # inside the actual Q tensor. The redundant load is wasted but
            # harmless; the resulting LDS write to row ∈ {6, 7} is never
            # consumed because every global store is gated by the output-time
            # ``mfma_row < QG`` predicate (the same gate canonical QG=8 uses
            # for its garbage rows 8..15).
            for c in range_constexpr(QG_LOAD_ITERS):
                row_chunk = lane + fx.Int32(c * WARP_SIZE)     # 0..127
                row = row_chunk >> fx.Int32(4)                 # 0..7
                col_b = row_chunk & fx.Int32(15)               # 0..15
                col_elem = col_b * fx.Int32(8)                 # 0..120 in bf16
                q_off_real = (
                    seq * c_sq
                    + (kv_h * c_qg + row) * c_qh
                    + col_elem
                )
                row_in_qg = row < fx.Int32(QG)
                q_off_elem = row_in_qg.select(q_off_real, col_elem)
                q_v = buffer_ops.buffer_load(
                    q_rsrc, q_off_elem // fx.Int32(2),
                    vec_width=4, dtype=T.i32,
                )
                q_lds_byte = row * fx.Int32(HEAD_SIZE * 2) + col_elem * fx.Int32(2)
                vector.store(
                    q_v, q_lds_i32,
                    [arith.index_cast(T.index, q_lds_byte // fx.Int32(4))],
                )
        else:
            # --- STEP B': in-register FWHT → Q_LDS ---
            # PiT = H (pure normalised Hadamard, symmetric) so H @ q = q @ H.
            # Lane i holds elements [2*i, 2*i+1] of one head at a time.
            # After all QG heads + barrier, Q_LDS is identical to what STEP B
            # would have produced from an externally pre-rotated q_rot tensor.
            _WHT_SCALE = arith.constant(1.0 / (HEAD_SIZE ** 0.5), type=T.f32)
            _ONE_F32_I32 = arith.constant(0x3F800000, type=T.i32)  # +1.0 bits
            _C16 = arith.constant(16, type=T.i32)
            _C31 = arith.constant(31, type=T.i32)
            _q_lds_smem = SmemPtr(base, q_off, T.i32,
                                  shape=(Q_LDS_BYTES // 4,))

            for h in range_constexpr(QG):
                # -- Load 2 packed bf16 (= 1 i32) for this lane / head --
                _q_elem_off = (
                    seq * c_sq
                    + (kv_h * c_qg + fx.Int32(h)) * c_qh
                    + lane * fx.Int32(2)
                )
                _q_raw = buffer_ops.buffer_load(
                    q_rsrc, _q_elem_off // fx.Int32(2),
                    vec_width=1, dtype=T.i32,
                )
                # Unpack i32 → 2 × bf16 → 2 × f32
                _lo_i16 = arith.trunci(T.i16, _q_raw)
                _hi_i16 = arith.trunci(
                    T.i16, arith.shrui(_q_raw, _C16)
                )
                _q_lo = arith.extf(T.f32, arith.bitcast(T.bf16, _lo_i16))
                _q_hi = arith.extf(T.f32, arith.bitcast(T.bf16, _hi_i16))

                # Stage 0: intra-lane butterfly (no shuffle required)
                _a = _q_lo + _q_hi
                _b = _q_lo - _q_hi
                _q_lo = _a
                _q_hi = _b

                # Stages 1-6: cross-lane butterfly (shuffle_xor + branchless ±)
                # result = q + sign * shuffle(q)  where sign ∈ {+1, -1}
                # sign = bitcast_f32(float_bits(1.0) XOR (lane_bit << 31))
                for _log2m in range_constexpr(6):   # masks 1, 2, 4, 8, 16, 32
                    _mask = 1 << _log2m
                    _other_lo = _q_lo.shuffle_xor(fx.Int32(_mask), c_w)
                    _other_hi = _q_hi.shuffle_xor(fx.Int32(_mask), c_w)
                    # lane_bit = 0 if this lane is "low" in the pair, else 1
                    _lane_bit = _ival(
                        (lane >> fx.Int32(_log2m)) & fx.Int32(1)
                    )
                    _sign_f32 = arith.bitcast(
                        T.f32,
                        arith.xori(
                            _ONE_F32_I32,
                            arith.shli(_lane_bit, _C31),
                        ),
                    )
                    _q_lo = _q_lo + _sign_f32 * _other_lo
                    _q_hi = _q_hi + _sign_f32 * _other_hi

                # Scale by 1/sqrt(HEAD_SIZE)
                _q_lo = _q_lo * _WHT_SCALE
                _q_hi = _q_hi * _WHT_SCALE

                # Pack 2 × f32 → 2 × bf16 → 1 × i32
                _lo_bf16_out = arith.truncf(T.bf16, _ival(_q_lo))
                _hi_bf16_out = arith.truncf(T.bf16, _ival(_q_hi))
                _lo_i32_out = arith.extui(
                    T.i32, arith.bitcast(T.i16, _lo_bf16_out)
                )
                _hi_i32_out = arith.extui(
                    T.i32, arith.bitcast(T.i16, _hi_bf16_out)
                )
                _packed = arith.ori(
                    _lo_i32_out,
                    arith.shli(_hi_i32_out, _C16),
                )

                # Write to Q_LDS: row h, cols [2*lane, 2*lane+1]
                _lds_i32_idx = fx.Int32(h * HEAD_SIZE // 2) + lane
                _q_lds_smem.store(
                    _packed,
                    [arith.index_cast(T.index, _ival(_lds_i32_idx))],
                )
        gpu.barrier()

        # ===== STEP C: Pre-load Q operands for QK_K_CHUNKS = 4 K-chunks ===
        # Wide-K MFMA: K_per_chunk = 32, K_per_lane = 32/4 = 8 bf16 per chunk.
        # Lane t holds 8 bf16 at row=mfma_row, K-cols
        #   chunk*32 + mfma_col_grp*8 + 0..7
        # Byte addr = mfma_row * HEAD_SIZE*2 + (chunk*32 + col_grp*8)*2
        #          = mfma_row * 256 + chunk*64 + col_grp*16
        # i64 idx  = mfma_row * 32 + chunk*8 + col_grp*2 (each i64 = 4 bf16)
        # Load 16 bytes (= 8 bf16 = 2 i64) per chunk via vec(2, i64).
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

        # qk_scaled + Q_HOIST: build the loop-invariant scaled-MFMA Q operand
        # once here. Lane t holds all 32 head-dims of group mfma_col_grp at
        # query row mfma_row: 4 sub-chunks of 8 bf16, each repacked to E4M3 i64,
        # assembled to vec8 i32. Held in registers and reused across the tile loop.
        q_op_hoisted = None
        if const_expr(QK_SCALED and Q_HOIST):
            _q_hoist_words = []
            for j in range_constexpr(4):
                _q_hoist_idx = (
                    mfma_row * fx.Int32(HEAD_SIZE * 2 // 8)
                    + mfma_col_grp * fx.Int32(8)
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
            q_op_hoisted = vector.bitcast(
                T.vec(8, T.i32),
                vector.from_elements(T.vec(4, T.i64), _q_hoist_words),
            )

        # ===== STEP D: Online softmax + PV state =========================
        running_max = NEG_INF
        running_sum = ZERO_F
        zero_v4 = arith.constant_vector(0.0, T.f32x4)
        acc_pv = [zero_v4 for _ in range(PV_N_CHUNKS)]

        # ===== STEP E: Sequence-len + partition base ====================
        seq_len = buffer_ops.buffer_load(sl_rsrc, seq, vec_width=1, dtype=T.i32)
        # ── Option A: bounded num_partitions + internal looping ─────────
        # With TGPP > 1 this CTA owns a contiguous slice of length
        # ``PARTITION_EXTENT_TOKENS`` that is iterated as ``TGPP`` groups
        # of 16 K-tiles each. ``partition_start`` is recomputed per
        # outer ``tg`` iteration below; the FA-2 online-softmax state
        # (running_max, running_sum, acc_pv) accumulates across all
        # ``TGPP * 16`` tiles for this CTA — semantically identical to
        # processing one large 16*TGPP-tile partition.
        partition_base = part * fx.Int32(PARTITION_EXTENT_TOKENS)

        # Per-K-tile dequant lane assignment:
        # 16 tokens × 64 packed bytes per token = 1024 bytes
        # 64 lanes × 16 bytes/lane = 1024 ✓
        # Lane t: token = t / 4, sub-chunk = t % 4 (0..3) → 16 bytes contiguous
        tok_in_tile = lane >> fx.Int32(2)
        chunk_in_tok = lane & fx.Int32(3)
        # ``slot`` (= absolute slot index within the cache block, range
        # 0.._BS-1) is recomputed per tile inside the K-tile loop below
        # so it reflects ``tile_in_block * TILE_SIZE + tok_in_tile``.
        #
        # fp8_g32 scale addressing: each lane owns exactly one group of 32
        # head-dims (== chunk_in_tok), so it needs exactly ONE K-scale byte
        # and ONE V-scale byte (the byte for group==chunk_in_tok). The 4
        # group scale bytes are 4-byte-aligned inside the slot (K@64, V@132),
        # so we load the 4-byte scale word as one i32 and extract our byte
        # via (word >> (chunk_in_tok*8)) & 0xFF.
        c_scale_byte_shift = chunk_in_tok * fx.Int32(8)

        # ===== STEP F: K-tile loop =======================================
        # With kv_block_size > TILE_SIZE there are _TILES_PER_BLOCK tiles
        # per cache block. Tiles within the same block share a block-table
        # entry but address different slot ranges within the block.
        #
        # ── Per-tile block-table OOB redirect ────────────────────────────
        # The K-tile loop is unrolled 16 times so every CTA issues 16
        # block-table reads regardless of (partition, seq_len). When a
        # partition extends past ``seq_len`` (e.g. num_partitions=2 for
        # a 256-token sequence — partition 1 covers tokens 256..511, all
        # masked out) the per-tile bt offset can land beyond the bt
        # allocation. With ``max_size=True`` the descriptor advertises
        # 4 GB so the HW returns whatever pre-existing HBM bytes live at
        # that offset; the resulting garbage ``phys_block`` is multiplied
        # by ``stride_cache_block`` and the subsequent kv_cache read
        # jumps into pages that may be unmapped (→ "Memory access fault
        # by GPU node") or may decode as NaN bytes (→ NaN segm_out).
        # Allocator-pattern dependent, but reproduces deterministically
        # at large B (e.g. B=64 seq=256 in the test sweep produces 8 192
        # NaN entries even on canonical QG=8).
        #
        # Fix: when the tile starts at or past ``seq_len`` (so its
        # qk_acc is going to be killed by the per-token
        # ``kv_tok < seq_len`` mask anyway), redirect the bt read to
        # ``bt[seq, 0]`` — always in bounds. The redundant phys_block
        # decode + kv_cache read is wasted work, but correctness is
        # preserved without changing the kernel's iteration count.
        # ===== Option A': scf.ForOp w/ iter_args (HIP-style) =========
        # Runtime-adaptive outer loop: trip count derives from
        # actual seq_len, NOT TGPP_max. At cudagraph-capture warmup
        # (seq_len=1) only 1 iteration runs; at 32K production
        # decode all TGPP iterations run. Kernel binary stays small
        # (single body, looped at runtime) — capture time matches
        # the legacy 16-tile-only kernel rather than scaling 4x.
        #
        # FA-2 state (running_max, running_sum, acc_pv[8]) threads
        # through scf iter_args; the body reads them at top, runs
        # the unchanged 16-tile inner body, and yields the new
        # state at bottom. After the loop, results are pulled out
        # of for_op.results back into the local Python names so the
        # downstream STEP G (output) is unmodified.
        c_kcb = fx.Int32(KV_COMPUTE_BLOCK)
        c_tgpp = fx.Int32(TGPP)
        c_zero_i32 = fx.Int32(0)
        c_one_i32 = fx.Int32(1)
        # remaining = seq_len - partition_base   (signed, may be <=0)
        remaining = seq_len - partition_base
        in_range = remaining > c_zero_i32
        # trip_raw = ceil(remaining / KV_COMPUTE_BLOCK)
        # (when in_range is false, the divisor branch produces
        # garbage that the select below overrides with zero, so we
        # don't pre-clamp)
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
        # Initial iter_args list: FA-2 accumulator state.
        # Order: running_max, running_sum, acc_pv[0..PV_N_CHUNKS-1].
        # NOTE: scf.ForOp requires ir.Value (with .type). The DSL
        # constants ZERO_F / ONE_F are fx.Float32 (Numeric) wrappers,
        # not raw ir.Value, so we unwrap via .ir_value() before
        # passing.  _ival() is defined once near the top of the kernel body.
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
                block_base = phys_block * c_block

                # ``slot`` is the absolute slot index within the cache block
                # (range 0.._BS-1). For BS=16 it equals tok_in_tile; for BS=32
                # it equals tile_in_block * 16 + tok_in_tile.
                #
                # fp8_g32 AoS slot base (bytes):
                #   block_base + slot*stride_pos + kv_h*stride_head
                slot = fx.Int32(tile_in_block * TILE_SIZE) + tok_in_tile
                slot_base_byte = (
                    block_base
                    + slot * c_stride_pos
                    + kv_h * c_stride_head
                )
                # K codes: 16 bytes (= group chunk_in_tok) at slot_base + chunk*16.
                k_byte = slot_base_byte + chunk_in_tok * fx.Int32(16)
                k_packed = buffer_ops.buffer_load(
                    kv_rsrc, k_byte // fx.Int32(4),
                    vec_width=4, dtype=T.i32,
                )

                # ---- HOISTED: issue V codes + K/V scale HBM loads early ---
                # These are async; their s_waitcnt is pushed by the compiler
                # past the K dequant + QK MFMA + softmax block, hiding most
                # of the V HBM latency behind compute.
                v_byte = slot_base_byte + c_vcode_off + chunk_in_tok * fx.Int32(16)
                v_packed = buffer_ops.buffer_load(
                    kv_rsrc, v_byte // fx.Int32(4),
                    vec_width=4, dtype=T.i32,
                )
                # UE8M0 group scales: load the 4-byte scale word (4-aligned)
                # and extract this lane's group byte. scale = 2^(byte-127) =
                # bitcast_f32(byte << 23); byte==0 -> +0.0 (zero sentinel).
                kscale_word = buffer_ops.buffer_load(
                    kv_rsrc, (slot_base_byte + c_kscale_off) // fx.Int32(4),
                    vec_width=1, dtype=T.i32,
                )
                vscale_word = buffer_ops.buffer_load(
                    kv_rsrc, (slot_base_byte + c_vscale_off) // fx.Int32(4),
                    vec_width=1, dtype=T.i32,
                )
                kscale_byte = (kscale_word >> c_scale_byte_shift) & fx.Int32(0xFF)
                vscale_byte = (vscale_word >> c_scale_byte_shift) & fx.Int32(0xFF)
                kscale_f32 = arith.bitcast(T.f32, _ival(kscale_byte << fx.Int32(23)))
                vscale_f32 = arith.bitcast(T.f32, _ival(vscale_byte << fx.Int32(23)))
                if ARCH_B:
                    c_arch_b = arith.constant(SCALE_C, type=T.f32)
                    kscale_f32 = kscale_f32 * c_arch_b
                    vscale_f32 = vscale_f32 * c_arch_b

                if const_expr(QK_FP8):
                    # qk_fp8 K dequant → LDS [token, head_dim] as E4M3 BYTES
                    # (1 byte/elem), UNSCALED (raw FP4 grid value). The UE8M0
                    # group scale is staged to scale_lds and folded post-MFMA.
                    # Lane (tok_in_tile, group=chunk_in_tok) writes 32 E4M3
                    # bytes = 4× i64 at byte tok*HEAD_SIZE + group*32 + w*8.
                    for w in range_constexpr(4):
                        word_i32 = vector.extract(k_packed, static_position=[w])
                        cents = []
                        for n in range_constexpr(8):
                            nibble = (word_i32 >> fx.Int32(n * 4)) & fx.Int32(0xF)
                            nibble_idx = arith.index_cast(T.index, nibble)
                            cents.append(cent_lds.load([nibble_idx]))
                        k_i64 = _f32x8_to_fp8_i64(cents)
                        k_i64_idx = (
                            tok_in_tile * fx.Int32(HEAD_SIZE // 8)
                            + chunk_in_tok * fx.Int32(4)
                            + fx.Int32(w)
                        )
                        vector.store(
                            vector.from_elements(T.vec(1, T.i64), [k_i64]),
                            kv_lds_i64,
                            [arith.index_cast(T.index, k_i64_idx)],
                        )
                    # Stage this lane's (token, group) UE8M0 K scale for the
                    # post-MFMA per-token fold.
                    scale_lds.store(
                        kscale_f32,
                        [arith.index_cast(
                            T.index,
                            tok_in_tile * fx.Int32(N_GROUPS) + chunk_in_tok,
                        )],
                    )
                elif const_expr(QK_SCALED):
                    # qk_scaled: store raw FP4 codes (natural contiguous order,
                    # 16 bytes = 32 nibbles per group) directly to KV LDS for the
                    # scaled MFMA A operand. No dequant needed here.
                    k_code_idx = (
                        tok_in_tile * fx.Int32(HEAD_SIZE // 8)
                        + chunk_in_tok * fx.Int32(4)
                    )
                    vector.store(
                        k_packed, kv_lds_i32,
                        [arith.index_cast(T.index, k_code_idx)],
                    )
                    # Stage raw UE8M0 byte (0..255) for the scaleA word assembly.
                    scale_lds_i32.store(
                        kscale_byte,
                        [arith.index_cast(
                            T.index,
                            tok_in_tile * fx.Int32(N_GROUPS) + chunk_in_tok,
                        )],
                    )
                else:
                    # K dequant → LDS [token, head_dim] (natural)
                    # Lane writes 32 bf16 (= 4×8) for token=tok_in_tile,
                    # head_dims chunk_in_tok*32..+31 (4 sub-chunks of 8).
                    tok_kreg = tok_in_tile * fx.Int32(HEAD_SIZE * 2 // 8)
                    chunk_kreg = tok_kreg + chunk_in_tok * fx.Int32(8)
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
                            [arith.index_cast(T.index, chunk_kreg + fx.Int32(w * 2))],
                        )
                gpu.barrier()

                if const_expr(QK_FP8):
                    # qk_fp8 QK: 4× mfma_f32_16x16x32_fp8_fp8 (K=32 == one
                    # UE8M0 group). A=K[token, head_dim] E4M3 (8 fp8/lane = 1
                    # i64 at byte mfma_row*HEAD_SIZE + chk*32 + col_grp*8),
                    # B=Q E4M3. Each group's fp32 accumulator is multiplied by
                    # the per-token UE8M0 scale vec4 (lane holds 4 tokens
                    # col_grp*4..+3) then summed:  scores = Σ_g scale_g·partial_g.
                    qk_acc = zero_v4
                    for chk in range_constexpr(QK_K_CHUNKS):
                        k_idx_i64 = (
                            mfma_row * fx.Int32(HEAD_SIZE // 8)
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
                        # Fold the per-token UE8M0 scale for this group: the 4
                        # fp32/lane are tokens col_grp*4+elem; multiply each by
                        # scale_lds[token, chk] and accumulate into qk_acc.
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
                    # qk_scaled QK: SINGLE native scaled MFMA over K=128.
                    #   A = K raw FP4 codes (vec4 i32 = 16 bytes) per
                    #       (token=mfma_row, group=mfma_col_grp) from K-code LDS.
                    #   B = Q E4M3 (vec8 i32 = 32 bytes): 4 sub-chunks of 8 bf16
                    #       from Q_LDS converted in-register (or hoisted).
                    #   scaleA = per-(token,group) UE8M0 byte at op_sel byte 0.
                    #   scaleB = 0x7F identity.
                    BLGP_E2M1 = 4   # A operand type code = fp4 e2m1
                    CBSZ_E4M3 = 0   # B operand type code = fp8 e4m3
                    IDENT = fx.Int32(0x7F)
                    k_code_idx = (
                        mfma_row * fx.Int32(HEAD_SIZE // 8)
                        + mfma_col_grp * fx.Int32(4)
                    )
                    k_op = vector.load_op(
                        T.vec(4, T.i32), kv_lds_i32,
                        [arith.index_cast(T.index, k_code_idx)],
                    )
                    if const_expr(Q_HOIST):
                        q_op = q_op_hoisted
                    else:
                        q_fp8_words = []
                        for j in range_constexpr(4):
                            q_idx_i64 = (
                                mfma_row * fx.Int32(HEAD_SIZE * 2 // 8)
                                + mfma_col_grp * fx.Int32(8)
                                + fx.Int32(j * 2)
                            )
                            qv = vector.load_op(
                                T.vec(2, T.i64), q_lds_i64,
                                [arith.index_cast(T.index, q_idx_i64)],
                            )
                            q_fp8_words.append(
                                _bf16x8_to_fp8_i64(
                                    vector.bitcast(T.vec(8, T.bf16), qv))
                            )
                        q_op = vector.bitcast(
                            T.vec(8, T.i32),
                            vector.from_elements(T.vec(4, T.i64), q_fp8_words),
                        )
                    scbyte = fx.Int32(scale_lds_i32.load(
                        [arith.index_cast(
                            T.index,
                            mfma_row * fx.Int32(N_GROUPS) + mfma_col_grp)]
                    ))
                    kscale = fx.Int32(0x7F7F7F00) | scbyte
                    qk_acc = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        T.vec(4, T.f32),
                        [k_op, q_op, zero_v4,
                         BLGP_E2M1, CBSZ_E4M3, 0, kscale, 0, IDENT],
                    )
                    if ARCH_B:
                        qk_acc = _vsplat_mul(
                            qk_acc, arith.constant(SCALE_C, type=T.f32))
                else:
                    # QK MFMA (CDNA4 wide-K): A=K[token, head_dim], B=Q (= Q^T).
                    # K read: lane t = K[token=mfma_row, head_dim=chunk*32+col_grp*8..+7]
                    # Same i64×2 indexing as Q.
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

                # qk_acc layout: lane t holds C[token=(t/16)*4..+3, query=t%16]
                # 4 fp32/lane = 4 different tokens at SAME query_row.

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
                # Per-row max: max over 4 fp32 in lane, then xor-shuffle 16, 32
                # (across the 4 col_grps that share same mfma_row).
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

                # Compute probs: p = exp((qk - new_max) * LOG2E)
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

                # ---- V dequant → LDS (V codes + V scale loads were hoisted) -
                # v_packed and vscale_f32 are already in flight from the top of
                # this iteration (vscale_f32 = 2^(v_scale_byte-127) for this
                # lane's group). fp8_g32 V dequant: cent_lds[nibble]*vscale_f32
                # (FP4 grid value × UE8M0 group scale) — no zero point.
                if USE_HW_TR:
                    # V dequant → LDS [token][head_dim] (ROW-MAJOR, no transpose).
                    # Each lane writes 32 contiguous bf16 (one token, head_dims
                    # chunk_in_tok*32..+31) as 4× ds_write_b128 = 4× vec(2,i64).
                    # Replaces 32× ds_write_b16 of the legacy transposed path.
                    v_lds_elem_base = (
                        tok_in_tile * fx.Int32(HEAD_SIZE)
                        + chunk_in_tok * fx.Int32(32)
                    )
                    for w in range_constexpr(4):
                        word_i32 = vector.extract(v_packed, static_position=[w])
                        if const_expr(V_CVT):
                            # Native CDNA4 scaled convert: 2-wide cvt_scalef32_pk_bf16_fp4.
                            # srcSelIndex s picks byte s of the 32-bit code word
                            # (nibbles 2s, 2s+1) → 2 bf16 with UE8M0 scale fused.
                            # 4 cvt calls/word replace 8×(LUT gather + mul + trunc).
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
                            v_i64 = vector.bitcast(T.vec(2, T.i64), v_bf16)
                        else:
                            bf16_elems = []
                            for n in range_constexpr(8):
                                nibble = (word_i32 >> fx.Int32(n * 4)) & fx.Int32(0xF)
                                nibble_idx = arith.index_cast(T.index, nibble)
                                cent_f32 = cent_lds.load([nibble_idx])
                                elem_f32 = cent_f32 * vscale_f32
                                elem_bf16 = arith.trunc_f(T.bf16, elem_f32)
                                bf16_elems.append(elem_bf16)
                            v_bf16 = vector.from_elements(T.vec(8, T.bf16), bf16_elems)
                            v_i64 = vector.bitcast(T.vec(2, T.i64), v_bf16)
                        v_lds_i64_idx = (
                            v_lds_elem_base + fx.Int32(w * 8)
                        ) // fx.Int32(4)
                        vector.store(
                            v_i64, kv_lds_i64,
                            [arith.index_cast(T.index, v_lds_i64_idx)],
                        )
                    # ── HW V transpose: cross-lane LDS sync ─────────────────────
                    # ds_read_tr16_b64 below introduces a cross-lane LDS read:
                    # consumer lane t reads bytes written by source lanes
                    # 0,4,8,12 (etc.). The compiler's automatic waitcnt insertion
                    # is per-lane and has no awareness of the HW transpose's
                    # cross-lane forwarding, so it can sink the next iteration's
                    # ds_read ahead of these ds_write_b128 ops in the schedule.
                    #
                    # Two-part fence (mirrors mla_fwd_decode_m16x8_fp8_fp8.py):
                    #   1. sched_barrier(0): MachineScheduler reorder barrier
                    #      (mask=0 → no instruction class may cross). Pure hint,
                    #      generates no runtime instruction. Use the simple
                    #      sched_barrier (NOT sched_group_barrier, which is an
                    #      IGLP marker that requires partner barriers and
                    #      corrupts the schedule when used in isolation).
                    #   2. s_waitcnt lgkmcnt=0: runtime drain of the LDS write
                    #      queue. vmcnt/expcnt left at no-wait so we don't
                    #      stall on speculatively-issued HBM loads (which can
                    #      include OOB-but-masked buffer_load reads from the
                    #      next K-tile's hoisted V prefetch).
                    #
                    # The SW path below does NOT need this because each lane
                    # only reads cells it itself wrote (no cross-lane dep), and
                    # the per-lane same-address waitcnt the compiler emits is
                    # correct.
                    #
                    # Empirical: targets the -3.3pp GSM8K regression on
                    # Qwen3-32B (padded num_partitions=64, ~16 K-tiles each)
                    # while leaving Qwen2.5-72B (16 partitions) unchanged.
                    rocdl.sched_barrier(0)
                    # encode_waitcnt(vmcnt=63, expcnt=7, lgkmcnt=0)
                    #   = 0xF | (7<<4) | (0<<8) | (3<<14) = 0xC07F
                    rocdl.s_waitcnt(0xC07F)
                else:
                    # Legacy transposed V_LDS path: 32 ds_write_b16 per lane.
                    for w in range_constexpr(4):
                        word_i32 = vector.extract(v_packed, static_position=[w])
                        for n in range_constexpr(8):
                            nibble = (word_i32 >> fx.Int32(n * 4)) & fx.Int32(0xF)
                            nibble_idx = arith.index_cast(T.index, nibble)
                            cent_f32 = cent_lds.load([nibble_idx])
                            elem_f32 = cent_f32 * vscale_f32
                            elem_bf16 = arith.trunc_f(T.bf16, elem_f32)
                            elem_i16 = arith.bitcast(T.i16, elem_bf16)
                            head_dim = chunk_in_tok * fx.Int32(32) + fx.Int32(w * 8 + n)
                            v_idx_i16 = head_dim * fx.Int32(TILE_SIZE) + tok_in_tile
                            v_vec = vector.from_elements(T.vec(1, T.i16), [elem_i16])
                            vector.store(v_vec, kv_lds_i16,
                                         [arith.index_cast(T.index, v_idx_i16)])
                gpu.barrier()

                # ---- PV MFMA: A=V[head_dim, token], B=P (=qk_acc bf16) -----
                # P operand B layout matches qk_acc: lane t holds 4 bf16 at
                # K=token=(t/16)*4..+3, N=query=t%16. Just trunc_f to bf16.
                p_bf16 = arith.trunc_f(T.vec(4, T.bf16), qk_acc)
                p_op = vector.bitcast(T.vec(4, T.i16), p_bf16)

                if USE_HW_TR:
                    # HW-transpose PV path: V_lds is row-major V[token][head_dim].
                    # MFMA A operand layout: lane t holds 4 bf16 at
                    #   M=head_dim=mfma_row + h*16, K=token=(t/16)*4..+3
                    # ds_read_tr16_b64 (4-element 16-bit transpose, per-16-lane block):
                    #   result[lane=t, elem=e] = Input[source_lane=e*4 + (t%16)//4,
                    #                                   col=t%4]
                    # Per-lane address: token_idx = lane // 4 (covers 0..15 across
                    # all four 16-lane MFMA blocks), hd_sub = (lane % 4)*4 selects
                    # the 4-element column window inside the h*16 chunk.
                    # Total LDS byte offset = kv_off + token_idx*HEAD_SIZE*2
                    #                        + (h*16 + hd_sub)*2
                    token_idx = lane >> fx.Int32(2)
                    hd_sub = (lane & fx.Int32(3)) * fx.Int32(4)
                    v_lane_byte = (
                        fx.Int32(kv_off)
                        + token_idx * fx.Int32(HEAD_SIZE * 2)
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
                    # Legacy transposed-LDS PV: 1 ds_read_b64 per chunk.
                    # Byte addr = (mfma_row + h*16) * TILE_SIZE * 2 + (col_grp*4) * 2
                    #          = (mfma_row + h*16) * 32 + col_grp*8
                    # i64 idx  = (mfma_row + h*16) * 4 + col_grp
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
        # Pull final accumulator values out of the for_op results.
        running_max = _for_op.results[0]
        running_sum = _for_op.results[1]
        acc_pv = list(_for_op.results[2:])

        # ===== STEP G: Output ===========================================
        # acc_pv[h] layout: lane t holds 4 fp32 at M=head_dim=(t/16)*4..+3 + h*16,
        #                                       N=query_row=t%16.
        # 4 fp32/lane = 4 contiguous head_dim values for ONE query_row.
        safe_sum = (running_sum > ZERO_F).select(running_sum, ONE_F)
        rcp = ONE_F / safe_sum

        c_os = fx.Int32(_stride_out_seq)
        c_oh = fx.Int32(_stride_out_head)
        c_op_ = fx.Int32(_stride_out_part)
        out_base = seq * c_os + kv_h * c_oh + part * c_op_

        # Gate all global stores by mfma_row < QG. For QG=16 the predicate is
        # tautological (mfma_row ∈ [0,15]); for QG=8 lanes 8..15 (whose MFMA
        # outputs are computational waste) skip the stores entirely.
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

    return fp8_g32_decode_v4_gqa6_kernel
