# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton fused TurboQuant decode attention.

Decode path: Triton stage1 (split-KV tiled attention scoring + value
accumulation) + stage2 (log-sum-exp reduction across splits).

Supports FP8 (E4M3) keys, 3-bit and 4-bit uniform quantized values.
"""

import ctypes
import math
import os
import time
import warnings

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.turboquant_runtime_stats import record_decode_call

_FP8_E4B15: int | None = None

# ---------------------------------------------------------------------------
# HIP kernels for ROCm: optimized Stage1 and Stage2 for MI300X/MI325X
#
# Simplified architecture (3 paths):
#   1. FUSED  — short seq: Grid=(B,Hq), no splits, direct bf16 output
#   2. SPLIT  — long seq:  Grid=(B,Hq,splits), 4-warp, mid_o + Stage2
#   3. TRITON — fallback:  CUDA, FP8, or no HIP .so
# ---------------------------------------------------------------------------
_HIP_SPLIT_LIB = None
_HIP_SPLIT_FN = None
_HIP_FUSED_LIB = None
_HIP_FUSED_FN = None
_HIP_STAGE2_LIB = None
_HIP_STAGE2_BF16_FN = None
_HIP_STAGE2_F32_FN = None

# (Legacy loader state removed — unified into _HIP_SPLIT/FUSED above)

_WARNED_HIP_SO_KEYS: set[str] = set()

_DISABLE_HIP_SO = os.environ.get("TQ_DISABLE_HIP_SO", "0") == "1"
_ALLOW_STALE_HIP_SO = os.environ.get("TQ_ALLOW_STALE_HIP_SO", "0") == "1"

# (Legacy V56 threshold removed — superseded by _FUSED_SEQ_THRESHOLD)


def _warn_hip_so_once(key: str, message: str) -> None:
    if key in _WARNED_HIP_SO_KEYS:
        return
    _WARNED_HIP_SO_KEYS.add(key)
    warnings.warn(message, RuntimeWarning, stacklevel=2)


def _hip_so_is_usable(so_path: str, *reference_files: str) -> bool:
    if _DISABLE_HIP_SO:
        _warn_hip_so_once(
            "hip-so-disabled",
            "TurboQuant HIP .so loaders are disabled by TQ_DISABLE_HIP_SO=1; "
            "falling back to the Triton path.",
        )
        return False

    if not os.path.exists(so_path):
        return False

    if _ALLOW_STALE_HIP_SO:
        return True

    ref_mtimes = [
        os.path.getmtime(path)
        for path in reference_files
        if path and os.path.exists(path)
    ]
    if not ref_mtimes:
        return True

    so_mtime = os.path.getmtime(so_path)
    if so_mtime < max(ref_mtimes):
        basename = os.path.basename(so_path)
        _warn_hip_so_once(
            f"stale:{basename}",
            "TurboQuant HIP .so appears older than its Python launcher; "
            f"skipping stale library `{basename}`. "
            "Rebuild the HIP kernels or set TQ_ALLOW_STALE_HIP_SO=1 "
            "to force loading it.",
        )
        return False
    return True


# ---- Shared argtypes for the Unified kernel v136 ----
# v136 accepts bf16 Q input and supports dual-mode operation:
#   splits > 1: write partials to mid_o (needs Stage2)
#   splits == 1: write final bf16 output directly (no Stage2)
_UNIFIED_ARGTYPES = (
    [ctypes.c_void_p] * 7  # q_rot_bf16, kv_cache, bt, seq_lens, centroids, mid_o, output
    + [ctypes.c_int] * 2   # stride_qb, stride_qh
    + [ctypes.c_int] * 3   # stride_cb, stride_cp, stride_ch
    + [ctypes.c_int]       # stride_bt
    + [ctypes.c_int] * 3   # stride_mb, stride_mh, stride_ms
    + [ctypes.c_int] * 2   # stride_ob, stride_oh
    + [ctypes.c_int] * 4   # num_kv_heads, block_size, num_kv_splits, kv_group_size
    + [ctypes.c_float]     # attn_scale
    + [ctypes.c_int]       # norm_correction
    + [ctypes.c_int] * 2   # B, Hq
    + [ctypes.c_void_p]    # hipStream_t stream
)

# Legacy argtypes (v132 and earlier) — kept for backward compat
_STAGE1_ARGTYPES = (
    [ctypes.c_void_p] * 6  # q_rot, kv_cache, bt, seq_lens, centroids, mid_o
    + [ctypes.c_int] * 2   # stride_qb, stride_qh
    + [ctypes.c_int] * 3   # stride_cb, stride_cp, stride_ch
    + [ctypes.c_int]       # stride_bt
    + [ctypes.c_int] * 3   # stride_mb, stride_mh, stride_ms
    + [ctypes.c_int] * 4   # num_kv_heads, block_size, num_kv_splits, kv_group_size
    + [ctypes.c_float]     # attn_scale
    + [ctypes.c_int]       # norm_correction
    + [ctypes.c_int] * 2   # B, Hq
    + [ctypes.c_void_p]    # hipStream_t stream
)


def _load_hip_split():
    """Load the unified kernel v136 (bf16 Q + direct output + split paths).

    Architecture: 256 threads = 4 wavefronts, Grid=(B, Hk, splits).
    Dual-mode operation:
      - splits > 1: write partials to mid_o, separate Stage2 reduces
      - splits == 1: write final bf16 output directly, no Stage2
    Accepts native bf16 Q input (halves Q HBM traffic vs fp32).
    Phase 1 (K scoring): All 4 waves do MFMA (each handles 2 of 8
    kb blocks), with cross-wave LDS reduction.
    Phase 2 (Value accumulation): Value MFMA with LDS redistribution.
    Based on v132 core with bf16 Q path and unified output logic.
    1.104x geomean E2E speedup over v132 production.
    """
    global _HIP_SPLIT_LIB, _HIP_SPLIT_FN
    if _HIP_SPLIT_FN is not None:
        return _HIP_SPLIT_FN
    if _HIP_SPLIT_LIB is False:
        return None
    if not current_platform.is_rocm():
        _HIP_SPLIT_LIB = False
        return None

    so_path = os.path.join(os.path.dirname(__file__), "tq_decode_split_hip.so")
    if not _hip_so_is_usable(so_path, __file__):
        _HIP_SPLIT_LIB = False
        return None

    try:
        lib = ctypes.CDLL(so_path)
        fn = lib.launch_tq_decode_unified
        fn.argtypes = _UNIFIED_ARGTYPES
        fn.restype = None
        _HIP_SPLIT_LIB = lib
        _HIP_SPLIT_FN = fn
        return fn
    except Exception:
        _HIP_SPLIT_LIB = False
        return None


def _load_hip_fused():
    """Load the fused Stage1+Stage2 kernel V4 (4-warp, 128 threads).

    Grid = (B, Hq) — no split dimension.  Each block processes the full
    sequence.  Accepts native bf16/fp16 q_rot (no fp32 conversion) and
    outputs in the same dtype.  The `dtype` param (0=bf16, 1=fp16) controls
    BOTH Q input interpretation and output type.
    """
    global _HIP_FUSED_LIB, _HIP_FUSED_FN
    if _HIP_FUSED_FN is not None:
        return _HIP_FUSED_FN
    if _HIP_FUSED_LIB is False:
        return None
    if not current_platform.is_rocm():
        _HIP_FUSED_LIB = False
        return None

    so_path = os.path.join(os.path.dirname(__file__), "tq_decode_fused_hip.so")
    if not _hip_so_is_usable(so_path, __file__):
        _HIP_FUSED_LIB = False
        return None

    try:
        lib = ctypes.CDLL(so_path)
        fn = lib.launch_tq_decode_fused
        fn.argtypes = (
            [ctypes.c_void_p] * 6  # q_rot, kv_cache, bt, seq_lens, centroids, output
            + [ctypes.c_int] * 2   # stride_qb, stride_qh
            + [ctypes.c_int] * 3   # stride_cb, stride_cp, stride_ch
            + [ctypes.c_int]       # stride_bt
            + [ctypes.c_int] * 2   # stride_ob, stride_oh
            + [ctypes.c_int]       # num_kv_heads
            + [ctypes.c_int]       # block_size
            + [ctypes.c_int]       # kv_group_size
            + [ctypes.c_float]     # attn_scale
            + [ctypes.c_int]       # norm_correction
            + [ctypes.c_int]       # dtype (0=bf16, 1=fp16) — Q input AND output
            + [ctypes.c_int] * 2   # B, Hq
            + [ctypes.c_void_p]    # hipStream_t stream
        )
        fn.restype = None
        _HIP_FUSED_LIB = lib
        _HIP_FUSED_FN = fn
        return fn
    except Exception:
        _HIP_FUSED_LIB = False
        return None


# Fused-vs-split dispatch threshold.
# Fused-vs-split dispatch threshold.
#
# After deploying v112 (all-waves MFMA + higher occupancy), the split path
# now beats fused across ALL tested configurations including short sequences:
#
# Benchmark (MI355X gfx950, Hq=64, Hk=8, April 2026, v112 split):
#   B=4  L=256:  fused=62.6µs  split=53.2µs  → split 1.18x faster
#   B=4  L=384:  fused=85.0µs  split=52.4µs  → split 1.62x faster
#   B=8  L=256:  fused=63.8µs  split=53.4µs  → split 1.20x faster
#   B=16 L=512:  fused=112µs   split=51.9µs  → split 2.17x faster
#   B=32 L=1024: fused=309µs   split=102µs   → split 3.03x faster
#   B=32 L=2048: fused=615µs   split=159µs   → split 3.86x faster
#
# The v112 split kernel's GQA grid=(B,Hk,splits) with all-waves MFMA
# and 50% occupancy dominates the fused kernel's grid=(B,Hq) in every
# regime.  Fused path is effectively disabled but kept for future use.


def _should_use_fused(
    batch_size: int, max_seq_len_hint: int,
) -> bool:
    """Batch-adaptive fused-vs-split decision.

    Returns True if fused kernel is expected to be faster.
    After v112 split deployment, fused is never faster — always return False.
    """
    # v112 split beats fused in every tested config (B=4..32, L=256..8192).
    # Disable fused path entirely.
    return False


# (Legacy _should_use_v56 removed — superseded by _should_use_fused)


def _resolve_num_kv_splits(
    max_num_kv_splits: int,
    eager_max_num_kv_splits: int,
    max_seq_len_hint: int,
    batch_size: int,
    allow_adaptive_kv_splits: bool,
) -> int:
    if max_num_kv_splits <= 1:
        return 1
    if not allow_adaptive_kv_splits:
        return max_num_kv_splits

    eager_cap = max(1, min(max_num_kv_splits, eager_max_num_kv_splits))
    if max_seq_len_hint <= 0:
        return eager_cap

    # Long-context decode is highly sensitive to KV split count on MI355X.
    # Prefer more splits once context exceeds 1K, while keeping small-context
    # batches closer to the old behavior.
    if max_seq_len_hint >= 4096:
        return eager_cap
    if max_seq_len_hint >= 2048:
        suggested = 16 if batch_size >= 16 else eager_cap
        return min(eager_cap, suggested)
    if max_seq_len_hint >= 1024:
        return min(eager_cap, 16)

    target_tokens_per_split = 64
    suggested = max(1, math.ceil(max_seq_len_hint / target_tokens_per_split))
    return min(eager_cap, suggested)


def _load_hip_stage2():
    """Load HIP Stage2 reduce kernel with bf16 output."""
    global _HIP_STAGE2_LIB, _HIP_STAGE2_BF16_FN, _HIP_STAGE2_F32_FN
    if _HIP_STAGE2_BF16_FN is not None:
        return _HIP_STAGE2_BF16_FN, _HIP_STAGE2_F32_FN
    if _HIP_STAGE2_LIB is False:
        return None, None

    if not current_platform.is_rocm():
        _HIP_STAGE2_LIB = False
        return None, None

    so_path = os.path.join(os.path.dirname(__file__), "tq_decode_stage2_hip.so")
    if not _hip_so_is_usable(so_path, __file__):
        _HIP_STAGE2_LIB = False
        return None, None

    try:
        lib = ctypes.CDLL(so_path)
        _argtypes = (
            [ctypes.c_void_p] * 3  # mid_o, output, seq_lens
            + [ctypes.c_int] * 5   # stride_mb, stride_mh, stride_ms, stride_ob, stride_oh
            + [ctypes.c_int]       # num_kv_splits
            + [ctypes.c_int] * 2   # B, Hq
            + [ctypes.c_void_p]    # stream
        )
        fn_bf16 = lib.launch_tq_decode_stage2_bf16
        fn_bf16.argtypes = _argtypes
        fn_bf16.restype = None
        fn_f32 = lib.launch_tq_decode_stage2_f32
        fn_f32.argtypes = _argtypes
        fn_f32.restype = None
        _HIP_STAGE2_LIB = lib
        _HIP_STAGE2_BF16_FN = fn_bf16
        _HIP_STAGE2_F32_FN = fn_f32
        return fn_bf16, fn_f32
    except Exception:
        _HIP_STAGE2_LIB = False
        return None, None


def _use_fp8_e4b15(device: int = 0) -> int:
    """Return 1 if device needs fp8e4b15 (Ampere/Ada, SM < 8.9), else 0.
    On non-CUDA platforms (e.g. XPU), always returns 0 (use e4nv format).
    """
    global _FP8_E4B15
    if _FP8_E4B15 is None:
        if current_platform.is_cuda_alike():
            cap = torch.cuda.get_device_capability(device)
            _FP8_E4B15 = 1 if cap < (8, 9) else 0
        else:
            _FP8_E4B15 = 0
    return _FP8_E4B15


# ---------------------------------------------------------------------------
# Custom op wrappers for HIP kernels (CUDA graph compatibility)
# ---------------------------------------------------------------------------
# Raw ctypes calls are opaque to PyTorch's CUDA graph capture.  Wrapping
# them as torch.library custom_ops lets the graph system see and replay
# the GPU work correctly.  The `register_fake` (meta) impl tells
# FakeTensorMode the output shape/dtype without running the kernel.
# ---------------------------------------------------------------------------

try:
    from torch.library import register_fake
except ImportError:
    try:
        from torch.library import impl_abstract as register_fake
    except ImportError:
        register_fake = None


def _register_hip_custom_ops():
    """Register HIP kernel wrappers as torch custom ops (once)."""
    if not current_platform.is_rocm() or register_fake is None:
        return

    # --- HIP Stage1 split (4-warp, unified, native bf16/fp16 Q input) ---
    @torch.library.custom_op("tq::hip_stage1_split", mutates_args=("mid_o",))
    def hip_stage1_split(
        q_rot: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        centroids: torch.Tensor,
        mid_o: torch.Tensor,
        num_kv_heads: int,
        block_size: int,
        num_kv_splits: int,
        kv_group_size: int,
        attn_scale: float,
        norm_correction: int,
        input_dtype: int,       # 0=bf16, 1=fp16 (Q input type)
    ) -> None:
        fn = _load_hip_split()
        if fn is None:
            return
        B, Hq = q_rot.shape[0], q_rot.shape[1]
        stream_ptr = torch.cuda.current_stream(q_rot.device).cuda_stream
        fn(
            q_rot.data_ptr(),
            kv_cache.data_ptr(),
            block_table.data_ptr(),
            seq_lens.data_ptr(),
            centroids.data_ptr(),
            mid_o.data_ptr(),
            q_rot.stride(0), q_rot.stride(1),
            kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
            block_table.stride(0),
            mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
            num_kv_heads, block_size, num_kv_splits, kv_group_size,
            attn_scale,
            norm_correction,
            input_dtype,
            B, Hq,
            ctypes.c_void_p(stream_ptr),
        )

    @register_fake("tq::hip_stage1_split")
    def hip_stage1_split_fake(
        q_rot, kv_cache, block_table, seq_lens, centroids, mid_o,
        num_kv_heads, block_size, num_kv_splits, kv_group_size,
        attn_scale, norm_correction, input_dtype,
    ) -> None:
        return None

    # --- HIP Fused Stage1+Stage2 (4-warp, bf16/fp16 output) ---
    @torch.library.custom_op("tq::hip_fused", mutates_args=("output",))
    def hip_fused(
        q_rot: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        centroids: torch.Tensor,
        output: torch.Tensor,
        num_kv_heads: int,
        block_size: int,
        kv_group_size: int,
        attn_scale: float,
        norm_correction: int,
        output_dtype: int,       # 0=bf16, 1=fp16
    ) -> None:
        fn = _load_hip_fused()
        if fn is None:
            return
        B, Hq = q_rot.shape[0], q_rot.shape[1]
        stream_ptr = torch.cuda.current_stream(q_rot.device).cuda_stream
        fn(
            q_rot.data_ptr(),
            kv_cache.data_ptr(),
            block_table.data_ptr(),
            seq_lens.data_ptr(),
            centroids.data_ptr(),
            output.data_ptr(),
            q_rot.stride(0), q_rot.stride(1),
            kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
            block_table.stride(0),
            output.stride(0), output.stride(1),
            num_kv_heads, block_size, kv_group_size,
            attn_scale,
            norm_correction,
            output_dtype,
            B, Hq,
            ctypes.c_void_p(stream_ptr),
        )

    @register_fake("tq::hip_fused")
    def hip_fused_fake(
        q_rot, kv_cache, block_table, seq_lens, centroids, output,
        num_kv_heads, block_size, kv_group_size,
        attn_scale, norm_correction, output_dtype,
    ) -> None:
        return None

    # --- HIP Stage2 (f32 output) ---
    @torch.library.custom_op("tq::hip_stage2_f32", mutates_args=("output",))
    def hip_stage2_f32(
        mid_o: torch.Tensor,
        output: torch.Tensor,
        seq_lens: torch.Tensor,
        num_kv_splits: int,
    ) -> None:
        _, fn_f32 = _load_hip_stage2()
        if fn_f32 is None:
            return
        B, Hq = output.shape[0], output.shape[1]
        stream_ptr = torch.cuda.current_stream(mid_o.device).cuda_stream
        fn_f32(
            mid_o.data_ptr(),
            output.data_ptr(),
            seq_lens.data_ptr(),
            mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
            output.stride(0), output.stride(1),
            num_kv_splits,
            B, Hq,
            ctypes.c_void_p(stream_ptr),
        )

    @register_fake("tq::hip_stage2_f32")
    def hip_stage2_f32_fake(mid_o, output, seq_lens, num_kv_splits) -> None:
        return None

    # --- HIP Stage2 (bf16 output — fused f32→bf16 conversion) ---
    @torch.library.custom_op("tq::hip_stage2_bf16", mutates_args=("output",))
    def hip_stage2_bf16(
        mid_o: torch.Tensor,
        output: torch.Tensor,
        seq_lens: torch.Tensor,
        num_kv_splits: int,
    ) -> None:
        fn_bf16, _ = _load_hip_stage2()
        if fn_bf16 is None:
            return
        B, Hq = output.shape[0], output.shape[1]
        stream_ptr = torch.cuda.current_stream(mid_o.device).cuda_stream
        fn_bf16(
            mid_o.data_ptr(),
            output.data_ptr(),
            seq_lens.data_ptr(),
            mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
            output.stride(0), output.stride(1),
            num_kv_splits,
            B, Hq,
            ctypes.c_void_p(stream_ptr),
        )

    @register_fake("tq::hip_stage2_bf16")
    def hip_stage2_bf16_fake(mid_o, output, seq_lens, num_kv_splits) -> None:
        return None


# Try to register custom ops at import time
try:
    _register_hip_custom_ops()
except Exception:
    pass  # Registration failed — fall back to raw ctypes calls


# ---------------------------------------------------------------------------
# Stage 1: Fused TQ score + value accumulation (BLOCK_KV tiled)
# ---------------------------------------------------------------------------


@triton.jit
def _tq_decode_stage1(
    # Precomputed query projection
    Q_rot_ptr,  # [B, Hq, D] float32
    # Compressed KV cache (combined K+V)
    KV_cache_ptr,  # [num_blocks, block_size, Hk, padded_slot] uint8
    # Block table and sequence info
    Block_table_ptr,  # [B, max_num_blocks] int32
    Seq_lens_ptr,  # [B] int32
    # TQ parameters
    Centroids_ptr,  # [n_centroids] float32
    # Output (intermediate for stage2)
    Mid_o_ptr,  # [B, Hq, NUM_KV_SPLITS, D+1] float32
    # Strides
    stride_qb,
    stride_qh,  # Q strides: [B, Hq, D]
    stride_cache_block,
    stride_cache_pos,
    stride_cache_head,  # KV cache
    stride_bt_b,  # block_table stride per batch
    stride_mid_b,
    stride_mid_h,
    stride_mid_s,  # mid_o strides
    # Constexpr dims
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # KV cache block_size (pages)
    NUM_KV_SPLITS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,  # Hq // Hk
    # TQ layout constants
    MSE_BITS: tl.constexpr,  # 3 or 4
    MSE_BYTES: tl.constexpr,  # ceil(D * mse_bits / 8)
    KPS: tl.constexpr,  # key_packed_size
    VQB: tl.constexpr,  # value_quant_bits (4 or 8=FP8)
    VAL_DATA_BYTES: tl.constexpr,  # ceil(D * vqb / 8) or D for FP8
    # Score constants
    ATTN_SCALE: tl.constexpr,  # 1/sqrt(D)
    # Block tile sizes
    BLOCK_D: tl.constexpr,  # next_power_of_2(HEAD_DIM)
    BLOCK_KV: tl.constexpr,  # tokens per tile (16)
    KEY_FP8: tl.constexpr,  # 1 if K is stored as FP8
    NORM_CORRECTION: tl.constexpr = 0,  # 1 = re-normalize centroids
    FP8_E4B15: tl.constexpr = 0,  # 1 = use e4b15 (Ampere/Ada), 0 = e4nv (Hopper+)
):
    bid = tl.program_id(0)  # batch index
    hid = tl.program_id(1)  # q_head index
    sid = tl.program_id(2)  # kv_split index

    kv_head = hid // KV_GROUP_SIZE

    # Sequence length for this batch
    seq_len = tl.load(Seq_lens_ptr + bid)

    # KV split range
    split_len = tl.cdiv(seq_len, NUM_KV_SPLITS)
    split_start = split_len * sid
    split_end = tl.minimum(split_start + split_len, seq_len)

    if split_start >= split_end:
        return

    # Dimension offsets
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    kv_range = tl.arange(0, BLOCK_KV)

    # Load query vector: q_rot — [BLOCK_D] float32
    q_base = bid * stride_qb + hid * stride_qh
    q_rot = tl.load(Q_rot_ptr + q_base + d_offs, mask=d_mask, other=0.0).to(tl.float32)

    # Precompute byte/bit index vectors for MSE gather loads
    if not KEY_FP8:
        mse_bit_off = d_offs * MSE_BITS
        mse_byte_idx = mse_bit_off // 8
        mse_bit_shift = mse_bit_off % 8
        mse_mask = (1 << MSE_BITS) - 1

        # OPTIMIZATION: Preload all 16 centroids to registers
        # This avoids indirect global memory loads in the inner loop
        c0 = tl.load(Centroids_ptr + 0)
        c1 = tl.load(Centroids_ptr + 1)
        c2 = tl.load(Centroids_ptr + 2)
        c3 = tl.load(Centroids_ptr + 3)
        c4 = tl.load(Centroids_ptr + 4)
        c5 = tl.load(Centroids_ptr + 5)
        c6 = tl.load(Centroids_ptr + 6)
        c7 = tl.load(Centroids_ptr + 7)
        c8 = tl.load(Centroids_ptr + 8)
        c9 = tl.load(Centroids_ptr + 9)
        c10 = tl.load(Centroids_ptr + 10)
        c11 = tl.load(Centroids_ptr + 11)
        c12 = tl.load(Centroids_ptr + 12)
        c13 = tl.load(Centroids_ptr + 13)
        c14 = tl.load(Centroids_ptr + 14)
        c15 = tl.load(Centroids_ptr + 15)

    # Precompute value bit/byte index vectors (loop-invariant)
    if VQB == 3:
        val_bit_off = d_offs * 3
        val_byte_idx = val_bit_off // 8
        val_bit_shift = val_bit_off % 8

    # Online softmax accumulators
    m_prev = -float("inf")
    l_prev = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    bt_base = bid * stride_bt_b

    # ================================================================
    # TILED LOOP: process BLOCK_KV tokens per iteration
    # ================================================================
    for start_n in range(split_start, split_end, BLOCK_KV):
        kv_offs = start_n + kv_range
        kv_mask = kv_offs < split_end

        page_idx = kv_offs // BLOCK_SIZE
        page_off = kv_offs % BLOCK_SIZE
        block_nums = tl.load(
            Block_table_ptr + bt_base + page_idx,
            mask=kv_mask,
            other=0,
        )

        slot_bases = (
            block_nums * stride_cache_block
            + page_off * stride_cache_pos
            + kv_head * stride_cache_head
        )

        # ============================================================
        # COMPUTE ATTENTION SCORES: [BLOCK_KV]
        # ============================================================
        if KEY_FP8:
            k_addrs = slot_bases[:, None] + d_offs[None, :]
            k_raw = tl.load(
                KV_cache_ptr + k_addrs,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            )
            if FP8_E4B15:
                k_float = k_raw.to(tl.float8e4b15, bitcast=True).to(tl.float32)
            else:
                k_float = k_raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            # OPTIMIZATION: d_mask is always True when HEAD_DIM == BLOCK_D
            scores = tl.sum(q_rot[None, :] * k_float, axis=1) * ATTN_SCALE
            scores = tl.where(kv_mask, scores, -float("inf"))
        else:
            # MSE unpack + norms
            mse_addrs0 = slot_bases[:, None] + mse_byte_idx[None, :]
            mse_raw0 = tl.load(
                KV_cache_ptr + mse_addrs0,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            mse_raw1 = tl.load(
                KV_cache_ptr + mse_addrs0 + 1,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            raw16 = mse_raw0 | (mse_raw1 << 8)
            mse_idx = (raw16 >> mse_bit_shift[None, :]) & mse_mask

            # Centroid gather + dot product
            # OPTIMIZATION: Use preloaded centroids with lookup table
            c_vals = tl.where(mse_idx == 0, c0,
                     tl.where(mse_idx == 1, c1,
                     tl.where(mse_idx == 2, c2,
                     tl.where(mse_idx == 3, c3,
                     tl.where(mse_idx == 4, c4,
                     tl.where(mse_idx == 5, c5,
                     tl.where(mse_idx == 6, c6,
                     tl.where(mse_idx == 7, c7,
                     tl.where(mse_idx == 8, c8,
                     tl.where(mse_idx == 9, c9,
                     tl.where(mse_idx == 10, c10,
                     tl.where(mse_idx == 11, c11,
                     tl.where(mse_idx == 12, c12,
                     tl.where(mse_idx == 13, c13,
                     tl.where(mse_idx == 14, c14,
                     c15)))))))))))))))

            # Norm correction: re-normalize centroid vector to unit norm
            if NORM_CORRECTION:
                # OPTIMIZATION: d_mask is always True when HEAD_DIM == BLOCK_D
                c_norm_sq = tl.sum(c_vals * c_vals, axis=1)
                c_inv_norm = tl.rsqrt(c_norm_sq + 1e-16)
                c_vals = c_vals * c_inv_norm[:, None]

            # OPTIMIZATION: d_mask is always True when HEAD_DIM == BLOCK_D
            term1 = tl.sum(q_rot[None, :] * c_vals, axis=1)

            # Load norms (fp16 -> fp32): norms are at MSE_BYTES offset
            norm_bases = slot_bases + MSE_BYTES
            n_lo = tl.load(KV_cache_ptr + norm_bases, mask=kv_mask, other=0).to(
                tl.uint16
            )
            n_hi = tl.load(KV_cache_ptr + norm_bases + 1, mask=kv_mask, other=0).to(
                tl.uint16
            )
            vec_norms = (n_lo | (n_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)

            scores = vec_norms * term1 * ATTN_SCALE
            scores = tl.where(kv_mask, scores, -float("inf"))

        # ============================================================
        # ONLINE SOFTMAX UPDATE (block-level)
        # ============================================================
        n_e_max = tl.maximum(tl.max(scores, 0), m_prev)
        re_scale = tl.exp(m_prev - n_e_max)
        p = tl.exp(scores - n_e_max)

        # ============================================================
        # VALUE LOAD + DEQUANTIZE: [BLOCK_KV, BLOCK_D]
        # ============================================================
        val_bases = slot_bases + KPS

        if VQB == 3:
            val_addrs0 = val_bases[:, None] + val_byte_idx[None, :]
            val_raw0 = tl.load(
                KV_cache_ptr + val_addrs0,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            val_raw1 = tl.load(
                KV_cache_ptr + val_addrs0 + 1,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            raw16 = val_raw0 | (val_raw1 << 8)
            v_idx = ((raw16 >> val_bit_shift[None, :]) & 0x7).to(tl.float32)

            sc_bases = val_bases + VAL_DATA_BYTES
            sc_lo = tl.load(KV_cache_ptr + sc_bases, mask=kv_mask, other=0).to(
                tl.uint16
            )
            sc_hi = tl.load(KV_cache_ptr + sc_bases + 1, mask=kv_mask, other=0).to(
                tl.uint16
            )
            v_scales = (
                (sc_lo | (sc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            )
            zr_lo = tl.load(KV_cache_ptr + sc_bases + 2, mask=kv_mask, other=0).to(
                tl.uint16
            )
            zr_hi = tl.load(KV_cache_ptr + sc_bases + 3, mask=kv_mask, other=0).to(
                tl.uint16
            )
            v_zeros = (zr_lo | (zr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            values = v_idx * v_scales[:, None] + v_zeros[:, None]
        else:  # VQB == 4
            vb_idx = d_offs // 2
            vb_shift = (d_offs % 2) * 4
            val_addrs = val_bases[:, None] + vb_idx[None, :]
            val_raw = tl.load(
                KV_cache_ptr + val_addrs,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            v_idx = ((val_raw >> vb_shift[None, :]) & 0xF).to(tl.float32)

            sc_bases = val_bases + VAL_DATA_BYTES
            sc_lo = tl.load(KV_cache_ptr + sc_bases, mask=kv_mask, other=0).to(
                tl.uint16
            )
            sc_hi = tl.load(KV_cache_ptr + sc_bases + 1, mask=kv_mask, other=0).to(
                tl.uint16
            )
            v_scales = (
                (sc_lo | (sc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            )
            zr_lo = tl.load(KV_cache_ptr + sc_bases + 2, mask=kv_mask, other=0).to(
                tl.uint16
            )
            zr_hi = tl.load(KV_cache_ptr + sc_bases + 3, mask=kv_mask, other=0).to(
                tl.uint16
            )
            v_zeros = (zr_lo | (zr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            values = v_idx * v_scales[:, None] + v_zeros[:, None]

        # ============================================================
        # WEIGHTED VALUE ACCUMULATION
        # ============================================================
        acc = acc * re_scale + tl.sum(p[:, None] * values, 0)
        l_prev = l_prev * re_scale + tl.sum(p, 0)
        m_prev = n_e_max

    # Store partial result
    out_base = bid * stride_mid_b + hid * stride_mid_h + sid * stride_mid_s
    safe_l = tl.where(l_prev > 0.0, l_prev, 1.0)
    tl.store(Mid_o_ptr + out_base + d_offs, acc / safe_l, mask=d_mask)
    lse = m_prev + tl.log(safe_l)
    tl.store(Mid_o_ptr + out_base + HEAD_DIM, lse)


# ---------------------------------------------------------------------------
# Pre-dequant kernel: Bulk dequant K (MSE+norms) and V to fp16
# ---------------------------------------------------------------------------


@triton.jit
def _tq_full_dequant_kv(
    KV_cache_ptr,
    Block_table_ptr,
    Centroids_ptr,
    K_out_ptr,  # [B, Hk, max_seq, D] fp16 or bf16
    V_out_ptr,  # [B, Hk, max_seq, D] fp16 or bf16
    stride_ko_b,
    stride_ko_h,
    stride_ko_s,
    stride_vo_b,
    stride_vo_h,
    stride_vo_s,
    stride_cache_block,
    stride_cache_pos,
    stride_cache_head,
    stride_bt_b,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    MSE_BYTES: tl.constexpr,
    KPS: tl.constexpr,
    VQB: tl.constexpr,
    VAL_DATA_BYTES: tl.constexpr,
    MSE_BITS: tl.constexpr,
    N_CENTROIDS: tl.constexpr,
    KEY_FP8: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NORM_CORRECTION: tl.constexpr = 0,
    FP8_E4B15: tl.constexpr = 0,  # 1 = use e4b15 (Ampere/Ada), 0 = e4nv (Hopper+)
    OUT_BF16: tl.constexpr = 0,  # 1 = output bfloat16, 0 = output float16
):
    """Full dequant: reconstruct K (MSE centroids * norm or FP8) and V.

    Output dtype follows OUT_BF16 flag: bf16 when 1, fp16 when 0.
    """
    pos = tl.program_id(0)
    bh = tl.program_id(1)
    bid = bh // NUM_KV_HEADS
    hid = bh % NUM_KV_HEADS

    page_idx = pos // BLOCK_SIZE
    page_off = pos % BLOCK_SIZE
    block_num = tl.load(Block_table_ptr + bid * stride_bt_b + page_idx)
    slot_base = (
        block_num * stride_cache_block
        + page_off * stride_cache_pos
        + hid * stride_cache_head
    )

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM

    # === K dequant ===
    out_dtype = tl.bfloat16 if OUT_BF16 else tl.float16
    ko_base = bid * stride_ko_b + hid * stride_ko_h + pos * stride_ko_s
    if KEY_FP8:
        k_raw = tl.load(KV_cache_ptr + slot_base + d_offs, mask=d_mask, other=0)
        if FP8_E4B15:
            k_recon = k_raw.to(tl.float8e4b15, bitcast=True).to(tl.float32)
        else:
            k_recon = k_raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        tl.store(K_out_ptr + ko_base + d_offs, k_recon.to(out_dtype), mask=d_mask)
    else:
        # MSE unpack (3-bit or 4-bit) + norms
        mse_bit_off = d_offs * MSE_BITS
        mse_byte_idx = mse_bit_off // 8
        mse_bit_shift = mse_bit_off % 8
        mse_umask = (1 << MSE_BITS) - 1

        mse_raw0 = tl.load(
            KV_cache_ptr + slot_base + mse_byte_idx, mask=d_mask, other=0
        ).to(tl.int32)
        mse_raw1 = tl.load(
            KV_cache_ptr + slot_base + mse_byte_idx + 1, mask=d_mask, other=0
        ).to(tl.int32)
        raw16 = mse_raw0 | (mse_raw1 << 8)
        mse_idx = (raw16 >> mse_bit_shift) & mse_umask

        k_mse = tl.load(Centroids_ptr + mse_idx, mask=d_mask, other=0.0)

        # Norm correction: re-normalize centroid vector to unit norm
        if NORM_CORRECTION:
            c_norm_sq = tl.sum(tl.where(d_mask, k_mse * k_mse, 0.0), axis=0)
            c_inv_norm = tl.rsqrt(c_norm_sq + 1e-16)
            k_mse = k_mse * c_inv_norm

        # Norms at MSE_BYTES offset (no QJL bytes)
        norm_base = slot_base + MSE_BYTES
        n_lo = tl.load(KV_cache_ptr + norm_base).to(tl.uint16)
        n_hi = tl.load(KV_cache_ptr + norm_base + 1).to(tl.uint16)
        vec_norm = (n_lo | (n_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)

        k_recon = vec_norm * k_mse
        tl.store(K_out_ptr + ko_base + d_offs, k_recon.to(out_dtype), mask=d_mask)

    # === V dequant ===
    val_base = slot_base + KPS
    if VQB == 4:
        vb_idx = d_offs // 2
        vb_shift = (d_offs % 2) * 4
        val_raw = tl.load(KV_cache_ptr + val_base + vb_idx, mask=d_mask, other=0).to(
            tl.int32
        )
        v_idx = ((val_raw >> vb_shift) & 0xF).to(tl.float32)

        sc_base = val_base + VAL_DATA_BYTES
        sc_lo = tl.load(KV_cache_ptr + sc_base).to(tl.uint16)
        sc_hi = tl.load(KV_cache_ptr + sc_base + 1).to(tl.uint16)
        v_scale = (sc_lo | (sc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        zr_lo = tl.load(KV_cache_ptr + sc_base + 2).to(tl.uint16)
        zr_hi = tl.load(KV_cache_ptr + sc_base + 3).to(tl.uint16)
        v_zero = (zr_lo | (zr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        v_vals = v_idx * v_scale + v_zero
    elif VQB == 3:
        # 3-bit value unpack: 8 values per 3 bytes
        val_bit_off = d_offs * 3
        val_byte_idx = val_bit_off // 8
        val_bit_shift = val_bit_off % 8
        val_raw0 = tl.load(
            KV_cache_ptr + val_base + val_byte_idx, mask=d_mask, other=0
        ).to(tl.int32)
        val_raw1 = tl.load(
            KV_cache_ptr + val_base + val_byte_idx + 1, mask=d_mask, other=0
        ).to(tl.int32)
        raw16 = val_raw0 | (val_raw1 << 8)
        v_idx = ((raw16 >> val_bit_shift) & 0x7).to(tl.float32)

        sc_base = val_base + VAL_DATA_BYTES
        sc_lo = tl.load(KV_cache_ptr + sc_base).to(tl.uint16)
        sc_hi = tl.load(KV_cache_ptr + sc_base + 1).to(tl.uint16)
        v_scale = (sc_lo | (sc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        zr_lo = tl.load(KV_cache_ptr + sc_base + 2).to(tl.uint16)
        zr_hi = tl.load(KV_cache_ptr + sc_base + 3).to(tl.uint16)
        v_zero = (zr_lo | (zr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        v_vals = v_idx * v_scale + v_zero
    else:
        v_vals = tl.zeros([BLOCK_D], dtype=tl.float32)

    vo_base = bid * stride_vo_b + hid * stride_vo_h + pos * stride_vo_s
    tl.store(V_out_ptr + vo_base + d_offs, v_vals.to(out_dtype), mask=d_mask)


# ---------------------------------------------------------------------------
# Stage 2: Reuse from triton_decode_attention.py
# ---------------------------------------------------------------------------
from vllm.v1.attention.ops.triton_decode_attention import (
    _fwd_kernel_stage2,
)

# ---------------------------------------------------------------------------
# Launcher — cached constants + fused GEMM
# ---------------------------------------------------------------------------

_layout_cache: dict = {}


def _get_layout(D, mse_bits, value_quant_bits, key_packed_size):
    """Get cached layout constants."""
    key = (D, mse_bits, value_quant_bits, key_packed_size)
    cfg = _layout_cache.get(key)
    if cfg is None:
        val_data_bytes = math.ceil(D * value_quant_bits / 8)
        cfg = {
            "mse_bytes": math.ceil(D * mse_bits / 8),
            "val_data_bytes": val_data_bytes,
            "mse_bits": mse_bits,
            "n_centroids": 2**mse_bits,
            "BLOCK_D": triton.next_power_of_2(D),
        }
        _layout_cache[key] = cfg
    return cfg


_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16}


def triton_turboquant_decode_attention(
    query: torch.Tensor,  # [B, Hq, D] — original query
    kv_cache: torch.Tensor,  # [num_blocks, block_size, Hk, padded_slot] uint8
    block_table: torch.Tensor,  # [B, max_num_blocks] int32
    seq_lens: torch.Tensor,  # [B] int32
    Pi: torch.Tensor,  # [D, D] float32
    centroids: torch.Tensor,  # [n_centroids] float32
    scale: float,
    mse_bits: int,
    key_packed_size: int,
    value_quant_bits: int,
    key_fp8: bool = False,
    norm_correction: bool = False,
    PiT: torch.Tensor | None = None,  # [D, D] pre-computed Pi.T contiguous
    # Pre-allocated buffers (optional, avoids per-call allocation)
    mid_o_buf: torch.Tensor | None = None,
    output_buf: torch.Tensor | None = None,
    lse_buf: torch.Tensor | None = None,
    q_rot_buf: torch.Tensor | None = None,  # [max_B, Hq, D] pre-allocated
    centroids_f32: torch.Tensor | None = None,  # pre-cached float32 centroids
    buf_holder: object | None = None,
    max_num_kv_splits: int = 32,  # fixed split count (must be constant for cudagraph)
    eager_max_num_kv_splits: int = 32,
    max_seq_len_hint: int = 0,
    allow_adaptive_kv_splits: bool = False,
    v56_max_seq_len: int = 0,  # 0 = disabled; GEMV cost is seq-independent
) -> torch.Tensor:
    """Launch fused TQ decode attention (Triton stage1 + stage2).

    Returns: output tensor [B, Hq, D] in query's dtype.
    """
    # ── Dtype whitelist ──────────────────────────────────────────────
    if query.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"TQ decode: query.dtype must be one of {_SUPPORTED_DTYPES}, "
            f"got {query.dtype}"
        )

    # ── Tensor type enforcement ──────────────────────────────────────
    if block_table.dtype != torch.int32:
        block_table = block_table.to(torch.int32)
    if seq_lens.dtype != torch.int32:
        seq_lens = seq_lens.to(torch.int32)

    B, Hq, D = query.shape
    Hk = kv_cache.shape[2]
    block_size = kv_cache.shape[1]
    kv_group_size = Hq // Hk
    device = query.device

    cfg = _get_layout(D, mse_bits, value_quant_bits, key_packed_size)

    NUM_KV_SPLITS = _resolve_num_kv_splits(
        max_num_kv_splits=max_num_kv_splits,
        eager_max_num_kv_splits=eager_max_num_kv_splits,
        max_seq_len_hint=max_seq_len_hint,
        batch_size=B,
        allow_adaptive_kv_splits=allow_adaptive_kv_splits,
    )

    # Pre-allocate mid_o buffer for Stage1 partial results
    if (
        mid_o_buf is not None
        and mid_o_buf.shape[0] >= B
        and mid_o_buf.shape[2] >= NUM_KV_SPLITS
    ):
        mid_o = mid_o_buf[:B, :Hq, :NUM_KV_SPLITS, :]
    else:
        mid_o = torch.empty(
            B,
            Hq,
            NUM_KV_SPLITS,
            D + 1,
            dtype=torch.float32,
            device=device,
        )
        if buf_holder is not None:
            buf_holder._tq_mid_o_buf = mid_o

    # Pre-cache centroids for HIP kernels
    if centroids_f32 is None:
        centroids_f32 = centroids.float().contiguous()

    # Ensure PiT is available. HIP paths derive native-dtype copies lazily.
    if not key_fp8 and PiT is None:
        PiT = Pi.T.contiguous()

    # -------------------------------------------------------------------
    # Stage 1: Kernel dispatch (3 paths)
    # - FUSED:  short seq (≤512), Grid=(B,Hq), native dtype output
    # - SPLIT:  long seq,  Grid=(B,Hq,splits), 4-warp, mid_o → Stage2
    # - TRITON: fallback for CUDA, FP8 path, or no HIP .so
    #
    # HIP kernels hardcode HEAD_DIM=128, MSE_BYTES=64 (4-bit MSE),
    # KPS=68, VAL_DATA_BYTES=64 (4-bit values).
    # block_size must be power-of-2 (16/32/64/128).
    # -------------------------------------------------------------------
    stream_ptr = torch.cuda.current_stream(device).cuda_stream

    _hip_safe = (
        not key_fp8
        and D == 128
        and mse_bits == 4
        and value_quant_bits == 4
    )

    host_qrot_us = None
    host_stage1_us = None
    host_stage2_us = None
    decode_path = "triton"
    stage2_backend = "triton_f32"
    stage1_custom_op = "none"
    stage2_custom_op = "none"

    # -------------------------------------------------------------------
    # FUSED PATH (V4): Stage1+Stage2 in a single kernel, native-dtype Q.
    # Grid=(B, Hq) — no split dimension, no mid_o buffer.
    # Directly outputs bf16 or fp16 (matching query dtype).  Eliminates:
    #   1. mid_o allocation & memory traffic (up to 8MB write+read)
    #   2. Stage2 kernel launch overhead
    #   3. query.float() cast (5.9µs saved)
    #   4. fp32 PiT GEMM → native dtype GEMM (faster on tensor cores)
    #   5. fp32 q_rot HBM traffic (halved: 2B vs 4B per element)
    # Best when B is large enough for CU saturation and seq is short.
    # -------------------------------------------------------------------
    hip_fused_fn = _load_hip_fused() if (
        _hip_safe
        and _should_use_fused(B, max_seq_len_hint)
    ) else None

    if hip_fused_fn is not None:
        # V4: native-dtype Q path — no fp32 conversion.
        # PiT in query's dtype for native GEMM (bf16×bf16 or fp16×fp16).
        # This eliminates: query.float() cast, fp32 GEMM overhead,
        # fp32 q_rot HBM traffic (halved: 2B vs 4B per element).
        _q_dtype = query.dtype
        PiT_native = getattr(buf_holder, "_tq_PiT_native", None)
        if PiT_native is None or PiT_native.dtype != _q_dtype:
            PiT_native = PiT.to(_q_dtype).contiguous()
            if buf_holder is not None:
                buf_holder._tq_PiT_native = PiT_native

        if (
            q_rot_buf is not None
            and q_rot_buf.shape[0] >= B
            and q_rot_buf.dtype == _q_dtype
        ):
            q_rot = q_rot_buf[:B]
            q_flat = query.reshape(B * Hq, D)
            t0 = time.perf_counter()
            torch.mm(q_flat, PiT_native, out=q_rot.reshape(B * Hq, D))
            host_qrot_us = (time.perf_counter() - t0) * 1e6
        else:
            q_flat = query.reshape(B * Hq, D)
            t0 = time.perf_counter()
            q_rot = (q_flat @ PiT_native).reshape(B, Hq, D).contiguous()
            host_qrot_us = (time.perf_counter() - t0) * 1e6
            if buf_holder is not None:
                buf_holder._tq_q_rot_buf = q_rot

        # Output dtype matches query dtype (bf16 or fp16)
        out_dtype = query.dtype  # torch.bfloat16 or torch.float16
        _output_dtype_flag = 0 if out_dtype == torch.bfloat16 else 1

        if (
            output_buf is not None
            and output_buf.shape[0] >= B
            and output_buf.dtype == out_dtype
        ):
            output = output_buf[:B, :Hq, :D]
        else:
            output = torch.empty(B, Hq, D, dtype=out_dtype, device=device)
            if buf_holder is not None:
                buf_holder._tq_output_buf = output

        _nc = 1 if norm_correction else 0
        _has_fused_op = (
            hasattr(torch.ops, "tq")
            and hasattr(torch.ops.tq, "hip_fused")
        )
        if _has_fused_op:
            t0 = time.perf_counter()
            torch.ops.tq.hip_fused(
                q_rot, kv_cache, block_table, seq_lens,
                centroids_f32, output,
                Hk, block_size, kv_group_size,
                scale, _nc, _output_dtype_flag,
            )
            host_stage1_us = (time.perf_counter() - t0) * 1e6
            stage1_custom_op = "torch_ops"
        else:
            t0 = time.perf_counter()
            hip_fused_fn(
                q_rot.data_ptr(),
                kv_cache.data_ptr(),
                block_table.data_ptr(),
                seq_lens.data_ptr(),
                centroids_f32.data_ptr(),
                output.data_ptr(),
                q_rot.stride(0), q_rot.stride(1),
                kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
                block_table.stride(0),
                output.stride(0), output.stride(1),
                Hk, block_size, kv_group_size,
                scale,
                _nc,
                _output_dtype_flag,
                B, Hq,
                ctypes.c_void_p(stream_ptr),
            )
            host_stage1_us = (time.perf_counter() - t0) * 1e6
            stage1_custom_op = "ctypes"
        decode_path = "hip_fused"
        stage2_backend = "fused"
        stage2_custom_op = "fused"

        record_decode_call(
            batch_size=B,
            max_seq_len=max_seq_len_hint,
            num_kv_splits=0,  # no splits in fused path
            path=decode_path,
            stage2_backend=stage2_backend,
            custom_op=f"{stage1_custom_op}/{stage2_custom_op}",
            output_dtype=str(output.dtype).replace("torch.", ""),
            host_qrot_us=host_qrot_us,
            host_stage1_us=host_stage1_us,
            host_stage2_us=0.0,
        )
        return output

    # -------------------------------------------------------------------
    # SPLIT/UNIFIED PATH: GEMM + Unified Kernel v136
    #
    # v136 unified kernel accepts bf16 Q input and supports dual-mode:
    #   splits > 1: write partials to mid_o → separate Stage2
    #   splits == 1: write final bf16 output directly → no Stage2
    # bf16 GEMM is ~40% cheaper than fp32 GEMM (~15µs vs ~20µs).
    # 1.104x geomean E2E speedup over v132 production.
    # -------------------------------------------------------------------
    # Compute q_rot (bf16 for HIP v136, fp32 for Triton fallback)
    hip_split_fn = _load_hip_split() if _hip_safe else None
    _q_dtype = query.dtype

    if key_fp8:
        q_rot = query.contiguous()
    elif hip_split_fn is not None:
        # v136: bf16 GEMM — native dtype path
        PiT_native = getattr(buf_holder, "_tq_PiT_native", None) if buf_holder else None
        if PiT_native is None or PiT_native.dtype != _q_dtype:
            PiT_native = PiT.to(_q_dtype).contiguous()
            if buf_holder is not None:
                buf_holder._tq_PiT_native = PiT_native

        if (
            q_rot_buf is not None
            and q_rot_buf.shape[0] >= B
            and q_rot_buf.dtype == _q_dtype
        ):
            q_rot = q_rot_buf[:B]
            q_flat = query.reshape(B * Hq, D)
            t0 = time.perf_counter()
            torch.mm(q_flat, PiT_native, out=q_rot.reshape(B * Hq, D))
            host_qrot_us = (time.perf_counter() - t0) * 1e6
        else:
            q_flat = query.reshape(B * Hq, D)
            t0 = time.perf_counter()
            q_rot = (q_flat @ PiT_native).reshape(B, Hq, D).contiguous()
            host_qrot_us = (time.perf_counter() - t0) * 1e6
            if buf_holder is not None:
                buf_holder._tq_q_rot_buf = q_rot
    else:
        # Triton fallback: fp32 GEMM (no dtype param support)
        if (
            q_rot_buf is not None
            and q_rot_buf.shape[0] >= B
            and q_rot_buf.dtype == torch.float32
        ):
            q_rot = q_rot_buf[:B]
            q_flat = query.reshape(B * Hq, D).float()
            t0 = time.perf_counter()
            torch.mm(q_flat, PiT, out=q_rot.reshape(B * Hq, D))
            host_qrot_us = (time.perf_counter() - t0) * 1e6
        else:
            q_float = query.float()
            t0 = time.perf_counter()
            q_rot = (q_float @ PiT).contiguous()
            host_qrot_us = (time.perf_counter() - t0) * 1e6
            if buf_holder is not None:
                buf_holder._tq_q_rot_buf = q_rot

    if hip_split_fn is not None:
        _nc = 1 if norm_correction else 0
        # v136 unified kernel ABI: accepts output ptr for direct-output mode
        # When NUM_KV_SPLITS > 1, output ptr is unused (mid_o path).
        # We pass a placeholder output; real output comes from Stage2.
        _placeholder_output = mid_o  # unused when splits > 1
        t0 = time.perf_counter()
        hip_split_fn(
                q_rot.data_ptr(),
                kv_cache.data_ptr(),
                block_table.data_ptr(),
                seq_lens.data_ptr(),
                centroids_f32.data_ptr(),
                mid_o.data_ptr(),
                _placeholder_output.data_ptr(),
                q_rot.stride(0), q_rot.stride(1),
                kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
                block_table.stride(0),
                mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
                0, 0,  # stride_ob, stride_oh — unused when splits > 1
                Hk, block_size, NUM_KV_SPLITS, kv_group_size,
                scale,
                _nc,
                B, Hq,
                ctypes.c_void_p(stream_ptr),
            )
        host_stage1_us = (time.perf_counter() - t0) * 1e6
        stage1_custom_op = "ctypes"
        decode_path = "hip_unified_v136"
    else:
        # Triton fallback (CUDA, FP8 path, or no HIP .so)
        fp8_e4b15 = _use_fp8_e4b15(device.index or 0)
        BLOCK_KV = 4
        grid = (B, Hq, NUM_KV_SPLITS)
        t0 = time.perf_counter()
        _tq_decode_stage1[grid](
            q_rot,
            kv_cache,
            block_table,
            seq_lens,
            centroids,
            mid_o,
            q_rot.stride(0),
            q_rot.stride(1),
            kv_cache.stride(0),
            kv_cache.stride(1),
            kv_cache.stride(2),
            block_table.stride(0),
            mid_o.stride(0),
            mid_o.stride(1),
            mid_o.stride(2),
            NUM_KV_HEADS=Hk,
            HEAD_DIM=D,
            BLOCK_SIZE=block_size,
            NUM_KV_SPLITS=NUM_KV_SPLITS,
            KV_GROUP_SIZE=kv_group_size,
            MSE_BITS=mse_bits,
            MSE_BYTES=cfg["mse_bytes"],
            KPS=key_packed_size,
            VQB=value_quant_bits,
            VAL_DATA_BYTES=cfg["val_data_bytes"],
            ATTN_SCALE=scale,
            BLOCK_D=cfg["BLOCK_D"],
            BLOCK_KV=BLOCK_KV,
            KEY_FP8=1 if key_fp8 else 0,
            NORM_CORRECTION=1 if norm_correction else 0,
            FP8_E4B15=fp8_e4b15,
            num_warps=1,
            num_stages=1,
        )
        host_stage1_us = (time.perf_counter() - t0) * 1e6
        decode_path = "triton_stage1"

    # -------------------------------------------------------------------
    # Stage 2: Reduce across KV splits
    # Try HIP Stage2 (faster than Triton on ROCm), else Triton fallback.
    #
    # When HIP bf16 Stage2 is available AND query is bf16, produce bf16
    # output directly (saves a separate f32→bf16 cast in the caller).
    # Otherwise produce f32; caller handles dtype conversion.
    # -------------------------------------------------------------------
    hip_s2_bf16, hip_s2_f32 = _load_hip_stage2()

    use_bf16_stage2 = (
        hip_s2_bf16 is not None
        and query.dtype == torch.bfloat16
    )

    if use_bf16_stage2:
        # Direct bf16 output — fused reduce + cast
        if output_buf is not None and output_buf.shape[0] >= B and output_buf.dtype == torch.bfloat16:
            output = output_buf[:B, :Hq, :D]
        else:
            output = torch.empty(B, Hq, D, dtype=torch.bfloat16, device=device)
            if buf_holder is not None:
                buf_holder._tq_output_buf = output

        if hasattr(torch.ops, "tq") and hasattr(torch.ops.tq, "hip_stage2_bf16"):
            t0 = time.perf_counter()
            torch.ops.tq.hip_stage2_bf16(mid_o, output, seq_lens, NUM_KV_SPLITS)
            host_stage2_us = (time.perf_counter() - t0) * 1e6
            stage2_custom_op = "torch_ops"
        else:
            t0 = time.perf_counter()
            hip_s2_bf16(
                mid_o.data_ptr(),
                output.data_ptr(),
                seq_lens.data_ptr(),
                mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
                output.stride(0), output.stride(1),
                NUM_KV_SPLITS,
                B, Hq,
                ctypes.c_void_p(stream_ptr),
            )
            host_stage2_us = (time.perf_counter() - t0) * 1e6
            stage2_custom_op = "ctypes"
        stage2_backend = "hip_bf16"
    elif hip_s2_f32 is not None:
        # f32 output — must verify cached buffer dtype matches
        if (
            output_buf is not None
            and output_buf.shape[0] >= B
            and output_buf.dtype == torch.float32
        ):
            output = output_buf[:B, :Hq, :D]
        else:
            output = torch.empty(B, Hq, D, dtype=torch.float32, device=device)
            if buf_holder is not None:
                buf_holder._tq_output_buf = output

        if hasattr(torch.ops, "tq") and hasattr(torch.ops.tq, "hip_stage2_f32"):
            t0 = time.perf_counter()
            torch.ops.tq.hip_stage2_f32(mid_o, output, seq_lens, NUM_KV_SPLITS)
            host_stage2_us = (time.perf_counter() - t0) * 1e6
            stage2_custom_op = "torch_ops"
        else:
            t0 = time.perf_counter()
            hip_s2_f32(
                mid_o.data_ptr(),
                output.data_ptr(),
                seq_lens.data_ptr(),
                mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
                output.stride(0), output.stride(1),
                NUM_KV_SPLITS,
                B, Hq,
                ctypes.c_void_p(stream_ptr),
            )
            host_stage2_us = (time.perf_counter() - t0) * 1e6
            stage2_custom_op = "ctypes"
        stage2_backend = "hip_f32"
    else:
        # Triton fallback (f32) — must verify cached buffer dtype matches
        if (
            output_buf is not None
            and output_buf.shape[0] >= B
            and output_buf.dtype == torch.float32
        ):
            output = output_buf[:B, :Hq, :D]
        else:
            output = torch.empty(B, Hq, D, dtype=torch.float32, device=device)
            if buf_holder is not None:
                buf_holder._tq_output_buf = output

        if lse_buf is not None and lse_buf.shape[0] >= B:
            lse = lse_buf[:B, :Hq]
        else:
            lse = torch.empty(B, Hq, dtype=torch.float32, device=device)
            if buf_holder is not None:
                buf_holder._tq_lse_buf = lse

        grid2 = (B, Hq)
        t0 = time.perf_counter()
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
            NUM_KV_SPLITS=NUM_KV_SPLITS,
            BLOCK_DV=cfg["BLOCK_D"],
            Lv=D,
            num_warps=4,
            num_stages=2,
        )
        host_stage2_us = (time.perf_counter() - t0) * 1e6
        stage2_backend = "triton_f32"

    record_decode_call(
        batch_size=B,
        max_seq_len=max_seq_len_hint,
        num_kv_splits=NUM_KV_SPLITS,
        path=decode_path,
        stage2_backend=stage2_backend,
        custom_op=f"{stage1_custom_op}/{stage2_custom_op}",
        output_dtype=str(output.dtype).replace("torch.", ""),
        host_qrot_us=host_qrot_us,
        host_stage1_us=host_stage1_us,
        host_stage2_us=host_stage2_us,
    )
    return output
