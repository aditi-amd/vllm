# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Triton kernels for TurboQuant KV store.

Three kernels:
1. _tq_fused_store_fp8: FP8 key scatter + value uniform quantization.
2. _tq_fused_store_mse: Fused bucketize + centroid gather + residual norm
   + MSE index packing + value quantization (eliminates 4 PyTorch kernel
   launches vs the old pack-only approach).
3. _tq_butterfly_store_mse: Fully-fused butterfly store.  Replaces the
   O(D²) PiT GEMV inside _tq_fully_fused_store_mse with an O(D log₂D)
   Walsh-Hadamard butterfly (7 stages for D=128) using tl.gather for the
   in-register cross-element exchange.  No PiT matrix is loaded.
   Gate: VLLM_TQ_STORE_WHT_BUTTERFLY=1.

The launcher `triton_turboquant_store` selects the appropriate kernel.
"""

import ctypes
import math
import os
import time

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_turboquant_decode import (
    _hip_so_is_usable,
    _use_fp8_e4b15,
)
from vllm.v1.attention.ops.turboquant_runtime_stats import record_store_call

# ═══════════════════════════════════════════════════════════════════════
# HIP-fused TQ Store loader (optional, AMD MI300X only)
# ═══════════════════════════════════════════════════════════════════════

_hip_store_fn = None
_hip_store_loaded = False


def _load_hip_store():
    """Load fused HIP TQ store kernel if available."""
    global _hip_store_fn, _hip_store_loaded
    if _hip_store_loaded:
        return _hip_store_fn
    _hip_store_loaded = True

    # Look for the .so in geak_tq_store/hip_kernel/ relative to repo root
    # Try multiple paths
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..",
                     "geak_tq_store", "hip_kernel", "tq_store_fused.so"),
        os.path.join(os.path.dirname(__file__),
                     "tq_store_fused.so"),
    ]
    so_path = None
    for c in candidates:
        c = os.path.realpath(c)
        if _hip_so_is_usable(c, __file__):
            so_path = c
            break

    if so_path is None:
        return None

    try:
        lib = ctypes.CDLL(so_path)
        fn = lib.launch_tq_store_fused
        fn.argtypes = [
            ctypes.c_void_p,  # key
            ctypes.c_void_p,  # value
            ctypes.c_void_p,  # PiT
            ctypes.c_void_p,  # centroids
            ctypes.c_void_p,  # midpoints
            ctypes.c_void_p,  # kv_cache
            ctypes.c_void_p,  # slot_map
            ctypes.c_int,     # stride_block
            ctypes.c_int,     # stride_pos
            ctypes.c_int,     # stride_head
            ctypes.c_int,     # H
            ctypes.c_int,     # block_size
            ctypes.c_int,     # NH
            ctypes.c_int,     # kv_dtype (0=bf16, 1=fp16)
            ctypes.c_void_p,  # stream
        ]
        fn.restype = None
        _hip_store_fn = fn
        return fn
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════
# Custom op wrapper for HIP store (CUDA graph compatibility)
# ═══════════════════════════════════════════════════════════════════════

try:
    from torch.library import register_fake
except ImportError:
    try:
        from torch.library import impl_abstract as register_fake
    except ImportError:
        register_fake = None


def _register_hip_store_custom_op():
    """Register HIP store kernel wrapper as torch custom op (once)."""
    if not current_platform.is_rocm() or register_fake is None:
        return

    @torch.library.custom_op("tq::hip_store_fused", mutates_args=("kv_cache",))
    def hip_store_fused(
        key: torch.Tensor,
        value: torch.Tensor,
        PiT: torch.Tensor,
        centroids: torch.Tensor,
        midpoints: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        stride_block: int,
        stride_pos: int,
        stride_head: int,
        H: int,
        block_size: int,
        NH: int,
        kv_dtype: int,
    ) -> None:
        fn = _load_hip_store()
        if fn is None:
            return
        stream = torch.cuda.current_stream(key.device).cuda_stream
        fn(
            key.data_ptr(), value.data_ptr(),
            PiT.data_ptr(), centroids.data_ptr(), midpoints.data_ptr(),
            kv_cache.data_ptr(), slot_mapping.data_ptr(),
            stride_block, stride_pos, stride_head,
            H, block_size, NH, kv_dtype,
            ctypes.c_void_p(stream),
        )

    @register_fake("tq::hip_store_fused")
    def hip_store_fused_fake(
        key, value, PiT, centroids, midpoints, kv_cache, slot_mapping,
        stride_block, stride_pos, stride_head, H, block_size, NH, kv_dtype,
    ) -> None:
        return None


try:
    _register_hip_store_custom_op()
except Exception:
    pass


# ═══════════════════════════════════════════════════════════════════════
# Shared: value uniform quantization + pack + scale/zero store
# ═══════════════════════════════════════════════════════════════════════


@triton.jit
def _store_quantized_value(
    Value_ptr,
    KV_cache_ptr,
    base,  # pid * D offset into Value_ptr
    slot_base,  # byte offset into KV_cache_ptr for this slot+head
    d_offs,  # tl.arange(0, BLOCK_D)
    d_mask,  # d_offs < D
    D: tl.constexpr,
    KPS: tl.constexpr,
    VQB: tl.constexpr,
    VAL_DATA_BYTES: tl.constexpr,
    BLOCK_VAL: tl.constexpr,
    BLOCK_GRP: tl.constexpr,
):
    """Uniform quantization of values to VQB bits, pack, and store with scale/zero."""
    val_cache_offset = KPS

    if VQB == 3:
        val_vec = tl.load(Value_ptr + base + d_offs, mask=d_mask, other=0.0).to(
            tl.float32
        )
        val_min = tl.min(tl.where(d_mask, val_vec, float("inf")), axis=0)
        val_max = tl.max(tl.where(d_mask, val_vec, -float("inf")), axis=0)
        v_scale = (val_max - val_min) / 7.0
        v_scale = tl.where(v_scale > 1e-8, v_scale, 1e-8)

        q_vals = tl.minimum(
            tl.maximum(((val_vec - val_min) / v_scale + 0.5).to(tl.int32), 0), 7
        )

        grp_offs = tl.arange(0, BLOCK_GRP)
        grp_mask = grp_offs < (D // 8)
        q_grp = tl.reshape(q_vals, [BLOCK_GRP, 8])
        shifts_3bit = tl.arange(0, 8) * 3
        packed_24 = tl.sum(q_grp << shifts_3bit[None, :], axis=1)
        b0 = (packed_24 & 0xFF).to(tl.uint8)
        b1 = ((packed_24 >> 8) & 0xFF).to(tl.uint8)
        b2 = ((packed_24 >> 16) & 0xFF).to(tl.uint8)
        tl.store(
            KV_cache_ptr + slot_base + val_cache_offset + grp_offs * 3,
            b0,
            mask=grp_mask,
        )
        tl.store(
            KV_cache_ptr + slot_base + val_cache_offset + grp_offs * 3 + 1,
            b1,
            mask=grp_mask,
        )
        tl.store(
            KV_cache_ptr + slot_base + val_cache_offset + grp_offs * 3 + 2,
            b2,
            mask=grp_mask,
        )

        sc_offset = val_cache_offset + VAL_DATA_BYTES
        sc_f16 = v_scale.to(tl.float16)
        sc_u16 = sc_f16.to(tl.uint16, bitcast=True)
        tl.store(KV_cache_ptr + slot_base + sc_offset, (sc_u16 & 0xFF).to(tl.uint8))
        tl.store(
            KV_cache_ptr + slot_base + sc_offset + 1,
            ((sc_u16 >> 8) & 0xFF).to(tl.uint8),
        )
        zr_f16 = val_min.to(tl.float16)
        zr_u16 = zr_f16.to(tl.uint16, bitcast=True)
        tl.store(KV_cache_ptr + slot_base + sc_offset + 2, (zr_u16 & 0xFF).to(tl.uint8))
        tl.store(
            KV_cache_ptr + slot_base + sc_offset + 3,
            ((zr_u16 >> 8) & 0xFF).to(tl.uint8),
        )

    else:  # VQB == 4
        val_vec = tl.load(Value_ptr + base + d_offs, mask=d_mask, other=0.0).to(
            tl.float32
        )
        val_min = tl.min(tl.where(d_mask, val_vec, float("inf")), axis=0)
        val_max = tl.max(tl.where(d_mask, val_vec, -float("inf")), axis=0)
        v_scale = (val_max - val_min) / 15.0
        v_scale = tl.where(v_scale > 1e-8, v_scale, 1e-8)

        val_offs = tl.arange(0, BLOCK_VAL)
        val_mask = val_offs < VAL_DATA_BYTES
        v0 = tl.load(
            Value_ptr + base + val_offs * 2,
            mask=val_mask & (val_offs * 2 < D),
            other=val_min,
        )
        v1 = tl.load(
            Value_ptr + base + val_offs * 2 + 1,
            mask=val_mask & (val_offs * 2 + 1 < D),
            other=val_min,
        )
        q0 = tl.minimum(
            tl.maximum(((v0 - val_min) / v_scale + 0.5).to(tl.int32), 0), 15
        )
        q1 = tl.minimum(
            tl.maximum(((v1 - val_min) / v_scale + 0.5).to(tl.int32), 0), 15
        )
        packed_val = (q0 | (q1 << 4)).to(tl.uint8)
        tl.store(
            KV_cache_ptr + slot_base + val_cache_offset + val_offs,
            packed_val,
            mask=val_mask,
        )

        sc_offset = val_cache_offset + VAL_DATA_BYTES
        sc_f16 = v_scale.to(tl.float16)
        sc_u16 = sc_f16.to(tl.uint16, bitcast=True)
        tl.store(KV_cache_ptr + slot_base + sc_offset, (sc_u16 & 0xFF).to(tl.uint8))
        tl.store(
            KV_cache_ptr + slot_base + sc_offset + 1,
            ((sc_u16 >> 8) & 0xFF).to(tl.uint8),
        )
        zr_f16 = val_min.to(tl.float16)
        zr_u16 = zr_f16.to(tl.uint16, bitcast=True)
        tl.store(KV_cache_ptr + slot_base + sc_offset + 2, (zr_u16 & 0xFF).to(tl.uint8))
        tl.store(
            KV_cache_ptr + slot_base + sc_offset + 3,
            ((zr_u16 >> 8) & 0xFF).to(tl.uint8),
        )


# ═══════════════════════════════════════════════════════════════════════
# FP8 key store + value uniform quantization
# ═══════════════════════════════════════════════════════════════════════


@triton.jit
def _tq_fused_store_fp8(
    Key_ptr,  # [NH, D] float16/bfloat16 — raw keys
    Value_ptr,  # [NH, D] float16/bfloat16 — raw values
    KV_cache_ptr,  # [total_bytes] uint8 (flattened view)
    Slot_mapping_ptr,  # [N] int32 — per-token slot indices
    # Cache strides (for computing byte offsets)
    stride_cache_block: tl.constexpr,
    stride_cache_pos: tl.constexpr,
    stride_cache_head: tl.constexpr,
    # Dimensions
    D: tl.constexpr,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    # TQ layout
    KPS: tl.constexpr,
    # Value quantization
    VQB: tl.constexpr,
    VAL_DATA_BYTES: tl.constexpr,
    # Packing block sizes
    BLOCK_VAL: tl.constexpr,
    BLOCK_GRP: tl.constexpr = 16,
    FP8_E4B15: tl.constexpr = 0,  # 1 = e4b15 (Ampere/Ada), 0 = e4nv (Hopper+)
):
    """FP8 key cast+scatter + value uniform quantization: one program per (token, head)."""
    pid = tl.program_id(0)
    token_idx = pid // H
    head_idx = pid % H

    slot = tl.load(Slot_mapping_ptr + token_idx)
    if slot < 0:
        return
    blk = slot // BLOCK_SIZE
    off = slot % BLOCK_SIZE
    slot_base = (
        blk * stride_cache_block + off * stride_cache_pos + head_idx * stride_cache_head
    )

    base = pid * D

    # ── FP8 KEY: cast to FP8 in-kernel and store ─────────────────
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D
    k_vals = tl.load(Key_ptr + base + d_offs, mask=d_mask, other=0.0)
    if FP8_E4B15:
        k_fp8 = k_vals.to(tl.float8e4b15)
    else:
        k_fp8 = k_vals.to(tl.float8e4nv)
    k_bytes = k_fp8.to(tl.uint8, bitcast=True)
    tl.store(KV_cache_ptr + slot_base + d_offs, k_bytes, mask=d_mask)

    # ── VALUE QUANTIZE + PACK ───────────────────────────────────────
    _store_quantized_value(
        Value_ptr,
        KV_cache_ptr,
        base,
        slot_base,
        d_offs,
        d_mask,
        D=D,
        KPS=KPS,
        VQB=VQB,
        VAL_DATA_BYTES=VAL_DATA_BYTES,
        BLOCK_VAL=BLOCK_VAL,
        BLOCK_GRP=BLOCK_GRP,
    )


# ═══════════════════════════════════════════════════════════════════════
# Fused MSE store: bucketize + centroid gather + residual norm + pack
# (eliminates 4 PyTorch kernel launches per layer vs pack-only kernel)
# ═══════════════════════════════════════════════════════════════════════


@triton.jit
def _tq_fused_store_mse(
    # Post-rotation inputs
    Y_ptr,  # [NH, D] float32 — rotated normalized keys (x_hat @ PiT)
    Norms_ptr,  # [NH] float32 — key vector norms (||k||)
    Value_ptr,  # [NH, D] float32 — raw values
    # Quantization tables
    Centroids_ptr,  # [n_centroids] float32
    Midpoints_ptr,  # [n_centroids-1] float32
    # Cache and indexing
    KV_cache_ptr,  # [total_bytes] uint8 (flattened view)
    Slot_mapping_ptr,  # [N] int32 — per-token slot indices
    # Cache strides
    stride_cache_block: tl.constexpr,
    stride_cache_pos: tl.constexpr,
    stride_cache_head: tl.constexpr,
    # Dimensions
    D: tl.constexpr,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    # TQ layout
    MSE_BYTES: tl.constexpr,
    KPS: tl.constexpr,
    # Value quantization
    VQB: tl.constexpr,
    VAL_DATA_BYTES: tl.constexpr,
    # Packing block sizes
    BLOCK_VAL: tl.constexpr,
    # MSE params
    MSE_BITS: tl.constexpr,
    N_CENTROIDS: tl.constexpr,
    BLOCK_GRP: tl.constexpr = 16,
):
    """Fused MSE quantize + pack + store.

    Performs bucketize, centroid gather, residual norm, MSE index packing,
    and value quantization in one kernel — eliminates 4 PyTorch kernel
    launches (bucketize, gather, subtract, norm) per layer vs pack-only.
    """
    pid = tl.program_id(0)
    token_idx = pid // H
    head_idx = pid % H

    slot = tl.load(Slot_mapping_ptr + token_idx)
    if slot < 0:
        return
    blk = slot // BLOCK_SIZE
    off = slot % BLOCK_SIZE
    slot_base = (
        blk * stride_cache_block + off * stride_cache_pos + head_idx * stride_cache_head
    )

    base = pid * D
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    # ── 1. INLINE BUCKETIZE ──────────────────────────────────────────
    y_vec = tl.load(Y_ptr + base + d_offs, mask=d_mask, other=0.0)
    idx = tl.zeros([BLOCK_D], dtype=tl.int32)
    for i in range(N_CENTROIDS - 1):
        mid_val = tl.load(Midpoints_ptr + i)
        idx += tl.where(y_vec >= mid_val, 1, 0)

    # ── 2. CENTROID GATHER + RESIDUAL NORM ────────────────────────────
    centroid_vals = tl.load(Centroids_ptr + idx, mask=d_mask, other=0.0)
    residual = y_vec - centroid_vals
    gamma = tl.sqrt(tl.sum(tl.where(d_mask, residual * residual, 0.0), axis=0))

    # ── 3. PACK MSE INDICES from register idx ─────────────────────────
    if MSE_BITS == 4:
        idx_pairs = tl.reshape(idx, [BLOCK_D // 2, 2])
        shifts_4 = tl.arange(0, 2) * 4
        packed = tl.sum((idx_pairs & 0xF) << shifts_4[None, :], axis=1).to(tl.uint8)
        mse_offs = tl.arange(0, BLOCK_D // 2)
        mse_mask = mse_offs < MSE_BYTES
        tl.store(KV_cache_ptr + slot_base + mse_offs, packed, mask=mse_mask)

    elif MSE_BITS == 3:
        grp_offs = tl.arange(0, BLOCK_GRP)
        grp_mask = grp_offs < (D // 8)
        idx_grp = tl.reshape(idx, [BLOCK_GRP, 8])
        shifts_3 = tl.arange(0, 8) * 3
        packed_24 = tl.sum((idx_grp & 0x7) << shifts_3[None, :], axis=1)
        b0 = (packed_24 & 0xFF).to(tl.uint8)
        b1 = ((packed_24 >> 8) & 0xFF).to(tl.uint8)
        b2 = ((packed_24 >> 16) & 0xFF).to(tl.uint8)
        tl.store(KV_cache_ptr + slot_base + grp_offs * 3, b0, mask=grp_mask)
        tl.store(KV_cache_ptr + slot_base + grp_offs * 3 + 1, b1, mask=grp_mask)
        tl.store(KV_cache_ptr + slot_base + grp_offs * 3 + 2, b2, mask=grp_mask)

    # ── 4. STORE NORMS (vec_norm + gamma as fp16) ─────────────────────
    norm_offset = MSE_BYTES

    vn_f16 = tl.load(Norms_ptr + pid).to(tl.float16)
    vn_u16 = vn_f16.to(tl.uint16, bitcast=True)
    tl.store(KV_cache_ptr + slot_base + norm_offset, (vn_u16 & 0xFF).to(tl.uint8))
    tl.store(
        KV_cache_ptr + slot_base + norm_offset + 1, ((vn_u16 >> 8) & 0xFF).to(tl.uint8)
    )

    gm_f16 = gamma.to(tl.float16)
    gm_u16 = gm_f16.to(tl.uint16, bitcast=True)
    tl.store(KV_cache_ptr + slot_base + norm_offset + 2, (gm_u16 & 0xFF).to(tl.uint8))
    tl.store(
        KV_cache_ptr + slot_base + norm_offset + 3, ((gm_u16 >> 8) & 0xFF).to(tl.uint8)
    )

    # ── 5. VALUE QUANTIZE + PACK ──────────────────────────────────────
    _store_quantized_value(
        Value_ptr,
        KV_cache_ptr,
        base,
        slot_base,
        d_offs,
        d_mask,
        D=D,
        KPS=KPS,
        VQB=VQB,
        VAL_DATA_BYTES=VAL_DATA_BYTES,
        BLOCK_VAL=BLOCK_VAL,
        BLOCK_GRP=BLOCK_GRP,
    )


# ═══════════════════════════════════════════════════════════════════════
# Fully-fused MSE store: raw key → norm → normalize → PiT rotation
#   → bucketize → centroid gather → residual norm → pack → value quant
# Single kernel launch replaces 3 PyTorch ops + 1 Triton kernel.
# ═══════════════════════════════════════════════════════════════════════


@triton.jit
def _tq_fully_fused_store_mse(
    # Raw inputs (bf16/fp16)
    Key_ptr,        # [NH, D] — raw keys
    Value_ptr,      # [NH, D] — raw values
    # Rotation matrix
    PiT_ptr,        # [D, D] float32 — rotation matrix (contiguous)
    # Quantization tables
    Centroids_ptr,  # [n_centroids] float32
    Midpoints_ptr,  # [n_centroids-1] float32
    # Cache and indexing
    KV_cache_ptr,   # [total_bytes] uint8 (flattened view)
    Slot_mapping_ptr,  # [N] int32
    # Cache strides
    stride_cache_block: tl.constexpr,
    stride_cache_pos: tl.constexpr,
    stride_cache_head: tl.constexpr,
    # Dimensions
    D: tl.constexpr,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    # TQ layout
    MSE_BYTES: tl.constexpr,
    KPS: tl.constexpr,
    # Value quantization
    VQB: tl.constexpr,
    VAL_DATA_BYTES: tl.constexpr,
    # Packing block sizes
    BLOCK_VAL: tl.constexpr,
    # MSE params
    MSE_BITS: tl.constexpr,
    N_CENTROIDS: tl.constexpr,
    BLOCK_GRP: tl.constexpr = 16,
):
    """Fully-fused MSE store: raw key to compressed cache in one kernel.

    Fuses: float cast + norm + normalize + PiT rotation + bucketize +
    centroid gather + residual norm + MSE pack + value quant + store.
    Eliminates 3 PyTorch kernel launches (norm, div, mm) per layer.
    """
    pid = tl.program_id(0)
    token_idx = pid // H
    head_idx = pid % H

    slot = tl.load(Slot_mapping_ptr + token_idx)
    if slot < 0:
        return
    blk = slot // BLOCK_SIZE
    off = slot % BLOCK_SIZE
    slot_base = (
        blk * stride_cache_block + off * stride_cache_pos
        + head_idx * stride_cache_head
    )

    base = pid * D
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    # ── 0. LOAD RAW KEY + NORM + NORMALIZE ─────────────────────────
    k_vec = tl.load(Key_ptr + base + d_offs, mask=d_mask, other=0.0).to(
        tl.float32
    )
    # Compute L2 norm
    k_sq_sum = tl.sum(tl.where(d_mask, k_vec * k_vec, 0.0), axis=0)
    vec_norm = tl.sqrt(k_sq_sum)
    # Normalize
    inv_norm = 1.0 / (vec_norm + 1e-8)
    x_hat = k_vec * inv_norm

    # ── 0b. ROTATE: y = x_hat @ PiT (in-kernel dot products) ──────
    # For each output dim j: y[j] = sum_i x_hat[i] * PiT[i, j]
    # We compute all D output values using a register-tiled approach.
    y_vec = tl.zeros([BLOCK_D], dtype=tl.float32)
    for i in range(D):
        xi = tl.sum(tl.where(d_offs == i, x_hat, 0.0), axis=0)
        pi_row = tl.load(PiT_ptr + i * D + d_offs, mask=d_mask, other=0.0)
        y_vec += xi * pi_row

    # ── 1. INLINE BUCKETIZE ──────────────────────────────────────────
    idx = tl.zeros([BLOCK_D], dtype=tl.int32)
    for i in range(N_CENTROIDS - 1):
        mid_val = tl.load(Midpoints_ptr + i)
        idx += tl.where(y_vec >= mid_val, 1, 0)

    # ── 2. CENTROID GATHER + RESIDUAL NORM ────────────────────────────
    centroid_vals = tl.load(Centroids_ptr + idx, mask=d_mask, other=0.0)
    residual = y_vec - centroid_vals
    gamma = tl.sqrt(tl.sum(tl.where(d_mask, residual * residual, 0.0), axis=0))

    # ── 3. PACK MSE INDICES ───────────────────────────────────────────
    if MSE_BITS == 4:
        idx_pairs = tl.reshape(idx, [BLOCK_D // 2, 2])
        shifts_4 = tl.arange(0, 2) * 4
        packed = tl.sum((idx_pairs & 0xF) << shifts_4[None, :], axis=1).to(
            tl.uint8
        )
        mse_offs = tl.arange(0, BLOCK_D // 2)
        mse_mask = mse_offs < MSE_BYTES
        tl.store(KV_cache_ptr + slot_base + mse_offs, packed, mask=mse_mask)

    elif MSE_BITS == 3:
        grp_offs = tl.arange(0, BLOCK_GRP)
        grp_mask = grp_offs < (D // 8)
        idx_grp = tl.reshape(idx, [BLOCK_GRP, 8])
        shifts_3 = tl.arange(0, 8) * 3
        packed_24 = tl.sum((idx_grp & 0x7) << shifts_3[None, :], axis=1)
        b0 = (packed_24 & 0xFF).to(tl.uint8)
        b1 = ((packed_24 >> 8) & 0xFF).to(tl.uint8)
        b2 = ((packed_24 >> 16) & 0xFF).to(tl.uint8)
        tl.store(KV_cache_ptr + slot_base + grp_offs * 3, b0, mask=grp_mask)
        tl.store(KV_cache_ptr + slot_base + grp_offs * 3 + 1, b1, mask=grp_mask)
        tl.store(KV_cache_ptr + slot_base + grp_offs * 3 + 2, b2, mask=grp_mask)

    # ── 4. STORE NORMS ────────────────────────────────────────────────
    norm_offset = MSE_BYTES

    vn_f16 = vec_norm.to(tl.float16)
    vn_u16 = vn_f16.to(tl.uint16, bitcast=True)
    tl.store(KV_cache_ptr + slot_base + norm_offset, (vn_u16 & 0xFF).to(tl.uint8))
    tl.store(
        KV_cache_ptr + slot_base + norm_offset + 1,
        ((vn_u16 >> 8) & 0xFF).to(tl.uint8),
    )

    gm_f16 = gamma.to(tl.float16)
    gm_u16 = gm_f16.to(tl.uint16, bitcast=True)
    tl.store(KV_cache_ptr + slot_base + norm_offset + 2, (gm_u16 & 0xFF).to(tl.uint8))
    tl.store(
        KV_cache_ptr + slot_base + norm_offset + 3,
        ((gm_u16 >> 8) & 0xFF).to(tl.uint8),
    )

    # ── 5. VALUE QUANTIZE + PACK ──────────────────────────────────────
    _store_quantized_value(
        Value_ptr,
        KV_cache_ptr,
        base,
        slot_base,
        d_offs,
        d_mask,
        D=D,
        KPS=KPS,
        VQB=VQB,
        VAL_DATA_BYTES=VAL_DATA_BYTES,
        BLOCK_VAL=BLOCK_VAL,
        BLOCK_GRP=BLOCK_GRP,
    )


# ═══════════════════════════════════════════════════════════════════════
# Butterfly-fused MSE store (VLLM_TQ_STORE_WHT_BUTTERFLY=1)
#
# Identical to _tq_fully_fused_store_mse except the O(D²) scalar GEMV
# rotation loop is replaced by a 7-stage Walsh-Hadamard butterfly:
#
#   for stage in range(LOG2_D):          # 7 iterations, unrolled
#       mask   = 1 << stage              # 1,2,4,8,16,32,64
#       pidx   = d_offs XOR mask         # partner index (in-register)
#       pval   = tl.gather(y, pidx, 0)   # fetch partner value
#       sign   = +1 if (d & mask)==0 else -1
#       y      = (y + sign * pval) * INV_SQRT2
#
# Total: 7 × 128 = 896 additions.  No PiT matrix is loaded from HBM.
# The result is mathematically identical to y = x_hat @ H (normalised
# Hadamard matrix), which equals x_hat @ PiT because PiT == H for TQ.
# ═══════════════════════════════════════════════════════════════════════

@triton.jit
def _tq_butterfly_store_mse(
    # Raw inputs (bf16/fp16)
    Key_ptr,        # [NH, D]
    Value_ptr,      # [NH, D]
    # Quantization tables (PiT NOT needed — butterfly needs no matrix)
    Centroids_ptr,  # [n_centroids] float32
    Midpoints_ptr,  # [n_centroids-1] float32
    # Cache and indexing
    KV_cache_ptr,         # [total_bytes] uint8 (flattened view)
    Slot_mapping_ptr,     # [N] int64
    # Cache strides
    stride_cache_block: tl.constexpr,
    stride_cache_pos: tl.constexpr,
    stride_cache_head: tl.constexpr,
    # Dimensions
    D: tl.constexpr,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    LOG2_D: tl.constexpr,   # log2(D) — number of butterfly stages
    # TQ layout
    MSE_BYTES: tl.constexpr,
    KPS: tl.constexpr,
    # Value quantization
    VQB: tl.constexpr,
    VAL_DATA_BYTES: tl.constexpr,
    # Packing block sizes
    BLOCK_VAL: tl.constexpr,
    # MSE params
    MSE_BITS: tl.constexpr,
    N_CENTROIDS: tl.constexpr,
    BLOCK_GRP: tl.constexpr = 16,
):
    """Fully-fused MSE store with WHT butterfly rotation.

    Replaces the O(D²) PiT GEMV in _tq_fully_fused_store_mse with a
    7-stage in-register Walsh-Hadamard butterfly (O(D log₂D) = 896 ops
    for D=128).  All other steps are identical: load → norm → normalize
    → butterfly → bucketize → pack → store norms → value quant.
    """
    pid = tl.program_id(0)
    token_idx = pid // H
    head_idx = pid % H

    slot = tl.load(Slot_mapping_ptr + token_idx)
    if slot < 0:
        return
    blk = slot // BLOCK_SIZE
    off = slot % BLOCK_SIZE
    slot_base = (
        blk * stride_cache_block + off * stride_cache_pos
        + head_idx * stride_cache_head
    )

    base = pid * D
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    # ── 0. LOAD RAW KEY + NORM + NORMALIZE ──────────────────────────
    k_vec = tl.load(Key_ptr + base + d_offs, mask=d_mask, other=0.0).to(
        tl.float32
    )
    k_sq_sum = tl.sum(tl.where(d_mask, k_vec * k_vec, 0.0), axis=0)
    vec_norm = tl.sqrt(k_sq_sum)
    inv_norm = 1.0 / (vec_norm + 1e-8)
    y_vec = k_vec * inv_norm   # unit-normalised key, shape [BLOCK_D]

    # ── 1. WHT BUTTERFLY: O(D log₂D), no HBM loads ──────────────────
    # Each of LOG2_D stages (7 for D=128) pairs every element d with
    # its XOR-partner d^mask.  The even element of each pair adds, the
    # odd element subtracts, and both scale by 1/√2.  After all stages
    # every output is a weighted ±1 sum of all inputs — equivalent to
    # multiplying by the normalised D×D Hadamard matrix H == PiT.
    for _stage in range(LOG2_D):   # compile-time unrolled (LOG2_D constexpr)
        _bmask = 1 << _stage           # 1, 2, 4, 8, 16, 32, 64
        _pidx  = (d_offs ^ _bmask).to(tl.int32)      # partner index
        _pval  = tl.gather(y_vec, _pidx, 0)           # fetch partner
        # sign applied to SELF: LOW element adds (+self + partner),
        # HIGH element subtracts (-self + partner) → matches k @ H_standard.
        _sign  = tl.where((d_offs & _bmask) == 0, 1.0, -1.0).to(tl.float32)
        y_vec  = (_sign * y_vec + _pval) * 0.7071067811865476  # * 1/sqrt(2)

    # Zero padding slots (BLOCK_D may be larger than D if D not power-of-2,
    # but for D=128 BLOCK_D==D so this is a no-op).
    y_vec = tl.where(d_mask, y_vec, 0.0)

    # ── 2. INLINE BUCKETIZE ─────────────────────────────────────────
    idx = tl.zeros([BLOCK_D], dtype=tl.int32)
    for i in range(N_CENTROIDS - 1):
        mid_val = tl.load(Midpoints_ptr + i)
        idx += tl.where(y_vec >= mid_val, 1, 0)

    # ── 3. CENTROID GATHER + RESIDUAL NORM ──────────────────────────
    centroid_vals = tl.load(Centroids_ptr + idx, mask=d_mask, other=0.0)
    residual = y_vec - centroid_vals
    gamma = tl.sqrt(tl.sum(tl.where(d_mask, residual * residual, 0.0), axis=0))

    # ── 4. PACK MSE INDICES ─────────────────────────────────────────
    if MSE_BITS == 4:
        idx_pairs = tl.reshape(idx, [BLOCK_D // 2, 2])
        shifts_4 = tl.arange(0, 2) * 4
        packed = tl.sum((idx_pairs & 0xF) << shifts_4[None, :], axis=1).to(
            tl.uint8
        )
        mse_offs = tl.arange(0, BLOCK_D // 2)
        mse_mask = mse_offs < MSE_BYTES
        tl.store(KV_cache_ptr + slot_base + mse_offs, packed, mask=mse_mask)

    elif MSE_BITS == 3:
        grp_offs = tl.arange(0, BLOCK_GRP)
        grp_mask = grp_offs < (D // 8)
        idx_grp = tl.reshape(idx, [BLOCK_GRP, 8])
        shifts_3 = tl.arange(0, 8) * 3
        packed_24 = tl.sum((idx_grp & 0x7) << shifts_3[None, :], axis=1)
        b0 = (packed_24 & 0xFF).to(tl.uint8)
        b1 = ((packed_24 >> 8) & 0xFF).to(tl.uint8)
        b2 = ((packed_24 >> 16) & 0xFF).to(tl.uint8)
        tl.store(KV_cache_ptr + slot_base + grp_offs * 3, b0, mask=grp_mask)
        tl.store(KV_cache_ptr + slot_base + grp_offs * 3 + 1, b1, mask=grp_mask)
        tl.store(KV_cache_ptr + slot_base + grp_offs * 3 + 2, b2, mask=grp_mask)

    # ── 5. STORE NORMS (vec_norm + gamma as fp16) ───────────────────
    norm_offset = MSE_BYTES

    vn_f16 = vec_norm.to(tl.float16)
    vn_u16 = vn_f16.to(tl.uint16, bitcast=True)
    tl.store(KV_cache_ptr + slot_base + norm_offset, (vn_u16 & 0xFF).to(tl.uint8))
    tl.store(
        KV_cache_ptr + slot_base + norm_offset + 1,
        ((vn_u16 >> 8) & 0xFF).to(tl.uint8),
    )

    gm_f16 = gamma.to(tl.float16)
    gm_u16 = gm_f16.to(tl.uint16, bitcast=True)
    tl.store(KV_cache_ptr + slot_base + norm_offset + 2, (gm_u16 & 0xFF).to(tl.uint8))
    tl.store(
        KV_cache_ptr + slot_base + norm_offset + 3,
        ((gm_u16 >> 8) & 0xFF).to(tl.uint8),
    )

    # ── 6. VALUE QUANTIZE + PACK ────────────────────────────────────
    _store_quantized_value(
        Value_ptr,
        KV_cache_ptr,
        base,
        slot_base,
        d_offs,
        d_mask,
        D=D,
        KPS=KPS,
        VQB=VQB,
        VAL_DATA_BYTES=VAL_DATA_BYTES,
        BLOCK_VAL=BLOCK_VAL,
        BLOCK_GRP=BLOCK_GRP,
    )


# ═══════════════════════════════════════════════════════════════════════
# Launcher
# ═══════════════════════════════════════════════════════════════════════

# Read once at import time so the gate is zero-overhead at call time.
_USE_STORE_BUTTERFLY: bool = (
    os.environ.get("VLLM_TQ_STORE_WHT_BUTTERFLY", "0") == "1"
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16}


def triton_turboquant_store(
    key: torch.Tensor,  # [N, H, D] — raw keys (post-RoPE)
    value: torch.Tensor,  # [N, H, D] — raw values
    kv_cache: torch.Tensor,  # [num_blocks, block_size, Hk, padded_slot] uint8
    slot_mapping: torch.Tensor,  # [N] int64
    PiT: torch.Tensor,  # [D, D] float32
    centroids: torch.Tensor,  # [n_centroids] float32
    midpoints: torch.Tensor,  # [n_centroids-1] float32
    mse_bits: int,
    key_packed_size: int,
    value_quant_bits: int,
    key_fp8: bool = False,
):
    """Launch TQ store kernel — FP8 uses _tq_fused_store_fp8, MSE uses _tq_fused_store_mse."""
    # ── Dtype whitelist ──────────────────────────────────────────────
    if key.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"TQ store: key.dtype must be one of {_SUPPORTED_DTYPES}, "
            f"got {key.dtype}"
        )
    if value.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"TQ store: value.dtype must be one of {_SUPPORTED_DTYPES}, "
            f"got {value.dtype}"
        )
    if key.dtype != value.dtype:
        raise ValueError(
            f"TQ store: key.dtype ({key.dtype}) must match value.dtype "
            f"({value.dtype})"
        )

    # ── Tensor type enforcement ──────────────────────────────────────
    if slot_mapping.dtype != torch.int64:
        slot_mapping = slot_mapping.to(torch.int64)

    N, H, D = key.shape
    NH = N * H
    block_size = kv_cache.shape[1]
    num_kv_heads = kv_cache.shape[2]
    padded_slot = kv_cache.shape[3]
    BLOCK_D = triton.next_power_of_2(D)
    mse_bytes = math.ceil(D * mse_bits / 8)
    n_centroids = 2**mse_bits

    val_data_bytes = math.ceil(D * value_quant_bits / 8)

    BLOCK_VAL = triton.next_power_of_2(val_data_bytes)

    # Cache strides
    stride_block = block_size * num_kv_heads * padded_slot
    stride_pos = num_kv_heads * padded_slot
    stride_head = padded_slot

    block_grp = triton.next_power_of_2(D // 8) if D >= 8 else 1

    # ── BUTTERFLY PATH: fused WHT rotation (VLLM_TQ_STORE_WHT_BUTTERFLY=1) ──
    # Requires D to be a power of 2 (always true for TQ: D=64 or D=128).
    # Replaces the O(D²) PiT GEMV with a 7-stage in-register WHT butterfly.
    # No PiT matrix is loaded.  Skipped for FP8 keys (no rotation needed).
    if _USE_STORE_BUTTERFLY and not key_fp8 and (D & (D - 1)) == 0:
        log2_d = int(math.log2(D))
        k_flat = key.reshape(NH, D).contiguous()
        v_flat = value.reshape(NH, D)
        grid = (NH,)
        t0 = time.perf_counter()
        _tq_butterfly_store_mse[grid](
            k_flat,
            v_flat,
            centroids,
            midpoints,
            kv_cache.view(-1),
            slot_mapping,
            stride_cache_block=stride_block,
            stride_cache_pos=stride_pos,
            stride_cache_head=stride_head,
            D=D,
            H=H,
            BLOCK_SIZE=block_size,
            BLOCK_D=BLOCK_D,
            LOG2_D=log2_d,
            MSE_BYTES=mse_bytes,
            KPS=key_packed_size,
            VQB=value_quant_bits,
            VAL_DATA_BYTES=val_data_bytes,
            BLOCK_VAL=BLOCK_VAL,
            MSE_BITS=mse_bits,
            N_CENTROIDS=n_centroids,
            BLOCK_GRP=block_grp,
            num_warps=4,
            num_stages=1,
        )
        record_store_call(
            backend="triton_butterfly",
            host_store_us=(time.perf_counter() - t0) * 1e6,
        )
        return

    # ── FP8 PATH: in-kernel FP8 cast + scatter via fp8 kernel ──
    if key_fp8:
        k_flat = key.reshape(NH, D).contiguous()
        v_flat = value.reshape(NH, D).contiguous()

        fp8_e4b15 = _use_fp8_e4b15(key.device.index or 0)

        grid = (NH,)
        t0 = time.perf_counter()
        _tq_fused_store_fp8[grid](
            k_flat,
            v_flat,
            kv_cache.view(-1),
            slot_mapping,
            stride_cache_block=stride_block,
            stride_cache_pos=stride_pos,
            stride_cache_head=stride_head,
            D=D,
            H=H,
            BLOCK_SIZE=block_size,
            BLOCK_D=BLOCK_D,
            KPS=key_packed_size,
            VQB=value_quant_bits,
            VAL_DATA_BYTES=val_data_bytes,
            BLOCK_VAL=BLOCK_VAL,
            BLOCK_GRP=block_grp,
            FP8_E4B15=fp8_e4b15,
            num_warps=4,
            num_stages=1,
        )
        record_store_call(
            backend="triton_fp8",
            host_store_us=(time.perf_counter() - t0) * 1e6,
        )
        return

    # ── MSE PATH: try fused HIP kernel first (6-9x faster) ──
    hip_fn = _load_hip_store()
    if hip_fn is not None and D == 128 and mse_bits == 4 and value_quant_bits == 4:
        # Fully fused HIP path: bf16/fp16→norm→GEMV→bucketize→pack in single kernel
        k_flat = key.reshape(NH, D).contiguous()
        v_flat = value.reshape(NH, D).contiguous()
        kv_dtype_flag = 0 if key.dtype == torch.bfloat16 else 1  # 0=bf16, 1=fp16

        if hasattr(torch.ops, "tq") and hasattr(torch.ops.tq, "hip_store_fused"):
            t0 = time.perf_counter()
            torch.ops.tq.hip_store_fused(
                k_flat, v_flat, PiT, centroids, midpoints,
                kv_cache, slot_mapping,
                stride_block, stride_pos, stride_head,
                H, block_size, NH, kv_dtype_flag,
            )
            host_store_us = (time.perf_counter() - t0) * 1e6
            backend = "hip_store_torch_ops"
        else:
            stream = torch.cuda.current_stream().cuda_stream
            t0 = time.perf_counter()
            hip_fn(
                k_flat.data_ptr(),
                v_flat.data_ptr(),
                PiT.data_ptr(),
                centroids.data_ptr(),
                midpoints.data_ptr(),
                kv_cache.data_ptr(),
                slot_mapping.data_ptr(),
                stride_block, stride_pos, stride_head,
                H, block_size, NH,
                kv_dtype_flag,
                ctypes.c_void_p(stream),
            )
            host_store_us = (time.perf_counter() - t0) * 1e6
            backend = "hip_store_ctypes"
        record_store_call(backend=backend, host_store_us=host_store_us)
        return

    # ── MSE PATH (fallback): external GEMM + fused bucketize/pack kernel ──
    # Normalize + rotation GEMM externally (cuBLAS/hipBLAS is faster
    # than in-kernel for large batches)
    k_flat = key.float().reshape(NH, D)
    norms = k_flat.norm(dim=1, keepdim=True)
    y = torch.mm(k_flat, PiT)  # k @ PiT (not normalized)
    y = y / (norms + 1e-8)     # normalize after matmul (saves one op)
    y = y.contiguous()

    v_flat = value.float().reshape(NH, D)

    # Fused kernel: bucketize + centroid gather + residual norm + pack
    grid = (NH,)
    t0 = time.perf_counter()
    _tq_fused_store_mse[grid](
        y,
        norms.squeeze(1),
        v_flat,
        centroids,
        midpoints,
        kv_cache.view(-1),
        slot_mapping,
        stride_cache_block=stride_block,
        stride_cache_pos=stride_pos,
        stride_cache_head=stride_head,
        D=D,
        H=H,
        BLOCK_SIZE=block_size,
        BLOCK_D=BLOCK_D,
        MSE_BYTES=mse_bytes,
        KPS=key_packed_size,
        VQB=value_quant_bits,
        VAL_DATA_BYTES=val_data_bytes,
        BLOCK_VAL=BLOCK_VAL,
        MSE_BITS=mse_bits,
        N_CENTROIDS=n_centroids,
        BLOCK_GRP=block_grp,
        num_warps=4,
        num_stages=1,
    )
    record_store_call(
        backend="triton_mse",
        host_store_us=(time.perf_counter() - t0) * 1e6,
    )
