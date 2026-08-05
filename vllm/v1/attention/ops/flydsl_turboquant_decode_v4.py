# SPDX-License-Identifier: Apache-2.0
"""FlyDSL TurboQuant decode v4 launcher (vLLM-side).

Drop-in replacement for ``triton_turboquant_decode_attention_v3`` for the
TQ decode profile HEAD_SIZE=128, MSE_BITS=4 K, VQB=4 V, N_CENTROIDS=16,
BLOCK_SIZE in {16, 32}.

Per-GQA kernel dispatch:
  * GQA group ∈ {8, 16} → canonical kernels.tq_decode_v4 (Qwen-class)
  * GQA group == 6      → kernels.tq_decode_v4_gqa6 sibling (MiniMax-M2.5)

Opt-in via ``VLLM_TQ_DECODE_V4=1``. Falls back to v3 if FlyDSL is not
importable (e.g. wrong arch, missing build tree). The GQA-6 sibling is
imported best-effort: missing it does not affect Qwen GQA-{8,16} paths.

Architecture:
    1. Q rotation: ``q_rot = (query.float() @ PiT).bfloat16()`` — same
       launcher-side rocBLAS GEMM v3 uses with ``VLLM_TQ_FUSE_Q_ROT=0``.
    2. FlyDSL kernel writes per-partition outputs into
       ``[N, Hk, P, QG, D]`` bf16 + ``[N, Hk, P, QG]`` fp32 max/sum buffers.
    3. A small Triton reducer combines partitions in v4's native layout
       (no permute/cast), writing the final ``[N, Hq, D]`` output.

Kernel module is built once per ``(num_kv_heads, num_partitions,
max_blocks_per_seq, scale)`` shape and cached.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Any

import torch
import triton
import triton.language as tl

from vllm.logger import init_logger

logger = init_logger(__name__)

# -- FlyDSL import bootstrap --------------------------------------------------
# FlyDSL lives outside the vLLM package tree. Add both the source and the
# built python_packages dir to sys.path lazily on first use, so import-time
# failures don't break vLLM startup on non-MI355X hosts.
_FLYDSL_ROOT = os.environ.get("VLLM_FLYDSL_ROOT", "/root/FlyDSL")
_FLYDSL_PKGS = os.environ.get(
    "VLLM_FLYDSL_PKGS", "/root/FlyDSL/build-fly/python_packages"
)


def _ensure_flydsl_paths() -> None:
    if _FLYDSL_PKGS not in sys.path:
        sys.path.insert(0, _FLYDSL_PKGS)
    if _FLYDSL_ROOT not in sys.path:
        sys.path.insert(0, _FLYDSL_ROOT)


_FLYDSL_AVAILABLE: bool | None = None
_TQ_MOD = None       # flydsl_kernels.tq_decode_v4 module (Qwen GQA-{8,16})
_TQ_MOD_GQA6 = None  # flydsl_kernels.tq_decode_v4_gqa6 module (MiniMax GQA-6, opt)
_TQ_MOD_HD256 = None       # flydsl_kernels.tq_decode_hd256 (HEAD_SIZE=256, opt)
_TQ_MOD_GQA6_HD256 = None  # flydsl_kernels.tq_decode_gqa6_hd256 (GQA-6 + 256, opt)
_FLYC = None    # flydsl.compiler
_FX = None      # flydsl.expr
_TYPING_T = None
_CC = None      # CompilationContext
_IR = None


def is_flydsl_available() -> bool:
    """Return True iff FlyDSL imports + canonical kernel module load successfully.

    The GQA-6 sibling kernel (``kernels.tq_decode_v4_gqa6``) is imported
    best-effort: if it's missing (older FlyDSL checkout that pre-dates
    MiniMax support) the canonical Qwen path stays fully functional and
    only GQA-6 dispatches will fail with a clear error at launch time.
    """
    global _FLYDSL_AVAILABLE, _TQ_MOD, _TQ_MOD_GQA6, _TQ_MOD_HD256
    global _TQ_MOD_GQA6_HD256
    global _FLYC, _FX, _TYPING_T, _CC, _IR
    if _FLYDSL_AVAILABLE is not None:
        return _FLYDSL_AVAILABLE
    try:
        _ensure_flydsl_paths()
        import flydsl.compiler as flyc  # noqa: F401
        import flydsl.expr as fx  # noqa: F401
        from flydsl.expr.typing import T  # noqa: F401
        from flydsl.compiler.kernel_function import CompilationContext
        from flydsl._mlir import ir
        from vllm.v1.attention.ops.flydsl_kernels import (
            tq_decode_v4 as tq_mod,
        )
        _FLYC = flyc
        _FX = fx
        _TYPING_T = T
        _CC = CompilationContext
        _IR = ir
        _TQ_MOD = tq_mod
        _FLYDSL_AVAILABLE = True
        logger.info_once("FlyDSL TQ decode v4 launcher: available")
    except Exception as ex:  # noqa: BLE001
        _FLYDSL_AVAILABLE = False
        logger.warning_once(
            "FlyDSL TQ decode v4 launcher: unavailable (%s). "
            "Falling back to Triton v3.", ex
        )
        return _FLYDSL_AVAILABLE
    # Best-effort GQA-6 sibling import (does NOT gate Qwen availability).
    try:
        from vllm.v1.attention.ops.flydsl_kernels import (
            tq_decode_v4_gqa6 as tq_mod_gqa6,
        )
        _TQ_MOD_GQA6 = tq_mod_gqa6
        logger.info_once(
            "FlyDSL TQ decode v4 GQA-6 sibling: available (MiniMax-class)"
        )
    except Exception as ex:  # noqa: BLE001
        _TQ_MOD_GQA6 = None
        logger.info_once(
            "FlyDSL TQ decode v4 GQA-6 sibling: not available (%s); "
            "GQA-6 models will fall back to Triton v3.", ex
        )
    # Best-effort HEAD_SIZE=256 sibling import (does NOT gate HEAD_SIZE=128
    # availability). Routes GQA-{8,16} layers with head_dim==256 (Qwen3.6-class,
    # Qwen3.5-397B) to the 256-wide-head kernel; missing it just means those
    # layers fall back to Triton v3.
    try:
        from vllm.v1.attention.ops.flydsl_kernels import (
            tq_decode_hd256 as tq_mod_hd256,
        )
        _TQ_MOD_HD256 = tq_mod_hd256
        logger.info_once("FlyDSL TQ decode v4 HEAD_SIZE=256 sibling: available")
    except Exception as ex:  # noqa: BLE001
        _TQ_MOD_HD256 = None
        logger.info_once(
            "FlyDSL TQ decode v4 HEAD_SIZE=256 sibling: not available (%s); "
            "head_dim=256 models will fall back to Triton v3.", ex
        )
    # Best-effort GQA-6 + HEAD_SIZE=256 sibling (Qwen3.6-27B full-attn layers).
    try:
        from vllm.v1.attention.ops.flydsl_kernels import (
            tq_decode_gqa6_hd256 as tq_mod_gqa6_hd256,
        )
        _TQ_MOD_GQA6_HD256 = tq_mod_gqa6_hd256
        logger.info_once(
            "FlyDSL TQ decode v4 GQA-6 HEAD_SIZE=256 sibling: available "
            "(Qwen3.6-27B class)"
        )
    except Exception as ex:  # noqa: BLE001
        _TQ_MOD_GQA6_HD256 = None
        logger.info_once(
            "FlyDSL TQ decode v4 GQA-6 HEAD_SIZE=256 sibling: not available "
            "(%s); GQA-6 head_dim=256 layers will fall back to Triton v3.", ex
        )
    return _FLYDSL_AVAILABLE


def is_flydsl_gqa6_available() -> bool:
    """True iff the optional GQA-6 sibling kernel module loaded.

    Used by the eligibility gate in turboquant_attn.py to decide whether
    a layer with num_kv_groups==6 can run on FlyDSL v4 or must fall back
    to Triton v3.
    """
    if _FLYDSL_AVAILABLE is None:
        is_flydsl_available()
    return _TQ_MOD_GQA6 is not None


def is_flydsl_hd256_available() -> bool:
    """True iff the optional HEAD_SIZE=256 sibling kernel module loaded.

    Used by the eligibility gate in turboquant_attn.py to decide whether a
    layer with head_dim==256 can run on FlyDSL v4 or must fall back to
    Triton v3.
    """
    if _FLYDSL_AVAILABLE is None:
        is_flydsl_available()
    return _TQ_MOD_HD256 is not None


def is_flydsl_gqa6_hd256_available() -> bool:
    """True iff the optional GQA-6 + HEAD_SIZE=256 sibling kernel module loaded.

    Used by the eligibility gate to decide whether a layer with
    num_kv_groups==6 AND head_dim==256 (Qwen3.6-27B full-attention layers)
    can run on FlyDSL v4 or must fall back to Triton v3.
    """
    if _FLYDSL_AVAILABLE is None:
        is_flydsl_available()
    return _TQ_MOD_GQA6_HD256 is not None


# -- Kernel module cache -------------------------------------------------------
# Compiling a FlyDSL kernel is expensive; cache by the constexprs that
# parameterize stride math: num_kv_heads, num_partitions, max_blocks_per_seq,
# scale. ``num_seqs`` is irrelevant to the kernel body so we omit it.
_KERN_CACHE: dict[tuple, Any] = {}


# -- Tier-1 launcher overhead state -------------------------------------------
# Replaces vllm `logger.{info,warning}_once` (which hashes the format string
# every call to dedup) with a true zero-overhead bool guard.
_LOG_INVOKED_ONCE: bool = False
_LOG_SINKS_WARNED: bool = False
_LOG_NORM_WARNED: bool = False


def _detect_max_capture_B() -> int:
    """Probe vLLM compilation config for the largest cudagraph capture size.

    Returns max(cudagraph_capture_sizes) if available, else falls back to
    ``VLLM_TQ_DECODE_V4_B_BUCKET`` (default 512 = vLLM default cap).

    Used to size the segm-pool bucket once, so that all distinct B's
    (whether captured in a graph, or hit eagerly in mixed batches) share
    a single allocation per shape — eliminating the per-B 52 MiB growth
    that previously dominated dense-capture VRAM cost (1.3 GiB at the
    25-size custom set, ~2.8 GiB at vLLM's full default sweep).
    """
    env = os.environ.get("VLLM_TQ_DECODE_V4_B_BUCKET")
    if env is not None:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    try:
        from vllm.config import get_current_vllm_config
        cfg = get_current_vllm_config()
        sizes = cfg.compilation_config.cudagraph_capture_sizes
        if sizes:
            return int(max(sizes))
    except Exception:  # noqa: BLE001
        pass
    return 512


class _SegmBufPool:
    """Single-bucket buffer pool for segm_out/segm_max/segm_sum/output.

    Each unique shape signature (Hk, Hq, num_partitions, QG, D, device,
    dtype) is backed by ONE allocation sized at ``max_B`` (= max captured
    cudagraph batch size, or env override). Per-call ``get(B, ...)``
    returns ``[:B]`` views — same data_ptr base, narrower N dim. The
    kernel grid is ``(B, ...)`` so it only ever writes the first B rows
    of segm_out / segm_max / segm_sum; downstream readers (Triton
    reducer + caller) only consume the first B rows of output.

    Why a single bucket: cudagraph records launch pointers at capture
    time. With per-B allocations, every captured size triggered a fresh
    52-MiB-each allocation (M2.5 64K shape) → 1.3 GiB at our 25-size
    capture set, ~2.8 GiB at vLLM's full default sweep up to B=512.
    With single-bucket, the pool tops out at one max_B-sized buffer per
    shape (~213 MiB at B_max=512 for M2.5), shared across ALL captured
    sizes and ALL eager mixed-batch calls. The data_ptr is stable from
    first allocation onward, so cudagraph capture is happy across all
    captured B's.

    Eliminates per-decode-step ``cudaMalloc`` (× 4) plus the device-side
    memset kernels behind ``torch.full(-inf)`` and ``torch.zeros``.
    The kernel always writes the FULL ``[B, Hk, P, QG, D]`` slice and
    every (n, kv_h, p, qg) position of segm_max / segm_sum (it stores
    ``-inf`` and ``0`` for empty partitions itself), so uninitialized
    bytes from the pool are safe to reuse for any B ≤ B_bucket.
    """

    __slots__ = ("_bufs", "_max_B")

    def __init__(self) -> None:
        self._bufs: dict[tuple, dict[str, torch.Tensor]] = {}
        self._max_B: int | None = None  # lazy-init on first get()

    def get(
        self,
        B: int,
        Hk: int,
        Hq: int,
        num_partitions: int,
        QG: int,
        D: int,
        device: torch.device,
        q_dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        if self._max_B is None:
            self._max_B = _detect_max_capture_B()
        # B may exceed the detected max if user set max-num-seqs higher
        # than the largest captured size — still safe, we'll grow once.
        # Note: growing AFTER cudagraph capture would invalidate captured
        # pointers, so this should only happen during warmup or in
        # always-eager configs.
        B_bucket = max(self._max_B, int(B))
        key = (
            int(Hk), int(Hq), int(num_partitions),
            int(QG), int(D), str(device), q_dtype,
        )
        bufs = self._bufs.get(key)
        if bufs is None or bufs["segm_out"].shape[0] < B_bucket:
            if bufs is not None and bufs["segm_out"].shape[0] < B_bucket:
                # First-time grow: warn so user knows cudagraphs may be
                # invalidated. In practice this means VLLM_TQ_DECODE_V4_
                # B_BUCKET should have been set higher.
                logger.warning_once(
                    "FlyDSL v4 _SegmBufPool: growing bucket from %d to %d "
                    "(B=%d). If you see this AFTER cudagraph warmup, the "
                    "previously-captured graphs hold stale pointers and "
                    "will GPU-fault. Set VLLM_TQ_DECODE_V4_B_BUCKET=%d "
                    "before launch to avoid this.",
                    bufs["segm_out"].shape[0], B_bucket, B, B_bucket,
                )
            bufs = {
                "segm_out": torch.empty(
                    (B_bucket, Hk, num_partitions, QG, D),
                    dtype=torch.bfloat16, device=device,
                ),
                "segm_max": torch.empty(
                    (B_bucket, Hk, num_partitions, QG),
                    dtype=torch.float32, device=device,
                ),
                "segm_sum": torch.empty(
                    (B_bucket, Hk, num_partitions, QG),
                    dtype=torch.float32, device=device,
                ),
                "output": torch.empty(
                    (B_bucket, Hq, D), dtype=q_dtype, device=device,
                ),
                # q_rot fix: stable pre-allocated buffer for the rotated query
                # tensor. On ROCm, fresh torch.empty / matmul-result allocations
                # after HIP graph capture can land in the graph memory pool and
                # cause GPU memory faults when passed as kernel arguments in the
                # eager mixed-batch path. By pooling q_rot here (same pattern as
                # segm_out/output above), the data_ptr is stable from first
                # allocation onward — same for all captured B's and eager calls.
                "q_rot": torch.empty(
                    (B_bucket, Hq, D), dtype=q_dtype, device=device,
                ),
                # q_float: fp32 copy of query (bf16→fp32 for mm input).
                # Pooled to avoid fresh allocations post-capture.
                "q_float": torch.empty(
                    (B_bucket, Hq, D), dtype=torch.float32, device=device,
                ),
                # q_rot_fp32: fp32 output of the rotation mm (before bf16 cast).
                # Using torch.mm(..., out=this) eliminates the fresh mm-result
                # allocation that would otherwise land in the HIP graph pool.
                "q_rot_fp32": torch.empty(
                    (B_bucket, Hq, D), dtype=torch.float32, device=device,
                ),
            }
            self._bufs[key] = bufs
            self._max_B = B_bucket
            logger.info_once(
                "FlyDSL v4 _SegmBufPool: allocated single-bucket "
                "shape=(Hk=%d, Hq=%d, P=%d, QG=%d, D=%d, dtype=%s) "
                "B_bucket=%d (covers all captured + eager B's). "
                "VRAM = %.1f MiB / shape.",
                Hk, Hq, num_partitions, QG, D, q_dtype,
                B_bucket,
                sum(t.numel() * t.element_size() for t in bufs.values())
                / (1 << 20),
            )
        # Return [:B] views — same data_ptr base, narrower N dim.
        return {
            "segm_out": bufs["segm_out"][:B],
            "segm_max": bufs["segm_max"][:B],
            "segm_sum": bufs["segm_sum"][:B],
            "output": bufs["output"][:B],
            "q_rot": bufs["q_rot"][:B],
            "q_float": bufs["q_float"][:B],
            "q_rot_fp32": bufs["q_rot_fp32"][:B],
        }

    def stats(self) -> dict[str, int]:
        return {
            "shapes": len(self._bufs),
            "max_B": int(self._max_B or 0),
            "bytes": sum(
                sum(t.numel() * t.element_size() for t in d.values())
                for d in self._bufs.values()
            ),
        }


_SEGM_POOL = _SegmBufPool()
_SEGM_DEBUG = os.environ.get("VLLM_TQ_V4_SEGM_DEBUG", "0") == "1"
# VLLM_TQ_V4_NAN_GUARD=1: after the reducer, scan the decode output for
# NaN/Inf. A single non-finite value here propagates into the hidden state and
# then into the KV store, spreading persistent corruption across the whole KV
# cache (and thus every subsequent request) — the suspected seed of the E2E
# progressive-accuracy collapse that shadow-compare cannot see (its v3 oracle
# reads the same already-corrupted KV). On first hit we log the full call
# context so the exact inputs can be reproduced offline.
_V4_NAN_GUARD = os.environ.get("VLLM_TQ_V4_NAN_GUARD", "0") == "1"
_V4_NAN_HITS = 0
_TENSOR_PROPS_LOGGED = False


_HW_TR_CACHED: bool | None = None
_WHT_BF_CACHED: bool | None = None


def _wht_butterfly_enabled() -> bool:
    """Return True iff the in-kernel WHT butterfly path is requested.

    Set ``VLLM_TQ_DECODE_V4_WHT_BUTTERFLY=1`` to enable.  When enabled
    the launcher skips the external T1.0 GEMM (``q @ PiT``) and passes
    the raw ``query`` tensor to a butterfly-capable kernel that computes
    ``H @ q`` in-register, eliminating the HBM round-trip for q_rot.
    """
    global _WHT_BF_CACHED
    if _WHT_BF_CACHED is not None:
        return _WHT_BF_CACHED
    _WHT_BF_CACHED = (
        os.environ.get("VLLM_TQ_DECODE_V4_WHT_BUTTERFLY", "0") == "1"
    )
    if _WHT_BF_CACHED:
        logger.info_once(
            "FlyDSL TQ v4: WHT butterfly ON "
            "(VLLM_TQ_DECODE_V4_WHT_BUTTERFLY=1)"
        )
    return _WHT_BF_CACHED


def _hw_tr_enabled() -> bool:
    """Resolve the HW V-transpose build flag.

    Default ON for gfx950+ (ds_read_tr16_b64 is bit-exact vs baseline and
    5-7% faster on Qwen-class shapes). Off elsewhere. Override with
    ``VLLM_TQ_DECODE_V4_HW_TR=0`` (force off) or ``=1`` (force on).
    """
    global _HW_TR_CACHED
    if _HW_TR_CACHED is not None:
        return _HW_TR_CACHED
    env = os.environ.get("VLLM_TQ_DECODE_V4_HW_TR")
    if env is not None:
        _HW_TR_CACHED = env == "1"
        return _HW_TR_CACHED
    try:
        _ensure_flydsl_paths()
        from flydsl.runtime.device import get_rocm_arch as _arch
        a = str(_arch() or "")
        _HW_TR_CACHED = a.startswith("gfx950")
    except Exception:  # noqa: BLE001
        _HW_TR_CACHED = False
    if _HW_TR_CACHED:
        logger.info_once("FlyDSL TQ v4: HW V transpose ON (default for gfx950+)")
    else:
        logger.info_once("FlyDSL TQ v4: HW V transpose OFF")
    return _HW_TR_CACHED


_GET_KERNEL_STATS = {"hits": 0, "misses": 0, "build_total_s": 0.0}


def _get_kernel(num_kv_heads: int, num_partitions: int,
                max_blocks_per_seq: int, scale: float,
                query_group_size: int, kv_block_size: int,
                use_hw_v_transpose: bool = False,
                num_seqs_hint: int = 1,
                tile_groups_per_partition: int = 1,
                use_wht_butterfly: bool = False,
                head_size: int = 128,
                sliding_window: int = 0,
                cache_block_stride_bytes: int = 0):
    # ``num_seqs_hint`` (= runtime B) is forwarded to the build for shape
    # awareness; it does NOT participate in the cache key because the
    # kernel body uses gpu.block_idx.x (= seq index at runtime) and is
    # GQA-symmetric across B at build time.
    #
    # ``tile_groups_per_partition`` (Option A) IS in the cache key — each
    # value compiles a different unrolled K-tile loop body.
    # ``sliding_window`` IS in the cache key: it toggles the base-shift + mask
    # codegen (SWA==0 is the unchanged full-attention binary).
    # ``cache_block_stride_bytes`` IS in the cache key: a padded KV cache
    # (Gemma 4 unified page) bakes a different block stride than the natural /
    # contiguous case (0), so they are distinct binaries.
    key = (num_kv_heads, int(num_partitions), int(max_blocks_per_seq),
           round(float(scale), 8), int(query_group_size), int(kv_block_size),
           bool(use_hw_v_transpose), int(tile_groups_per_partition),
           bool(use_wht_butterfly), int(head_size), int(sliding_window),
           int(cache_block_stride_bytes))
    cached = _KERN_CACHE.get(key)
    if cached is not None:
        _GET_KERNEL_STATS["hits"] += 1
        return cached
    assert is_flydsl_available()
    # Time the FlyDSL build path so we can quantify cudagraph capture cost.
    import time as _t
    _build_t0 = _t.perf_counter()

    # Per-(head_size, GQA) dispatch:
    #   * head_size == 256 → tq_decode_hd256 / tq_decode_gqa6_hd256 siblings
    #     (Qwen3.6-27B / Qwen3.5-397B). Keeps the HEAD_SIZE=128 v4 kernels
    #     byte-identical.
    #   * head_size == 128, GQA-6 (MiniMax-M2.5) → tq_decode_v4_gqa6 sibling.
    #   * head_size == 128, GQA-{8,16} (Qwen) → canonical tq_decode_v4 kernel.
    qg = int(query_group_size)
    hs = int(head_size)
    if hs == 256:
        if qg == 6:
            if _TQ_MOD_GQA6_HD256 is None:
                raise RuntimeError(
                    "FlyDSL TQ v4 GQA-6 HEAD_SIZE=256 sibling kernel "
                    "(vllm.v1.attention.ops.flydsl_kernels."
                    "tq_decode_gqa6_hd256) failed to import; check that "
                    "FlyDSL is importable."
                )
            kmod = _TQ_MOD_GQA6_HD256
            kfn = kmod.build_tq_decode_gqa6_hd256_module(
                num_seqs=int(num_seqs_hint),
                num_kv_heads=num_kv_heads,
                num_partitions=num_partitions,
                max_blocks_per_seq=max_blocks_per_seq,
                softmax_scale=float(scale),
                query_group_size=qg,
                kv_block_size=int(kv_block_size),
                use_hw_v_transpose=bool(use_hw_v_transpose),
                tile_groups_per_partition=int(tile_groups_per_partition),
                use_wht_butterfly=bool(use_wht_butterfly),
            )
        else:
            if _TQ_MOD_HD256 is None:
                raise RuntimeError(
                    "FlyDSL TQ v4 HEAD_SIZE=256 sibling kernel "
                    "(vllm.v1.attention.ops.flydsl_kernels.tq_decode_hd256) "
                    "failed to import; check that FlyDSL is importable."
                )
            kmod = _TQ_MOD_HD256
            kfn = kmod.build_tq_decode_hd256_module(
                num_seqs=int(num_seqs_hint),
                num_kv_heads=num_kv_heads,
                num_partitions=num_partitions,
                max_blocks_per_seq=max_blocks_per_seq,
                softmax_scale=float(scale),
                query_group_size=qg,
                kv_block_size=int(kv_block_size),
                use_hw_v_transpose=bool(use_hw_v_transpose),
                tile_groups_per_partition=int(tile_groups_per_partition),
                use_wht_butterfly=bool(use_wht_butterfly),
                sliding_window=int(sliding_window),
                cache_block_stride_bytes=int(cache_block_stride_bytes),
            )
    elif qg == 6:
        if _TQ_MOD_GQA6 is None:
            raise RuntimeError(
                "FlyDSL TQ v4 GQA-6 sibling module is not importable; "
                "ensure vllm/v1/attention/ops/flydsl_kernels/"
                "tq_decode_v4_gqa6.py is present in the vLLM tree."
            )
        kmod = _TQ_MOD_GQA6
        kfn = kmod.build_tq_decode_v4_gqa6_module(
            num_seqs=int(num_seqs_hint),
            num_kv_heads=num_kv_heads,
            num_partitions=num_partitions,
            max_blocks_per_seq=max_blocks_per_seq,
            softmax_scale=float(scale),
            query_group_size=qg,
            kv_block_size=int(kv_block_size),
            use_hw_v_transpose=bool(use_hw_v_transpose),
            tile_groups_per_partition=int(tile_groups_per_partition),
        )
    else:
        kmod = _TQ_MOD
        kfn = kmod.build_tq_decode_v4_module(
            num_seqs=int(num_seqs_hint),
            num_kv_heads=num_kv_heads,
            num_partitions=num_partitions,
            max_blocks_per_seq=max_blocks_per_seq,
            softmax_scale=float(scale),
            query_group_size=qg,
            kv_block_size=int(kv_block_size),
            use_hw_v_transpose=bool(use_hw_v_transpose),
            tile_groups_per_partition=int(tile_groups_per_partition),
            use_wht_butterfly=bool(use_wht_butterfly),
        )
    al = kmod.allocator
    block_threads = kmod.BLOCK_THREADS

    flyc = _FLYC
    fx = _FX
    T = _TYPING_T
    CompilationContext = _CC
    ir_mod = _IR

    @flyc.jit
    def _launch(out, es, ml, q, kvc, cents, bt, sl,
                gx: fx.Int32, gy: fx.Int32, gz: fx.Int32,
                stream: fx.Stream):
        # Re-finalize the LDS allocator for this launch (idempotent per build).
        al.finalized = False
        ctx = CompilationContext.get_current()
        with ir_mod.InsertionPoint(ctx.gpu_module_body):
            al.finalize()
        from flydsl.expr import arith
        grid_x = arith.index_cast(T.index, gx.ir_value())
        grid_y = arith.index_cast(T.index, gy.ir_value())
        grid_z = arith.index_cast(T.index, gz.ir_value())
        kfn(out, es, ml, q, kvc, cents, bt, sl).launch(
            grid=(grid_x, grid_y, grid_z),
            block=(block_threads, 1, 1), stream=stream,
        )

    _KERN_CACHE[key] = _launch
    _build_dt = _t.perf_counter() - _build_t0
    _GET_KERNEL_STATS["misses"] += 1
    _GET_KERNEL_STATS["build_total_s"] += _build_dt
    logger.info(
        "FlyDSL v4 _get_kernel BUILD #%d dt=%.2fs (cumulative=%.1fs) key=%s",
        _GET_KERNEL_STATS["misses"], _build_dt,
        _GET_KERNEL_STATS["build_total_s"], key,
    )
    return _launch


# -- Partition reducer (Triton, native v4 layout) ------------------------------
# Reduces the FA2 split-KV partials produced by the FlyDSL kernel.
#   segm_out [N, Hk, P, QG, D] bf16
#   segm_max [N, Hk, P, QG]    fp32
#   segm_sum [N, Hk, P, QG]    fp32
# → output [N, Hq=Hk*QG, D] of any (bf16/fp16/fp32) dtype.
@triton.jit
def _reduce_partitions_v4(
    output_ptr,              # [N, Hq, D] in OUT_DTYPE
    segm_out_ptr,            # [N, Hk, P, QG, D] bf16
    segm_max_ptr,            # [N, Hk, P, QG] fp32
    segm_sum_ptr,            # [N, Hk, P, QG] fp32
    out_stride_n: tl.int64,  # stride on N axis (in elems of OUT_DTYPE)
    out_stride_h: tl.int64,  # stride on Hq axis
    NUM_KV_HEADS: tl.constexpr,
    QG: tl.constexpr,
    NUM_PARTS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
):
    # grid = (N, Hq)
    n = tl.program_id(0)
    hq = tl.program_id(1)
    kv_h = hq // QG
    qg = hq % QG

    # Per-partition pointers for this (n, kv_h, qg) row.
    # Strides in elems of source dtype.
    msum_base = (
        n * (NUM_KV_HEADS * NUM_PARTS * QG)
        + kv_h * (NUM_PARTS * QG)
        + qg
    )
    so_base = (
        n * (NUM_KV_HEADS * NUM_PARTS * QG * HEAD_SIZE)
        + kv_h * (NUM_PARTS * QG * HEAD_SIZE)
        + qg * HEAD_SIZE
    )

    p_off = tl.arange(0, NUM_PARTS)
    d_off = tl.arange(0, HEAD_SIZE)

    # Load per-partition max/sum.
    m_idx = msum_base + p_off * QG
    seg_max = tl.load(segm_max_ptr + m_idx)
    seg_sum = tl.load(segm_sum_ptr + m_idx)

    # Mask out empty partitions (max == -inf).
    valid = seg_max > float("-inf")
    overall_max = tl.max(tl.where(valid, seg_max, float("-inf")))

    # Rescale exp sums.
    rescale = tl.where(valid, tl.exp(seg_max - overall_max), 0.0)
    seg_sum_rescaled = seg_sum * rescale
    overall_sum = tl.sum(seg_sum_rescaled)

    # Load segment outputs and combine.
    so_idx = (
        so_base
        + p_off[:, None] * (QG * HEAD_SIZE)
        + d_off[None, :]
    )
    seg_out = tl.load(segm_out_ptr + so_idx).to(tl.float32)
    # acc = sum_p( seg_out_p * (seg_sum_p * rescale_p) ) / overall_sum
    weighted = seg_out * seg_sum_rescaled[:, None]
    acc_sum = tl.sum(weighted, axis=0)
    acc = tl.where(overall_sum > 0.0, acc_sum / overall_sum, 0.0)

    # Store.
    out_off = n * out_stride_n + hq * out_stride_h + d_off
    tl.store(output_ptr + out_off, acc.to(output_ptr.dtype.element_ty))


# -- Public launcher ----------------------------------------------------------
def flydsl_turboquant_decode_attention_v4(
    query: torch.Tensor,            # [B, Hq, D] bf16/fp16
    kv_cache: torch.Tensor,         # [num_blocks, BS, Hk, slot_size_aligned]
    block_table: torch.Tensor,      # [B, max_blocks_per_seq] int32
    seq_lens: torch.Tensor,         # [B] int32
    Pi: torch.Tensor,               # [D, D] fp32
    centroids: torch.Tensor,        # [N_CENTROIDS] fp32
    scale: float,
    mse_bits: int,
    key_packed_size: int,
    value_quant_bits: int,
    value_packed_size: int,
    key_fp8: bool = False,
    norm_correction: bool = False,
    PiT: torch.Tensor | None = None,
    max_seq_len: int = 0,
    mid_o_buf: Any = None,
    output_buf: torch.Tensor | None = None,
    lse_buf: Any = None,
    buf_holder: Any = None,
    max_num_kv_splits: int = 32,
    sinks: torch.Tensor | None = None,
    sliding_window: int | None = None,
) -> torch.Tensor:
    """v3-compatible launcher backed by the FlyDSL v4 decode kernel.

    Constraints:
      * key_fp8 == False
      * mse_bits == 4
      * value_quant_bits == 4
      * centroids.numel() == 16
      * D == 128
      * block_size in {16, 32}
      * Hq // Hk in {6, 8, 16}
        - 8/16 → canonical tq_decode_v4 kernel (Qwen2.5-72B / Qwen3-32B)
        - 6    → tq_decode_v4_gqa6 sibling kernel (MiniMax-M2.5)

    Sinks are NYI and silently ignored if set. norm_correction is honored
    implicitly via the pre-folded stored K-norm (see footer comment).
    """
    del mid_o_buf, lse_buf, key_packed_size, value_packed_size
    if not is_flydsl_available():
        raise RuntimeError(
            "VLLM_TQ_DECODE_V4 requested but FlyDSL is not available; "
            "set VLLM_TQ_DECODE_V4=0 or fix VLLM_FLYDSL_ROOT/VLLM_FLYDSL_PKGS."
        )
    assert not key_fp8, "FlyDSL v4 supports MSE-key path only"
    assert mse_bits == 4, f"FlyDSL v4 expects mse_bits=4, got {mse_bits}"
    assert value_quant_bits == 4, (
        f"FlyDSL v4 expects value_quant_bits=4, got {value_quant_bits}"
    )

    B, Hq, D = query.shape
    Hk = kv_cache.shape[2]
    block_size = kv_cache.shape[1]
    QG = Hq // Hk
    # HEAD_SIZE=128 uses the canonical/GQA-6 v4 kernels; HEAD_SIZE=256 uses the
    # tq_decode_hd256 / tq_decode_gqa6_hd256 siblings. Both share the same TQ
    # profile constants (N_CENTROIDS, KV_COMPUTE_BLOCK); only per-head width
    # differs.
    assert D in (_TQ_MOD.HEAD_SIZE, 256), (
        f"FlyDSL v4 supports head_dim 128 or 256, got {D}"
    )
    if D == 256 and _TQ_MOD_HD256 is None:
        raise RuntimeError(
            "FlyDSL v4 launcher: head_dim=256 requested but the "
            "tq_decode_hd256 sibling module is not available. Ensure "
            "vllm/v1/attention/ops/flydsl_kernels/tq_decode_hd256.py is "
            "present, or set VLLM_TQ_DECODE_V4=0 to fall back to Triton v3."
        )
    # head_dim=128 kernels support {16,32}; head_dim=256 (incl. hybrid models
    # like Qwen3.6-27B, which force a larger mamba-aligned block) also accept
    # 64/128/256. Keeps the 128-head path constraint unchanged.
    if D == 256:
        assert block_size in (16, 32, 64, 128, 256), (
            f"v4 head_dim=256 supports kv_block_size 16/32/64/128/256, "
            f"got {block_size}"
        )
    else:
        assert block_size in (16, 32), (
            f"v4 supports kv_block_size 16 or 32, got {block_size}"
        )
    # QG=2 (Gemma 4 sliding layers, 32 q / 16 kv) is under evaluation on the
    # HEAD_SIZE=256 canonical kernel: the 16-row MFMA half-fill + mfma_row<QG
    # store gate already generalize below GQA-8. Only the D=256 canonical path
    # is validated for QG=2; D=128 still requires {6,8,16}.
    if D == 256:
        assert QG in (2, 6, 8, 16), (
            f"v4 head_dim=256 supports GQA factor 2, 6, 8 or 16, got {QG}"
        )
    else:
        assert QG in (6, 8, 16), f"v4 supports GQA factor 6, 8 or 16, got {QG}"
    if D == 256 and QG == 6 and _TQ_MOD_GQA6_HD256 is None:
        raise RuntimeError(
            "FlyDSL v4 launcher: GQA-6 head_dim=256 requested (Qwen3.6-class) "
            "but the tq_decode_gqa6_hd256 sibling module is not available. "
            "Ensure vllm/v1/attention/ops/flydsl_kernels/"
            "tq_decode_gqa6_hd256.py is present, or set VLLM_TQ_DECODE_V4=0 "
            "to fall back to Triton v3."
        )
    if D == 128 and QG == 6 and _TQ_MOD_GQA6 is None:
        raise RuntimeError(
            "FlyDSL v4 launcher: GQA-6 requested (MiniMax-class) but the "
            "tq_decode_v4_gqa6 sibling module is not available. Ensure "
            "vllm/v1/attention/ops/flydsl_kernels/tq_decode_v4_gqa6.py is "
            "present, or set VLLM_TQ_DECODE_V4=0 to fall back to Triton v3."
        )
    assert centroids.numel() == _TQ_MOD.N_CENTROIDS, (
        f"centroids.numel={centroids.numel()} != "
        f"{_TQ_MOD.N_CENTROIDS}"
    )

    # ---- T1.1 / T1.4: per-layer cache for PiT_f32 + contiguous centroids ---
    # PiT and centroids are model constants (set once at layer warmup).
    # Avoid the per-decode-step transpose+cast and `.contiguous()` no-op
    # check by stashing the pre-cooked tensors on ``buf_holder`` (= layer).
    PiT_f32: torch.Tensor
    centroids_c: torch.Tensor
    if buf_holder is not None:
        PiT_f32 = getattr(buf_holder, "_tq_v4_PiT_f32", None)
        if PiT_f32 is None:
            _PiT_src = PiT if PiT is not None else Pi.T.contiguous()
            PiT_f32 = (
                _PiT_src if _PiT_src.dtype == torch.float32
                else _PiT_src.to(torch.float32)
            )
            buf_holder._tq_v4_PiT_f32 = PiT_f32
        centroids_c = getattr(buf_holder, "_tq_v4_centroids_c", None)
        if centroids_c is None:
            centroids_c = centroids.contiguous()
            buf_holder._tq_v4_centroids_c = centroids_c
    else:
        # Defensive fallback (unit tests may pass ``buf_holder=None``).
        _PiT_src = PiT if PiT is not None else Pi.T.contiguous()
        PiT_f32 = (
            _PiT_src if _PiT_src.dtype == torch.float32
            else _PiT_src.to(torch.float32)
        )
        centroids_c = centroids.contiguous()

    # ---- T1.0: pooled q_rot (Bug-1 fix: stable data_ptr post-capture) ------
    # On ROCm, fresh tensor allocations made AFTER HIP graph capture can land
    # in the graph memory pool and cause GPU memory faults when passed as kernel
    # pointer arguments in the eager mixed-batch path. We pool q_rot and the
    # fp32 intermediate the same way segm_out/output are pooled: one max-B
    # allocation per shape, reused across all B's.
    #
    # Note: pool_bufs is computed BELOW after num_partitions is determined;
    # q_rot needs pool_bufs["q_float"] and pool_bufs["q_rot"]. We compute the
    # rotation inline after the pool is fetched (see T1.2).

    # ---- Partition count (FA2 split-KV) ----------------------------------
    #
    # v4 runs INSIDE FULL cudagraph (TurboQuantMetadataBuilder._cudagraph_
    # support = UNIFORM_BATCH). That means the gridDim baked at capture
    # time MUST equal the gridDim at replay — the kernel launch parameters
    # are recorded into the captured graph. So `num_partitions` (which
    # becomes gridDim.z) MUST be derived from a stable source that produces
    # the same value at capture and at runtime.
    #
    # We use the worst-case context length implied by the block table's
    # allocation (block_table.shape[1] * block_size) as that source. It's
    # bounded by `max_model_len` (vLLM allocates the block table for the
    # configured max), and it's identical at capture and at runtime
    # because `block_table.shape[1]` is fixed once the model is loaded.
    # Per-tile OOB redirects + the masked FA-2 reducer ensure that for
    # short sequences the extra partitions contribute zero to the output.
    #
    # Override (for eager-mode testing only): set
    # VLLM_TQ_DECODE_V4_DYNAMIC_PARTS=1 to size from per-step actual
    # `max_seq_len` instead. This will crash if cudagraph is enabled
    # because the captured gridDim won't match runtime; safe ONLY with
    # --enforce-eager or unit-test harnesses.
    kv_compute_block = _TQ_MOD.KV_COMPUTE_BLOCK
    worst_case_max_seq_len = int(block_table.shape[1]) * int(block_size)
    if max_seq_len <= 0:
        max_seq_len = worst_case_max_seq_len
    if os.environ.get("VLLM_TQ_DECODE_V4_DYNAMIC_PARTS", "0") == "1":
        sizing_max_seq_len = int(max_seq_len)
    else:
        sizing_max_seq_len = worst_case_max_seq_len

    # Sliding-window layers read at most ``window`` keys plus one aligned
    # KV_COMPUTE_BLOCK of slack (the kernel shifts every partition base to the
    # aligned window start). Sizing partitions to that span instead of the full
    # context is the whole point of SWA on the decode path: at 128K with
    # window=1024 this drops num_partitions/TGPP from covering 131072 tokens to
    # ~1280, i.e. the kernel stops walking 100x of masked-out tiles. ``window``
    # is a fixed per-layer constant, so this keeps num_partitions cudagraph-
    # stable. Only applied on the SWA-capable HEAD_SIZE=256 path.
    _swa = int(sliding_window) if sliding_window else 0
    if _swa > 0:
        # Only the canonical HEAD_SIZE=256 kernel (QG in {2,8,16}) implements
        # the base-shift + window mask. Routing a sliding-window layer through
        # any other build path (hd128, gqa6_hd256, or a future hd512) would
        # silently run full attention — correct only until context exceeds the
        # window — so fail loudly instead. The backend gate already restricts
        # the SWA relaxation to head_size==256; this is defense in depth for
        # direct launcher callers.
        if D != 256 or QG == 6:
            raise NotImplementedError(
                f"FlyDSL v4 sliding_window is implemented only on the "
                f"canonical head_dim=256 kernel (QG in 2/8/16); got "
                f"head_dim={D}, QG={QG}. Route this layer to v3."
            )
        # Correctness-first SWA (matches the kernel change in
        # tq_decode_hd256.py STEP E): the kernel no longer base-shifts to the
        # aligned window start — it partitions from absolute 0 and windows
        # purely via the per-token mask, exactly like the v3 SoA decode. So the
        # partitions must cover the FULL sequence, not just the window span;
        # sizing to ``swa_span`` here would leave the tail of long sequences
        # uncovered (the window lives at the END of the sequence, not the
        # start). We therefore keep ``sizing_max_seq_len`` at the worst-case
        # context. The window-span pruning optimization was the source of the
        # seq_len>window divergence vs v3 and is intentionally removed until it
        # can be reintroduced as a proven-safe leading-tile skip. The guard
        # above still restricts SWA to the hd256 QG∈{2,8,16} kernel.
        _ = _swa + kv_compute_block  # (span kept for reference; no longer caps)

    # ── Option A: bounded num_partitions + internal tile-group looping ──
    # The kernel previously hardcoded one partition = 256 tokens (=
    # KV_COMPUTE_BLOCK), forcing num_partitions to scale linearly with
    # max_model_len. For 32K that gave grid.z = 256 and a 19.6 GiB
    # cudagraph capture (Qwen 72B 32K, MI355X). The new kernel takes
    # ``tile_groups_per_partition`` (TGPP); each CTA processes
    # ``TGPP * 16`` K-tiles = ``TGPP * KV_COMPUTE_BLOCK`` tokens with the
    # FA-2 online-softmax state accumulating across all of them. This
    # mirrors what Triton v3 / HIP SoA-fusion already do.
    #
    # Strategy: cap num_partitions at ``MAX_PARTITIONS`` (default 32 — the
    # same constant Triton v3's launcher uses, see
    # triton_turboquant_unified_attention.py L1224 ``num_kv_splits=16``
    # baseline + L370 HIP launcher ``max_num_kv_splits=32``), then derive
    # TGPP so that ``num_partitions * TGPP * KV_COMPUTE_BLOCK >=
    # sizing_max_seq_len``.
    #
    # Examples (block_size=32 → worst_case max_bps*32):
    #   max_model_len=8K:   required=32, parts=32, TGPP=1  (no waste)
    #   max_model_len=16K:  required=64, parts=32, TGPP=2  (no waste)
    #   max_model_len=32K:  required=128, parts=32, TGPP=4 (no waste)
    #   max_model_len=64K:  required=256, parts=32, TGPP=8 (no waste)
    #   max_model_len=128K: required=512, parts=32, TGPP=16
    #
    # Override via VLLM_TQ_DECODE_V4_MAX_PARTITIONS (default 32). Smaller
    # gives even less graph memory but more work per CTA; larger gives
    # more parallelism but more graph memory.
    MAX_PARTITIONS = int(os.environ.get(
        "VLLM_TQ_DECODE_V4_MAX_PARTITIONS", "32"))
    MAX_PARTITIONS = max(2, MAX_PARTITIONS)
    required_num_partitions = (
        sizing_max_seq_len + kv_compute_block - 1) // kv_compute_block
    # max_num_kv_splits acts as a parallelism floor (ensure at least this
    # many CTAs along grid.z), but is itself capped by MAX_PARTITIONS.
    parallelism_floor = min(MAX_PARTITIONS, max(1, max_num_kv_splits))
    # Cap num_partitions at MAX_PARTITIONS, but never go below the floor.
    num_partitions_actual = max(
        parallelism_floor,
        min(MAX_PARTITIONS, required_num_partitions),
    )
    # Round up to next power of 2 for the Triton reducer's tl.arange(0, N)
    # constraint (Triton requires power-of-2 ≥ 2).
    num_partitions = max(2, triton.next_power_of_2(num_partitions_actual))
    # Now derive tile_groups_per_partition (TGPP) so total coverage
    # ``num_partitions * TGPP * KV_COMPUTE_BLOCK`` is >= sizing_max_seq_len.
    # This guarantees no work is dropped at runtime regardless of seq_len.
    # Round TGPP up to the next power-of-2 so the JIT cache key is bounded
    # to a small set of values (e.g. {1, 2, 4, 8, 16}) — limits the number
    # of distinct kernel binaries that need to be compiled across requests.
    _tgpp_required = max(
        1,
        (required_num_partitions + num_partitions - 1) // num_partitions,
    )
    tile_groups_per_partition = int(triton.next_power_of_2(_tgpp_required))

    # ---- T1.2: pooled buffers (no per-call cudaMalloc / memset) ----------
    # The FlyDSL kernel writes the FULL [B, Hk, P, QG, D] segm_out and the
    # FULL [B, Hk, P, QG] segm_max / segm_sum (running_max=-inf, sum=0 for
    # empty partitions are stored by the kernel itself), so uninitialized
    # buffers from the pool are safe to reuse.
    device = query.device
    pool_bufs = _SEGM_POOL.get(
        B, Hk, Hq, num_partitions, QG, D, device, query.dtype,
    )
    segm_out = pool_bufs["segm_out"]
    segm_max = pool_bufs["segm_max"]
    segm_sum = pool_bufs["segm_sum"]
    if output_buf is None:
        output = pool_bufs["output"]
    else:
        output = output_buf[:B] if output_buf.shape[0] != B else output_buf

    # ---- T1.0 (continued): q rotation -----------------------------------
    # Normal path: GEMM q @ PiT via stable pooled buffers.
    # Butterfly path (VLLM_TQ_DECODE_V4_WHT_BUTTERFLY=1): skip the GEMM
    # entirely; the kernel computes H @ q in-register (STEP B').  We still
    # allocate/preserve the pool slots so the pool state is consistent, but
    # no computation is performed on them.
    use_wht_bf = _wht_butterfly_enabled()
    if use_wht_bf and D == 256:
        logger.warning_once(
            "FlyDSL head_dim=256: WHT butterfly not supported yet; using the "
            "q@PiT GEMM rotation path instead "
            "(VLLM_TQ_DECODE_V4_WHT_BUTTERFLY ignored for 256-wide heads)."
        )
        use_wht_bf = False
    if not use_wht_bf:
        _q_float = pool_bufs["q_float"]          # [B, Hq, D] fp32, stable
        _q_rot_f32 = pool_bufs["q_rot_fp32"]     # [B, Hq, D] fp32, stable mm output
        _q_rot_out = pool_bufs["q_rot"]          # [B, Hq, D] query.dtype, stable
        _q_float.copy_(query)                    # bf16 → fp32 in-place, no alloc
        # mm into stable fp32 buffer via out= to avoid fresh allocations.
        # PiT_f32 is cached on buf_holder (stable ptr since first layer warmup).
        torch.mm(
            _q_float.view(B * Hq, D), PiT_f32,
            out=_q_rot_f32.view(B * Hq, D),     # in-place into pool buffer
        )
        _q_rot_out.copy_(_q_rot_f32)             # fp32 → bf16, into stable buf
        q_for_kernel = _q_rot_out                # stable ptr for kernel launch
    else:
        # Butterfly: pass raw query directly.  query is [B, Hq, D] bf16,
        # contiguous (guaranteed by the attention backend).
        q_for_kernel = query

    # ---- FlyDSL kernel launch -------------------------------------------
    max_bps = int(block_table.shape[1])
    use_hw_tr = _hw_tr_enabled()
    # Padded-cache block stride (Gemma 4 unified page). vLLM hands the D=256
    # sliding layers a strided KV view whose stride(0) exceeds the natural
    # [data|meta] block (the sliding page is padded up to the D=512 global
    # page). The kernel bakes the block stride as a constant, so we must pass
    # the true stride; the natural computation would read the wrong block and
    # produce token-salad. Pass 0 whenever the cache is contiguous (Qwen /
    # hd128) so those kernel binaries stay bit-identical.
    _cache_block_stride_bytes = 0
    if D == 256:
        _actual_block_stride = int(kv_cache.stride(0)) * int(kv_cache.element_size())
        _natural_block_stride = (
            block_size * Hk * _TQ_MOD_HD256.DATA_BYTES_PER_SLOT
            + Hk * _TQ_MOD_HD256.NUM_SOA_FIELDS * block_size * 2
        )
        if _actual_block_stride != _natural_block_stride:
            _cache_block_stride_bytes = _actual_block_stride
    launch = _get_kernel(
        Hk, num_partitions, max_bps, scale, QG, block_size,
        use_hw_v_transpose=use_hw_tr,
        num_seqs_hint=int(B),
        tile_groups_per_partition=int(tile_groups_per_partition),
        use_wht_butterfly=use_wht_bf,
        head_size=int(D),
        sliding_window=_swa,
        cache_block_stride_bytes=_cache_block_stride_bytes,
    )
    # T1.3: zero-overhead one-shot info log (replaces logger.info_once which
    # hashes its format string on every call to dedup).
    global _LOG_INVOKED_ONCE, _LOG_SINKS_WARNED, _LOG_NORM_WARNED
    if not _LOG_INVOKED_ONCE:
        _LOG_INVOKED_ONCE = True
        logger.info(
            "FlyDSL v4 launcher invoked (UNIFORM_BATCH cudagraph): "
            "B=%d Hk=%d Hq=%d D=%d QG=%d num_partitions=%d (actual=%d, "
            "cap=%d) TGPP=%d max_bps=%d block_size=%d max_seq_len=%d "
            "hw_v_transpose=%s wht_butterfly=%s "
            "(coverage=%d tokens, worst_case=%d tokens, sizing=%s)",
            B, Hk, Hq, D, QG, num_partitions, num_partitions_actual,
            MAX_PARTITIONS, tile_groups_per_partition,
            max_bps, int(block_size), int(max_seq_len), use_hw_tr,
            use_wht_bf,
            num_partitions * tile_groups_per_partition * kv_compute_block,
            worst_case_max_seq_len,
            "per-step actual" if (
                os.environ.get("VLLM_TQ_DECODE_V4_DYNAMIC_PARTS", "0") == "1"
            ) else "worst_case",
        )
    launch(
        segm_out, segm_sum, segm_max,
        q_for_kernel, kv_cache, centroids_c,
        block_table, seq_lens,
        B, Hk, num_partitions,
        torch.cuda.current_stream(),
    )

    if _SEGM_DEBUG and not _TENSOR_PROPS_LOGGED:
        # The offline replay rebuilds every tensor as a fresh contiguous
        # allocation, which would mask a non-contiguous view, odd stride or
        # dtype mismatch that the kernel mis-reads in the live server.
        globals()["_TENSOR_PROPS_LOGGED"] = True
        def _props(name, t):
            if not isinstance(t, torch.Tensor):
                return f"{name}=<{type(t).__name__}>"
            return (f"{name}: shape={tuple(t.shape)} stride={t.stride()} "
                    f"dtype={t.dtype} contig={t.is_contiguous()} "
                    f"ptr%256={t.data_ptr() % 256} off={t.storage_offset()}")
        logger.warning(
            "FlyDSL v4 TENSOR PROPS (B=%d):\n  %s", B, "\n  ".join([
                _props("query", query), _props("q_for_kernel", q_for_kernel),
                _props("kv_cache", kv_cache),
                _props("block_table", block_table),
                _props("seq_lens", seq_lens),
                _props("segm_out", segm_out), _props("segm_max", segm_max),
                _props("output", output),
            ]))

    if _SEGM_DEBUG:
        # A row that ends up all-zero in the output got no valid partition
        # from the kernel. Distinguish "kernel ran and found nothing"
        # (segm_max written as -inf) from "kernel never touched the row"
        # (stale pooled bytes) before the reducer collapses the evidence.
        # Mirror the reducer's own condition: a row goes to exact zeros when
        # every (kv_head, qgroup) has no partition with a positive weight.
        _m = segm_max[:B]
        _s = segm_sum[:B]
        _valid = _m > float("-inf")
        _eff = (_s * _valid).sum(dim=2)          # [B, Hk, QG]
        _dead_rows = (_eff <= 0).reshape(B, -1).all(dim=1)
        if bool(_dead_rows.any()):
            _bad = [b for b in range(B) if bool(_dead_rows[b])]
            logger.warning(
                "FlyDSL v4 SEGM: DEAD ROWS %s (call ctx B=%d Hk=%d P=%d "
                "QG=%d) seq_lens=%s valid_parts_per_row=%s "
                "sum_absmax_per_row=%s max_finite_per_row=%s "
                "q_absmax_per_row=%s",
                _bad, B, Hk, num_partitions, QG, seq_lens[:B].tolist(),
                _valid.reshape(B, -1).sum(dim=1).tolist(),
                [round(float(_s[b].abs().max()), 6) for b in range(B)],
                [int(torch.isfinite(_m[b]).sum()) for b in range(B)],
                [round(float(q_for_kernel[b].abs().max()), 4)
                 for b in range(B)],
            )

    # ---- Reduce partitions -> [B, Hq, D] --------------------------------
    _reduce_partitions_v4[(B, Hq)](
        output_ptr=output,
        segm_out_ptr=segm_out,
        segm_max_ptr=segm_max,
        segm_sum_ptr=segm_sum,
        out_stride_n=output.stride(0),
        out_stride_h=output.stride(1),
        NUM_KV_HEADS=Hk,
        QG=QG,
        NUM_PARTS=num_partitions,
        HEAD_SIZE=D,
    )
    if _V4_NAN_GUARD:
        global _V4_NAN_HITS
        _ob = output[:B]
        _nonfin = ~torch.isfinite(_ob)
        if bool(_nonfin.any()):
            _V4_NAN_HITS += 1
            _rows = _nonfin.reshape(B, -1).any(dim=1)
            _bad = [b for b in range(B) if bool(_rows[b])]
            logger.warning(
                "FlyDSL v4 NAN-GUARD hit #%d: non-finite output rows=%s "
                "(B=%d Hk=%d Hq=%d P=%d QG=%d D=%d) seq_lens=%s "
                "n_nonfinite=%d q_absmax=%s segm_max_finite_per_bad=%s "
                "segm_sum_absmax_per_bad=%s",
                _V4_NAN_HITS, _bad, B, Hk, Hq, num_partitions, QG, D,
                seq_lens[:B].tolist(), int(_nonfin.sum()),
                [round(float(q_for_kernel[b].abs().max()), 4) for b in _bad],
                [int(torch.isfinite(segm_max[b]).sum()) for b in _bad],
                [round(float(segm_sum[b].abs().max()), 6) for b in _bad],
            )
            # segm_out NaN localization: is the non-finite value already in the
            # kernel's segm_out (kernel bug) or introduced by the reducer?
            _so = segm_out[:B]
            _so_bad = ~torch.isfinite(_so)
            logger.warning(
                "  NAN-GUARD segm_out non-finite=%d (kernel-side) ; "
                "per-bad-row segm_out_nonfinite=%s",
                int(_so_bad.sum()),
                [int(_so_bad[b].sum()) for b in _bad],
            )
            _dump = os.environ.get("VLLM_TQ_V4_NAN_DUMP", "")
            if _dump and _V4_NAN_HITS == 1:
                try:
                    b0 = _bad[0]
                    _bps = block_table.shape[1]
                    _nblk = (int(seq_lens[b0]) + block_size - 1) // block_size
                    _bt0 = block_table[b0, :_nblk].to(torch.long)
                    torch.save({
                        "query": query[b0:b0 + 1].detach().cpu(),
                        "seq_len": int(seq_lens[b0]),
                        "block_table_row": block_table[b0:b0 + 1].detach().cpu(),
                        "kv_blocks": kv_cache.index_select(
                            0, _bt0).detach().cpu(),
                        "kv_block_ids": _bt0.detach().cpu(),
                        "kv_stride": tuple(kv_cache.stride()),
                        "kv_shape": tuple(kv_cache.shape),
                        "centroids": centroids.detach().cpu(),
                        "Pi": Pi.detach().cpu(),
                        "PiT": (PiT.detach().cpu() if PiT is not None else None),
                        "scale": float(scale),
                        "sliding_window": int(_swa),
                        "num_partitions": int(num_partitions),
                        "QG": int(QG), "Hk": int(Hk), "D": int(D),
                        "block_size": int(block_size),
                        "segm_out_b0": segm_out[b0].detach().cpu(),
                        "segm_max_b0": segm_max[b0].detach().cpu(),
                        "segm_sum_b0": segm_sum[b0].detach().cpu(),
                    }, _dump)
                    logger.warning("  NAN-GUARD dumped seed inputs to %s "
                                   "(row=%d seq_len=%d)", _dump, b0,
                                   int(seq_lens[b0]))
                except Exception as _de:  # noqa: BLE001
                    logger.warning("  NAN-GUARD dump failed: %r", _de)
    if sinks is not None and not _LOG_SINKS_WARNED:
        _LOG_SINKS_WARNED = True
        logger.warning(
            "FlyDSL v4 launcher: sinks ignored (NYI). Disable sinks or "
            "use VLLM_TQ_DECODE_V3 if sinks are required."
        )
    # ── norm_correction is honored IMPLICITLY ───────────────────────────
    # When the model was stored with norm_correction=True (the *_nc presets
    # turboquant_4bit_nc / k3v4_nc / 3bit_nc), the per-token K-norm scalar
    # was pre-folded to ||k_t|| / ||c_t|| at store time by
    # triton_turboquant_store._store_packed_key step 3 (see lines 339-349:
    # `vn_f32 = vn_f32 * c_inv_norm`). The decode kernel just multiplies
    # `c_vals * stored_knorm` (kernels/tq_decode_v4.py line 388:
    # `cent_f32 * knorm_f32`), which then equals
    # `(c_vals / ||c_t||) * ||k_t||` — exactly the unit-norm-renormalized
    # centroid times the original key norm. v3 does the identical multiply
    # (triton_turboquant_decode.py line 416-417). Therefore both decoders
    # honor norm_correction equivalently and there is nothing to "do" at
    # decode time. The launcher's `norm_correction` arg is only kept for
    # API parity. We emit a one-shot INFO log to make the contract explicit
    # rather than the previous misleading "NYI" warning.
    if norm_correction and not _LOG_NORM_WARNED:
        _LOG_NORM_WARNED = True
        logger.info(
            "FlyDSL v4 launcher: norm_correction honored implicitly via "
            "pre-folded stored K-norm (cf. triton_turboquant_store step 3); "
            "no decode-time work required, identical to v3 behavior."
        )
    return output
