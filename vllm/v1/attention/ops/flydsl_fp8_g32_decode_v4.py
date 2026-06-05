# SPDX-License-Identifier: Apache-2.0
"""FlyDSL fp8_g32 decode v4 launcher (vLLM-side).

Drop-in alternative to ``fp8_g32_unified_attention`` (Triton v3) for the
fp8_g32 decode profile: HEAD_SIZE=128, FP4 E2M1 codes + UE8M0 per-group-of-32
scales, BLOCK_SIZE in {16, 32}, GQA group in {8, 16}.

This launcher is a direct sibling of ``flydsl_turboquant_decode_v4.py`` and
reuses its framework verbatim (FlyDSL bootstrap, single-bucket segm-pool,
kernel-module cache, Triton partition reducer). Only the bits that differ for
the fp8_g32 algorithm change:

  * Kernel module is ``kernels.fp8_g32_decode_v4`` (built with the cache's
    ``padded_slot`` slot stride + Arch-A/B mode).
  * "Centroids" passed to the kernel = the fixed FP4 E2M1 value table
    (``FP4_BITS_TO_VALUE``), so the LDS LUT directly decodes a nibble to its
    FP4 value. No learned centroids.
  * Q rotation matches the reference + Triton v3: ``q_rot = query.float() @
    PiT`` (PiT = the Sylvester Hadamard), then round-tripped through FP8 E4M3
    (the precision haircut the QK ``dot_scaled`` applies in v3) before being
    fed as bf16 to the bf16 QK MFMA.

Opt-in via ``VLLM_FP8_G32_DECODE_V4=1``. Falls back to the Triton path if
FlyDSL is not importable (e.g. wrong arch, missing build tree).
"""
from __future__ import annotations

import os
import sys
from typing import Any

import torch
import triton

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
_FP8_MOD = None      # kernels.fp8_g32_decode_v4 module (Qwen GQA-{8,16})
_FP8_MOD_GQA6 = None  # kernels.fp8_g32_decode_v4_gqa6 module (MiniMax GQA-6); may stay None
_GQA6_AVAILABLE: bool | None = None
_FLYC = None    # flydsl.compiler
_FX = None      # flydsl.expr
_TYPING_T = None
_CC = None      # CompilationContext
_IR = None


def is_flydsl_available() -> bool:
    """Return True iff FlyDSL imports + the fp8_g32 kernel module load."""
    global _FLYDSL_AVAILABLE, _FP8_MOD
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
        import kernels.fp8_g32_decode_v4 as fp8_mod
        _FLYC = flyc
        _FX = fx
        _TYPING_T = T
        _CC = CompilationContext
        _IR = ir
        _FP8_MOD = fp8_mod
        _FLYDSL_AVAILABLE = True
        logger.info_once("FlyDSL fp8_g32 decode v4 launcher: available")
        # Best-effort load of the GQA-6 sibling (MiniMax-M2.5). Its absence
        # must NOT disable the canonical Qwen GQA-{8,16} path, so import it
        # separately and only flip the gqa6 flag on success.
        global _FP8_MOD_GQA6, _GQA6_AVAILABLE
        try:
            import kernels.fp8_g32_decode_v4_gqa6 as fp8_mod_gqa6
            _FP8_MOD_GQA6 = fp8_mod_gqa6
            _GQA6_AVAILABLE = True
            logger.info_once(
                "FlyDSL fp8_g32 decode v4 GQA-6 sibling: available")
        except Exception as ex_gqa6:  # noqa: BLE001
            _GQA6_AVAILABLE = False
            logger.warning_once(
                "FlyDSL fp8_g32 decode v4 GQA-6 sibling: unavailable (%s). "
                "MiniMax-class (QG=6) will fall back to Triton fp8_g32 v3.",
                ex_gqa6,
            )
    except Exception as ex:  # noqa: BLE001
        _FLYDSL_AVAILABLE = False
        logger.warning_once(
            "FlyDSL fp8_g32 decode v4 launcher: unavailable (%s). "
            "Falling back to Triton fp8_g32 v3.", ex
        )
    return _FLYDSL_AVAILABLE


def is_flydsl_fp8_gqa6_available() -> bool:
    """True iff FlyDSL is available AND the GQA-6 (MiniMax) sibling loaded.

    Used by the backend eligibility gate to allow QG=6 only when the
    fp8_g32_decode_v4_gqa6 kernel module is importable; otherwise QG=6
    falls back to the Triton fp8_g32 v3 path.
    """
    if not is_flydsl_available():
        return False
    return bool(_GQA6_AVAILABLE)


# -- Kernel module cache -------------------------------------------------------
# Compiling a FlyDSL kernel is expensive; cache by the constexprs that
# parameterize stride math: num_kv_heads, num_partitions, max_blocks_per_seq,
# scale, query_group_size, kv_block_size, padded_slot, hw_v_transpose,
# tile_groups_per_partition, arch_b.
_KERN_CACHE: dict[tuple, Any] = {}

_LOG_INVOKED_ONCE: bool = False
_LOG_SINKS_WARNED: bool = False

# Per-device cached FP4 E2M1 value table (used as the kernel "centroids" LUT).
_FP4_LUT_CACHE: dict[tuple[str, torch.dtype], torch.Tensor] = {}


def _fp4_value_lut(device: torch.device) -> torch.Tensor:
    """Return the fixed 16-entry FP4 E2M1 value table as fp32 [16] on device.

    Indexed by the stored 4-bit E2M1 code, so ``lut[nibble]`` is the dequant
    value directly (matches ``fp8_levels.FP4_BITS_TO_VALUE``).
    """
    from vllm.v1.attention.ops.fp8_g32.fp8_levels import FP4_BITS_TO_VALUE
    key = (str(device), torch.float32)
    t = _FP4_LUT_CACHE.get(key)
    if t is None:
        t = torch.tensor(
            FP4_BITS_TO_VALUE, dtype=torch.float32, device=device
        ).contiguous()
        _FP4_LUT_CACHE[key] = t
    return t


# Per-device cached qperm head-dim permutation index (qk_scaled path only).
# qperm[phys] = the natural head-dim placed at physical operand position phys.
# It realigns the fixed FP4xFP8 scaled-MFMA contraction so that, with K fed
# native-contiguous, Q's columns pair correctly. Folded into PiT on the host
# (PiT[:, qperm]) so the Q rotation matmul emits already-permuted q_rot at zero
# extra runtime cost. Derived + validated in tmp_qk_scaled_mfma.py (contigK).
_QPERM_CACHE: dict[str, torch.Tensor] = {}


def _kmap_phys(group: int, s: int) -> int:
    # Physical operand position of group's element s for the K=128 fp4/fp8
    # scaled MFMA (matches tmp_qk_scaled_mfma.py::_kmap).
    return 32 * (s >> 4) + 64 * (group % 2) + 16 * (group // 2) + (s & 15)


def _qperm_index(device: torch.device, D: int = 128,
                 group_size: int = 32) -> torch.Tensor:
    key = str(device)
    t = _QPERM_CACHE.get(key)
    if t is None:
        n_groups = D // group_size
        qperm = [0] * D
        for g in range(n_groups):
            for s in range(group_size):
                qperm[_kmap_phys(g, s)] = g * group_size + s
        t = torch.tensor(qperm, dtype=torch.long, device=device).contiguous()
        _QPERM_CACHE[key] = t
    return t


def _detect_max_capture_B() -> int:
    env = os.environ.get("VLLM_FP8_G32_DECODE_V4_B_BUCKET")
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
    """Single-bucket buffer pool for segm_out/segm_max/segm_sum/output + the
    pooled Q-rotation intermediates. Identical to the TQ v4 pool except that
    fp8_g32 needs an extra ``q_fp8`` (float8_e4m3fn) slot for the Q precision
    haircut (``q_rot_fp32 -> fp8_e4m3 -> bf16``) so that no fresh allocation
    lands in the HIP graph memory pool post-capture.
    """

    __slots__ = ("_bufs", "_max_B")

    def __init__(self) -> None:
        self._bufs: dict[tuple, dict[str, torch.Tensor]] = {}
        self._max_B: int | None = None

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
        B_bucket = max(self._max_B, int(B))
        key = (
            int(Hk), int(Hq), int(num_partitions),
            int(QG), int(D), str(device), q_dtype,
        )
        bufs = self._bufs.get(key)
        if bufs is None or bufs["segm_out"].shape[0] < B_bucket:
            if bufs is not None and bufs["segm_out"].shape[0] < B_bucket:
                logger.warning_once(
                    "FlyDSL fp8_g32 v4 _SegmBufPool: growing bucket from %d "
                    "to %d (B=%d). If this happens AFTER cudagraph warmup the "
                    "captured graphs hold stale pointers and will GPU-fault. "
                    "Set VLLM_FP8_G32_DECODE_V4_B_BUCKET=%d before launch.",
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
                "q_rot": torch.empty(
                    (B_bucket, Hq, D), dtype=q_dtype, device=device,
                ),
                "q_float": torch.empty(
                    (B_bucket, Hq, D), dtype=torch.float32, device=device,
                ),
                "q_rot_fp32": torch.empty(
                    (B_bucket, Hq, D), dtype=torch.float32, device=device,
                ),
                # fp8_g32-specific: E4M3 haircut intermediate (pooled so the
                # cast does not allocate a fresh tensor post-capture).
                "q_fp8": torch.empty(
                    (B_bucket, Hq, D),
                    dtype=torch.float8_e4m3fn, device=device,
                ),
            }
            self._bufs[key] = bufs
            self._max_B = B_bucket
            logger.info_once(
                "FlyDSL fp8_g32 v4 _SegmBufPool: allocated single-bucket "
                "shape=(Hk=%d, Hq=%d, P=%d, QG=%d, D=%d, dtype=%s) "
                "B_bucket=%d. VRAM = %.1f MiB / shape.",
                Hk, Hq, num_partitions, QG, D, q_dtype, B_bucket,
                sum(t.numel() * t.element_size() for t in bufs.values())
                / (1 << 20),
            )
        return {
            "segm_out": bufs["segm_out"][:B],
            "segm_max": bufs["segm_max"][:B],
            "segm_sum": bufs["segm_sum"][:B],
            "output": bufs["output"][:B],
            "q_rot": bufs["q_rot"][:B],
            "q_float": bufs["q_float"][:B],
            "q_rot_fp32": bufs["q_rot_fp32"][:B],
            "q_fp8": bufs["q_fp8"][:B],
        }


_SEGM_POOL = _SegmBufPool()

_HW_TR_CACHED: bool | None = None


def _hw_tr_enabled() -> bool:
    """Resolve the HW V-transpose build flag (default ON for gfx950+)."""
    global _HW_TR_CACHED
    if _HW_TR_CACHED is not None:
        return _HW_TR_CACHED
    env = os.environ.get("VLLM_FP8_G32_DECODE_V4_HW_TR")
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
        logger.info_once(
            "FlyDSL fp8_g32 v4: HW V transpose ON (default for gfx950+)")
    else:
        logger.info_once("FlyDSL fp8_g32 v4: HW V transpose OFF")
    return _HW_TR_CACHED


_GET_KERNEL_STATS = {"hits": 0, "misses": 0, "build_total_s": 0.0}


def _get_kernel(num_kv_heads: int, num_partitions: int,
                max_blocks_per_seq: int, scale: float,
                query_group_size: int, kv_block_size: int,
                padded_slot: int,
                use_hw_v_transpose: bool = False,
                num_seqs_hint: int = 1,
                tile_groups_per_partition: int = 1,
                arch_b: bool = False,
                scale_c: float = 0.156,
                qk_fp8: bool = False,
                qk_scaled: bool = False,
                v_cvt: bool = False,
                q_hoist: bool = False):
    key = (num_kv_heads, int(num_partitions), int(max_blocks_per_seq),
           round(float(scale), 8), int(query_group_size), int(kv_block_size),
           int(padded_slot), bool(use_hw_v_transpose),
           int(tile_groups_per_partition), bool(arch_b),
           round(float(scale_c), 8), bool(qk_fp8), bool(qk_scaled),
           bool(v_cvt), bool(q_hoist))
    cached = _KERN_CACHE.get(key)
    if cached is not None:
        _GET_KERNEL_STATS["hits"] += 1
        return cached
    assert is_flydsl_available()
    import time as _t
    _build_t0 = _t.perf_counter()

    # GQA-6 (MiniMax) dispatches to the sibling module/build fn; Qwen
    # GQA-{8,16} use the canonical kernel. The two modules expose distinct
    # MLIR smem symbols + kernel names so both can be JIT-resident at once.
    if int(query_group_size) == 6:
        assert _FP8_MOD_GQA6 is not None, (
            "query_group_size=6 requires the fp8_g32_decode_v4_gqa6 sibling, "
            "which failed to import; check is_flydsl_fp8_gqa6_available()."
        )
        kmod = _FP8_MOD_GQA6
        # Forward every perf knob (hw V-transpose, FA-2 split-K, Arch A/B,
        # qk_fp8 Step-A MFMA) so GQA-6 has the SAME optimizations as the
        # canonical GQA-{8,16} kernel. use_wht_butterfly is left at its
        # default (False): the launcher always feeds pre-rotated q_rot, so
        # the in-kernel butterfly is unused on this path.
        kfn = kmod.build_fp8_g32_decode_v4_gqa6_module(
            num_seqs=int(num_seqs_hint),
            num_kv_heads=num_kv_heads,
            num_partitions=num_partitions,
            padded_slot=int(padded_slot),
            max_blocks_per_seq=max_blocks_per_seq,
            softmax_scale=float(scale),
            query_group_size=int(query_group_size),
            kv_block_size=int(kv_block_size),
            use_hw_v_transpose=bool(use_hw_v_transpose),
            tile_groups_per_partition=int(tile_groups_per_partition),
            arch_b=bool(arch_b),
            scale_c=float(scale_c),
            qk_fp8=bool(qk_fp8),
            qk_scaled=bool(qk_scaled),
            v_cvt=bool(v_cvt),
            q_hoist=bool(q_hoist),
        )
    else:
        kmod = _FP8_MOD
        kfn = kmod.build_fp8_g32_decode_v4_module(
            num_seqs=int(num_seqs_hint),
            num_kv_heads=num_kv_heads,
            num_partitions=num_partitions,
            padded_slot=int(padded_slot),
            max_blocks_per_seq=max_blocks_per_seq,
            softmax_scale=float(scale),
            query_group_size=int(query_group_size),
            kv_block_size=int(kv_block_size),
            use_hw_v_transpose=bool(use_hw_v_transpose),
            tile_groups_per_partition=int(tile_groups_per_partition),
            use_wht_butterfly=False,
            arch_b=bool(arch_b),
            scale_c=float(scale_c),
            qk_fp8=bool(qk_fp8),
            qk_scaled=bool(qk_scaled),
            v_cvt=bool(v_cvt),
            q_hoist=bool(q_hoist),
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
        "FlyDSL fp8_g32 v4 _get_kernel BUILD #%d dt=%.2fs (cumulative=%.1fs) "
        "key=%s", _GET_KERNEL_STATS["misses"], _build_dt,
        _GET_KERNEL_STATS["build_total_s"], key,
    )
    return _launch


# -- Partition reducer (Triton, native v4 layout) ------------------------------
# Identical to the TQ v4 reducer (layout-only, no algorithm specifics).
import triton.language as tl  # noqa: E402


@triton.jit
def _reduce_partitions_v4(
    output_ptr,              # [N, Hq, D] in OUT_DTYPE
    segm_out_ptr,            # [N, Hk, P, QG, D] bf16
    segm_max_ptr,            # [N, Hk, P, QG] fp32
    segm_sum_ptr,            # [N, Hk, P, QG] fp32
    out_stride_n: tl.int64,
    out_stride_h: tl.int64,
    NUM_KV_HEADS: tl.constexpr,
    QG: tl.constexpr,
    NUM_PARTS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
):
    n = tl.program_id(0)
    hq = tl.program_id(1)
    kv_h = hq // QG
    qg = hq % QG

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

    m_idx = msum_base + p_off * QG
    seg_max = tl.load(segm_max_ptr + m_idx)
    seg_sum = tl.load(segm_sum_ptr + m_idx)

    valid = seg_max > float("-inf")
    overall_max = tl.max(tl.where(valid, seg_max, float("-inf")))

    rescale = tl.where(valid, tl.exp(seg_max - overall_max), 0.0)
    seg_sum_rescaled = seg_sum * rescale
    overall_sum = tl.sum(seg_sum_rescaled)

    so_idx = (
        so_base
        + p_off[:, None] * (QG * HEAD_SIZE)
        + d_off[None, :]
    )
    seg_out = tl.load(segm_out_ptr + so_idx).to(tl.float32)
    weighted = seg_out * seg_sum_rescaled[:, None]
    acc_sum = tl.sum(weighted, axis=0)
    acc = tl.where(overall_sum > 0.0, acc_sum / overall_sum, 0.0)

    out_off = n * out_stride_n + hq * out_stride_h + d_off
    tl.store(output_ptr + out_off, acc.to(output_ptr.dtype.element_ty))


# -- Public launcher ----------------------------------------------------------
def flydsl_fp8_g32_decode_attention_v4(
    query: torch.Tensor,            # [B, Hq, D] bf16/fp16 (raw, unrotated)
    kv_cache: torch.Tensor,         # [num_blocks, BS, Hk, padded_slot] uint8
    block_table: torch.Tensor,      # [B, max_blocks_per_seq] int32
    seq_lens: torch.Tensor,         # [B] int32
    scale: float,
    PiT: torch.Tensor | None = None,  # [D, D] fp32 Hadamard (= layer._tq_PiT)
    max_seq_len: int = 0,
    output_buf: torch.Tensor | None = None,
    buf_holder: Any = None,
    max_num_kv_splits: int = 32,
    sinks: torch.Tensor | None = None,
) -> torch.Tensor:
    """Triton-v3-compatible launcher backed by the FlyDSL fp8_g32 decode kernel.

    Constraints:
      * D == 128
      * block_size in {16, 32}
      * Hq // Hk in {8, 16}
      * sinks is None (NYI on this path)

    The fp8_g32 cache slot layout (AoS) is read from ``kv_cache.shape[3]``
    (= padded_slot). Arch A/B + constant ``c`` are read from fp8_levels.
    """
    if not is_flydsl_available():
        raise RuntimeError(
            "VLLM_FP8_G32_DECODE_V4 requested but FlyDSL is not available; "
            "set VLLM_FP8_G32_DECODE_V4=0 or fix "
            "VLLM_FLYDSL_ROOT/VLLM_FLYDSL_PKGS."
        )

    from vllm.v1.attention.ops.fp8_g32.fp8_levels import (
        get_constant_c,
        is_arch_b,
    )

    B, Hq, D = query.shape
    Hk = kv_cache.shape[2]
    block_size = kv_cache.shape[1]
    padded_slot = int(kv_cache.shape[3])
    QG = Hq // Hk
    assert D == _FP8_MOD.HEAD_SIZE, (
        f"fp8_g32 v4 expects D={_FP8_MOD.HEAD_SIZE}, got {D}"
    )
    assert block_size in (16, 32), (
        f"fp8_g32 v4 supports kv_block_size 16 or 32, got {block_size}"
    )
    if QG == 6:
        # MiniMax-class GQA-6 routes to the sibling kernel; require it loaded.
        assert is_flydsl_fp8_gqa6_available(), (
            "fp8_g32 v4 GQA-6 (QG=6) requires the fp8_g32_decode_v4_gqa6 "
            "sibling, which is unavailable; caller should not have dispatched "
            "here (see is_flydsl_fp8_gqa6_available())."
        )
    else:
        assert QG in (8, 16), (
            f"fp8_g32 v4 supports GQA factor 6, 8 or 16, got {QG}"
        )

    arch_b = bool(is_arch_b())
    scale_c = float(get_constant_c())

    device = query.device

    # ---- PiT (Hadamard) — cache the contiguous fp32 form on the layer -----
    PiT_f32: torch.Tensor
    if buf_holder is not None:
        PiT_f32 = getattr(buf_holder, "_fp8_v4_PiT_f32", None)
        if PiT_f32 is None:
            assert PiT is not None, (
                "fp8_g32 v4 launcher requires PiT (the Hadamard rotation)"
            )
            PiT_f32 = (
                PiT if PiT.dtype == torch.float32 else PiT.to(torch.float32)
            )
            if not PiT_f32.is_contiguous():
                PiT_f32 = PiT_f32.contiguous()
            buf_holder._fp8_v4_PiT_f32 = PiT_f32
    else:
        assert PiT is not None, (
            "fp8_g32 v4 launcher requires PiT (the Hadamard rotation)"
        )
        PiT_f32 = (
            PiT if PiT.dtype == torch.float32 else PiT.to(torch.float32)
        )
        if not PiT_f32.is_contiguous():
            PiT_f32 = PiT_f32.contiguous()

    centroids_c = _fp4_value_lut(device)

    # ---- Partition count (FA2 split-KV) — identical policy to TQ v4 -------
    kv_compute_block = _FP8_MOD.KV_COMPUTE_BLOCK
    worst_case_max_seq_len = int(block_table.shape[1]) * int(block_size)
    if max_seq_len <= 0:
        max_seq_len = worst_case_max_seq_len
    if os.environ.get("VLLM_FP8_G32_DECODE_V4_DYNAMIC_PARTS", "0") == "1":
        sizing_max_seq_len = int(max_seq_len)
    else:
        sizing_max_seq_len = worst_case_max_seq_len

    MAX_PARTITIONS = int(os.environ.get(
        "VLLM_FP8_G32_DECODE_V4_MAX_PARTITIONS", "32"))
    MAX_PARTITIONS = max(2, MAX_PARTITIONS)
    required_num_partitions = (
        sizing_max_seq_len + kv_compute_block - 1) // kv_compute_block
    parallelism_floor = min(MAX_PARTITIONS, max(1, max_num_kv_splits))
    num_partitions_actual = max(
        parallelism_floor,
        min(MAX_PARTITIONS, required_num_partitions),
    )
    num_partitions = max(2, triton.next_power_of_2(num_partitions_actual))
    _tgpp_required = max(
        1,
        (required_num_partitions + num_partitions - 1) // num_partitions,
    )
    tile_groups_per_partition = int(triton.next_power_of_2(_tgpp_required))

    # ---- Pooled buffers ---------------------------------------------------
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

    # ---- QK MFMA path selection ------------------------------------------
    # Default ON: native CDNA4 scaled FP4xFP8 MFMA QK path (gfx950). Set
    # VLLM_FP8_G32_DECODE_V4_QK_SCALED=0 to fall back to the bf16 MFMA path.
    qk_fp8 = os.environ.get("VLLM_FP8_G32_DECODE_V4_QK_FP8", "0") == "1"
    qk_scaled = os.environ.get("VLLM_FP8_G32_DECODE_V4_QK_SCALED", "1") == "1"
    assert not (qk_fp8 and qk_scaled), (
        "VLLM_FP8_G32_DECODE_V4_QK_FP8 and _QK_SCALED are mutually exclusive"
    )
    # ---- V dequant path selection ----------------------------------------
    # Default ON: native CDNA4 scaled FP4->bf16 CVT for V dequant (gfx950);
    # requires the HW V-transpose LDS layout. Set
    # VLLM_FP8_G32_DECODE_V4_V_CVT=0 to fall back to the software LUT path.
    v_cvt = os.environ.get("VLLM_FP8_G32_DECODE_V4_V_CVT", "1") == "1"
    # Default ON: hoist the loop-invariant scaled-MFMA Q operand out of the
    # K-tile loop (build fp8 Q once in STEP C, reuse). No-op unless qk_scaled.
    # Set VLLM_FP8_G32_DECODE_V4_Q_HOIST=0 to disable.
    q_hoist = os.environ.get("VLLM_FP8_G32_DECODE_V4_Q_HOIST", "1") == "1"

    # ---- Q rotation + FP8 E4M3 haircut -----------------------------------
    # q_rot = query.float() @ PiT  -> cast to FP8 E4M3 (precision haircut the
    # QK dot_scaled applies in Triton v3 / the reference) -> bf16 for the
    # bf16 QK MFMA. All via stable pooled buffers (no post-capture alloc).
    #
    # qk_scaled: the head-dim qperm permutation is folded into PiT's COLUMNS
    # (PiT_used = PiT[:, qperm]) so the matmul directly produces native-
    # contiguous-operand-order q_rot for the scaled FP4xFP8 MFMA — no extra
    # gather, no new buffers, no dtype change.
    PiT_used = PiT_f32
    if qk_scaled:
        qperm_idx = _qperm_index(device, D=D)
        if buf_holder is not None:
            PiT_used = getattr(buf_holder, "_fp8_v4_PiT_perm_f32", None)
            if PiT_used is None:
                PiT_used = PiT_f32.index_select(1, qperm_idx).contiguous()
                buf_holder._fp8_v4_PiT_perm_f32 = PiT_used
        else:
            PiT_used = PiT_f32.index_select(1, qperm_idx).contiguous()
    _q_float = pool_bufs["q_float"]
    _q_rot_f32 = pool_bufs["q_rot_fp32"]
    _q_fp8 = pool_bufs["q_fp8"]
    _q_rot_out = pool_bufs["q_rot"]
    _q_float.copy_(query)
    torch.mm(
        _q_float.view(B * Hq, D), PiT_used,
        out=_q_rot_f32.view(B * Hq, D),
    )
    _q_fp8.copy_(_q_rot_f32)       # fp32 -> e4m3 (saturating round)
    _q_rot_out.copy_(_q_fp8)       # e4m3 -> bf16 (exact: e4m3 ⊂ bf16)
    q_for_kernel = _q_rot_out

    # ---- FlyDSL kernel launch --------------------------------------------
    max_bps = int(block_table.shape[1])
    use_hw_tr = _hw_tr_enabled()
    launch = _get_kernel(
        Hk, num_partitions, max_bps, scale, QG, block_size, padded_slot,
        use_hw_v_transpose=use_hw_tr,
        num_seqs_hint=int(B),
        tile_groups_per_partition=int(tile_groups_per_partition),
        arch_b=arch_b,
        scale_c=scale_c,
        qk_fp8=qk_fp8,
        qk_scaled=qk_scaled,
        v_cvt=v_cvt,
        q_hoist=q_hoist,
    )
    global _LOG_INVOKED_ONCE, _LOG_SINKS_WARNED
    if not _LOG_INVOKED_ONCE:
        _LOG_INVOKED_ONCE = True
        logger.info(
            "FlyDSL fp8_g32 v4 launcher invoked: B=%d Hk=%d Hq=%d D=%d QG=%d "
            "num_partitions=%d (actual=%d, cap=%d) TGPP=%d max_bps=%d "
            "block_size=%d padded_slot=%d max_seq_len=%d hw_v_transpose=%s "
            "arch_b=%s c=%.4f (coverage=%d tokens, worst_case=%d tokens)",
            B, Hk, Hq, D, QG, num_partitions, num_partitions_actual,
            MAX_PARTITIONS, tile_groups_per_partition, max_bps,
            int(block_size), padded_slot, int(max_seq_len), use_hw_tr,
            arch_b, scale_c,
            num_partitions * tile_groups_per_partition * kv_compute_block,
            worst_case_max_seq_len,
        )
    launch(
        segm_out, segm_sum, segm_max,
        q_for_kernel, kv_cache, centroids_c,
        block_table, seq_lens,
        B, Hk, num_partitions,
        torch.cuda.current_stream(),
    )

    # ---- Reduce partitions -> [B, Hq, D] ---------------------------------
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
    if sinks is not None and not _LOG_SINKS_WARNED:
        _LOG_SINKS_WARNED = True
        logger.warning(
            "FlyDSL fp8_g32 v4 launcher: sinks ignored (NYI). Disable sinks "
            "or use VLLM_FP8_G32_V3 if sinks are required."
        )
    return output
