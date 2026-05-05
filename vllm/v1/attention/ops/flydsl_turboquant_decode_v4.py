# SPDX-License-Identifier: Apache-2.0
"""FlyDSL TurboQuant decode v4 launcher (vLLM-side).

Drop-in replacement for ``triton_turboquant_decode_attention_v3`` for the
Qwen-class TQ decode profile (HEAD_SIZE=128, GQA group=16, MSE_BITS=4 K,
VQB=4 V, N_CENTROIDS=16, BLOCK_SIZE=16).

Opt-in via ``VLLM_TQ_DECODE_V4=1``. Falls back to v3 if FlyDSL is not
importable (e.g. wrong arch, missing build tree).

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
_TQ_MOD = None  # kernels.tq_decode_v4 module
_FLYC = None    # flydsl.compiler
_FX = None      # flydsl.expr
_TYPING_T = None
_CC = None      # CompilationContext
_IR = None


def is_flydsl_available() -> bool:
    """Return True iff FlyDSL imports + kernel module load successfully."""
    global _FLYDSL_AVAILABLE, _TQ_MOD, _FLYC, _FX, _TYPING_T, _CC, _IR
    if _FLYDSL_AVAILABLE is not None:
        return _FLYDSL_AVAILABLE
    try:
        _ensure_flydsl_paths()
        import flydsl.compiler as flyc  # noqa: F401
        import flydsl.expr as fx  # noqa: F401
        from flydsl.expr.typing import T  # noqa: F401
        from flydsl.compiler.kernel_function import CompilationContext
        from flydsl._mlir import ir
        import kernels.tq_decode_v4 as tq_mod
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


class _SegmBufPool:
    """Per-shape buffer pool for segm_out/segm_max/segm_sum/output.

    Eliminates per-decode-step ``cudaMalloc`` (× 4) plus the device-side
    memset kernels behind ``torch.full(-inf)`` and ``torch.zeros``.

    Buffers are keyed by the full shape signature, so cudagraph capture
    sees a stable address per shape (which is what cudagraph requires).
    The kernel always writes the FULL ``[B, Hk, P, QG, D]`` slice and
    every (n, kv_h, p, qg) position of segm_max / segm_sum (it stores
    ``-inf`` and ``0`` for empty partitions itself), so no reset is
    needed between calls at the same shape.
    """

    __slots__ = ("_bufs",)

    def __init__(self) -> None:
        self._bufs: dict[tuple, dict[str, torch.Tensor]] = {}

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
        key = (
            int(B), int(Hk), int(Hq), int(num_partitions),
            int(QG), int(D), str(device), q_dtype,
        )
        bufs = self._bufs.get(key)
        if bufs is None:
            bufs = {
                "segm_out": torch.empty(
                    (B, Hk, num_partitions, QG, D),
                    dtype=torch.bfloat16, device=device,
                ),
                "segm_max": torch.empty(
                    (B, Hk, num_partitions, QG),
                    dtype=torch.float32, device=device,
                ),
                "segm_sum": torch.empty(
                    (B, Hk, num_partitions, QG),
                    dtype=torch.float32, device=device,
                ),
                "output": torch.empty(
                    (B, Hq, D), dtype=q_dtype, device=device,
                ),
            }
            self._bufs[key] = bufs
        return bufs

    def stats(self) -> dict[str, int]:
        return {
            "shapes": len(self._bufs),
            "bytes": sum(
                sum(t.numel() * t.element_size() for t in d.values())
                for d in self._bufs.values()
            ),
        }


_SEGM_POOL = _SegmBufPool()


_HW_TR_CACHED: bool | None = None


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


def _get_kernel(num_kv_heads: int, num_partitions: int,
                max_blocks_per_seq: int, scale: float,
                query_group_size: int, kv_block_size: int,
                use_hw_v_transpose: bool = False):
    key = (num_kv_heads, int(num_partitions), int(max_blocks_per_seq),
           round(float(scale), 8), int(query_group_size), int(kv_block_size),
           bool(use_hw_v_transpose))
    cached = _KERN_CACHE.get(key)
    if cached is not None:
        return cached
    assert is_flydsl_available()
    kfn = _TQ_MOD.build_tq_decode_v4_module(
        num_seqs=1,  # not used in body
        num_kv_heads=num_kv_heads,
        num_partitions=num_partitions,
        max_blocks_per_seq=max_blocks_per_seq,
        softmax_scale=float(scale),
        query_group_size=int(query_group_size),
        kv_block_size=int(kv_block_size),
        use_hw_v_transpose=bool(use_hw_v_transpose),
    )
    al = _TQ_MOD.allocator
    block_threads = _TQ_MOD.BLOCK_THREADS

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
) -> torch.Tensor:
    """v3-compatible launcher backed by the FlyDSL v4 decode kernel.

    Constraints (Qwen-class profile only):
      * key_fp8 == False
      * mse_bits == 4
      * value_quant_bits == 4
      * centroids.numel() == 16
      * D == 128
      * block_size == 16
      * Hq // Hk in {8, 16}  (Qwen 72B GQA-8 or Qwen 32B GQA-16)

    Sinks/norm_correction are not implemented and silently ignored if
    set; the caller is expected to disable them when toggling v4.
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
    assert D == _TQ_MOD.HEAD_SIZE
    assert block_size in (16, 32), (
        f"v4 supports kv_block_size 16 or 32, got {block_size}"
    )
    assert QG in (8, 16), f"v4 supports GQA factor 8 or 16, got {QG}"
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

    q_rot = (query.float() @ PiT_f32).to(query.dtype).contiguous()

    # ---- Partition count (FA2 split-KV) ----------------------------------
    if max_seq_len <= 0:
        # Cheap upper bound from block table; no GPU sync.
        max_seq_len = int(block_table.shape[1]) * int(block_size)
    kv_compute_block = _TQ_MOD.KV_COMPUTE_BLOCK
    num_partitions_actual = max(
        1, min(max_num_kv_splits,
               (int(max_seq_len) + kv_compute_block - 1) // kv_compute_block)
    )
    # Round up to next power of 2 for the Triton reducer's tl.arange(0, N)
    # constraint (Triton requires power-of-2 ≥ 2). The FlyDSL kernel iterates
    # exactly num_partitions partitions; for padded partitions whose K-tile
    # start exceeds seq_len, the kernel's per-tile mask rejects all tokens and
    # writes -inf / 0 to segm_max / segm_sum (per the empty-partition
    # contract in this module's docstring).
    num_partitions = max(2, triton.next_power_of_2(num_partitions_actual))

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

    # ---- FlyDSL kernel launch -------------------------------------------
    max_bps = int(block_table.shape[1])
    use_hw_tr = _hw_tr_enabled()
    launch = _get_kernel(
        Hk, num_partitions, max_bps, scale, QG, block_size,
        use_hw_v_transpose=use_hw_tr,
    )
    # T1.3: zero-overhead one-shot info log (replaces logger.info_once which
    # hashes its format string on every call to dedup).
    global _LOG_INVOKED_ONCE, _LOG_SINKS_WARNED, _LOG_NORM_WARNED
    if not _LOG_INVOKED_ONCE:
        _LOG_INVOKED_ONCE = True
        logger.info(
            "FlyDSL v4 launcher invoked: B=%d Hk=%d Hq=%d D=%d QG=%d "
            "num_partitions=%d (actual=%d) max_bps=%d block_size=%d "
            "max_seq_len=%d hw_v_transpose=%s (Tier-1: PiT_f32 cache, "
            "segm pool, logger dedup, centroids cache)",
            B, Hk, Hq, D, QG, num_partitions, num_partitions_actual,
            max_bps, int(block_size), int(max_seq_len), use_hw_tr,
        )
    launch(
        segm_out, segm_sum, segm_max,
        q_rot, kv_cache, centroids_c,
        block_table, seq_lens,
        B, Hk, num_partitions,
        torch.cuda.current_stream(),
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
