# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant attention backend for vLLM.

Prefill: Standard scaled dot-product attention on uncompressed K/V,
         then quantize K and store K+V into combined cache slot.
Decode:  Compute TQ attention scores from compressed cache,
         unpack FP16 values, softmax + weighted sum.

Cache layout (no leading 2 dimension):
  (num_blocks, block_size, num_kv_heads, slot_size)
  where slot_size = key_packed_size + value_fp16_size

Per-head per-position slot layout:
  [key_packed (kps bytes) | value_fp16 (D*2 bytes)]
  For turboquant_k3v4_nc head_dim=256: [100 bytes key | 512 bytes value] = 612
"""

import functools
import math
import os
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
import torch.nn.functional as F

from vllm.config import get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.turboquant.centroids import (
    get_centroids,
)
from vllm.triton_utils import triton
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.fa_utils import (
    is_flash_attn_varlen_func_available,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.attention.ops.triton_turboquant_decode import (
    _tq_full_dequant_kv,
    _use_fp8_e4b15,
    triton_turboquant_decode_attention,
)
from vllm.v1.attention.ops.triton_turboquant_decode_v2 import (
    triton_turboquant_decode_attention_v2,
)
from vllm.v1.attention.ops.triton_turboquant_store import triton_turboquant_store


def _lazy_fp4_g32_imports():
    """Lazy imports for FP4-g32 kernels — avoids Triton import cost on paths
    that don't use fp4_kv_g32. Returns (fp4_g32_store, fp4_g32_decode_attention)."""
    from vllm.v1.attention.ops.fp4_g32.triton_store import fp4_g32_store
    from vllm.v1.attention.ops.fp4_g32.triton_decode import (
        fp4_g32_decode_attention,
        fp4_g32_dequant_cached_kv,
    )
    return fp4_g32_store, fp4_g32_decode_attention, fp4_g32_dequant_cached_kv


def _lazy_fp4_g32_v3_import():
    """Lazy import for the v3-based FP4-g32 unified attention kernel."""
    from vllm.v1.attention.ops.fp4_g32.triton_unified_attention import (
        fp4_g32_unified_attention,
    )
    return fp4_g32_unified_attention


def _lazy_fp8_g32_imports():
    """Lazy imports for FP8-g32 kernels — avoids Triton import cost on paths
    that don't use fp8_kv_g32. Returns
    (fp8_g32_store, fp8_g32_decode_attention, fp8_g32_full_dequant_kv)."""
    from vllm.v1.attention.ops.fp8_g32.triton_store import fp8_g32_store
    from vllm.v1.attention.ops.fp8_g32.triton_decode import (
        fp8_g32_decode_attention,
        fp8_g32_full_dequant_kv,
    )
    return fp8_g32_store, fp8_g32_decode_attention, fp8_g32_full_dequant_kv


def _lazy_fp8_g32_v3_import():
    """Lazy import for FP8-g32 V3 unified attention kernel.

    Mirrors `_lazy_fp4_g32_v3_import`. Returns the launcher
    ``fp8_g32_unified_attention``. This kernel uses AMD CDNA4's hardware
    scaled F8F6F4 MFMA via ``tl.dot_scaled`` for QK (FP8 E4M3 query × FP4
    E2M1 keys with E8M0 per-group-32 scales). PV stays bf16 ``tl.dot``
    because our V scale axis doesn't match microscaling rhs layout.
    """
    from vllm.v1.attention.ops.fp8_g32.triton_unified_attention import (
        fp8_g32_unified_attention,
    )
    return fp8_g32_unified_attention


def _fp4_get_layer_dequant_bufs(
    layer: Any,
    Hk: int,
    max_seq: int,
    D: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fallback layer-cached dequant target buffers for FP4-g32 continuation
    prefill — used when WorkspaceManager is unavailable or too small.

    Mirrors v1's `_tq_k_dequant_buf` / `_tq_v_dequant_buf` pattern. Allocated
    once at worst-case size; never via bare `torch.empty` during graph capture.
    """
    _kbuf = getattr(layer, "_fp4_k_dequant_buf", None)
    if _kbuf is None or _kbuf.shape[2] < max_seq or _kbuf.dtype != dtype:
        _kbuf = torch.empty(1, Hk, max_seq, D, dtype=dtype, device=device)
        layer._fp4_k_dequant_buf = _kbuf
    _vbuf = getattr(layer, "_fp4_v_dequant_buf", None)
    if _vbuf is None or _vbuf.shape[2] < max_seq or _vbuf.dtype != dtype:
        _vbuf = torch.empty(1, Hk, max_seq, D, dtype=dtype, device=device)
        layer._fp4_v_dequant_buf = _vbuf
    return _kbuf, _vbuf


def _lazy_soa_store_imports():
    """Lazy imports to avoid circular import: turboquant_soa_fusion.__init__
    imports from turboquant_attn, so we cannot import it at module level."""
    from vllm.v1.attention.ops.turboquant_soa_fusion.triton_turboquant_store import (
        triton_turboquant_store as soa_store,
    )
    from vllm.v1.attention.ops.turboquant_soa_fusion.external_ops import (
        _load_soa_fused_store as load_soa_fn,
        _soa_fused_store_safe as soa_safe,
    )
    from vllm.v1.attention.ops.turboquant_soa_fusion.triton_turboquant_decode import (
        _tq_full_dequant_kv as soa_dequant,
    )
    return soa_store, load_soa_fn, soa_safe, soa_dequant


def _lazy_soa_decode_v3():
    """Return the SoA-layout-aware v3 decode kernel.

    Must be kept lazy (same reason as _lazy_soa_store_imports): the
    turboquant_soa_fusion package imports from turboquant_attn at init time,
    so importing it at module level causes a circular import.
    """
    from vllm.v1.attention.ops.turboquant_soa_fusion.external_ops import (
        triton_turboquant_decode_attention_v3 as soa_v3,
    )
    return soa_v3
from vllm.v1.attention.ops.triton_turboquant_unified_attention import (
    triton_turboquant_decode_attention_v3,
)
from vllm.v1.attention.ops.flydsl_turboquant_decode_v4 import (
    flydsl_turboquant_decode_attention_v4,
    is_flydsl_available as _flydsl_v4_available,
    is_flydsl_gqa6_available as _flydsl_v4_gqa6_available,
    is_flydsl_hd256_available as _flydsl_v4_hd256_available,
    is_flydsl_gqa6_hd256_available as _flydsl_v4_gqa6_hd256_available,
)
from vllm.v1.attention.ops.flydsl_fp8_g32_decode_v4 import (
    flydsl_fp8_g32_decode_attention_v4,
    is_flydsl_available as _flydsl_fp8_v4_available,
    is_flydsl_fp8_gqa6_available as _flydsl_fp8_v4_gqa6_available,
    is_flydsl_fp8_hd256_available as _flydsl_fp8_v4_hd256_available,
)
# v5 FUSED: isolated fused-epilogue variant (single-kernel decode+combine).
# Imported lazily-safe here; only USED when VLLM_FP8_G32_DECODE_V5_FUSED=1.
# Its availability probes are independent of v4 so a fused-import failure can
# never disable the shipped UQ-adaptive v4 path.
from vllm.v1.attention.ops.flydsl_fp8_g32_decode_v5_fused import (
    flydsl_fp8_g32_decode_attention_v5_fused,
    is_flydsl_fp8_hd256_available as _flydsl_fp8_v5_hd256_available,
)
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)

logger = init_logger(__name__)

# Opt-in flag to dispatch decode path to the v2 Triton kernel.
# v1 remains the default. Set VLLM_TQ_DECODE_V2=1 to enable v2.
# Set VLLM_TQ_DECODE_V3=1 to enable v3 (unified prefill+decode kernel with
# 2D/3D split-KV dispatch and BLOCK_M=128 prefill heuristic). v3 supersedes
# v2 when enabled.
# Set VLLM_TQ_DECODE_V4=1 to enable v4 (FlyDSL CDNA4 wide-K MFMA decode
# kernel; MI355X / gfx950 only, MSE-key path, HEAD_SIZE=128, GQA in
# {6, 8, 16}). GQA={8,16} routes to canonical kernel (Qwen-class);
# GQA=6 routes to tq_decode_v4_gqa6 sibling (MiniMax-M2.5). v4 supersedes
# v3 when enabled. Continuation prefill still falls back to v3 since v4
# is decode-only.
_USE_TQ_V2 = os.environ.get("VLLM_TQ_DECODE_V2", "0") == "1"
_USE_TQ_V3 = os.environ.get("VLLM_TQ_DECODE_V3", "0") == "1"
_USE_TQ_V4 = os.environ.get("VLLM_TQ_DECODE_V4", "0") == "1"
_USE_TQ_SOA_FUSION = os.environ.get("VLLM_TQ_SOA_FUSION", "0") == "1"
# Opt-in flag for the v3-based FP4-g32 unified attention kernel. When set,
# the FP4 path routes both decode and continuation-prefill through the new
# `fp4_g32_unified_attention` (`triton_unified_attention.py`), which mirrors
# v3's `tl.dot`-based MFMA structure (GQA stacking into BLOCK_M, 2D/3D
# split-KV dispatch, fused Q rotation, main/tail split). When unset, the
# FP4 path keeps the legacy v1-based decode + dequant-and-flash_attn path.
_USE_FP4_G32_V3 = os.environ.get("VLLM_FP4_G32_V3", "0") == "1"
# Opt-in flag for the v3-based FP8-g32 unified attention kernel. When set,
# the FP8 path routes both decode and small-continuation-prefill through
# `fp8_g32_unified_attention` (`triton_unified_attention.py`), which lowers
# QK to AMD CDNA4's hardware scaled F8F6F4 MFMA via `tl.dot_scaled`
# (FP8 E4M3 × FP4 E2M1 with E8M0 per-group-32 scales). PV stays bf16
# `tl.dot` (same path as fp4_g32 V3). When unset, the FP8 path keeps the
# legacy v1-based decode + dequant-and-flash_attn path.
_USE_FP8_G32_V3 = os.environ.get("VLLM_FP8_G32_V3", "1") == "1"
# Opt-in flag for the FlyDSL fp8_g32 decode kernel (MI355X / gfx950 only,
# HEAD_SIZE=128, GQA in {8, 16}, no sinks/SWA). When set and FlyDSL is
# importable + the layer is eligible, the fp8_g32 DECODE path routes to
# `flydsl_fp8_g32_decode_attention_v4` (the FlyDSL port of the bug-free TQ v4
# kernel adapted to FP4 E2M1 + UE8M0 group scales). Ineligible layers and
# continuation-prefill fall back to the fp8_g32 Triton path (V3 if also set,
# else V1). Independent of VLLM_FP8_G32_V3.
_USE_FP8_G32_V4 = os.environ.get("VLLM_FP8_G32_DECODE_V4", "1") == "1"
# Opt-in flag for the FUSED single-kernel fp8_g32 decode (v5). When set (and
# v4 is also enabled + the layer is D=256 hd256-eligible), the hd256 decode
# routes to `flydsl_fp8_g32_decode_attention_v5_fused`, which folds the
# partition combine (and store) into the decode epilogue -- eliminating the
# separate Triton reduce kernel and the segm_* HBM round-trip. Default OFF:
# when unset the path is byte-for-byte the v4 UQ-adaptive kernel. This flag is
# strictly additive and never alters the v4 code path.
_USE_FP8_G32_V5_FUSED = os.environ.get(
    "VLLM_FP8_G32_DECODE_V5_FUSED", "1") == "1"
# Diagnostic: shadow-compare the FlyDSL fp8_g32 v4 decode against the Triton v3
# unified decode on IDENTICAL inputs every decode step, logging the first/each
# step where they diverge (cos < threshold). Localizes the D=256 autoregressive
# break that single-step offline parity misses. FlyDSL output is still returned
# (behavior unchanged) so the real trajectory is preserved.
_FP8_SHADOW_CMP = os.environ.get("VLLM_FP8_SHADOW_CMP", "0") == "1"
_FP8_SHADOW_LOG = os.environ.get(
    "VLLM_FP8_SHADOW_LOG", "/shareddata/adrana/workspace/reports/fp8_shadow.log")
_FP8_SHADOW_THRESH = float(os.environ.get("VLLM_FP8_SHADOW_THRESH", "0.999"))
_FP8_SHADOW_STATE: dict = {"calls": 0, "diverged": 0, "fh": None}


def _shadow_compare_fp8(layer_self, query, kv_cache, attn_metadata, PiT,
                        fly_out):
    """Run Triton v3 fp8_g32 decode on the SAME inputs and log divergence.

    Compares per-sequence cosine similarity between the FlyDSL output (already
    computed, ``fly_out``) and the Triton v3 golden output. Logs any step where
    min per-seq cos < threshold, with seq_lens/batch, to localize the D=256
    autoregressive break. Read-only w.r.t. the returned FlyDSL result.
    """
    import torch as _t
    from vllm.v1.attention.ops.fp8_g32.triton_unified_attention import (
        fp8_g32_unified_attention as _tri_v3,
    )
    st = _FP8_SHADOW_STATE
    st["calls"] += 1
    q = query
    if q.dim() == 3:
        B = q.shape[0]
    else:  # [num_tokens, Hq*D] or flattened — assume [B, Hq, D]
        B = q.shape[0]
    dev = q.device
    qsl = _t.arange(B + 1, dtype=_t.int32, device=dev)
    tri = _tri_v3(
        query=q, kv_cache=kv_cache, block_table=attn_metadata.block_table,
        seq_lens=attn_metadata.seq_lens, query_start_loc=qsl,
        scale=layer_self.scale, PiT=PiT, max_query_len=1,
        max_seq_len=int(attn_metadata.max_seq_len),
        sinks=getattr(layer_self, "sinks", None), sliding_window=None,
    )
    f = fly_out.reshape(B, -1).float()
    t = tri.reshape(B, -1).float()
    c = _t.nn.functional.cosine_similarity(f, t, dim=1)  # [B]
    seq_lens = attn_metadata.seq_lens[:B].tolist()
    cmin = float(c.min()); cmean = float(c.mean())
    worst = int(c.argmin())
    if st["fh"] is None:
        st["fh"] = open(_FP8_SHADOW_LOG, "w")
        st["fh"].write("call\tB\tcos_min\tcos_mean\tworst_seq\tworst_seqlen\t"
                       "maxerr\tall_seqlens\n")
    if cmin < _FP8_SHADOW_THRESH:
        st["diverged"] += 1
        me = float((f - t).abs().max())
        st["fh"].write(
            f"{st['calls']}\t{B}\t{cmin:.5f}\t{cmean:.5f}\t{worst}\t"
            f"{seq_lens[worst]}\t{me:.4f}\t{seq_lens}\n")
        st["fh"].flush()
        logger.warning_once(
            "fp8 SHADOW DIVERGE: cos_min=%.5f at seqlen=%d (B=%d) — see %s",
            cmin, seq_lens[worst], B, _FP8_SHADOW_LOG)


# Diagnostic: shadow-compare the FlyDSL v4 TQ decode against the Triton v3
# unified TQ decode on IDENTICAL inputs (same kv_cache/centroids/Pi), logging
# per-seq cosine divergence. Localizes the D=256 tq_decode_gqa6_hd256 kernel
# bug (broken in eager at ALL seqlens). FlyDSL output is still returned.
_TQ_SHADOW_CMP = os.environ.get("VLLM_TQ_SHADOW_CMP", "0") == "1"
_V4_SWA_OFF = os.environ.get("VLLM_TQ_V4_SWA_OFF", "0") == "1"
# Diagnostic: force the FlyDSL v4 decode to allocate its own scratch buffers
# (mid_o/output/lse) instead of the shared WorkspaceManager arena + the
# per-layer _SegmBufPool. If E2E accuracy recovers only with this set, the v4
# kernel is writing out of bounds into neighbouring shared state (the returned
# output_buf stays correct, so the shadow-compare — which uses fresh None
# buffers — cannot see the corruption).
_V4_FRESH_BUFS = os.environ.get("VLLM_TQ_V4_FRESH_BUFS", "0") == "1"
_TQ_SHADOW_LOG = os.environ.get(
    "VLLM_TQ_SHADOW_LOG", "/shareddata/adrana/workspace/reports/tq_shadow.log")
_TQ_SHADOW_THRESH = float(os.environ.get("VLLM_TQ_SHADOW_THRESH", "0.999"))
_TQ_SHADOW_STATE: dict = {"calls": 0, "diverged": 0, "fh": None}
_TQ_SHADOW_DUMP = os.environ.get("VLLM_TQ_SHADOW_DUMP", "")
_TQ_SHADOW_REINVOKE = os.environ.get("VLLM_TQ_SHADOW_REINVOKE", "0") == "1"


def _reinvoke_probe(st, impl, query, kv_cache, attn_metadata, Pi, PiT,
                    centroids, f_ref, t_ref, worst):
    """Re-run the identical FlyDSL v4 call in-process on a fresh output buffer.

    Distinguishes "the kernel computed nothing for this row" from "the reduce
    wrote somewhere other than the caller's output buffer": the offline replay
    of these very inputs is always correct, so the divergence has to come from
    live process state rather than the data.
    """
    import torch as _t
    try:
        from vllm.v1.attention.ops.flydsl_turboquant_decode_v4 import (
            flydsl_turboquant_decode_attention_v4 as _v4,
        )
        B = int(query.shape[0])
        again = _v4(
            query=query, kv_cache=kv_cache,
            block_table=attn_metadata.block_table,
            seq_lens=attn_metadata.seq_lens,
            Pi=Pi, centroids=centroids, scale=impl.scale,
            mse_bits=impl.tq_config.key_mse_bits,
            key_packed_size=impl.tq_config.key_packed_size,
            value_quant_bits=impl.tq_config.effective_value_quant_bits,
            value_packed_size=impl.tq_config.value_packed_size,
            max_seq_len=int(attn_metadata.max_seq_len),
            key_fp8=impl.tq_config.key_fp8,
            norm_correction=impl.tq_config.norm_correction,
            PiT=PiT, mid_o_buf=None, output_buf=None, lse_buf=None,
            buf_holder=None, max_num_kv_splits=impl.max_num_kv_splits,
            sinks=None,
        )
        a = again.reshape(B, -1).float()
        nz_first = [int((f_ref[b] != 0).sum()) for b in range(B)]
        nz_again = [int((a[b] != 0).sum()) for b in range(B)]
        cos_again = _t.nn.functional.cosine_similarity(a, t_ref, dim=1)
        logger.warning(
            "tq REINVOKE probe #%d (call=%d worst=%d): "
            "orig_nonzero=%s reinvoke_nonzero=%s reinvoke_cos=%s",
            st["reinvokes"], st["calls"], worst, nz_first, nz_again,
            [round(float(c), 5) for c in cos_again],
        )
    except Exception as e:  # diagnostics must never kill the server
        logger.warning("tq REINVOKE probe failed: %r", e)


def _dump_tq_divergence(st, impl, query, kv_cache, attn_metadata, Pi, PiT,
                        centroids, fly_out, tri, worst):
    """Snapshot the exact kernel inputs for the first diverging decode call.

    Only the KV blocks reachable from the worst sequence's block table are
    saved, with the block table remapped onto that dense subset, so the dump
    stays a few hundred MB instead of the full cache.
    """
    import torch as _t
    try:
        B = int(query.shape[0])
        blk_sz = _infer_tq_block_size(kv_cache)
        seq_lens = attn_metadata.seq_lens[:B]
        bt_full = attn_metadata.block_table[:B]
        seq_len = int(seq_lens[worst])

        # Union of every block reachable from the whole batch, so the batch
        # context (not just the failing row) can be replayed offline.
        per_row = [bt_full[b, :((int(seq_lens[b]) + blk_sz - 1) // blk_sz)]
                   for b in range(B)]
        blk_ids = _t.unique(_t.cat(per_row).to(_t.long))
        remap = _t.full((int(blk_ids.max()) + 1,), -1, dtype=_t.long,
                        device=blk_ids.device)
        remap[blk_ids] = _t.arange(blk_ids.numel(), device=blk_ids.device)
        bt_remap = _t.zeros_like(bt_full, dtype=_t.int32)
        for b in range(B):
            n = per_row[b].numel()
            bt_remap[b, :n] = remap[per_row[b].to(_t.long)].to(_t.int32)

        def _gather(c):
            if isinstance(c, (tuple, list)):
                return [_gather(x) for x in c]
            return c.index_select(0, blk_ids.to(c.device)).clone().cpu()

        payload = {
            "call": st["calls"], "seq_len": seq_len, "block_size": blk_sz,
            "B": B, "worst": worst,
            "seq_lens": seq_lens.clone().cpu(),
            "max_seq_len": int(attn_metadata.max_seq_len),
            "query": query[:B].clone().cpu(),
            "kv_blocks": _gather(kv_cache),
            "block_table": bt_remap.cpu(),
            "orig_block_ids": blk_ids.cpu(),
            "Pi": None if Pi is None else Pi.clone().cpu(),
            "PiT": None if PiT is None else PiT.clone().cpu(),
            "centroids": (None if centroids is None
                          else centroids.clone().cpu()),
            "fly_out": fly_out[:B].clone().cpu(),
            "tri_out": tri.reshape(query.shape[0], -1)[:B].clone().cpu(),
            "scale": impl.scale,
            "max_num_kv_splits": impl.max_num_kv_splits,
            "cfg": {
                "key_mse_bits": impl.tq_config.key_mse_bits,
                "key_packed_size": impl.tq_config.key_packed_size,
                "value_quant_bits":
                    impl.tq_config.effective_value_quant_bits,
                "value_packed_size": impl.tq_config.value_packed_size,
                "key_fp8": impl.tq_config.key_fp8,
                "norm_correction": impl.tq_config.norm_correction,
            },
        }
        _t.save(payload, _TQ_SHADOW_DUMP)
        logger.warning("tq SHADOW DUMP written to %s (call=%d seq_len=%d)",
                       _TQ_SHADOW_DUMP, st["calls"], seq_len)
    except Exception as e:  # diagnostics must never kill the server
        logger.warning("tq SHADOW DUMP failed: %r", e)


def _infer_tq_block_size(kv_cache):
    c = kv_cache
    while isinstance(c, (tuple, list)):
        c = c[0]
    return int(c.shape[1]) if c.dim() >= 2 else 1


def _shadow_compare_tq(impl, query, kv_cache, attn_metadata, Pi, PiT,
                       centroids, fly_out):
    """Run Triton v3 TQ decode on the SAME inputs and log divergence vs FlyDSL.

    Read-only w.r.t. the returned FlyDSL result. Golden = Triton v3
    (triton_turboquant_unified_attention), the bf16-parity reference.
    """
    import torch as _t
    st = _TQ_SHADOW_STATE
    st["calls"] += 1
    q = query
    B = q.shape[0]
    dev = q.device
    tri = impl._dispatch_decode_v3(
        query=q, kv_cache=kv_cache, block_table=attn_metadata.block_table,
        seq_lens=attn_metadata.seq_lens,
        Pi=Pi, centroids=centroids, scale=impl.scale,
        mse_bits=impl.tq_config.key_mse_bits,
        key_packed_size=impl.tq_config.key_packed_size,
        value_quant_bits=impl.tq_config.effective_value_quant_bits,
        value_packed_size=impl.tq_config.value_packed_size,
        max_seq_len=int(attn_metadata.max_seq_len),
        key_fp8=impl.tq_config.key_fp8,
        norm_correction=impl.tq_config.norm_correction,
        PiT=PiT, mid_o_buf=None, output_buf=None, lse_buf=None,
        buf_holder=None, max_num_kv_splits=impl.max_num_kv_splits,
        sinks=getattr(impl, "sinks", None),
        # CRITICAL: the golden v3 reference must window the decode exactly like
        # the production V3 server does (see the _USE_TQ_V3 path, which passes
        # sliding_window=self.sliding_window). Omitting it here made the oracle
        # do FULL attention, so a full-attention (SWA-off/broken) v4 matched the
        # reference at cos>=0.999 while both silently diverged from the correct
        # windowed decode — masking the real seq_len>window SWA bug. Pass the
        # layer's window so the shadow validates windowed-v4 vs windowed-v3.
        sliding_window=getattr(impl, "sliding_window", None),
    )
    f = fly_out.reshape(B, -1).float()
    t = tri.reshape(B, -1).float()
    c = _t.nn.functional.cosine_similarity(f, t, dim=1)  # [B]
    seq_lens = attn_metadata.seq_lens[:B].tolist()
    cmin = float(c.min()); cmean = float(c.mean())
    worst = int(c.argmin())
    if st["fh"] is None:
        st["fh"] = open(_TQ_SHADOW_LOG, "w")
        st["fh"].write("call\tB\tcos_min\tcos_mean\tworst_seq\tworst_seqlen\t"
                       "maxerr\tf_nan\tf_inf\tt_nan\tf_absmax\tall_seqlens\n")
    if cmin < _TQ_SHADOW_THRESH:
        st["diverged"] += 1
        finite = _t.isfinite(f)
        me = float((f - t)[finite & _t.isfinite(t)].abs().max()) if bool(
            finite.any()) else float("nan")
        f_nan = int(_t.isnan(f).sum())
        f_inf = int(_t.isinf(f).sum())
        t_nan = int(_t.isnan(t).sum())
        f_absmax = float(f[finite].abs().max()) if bool(finite.any()) else 0.0
        st["fh"].write(
            f"{st['calls']}\t{B}\t{cmin:.5f}\t{cmean:.5f}\t{worst}\t"
            f"{seq_lens[worst]}\t{me:.4f}\t{f_nan}\t{f_inf}\t{t_nan}\t"
            f"{f_absmax:.4f}\t{seq_lens}\n")
        st["fh"].flush()
        if _TQ_SHADOW_DUMP and not st.get("dumped"):
            st["dumped"] = True
            _dump_tq_divergence(st, impl, query, kv_cache, attn_metadata,
                                Pi, PiT, centroids, fly_out, tri, worst)
        if _TQ_SHADOW_REINVOKE and st.get("reinvokes", 0) < 5:
            st["reinvokes"] = st.get("reinvokes", 0) + 1
            _reinvoke_probe(st, impl, query, kv_cache, attn_metadata,
                            Pi, PiT, centroids, f, t, worst)
        logger.warning_once(
            "tq SHADOW DIVERGE: cos_min=%.5f at seqlen=%d (B=%d) — see %s",
            cmin, seq_lens[worst], B, _TQ_SHADOW_LOG)
if _USE_TQ_V4 and not _flydsl_v4_available():
    logger.warning(
        "VLLM_TQ_DECODE_V4 requested but FlyDSL is unavailable; "
        "falling back to v3."
    )
    _USE_TQ_V4 = False
if _USE_FP8_G32_V4 and not _flydsl_fp8_v4_available():
    logger.warning(
        "VLLM_FP8_G32_DECODE_V4 requested but FlyDSL is unavailable; "
        "falling back to the fp8_g32 Triton path."
    )
    _USE_FP8_G32_V4 = False

_HAS_FLASH_ATTN = is_flash_attn_varlen_func_available()
if _HAS_FLASH_ATTN:
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

# ROCm's CK flash-attention kernel rejects head dims above 256 with
# "CK only supports head dimension at most 256". Gemma 4's full-attention
# layers use global_head_dim=512, so prefill on those layers must route to
# the Triton unified kernel, which is generic over head size.
_CK_MAX_HEAD_DIM = 256

# Continuation-prefill scratch buffers, shared across all layers and grown on
# demand. Flat, so one allocation serves every (Hk, D) via a view.
#
# Continuation prefill needs two K/V-sized scratch pairs per layer: the dequant
# target for the cached prefix, and the "full" concatenation of that prefix with
# the current chunk. Both used to be cached per layer at max_model_len capacity.
# On Gemma 4 at 145K each is 16 heads x 145408 x 256 x 2 B = 1.19 GiB, so
# 50 sliding layers x 2 pairs x 2 tensors is ~230 GiB requested *outside* the
# gpu_memory_utilization budget: the engine died with HIP OOM while PyTorch
# held 283 of 288 GiB, and 60 layers x that footprint is what made 128K
# unreachable even though the KV cache itself fit comfortably.
#
# Sharing is safe for the same reason the WorkspaceManager shares its own
# buffers: layers run sequentially on a single stream and each buffer is fully
# consumed within the call that fills it. Allocation still happens once and is
# then reused, which preserves the stable-address property the per-layer cache
# was introduced for (variable-size torch.empty per call collided with the
# ROCm HIP graph pool). ``slot`` keeps the two purposes in separate buffers so
# the dequant source can never alias the concatenation target.
_shared_prefill_bufs: dict[Any, tuple[torch.Tensor, torch.Tensor]] = {}


def _get_shared_prefill_bufs(
    slot: str, numel: int, device: Any, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a (k, v) flat buffer pair of at least ``numel`` elements."""
    key = (slot, device, dtype)
    k, v = _shared_prefill_bufs.get(key, (None, None))
    if k is None or k.numel() < numel:
        # Grow-only: a request larger than any seen so far replaces the pair.
        k = torch.empty(numel, dtype=dtype, device=device)
        v = torch.empty(numel, dtype=dtype, device=device)
        _shared_prefill_bufs[key] = (k, v)
    return k, v

logger.info_once(
    "TurboQuant has flash attn: %s, decode kernel: %s, fp4_g32_v3: %s, "
    "fp8_g32_v3: %s, fp8_g32_v4(flydsl): %s",
    _HAS_FLASH_ATTN,
    "v4(flydsl)" if _USE_TQ_V4 else "v3" if _USE_TQ_V3 else "v2" if _USE_TQ_V2 else "v1",
    _USE_FP4_G32_V3,
    _USE_FP8_G32_V3,
    _USE_FP8_G32_V4,
)
# Continuation prefill: for small continuation chunks (q_len ≤ threshold),
# use the TQ decode kernel directly instead of full-dequant + flash_attn.
# do_kv_cache_update already stored all tokens to TQ cache, so the decode
# kernel can read them efficiently. This avoids O(cached_len) dequant work
# per continuation, eliminating the O(N²/chunk_size) collapse at long context.
# Set VLLM_TQ_CONTINUATION_DECODE_THRESHOLD=0 to force every continuation
# through full-dequant + flash_attn instead of the v3 reader (slower, but the
# v3 reader is layout-sensitive).
_CONTINUATION_DECODE_THRESHOLD = int(
    os.environ.get("VLLM_TQ_CONTINUATION_DECODE_THRESHOLD", "128")
)


def _build_hadamard(d: int, device_str: str) -> torch.Tensor:
    """Orthonormal Hadamard matrix (Sylvester construction), cached per (d, device).

    Precomputed D×D matrix enables matmul-based WHT — single cuBLAS GEMM
    instead of log2(D) butterfly kernel launches. 64KB for D=128.
    """
    # Normalize device string so "cuda" and "cuda:0" hit the same cache entry.
    return _build_hadamard_cached(d, str(torch.device(device_str)))


@functools.cache
def _build_hadamard_cached(d: int, device_str: str) -> torch.Tensor:
    H = torch.tensor([[1.0]])
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / math.sqrt(d)).to(torch.device(device_str))


class TurboQuantAttentionBackend(AttentionBackend):
    """Attention backend using TurboQuant KV-cache compression."""

    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = False

    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "turboquant_k8v4",
        "turboquant_4bit_nc",
        "turboquant_k3v4_nc",
        "turboquant_3bit_nc",
        "fp4_kv_g32",
        "fp8_kv_g32",
    ]

    @staticmethod
    def get_name() -> str:
        return "TURBOQUANT"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [16, 32, 64, 128]

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_per_head_quant_scales(cls) -> bool:
        return False

    @staticmethod
    def get_impl_cls() -> type["TurboQuantAttentionImpl"]:
        if _USE_TQ_SOA_FUSION:
            from vllm.v1.attention.ops.turboquant_soa_fusion import (
                FusionTurboQuantAttentionImpl,
            )
            return FusionTurboQuantAttentionImpl
        return TurboQuantAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["TurboQuantMetadataBuilder"]:
        return TurboQuantMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "turboquant_4bit_nc",
    ) -> tuple[int, ...]:
        """Combined K+V cache shape — no leading 2 dimension.

        Standard attention backends use (2, num_blocks, block_size, num_kv_heads,
        head_dim) with a leading 2 to separate K and V. TurboQuant packs K+V
        into a single interleaved slot per head per position, so the cache is:

            (num_blocks, block_size, num_kv_heads, slot_size_aligned)

        Each slot = [key_packed | value_packed | padding].
        This is safe because TQ has its own get_kv_cache_shape override and
        never shares cache tensors with other backends. Layers that fall back
        to native dtype via kv_cache_dtype_skip_layers get their own
        standard-shaped cache allocation.

        head_size is the model's real head_dim. slot_size_aligned is computed
        from the TQ config to ensure correct cache allocation for all head dims.
        """
        from vllm.model_executor.layers.quantization.turboquant.config import (
            TurboQuantConfig,
        )

        if cache_dtype_str == "fp4_kv_g32":
            from vllm.v1.attention.ops.fp4_g32.fp4_levels import (
                get_group_size,
                get_token_norm,
                slot_size as _fp4_slot_size,
            )
            return (num_blocks, block_size, num_kv_heads,
                    _fp4_slot_size(head_size, get_group_size(), get_token_norm()))
        if cache_dtype_str == "fp8_kv_g32":
            from vllm.v1.attention.ops.fp8_g32.fp8_levels import (
                get_group_size as _fp8_g,
                get_token_norm as _fp8_tn,
                slot_size as _fp8_slot_size,
            )
            return (num_blocks, block_size, num_kv_heads,
                    _fp8_slot_size(head_size, _fp8_g(), _fp8_tn()))
        tq_config = TurboQuantConfig.from_cache_dtype(cache_dtype_str, head_size)
        return (num_blocks, block_size, num_kv_heads, tq_config.slot_size_aligned)

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype is None:
            return False
        return (
            kv_cache_dtype.startswith("turboquant_")
            or kv_cache_dtype == "fp4_kv_g32"
            or kv_cache_dtype == "fp8_kv_g32"
        )

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        # head_size from spec is effective_head_size (padded_slot//2),
        # not the model's actual head_dim. Accept any positive value.
        return head_size > 0

    @classmethod
    def supports_sink(cls) -> bool:
        """Return True to indicate TurboQuant supports sink tokens.

        Sink tokens provide stable attention anchors at the start of context.
        The TQ decode kernel initializes online softmax with pre-computed
        sink attention logits for proper attention distribution.
        """
        return True


@dataclass
class TurboQuantMetadata(AttentionMetadata):
    """Metadata for TurboQuant attention."""

    seq_lens: torch.Tensor  # (num_reqs,) — total context length per request
    slot_mapping: torch.Tensor  # (num_tokens,) — cache slot for each token
    block_table: torch.Tensor  # (num_reqs, max_num_blocks)
    query_start_loc: torch.Tensor  # (num_reqs + 1,) — cu_seqlens for queries
    num_actual_tokens: int = 0  # actual tokens (excluding padding)
    max_query_len: int = 0  # longest query in batch
    max_seq_len: int = 0  # longest context in batch
    is_prefill: bool = False
    num_decodes: int = 0  # number of decode requests (first in batch)
    num_decode_tokens: int = 0  # tokens from decode requests


class TurboQuantMetadataBuilder(AttentionMetadataBuilder[TurboQuantMetadata]):
    """Builds TurboQuantMetadata from scheduler output."""

    # v1/v2/v3, HIP SoA fusion, and v4 all bake their gridDim from a stable source
    # (compile-time MAX_NUM_KV_SPLITS, pre-computed metadata, or worst-case derivation
    # from block_table.shape * block_size). The v4 launcher uses a deterministic
    # worst-case sizing (mirrored on capture and runtime) so the captured gridDim is
    # always valid at replay. Continuation prefill needs workspace warmup that only
    # happens during cudagraph capture, so v4 must stay IN UNIFORM_BATCH cudagraph.
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> TurboQuantMetadata:
        attn_metadata = self.build(0, common_attn_metadata)
        # Set seq_lens to 1 so CUDA graph capture is fast
        # (real seq_lens are filled at replay time).
        attn_metadata.seq_lens.fill_(1)
        return attn_metadata

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        """Build TurboQuantMetadata from common attention metadata."""
        cam = common_attn_metadata

        # With reorder_batch_threshold=1, the model runner guarantees
        # decodes come first in the batch. split_decodes_and_prefills
        # finds the boundary (operates on CPU tensors — no GPU sync).
        assert self.reorder_batch_threshold is not None
        num_decodes, num_prefills, num_decode_tokens, _ = split_decodes_and_prefills(
            cam, decode_threshold=self.reorder_batch_threshold
        )

        return TurboQuantMetadata(
            seq_lens=cam.seq_lens,
            slot_mapping=cam.slot_mapping,
            block_table=cam.block_table_tensor,
            query_start_loc=cam.query_start_loc,
            num_actual_tokens=cam.num_actual_tokens,
            max_query_len=cam.max_query_len,
            max_seq_len=cam.max_seq_len,
            is_prefill=(cam.max_query_len > 1),
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
        )


class TurboQuantAttentionImpl(AttentionImpl["TurboQuantMetadata"]):
    """TurboQuant attention implementation.

    Vectorized PyTorch: batch quantize/store, vectorized bit-unpack
    decode with einsum scores and value gather.
    """

    supports_quant_query_input: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        **kwargs,
    ):
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self._is_fp4_g32 = (kv_cache_dtype == "fp4_kv_g32")
        self._is_fp8_g32 = (kv_cache_dtype == "fp8_kv_g32")
        # fp8_g32 reuses the fp4_g32 control flow (no Lloyd-Max centroids,
        # Hadamard-rotated K, decode kernel + dequant-for-large-continuation).
        # Decode/continuation routing branches on _is_fp4_g32 OR _is_fp8_g32.
        self._is_g32 = self._is_fp4_g32 or self._is_fp8_g32
        # Sliding window (per-layer): wired to the V3 unified kernels for
        # SWA tile-pruning + per-tile masking. None or 0 disables SWA.
        #
        # ``VLLM_TQ_DISABLE_SWA=1`` is a debug knob for A/B testing the V3
        # kernel against full-attention. On SWA-trained models (e.g.
        # gpt-oss with sliding_window=128 every-other layer), forcing the
        # attention math to drop the SWA mask makes the SWA layers
        # extrapolate beyond their training window and produces incoherent
        # output as soon as context exceeds sliding_window tokens — this is
        # an architectural limitation, not a kernel bug.
        #
        # Therefore ``VLLM_TQ_DISABLE_SWA=1`` only flips the SWA *cache*
        # spec to full-attention (cache holds all positions, no rotation)
        # while keeping ``self.sliding_window`` plumbed through to the V3
        # kernel mask. Net effect:
        #   * Math identical to natural mode -> coherent output at any ctx.
        #   * Cache layout differs (full vs rotated) — useful for memory-
        #     utilization A/B against the SWA-rotated TQSlidingWindowSpec.
        #
        # ``VLLM_TQ_FORCE_FULL_ATTN=1`` is an opt-in escape hatch that
        # additionally zeroes the kernel mask — intended for non-SWA
        # sinks-only models. On SWA-trained models it produces incoherent
        # output past sliding_window tokens (documented).
        self.sliding_window = sliding_window
        if os.environ.get("VLLM_TQ_FORCE_FULL_ATTN", "0") == "1":
            if sliding_window is not None and sliding_window > 0:
                logger.warning_once(
                    "VLLM_TQ_FORCE_FULL_ATTN=1: zeroing sliding_window=%s "
                    "for V3 kernel mask (SWA tile pruning + per-tile mask "
                    "off). On SWA-trained models (gpt-oss) this produces "
                    "incoherent outputs beyond sliding_window tokens; this "
                    "is math-correct but architecturally out-of-"
                    "distribution.",
                    sliding_window,
                )
            self.sliding_window = None

        from vllm.model_executor.layers.quantization.turboquant.config import (
            TurboQuantConfig,
        )

        if not self._is_g32:
            self.tq_config = TurboQuantConfig.from_cache_dtype(kv_cache_dtype, head_size)
            # Pre-compute kernel constants from config (avoid repeated arithmetic)
            cfg = self.tq_config
            self._mse_bytes = (
                math.ceil(head_size * cfg.key_mse_bits / 8)
                if not cfg.key_fp8
                else head_size
            )
            self._val_data_bytes = math.ceil(head_size * cfg.effective_value_quant_bits / 8)
            self._n_centroids = cfg.n_centroids if not cfg.key_fp8 else 1
        else:
            self.tq_config = None
            self._mse_bytes = 0
            self._val_data_bytes = 0
            self._n_centroids = 0

        # Fixed NUM_KV_SPLITS (grid dims must be constant for cudagraph,
        # and benchmarks show no regression vs dynamic in eager mode).
        vllm_config = get_current_vllm_config()
        self.max_num_kv_splits = (
            vllm_config.attention_config.tq_max_kv_splits_for_cuda_graph
        )
        # Cache max_model_len now (config is available at __init__ time but
        # NOT during CUDA graph capture when _ensure_on_device is re-entered).
        self._max_model_len = vllm_config.model_config.max_model_len

        # Sink tokens support: store reference from kwargs if provided.
        # Sinks are pre-computed attention logits [Hq] that anchor attention
        # to sink tokens at the start of context (used by GPT-OSS and similar).
        # Note: sinks is passed directly via **extra_impl_args spread, not nested.
        self.sinks = kwargs.get("sinks")

        # SOA store flag: when VLLM_TQ_SOA_FUSION_STORE=1, the store writes SoA
        # layout (data region + metadata region separated per block). The FlyDSL v4
        # decode kernel reads exclusively from SoA layout, so this flag must be True
        # when VLLM_TQ_DECODE_V4=1.
        import os as _os_init
        self._soa_store = _os_init.environ.get("VLLM_TQ_SOA_FUSION_STORE", "0") == "1"

    def _ensure_on_device(self, layer, device):
        """One-time derivation of TQ buffers (rotation matrix, midpoints).

        The Hadamard rotation is shared across all layers: random sign
        flips do not improve Lloyd-Max quantization quality because the
        quantizer is symmetric around zero (sign-flipping a coordinate
        maps it to the mirror centroid with identical distortion).
        """
        # Pre-allocate _arange_cache and _cu_2 on the correct device before
        # any CUDA graph capture.  These are lazily initialised in forward();
        # if the lazy path fires during piecewise graph replay, the new
        # torch.arange / torch.zeros allocation receives an address already
        # owned by the captured HIP graph pool on ROCm → GPU memory fault or
        # garbage outputs.  Sizing to max_model_len+2 ensures the cache is
        # never re-grown during replay (_ac.shape[0] > any valid max_seq_len).
        # Use self._max_model_len (set in __init__) — get_current_vllm_config()
        # cannot be called here because _ensure_on_device is re-entered during
        # CUDA graph capture when the config context is not set.
        _max_len = self._max_model_len
        _already_ok = (
            hasattr(self, "_arange_cache")
            and self._arange_cache.device.type == str(device).split(":")[0]
            and self._arange_cache.shape[0] >= _max_len + 2
        )
        if not _already_ok:
            self._arange_cache = torch.arange(
                0, _max_len + 2, device=device, dtype=torch.int32
            )
        if not hasattr(self, "_cu_2") or self._cu_2.device != torch.device(device):
            self._cu_2 = torch.zeros(2, device=device, dtype=torch.int32)

        # Pre-warm the WorkspaceManager to the maximum size needed by
        # _decode_attention BEFORE any CUDA graph capture begins.
        #
        # _decode_attention calls WorkspaceManager.get_simultaneous() with
        # shape (B, Hq, S, D+1) × fp32 + (B, Hq, D) × query_dtype + (B, Hq) × fp32.
        # If the workspace grows DURING the piecewise capture loop (when a larger
        # batch size is warmed up mid-capture), some already-captured batch sizes
        # have the OLD (now freed) workspace address baked in → GPU fault on replay.
        # Pre-warming at the maximum captured batch size + maximum kv splits here
        # ensures the workspace reaches its final size before any capture starts.
        if is_workspace_manager_initialized() and not current_workspace_manager().is_locked():
            B_max = self._max_capture_batch_size()
            D = self.head_size
            Hq = self.num_heads
            S = self.max_num_kv_splits
            _pre_warm_bytes = (
                B_max * Hq * (S * (D + 1) + D) * 4  # fp32 mid_o + fp32 lse
                + B_max * Hq * D * 2                  # query_dtype output (bf16=2B)
                + 512                                  # alignment padding
            )
            _ws = current_workspace_manager()
            # Touch the workspace to trigger growth to this size before capture.
            # _ensure_workspace_size is not exposed, so use get_simultaneous with
            # a single flat allocation of the required size.
            try:
                _ws.get_simultaneous(((_pre_warm_bytes,), torch.uint8))
            except AssertionError:
                pass  # Already locked (shouldn't happen here, but be safe)

        if not hasattr(layer, "_tq_cached"):
            D = self.head_size

            # Pure Hadamard: orthonormal + symmetric (H = H^T), enabling
            # in-kernel butterfly fusion and trivial inverse for continuation.
            H = _build_hadamard(D, str(device))
            layer._tq_PiT = H
            layer._tq_Pi = H
            # fp16 copy for rotation in continuation prefill path
            layer._tq_Pi_half = H.to(torch.float16)

            if not self._is_g32:
                # Centroids for Lloyd-Max quantization (TQ formats only).
                layer._tq_centroids = get_centroids(D, self.tq_config.centroid_bits).to(
                    device=device, dtype=torch.float32
                )
                c_sorted, _ = layer._tq_centroids.sort()
                layer._tq_midpoints = (c_sorted[:-1] + c_sorted[1:]) / 2
            layer._tq_cached = True

    def _max_capture_batch_size(self) -> int:
        """Return the largest batch size we might see at runtime decode.

        Used to pre-warm the WorkspaceManager before capture begins so that
        workspace growth cannot happen mid-capture (invalidating baked-in
        pointers) NOR after capture lock (assertion in
        _ensure_workspace_size).

        We must take max(cudagraph_capture_sizes, scheduler.max_num_seqs):
        when a forward pass exceeds the largest captured graph size, vllm
        falls back to the eager path — but the workspace is still locked
        and that eager path can hit batch sizes up to max_num_seqs. Without
        this max, lm-eval-style continuous batching that pushes B past 512
        triggers the post-capture-lock assertion.

        Falls back to a generous heuristic (1024) if config is unavailable.
        """
        try:
            from vllm.v1.utils import get_current_vllm_config as _gcvc
            cfg = _gcvc()
            candidates: list[int] = []
            sizes = cfg.compilation_config.cudagraph_capture_sizes
            if sizes:
                candidates.append(int(max(sizes)))
            sched = getattr(cfg, "scheduler_config", None)
            if sched is not None and getattr(sched, "max_num_seqs", None):
                candidates.append(int(sched.max_num_seqs))
            if candidates:
                return max(candidates)
        except Exception:
            pass
        return 1024

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Store compressed K/V into the combined TQ cache.

        Called as a separate custom op (unified_kv_cache_update) BEFORE
        the attention forward, matching FlashAttention's split pattern.
        slot_mapping is already sliced to num_actual_tokens by the caller.
        """
        N = slot_mapping.shape[0]
        if N <= 0:
            return

        device = key.device
        self._ensure_on_device(layer, device)

        k = key[:N].view(N, self.num_kv_heads, self.head_size)
        v = value[:N].view(N, self.num_kv_heads, self.head_size)
        self._store_kv(k, v, kv_cache, slot_mapping, layer)

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: "TurboQuantMetadata",
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens = query.shape[0]

        if output is None:
            output = torch.zeros(
                num_tokens,
                self.num_heads * self.head_size,
                dtype=query.dtype,
                device=query.device,
            )

        if attn_metadata is None:
            return output.fill_(0)

        # Slice to actual tokens
        N = attn_metadata.num_actual_tokens
        if N <= 0:
            return output.fill_(0)


        q = query[:N].view(N, self.num_heads, self.head_size)

        # ── DEBUG: env-gated raw q/k/v capture for D=256 recipe analysis ──
        import os as _os_dbg
        _dump_dir = _os_dbg.environ.get("VLLM_FP8_DUMP_QKV")
        if _dump_dir and self.head_size == 256:
            _cnt = getattr(TurboQuantAttentionImpl, "_dbg_dump_cnt", 0)
            if _cnt < 8:
                try:
                    _k = key[:N].view(N, self.num_kv_heads, self.head_size)
                    _v = value[:N].view(N, self.num_kv_heads, self.head_size)
                    torch.save(
                        {"q": q.detach().float().cpu(),
                         "k": _k.detach().float().cpu(),
                         "v": _v.detach().float().cpu(),
                         "is_prefill": bool(attn_metadata.is_prefill),
                         "N": int(N)},
                        f"{_dump_dir}/qkv_{_cnt:02d}.pt",
                    )
                    TurboQuantAttentionImpl._dbg_dump_cnt = _cnt + 1
                except Exception:
                    pass

        # Get TQ buffers, ensure on device (one-time migration).
        # Use Any-typed alias for dynamic _tq_* attrs set by _ensure_on_device.
        tq_layer: Any = layer
        device = q.device
        self._ensure_on_device(tq_layer, device)
        Pi = tq_layer._tq_Pi
        PiT = tq_layer._tq_PiT
        centroids = getattr(tq_layer, "_tq_centroids", None)

        # Compute attention (KV cache was already updated by do_kv_cache_update)
        # With reorder_batch_threshold=1, decodes come first in the batch.
        # num_decodes/num_decode_tokens from metadata give the split point.
        num_decodes = attn_metadata.num_decodes
        num_decode_tokens = attn_metadata.num_decode_tokens

        if not attn_metadata.is_prefill:
            # Pure decode batch — fast path
            attn_out = self._decode_attention(
                q, kv_cache, attn_metadata, Pi, centroids, PiT, layer
            )
        elif num_decodes == 0:
            # Pure prefill batch
            k = key[:N].view(N, self.num_kv_heads, self.head_size)
            v = value[:N].view(N, self.num_kv_heads, self.head_size)
            attn_out = self._prefill_attention(
                q,
                k,
                v,
                kv_cache,
                attn_metadata,
                Pi,
                centroids,
                PiT,
                layer=layer,
            )
        else:
            # Mixed batch: decodes first (guaranteed by reorder_batch).
            # Bug-2 fix: avoid fresh torch.zeros allocation for attn_out —
            # on ROCm, post-capture allocations can land in the HIP graph
            # memory pool and cause GPU faults in the eager mixed-batch path.
            # Write decode and prefill results directly into the pre-allocated
            # output buffer instead.

            # --- Decode portion (first num_decodes requests) ---
            # Use full-batch max_seq_len as safe upper bound (no GPU sync).
            decode_meta = TurboQuantMetadata(
                seq_lens=attn_metadata.seq_lens[:num_decodes],
                slot_mapping=attn_metadata.slot_mapping[:num_decode_tokens],
                block_table=attn_metadata.block_table[:num_decodes],
                query_start_loc=attn_metadata.query_start_loc[: num_decodes + 1],
                num_actual_tokens=num_decode_tokens,
                max_query_len=1,
                max_seq_len=attn_metadata.max_seq_len,
                is_prefill=False,
            )
            _decode_out = self._decode_attention(
                q[:num_decode_tokens], kv_cache, decode_meta, Pi, centroids, PiT, layer
            )
            # Write decode result directly into output (no intermediate attn_out)
            if output.ndim == 3:
                output[:num_decode_tokens] = _decode_out.to(output.dtype)
            else:
                output[:num_decode_tokens] = _decode_out.reshape(
                    num_decode_tokens, -1
                ).to(output.dtype)

            # --- Prefill portion (remaining requests) ---
            # CRITICAL: use prefill-specific max_seq_len so flash_attn's
            # fast path (max_query_len == max_seq_len) triggers for
            # first-chunk prefills. Using full-batch max_seq_len breaks
            # this because decode requests inflate max_seq_len.
            prefill_seq_lens = attn_metadata.seq_lens[num_decodes:]
            # Use CPU-side max to avoid GPU→CPU sync from .item()
            prefill_max_seq = max(attn_metadata.seq_lens[num_decodes:].tolist())
            prefill_qsl = (
                attn_metadata.query_start_loc[num_decodes:] - num_decode_tokens
            )
            prefill_meta = TurboQuantMetadata(
                seq_lens=prefill_seq_lens,
                slot_mapping=attn_metadata.slot_mapping[num_decode_tokens:N],
                block_table=attn_metadata.block_table[num_decodes:],
                query_start_loc=prefill_qsl,
                num_actual_tokens=N - num_decode_tokens,
                max_query_len=attn_metadata.max_query_len,
                max_seq_len=prefill_max_seq,
                is_prefill=True,
            )
            k = key[:N].view(N, self.num_kv_heads, self.head_size)
            v = value[:N].view(N, self.num_kv_heads, self.head_size)
            _prefill_out = self._prefill_attention(
                q[num_decode_tokens:],
                k[num_decode_tokens:],
                v[num_decode_tokens:],
                kv_cache,
                prefill_meta,
                Pi,
                centroids,
                PiT,
                layer=layer,
            )
            # Write prefill result directly into output
            n_pref = N - num_decode_tokens
            if output.ndim == 3:
                output[num_decode_tokens:N] = _prefill_out.to(output.dtype)
            else:
                output[num_decode_tokens:N] = _prefill_out.reshape(
                    n_pref, -1
                ).to(output.dtype)
            # Return early — output already filled, skip the common copy below
            return output

        # Write into output buffer: attn_out is (N, Hq, D)
        # output may be 2D (N, Hq*D) or 3D (N, Hq, D)
        if output.ndim == 3:
            output[:N] = attn_out.to(output.dtype)
        else:
            output[:N] = attn_out.reshape(N, -1).to(output.dtype)
        return output

    # ------------------------------------------------------------------ #
    #  Layout-aware v3 decode dispatcher                                  #
    # ------------------------------------------------------------------ #
    def _dispatch_decode_v3(self, **kwargs):
        """Call the v3 decode kernel that matches the current cache layout.

        When the cache was written in SoA layout (VLLM_TQ_SOA_FUSION_STORE=1,
        required for the v4 FlyDSL decode kernel), the continuation-prefill
        path must also use a SoA-aware v3 reader.  Using the default AoS v3
        on a SoA-written cache mis-addresses k_norm / v_scale / v_zero and
        produces garbage attention output for every cached-prefix turn.

        AoS legs (pure v3 / default store) are unchanged: self._soa_store is
        False, so they keep the same AoS v3 they always used.
        """
        if not self._soa_store:
            return triton_turboquant_decode_attention_v3(**kwargs)

        # The SoA v3 wrapper forwards to the SoA unified launcher, whose
        # signature differs slightly from the AoS v3 (e.g. it has no
        # `sliding_window` parameter). Filter kwargs to the params the SoA
        # launcher accepts so a benign extra (e.g. sliding_window=None) cannot
        # crash the engine; raise loudly if a *meaningful* (non-None) kwarg
        # would be silently dropped.
        import inspect
        from vllm.v1.attention.ops.turboquant_soa_fusion.triton_turboquant_unified_attention import (  # noqa: E501
            triton_turboquant_decode_attention_v3 as _soa_unified_v3,
        )
        accepted = set(inspect.signature(_soa_unified_v3).parameters)
        dropped_meaningful = [
            k for k, v in kwargs.items() if k not in accepted and v is not None
        ]
        if dropped_meaningful:
            raise NotImplementedError(
                "SoA v3 decode does not support kwargs "
                f"{sorted(dropped_meaningful)} (set on a SoA-store TurboQuant "
                "config). Extend the SoA launcher or disable the unsupported "
                "feature."
            )
        filtered = {k: v for k, v in kwargs.items() if k in accepted}
        return _lazy_soa_decode_v3()(**filtered)

    # ------------------------------------------------------------------ #
    #  Store K/V into combined cache (vectorized)                         #
    # ------------------------------------------------------------------ #
    def _store_kv(
        self,
        key: torch.Tensor,  # (N, Hk, D)
        value: torch.Tensor,  # (N, Hk, D)
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot_size)
        slot_mapping: torch.Tensor,
        layer: Any,
    ):
        """Quantize + store via fused Triton kernel."""
        if self._is_fp4_g32:
            from vllm.v1.attention.ops.fp4_g32.triton_store import fp4_g32_store
            fp4_g32_store(
                key,
                value,
                kv_cache,
                slot_mapping,
                PiT=getattr(layer, "_tq_PiT", None),
            )
            return
        if self._is_fp8_g32:
            fp8_store, _, _ = _lazy_fp8_g32_imports()
            fp8_store(
                key,
                value,
                kv_cache,
                slot_mapping,
                PiT=getattr(layer, "_tq_PiT", None),
            )
            return
        if self._soa_store:
            # SOA layout: data region + metadata region separated per block.
            # Required by FlyDSL v4 decode kernel (DATA_BYTES_PER_SLOT=128).
            #
            # NOTE: tq_store_fused_soa.hip (the "HIP SOA store") is intentionally
            # bypassed here — despite its name it writes AoS layout (confirmed by
            # its own source comments: "Fused TQ Store HIP Kernel — AoS Layout").
            # Only the Triton SOA store correctly writes SoA (data region +
            # metadata region, DATA_BYTES_PER_SLOT=128) compatible with v4.
            _soa_triton_turboquant_store, _, _, _ = _lazy_soa_store_imports()
            _soa_triton_turboquant_store(
                key=key,
                value=value,
                kv_cache=kv_cache,
                slot_mapping=slot_mapping,
                PiT=layer._tq_PiT,
                midpoints=layer._tq_midpoints,
                mse_bits=self.tq_config.key_mse_bits,
                key_packed_size=self.tq_config.key_packed_size,
                value_quant_bits=self.tq_config.effective_value_quant_bits,
                key_fp8=self.tq_config.key_fp8,
                centroids=layer._tq_centroids,
                norm_correction=self.tq_config.norm_correction,
            )
        else:
            triton_turboquant_store(
                key,
                value,
                kv_cache,
                slot_mapping,
                layer._tq_PiT,
                layer._tq_centroids,
                layer._tq_midpoints,
                mse_bits=self.tq_config.key_mse_bits,
                key_packed_size=self.tq_config.key_packed_size,
                value_quant_bits=self.tq_config.effective_value_quant_bits,
                key_fp8=self.tq_config.key_fp8,
            )

    # ------------------------------------------------------------------ #
    #  Prefill: SDPA on raw Q/K/V with causal mask                        #
    # ------------------------------------------------------------------ #
    def _prefill_attention(
        self,
        query: torch.Tensor,  # (N, Hq, D)
        key: torch.Tensor,  # (N, Hk, D)
        value: torch.Tensor,  # (N, Hk, D)
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot_size)
        attn_metadata: TurboQuantMetadata,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
        PiT: torch.Tensor | None = None,
        layer: Any = None,
    ) -> torch.Tensor:
        N, Hq, D = query.shape
        Hk = key.shape[1]

        # Fast path: first-chunk prefills (all K/V in batch).
        # max_query_len == max_seq_len means no request has prior cached KV.
        # When sinks are present, skip this fast path because the paged attention
        # block table setup is incorrect for batched sequences (all sequences
        # would incorrectly attend to concatenated K/V from all requests).
        # This batched call passes causal=True with no window_size, so it is
        # only valid for full-attention layers: a sliding-window layer would
        # silently attend outside its window. That was unreachable while the
        # only SWA model (gpt-oss) had sinks, which already skip this path;
        # Gemma 4 is SWA *without* sinks, so gate on the window explicitly.
        # D is also capped by CK (see _CK_MAX_HEAD_DIM).
        if (
            self.sinks is None
            and not (self.sliding_window and self.sliding_window > 0)
            and D <= _CK_MAX_HEAD_DIM
            and attn_metadata.max_query_len == attn_metadata.max_seq_len
            and _HAS_FLASH_ATTN
        ):
            return flash_attn_varlen_func(
                q=query,
                k=key,
                v=value,
                cu_seqlens_q=attn_metadata.query_start_loc,
                cu_seqlens_k=attn_metadata.query_start_loc,
                max_seqlen_q=attn_metadata.max_query_len,
                max_seqlen_k=attn_metadata.max_query_len,
                softmax_scale=self.scale,
                causal=True,
            )

        # Continuation or no flash_attn: per-request attention.
        # For continuation chunks (seq_len > q_len), we must attend to
        # previously cached K/V from the TQ cache, not just the current
        # chunk's raw K/V.
        Hk = key.shape[1]
        use_gqa = Hk < Hq
        query_start_loc = attn_metadata.query_start_loc
        num_reqs = query_start_loc.shape[0] - 1

        output = torch.zeros(N, Hq, D, device=query.device, dtype=query.dtype)

        # Convert to Python lists once (single CPU-GPU sync) instead of
        # per-request .item() calls that each force a sync.
        qsl = query_start_loc.tolist()
        seq_lens_list = attn_metadata.seq_lens.tolist()

        # Pre-allocate cu_seqlens for single-request flash_attn calls
        # to avoid per-request host→device tensor creation.
        if not hasattr(self, "_cu_2"):
            self._cu_2 = torch.zeros(2, device=query.device, dtype=torch.int32)
        # Cache arange on self (avoid per-call kernel launch).
        _max_seq = attn_metadata.max_seq_len
        _ac: torch.Tensor | None = getattr(self, "_arange_cache", None)
        if _ac is None or _ac.shape[0] <= _max_seq:
            _ac = torch.arange(
                0, _max_seq + 1, device=query.device, dtype=attn_metadata.seq_lens.dtype
            )
            self._arange_cache = _ac
        _arange_cache: torch.Tensor = _ac

        for i in range(num_reqs):
            q_start = qsl[i]
            q_end = qsl[i + 1]
            q_len = q_end - q_start
            if q_len <= 0:
                continue

            seq_len = seq_lens_list[i]
            q_seq = query[q_start:q_end]  # (q_len, Hq, D)
            k_seq = key[q_start:q_end]  # (q_len, Hk, D)
            v_seq = value[q_start:q_end]  # (q_len, Hk, D)


            if q_len == seq_len:
                # First-chunk prefill: all K/V are in the current batch.
                if self.sinks is not None or D > _CK_MAX_HEAD_DIM:
                    # Use unified attention for sink support, and for head
                    # dims CK cannot handle (Gemma 4 global layers, D=512).
                    # This kernel is generic over head size and honors
                    # window_size, so it is correct for both cases.
                    out = torch.empty_like(q_seq)
                    k_cache = k_seq.unsqueeze(0)  # [1, q_len, Hk, D]
                    v_cache = v_seq.unsqueeze(0)  # [1, q_len, Hk, D]
                    cu_single = torch.tensor(
                        [0, q_len], dtype=torch.int32, device=query.device
                    )
                    seq_lens_single = torch.tensor(
                        [q_len], dtype=torch.int32, device=query.device
                    )
                    block_table_single = torch.zeros(
                        (1, 1), dtype=torch.int32, device=query.device
                    )
                    # SWA: pass (window-1, 0) for left-only causal sliding
                    # window when set (gpt-oss every-other layer); else
                    # (-1, -1) for full causal attention.
                    _win = (
                        (self.sliding_window - 1, 0)
                        if self.sliding_window and self.sliding_window > 0
                        else (-1, -1)
                    )
                    unified_attention(
                        q=q_seq,
                        k=k_cache,
                        v=v_cache,
                        out=out,
                        cu_seqlens_q=cu_single,
                        max_seqlen_q=q_len,
                        seqused_k=seq_lens_single,
                        max_seqlen_k=q_len,
                        softmax_scale=self.scale,
                        causal=True,
                        window_size=_win,
                        block_table=block_table_single,
                        softcap=0.0,
                        q_descale=None,
                        k_descale=None,
                        v_descale=None,
                        sinks=self.sinks,
                    )
                elif _HAS_FLASH_ATTN:
                    self._cu_2[1] = q_len
                    cu = self._cu_2
                    _fa_window = (
                        (self.sliding_window - 1, 0)
                        if self.sliding_window and self.sliding_window > 0
                        else None
                    )
                    out = flash_attn_varlen_func(
                        q=q_seq,
                        k=k_seq,
                        v=v_seq,
                        cu_seqlens_q=cu,
                        cu_seqlens_k=cu,
                        max_seqlen_q=q_len,
                        max_seqlen_k=q_len,
                        softmax_scale=self.scale,
                        causal=True,
                        **({"window_size": _fa_window} if _fa_window is not None else {}),
                    )
                else:
                    q_t = q_seq.transpose(0, 1).contiguous()
                    k_t = k_seq.transpose(0, 1).contiguous()
                    v_t = v_seq.transpose(0, 1).contiguous()
                    out = F.scaled_dot_product_attention(
                        q_t,
                        k_t,
                        v_t,
                        is_causal=True,
                        scale=self.scale,
                        enable_gqa=use_gqa,
                    ).transpose(0, 1)
                output[q_start:q_end] = out.to(query.dtype)
            else:
                # Continuation chunk: tokens already stored to cache
                # by do_kv_cache_update. Use decode kernel directly.
                cached_len = seq_len - q_len
                synth_seq_lens = _arange_cache[cached_len + 1 : seq_len + 1]
                synth_bt = attn_metadata.block_table[i : i + 1].expand(q_len, -1)
                if self._is_fp8_g32:
                    # fp8_g32 continuation prefill:
                    # - V3 + small chunk (≤128): route to the new V3 unified
                    #   attention kernel (scaled F8F6F4 MFMA QK + bf16 PV),
                    #   mirroring fp4_g32 V3.
                    # - V1 + small chunk (≤128): route to the legacy v1-style
                    #   stage1 decode kernel.
                    # - Any large chunk (>128): route to the shared dequant +
                    #   flash_attn path (`_fp4_g32_continuation_prefill`),
                    #   which auto-dispatches the fp8 dequant kernel via
                    #   `_is_fp8_g32`.
                    if (
                        _USE_FP8_G32_V3
                        and q_len <= _CONTINUATION_DECODE_THRESHOLD
                    ):
                        _fp8_g32_unified = _lazy_fp8_g32_v3_import()
                        cu_q = _arange_cache[: q_len + 1].to(torch.int32)
                        out = _fp8_g32_unified(
                            query=q_seq,
                            kv_cache=kv_cache,
                            block_table=synth_bt,
                            seq_lens=synth_seq_lens,
                            query_start_loc=cu_q,
                            scale=self.scale,
                            PiT=PiT,
                            max_query_len=1,
                            max_seq_len=int(seq_len),
                            sinks=self.sinks,
                            sliding_window=self.sliding_window,
                        )
                    elif q_len <= _CONTINUATION_DECODE_THRESHOLD:
                        _, _fp8_decode, _ = _lazy_fp8_g32_imports()
                        out = _fp8_decode(
                            query=q_seq,
                            kv_cache=kv_cache,
                            block_table=synth_bt,
                            seq_lens=synth_seq_lens,
                            scale=self.scale,
                            max_num_kv_splits=self.max_num_kv_splits,
                            PiT=PiT,
                            sinks=self.sinks,
                        )
                    else:
                        out = self._fp4_g32_continuation_prefill(
                            layer=layer,
                            query=q_seq,
                            key_chunk=k_seq,
                            val_chunk=v_seq,
                            kv_cache=kv_cache,
                            block_table=attn_metadata.block_table[i : i + 1],
                            cached_len=cached_len,
                            seq_len=seq_len,
                            PiT=PiT,
                        )
                elif self._is_fp4_g32:
                    if _USE_FP4_G32_V3:
                        # v3 dispatch mirrors TQ44 V3 (and v1/v2): the V3
                        # unified attention kernel is decode-optimized. For
                        # LARGE continuation chunks (q_len > 128) the standard
                        # TurboQuant pattern is "dequant cache once → bf16
                        # scratch → flash_attn", which beats running the
                        # unified kernel at large q. Small continuations
                        # (q_len ≤ 128) stay on the V3 synth-decode path
                        # because each query token gets its own seq and
                        # BLOCK_M=16 amortizes well.
                        _fp4_g32_unified = _lazy_fp4_g32_v3_import()
                        if q_len <= _CONTINUATION_DECODE_THRESHOLD:
                            cu_q = _arange_cache[: q_len + 1].to(torch.int32)
                            out = _fp4_g32_unified(
                                query=q_seq,
                                kv_cache=kv_cache,
                                block_table=synth_bt,
                                seq_lens=synth_seq_lens,
                                query_start_loc=cu_q,
                                scale=self.scale,
                                PiT=PiT,
                                max_query_len=1,
                                max_seq_len=int(seq_len),
                                sinks=self.sinks,
                                sliding_window=self.sliding_window,
                            )
                        else:
                            out = self._fp4_g32_continuation_prefill(
                                layer=layer,
                                query=q_seq,
                                key_chunk=k_seq,
                                val_chunk=v_seq,
                                kv_cache=kv_cache,
                                block_table=attn_metadata.block_table[i : i + 1],
                                cached_len=cached_len,
                                seq_len=seq_len,
                                PiT=PiT,
                            )
                    else:
                        _, _fp4_g32_decode, _ = _lazy_fp4_g32_imports()
                        if q_len <= _CONTINUATION_DECODE_THRESHOLD:
                            # Small continuation (≤128 tokens): decode kernel per query.
                            # Each query gets its own seq_len for causal masking.
                            out = _fp4_g32_decode(
                                query=q_seq,
                                kv_cache=kv_cache,
                                block_table=synth_bt,
                                seq_lens=synth_seq_lens,
                                scale=self.scale,
                                max_num_kv_splits=self.max_num_kv_splits,
                                PiT=PiT,
                                sinks=self.sinks,
                            )
                        else:
                            # Large continuation (>128 tokens): dequant cached KV once,
                            # then run flash_attn — mirrors v1's _continuation_prefill.
                            # This avoids q_len independent KV scans (O(q_len × ctx) work)
                            # and instead does O(ctx) dequant + O(q_len × ctx) flash_attn.
                            out = self._fp4_g32_continuation_prefill(
                                layer=layer,
                                query=q_seq,
                                key_chunk=k_seq,
                                val_chunk=v_seq,
                                kv_cache=kv_cache,
                                block_table=attn_metadata.block_table[i : i + 1],
                                cached_len=cached_len,
                                seq_len=seq_len,
                                PiT=PiT,
                            )
                elif q_len <= _CONTINUATION_DECODE_THRESHOLD:
                    # Fast path: treat each query as a decode request
                    # with incremental seq_lens for causal masking.
                    if _USE_TQ_V3 or _USE_TQ_V4:
                        # Continuation prefill stays on v3 even when v4 is the
                        # main decode kernel (v4 is decode-batch only, q_len=1
                        # implicit). Keeping v3 here makes the v3-leg vs v4-leg
                        # differ ONLY in the decode-batch kernel — clean A/B.
                        #
                        # IMPORTANT: use _dispatch_decode_v3 so that when the
                        # cache was written in SoA layout (VLLM_TQ_SOA_FUSION_STORE=1,
                        # required for v4), we read it with the SoA-aware v3.
                        # The default AoS v3 mis-addresses k_norm/v_scale/v_zero
                        # in a SoA cache → garbage cached-prefix output → accuracy
                        # collapse on every multi-turn / prefix-cached request.
                        out = self._dispatch_decode_v3(
                            query=q_seq,
                            kv_cache=kv_cache,
                            block_table=synth_bt,
                            seq_lens=synth_seq_lens,
                            Pi=Pi,
                            centroids=centroids,
                            scale=self.scale,
                            mse_bits=self.tq_config.key_mse_bits,
                            key_packed_size=self.tq_config.key_packed_size,
                            value_quant_bits=(
                                self.tq_config.effective_value_quant_bits
                            ),
                            value_packed_size=self.tq_config.value_packed_size,
                            max_seq_len=int(seq_len),
                            key_fp8=self.tq_config.key_fp8,
                            norm_correction=self.tq_config.norm_correction,
                            PiT=PiT,
                            sinks=self.sinks,
                            sliding_window=self.sliding_window,
                        )
                    elif _USE_TQ_V2:
                        # v2 kernel does not support sinks yet; sink plumbing
                        # lives on v1 (and soon v3). v2 is opt-in for perf
                        # experiments only.
                        out = triton_turboquant_decode_attention_v2(
                            query=q_seq,
                            kv_cache=kv_cache,
                            block_table=synth_bt,
                            seq_lens=synth_seq_lens,
                            Pi=Pi,
                            centroids=centroids,
                            scale=self.scale,
                            mse_bits=self.tq_config.key_mse_bits,
                            key_packed_size=self.tq_config.key_packed_size,
                            value_quant_bits=(
                                self.tq_config.effective_value_quant_bits
                            ),
                            value_packed_size=self.tq_config.value_packed_size,
                            max_seq_len=int(seq_len),
                            key_fp8=self.tq_config.key_fp8,
                            norm_correction=self.tq_config.norm_correction,
                            PiT=PiT,
                        )
                    else:
                        # v1 baseline: continuation fast-path using the
                        # Triton v1 decode kernel per synthetic query token.
                        out = triton_turboquant_decode_attention(
                            query=q_seq,
                            kv_cache=kv_cache,
                            block_table=synth_bt,
                            seq_lens=synth_seq_lens,
                            Pi=Pi,
                            centroids=centroids,
                            scale=self.scale,
                            mse_bits=self.tq_config.key_mse_bits,
                            key_packed_size=self.tq_config.key_packed_size,
                            value_quant_bits=(
                                self.tq_config.effective_value_quant_bits
                            ),
                            key_fp8=self.tq_config.key_fp8,
                            norm_correction=self.tq_config.norm_correction,
                            PiT=PiT,
                            sinks=self.sinks,
                        )
                else:
                    # Large continuation: dequant cached K/V and use
                    # flash_attn for better throughput.
                    out = self._continuation_prefill(
                        layer,
                        q_seq,
                        k_seq,
                        v_seq,
                        kv_cache,
                        attn_metadata.block_table[i : i + 1],
                        cached_len,
                        seq_len,
                        Pi,
                        centroids,
                    )
                output[q_start:q_end] = out.to(query.dtype)

        return output

    def _continuation_prefill(
        self,
        layer: Any,
        query: torch.Tensor,  # (q_len, Hq, D)
        key_chunk: torch.Tensor,  # (q_len, Hk, D)
        val_chunk: torch.Tensor,  # (q_len, Hk, D)
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot_size)
        block_table: torch.Tensor,  # (1, max_num_blocks)
        cached_len: int,
        seq_len: int,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
    ) -> torch.Tensor:
        """Handle continuation chunk by dequanting cached K/V from TQ cache.

        Dequants previously cached K/V, concatenates with the current
        chunk's raw K/V, then runs flash_attn with causal masking.
        """
        q_len, Hq, D = query.shape
        Hk = key_chunk.shape[1]
        device = query.device
        block_size = kv_cache.shape[1]
        BLOCK_D = triton.next_power_of_2(D)

        mse_bytes = self._mse_bytes
        val_data_bytes = self._val_data_bytes

        # Dequant cached K/V from TQ cache
        # Allocate slightly over to align to block_size for the grid.
        # Reuse cached buffers to avoid per-call allocation (~16MB at 8K).
        alloc_len = math.ceil(cached_len / block_size) * block_size
        buf_shape = (1, Hk, alloc_len, D)
        # Use WorkspaceManager for dequant buffers.
        # Shared across all layers — saves 60× memory at long context.
        # Required for CUDA Graph capture (per-layer growth incompatible with CG).
        #
        # Fallback: if the workspace was locked at a smaller size than needed
        # (happens when the CUDA-graph profile_seq_lens undershot max_model_len),
        # allocate directly. _continuation_prefill is always eager (never inside
        # a captured graph), so a per-call torch.empty is safe here.
        if is_workspace_manager_initialized():
            try:
                k_buf, v_buf = current_workspace_manager().get_simultaneous(
                    (buf_shape, torch.float16),
                    (buf_shape, torch.float16),
                )
            except AssertionError:
                # WorkspaceManager locked too small — use the shared fallback
                # buffers rather than one pair per layer, which would allocate
                # ~60x more outside the memory budget. Sized to this call's
                # alloc_len, not to max_model_len.
                _n = Hk * alloc_len * D
                _kf, _vf = _get_shared_prefill_bufs(
                    "dequant", _n, device, torch.float16
                )
                k_buf = _kf[:_n].view(buf_shape)
                v_buf = _vf[:_n].view(buf_shape)
        else:
            _n = Hk * alloc_len * D
            _kf, _vf = _get_shared_prefill_bufs("dequant", _n, device, torch.float16)
            k_buf = _kf[:_n].view(buf_shape)
            v_buf = _vf[:_n].view(buf_shape)
        # Skip .zero_() — kernel writes all positions up to cached_len,
        # and we only read [:cached_len] afterwards.
        k_cached = k_buf[:, :, :alloc_len, :]
        v_cached = v_buf[:, :, :alloc_len, :]


        # Opt#3 SoA layout constants (must match store-side computation).
        key_fp8 = self.tq_config.key_fp8
        key_data_bytes = D if key_fp8 else mse_bytes
        data_bytes_per_slot = key_data_bytes + val_data_bytes
        meta_region_offset = block_size * Hk * data_bytes_per_slot
        num_soa_fields = 2 if key_fp8 else 3
        soa_k_norm = 0
        soa_v_scale = 0 if key_fp8 else 1
        soa_v_zero = 1 if key_fp8 else 2
        kv_cache_u16 = kv_cache.view(torch.uint16)

        grid = (alloc_len, 1 * Hk)
        if self._soa_store:
            # SoA layout: use the SOA-aware dequant kernel.
            _, _, _, _soa_tq_full_dequant_kv = _lazy_soa_store_imports()
            _soa_tq_full_dequant_kv[grid](
                kv_cache,
                kv_cache_u16,
                block_table,
                centroids,
                k_cached,
                v_cached,
                k_cached.stride(0),
                k_cached.stride(1),
                k_cached.stride(2),
                v_cached.stride(0),
                v_cached.stride(1),
                v_cached.stride(2),
                kv_cache.stride(0),
                block_table.stride(0),
                HEAD_DIM=D,
                BLOCK_SIZE=block_size,
                NUM_KV_HEADS=Hk,
                MSE_BYTES=mse_bytes,
                VQB=self.tq_config.effective_value_quant_bits,
                VAL_DATA_BYTES=val_data_bytes,
                MSE_BITS=self.tq_config.key_mse_bits,
                KEY_FP8=1 if key_fp8 else 0,
                KEY_DATA_BYTES=key_data_bytes,
                META_REGION_OFFSET=meta_region_offset,
                NUM_SOA_FIELDS=num_soa_fields,
                SOA_K_NORM=soa_k_norm,
                SOA_V_SCALE=soa_v_scale,
                SOA_V_ZERO=soa_v_zero,
                BLOCK_D=BLOCK_D,
                NORM_CORRECTION=1 if self.tq_config.norm_correction else 0,
                FP8_E4B15=_use_fp8_e4b15(device.index or 0),
                num_warps=4,
            )
        else:
            _tq_full_dequant_kv[grid](
                kv_cache,
                block_table,
                centroids,
                k_cached,
                v_cached,
                k_cached.stride(0),
                k_cached.stride(1),
                k_cached.stride(2),
                v_cached.stride(0),
                v_cached.stride(1),
                v_cached.stride(2),
                kv_cache.stride(0),
                kv_cache.stride(1),
                kv_cache.stride(2),
                block_table.stride(0),
                HEAD_DIM=D,
                BLOCK_SIZE=block_size,
                NUM_KV_HEADS=Hk,
                MSE_BYTES=mse_bytes,
                KPS=self.tq_config.key_packed_size,
                VQB=self.tq_config.effective_value_quant_bits,
                VAL_DATA_BYTES=val_data_bytes,
                MSE_BITS=self.tq_config.key_mse_bits,
                N_CENTROIDS=2 ** self.tq_config.key_mse_bits,
                KEY_FP8=1 if key_fp8 else 0,
                BLOCK_D=BLOCK_D,
                NORM_CORRECTION=1 if self.tq_config.norm_correction else 0,
                FP8_E4B15=_use_fp8_e4b15(device.index or 0),
                OUT_BF16=0,
                num_warps=4,
            )

        # Inverse-rotate MSE keys back to original space
        if not self.tq_config.key_fp8:
            # fp16 matmul for rotation (2× less bandwidth, uses fp16 tensor cores)
            # Use a layer-cached output buffer for the matmul to avoid variable-size
            # torch allocations that can conflict with the ROCm HIP graph pool.
            Pi_half = layer._tq_Pi_half
            k_flat = k_cached[0, :, :cached_len, :].reshape(-1, D)
            _kflat_rows = k_flat.shape[0]  # = Hk * cached_len
            _kflat_max = block_table.shape[1] * block_size * Hk
            _kflat_buf: torch.Tensor | None = getattr(layer, "_tq_kflat_buf", None)
            if _kflat_buf is None or _kflat_buf.shape[0] < _kflat_max:
                _kflat_buf = torch.empty(_kflat_max, D, dtype=torch.float16, device=device)
                layer._tq_kflat_buf = _kflat_buf
            torch.mm(k_flat, Pi_half, out=_kflat_buf[:_kflat_rows])
            k_flat = _kflat_buf[:_kflat_rows]
            k_cached_trim = k_flat.reshape(Hk, cached_len, D).transpose(
                0, 1
            )  # (cached_len, Hk, D) — already fp16
        else:
            k_cached_trim = k_cached[0, :, :cached_len, :].transpose(
                0, 1
            )  # (cached_len, Hk, D)

        # Skip .contiguous() — the copy into k_full/v_full handles layout
        v_cached_trim = v_cached[0, :, :cached_len, :].transpose(0, 1)

        # Concatenate cached + current chunk K/V (match query dtype).
        # Cache k_full/v_full on the layer object so they are allocated once
        # at worst-case size during warmup and reused on every subsequent call
        # — including during piecewise CUDA graph replay.  This matches the
        # pattern FusionTurboQuantAttentionImpl uses for its k_buf/v_buf and
        # eliminates the variable-size torch.empty() calls that cause ROCm HIP
        # graph pool address collisions (→ garbage outputs at 32K, GPU fault at
        # 128K).  Worst-case capacity = all KV blocks × block_size.
        # Shared across layers rather than cached per layer: at 145K the
        # per-layer version wanted 1.19 GiB x 2 x 60 layers outside the memory
        # budget (see _get_shared_prefill_bufs). Capacity tracks seq_len, not
        # max_model_len, so short requests do not reserve the worst case.
        qdtype = query.dtype
        _n = seq_len * Hk * D
        _kfull_buf, _vfull_buf = _get_shared_prefill_bufs(
            "kvfull", _n, device, qdtype
        )
        k_full = _kfull_buf[:_n].view(seq_len, Hk, D)
        v_full = _vfull_buf[:_n].view(seq_len, Hk, D)
        k_full[:cached_len] = k_cached_trim.to(qdtype)
        k_full[cached_len:] = key_chunk
        v_full[:cached_len] = v_cached_trim.to(qdtype)
        v_full[cached_len:] = val_chunk


        # Attention: q_len queries attending to seq_len K/V with causal mask
        #
        # Sinks gating: flash_attn on ROCm is locked to FA2 (TurboQuant
        # incompatibility with FA3) and FA2 has no sinks parameter, so models
        # like gpt-oss that fold a per-head sink logit into the softmax
        # denominator can NOT go through flash_attn here — the resulting
        # softmax overflows at long context and crashes downstream MoE.
        # When sinks are present, route through the upstream Triton
        # ``unified_attention`` kernel (same path used for first-chunk
        # prefill above and by upstream RocmAttentionImpl); it natively
        # supports sinks and handles cu_seqlens_q != cu_seqlens_k as a
        # lower-right causal mask (q at absolute position cached_len+i
        # attends K[0..cached_len+i]).
        # D > _CK_MAX_HEAD_DIM joins the sinks case here for the same reason as
        # in _prefill_attention: CK's flash kernel below cannot handle it,
        # while this Triton kernel is generic over head size.
        if self.sinks is not None or D > _CK_MAX_HEAD_DIM:
            out = torch.empty_like(query)
            cu_q = torch.tensor([0, q_len], dtype=torch.int32, device=device)
            seqused_k = torch.tensor([seq_len], dtype=torch.int32, device=device)
            # Single virtual block of size seq_len: unified_attention reads
            # block_size dynamically from v.shape[1], so wrapping the
            # contiguous k_full/v_full as [1, seq_len, Hk, D] with a 1-entry
            # block_table works without paging logic.
            block_table_single = torch.zeros((1, 1), dtype=torch.int32, device=device)
            _win = (
                (self.sliding_window - 1, 0)
                if self.sliding_window and self.sliding_window > 0
                else (-1, -1)
            )
            unified_attention(
                q=query,
                k=k_full.unsqueeze(0),
                v=v_full.unsqueeze(0),
                out=out,
                cu_seqlens_q=cu_q,
                max_seqlen_q=q_len,
                seqused_k=seqused_k,
                max_seqlen_k=seq_len,
                softmax_scale=self.scale,
                causal=True,
                window_size=_win,
                block_table=block_table_single,
                softcap=0.0,
                q_descale=None,
                k_descale=None,
                v_descale=None,
                sinks=self.sinks,
            )
            return out
        if _HAS_FLASH_ATTN:
            cu_seqlens_q = torch.tensor([0, q_len], device=device, dtype=torch.int32)
            cu_seqlens_k = torch.tensor([0, seq_len], device=device, dtype=torch.int32)
            _fa_window = (
                (self.sliding_window - 1, 0)
                if self.sliding_window and self.sliding_window > 0
                else None
            )
            _fa_out = flash_attn_varlen_func(
                q=query,
                k=k_full,
                v=v_full,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=q_len,
                max_seqlen_k=seq_len,
                softmax_scale=self.scale,
                causal=True,
                **({"window_size": _fa_window} if _fa_window is not None else {}),
            )
            return _fa_out
        else:
            # SDPA fallback: expand KV for GQA, build causal mask
            q_t = query.transpose(0, 1).unsqueeze(0)  # (1, Hq, q_len, D)
            k_t = k_full.transpose(0, 1).unsqueeze(0)  # (1, Hk, seq_len, D)
            v_t = v_full.transpose(0, 1).unsqueeze(0)  # (1, Hk, seq_len, D)
            # Build causal mask: query position p can attend to K position j
            # where j <= cached_len + p (p is 0-indexed within chunk)
            q_pos = torch.arange(q_len, device=device).unsqueeze(1) + cached_len
            k_pos = torch.arange(seq_len, device=device).unsqueeze(0)
            mask = k_pos <= q_pos  # (q_len, seq_len)
            out = F.scaled_dot_product_attention(
                q_t,
                k_t,
                v_t,
                attn_mask=mask,
                scale=self.scale,
                enable_gqa=(Hk < Hq),
            )  # (1, Hq, q_len, D)
            return out[0].transpose(0, 1)  # (q_len, Hq, D)

    # ------------------------------------------------------------------ #
    #  FP4 large-continuation prefill: dequant cache + flash_attn        #
    # ------------------------------------------------------------------ #
    def _fp4_g32_continuation_prefill(
        self,
        layer: Any,
        query: torch.Tensor,       # [q_len, Hq, D] raw (not yet rotated)
        key_chunk: torch.Tensor,   # [q_len, Hk, D] raw
        val_chunk: torch.Tensor,   # [q_len, Hk, D] raw
        kv_cache: torch.Tensor,    # paged fp4 cache uint8
        block_table: torch.Tensor, # [1, max_num_blocks] int32
        cached_len: int,
        seq_len: int,
        PiT: torch.Tensor,         # [D, D] fp32 Hadamard
    ) -> torch.Tensor:
        """FP4-g32 continuation prefill — mirrors v1's `_continuation_prefill`.

        Pipeline (same as v1, only the dequant kernel differs):
          1. Acquire workspace buffers for dequant target [1, Hk, alloc_len, D]
          2. Triton dequant kernel writes K (rotated) and V (raw) into buffers
          3. Layer-cached `k_full`/`v_full` [max_seq, Hk, D] are filled:
             - [:cached_len] ← dequant buffer (transpose copy)
             - [cached_len:] ← rotated current chunk K / raw V
          4. flash_attn_varlen_func with causal mask

        FP4 advantage over v1: K is stored in Hadamard-rotated space, so we
        skip v1's `k_flat @ Pi_half` inverse-rotation fp16 matmul step. The
        rotation happens once on the small current chunk (q_len ≤ 8K) and on
        the q_len queries (Q @ PiT), not on the full cached_len.
        """
        if self._is_fp8_g32:
            _, _, fp_full_dequant = _lazy_fp8_g32_imports()
        else:
            from vllm.v1.attention.ops.fp4_g32.triton_decode import (
                fp4_g32_full_dequant_kv as fp_full_dequant,
            )

        q_len, Hq, D = query.shape
        Hk = key_chunk.shape[1]
        device = query.device
        qdtype = query.dtype
        block_size = kv_cache.shape[1]
        alloc_len = math.ceil(cached_len / block_size) * block_size

        # 1. Acquire dequant target buffers [1, Hk, alloc_len, D].
        # Workspace-shared across layers — required for CUDA Graph safety and
        # massive memory savings at long context (saves Nlayer× allocations).
        buf_shape = (1, Hk, alloc_len, D)
        if is_workspace_manager_initialized():
            try:
                k_buf, v_buf = current_workspace_manager().get_simultaneous(
                    (buf_shape, qdtype),
                    (buf_shape, qdtype),
                )
            except AssertionError:
                # WorkspaceManager too small (CG profile undershot max_model_len)
                # → fall back to layer-cached grow-only buffers. Never use bare
                # torch.empty here: address could land in captured HIP graph pool.
                k_buf, v_buf = _fp4_get_layer_dequant_bufs(
                    layer, Hk, block_table.shape[1] * block_size, D, qdtype, device
                )
        else:
            k_buf, v_buf = _fp4_get_layer_dequant_bufs(
                layer, Hk, block_table.shape[1] * block_size, D, qdtype, device
            )

        # 2. Triton dequant kernel writes K (rotated) and V (raw) into bufs.
        fp_full_dequant(
            kv_cache=kv_cache,
            block_table=block_table,
            k_out=k_buf,
            v_out=v_buf,
            alloc_len=alloc_len,
        )

        # 3. Acquire layer-cached k_full/v_full [max_possible_seq, Hk, D] in qdtype.
        # Same lifecycle as v1: allocated once at worst-case size during the
        # first call, reused on every subsequent call (including CG replay).
        _kfull_cap = block_table.shape[1] * block_size
        _kfull_buf = getattr(layer, "_fp4_kfull_buf", None)
        if (
            _kfull_buf is None
            or _kfull_buf.shape[0] < _kfull_cap
            or _kfull_buf.dtype != qdtype
        ):
            _kfull_buf = torch.empty(_kfull_cap, Hk, D, dtype=qdtype, device=device)
            layer._fp4_kfull_buf = _kfull_buf
        _vfull_buf = getattr(layer, "_fp4_vfull_buf", None)
        if (
            _vfull_buf is None
            or _vfull_buf.shape[0] < _kfull_cap
            or _vfull_buf.dtype != qdtype
        ):
            _vfull_buf = torch.empty(_kfull_cap, Hk, D, dtype=qdtype, device=device)
            layer._fp4_vfull_buf = _vfull_buf
        k_full = _kfull_buf[:seq_len]
        v_full = _vfull_buf[:seq_len]

        # 4. Copy dequanted cache into k_full/v_full prefix.
        # `k_buf[0, :, :cached_len, :].transpose(0, 1)` is a strided view
        # [cached_len, Hk, D]; the assignment dispatches a contiguous copy.
        k_full[:cached_len] = k_buf[0, :, :cached_len, :].transpose(0, 1)
        v_full[:cached_len] = v_buf[0, :, :cached_len, :].transpose(0, 1)

        # 5. Rotate current chunk K and Q (small matmul, q_len ≤ 8K).
        # K_chunk_rot = K_chunk @ PiT → store in k_full[cached_len:].
        # Q_rot     = Q       @ PiT → input to flash_attn.
        # Hadamard is symmetric orthogonal so Q_rot · K_rot.T == Q · K.T (math
        # is identity-preserving for the attention scores).
        k_full[cached_len:] = (
            key_chunk.to(torch.float32) @ PiT
        ).to(qdtype)
        v_full[cached_len:] = val_chunk
        q_rot = (query.to(torch.float32) @ PiT).to(qdtype)  # [q_len, Hq, D]

        # 6. Flash attention with lower-right causal mask.
        # flash_attn_varlen_func with unequal Q/K lengths applies the correct
        # mask: Q[i] (absolute position cached_len+i) attends K[0..cached_len+i].
        #
        # Sinks gating (mirrors `_continuation_prefill`): FA2 on ROCm has no
        # sinks parameter, so for gpt-oss-style sinks models we route through
        # the Triton ``unified_attention`` kernel — same path upstream
        # RocmAttentionImpl uses on ROCm for sinks. Q/K are already in
        # Hadamard-rotated space (q_rot, k_full) so the dot products match
        # the original-space attention scores.
        if self.sinks is not None or D > _CK_MAX_HEAD_DIM:
            out = torch.empty_like(q_rot)
            cu_q = torch.tensor([0, q_len], dtype=torch.int32, device=device)
            seqused_k = torch.tensor([seq_len], dtype=torch.int32, device=device)
            # Wrap contiguous k_full/v_full as a single virtual block of
            # size seq_len; unified_attention reads block_size dynamically
            # from v.shape[1].
            block_table_single = torch.zeros((1, 1), dtype=torch.int32, device=device)
            _win = (
                (self.sliding_window - 1, 0)
                if self.sliding_window and self.sliding_window > 0
                else (-1, -1)
            )
            unified_attention(
                q=q_rot,
                k=k_full.unsqueeze(0),
                v=v_full.unsqueeze(0),
                out=out,
                cu_seqlens_q=cu_q,
                max_seqlen_q=q_len,
                seqused_k=seqused_k,
                max_seqlen_k=seq_len,
                softmax_scale=self.scale,
                causal=True,
                window_size=_win,
                block_table=block_table_single,
                softcap=0.0,
                q_descale=None,
                k_descale=None,
                v_descale=None,
                sinks=self.sinks,
            )
            return out.to(qdtype)
        if _HAS_FLASH_ATTN:
            cu_q = torch.tensor([0, q_len], dtype=torch.int32, device=device)
            cu_k = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
            _fa_window = (
                (self.sliding_window - 1, 0)
                if self.sliding_window and self.sliding_window > 0
                else None
            )
            out = flash_attn_varlen_func(
                q=q_rot,
                k=k_full,
                v=v_full,
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                max_seqlen_q=q_len,
                max_seqlen_k=seq_len,
                softmax_scale=self.scale,
                causal=True,
                **({"window_size": _fa_window} if _fa_window is not None else {}),
            )
        else:
            kv_group = Hq // Hk
            q_t = q_rot.transpose(0, 1).unsqueeze(0)
            k_t = k_full.transpose(0, 1).unsqueeze(0)
            v_t = v_full.transpose(0, 1).unsqueeze(0)
            if kv_group > 1:
                k_t = k_t.expand(1, Hq, -1, -1)
                v_t = v_t.expand(1, Hq, -1, -1)
            import torch.nn.functional as _F
            out = _F.scaled_dot_product_attention(
                q_t, k_t, v_t,
                is_causal=True,
                scale=self.scale,
            ).squeeze(0).transpose(0, 1)

        return out.to(qdtype)

    # ------------------------------------------------------------------ #
    #  Decode: Triton TQ decode attention                                 #
    # ------------------------------------------------------------------ #
    def _decode_attention(
        self,
        query: torch.Tensor,  # (B, Hq, D)
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot_size)
        attn_metadata: TurboQuantMetadata,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
        PiT: torch.Tensor | None = None,
        layer: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        # Acquire shared decode scratch buffers from WorkspaceManager.
        # Layers execute sequentially so one set of buffers is sufficient.
        # Falls back to kernel-internal allocation if workspace unavailable.
        B = query.shape[0]
        D = self.head_size
        S = self.max_num_kv_splits
        Hq = self.num_heads
        mid_o_buf = output_buf = lse_buf = None
        if is_workspace_manager_initialized():
            # output_buf in query dtype — matches the in-kernel fp16 cast in stage2.
            mid_o_buf, output_buf, lse_buf = (
                current_workspace_manager().get_simultaneous(
                    ((B, Hq, S, D + 1), torch.float32),
                    ((B, Hq, D), query.dtype),
                    ((B, Hq), torch.float32),
                )
            )

        if self._is_fp4_g32:
            if _USE_FP4_G32_V3:
                # v3-based unified decode: real MFMA QK/PV via tl.dot, GQA
                # stacking, 2D/3D split-KV dispatch. ~2.7-2.9x faster than
                # v1-based decode on MI300X at long context (verified
                # bit-similar bf16 output vs v1 in ops tests).
                # Mirrors TQ V3's pattern: fresh `torch.arange` per call (CUDA
                # graph capture pool handles per-B-size variants correctly).
                _fp4_g32_unified = _lazy_fp4_g32_v3_import()
                cu_q = torch.arange(
                    B + 1,
                    dtype=attn_metadata.seq_lens.dtype,
                    device=query.device,
                )
                _fp4_result = _fp4_g32_unified(
                    query=query,
                    kv_cache=kv_cache,
                    block_table=attn_metadata.block_table,
                    seq_lens=attn_metadata.seq_lens,
                    query_start_loc=cu_q,
                    scale=self.scale,
                    PiT=PiT,
                    max_query_len=1,
                    max_seq_len=int(attn_metadata.max_seq_len),
                    sinks=self.sinks,
                    sliding_window=self.sliding_window,
                    output=output_buf[:B] if output_buf is not None else None,
                )
                return _fp4_result

            _, _fp4_g32_decode, _ = _lazy_fp4_g32_imports()
            return _fp4_g32_decode(
                query=query,
                kv_cache=kv_cache,
                block_table=attn_metadata.block_table,
                seq_lens=attn_metadata.seq_lens,
                scale=self.scale,
                max_num_kv_splits=self.max_num_kv_splits,
                PiT=PiT,
                sinks=self.sinks,
                mid_o_buf=mid_o_buf,
                output_buf=output_buf,
                lse_buf=lse_buf,
            )

        if self._is_fp8_g32:
            if _USE_FP8_G32_V4:
                # FlyDSL fp8_g32 decode kernel: MI355X/gfx950 only,
                # HEAD_SIZE=128, no sinks, no SWA. GQA factor:
                #   * 8, 16 → canonical fp8_g32_decode_v4    (Qwen2.5/3)
                #   * 6     → fp8_g32_decode_v4_gqa6 sibling (MiniMax-M2.5),
                #             only when that sibling imported successfully.
                # Falls back to the Triton fp8_g32 path below when the layer
                # is ineligible. The kernel is a port of the bug-free TQ v4
                # FlyDSL kernel adapted to FP4 E2M1 codes + UE8M0 per-group-32
                # scales (Arch A/B read from fp8_levels).
                _g = self.num_kv_groups
                if self.head_size == 256:
                    # HEAD_SIZE=256 (Qwen3.6-class) routes to the fp8_g32 hd256
                    # siblings (base bf16-QK path): GQA-6 -> gqa6 hd256,
                    # GQA-{8,16} -> canonical hd256.
                    fp8_v4_eligible = (
                        _g in (6, 8, 16)
                        and _flydsl_fp8_v4_hd256_available(_g)
                        and self.sinks is None
                        and not (self.sliding_window and self.sliding_window > 0)
                    )
                else:
                    fp8_v4_gqa_ok = (_g in (8, 16)) or (
                        _g == 6 and _flydsl_fp8_v4_gqa6_available()
                    )
                    fp8_v4_eligible = (
                        self.head_size == 128
                        and fp8_v4_gqa_ok
                        and self.sinks is None
                        and not (self.sliding_window and self.sliding_window > 0)
                    )
                if fp8_v4_eligible:
                    # v5 FUSED opt-in: only for the hd256 (D=256) path, only
                    # when the flag is set AND the fused sibling is importable.
                    # Any failure falls back to the v4 UQ-adaptive kernel so
                    # the shipped path is never at risk.
                    _use_v5 = (
                        _USE_FP8_G32_V5_FUSED
                        and self.head_size == 256
                        and _flydsl_fp8_v5_hd256_available(
                            self.num_kv_groups)
                    )
                    _fp8_decode_fn = (
                        flydsl_fp8_g32_decode_attention_v5_fused
                        if _use_v5
                        else flydsl_fp8_g32_decode_attention_v4
                    )
                    _fly_out = _fp8_decode_fn(
                        query=query,
                        kv_cache=kv_cache,
                        block_table=attn_metadata.block_table,
                        seq_lens=attn_metadata.seq_lens,
                        scale=self.scale,
                        PiT=PiT,
                        max_seq_len=attn_metadata.max_seq_len,
                        output_buf=output_buf,
                        buf_holder=layer,
                        max_num_kv_splits=self.max_num_kv_splits,
                        sinks=self.sinks,
                    )
                    if _FP8_SHADOW_CMP:
                        try:
                            _shadow_compare_fp8(
                                self, query, kv_cache, attn_metadata,
                                PiT, _fly_out)
                        except Exception as _sce:  # noqa: BLE001
                            logger.warning_once(
                                "fp8 shadow-compare failed: %s", _sce)
                    return _fly_out
                logger.warning_once(
                    "fp8_g32 v4 eligibility failed (head_size=%s "
                    "num_kv_groups=%s sinks=%s swa=%s) — falling back to the "
                    "fp8_g32 Triton path",
                    self.head_size, self.num_kv_groups,
                    self.sinks is not None,
                    bool(self.sliding_window and self.sliding_window > 0),
                )
            if _USE_FP8_G32_V3:
                # v3-based unified decode: hardware F8F6F4 scaled MFMA on QK
                # via tl.dot_scaled (FP8 E4M3 × FP4 E2M1 + E8M0 scales) +
                # bf16 tl.dot on PV. GQA stacking + 2D/3D split-KV dispatch.
                # Bit-similar to v1 decode in ops tests (cos_sim ≥ 0.999998
                # vs v1, ≥ 0.99998 vs reference). Mirrors fp4_g32 V3
                # dispatch pattern.
                _fp8_g32_unified = _lazy_fp8_g32_v3_import()
                cu_q = torch.arange(
                    B + 1,
                    dtype=attn_metadata.seq_lens.dtype,
                    device=query.device,
                )
                _fp8_result = _fp8_g32_unified(
                    query=query,
                    kv_cache=kv_cache,
                    block_table=attn_metadata.block_table,
                    seq_lens=attn_metadata.seq_lens,
                    query_start_loc=cu_q,
                    scale=self.scale,
                    PiT=PiT,
                    max_query_len=1,
                    max_seq_len=int(attn_metadata.max_seq_len),
                    sinks=self.sinks,
                    sliding_window=self.sliding_window,
                    output=output_buf[:B] if output_buf is not None else None,
                )
                return _fp8_result

            # Legacy v1-based decode + (reused fp4_g32) stage-2 reduce.
            _, _fp8_decode, _ = _lazy_fp8_g32_imports()
            return _fp8_decode(
                query=query,
                kv_cache=kv_cache,
                block_table=attn_metadata.block_table,
                seq_lens=attn_metadata.seq_lens,
                scale=self.scale,
                max_num_kv_splits=self.max_num_kv_splits,
                PiT=PiT,
                sinks=self.sinks,
                mid_o_buf=mid_o_buf,
                output_buf=output_buf,
                lse_buf=lse_buf,
            )

        if _USE_TQ_V4:
            # FlyDSL v4 decode kernel: MI355X/gfx950 only, MSE-key path,
            # HEAD_SIZE=128, GQA in {6, 8, 16}. Falls back to v3 for FP8
            # keys or when sinks are required.
            #
            # GQA dispatch (handled inside the launcher):
            #   * 8/16 → canonical tq_decode_v4 kernel  (Qwen 72B / 32B)
            #   * 6    → tq_decode_v4_gqa6 sibling      (MiniMax-M2.5)
            #
            # GQA-6 also requires the optional sibling module to be
            # importable from the FlyDSL checkout — if it's missing,
            # gate the layer out of v4 here so we route cleanly to v3
            # instead of erroring at launch.
            #
            # ``norm_correction`` does NOT block v4: the correction is
            # pre-folded into the stored K-norm value at storage time
            # (see ``triton_turboquant_store._store_packed_key`` step 3),
            # so the decode kernel just multiplies ``c_vals * stored_knorm``
            # regardless of whether the model uses norm_correction or not.
            # v3 keeps NORM_CORRECTION as a constexpr only for API parity.
            #   * HEAD_SIZE=256 (Qwen3.6-27B / Qwen3.5-397B): GQA-{8,16} →
            #     tq_decode_hd256 sibling; GQA-6 → tq_decode_gqa6_hd256 sibling.
            #     Both are optional modules gated here so missing ones route to
            #     v3 rather than erroring at launch.
            _gqa = self.num_kv_groups
            if self.head_size == 256:
                if _gqa in (2, 8, 16):
                    # QG=2 (Gemma 4 sliding layers) reuses the canonical hd256
                    # module's MFMA half-fill + mfma_row<QG store gate, which
                    # already generalize below GQA-8. Validated numerically vs
                    # exact fp32 attention (bench_hd256_gqa2_sliding.py).
                    v4_hs_ok = _flydsl_v4_hd256_available()
                    v4_gqa_ok = True
                elif _gqa == 6:
                    v4_hs_ok = _flydsl_v4_gqa6_hd256_available()
                    v4_gqa_ok = True
                else:
                    v4_hs_ok = False
                    v4_gqa_ok = False
            else:
                v4_hs_ok = self.head_size == 128
                v4_gqa_ok = (_gqa in (8, 16)) or (
                    _gqa == 6 and _flydsl_v4_gqa6_available()
                )
            # SWA routing. The canonical HEAD_SIZE=256 kernel (QG in 2/8/16)
            # now implements decode sliding-window (base-shift to the aligned
            # window start + per-token window mask; validated in
            # tests/kernels/turboquant_v4/test_hd256_swa.py). Every other v4
            # path (hd128, gqa6_hd256, and any future hd512) still lacks a
            # window mask, so a SWA layer there would silently attend outside
            # its window and diverge once context exceeds the window — those
            # must stay on V3.
            _swa_active = bool(self.sliding_window and self.sliding_window > 0)
            _swa_ok_on_v4 = self.head_size == 256 and _gqa in (2, 8, 16)
            v4_eligible = (
                not self.tq_config.key_fp8
                and self.tq_config.key_mse_bits == 4
                and self.tq_config.effective_value_quant_bits == 4
                and v4_hs_ok
                and v4_gqa_ok
                and self.sinks is None
                and (not _swa_active or _swa_ok_on_v4)
            )
            if v4_eligible:
                _v4_mid = None if _V4_FRESH_BUFS else mid_o_buf
                _v4_out = None if _V4_FRESH_BUFS else output_buf
                _v4_lse = None if _V4_FRESH_BUFS else lse_buf
                _v4_holder = None if _V4_FRESH_BUFS else layer
                result = flydsl_turboquant_decode_attention_v4(
                    query=query,
                    kv_cache=kv_cache,
                    block_table=attn_metadata.block_table,
                    seq_lens=attn_metadata.seq_lens,
                    Pi=Pi,
                    centroids=centroids,
                    scale=self.scale,
                    mse_bits=self.tq_config.key_mse_bits,
                    key_packed_size=self.tq_config.key_packed_size,
                    value_quant_bits=self.tq_config.effective_value_quant_bits,
                    value_packed_size=self.tq_config.value_packed_size,
                    max_seq_len=attn_metadata.max_seq_len,
                    key_fp8=self.tq_config.key_fp8,
                    norm_correction=self.tq_config.norm_correction,
                    PiT=PiT,
                    mid_o_buf=_v4_mid,
                    output_buf=_v4_out,
                    lse_buf=_v4_lse,
                    buf_holder=_v4_holder,
                    max_num_kv_splits=self.max_num_kv_splits,
                    sinks=self.sinks,
                    # VLLM_TQ_V4_SWA_OFF=1 forces full-attention decode on the
                    # v4 sliding layers (diagnostic: matches the SoA-v3 decode
                    # baseline, which itself does not window during decode).
                    sliding_window=(
                        int(self.sliding_window)
                        if (_swa_active and not _V4_SWA_OFF)
                        else 0
                    ),
                )
            else:
                # Per-config gate failed — route to v3.
                logger.warning_once(
                    "v4 eligibility failed (key_fp8=%s mse_bits=%s vqb=%s "
                    "head_size=%s num_kv_groups=%s sinks=%s) — "
                    "falling back to v3",
                    self.tq_config.key_fp8,
                    self.tq_config.key_mse_bits,
                    self.tq_config.effective_value_quant_bits,
                    self.head_size,
                    self.num_kv_groups,
                    self.sinks is not None,
                )
                result = self._dispatch_decode_v3(
                    query=query,
                    kv_cache=kv_cache,
                    block_table=attn_metadata.block_table,
                    seq_lens=attn_metadata.seq_lens,
                    Pi=Pi,
                    centroids=centroids,
                    scale=self.scale,
                    mse_bits=self.tq_config.key_mse_bits,
                    key_packed_size=self.tq_config.key_packed_size,
                    value_quant_bits=self.tq_config.effective_value_quant_bits,
                    value_packed_size=self.tq_config.value_packed_size,
                    max_seq_len=attn_metadata.max_seq_len,
                    key_fp8=self.tq_config.key_fp8,
                    norm_correction=self.tq_config.norm_correction,
                    PiT=PiT,
                    mid_o_buf=mid_o_buf,
                    output_buf=output_buf,
                    lse_buf=lse_buf,
                    buf_holder=layer,
                    max_num_kv_splits=self.max_num_kv_splits,
                    sinks=self.sinks,
                    sliding_window=self.sliding_window,
                )
        elif _USE_TQ_V3:
            result = self._dispatch_decode_v3(
                query=query,
                kv_cache=kv_cache,
                block_table=attn_metadata.block_table,
                seq_lens=attn_metadata.seq_lens,
                Pi=Pi,
                centroids=centroids,
                scale=self.scale,
                mse_bits=self.tq_config.key_mse_bits,
                key_packed_size=self.tq_config.key_packed_size,
                value_quant_bits=self.tq_config.effective_value_quant_bits,
                value_packed_size=self.tq_config.value_packed_size,
                max_seq_len=attn_metadata.max_seq_len,
                key_fp8=self.tq_config.key_fp8,
                norm_correction=self.tq_config.norm_correction,
                PiT=PiT,
                mid_o_buf=mid_o_buf,
                output_buf=output_buf,
                lse_buf=lse_buf,
                buf_holder=layer,
                max_num_kv_splits=self.max_num_kv_splits,
                sinks=self.sinks,
                sliding_window=self.sliding_window,
            )
        elif _USE_TQ_V2:
            # v2 kernel does not support sinks yet; sink plumbing lives on v1
            # (and soon v3). v2 is opt-in for perf experiments only.
            result = triton_turboquant_decode_attention_v2(
                query=query,
                kv_cache=kv_cache,
                block_table=attn_metadata.block_table,
                seq_lens=attn_metadata.seq_lens,
                Pi=Pi,
                centroids=centroids,
                scale=self.scale,
                mse_bits=self.tq_config.key_mse_bits,
                key_packed_size=self.tq_config.key_packed_size,
                value_quant_bits=self.tq_config.effective_value_quant_bits,
                value_packed_size=self.tq_config.value_packed_size,
                max_seq_len=attn_metadata.max_seq_len,
                key_fp8=self.tq_config.key_fp8,
                norm_correction=self.tq_config.norm_correction,
                PiT=PiT,
                mid_o_buf=mid_o_buf,
                output_buf=output_buf,
                lse_buf=lse_buf,
                buf_holder=layer,
                max_num_kv_splits=self.max_num_kv_splits,
            )
        else:
            result = triton_turboquant_decode_attention(
                query=query,
                kv_cache=kv_cache,
                block_table=attn_metadata.block_table,
                seq_lens=attn_metadata.seq_lens,
                Pi=Pi,
                centroids=centroids,
                scale=self.scale,
                mse_bits=self.tq_config.key_mse_bits,
                key_packed_size=self.tq_config.key_packed_size,
                value_quant_bits=self.tq_config.effective_value_quant_bits,
                key_fp8=self.tq_config.key_fp8,
                norm_correction=self.tq_config.norm_correction,
                PiT=PiT,
                mid_o_buf=mid_o_buf,
                output_buf=output_buf,
                lse_buf=lse_buf,
                buf_holder=layer,
                max_num_kv_splits=self.max_num_kv_splits,
                sinks=self.sinks,
            )
        if _TQ_SHADOW_CMP and _USE_TQ_V4 and not self.tq_config.key_fp8:
            try:
                _shadow_compare_tq(
                    self, query, kv_cache, attn_metadata, Pi, PiT,
                    centroids, result)
            except Exception as _sce:  # noqa: BLE001
                logger.warning_once("tq shadow-compare failed: %s", _sce)
        return result
