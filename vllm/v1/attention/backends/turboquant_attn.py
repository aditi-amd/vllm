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
from vllm.v1.attention.ops.triton_turboquant_unified_attention import (
    triton_turboquant_decode_attention_v3,
)
from vllm.v1.attention.ops.flydsl_turboquant_decode_v4 import (
    flydsl_turboquant_decode_attention_v4,
    is_flydsl_available as _flydsl_v4_available,
    is_flydsl_gqa6_available as _flydsl_v4_gqa6_available,
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
if _USE_TQ_V4 and not _flydsl_v4_available():
    logger.warning(
        "VLLM_TQ_DECODE_V4 requested but FlyDSL is unavailable; "
        "falling back to v3."
    )
    _USE_TQ_V4 = False

_HAS_FLASH_ATTN = is_flash_attn_varlen_func_available()
if _HAS_FLASH_ATTN:
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

logger.info_once(
    "TurboQuant has flash attn: %s, decode kernel: %s, fp4_g32_v3: %s",
    _HAS_FLASH_ATTN,
    "v4(flydsl)" if _USE_TQ_V4 else "v3" if _USE_TQ_V3 else "v2" if _USE_TQ_V2 else "v1",
    _USE_FP4_G32_V3,
)
# Continuation prefill: for small continuation chunks (q_len ≤ threshold),
# use the TQ decode kernel directly instead of full-dequant + flash_attn.
# do_kv_cache_update already stored all tokens to TQ cache, so the decode
# kernel can read them efficiently. This avoids O(cached_len) dequant work
# per continuation, eliminating the O(N²/chunk_size) collapse at long context.
_CONTINUATION_DECODE_THRESHOLD = 128


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
        tq_config = TurboQuantConfig.from_cache_dtype(cache_dtype_str, head_size)
        return (num_blocks, block_size, num_kv_heads, tq_config.slot_size_aligned)

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype is None:
            return False
        return kv_cache_dtype.startswith("turboquant_") or kv_cache_dtype == "fp4_kv_g32"

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

        from vllm.model_executor.layers.quantization.turboquant.config import (
            TurboQuantConfig,
        )

        if not self._is_fp4_g32:
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

            if not self._is_fp4_g32:
                # Centroids for Lloyd-Max quantization (TQ formats only).
                layer._tq_centroids = get_centroids(D, self.tq_config.centroid_bits).to(
                    device=device, dtype=torch.float32
                )
                c_sorted, _ = layer._tq_centroids.sort()
                layer._tq_midpoints = (c_sorted[:-1] + c_sorted[1:]) / 2
            layer._tq_cached = True

    def _max_capture_batch_size(self) -> int:
        """Return the largest batch size that will be captured in CUDA graphs.

        Used to pre-warm the WorkspaceManager before capture begins so that
        workspace growth cannot happen mid-capture and invalidate baked-in ptrs.
        Falls back to a generous heuristic (512) if config is unavailable.
        """
        try:
            from vllm.v1.utils import get_current_vllm_config as _gcvc
            cfg = _gcvc()
            sizes = cfg.compilation_config.cudagraph_capture_sizes
            if sizes:
                return int(max(sizes))
        except Exception:
            pass
        return 512

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
        if (
            self.sinks is None
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
                if self.sinks is not None:
                    # Use unified attention for sink support
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
                        window_size=(-1, -1),
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
                if self._is_fp4_g32:
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
                        out = triton_turboquant_decode_attention_v3(
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
                # WorkspaceManager too small — fall back to layer-cached buffer
                # (same grow-only pattern as FusionTurboQuantAttentionImpl).
                # Never use a bare torch.empty here: that allocation can receive
                # an address inside the captured HIP graph pool on ROCm.
                _kbuf = getattr(layer, "_tq_k_dequant_buf", None)
                _vbuf = getattr(layer, "_tq_v_dequant_buf", None)
                if _kbuf is None or _kbuf.shape[2] < alloc_len:
                    _kbuf = torch.empty(
                        (1, Hk, block_table.shape[1] * block_size, D),
                        dtype=torch.float16, device=device,
                    )
                    layer._tq_k_dequant_buf = _kbuf
                if _vbuf is None or _vbuf.shape[2] < alloc_len:
                    _vbuf = torch.empty(
                        (1, Hk, block_table.shape[1] * block_size, D),
                        dtype=torch.float16, device=device,
                    )
                    layer._tq_v_dequant_buf = _vbuf
                k_buf = _kbuf
                v_buf = _vbuf
        else:
            _kbuf = getattr(layer, "_tq_k_dequant_buf", None)
            _vbuf = getattr(layer, "_tq_v_dequant_buf", None)
            if _kbuf is None or _kbuf.shape[2] < alloc_len:
                _kbuf = torch.empty(
                    (1, Hk, block_table.shape[1] * block_size, D),
                    dtype=torch.float16, device=device,
                )
                layer._tq_k_dequant_buf = _kbuf
            if _vbuf is None or _vbuf.shape[2] < alloc_len:
                _vbuf = torch.empty(
                    (1, Hk, block_table.shape[1] * block_size, D),
                    dtype=torch.float16, device=device,
                )
                layer._tq_v_dequant_buf = _vbuf
            k_buf = _kbuf
            v_buf = _vbuf
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
        qdtype = query.dtype
        _kfull_cap = block_table.shape[1] * block_size
        _kfull_buf: torch.Tensor | None = getattr(layer, "_tq_kfull_buf", None)
        if _kfull_buf is None or _kfull_buf.shape[0] < _kfull_cap or _kfull_buf.dtype != qdtype:
            _kfull_buf = torch.empty(_kfull_cap, Hk, D, dtype=qdtype, device=device)
            layer._tq_kfull_buf = _kfull_buf
        _vfull_buf: torch.Tensor | None = getattr(layer, "_tq_vfull_buf", None)
        if _vfull_buf is None or _vfull_buf.shape[0] < _kfull_cap or _vfull_buf.dtype != qdtype:
            _vfull_buf = torch.empty(_kfull_cap, Hk, D, dtype=qdtype, device=device)
            layer._tq_vfull_buf = _vfull_buf
        k_full = _kfull_buf[:seq_len]
        v_full = _vfull_buf[:seq_len]
        k_full[:cached_len] = k_cached_trim.to(qdtype)
        k_full[cached_len:] = key_chunk
        v_full[:cached_len] = v_cached_trim.to(qdtype)
        v_full[cached_len:] = val_chunk

        # Attention: q_len queries attending to seq_len K/V with causal mask
        if _HAS_FLASH_ATTN:
            cu_seqlens_q = torch.tensor([0, q_len], device=device, dtype=torch.int32)
            cu_seqlens_k = torch.tensor([0, seq_len], device=device, dtype=torch.int32)
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
        from vllm.v1.attention.ops.fp4_g32.triton_decode import (
            fp4_g32_full_dequant_kv,
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
        fp4_g32_full_dequant_kv(
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
        if _HAS_FLASH_ATTN:
            cu_q = torch.tensor([0, q_len], dtype=torch.int32, device=device)
            cu_k = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
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
                return _fp4_g32_unified(
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
                    output=output_buf[:B] if output_buf is not None else None,
                )

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
            _gqa = self.num_kv_groups
            v4_gqa_ok = (_gqa in (8, 16)) or (
                _gqa == 6 and _flydsl_v4_gqa6_available()
            )
            v4_eligible = (
                not self.tq_config.key_fp8
                and self.tq_config.key_mse_bits == 4
                and self.tq_config.effective_value_quant_bits == 4
                and self.head_size == 128
                and v4_gqa_ok
                and self.sinks is None
            )
            if v4_eligible:
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
                    mid_o_buf=mid_o_buf,
                    output_buf=output_buf,
                    lse_buf=lse_buf,
                    buf_holder=layer,
                    max_num_kv_splits=self.max_num_kv_splits,
                    sinks=self.sinks,
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
                result = triton_turboquant_decode_attention_v3(
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
                )
        elif _USE_TQ_V3:
            result = triton_turboquant_decode_attention_v3(
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
        return result
