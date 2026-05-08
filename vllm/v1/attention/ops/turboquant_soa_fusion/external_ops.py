"""Bridge to the SoA/unified TurboQuant kernels.

All Python/Triton SoA kernels (store, decode, unified attention) are now
self-contained within this package. No external checkout is required.

The env override ``VLLM_TQ_SOA_FUSION_SOURCE_ROOT`` is retained for
development convenience but is no longer the default load path.
"""

from __future__ import annotations

import ctypes
import math
import os
import warnings
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import torch

from vllm.platforms import current_platform

_WARNED_HIP_V3_SCALAR_KEYS: set[str] = set()


def _warn_hip_v3_scalar_once(key: str, message: str) -> None:
    if key in _WARNED_HIP_V3_SCALAR_KEYS:
        return
    _WARNED_HIP_V3_SCALAR_KEYS.add(key)
    warnings.warn(message, RuntimeWarning, stacklevel=2)


def _load_module_from_external(module_name: str, file_name: str) -> ModuleType:
    """Load a module from an external source root (development override)."""
    import importlib.util
    import sys

    source_root = os.environ.get("VLLM_TQ_SOA_FUSION_SOURCE_ROOT")
    if not source_root:
        raise ImportError("VLLM_TQ_SOA_FUSION_SOURCE_ROOT not set")
    source_path = Path(source_root) / file_name
    if not source_path.exists():
        raise ImportError(f"External source file missing: {source_path}")

    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {source_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache
def _decode_module() -> ModuleType:
    if os.environ.get("VLLM_TQ_SOA_FUSION_SOURCE_ROOT"):
        return _load_module_from_external(
            "vllm.v1.attention.ops.turboquant_soa_fusion._external_decode",
            "triton_turboquant_decode.py",
        )
    # Default: use the self-contained copy within this package
    from . import triton_turboquant_decode as mod
    return mod


@lru_cache
def _store_module() -> ModuleType:
    if os.environ.get("VLLM_TQ_SOA_FUSION_SOURCE_ROOT"):
        return _load_module_from_external(
            "vllm.v1.attention.ops.turboquant_soa_fusion._external_store",
            "triton_turboquant_store.py",
        )
    from . import triton_turboquant_store as mod
    return mod


@lru_cache
def _unified_module() -> ModuleType:
    if os.environ.get("VLLM_TQ_SOA_FUSION_SOURCE_ROOT"):
        return _load_module_from_external(
            "vllm.v1.attention.ops.turboquant_soa_fusion._external_unified_attention",
            "triton_turboquant_unified_attention.py",
        )
    from . import triton_turboquant_unified_attention as mod
    return mod


@lru_cache
def _load_hip_v3_scalar():
    if os.environ.get("VLLM_TQ_SOA_FUSION_DECODE_SCALAR", "0") != "1":
        return None
    if os.environ.get("TQ_DISABLE_HIP_SO", "0") == "1":
        _warn_hip_v3_scalar_once(
            "disabled",
            "TurboQuant SoA HIP scalar decode is disabled by "
            "TQ_DISABLE_HIP_SO=1; falling back to Triton v3.",
        )
        return None
    if not current_platform.is_rocm():
        return None

    so_path = Path(
        os.environ.get(
            "VLLM_TQ_SOA_FUSION_SCALAR_SO_PATH",
            str(Path(__file__).with_name("hip_v3_scalar.so")),
        )
    )
    if not so_path.exists():
        _warn_hip_v3_scalar_once(
            "missing",
            f"TurboQuant SoA HIP scalar decode library is missing: {so_path}; "
            "falling back to Triton v3.",
        )
        return None

    try:
        lib = ctypes.CDLL(str(so_path))
        fn = lib.launch_tq_v3_scalar_decode
        fn.argtypes = (
            [ctypes.c_void_p] * 7
            + [ctypes.c_int] * 13
            + [ctypes.c_float]
            + [ctypes.c_int] * 4
            + [ctypes.c_void_p]
        )
        fn.restype = None
        return fn
    except Exception as exc:
        _warn_hip_v3_scalar_once(
            "load-failed",
            f"Failed to load TurboQuant SoA HIP scalar decode: {exc}; "
            "falling back to Triton v3.",
        )
        return None


@lru_cache
def _load_hip_v3_mfma_qk():
    if os.environ.get("VLLM_TQ_SOA_FUSION_DECODE_MFMA_QK", "0") != "1":
        return None
    if os.environ.get("TQ_DISABLE_HIP_SO", "0") == "1":
        _warn_hip_v3_scalar_once(
            "mfma-disabled",
            "TurboQuant SoA HIP MFMA-QK decode is disabled by "
            "TQ_DISABLE_HIP_SO=1; falling back to scalar/Triton v3.",
        )
        return None
    if not current_platform.is_rocm():
        return None

    so_path = Path(
        os.environ.get(
            "VLLM_TQ_SOA_FUSION_MFMA_QK_SO_PATH",
            str(Path(__file__).with_name("hip_v3_mfma_qk.so")),
        )
    )
    if not so_path.exists():
        _warn_hip_v3_scalar_once(
            "mfma-missing",
            f"TurboQuant SoA HIP MFMA-QK decode library is missing: {so_path}; "
            "falling back to scalar/Triton v3.",
        )
        return None

    try:
        lib = ctypes.CDLL(str(so_path))
        fn = lib.launch_tq_v3_mfma_qk_decode
        fn.argtypes = (
            [ctypes.c_void_p] * 7
            + [ctypes.c_int] * 13
            + [ctypes.c_float]
            + [ctypes.c_int] * 3
            + [ctypes.c_void_p]
        )
        fn.restype = None
        return fn
    except Exception as exc:
        _warn_hip_v3_scalar_once(
            "mfma-load-failed",
            f"Failed to load TurboQuant SoA HIP MFMA-QK decode: {exc}; "
            "falling back to scalar/Triton v3.",
        )
        return None


@lru_cache
def _load_hip_v3_flash_tq():
    if os.environ.get("VLLM_TQ_SOA_FUSION_DECODE_FLASH_TQ", "0") != "1":
        return None
    if os.environ.get("TQ_DISABLE_HIP_SO", "0") == "1":
        _warn_hip_v3_scalar_once(
            "flash-tq-disabled",
            "TurboQuant SoA HIP FlashTQ decode is disabled by "
            "TQ_DISABLE_HIP_SO=1; falling back to MFMA/scalar/Triton v3.",
        )
        return None
    if not current_platform.is_rocm():
        return None

    so_path = Path(
        os.environ.get(
            "VLLM_TQ_SOA_FUSION_FLASH_TQ_SO_PATH",
            str(Path(__file__).with_name("hip_v3_flash_tq.so")),
        )
    )
    if not so_path.exists():
        _warn_hip_v3_scalar_once(
            "flash-tq-missing",
            f"TurboQuant SoA HIP FlashTQ decode library is missing: {so_path}; "
            "falling back to MFMA/scalar/Triton v3.",
        )
        return None

    try:
        lib = ctypes.CDLL(str(so_path))
        fn = lib.launch_tq_v3_flash_tq_decode
        fn.argtypes = (
            [ctypes.c_void_p] * 7
            + [ctypes.c_int] * 13
            + [ctypes.c_float]
            + [ctypes.c_int] * 3
            + [ctypes.c_void_p]
        )
        fn.restype = None
        return fn
    except Exception as exc:
        _warn_hip_v3_scalar_once(
            "flash-tq-load-failed",
            f"Failed to load TurboQuant SoA HIP FlashTQ decode: {exc}; "
            "falling back to MFMA/scalar/Triton v3.",
        )
        return None


@lru_cache
def _load_soa_bf16q_pv_mfma_from_path(so_path: str):
    try:
        lib = ctypes.CDLL(so_path)
        fn = lib.launch_tq_soa_bf16q_pv_mfma_decode
        fn.argtypes = (
            [ctypes.c_void_p] * 7
            + [ctypes.c_int] * 13
            + [ctypes.c_float]
            + [ctypes.c_int] * 3
            + [ctypes.c_void_p]
        )
        fn.restype = None
        return fn
    except Exception as exc:
        _warn_hip_v3_scalar_once(
            "bf16q-pv-mfma-load-failed",
            f"Failed to load TurboQuant SoA bf16Q/PV-MFMA decode: {exc}; "
            "falling back to FlashTQ/MFMA/scalar/Triton v3.",
        )
        return None


def _load_soa_bf16q_pv_mfma_gqa6():
    """Load the gqa=6 variant of the bf16Q/PV MFMA decode kernel.

    Compiled from soa_bf16q_pv_mfma_decode_gqa6.hip with:
      KV_GROUP_SIZE=6, THREADS=192 (=32*6), WAVES=3 (=192/64)
    All other constants (BLOCK_N, HEAD_DIM, DIMS_PER_THREAD, etc.) unchanged.
    """
    if os.environ.get("VLLM_TQ_SOA_FUSION_DECODE_BF16Q_PV_MFMA", "0") != "1":
        return None
    if os.environ.get("TQ_DISABLE_HIP_SO", "0") == "1":
        return None
    if not current_platform.is_rocm():
        return None
    so_path = Path(
        os.environ.get(
            "VLLM_TQ_SOA_FUSION_BF16Q_PV_MFMA_GQA6_SO_PATH",
            str(Path(__file__).with_name("soa_bf16q_pv_mfma_decode_gqa6.so")),
        )
    )
    if not so_path.exists():
        return None
    return _load_soa_bf16q_pv_mfma_from_path(str(so_path))


def _load_soa_bf16q_pv_mfma():
    if os.environ.get("VLLM_TQ_SOA_FUSION_DECODE_BF16Q_PV_MFMA", "0") != "1":
        return None
    if os.environ.get("TQ_DISABLE_HIP_SO", "0") == "1":
        _warn_hip_v3_scalar_once(
            "bf16q-pv-mfma-disabled",
            "TurboQuant SoA bf16Q/PV-MFMA decode is disabled by "
            "TQ_DISABLE_HIP_SO=1; falling back to FlashTQ/MFMA/scalar/Triton v3.",
        )
        return None
    if not current_platform.is_rocm():
        return None

    so_path = Path(
        os.environ.get(
            "VLLM_TQ_SOA_FUSION_BF16Q_PV_MFMA_SO_PATH",
            str(Path(__file__).with_name("soa_bf16q_pv_mfma_decode.so")),
        )
    )
    if not so_path.exists():
        _warn_hip_v3_scalar_once(
            "bf16q-pv-mfma-missing",
            f"TurboQuant SoA bf16Q/PV-MFMA decode library is missing: "
            f"{so_path}; falling back to FlashTQ/MFMA/scalar/Triton v3.",
        )
        return None
    return _load_soa_bf16q_pv_mfma_from_path(str(so_path))


# ──────────────────────────────────────────────────────────────────────────────
# Fused SoA store kernel  (tq_store_fused_soa.so)
# Gate: VLLM_TQ_SOA_FUSION_STORE=1
# Fuses cast+norm+GEMV+bucketize+value_quant into one HIP kernel.
# Constraints: D=128, mse_bits=4, value_quant_bits=4, not FP8 key.
# ──────────────────────────────────────────────────────────────────────────────

_SOA_STORE_LIB = None
_SOA_STORE_FN = None
_SOA_STORE_LOADED = False


def _load_soa_fused_store():
    global _SOA_STORE_LIB, _SOA_STORE_FN, _SOA_STORE_LOADED
    if _SOA_STORE_LOADED:
        return _SOA_STORE_FN
    _SOA_STORE_LOADED = True
    if os.environ.get("VLLM_TQ_SOA_FUSION_STORE", "0") != "1":
        return None
    if not current_platform.is_rocm():
        return None
    so_path = Path(__file__).with_name("tq_store_fused_soa.so")
    if not so_path.exists():
        _warn_hip_v3_scalar_once(
            "soa-fused-store-missing",
            f"TurboQuant SoA fused store .so missing: {so_path}; "
            "falling back to Triton store.",
        )
        return None
    try:
        lib = ctypes.CDLL(str(so_path))
        fn = lib.launch_tq_store_fused_soa
        fn.argtypes = [
            ctypes.c_void_p,  # key (bf16/fp16) [NH, D]
            ctypes.c_void_p,  # value (bf16/fp16) [NH, D]
            ctypes.c_void_p,  # PiT float32 [D, D]
            ctypes.c_void_p,  # centroids float32 [16]
            ctypes.c_void_p,  # midpoints float32 [15]
            ctypes.c_void_p,  # kv_cache uint8
            ctypes.c_void_p,  # kv_cache_u16 uint16 alias
            ctypes.c_void_p,  # slot_map int64 [N]
            ctypes.c_int,     # stride_cache_block: kv_cache.stride(0) bytes per block
            ctypes.c_int,     # stride_cache_head: kv_cache.stride(2) bytes per slot
            ctypes.c_int,     # H (num_kv_heads)
            ctypes.c_int,     # block_size
            ctypes.c_int,     # NH
            ctypes.c_int,     # kv_dtype: 0=bf16, 1=fp16
            ctypes.c_void_p,  # stream
        ]
        fn.restype = None
        _SOA_STORE_LIB = lib
        _SOA_STORE_FN = fn
        return fn
    except Exception as exc:
        _warn_hip_v3_scalar_once(
            "soa-fused-store-load-failed",
            f"Failed to load TurboQuant SoA fused store: {exc}; "
            "falling back to Triton store.",
        )
        return None


def _soa_fused_store_safe(
    key: torch.Tensor,
    kv_cache: torch.Tensor,
    mse_bits: int,
    value_quant_bits: int,
    key_fp8: bool,
) -> bool:
    return (
        current_platform.is_rocm()
        and key.shape[-1] == 128
        and mse_bits == 4
        and value_quant_bits == 4
        and not key_fp8
        and key.dtype in (torch.bfloat16, torch.float16)
        and kv_cache.dtype == torch.uint8
    )


# ──────────────────────────────────────────────────────────────────────────────
# WHT butterfly kernel  (wht_butterfly.so)
# Gate: VLLM_TQ_SOA_FUSION_WHT_BUTTERFLY=1
# Replaces q @ PiT GEMV with an in-register Hadamard butterfly (5x faster).
# Constraints: D=128, bf16 input/output only.
# ──────────────────────────────────────────────────────────────────────────────

_WHT_BF_LIB = None
_WHT_BF_FN = None
_WHT_BF_LOADED = False


def _load_wht_butterfly():
    global _WHT_BF_LIB, _WHT_BF_FN, _WHT_BF_LOADED
    if _WHT_BF_LOADED:
        return _WHT_BF_FN
    _WHT_BF_LOADED = True
    if os.environ.get("VLLM_TQ_SOA_FUSION_WHT_BUTTERFLY", "0") != "1":
        return None
    if not current_platform.is_rocm():
        return None
    so_path = Path(__file__).with_name("wht_butterfly.so")
    if not so_path.exists():
        _warn_hip_v3_scalar_once(
            "wht-butterfly-missing",
            f"TurboQuant WHT butterfly .so missing: {so_path}; "
            "falling back to FP32 GEMV q_rot.",
        )
        return None
    try:
        lib = ctypes.CDLL(str(so_path))
        fn = lib.launch_wht_butterfly
        fn.argtypes = [
            ctypes.c_void_p,  # q  bf16 [M, D]
            ctypes.c_void_p,  # q_rot bf16 [M, D] (output)
            ctypes.c_void_p,  # signs bf16 [D]
            ctypes.c_int,     # M = B * Hq
            ctypes.c_int,     # D = HEAD_DIM
            ctypes.c_void_p,  # stream
        ]
        fn.restype = None
        _WHT_BF_LIB = lib
        _WHT_BF_FN = fn
        return fn
    except Exception as exc:
        _warn_hip_v3_scalar_once(
            "wht-butterfly-load-failed",
            f"Failed to load TurboQuant WHT butterfly: {exc}; "
            "falling back to FP32 GEMV q_rot.",
        )
        return None


def _apply_wht_butterfly(
    query: torch.Tensor,  # [B, Hq, D] bf16
    signs: torch.Tensor,  # [D] bf16/fp32
) -> torch.Tensor | None:
    """Apply WHT butterfly as drop-in replacement for FP32 GEMV q_rot.

    Returns rotated query [B, Hq, D] bf16, or None if butterfly is unavailable
    or conditions are not met (falls back to caller's GEMV path).
    """
    fn = _load_wht_butterfly()
    if fn is None:
        return None
    if query.dtype != torch.bfloat16 or query.shape[-1] != 128:
        return None
    B, Hq, D = query.shape
    M = B * Hq
    q_flat = query.reshape(M, D).contiguous()
    signs_bf16 = signs.to(torch.bfloat16).contiguous()
    q_rot = torch.empty_like(q_flat)
    stream_ptr = torch.cuda.current_stream(query.device).cuda_stream
    fn(
        ctypes.c_void_p(q_flat.data_ptr()),
        ctypes.c_void_p(q_rot.data_ptr()),
        ctypes.c_void_p(signs_bf16.data_ptr()),
        M,
        D,
        ctypes.c_void_p(stream_ptr),
    )
    return q_rot.reshape(B, Hq, D)


def _dtype_code(dtype: torch.dtype) -> int | None:
    if dtype is torch.bfloat16:
        return 0
    if dtype is torch.float16:
        return 1
    if dtype is torch.float32:
        return 2
    return None


def _hip_v3_scalar_safe(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    centroids: torch.Tensor,
    mse_bits: int,
    value_quant_bits: int,
    key_fp8: bool,
) -> bool:
    return (
        current_platform.is_rocm()
        and query.dim() == 3
        and query.shape[-1] == 128
        and kv_cache.dtype == torch.uint8
        and kv_cache.dim() == 4
        and block_table.dtype == torch.int32
        and seq_lens.dtype == torch.int32
        and centroids.dtype == torch.float32
        and (not key_fp8)
        and mse_bits == 4
        and value_quant_bits == 4
        and _dtype_code(query.dtype) is not None
    )


def _rotated_query_fp32(
    query: torch.Tensor,
    Pi: torch.Tensor,
    PiT: torch.Tensor | None,
) -> torch.Tensor:
    if PiT is None:
        PiT = Pi.T.contiguous()
    PiT_f32 = PiT if PiT.dtype == torch.float32 else PiT.to(torch.float32)
    if not PiT_f32.is_contiguous():
        PiT_f32 = PiT_f32.contiguous()
    return (query.float() @ PiT_f32).contiguous()


def _decode_num_splits(
    seq_lens: torch.Tensor,
    block_size: int,
    max_seq_len: int,
    max_num_kv_splits: int,
) -> int:
    # Prefer the pre-computed max_seq_len (a Python int) to avoid a GPU→CPU
    # sync via .item().  During CUDAGraph capture the stream is in recording
    # mode and .item() would raise hipErrorStreamCaptureUnsupported.
    if max_seq_len > 0:
        max_seq_len_hint = max_seq_len
    elif torch.cuda.is_current_stream_capturing():
        # Fallback: assume long context so we pick the 3-D split path.
        max_seq_len_hint = 2048
    else:
        max_seq_len_hint = int(seq_lens.max().item())
    use_3d = max_seq_len_hint >= 1024 and max_num_kv_splits > 1
    num_splits = max(1, max_num_kv_splits if use_3d else 1)
    max_possible_splits = max(1, math.ceil(max_seq_len_hint / block_size))
    return min(num_splits, max_possible_splits)


def _maybe_hip_v3_mfma_like_decode(
    fn,
    *args,
    q_rot_dtype: torch.dtype | None = None,
    require_query_dtype: torch.dtype | None = None,
    **kwargs,
):
    if fn is None:
        return None

    query = kwargs["query"]
    kv_cache = kwargs["kv_cache"]
    block_table = kwargs["block_table"]
    seq_lens = kwargs["seq_lens"]
    Pi = kwargs["Pi"]
    centroids = kwargs["centroids"]
    scale = float(kwargs["scale"])
    mse_bits = int(kwargs["mse_bits"])
    value_quant_bits = int(kwargs["value_quant_bits"])
    key_fp8 = bool(kwargs.get("key_fp8", False))
    PiT = kwargs.get("PiT", None)
    output_buf = kwargs.get("output_buf", None)
    max_seq_len = int(kwargs.get("max_seq_len", 0) or 0)
    max_num_kv_splits = int(kwargs.get("max_num_kv_splits", 32))
    signs = kwargs.get("signs", None)

    if require_query_dtype is not None and query.dtype != require_query_dtype:
        return None
    if not _hip_v3_scalar_safe(
        query, kv_cache, block_table, seq_lens, centroids, mse_bits,
        value_quant_bits, key_fp8,
    ):
        return None

    B, Hq, _ = query.shape
    Hk = kv_cache.shape[2]
    block_size = kv_cache.shape[1]
    kv_group_size = Hq // Hk
    if kv_group_size not in (6, 8):
        return None

    # Try WHT butterfly (5x faster than FP32 GEMV) when signs are available.
    # Falls back to FP32 GEMV if butterfly is disabled or conditions not met.
    if signs is not None:
        q_rot = _apply_wht_butterfly(query, signs)
    else:
        q_rot = None
    if q_rot is None:
        q_rot = _rotated_query_fp32(query, Pi, PiT)
        if q_rot_dtype is not None:
            q_rot = q_rot.to(q_rot_dtype)
    q_rot = q_rot.contiguous()
    output = output_buf[:B] if output_buf is not None else torch.empty_like(query)
    if not output.is_contiguous():
        output = output.contiguous()
    out_dtype = _dtype_code(output.dtype)
    if out_dtype not in (0, 1):
        return None

    num_splits = _decode_num_splits(
        seq_lens, block_size, max_seq_len, max_num_kv_splits
    )
    mid_o = torch.empty(
        (B, Hq, num_splits, query.shape[-1] + 1),
        dtype=torch.float32,
        device=query.device,
    )

    stream_ptr = torch.cuda.current_stream(query.device).cuda_stream
    fn(
        ctypes.c_void_p(q_rot.data_ptr()),
        ctypes.c_void_p(kv_cache.data_ptr()),
        ctypes.c_void_p(block_table.data_ptr()),
        ctypes.c_void_p(seq_lens.data_ptr()),
        ctypes.c_void_p(centroids.data_ptr()),
        ctypes.c_void_p(mid_o.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        q_rot.stride(0),
        q_rot.stride(1),
        kv_cache.stride(0),
        block_table.stride(0),
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        output.stride(0),
        output.stride(1),
        Hk,
        block_size,
        num_splits,
        kv_group_size,
        scale,
        out_dtype,
        B,
        Hq,
        ctypes.c_void_p(stream_ptr),
    )
    return output


def _maybe_soa_bf16q_pv_mfma_decode(*args, **kwargs):
    # Dispatch to the matching compiled kernel by kv_group_size.
    # KV_GROUP_SIZE is baked into the kernel at compile time, so each gqa
    # value needs its own .so. Unsupported values fall back to Triton v3.
    query = kwargs.get("query")
    kv_cache = kwargs.get("kv_cache")
    if query is not None and kv_cache is not None:
        kv_group_size = query.shape[1] // kv_cache.shape[2]
        if kv_group_size == 6:
            fn = _load_soa_bf16q_pv_mfma_gqa6()
        elif kv_group_size == 8:
            fn = _load_soa_bf16q_pv_mfma()
        else:
            return None  # unsupported gqa — fall back to Triton v3
    else:
        fn = _load_soa_bf16q_pv_mfma()
    return _maybe_hip_v3_mfma_like_decode(
        fn,
        *args,
        q_rot_dtype=torch.bfloat16,
        require_query_dtype=torch.bfloat16,
        **kwargs,
    )


def _maybe_hip_v3_flash_tq_decode(*args, **kwargs):
    return _maybe_hip_v3_mfma_like_decode(
        _load_hip_v3_flash_tq(), *args, **kwargs)


def _maybe_hip_v3_mfma_qk_decode(*args, **kwargs):
    return _maybe_hip_v3_mfma_like_decode(
        _load_hip_v3_mfma_qk(), *args, **kwargs)


def _maybe_hip_v3_scalar_decode(*args, **kwargs):
    fn = _load_hip_v3_scalar()
    if fn is None:
        return None

    query = kwargs["query"]
    kv_cache = kwargs["kv_cache"]
    block_table = kwargs["block_table"]
    seq_lens = kwargs["seq_lens"]
    Pi = kwargs["Pi"]
    centroids = kwargs["centroids"]
    scale = float(kwargs["scale"])
    mse_bits = int(kwargs["mse_bits"])
    value_quant_bits = int(kwargs["value_quant_bits"])
    key_fp8 = bool(kwargs.get("key_fp8", False))
    PiT = kwargs.get("PiT", None)
    output_buf = kwargs.get("output_buf", None)
    max_seq_len = int(kwargs.get("max_seq_len", 0) or 0)
    max_num_kv_splits = int(kwargs.get("max_num_kv_splits", 32))

    if not _hip_v3_scalar_safe(
        query, kv_cache, block_table, seq_lens, centroids, mse_bits,
        value_quant_bits, key_fp8,
    ):
        return None

    # Scalar v3 validates the HIP SoA decode contract first. Fused Q rotation
    # moves into HIP in the following MFMA phase.
    q_rot = _rotated_query_fp32(query, Pi, PiT).to(query.dtype).contiguous()
    output = output_buf[: query.shape[0]] if output_buf is not None else torch.empty_like(query)
    if not output.is_contiguous():
        output = output.contiguous()

    B, Hq, _ = q_rot.shape
    Hk = kv_cache.shape[2]
    block_size = kv_cache.shape[1]
    kv_group_size = Hq // Hk
    num_splits = _decode_num_splits(
        seq_lens, block_size, max_seq_len, max_num_kv_splits
    )

    mid_o = torch.empty(
        (B, Hq, num_splits, query.shape[-1] + 1),
        dtype=torch.float32,
        device=query.device,
    )
    q_dtype = _dtype_code(q_rot.dtype)
    out_dtype = _dtype_code(output.dtype)
    if q_dtype is None or out_dtype is None:
        return None

    stream_ptr = torch.cuda.current_stream(query.device).cuda_stream
    fn(
        ctypes.c_void_p(q_rot.data_ptr()),
        ctypes.c_void_p(kv_cache.data_ptr()),
        ctypes.c_void_p(block_table.data_ptr()),
        ctypes.c_void_p(seq_lens.data_ptr()),
        ctypes.c_void_p(centroids.data_ptr()),
        ctypes.c_void_p(mid_o.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        q_rot.stride(0),
        q_rot.stride(1),
        kv_cache.stride(0),
        block_table.stride(0),
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        output.stride(0),
        output.stride(1),
        Hk,
        block_size,
        num_splits,
        kv_group_size,
        scale,
        q_dtype,
        out_dtype,
        B,
        Hq,
        ctypes.c_void_p(stream_ptr),
    )
    return output


def _use_fp8_e4b15(device: int = 0) -> int:
    return _decode_module()._use_fp8_e4b15(device)


def _get_tq_full_dequant_kv():
    """Lazy accessor for _tq_full_dequant_kv to avoid eager module load."""
    return _decode_module()._tq_full_dequant_kv


# Lazy proxy: accessed as _tq_full_dequant_kv but deferred until first use.
# The underlying object is a Triton JIT kernel invoked via kernel[grid](...),
# so __getitem__ is the primary dispatch path.
class _LazyDequantKV:
    """Proxy that lazily loads _tq_full_dequant_kv on first access."""

    def __init__(self):
        self._cached = None

    def _resolve(self):
        if self._cached is None:
            self._cached = _get_tq_full_dequant_kv()
        return self._cached

    def __getitem__(self, key):
        return self._resolve()[key]

    def __getattr__(self, name):
        return getattr(self._resolve(), name)

    def __call__(self, *args, **kwargs):
        return self._resolve()(*args, **kwargs)


_tq_full_dequant_kv = _LazyDequantKV()


def triton_turboquant_store(*args, **kwargs):
    return _store_module().triton_turboquant_store(*args, **kwargs)


def triton_turboquant_unified_attention(*args, **kwargs):
    return _unified_module().triton_turboquant_unified_attention(*args, **kwargs)


def triton_turboquant_decode_attention_v3(*args, **kwargs):
    hip_out = _maybe_soa_bf16q_pv_mfma_decode(*args, **kwargs)
    if hip_out is not None:
        return hip_out
    hip_out = _maybe_hip_v3_flash_tq_decode(*args, **kwargs)
    if hip_out is not None:
        return hip_out
    hip_out = _maybe_hip_v3_mfma_qk_decode(*args, **kwargs)
    if hip_out is not None:
        return hip_out
    hip_out = _maybe_hip_v3_scalar_decode(*args, **kwargs)
    if hip_out is not None:
        return hip_out
    return _unified_module().triton_turboquant_decode_attention_v3(*args, **kwargs)
