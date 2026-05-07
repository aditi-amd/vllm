# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental full-stack TurboQuant fusion implementation.

This implementation keeps the local metadata builder / scheduling glue, but
swaps the core TurboQuant contract to the SoA store + unified
attention path. The experimental switch is opt-in and does not perturb the
existing production backend.
"""

from __future__ import annotations

import functools
import json
import math
import os
from typing import Any

import torch
import torch.nn.functional as F

from vllm.triton_utils import triton
from vllm.v1.attention.backends.fa_utils import (
    is_flash_attn_varlen_func_available,
)
from vllm.v1.attention.backends.turboquant_attn import (
    TurboQuantAttentionImpl as LegacyTurboQuantAttentionImpl,
)

from .external_ops import (
    _tq_full_dequant_kv,
    _use_fp8_e4b15,
    triton_turboquant_decode_attention_v3,
    triton_turboquant_store,
)

_HAS_FLASH_ATTN = is_flash_attn_varlen_func_available()
if _HAS_FLASH_ATTN:
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

_CONTINUATION_DECODE_THRESHOLD = 64


@functools.cache
def _load_campaign_config(config_path: str) -> dict[str, Any]:
    if not config_path:
        return {}
    try:
        with open(config_path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _campaign_config() -> dict[str, Any]:
    return _load_campaign_config(os.environ.get("VLLM_TQ_SOA_FUSION_CAMPAIGN_CONFIG", ""))


def _campaign_option(name: str, default: Any) -> Any:
    return _campaign_config().get(name, default)


def _effective_max_num_kv_splits(default_value: int) -> int:
    cap = _campaign_option("max_num_kv_splits_cap", 0)
    try:
        cap_int = int(cap)
    except (TypeError, ValueError):
        cap_int = 0
    if cap_int > 0:
        return min(default_value, cap_int)
    return default_value


def _build_hadamard(d: int, device_str: str) -> torch.Tensor:
    """Orthonormal Hadamard matrix cached per logical device."""
    return _build_hadamard_cached(d, str(torch.device(device_str)))


@functools.cache
def _build_hadamard_cached(d: int, device_str: str) -> torch.Tensor:
    H = torch.tensor([[1.0]])
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / math.sqrt(d)).to(torch.device(device_str))


class FusionTurboQuantAttentionImpl(LegacyTurboQuantAttentionImpl):
    """Local backend wrapper over the SoA/unified TurboQuant contract."""

    def _ensure_on_device(self, layer, device):
        """Materialize v3-compatible cached tensors on the target device."""
        centroids = layer._tq_centroids
        if centroids.device != device or centroids.dtype != torch.float32:
            layer._tq_centroids = centroids.to(device=device, dtype=torch.float32)

        if getattr(layer, "_tq_fusion_cached", False):
            return

        H = _build_hadamard(self.head_size, str(device)).contiguous()
        layer._tq_PiT = H
        layer._tq_Pi = H

        c_sorted, _ = layer._tq_centroids.sort()
        layer._tq_midpoints = (c_sorted[:-1] + c_sorted[1:]) / 2
        layer._tq_centroids_f32 = layer._tq_centroids.contiguous()
        layer._tq_PiT_f32 = H.float().contiguous()
        layer._tq_Pi_f32 = H.float().contiguous()

        if not hasattr(layer, "_tq_mid_o_buf"):
            layer._tq_mid_o_buf = None
        if not hasattr(layer, "_tq_output_buf"):
            layer._tq_output_buf = None
        if not hasattr(layer, "_tq_lse_buf"):
            layer._tq_lse_buf = None
        if not hasattr(layer, "_tq_q_rot_buf"):
            layer._tq_q_rot_buf = None
        if not hasattr(layer, "_tq_k_dequant_buf"):
            layer._tq_k_dequant_buf = None
            layer._tq_v_dequant_buf = None

        layer._tq_fusion_cached = True
        layer._tq_cached = True

    def _store_kv(
        self,
        key: torch.Tensor,  # (N, Hk, D)
        value: torch.Tensor,  # (N, Hk, D)
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        centroids: torch.Tensor,
        layer: Any | None = None,
    ) -> None:
        assert layer is not None, "Fusion store path expects the attention layer."
        triton_turboquant_store(
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
            centroids=centroids,
            norm_correction=self.tq_config.norm_correction,
        )

    def _prefill_attention(
        self,
        query: torch.Tensor,  # (N, Hq, D)
        key: torch.Tensor,  # (N, Hk, D)
        value: torch.Tensor,  # (N, Hk, D)
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot_size)
        attn_metadata,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
        PiT: torch.Tensor | None = None,
        layer: Any = None,
    ) -> torch.Tensor:
        N, Hq, D = query.shape

        if _HAS_FLASH_ATTN and attn_metadata.max_query_len == attn_metadata.max_seq_len:
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

        Hk = key.shape[1]
        use_gqa = Hk < Hq
        query_start_loc = attn_metadata.query_start_loc
        num_reqs = query_start_loc.shape[0] - 1

        output = torch.zeros(N, Hq, D, device=query.device, dtype=query.dtype)
        qsl = query_start_loc.tolist()
        seq_lens_list = attn_metadata.seq_lens.tolist()
        cu_seqlens_single = torch.zeros(2, device=query.device, dtype=torch.int32)

        decode_threshold = int(
            _campaign_option("short_decode_threshold", _CONTINUATION_DECODE_THRESHOLD)
        )
        max_num_kv_splits = _effective_max_num_kv_splits(self.max_num_kv_splits)

        for i in range(num_reqs):
            q_start = qsl[i]
            q_end = qsl[i + 1]
            q_len = q_end - q_start
            if q_len <= 0:
                continue

            seq_len = seq_lens_list[i]
            q_seq = query[q_start:q_end]
            k_seq = key[q_start:q_end]
            v_seq = value[q_start:q_end]

            if q_len == seq_len:
                if _HAS_FLASH_ATTN:
                    cu_seqlens_single[1] = q_len
                    out = flash_attn_varlen_func(
                        q=q_seq,
                        k=k_seq,
                        v=v_seq,
                        cu_seqlens_q=cu_seqlens_single,
                        cu_seqlens_k=cu_seqlens_single,
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
                continue

            cached_len = seq_len - q_len
            if q_len <= decode_threshold:
                synth_seq_lens = torch.arange(
                    cached_len + 1,
                    seq_len + 1,
                    device=query.device,
                    dtype=attn_metadata.seq_lens.dtype,
                )
                synth_bt = attn_metadata.block_table[i : i + 1].expand(q_len, -1)
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
                    value_quant_bits=self.tq_config.effective_value_quant_bits,
                    value_packed_size=self.tq_config.value_packed_size,
                    max_seq_len=int(seq_len),
                    key_fp8=self.tq_config.key_fp8,
                    norm_correction=self.tq_config.norm_correction,
                    PiT=PiT,
                    buf_holder=layer,
                    max_num_kv_splits=max_num_kv_splits,
                )
            else:
                _centroids_f32 = (
                    getattr(layer, "_tq_centroids_f32", None) if layer is not None else None
                )
                _PiT_f32 = (
                    getattr(layer, "_tq_PiT_f32", None) if layer is not None else None
                )
                _Pi_f32 = (
                    getattr(layer, "_tq_Pi_f32", None) if layer is not None else None
                )
                out = self._continuation_prefill(
                    query=q_seq,
                    key_chunk=k_seq,
                    val_chunk=v_seq,
                    kv_cache=kv_cache,
                    block_table=attn_metadata.block_table[i : i + 1],
                    cached_len=cached_len,
                    seq_len=seq_len,
                    Pi=Pi,
                    centroids=centroids,
                    PiT=PiT,
                    centroids_f32=_centroids_f32,
                    PiT_f32=_PiT_f32,
                    Pi_f32=_Pi_f32,
                    layer=layer,
                )
            output[q_start:q_end] = out.to(query.dtype)

        return output

    def _continuation_prefill(
        self,
        query: torch.Tensor,  # (q_len, Hq, D)
        key_chunk: torch.Tensor,  # (q_len, Hk, D)
        val_chunk: torch.Tensor,  # (q_len, Hk, D)
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot_size)
        block_table: torch.Tensor,  # (1, max_num_blocks)
        cached_len: int,
        seq_len: int,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
        PiT: torch.Tensor | None = None,
        centroids_f32: torch.Tensor | None = None,
        PiT_f32: torch.Tensor | None = None,
        Pi_f32: torch.Tensor | None = None,
        layer: Any = None,
    ) -> torch.Tensor:
        """Continuation chunk path with SoA full-dequant and rotated reuse.

        The cached keys are already stored in rotated space by the SoA store.
        For MSE-key paths, we keep the local optimization that rotates the small
        continuation chunk into the same space instead of inverse-rotating the
        full cached history.
        """

        q_len, Hq, D = query.shape
        Hk = key_chunk.shape[1]
        device = query.device
        block_size = kv_cache.shape[1]
        BLOCK_D = triton.next_power_of_2(D)

        mse_bytes = self._mse_bytes
        val_data_bytes = self._val_data_bytes
        key_fp8 = self.tq_config.key_fp8
        qdtype = query.dtype

        _centroids_f32 = (
            centroids_f32 if centroids_f32 is not None else centroids.to(torch.float32)
        )
        _PiT_f32 = PiT_f32 if PiT_f32 is not None else (
            PiT.to(torch.float32) if PiT is not None else None
        )
        _Pi_f32 = Pi_f32 if Pi_f32 is not None else Pi.to(torch.float32)

        alloc_len = math.ceil(cached_len / block_size) * block_size
        buf_shape = (1, Hk, alloc_len, D)
        if layer is not None:
            k_buf = getattr(layer, "_tq_k_dequant_buf", None)
            v_buf = getattr(layer, "_tq_v_dequant_buf", None)
        else:
            k_buf = None
            v_buf = None

        if k_buf is None or v_buf is None or k_buf.shape[2] < alloc_len:
            k_buf = torch.empty(buf_shape, dtype=torch.float16, device=device)
            v_buf = torch.empty(buf_shape, dtype=torch.float16, device=device)
            if layer is not None:
                layer._tq_k_dequant_buf = k_buf
                layer._tq_v_dequant_buf = v_buf

        k_cached = k_buf[:, :, :alloc_len, :].zero_()
        v_cached = v_buf[:, :, :alloc_len, :].zero_()

        key_data_bytes = D if key_fp8 else mse_bytes
        data_bytes_per_slot = key_data_bytes + val_data_bytes
        meta_region_offset = block_size * Hk * data_bytes_per_slot
        num_soa_fields = 2 if key_fp8 else 3
        soa_k_norm = 0
        soa_v_scale = 0 if key_fp8 else 1
        soa_v_zero = 1 if key_fp8 else 2
        kv_cache_u16 = kv_cache.view(torch.uint16)

        grid = (alloc_len, Hk)
        _tq_full_dequant_kv[grid](
            kv_cache,
            kv_cache_u16,
            block_table,
            _centroids_f32,
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

        continuation_mode = str(_campaign_option("continuation_mode", "rotated_cache"))
        use_rotated = (
            continuation_mode != "full_dequant"
            and (not key_fp8)
            and (_PiT_f32 is not None)
        )
        if use_rotated:
            k_cached_rot = (
                k_cached[0, :, :cached_len, :]
                .transpose(0, 1)
                .contiguous()
                .to(qdtype)
            )
            k_chunk_rot = (
                key_chunk.reshape(-1, D).float() @ _PiT_f32
            ).to(qdtype).reshape(q_len, Hk, D)
            q_rot = (
                query.reshape(-1, D).float() @ _PiT_f32
            ).to(qdtype).reshape(q_len, Hq, D)
            k_full = torch.cat([k_cached_rot, k_chunk_rot], dim=0)
            v_full = torch.cat(
                [
                    v_cached[0, :, :cached_len, :].transpose(0, 1).contiguous().to(qdtype),
                    val_chunk,
                ],
                dim=0,
            )
            q_attn = q_rot
        else:
            if not key_fp8:
                k_flat = k_cached[0, :, :cached_len, :].reshape(-1, D).float()
                k_flat = k_flat @ _Pi_f32
                k_cached_trim = (
                    k_flat.to(qdtype).reshape(Hk, cached_len, D).transpose(0, 1)
                )
            else:
                k_cached_trim = (
                    k_cached[0, :, :cached_len, :].transpose(0, 1).contiguous().to(qdtype)
                )

            v_cached_trim = (
                v_cached[0, :, :cached_len, :].transpose(0, 1).contiguous().to(qdtype)
            )
            k_full = torch.cat([k_cached_trim, key_chunk], dim=0)
            v_full = torch.cat([v_cached_trim, val_chunk], dim=0)
            q_attn = query

        if _HAS_FLASH_ATTN:
            cu_seqlens_q = torch.tensor([0, q_len], device=device, dtype=torch.int32)
            cu_seqlens_k = torch.tensor([0, seq_len], device=device, dtype=torch.int32)
            return flash_attn_varlen_func(
                q=q_attn,
                k=k_full,
                v=v_full,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=q_len,
                max_seqlen_k=seq_len,
                softmax_scale=self.scale,
                causal=True,
            )

        q_t = q_attn.transpose(0, 1).unsqueeze(0)
        k_t = k_full.transpose(0, 1).unsqueeze(0)
        v_t = v_full.transpose(0, 1).unsqueeze(0)
        q_pos = torch.arange(q_len, device=device).unsqueeze(1) + cached_len
        k_pos = torch.arange(seq_len, device=device).unsqueeze(0)
        mask = k_pos <= q_pos
        out = F.scaled_dot_product_attention(
            q_t,
            k_t,
            v_t,
            attn_mask=mask,
            scale=self.scale,
            enable_gqa=(Hk < Hq),
        )
        return out[0].transpose(0, 1)

    def _decode_attention(
        self,
        query: torch.Tensor,  # (B, Hq, D)
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot_size)
        attn_metadata,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
        PiT: torch.Tensor | None = None,
        layer: Any = None,
    ) -> torch.Tensor:
        output_buf = None
        output_buf_policy = str(_campaign_option("output_buf_policy", "reuse_if_match"))
        if layer is not None and output_buf_policy != "disable":
            maybe_output_buf = getattr(layer, "_tq_output_buf", None)
            if maybe_output_buf is not None and maybe_output_buf.dtype == query.dtype:
                output_buf = maybe_output_buf

        return triton_turboquant_decode_attention_v3(
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
            key_fp8=self.tq_config.key_fp8,
            norm_correction=self.tq_config.norm_correction,
            PiT=PiT,
            max_seq_len=int(attn_metadata.max_seq_len),
            output_buf=output_buf,
            buf_holder=layer,
            max_num_kv_splits=_effective_max_num_kv_splits(self.max_num_kv_splits),
        )
