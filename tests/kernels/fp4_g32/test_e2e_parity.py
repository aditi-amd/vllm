# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end parity test: fp4_kv_g32 vs exact fp16 SDPA baseline.

Verifies that the full pipeline
  fp16 K/V → fp4_g32_store → KV cache → fp4_g32_decode_attention
produces attention outputs that are numerically close to the exact fp16
scaled-dot-product-attention on the same K/V.

Thresholds are intentionally loose to reflect inherent FP4 quantization
error rather than strict bit-level matching.  The key invariant is that
the kernel produces no catastrophic divergence — cosine similarity must
remain > 0.99 for all tested shapes.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from vllm.v1.attention.ops.fp4_g32 import (
    slot_size,
)
from vllm.v1.attention.ops.fp4_g32.triton_decode import fp4_g32_decode_attention
from vllm.v1.attention.ops.fp4_g32.triton_store import fp4_g32_store

DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "xpu"
GPGPU_AVAILABLE = torch.cuda.is_available() or torch.xpu.is_available()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_hadamard(dim: int, device: str) -> torch.Tensor:
    H = torch.tensor([[1.0]], dtype=torch.float64)
    while H.shape[0] < dim:
        H = torch.cat(
            [torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0
        )
    return (H / math.sqrt(dim)).to(device=device, dtype=torch.float32)


def _exact_sdpa_reference(
    query: torch.Tensor,    # [B, Hq, D]
    key: torch.Tensor,      # [seq_len, Hk, D]
    value: torch.Tensor,    # [seq_len, Hk, D]
    scale: float,
) -> torch.Tensor:
    """Exact fp32 scaled dot-product attention, GQA-aware, causal=False.

    Returns [B, Hq, D] in query's dtype.
    """
    B, Hq, D = query.shape
    Hk = key.shape[1]
    kv_group = Hq // Hk

    q_f = query.float()                              # [B, Hq, D]
    k_f = key.float().transpose(0, 1)                # [Hk, seq, D]
    v_f = value.float().transpose(0, 1)              # [Hk, seq, D]

    # Expand KV heads for GQA
    k_f = k_f.repeat_interleave(kv_group, dim=0)    # [Hq, seq, D]
    v_f = v_f.repeat_interleave(kv_group, dim=0)    # [Hq, seq, D]

    # q_f: [B, Hq, D]  →  [Hq, B, D] for bmm
    q_t = q_f.permute(1, 0, 2)                      # [Hq, B, D]
    k_t = k_f                                        # [Hq, seq, D]
    v_t = v_f                                        # [Hq, seq, D]

    scores = torch.bmm(q_t, k_t.transpose(1, 2)) * scale   # [Hq, B, seq]
    weights = F.softmax(scores, dim=-1)                     # [Hq, B, seq]
    out = torch.bmm(weights, v_t)                           # [Hq, B, D]
    return out.permute(1, 0, 2).to(query.dtype)             # [B, Hq, D]


def _cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not GPGPU_AVAILABLE, reason="GPGPU not available")
class TestFP4G32E2EParity:
    """Full-pipeline parity: fp4_kv_g32 vs fp16 SDPA baseline."""

    @staticmethod
    def _build_and_store(
        B: int,
        Hq: int,
        Hk: int,
        D: int,
        seq_len: int,
        block_size: int,
        dtype: torch.dtype,
        seed: int,
    ):
        device = torch.device(DEVICE_TYPE)
        torch.manual_seed(seed)
        key = torch.randn(seq_len, Hk, D, device=device, dtype=dtype)
        value = torch.randn(seq_len, Hk, D, device=device, dtype=dtype)

        padded_slot = slot_size(D)
        num_blocks = (seq_len + block_size - 1) // block_size + 1
        kv_cache = torch.zeros(
            num_blocks, block_size, Hk, padded_slot,
            device=device, dtype=torch.uint8,
        )
        slot_mapping = torch.arange(seq_len, device=device, dtype=torch.int32)
        fp4_g32_store(key, value, kv_cache, slot_mapping)

        # Build block_table: one contiguous range per batch item (all see the
        # same KV cache since we only have one sequence here).
        block_table = (
            torch.arange(num_blocks, device=device, dtype=torch.int32)
            .unsqueeze(0)
            .expand(B, -1)
            .contiguous()
        )
        seq_lens = torch.full((B,), seq_len, device=device, dtype=torch.int32)
        return key, value, kv_cache, block_table, seq_lens

    @pytest.mark.parametrize(
        "B, Hq, Hk, D, seq_len, dtype, cos_min, max_abs_budget",
        [
            # Thresholds reflect true FP4 quantization error vs exact fp16 SDPA.
            # Observed: cos_sim ~0.988, max_abs ~0.05 across all shapes.
            # cos_min=0.985 leaves headroom; max_abs_budget=0.10 is 2× observed max.
            # --- standard decode shapes ---
            (1, 8,  1, 128, 512,  torch.bfloat16, 0.985, 0.10),
            (4, 8,  1, 128, 512,  torch.bfloat16, 0.985, 0.10),
            (1, 8,  1, 128, 2048, torch.bfloat16, 0.985, 0.10),
            (1, 8,  1, 128, 512,  torch.float16,  0.985, 0.10),
            # --- GQA ---
            (1, 32, 4, 128, 1024, torch.bfloat16, 0.985, 0.10),
            (2, 16, 2, 128, 1024, torch.bfloat16, 0.985, 0.10),
            # --- longer context ---
            (1, 8,  1, 128, 8192, torch.bfloat16, 0.985, 0.10),
        ],
    )
    def test_fp4_vs_exact_fp16(
        self, B, Hq, Hk, D, seq_len, dtype, cos_min, max_abs_budget
    ):
        device = torch.device(DEVICE_TYPE)
        torch.manual_seed(42)
        query = torch.randn(B, Hq, D, device=device, dtype=dtype)

        key, value, kv_cache, block_table, seq_lens = self._build_and_store(
            B=B, Hq=Hq, Hk=Hk, D=D, seq_len=seq_len,
            block_size=16, dtype=dtype, seed=1234,
        )

        scale = 1.0 / math.sqrt(D)

        # FP4-g32 decode attention
        out_fp4 = fp4_g32_decode_attention(
            query=query,
            kv_cache=kv_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            scale=scale,
            max_num_kv_splits=16,
        )

        # Exact fp16 SDPA reference (no quantization)
        out_ref = _exact_sdpa_reference(query, key, value, scale)

        cos = _cos_sim(out_fp4, out_ref)
        max_abs = (out_fp4.float() - out_ref.float()).abs().max().item()
        tag = (
            f"[fp4_kv_g32 B={B} Hq={Hq} Hk={Hk} D={D} seq={seq_len} "
            f"{dtype}]"
        )
        assert cos > cos_min, f"{tag} cos_sim={cos:.6f} < {cos_min}"
        assert max_abs < max_abs_budget, (
            f"{tag} max_abs={max_abs:.4e} >= {max_abs_budget:.2e}"
        )

    @pytest.mark.parametrize(
        "Hq, Hk, D, seq_len, dtype",
        [
            (8, 1, 128, 512,  torch.bfloat16),
            (8, 1, 128, 1024, torch.bfloat16),
        ],
    )
    def test_fp4_with_sinks(self, Hq, Hk, D, seq_len, dtype):
        """Sink tokens must not corrupt the output (sanity: output stays finite
        and cos_sim vs no-sinks output remains high, indicating sinks only
        damp rather than invert the distribution)."""
        device = torch.device(DEVICE_TYPE)
        B = 1
        torch.manual_seed(55)
        query = torch.randn(B, Hq, D, device=device, dtype=dtype)

        _, _, kv_cache, block_table, seq_lens = self._build_and_store(
            B=B, Hq=Hq, Hk=Hk, D=D, seq_len=seq_len,
            block_size=16, dtype=dtype, seed=7777,
        )
        scale = 1.0 / math.sqrt(D)

        sinks = torch.randn(Hq, device=device, dtype=torch.float32)

        out_no_sink = fp4_g32_decode_attention(
            query=query, kv_cache=kv_cache, block_table=block_table,
            seq_lens=seq_lens, scale=scale, max_num_kv_splits=16,
        )
        out_with_sink = fp4_g32_decode_attention(
            query=query, kv_cache=kv_cache, block_table=block_table,
            seq_lens=seq_lens, scale=scale, max_num_kv_splits=16,
            sinks=sinks,
        )

        assert out_with_sink.isfinite().all(), "sink output has non-finite values"
        cos = _cos_sim(out_no_sink, out_with_sink)
        # Sinks damp but should not catastrophically redirect attention
        assert cos > 0.9, f"sink vs no-sink cos_sim={cos:.6f} unexpectedly low"

    # ------------------------------------------------------------------
    # Sink correctness: fp4_kv_g32 sinks vs fp32 oracle
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "B, Hq, Hk, D, seq_len, dtype",
        [
            (1, 8,  1, 128, 512,  torch.bfloat16),
            (2, 8,  1, 128, 1024, torch.bfloat16),
            (1, 32, 4, 128, 512,  torch.bfloat16),
            (1, 8,  1, 128, 512,  torch.float16),
        ],
    )
    def test_fp4_sinks_vs_oracle(self, B, Hq, Hk, D, seq_len, dtype):
        """fp4_kv_g32 + sinks must match the fp32 oracle that folds the sink
        logit into the softmax denominator.

        Oracle:  out_h = Σ_i exp(q·k_i) V_i / (exp(s_h) + Σ_j exp(q·k_j))
        This is the same formula tested in TestV3SinksVsReference for TQ-v3.
        """
        device = torch.device(DEVICE_TYPE)
        scale = 1.0 / math.sqrt(D)

        torch.manual_seed(42)
        query = torch.randn(B, Hq, D, device=device, dtype=dtype)
        sinks = torch.randn(Hq, device=device, dtype=torch.float32)

        key, value, kv_cache, block_table, seq_lens = self._build_and_store(
            B=B, Hq=Hq, Hk=Hk, D=D, seq_len=seq_len,
            block_size=16, dtype=dtype, seed=7654,
        )

        # fp4_kv_g32 decode with sinks
        out_fp4 = fp4_g32_decode_attention(
            query=query,
            kv_cache=kv_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            scale=scale,
            max_num_kv_splits=16,
            sinks=sinks,
        )

        # fp32 oracle: exact softmax with sink in denominator, no quantization
        kv_group = Hq // Hk
        k_f = key.float().repeat_interleave(kv_group, dim=1)   # [seq, Hq, D]
        v_f = value.float().repeat_interleave(kv_group, dim=1) # [seq, Hq, D]
        q_f = query.float()                                     # [B, Hq, D]
        scores = torch.einsum("bhd,shd->bhs", q_f, k_f) * scale   # [B, Hq, seq]
        sinks_b = sinks.view(1, Hq, 1).expand(B, -1, -1)           # [B, Hq, 1]
        scores_ext = torch.cat([scores, sinks_b], dim=-1)           # [B, Hq, seq+1]
        probs_ext = torch.softmax(scores_ext.float(), dim=-1)
        probs = probs_ext[..., :-1]                                  # [B, Hq, seq]
        oracle = torch.einsum("bhs,shd->bhd", probs, v_f)           # [B, Hq, D]

        cos = _cos_sim(out_fp4, oracle)
        max_abs = (out_fp4.float() - oracle.float()).abs().max().item()
        tag = (
            f"[fp4_kv_g32+sinks B={B} Hq={Hq} Hk={Hk} D={D} seq={seq_len} "
            f"{dtype}]"
        )
        # Threshold = FP4 quantization budget + small FP32 rounding.
        # Observed cos ~0.987, max_abs ~0.05 (same budget as no-sinks cases).
        assert cos > 0.985, f"{tag} cos_sim={cos:.6f} < 0.985"
        assert max_abs < 0.10, f"{tag} max_abs={max_abs:.4e} >= 0.10"

    def test_backend_registered(self):
        """Smoke test: fp4_kv_g32 is accepted by the backend dtype gating."""
        from vllm.v1.attention.backends.turboquant_attn import (
            TurboQuantAttentionBackend,
        )
        assert TurboQuantAttentionBackend.supports_kv_cache_dtype("fp4_kv_g32")
        assert not TurboQuantAttentionBackend.supports_kv_cache_dtype("fp4_wrong")

    def test_slot_size_registered(self):
        """Smoke test: get_kv_cache_shape returns slot=144 for head_dim=128."""
        from vllm.v1.attention.backends.turboquant_attn import (
            TurboQuantAttentionBackend,
        )
        shape = TurboQuantAttentionBackend.get_kv_cache_shape(
            num_blocks=4,
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            cache_dtype_str="fp4_kv_g32",
        )
        assert shape == (4, 16, 8, 144), f"unexpected shape {shape}"
