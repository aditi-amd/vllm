# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the fp8_g32 Triton decode kernel.

Mirrors `tests/kernels/fp4_g32/test_decode_vs_reference.py`.

Tolerances are slightly looser than the fp4_g32 tests because:
1. The E8M0 (power-of-2) scale snap rounds the per-block scale further
   than fp4_g32's fp16 cast — up to a 25 % per-block multiplicative
   error at octave boundaries.
2. Q is round-tripped through FP8 E4M3 (scale = 1), giving Q an
   effective precision of 3 mantissa bits + 4 exponent bits.
"""

from __future__ import annotations

import pytest
import torch

from vllm.v1.attention.ops.fp8_g32.fp8_levels import slot_size
from vllm.v1.attention.ops.fp8_g32.reference import reference_fp8_g32_attention
from vllm.v1.attention.ops.fp8_g32.triton_decode import fp8_g32_decode_attention
from vllm.v1.attention.ops.fp8_g32.triton_store import fp8_g32_store


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA/HIP device"
)


def _build_paged_cache(
    key_BNHD: torch.Tensor,
    value_BNHD: torch.Tensor,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack [B, N, Hk, D] K/V into a paged uint8 fp8_g32 cache."""
    B, N, Hk, D = key_BNHD.shape
    blocks_per_seq = (N + block_size - 1) // block_size
    num_blocks = B * blocks_per_seq + 1

    slot_b = slot_size(D)
    kv_cache = torch.zeros(
        num_blocks, block_size, Hk, slot_b, dtype=torch.uint8, device=device
    )

    block_table = torch.zeros(B, blocks_per_seq, dtype=torch.int32, device=device)
    for bi in range(B):
        for blk in range(blocks_per_seq):
            block_table[bi, blk] = bi * blocks_per_seq + blk

    slot_mapping = torch.full((B * N,), -1, dtype=torch.int64, device=device)
    k_flat = torch.empty(B * N, Hk, D, dtype=key_BNHD.dtype, device=device)
    v_flat = torch.empty_like(k_flat)
    for bi in range(B):
        for ti in range(N):
            global_idx = bi * N + ti
            block_num = int(block_table[bi, ti // block_size].item())
            slot = block_num * block_size + ti % block_size
            slot_mapping[global_idx] = slot
            k_flat[global_idx] = key_BNHD[bi, ti]
            v_flat[global_idx] = value_BNHD[bi, ti]

    fp8_g32_store(k_flat, v_flat, kv_cache, slot_mapping)
    torch.cuda.synchronize()

    return kv_cache, block_table, slot_mapping


def _attention_quality(out_triton: torch.Tensor, out_ref: torch.Tensor) -> dict:
    a = out_triton.float().flatten()
    b = out_ref.float().flatten()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    diff = (a - b).abs()
    return {
        "cos_sim": cos,
        "mean_abs": diff.mean().item(),
        "max_abs": diff.max().item(),
        "norm_ratio": (a.norm() / b.norm()).item(),
    }


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "B,Hq,Hk,N,D",
    [
        (1, 1, 1, 16, 128),
        (1, 4, 2, 128, 128),
        (2, 8, 4, 256, 128),
        (1, 4, 1, 1024, 128),
    ],
)
def test_decode_matches_reference(dtype, B, Hq, Hk, N, D):
    """Triton decode output is close to the reference attention output.

    Looser tolerances than fp4_g32 because UE8M0 + FP8 Q accumulate
    more rounding than fp16 scale + bf16 Q.
    """
    device = torch.device("cuda")
    torch.manual_seed(0x8032)

    block_size = 32
    scale = 1.0 / (D**0.5)

    query = torch.randn(B, Hq, D, dtype=dtype, device=device)
    key = torch.randn(B, N, Hk, D, dtype=dtype, device=device)
    value = torch.randn(B, N, Hk, D, dtype=dtype, device=device)

    kv_cache, block_table, _ = _build_paged_cache(key, value, block_size, device)
    seq_lens = torch.full((B,), N, dtype=torch.int32, device=device)

    out_triton = fp8_g32_decode_attention(
        query=query,
        kv_cache=kv_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        scale=scale,
        max_num_kv_splits=8,
    )
    torch.cuda.synchronize()

    out_ref = reference_fp8_g32_attention(
        query=query, key=key, value=value, scale=scale
    )

    q = _attention_quality(out_triton, out_ref)
    print(
        f"\n[B={B} Hq={Hq} Hk={Hk} N={N} D={D} {dtype}] "
        f"cos_sim={q['cos_sim']:.6f}  mean={q['mean_abs']:.5f}  "
        f"max={q['max_abs']:.5f}  norm_ratio={q['norm_ratio']:.5f}"
    )
    # Looser thresholds than fp4_g32 (UE8M0 + FP8 Q add ~2× the rounding).
    assert q["cos_sim"] > 0.98, q
    assert q["mean_abs"] < 0.08, q
    assert q["max_abs"] < 0.8, q
    assert 0.93 < q["norm_ratio"] < 1.07, q


def test_decode_with_sinks():
    """Sink support: passing a sinks tensor changes the output predictably."""
    device = torch.device("cuda")
    torch.manual_seed(0x8517)

    B, Hq, Hk, N, D = 1, 4, 2, 256, 128
    block_size = 32
    dtype = torch.bfloat16
    scale = 1.0 / (D**0.5)

    query = torch.randn(B, Hq, D, dtype=dtype, device=device)
    key = torch.randn(B, N, Hk, D, dtype=dtype, device=device)
    value = torch.randn(B, N, Hk, D, dtype=dtype, device=device)

    kv_cache, block_table, _ = _build_paged_cache(key, value, block_size, device)
    seq_lens = torch.full((B,), N, dtype=torch.int32, device=device)
    sinks = torch.randn(Hq, dtype=torch.float32, device=device) * 2.0

    out_triton = fp8_g32_decode_attention(
        query=query, kv_cache=kv_cache, block_table=block_table,
        seq_lens=seq_lens, scale=scale, sinks=sinks, max_num_kv_splits=8,
    )
    out_ref = reference_fp8_g32_attention(
        query=query, key=key, value=value, scale=scale, sinks=sinks
    )
    q = _attention_quality(out_triton, out_ref)
    print(
        f"\n[sinks] cos_sim={q['cos_sim']:.6f}  mean={q['mean_abs']:.5f}  "
        f"max={q['max_abs']:.5f}  norm_ratio={q['norm_ratio']:.5f}"
    )
    assert q["cos_sim"] > 0.98, q
    assert q["mean_abs"] < 0.08, q
    assert q["max_abs"] < 0.8, q

    # With strongly-negative sinks the kernel should match no-sinks.
    sinks_low = torch.full((Hq,), -50.0, dtype=torch.float32, device=device)
    out_low = fp8_g32_decode_attention(
        query=query, kv_cache=kv_cache, block_table=block_table,
        seq_lens=seq_lens, scale=scale, sinks=sinks_low, max_num_kv_splits=8,
    )
    out_no_sink = fp8_g32_decode_attention(
        query=query, kv_cache=kv_cache, block_table=block_table,
        seq_lens=seq_lens, scale=scale, max_num_kv_splits=8,
    )
    q_low = _attention_quality(out_low, out_no_sink)
    assert q_low["cos_sim"] > 0.999, (
        f"sinks=-50 should be ~equivalent to no-sinks, got {q_low}"
    )
