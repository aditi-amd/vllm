# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerics test for the speculative-decode (MTP) verify expansion.

The TurboQuant backend serves an MTP verify batch (B requests, each with
K = 1+num_speculative_tokens causal query tokens against a shared context)
by expanding it into B*K single-query decodes with *staggered* per-query
context lengths and a block table repeated K times, then calling the ordinary
single-query fp8_g32 decode kernel (see
`TurboQuantAttentionImpl._spec_decode_region`).

This test reproduces that exact expansion math and checks each verify query's
output against the trusted reference computed with the query's own truncated
context. It guards the staggered-`seq_lens` / request-major layout logic, which
is the only new code on the UQ+MTP path.
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


def _build_paged_cache(key_BNHD, value_BNHD, block_size, device):
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
            gidx = bi * N + ti
            block_num = int(block_table[bi, ti // block_size].item())
            slot = block_num * block_size + ti % block_size
            slot_mapping[gidx] = slot
            k_flat[gidx] = key_BNHD[bi, ti]
            v_flat[gidx] = value_BNHD[bi, ti]

    fp8_g32_store(k_flat, v_flat, kv_cache, slot_mapping)
    torch.cuda.synchronize()
    return kv_cache, block_table


def _quality(a_out, b_out):
    a = a_out.float().flatten()
    b = b_out.float().flatten()
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
    "B,Hq,Hk,N,D,K",
    [
        (1, 4, 2, 128, 128, 2),
        (2, 8, 4, 256, 128, 3),
        (1, 4, 1, 1024, 128, 3),
        (3, 8, 2, 512, 128, 4),
    ],
)
def test_spec_decode_expansion_matches_reference(dtype, B, Hq, Hk, N, D, K):
    """Vectorized B*K expansion == per-query reference with truncated context.

    Request b's K verify queries attend causally to context lengths
    [N-K+1, ..., N]; the KV of all N tokens (including the K new ones) is
    already stored in the cache.
    """
    device = torch.device("cuda")
    torch.manual_seed(0xA17 + K)

    block_size = 32
    scale = 1.0 / (D**0.5)

    key = torch.randn(B, N, Hk, D, dtype=dtype, device=device)
    value = torch.randn(B, N, Hk, D, dtype=dtype, device=device)
    # One query vector per (request, verify-slot).
    q_bk = torch.randn(B, K, Hq, D, dtype=dtype, device=device)

    kv_cache, block_table = _build_paged_cache(key, value, block_size, device)
    seq_lens_dec = torch.full((B,), N, dtype=torch.int32, device=device)

    # ---- expansion math, identical to _spec_decode_region ----
    offs = torch.arange(K, device=device, dtype=seq_lens_dec.dtype) - (K - 1)
    synth_seq_lens = (seq_lens_dec[:, None] + offs[None, :]).reshape(-1).clamp_(min=1)
    synth_bt = block_table.repeat_interleave(K, dim=0)
    q_flat = q_bk.reshape(B * K, Hq, D)  # request-major

    out_vec = fp8_g32_decode_attention(
        query=q_flat,
        kv_cache=kv_cache,
        block_table=synth_bt,
        seq_lens=synth_seq_lens,
        scale=scale,
        max_num_kv_splits=8,
    ).reshape(B, K, Hq, D)
    torch.cuda.synchronize()

    # ---- per-query reference with truncated context ----
    worst = {"cos_sim": 1.0, "mean_abs": 0.0, "max_abs": 0.0, "norm_ratio": 1.0}
    for b in range(B):
        for k in range(K):
            L = N - K + 1 + k
            ref = reference_fp8_g32_attention(
                query=q_bk[b, k : k + 1].reshape(1, Hq, D),
                key=key[b : b + 1, :L],
                value=value[b : b + 1, :L],
                scale=scale,
            )  # (1, Hq, D)
            q = _quality(out_vec[b, k], ref[0])
            if q["cos_sim"] < worst["cos_sim"]:
                worst = q

    print(
        f"\n[B={B} Hq={Hq} Hk={Hk} N={N} D={D} K={K} {dtype}] worst-query "
        f"cos_sim={worst['cos_sim']:.6f} mean={worst['mean_abs']:.5f} "
        f"max={worst['max_abs']:.5f} norm_ratio={worst['norm_ratio']:.5f}"
    )
    # Same tolerances as the single-query decode test (UE8M0 + FP8 Q rounding).
    assert worst["cos_sim"] > 0.98, worst
    assert worst["mean_abs"] < 0.08, worst
    assert worst["max_abs"] < 0.8, worst
    assert 0.93 < worst["norm_ratio"] < 1.07, worst
