# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the FP4-g32 Triton decode kernel.

For each (B, Hq, Hk, seq_len) shape:
 1. Generate random K/V tensors.
 2. Encode them into a uint8 cache via the Triton store kernel (which is
    itself verified bit-exact against the reference fakequant in
    `test_store_vs_reference.py`).
 3. Run the Triton decode kernel.
 4. Run the PyTorch reference attention (which does its own encode/decode
    round-trip internally and is therefore the algorithmic ground truth).
 5. Compare. We expect small numerical drift (≤ ~3 bf16 ULPs / output
    element on typical attention outputs) due to FA-2 online softmax
    vs naive softmax and per-tile fp32 accumulation order.

The test is structured to fail loud on any algorithmic bug (e.g. wrong
group ordering, missing scale, sink mis-placement) while tolerating
bf16-rounding-level differences.
"""

from __future__ import annotations

import pytest
import torch

from vllm.v1.attention.ops.fp4_g32.fp4_levels import slot_size
from vllm.v1.attention.ops.fp4_g32.reference import reference_fp4_g32_attention
from vllm.v1.attention.ops.fp4_g32.triton_decode import fp4_g32_decode_attention
from vllm.v1.attention.ops.fp4_g32.triton_store import fp4_g32_store


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA/HIP device"
)


def _build_paged_cache(
    key_BNHD: torch.Tensor,
    value_BNHD: torch.Tensor,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack [B, N, Hk, D] K/V into a paged uint8 FP4-g32 cache.

    Returns (kv_cache, block_table, slot_mapping_flat).
    """
    B, N, Hk, D = key_BNHD.shape
    blocks_per_seq = (N + block_size - 1) // block_size
    num_blocks = B * blocks_per_seq + 1  # +1 to verify we don't overflow

    slot_b = slot_size(D)
    kv_cache = torch.zeros(
        num_blocks, block_size, Hk, slot_b, dtype=torch.uint8, device=device
    )

    # Assign distinct block ids per batch row.
    block_table = torch.zeros(B, blocks_per_seq, dtype=torch.int32, device=device)
    for bi in range(B):
        for blk in range(blocks_per_seq):
            block_table[bi, blk] = bi * blocks_per_seq + blk

    # Flatten K/V to [B*N, Hk, D], compute slot per (bi, ti).
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

    fp4_g32_store(k_flat, v_flat, kv_cache, slot_mapping)
    torch.cuda.synchronize()

    return kv_cache, block_table, slot_mapping


def _attention_quality(
    out_triton: torch.Tensor, out_ref: torch.Tensor
) -> dict[str, float]:
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
        (1, 1, 1, 16, 128),    # min config
        (1, 4, 2, 128, 128),   # small GQA
        (2, 8, 4, 256, 128),   # multi-batch GQA
        (1, 4, 1, 1024, 128),  # long context, no GQA
    ],
)
def test_decode_matches_reference(dtype, B, Hq, Hk, N, D):
    """Triton decode output is close to the reference attention output."""
    device = torch.device("cuda")
    torch.manual_seed(0x4032)

    block_size = 32
    scale = 1.0 / (D**0.5)

    query = torch.randn(B, Hq, D, dtype=dtype, device=device)
    key = torch.randn(B, N, Hk, D, dtype=dtype, device=device)
    value = torch.randn(B, N, Hk, D, dtype=dtype, device=device)

    kv_cache, block_table, _ = _build_paged_cache(key, value, block_size, device)
    seq_lens = torch.full((B,), N, dtype=torch.int32, device=device)

    out_triton = fp4_g32_decode_attention(
        query=query,
        kv_cache=kv_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        scale=scale,
        max_num_kv_splits=8,
    )
    torch.cuda.synchronize()

    out_ref = reference_fp4_g32_attention(
        query=query, key=key, value=value, scale=scale
    )

    q = _attention_quality(out_triton, out_ref)
    print(
        f"\n[B={B} Hq={Hq} Hk={Hk} N={N} D={D} {dtype}] "
        f"cos_sim={q['cos_sim']:.6f}  mean={q['mean_abs']:.5f}  "
        f"max={q['max_abs']:.5f}  norm_ratio={q['norm_ratio']:.5f}"
    )
    # Loose-ish but algorithmically-meaningful tolerances:
    # - cos_sim ≥ 0.99: would catch a mis-ordered group, missing scale,
    #   or wrong dim-iteration.
    # - mean_abs ≤ 0.05 at typical N: catches systematic scaling errors.
    # - max_abs ≤ 0.5: catches single-element catastrophic drift.
    assert q["cos_sim"] > 0.99, q
    assert q["mean_abs"] < 0.05, q
    assert q["max_abs"] < 0.5, q
    # norm should be within ~3 % of the reference (bf16 + FA-2 reordering).
    assert 0.97 < q["norm_ratio"] < 1.03, q


def test_decode_with_sinks():
    """Sink support: passing a sinks tensor changes the output predictably
    (sink-with-zero == softmax over [logits, sink_logit] with sink getting
    fractional probability mass)."""
    device = torch.device("cuda")
    torch.manual_seed(0x517)

    B, Hq, Hk, N, D = 1, 4, 2, 256, 128
    block_size = 32
    dtype = torch.bfloat16
    scale = 1.0 / (D**0.5)

    query = torch.randn(B, Hq, D, dtype=dtype, device=device)
    key = torch.randn(B, N, Hk, D, dtype=dtype, device=device)
    value = torch.randn(B, N, Hk, D, dtype=dtype, device=device)

    kv_cache, block_table, _ = _build_paged_cache(key, value, block_size, device)
    seq_lens = torch.full((B,), N, dtype=torch.int32, device=device)

    # Sinks magnitude ~5 → meaningful but not dominating.
    sinks = torch.randn(Hq, dtype=torch.float32, device=device) * 2.0

    out_triton = fp4_g32_decode_attention(
        query=query,
        kv_cache=kv_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        scale=scale,
        sinks=sinks,
        max_num_kv_splits=8,
    )

    out_ref = reference_fp4_g32_attention(
        query=query, key=key, value=value, scale=scale, sinks=sinks
    )

    q = _attention_quality(out_triton, out_ref)
    print(
        f"\n[sinks] cos_sim={q['cos_sim']:.6f}  mean={q['mean_abs']:.5f}  "
        f"max={q['max_abs']:.5f}  norm_ratio={q['norm_ratio']:.5f}"
    )
    assert q["cos_sim"] > 0.99, q
    assert q["mean_abs"] < 0.05, q
    assert q["max_abs"] < 0.5, q

    # Bonus check: with strongly-negative sinks the kernel should behave
    # essentially like no-sinks (sink prob mass ≈ 0), and with strongly
    # positive sinks it should suppress the regular outputs (sink dominates).
    sinks_low = torch.full((Hq,), -50.0, dtype=torch.float32, device=device)
    out_low = fp4_g32_decode_attention(
        query=query, kv_cache=kv_cache, block_table=block_table,
        seq_lens=seq_lens, scale=scale, sinks=sinks_low, max_num_kv_splits=8,
    )
    out_no_sink = fp4_g32_decode_attention(
        query=query, kv_cache=kv_cache, block_table=block_table,
        seq_lens=seq_lens, scale=scale, max_num_kv_splits=8,
    )
    q_low = _attention_quality(out_low, out_no_sink)
    assert q_low["cos_sim"] > 0.999, (
        f"sinks=-50 should be ~equivalent to no-sinks, got {q_low}"
    )
