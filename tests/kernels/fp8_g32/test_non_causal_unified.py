"""Non-causal (bidirectional) fp8_g32 unified attention, as DFlash needs it.

Validated without a second reference implementation, by exploiting an exact
equivalence: under a non-causal mask every query in the block attends to the
whole sequence, which is precisely what a SINGLE causal query at the last
position already does. So for each block position j,

    non_causal(q_0..q_{Q-1})[j]  ==  causal(q_j alone, query_len=1)

Any real masking bug (leaking the causal cut into the non-causal path, or
stopping the tile loop short at max_seq_prefix_len) breaks this for j < Q-1.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "turboquant_v4")
)

from test_fp8_g32_v4_parity import (  # noqa: E402
    HEAD_SIZE,
    KV_BLOCK_SIZE,
    QG,
    build_fp8_g32_cache,
)
from vllm.v1.attention.ops.fp8_g32.reference import hadamard_matrix  # noqa: E402

try:
    from vllm.v1.attention.ops.fp8_g32.triton_unified_attention import (
        fp8_g32_unified_attention,
    )
except ImportError:  # pragma: no cover
    fp8_g32_unified_attention = None


def _run(query, kv_cache, bt, seq_lens, cu_q, scale, PiT, max_q, N, causal):
    return fp8_g32_unified_attention(
        query=query,
        kv_cache=kv_cache,
        block_table=bt,
        seq_lens=seq_lens,
        query_start_loc=cu_q,
        scale=scale,
        PiT=PiT,
        max_query_len=max_q,
        max_seq_len=N,
        sinks=None,
        sliding_window=None,
        causal=causal,
    )


@pytest.mark.skipif(
    fp8_g32_unified_attention is None, reason="fp8_g32 Triton path unavailable"
)
@pytest.mark.parametrize("B,Hk,N,Q", [(2, 1, 256, 4), (1, 2, 512, 8), (3, 1, 128, 3)])
def test_non_causal_matches_single_query_causal(B, Hk, N, Q):
    device = "cuda"
    torch.manual_seed(0x0C0FFEE)
    D = HEAD_SIZE
    scale = 1.0 / (D**0.5)
    max_bps = (N + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE + 4

    _, kv_cache, bt, sl, _, _ = build_fp8_g32_cache(
        B, Hk, N, max_bps, device=device
    )
    PiT = hadamard_matrix(D, torch.device(device), torch.float32).contiguous()
    query = torch.randn(B * Q, Hk * QG, D, dtype=torch.bfloat16, device=device)
    cu_q = torch.arange(0, B * Q + 1, Q, dtype=sl.dtype, device=device)

    out_nc = _run(query, kv_cache, bt, sl, cu_q, scale, PiT, Q, N, causal=False)

    # Same rows, each replayed as a lone causal query over the full sequence.
    cu_1 = torch.tensor([0, 1], dtype=sl.dtype, device=device)
    for s in range(B):
        sl_1 = sl[s : s + 1].contiguous()
        for j in range(Q):
            row = query[s * Q + j].unsqueeze(0).contiguous()
            ref = _run(
                row, kv_cache, bt[s : s + 1].contiguous(), sl_1, cu_1,
                scale, PiT, 1, N, causal=True,
            )
            got = out_nc[s * Q + j]
            torch.testing.assert_close(
                got.float(), ref[0].float(), rtol=2e-2, atol=2e-2,
                msg=lambda m, s=s, j=j: f"seq {s} block pos {j}:\n{m}",
            )


@pytest.mark.skipif(
    fp8_g32_unified_attention is None, reason="fp8_g32 Triton path unavailable"
)
def test_causal_path_unchanged():
    """causal=True must be bit-identical to the pre-existing default."""
    device = "cuda"
    torch.manual_seed(0x1234)
    B, Hk, N, Q = 2, 1, 256, 4
    D = HEAD_SIZE
    scale = 1.0 / (D**0.5)
    _, kv_cache, bt, sl, _, _ = build_fp8_g32_cache(
        B, Hk, N, (N + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE + 4, device=device
    )
    PiT = hadamard_matrix(D, torch.device(device), torch.float32).contiguous()
    query = torch.randn(B * Q, Hk * QG, D, dtype=torch.bfloat16, device=device)
    cu_q = torch.arange(0, B * Q + 1, Q, dtype=sl.dtype, device=device)

    explicit = _run(query, kv_cache, bt, sl, cu_q, scale, PiT, Q, N, causal=True)
    default = fp8_g32_unified_attention(
        query=query, kv_cache=kv_cache, block_table=bt, seq_lens=sl,
        query_start_loc=cu_q, scale=scale, PiT=PiT, max_query_len=Q,
        max_seq_len=N, sinks=None, sliding_window=None,
    )
    assert torch.equal(explicit, default)
