#!/usr/bin/env python3
"""Validate the sliding-window port in the SoA v3 unified attention kernel.

The SoA kernel was forked from the AoS unified kernel with sliding-window
support omitted ("deferred to follow-ups"), so any model with per-layer
windows (Gemma 4: window=1024 on 50 of 60 layers) raised NotImplementedError
in the backend. This checks the ported SLIDING_WINDOW path against exact
attention math over the dequantized K/V the fixture builds.

For decode the single query sits at absolute position T-1, so a window W
admits keys [T-W, T-1]. Cases deliberately include a W that is not a
multiple of TILE_SIZE (16 for decode), which is what exercises the
per-element boundary mask rather than just the tile pruning, and a W larger
than the context, which must degenerate to full attention.

Run: python tests/kernels/turboquant_v4/test_soa_v3_sliding_window.py
"""
from __future__ import annotations

import os
import sys

# Fixture reads these at import time; match a Gemma-4 sliding layer's shape
# (block_size 32, GQA 2) rather than the module defaults (16 / 16).
os.environ.setdefault("BS", "32")
os.environ.setdefault("QG", "2")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))

# The installed `vllm` distribution resolves to a *sibling worktree*, so
# running this as a script (which puts _HERE, not the repo root, on the
# path) would silently test the wrong tree. Pin this branch explicitly, and
# import from it before the fixture inserts its own worktree path.
sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402

from vllm.v1.attention.ops.turboquant_soa_fusion.triton_turboquant_unified_attention import (  # noqa: E402,E501
    triton_turboquant_decode_attention_v3 as soa_decode_v3,
)

sys.path.insert(0, _HERE)
from test_v4_vs_v3_fast import (  # noqa: E402
    HEAD_SIZE,
    KEY_DATA_BYTES,
    QG,
    make_inputs_fast,
)


def _reference_decode(
    q: torch.Tensor,  # [S, Hq, D] bf16
    K_ref: torch.Tensor,  # [S, Hk, T, D] fp32
    V_ref: torch.Tensor,  # [S, Hk, T, D] fp32
    scale: float,
    window: int,
) -> torch.Tensor:
    """Exact fp32 decode attention with a causal sliding window."""
    S, Hq, D = q.shape
    T = K_ref.shape[2]
    # Decode query is at absolute position T-1; window W admits [T-W, T-1].
    lo = 0 if window <= 0 else max(0, T - window)

    out = torch.empty(S, Hq, D, dtype=torch.float32)
    qf = q.float().cpu()
    for s in range(S):
        for hq in range(Hq):
            hk = hq // QG
            k = K_ref[s, hk, lo:T]  # [W, D]
            v = V_ref[s, hk, lo:T]  # [W, D]
            logits = scale * (k @ qf[s, hq])  # [W]
            p = torch.softmax(logits.double(), dim=0)
            out[s, hq] = (p @ v.double()).float()
    return out


def _run(num_seqs: int, num_kv_heads: int, seq_len: int, window: int):
    max_bps = (seq_len + 32 - 1) // 32 + 4
    centroids, q_bf16, kv_cache, bt, sl, K_ref, V_ref = make_inputs_fast(
        num_seqs, num_kv_heads, seq_len, max_bps
    )
    Pi = torch.eye(HEAD_SIZE, dtype=torch.float32, device="cuda")
    scale = 1.0 / (HEAD_SIZE**0.5)

    got = soa_decode_v3(
        query=q_bf16,
        kv_cache=kv_cache,
        block_table=bt,
        seq_lens=sl,
        Pi=Pi,
        centroids=centroids,
        scale=scale,
        mse_bits=4,
        key_packed_size=KEY_DATA_BYTES + 2,
        value_quant_bits=4,
        value_packed_size=KEY_DATA_BYTES + 4,
        key_fp8=False,
        PiT=Pi.T.contiguous(),
        max_seq_len=seq_len,
        max_num_kv_splits=32,
        sinks=None,
        sliding_window=window,
    )
    want = _reference_decode(q_bf16, K_ref, V_ref, scale, window)
    return (got.float().cpu() - want).abs().max().item()


def main() -> None:
    import vllm

    assert os.path.dirname(vllm.__file__).startswith(_REPO_ROOT), (
        f"testing the wrong tree: {vllm.__file__}"
    )

    # bf16 Q against an fp32/fp64 reference; the fixture's own v3-vs-v4
    # comparisons use 5e-3, so hold the same bar.
    TOL = 5e-3
    cases = [
        # (seqs, kv_heads, seq_len, window, label)
        (2, 4, 512, 0, "window=0 (full attn, regression guard)"),
        (2, 4, 512, 128, "window=128, tile-aligned"),
        (2, 4, 512, 100, "window=100, NOT tile-aligned (boundary mask)"),
        (2, 4, 512, 1024, "window > context (degenerates to full)"),
        (1, 2, 1024, 1024, "window == context, Gemma-4 window size"),
        (3, 8, 2048, 1024, "long context, heavy pruning (~50% tiles)"),
        (2, 4, 96, 33, "short ctx, odd window"),
    ]
    print(f"{'case':<46} {'max_abs':>10}   verdict")
    print("-" * 72)
    failures = []
    for seqs, hk, T, w, label in cases:
        err = _run(seqs, hk, T, w)
        ok = err < TOL
        if not ok:
            failures.append((label, err))
        print(f"{label:<46} {err:10.2e}   {'PASS' if ok else 'FAIL'}")

    print("-" * 72)
    if failures:
        print(f"FAILED {len(failures)}/{len(cases)} (tolerance {TOL:g})")
        raise SystemExit(1)
    print(f"All {len(cases)} cases passed (tolerance {TOL:g})")


if __name__ == "__main__":
    main()
