#!/usr/bin/env python3
"""Correctness: FlyDSL hd256 QG=2 SLIDING-WINDOW decode vs windowed exact attention.

The decode query sits at position seq_len-1 and, under a window W, may attend
only to keys [max(0, seq_len-W), seq_len). This test drives seq_len > W so the
window actually truncates (a full-attention kernel would be visibly wrong), and
compares the FlyDSL v4 output to an fp32 reference that applies the same window.

Cases mirror test_soa_v3_sliding_window.py:
  - W >= seq_len  -> window inactive, must equal full attention
  - W < seq_len, W a multiple of the 256-token compute block (aligned start)
  - W < seq_len, W NOT aligned (stresses the per-token boundary mask over the
    aligned-down slack)
  - long context, heavy pruning (window is a small fraction of seq_len)

Usage: HIP_VISIBLE_DEVICES=7 python test_hd256_swa.py
"""
import os
import sys

import torch

sys.path.insert(0, "/shareddata/adrana/workspace/vllm-pr-fp8hd256")
sys.path.insert(0, os.path.dirname(__file__))

from bench_hd256_gqa2_sliding import (  # noqa: E402
    make_inputs, HEAD_SIZE, QG, NUM_KV_HEADS, KV_BLOCK_SIZE,
)
from vllm.v1.attention.ops.flydsl_turboquant_decode_v4 import (  # noqa: E402
    flydsl_turboquant_decode_attention_v4,
    is_flydsl_available,
)


def windowed_reference(q_bf16, K_ref, V_ref, sl, scale, window):
    """Exact fp32 attention with decode sliding window [max(0,ql-W), ql)."""
    num_seqs, Hq, D = q_bf16.shape
    Hk = K_ref.shape[1]
    QG_ = Hq // Hk
    q = q_bf16.float().reshape(num_seqs, Hk, QG_, D)
    out = torch.zeros(num_seqs, Hq, D, dtype=torch.float32)
    for s in range(num_seqs):
        ql = int(sl[s].item())
        lo = max(0, ql - window) if window > 0 else 0
        for h in range(Hk):
            K = K_ref[s, h, lo:ql]
            V = V_ref[s, h, lo:ql]
            qq = q[s, h]
            scores = (qq @ K.T) * scale
            m = scores.max(dim=-1, keepdim=True).values
            e = torch.exp(scores - m)
            p = e / e.sum(dim=-1, keepdim=True)
            out[s, h * QG_:(h + 1) * QG_] = p @ V
    return out


def pad_cache(kv_cache_4d, padded_block_bytes):
    """Reproduce Gemma 4's unified-page padded KV view.

    vLLM pads the D=256 sliding page (natural 134144 B) up to the D=512 global
    page (198912 B), handing the kernel a strided view whose stride(0) exceeds
    the natural block. We allocate a padded buffer, copy each natural block into
    its leading bytes, and return an as_strided view with the padded block
    stride (matching the real strides=[198912, 4192, 262, 1]).
    """
    nb, bs, hk, slot = kv_cache_4d.shape
    natural = bs * hk * slot
    assert padded_block_bytes >= natural
    flat = kv_cache_4d.reshape(nb, natural)
    padded = torch.zeros(nb, padded_block_bytes, dtype=torch.uint8,
                         device=kv_cache_4d.device)
    padded[:, :natural] = flat
    return torch.as_strided(
        padded, size=(nb, bs, hk, slot),
        stride=(padded_block_bytes, hk * slot, slot, 1),
    )


def run(num_seqs, seq_len, window, padded=False):
    max_bps = (seq_len + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE + 4
    centroids, q_bf16, kv_cache, bt, sl, K_ref, V_ref = make_inputs(
        num_seqs, NUM_KV_HEADS, seq_len, max_bps
    )
    if padded:
        # 198912 = Gemma's global D=512 page (32*4*518 or equivalently the
        # padded sliding page). Any value > natural exercises the stride path.
        kv_cache = pad_cache(kv_cache, 198912)
    Pi = torch.eye(HEAD_SIZE, dtype=torch.float32, device="cuda")
    PiT = Pi.T.contiguous()
    o_v4 = flydsl_turboquant_decode_attention_v4(
        query=q_bf16, kv_cache=kv_cache, block_table=bt, seq_lens=sl,
        Pi=Pi, centroids=centroids, scale=1.0 / (HEAD_SIZE ** 0.5),
        mse_bits=4, key_packed_size=(HEAD_SIZE // 2) + 2, value_quant_bits=4,
        value_packed_size=(HEAD_SIZE // 2) + 4, key_fp8=False,
        norm_correction=False, PiT=PiT, max_seq_len=seq_len,
        max_num_kv_splits=32, sinks=None, sliding_window=window,
    )
    ref = windowed_reference(q_bf16.cpu(), K_ref, V_ref, sl.cpu(),
                             1.0 / (HEAD_SIZE ** 0.5), window)
    return (o_v4.cpu().float() - ref).abs().max().item()


def main():
    assert is_flydsl_available()
    TOL = 5e-3
    print(f"=== hd256 QG={QG} sliding-window correctness (tol {TOL}) ===")
    cases = [
        ("W>=seq (inactive)         ", 2, 512, 1024, False),
        ("W==seq                    ", 2, 1024, 1024, False),
        ("W<seq, aligned (W=1024)   ", 2, 2048, 1024, False),
        ("W<seq, aligned (W=512)    ", 2, 2048, 512, False),
        ("W<seq, NOT aligned (W=700)", 2, 2048, 700, False),
        ("W<seq, NOT aligned (W=100)", 2, 1536, 100, False),
        ("long ctx, heavy prune     ", 2, 8192, 1024, False),
        ("batch, aligned            ", 8, 3000, 1024, False),
        # PADDED cache (Gemma unified page): the real E2E layout.
        ("PADDED W<seq aligned      ", 2, 2048, 1024, True),
        ("PADDED W>=seq (inactive)  ", 2, 512, 1024, True),
        ("PADDED long ctx prune     ", 2, 8192, 1024, True),
        ("PADDED batch NOT aligned  ", 8, 3000, 700, True),
    ]
    worst = 0.0
    npass = 0
    for name, B, S, W, pad in cases:
        try:
            d = run(B, S, W, padded=pad)
            ok = d < TOL
            npass += ok
            worst = max(worst, d)
            print(f"[{'PASS' if ok else 'FAIL'}] {name}  B={B} seq={S:>5} "
                  f"W={W:>4}  maxdiff={d:.3e}")
        except Exception as ex:  # noqa: BLE001
            print(f"[ERR ] {name}  B={B} seq={S} W={W}: {type(ex).__name__}: {ex}")
    print(f"\n{npass}/{len(cases)} passed; worst maxdiff={worst:.3e}")
    print("RESULT:", "ALL PASS" if npass == len(cases) else "FAILURES PRESENT")


if __name__ == "__main__":
    main()
