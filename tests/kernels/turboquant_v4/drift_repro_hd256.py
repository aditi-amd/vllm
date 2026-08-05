#!/usr/bin/env python3
"""Isolated E2E-faithful DRIFT repro for FlyDSL hd256 QG=2 (Gemma sliding).

Goal: reproduce the server-side *progressive* GSM8K collapse (95->84->53 over
repeated identical runs) in a single process, with NO vLLM scheduler / cudagraph
/ store kernel involved.

Method:
  * Build ONE fixed "probe" decode call (query/kv_cache/bt/sl/centroids/Pi),
    record its golden output.
  * Repeatedly run many "churn" launcher calls with VARYING B / seq_len against
    a padded (Gemma unified-page) cache, sharing the SAME module-level segm pool
    and a SINGLE persistent buf_holder (== one layer) -- exactly like the server.
  * Every CHECK_EVERY iters, re-run the EXACT probe call and compare to golden;
    also checksum the persistent tensors (centroids, PiT, probe kv_cache).

If the probe output DRIFTS or a checksum changes, a real OOB write into
persistent memory exists and we localize it next with compute-sanitizer.
If nothing drifts, the corruption needs the full vLLM context (store/cudagraph/
multi-layer) and the hunt redirects there.

Usage: HIP_VISIBLE_DEVICES=2 python drift_repro_hd256.py
"""
import os
import sys

import torch

sys.path.insert(0, "/shareddata/adrana/workspace/vllm-pr-fp8hd256")
sys.path.insert(0, os.path.dirname(__file__))

from bench_hd256_gqa2_sliding import (  # noqa: E402
    make_inputs, HEAD_SIZE, QG, NUM_KV_HEADS, KV_BLOCK_SIZE,
)
from test_hd256_swa import pad_cache  # noqa: E402
from vllm.v1.attention.ops.flydsl_turboquant_decode_v4 import (  # noqa: E402
    flydsl_turboquant_decode_attention_v4,
    is_flydsl_available,
)

SCALE = 1.0 / (HEAD_SIZE ** 0.5)
PADDED_BLOCK = 198912
WINDOW = 1024
# E2E block-table width = max_model_len/block_size = 32768/32 = 1024. This makes
# worst_case_max_seq_len=32768 -> num_partitions=32, TGPP=4 even for SHORT actual
# sequences (most partitions empty). The tiny-block-table repro missed this.
E2E_MAX_BPS = 1024


class _Holder:
    """Stand-in for a vLLM attention layer (buf_holder for the launcher)."""


def _call(q, kv_cache, bt, sl, centroids, Pi, PiT, holder, max_seq_len, window):
    return flydsl_turboquant_decode_attention_v4(
        query=q, kv_cache=kv_cache, block_table=bt, seq_lens=sl,
        Pi=Pi, centroids=centroids, scale=SCALE,
        mse_bits=4, key_packed_size=(HEAD_SIZE // 2) + 2, value_quant_bits=4,
        value_packed_size=(HEAD_SIZE // 2) + 4, key_fp8=False,
        norm_correction=False, PiT=PiT, max_seq_len=max_seq_len,
        max_num_kv_splits=32, sinks=None, sliding_window=window,
        buf_holder=holder,
    )


def _cksum(t):
    return float(t.detach().float().abs().sum().item())


def main():
    assert is_flydsl_available()
    torch.manual_seed(0)
    dev = "cuda"

    # ---- persistent probe (fixed, padded, Gemma layout) ----
    p_seq = 1500
    p_bps = E2E_MAX_BPS  # E2E-faithful: forces num_partitions=32, TGPP=4
    centroids, p_q, p_kv, p_bt, p_sl, _, _ = make_inputs(
        2, NUM_KV_HEADS, p_seq, p_bps, seed=0x1234)
    p_kv = pad_cache(p_kv, PADDED_BLOCK)
    Pi = torch.eye(HEAD_SIZE, dtype=torch.float32, device=dev)
    PiT = Pi.T.contiguous()
    holder = _Holder()  # single persistent layer, like the server

    golden = _call(p_q, p_kv, p_bt, p_sl, centroids, Pi, PiT, holder,
                   p_seq, WINDOW).detach().clone()
    ck0 = (_cksum(centroids), _cksum(PiT), _cksum(p_kv), _cksum(p_q))

    # ---- churn configs (vary B across capture sizes + seq lengths) ----
    churn = [
        (1, 512), (2, 1500), (4, 900), (8, 3000), (16, 2048),
        (32, 1200), (8, 8192), (2, 16000), (4, 32000), (24, 700),
    ]
    # pre-build churn inputs once (persistent buffers, reused = server-like)
    built = []
    for B, S in churn:
        c2, q2, kv2, bt2, sl2, _, _ = make_inputs(
            B, NUM_KV_HEADS, S, E2E_MAX_BPS, seed=0xABC0 + B + S)
        kv2 = pad_cache(kv2, PADDED_BLOCK)
        built.append((q2, kv2, bt2, sl2, c2, S))

    N_ITERS = 4000
    CHECK_EVERY = 250
    worst_drift = 0.0
    print(f"=== hd256 QG={QG} DRIFT repro: {N_ITERS} churn iters, "
          f"probe seq={p_seq} W={WINDOW}, padded ===")
    for it in range(1, N_ITERS + 1):
        q2, kv2, bt2, sl2, c2, S = built[it % len(built)]
        # churn uses the SAME persistent centroids/Pi (like a real layer) so
        # the PiT_f32 cache path is identical to the probe.
        _call(q2, kv2, bt2, sl2, centroids, Pi, PiT, holder, S, WINDOW)
        if it % CHECK_EVERY == 0:
            cur = _call(p_q, p_kv, p_bt, p_sl, centroids, Pi, PiT, holder,
                        p_seq, WINDOW)
            drift = (cur.float() - golden.float()).abs().max().item()
            worst_drift = max(worst_drift, drift)
            ck = (_cksum(centroids), _cksum(PiT), _cksum(p_kv), _cksum(p_q))
            dck = [abs(a - b) for a, b in zip(ck, ck0)]
            print(f"  iter {it:>5}: probe_drift={drift:.3e}  "
                  f"d[centroids,PiT,kv,q]="
                  f"[{dck[0]:.2e},{dck[1]:.2e},{dck[2]:.2e},{dck[3]:.2e}]")
    print(f"\nworst probe drift over run = {worst_drift:.3e}")
    print("RESULT:", "DRIFT DETECTED (real OOB)" if worst_drift > 1e-3
          else "NO DRIFT (launcher-isolated is clean)")


if __name__ == "__main__":
    main()
