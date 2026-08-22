#!/usr/bin/env python3
"""L2 compounding/drift gate for the fp8_g32 hd256 decode kernel (QG=8).

Mirrors drift_repro_hd256.py (TQ QG=2) for the fp8_g32 path. Catches the class
of bug single-step parity misses: a churn of many decode calls (varying B/seq)
sharing ONE persistent buf_holder + the module-level segm pool — exactly like
the server — that silently corrupts persistent memory (OOB write) and makes a
fixed "probe" decode drift over time. This is the failure mode behind the
qk_fp8 GSM8K collapse (single-step cos 0.99999 but progressive E2E decay).

Run after ANY Tier-2 (numerically-different) kernel change:
  HIP_VISIBLE_DEVICES=0 VLLM_FP8_G32_V3=1 VLLM_FP8_G32_DECODE_V4=1 \
  VLLM_FP8_G32_DECODE_V4_QK_SCALED=1 VLLM_FLYDSL_ROOT=/root/FlyDSL \
  VLLM_FLYDSL_PKGS=/root/FlyDSL/build-fly/python_packages \
  python tests/kernels/turboquant_v4/drift_fp8_g32_hd256.py

PASS = probe output stays within DRIFT_TOL of golden and persistent checksums
don't move. FAIL = real OOB / compounding corruption.
"""
from __future__ import annotations

import os
import sys
import types

import torch

sys.path.insert(0, "/shareddata/adrana/workspace/vllm-pr-fp8hd256")
sys.path.insert(0, os.path.dirname(__file__))

from dev_loop import (  # noqa: E402
    D, QG, HK, BS, _build_parity_cache,
)
from vllm.v1.attention.ops.fp8_g32.fp8_levels import (  # noqa: E402
    get_group_size, slot_size,
)
from vllm.v1.attention.ops.fp8_g32.reference import hadamard_matrix  # noqa: E402
from vllm.v1.attention.ops.flydsl_fp8_g32_decode_v4 import (  # noqa: E402
    flydsl_fp8_g32_decode_attention_v4, is_flydsl_available,
)

SCALE = 1.0 / (D ** 0.5)
# Wide block table so worst_case_max_seq_len forces num_partitions=32, TGPP>1
# even for short actual sequences (matches the server sizing path).
E2E_MAX_BPS = 1024
DRIFT_TOL = 1e-3


def _churn_inputs(b, seq, device="cuda", seed=0):
    """Fast random-byte cache (values irrelevant for churn; only shapes matter
    for exercising the shared segm pool + buf_holder buckets)."""
    Hk, Hq, gs = HK, HK * QG, get_group_size()
    padded_slot = slot_size(D, gs)
    bps = (seq + BS - 1) // BS
    g = torch.Generator(device=device).manual_seed(seed)
    q = (torch.randn(b, Hq, D, device=device, generator=g) * 0.5).to(
        torch.bfloat16)
    kv = torch.randint(0, 256, (b * bps + 8, BS, Hk, padded_slot),
                       dtype=torch.uint8, device=device, generator=g)
    bt = torch.zeros(b, E2E_MAX_BPS, dtype=torch.int32, device=device)
    bt[:, :bps] = (torch.arange(b * bps, dtype=torch.int32, device=device)
                   .reshape(b, bps) + 1)
    sl = torch.full((b,), seq, dtype=torch.int32, device=device)
    PiT = hadamard_matrix(D, torch.device(device), torch.float32).contiguous()
    return q, kv, bt, sl, PiT, seq


def _cksum(t):
    return float(t.detach().float().abs().sum().item())


def main():
    assert is_flydsl_available()
    dev = "cuda"
    holder = types.SimpleNamespace()  # single persistent layer, like the server

    # ---- fixed probe (real-encoded, so drift shows in a meaningful output) ---
    p_q, p_kv, p_bt, p_sl, _K, _V, p_PiT = _build_parity_cache(
        2, 1536, device=dev, seed=0x1234)
    golden = flydsl_fp8_g32_decode_attention_v4(
        query=p_q, kv_cache=p_kv, block_table=p_bt, seq_lens=p_sl, scale=SCALE,
        PiT=p_PiT, max_seq_len=1536, max_num_kv_splits=32, sinks=None,
        buf_holder=holder).detach().clone()
    ck0 = (_cksum(p_PiT), _cksum(p_kv), _cksum(p_q))

    churn = [(1, 512), (2, 1536), (4, 900), (8, 3000), (16, 2048),
             (32, 1200), (8, 8192), (2, 16000), (4, 32000), (24, 700)]
    built = [_churn_inputs(b, s, seed=0xABC0 + b + s) for b, s in churn]

    N_ITERS = int(os.environ.get("DRIFT_ITERS", "3000"))
    CHECK_EVERY = 250
    worst = 0.0
    print(f"=== fp8_g32 hd256 QG={QG} DRIFT: {N_ITERS} churn iters, "
          f"probe seq=1536 ===")
    for it in range(1, N_ITERS + 1):
        q2, kv2, bt2, sl2, PiT2, S = built[it % len(built)]
        flydsl_fp8_g32_decode_attention_v4(
            query=q2, kv_cache=kv2, block_table=bt2, seq_lens=sl2, scale=SCALE,
            PiT=PiT2, max_seq_len=S, max_num_kv_splits=32, sinks=None,
            buf_holder=holder)
        if it % CHECK_EVERY == 0:
            cur = flydsl_fp8_g32_decode_attention_v4(
                query=p_q, kv_cache=p_kv, block_table=p_bt, seq_lens=p_sl,
                scale=SCALE, PiT=p_PiT, max_seq_len=1536, max_num_kv_splits=32,
                sinks=None, buf_holder=holder)
            drift = (cur.float() - golden.float()).abs().max().item()
            worst = max(worst, drift)
            ck = (_cksum(p_PiT), _cksum(p_kv), _cksum(p_q))
            dck = [abs(a - b) for a, b in zip(ck, ck0)]
            print(f"  iter {it:>5}: probe_drift={drift:.3e}  "
                  f"d[PiT,kv,q]=[{dck[0]:.2e},{dck[1]:.2e},{dck[2]:.2e}]")
    print(f"\nworst probe drift = {worst:.3e}  (tol {DRIFT_TOL:.0e})")
    ok = worst <= DRIFT_TOL
    print("RESULT:", "PASS (no drift)" if ok else "FAIL (drift/OOB)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
