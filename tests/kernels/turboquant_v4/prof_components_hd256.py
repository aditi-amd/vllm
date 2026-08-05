#!/usr/bin/env python3
"""Component-level profiler for the FlyDSL hd256 QG=2 decode launcher.

Uses torch.profiler to attribute per-call GPU time to: q-rotation GEMM
(aten::mm + copies), the FlyDSL decode kernel, and the Triton partition
reducer. Tells us which piece to optimize to beat the bf16 baseline.
"""
import os
import sys

import torch
from torch.profiler import profile, ProfilerActivity

sys.path.insert(0, "/shareddata/adrana/workspace/vllm-pr-fp8hd256")

from tests.kernels.turboquant_v4.bench_hd256_gqa2_sliding import (
    make_inputs, HEAD_SIZE, NUM_KV_HEADS, QG, KV_BLOCK_SIZE,
    KEY_PACKED, VAL_PACKED,
)
from vllm.v1.attention.ops.flydsl_turboquant_decode_v4 import (
    flydsl_turboquant_decode_attention_v4, is_flydsl_available,
)


def main():
    assert is_flydsl_available()
    dev = "cuda"
    B = int(os.environ.get("B", "32"))
    seq_len = int(os.environ.get("SEQ", "1024"))
    max_bps = (seq_len + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE + 4
    centroids, q_bf16, kv_cache, bt, sl, _, _ = make_inputs(
        B, NUM_KV_HEADS, seq_len, max_bps)
    q_bf16 = q_bf16.to(torch.bfloat16)
    Pi = torch.eye(HEAD_SIZE, dtype=torch.float32, device=dev)
    PiT = Pi.T.contiguous()
    common = dict(
        kv_cache=kv_cache, block_table=bt, seq_lens=sl,
        Pi=Pi, centroids=centroids, scale=1.0 / (HEAD_SIZE ** 0.5),
        mse_bits=4, key_packed_size=KEY_PACKED, value_quant_bits=4,
        value_packed_size=VAL_PACKED, key_fp8=False,
        norm_correction=False, PiT=PiT, max_seq_len=seq_len,
        max_num_kv_splits=32, sinks=None,
    )

    for _ in range(30):
        flydsl_turboquant_decode_attention_v4(query=q_bf16, **common)
    torch.cuda.synchronize()

    N = 100
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(N):
            flydsl_turboquant_decode_attention_v4(query=q_bf16, **common)
        torch.cuda.synchronize()

    print(f"=== torch.profiler  B={B} seq={seq_len} hd256 QG={QG} "
          f"Hk={NUM_KV_HEADS} (per {N} iters) ===")
    ka = prof.key_averages()
    # Sum device time per kernel; print top entries.
    rows = []
    for e in ka:
        dev_us = getattr(e, "device_time_total", 0.0) or getattr(
            e, "cuda_time_total", 0.0)
        if dev_us and dev_us > 0:
            rows.append((dev_us / N, e.key, e.count / N))
    rows.sort(reverse=True)
    tot = sum(r[0] for r in rows)
    print(f"{'us/call':>10} {'pct':>6}  {'launches/call':>13}  name")
    print("-" * 90)
    for us, name, cnt in rows[:25]:
        print(f"{us:10.2f} {100*us/tot:5.1f}%  {cnt:13.2f}  {name[:60]}")
    print("-" * 90)
    print(f"{tot:10.2f}  total device us/call")


if __name__ == "__main__":
    main()
