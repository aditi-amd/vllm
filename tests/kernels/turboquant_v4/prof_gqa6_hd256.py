#!/usr/bin/env python3
"""Isolated single-shape driver for the GQA-6 hd256 v4 decode kernel.

Runs ONLY the FlyDSL v4 kernel on one representative Qwen3.6 full-attention
decode shape (B, seq_len, QG=6, Hk=4, D=256), in a tight loop, so rocprofv3
--pmc attributes hardware counters (LDS bank conflicts, VALU, etc.) to the
single decode dispatch. No Triton reference, no 27B model load.

Usage:
  HIP_VISIBLE_DEVICES=2 ITERS=200 B=64 SEQ=1024 python prof_gqa6_hd256.py
"""
import os
import sys

import torch

sys.path.insert(0, "/shareddata/adrana/workspace/vllm-pr-fp8hd256")
sys.path.insert(0, os.path.dirname(__file__))

from bench_hd256_gqa2_sliding import make_inputs, KEY_PACKED, VAL_PACKED, HEAD_SIZE  # noqa: E402
from vllm.v1.attention.ops.flydsl_turboquant_decode_v4 import (  # noqa: E402
    flydsl_turboquant_decode_attention_v4,
    is_flydsl_available,
)


def main():
    assert is_flydsl_available(), "FlyDSL not available"
    B = int(os.environ.get("B", "64"))
    SEQ = int(os.environ.get("SEQ", "1024"))
    ITERS = int(os.environ.get("ITERS", "200"))
    HK = int(os.environ.get("HK", "4"))
    max_bps = (SEQ + 32 - 1) // 32 + 4
    centroids, q_bf16, kv_cache, bt, sl, _, _ = make_inputs(B, HK, SEQ, max_bps)
    Pi = torch.eye(HEAD_SIZE, dtype=torch.float32, device="cuda")
    PiT = Pi.T.contiguous()
    common = dict(
        kv_cache=kv_cache, block_table=bt, seq_lens=sl,
        Pi=Pi, centroids=centroids, scale=1.0 / (HEAD_SIZE ** 0.5),
        mse_bits=4, key_packed_size=KEY_PACKED, value_quant_bits=4,
        value_packed_size=VAL_PACKED, key_fp8=False,
        norm_correction=False, PiT=PiT, max_seq_len=SEQ,
        max_num_kv_splits=32, sinks=None,
    )
    # warmup / JIT build
    for _ in range(10):
        flydsl_turboquant_decode_attention_v4(query=q_bf16, **common)
    torch.cuda.synchronize()
    print(f"PROF_REGION_START B={B} SEQ={SEQ} QG=6 HK={HK} ITERS={ITERS}", flush=True)
    for _ in range(ITERS):
        flydsl_turboquant_decode_attention_v4(query=q_bf16, **common)
    torch.cuda.synchronize()
    print("PROF_REGION_END", flush=True)


if __name__ == "__main__":
    main()
