#!/usr/bin/env python3
"""Fast offline microbench + component profiler for the fp8_g32 hd256 decode
kernel (Qwen3.8 per-GPU shape under TP=8: Hq=8, Hk=1, QG=8, D=256, BS=32).

Timing depends only on tensor shapes / seq_lens, NOT on the packed byte values,
so we skip the slow O(B*seq) encode/scatter and fill the cache with random
bytes. Correctness is covered separately by test_fp8_g32_v4_parity.py.

Attributes per-call GPU time to: FlyDSL decode kernel, the split-KV partition
reducer, and the q-rotation GEMM/copies.

Usage:
  HIP_VISIBLE_DEVICES=0 VLLM_FP8_G32_DECODE_V4=1 \
  VLLM_FP8_G32_DECODE_V4_QK_SCALED=1 VLLM_FP8_G32_V3=1 \
  VLLM_FLYDSL_ROOT=/root/FlyDSL \
  VLLM_FLYDSL_PKGS=/root/FlyDSL/build-fly/python_packages \
  python tests/kernels/turboquant_v4/bench_fp8_g32_hd256_decode.py
"""
from __future__ import annotations

import os
import sys

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, "/shareddata/adrana/workspace/vllm-pr-fp8hd256")

from vllm.v1.attention.ops.fp8_g32.fp8_levels import get_group_size, slot_size
from vllm.v1.attention.ops.fp8_g32.reference import hadamard_matrix
from vllm.v1.attention.ops.flydsl_fp8_g32_decode_v4 import (
    flydsl_fp8_g32_decode_attention_v4,
    is_flydsl_available,
    is_flydsl_fp8_hd256_available,
)

D = 256
QG = int(os.environ.get("QG", "8"))       # per-GPU: 8 Q heads / 1 KV head
HK = int(os.environ.get("HK", "1"))
BS = int(os.environ.get("BS", "32"))
B = int(os.environ.get("B", "16"))         # concurrency (num_seqs)
KV_SPLITS = int(os.environ.get("KV_SPLITS", "32"))


def build_inputs(seq_len: int, device="cuda"):
    Hk = HK
    Hq = Hk * QG
    gs = get_group_size()
    padded_slot = slot_size(D, gs)
    blocks_per_seq = (seq_len + BS - 1) // BS
    max_bps = blocks_per_seq + 4
    total_blocks = B * blocks_per_seq + 8

    q = (torch.randn(B, Hq, D, device=device, dtype=torch.float32) * 0.5).to(
        torch.bfloat16)
    kv_cache = torch.randint(
        0, 256, (total_blocks, BS, Hk, padded_slot), dtype=torch.uint8,
        device=device)
    bt = torch.zeros(B, max_bps, dtype=torch.int32, device=device)
    bt[:, :blocks_per_seq] = (
        torch.arange(B * blocks_per_seq, dtype=torch.int32, device=device)
        .reshape(B, blocks_per_seq) + 1)
    sl = torch.full((B,), seq_len, dtype=torch.int32, device=device)
    PiT = hadamard_matrix(D, torch.device(device), torch.float32).contiguous()
    return q, kv_cache, bt, sl, PiT


def run(seq_len: int, iters: int = 100):
    dev = "cuda"
    scale = 1.0 / (D ** 0.5)
    q, kv_cache, bt, sl, PiT = build_inputs(seq_len, dev)
    common = dict(kv_cache=kv_cache, block_table=bt, seq_lens=sl, scale=scale,
                  PiT=PiT, max_seq_len=seq_len, max_num_kv_splits=KV_SPLITS,
                  sinks=None)
    for _ in range(30):
        flydsl_fp8_g32_decode_attention_v4(query=q, **common)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        for _ in range(iters):
            flydsl_fp8_g32_decode_attention_v4(query=q, **common)
        torch.cuda.synchronize()

    rows = []
    for e in p.key_averages():
        dev_us = getattr(e, "device_time_total", 0.0) or getattr(
            e, "cuda_time_total", 0.0)
        if dev_us and dev_us > 0:
            rows.append((dev_us / iters, e.key, e.count / iters))
    rows.sort(reverse=True)
    tot = sum(r[0] for r in rows)
    print(f"\n=== fp8_g32 hd256 decode  B={B} Hk={HK} QG={QG} D={D} BS={BS} "
          f"seq={seq_len} splits={KV_SPLITS}  ({iters} iters) ===")
    print(f"{'us/call':>10} {'pct':>6} {'launch/call':>11}  name")
    print("-" * 84)
    for us, name, cnt in rows[:14]:
        print(f"{us:10.2f} {100*us/tot:5.1f}% {cnt:11.2f}  {name[:56]}")
    print("-" * 84)
    print(f"{tot:10.2f}  total device us/call   (seq={seq_len})")
    return seq_len, tot


def main():
    assert torch.cuda.is_available(), "needs a GPU"
    assert is_flydsl_available(), "FlyDSL not importable"
    assert is_flydsl_fp8_hd256_available(QG), f"hd256 QG={QG} sibling unavailable"
    seqs = [int(x) for x in os.environ.get("SEQS", "8192,16384,32768").split(",")]
    summary = [run(s) for s in seqs]
    print("\n===== total device us/call vs context =====")
    for s, t in summary:
        print(f"  seq={s:>6}: {t:8.2f} us/call")


if __name__ == "__main__":
    main()
