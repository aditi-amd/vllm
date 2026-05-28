#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode-attention microbenchmark: fp4_kv_g32 vs TurboQuant v1.

Measures raw kernel throughput (tokens/s KV processed) for the decode
attention step only — no server, no model load, no prefill.

Default shapes match MiniMax-M2.5:
    Hq=48, Hk=8, head_dim=128, block_size=32

Usage:
    # Default sweep (batch=1,4,16 × seqlen=128,512,2048)
    python benchmarks/bench_fp4_vs_tqv1_decode.py

    # Custom sweep
    python benchmarks/bench_fp4_vs_tqv1_decode.py \\
        --batch 1 4 16 --seqlen 256 1024 4096 --warmup 5 --iters 20

    # fp4_kv_g32 only (skip TQ v1 setup)
    python benchmarks/bench_fp4_vs_tqv1_decode.py --skip-tq

    # JSON output
    python benchmarks/bench_fp4_vs_tqv1_decode.py --json results.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import torch


# ─────────────────────────────────────────────────────────────────────────── #
# Helpers
# ─────────────────────────────────────────────────────────────────────────── #

def _sync():
    """Synchronise the GPU before timing."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif hasattr(torch, "hip") and torch.version.hip is not None:
        torch.cuda.synchronize()  # HIP uses the CUDA runtime API


def _time_kernel(fn, warmup: int, iters: int) -> float:
    """Return mean wall-clock time per call (seconds)."""
    for _ in range(warmup):
        fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync()
    return (time.perf_counter() - t0) / iters


# ─────────────────────────────────────────────────────────────────────────── #
# Synthetic data builders
# ─────────────────────────────────────────────────────────────────────────── #

def _build_tqv1_kv_cache(
    B: int, Hk: int, D: int, block_size: int, seq_lens: torch.Tensor,
    device: torch.device,
):
    """Build a synthetic TQ v1 KV cache (turboquant_4bit_nc layout)."""
    from vllm.model_executor.layers.quantization.turboquant.config import (
        TurboQuantConfig,
    )
    tq_cfg = TurboQuantConfig.from_cache_dtype("turboquant_4bit_nc", D)
    slot = tq_cfg.slot_size_aligned

    max_seq = int(seq_lens.max().item())
    num_blocks_per_seq = math.ceil(max_seq / block_size)
    num_blocks = B * num_blocks_per_seq

    # kv_cache is uint8 but the TQ decode kernel views it as uint16
    assert slot % 2 == 0, "TQ slot must be even for uint16 view"
    kv_cache = torch.randint(0, 256, (num_blocks, block_size, Hk, slot),
                             dtype=torch.uint8, device=device)
    block_table = torch.arange(
        B * num_blocks_per_seq, dtype=torch.int32, device=device
    ).view(B, num_blocks_per_seq)
    return kv_cache, block_table, tq_cfg


def _build_fp4_kv_cache(
    B: int, Hk: int, D: int, block_size: int, seq_lens: torch.Tensor,
    device: torch.device,
):
    """Build a synthetic fp4_kv_g32 KV cache (144-byte slot for D=128)."""
    from vllm.v1.attention.ops.fp4_g32.fp4_levels import slot_size
    slot = slot_size(D)

    max_seq = int(seq_lens.max().item())
    num_blocks_per_seq = math.ceil(max_seq / block_size)
    num_blocks = B * num_blocks_per_seq

    kv_cache = torch.randint(0, 256, (num_blocks, block_size, Hk, slot),
                             dtype=torch.uint8, device=device)
    block_table = torch.arange(
        B * num_blocks_per_seq, dtype=torch.int32, device=device
    ).view(B, num_blocks_per_seq)
    return kv_cache, block_table


# ─────────────────────────────────────────────────────────────────────────── #
# Benchmark runners
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class BenchResult:
    label: str
    B: int
    seqlen: int
    Hq: int
    Hk: int
    D: int
    mean_ms: float
    tokens_per_sec: float   # total KV tokens processed per second
    qk_macs: float          # QK dot-product MACs for reference

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "batch": self.B,
            "seqlen": self.seqlen,
            "Hq": self.Hq,
            "Hk": self.Hk,
            "D": self.D,
            "mean_ms": round(self.mean_ms * 1e3, 3),
            "tokens_per_sec": round(self.tokens_per_sec, 1),
            "qk_gflops": round(self.qk_macs * 2 / 1e9, 4),
        }


def run_tqv1(
    B: int, seqlen: int,
    Hq: int = 48, Hk: int = 8, D: int = 128,
    block_size: int = 32,
    warmup: int = 5, iters: int = 20,
    device: torch.device = None,
) -> BenchResult:
    from vllm.v1.attention.ops.triton_turboquant_decode import (
        triton_turboquant_decode_attention,
    )

    if device is None:
        device = torch.device("cuda:0")

    seq_lens = torch.full((B,), seqlen, dtype=torch.int32, device=device)
    query = torch.randn(B, Hq, D, dtype=torch.float16, device=device)
    Pi = torch.eye(D, dtype=torch.float32, device=device)
    PiT = Pi.T.contiguous()

    kv_cache, block_table, tq_cfg = _build_tqv1_kv_cache(
        B, Hk, D, block_size, seq_lens, device
    )

    from vllm.model_executor.layers.quantization.turboquant.centroids import (
        get_centroids,
    )
    centroids = get_centroids(D, tq_cfg.centroid_bits).to(device)

    def _call():
        return triton_turboquant_decode_attention(
            query=query,
            kv_cache=kv_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            Pi=Pi,
            centroids=centroids,
            scale=1.0 / math.sqrt(D),
            mse_bits=tq_cfg.mse_bits,
            key_packed_size=tq_cfg.key_packed_size,
            value_quant_bits=tq_cfg.value_quant_bits,
            key_fp8=False,
            norm_correction=tq_cfg.norm_correction,
            PiT=PiT,
        )

    mean_s = _time_kernel(_call, warmup, iters)
    total_kv_tokens = B * seqlen
    qk_macs = float(B * Hq * seqlen * D)  # B * Hq * S * D per QK^T
    return BenchResult(
        label="tq_v1",
        B=B, seqlen=seqlen, Hq=Hq, Hk=Hk, D=D,
        mean_ms=mean_s,
        tokens_per_sec=total_kv_tokens / mean_s,
        qk_macs=qk_macs,
    )


def run_fp4_kv_g32(
    B: int, seqlen: int,
    Hq: int = 48, Hk: int = 8, D: int = 128,
    block_size: int = 32,
    warmup: int = 5, iters: int = 20,
    device: torch.device = None,
) -> BenchResult:
    from vllm.v1.attention.ops.fp4_g32.triton_decode import (
        fp4_g32_decode_attention,
        _get_pit,
    )

    if device is None:
        device = torch.device("cuda:0")

    seq_lens = torch.full((B,), seqlen, dtype=torch.int32, device=device)
    query = torch.randn(B, Hq, D, dtype=torch.bfloat16, device=device)
    PiT = _get_pit(D, device, torch.float32)

    kv_cache, block_table = _build_fp4_kv_cache(
        B, Hk, D, block_size, seq_lens, device
    )

    def _call():
        return fp4_g32_decode_attention(
            query=query,
            kv_cache=kv_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            scale=1.0 / math.sqrt(D),
            PiT=PiT,
        )

    mean_s = _time_kernel(_call, warmup, iters)
    total_kv_tokens = B * seqlen
    qk_macs = float(B * Hq * seqlen * D)
    return BenchResult(
        label="fp4_kv_g32",
        B=B, seqlen=seqlen, Hq=Hq, Hk=Hk, D=D,
        mean_ms=mean_s,
        tokens_per_sec=total_kv_tokens / mean_s,
        qk_macs=qk_macs,
    )


# ─────────────────────────────────────────────────────────────────────────── #
# Reporting
# ─────────────────────────────────────────────────────────────────────────── #

def _print_table(results: list[BenchResult]):
    col_w = 16
    header = (
        f"  {'Kernel':<14} {'B':>4} {'Slen':>6} "
        f"{'ms/call':>9} {'TPS':>12} {'Δ TPS':>9}"
    )
    sep = "-" * len(header)
    print()
    print("=" * len(header))
    print("  Decode attention kernel benchmark — MiniMax-M2.5 shapes")
    print("  (Hq=48, Hk=8, head_dim=128, block_size=32)")
    print(sep)
    print(header)
    print(sep)

    # Group by (B, seqlen) for delta computation
    grouped: dict[tuple[int,int], dict[str, BenchResult]] = {}
    for r in results:
        key = (r.B, r.seqlen)
        if key not in grouped:
            grouped[key] = {}
        grouped[key][r.label] = r

    for (B, seqlen), legs in sorted(grouped.items()):
        tq   = legs.get("tq_v1")
        fp4  = legs.get("fp4_kv_g32")
        for label in ("tq_v1", "fp4_kv_g32"):
            r = legs.get(label)
            if r is None:
                continue
            if tq and fp4:
                if label == "fp4_kv_g32":
                    delta = f"{100*(fp4.tokens_per_sec/tq.tokens_per_sec - 1):+.1f}%"
                else:
                    delta = "baseline"
            else:
                delta = "n/a"
            print(
                f"  {r.label:<14} {r.B:>4} {r.seqlen:>6} "
                f"{r.mean_ms*1e3:>8.2f}ms {r.tokens_per_sec:>12,.0f} {delta:>9}"
            )
        if len(grouped) > 1:
            print(sep)

    print("=" * len(header))
    print()


# ─────────────────────────────────────────────────────────────────────────── #
# main
# ─────────────────────────────────────────────────────────────────────────── #

def main():
    parser = argparse.ArgumentParser(
        description="Decode attention microbenchmark: fp4_kv_g32 vs TQ v1"
    )
    parser.add_argument("--batch",   type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--seqlen",  type=int, nargs="+", default=[128, 512, 2048])
    parser.add_argument("--hq",      type=int, default=48, help="# Q heads")
    parser.add_argument("--hk",      type=int, default=8,  help="# KV heads")
    parser.add_argument("--head-dim",type=int, default=128, dest="head_dim")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--warmup",  type=int, default=5)
    parser.add_argument("--iters",   type=int, default=20)
    parser.add_argument("--skip-tq",   action="store_true",
                        help="Skip TQ v1 (e.g. centroids not available)")
    parser.add_argument("--skip-fp4",  action="store_true",
                        help="Skip fp4_kv_g32")
    parser.add_argument("--gpu",     type=int, default=0, help="CUDA device index")
    parser.add_argument("--json",    type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: no CUDA/ROCm device found", file=sys.stderr)
        sys.exit(1)

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    print(f"Device: {torch.cuda.get_device_name(args.gpu)}")
    print(f"Shapes: Hq={args.hq}, Hk={args.hk}, D={args.head_dim}, "
          f"block_size={args.block_size}")
    print(f"Sweep:  batch={args.batch}  seqlen={args.seqlen}")
    print(f"Timing: {args.warmup} warmup + {args.iters} timed iters")

    all_results: list[BenchResult] = []
    common = dict(
        Hq=args.hq, Hk=args.hk, D=args.head_dim,
        block_size=args.block_size,
        warmup=args.warmup, iters=args.iters,
        device=device,
    )

    for B in args.batch:
        for seqlen in args.seqlen:
            print(f"\n  B={B}, seqlen={seqlen} ...", flush=True)
            if not args.skip_tq:
                try:
                    r = run_tqv1(B=B, seqlen=seqlen, **common)
                    all_results.append(r)
                    print(f"    tq_v1        {r.mean_ms*1e3:8.2f} ms  "
                          f"{r.tokens_per_sec:,.0f} tok/s")
                except Exception as e:
                    print(f"    tq_v1 FAILED: {e}")

            if not args.skip_fp4:
                try:
                    r = run_fp4_kv_g32(B=B, seqlen=seqlen, **common)
                    all_results.append(r)
                    print(f"    fp4_kv_g32   {r.mean_ms*1e3:8.2f} ms  "
                          f"{r.tokens_per_sec:,.0f} tok/s")
                except Exception as e:
                    print(f"    fp4_kv_g32 FAILED: {e}")

    _print_table(all_results)

    if args.json:
        out = [r.as_dict() for r in all_results]
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Results saved to {args.json}")


if __name__ == "__main__":
    main()
