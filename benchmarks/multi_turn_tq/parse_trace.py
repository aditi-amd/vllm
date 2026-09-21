#!/usr/bin/env python3
"""Attribute GPU kernel time in a vLLM torch-profiler trace.

Reads the Chrome/Perfetto trace(s) produced by a server launched with
--profiler-config '{"profiler":"torch",...}' and sums GPU-kernel durations,
bucketed into: dequant / attention / moe / gemm / other. The point is to size
the *prefill dequant tax* (bucket "dequant") relative to attention and the rest,
and to compare the UQ leg (visible bulk _tq_full_dequant_kv) against the KV8 leg
(dequant fused in attention, so ~no dequant bucket).

Usage:
    python parse_trace.py <trace_dir_or_file> [--top 40]

Accepts a directory (all *.json / *.json.gz inside are aggregated) or a single
trace file. No third-party deps.
"""
import argparse
import glob
import gzip
import json
import os
import re
from collections import defaultdict

# Kernel-name buckets. Order matters: first match wins. Patterns are broad on
# purpose because FlyDSL/HIP kernel names are often opaque — inspect the per-kernel
# top-N table (printed below) and refine these if a hot kernel lands in "other".
BUCKETS = [
    # KV dequant + the K-side inverse-Hadamard transform that only exists because KV is quantized.
    ("dequant", re.compile(r"full_dequant|_dequant_kv|unpack|cvt_scalef32|scalef32_pk", re.I)),
    # Full (softmax) attention: prefill FMHA + fp8_g32/hd256 decode + vLLM unified/paged attention.
    ("full_attn", re.compile(r"fmha|flash_?attn|fp8_g32_decode|hd256|kernel_unified_attention|paged_attn|paged_attention", re.I)),
    # Gated-DeltaNet linear-attention layers (48/64 layers on qwen3_5). merge_16x16_to_64x64_inverse
    # is the chunk-state merge (present in BOTH legs, so it is NOT KV dequant).
    ("linear_attn", re.compile(r"gated_delta|conv1d|chunk_fwd|chunk_gated|recompute_w_u|post_conv|merge_16x16_to_64x64", re.I)),
    # MoE FFN.
    ("moe", re.compile(r"\bmoe\b|expert|topk|grouped_gemm|act_and_mul|silu", re.I)),
    # Dense GEMMs: rocBLAS Cijk + skinny wvSplitK + fused triton matmuls.
    ("gemm", re.compile(r"gemm|cijk|cutlass|wvsplitk|split_?k|hgemm|f8f6f4|\bmfma\b", re.I)),
    ("norm_rope", re.compile(r"rms_norm|layernorm|rotary|rope|embedding", re.I)),
    ("memory", re.compile(r"copybuffer|memcpy|copy_kernel|batch_memcpy", re.I)),
]


def bucket_of(name: str) -> str:
    for label, rx in BUCKETS:
        if rx.search(name):
            return label
    return "other"


def is_gpu_kernel(ev: dict) -> bool:
    # torch profiler chrome trace: GPU kernels are complete events (ph == "X")
    # categorised as "kernel" (roctracer/kineto). Exclude host ops and memcpy.
    if ev.get("ph") != "X":
        return False
    cat = str(ev.get("cat", "")).lower()
    return cat in ("kernel", "gpu_kernel") or "kernel" in cat


def load_events(path: str):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("traceEvents", [])
    return data


def collect_files(target: str):
    if os.path.isdir(target):
        files = sorted(
            glob.glob(os.path.join(target, "*.json"))
            + glob.glob(os.path.join(target, "*.json.gz"))
            + glob.glob(os.path.join(target, "*.pt.trace.json"))
            + glob.glob(os.path.join(target, "*.pt.trace.json.gz"))
        )
        if not files:
            raise SystemExit(f"no trace files (*.json / *.json.gz) under {target}")
        return files
    return [target]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="trace directory or single trace file")
    ap.add_argument("--top", type=int, default=40, help="show top-N kernels by total time")
    args = ap.parse_args()

    per_kernel_time = defaultdict(float)
    per_kernel_count = defaultdict(int)

    for fp in collect_files(args.target):
        try:
            events = load_events(fp)
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] could not read {fp}: {e}")
            continue
        n = 0
        for ev in events:
            if is_gpu_kernel(ev):
                name = ev.get("name", "<unknown>")
                per_kernel_time[name] += float(ev.get("dur", 0.0))
                per_kernel_count[name] += 1
                n += 1
        print(f"  parsed {os.path.basename(fp)}: {n} gpu-kernel events")

    total = sum(per_kernel_time.values())
    if total <= 0:
        raise SystemExit(
            "no GPU-kernel events found. Check that the server used "
            "--profiler-config profiler=torch and that /stop_profile flushed."
        )

    # bucket totals
    bucket_time = defaultdict(float)
    for name, t in per_kernel_time.items():
        bucket_time[bucket_of(name)] += t

    us = 1.0
    ms = 1000.0
    print("\n================ GPU-time by bucket ================")
    print(f"{'bucket':<12}{'time (ms)':>14}{'% of GPU':>12}")
    order = ["dequant", "full_attn", "linear_attn", "moe", "gemm", "norm_rope", "memory", "other"]
    for b in order:
        t = bucket_time.get(b, 0.0)
        print(f"{b:<12}{t/ms:>14.3f}{100*t/total:>11.2f}%")
    print(f"{'TOTAL':<12}{total/ms:>14.3f}{100.0:>11.2f}%")

    # headline ratios
    deq = bucket_time.get("dequant", 0.0)
    attn = bucket_time.get("full_attn", 0.0)
    print("\n---------------- headline ratios ----------------")
    print(f"dequant / total GPU       : {100*deq/total:.2f}%")
    denom = deq + attn
    if denom > 0:
        print(f"dequant / (dequant+attn)  : {100*deq/denom:.2f}%")
    print("  (UQ: expect a non-zero dequant bucket; KV8: expect ~0 — fused in attention)")

    print(f"\n================ top {args.top} kernels ================")
    print(f"{'time(ms)':>10}{'  cnt':>7}{'  bucket':>10}  name")
    for name, t in sorted(per_kernel_time.items(), key=lambda kv: kv[1], reverse=True)[: args.top]:
        print(f"{t/ms:>10.3f}{per_kernel_count[name]:>7}{bucket_of(name):>10}  {name[:90]}")


if __name__ == "__main__":
    main()
