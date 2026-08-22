#!/usr/bin/env python3
"""Minimal single-kernel driver for rocprof/omniperf: runs ONLY the fp8_g32
hd256 decode kernel at one context, many iters, so the profiler has a clean
target. No sweep, no parity, no store. Env: SEQ (default 16384), N (default 50).
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, "/shareddata/adrana/workspace/vllm-pr-fp8hd256")
from tests.kernels.turboquant_v4.dev_loop import build_inputs  # noqa: E402
from vllm.v1.attention.ops.flydsl_fp8_g32_decode_v4 import (  # noqa: E402
    flydsl_fp8_g32_decode_attention_v4,
)

SEQ = int(os.environ.get("SEQ", "16384"))
N = int(os.environ.get("N", "50"))


def main():
    common = build_inputs(SEQ, int(os.environ.get("SPLITS", "32")))
    for _ in range(20):
        flydsl_fp8_g32_decode_attention_v4(**common)
    torch.cuda.synchronize()
    for _ in range(N):
        flydsl_fp8_g32_decode_attention_v4(**common)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
