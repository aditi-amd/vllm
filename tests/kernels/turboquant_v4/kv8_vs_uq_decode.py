#!/usr/bin/env python3
"""Head-to-head decode benchmark: UQ (fp8_g32 FlyDSL) vs KV8 (Triton unified 3D).

Motivation
----------
The Qwen3.8 Perfetto traces show UQ decode at 651.7 us/call while KV8's pure-decode
kernel (kernel_unified_attention_3d_num_query_heads_8) shows a bimodal
69 x 1564us / 184 x 29.3us. Those means are NOT comparable: the traces don't record
per-call context/batch, so we can't tell which calls ran at which context. Trace
archaeology cannot answer "is UQ as fast as KV8".

This bench removes the ambiguity by running BOTH kernels at IDENTICAL shapes
(same batch, same context, same head config) so the comparison is apples-to-apples
and iterable in seconds.

Per-rank Qwen3.8 decode config (matches the traces: num_query_heads_8, hd256):
  Hq=8 query heads, Hk=1 KV head (GQA-8), head_size=256, 1 query token/seq.

KV8 leg  : vLLM default fp8 KV cache (per-tensor) via the Triton unified 3D kernel.
UQ leg   : fp8_g32 (4-bit KV + UE8M0 group scales) via the FlyDSL decode kernel.

Note UQ reads ~half the KV bytes of KV8, so at equal efficiency UQ should WIN.

Usage:
  HIP_VISIBLE_DEVICES=0 VLLM_FP8_G32_V3=1 VLLM_FP8_G32_DECODE_V4=1 \
  VLLM_FP8_G32_DECODE_V4_QK_SCALED=1 VLLM_FLYDSL_ROOT=/root/FlyDSL \
  VLLM_FLYDSL_PKGS=/root/FlyDSL/build-fly/python_packages \
  python tests/kernels/turboquant_v4/kv8_vs_uq_decode.py

Env: SEQS=16384,32768,131072  B=16  SEGS=16 (KV8 softmax segments)  ITERS=30
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, "/shareddata/adrana/workspace/vllm-pr-fp8hd256")

from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import KVQuantMode

HQ = int(os.environ.get("HQ", "8"))       # query heads per rank
HK = int(os.environ.get("HK", "1"))       # kv heads per rank
D = int(os.environ.get("D", "256"))
B = int(os.environ.get("B", "16"))
ITERS = int(os.environ.get("ITERS", "30"))
SEGS = int(os.environ.get("SEGS", "16"))
KV8_BLOCK = int(os.environ.get("KV8_BLOCK", "16"))
DEV = "cuda"
FP8 = torch.float8_e4m3fnuz if torch.version.hip else torch.float8_e4m3fn


def _next_pow2(x):
    return 1 << (x - 1).bit_length()


# ---------------- KV8 (Triton unified attention, fp8 KV) ---------------------
def kv8_setup(seq):
    bps = (seq + KV8_BLOCK - 1) // KV8_BLOCK
    num_blocks = B * bps + 8
    q = torch.randn(B, HQ, D, device=DEV, dtype=torch.bfloat16)
    kc = torch.randn(num_blocks, KV8_BLOCK, HK, D, device=DEV,
                     dtype=torch.bfloat16).to(FP8)
    vc = torch.randn(num_blocks, KV8_BLOCK, HK, D, device=DEV,
                     dtype=torch.bfloat16).to(FP8)
    out = torch.empty(B, HQ, D, device=DEV, dtype=torch.bfloat16)
    # decode: exactly 1 query token per sequence
    cu_q = torch.arange(B + 1, device=DEV, dtype=torch.int32)
    kv_lens = torch.full((B,), seq, device=DEV, dtype=torch.int32)
    bt = (torch.arange(B * bps, device=DEV, dtype=torch.int32)
          .reshape(B, bps) % num_blocks)
    dpad = _next_pow2(D)
    # 3D path requires num_seqs <= seq_threshold_3D and max_seqlen_q == 1
    thr = B
    segm_out = torch.empty((thr, HQ, SEGS, dpad), device=DEV, dtype=torch.float32)
    segm_max = torch.empty((thr, HQ, SEGS), device=DEV, dtype=torch.float32)
    segm_sum = torch.empty((thr, HQ, SEGS), device=DEV, dtype=torch.float32)
    one = torch.ones(1, device=DEV, dtype=torch.float32)
    return dict(q=q, k=kc, v=vc, out=out, cu_seqlens_q=cu_q, max_seqlen_q=1,
                seqused_k=kv_lens, max_seqlen_k=seq, softmax_scale=D ** -0.5,
                causal=True, window_size=(-1, -1), block_table=bt, softcap=0,
                q_descale=None, k_descale=one, v_descale=one,
                seq_threshold_3D=thr, num_par_softmax_segments=SEGS,
                softmax_segm_output=segm_out, softmax_segm_max=segm_max,
                softmax_segm_expsum=segm_sum,
                kv_quant_mode=KVQuantMode.FP8_PER_TENSOR)


def kv8_bytes(seq):
    # fp8: 1 byte/elem, K and V, all kv heads
    return B * seq * HK * D * 2 * 1


# ---------------- UQ (FlyDSL fp8_g32) --------------------------------------
def uq_setup(seq, splits):
    from vllm.v1.attention.ops.fp8_g32.fp8_levels import get_group_size, slot_size
    from vllm.v1.attention.ops.fp8_g32.reference import hadamard_matrix
    import types
    BS = int(os.environ.get("BS", "32"))
    gs = get_group_size()
    padded = slot_size(D, gs)
    bps = (seq + BS - 1) // BS
    total = B * bps + 8
    q = (torch.randn(B, HQ, D, device=DEV) * 0.5).to(torch.bfloat16)
    kv = torch.randint(0, 256, (total, BS, HK, padded), dtype=torch.uint8, device=DEV)
    bt = torch.zeros(B, bps + 4, dtype=torch.int32, device=DEV)
    bt[:, :bps] = (torch.arange(B * bps, dtype=torch.int32, device=DEV)
                   .reshape(B, bps) + 1)
    sl = torch.full((B,), seq, dtype=torch.int32, device=DEV)
    PiT = hadamard_matrix(D, torch.device(DEV), torch.float32).contiguous()
    return dict(query=q, kv_cache=kv, block_table=bt, seq_lens=sl,
                scale=D ** -0.5, PiT=PiT, max_seq_len=seq,
                max_num_kv_splits=splits, sinks=None,
                buf_holder=types.SimpleNamespace())


def uq_bytes(seq):
    # fp8_g32: 4-bit codes + UE8M0 group scales ~= 0.5 + 1/32 bytes per elem
    return int(B * seq * HK * D * 2 * (0.5 + 1.0 / 32))


def timeit(fn, iters=ITERS):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000.0 / iters   # us


def main():
    seqs = [int(x) for x in os.environ.get("SEQS", "16384,32768,131072").split(",")]
    splits = int(os.environ.get("SPLITS", "256"))
    from vllm.v1.attention.ops.flydsl_fp8_g32_decode_v4 import (
        flydsl_fp8_g32_decode_attention_v4 as uq_attn,
    )
    print(f"config: B={B} Hq={HQ} Hk={HK} D={D} kv8_block={KV8_BLOCK} "
          f"segs={SEGS} uq_splits={splits} iters={ITERS}")
    print(f"{'seq':>8} {'KV8 us':>9} {'UQ us':>9} {'UQ/KV8':>7} "
          f"{'KV8 GB/s':>9} {'UQ GB/s':>9} {'KV8 MB':>8} {'UQ MB':>7}")
    for sq in seqs:
        ka = kv8_setup(sq)
        t_kv8 = timeit(lambda: unified_attention(**ka))
        ua = uq_setup(sq, splits)
        t_uq = timeit(lambda: uq_attn(**ua))
        kb, ub = kv8_bytes(sq), uq_bytes(sq)
        print(f"{sq:>8} {t_kv8:>9.2f} {t_uq:>9.2f} {t_uq / t_kv8:>7.2f}x "
              f"{kb / (t_kv8 * 1e3):>9.0f} {ub / (t_uq * 1e3):>9.0f} "
              f"{kb / 1e6:>8.1f} {ub / 1e6:>7.1f}")


if __name__ == "__main__":
    main()
