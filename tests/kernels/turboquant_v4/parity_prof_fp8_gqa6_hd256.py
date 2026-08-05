#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Correctness + isolated-profile harness for the fp8_g32 GQA-6 hd256 v4 kernel.

Adapted from test_fp8_g32_v4_parity.py to HEAD_SIZE=256 / GQA-6 (Qwen3.6-27B
full-attention shape). Two modes:
  * default: correctness (cos vs PyTorch golden) across a few shapes.
  * PROF=1 : single-shape tight loop over the launcher only (for rocprofv3
             --pmc). Set B/SEQ/HK/ITERS via env.

Usage:
  correctness: HIP_VISIBLE_DEVICES=2 python parity_prof_fp8_gqa6_hd256.py
  profile    : PROF=1 B=64 SEQ=1024 ITERS=40 HIP_VISIBLE_DEVICES=2 \
               python parity_prof_fp8_gqa6_hd256.py
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, "/shareddata/adrana/workspace/vllm-pr-fp8hd256")

from vllm.v1.attention.ops.fp8_g32.fp8_levels import (  # noqa: E402
    get_constant_c, get_group_size, is_arch_b,
    k_scales_offset, slot_size, v_codes_offset, v_scales_offset,
)
from vllm.v1.attention.ops.fp8_g32.reference import (  # noqa: E402
    fp8_g32_encode, hadamard_matrix, reference_fp8_g32_attention,
)
from vllm.v1.attention.ops.flydsl_fp8_g32_decode_v4 import (  # noqa: E402
    flydsl_fp8_g32_decode_attention_v4, is_flydsl_available,
)

HEAD_SIZE = 256
KV_BLOCK_SIZE = int(os.environ.get("BS", "32"))
QG = int(os.environ.get("QG", "6"))


def _cos(a, b):
    a = a.float().reshape(-1); b = b.float().reshape(-1)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def build_cache(num_seqs, Hk, seq_len, max_bps, seed=0xC0FFEE, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    Hq = Hk * QG
    D = HEAD_SIZE
    gs = get_group_size()
    code_bytes = D // 2                     # FP4 = 4 bits/elem
    q = (torch.randn(num_seqs, Hq, D, generator=g, dtype=torch.float32,
                     device=device) * 0.5)
    q_bf16 = q.to(torch.bfloat16)
    K_raw = (torch.randn(num_seqs, seq_len, Hk, D, generator=g,
                         dtype=torch.float32, device=device) * 0.6).to(torch.bfloat16)
    V_raw = (torch.randn(num_seqs, seq_len, Hk, D, generator=g,
                         dtype=torch.float32, device=device) * 0.6).to(torch.bfloat16)
    enc_k = fp8_g32_encode(K_raw.transpose(1, 2).contiguous(), rotate=True)
    enc_v = fp8_g32_encode(V_raw.transpose(1, 2).contiguous(), rotate=False)
    k_codes, k_scales = enc_k.codes_packed, enc_k.scale_bytes
    v_codes, v_scales = enc_v.codes_packed, enc_v.scale_bytes

    padded_slot = slot_size(D, gs)
    KSC = k_scales_offset(D, gs)
    VCO = v_codes_offset(D, gs)
    VSC = v_scales_offset(D, gs)
    n_groups = D // gs

    blocks_per_seq = (seq_len + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE
    bt = torch.zeros(num_seqs, max_bps, dtype=torch.int32, device=device)
    bt[:, :blocks_per_seq] = (
        torch.arange(num_seqs * blocks_per_seq, dtype=torch.int32,
                     device=device).reshape(num_seqs, blocks_per_seq) + 1
    )
    total_blocks = num_seqs * blocks_per_seq + 8
    kv_cache = torch.zeros(total_blocks, KV_BLOCK_SIZE, Hk, padded_slot,
                           dtype=torch.uint8, device=device)
    for s in range(num_seqs):
        for t in range(seq_len):
            blk = int(bt[s, t // KV_BLOCK_SIZE].item())
            slot = t % KV_BLOCK_SIZE
            cell = kv_cache[blk, slot]
            cell[:, 0:code_bytes] = k_codes[s, :, t]
            cell[:, KSC:KSC + n_groups] = k_scales[s, :, t]
            cell[:, VCO:VCO + code_bytes] = v_codes[s, :, t]
            cell[:, VSC:VSC + n_groups] = v_scales[s, :, t]
    sl = torch.full((num_seqs,), seq_len, dtype=torch.int32, device=device)
    return q_bf16, kv_cache, bt, sl, K_raw, V_raw


def run_case(num_seqs, Hk, seq_len, timed=False, iters=40):
    D = HEAD_SIZE
    scale = 1.0 / (D ** 0.5)
    max_bps = (seq_len + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE + 4
    q_bf16, kv_cache, bt, sl, K_raw, V_raw = build_cache(
        num_seqs, Hk, seq_len, max_bps)
    PiT = hadamard_matrix(D, torch.device("cuda"), torch.float32).contiguous()
    common = dict(kv_cache=kv_cache, block_table=bt, seq_lens=sl, scale=scale,
                  PiT=PiT, max_seq_len=seq_len, buf_holder=None,
                  max_num_kv_splits=32, sinks=None)
    if timed:
        for _ in range(10):
            flydsl_fp8_g32_decode_attention_v4(query=q_bf16, **common)
        torch.cuda.synchronize()
        print(f"PROF_REGION_START B={num_seqs} SEQ={seq_len} QG={QG} "
              f"HK={Hk} ITERS={iters}", flush=True)
        for _ in range(iters):
            flydsl_fp8_g32_decode_attention_v4(query=q_bf16, **common)
        torch.cuda.synchronize()
        print("PROF_REGION_END", flush=True)
        return None
    out_ref = reference_fp8_g32_attention(
        query=q_bf16, key=K_raw, value=V_raw, scale=scale,
        constant_c=get_constant_c(), arch_b=is_arch_b())
    out_v4 = flydsl_fp8_g32_decode_attention_v4(query=q_bf16, **common)
    cos = _cos(out_v4.cpu(), out_ref.cpu())
    mad = (out_v4.cpu().float() - out_ref.cpu().float()).abs().max().item()
    return cos, mad


def main():
    assert is_flydsl_available(), "FlyDSL not importable"
    print(f"=== fp8_g32 GQA-6 hd256 (BS={KV_BLOCK_SIZE} QG={QG} "
          f"arch_b={is_arch_b()} c={get_constant_c():.4f}) ===", flush=True)
    if os.environ.get("PROF", "0") == "1":
        run_case(int(os.environ.get("B", "64")),
                 int(os.environ.get("HK", "4")),
                 int(os.environ.get("SEQ", "1024")),
                 timed=True, iters=int(os.environ.get("ITERS", "40")))
        return
    print(f"{'B':>4} {'Hk':>3} {'seq':>6}  {'cos(ref)':>10} {'max|Δ|':>10}")
    print("-" * 40)
    worst = 1.0
    for num_seqs, Hk, seq_len in [(1, 4, 256), (2, 4, 1024), (4, 4, 512)]:
        cos, mad = run_case(num_seqs, Hk, seq_len)
        worst = min(worst, cos)
        print(f"{num_seqs:>4} {Hk:>3} {seq_len:>6}  {cos:>10.6f} {mad:>10.4e}")
    print(f"\nworst cos(ref) = {worst:.6f}  "
          f"{'PASS' if worst > 0.99 else 'FAIL'} (tol 0.99)")


if __name__ == "__main__":
    main()
