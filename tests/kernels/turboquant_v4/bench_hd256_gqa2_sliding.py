#!/usr/bin/env python3
"""Microbench: FlyDSL hd256 QG=2 decode vs Triton v3, on the Gemma 4 sliding-layer shape.

Purpose (pre-implementation go/no-go for FlyDSL-on-Gemma):
  Gemma 4's 50 sliding layers are D=256, num_kv_heads=16, GQA=2, window=1024.
  A window=1024 layer never reads more than 1024 KV tokens in decode, so timing
  at seq_len<=1024 IS the steady-state sliding-layer cost -- SWA tile-pruning
  changes correctness at long context, not the per-call decode time we measure
  here. D=512 global layers are a separate decision and are not exercised.

  Both kernels read the identical SoA-packed cache; the launcher auto-dispatches
  D=256/QG=2 to build_tq_decode_hd256_module(query_group_size=2). We (1) prove
  the FlyDSL QG=2 output matches an exact fp32 attention reference (a fast-but-
  wrong kernel is a meaningless speed number), then (2) time v3 vs v4.

Adapted from test_v4_vs_v3_fast.py (hd128). Only the per-head SoA byte layout
changes: D=256 => K/V nibble data = 128 B each, slot = 130 + 132 = 262 B.

Usage: HIP_VISIBLE_DEVICES=7 python bench_hd256_gqa2_sliding.py
"""
import os
import sys
import time

import torch

sys.path.insert(0, "/shareddata/adrana/workspace/vllm-pr-fp8hd256")

from vllm.v1.attention.ops.triton_turboquant_unified_attention import (
    triton_turboquant_decode_attention_v3,
)
from vllm.v1.attention.ops.flydsl_turboquant_decode_v4 import (
    flydsl_turboquant_decode_attention_v4,
    is_flydsl_available,
)

HEAD_SIZE = 256
KV_BLOCK_SIZE = int(os.environ.get("BS", "32"))   # Gemma serves block-size 32
N_CENTROIDS = 16
QG = int(os.environ.get("QG", "2"))               # Gemma sliding: 32 q / 16 kv
NUM_KV_HEADS = int(os.environ.get("HK", "16"))

KEY_DATA_BYTES = HEAD_SIZE // 2                    # 128: 4-bit nibbles over 256
VAL_DATA_BYTES = HEAD_SIZE // 2                    # 128
DATA_BYTES_PER_SLOT = KEY_DATA_BYTES + VAL_DATA_BYTES  # 256
# Per-head slot = key_packed(130) + value_packed(132) = 262 (matches the SoA
# store: 128 B nibble data + fp16 meta each side).
KEY_PACKED = KEY_DATA_BYTES + 2                    # 130
VAL_PACKED = VAL_DATA_BYTES + 4                    # 132
SLOT_SIZE = KEY_PACKED + VAL_PACKED                # 262
NUM_SOA_FIELDS = 3
SOA_K_NORM, SOA_V_SCALE, SOA_V_ZERO = 0, 1, 2


def make_inputs(num_seqs, num_kv_heads, seq_len, max_bps, seed=0xC0FFEE,
                device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    Hk = num_kv_heads
    Hq = Hk * QG

    centroids = (torch.randn(N_CENTROIDS, generator=g,
                             dtype=torch.float32, device=device) * 0.5)
    q = torch.randn(num_seqs, Hq, HEAD_SIZE, generator=g,
                    dtype=torch.float32, device=device) * 0.1
    q_bf16 = q.to(torch.bfloat16)

    blocks_per_seq = (seq_len + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE
    bt = torch.zeros(num_seqs, max_bps, dtype=torch.int32, device=device)
    bt[:, :blocks_per_seq] = (
        torch.arange(num_seqs * blocks_per_seq, dtype=torch.int32,
                     device=device).reshape(num_seqs, blocks_per_seq) + 1
    )
    total_blocks = num_seqs * blocks_per_seq + 8
    bytes_per_block = KV_BLOCK_SIZE * Hk * SLOT_SIZE

    k_idx = torch.randint(0, N_CENTROIDS, (num_seqs, Hk, seq_len, HEAD_SIZE),
                          generator=g, dtype=torch.uint8, device=device)
    v_idx = torch.randint(0, N_CENTROIDS, (num_seqs, Hk, seq_len, HEAD_SIZE),
                          generator=g, dtype=torch.uint8, device=device)
    knorm = (torch.rand(num_seqs, Hk, seq_len, generator=g, device=device)
             * 0.5 + 0.5).to(torch.float16)
    vscale = (torch.rand(num_seqs, Hk, seq_len, generator=g, device=device)
              * 0.05 + 0.01).to(torch.float16)
    vzero = ((torch.rand(num_seqs, Hk, seq_len, generator=g, device=device)
              - 0.5) * 0.1).to(torch.float16)

    K_ref = (centroids[k_idx.long()].float() * knorm.float().unsqueeze(-1))
    V_ref = (v_idx.float() * vscale.float().unsqueeze(-1)
             + vzero.float().unsqueeze(-1))

    k_packed = (k_idx[..., 0::2] | (k_idx[..., 1::2] << 4))  # [N,Hk,T,128]
    v_packed = (v_idx[..., 0::2] | (v_idx[..., 1::2] << 4))  # [N,Hk,T,128]

    kv_cache = torch.zeros(total_blocks, bytes_per_block,
                           dtype=torch.uint8, device=device)

    seq_idx = torch.arange(num_seqs, device=device).view(num_seqs, 1).expand(
        num_seqs, seq_len)
    tok_idx = torch.arange(seq_len, device=device).view(1, seq_len).expand(
        num_seqs, seq_len)
    blk_for_tok = bt[seq_idx, tok_idx // KV_BLOCK_SIZE]
    slot_for_tok = tok_idx % KV_BLOCK_SIZE

    h_idx_arr = torch.arange(Hk, device=device)
    base_data = (slot_for_tok.unsqueeze(1) * Hk * DATA_BYTES_PER_SLOT
                 + h_idx_arr.view(1, Hk, 1) * DATA_BYTES_PER_SLOT)
    blk_b = blk_for_tok.unsqueeze(1).expand(num_seqs, Hk, seq_len)
    dst_base = blk_b * bytes_per_block + base_data
    rng_k = torch.arange(KEY_DATA_BYTES, device=device)
    rng_v = torch.arange(VAL_DATA_BYTES, device=device) + KEY_DATA_BYTES
    k_dst = (dst_base.unsqueeze(-1) + rng_k.view(1, 1, 1, -1)).reshape(-1)
    v_dst = (dst_base.unsqueeze(-1) + rng_v.view(1, 1, 1, -1)).reshape(-1)
    flat = kv_cache.view(-1)
    flat[k_dst] = k_packed.reshape(-1)
    flat[v_dst] = v_packed.reshape(-1)

    META_OFF = KV_BLOCK_SIZE * Hk * DATA_BYTES_PER_SLOT
    meta_view = kv_cache.view(torch.float16).view(total_blocks, -1)
    meta_off_hw = META_OFF // 2
    h_arr = h_idx_arr.view(1, Hk, 1)
    slot_arr = slot_for_tok.unsqueeze(1)
    common_off = h_arr * NUM_SOA_FIELDS * KV_BLOCK_SIZE + slot_arr
    blk_flat = blk_for_tok.unsqueeze(1).expand(num_seqs, Hk, seq_len)
    knorm_off = meta_off_hw + common_off + SOA_K_NORM * KV_BLOCK_SIZE
    vscale_off = meta_off_hw + common_off + SOA_V_SCALE * KV_BLOCK_SIZE
    vzero_off = meta_off_hw + common_off + SOA_V_ZERO * KV_BLOCK_SIZE
    meta_view[blk_flat.reshape(-1).long(), knorm_off.reshape(-1).long()] = knorm.reshape(-1)
    meta_view[blk_flat.reshape(-1).long(), vscale_off.reshape(-1).long()] = vscale.reshape(-1)
    meta_view[blk_flat.reshape(-1).long(), vzero_off.reshape(-1).long()] = vzero.reshape(-1)

    sl = torch.full((num_seqs,), seq_len, dtype=torch.int32, device=device)
    kv_cache_4d = kv_cache.view(total_blocks, KV_BLOCK_SIZE, Hk, SLOT_SIZE)
    return centroids, q_bf16, kv_cache_4d, bt, sl, K_ref.cpu(), V_ref.cpu()


def py_reference(q_bf16, K_ref, V_ref, sl, scale):
    num_seqs, Hq, D = q_bf16.shape
    Hk = K_ref.shape[1]
    QG_ = Hq // Hk
    q = q_bf16.float().reshape(num_seqs, Hk, QG_, D)
    out = torch.zeros(num_seqs, Hq, D, dtype=torch.float32)
    for s in range(num_seqs):
        for h in range(Hk):
            ql = int(sl[s].item())
            K = K_ref[s, h, :ql]
            V = V_ref[s, h, :ql]
            qq = q[s, h]
            scores = (qq @ K.T) * scale
            m = scores.max(dim=-1, keepdim=True).values
            e = torch.exp(scores - m)
            p = e / e.sum(dim=-1, keepdim=True)
            out[s, h * QG_:(h + 1) * QG_] = p @ V
    return out


def bench(fn, n_warmup=10, n_iter=100):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iter * 1e6


def run_case(num_seqs, num_kv_heads, seq_len, validate=False):
    max_bps = (seq_len + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE + 4
    centroids, q_bf16, kv_cache, bt, sl, K_ref, V_ref = make_inputs(
        num_seqs, num_kv_heads, seq_len, max_bps
    )
    Pi = torch.eye(HEAD_SIZE, dtype=torch.float32, device="cuda")
    PiT = Pi.T.contiguous()
    common = dict(
        kv_cache=kv_cache, block_table=bt, seq_lens=sl,
        Pi=Pi, centroids=centroids, scale=1.0 / (HEAD_SIZE ** 0.5),
        mse_bits=4, key_packed_size=KEY_PACKED, value_quant_bits=4,
        value_packed_size=VAL_PACKED, key_fp8=False,
        norm_correction=False, PiT=PiT, max_seq_len=seq_len,
        max_num_kv_splits=32, sinks=None,
    )

    o_v3 = triton_turboquant_decode_attention_v3(query=q_bf16, **common)
    o_v4 = flydsl_turboquant_decode_attention_v4(query=q_bf16, **common)
    pair_diff = (o_v3.cpu().float() - o_v4.cpu().float()).abs().max().item()

    if validate:
        ref = py_reference(q_bf16.cpu(), K_ref, V_ref, sl.cpu(),
                           1.0 / (HEAD_SIZE ** 0.5))
        v3_diff = (o_v3.cpu().float() - ref).abs().max().item()
        v4_diff = (o_v4.cpu().float() - ref).abs().max().item()
    else:
        v3_diff = v4_diff = float("nan")

    v3_us = bench(lambda: triton_turboquant_decode_attention_v3(query=q_bf16, **common))
    v4_us = bench(lambda: flydsl_turboquant_decode_attention_v4(query=q_bf16, **common))
    return v3_us, v4_us, pair_diff, v3_diff, v4_diff


def main():
    assert is_flydsl_available(), "FlyDSL not available"
    print(f"=== hd256 QG={QG} Hk={NUM_KV_HEADS} BS={KV_BLOCK_SIZE} "
          f"(Gemma 4 sliding-layer shape) ===")

    print("\n--- Correctness gate (seq=1024, B=2) vs exact fp32 attention ---")
    v3, v4, pair, v3d, v4d = run_case(2, NUM_KV_HEADS, 1024, validate=True)
    print(f"v3 vs ref: {v3d:.4e}")
    print(f"v4 vs ref: {v4d:.4e}   <-- FlyDSL QG=2 correctness")
    print(f"v4 vs v3 : {pair:.4e}")
    TOL = 5e-3
    ok = v4d < TOL
    print(f"CORRECTNESS: {'PASS' if ok else 'FAIL'} (tol {TOL})")
    if not ok:
        print("Stopping: fast-but-wrong is meaningless. Fix correctness first.")
        return
    if os.environ.get("VALIDATE_ONLY", "0") == "1":
        print("VALIDATE_ONLY=1: correctness confirmed, skipping timing sweep.")
        return

    print("\n--- Timing sweep (decode; seq_len<=1024 = sliding-window regime) ---")
    print(f"{'B':>4} {'seq':>6}  {'v3 us':>9} {'v4 us':>9} {'v4/v3':>7}  "
          f"{'speedup':>8}  {'pair_diff':>10}")
    print("-" * 64)
    for num_seqs in [1, 4, 16, 64, 128]:
        for seq_len in [128, 256, 512, 1024]:
            try:
                v3, v4, pair, _, _ = run_case(num_seqs, NUM_KV_HEADS, seq_len)
                spd = v3 / v4
                print(f"{num_seqs:>4d} {seq_len:>6d}  {v3:9.1f} {v4:9.1f} "
                      f"{v4/v3:7.3f}  {spd:7.2f}x  {pair:.4e}")
            except Exception as ex:  # noqa: BLE001
                print(f"{num_seqs:>4d} {seq_len:>6d}  FAIL: {ex}")
    print("\nspeedup > 1.0 => FlyDSL faster than Triton v3 on this shape.")


if __name__ == "__main__":
    main()
