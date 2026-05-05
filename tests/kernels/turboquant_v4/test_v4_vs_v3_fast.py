#!/usr/bin/env python3
"""Vectorized version of test_v4_vs_v3.py — fixture in tensor ops, not loops."""
import sys
import time

import torch

sys.path.insert(0, "/shareddata/adrana/workspace/vllm-pr")

from vllm.v1.attention.ops.triton_turboquant_unified_attention import (
    triton_turboquant_decode_attention_v3,
)
from vllm.v1.attention.ops.flydsl_turboquant_decode_v4 import (
    flydsl_turboquant_decode_attention_v4,
    is_flydsl_available,
)


HEAD_SIZE = 128
KV_BLOCK_SIZE = int(__import__("os").environ.get("BS", "16"))
N_CENTROIDS = 16
QG = int(__import__("os").environ.get("QG", "16"))
KEY_DATA_BYTES = HEAD_SIZE // 2
DATA_BYTES_PER_SLOT = HEAD_SIZE
NUM_SOA_FIELDS = 3
SOA_K_NORM, SOA_V_SCALE, SOA_V_ZERO = 0, 1, 2


def make_inputs_fast(num_seqs, num_kv_heads, seq_len, max_bps, seed=0xC0FFEE,
                     device="cuda"):
    """Vectorized cache builder; produces same layout as the slow version."""
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

    # vLLM cache layout: [num_blocks, BS, Hk, slot_size_aligned]
    slot_size_aligned = (KEY_DATA_BYTES + 2) + (KEY_DATA_BYTES + 4)  # 134
    if slot_size_aligned % 2:
        slot_size_aligned += 1
    bytes_per_block = KV_BLOCK_SIZE * Hk * slot_size_aligned

    # Generate all (seq, tok, head) data at once.
    total_slots = num_seqs * seq_len * Hk
    k_idx = torch.randint(0, N_CENTROIDS,
                          (num_seqs, Hk, seq_len, HEAD_SIZE),
                          generator=g, dtype=torch.uint8, device=device)
    v_idx = torch.randint(0, N_CENTROIDS,
                          (num_seqs, Hk, seq_len, HEAD_SIZE),
                          generator=g, dtype=torch.uint8, device=device)
    knorm = (torch.rand(num_seqs, Hk, seq_len,
                        generator=g, device=device) * 0.5 + 0.5).to(torch.float16)
    vscale = (torch.rand(num_seqs, Hk, seq_len,
                         generator=g, device=device) * 0.05 + 0.01).to(torch.float16)
    vzero = ((torch.rand(num_seqs, Hk, seq_len,
                         generator=g, device=device) - 0.5) * 0.1).to(torch.float16)

    K_ref = (centroids[k_idx.long()].float()
             * knorm.float().unsqueeze(-1))             # [N, Hk, T, D]
    V_ref = (v_idx.float() * vscale.float().unsqueeze(-1)
             + vzero.float().unsqueeze(-1))             # [N, Hk, T, D]

    # Pack 4-bit nibbles: shape [..., D/2] uint8
    k_packed = (k_idx[..., 0::2] | (k_idx[..., 1::2] << 4))  # [N, Hk, T, 64]
    v_packed = (v_idx[..., 0::2] | (v_idx[..., 1::2] << 4))  # [N, Hk, T, 64]

    # Allocate cache as flat byte buffer.
    kv_cache = torch.zeros(total_blocks, bytes_per_block,
                           dtype=torch.uint8, device=device)

    # Compute target (block, slot) for every (seq, tok).
    seq_idx = torch.arange(num_seqs, device=device).view(num_seqs, 1).expand(
        num_seqs, seq_len)
    tok_idx = torch.arange(seq_len, device=device).view(1, seq_len).expand(
        num_seqs, seq_len)
    blk_for_tok = bt[seq_idx, tok_idx // KV_BLOCK_SIZE]  # [N, T]
    slot_for_tok = tok_idx % KV_BLOCK_SIZE                # [N, T]

    # ---- Write data region (K then V) for all (s, h, t) ------------
    # data offset = slot * Hk * 128 + h * 128
    h_idx_arr = torch.arange(Hk, device=device)
    base_data = (slot_for_tok.unsqueeze(1) * Hk * DATA_BYTES_PER_SLOT
                 + h_idx_arr.view(1, Hk, 1) * DATA_BYTES_PER_SLOT)  # [N, Hk, T]
    # Place packed K/V (64+64=128 bytes per slot per head).
    # Flatten target indices: dest = blk * bytes_per_block + base_data + 0..127
    blk_b = blk_for_tok.unsqueeze(1).expand(num_seqs, Hk, seq_len)  # [N, Hk, T]
    dst_base = blk_b * bytes_per_block + base_data                    # [N, Hk, T]
    # K bytes positions
    rng_k = torch.arange(KEY_DATA_BYTES, device=device)
    rng_v = torch.arange(KEY_DATA_BYTES, device=device) + KEY_DATA_BYTES
    k_dst = (dst_base.unsqueeze(-1) + rng_k.view(1, 1, 1, -1)).reshape(-1)
    v_dst = (dst_base.unsqueeze(-1) + rng_v.view(1, 1, 1, -1)).reshape(-1)
    flat = kv_cache.view(-1)
    flat[k_dst] = k_packed.reshape(-1)
    flat[v_dst] = v_packed.reshape(-1)

    # ---- Write meta region (k_norm, v_scale, v_zero) ---------------
    META_OFF = KV_BLOCK_SIZE * Hk * DATA_BYTES_PER_SLOT
    # meta is fp16 indexed by halfwords:
    # meta_view base = META_OFF/2; per-head stride = NUM_SOA_FIELDS * BS
    # k_norm at h*3*BS + 0*BS + slot
    # v_scale at h*3*BS + 1*BS + slot
    # v_zero  at h*3*BS + 2*BS + slot
    meta_view = kv_cache.view(torch.float16).view(total_blocks, -1)
    meta_off_hw = META_OFF // 2  # halfwords
    h_arr = h_idx_arr.view(1, Hk, 1)                           # [1, Hk, 1]
    slot_arr = slot_for_tok.unsqueeze(1)                        # [N, 1, T]
    common = h_arr * NUM_SOA_FIELDS * KV_BLOCK_SIZE + slot_arr  # [N, Hk, T]
    blk_flat = blk_for_tok.unsqueeze(1).expand(num_seqs, Hk, seq_len)
    knorm_off = meta_off_hw + common + SOA_K_NORM * KV_BLOCK_SIZE
    vscale_off = meta_off_hw + common + SOA_V_SCALE * KV_BLOCK_SIZE
    vzero_off = meta_off_hw + common + SOA_V_ZERO * KV_BLOCK_SIZE
    meta_view[blk_flat.reshape(-1).long(), knorm_off.reshape(-1).long()] = knorm.reshape(-1)
    meta_view[blk_flat.reshape(-1).long(), vscale_off.reshape(-1).long()] = vscale.reshape(-1)
    meta_view[blk_flat.reshape(-1).long(), vzero_off.reshape(-1).long()] = vzero.reshape(-1)

    sl = torch.full((num_seqs,), seq_len, dtype=torch.int32, device=device)
    # Reshape to 4D for v3 compatibility: [num_blocks, BS, Hk, slot_size_aligned]
    kv_cache_4d = kv_cache.view(total_blocks, KV_BLOCK_SIZE, Hk, slot_size_aligned)
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
    centroids, q_bf16, kv_cache, bt, sl, K_ref, V_ref = make_inputs_fast(
        num_seqs, num_kv_heads, seq_len, max_bps
    )
    Pi = torch.eye(HEAD_SIZE, dtype=torch.float32, device="cuda")
    PiT = Pi.T.contiguous()
    common = dict(
        kv_cache=kv_cache, block_table=bt, seq_lens=sl,
        Pi=Pi, centroids=centroids, scale=1.0/(HEAD_SIZE**0.5),
        mse_bits=4, key_packed_size=KEY_DATA_BYTES+2, value_quant_bits=4,
        value_packed_size=KEY_DATA_BYTES+4, key_fp8=False,
        norm_correction=False, PiT=PiT, max_seq_len=seq_len,
        max_num_kv_splits=32, sinks=None,
    )

    o_v3 = triton_turboquant_decode_attention_v3(query=q_bf16, **common)
    o_v4 = flydsl_turboquant_decode_attention_v4(query=q_bf16, **common)
    pair_diff = (o_v3.cpu().float() - o_v4.cpu().float()).abs().max().item()

    if validate:
        out_ref = py_reference(q_bf16.cpu(), K_ref, V_ref, sl.cpu(),
                               1.0/(HEAD_SIZE**0.5))
        v3_diff = (o_v3.cpu().float() - out_ref).abs().max().item()
        v4_diff = (o_v4.cpu().float() - out_ref).abs().max().item()
    else:
        v3_diff = v4_diff = float("nan")

    v3_us = bench(lambda: triton_turboquant_decode_attention_v3(query=q_bf16, **common))
    v4_us = bench(lambda: flydsl_turboquant_decode_attention_v4(query=q_bf16, **common))
    return v3_us, v4_us, pair_diff, v3_diff, v4_diff


def main():
    assert is_flydsl_available()

    # Sanity validation on small case.
    print("=== Validation (seq=1024, B=2, Hk=8) ===")
    v3, v4, pair, v3d, v4d = run_case(2, 8, 1024, validate=True)
    print(f"v3 vs ref: {v3d:.4e}")
    print(f"v4 vs ref: {v4d:.4e}")
    print(f"v4 vs v3 : {pair:.4e}")
    print(f"v3 us={v3:.1f}  v4 us={v4:.1f}  ratio={v4/v3:.3f}")

    print()
    print("=== Sweep (Qwen-32B class: Hk=8, QG=16, Hq=128) ===")
    print(f"{'B':>4} {'seq':>6}  "
          f"{'v3 us':>8} {'v4 us':>8} {'v4/v3':>7}  {'pair_diff':>10}")
    print("-" * 55)
    for num_seqs in [1, 4, 16, 64]:
        for seq_len in [256, 1024, 4096, 8192]:
            try:
                v3, v4, pair, _, _ = run_case(num_seqs, 8, seq_len)
                print(f"{num_seqs:>4d} {seq_len:>6d}  "
                      f"{v3:8.1f} {v4:8.1f} {v4/v3:7.3f}  {pair:.4e}")
            except Exception as ex:  # noqa: BLE001
                print(f"{num_seqs:>4d} {seq_len:>6d}  FAIL: {ex}")


if __name__ == "__main__":
    main()
