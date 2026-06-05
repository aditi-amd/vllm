#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Parity test for the FlyDSL fp8_g32 decode v4 kernel.

Builds an fp8_g32 KV cache with the *real* ``fp8_g32_encode`` (so the on-disk
bytes are bit-identical to what the reference dequants), runs the FlyDSL
launcher, and compares against:

  1. ``reference_fp8_g32_attention``  — the PyTorch golden (Q Hadamard-rotated
     + FP8-E4M3 haircut, FP4+UE8M0 K/V encode→decode).
  2. ``fp8_g32_unified_attention``    — the Triton v3 kernel (when importable).

Run directly (needs an MI355X / gfx950 box with FlyDSL built):

    VLLM_FP8_G32_DECODE_V4=1 python tests/kernels/turboquant_v4/\
test_fp8_g32_v4_parity.py
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, "/shareddata/adrana/workspace/vllm-pr")

from vllm.v1.attention.ops.fp8_g32.fp8_levels import (  # noqa: E402
    get_constant_c,
    get_group_size,
    is_arch_b,
    k_scales_offset,
    slot_size,
    v_codes_offset,
    v_scales_offset,
)
from vllm.v1.attention.ops.fp8_g32.reference import (  # noqa: E402
    fp8_g32_encode,
    hadamard_matrix,
    reference_fp8_g32_attention,
)
from vllm.v1.attention.ops.flydsl_fp8_g32_decode_v4 import (  # noqa: E402
    flydsl_fp8_g32_decode_attention_v4,
    is_flydsl_available,
)

HEAD_SIZE = 128
KV_BLOCK_SIZE = int(os.environ.get("BS", "16"))
QG = int(os.environ.get("QG", "16"))


def _cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def build_fp8_g32_cache(num_seqs, num_kv_heads, seq_len, max_bps,
                        seed=0xC0FFEE, device="cuda"):
    """Build an fp8_g32 AoS cache from random K/V via the real encoder.

    Returns (q_bf16, kv_cache, block_table, seq_lens, K_raw, V_raw) where
    K_raw/V_raw are the *unrotated* bf16 tensors fed to the reference.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    Hk = num_kv_heads
    Hq = Hk * QG
    D = HEAD_SIZE
    gs = get_group_size()

    q = (torch.randn(num_seqs, Hq, D, generator=g,
                     dtype=torch.float32, device=device) * 0.5)
    q_bf16 = q.to(torch.bfloat16)

    # Raw K/V in reference layout [B, N, Hk, D].
    K_raw = (torch.randn(num_seqs, seq_len, Hk, D, generator=g,
                         dtype=torch.float32, device=device) * 0.6).to(torch.bfloat16)
    V_raw = (torch.randn(num_seqs, seq_len, Hk, D, generator=g,
                         dtype=torch.float32, device=device) * 0.6).to(torch.bfloat16)

    # Encode into [B, Hk, N, D] codes/scales (K rotated, V natural) — exactly
    # the reference's internal encode_decode inputs.
    enc_k = fp8_g32_encode(K_raw.transpose(1, 2).contiguous(), rotate=True)
    enc_v = fp8_g32_encode(V_raw.transpose(1, 2).contiguous(), rotate=False)
    k_codes = enc_k.codes_packed       # [B, Hk, N, D//2] uint8
    k_scales = enc_k.scale_bytes       # [B, Hk, N, D//gs] uint8
    v_codes = enc_v.codes_packed
    v_scales = enc_v.scale_bytes

    padded_slot = slot_size(D, gs)     # 136
    KSC = k_scales_offset(D, gs)       # 64
    VCO = v_codes_offset(D, gs)        # 68
    VSC = v_scales_offset(D, gs)       # 132
    n_groups = D // gs

    blocks_per_seq = (seq_len + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE
    bt = torch.zeros(num_seqs, max_bps, dtype=torch.int32, device=device)
    bt[:, :blocks_per_seq] = (
        torch.arange(num_seqs * blocks_per_seq, dtype=torch.int32,
                     device=device).reshape(num_seqs, blocks_per_seq) + 1
    )
    total_blocks = num_seqs * blocks_per_seq + 8

    kv_cache = torch.zeros(
        total_blocks, KV_BLOCK_SIZE, Hk, padded_slot,
        dtype=torch.uint8, device=device,
    )

    # Scatter every (seq, tok, head) slot into its (block, slot, head) cell.
    for s in range(num_seqs):
        for t in range(seq_len):
            blk = int(bt[s, t // KV_BLOCK_SIZE].item())
            slot = t % KV_BLOCK_SIZE
            cell = kv_cache[blk, slot]                  # [Hk, padded_slot]
            cell[:, 0:64] = k_codes[s, :, t]
            cell[:, KSC:KSC + n_groups] = k_scales[s, :, t]
            cell[:, VCO:VCO + 64] = v_codes[s, :, t]
            cell[:, VSC:VSC + n_groups] = v_scales[s, :, t]

    sl = torch.full((num_seqs,), seq_len, dtype=torch.int32, device=device)
    return q_bf16, kv_cache, bt, sl, K_raw, V_raw


def run_case(num_seqs, num_kv_heads, seq_len):
    device = "cuda"
    D = HEAD_SIZE
    scale = 1.0 / (D ** 0.5)
    max_bps = (seq_len + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE + 4
    q_bf16, kv_cache, bt, sl, K_raw, V_raw = build_fp8_g32_cache(
        num_seqs, num_kv_heads, seq_len, max_bps, device=device,
    )

    PiT = hadamard_matrix(D, torch.device(device), torch.float32).contiguous()

    out_ref = reference_fp8_g32_attention(
        query=q_bf16, key=K_raw, value=V_raw, scale=scale,
        constant_c=get_constant_c(), arch_b=is_arch_b(),
    )

    out_v4 = flydsl_fp8_g32_decode_attention_v4(
        query=q_bf16,
        kv_cache=kv_cache,
        block_table=bt,
        seq_lens=sl,
        scale=scale,
        PiT=PiT,
        max_seq_len=seq_len,
        buf_holder=None,
        max_num_kv_splits=32,
        sinks=None,
    )

    cos = _cos_sim(out_v4.cpu(), out_ref.cpu())
    max_abs = (out_v4.cpu().float() - out_ref.cpu().float()).abs().max().item()

    # Optional Triton v3 comparison (best-effort).
    cos_v3 = float("nan")
    try:
        from vllm.v1.attention.ops.fp8_g32.triton_unified_attention import (
            fp8_g32_unified_attention,
        )
        cu_q = torch.arange(num_seqs + 1, dtype=sl.dtype, device=device)
        out_v3 = fp8_g32_unified_attention(
            query=q_bf16, kv_cache=kv_cache, block_table=bt, seq_lens=sl,
            query_start_loc=cu_q, scale=scale, PiT=PiT, max_query_len=1,
            max_seq_len=seq_len, sinks=None, sliding_window=None,
        )
        cos_v3 = _cos_sim(out_v4.cpu(), out_v3.cpu())
    except Exception as ex:  # noqa: BLE001
        print(f"  (Triton v3 compare skipped: {ex})")

    return cos, max_abs, cos_v3


def main():
    assert torch.cuda.is_available(), "needs a GPU"
    assert is_flydsl_available(), "FlyDSL not importable"

    print(f"=== fp8_g32 v4 parity (BS={KV_BLOCK_SIZE} QG={QG} "
          f"arch_b={is_arch_b()} c={get_constant_c():.4f}) ===")
    print(f"{'B':>4} {'Hk':>3} {'seq':>6}  {'cos(ref)':>10} "
          f"{'max|Δ|':>10} {'cos(v3)':>10}")
    print("-" * 52)
    fail = False
    for num_seqs, Hk, seq_len in [
        (1, 8, 256), (2, 8, 1024), (4, 8, 2048), (2, 4, 512),
    ]:
        try:
            cos, max_abs, cos_v3 = run_case(num_seqs, Hk, seq_len)
            ok = cos >= 0.999
            fail = fail or not ok
            flag = "" if ok else "  <-- LOW COS"
            print(f"{num_seqs:>4d} {Hk:>3d} {seq_len:>6d}  {cos:>10.6f} "
                  f"{max_abs:>10.3e} {cos_v3:>10.6f}{flag}")
        except Exception as ex:  # noqa: BLE001
            fail = True
            print(f"{num_seqs:>4d} {Hk:>3d} {seq_len:>6d}  FAIL: {ex}")
    print()
    print("RESULT:", "FAIL" if fail else "PASS")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
