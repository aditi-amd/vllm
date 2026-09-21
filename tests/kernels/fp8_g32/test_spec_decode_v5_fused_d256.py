# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolate whether the greedy-match UQ+MTP drift is a *kernel* issue

(the real D=256 v5-fused FlyDSL decode kernel misbehaving on the batch shape
MTP verify produces) or an *MTP-glue* issue (Eagle/rejection-sampling logic
outside the attention kernel).

Two checks, both against the exact PyTorch reference
(`reference_fp8_g32_attention`), using the REAL production kernel
(`flydsl_fp8_g32_decode_attention_v5_fused`), not the Triton D=128 stand-in
used by `test_spec_decode_vs_reference.py`:

1. ``test_k1_degenerate_matches_plain_decode``: _spec_decode_region's math
   with K=1 must be *bit-identical* to an ordinary decode call at the same
   B (no expansion actually happens: offs=[0], synth_seq_lens==seq_lens,
   num_decodes unchanged). Any diff here means the metadata-construction
   path itself is broken, independent of MTP or batch-size effects.

2. ``test_bk_expansion_matches_reference``: the real K>1 vectorized B*K
   expansion vs the exact reference, for D=256 / GQA-8 (Qwen3.8 per-TP-rank
   shape). This is the same math `test_spec_decode_vs_reference.py` already
   validated on the D=128 Triton kernel, now run through the real D=256
   kernel to see if the split-KV partition count (which is chosen from B,
   see VLLM_FP8_G32_DECODE_V4_LOWB_THRESHOLD) drifts the result once B is
   expanded B -> B*K.
"""

from __future__ import annotations

import pytest
import torch

from vllm.v1.attention.ops import flydsl_fp8_g32_decode_v5_fused as v5
from vllm.v1.attention.ops.flydsl_fp8_g32_decode_v5_fused import (
    flydsl_fp8_g32_decode_attention_v5_fused,
    is_flydsl_available,
    is_flydsl_fp8_hd256_available,
)
from vllm.v1.attention.ops.fp8_g32.fp8_levels import slot_size
from vllm.v1.attention.ops.fp8_g32.reference import reference_fp8_g32_attention
from vllm.v1.attention.ops.fp8_g32.triton_store import fp8_g32_store

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA/HIP device"
)


def _skip_if_unavailable(query_group_size: int) -> None:
    if not is_flydsl_available():
        pytest.skip("FlyDSL not available on this host")
    if not is_flydsl_fp8_hd256_available(query_group_size):
        pytest.skip("FlyDSL fp8_g32 hd256 (D=256) sibling not available")


def _build_paged_cache(key_BNHD, value_BNHD, block_size, device):
    """Pack [B, N, Hk, D] K/V into a paged uint8 fp8_g32 cache."""
    B, N, Hk, D = key_BNHD.shape
    blocks_per_seq = (N + block_size - 1) // block_size
    num_blocks = B * blocks_per_seq + 1

    slot_b = slot_size(D)
    kv_cache = torch.zeros(
        num_blocks, block_size, Hk, slot_b, dtype=torch.uint8, device=device
    )
    block_table = torch.zeros(B, blocks_per_seq, dtype=torch.int32, device=device)
    for bi in range(B):
        for blk in range(blocks_per_seq):
            block_table[bi, blk] = bi * blocks_per_seq + blk

    slot_mapping = torch.full((B * N,), -1, dtype=torch.int64, device=device)
    k_flat = torch.empty(B * N, Hk, D, dtype=key_BNHD.dtype, device=device)
    v_flat = torch.empty_like(k_flat)
    for bi in range(B):
        for ti in range(N):
            gidx = bi * N + ti
            block_num = int(block_table[bi, ti // block_size].item())
            slot = block_num * block_size + ti % block_size
            slot_mapping[gidx] = slot
            k_flat[gidx] = key_BNHD[bi, ti]
            v_flat[gidx] = value_BNHD[bi, ti]

    fp8_g32_store(k_flat, v_flat, kv_cache, slot_mapping)
    torch.cuda.synchronize()
    return kv_cache, block_table


def _quality(a_out, b_out):
    a = a_out.float().flatten()
    b = b_out.float().flatten()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    diff = (a - b).abs()
    return {
        "cos_sim": cos,
        "mean_abs": diff.mean().item(),
        "max_abs": diff.max().item(),
        "norm_ratio": (a.norm() / b.norm()).item(),
    }


def _run_v5(query, kv_cache, block_table, seq_lens, scale, PiT, max_seq_len):
    return flydsl_fp8_g32_decode_attention_v5_fused(
        query=query,
        kv_cache=kv_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        scale=scale,
        PiT=PiT,
        max_seq_len=max_seq_len,
        buf_holder=None,
        max_num_kv_splits=32,
    )


def test_fuse_qrot_opt_out_matches_in_kernel_path(monkeypatch):
    """The documented V5_FUSE_QROT=0 escape hatch must pass pre-rotated Q."""
    B, Hq, Hk, D, N = 4, 8, 1, 256, 4096
    _skip_if_unavailable(Hq // Hk)
    device = torch.device("cuda")
    torch.manual_seed(0xF053)
    dtype = torch.bfloat16
    block_size = 32
    scale = 1.0 / (D**0.5)

    key = torch.randn(B, N, Hk, D, dtype=dtype, device=device)
    value = torch.randn(B, N, Hk, D, dtype=dtype, device=device)
    query = torch.randn(B, Hq, D, dtype=dtype, device=device)
    kv_cache, block_table = _build_paged_cache(
        key, value, block_size, device
    )
    seq_lens = torch.full((B,), N, dtype=torch.int32, device=device)

    from vllm.v1.attention.ops.fp8_g32.reference import hadamard_matrix

    PiT = hadamard_matrix(D, device, torch.float32)

    monkeypatch.setattr(v5, "_FUSE_QROT_INKERNEL", True)
    v5._KERN_CACHE.clear()
    out_inline = _run_v5(
        query, kv_cache, block_table, seq_lens, scale, PiT, N
    ).clone()

    monkeypatch.setattr(v5, "_FUSE_QROT_INKERNEL", False)
    v5._KERN_CACHE.clear()
    out_prologue = _run_v5(
        query, kv_cache, block_table, seq_lens, scale, PiT, N
    ).clone()
    torch.cuda.synchronize()

    assert torch.equal(out_inline, out_prologue)


@pytest.mark.parametrize("block_size", [32, 64, 256])
def test_v7_strided_matches_blocked_and_reference(monkeypatch, block_size):
    """Exercise v7 scheduling with the non-contiguous server QKV stride."""
    B, Hq, Hk, D, N = 4, 8, 1, 256, 4096
    _skip_if_unavailable(Hq // Hk)
    device = torch.device("cuda")
    torch.manual_seed(0x57A1D + block_size)
    dtype = torch.bfloat16
    scale = 1.0 / (D**0.5)

    key = torch.randn(B, N, Hk, D, dtype=dtype, device=device)
    value = torch.randn(B, N, Hk, D, dtype=dtype, device=device)
    qkv = torch.randn(B, Hq + 2 * Hk, D, dtype=dtype, device=device)
    query = qkv[:, :Hq, :]
    assert query.stride(0) == (Hq + 2 * Hk) * D

    kv_cache, block_table = _build_paged_cache(
        key, value, block_size, device
    )
    seq_lens = torch.full((B,), N, dtype=torch.int32, device=device)

    from vllm.v1.attention.ops.fp8_g32.reference import hadamard_matrix

    PiT = hadamard_matrix(D, device, torch.float32)

    monkeypatch.setenv("VLLM_FP8_G32_DECODE_V5_STRIDED_TG", "1")
    v5._ENV_SNAPSHOT.clear()
    v5._KERN_CACHE.clear()
    out_strided = _run_v5(
        query, kv_cache, block_table, seq_lens, scale, PiT, N
    ).clone()

    monkeypatch.setenv("VLLM_FP8_G32_DECODE_V5_STRIDED_TG", "0")
    v5._ENV_SNAPSHOT.clear()
    v5._KERN_CACHE.clear()
    out_blocked = _run_v5(
        query, kv_cache, block_table, seq_lens, scale, PiT, N
    ).clone()
    torch.cuda.synchronize()

    ref = reference_fp8_g32_attention(
        query=query,
        key=key,
        value=value,
        scale=scale,
    )
    strided_quality = _quality(out_strided, ref)
    blocked_quality = _quality(out_blocked, ref)
    mutual_quality = _quality(out_strided, out_blocked)

    assert strided_quality["cos_sim"] > 0.999
    assert blocked_quality["cos_sim"] > 0.999
    assert mutual_quality["cos_sim"] > 0.99999


@pytest.mark.parametrize("B", [1, 2, 4, 8, 9, 16])
def test_k1_degenerate_matches_plain_decode(B):
    """K=1 spec-expansion metadata must be a no-op vs plain decode at the
    SAME batch size B (this does not exercise the B->B*K partition-count
    sensitivity at all -- it only guards the metadata plumbing)."""
    Hq, Hk, D, N = 8, 1, 256, 4096
    _skip_if_unavailable(Hq // Hk)
    device = torch.device("cuda")
    torch.manual_seed(0xBEEF + B)
    dtype = torch.bfloat16
    block_size = 32
    scale = 1.0 / (D**0.5)

    key = torch.randn(B, N, Hk, D, dtype=dtype, device=device)
    value = torch.randn(B, N, Hk, D, dtype=dtype, device=device)
    query = torch.randn(B, Hq, D, dtype=dtype, device=device)
    kv_cache, block_table = _build_paged_cache(key, value, block_size, device)
    seq_lens = torch.full((B,), N, dtype=torch.int32, device=device)

    from vllm.v1.attention.ops.fp8_g32.reference import hadamard_matrix
    PiT = hadamard_matrix(D, device, torch.float32)

    out_a = _run_v5(query, kv_cache, block_table, seq_lens, scale, PiT, N).clone()
    out_b = _run_v5(query, kv_cache, block_table, seq_lens, scale, PiT, N).clone()
    torch.cuda.synchronize()

    assert torch.equal(out_a, out_b), "K=1 decode is not even deterministic run-to-run!"
    print(f"\n[K=1 degenerate B={B}] bit-identical across repeated calls: OK")


@pytest.mark.parametrize(
    "B,K",
    [
        (1, 2), (1, 3),
        (4, 2), (4, 3),
        (8, 2), (8, 3),   # B*K crosses the LOWB_THRESHOLD=8 default boundary
        (9, 2),           # B alone already > 8 -> HIB_CAP even before *K
    ],
)
def test_bk_expansion_matches_reference(B, K):
    """Real D=256 v5-fused kernel: B*K verify expansion vs exact reference.

    Mirrors test_spec_decode_vs_reference.py's expansion math but on the
    actual production kernel (GQA-8, D=256) instead of the D=128 Triton
    stand-in, to see whether the batch-size-dependent split-KV partition
    count (MAX_PARTITIONS flips 512->256 past B=8) measurably drifts the
    verify-step output relative to what a true single-token decode of the
    same request would produce.
    """
    Hq, Hk, D, N = 8, 1, 256, 4096
    _skip_if_unavailable(Hq // Hk)
    device = torch.device("cuda")
    torch.manual_seed(0xA17 + B * 100 + K)
    dtype = torch.bfloat16
    block_size = 32
    scale = 1.0 / (D**0.5)

    key = torch.randn(B, N, Hk, D, dtype=dtype, device=device)
    value = torch.randn(B, N, Hk, D, dtype=dtype, device=device)
    q_bk = torch.randn(B, K, Hq, D, dtype=dtype, device=device)

    kv_cache, block_table = _build_paged_cache(key, value, block_size, device)
    seq_lens_dec = torch.full((B,), N, dtype=torch.int32, device=device)

    from vllm.v1.attention.ops.fp8_g32.reference import hadamard_matrix
    PiT = hadamard_matrix(D, device, torch.float32)

    # ---- expansion math, identical to _spec_decode_region ----
    offs = torch.arange(K, device=device, dtype=seq_lens_dec.dtype) - (K - 1)
    synth_seq_lens = (seq_lens_dec[:, None] + offs[None, :]).reshape(-1).clamp_(min=1)
    synth_bt = block_table.repeat_interleave(K, dim=0)
    q_flat = q_bk.reshape(B * K, Hq, D)  # request-major

    out_vec = _run_v5(
        q_flat, kv_cache, synth_bt, synth_seq_lens, scale, PiT, N
    ).reshape(B, K, Hq, D).clone()
    torch.cuda.synchronize()

    # ---- exact per-query reference with truncated context ----
    worst = {"cos_sim": 1.0, "mean_abs": 0.0, "max_abs": 0.0, "norm_ratio": 1.0}
    # ---- SAME KERNEL, but called as if this were a true single-token
    #      decode of the request in isolation (B=1) -- isolates whether the
    #      partition-count-vs-B heuristic itself is the source of drift,
    #      as opposed to generic UE8M0/FP4 rounding noise.
    worst_vs_b1 = dict(worst)
    for b in range(B):
        for k in range(K):
            L = N - K + 1 + k
            ref = reference_fp8_g32_attention(
                query=q_bk[b, k : k + 1].reshape(1, Hq, D),
                key=key[b : b + 1, :L],
                value=value[b : b + 1, :L],
                scale=scale,
            )  # (1, Hq, D)
            q = _quality(out_vec[b, k], ref[0])
            if q["cos_sim"] < worst["cos_sim"]:
                worst = q

            # Re-run the REAL kernel at B=1 (true decode batch size) for
            # this single (b,k) query against its truncated context.
            k1_cache, k1_bt = _build_paged_cache(
                key[b : b + 1, :L], value[b : b + 1, :L], block_size, device
            )
            seq_len_1 = torch.full((1,), L, dtype=torch.int32, device=device)
            out_b1 = _run_v5(
                q_bk[b, k : k + 1].reshape(1, Hq, D),
                k1_cache, k1_bt, seq_len_1, scale, PiT, L,
            )
            qb1 = _quality(out_vec[b, k], out_b1[0])
            if qb1["cos_sim"] < worst_vs_b1["cos_sim"]:
                worst_vs_b1 = qb1

    print(
        f"\n[B={B} K={K} D=256 GQA=8, v5-fused] "
        f"vs EXACT ref: cos_sim={worst['cos_sim']:.6f} mean={worst['mean_abs']:.5f} "
        f"max={worst['max_abs']:.5f} norm_ratio={worst['norm_ratio']:.5f}"
    )
    print(
        f"[B={B} K={K} D=256 GQA=8, v5-fused] "
        f"vs SAME KERNEL @B=1 (true decode batch size): "
        f"cos_sim={worst_vs_b1['cos_sim']:.6f} mean={worst_vs_b1['mean_abs']:.5f} "
        f"max={worst_vs_b1['max_abs']:.5f}"
    )
    # Same tolerance band as the D=128 Triton expansion test.
    assert worst["cos_sim"] > 0.98, worst
    assert worst["mean_abs"] < 0.08, worst
    assert worst["max_abs"] < 0.8, worst
    assert 0.93 < worst_vs_b1["cos_sim"] < 1.0001, (
        "expanded B*K output disagrees with the SAME kernel run at the true "
        f"decode batch size B=1 -- batch-size-dependent kernel numerics "
        f"(split-KV partition count) is the likely drift source: {worst_vs_b1}"
    )
