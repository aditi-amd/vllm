#!/usr/bin/env python3
"""Warm-process dev loop for the fp8_g32 hd256 decode kernel.

One process amortizes import + FlyDSL JIT across an entire sweep. Uses CUDA
events (not torch.profiler) for fast total-time timing; a --prof mode gives the
per-kernel component breakdown; a quick parity gate (small seq) guards
correctness. Pair with watch_dev.sh to auto-rerun on kernel edits.

Modes (MODE env or arg1):
  time    (default) CUDA-event timing over SEQS x SPLITS grid + quick parity
  prof    torch.profiler component breakdown at each SEQ
  parity  parity gate only (D=256, QG=8, small seq)
  ab      A/B two configs by SPLITS (in-process); flag A/B (qk_fp8) = 2 launches

Env knobs: SEQS=8192,16384,32768  SPLITS=32  SPLITS_B=8  B=16 HK=1 QG=8 BS=32
           ITERS=30  SKIP_PARITY=0

Example:
  HIP_VISIBLE_DEVICES=0 VLLM_FP8_G32_V3=1 VLLM_FP8_G32_DECODE_V4=1 \
  VLLM_FP8_G32_DECODE_V4_QK_SCALED=1 VLLM_FLYDSL_ROOT=/root/FlyDSL \
  VLLM_FLYDSL_PKGS=/root/FlyDSL/build-fly/python_packages \
  python tests/kernels/turboquant_v4/dev_loop.py time
"""
from __future__ import annotations

import os
import sys
import types

import torch

sys.path.insert(0, "/shareddata/adrana/workspace/vllm-pr-fp8hd256")

from vllm.v1.attention.ops.fp8_g32.fp8_levels import (
    get_constant_c,
    get_group_size,
    is_arch_b,
    k_scales_offset,
    slot_size,
    v_codes_offset,
    v_scales_offset,
)
from vllm.v1.attention.ops.fp8_g32.reference import (
    fp8_g32_encode,
    hadamard_matrix,
    reference_fp8_g32_attention,
)
from vllm.v1.attention.ops.fp8_g32.triton_store import fp8_g32_store
from vllm.v1.attention.ops.flydsl_fp8_g32_decode_v4 import (
    flydsl_fp8_g32_decode_attention_v4,
    is_flydsl_available,
    is_flydsl_fp8_hd256_available,
)

D = 256
QG = int(os.environ.get("QG", "8"))
HK = int(os.environ.get("HK", "1"))
BS = int(os.environ.get("BS", "32"))
B = int(os.environ.get("B", "16"))
ITERS = int(os.environ.get("ITERS", "30"))
PEAK_BW = float(os.environ.get("PEAK_BW", "5300"))  # GB/s (MI355 HBM)


def build_inputs(seq_len, splits, device="cuda", seed=None):
    Hk, Hq, gs = HK, HK * QG, get_group_size()
    padded_slot = slot_size(D, gs)
    bps = (seq_len + BS - 1) // BS
    # BT_SEQ models the SERVER condition: the block table is allocated for
    # max_model_len while the ACTUAL context (seq_lens) is much shorter. The
    # launcher derives worst_case_max_seq_len from block_table.shape[1], so with
    # DYNAMIC_PARTS=0 it sizes partitions/TGPP for BT_SEQ, not seq_len -- i.e.
    # every decode call pays worst-case work. Default BT_SEQ=seq_len keeps the
    # historical bench behavior (block table sized to the actual context).
    bt_seq = int(os.environ.get("BT_SEQ", "0")) or seq_len
    bt_bps = max(bps, (bt_seq + BS - 1) // BS)
    total_blocks = B * bps + 8
    g = None if seed is None else torch.Generator(device=device).manual_seed(seed)
    q = (torch.randn(B, Hq, D, device=device, generator=g) * 0.5).to(
        torch.bfloat16)
    kv = torch.randint(0, 256, (total_blocks, BS, Hk, padded_slot),
                       dtype=torch.uint8, device=device, generator=g)
    bt = torch.zeros(B, bt_bps + 4, dtype=torch.int32, device=device)
    bt[:, :bps] = (torch.arange(B * bps, dtype=torch.int32, device=device)
                   .reshape(B, bps) + 1)
    sl = torch.full((B,), seq_len, dtype=torch.int32, device=device)
    PiT = hadamard_matrix(D, torch.device(device), torch.float32).contiguous()
    # Persistent buf_holder so PiT/PiT_perm caching matches the real model
    # (layer object). Without it the launcher recomputes the qperm index_select
    # + contiguous copy every call — a bench-only artifact.
    return dict(query=q, kv_cache=kv, block_table=bt, seq_lens=sl,
                scale=1.0 / (D ** 0.5), PiT=PiT, max_seq_len=seq_len,
                max_num_kv_splits=splits, sinks=None,
                buf_holder=types.SimpleNamespace())


_GOLDEN = os.path.join(os.path.dirname(__file__), "dev_loop_golden.pt")


def golden_case(save):
    """Bit-exact gate: deterministic REAL-encoded inputs -> compare vs saved
    golden. max|Δ|==0 means the change is bitwise identical (safe occupancy-only
    edit). Uses the parity builder (real fp8_g32 encode) so the output is
    numerically meaningful — a random-byte cache decodes to all-zeros and would
    catch nothing.
    """
    seq_len = int(os.environ.get("GOLDEN_SEQ", "2048"))
    ns = min(B, 8)
    q, kv, bt, sl, _K, _V, PiT = _build_parity_cache(ns, seq_len, seed=0xC0FFEE)
    out = flydsl_fp8_g32_decode_attention_v4(
        query=q, kv_cache=kv, block_table=bt, seq_lens=sl,
        scale=1.0 / (D ** 0.5), PiT=PiT, max_seq_len=seq_len,
        max_num_kv_splits=int(os.environ.get("SPLITS", "32")),
        sinks=None).float().cpu()
    if save:
        torch.save(out, _GOLDEN)
        print(f"golden saved: {_GOLDEN}  shape={tuple(out.shape)} "
              f"seq={seq_len} B={B} QG={QG}")
        return True
    if not os.path.exists(_GOLDEN):
        print("no golden saved; run: dev_loop golden-save (on baseline first)")
        return False
    ref = torch.load(_GOLDEN)
    if ref.shape != out.shape:
        print(f"golden shape mismatch {tuple(ref.shape)} vs {tuple(out.shape)}")
        return False
    d = (out - ref).abs()
    mx = d.max().item()
    cos = torch.nn.functional.cosine_similarity(
        out.reshape(-1), ref.reshape(-1), dim=0).item()
    exact = mx == 0.0
    print(f"vs golden: max|Δ|={mx:.3e}  cos={cos:.8f}  "
          f"{'BIT-IDENTICAL ✓' if exact else ('numeric diff' if cos>=0.9999 else 'FAIL ✗')}")
    return exact or cos >= 0.9999


def _median_us(fn):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    st = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS + 1)]
    st[0].record()
    for i in range(ITERS):
        fn()
        st[i + 1].record()
    torch.cuda.synchronize()
    t = sorted(st[i].elapsed_time(st[i + 1]) * 1e3 for i in range(ITERS))
    return t[len(t) // 2]


def time_case(seq_len, splits):
    common = build_inputs(seq_len, splits)
    return _median_us(lambda: flydsl_fp8_g32_decode_attention_v4(**common))


def decode_bytes(seq_len):
    """KV bytes the decode kernel must read per call (4-bit codes+scales)."""
    return B * seq_len * HK * slot_size(D, get_group_size())


def floor_us(seq_len):
    return decode_bytes(seq_len) / (PEAK_BW * 1e3)  # bytes / (GB/s * 1e3 B/us)


def store_case(device="cuda"):
    Hk, gs = HK, get_group_size()
    padded_slot = slot_size(D, gs)
    k = (torch.randn(B, Hk, D, device=device) * 0.6).to(torch.bfloat16)
    v = (torch.randn(B, Hk, D, device=device) * 0.6).to(torch.bfloat16)
    kv = torch.zeros(B + 8, BS, Hk, padded_slot, dtype=torch.uint8,
                     device=device)
    slot = (torch.arange(B, dtype=torch.int64, device=device) * BS)
    return _median_us(lambda: fp8_g32_store(k, v, kv, slot))


def prof_case(seq_len, splits):
    from torch.profiler import ProfilerActivity, profile
    common = build_inputs(seq_len, splits)
    for _ in range(20):
        flydsl_fp8_g32_decode_attention_v4(**common)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        for _ in range(100):
            flydsl_fp8_g32_decode_attention_v4(**common)
        torch.cuda.synchronize()
    rows = []
    for e in p.key_averages():
        us = getattr(e, "device_time_total", 0.0) or getattr(
            e, "cuda_time_total", 0.0)
        if us and us > 0:
            rows.append((us / 100, e.key, e.count / 100))
    rows.sort(reverse=True)
    tot = sum(r[0] for r in rows)
    print(f"\n--- prof seq={seq_len} splits={splits}  ({tot:.1f} us/call) ---")
    print(f"{'us/call':>9} {'pct':>6} {'launch':>7}  name")
    for us, name, cnt in rows[:12]:
        print(f"{us:9.2f} {100*us/tot:5.1f}% {cnt:7.2f}  {name[:52]}")


def _build_parity_cache(num_seqs, seq_len, device="cuda", seed=0xC0FFEE):
    """D-generic fp8_g32 cache from the real encoder (small seq for gate)."""
    g = torch.Generator(device=device).manual_seed(seed)
    Hk, Hq, gs = HK, HK * QG, get_group_size()
    q = (torch.randn(num_seqs, Hq, D, generator=g, device=device) * 0.5).to(
        torch.bfloat16)
    K = (torch.randn(num_seqs, seq_len, Hk, D, generator=g, device=device)
         * 0.6).to(torch.bfloat16)
    V = (torch.randn(num_seqs, seq_len, Hk, D, generator=g, device=device)
         * 0.6).to(torch.bfloat16)
    enc_k = fp8_g32_encode(K.transpose(1, 2).contiguous(), rotate=True)
    enc_v = fp8_g32_encode(V.transpose(1, 2).contiguous(), rotate=False)
    padded_slot, KSC = slot_size(D, gs), k_scales_offset(D, gs)
    VCO, VSC = v_codes_offset(D, gs), v_scales_offset(D, gs)
    ng, half = D // gs, D // 2
    bps = (seq_len + BS - 1) // BS
    bt = torch.zeros(num_seqs, bps + 4, dtype=torch.int32, device=device)
    bt[:, :bps] = (torch.arange(num_seqs * bps, dtype=torch.int32,
                   device=device).reshape(num_seqs, bps) + 1)
    kv = torch.zeros(num_seqs * bps + 8, BS, Hk, padded_slot,
                     dtype=torch.uint8, device=device)
    for s in range(num_seqs):
        for t in range(seq_len):
            cell = kv[int(bt[s, t // BS].item()), t % BS]
            cell[:, 0:half] = enc_k.codes_packed[s, :, t]
            cell[:, KSC:KSC + ng] = enc_k.scale_bytes[s, :, t]
            cell[:, VCO:VCO + half] = enc_v.codes_packed[s, :, t]
            cell[:, VSC:VSC + ng] = enc_v.scale_bytes[s, :, t]
    sl = torch.full((num_seqs,), seq_len, dtype=torch.int32, device=device)
    PiT = hadamard_matrix(D, torch.device(device), torch.float32).contiguous()
    return q, kv, bt, sl, K, V, PiT


def parity_quick():
    if os.environ.get("SKIP_PARITY") == "1":
        return True
    try:
        ok = True
        print(f"\n--- parity (D={D} QG={QG}) ---")
        for ns, sq in [(1, 256), (2, 1024)]:
            q, kv, bt, sl, K, V, PiT = _build_parity_cache(ns, sq)
            scale = 1.0 / (D ** 0.5)
            ref = reference_fp8_g32_attention(
                query=q, key=K, value=V, scale=scale,
                constant_c=get_constant_c(), arch_b=is_arch_b())
            out = flydsl_fp8_g32_decode_attention_v4(
                query=q, kv_cache=kv, block_table=bt, seq_lens=sl, scale=scale,
                PiT=PiT, max_seq_len=sq, max_num_kv_splits=32, sinks=None)
            a, b = out.float().reshape(-1), ref.float().reshape(-1)
            cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
            mx = (out.cpu().float() - ref.cpu().float()).abs().max().item()
            ok = ok and cos >= 0.999
            flag = "" if cos >= 0.999 else "  <-- LOW"
            print(f"  B={ns} seq={sq:>5}  cos={cos:.6f} max|Δ|={mx:.2e}{flag}")
        print("  parity:", "PASS" if ok else "FAIL")
        return ok
    except Exception as ex:  # noqa: BLE001
        print(f"  parity skipped: {ex}")
        return True


def main():
    assert torch.cuda.is_available() and is_flydsl_available()
    assert is_flydsl_fp8_hd256_available(QG)
    mode = (sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MODE", "time"))
    seqs = [int(x) for x in os.environ.get("SEQS", "8192,16384,32768").split(",")]
    splits = int(os.environ.get("SPLITS", "32"))

    if mode == "prof":
        for s in seqs:
            prof_case(s, splits)
    elif mode == "golden-save":
        golden_case(save=True)
    elif mode == "golden":
        ok = golden_case(save=False)
        sys.exit(0 if ok else 1)
    elif mode == "parity":
        parity_quick()
    elif mode == "ab":
        sb = int(os.environ.get("SPLITS_B", "8"))
        print(f"{'seq':>7} {'A(sp=%d)' % splits:>10} {'B(sp=%d)' % sb:>10} "
              f"{'B/A':>6}")
        for s in seqs:
            a, b = time_case(s, splits), time_case(s, sb)
            print(f"{s:>7} {a:>10.2f} {b:>10.2f} {b/a:>6.2f}")
    else:  # time
        st = store_case()
        print(f"store kernel: {st:6.2f} us/call  (B={B} Hk={HK} D={D})")
        print(f"{'seq':>7} {'decode':>9} {'+store':>9} {'floor':>8} "
              f"{'GB/s':>7} {'x_floor':>8}   (splits={splits} B={B} QG={QG})")
        for s in seqs:
            d = time_case(s, splits)
            fl = floor_us(s)
            gbs = decode_bytes(s) / (d * 1e3)
            print(f"{s:>7} {d:>9.2f} {d+st:>9.2f} {fl:>8.2f} {gbs:>7.0f} "
                  f"{d/fl:>7.1f}x")
        print("KV8 ref (from trace, ~100k+ ctx): ~29 us/call unified")
        parity_quick()


if __name__ == "__main__":
    main()
