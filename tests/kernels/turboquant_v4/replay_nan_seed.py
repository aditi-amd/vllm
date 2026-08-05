#!/usr/bin/env python3
"""Deterministically replay the captured NaN-seed decode call (v4 hd256 SWA).

The live server dumped the exact single-row inputs of the FIRST decode call
whose kernel-side segm_out went non-finite. This script reconstructs that call
against a fresh CONTIGUOUS kv_cache (block table remapped to arange) and:

  1. confirms the kernel reproduces the non-finite segm_out deterministically,
  2. re-runs with sliding_window=0 (SWA off) to test if the window mask is the
     trigger,
  3. localizes which (kv_head, partition) is non-finite.

Usage: HIP_VISIBLE_DEVICES=1 python replay_nan_seed.py [dump.pt]
"""
import os
import sys

import torch

sys.path.insert(0, "/shareddata/adrana/workspace/vllm-pr-fp8hd256")
sys.path.insert(0, os.path.dirname(__file__))

from vllm.v1.attention.ops.flydsl_turboquant_decode_v4 import (  # noqa: E402
    flydsl_turboquant_decode_attention_v4,
)

DUMP = sys.argv[1] if len(sys.argv) > 1 else \
    "/shareddata/adrana/workspace/reports/v4_nan_seed.pt"
E2E_MAX_BPS = 1024  # match server num_partitions=32


def build(d):
    dev = "cuda"
    kv_blocks = d["kv_blocks"].to(dev)          # [nblk, BS, Hk, slot] contiguous
    nblk = kv_blocks.shape[0]
    q = d["query"].to(dev)                       # [1, Hq, D]
    seq_len = int(d["seq_len"])
    bs = int(d["block_size"])
    # fresh contiguous cache with the gathered blocks placed at 0..nblk-1
    kv = kv_blocks.contiguous()
    bt = torch.zeros(1, E2E_MAX_BPS, dtype=torch.int32, device=dev)
    bt[0, :nblk] = torch.arange(nblk, dtype=torch.int32, device=dev)
    sl = torch.tensor([seq_len], dtype=torch.int32, device=dev)
    centroids = d["centroids"].to(dev)
    Pi = d["Pi"].to(dev)
    PiT = d["PiT"].to(dev) if d["PiT"] is not None else None
    return dict(q=q, kv=kv, bt=bt, sl=sl, centroids=centroids, Pi=Pi, PiT=PiT,
                scale=float(d["scale"]), swa=int(d["sliding_window"]),
                D=int(d["D"]), QG=int(d["QG"]), Hk=int(d["Hk"]),
                seq_len=seq_len)


def call(b, window):
    return flydsl_turboquant_decode_attention_v4(
        query=b["q"], kv_cache=b["kv"], block_table=b["bt"], seq_lens=b["sl"],
        Pi=b["Pi"], centroids=b["centroids"], scale=b["scale"],
        mse_bits=4, key_packed_size=(b["D"] // 2) + 2, value_quant_bits=4,
        value_packed_size=(b["D"] // 2) + 4, key_fp8=False,
        norm_correction=False, PiT=b["PiT"], max_seq_len=b["seq_len"],
        max_num_kv_splits=32, sinks=None, sliding_window=window,
        buf_holder=None,
    )


def report(tag, out):
    o = out.detach().float()
    nf = ~torch.isfinite(o)
    print(f"  [{tag}] out shape={tuple(o.shape)} nonfinite={int(nf.sum())} "
          f"absmax_finite={float(o[torch.isfinite(o)].abs().max()) if torch.isfinite(o).any() else float('nan'):.4g}")
    return int(nf.sum())


def main():
    d = torch.load(DUMP, map_location="cpu")
    print(f"=== replay NaN seed: seq_len={d['seq_len']} window={d['sliding_window']} "
          f"QG={d['QG']} D={d['D']} Hk={d['Hk']} nblk={d['kv_blocks'].shape[0]} ===")
    # server-side segm_out for the bad row (ground truth of the failure)
    so = d["segm_out_b0"].float()   # [Hk, P, QG, D]
    so_nf = ~torch.isfinite(so)
    bad_hp = [(h, p) for h in range(so.shape[0]) for p in range(so.shape[1])
              if bool(so_nf[h, p].any())]
    print(f"  server segm_out non-finite (Hk,P) pairs = {bad_hp[:12]}"
          f"{' ...' if len(bad_hp) > 12 else ''}  total={len(bad_hp)}")

    b = build(d)
    print("\n-- replay with window ON (as server) --")
    o_on = call(b, b["swa"])
    n_on = report("SWA-ON", o_on)

    # ---- mechanism test: zero the STALE slots beyond seq_len in last block ----
    # Hypothesis: slots >= (seq_len % block_size) in the final block hold stale
    # recycled-block bytes whose V dequants to Inf/NaN; masked QK gives P=0 and
    # the PV MFMA computes 0*Inf = NaN. Zeroing those slots should kill the NaN.
    import copy
    bz = build(d)
    bs = int(d["block_size"])
    sl = int(d["seq_len"])
    last_blk = sl // bs           # index into gathered blocks (arange-mapped)
    first_stale_slot = sl % bs    # slots [first_stale_slot, bs) are beyond seq
    kvz = bz["kv"].clone()
    if first_stale_slot != 0 and last_blk < kvz.shape[0]:
        # kv shape [nblk, BS, Hk, slot]; zero the tail slots of the last block
        kvz[last_blk, first_stale_slot:, :, :] = 0
        # inspect a stale slot's raw bytes before we zeroed (from original)
        stale = bz["kv"][last_blk, first_stale_slot, :, :]
        print(f"\n  last_blk={last_blk} first_stale_slot={first_stale_slot} "
              f"stale_slot abs-max(raw uint8 view)={float(stale.float().abs().max()):.1f}")
    bz["kv"] = kvz
    print("\n-- replay with beyond-seq slots ZEROED --")
    o_z = call(bz, bz["swa"])
    n_z = report("TAIL-ZEROED", o_z)
    if n_on > 0 and n_z == 0:
        print("  => CONFIRMED: stale beyond-seq slots are the NaN source "
              "(zeroing them fixes it)")
    print("\n-- replay with window OFF (sliding_window=0) --")
    o_off = call(b, 0)
    n_off = report("SWA-OFF", o_off)

    print("\nRESULT:",
          "REPRODUCED (kernel NaN)" if n_on > 0 else "NOT reproduced (env/state dependent)")
    if n_on > 0 and n_off == 0:
        print("  => SWA WINDOW MASK is the trigger (NaN vanishes with window off)")
    elif n_on > 0 and n_off > 0:
        print("  => NaN present with and without window => not SWA-mask-specific")


if __name__ == "__main__":
    main()
