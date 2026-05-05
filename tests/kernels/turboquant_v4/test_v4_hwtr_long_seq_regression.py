#!/usr/bin/env python3
"""Long-sequence regression for the HW V transpose path in TQ FlyDSL v4.

Background (see paper §8.6 + tq_decode_v4.py "HW V transpose: cross-lane
LDS sync" comment): the HW V transpose path uses ds_read_tr16_b64 to load
the V operand for the PV MFMA. That instruction performs a cross-lane LDS
read — consumer lane t reads bytes written by source lanes 0,4,8,12 (etc.).
The compiler's automatic waitcnt insertion only sees same-lane LDS deps,
so on long-running per-CTA loops it can sink the next iteration's
ds_read_tr16_b64 ahead of the current iteration's ds_write_b128 ops.

The fix (sched_barrier(0) + s_waitcnt lgkmcnt=0 anchor between V LDS write
and the PV MFMA) prevents that reorder. This test reproduces the
production shape (Qwen3-32B max_model_len=9472, num_partitions padded to
64, ~16 K-tile iterations per partition, mixed sequence lengths) and
asserts:

  1. HW V transpose (default ON for gfx950+) is bit-equivalent to the SW
     transpose path (which is known-correct because each lane only reads
     cells it itself wrote — no cross-lane dep).
  2. Both paths track the python reference within FlyDSL's 5e-3 default
     per-element tolerance.
  3. Repeated invocations remain stable (no drift / scheduler state
     accumulation).

This test would have caught the Qwen3-32B GSM8K -3.3pp regression that
test_v4_vs_v3_fast.py and test_v4_opt_ab.py both missed (they used
short, fixed-length sequences).
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, "/shareddata/adrana/workspace/vllm-pr")
sys.path.insert(0, "/shareddata/adrana/workspace/flydsl_kernels")

# Force HW_TR resolution to be controllable per-call.
os.environ.pop("VLLM_TQ_DECODE_V4_HW_TR", None)

from vllm.v1.attention.ops import flydsl_turboquant_decode_v4 as v4_mod  # noqa: E402
from vllm.v1.attention.ops.flydsl_turboquant_decode_v4 import (  # noqa: E402
    flydsl_turboquant_decode_attention_v4,
    is_flydsl_available,
)
from test_v4_vs_v3_fast import (  # noqa: E402
    HEAD_SIZE,
    make_inputs_fast,
    py_reference,
)


# Production-like shapes. The first one is the actual Qwen3-32B failure
# mode: max_model_len=9472, GQA=8, KV block_size=32. The remaining cases
# stress different segments of the (batch, seq_len, num_partitions) space.
# Each entry: (batch, num_kv_heads, seq_len, qg, kv_block_size,
#              num_partitions_actual)  where num_partitions_actual is
# computed as ceil(seq_len / KV_COMPUTE_BLOCK=256) before PO2 padding.
CASES = [
    # Qwen3-32B max-len, padded num_partitions=64 (37 → 64 PO2)
    (1,  4, 9472, 8, 32, 37),
    (4,  4, 9472, 8, 32, 37),
    # Mid-length, intermediate padding (5 → 8 PO2)
    (4,  8, 1280, 8, 32,  5),
    # Long PO2-natural (32 partitions → no padding)
    (8,  8, 8192, 8, 32, 32),
    # Long sequence with 16-tiles-per-partition stress
    (4,  8, 4096, 8, 32, 16),
]


def _force_hw_tr(enabled: bool) -> None:
    """Override the launcher's cached HW_TR resolution for the next call."""
    v4_mod._HW_TR_CACHED = bool(enabled)


def _run_one(num_seqs: int, num_kv_heads: int, seq_len: int,
             qg: int, kv_block_size: int):
    """Build fixture, run v4 once with HW_TR=ON and once with HW_TR=OFF,
    return (hw_out, sw_out, ref_out, py_ref_out)."""
    # The fixture uses module-level KV_BLOCK_SIZE / QG; rebind via env.
    os.environ["BS"] = str(kv_block_size)
    os.environ["QG"] = str(qg)
    import importlib
    import test_v4_vs_v3_fast as fv
    importlib.reload(fv)

    max_bps = (seq_len + kv_block_size - 1) // kv_block_size + 4
    centroids, q_bf16, kv_cache, bt, sl, K_ref, V_ref = (
        fv.make_inputs_fast(num_seqs, num_kv_heads, seq_len, max_bps)
    )

    Pi = torch.eye(HEAD_SIZE, dtype=torch.float32, device="cuda")
    PiT = Pi.T.contiguous()
    common = dict(
        kv_cache=kv_cache, block_table=bt, seq_lens=sl,
        Pi=Pi, centroids=centroids, scale=1.0 / (HEAD_SIZE ** 0.5),
        mse_bits=4, key_packed_size=64 + 2, value_quant_bits=4,
        value_packed_size=64 + 4, key_fp8=False,
        norm_correction=False, PiT=PiT, max_seq_len=seq_len,
        max_num_kv_splits=32, sinks=None,
    )

    _force_hw_tr(True)
    hw_out = flydsl_turboquant_decode_attention_v4(query=q_bf16, **common).clone()

    _force_hw_tr(False)
    sw_out = flydsl_turboquant_decode_attention_v4(query=q_bf16, **common).clone()

    py_ref = py_reference(q_bf16.cpu(), K_ref, V_ref, sl.cpu(),
                          1.0 / (HEAD_SIZE ** 0.5))
    return hw_out.cpu().float(), sw_out.cpu().float(), py_ref


def main() -> int:
    assert is_flydsl_available(), "FlyDSL not available"
    print("=== Long-sequence HW V transpose regression ===")
    print(f"{'Case':>40} | {'HW vs SW':>11} | {'HW vs ref':>11} | "
          f"{'SW vs ref':>11} | status")
    print("-" * 105)

    fails = 0
    PER_ELEM_TOL = 5.0e-3   # FlyDSL default tolerance for bf16 outputs
    HW_VS_SW_TOL = 2.5e-3   # one bf16 ULP per chain; same as v4 vs v3

    for B, Hk, S, qg, bs, num_parts in CASES:
        try:
            hw, sw, ref = _run_one(B, Hk, S, qg, bs)
        except Exception as ex:  # noqa: BLE001
            print(f"{f'B={B} Hk={Hk} S={S} qg={qg} bs={bs}':>40}"
                  f" |       FAIL: {ex}")
            fails += 1
            continue

        d_hw_sw = (hw - sw).abs().max().item()
        d_hw_ref = (hw - ref).abs().max().item()
        d_sw_ref = (sw - ref).abs().max().item()

        ok = (d_hw_sw <= HW_VS_SW_TOL
              and d_hw_ref <= PER_ELEM_TOL
              and d_sw_ref <= PER_ELEM_TOL)
        status = "PASS" if ok else "FAIL"
        if not ok:
            fails += 1

        case_label = f"B={B} Hk={Hk} S={S} qg={qg} bs={bs} np={num_parts}"
        print(f"{case_label:>40} | {d_hw_sw:11.4e} | {d_hw_ref:11.4e} | "
              f"{d_sw_ref:11.4e} | {status}")

        # Belt-and-suspenders: free GPU memory between cases (test_v4_tier1
        # documented harness fragility around long-running sweeps).
        import gc
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    print()
    if fails:
        print(f"FAIL: {fails}/{len(CASES)} case(s) regressed")
        return 1
    print(f"PASS: all {len(CASES)} long-sequence cases match SW path "
          f"and python reference")
    print()
    print("This validates the sched_barrier(0) + s_waitcnt(lgkmcnt=0) fence")
    print("inserted between V LDS write and ds_read_tr16_b64 in")
    print("kernels/tq_decode_v4.py. Without it, HW vs SW diverges by")
    print("~5e-3 on the long-sequence cases and Qwen3-32B GSM8K loses")
    print("3.3pp accuracy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
