# SPDX-License-Identifier: Apache-2.0
"""Prove norm_correction is a decode-time no-op for both v3 and v4.

Question: when ``--kv-cache-dtype turboquant_4bit_nc`` (norm_correction=True),
does v4 *actually* honor the correction, or did v4 silently regress accuracy?

Answer (this test): the v4 kernel ignores the launcher's norm_correction
flag, but so does v3. Both kernels just multiply ``c_vals * stored_knorm``;
the correction (1/||c_t||) is pre-folded into ``stored_knorm`` at STORE time
by triton_turboquant_store._store_packed_key (line 339-349:
``vn_f32 = vn_f32 * c_inv_norm`` when NORM_CORRECTION=1).

So at decode time, flipping ``norm_correction=True/False`` should produce
**bit-identical** output for v3 and v4 — proving the launcher flag is purely
API parity.

Pass criteria:
  - v3(norm_correction=True) ≡ v3(norm_correction=False)   (max_abs == 0)
  - v4(norm_correction=True) ≡ v4(norm_correction=False)   (max_abs == 0)
  - v3 ≡ v4 within standard tolerance                       (max_abs < 5e-3
    per FlyDSL test_pa.py default get_tolerance())
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, "/shareddata/adrana/workspace/vllm-pr")
sys.path.insert(0, "/shareddata/adrana/workspace/flydsl_kernels")

import torch

from test_v4_vs_v3_fast import (
    HEAD_SIZE, KEY_DATA_BYTES, make_inputs_fast,
)
from vllm.v1.attention.ops.flydsl_turboquant_decode_v4 import (
    flydsl_turboquant_decode_attention_v4,
)
from vllm.v1.attention.ops.turboquant_soa_fusion.external_ops import (
    triton_turboquant_decode_attention_v3,
)


def _make_args(num_seqs: int, num_kv_heads: int, seq_len: int):
    """Build a fixed cache + inputs for both kernel paths."""
    max_bps = (seq_len + 16 - 1) // 16 + 4
    centroids, q_bf16, kv_cache, bt, sl, *_ = make_inputs_fast(
        num_seqs, num_kv_heads, seq_len, max_bps
    )
    Pi = torch.eye(HEAD_SIZE, dtype=torch.float32, device="cuda")
    PiT = Pi.T.contiguous()
    base = dict(
        kv_cache=kv_cache, block_table=bt, seq_lens=sl,
        Pi=Pi, centroids=centroids, scale=1.0 / (HEAD_SIZE ** 0.5),
        mse_bits=4, key_packed_size=KEY_DATA_BYTES + 2, value_quant_bits=4,
        value_packed_size=KEY_DATA_BYTES + 4, key_fp8=False,
        PiT=PiT, max_seq_len=seq_len, max_num_kv_splits=32, sinks=None,
    )
    return q_bf16, base


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def main() -> None:
    # FlyDSL test_pa.py default tolerance for fixed-len, no-sliding-window:
    # diff_tolerance = 5e-3
    TOL = 5e-3
    cases = [
        # (B, Hk, seq_len)     — covers single-partition + multi-partition
        (1,  8,  256),
        (2,  8, 1024),
        (4,  8, 4096),
        # GQA-8 (Qwen2.5-72B-class) is encoded by the test's QG=16 default;
        # for QG=8 cases, the harness module-level QG would need tweaking.
    ]
    print("Goal: confirm norm_correction is a decode-time no-op for v3 AND v4.")
    print("If True/False give bit-identical decode output, the correction is")
    print("purely a STORE-time concern (pre-folded into stored_knorm) and")
    print("the v4 launcher's 'NYI' warning was misleading.\n")
    print(f"{'Case':>22} | {'v3 nc=T vs F':>14} | {'v4 nc=T vs F':>14} | "
          f"{'v3 vs v4 (nc=T)':>16} | status")
    print("-" * 100)

    fails = 0
    for B, Hk, S in cases:
        q, base = _make_args(B, Hk, S)
        o_v3_T = triton_turboquant_decode_attention_v3(
            query=q, norm_correction=True, **base
        ).clone()
        o_v3_F = triton_turboquant_decode_attention_v3(
            query=q, norm_correction=False, **base
        ).clone()
        o_v4_T = flydsl_turboquant_decode_attention_v4(
            query=q, norm_correction=True, **base
        ).clone()
        o_v4_F = flydsl_turboquant_decode_attention_v4(
            query=q, norm_correction=False, **base
        ).clone()

        d_v3 = _max_abs(o_v3_T, o_v3_F)
        d_v4 = _max_abs(o_v4_T, o_v4_F)
        d_xx = _max_abs(o_v3_T, o_v4_T)

        # Bit-exact between True/False is the strict claim. Tolerate < 1e-9.
        v3_ok = d_v3 < 1e-9
        v4_ok = d_v4 < 1e-9
        xx_ok = d_xx < TOL

        status = "PASS" if (v3_ok and v4_ok and xx_ok) else "FAIL"
        if not (v3_ok and v4_ok and xx_ok):
            fails += 1

        case = f"B={B} Hk={Hk} S={S}"
        print(f"{case:>22} | {d_v3:>14.4e} | {d_v4:>14.4e} | "
              f"{d_xx:>16.4e} | {status}")

    print()
    if fails:
        print(f"FAIL: {fails}/{len(cases)} case(s) regressed")
        print("If v3 nc=T vs F differs, v3 is NOT a no-op (would invalidate the")
        print("inline comment claim at turboquant_attn.py:994-999).")
        print("If v4 nc=T vs F differs, v4 IS doing decode-time work for the flag")
        print("(would invalidate this fix).")
        sys.exit(1)
    print("ALL PASS:")
    print("  • v3 ignores norm_correction at decode (bit-identical T vs F)")
    print("  • v4 ignores norm_correction at decode (bit-identical T vs F)")
    print(f"  • v3 vs v4 within FlyDSL default tolerance ({TOL:.0e})")
    print()
    print("CONCLUSION:")
    print("  The kv_cache_dtype suffix '_nc' (norm_correction) is a STORE-time")
    print("  property — by the time data hits decode, it's already folded into")
    print("  the stored K-norm scalar. The decode kernel just does")
    print("  K = c * stored_knorm, where stored_knorm = ||k|| / ||c|| (when nc=True)")
    print("  or stored_knorm = ||k|| (when nc=False). Both v3 and v4 do the")
    print("  identical multiply, so the gap on GSM8K v4 vs v3 (-3.3pp) is NOT")
    print("  caused by missing norm_correction in v4.")


if __name__ == "__main__":
    main()
