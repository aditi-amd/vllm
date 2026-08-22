"""Numeric A/B for the split-KV partition-cap change (MAX_PARTITIONS 32 -> 256).

Raising the cap changes how many partitions the KV range is split into, which
changes the ORDER of the cross-partition softmax reduction. That is not a
bit-identical edit, so the golden gate cannot cover it; this script quantifies
the numeric effect instead, against the fp32 reference.

Context must exceed KV_COMPUTE_BLOCK * 32 (= 8192) for the two caps to actually
differ:
  seq=16384 -> required=64 partitions
               cap=32  -> P=32, TGPP=2   (serialized tile-groups)
               cap=256 -> P=64, TGPP=1   (parallel workgroups)

Run:
  PYTHONPATH=<vllm_src> HIP_VISIBLE_DEVICES=0 python3 cap_numeric_ab.py
Env: SEQ (default 16384), QK_SCALED (default 1 = production default path).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))

from dev_loop import D, _build_parity_cache  # noqa: E402

from vllm.v1.attention.ops.fp8_g32.fp8_levels import (  # noqa: E402
    get_constant_c,
    is_arch_b,
)
from vllm.v1.attention.ops.fp8_g32.reference import (  # noqa: E402
    reference_fp8_g32_attention,
)
from vllm.v1.attention.ops.flydsl_fp8_g32_decode_v4 import (  # noqa: E402
    flydsl_fp8_g32_decode_attention_v4,
)


def run_at_cap(cap, q, kv, bt, sl, PiT, seq, scale):
    os.environ["VLLM_FP8_G32_DECODE_V4_MAX_PARTITIONS"] = str(cap)
    return flydsl_fp8_g32_decode_attention_v4(
        query=q, kv_cache=kv, block_table=bt, seq_lens=sl, scale=scale,
        PiT=PiT, max_seq_len=seq, max_num_kv_splits=32, sinks=None,
    ).float().cpu()


def cmp(name, a, b):
    d = (a - b).abs()
    cos = torch.nn.functional.cosine_similarity(
        a.reshape(-1), b.reshape(-1), dim=0).item()
    rel = (d.max() / b.abs().max()).item()
    print(f"  {name:<28} cos={cos:.8f}  max|d|={d.max().item():.3e}  "
          f"rel={rel:.3e}")
    return cos


def main():
    seq = int(os.environ.get("SEQ", "16384"))
    scale = 1.0 / (D ** 0.5)
    print(f"building real fp8_g32-encoded cache: seq={seq} (this is slow) ...")
    q, kv, bt, sl, K, V, PiT = _build_parity_cache(1, seq)

    ref = reference_fp8_g32_attention(
        query=q, key=K, value=V, scale=scale,
        constant_c=get_constant_c(), arch_b=is_arch_b()).float().cpu()

    out32 = run_at_cap(32, q, kv, bt, sl, PiT, seq, scale)
    out256 = run_at_cap(256, q, kv, bt, sl, PiT, seq, scale)

    print(f"\n--- cap numeric A/B (seq={seq}, "
          f"qk_scaled={os.environ.get('VLLM_FP8_G32_DECODE_V4_QK_SCALED','1')}) ---")
    c32 = cmp("cap=32   vs fp32 ref", out32, ref)
    c256 = cmp("cap=256  vs fp32 ref", out256, ref)
    cmp("cap=32   vs cap=256", out32, out256)

    # The gate: the fix must not be measurably further from the reference than
    # the pre-fix path. Equal-or-better, within 1e-4 of parity tolerance.
    ok = c256 >= 0.9999 and c256 >= c32 - 1e-4
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'} "
          f"(cap256 cos {c256:.8f} vs cap32 cos {c32:.8f})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
