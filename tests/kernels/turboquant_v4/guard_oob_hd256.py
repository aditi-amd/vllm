"""Guard-canary harness to localize the v4 hd256 (GQA=2) out-of-bounds write.

Wraps every _SegmBufPool buffer inside a larger sentinel-filled allocation and
checks, after each decode, whether the leading/trailing guard regions were
overwritten. Any change localizes the offending buffer + direction + distance.

Run:
  VLLM_FLYDSL_ROOT=/root/FlyDSL VLLM_FLYDSL_PKGS=/root/FlyDSL/build-fly/python_packages \
  PYTHONPATH=/root/FlyDSL/build-fly/python_packages HIP_VISIBLE_DEVICES=2 \
  python tests/kernels/turboquant_v4/guard_oob_hd256.py
"""
import os
import sys

import torch

sys.path.insert(0, "/shareddata/adrana/workspace/vllm-pr-fp8hd256")
sys.path.insert(0, os.path.dirname(__file__))

import vllm.v1.attention.ops.flydsl_turboquant_decode_v4 as _mod
from vllm.v1.attention.ops.flydsl_turboquant_decode_v4 import (
    flydsl_turboquant_decode_attention_v4,
)
from bench_hd256_gqa2_sliding import (  # noqa: E402
    make_inputs, HEAD_SIZE, QG, NUM_KV_HEADS, KV_BLOCK_SIZE,
)
from test_hd256_swa import pad_cache  # noqa: E402

PAD = 4096  # guard elements on each side
SENT = {torch.bfloat16: 12345.0, torch.float32: 98765.0}


class GuardedPool:
    """Drop-in replacement for _SEGM_POOL that guards every returned buffer."""

    def __init__(self):
        self.guards = []  # list of (name, full_tensor, lo, hi, sentinel)

    def _alloc(self, name, shape, dtype, device):
        n = 1
        for s in shape:
            n *= int(s)
        sent = SENT[dtype]
        full = torch.full((n + 2 * PAD,), sent, dtype=dtype, device=device)
        body = full[PAD:PAD + n].view(*shape)  # contiguous, correct strides
        self.guards.append((name, full, PAD, PAD + n, sent))
        return body

    def get(self, B, Hk, Hq, num_partitions, QG, D, device, q_dtype):
        self.guards.clear()
        a = self._alloc
        return {
            "segm_out": a("segm_out", (B, Hk, num_partitions, QG, D),
                          torch.bfloat16, device),
            "segm_max": a("segm_max", (B, Hk, num_partitions, QG),
                          torch.float32, device),
            "segm_sum": a("segm_sum", (B, Hk, num_partitions, QG),
                          torch.float32, device),
            "output": a("output", (B, Hq, D), q_dtype, device),
            "q_rot": a("q_rot", (B, Hq, D), q_dtype, device),
            "q_float": a("q_float", (B, Hq, D), torch.float32, device),
            "q_rot_fp32": a("q_rot_fp32", (B, Hq, D), torch.float32, device),
        }

    def check(self, tag):
        hits = []
        for name, full, lo, hi, sent in self.guards:
            lead = full[:lo]
            trail = full[hi:]
            lead_bad = int((lead != sent).sum().item())
            trail_bad = int((trail != sent).sum().item())
            if lead_bad or trail_bad:
                # find first/last corrupted offset relative to body
                td = (trail != sent).nonzero().flatten()
                ld = (lead != sent).nonzero().flatten()
                first_trail = int(td[0].item()) if td.numel() else -1
                first_lead = int(ld[0].item() - lo) if ld.numel() else -1
                hits.append(
                    f"    [{name}] lead_bad={lead_bad} (firstoff={first_lead}) "
                    f"trail_bad={trail_bad} (firsttrailoff=+{first_trail})"
                )
        if hits:
            print(f"  {tag}: GUARD CORRUPTION")
            for h in hits:
                print(h)
        else:
            print(f"  {tag}: clean")
        return bool(hits)


def run_case(gp, num_seqs, seq_len, window, padded):
    max_bps = (seq_len + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE + 4
    centroids, q_bf16, kv_cache, bt, sl, *_ = make_inputs(
        num_seqs, NUM_KV_HEADS, seq_len, max_bps
    )
    if padded:
        kv_cache = pad_cache(kv_cache, 198912)
    Pi = torch.eye(HEAD_SIZE, dtype=torch.float32, device="cuda")
    PiT = Pi.T.contiguous()
    _ = flydsl_turboquant_decode_attention_v4(
        query=q_bf16, kv_cache=kv_cache, block_table=bt, seq_lens=sl,
        Pi=Pi, centroids=centroids, scale=1.0 / (HEAD_SIZE ** 0.5),
        mse_bits=4, key_packed_size=(HEAD_SIZE // 2) + 2, value_quant_bits=4,
        value_packed_size=(HEAD_SIZE // 2) + 4, key_fp8=False,
        norm_correction=False, PiT=PiT, max_seq_len=seq_len,
        max_num_kv_splits=32, sinks=None, sliding_window=window,
    )
    torch.cuda.synchronize()
    tag = f"B={num_seqs} seq={seq_len} W={window} pad={padded}"
    return gp.check(tag)


def main():
    gp = GuardedPool()
    _mod._SEGM_POOL = gp  # monkeypatch the module-level singleton
    print(f"=== hd256 QG={QG} guard-canary OOB scan (PAD={PAD}) ===")
    cases = [
        (2, 512, 1024, False),
        (2, 2048, 1024, False),
        (2, 8192, 1024, False),
        (8, 3000, 1024, False),
        (2, 2048, 1024, True),
        (2, 8192, 1024, True),
        (8, 3000, 700, True),
        (16, 4096, 1024, True),
        (32, 2048, 1024, True),
    ]
    any_bad = False
    for c in cases:
        any_bad |= run_case(gp, *c)
    print("RESULT:", "OOB DETECTED" if any_bad else "no guard corruption")


if __name__ == "__main__":
    main()
