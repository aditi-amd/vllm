#!/usr/bin/env python3
"""Isolate decode-kernel cost in the 100K/8K traces (rank0).

UQ decode  -> dedicated kernel `fp8_g32_decode_hd256*` (+ split-KV reduce/merge).
KV8 decode -> fused into `kernel_unified_attention_2d/3d` (same kernel as prefill),
              so we split its calls by duration: prefill calls are long (process
              the whole 8K/100K prompt), decode calls are short (1 query token).
"""
import gzip, json, sys, statistics as st
from collections import defaultdict

def load_events(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        data = json.load(f)
    ev = data.get("traceEvents", data if isinstance(data, list) else [])
    # GPU kernel events: ph 'X' with 'dur', category kernel, and a device tid.
    out = []
    for e in ev:
        if e.get("ph") != "X" or "dur" not in e:
            continue
        cat = e.get("cat", "")
        if cat not in ("kernel", "Kernel", "gpu_op", "gpu_user_annotation"):
            # torch/roctracer marks device kernels as cat 'kernel'
            if "args" not in e or "device" not in e.get("args", {}):
                continue
        out.append((e.get("name", ""), float(e["dur"])))
    return out

def stats(durs):
    if not durs:
        return "n=0"
    durs = sorted(durs)
    n = len(durs)
    return (f"n={n:5d}  sum={sum(durs)/1000:8.1f}ms  mean={st.mean(durs):7.1f}us  "
            f"med={durs[n//2]:7.1f}us  p90={durs[int(n*0.9)]:7.1f}us  max={durs[-1]:7.1f}us")

def analyze(path, leg):
    ev = load_events(path)
    # group by kernel name
    by = defaultdict(list)
    for name, dur in ev:
        by[name].append(dur)

    print(f"\n================= {leg}: {path.split('/')[-2]} =================")
    print(f"  total gpu-kernel events: {len(ev)}")

    if leg == "uq":
        # dedicated decode kernel(s)
        for key in sorted(by):
            kl = key.lower()
            if "fp8_g32_decode" in kl or ("decode" in kl and "hd256" in kl):
                print(f"  [decode]   {key[:70]:70} {stats(by[key])}")
        # split-KV reduce / merge partition kernels
        for key in sorted(by):
            kl = key.lower()
            if ("reduce" in kl and "partition" in kl) or "merge_attn" in kl or \
               "flash" in kl and "reduce" in kl or "_reduce_" in kl:
                print(f"  [splitKV]  {key[:70]:70} {stats(by[key])}")
    else:
        # KV8: unified attention handles BOTH prefill and decode. Split by duration.
        uni = []
        for key in by:
            if "unified_attention" in key.lower():
                uni += [(key, d) for d in by[key]]
        alldur = [d for _, d in uni]
        if alldur:
            # decode calls are the short ones. Use a split threshold: decode (1 token)
            # kernels are far shorter than prefill (thousands of tokens). Split at the
            # gap — here we bucket calls under 400us as decode, over as prefill, and
            # also print the raw duration histogram so the split is auditable.
            allsorted = sorted(alldur)
            thr = 400.0
            dec = [d for d in alldur if d < thr]
            pre = [d for d in alldur if d >= thr]
            print(f"  [unified_attn ALL]  {stats(alldur)}")
            print(f"  [unified prefill>=400us] {stats(pre)}")
            print(f"  [unified decode  <400us] {stats(dec)}")
            # histogram
            import bisect
            edges = [0,25,50,100,200,400,800,1600,3200,6400,999999]
            h=[0]*(len(edges)-1)
            for d in allsorted:
                h[bisect.bisect_right(edges,d)-1]+=1
            print("  duration histogram (us):")
            for i in range(len(h)):
                if h[i]:
                    print(f"      [{edges[i]:>6}-{edges[i+1]:>6}) : {h[i]}")

if __name__ == "__main__":
    for path, leg in [(sys.argv[i], sys.argv[i+1]) for i in range(1, len(sys.argv), 2)]:
        analyze(path, leg)
