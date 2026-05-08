# FlyDSL Dispatch Overhead: Root Cause Analysis

## Background

FlyDSL is described as an AOT (Ahead-Of-Time) compilation framework: Python DSL →
MLIR → HSACO GPU binary. In practice, vLLM's integration revealed significant
overhead during CUDA graph capture that does not disappear after the first run.
This document explains why, distinguishes the three separate cost phases, and
describes the workaround we implemented.

---

## The Three Phases of FlyDSL Kernel Dispatch

### Phase 1 — MLIR → HSACO compilation (truly AOT, disk-cached)

This is the step FlyDSL advertises as AOT. The Python DSL is lowered through
MLIR to a ROCm HSACO binary and written to a disk cache
(`~/.cache/flydsl/...`).

| Cold (first ever run) | Warm (HSACO on disk) |
|---|---|
| ~34 s/rank | ~0.02 s/rank |

On the first server boot for a given `(num_kv_heads, num_partitions,
max_blocks_per_seq)` key, all 4 TP ranks compile independently, totalling
~136 s. Subsequent boots read the cached HSACO and this phase is effectively
free. **This phase is not the problem after the first run.**

### Phase 2 — `@flyc.jit` Python dispatch specialization (the real bottleneck)

`@flyc.jit` is modelled on Triton's `@triton.jit`: it specializes the kernel
invocation lazily on the first call for each unique input signature (batch size
B, tensor shapes, strides, dtypes). Even when the HSACO is already cached, for
each new B FlyDSL must:

1. Load the HSACO from disk into a GPU module object.
2. Inspect the kernel's MLIR ABI and build a **launch descriptor** — a Python
   object that maps every tensor argument to its ROCm pointer, shape, stride,
   and dtype at the MLIR ABI level.
3. Allocate and register any kernel-internal scratchpad buffers with the ROCm
   runtime.
4. Store the descriptor in `@flyc.jit`'s Python dict, keyed by the full input
   signature.

This binding step cannot be pre-done at true AOT time because **the tensor
shapes (specifically batch size B) are not known until vLLM requests them at
capture time**.

**Measured cost: ~900 ms average, up to ~2.8 s, per novel batch size.**

With vLLM's default `cudagraph_capture_sizes` list of 51 entries, this
contributes ~46 s to every warm capture. During live serving, any batch size
not pre-captured hits this 900 ms path before the first response — a severe
latency spike.

### Phase 3 — CUDA graph recording (minor, ~50 ms total)

For each capture size, vLLM records a `torch.cuda.CUDAGraph`. The v4 kernel
adds a Z-dimension of `num_partitions=32` CTAs per sequence, making the graph
slightly larger than v3. This contributes negligibly compared to Phase 2.

---

## Why v3 (Triton) and HIP Do Not Have This Problem

### Triton v3

`@triton.jit` specialization is a hash lookup into a pre-compiled PTX module.
The kernel binary is already registered with the CUDA/ROCm driver. Launching a
new batch size B just passes a different `grid=(B*Hk, 1, 1)` — no ABI
rebinding is needed. Cost: **~microseconds per novel B**.

### HIP SoA fusion

The HIP kernel is a precompiled `.so` exported as a C function with a fixed
binary ABI. vLLM calls it via `ctypes.CDLL` directly. There is no Python
dispatch layer, no shape specialization, no launch descriptor construction.
Cost: **zero per novel B**.

---

## Measured Impact (Qwen2.5-72B, 32K/1K, TP=4)

| Config | Capture time | Novel-B serving overhead | End-to-end TPS |
|---|---|---|---|
| Triton v3 | 12 s | ~μs | 220 t/s |
| HIP SoA fusion | 8 s | 0 | 267 t/s |
| v4 cold cache | **152 s** | ~900 ms/B | — |
| v4 warm, default capture (51 sizes) | 47 s | ~900 ms/B | 182 t/s |
| v4 warm, **dense capture [1..64]** | 14 s | **0** (all pre-specialized) | **275 t/s** |

---

## Workaround: Dense `cudagraph_capture_sizes`

The root cause cannot be fixed without changes to the FlyDSL framework itself.
The workaround moves all Phase 2 specialization into the initial capture phase
(which is disk-cached after the first run) by explicitly listing every batch
size that the benchmark will encounter:

```python
DENSE_SIZES = list(range(1, 65)) + [72, 80, 88, ..., 512]
compilation_config = {
    "cudagraph_mode": "FULL_AND_PIECEWISE",
    "cudagraph_capture_sizes": DENSE_SIZES,
}
```

After the first server boot (warm cache), capture takes ~14 s (matching v3)
and no novel-B overhead occurs during serving. This lifted v4 throughput from
182 → 275 t/s on Qwen (+35%) and is the configuration used in all reported
benchmark numbers.

**Limitation:** if a request arrives with a batch size not in the dense list,
the 900 ms path is hit again. For C=64 workloads this is avoided by including
[1..64] in the list.

---

## What a Proper Fix Would Look Like

A true production-grade AOT path would:

1. Pre-build all launch descriptors at model load time for all known B values
   (the `cudagraph_capture_sizes` list is available upfront).
2. Serialize the descriptors to disk alongside the HSACO, keyed by
   `(num_kv_heads, num_partitions, max_blocks_per_seq, B)`.
3. At capture time, deserialize descriptors with a dict lookup (~μs) rather
   than rebuilding from MLIR ABI (~900 ms).

This would make FlyDSL's dispatch cost equivalent to Triton's without requiring
any changes to the kernel DSL itself — only to the `@flyc.jit` specialization
cache layer.

---

## Operational Q&A

### 1. Do benchmark runs include warmup, and does it help v4?

Yes — the `run.sh` script passes `--num-warmups 8` to the benchmark client:
8 requests are sent and discarded before timing starts. But this is
**request warmup, not FlyDSL specialization warmup**. The distinction matters:

| What "warmup" covers | v3 / HIP | v4 default capture | v4 dense capture |
|---|---|---|---|
| GPU cache warm-up (L2, TLB) | ✓ | ✓ | ✓ |
| CUDA graph replay path warm | ✓ | ✓ | ✓ |
| Novel-B 900 ms penalty | N/A | Only if B seen in warmup | ✓ all pre-done at capture |

With default capture, if the 8 warmup requests use B values already in the
51-entry capture list, the 900 ms penalty never shows up in warmup — but it
hits the timed requests the first time a new B appears mid-benchmark. With
dense capture, every B from 1 to 64 was pre-specialized at capture time, so
both warmup and timed requests run at full speed.

**The 275 t/s Qwen and 343 t/s MiniMax numbers are clean steady-state
throughput with no hidden latency spikes.**

---

### 2. Why does v4 beat HIP by 30% on MiniMax but only 3% on Qwen?

The kernel-level advantage of v4 is consistent — microbenchmarks show v4
3–4× faster than v3 at both 8K and 32K sequence lengths. End-to-end TPS gain
depends on what fraction of total step time is in the attention kernel, and
how well the other kernels (HIP, v3) were tuned for that model's GQA shape.

**MiniMax-M2.5 (GQA-6: 48 query heads, 8 KV heads, group size 6):**
- Triton v3 was tuned for GQA-8; GQA-6 fits awkwardly into its tiling.
- HIP's `soa_bf16q_pv_mfma_decode_gqa6.so` is a separate, less-optimised
  companion kernel — added later and not as battle-hardened as the Qwen path.
- v4's tiling is group-size-agnostic — GQA-6 is handled identically to GQA-8
  by the scf.ForOp inner loop.
- MiniMax is also a MoE model with large FFN experts — attention is a higher
  fraction of per-token compute relative to dense models.
- Result: **v4 +30% vs HIP, +77% vs v3, +15% above BF16 baseline**.

**Qwen2.5-72B (GQA-8: 64 query heads, 8 KV heads, group size 8):**
- HIP's `soa_bf16q_pv_mfma_decode.so` was specifically tuned for GQA-8 and
  is the most optimised path in the HIP codebase.
- Dense transformer — attention is a smaller fraction of total step compute.
- v4 wins, but headroom above HIP is narrow: **+3% vs HIP, +25% vs v3**.

---

### 3. Why add dense capture sizes for v4 but not for HIP or v3?

The 51 default vLLM capture sizes were already sufficient for HIP and v3
because their per-novel-B dispatch cost is effectively zero:

| Kernel | Cost per novel B | 51 default sizes sufficient? | Dense sizes needed? |
|---|---|---|---|
| Triton v3 | ~μs (hash lookup) | ✓ Yes | No |
| HIP `.so` | 0 (ctypes call) | ✓ Yes | No |
| FlyDSL v4 | ~900 ms (ABI rebind) | ✗ No | **Yes** |

The 51 default sizes cover round numbers: powers of 2 and a few multiples
(`[1, 2, 4, 8, 16, 32, 64, ...]`). With C=64 concurrent requests draining at
different rates, vLLM's scheduler creates irregular batch sizes like 3, 5, 7,
11, 17, 23, ... during the timed phase. For HIP and v3, hitting B=17 for the
first time costs nothing. For v4, it costs ~900 ms the first time — a visible
latency spike in TPOT.

The dense list `[1, 2, 3, ..., 64, 72, 80, ..., 512]` pre-captures every
integer B up to the concurrency limit, so during timed execution there are no
novel B values — every possible batch size was already seen and specialised
during capture.

Adding dense sizes for HIP would provide no benefit while making HIP's capture
slightly slower (more `torch.cuda.CUDAGraph` records to make). It is a
v4-specific requirement arising solely from the `@flyc.jit` dispatch layer.

---

## Summary

| Cost source | Phase | Cold | Warm (current) | Warm (ideal fix) |
|---|---|---|---|---|
| MLIR → HSACO compile | 1 | ~136 s (4 ranks) | ~0.08 s | ~0.08 s |
| `@flyc.jit` shape specialization | 2 | ~46 s (51 sizes) | ~46 s (51 sizes) | ~0 s |
| Dense capture pre-specialization | 2 | ~110 s (77 sizes) | ~14 s | ~0 s |
| CUDA graph recording | 3 | ~5 s | ~5 s | ~5 s |

The dense capture workaround is the right short-term fix. The long-term fix is
a serializable launch descriptor cache in the FlyDSL `@flyc.jit` layer.
