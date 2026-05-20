# TurboQuant Kernel Setup and Launch Guide

---

## Part 1 — FlyDSL Setup (required for v4 only)

FlyDSL is a separate compiler framework that must be built once per machine.
Skip this part if you only plan to use HIP, Triton v3, or v1.

### Prerequisites
- AMD MI355X (gfx950) — validated on MI355X only; porting to MI300X is possible
  but requires kernel changes for the CDNA4 wide-K MFMA instruction
- ROCm 7.x (`hipcc` on `PATH`)
- Python 3.12
- Linux x86_64, glibc ≥ 2.35

### Build

```bash
git clone /shareddata/adrana/FlyDSL /opt/FlyDSL
cd /opt/FlyDSL
git checkout 41500b0      # tested SHA

mkdir -p build-fly && cd build-fly
cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON
ninja -j$(nproc)          # ~5 minutes
```

### Verify

```bash
python3 -c "
import sys
sys.path.insert(0, '/opt/FlyDSL/build-fly/python_packages')
import flydsl
print('FlyDSL OK:', flydsl.__version__)
"
```

### Export paths (add to your shell rc or set before launching)

```bash
export VLLM_FLYDSL_ROOT=/opt/FlyDSL
export VLLM_FLYDSL_PKGS=/opt/FlyDSL/build-fly/python_packages
export PYTHONPATH="${VLLM_FLYDSL_ROOT}:${VLLM_FLYDSL_PKGS}${PYTHONPATH:+:${PYTHONPATH}}"
```

---

## Part 2 — Kernel Options

All kernels require these vllm serve flags:

```
--attention-backend ROCM_AITER_UNIFIED_ATTN
--kv-cache-dtype turboquant_4bit_nc
--block-size 32
--no-enable-prefix-caching
--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
```

---

### FlyDSL v4

Best throughput. Requires FlyDSL setup above. SoA KV layout.

```bash
VLLM_TQ_DECODE_V4=1
VLLM_TQ_DECODE_V3=0
VLLM_TQ_DECODE_V2=0
VLLM_TQ_SOA_FUSION_STORE=1    # write SoA layout that v4 decode reads
VLLM_TQ_SOA_FUSION=0          # must be OFF (that flag is for HIP path)
HSA_NO_SCRATCH_RECLAIM=1
VLLM_ROCM_USE_AITER=1
```

#### FlyDSL v4 — butterfly optimizations (opt-in)

Two in-kernel Walsh-Hadamard butterfly opts can be enabled independently on top
of the base v4 config above.  Both default OFF for backward compatibility.

**Decode Q-rotation butterfly** (`+6.6% TPS, −10.8 ms ITL` on MiniMax-M2.5 32K/1K C=64):

```bash
VLLM_TQ_DECODE_V4_WHT_BUTTERFLY=1
```

Replaces the launcher-side rocBLAS GEMM (`q_rot = q @ PiT`) with a 7-stage
in-register Fast Walsh-Hadamard Transform (FWHT) computed inside the FlyDSL
kernel (STEP B').  The raw `query` tensor is passed instead of the pre-rotated
`q_rot`, eliminating the external GEMM and the `q_rot` HBM round-trip entirely.
Constraints: D=128, `bf16` query.

**Store K-rotation butterfly** (negligible E2E gain at C=64; eliminates HBM load of PiT matrix at store time):

```bash
VLLM_TQ_STORE_WHT_BUTTERFLY=1
```

Replaces the O(D²) PiT GEMV inside `_tq_fully_fused_store_mse` with a 7-stage
in-register WHT butterfly (O(D log₂D) = 896 additions for D=128).  No PiT
matrix is loaded from HBM at store time.  Constraints: D must be a power of 2,
non-FP8 keys only.

**Recommended production config (both opts on):**

```bash
VLLM_TQ_DECODE_V4=1 VLLM_TQ_DECODE_V3=0 VLLM_TQ_DECODE_V2=0 \
VLLM_TQ_SOA_FUSION_STORE=1 VLLM_TQ_SOA_FUSION=0 \
VLLM_TQ_DECODE_V4_WHT_BUTTERFLY=1 \
VLLM_TQ_STORE_WHT_BUTTERFLY=1 \
HSA_NO_SCRATCH_RECLAIM=1 VLLM_ROCM_USE_AITER=1
```

Benchmark results (MiniMax-M2.5, 32K/1K, C=64, N=80, TP=2, MI355X):

| Config                         |  TPS  | vs baseline | TPOT (ms) | ITL (ms) |
|--------------------------------|------:|------------:|----------:|---------:|
| baseline  (dec=0, store=0)     | 343.1 |      —      |   117.98  |   66.25  |
| dec-bf    (dec=1, store=0)     | 365.6 |    +6.56%   |   109.58  |   55.41  |
| store-bf  (dec=0, store=1)     | 345.2 |    +0.62%   |   117.21  |   65.42  |
| both-bf   (dec=1, store=1)     | 366.0 |    +6.68%   |   109.51  |   55.39  |

---

### HIP v3

Production-tested. Uses compiled HIP `.so` for decode + WHT butterfly Q rotation.
No external build required — `.so` files are pre-built in the repo.

```bash
VLLM_TQ_DECODE_V4=0
VLLM_TQ_DECODE_V3=1
VLLM_TQ_DECODE_V2=0
VLLM_TQ_SOA_FUSION=1                        # activates HIP implementation class
VLLM_TQ_SOA_FUSION_DECODE_BF16Q_PV_MFMA=1  # use MFMA decode .so
VLLM_TQ_SOA_FUSION_STORE=0                  # HIP class handles its own store
HSA_NO_SCRATCH_RECLAIM=1
VLLM_ROCM_USE_AITER=1
```

---

### Triton v3

Pure Triton decode. No HIP or FlyDSL dependencies.

```bash
VLLM_TQ_DECODE_V4=0
VLLM_TQ_DECODE_V3=1
VLLM_TQ_DECODE_V2=0
VLLM_TQ_SOA_FUSION=0
VLLM_TQ_SOA_FUSION_STORE=0
HSA_NO_SCRATCH_RECLAIM=1
VLLM_ROCM_USE_AITER=1
```

---

### v1 (legacy)

Original Triton implementation. Slowest, kept for reference only.

```bash
VLLM_TQ_DECODE_V4=0
VLLM_TQ_DECODE_V3=0
VLLM_TQ_DECODE_V2=0
VLLM_TQ_SOA_FUSION=0
VLLM_TQ_SOA_FUSION_STORE=0
HSA_NO_SCRATCH_RECLAIM=1
VLLM_ROCM_USE_AITER=1
```

---

## Part 3 — Full launch example (MiniMax-M2.5, FlyDSL v4 + butterfly, TP=2)

```bash
HIP_VISIBLE_DEVICES=4,5 \
VLLM_FLYDSL_ROOT=/opt/FlyDSL \
VLLM_FLYDSL_PKGS=/opt/FlyDSL/build-fly/python_packages \
PYTHONPATH="${VLLM_FLYDSL_ROOT}:${VLLM_FLYDSL_PKGS}" \
VLLM_TQ_DECODE_V4=1 VLLM_TQ_DECODE_V3=0 VLLM_TQ_DECODE_V2=0 \
VLLM_TQ_SOA_FUSION_STORE=1 VLLM_TQ_SOA_FUSION=0 \
VLLM_TQ_DECODE_V4_WHT_BUTTERFLY=1 \
VLLM_TQ_STORE_WHT_BUTTERFLY=1 \
HSA_NO_SCRATCH_RECLAIM=1 VLLM_ROCM_USE_AITER=1 \
vllm serve /shareddata/larryli2/MiniMax-M2.5 \
    --tensor-parallel-size 2 \
    --attention-backend ROCM_AITER_UNIFIED_ATTN \
    --kv-cache-dtype turboquant_4bit_nc \
    --block-size 32 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 9472 \
    --no-enable-prefix-caching \
    --trust-remote-code \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
    --port 8000
```
