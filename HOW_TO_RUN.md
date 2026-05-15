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

## Part 3 — Full launch example (MiniMax-M2.5, FlyDSL v4, TP=2)

```bash
HIP_VISIBLE_DEVICES=4,5 \
VLLM_FLYDSL_ROOT=/opt/FlyDSL \
VLLM_FLYDSL_PKGS=/opt/FlyDSL/build-fly/python_packages \
PYTHONPATH="${VLLM_FLYDSL_ROOT}:${VLLM_FLYDSL_PKGS}" \
VLLM_TQ_DECODE_V4=1 VLLM_TQ_DECODE_V3=0 VLLM_TQ_DECODE_V2=0 \
VLLM_TQ_SOA_FUSION_STORE=1 VLLM_TQ_SOA_FUSION=0 \
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
