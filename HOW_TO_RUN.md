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

### FP4-g32 (fp4_kv_g32)

**FP4 E2M1 codes + fp16 per-group-of-32 scales.**
Pure Triton, no FlyDSL or HIP dependency. Works on any ROCm GPU.

Design notes:
- K and V are stored as 4-bit FP4 E2M1 codes with fp16 scales (one scale per 32 elements per group).
- Slot size: **144 bytes** per (token, head) — K codes 64 B + K scales 8 B + V codes 64 B + V scales 8 B.
- K is Hadamard-rotated before quantization (same rotation as TurboQuant).
- Scale format is **fp16**, not UE8M0 — avoids the −13 pp accuracy penalty from power-of-two-only scales.
- There is **no SoA cache layout for FP4** today (unlike TQ44, which has a `VLLM_TQ_SOA_FUSION` SoA path).
  Per-token K/V scales are stored AoS, interleaved with the codes inside each slot.

#### Two decode paths

| Path | Flag | Kernel | Status |
|---|---|---|---|
| **V3 (recommended)** | `VLLM_FP4_G32_V3=1` | `fp4_g32_unified_attention` (`tl.dot` MFMA, GQA stacking, 2D/3D split-KV, main/tail, fused Q rot) | Current ship target |
| V1 (legacy) | unset (default) | `fp4_g32_decode_attention` (per-head + scalar broadcast-mul-sum) + `_fp4_g32_continuation_prefill` (full dequant + flash_attn) | Fallback while V3 burns in |

V3 mirrors TQ44 V3's `tl.dot`-based MFMA structure verbatim — only the K/V tile loaders differ. V1 is the original per-head decode + dequant-then-flash_attn path; significantly slower at long context (>32K).

#### Performance characterization

Decode attention with FP4-g32 V3 currently runs **~17–32% slower** than TQ44 V3 at 128K context (LCB-128K, MI355). The cost is *not* memory-bandwidth-bound on the data side — it sits in the K/V tile *build step* before MFMA:

- **4 fp16 scales per token** (vs 1 norm for TQ44 V3) → 8 B/token K-scale metadata vs 2 B; same for V → ~2.7× scale metadata bandwidth.
- **Per-group broadcast multiply** `K_g[16,4,32] × scales[16,4,1]` doesn't fold cleanly into the `[BLOCK_M=16, HEAD_DIM=128]` MFMA tile shape; Triton emits extra reshape and 4-way piecewise-broadcast instructions that TQ44's single scalar-per-token broadcast avoids.
- V also pays an FP4 LUT gather per element (TQ44 V is uniform INT4 so just `idx*scale + zero`, no LUT).

The MFMA instruction (`v_mfma_f32_16x16x32_bf16` on gfx950) is identical for both paths — the gap is purely in the dequant plumbing on the way into MFMA.

The intended production target is the hardware FP4 MFMA path via aiter `batched_gemm_a16wfp4` on MI355: K is consumed as FP4 directly by the tensor core, decode happens inline at load time, and the per-group scale fuses on the **accumulator** output side instead of pre-multiplying K. The Triton path here is the temporary fallback while that hardware path matures.

#### Launch

```bash
VLLM_FP4_G32_V3=1                # opt into V3 unified attention (recommended)
HSA_NO_SCRATCH_RECLAIM=1
VLLM_ROCM_USE_AITER=0            # AITER attention backend has no effect on FP4
                                 # quantized layers (they always go through the
                                 # FP4-g32 kernel); leave off to avoid confusion

vllm serve <model> \
    --kv-cache-dtype fp4_kv_g32 \
    --block-size 16 \
    --no-enable-prefix-caching \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
```

#### V3 tuning knobs

| Env var | Default | Purpose |
|---|---|---|
| `VLLM_FP4_G32_V3` | `0` | Enable V3 unified MFMA path (set to `1`) |
| `VLLM_FP4_G32_FUSE_Q_ROT` | `0` | Fuse Q@PiT rotation inside the kernel via bf16 MFMA. Off by default (3/11 borderline answer flips at 128K in lcb_128k smoke). A/B for perf only. |
| `VLLM_FP4_G32_TILE_SIZE_DECODE` | unset → 16 | Decode tile width. Larger values (32) cause severe VGPR spilling on AMD MI3xx at long context. |
| `VLLM_FP4_G32_NUM_STAGES` | `1` (HIP) / `2` (CUDA) | Software pipeline depth for the 2D/3D kernels. |
| `VLLM_FP4_G32_NUM_WARPS_3D` | `2` | Warp count for the 3D split-KV decode kernel. |
| `VLLM_FP4_G32` | — | **Deprecated / no-op**; use `VLLM_FP4_G32_V3=1` instead. |
| `FP4_KV_GROUP_SIZE` | `32` | Per-group element count (16 or 32). 32 is recommended (MSE-optimal, matches MFMA K=32 on gfx950). |
| `FP4FP16_CONSTANT_C` | `0.156` | MSE-optimal scale constant `s = c * absmax`. Don't tune unless you've calibrated. |
| `FP4_KV_TOKEN_NORM` | `0` | Optional per-token L2 normfold before per-group quant. Off by default. |

#### Accuracy budget (vs exact fp16 SDPA, head_dim=128)

| Context length | Cosine similarity | Max abs error |
|---|---|---|
| 512 tokens | ~0.989 | ~0.05 |
| 2048 tokens | ~0.988 | ~0.026 |
| 8192 tokens | ~0.989 | ~0.009 |

Cosine similarity does not degrade with context length (longer contexts average more
keys, reducing the relative impact of any single quantization error).

#### Long-context accuracy (LCB-128K, n=80)

| Model | TQ44 V3 | FP4 g32 V3 | FP4 advantage |
|---|---|---|---|
| MiniMax-M2.5 | 75.0 / 78.1 (no-AITER / AITER) | 78.1 | flat to small |
| Qwen2.5-72B | 51.6 | **56.3** | **+4.7 pp** |

FP4-g32 wins by larger margins on models with more concentrated KV spectra (lower `d_eff` — Qwen2.5-7B `d_eff=4.2` vs MiniMax `d_eff=6.7`), where per-group scaling captures dominant directions better than V3's single per-token norm.

#### Running tests

```bash
# Unit + kernel tests
python -m pytest tests/kernels/fp4_g32/ -v

# End-to-end parity vs fp16 SDPA
python -m pytest tests/kernels/fp4_g32/test_e2e_parity.py -v
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
