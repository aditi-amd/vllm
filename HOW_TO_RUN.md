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
VLLM_ROCM_USE_AITER=0         # OFF: triggered TQ44 KV-cache coherence
                              # regression on MiniMax (incoherent decode).
                              # The AITER attention backend is still used via
                              # --attention-backend ROCM_AITER_UNIFIED_ATTN.
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

**Recommended config — butterfly OFF for now (default, safe):**

Both butterfly opts are kept **OFF** by default for now. The store/decode
rotation paths must stay paired, and the butterfly variant has not yet been
validated against the full multi-turn / prefix-cache read path. Run with the
flags at `0` unless you are deliberately enabling butterfly under the
instructions below.

```bash
VLLM_TQ_DECODE_V4=1 VLLM_TQ_DECODE_V3=0 VLLM_TQ_DECODE_V2=0 \
VLLM_TQ_SOA_FUSION_STORE=1 VLLM_TQ_SOA_FUSION=0 \
VLLM_TQ_DECODE_V4_WHT_BUTTERFLY=0 \
VLLM_TQ_STORE_WHT_BUTTERFLY=0 \
HSA_NO_SCRATCH_RECLAIM=1 VLLM_ROCM_USE_AITER=0
```

**Enabling butterfly (opt-in, advanced):** the two butterfly opts above can be
turned on for the throughput gains shown — but only together and under the
constraints documented in this section (decode: D=128 + `bf16` query; store:
D a power of 2 + non-FP8 keys). Keep `VLLM_TQ_DECODE_V4_WHT_BUTTERFLY` and
`VLLM_TQ_STORE_WHT_BUTTERFLY` set to the **same** value so the store-side and
decode-side rotations stay paired; enabling only one mismatches the rotation
and corrupts decode output. Validate accuracy (incl. a multi-turn / prefix-
cache run) before using butterfly in production.

Note: `VLLM_ROCM_USE_AITER=0` is the safe default — `=1` caused TQ44
incoherent-decode on MiniMax in a May ablation. The AITER attention
backend is still active via `--attention-backend ROCM_AITER_UNIFIED_ATTN`.

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
VLLM_TQ_SOA_FUSION_STORE=1
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

**As of 2026-05-28 (MI355X / gfx950) FP4-g32 V3 BEATS TQ44 V3 on long-context decode.**

Clean microbench on an idle MI355 GPU (no co-tenant, 30 measured calls, 10 warmup):

| seqlen (B=1) | FP4-g32 V3 | TQ44 V3 | FP4 vs TQ44 |
|---|---|---|---|
| 8K   | 0.093 ms | 0.092 ms | +1.8% (parity) |
| 32K  | 0.102 ms | 0.140 ms | **−26.9%** |
| 64K  | 0.181 ms | 0.245 ms | **−26.2%** |
| 128K | 0.336 ms | 0.454 ms | **−26.0%** |

Microbench: `tmp_rocprof_microbench.py`, Hq=64 Hk=8 D=128. The −26% lead is essentially flat from 32K onward — both schemes scale linearly with KV length so the relative win is structural, not asymptotic.

NUM_KV_SPLITS=16 → 64 alone improves the FP4 stage-1 kernel by **−64% to −68%** (idle GPU, 32K-128K) — the doubling of grid parallelism on MI355 from 128 → 512 workgroups.

**Model-level confirmation (Qwen2.5-72B / LCB-128K / 20-prompt eval, MI355X x2 TP=2):**

| Run | NUM_KV_SPLITS | Wall time | Throughput delta | Accuracy (n=11 scoreable) |
|---|---|---|---|---|
| OLD default | 16 | 1728 s | (baseline)      | 45.5% (5/11) |
| NEW default | 64 | 1235 s | **+39.9% faster** / **−28.5% wall** | 54.5% (6/11) |

The accuracy delta is well within sample-variance for n=11 (one prompt flipped due to fp32 reduction-order changes at bf16 ULP, both runs returned bit-exact-equivalent outputs in microbench). Errors=9 in both runs come from prompts > 131K input tokens (dataset issue, identical for both).

The headline win came from **doubling NUM_KV_SPLITS from 16 to 64** (HIP default), which lifts the 3D split-KV grid from ~128 workgroups to ~512 workgroups — CDNA4 (gfx950) has the CU budget to absorb the extra parallelism, where the prior default (16) was leaving most CUs idle on long-context decode. See "V3 tuning knobs" below.

**Earlier characterization (still useful context for the dequant plumbing):**
- **4 fp16 scales per token** (vs 1 norm for TQ44 V3) → 8 B/token K-scale metadata vs 2 B; same for V → ~2.7× scale metadata bandwidth.
- **Per-group broadcast multiply** `K_g[16,4,32] × scales[16,4,1]` doesn't fold cleanly into the `[BLOCK_M=16, HEAD_DIM=128]` MFMA tile shape; Triton emits extra reshape and 4-way piecewise-broadcast instructions that TQ44's single scalar-per-token broadcast avoids.
- V also pays an FP4 LUT gather per element (TQ44 V is uniform INT4 so just `idx*scale + zero`, no LUT).

The MFMA instruction (`v_mfma_f32_16x16x32_bf16` on gfx950) is identical for both paths today.

The next horizon is the hardware FP4 MFMA path via `v_mfma_(scale_)f32_16x16x128_f8f6f4` on CDNA4: K is consumed as FP4 directly by the tensor core, decode happens inline at load time. There are obstacles (per-group fp16 scales vs hardware E8M0 expectations) that need separate scheme/encoder work — see the "Cross-scheme follow-ups" section.

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
| `VLLM_FP4_G32_FUSE_Q_ROT` | `0` | Fuse Q@PiT rotation inside the kernel via bf16 MFMA. Off by default (3/11 borderline answer flips at 128K in lcb_128k smoke; also significantly slower at batch ≥ 8). A/B for perf only. |
| `VLLM_FP4_G32_TILE_SIZE_DECODE` | unset → 16 | Decode tile width. Larger values (32) cause severe VGPR spilling on AMD MI3xx at long context. |
| `VLLM_FP4_G32_NUM_STAGES_2D` | `1` (HIP) / `2` (CUDA) | Software pipeline depth for the 2D path (prefill / chunked). On HIP, stages=2 measured −2% on prefill, so default stays 1. |
| `VLLM_FP4_G32_NUM_STAGES_3D` | `2` | Software pipeline depth for the 3D split-KV path (decode). On HIP, stages=2 measured **+9–15%** vs stages=1 across batch=1..32 / ctx=32K..128K with bit-identical accuracy — now the default. Override with stages=3 for extreme low-batch (1seq) decode (+35–47% extra at ctx≥32K, but variable at higher batch). |
| `VLLM_FP4_G32_NUM_STAGES` | unset | **Global override** of both the 2D and 3D `num_stages` defaults. Use the `_2D` / `_3D` variants for finer control. |
| `VLLM_FP4_G32_NUM_KV_SPLITS` | `64` (HIP) / `16` (CUDA) | KV split count for the 3D path. **The big MI355 win.** Old default was 16, new HIP default is 64. Microbench MI355 (B=1, Hq=64): 16→4.48 ms, 32→2.04 ms, **64→1.65 ms** (−63% vs 16, −19% vs 32 = TQ44's level), 128→1.22 ms (best at B=1 only; regresses at B≥2). 64 is best-or-tied across the full sweep B∈{1,2,4,8} × seqlen∈{8K..128K}, so it's the universal default. Coerced to nearest-power-of-2 ≤ requested (Triton `tl.arange` constraint). Bit-exact within bf16 ULP (max abs diff 1.22e-4). |
| `VLLM_FP4_G32_FAST_REDUCE` | `0` | **Experimental online-softmax stage-2 reducer.** Ports TQ44 V3's `_fwd_kernel_stage2` algorithm onto FP4's 3-buffer (segm_output / segm_max / segm_expsum) layout. On MI355 it measured slightly slower (median 56 us) than the default vectorized `reduce_segments` (median 44 us) at the small NUM_SEGMENTS used here, because TQ44's win comes from its packed `Mid_O[B,Hq,splits,D+1]` layout (one combined load per iter) which FP4 V3 does not have. Kept off by default; revisit when/if FP4 stage-1 is restructured to write a packed buffer. |
| `VLLM_FP4_G32_NUM_WARPS_3D` | `2` | Warp count for the 3D split-KV decode kernel. Microbench: warps=4 helps batch=1 only (+3%); regresses at batch≥8 by 30%+. Leave at 2 unless you specifically run batch=1. |
| `VLLM_FP4_G32_AMD_HINTS` | `0` | Apply AMD-specific Triton hints (`waves_per_eu=2`, `matrix_instr_nonkdim=16`, `kpack=2`) to the 3D split-KV stage-1 kernel. Same set used by AMD's own `triton_decode_attention._fwd_kernel_stage2`. Default OFF — the cross-shape sweep on MI355 was unstable: −23% at (B=1, 64K) but **+52% catastrophic** at (B=8, 32K). Opt in per-shape via `=1`, fine-tune with `VLLM_FP4_G32_WAVES_PER_EU` / `VLLM_FP4_G32_MFMA_NONKDIM` / `VLLM_FP4_G32_KPACK`. |
| `VLLM_FP4_G32_BF16_DEQUANT` | `0` | **Optimization B (kept for reference; do NOT enable).** Skips the fp32 round-trip in K/V dequant and does the per-group multiply in bf16. Microbench result: −10..−18% perf with rel-Δ≈2% accuracy drift (bf16 mantissa precision in the scale multiply). The fp32-multiply-then-cast path that AMD/Triton compiles is faster than bf16-throughout. Documented as a negative result. |
| `VLLM_FP4_G32_ACC_SCALE_FUSION` | `0` | **Optimization C (kept for reference; do NOT enable on AMD/Triton today).** Splits the QK dot into G=4 group-dots and applies per-group scales post-MFMA on the [BLOCK_M, TILE] partial; PV split on the output axis. Mathematically lossless. Microbench result on MI300X: **slower** (−8..−33%) because Triton/AMD MFMA prefers one large `[BM,BD]@[BD,T]` dot over four small `[BM,GS]@[GS,T]` dots — the per-dot launch + reshape/permute/split overhead dominates the savings from skipping the per-group broadcast multiply. The right path here is the AITER `batched_gemm_a16wfp4` hardware FP4 MFMA on MI355 (handles the per-group fusion natively). |
| `VLLM_FP4_G32` | — | **Deprecated / no-op**; use `VLLM_FP4_G32_V3=1` instead. |
| `FP4_KV_GROUP_SIZE` | `32` | Per-group element count (16 or 32). 32 is recommended (MSE-optimal, matches MFMA K=32 on gfx950). |
| `FP4FP16_CONSTANT_C` | `0.156` | MSE-optimal scale constant `s = c * absmax`. Don't tune unless you've calibrated. |
| `FP4_KV_TOKEN_NORM` | `0` | Optional per-token L2 normfold before per-group quant. Off by default. |

#### Optimization sweep — what worked, what didn't

Microbench at decode shape (`tmp_rocprof_microbench.py`).

| Optimization | Perf delta (decode) | Accuracy delta | Verdict |
|---|---|---|---|
| **`NUM_KV_SPLITS=64`** (HIP) — new default | **−42% to −73% per call** (MI355, B=1, 64K-128K) | bit-exact (bf16 ULP) | ✅ **shipped as new HIP default** |
| **`NUM_STAGES_3D=2`** (now default) | **+9..+15%** universally | bit-identical | ✅ shipped as new default |
| `NUM_STAGES_3D=3` | +35..+47% at batch=1; +0..−8% at batch≥8 | bit-identical | optional (low-batch only) |
| `TILE_SIZE_DECODE=32` | +3..+13% at long ctx, with stages=2 | bit-identical | safe but small extra win |
| `BF16_DEQUANT=1` (Opt B) | −4..−18% | rel-Δ≈2% | ❌ regression (compiler quirk) |
| `ACC_SCALE_FUSION=1` (Opt C) | −8..−33% | bit-identical | ❌ regression (small-dot overhead > savings) |
| `FUSE_Q_ROT=1` | −47..−40% at batch≥8 | rel-Δ≈6e-3 | ❌ regression (register pressure) |
| `NUM_WARPS_3D=4` | +3% at batch=1; −15..−30% at batch≥8 | bit-identical | ❌ batch-dependent |
| `FAST_REDUCE=1` (online stage-2) | −27% at MI355 | bit-exact | ❌ regression on FP4's 3-buffer layout (would need packed Mid_O to win) |
| `AMD_HINTS=1` (waves_per_eu / mfma_nonkdim / kpack) | mixed: −23% at (B=1, 64K), **+52% at (B=8, 32K)** | bit-identical | ❌ unstable across shapes; per-shape opt-in only |

#### Cross-scheme parity status for TQ44 V3 (corrected)

Three optimizations were originally listed as cross-scheme follow-ups for TQ44 V3. Re-checking the actual code state, they are all already addressed — either shipped, already env-flagged, or already at parity with FP4. **No new code changes were needed for TQ44 V3.** This section documents the actual state so the lead numbers are interpreted correctly.

| Optimization | TQ44 V3 today | FP4-g32 V3 today | Already parity? |
|---|---|---|---|
| `NUM_KV_SPLITS=64` | **64** — backend passes via `vllm/config/attention.py: tq_max_kv_splits_for_cuda_graph=64` ("Opt B"). The function-default `32` in `triton_turboquant_decode_v2.py` is dead code on the production path. | `64` (env default on HIP) | ✅ |
| `num_stages` on stage-1 | env-flagged via **`VLLM_TQ_NUM_STAGES_3D`** at `triton_turboquant_unified_attention.py:1090`; default `1` on HIP / `2` elsewhere | env-flagged via `VLLM_FP4_G32_NUM_STAGES_3D`; default `2` on HIP | ⚠️ different defaults; opt in to match |
| Q-rotation `q @ Pi.T` precision | `(query.float() @ PiT).to(query.dtype).contiguous()` at `triton_turboquant_unified_attention.py:976` — **fp32 GEMM with bf16 output cast** | `(query.float() @ PiT).to(query.dtype).contiguous()` at `triton_unified_attention.py:1591` — **identical** | ✅ |

**Actionable parity tuning for TQ44 V3 on MI355**:

```bash
VLLM_TQ_DECODE_V3=1 \
VLLM_TQ_NUM_STAGES_3D=2 \         # match FP4-g32's pipelining (existing flag)
vllm serve ...
```

This is the **only** runtime difference left. Setting `VLLM_TQ_NUM_STAGES_3D=2` brings TQ44 V3 to full parity with FP4-g32 V3 on tuning knobs.

**Q-rotation note**: a true-bf16 GEMM path (both operands bf16, not just output cast) was tried as "Opt L1" on TQ44 V3 and gave ~0.5 ms ITL win on 8K serving, but broke `TestV1V3TightEquivalence::test_v1_v3_decode_tight` at 2.44e-3 vs 1.5e-3 threshold (1 bf16 ULP from rounding PiT before the matmul). It was reverted for precision parity with V1. The same precision concern applies to FP4-g32 if it ever moves to true-bf16 GEMM — `VLLM_FP4_G32_FUSE_Q_ROT=1` saw 3/11 answer flips at 128K context for the same reason. Neither scheme has a true-bf16-GEMM Q-rot path in production today.

**Important**: enabling `VLLM_TQ_NUM_STAGES_3D=2` will likely re-close most of the 20–42% kernel-level lead FP4-g32 V3 currently has, because part of that lead came from this tuning difference (FP4 defaults to `2`, TQ44 defaults to `1` on HIP) rather than from the FP4 quantization scheme itself. The structurally durable part of FP4's lead (the part that survives at matched tuning) is the FP4 fixed-grid dequant path (no DRAM centroid LUT gather, no per-token L2-norm reduction) plus the future native FP4 MFMA path (`v_mfma_(scale_)f32_16x16x128_f8f6f4`), which only consumes E2M1-encoded codes — TQ44's learned-centroid 4-bit codes cannot use that hardware path.

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

### FP8-g32 (fp8_kv_g32)

**FP4 E2M1 codes + UE8M0 (1-byte E8M0) per-group-of-32 scales.**
Pure Triton, no FlyDSL or HIP dependency. Native scaled F8F6F4 MFMA on QK via
`tl.dot_scaled` → `v_mfma_scale_f32_16x16x128_f8f6f4` on CDNA4 (gfx950/MI355X).

Design notes:
- Same FP4 E2M1 codes as fp4_g32, but scales are stored as **1-byte UE8M0** (power-of-2
  only) instead of 2-byte fp16. Slot size: **136 bytes** per (token, head) vs 144 B in fp4_g32.
- K path uses `tl.dot_scaled` directly: raw FP4 codes + raw E8M0 scale bytes consumed by
  the native scaled MFMA without software dequant.
- V path uses **arithmetic FP4 decode** (default) → bf16 multiply by E8M0 scale → bf16 `tl.dot`.
  Arithmetic decode reconstructs each FP4 E2M1 value from its 4-bit code using integer ALU
  bit-assembly (`sign × 2^exp × mantissa`) instead of a 16-entry LUT gather, eliminating the
  costly per-element vector fetch. Controlled by `VLLM_FP8_G32_V_ARITH` (default `1` = ON).
  Bit-identical to the LUT path; verified across all 16 FP4 codes.
- K is Hadamard-rotated before quantization (same as TQ44 / fp4_g32).
- Layout is **AoS** (codes + scales interleaved per slot). A SoA layout migration
  (Path 4) is planned to match TQ44 V3's coalesced scale loads.

#### Launch (canonical defaults — just set one env var)

```bash
HIP_VISIBLE_DEVICES=0,1 \
VLLM_ROCM_USE_AITER=1 \
HSA_NO_SCRATCH_RECLAIM=1 \
VLLM_FP8_G32_V3=1 \
    python -m vllm.entrypoints.openai.api_server \
        --model /shareddata/Qwen/Qwen2.5-72B-Instruct \
        --port 9421 \
        --tensor-parallel-size 2 \
        --gpu-memory-utilization 0.85 \
        --max-model-len 9472 \
        --kv-cache-dtype fp8_kv_g32 \
        --block-size 32 \
        --trust-remote-code \
        --no-enable-prefix-caching \
        --attention-backend ROCM_AITER_UNIFIED_ATTN \
        --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
```

`VLLM_FP8_G32_V3=1` is the only required env var. All other knobs default to the
production-ready values below.

#### Accuracy A/B vs TQ44 V3 (LCB-128K, ~30 min, needs 4 GPUs)

```bash
bash benchmarks/lcb_qwen72b_128k_tq44_vs_fp8.sh
```

Runs TQ44 V3 and fp8_g32 V3 in parallel on separate GPU pairs (GPUs 0,1 vs 2,3
by default). Prints accuracy side by side at the end.

#### Throughput A/B vs TQ44 V3 (ISL=8192, OSL=1024, C=64, ~15 min, needs 4 GPUs)

```bash
python tmp_perf_qwen72b_8k1k_c64.py
```

Reports OutTPS / TPOT / TTFT / ITL side by side for TQ44 V3 vs fp8_g32 V3.

#### Tuning knobs

| Env var | Default | Purpose |
|---|---|---|
| `VLLM_FP8_G32_V3` | `0` | Enable the V3 unified kernel (set to `1`) |
| `VLLM_FP8_G32_V3_DOT_KIND` | `dot_scaled` | QK path: `dot_scaled` (native scaled F8F6F4 MFMA, recommended) or `fp8mfma` (software dequant to FP8 + plain dot) |
| `VLLM_FP8_G32_NUM_KV_SPLITS` | `16` | 3D split-KV parallelism. Matches TQ44 V3 default; ~36 tile iterations per CTA at 8K context |
| `VLLM_FP8_G32_NUM_STAGES_3D` | `3` (HIP) / `2` (non-HIP) | Software pipeline depth for decode stage-1. 3 is +2.2% OutTPS vs 2 on Qwen-72B 8K C=64 |
| `VLLM_FP8_G32_FUSE_Q_ROT` | `0` | Fuse Q@PiT rotation into kernel prologue. Wins at low batch; **regresses at C=64** on Qwen-72B. Opt-in only |
| `VLLM_FP8_G32_V_BF16` | `0` | Keep V dequant multiply in bf16 (skip fp32 round-trip). Bit-exact (UE8M0 scales are exact pow2). Wash at C=64 serving |
| `VLLM_FP8_G32_V_DOT_SCALED` | `0` | Use fp8×fp8 `dot_scaled` for PV (native scaled FP8 MFMA, 2× bf16 ceiling). **Forces TILE_SIZE=128 and NUM_STAGES_3D=1** — regresses at C=64 due to launch overhead. Opt-in only |
| `VLLM_FP8_G32_AMD_HINTS` | `0` | Pass `waves_per_eu=2`, `matrix_instr_nonkdim=16`, `kpack=2` to the AMD Triton backend. Experimental per-workload tuning |
| `VLLM_FP8_G32_SOA_SCALES` | `0` | **Path 4 (SoA scales).** Store per-group K/V scales in a contiguous per-block region instead of interleaved per-slot. Collapses the per-tile scattered scale loads into one coalesced load (mirrors TQ44 V3). Bit-identical (same codes/scales, only byte placement changes). Store + both decode kernels read the same module-load flag, so they always agree within a process. Default OFF pending the serving A/B vs TQ44 V3. Targets the main remaining decode gap vs TQ44 V3 |
| `VLLM_FP8_G32_V_ARITH` | **`1` (ON by default)** | Arithmetic FP4 V-decode: reconstructs each FP4 E2M1 code via integer ALU bit-assembly instead of a 16-entry LUT gather. Reduces VFetchInsts by ~48% on the decode kernel (1272 → 662). Bit-identical to the LUT path across all 16 codes. Set to `0` to fall back to LUT gather (debugging only) |
| `VLLM_FP8_G32_ARCHB` | `0` | Use Arch B codebook (multiply E8M0 scale by constant `c`). Default is Arch A. Do not change without re-quantizing the model |

#### What's faster, what's slower vs TQ44 V3

| Dimension | fp8_g32 V3 | TQ44 V3 | Notes |
|---|---|---|---|
| Slot size | 136 B | 136 B | Same (fp8_g32 saves 8 B vs fp4_g32's 144 B) |
| QK MFMA | native scaled F8F6F4 (K=128) | bf16 (K=32) | fp8_g32 has higher instruction throughput ceiling |
| V MFMA | bf16 (K=32) | bf16 (K=32) | Same |
| V decode | **arithmetic** (default, `V_ARITH=1`) — integer ALU bit-assembly, no LUT gather | affine `idx*scale+zero` (irreducible) | fp8_g32 wins: ~48% fewer VFetchInsts |
| K-scale loads | **AoS** — 1 byte per group scattered per slot | **SoA** — contiguous per block | TQ44 V3 wins: ~25% vs ~3% L1 efficiency |
| V-scale loads | **AoS** — scattered per slot | **SoA** — contiguous per block | TQ44 V3 wins: same reason |
| Serving Qwen-72B 8K C=64 | **+0.7% OutTPS** (511.9 vs 508.5) | baseline | V_ARITH flipped the delta: −0.6% → +0.7% |

**Serving benchmark results (Qwen2.5-72B, ISL=8K, OSL=1K, C=64, TP=2, MI355X):**

| Kernel | OutTPS | vs TQ44 V3 |
|---|---|---|
| TQ44 V3 optimized | 508.5 | baseline |
| fp8_g32 V3 (V_ARITH=OFF) | 504.0 | −0.9% |
| **fp8_g32 V3 (V_ARITH=ON, default)** | **511.9** | **+0.7% ✅** |

V_ARITH is the decisive optimization: replacing the per-element LUT gather in the V decode path
with integer ALU bit-assembly reduces VFetchInsts by ~48% (1272 → 662), a ~5.5% kernel
speedup, flipping fp8_g32 from slightly behind to slightly ahead of the optimized TQ44 V3
Triton kernel at serving concurrency (C=64).

**LCB-128K accuracy (Qwen2.5-72B and MiniMax-M2.5, MI355X x2, TP=2):**

| Model | TQ44 V3 optimized | fp8_g32 V3 (V_ARITH=ON) | Δ |
|---|---|---|---|
| Qwen2.5-72B | 56.52% strict | 57.61% strict | +1.1 pp |
| MiniMax-M2.5 | 64.13% strict | **68.48%** strict | **+4.4 pp** |

The MiniMax accuracy advantage is a quantization quality difference: fp8_g32's UE8M0
per-group-of-32 scales adapt better to MiniMax's attention value distributions than TQ44's
learned 4-bit codebook. V_ARITH itself is bit-identical to the LUT path and does not affect
accuracy.

**Note — SoA scales (Path 4):** `VLLM_FP8_G32_SOA_SCALES` migrates K/V scale byte placement
from AoS to SoA. Implemented, validated bit-identical, but a wash at C=64 serving — after
V_ARITH the bottleneck shifted away from scale loads. Available as opt-in; not recommended
for production.

---

### FP8-g32 FlyDSL V4 (recommended on MI355X / gfx950)

Best fp8_g32 throughput. Direct FlyDSL port of the bug-free TQ V4 decode kernel
with three CDNA4 hardware optimizations now ON by default. Requires the FlyDSL
build (see Part 1) and gfx950. Supports GQA group sizes 8 and 16 (Qwen2.5/3,
canonical kernel) and 6 (MiniMax-M2.5, sibling kernel).

#### Launch — defaults are production-ready

```bash
HIP_VISIBLE_DEVICES=0,1 \
VLLM_FLYDSL_ROOT=/opt/FlyDSL \
VLLM_FLYDSL_PKGS=/opt/FlyDSL/build-fly/python_packages \
HSA_NO_SCRATCH_RECLAIM=1 \
VLLM_FP8_G32_DECODE_V4=1 \
VLLM_FP8_G32_V3=1 \
    vllm serve /shareddata/Qwen/Qwen2.5-72B-Instruct \
        --tensor-parallel-size 2 \
        --gpu-memory-utilization 0.85 \
        --kv-cache-dtype fp8_kv_g32 \
        --block-size 32 \
        --trust-remote-code \
        --no-enable-prefix-caching \
        --attention-backend ROCM_AITER_UNIFIED_ATTN \
        --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
```

`VLLM_FP8_G32_DECODE_V4=1` enables the FlyDSL decode kernel for eligible
layers (HEAD_SIZE=128, GQA in {6, 8, 16}, no sinks/SWA). `VLLM_FP8_G32_V3=1`
is kept for the Triton fallback (continuation prefill, ineligible layers).
All three CDNA4 optimizations below are ON by default — no further env vars
required.

#### CDNA4 hardware optimizations (defaults ON)

| Env var | New default | Purpose |
|---|---|---|
| `VLLM_FP8_G32_DECODE_V4_QK_SCALED` | **`1`** | Native scaled FP4xFP8 MFMA for QK via `mfma_scale_f32_16x16x128_f8f6f4` on gfx950. Replaces the bf16 MFMA QK path with a single K=128 scaled MFMA — higher instruction throughput ceiling. Set to `0` to fall back to the bf16 path. |
| `VLLM_FP8_G32_DECODE_V4_V_CVT` | **`1`** | Native scaled FP4→bf16 hardware CVT (`cvt_scalef32_pk_bf16_fp4`) for V dequant. Replaces the software LUT FP4 decode + scale multiply with a single hardware CVT. Requires the HW V-transpose LDS layout (gfx950). Set to `0` to fall back to the software LUT. |
| `VLLM_FP8_G32_DECODE_V4_Q_HOIST` | **`1`** | Hoist the loop-invariant scaled-MFMA Q operand (FP8 build in STEP C) out of the K-tile loop and reuse it across tiles. No-op unless `_QK_SCALED=1`. Set to `0` to disable. |

To run the leg with all three optimizations OFF (bf16 MFMA QK + software-LUT V dequant) for A/B comparison:

```bash
VLLM_FP8_G32_DECODE_V4_QK_SCALED=0 \
VLLM_FP8_G32_DECODE_V4_V_CVT=0 \
VLLM_FP8_G32_DECODE_V4_Q_HOIST=0 \
... (rest of launch line)
```

#### Validated results

**Serving throughput (MiniMax-M2.5, ISL=32K, OSL=1K, C=64, N=128, TP=2, MI355X — 5-way decode-kernel sweep):**

| Kernel | OutTPS | TPOT p50 | ITL p50 | vs TQ44 V3 |
|---|---:|---:|---:|---:|
| v1 (legacy Triton) | 86.8 | 686.1 ms | 595.8 ms | −59% |
| TQ44 V3 (Triton) | 210.7 | 281.9 ms | 170.8 ms | baseline |
| TQ44 V4 FlyDSL (+ butterfly) | 254.0 | 188.3 ms | 64.9 ms | +20.5% |
| fp8_g32 V3 (Triton) | 199.2 | 297.9 ms | 189.9 ms | −5.5% |
| **fp8_g32 V4 FlyDSL (defaults: QK_SCALED + V_CVT + Q_HOIST)** | **303.4** | **173.7 ms** | **54.6 ms** | **+44.0%** |

The fp8_g32 V4 FlyDSL leg with all three optimizations on is the fastest
decode path measured on MI355X at C=64 serving — beating TQ44 V4 FlyDSL by
+19.4% OutTPS and the optimized TQ44 V3 Triton kernel by +44.0%.

**LCB-128K accuracy (Qwen2.5-72B, YaRN ×4 → 131072, TP=2, MI355X):**

| Configuration | acc_strict | acc_answered | n correct / null |
|---|---:|---:|---:|
| fp8_g32 V4 FlyDSL (defaults ON) | **61.96%** | **72.15%** | 57 / 12 |

Accuracy is on par with fp8_g32 V3 Triton (the three hardware
optimizations are bit-similar to within bf16 ULP of the reference path).

---

## Part 3 — Full launch example (MiniMax-M2.5, FlyDSL v4, TP=2)

Butterfly is kept OFF here (the safe default for now). To enable it, set both
`VLLM_TQ_DECODE_V4_WHT_BUTTERFLY` and `VLLM_TQ_STORE_WHT_BUTTERFLY` to `1`
together — see "Enabling butterfly (opt-in, advanced)" above for the
constraints.

```bash
HIP_VISIBLE_DEVICES=4,5 \
VLLM_FLYDSL_ROOT=/opt/FlyDSL \
VLLM_FLYDSL_PKGS=/opt/FlyDSL/build-fly/python_packages \
PYTHONPATH="${VLLM_FLYDSL_ROOT}:${VLLM_FLYDSL_PKGS}" \
VLLM_TQ_DECODE_V4=1 VLLM_TQ_DECODE_V3=0 VLLM_TQ_DECODE_V2=0 \
VLLM_TQ_SOA_FUSION_STORE=1 VLLM_TQ_SOA_FUSION=0 \
VLLM_TQ_DECODE_V4_WHT_BUTTERFLY=0 \
VLLM_TQ_STORE_WHT_BUTTERFLY=0 \
HSA_NO_SCRATCH_RECLAIM=1 VLLM_ROCM_USE_AITER=0 \
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
