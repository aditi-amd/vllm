#!/usr/bin/env bash
# MiniMax-M2.5 OOM-cliff study — TP=2, 5-way apples-to-apples.
#
# Config chosen to put BF16 KV cache at saturation:
#   - TP=2, EP=2: model=215 GB → 108 GB/GPU, leaving ~155 GB/GPU for KV
#   - ISL=64K, OSL=1K: 8.3 GB KV/req (BF16) → cliff at C≈37
#     C=16 = below cliff, C=32 = at cliff, C=64 = deep into cliff
#   - TQ 4-bit (4× KV compression) pushes cliff to C≈150 → no saturation
#
# All 5 legs use the SAME repo (feat/tq-hip-full), SAME backend
# (ROCM_AITER_UNIFIED_ATTN + soft-pin), SAME GPUs — true apples-to-apples.
# No branch switching needed.
#
# Legs:
#   1. BF16         — kv_dtype=auto, all TQ flags off
#   2. TQ v1        — kv_dtype=turboquant_4bit_nc, no V2/V3/V4 (plain Triton)
#   3. TQ v3        — VLLM_TQ_DECODE_V3=1
#   4. TQ HIP       — VLLM_TQ_SOA_FUSION=1 + BF16Q_PV_MFMA=1
#   5. TQ v4        — VLLM_TQ_DECODE_V4=1 + dense cudagraph capture
#
# GQA-6 note: MiniMax-M2.5 has 8 KV heads, 48 query heads → group_size=6.
#   HIP leg uses the GQA-6 soa_bf16q_pv_mfma_decode_gqa6.so kernel.
#   v4 leg uses the kernels.tq_decode_v4_gqa6 FlyDSL sibling.
#
# Runs on GPUs 4,7 (clean, 309 GB each) in parallel with Qwen TP=2 on 5,6.

set -uo pipefail

REPO="/shareddata/adrana/workspace/vllm-pr"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SH="${SCRIPT_DIR}/results_m25_chart_replica_mi355_8k1k/run.sh"

MODEL="${MODEL:-/shareddata/larryli2/MiniMax-M2.5}"
TP=2; EP_SIZE=2
ISL=65536; OSL=1024; MAX_MODEL_LEN=$((ISL + OSL + 512))
NWARMUP=4; GPU_MEM_UTIL=0.85; BLOCK_SIZE=32
CSWEEP="16 32 64"
NPROMPT=64
GPUS="${GPUS:-4,7}"; PORT="${PORT:-9098}"

OUTDIR="${SCRIPT_DIR}/results_oom_cliff_minimax_TP2_5way"
mkdir -p "${OUTDIR}/logs"
LOGFILE="${OUTDIR}/run.log"

export VLLM_FLYDSL_ROOT="${VLLM_FLYDSL_ROOT:-/root/FlyDSL}"
export VLLM_FLYDSL_PKGS="${VLLM_FLYDSL_PKGS:-/root/FlyDSL/build-fly/python_packages}"

DEFAULT_COMP_CFG='{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
# Dense capture covers every B in [1..64] — eliminates JIT spec overhead for v4
DENSE_SIZES='[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,72,80,88,96,104,112,120,128]'
DENSE_COMP_CFG='{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":'"${DENSE_SIZES}"'}'

exec > >(tee -a "${LOGFILE}") 2>&1
ts() { date -Iseconds; }
log() { echo "[$(ts)] $*"; }

log "================================================================"
log "MiniMax-M2.5 OOM-cliff  TP=2  ISL=64K/1K  5-way  GPUs=${GPUS}"
log "  Legs: BF16 | TQ-v1 | TQ-v3 | TQ-HIP (GQA-6) | TQ-v4 (GQA-6)"
log "  C sweep: ${CSWEEP}  cliff at C≈37 for BF16, C≈150 for TQ"
log "  feat/tq-hip-full  — apples-to-apples, same backend all legs"
log "================================================================"

# ─── Sanity checks ───────────────────────────────────────────────────────────
if grep -q "incompatible with this layer" "${REPO}/vllm/platforms/rocm.py" 2>/dev/null; then
    log "Soft-pin: OK ✓"
else
    log "FATAL: soft-pin missing on feat/tq-hip-full"; exit 1
fi

FIRST_GPU="${GPUS%%,*}"
free_gb=$(HIP_VISIBLE_DEVICES="${FIRST_GPU}" python3 -c \
    "import torch; f,_=torch.cuda.mem_get_info(0); print(int(f/1e9))" 2>/dev/null || echo 0)
log "GPU${FIRST_GPU} pre-flight: ${free_gb} GB free"
[[ ${free_gb} -lt 250 ]] && { log "FATAL: GPU${FIRST_GPU} not clean (${free_gb} GB)"; exit 1; }

# Verify HIP GQA-6 .so is present
GQA6_SO="${REPO}/vllm/v1/attention/ops/turboquant_soa_fusion/soa_bf16q_pv_mfma_decode_gqa6.so"
if [[ -f "${GQA6_SO}" ]]; then
    log "HIP GQA-6 .so: OK ✓ ($(du -sh ${GQA6_SO} | cut -f1))"
else
    log "FATAL: GQA-6 .so missing at ${GQA6_SO}"; exit 1
fi

# Verify FlyDSL GQA-6 kernel
python3 -c "
import sys; sys.path.insert(0, '${REPO}')
from vllm.v1.attention.ops.flydsl_turboquant_decode_v4 import is_flydsl_gqa6_available, _try_import_flydsl
_try_import_flydsl()
print('FlyDSL GQA-6: OK ✓' if is_flydsl_gqa6_available() else 'FlyDSL GQA-6: MISSING (v4 will fall back to Triton v3 for GQA-6)')
" 2>/dev/null | grep "GQA-6" | tee -a "${LOGFILE}" || true

# ─── Generic leg runner ───────────────────────────────────────────────────────
run_leg() {
    local label="$1" tag="$2" kv_dtype="$3" comp_cfg="$4"; shift 4

    log "----------------------------------------------------------------"
    log "${label}  TP=${TP} EP=${EP_SIZE}  GPUs=${GPUS}  tag=${tag}"
    log "----------------------------------------------------------------"
    rm -rf "${OUTDIR}/${tag}"
    mkdir -p "${OUTDIR}/${tag}"
    local t0=$(date +%s)

    (cd "${REPO}" && setsid env \
        GPUS="${GPUS}" HIP_VISIBLE_DEVICES="${GPUS}" \
        MODEL="${MODEL}" TP="${TP}" EP_SIZE="${EP_SIZE}" REPO="${REPO}" \
        ISL="${ISL}" OSL="${OSL}" MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
        NPROMPT="${NPROMPT}" CSWEEP="${CSWEEP}" \
        GPU_MEM_UTIL="${GPU_MEM_UTIL}" BLOCK_SIZE="${BLOCK_SIZE}" \
        NWARMUP="${NWARMUP}" \
        PORT="${PORT}" RUN_TAG="${tag}" OUTDIR="${OUTDIR}" \
        SKIP_DONE=0 KV_DTYPE="${kv_dtype}" \
        ATTN_BACKEND="ROCM_AITER_UNIFIED_ATTN" \
        HSA_NO_SCRATCH_RECLAIM=1 \
        COMPILATION_CONFIG="${comp_cfg}" \
        PYTHONPATH="${REPO}:${PYTHONPATH:-}" \
        VLLM_FLYDSL_ROOT="${VLLM_FLYDSL_ROOT}" \
        VLLM_FLYDSL_PKGS="${VLLM_FLYDSL_PKGS}" \
        "$@" \
        bash "${RUN_SH}") > "${OUTDIR}/logs/${tag}.log" 2>&1
    local rc=$? elapsed=$(( $(date +%s) - t0 ))

    if [[ $rc -ne 0 ]]; then
        log "  ${label}: FAILED (rc=${rc}, ${elapsed}s)"
        tail -6 "${OUTDIR}/logs/${tag}.log" | sed 's/^/  /' | tee -a "${LOGFILE}" || true
    else
        log "  ${label}: DONE (${elapsed}s)"
        python3 -c "
import json, pathlib
for c in (16, 32, 64):
    p = pathlib.Path('${OUTDIR}/${tag}/c%d.json' % c)
    if p.exists():
        d = json.loads(p.read_text())
        print(f'    C={c}: TPS={d[\"output_throughput\"]:.1f}  TPOT={d[\"median_tpot_ms\"]:.0f}ms  TTFT={d[\"mean_ttft_ms\"]/1000:.1f}s  E2E={d[\"mean_e2el_ms\"]/1000:.1f}s')
    else:
        print(f'    C={c}: no result')
" 2>/dev/null | tee -a "${LOGFILE}" || true
    fi

    pkill -9 -f "[v]llm serve.*--port[= ]${PORT}\b" 2>/dev/null || true
    log "  Settling 45s for HSA release..."
    sleep 45
    local post_free
    post_free=$(HIP_VISIBLE_DEVICES="${FIRST_GPU}" python3 -c \
        "import torch; f,_=torch.cuda.mem_get_info(0); print(f'{f/1e9:.0f}')" 2>/dev/null || echo "?")
    log "  GPU${FIRST_GPU} free after settle: ${post_free} GB"
}

# ─── Leg 1: BF16 ─────────────────────────────────────────────────────────────
run_leg "BF16 baseline" "m25_tp2_bf16" "auto" "${DEFAULT_COMP_CFG}" \
    VLLM_TQ_DECODE_V3=0 VLLM_TQ_DECODE_V2=0 VLLM_TQ_DECODE_V4=0 \
    VLLM_TQ_FP16_CENTROIDS=0 VLLM_TQ_LUT_RESIDENT=0 VLLM_TQ_SOA_FUSION=0

# ─── Leg 2: TQ v1 (plain Triton, default TQ) ─────────────────────────────────
run_leg "TQ Triton v1 (default)" "m25_tp2_tqv1" "turboquant_4bit_nc" "${DEFAULT_COMP_CFG}" \
    VLLM_TQ_DECODE_V3=0 VLLM_TQ_DECODE_V2=0 VLLM_TQ_DECODE_V4=0 \
    VLLM_TQ_FP16_CENTROIDS=0 VLLM_TQ_LUT_RESIDENT=0 VLLM_TQ_SOA_FUSION=0

# ─── Leg 3: TQ v3 (Triton v3) ────────────────────────────────────────────────
run_leg "TQ Triton v3" "m25_tp2_tqv3" "turboquant_4bit_nc" "${DEFAULT_COMP_CFG}" \
    VLLM_TQ_DECODE_V3=1 VLLM_TQ_DECODE_V2=0 VLLM_TQ_DECODE_V4=0 \
    VLLM_TQ_FP16_CENTROIDS=0 VLLM_TQ_LUT_RESIDENT=0 VLLM_TQ_SOA_FUSION=0

# ─── Leg 4: TQ HIP SoA-fusion (GQA-6 kernel) ─────────────────────────────────
run_leg "TQ HIP SoA-fusion GQA-6" "m25_tp2_tqhip" "turboquant_4bit_nc" "${DEFAULT_COMP_CFG}" \
    VLLM_TQ_DECODE_V3=0 VLLM_TQ_DECODE_V2=0 VLLM_TQ_DECODE_V4=0 \
    VLLM_TQ_SOA_FUSION=1 VLLM_TQ_SOA_FUSION_DECODE_BF16Q_PV_MFMA=1 \
    VLLM_TQ_SOA_FUSION_STORE=1 VLLM_TQ_SOA_FUSION_WHT_BUTTERFLY=1

# ─── Leg 5: TQ FlyDSL v4 GQA-6 (dense cudagraph) ────────────────────────────
run_leg "TQ FlyDSL v4 GQA-6 (dense)" "m25_tp2_tqv4" "turboquant_4bit_nc" "${DENSE_COMP_CFG}" \
    VLLM_TQ_DECODE_V3=0 VLLM_TQ_DECODE_V2=0 VLLM_TQ_DECODE_V4=1 \
    VLLM_TQ_FP16_CENTROIDS=0 VLLM_TQ_LUT_RESIDENT=0 VLLM_TQ_SOA_FUSION=0

# ─── Summary ─────────────────────────────────────────────────────────────────
log "================================================================"
log "MiniMax TP=2 5-way SUMMARY"
log "================================================================"
python3 - <<'PYEOF'
import json
from pathlib import Path

OUTDIR = Path("/shareddata/adrana/workspace/vllm-pr/benchmarks/results_oom_cliff_minimax_TP2_5way")
legs = [
    ("BF16",   "m25_tp2_bf16"),
    ("TQ-v1",  "m25_tp2_tqv1"),
    ("TQ-v3",  "m25_tp2_tqv3"),
    ("TQ-HIP", "m25_tp2_tqhip"),
    ("TQ-v4",  "m25_tp2_tqv4"),
]

def load(tag, c):
    p = OUTDIR / tag / f"c{c}.json"
    return json.loads(p.read_text()) if p.exists() else None

def fmt(d, key, scale=1, spec=".1f"):
    return format(d[key] * scale, spec) if d else "n/a"

print(f"\n  {'Leg':<10} {'C16 TPS':>8} {'C32 TPS':>8} {'C64 TPS':>8}  "
      f"{'C16 TPOT':>9} {'C32 TPOT':>9} {'C64 TPOT':>9}  "
      f"{'C16 TTFT':>9} {'C32 TTFT':>9} {'C64 TTFT':>9}  "
      f"{'C16 E2E':>8} {'C32 E2E':>8} {'C64 E2E':>8}")
print("  " + "-"*130)

for name, tag in legs:
    d = {c: load(tag, c) for c in (16, 32, 64)}
    print(f"  {name:<10} "
          f"{fmt(d[16],'output_throughput'):>8} {fmt(d[32],'output_throughput'):>8} {fmt(d[64],'output_throughput'):>8}  "
          f"{fmt(d[16],'median_tpot_ms',spec='.0f'):>8}ms {fmt(d[32],'median_tpot_ms',spec='.0f'):>8}ms {fmt(d[64],'median_tpot_ms',spec='.0f'):>8}ms  "
          f"{fmt(d[16],'mean_ttft_ms',1/1000):>8}s {fmt(d[32],'mean_ttft_ms',1/1000):>8}s {fmt(d[64],'mean_ttft_ms',1/1000):>8}s  "
          f"{fmt(d[16],'mean_e2el_ms',1/1000):>7}s {fmt(d[32],'mean_e2el_ms',1/1000):>7}s {fmt(d[64],'mean_e2el_ms',1/1000):>7}s")

# vs BF16
print()
bf = {c: load("m25_tp2_bf16", c) for c in (16, 32, 64)}
if all(v is not None for v in bf.values()):
    print(f"  {'Leg':<10}  {'vs BF16 C=16':>14}  {'vs BF16 C=32':>14}  {'vs BF16 C=64':>14}")
    print("  " + "-"*60)
    for name, tag in legs[1:]:
        row = f"  {name:<10}"
        for c in (16, 32, 64):
            d = load(tag, c)
            if d and bf[c]:
                row += f"  {d['output_throughput']/bf[c]['output_throughput']:>12.2f}x"
            else:
                row += f"  {'n/a':>12}"
        print(row)
PYEOF
log "Done. Results: ${OUTDIR}/"
