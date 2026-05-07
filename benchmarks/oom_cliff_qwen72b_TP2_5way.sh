#!/usr/bin/env bash
# Qwen2.5-72B OOM-cliff study — TP=2, 5-way apples-to-apples.
#
# Identical config to the TP=1 4-way run EXCEPT:
#   - TP=2  (2× KV headroom → OOM cliff should not appear)
#   - Adds v1 (plain TQ Triton, no V2/V3/V4) as 5th leg
#   - All 5 legs from feat/tq-hip-full (no branch switching)
#   - GPUs 5,6 (309 GB each, confirmed clean)
#   - Single server start per backend; CSWEEP runs C=16,32,64 in one pass
#
# Purpose: confirm the TP=1 throughput gains are memory-pressure driven —
# at TP=2 KV cache never saturates, so all backends should converge.
#
# Result tags: qwen_tp2_{bf16,tqv1,tqv3,tqhip,tqv4}

set -uo pipefail

REPO="/shareddata/adrana/workspace/vllm-pr"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SH="${SCRIPT_DIR}/results_m25_chart_replica_mi355_8k1k/run.sh"

MODEL="${MODEL:-/shareddata/Qwen/Qwen2.5-72B-Instruct}"
TP=2; EP_SIZE=1
ISL=24576; OSL=2048; MAX_MODEL_LEN=$((ISL + OSL + 256))
NWARMUP=4; GPU_MEM_UTIL=0.85; BLOCK_SIZE=32
CSWEEP="16 32 64"   # all three C values in one server pass
NPROMPT=64          # pre-generate 64 prompts; run.sh slices to C for each point
GPUS="${GPUS:-5,6}"; PORT="${PORT:-9097}"

OUTDIR="${SCRIPT_DIR}/results_oom_cliff_qwen72b_TP2_5way"
mkdir -p "${OUTDIR}/logs"
LOGFILE="${OUTDIR}/run.log"

export VLLM_FLYDSL_ROOT="${VLLM_FLYDSL_ROOT:-/root/FlyDSL}"
export VLLM_FLYDSL_PKGS="${VLLM_FLYDSL_PKGS:-/root/FlyDSL/build-fly/python_packages}"

DEFAULT_COMP_CFG='{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
DENSE_SIZES='[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,24,32,40,48,56,64,80,96,128]'
DENSE_COMP_CFG='{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":'"${DENSE_SIZES}"'}'

exec > >(tee -a "${LOGFILE}") 2>&1
ts() { date -Iseconds; }
log() { echo "[$(ts)] $*"; }

log "================================================================"
log "Qwen2.5-72B OOM-cliff  TP=2  24K/2K  5-way  GPUs=${GPUS}"
log "  Legs: BF16 | TQ-v1 | TQ-v3 | TQ-HIP | TQ-v4"
log "  C sweep: ${CSWEEP}  — feat/tq-hip-full"
log "================================================================"

# Sanity checks
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

# ─── generic leg runner ───────────────────────────────────────────────────────
# Runs CSWEEP="16 32 64" in one shot via run.sh.
run_leg() {
    local label="$1" tag="$2" kv_dtype="$3" comp_cfg="$4"; shift 4

    log "----------------------------------------------------------------"
    log "${label}  TP=${TP}  GPUs=${GPUS}  tag=${tag}"
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
        tail -5 "${OUTDIR}/logs/${tag}.log" | sed 's/^/  /' | tee -a "${LOGFILE}" || true
    else
        log "  ${label}: DONE (${elapsed}s)"
        python3 -c "
import json, pathlib
for c in (16, 32, 64):
    p = pathlib.Path('${OUTDIR}/${tag}/c%d.json' % c)
    if p.exists():
        d = json.loads(p.read_text())
        print(f'    C={c}: TPS={d[\"output_throughput\"]:.1f}  TPOT={d[\"median_tpot_ms\"]:.0f}ms  TTFT={d[\"mean_ttft_ms\"]/1000:.1f}s')
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
run_leg "BF16 baseline" "qwen_tp2_bf16" "auto" "${DEFAULT_COMP_CFG}" \
    VLLM_TQ_DECODE_V3=0 VLLM_TQ_DECODE_V2=0 VLLM_TQ_DECODE_V4=0 \
    VLLM_TQ_FP16_CENTROIDS=0 VLLM_TQ_LUT_RESIDENT=0 VLLM_TQ_SOA_FUSION=0

# ─── Leg 2: TQ v1 (plain Triton, no V3/V4) ───────────────────────────────────
run_leg "TQ Triton v1" "qwen_tp2_tqv1" "turboquant_4bit_nc" "${DEFAULT_COMP_CFG}" \
    VLLM_TQ_DECODE_V3=0 VLLM_TQ_DECODE_V2=0 VLLM_TQ_DECODE_V4=0 \
    VLLM_TQ_FP16_CENTROIDS=0 VLLM_TQ_LUT_RESIDENT=0 VLLM_TQ_SOA_FUSION=0

# ─── Leg 3: TQ v3 ────────────────────────────────────────────────────────────
run_leg "TQ Triton v3" "qwen_tp2_tqv3" "turboquant_4bit_nc" "${DEFAULT_COMP_CFG}" \
    VLLM_TQ_DECODE_V3=1 VLLM_TQ_DECODE_V2=0 VLLM_TQ_DECODE_V4=0 \
    VLLM_TQ_FP16_CENTROIDS=0 VLLM_TQ_LUT_RESIDENT=0 VLLM_TQ_SOA_FUSION=0

# ─── Leg 4: TQ HIP SoA-fusion ────────────────────────────────────────────────
run_leg "TQ HIP SoA-fusion" "qwen_tp2_tqhip" "turboquant_4bit_nc" "${DEFAULT_COMP_CFG}" \
    VLLM_TQ_DECODE_V3=0 VLLM_TQ_DECODE_V2=0 VLLM_TQ_DECODE_V4=0 \
    VLLM_TQ_SOA_FUSION=1 VLLM_TQ_SOA_FUSION_DECODE_BF16Q_PV_MFMA=1 \
    VLLM_TQ_SOA_FUSION_STORE=1 VLLM_TQ_SOA_FUSION_WHT_BUTTERFLY=1

# ─── Leg 5: TQ FlyDSL v4 (dense cudagraph) ───────────────────────────────────
run_leg "TQ FlyDSL v4" "qwen_tp2_tqv4" "turboquant_4bit_nc" "${DENSE_COMP_CFG}" \
    VLLM_TQ_DECODE_V3=0 VLLM_TQ_DECODE_V2=0 VLLM_TQ_DECODE_V4=1 \
    VLLM_TQ_FP16_CENTROIDS=0 VLLM_TQ_LUT_RESIDENT=0 VLLM_TQ_SOA_FUSION=0

# ─── Summary ─────────────────────────────────────────────────────────────────
log "================================================================"
log "TP=2 5-way SUMMARY"
log "================================================================"
python3 - <<'PYEOF'
import json
from pathlib import Path

OUTDIR = Path("/shareddata/adrana/workspace/vllm-pr/benchmarks/results_oom_cliff_qwen72b_TP2_5way")
legs = [
    ("BF16",   "qwen_tp2_bf16"),
    ("TQ-v1",  "qwen_tp2_tqv1"),
    ("TQ-v3",  "qwen_tp2_tqv3"),
    ("TQ-HIP", "qwen_tp2_tqhip"),
    ("TQ-v4",  "qwen_tp2_tqv4"),
]

def load(tag, c):
    p = OUTDIR / tag / f"c{c}.json"
    return json.loads(p.read_text()) if p.exists() else None

def f(d, key, scale=1, fmt=".1f"):
    return format(d[key]*scale, fmt) if d else "n/a"

print(f"\n  {'Leg':<10} {'C16 TPS':>8} {'C32 TPS':>8} {'C64 TPS':>8}  "
      f"{'C16 TPOT':>9} {'C32 TPOT':>9} {'C64 TPOT':>9}  "
      f"{'C16 TTFT':>9} {'C32 TTFT':>9} {'C64 TTFT':>9}")
print("  " + "-"*105)
for name, tag in legs:
    d = {c: load(tag, c) for c in (16,32,64)}
    print(f"  {name:<10} "
          f"{f(d[16],'output_throughput'):>8} {f(d[32],'output_throughput'):>8} {f(d[64],'output_throughput'):>8}  "
          f"{f(d[16],'median_tpot_ms',fmt='.0f'):>8}ms {f(d[32],'median_tpot_ms',fmt='.0f'):>8}ms {f(d[64],'median_tpot_ms',fmt='.0f'):>8}ms  "
          f"{f(d[16],'mean_ttft_ms',1/1000,'.1f'):>8}s {f(d[32],'mean_ttft_ms',1/1000,'.1f'):>8}s {f(d[64],'mean_ttft_ms',1/1000,'.1f'):>8}s")

# vs BF16
print()
bf = {c: load("qwen_tp2_bf16", c) for c in (16,32,64)}
if all(bf.values()):
    print(f"  {'Leg':<10} {'vs BF16 @C=16':>14} {'vs BF16 @C=32':>14} {'vs BF16 @C=64':>14}")
    print("  " + "-"*60)
    for name, tag in legs[1:]:
        row = f"  {name:<10}"
        for c in (16,32,64):
            d = load(tag, c)
            row += f"  {d['output_throughput']/bf[c]['output_throughput']:>12.2f}x" if d else f"  {'n/a':>12}"
        print(row)
PYEOF
log "Done. Results: ${OUTDIR}/"
