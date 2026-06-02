#!/usr/bin/env bash
# Qwen2.5-72B-Instruct  LCB-128K  2-way PARALLEL
#   Leg 1: TQ44 V3  (turboquant_4bit_nc, VLLM_TQ_DECODE_V3=1)        HIPs 0,1, port 9420
#   Leg 2: fp8_g32 V3 (fp8_kv_g32,        VLLM_FP8_G32_V3=1)         HIPs 2,3, port 9421
#
# Mirrors the gsm8k accuracy A/B env exactly (no SoA, no HIP); both legs run
# AITER unified attention + FULL_AND_PIECEWISE cudagraphs.
#
# Qwen2.5-72B native ctx = 32K; YaRN factor=4 -> 128K (Qwen-recommended).
# LCB-128K = 113 long-code QA problems; each prompt is up to 128K tokens.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/lcb_local_eval.py"
MODEL="/shareddata/Qwen/Qwen2.5-72B-Instruct"
DATASET="/shareddata/adrana/workspace/long-code-bench/data/LQA/128K.json"
OUTROOT="${SCRIPT_DIR}/results_lcb_qwen72b_128k_tq44_vs_fp8"
GPUS_TQ="${GPUS_TQ:-0,1}"
GPUS_FP8="${GPUS_FP8:-2,3}"
PORT_TQ="${PORT_TQ:-9420}"
PORT_FP8="${PORT_FP8:-9421}"
CONCURRENCY="${CONCURRENCY:-4}"
MAX_TOKENS="${MAX_TOKENS:-128}"

mkdir -p "${OUTROOT}/logs"

ts() { date -Iseconds; }

YARN_OVERRIDE='{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}'

cleanup_server() {
    local pid="$1" port="$2" gpus="$3"
    [[ -n "${pid}" ]] && kill -15 "${pid}" 2>/dev/null || true
    sleep 4
    [[ -n "${pid}" ]] && kill -9 "${pid}" 2>/dev/null || true
    pkill -9 -f "vllm.entrypoints.*--port ${port}" 2>/dev/null || true
    sleep 4
}

run_leg() {
    local LABEL="$1" GPUS="$2" PORT="$3" KV="$4" EXTRA_ENV="$5"
    local SLOG="${OUTROOT}/logs/server_${LABEL}.log"
    local ELOG="${OUTROOT}/logs/eval_${LABEL}.log"
    local OUT="${OUTROOT}/qwen72b_${LABEL}.json"

    echo "[$(ts)] [${LABEL}] BOOT  HIP=${GPUS}  port=${PORT}  kv=${KV}  env=[${EXTRA_ENV}]" \
        | tee -a "${OUTROOT}/driver.log"

    env HIP_VISIBLE_DEVICES="${GPUS}" VLLM_ROCM_USE_AITER=1 \
        HSA_NO_SCRATCH_RECLAIM=1 \
        ${EXTRA_ENV} \
        python3 -m vllm.entrypoints.openai.api_server \
            --model "${MODEL}" --port "${PORT}" \
            --tensor-parallel-size 2 \
            --gpu-memory-utilization 0.85 \
            --max-model-len 131072 \
            --kv-cache-dtype "${KV}" \
            --block-size 32 --trust-remote-code \
            --no-enable-prefix-caching --no-enable-log-requests \
            --hf-overrides "${YARN_OVERRIDE}" \
            --attention-backend ROCM_AITER_UNIFIED_ATTN \
            --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
            > "${SLOG}" 2>&1 &
    local PID=$!
    echo "[$(ts)] [${LABEL}] server PID=${PID}" | tee -a "${OUTROOT}/driver.log"

    local elapsed=0
    while ! curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; do
        sleep 20; elapsed=$((elapsed+20))
        if ! kill -0 "${PID}" 2>/dev/null; then
            echo "[$(ts)] [${LABEL}] FATAL: server died (see ${SLOG})" | tee -a "${OUTROOT}/driver.log"
            return 1
        fi
        [[ $elapsed -ge 2400 ]] && {
            echo "[$(ts)] [${LABEL}] FATAL: server boot timeout (>40 min)"
            kill -9 "${PID}"; return 1
        }
        (( elapsed % 60 == 0 )) && echo "[$(ts)] [${LABEL}] ...loading (${elapsed}s)" | tee -a "${OUTROOT}/driver.log"
    done
    echo "[$(ts)] [${LABEL}] server ready (${elapsed}s)" | tee -a "${OUTROOT}/driver.log"

    python3 "${RUNNER}" \
        --port "${PORT}" --model "${MODEL}" \
        --label "qwen72b_${LABEL}" \
        --output "${OUT}" \
        --data-path "${DATASET}" \
        --concurrency "${CONCURRENCY}" \
        --max-tokens "${MAX_TOKENS}" \
        > "${ELOG}" 2>&1
    local RC=$?
    echo "[$(ts)] [${LABEL}] eval rc=${RC}" | tee -a "${OUTROOT}/driver.log"
    cleanup_server "${PID}" "${PORT}" "${GPUS}"
    return $RC
}

echo "[$(ts)] Qwen72B LCB-128K 2-way PARALLEL on HIP=${GPUS_TQ}/${GPUS_FP8}" | tee -a "${OUTROOT}/driver.log"

# Launch both legs in parallel
run_leg "tq44_v3"     "${GPUS_TQ}"  "${PORT_TQ}"  "turboquant_4bit_nc" \
    "VLLM_TQ_DECODE_V4=0 VLLM_TQ_DECODE_V3=1 VLLM_TQ_DECODE_V2=0 VLLM_TQ_SOA_FUSION=0 VLLM_TQ_SOA_FUSION_STORE=0" &
PID_TQ=$!

run_leg "fp8_g32_v3"  "${GPUS_FP8}" "${PORT_FP8}" "fp8_kv_g32" \
    "VLLM_FP8_G32_V3=1" &
PID_FP8=$!

echo "[$(ts)] both legs launched: tq44 PID=${PID_TQ}, fp8 PID=${PID_FP8}" | tee -a "${OUTROOT}/driver.log"

wait "${PID_TQ}"
RC_TQ=$?
wait "${PID_FP8}"
RC_FP8=$?

echo "[$(ts)] tq44_v3 rc=${RC_TQ}, fp8_g32_v3 rc=${RC_FP8}" | tee -a "${OUTROOT}/driver.log"

python3 - <<PYEOF | tee -a "${OUTROOT}/driver.log"
import json
print()
print("=" * 110)
print("Qwen2.5-72B-Instruct  LCB-128K  2-way   (TQ44 V3  vs  fp8_g32 V3)")
print("=" * 110)
def show(label, p):
    try:
        d = json.load(open(p))
        print(f"  {label:<22} corr={d['n_correct']:>3} wrong={d['n_wrong']:>3} "
              f"null={d['n_null']:>3} oom={d.get('n_oom',0):>3} "
              f"strict={d['accuracy_strict']*100:>5.2f}% "
              f"answered={d['accuracy_answered']*100:>5.2f}% "
              f"dur={d.get('duration_s',0):.0f}s")
    except FileNotFoundError:
        print(f"  {label:<22}    (missing: {p})")
root = "${OUTROOT}"
show("TQ44 V3              ", f"{root}/qwen72b_tq44_v3.json")
show("fp8_g32 V3           ", f"{root}/qwen72b_fp8_g32_v3.json")
print("=" * 110)
PYEOF

echo "[$(ts)] Qwen72B LCB-128K 2-way PARALLEL DONE" | tee -a "${OUTROOT}/driver.log"
