#!/usr/bin/env bash
# Apples-to-apples GSM8K benchmark: fp4_kv_g32 (FP4 E2M1 + FP16 scales)
# vs TurboQuant 4-bit NC v1 (Lloyd-Max codebook, Triton v1 decode).
#
# Both legs use the same model, same 4-GPU TP, same GSM8K questions, same
# seed — only the KV cache dtype changes.
#
# Legs:
#   tqv1       -- kv-cache-dtype=turboquant_4bit_nc, Triton v1 decoder
#   fp4_kv_g32 -- kv-cache-dtype=fp4_kv_g32, FP4 E2M1 + FP16 scale Triton decoder
#
# Usage:
#   bash run_gsm8k_minimax_fp4_vs_tqv1.sh
#   NUM_QUESTIONS=1319 bash run_gsm8k_minimax_fp4_vs_tqv1.sh   # full GSM8K
#   GPUS=0,1,2,3 bash run_gsm8k_minimax_fp4_vs_tqv1.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-/shareddata/adrana/workspace/vllm-pr}"
MODEL="${MODEL:-/shareddata/larryli2/MiniMax-M2.5}"
PORT="${PORT:-9877}"
TP="${TP:-4}"
GPUS="${GPUS:-4,5,6,7}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2560}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
NUM_QUESTIONS="${NUM_QUESTIONS:-200}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-1200}"

LOG_DIR="${SCRIPT_DIR}/results"
TS=$(date +%Y%m%d_%H%M%S)
SUMMARY_LOG="${LOG_DIR}/fp4_vs_tqv1_minimax_${TS}.log"
mkdir -p "${LOG_DIR}"

exec > >(tee -a "${SUMMARY_LOG}") 2>&1
ts()  { date -Iseconds; }
log() { echo "[$(ts)] $*"; }

log "=== fp4_kv_g32 vs TQ v1 — MiniMax-M2.5 / N=${NUM_QUESTIONS} / GPUs=${GPUS} TP=${TP} ==="

# --------------------------------------------------------------------------- #
# run_leg <mode> <tag>
#   mode: tqv1 | fp4_kv_g32
# --------------------------------------------------------------------------- #
run_leg() {
    local mode="$1" tag="$2"
    log "----- Leg: ${tag} (MODE=${mode}) -----"

    local kv_flag="" server_log eval_log result_json
    local -a hip_envs=()

    case "${mode}" in
        tqv1)
            kv_flag="--kv-cache-dtype turboquant_4bit_nc"
            hip_envs=(
                "VLLM_TQ_DECODE_V3=0"
                "VLLM_TQ_DECODE_V2=0"
                "VLLM_TQ_FP16_CENTROIDS=0"
                "VLLM_TQ_LUT_RESIDENT=0"
                "VLLM_TQ_SOA_FUSION=0"
            )
            ;;
        fp4_kv_g32)
            kv_flag="--kv-cache-dtype fp4_kv_g32"
            hip_envs=()
            ;;
        *)
            echo "Unknown MODE=${mode}" >&2; exit 1 ;;
    esac

    server_log="${LOG_DIR}/server_${tag}_${TS}.log"
    eval_log="${LOG_DIR}/eval_${tag}_${TS}.log"
    result_json="${LOG_DIR}/gsm8k_${tag}_${TS}.json"

    log "Server log: ${server_log}"

    # ---- launch server ----
    env \
        HIP_VISIBLE_DEVICES="${GPUS}" \
        HSA_NO_SCRATCH_RECLAIM=1 \
        "${hip_envs[@]}" \
        vllm serve "${MODEL}" \
            --port "${PORT}" \
            --tensor-parallel-size "${TP}" \
            --gpu-memory-utilization "${GPU_MEM_UTIL}" \
            --max-model-len "${MAX_MODEL_LEN}" \
            --max-num-seqs "${MAX_NUM_SEQS}" \
            ${kv_flag} \
            --block-size 32 \
            --no-enable-prefix-caching \
            --no-enable-log-requests \
            --trust-remote-code \
        > "${server_log}" 2>&1 &

    local server_pid=$!
    log "Server PID=${server_pid}"

    log "Waiting for /v1/models on port ${PORT} (timeout ${SERVER_TIMEOUT}s)..."
    local deadline=$(( $(date +%s) + SERVER_TIMEOUT ))
    while (( $(date +%s) < deadline )); do
        if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
            log "Server ready"
            break
        fi
        if ! kill -0 "${server_pid}" 2>/dev/null; then
            log "Server died during startup. Last 60 lines:"
            tail -60 "${server_log}"
            return 1
        fi
        sleep 10
    done

    if ! curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
        log "Server did not come up. Last 80 lines:"
        tail -80 "${server_log}"
        kill "${server_pid}" 2>/dev/null || true
        return 1
    fi

    # ---- eval ----
    log "Running GSM8K eval (${NUM_QUESTIONS} questions)..."
    cd "${REPO}"
    python3 tests/evals/gsm8k/gsm8k_eval.py \
        --host http://127.0.0.1 \
        --port "${PORT}" \
        --num-questions "${NUM_QUESTIONS}" \
        --num-shots 5 \
        --max-tokens 256 \
        --temperature 0.0 \
        --seed 42 \
        --save-results "${result_json}" \
        > "${eval_log}" 2>&1 \
        && log "  ${tag}: eval DONE" \
        || log "  ${tag}: eval FAILED (continuing)"

    # ---- tear down ----
    log "Stopping server PID=${server_pid}"
    kill "${server_pid}" 2>/dev/null || true
    sleep 5
    kill -9 "${server_pid}" 2>/dev/null || true
    pkill -9 -f "vllm serve.*--port ${PORT}" 2>/dev/null || true

    log "Settling 30s for GPU memory reclaim..."
    sleep 30
}

run_leg tqv1       "minimax_tqv1_${TS}"
run_leg fp4_kv_g32 "minimax_fp4_kv_g32_${TS}"

# --------------------------------------------------------------------------- #
# Summary table
# --------------------------------------------------------------------------- #
log "----- Summary -----"
LOG_DIR="${LOG_DIR}" TS="${TS}" python3 - <<'PYEOF'
import glob, json, os

ld = os.environ["LOG_DIR"]
ts = os.environ["TS"]

def load_leg(tag_prefix):
    matches = sorted(glob.glob(f"{ld}/gsm8k_{tag_prefix}_{ts}_*.json"))
    if not matches:
        return None
    return json.load(open(matches[-1]))

rows = [
    ("TQ v1  (turboquant_4bit_nc / Triton v1)", load_leg(f"minimax_tqv1")),
    ("fp4_kv_g32 (FP4 E2M1 + FP16 scales)   ", load_leg(f"minimax_fp4_kv_g32")),
]

print()
print("=" * 100)
print(f"  Benchmark: fp4_kv_g32 vs TurboQuant v1 — MiniMax-M2.5")
print("-" * 100)
print(f"  {'Leg':<44} {'N':>5} {'Acc':>9} {'Invalid':>9} {'TPS':>9} {'QPS':>7}")
print("-" * 100)
for label, d in rows:
    if d is None:
        print(f"  {label:<44}  -- no result file --")
        continue
    acc = d.get("accuracy", 0)
    inv = d.get("invalid_rate", 0)
    tps = d.get("tokens_per_second", 0)
    qps = d.get("questions_per_second", 0)
    n   = d.get("num_questions", 0)
    print(f"  {label:<44} {n:>5} {acc*100:>7.2f}%  {inv*100:>6.2f}%  {tps:>8.1f} {qps:>6.2f}")
print("=" * 100)

tqv1_d, fp4_d = rows[0][1], rows[1][1]
if tqv1_d and fp4_d:
    dacc = (fp4_d.get("accuracy", 0) - tqv1_d.get("accuracy", 0)) * 100
    dtps = fp4_d.get("tokens_per_second", 0) - tqv1_d.get("tokens_per_second", 0)
    dtps_pct = 100 * dtps / max(tqv1_d.get("tokens_per_second", 1), 1e-9)
    print(f"  ΔAcc (fp4_kv_g32 − TQ v1):   {dacc:+.2f} pp")
    print(f"  ΔTPS (fp4_kv_g32 − TQ v1):   {dtps:+.1f} tok/s  ({dtps_pct:+.1f}%)")
PYEOF

log "=== 2-way benchmark complete ==="
log "Summary log: ${SUMMARY_LOG}"
