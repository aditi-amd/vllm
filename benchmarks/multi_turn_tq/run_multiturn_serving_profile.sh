#!/usr/bin/env bash
# Continuous-traffic (upstream) multi-turn torch-profiling for Qwen3.6-27B:
#   UQ fp8g32 FlyDSL (optimized)  vs  KV8 (vLLM default fp8).
#
# Unlike bench_multiturn_enhanced.py (round-barrier / synchronized), this drives
# the UPSTREAM vllm/benchmarks/multi_turn/benchmark_serving_multi_turn.py, which
# uses independent per-client processes sending back-to-back requests => a
# realistic staggered mix of prefill + decode at steady state (agentic-like).
#
# Profiling is TIME-BRACKETED (not round-bracketed): a background thread waits
# PROFILE_DELAY sec for steady state, POSTs /start_profile, captures
# PROFILE_DURATION sec, then POSTs /stop_profile. This bounds trace size even
# with 1k-token outputs at C=16.
#
# Usage:
#   ./run_multiturn_serving_profile.sh <uq|kv8> [gpu_id] [port]
#
# Two legs in parallel on GPUs 4 and 5:
#   ./run_multiturn_serving_profile.sh uq  4 6790
#   ./run_multiturn_serving_profile.sh kv8 5 6791
set -euo pipefail

LEG="${1:?leg required: uq|kv8}"
GPU="${2:-4}"
PORT="${3:-6790}"

# ---- asserted constants ---------------------------------------------------
REPO="/shareddata/adrana/workspace/vllm-pr-fp8hd256"
MODEL_PATH="/shareddata/adrana/workspace/models/Qwen3.6-27B"
SERVED_NAME="qwen36"
TP=1
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"   # must exceed 8k prefix growth over turns
GPU_MEM_UTIL=0.85
BLOCK_SIZE=32
ATTN_BACKEND="ROCM_AITER_UNIFIED_ATTN"

# ---- load / profiling knobs ----------------------------------------------
CLIENTS="${CLIENTS:-16}"                 # C=16 agentic concurrency
MAX_ACTIVE="${MAX_ACTIVE:-16}"
REQUEST_RATE="${REQUEST_RATE:-0}"        # 0 => clients send back-to-back (max steady load)
MAX_TURNS="${MAX_TURNS:-8}"
INPUT_JSON="${REPO}/benchmarks/multi_turn/agentic_8k1k.json"   # 8k-ish input / ~1k output
PROFILE_DELAY="${PROFILE_DELAY:-25}"     # sec before /start_profile (steady state)
PROFILE_DURATION="${PROFILE_DURATION:-8}" # sec captured window

STAMP="$(date +%Y%m%d_%H%M%S)"
TRACE_DIR="${REPO}/benchmarks/multi_turn_tq/traces/serving_${LEG}_${STAMP}"
OUTDIR="${REPO}/benchmarks/multi_turn_tq/results/multiturn_serving_profile"
LOGDIR="${REPO}/benchmarks/multi_turn_tq/logs"
mkdir -p "$TRACE_DIR" "$OUTDIR" "$LOGDIR"
SERVER_LOG="${LOGDIR}/server_serving_${LEG}_${STAMP}.log"
BENCH_LOG="${LOGDIR}/bench_serving_${LEG}_${STAMP}.log"

# ---- per-leg env + flags --------------------------------------------------
COMMON_ENV=(
  "HIP_VISIBLE_DEVICES=${GPU}"
  "PYTHONPATH=${REPO}"
  "VLLM_ROCM_USE_AITER=1"
  "HSA_NO_SCRATCH_RECLAIM=1"
  "VLLM_RPC_TIMEOUT=1800000"        # so /stop_profile flush never times out
)

if [[ "$LEG" == "uq" ]]; then
  LEG_ENV=(
    "VLLM_FP8_G32_V3=1"
    "VLLM_FP8_G32_DECODE_V4=1"
    "VLLM_FP8_G32_DECODE_V4_QK_SCALED=1"
    "VLLM_FLYDSL_ROOT=/root/FlyDSL"
    "VLLM_FLYDSL_PKGS=/root/FlyDSL/build-fly/python_packages"
  )
  KV_DTYPE="fp8_kv_g32"
elif [[ "$LEG" == "kv8" ]]; then
  LEG_ENV=()
  KV_DTYPE="fp8"
else
  echo "unknown leg '$LEG' (want uq|kv8)" >&2; exit 2
fi

PROFILER_CONFIG=$(printf '{"profiler":"torch","torch_profiler_dir":"%s","torch_profiler_with_stack":false}' "$TRACE_DIR")

echo "=== launching $LEG server on GPU $GPU port $PORT ==="
echo "    kv-cache-dtype=$KV_DTYPE  trace_dir=$TRACE_DIR"

# prefix caching forced ON: agentic multi-turn reuses the common prefix (warm re-prefill).
env "${COMMON_ENV[@]}" "${LEG_ENV[@]}" \
  vllm serve "$MODEL_PATH" \
    --served-model-name "$SERVED_NAME" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --block-size "$BLOCK_SIZE" \
    --kv-cache-dtype "$KV_DTYPE" \
    --attention-backend "$ATTN_BACKEND" \
    --trust-remote-code \
    --enable-prefix-caching \
    --enable-prompt-tokens-details \
    --port "$PORT" \
    --profiler-config "$PROFILER_CONFIG" \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "    server pid=$SERVER_PID  log=$SERVER_LOG"

cleanup() { echo "killing server $SERVER_PID"; kill "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

echo "=== waiting for server readiness (/v1/models) ==="
for i in $(seq 1 900); do
  if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "    ready after ${i}s"; break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "SERVER DIED during startup; tail of log:"; tail -40 "$SERVER_LOG"; exit 1
  fi
  sleep 1
done

echo "=== running continuous multi-turn load (C=$CLIENTS) + timed profile window ==="
env "${COMMON_ENV[@]}" python "${REPO}/benchmarks/multi_turn/benchmark_serving_multi_turn.py" \
  --input-file "$INPUT_JSON" \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_NAME" \
  --url "http://localhost:${PORT}" \
  --num-clients "$CLIENTS" \
  --max-active-conversations "$MAX_ACTIVE" \
  --request-rate "$REQUEST_RATE" \
  --max-turns "$MAX_TURNS" \
  --no-early-stop \
  --warmup-step \
  --profile \
  --profile-delay-sec "$PROFILE_DELAY" \
  --profile-duration-sec "$PROFILE_DURATION" \
  --output-file "${OUTDIR}/conv_${LEG}_${STAMP}.json" \
  --stats-json-output "${OUTDIR}/stats_${LEG}_${STAMP}.json" \
  2>&1 | tee "$BENCH_LOG"

echo "=== done. traces in: $TRACE_DIR ==="
ls -lh "$TRACE_DIR" || true
echo "parse with: python ${REPO}/benchmarks/multi_turn_tq/parse_trace.py '$TRACE_DIR'"
