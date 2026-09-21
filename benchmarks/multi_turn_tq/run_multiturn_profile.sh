#!/usr/bin/env bash
# Multi-turn torch-profiling for Qwen3.6-27B: UQ fp8g32 FlyDSL (optimized) vs KV8 (vLLM default fp8).
#
# Measures the *prefill dequant tax*: in agentic multi-turn, every warm-cache
# re-prefill must dequantize the cached KV before attention. UQ currently does a
# separate bulk dequant (_tq_full_dequant_kv); KV8 fuses dequant in-kernel. This
# captures a torch trace of the warm rounds so the two can be compared.
#
# Usage:
#   ./run_multiturn_profile.sh <uq|kv8> [gpu_id] [port]
#
# Example (two legs in parallel on GPUs 4 and 5):
#   ./run_multiturn_profile.sh uq  4 6790
#   ./run_multiturn_profile.sh kv8 5 6791
#
# Every value below is asserted from the repo (fp8g32_v4_opt.json / fp8.json,
# prior Qwen3.6 runs = TP1, models/Qwen3.6-27B), not guessed.
set -euo pipefail

LEG="${1:?leg required: uq|kv8}"
GPU="${2:-4}"
PORT="${3:-6790}"

# ---- asserted constants ---------------------------------------------------
REPO="/shareddata/adrana/workspace/vllm-pr-fp8hd256"
MODEL_PATH="${MODEL_PATH:-/shareddata/adrana/workspace/models/Qwen3.6-27B}"
SERVED_NAME="${SERVED_NAME:-qwen36}"
TP="${TP:-1}"              # prior Qwen3.6 runs used TP=1; Qwen3.8-MXFP4 needs TP=8
GPUS="${GPUS:-$GPU}"       # GPU visibility set (single id, or comma list for TP>1)
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"   # override via env; model supports up to 262144
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"   # 0.85 Qwen3.6; override 0.90 for Qwen3.8-MXFP4 TP=8
BLOCK_SIZE=32              # fp8g32_v4_opt.json; forced on both legs for apples-to-apples
ATTN_BACKEND="ROCM_AITER_UNIFIED_ATTN"   # same backend on both legs (apples-to-apples)

# ---- multi-turn / profiling knobs (bounded to keep the trace small) -------
CLIENTS="${CLIENTS:-4}"            # enough to force mixed decode+prefill; small trace
ROUNDS="${ROUNDS:-6}"             # first ROUNDS-2 are warmup (cache fill), last 2 profiled
COMMON_PREFIX_TOK="${COMMON_PREFIX_TOK:-1000}"
PREFIX_TOK="${PREFIX_TOK:-2000}"
SUBQ_TOK="${SUBQ_TOK:-2000}"      # large sub-questions => real context growth (agentic-like)
OUTPUT_TOK="${OUTPUT_TOK:-64}"    # short output bounds decode bloat in the trace (AMD-notebook trick)

STAMP="$(date +%Y%m%d_%H%M%S)"
TRACE_DIR="${REPO}/benchmarks/multi_turn_tq/traces/${LEG}_${STAMP}"
OUTDIR="${REPO}/benchmarks/multi_turn_tq/results/multiturn_profile"
LOGDIR="${REPO}/benchmarks/multi_turn_tq/logs"
mkdir -p "$TRACE_DIR" "$OUTDIR" "$LOGDIR"
SERVER_LOG="${LOGDIR}/server_${LEG}_${STAMP}.log"

# ---- per-leg env + flags --------------------------------------------------
COMMON_ENV=(
  "HIP_VISIBLE_DEVICES=${GPUS}"
  "PYTHONPATH=${REPO}"
  "VLLM_ROCM_USE_AITER=1"
  "HSA_NO_SCRATCH_RECLAIM=1"
  "VLLM_RPC_TIMEOUT=1800000"        # belt-and-suspenders so /stop_profile flush never times out
)

if [[ "$LEG" == "uq" ]]; then
  # UltraQuant fp8g32 FlyDSL optimized == SemiAnalysis 'uq_opt' arm (fp8g32_v4_opt.json)
  LEG_ENV=(
    "VLLM_FP8_G32_V3=1"
    "VLLM_FP8_G32_DECODE_V4=1"
    "VLLM_FP8_G32_DECODE_V4_QK_SCALED=1"
    "VLLM_FLYDSL_ROOT=/root/FlyDSL"
    "VLLM_FLYDSL_PKGS=/root/FlyDSL/build-fly/python_packages"
  )
  KV_DTYPE="fp8_kv_g32"
elif [[ "$LEG" == "kv8" ]]; then
  # vLLM default FP8 KV (fp8.json): dequant is fused in-attention, no bulk pass
  LEG_ENV=()
  KV_DTYPE="fp8"
else
  echo "unknown leg '$LEG' (want uq|kv8)" >&2; exit 2
fi

PROFILER_CONFIG=$(printf '{"profiler":"torch","torch_profiler_dir":"%s","torch_profiler_with_stack":false}' "$TRACE_DIR")

echo "=== launching $LEG server on GPU $GPU port $PORT ==="
echo "    kv-cache-dtype=$KV_DTYPE  trace_dir=$TRACE_DIR"

# NOTE: prefix caching is forced ON here even though fp8g32_v4_opt.json disables
# it — the dequant-tax measurement REQUIRES warm-cache re-prefills.
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
for i in $(seq 1 600); do
  if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "    ready after ${i}s"; break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "SERVER DIED during startup; tail of log:"; tail -40 "$SERVER_LOG"; exit 1
  fi
  sleep 1
done

echo "=== running multi-turn load with profiling (last 2 of $ROUNDS rounds) ==="
env "${COMMON_ENV[@]}" python "${REPO}/benchmarks/multi_turn_tq/bench_multiturn_enhanced.py" \
  --host localhost --port "$PORT" --model "$SERVED_NAME" \
  --num-clients "$CLIENTS" --num-rounds "$ROUNDS" --max-parallel "$CLIENTS" \
  --common-prefix-tokens "$COMMON_PREFIX_TOK" --prefix-tokens "$PREFIX_TOK" \
  --sub-question-tokens "$SUBQ_TOK" --output-tokens "$OUTPUT_TOK" \
  --profile --profile-start-round "${PROFILE_START_ROUND:-$((ROUNDS-2))}" \
  --tag "$LEG" --output-dir "$OUTDIR"

echo "=== done. traces in: $TRACE_DIR ==="
ls -lh "$TRACE_DIR" || true
echo "parse with: python ${REPO}/benchmarks/multi_turn_tq/parse_trace.py '$TRACE_DIR'"
