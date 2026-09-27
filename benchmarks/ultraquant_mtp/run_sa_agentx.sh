#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# One SemiAnalysis AgentX-MVP point (aiperf, 3600 s) against one vLLM server,
# with --max-num-seqs equal to the client concurrency.
#
# Usage: run_sa_agentx.sh MODEL KV_CACHE_DTYPE NUM_SPEC_TOKENS CONCURRENCY OUT_DIR [vllm serve args...]
#   NUM_SPEC_TOKENS=0 serves without MTP. Pick the TP=2 GPU pair with
#   HIP_VISIBLE_DEVICES. Example (UQ+MTP, Qwen3.6-27B, C=32):
#     HIP_VISIBLE_DEVICES=0,1 ./run_sa_agentx.sh /models/Qwen3.6-27B \
#         ultraquant_4bit 3 32 out/sa_uq_mtp_c32
set -euo pipefail

# Tool locations; set for your machine.
SA_DIR=/shareddata/adrana/workspace/SemiAnalysis
INFERENCEX_DIR=$SA_DIR/InferenceX
AIPERF=$SA_DIR/aiperf-venv/bin/aiperf
AITER_SRC=/shareddata/adrana/workspace/aiter-v0.1.21.post2
VLLM_SRC=$(cd "$(dirname "$0")/../.." && pwd)

MODEL=$1 KV_CACHE_DTYPE=$2 NUM_SPEC_TOKENS=$3 CONC=$4 OUT=$5
shift 5
PORT=8040
MAX_MODEL_LEN=262144
DURATION=3600

export PYTHONNOUSERSITE=1 VLLM_ROCM_USE_AITER=1 HSA_NO_SCRATCH_RECLAIM=1
export PYTHONPATH="$AITER_SRC:$VLLM_SRC"
export HF_HUB_CACHE=${HF_HUB_CACHE:-$SA_DIR/.hf-cache}
export AIPERF_UI_REALTIME_METRICS_ENABLED=true
export AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800
export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800
export AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=0
OUT=$(mkdir -p "$OUT" && cd "$OUT" && pwd)

SPEC_ARGS=()
if [ "$NUM_SPEC_TOKENS" != "0" ]; then
  SPEC_ARGS=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$NUM_SPEC_TOKENS}")
fi

vllm serve "$MODEL" --port "$PORT" --tensor-parallel-size 2 \
  --attention-backend ROCM_AITER_UNIFIED_ATTN --kv-cache-dtype "$KV_CACHE_DTYPE" \
  --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization 0.90 \
  --enable-prefix-caching --enable-prompt-tokens-details --no-enable-log-requests \
  --max-num-seqs "$CONC" --max-num-batched-tokens 8192 \
  --language-model-only --reasoning-parser qwen3 --trust-remote-code \
  "${SPEC_ARGS[@]}" "$@" > "$OUT/server.log" 2>&1 &
SERVER_PID=$!

descendants() { local c; for c in $(pgrep -P "$1"); do echo "$c"; descendants "$c"; done; }
stop_server() {
  local tree
  tree="$SERVER_PID $(descendants "$SERVER_PID")"
  # shellcheck disable=SC2086
  kill $tree 2>/dev/null || true
  sleep 15
  # shellcheck disable=SC2086
  kill -9 $tree 2>/dev/null || true
}
trap stop_server EXIT

until curl -s "http://localhost:$PORT/v1/models" | grep -q "$MODEL"; do
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "server exited; see $OUT/server.log"; exit 1; }
  sleep 5
done

(cd "$INFERENCEX_DIR" && "$AIPERF" profile --scenario inferencex-agentx-mvp \
  --url "http://localhost:$PORT" --endpoint /v1/chat/completions --endpoint-type chat --streaming \
  --model "$MODEL" --concurrency "$CONC" --benchmark-duration "$DURATION" \
  --stats-interval 30 --random-seed 42 --failed-request-threshold 0.10 \
  --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 \
  --warmup-requests-per-lane 10 --trace-idle-gap-cap-seconds 300 \
  --warmup-grace-period 1800 \
  --use-server-token-count --no-gpu-telemetry --tokenizer-trust-remote-code \
  --max-context-length "$MAX_MODEL_LEN" \
  --num-dataset-entries 393 --slice-duration 1.0 \
  --output-artifact-dir "$OUT/aiperf_artifacts" \
  --public-dataset semianalysis_cc_traces_weka_062126_256k) > "$OUT/benchmark.log" 2>&1

curl -s "http://localhost:$PORT/metrics" \
  | grep -E '^vllm:(spec_decode|prefix_cache|num_preemptions)' > "$OUT/metrics_end.txt" || true
echo "results: $OUT/aiperf_artifacts (profile_export_aiperf.json)"
