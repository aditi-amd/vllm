#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# GPQA-Diamond (198 items, model-card sampling) against one vLLM server.
#
# Usage: run_gpqa_diamond.sh MODEL KV_CACHE_DTYPE NUM_SPEC_TOKENS OUT_DIR [vllm serve args...]
#   NUM_SPEC_TOKENS=0 serves without MTP. Pick the TP=2 GPU pair with
#   HIP_VISIBLE_DEVICES. Example (UQ+MTP, Qwen3.6-27B):
#     HIP_VISIBLE_DEVICES=0,1 ./run_gpqa_diamond.sh /models/Qwen3.6-27B \
#         ultraquant_4bit 3 out/gpqa_uq_mtp --max-num-seqs 32
#
# Requires the eval-harness GPQA driver with card-sampling support
# (drivers/gpqa_native.py) and configs/benchmarks/gpqa_diamond_card.json.
set -euo pipefail

# Tool locations; set for your machine.
EVAL_HARNESS=/shareddata/adrana/workspace/eval-harness
AITER_SRC=/shareddata/adrana/workspace/aiter-v0.1.21.post2
VLLM_SRC=$(cd "$(dirname "$0")/../.." && pwd)

MODEL=$1 KV_CACHE_DTYPE=$2 NUM_SPEC_TOKENS=$3 OUT=$4
shift 4
PORT=8061
SEED=7
MAX_MODEL_LEN=163840

export PYTHONNOUSERSITE=1 VLLM_ROCM_USE_AITER=1 HSA_NO_SCRATCH_RECLAIM=1
export PYTHONPATH="$AITER_SRC:$VLLM_SRC"
mkdir -p "$OUT"

SPEC_ARGS=()
if [ "$NUM_SPEC_TOKENS" != "0" ]; then
  SPEC_ARGS=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$NUM_SPEC_TOKENS}")
fi

vllm serve "$MODEL" --port "$PORT" --tensor-parallel-size 2 \
  --attention-backend ROCM_AITER_UNIFIED_ATTN --kv-cache-dtype "$KV_CACHE_DTYPE" \
  --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization 0.75 \
  --reasoning-parser qwen3 --language-model-only \
  --no-enable-prefix-caching --no-enable-log-requests --trust-remote-code \
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

until curl -s "http://127.0.0.1:$PORT/v1/models" | grep -q "$MODEL"; do
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "server exited; see $OUT/server.log"; exit 1; }
  sleep 5
done

python "$EVAL_HARNESS/drivers/gpqa_native.py" \
  --port "$PORT" --model "$MODEL" --max_model_len "$MAX_MODEL_LEN" \
  --bench_config "$EVAL_HARNESS/configs/benchmarks/gpqa_diamond_card.json" \
  --out_dir "$OUT" --seed "$SEED" > "$OUT/eval.log" 2>&1

curl -s "http://127.0.0.1:$PORT/metrics" | grep -E '^vllm:spec_decode' > "$OUT/spec_metrics.txt" || true
python -c "import json; j = json.load(open('$OUT/summary.json')); \
print(f\"accuracy={j['accuracy']} correct={j['correct']}/{j['total']}\")"
