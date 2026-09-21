#!/bin/bash
# =============================================================================
# Sequential Qwen3.8-2.4T-A95B-MXFP4-Quark profiling: replicate the Qwen3.6
# 100K/5 -> 8K/5 deterministic 2-turn run (C=4), UQ vs KV8, torch profiler.
#
# Qwen3.8 needs TP=8 (all 8 GPUs, ~159 GiB/GPU MXFP4 weights) so the UQ and
# KV8 legs CANNOT run in parallel -> strictly sequential, 4 server loads:
#   1) uq  start_round=0  (whole 100K+8K trace)
#   2) uq  start_round=1  (round-1 re-prefill isolated)
#   3) kv8 start_round=0
#   4) kv8 start_round=1
#
# Apples-to-apples contract (identical across legs): TP, GPUs, gpu-mem-util,
# max-model-len, block-size, attention backend, workload, seed. Only KV dtype
# + its decode env differ (the variable under test). UQ decode env is the SAME
# as the Qwen3.6 profiling run (V3=1, DECODE_V4=1, QK_SCALED=1).
# =============================================================================
set -u

REPO=/shareddata/adrana/workspace/vllm-pr-fp8hd256
LAUNCHER="$REPO/benchmarks/multi_turn_tq/run_multiturn_profile.sh"
PORT=${PORT:-8040}

# ---- Qwen3.8 model + TP=8 (exported to the launcher via env overrides) ------
export MODEL_PATH=/shareddata/amd/Qwen3.8-2.4T-A95B-MXFP4-Quark
export SERVED_NAME=qwen38
export TP=8
export GPUS=0,1,2,3,4,5,6,7
export MAX_MODEL_LEN=131072          # must exceed 100K + 8K + output + headroom
export GPU_MEM_UTIL=0.90             # headroom for 1.3TB MXFP4 weights at TP=8

# ---- workload: EXACT Qwen3.6 100K/8K deterministic 2-turn config ------------
export CLIENTS=4
export ROUNDS=2
export COMMON_PREFIX_TOK=4000        # + PREFIX_TOK => ~100K round-0 fresh prefill
export PREFIX_TOK=96000
export SUBQ_TOK=8000                 # round-1 re-prefill over cached 100K
export OUTPUT_TOK=5

reclaim() {
  echo ">>> reclaiming GPUs ..."
  pkill -9 -f "vllm serve"            2>/dev/null || true
  pkill -9 -f "VLLM::EngineCore"      2>/dev/null || true
  pkill -9 -f "resource_tracker.*main" 2>/dev/null || true
  for i in $(seq 1 40); do
    pgrep -f "vllm serve" >/dev/null || pgrep -f "VLLM::EngineCore" >/dev/null || break
    sleep 3
  done
  sleep 10
}

run_pass() {
  local leg=$1 sr=$2
  echo "==================================================================="
  echo ">>> PASS: leg=$leg  profile_start_round=$sr  $(date)"
  echo "==================================================================="
  reclaim
  PROFILE_START_ROUND="$sr" bash "$LAUNCHER" "$leg" 0 "$PORT"
  echo ">>> PASS DONE: leg=$leg sr=$sr exit=$? $(date)"
}

run_pass uq  0
run_pass uq  1
run_pass kv8 0
run_pass kv8 1
reclaim
echo "SENTINEL_QWEN38_SEQ_DONE $(date)"
