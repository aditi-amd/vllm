#!/bin/bash
# =============================================================================
# Qwen3.8-2.4T-A95B-MXFP4 DECODE-ISOLATION profiling (replicates the Qwen3.6
# decode-iso run: C=16, 1 round, 8K context, 512-token decode).
# Short 8K prefill + long 512-token decode => trace is decode-kernel dominated.
# UQ vs KV8, TP=8, strictly sequential (both legs need all 8 GPUs).
# =============================================================================
set -u

REPO=/shareddata/adrana/workspace/vllm-pr-fp8hd256
LAUNCHER="$REPO/benchmarks/multi_turn_tq/run_multiturn_profile.sh"
PORT=${PORT:-8041}

export MODEL_PATH=/shareddata/amd/Qwen3.8-2.4T-A95B-MXFP4-Quark
export SERVED_NAME=qwen38
export TP=8
export GPUS=0,1,2,3,4,5,6,7
export MAX_MODEL_LEN=32768
# NOTE: GPUs 6/7 carry ~70 GiB of orphaned VRAM leaked by a dead pre-session job
# (PIDs gone, KFD never reclaimed). Worst-GPU free ~217 GiB, so 0.90 (259 GiB)
# fails the startup memory check. 0.72 (~207 GiB) fits with margin; decode-iso
# needs little KV so this does not affect the decode-kernel measurement.
export GPU_MEM_UTIL=0.72

# decode-isolation workload (EXACT Qwen3.6 decode-iso config)
export CLIENTS=16
export ROUNDS=1
export COMMON_PREFIX_TOK=4000
export PREFIX_TOK=4000            # + common => 8K context
export SUBQ_TOK=0
export OUTPUT_TOK=512             # long decode => decode-dominated trace

reclaim() {
  echo ">>> reclaiming GPUs ..."
  pkill -9 -f "vllm serve"             2>/dev/null || true
  pkill -9 -f "VLLM::EngineCore"       2>/dev/null || true
  pkill -9 -f "resource_tracker.*main" 2>/dev/null || true
  # wait until our processes are gone AND VRAM has settled (min free across all
  # GPUs stops rising), so a killed leg's workers finish releasing memory before
  # the next leg snapshots free VRAM.
  for i in $(seq 1 60); do
    if ! pgrep -f "vllm serve" >/dev/null && ! pgrep -f "VLLM::EngineCore" >/dev/null; then
      minfree=$(rocm-smi --showmeminfo vram 2>/dev/null | awk '/GPU\[[0-7]\].*Used/{u=$NF/1073741824; f=287.98-u; if(m==""||f<m)m=f} END{printf "%.0f",m}')
      echo "    procs clear; min free across GPUs = ${minfree} GiB"
      [ "${minfree:-0}" -ge 200 ] && break
    fi
    sleep 3
  done
  sleep 8
}

run_pass() {
  local leg=$1
  echo "==================================================================="
  echo ">>> DECODE PASS: leg=$leg  $(date)"
  echo "==================================================================="
  reclaim
  PROFILE_START_ROUND=0 bash "$LAUNCHER" "$leg" 0 "$PORT"
  echo ">>> DECODE PASS DONE: leg=$leg exit=$? $(date)"
}

run_pass uq
run_pass kv8
reclaim
echo "SENTINEL_QWEN38_DECODE_DONE $(date)"
