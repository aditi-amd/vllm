#!/bin/bash
# =============================================================================
# Qwen3.8-MXFP4 DECODE-ISOLATION CONTEXT SWEEP (the before/after target curve).
# Fixed context in {8k,16k,32k}, 512-token decode, C=16, UQ vs KV8, TP=8.
# One server load per (leg,context) so each gets its own clean torch trace.
# Goal: per-call fp8_g32_decode (UQ) vs unified_attention (KV8) as a function
# of context -> shows whether UQ decode cost scales and by how much.
# =============================================================================
set -u

REPO=/shareddata/adrana/workspace/vllm-pr-fp8hd256
LAUNCHER="$REPO/benchmarks/multi_turn_tq/run_multiturn_profile.sh"
MANIFEST="$REPO/benchmarks/multi_turn_tq/results/decode_sweep_manifest.tsv"
PORT=${PORT:-8042}
TRACES_DIR="$REPO/benchmarks/multi_turn_tq/traces"

export MODEL_PATH=/shareddata/amd/Qwen3.8-2.4T-A95B-MXFP4-Quark
export SERVED_NAME=qwen38
export TP=8
export GPUS=0,1,2,3,4,5,6,7
export MAX_MODEL_LEN=40960          # >= 32k ctx + 512 decode + headroom
export GPU_MEM_UTIL=0.90            # GPUs are clean now

# decode-isolation workload knobs (context set per-point below)
export CLIENTS=16
export ROUNDS=1
export SUBQ_TOK=0
export OUTPUT_TOK=512               # long decode => decode-dominated trace
export PROFILE_START_ROUND=0        # capture the single round (prefill+512 decode)

CONTEXTS=(8000 16000 32000)
LEGS=(uq kv8)

mkdir -p "$(dirname "$MANIFEST")"
echo -e "leg\tcontext\ttrace_dir\tstamp" > "$MANIFEST"

reclaim() {
  echo ">>> reclaiming GPUs ..."
  pkill -9 -f "vllm serve"             2>/dev/null || true
  pkill -9 -f "VLLM::EngineCore"       2>/dev/null || true
  pkill -9 -f "resource_tracker.*main" 2>/dev/null || true
  for i in $(seq 1 60); do
    if ! pgrep -f "vllm serve" >/dev/null && ! pgrep -f "VLLM::EngineCore" >/dev/null; then
      minfree=$(rocm-smi --showmeminfo vram 2>/dev/null | awk '/GPU\[[0-7]\].*Used/{u=$NF/1073741824; f=287.98-u; if(m==""||f<m)m=f} END{printf "%.0f",m}')
      echo "    procs clear; min free across GPUs = ${minfree} GiB"
      [ "${minfree:-0}" -ge 250 ] && break
    fi
    sleep 3
  done
  sleep 8
}

for leg in "${LEGS[@]}"; do
  for ctx in "${CONTEXTS[@]}"; do
    echo "==================================================================="
    echo ">>> SWEEP: leg=$leg context=$ctx  $(date)"
    echo "==================================================================="
    reclaim
    export COMMON_PREFIX_TOK=1000
    export PREFIX_TOK=$((ctx-1000))          # common+prefix = ctx tokens
    before=$(ls -1d "$TRACES_DIR/${leg}_"* 2>/dev/null | sort | tail -1)
    PROFILE_START_ROUND=0 bash "$LAUNCHER" "$leg" 0 "$PORT"
    rc=$?
    after=$(ls -1d "$TRACES_DIR/${leg}_"* 2>/dev/null | sort | tail -1)
    stamp=$(basename "$after" | sed "s/${leg}_//")
    echo -e "${leg}\t${ctx}\t${after}\t${stamp}" >> "$MANIFEST"
    echo ">>> SWEEP DONE: leg=$leg ctx=$ctx exit=$rc trace=$after $(date)"
  done
done
reclaim
echo "SENTINEL_QWEN38_DECODE_SWEEP_DONE $(date)"
echo "manifest: $MANIFEST"; cat "$MANIFEST"
