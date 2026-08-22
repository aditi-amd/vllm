#!/usr/bin/env bash
# Auto-rerun the dev loop whenever the fp8_g32 decode kernel/launcher changes.
# Edit the kernel -> save -> timing + parity rerun automatically (warm-ish).
#
# Usage:  HIP_VISIBLE_DEVICES=0 ./watch_dev.sh [MODE]   (MODE default: time)
set -u
MODE="${1:-time}"
ROOT=/shareddata/adrana/workspace/vllm-pr-fp8hd256
export VLLM_FP8_G32_V3=1 VLLM_FP8_G32_DECODE_V4=1 \
       VLLM_FP8_G32_DECODE_V4_QK_SCALED=1 \
       VLLM_FLYDSL_ROOT=/root/FlyDSL \
       VLLM_FLYDSL_PKGS=/root/FlyDSL/build-fly/python_packages
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"

WATCH=(
  "$ROOT/vllm/v1/attention/ops/flydsl_fp8_g32_decode_v4.py"
  "$ROOT/tests/kernels/turboquant_v4/dev_loop.py"
  /root/FlyDSL/kernels/fp8_g32_decode_v4.py
  /root/FlyDSL/kernels/fp8_g32_decode_hd256.py
)
run() { echo "==== $(date +%T) rerun ($MODE) ===="; \
        python "$ROOT/tests/kernels/turboquant_v4/dev_loop.py" "$MODE"; }

run
if command -v inotifywait >/dev/null 2>&1; then
  while inotifywait -q -e close_write "${WATCH[@]}" 2>/dev/null; do run; done
else
  echo "(inotifywait missing; polling mtime every 2s)"
  declare -A M
  while true; do
    ch=0
    for f in "${WATCH[@]}"; do
      [ -f "$f" ] || continue
      t=$(stat -c %Y "$f" 2>/dev/null)
      [ "${M[$f]:-}" != "$t" ] && { M[$f]=$t; ch=1; }
    done
    [ "$ch" = 1 ] && run
    sleep 2
  done
fi
