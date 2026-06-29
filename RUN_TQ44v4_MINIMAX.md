# Run TQ44v4 on MiniMax-M2.5 (FlyDSL v4)

Minimal steps to serve **MiniMax-M2.5** with the **TurboQuant-44 v4** 4-bit KV kernel
on **AMD MI355X (gfx950)**. Branch: **`feat/fp8scalekernel`** (commit `0136dd2ae`).

> Full kernel reference / options: see `HOW_TO_RUN.md`. This is the short path.

## 0. Prereqs
- AMD MI355X (gfx950), ROCm 7.x (`hipcc` on `PATH`), Python 3.12, Linux x86_64.

## 1. Build the vLLM fork
```bash
git clone -b feat/fp8scalekernel https://github.com/aditi-amd/vllm.git
cd vllm
pip install -e . --no-build-isolation     # builds the HIP TQ44 kernels
cd ..
```

## 2. Build FlyDSL (required for the v4 decode kernel)
```bash
git clone https://github.com/ROCm/FlyDSL.git /opt/FlyDSL   # or your internal mirror
cd /opt/FlyDSL
git checkout 41500b0                       # tested SHA
mkdir -p build-fly && cd build-fly
cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON
ninja -j"$(nproc)"                         # ~5 min -> build-fly/python_packages/flydsl
cd -
```

## 3. Set the FlyDSL path
```bash
export VLLM_FLYDSL_ROOT=/opt/FlyDSL
export VLLM_FLYDSL_PKGS=/opt/FlyDSL/build-fly/python_packages
export PYTHONPATH="${VLLM_FLYDSL_ROOT}:${VLLM_FLYDSL_PKGS}${PYTHONPATH:+:$PYTHONPATH}"
python3 -c "import flydsl; print('FlyDSL OK:', flydsl.__version__)"
```

## 4. Common env (MiniMax-M2.5, MI355X — same for baseline and TQ44v4)
```bash
export MODEL=/path/to/MiniMax-M2.5         # your local checkpoint
export HIP_VISIBLE_DEVICES=4,5             # two free gfx950 GPUs (TP=2)
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1
export HSA_NO_SCRATCH_RECLAIM=1            # older MEC firmware (<177)
# Dense CUDA-graph capture — important for decode throughput
export CAP='{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,24,32,40,48,56,64,80,96,128]}'
```

## 5. Serve — BF16 baseline
```bash
vllm serve "$MODEL" --port 8000 \
    --tensor-parallel-size=2 --enable-expert-parallel \
    --gpu-memory-utilization 0.85 --max-model-len 34816 \
    --kv-cache-dtype auto --block-size=32 \
    --attention-backend ROCM_AITER_UNIFIED_ATTN \
    --no-enable-prefix-caching --no-enable-log-requests \
    --trust-remote-code --compilation-config "$CAP"
```

## 6. Serve — TQ44v4 (FlyDSL v4, 4-bit KV)
Identical flags; only `--kv-cache-dtype` + the TQ decode env differ. Requires the FlyDSL
env from step 3 exported in this shell.
```bash
VLLM_TQ_DECODE_V4=1 VLLM_TQ_DECODE_V3=0 VLLM_TQ_DECODE_V2=0 \
VLLM_TQ_FP16_CENTROIDS=0 VLLM_TQ_LUT_RESIDENT=0 \
VLLM_TQ_SOA_FUSION=0 VLLM_TQ_SOA_FUSION_STORE=1 \
VLLM_TQ_SOA_FUSION_DECODE_BF16Q_PV_MFMA=0 VLLM_TQ_SOA_FUSION_WHT_BUTTERFLY=0 \
VLLM_TQ_DECODE_V4_WHT_BUTTERFLY=0 \
vllm serve "$MODEL" --port 8000 \
    --tensor-parallel-size=2 --enable-expert-parallel \
    --gpu-memory-utilization 0.85 --max-model-len 34816 \
    --kv-cache-dtype turboquant_4bit_nc --block-size=32 \
    --attention-backend ROCM_AITER_UNIFIED_ATTN \
    --no-enable-prefix-caching --no-enable-log-requests \
    --trust-remote-code --compilation-config "$CAP"
```
Decode auto-overrides to the **TURBOQUANT (FlyDSL v4)** backend; `--attention-backend`
applies to prefill / non-TQ layers. **Butterfly:** set `VLLM_TQ_DECODE_V4_WHT_BUTTERFLY=1`
for ~**+6.6% TPS / −11 ms ITL**.

## 7. Sanity check
```bash
curl -s http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"'"$MODEL"'","prompt":"Hello, the capital of France is","max_tokens":16}'
```

## 8. Benchmark (32K in / 1K out, C=64, N=80) — identical for both
```bash
vllm bench serve --backend openai \
    --base-url http://127.0.0.1:8000 --endpoint /v1/completions \
    --model "$MODEL" --dataset-name random \
    --random-input-len 32768 --random-output-len 1024 --random-range-ratio 0 \
    --max-concurrency 64 --num-prompts 80 \
    --request-rate inf --ignore-eos --num-warmups 8 \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99 \
    --seed 42 --save-result --result-filename m25_32k1k_c64.json
```

## Notes
- **Apples-to-apples:** same backend (`ROCM_AITER_UNIFIED_ATTN`), EP, dense capture, and
  `--num-warmups 8` for both legs; the only variable is `--kv-cache-dtype`.
- **Reference (2× MI355X, clean node, 32K/1K C=64):** TQ44v4 ≈ **343 tok/s / 118 ms TPOT**
  (blog parity); with butterfly ≈ **366 tok/s / 110 ms**. Skipping dense capture or
  `VLLM_ROCM_USE_AITER=1` drops this materially.
- `max-model-len 34816` = 32768 + 1024 + 1024 headroom. v4 is gfx950-only.
