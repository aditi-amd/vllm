# How to Run Benchmarks

This document explains how to run the benchmark coverage for this vLLM branch.
It assumes you are already in a clone of this repository.

Two benchmark paths are covered:

- vLLM `benchmarks/multi_turn`: benchmark included in this repository. Use it
  to measure generic multi-turn serving behavior with synthetic or
  ShareGPT-style conversations: throughput, TTFT, TPOT, latency, token counts,
  and turn counts.
- `kv-cache-bench`: internal AMD LongCodeBench serving harness. It requires AMD
  Git Enterprise access. Use it to measure long-context, multi-turn behavior:
  accuracy, TTFT/TPOT, prefix-cache reuse, GPU/CPU KV-cache hit rates,
  SLO-gated ramping, and steady-state concurrency.

Do not compare benchmark outputs unless the model, prompt source, max model
length, GPU count, tensor parallel size, concurrency, and KV-cache settings all
match.

## External References

- vLLM upstream `benchmarks/multi_turn`:
  `https://github.com/vllm-project/vllm/tree/main/benchmarks/multi_turn`
- AMD internal benchmark `kv-cache-bench`:
  `https://gitenterprise.xilinx.com/AMDNeuralOpt/kv-cache-bench.git`
- LongCodeBench code: `https://github.com/Zteefano/long-code-bench`
- LongCodeBench dataset: `https://huggingface.co/datasets/Steefano/LCB`

## Runtime Prerequisite

You need a server runtime environment that can import this vLLM branch with ROCm
and PyTorch. The benchmark client virtual environment created below is not
sufficient to launch the vLLM server.

## Prepare Benchmark 1 Client Environment

From the root of this repository:

```bash
export VLLM_REPO="$(git rev-parse --show-toplevel)"
```

Install the vLLM multi-turn client requirements into `.bench-venv`:

```bash
python3 -m venv "$VLLM_REPO/.bench-venv" && . "$VLLM_REPO/.bench-venv/bin/activate" && python3 -m pip install --upgrade pip && python3 -m pip install -r "$VLLM_REPO/benchmarks/multi_turn/requirements.txt"
```

Verify the benchmark client environment:

```bash
. "$VLLM_REPO/.bench-venv/bin/activate"
python3 -c "import aiohttp, numpy, pandas, transformers, tqdm, xlsxwriter; print('benchmark client env ok')"
```

This client environment may print warnings such as `PyTorch was not found` when
loading tokenizer-related packages. That is expected for client-only usage. It is
not the environment used to launch the vLLM server.

## Prepare vLLM Server Runtime

Use the Python environment or container where this branch of vLLM is built with
ROCm/PyTorch support. Do not rely on `.bench-venv` for server launch.

In the server runtime environment:

```bash
deactivate 2>/dev/null || true
cd "$VLLM_REPO"
export PYTHONPATH="$VLLM_REPO${PYTHONPATH:+:$PYTHONPATH}"
python3 -c "import torch, vllm; print('torch', torch.__version__); print('vllm', vllm.__version__)"
PYTHONPATH="$VLLM_REPO${PYTHONPATH:+:$PYTHONPATH}" vllm --help >/dev/null
```

If either command fails, build or install this vLLM branch in the server runtime
before starting either benchmark. If the Python import works but the `vllm` CLI
imports a different installation, keep `PYTHONPATH="$VLLM_REPO"` on every server
launch command below or install this branch into the server environment.

## Preflight Checks

Set the smoke-test values. Replace only the site-specific paths and GPU IDs:

```bash
export MODEL_PATH=<path-to-MiniMax-M2.5-or-accessible-HF-id>
export SERVED_MODEL_NAME=MiniMax-M2.5
export PORT_MULTI_TURN=19200
export TP_SIZE=2
export HIP_VISIBLE_DEVICES=<comma-separated-gpu-ids>
export MAX_MODEL_LEN=131072
```

Variable meanings:

- `MODEL_PATH` is the model path or Hugging Face id passed to vLLM and the
  benchmark client.
- `SERVED_MODEL_NAME` must match the name exposed by the running vLLM server and
  used by benchmark clients.
- `PORT_MULTI_TURN` is the vLLM server port for Benchmark 1.
- `TP_SIZE` is vLLM tensor parallel size. It should match the intended GPU count
  for the model.
- `HIP_VISIBLE_DEVICES` selects which GPUs vLLM can see.

Check GPUs and memory:

```bash
rocm-smi
rocm-smi --showmeminfo vram --showpids
echo "Using HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}"
HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" python3 - <<'PY'
import torch
for i in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(i)
    print(f"visible_gpu={i} free_gib={free/1024**3:.1f} total_gib={total/1024**3:.1f}")
PY
```

Choose GPUs with enough free memory for the selected model. If startup fails due
to insufficient free memory, use less busy GPUs, reduce GPU memory utilization,
or choose a smaller smoke-test model/config.

Rough memory check: vLLM reserves about
`total_vram_gib * gpu_memory_utilization` per visible GPU. For MiniMax-M2.5,
use GMU `0.90`: MI300X 192 GiB reserves ~173 GiB/GPU (~346 GiB total at TP=2);
MI355X 288 GiB reserves ~259 GiB/GPU (~518 GiB total at TP=2). If
`rocm-smi --showpids` shows other processes on the selected GPUs, choose
different GPU IDs or stop those jobs before benchmarking.

Check the port:

```bash
python3 - <<'PY'
import os, socket
port = int(os.environ["PORT_MULTI_TURN"])
sock = socket.socket()
try:
    sock.bind(("127.0.0.1", port))
finally:
    sock.close()
print(f"PORT_MULTI_TURN={port} is free")
PY
```

Check the model path:

```bash
test -e "$MODEL_PATH" || echo "MODEL_PATH is not a local path; ensure Hugging Face access works"
```

## Benchmark 1: vLLM `benchmarks/multi_turn`

This benchmark is included at `benchmarks/multi_turn`. Its default synthetic
configuration, `generate_multi_turn.json`, references `pg1184.txt`.

Prepare the text file only if it is missing:

```bash
cd "$VLLM_REPO/benchmarks/multi_turn"
if [ ! -f pg1184.txt ]; then
  wget https://www.gutenberg.org/ebooks/1184.txt.utf-8 -O pg1184.txt
fi
```

This download requires network access to Project Gutenberg. If network access is
not available, provide another local text file and update the `text_files` field
in `generate_multi_turn.json` to point to it.

Start vLLM for MiniMax-M2.5 from the server runtime environment:

```bash
deactivate 2>/dev/null || true
cd "$VLLM_REPO"
HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" \
PYTHONPATH="$VLLM_REPO${PYTHONPATH:+:$PYTHONPATH}" vllm serve "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --port "$PORT_MULTI_TURN" \
  --tensor-parallel-size "$TP_SIZE" \
  --trust-remote-code \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization 0.90 \
  --kv-cache-dtype fp8_kv_g32
```

For other models, use the model-specific flags required by that model and KV
scheme. A plain `vllm serve "$MODEL_PATH"` command is not sufficient when you
intend to benchmark a specific KV-cache dtype or kernel path.

Wait for `Application startup complete`, then verify:

```bash
curl -fsS "http://127.0.0.1:${PORT_MULTI_TURN}/v1/models"
```

Run the benchmark from the client environment:

```bash
. "$VLLM_REPO/.bench-venv/bin/activate"
cd "$VLLM_REPO/benchmarks/multi_turn"
python3 benchmark_serving_multi_turn.py \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --url "http://127.0.0.1:${PORT_MULTI_TURN}" \
  --input-file generate_multi_turn.json \
  --num-clients 1 \
  --max-active-conversations 2 \
  --stats-json-output multi_turn_stats.json
```

Stop the vLLM server before starting the next benchmark. If it is running in the
foreground, press `Ctrl-C`. If it is running in a terminal manager or background
process, stop that process and confirm no workers remain:

```bash
pkill -TERM -f "vllm.*--port ${PORT_MULTI_TURN}" || true
sleep 20
pgrep -af "vllm.*--port ${PORT_MULTI_TURN}" || true
rocm-smi --showpids
```

Validate Benchmark 1:

```bash
cd "$VLLM_REPO/benchmarks/multi_turn"
python3 - <<'PY'
import json
path = "multi_turn_stats.json"
data = json.load(open(path))
assert data, "stats file is empty"
completed = [r for r in data if (r.get("output_num_tokens") or 0) > 0]
print("requests:", len(data), "completed_with_output:", len(completed))
assert completed, "no requests with output tokens"
for key in ("error", "exception", "failed"):
    offenders = [r for r in data if r.get(key)]
    assert not offenders, f"{key} present in stats"
PY
```

For ShareGPT-style data, convert the source dataset first:

```bash
cd "$VLLM_REPO/benchmarks/multi_turn"
python3 convert_sharegpt_to_openai.py \
  <path-to-sharegpt-json> \
  sharegpt_conv_128.json \
  --seed 99 --max-items 128
```

Then pass `--input-file sharegpt_conv_128.json`.

## Benchmark 2: `kv-cache-bench`

`kv-cache-bench` is internal AMD infrastructure. Confirm Git Enterprise access:

```bash
git ls-remote https://gitenterprise.xilinx.com/AMDNeuralOpt/kv-cache-bench.git HEAD
```

Expected success output includes `<commit-sha> HEAD`. If this fails with
`could not read Username`, `Authentication failed`, or a DNS/network error,
connect to AMD network/VPN, sign in with SSO, configure an approved HTTPS
credential/PAT or SSH key, and rerun until it succeeds.

Prepare `kv-cache-bench` and its client requirements:

```bash
export KV_CACHE_BENCH_REPO="$(dirname "$VLLM_REPO")/kv-cache-bench"
([ -d "$KV_CACHE_BENCH_REPO/.git" ] || git clone https://gitenterprise.xilinx.com/AMDNeuralOpt/kv-cache-bench.git "$KV_CACHE_BENCH_REPO") && . "$VLLM_REPO/.bench-venv/bin/activate" && python3 -m pip install -r "$KV_CACHE_BENCH_REPO/requirements.txt"
git -C "$KV_CACHE_BENCH_REPO" remote -v
git -C "$KV_CACHE_BENCH_REPO" branch --show-current
git -C "$KV_CACHE_BENCH_REPO" rev-parse HEAD
```

Set Benchmark 2 values:

```bash
export MODEL_KEY=MiniMax-M2.5
export SCHEME_KEY=ultraquant
export PORT_KV_BENCH=19201
export LONG_CODE_BENCH_REPO="$(dirname "$VLLM_REPO")/long-code-bench"
export LCB_FILE="$LONG_CODE_BENCH_REPO/data/LQA/128K.json"
export BENCH_SECONDS=600
export START_USERS=1
export MAX_USERS=1
```

Check model and scheme keys:

```bash
python3 - <<'PY'
import os, yaml
repo = os.environ["KV_CACHE_BENCH_REPO"]
model_key = os.environ["MODEL_KEY"]
scheme_key = os.environ["SCHEME_KEY"]
models = yaml.safe_load(open(f"{repo}/configs/models.yaml"))["models"]
schemes = yaml.safe_load(open(f"{repo}/configs/schemes.yaml"))["schemes"]
assert model_key in models, f"missing model key: {model_key}"
assert scheme_key in schemes, f"missing scheme key: {scheme_key}"
print("model config:", models[model_key])
print("scheme config:", schemes[scheme_key])
PY
```

Check the Benchmark 2 port:

```bash
python3 - <<'PY'
import os, socket
port = int(os.environ["PORT_KV_BENCH"])
sock = socket.socket()
try:
    sock.bind(("127.0.0.1", port))
finally:
    sock.close()
print(f"PORT_KV_BENCH={port} is free")
PY
```

Prepare LongCodeBench data:

```bash
([ -d "$LONG_CODE_BENCH_REPO/.git" ] || git clone https://github.com/Zteefano/long-code-bench.git "$LONG_CODE_BENCH_REPO") && . "$VLLM_REPO/.bench-venv/bin/activate" && python3 -m pip install -U huggingface_hub && hf download Steefano/LCB --repo-type dataset --local-dir "$LONG_CODE_BENCH_REPO/data" && python3 - <<'PY'
from pathlib import Path
from zipfile import ZipFile
data_dir = Path(__import__("os").environ["LONG_CODE_BENCH_REPO"]) / "data"
with ZipFile(data_dir / "LongCodeQA.zip") as zf:
    zf.extractall(data_dir)
PY
test -f "$LCB_FILE"
```

Start vLLM through the benchmark harness from the server runtime environment:

```bash
deactivate 2>/dev/null || true
cd "$KV_CACHE_BENCH_REPO"
HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" PYTHONPATH="$VLLM_REPO${PYTHONPATH:+:$PYTHONPATH}" \
  ./scripts/launch_vllm.sh "$MODEL_KEY" "$SCHEME_KEY" \
  --port "$PORT_KV_BENCH" \
  --vllm-pythonpath "$VLLM_REPO"
```

For the smoke-test values above, the selected config is `MiniMax-M2.5` with
scheme `ultraquant`. In `kv-cache-bench/configs/schemes.yaml`, `ultraquant`
passes `--kv-cache-dtype fp8_kv_g32`, which is the fp8 scale-kernel KV-cache
path covered by this branch. In `kv-cache-bench/configs/models.yaml`,
`MiniMax-M2.5` sets the expected 128K context, TP=2, `trust_remote_code`, and
GPU memory utilization for the benchmark harness.

Wait until the server log prints:

```text
Application startup complete.
```

Large models can take several minutes to load weights and capture graphs. After
readiness, verify the server:

```bash
curl -fsS "http://127.0.0.1:${PORT_KV_BENCH}/v1/models"
curl -fsS "http://127.0.0.1:${PORT_KV_BENCH}/metrics" >/dev/null
```

Run the LongCodeBench driver from the benchmark client environment:

```bash
. "$VLLM_REPO/.bench-venv/bin/activate"
cd "$KV_CACHE_BENCH_REPO"
python3 -m benchmarks.multiturn_lcb \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --url "http://127.0.0.1:${PORT_KV_BENCH}" \
  --tag fp8scalekernel --ctx 128K \
  --max-model-len "$MAX_MODEL_LEN" \
  --start-users "$START_USERS" --max-users "$MAX_USERS" \
  --recycle --test-duration "$BENCH_SECONDS" \
  --lcb-file "$LCB_FILE" \
  --metrics-url "http://127.0.0.1:${PORT_KV_BENCH}/metrics" \
  --output-dir results/
```

This is a smoke test. `BENCH_SECONDS=600` gives the 128K LongCodeBench turns
enough time to complete and produce scored accuracy. Reduce it only for quick
local iteration if you are willing to accept an invalid zero-turn summary.
If an `asyncio.run() shutdown` exception prints after `DONE`, validate the
summary below; it is non-fatal when the summary passes.

Stop the server after the run and confirm no vLLM workers remain:

```bash
pkill -TERM -f "vllm.entrypoints.openai.api_server.*--port ${PORT_KV_BENCH}" || true
pkill -TERM -f "launch_vllm.sh.*--port ${PORT_KV_BENCH}" || true
sleep 20
pgrep -af "vllm.*--port ${PORT_KV_BENCH}|launch_vllm.sh.*--port ${PORT_KV_BENCH}" || true
rocm-smi --showpids
```

## Validate Results

Do not treat client exit code `0` as proof that a benchmark succeeded. Validate
the output and the server log.

For `kv-cache-bench`, find the newest summary and check minimum validity:

```bash
python3 - <<'PY'
import glob, json, os
summaries = sorted(glob.glob(f"{os.environ['KV_CACHE_BENCH_REPO']}/results/*.summary.json"))
assert summaries, "no summary files found"
path = summaries[-1]
data = json.load(open(path))
print(path)
print(json.dumps({k: data.get(k) for k in [
    "n_turns", "n_errored", "tokens_per_sec_per_gpu", "accuracy"
]}, indent=2))
period_output = sum((p.get("output_tokens") or 0)
                    for p in data.get("per_assessment_period", []))
turn_output = sum((t.get("completion_tokens_mean") or 0) * (t.get("n") or 0)
                  for t in data.get("per_turn_index", []))
print("estimated_output_tokens:", max(period_output, turn_output))
assert data.get("n_errored", 0) == 0, "errored turns present"
assert (data.get("tokens_per_sec_per_gpu") or 0) > 0, "zero throughput"
assert max(period_output, turn_output) > 0, "zero output tokens"
PY
```

Also check the server log for HTTP 500s, GPU memory faults, and shutdowns during
the run. Graph only validated successful summaries:

```bash
cd "$KV_CACHE_BENCH_REPO"
python3 -m graphing.holistic --input results/*.summary.json --output holistic.png
```

## Common Failures

- Unknown `MODEL_KEY` or `SCHEME_KEY`: inspect
  `kv-cache-bench/configs/models.yaml` and `kv-cache-bench/configs/schemes.yaml`.
- Startup OOM or "free memory is less than desired GPU memory utilization": pick
  less busy GPUs with `HIP_VISIBLE_DEVICES`, reduce `gpu_memory_utilization` in
  the model config, or use a smaller smoke-test model/config.
- Benchmark starts before server readiness: wait for `Application startup
  complete` and confirm `/v1/models` responds.
- `vllm serve` imports the wrong vLLM or fails while Python import succeeds:
  launch with `PYTHONPATH="$VLLM_REPO"` or install this branch into the server
  runtime environment.
- Port already in use: stop the previous vLLM server or use separate
  `PORT_MULTI_TURN` and `PORT_KV_BENCH` values.
- Benchmark unexpectedly uses `kv_cache_dtype=auto`: the server command is
  missing the intended KV-cache dtype. For the MiniMax-M2.5 fp8 scale-kernel
  smoke test, use `SCHEME_KEY=ultraquant` or pass `--kv-cache-dtype fp8_kv_g32`
  in the explicit vLLM server command.
- Graphing fails on `None` cache-hit values: the summary is likely from a failed
  run. Validate the summary before graphing.
- Do not use `--enforce-eager` to work around server launch issues; it changes
  serving behavior and invalidates representative benchmark results.
