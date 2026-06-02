"""Parallel 8K/1K C=64 serving perf A/B for Qwen2.5-72B on MI355X.

Two legs, run in parallel on separate GPU pairs (TP=2 each):
  1. TQ44 V3 + SoA STORE  (Triton V3 reading SoA-written K cache; no HIP dep)
  2. fp8_g32 V3            (same as gsm8k accuracy run)

Both share:
  - AITER unified attention backend (bf16 prefill + non-TQ layers)
  - FULL_AND_PIECEWISE cudagraphs
  - block-size=32, gpu_memory_utilization=0.85
  - ISL=8192, OSL=1024, fixed length (range_ratio=0)
  - C=64 concurrency, N=128 prompts (smaller than full sweep for ~10 min wallclock)

Output: per-leg c64.json with OutTPS / TPOT / TTFT / ITL + side-by-side table.
"""
from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# ---------- CONFIG ----------
MODEL = "/shareddata/Qwen/Qwen2.5-72B-Instruct"
REPO  = "/shareddata/adrana/workspace/vllm-pr"
TP    = 2
ISL   = 8192
OSL   = 1024
MML   = ISL + OSL + 256
C     = 64
N     = 128
NWARMUP = 8
GPU_MEM_UTIL = 0.85
BLOCK_SIZE = 32

OUTDIR = Path("/tmp/qwen72b_8k1k_c64_perf")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Both legs use these env vars (matches gsm8k accuracy run, plus SoA store for leg 1)
BASE_ENV = {
    "VLLM_ROCM_USE_AITER": "1",
    "VLLM_ROCM_QUICK_REDUCE_QUANTIZATION": "INT4",
    "VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT": "1",
    "HSA_NO_SCRATCH_RECLAIM": "1",
}

LEG_TQ44_V3_SOA = {
    **BASE_ENV,
    "VLLM_TQ_DECODE_V4": "0",
    "VLLM_TQ_DECODE_V3": "1",
    "VLLM_TQ_DECODE_V2": "0",
    "VLLM_TQ_SOA_FUSION": "0",          # HIP-class OFF (user requested no HIP dep)
    "VLLM_TQ_SOA_FUSION_STORE": "1",    # write SoA layout for Triton V3 decode
    "VLLM_TQ_FP16_CENTROIDS": "0",
    "VLLM_TQ_LUT_RESIDENT": "0",
}

LEG_FP8_G32_V3 = {
    **BASE_ENV,
    "VLLM_FP8_G32_V3": "1",
    "VLLM_FP8_G32_ARCHB": "0",
    # Final config: use the NEW defaults (NUM_KV_SPLITS=16, NUM_STAGES_3D=3)
    # without explicit env overrides — verifies the production-ready
    # behavior end-to-end.
}

# Path 2: fp8×fp8 dot_scaled on PV. Forces TILE_SIZE=32. Expected to gain
# ~2× the V MFMA throughput on CDNA4 (native scaled fp8 MFMA).
LEG_FP8_G32_V_DOT_SCALED = {
    **BASE_ENV,
    "VLLM_FP8_G32_V3": "1",
    "VLLM_FP8_G32_ARCHB": "0",
    "VLLM_FP8_G32_V_DOT_SCALED": "1",
}

CC_JSON = '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'


def _start_server(label: str, gpus: str, port: int, kv_dtype: str,
                  env_extra: dict[str, str]) -> tuple[subprocess.Popen, Path]:
    """Launch vllm serve in background, return (proc, server_log_path)."""
    tag_dir = OUTDIR / label
    tag_dir.mkdir(exist_ok=True)
    server_log = tag_dir / "server.log"

    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = gpus
    env.pop("ROCR_VISIBLE_DEVICES", None)
    env.update(env_extra)

    cmd = [
        "vllm", "serve", MODEL,
        "--port", str(port),
        "--tensor-parallel-size", str(TP),
        "--gpu-memory-utilization", str(GPU_MEM_UTIL),
        "--max-model-len", str(MML),
        "--kv-cache-dtype", kv_dtype,
        "--block-size", str(BLOCK_SIZE),
        "--no-enable-prefix-caching",
        "--attention-backend", "ROCM_AITER_UNIFIED_ATTN",
        "--no-enable-log-requests",
        "--trust-remote-code",
        "--compilation-config", CC_JSON,
    ]
    print(f"[{label}] starting vllm serve on HIPs={gpus}, port={port}, kv={kv_dtype}")
    print(f"[{label}]   env_extra={ {k:v for k,v in env_extra.items() if k.startswith('VLLM_TQ') or k.startswith('VLLM_FP8')} }")
    print(f"[{label}]   log -> {server_log}")
    log_f = open(server_log, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT,
                            cwd=REPO, preexec_fn=os.setsid)
    proc._label = label
    proc._log_f = log_f
    proc._server_log = server_log
    proc._port = port
    return proc, server_log


def _wait_ready(proc: subprocess.Popen, port: int, timeout: int = 1800) -> bool:
    """Poll /v1/models until reachable or proc dies / timeout."""
    label = proc._label
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            print(f"[{label}] FATAL: server exited (code {proc.returncode}) before ready")
            return False
        try:
            with urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=2) as r:
                if r.status == 200:
                    elapsed = int(time.time() - t0)
                    print(f"[{label}] server ready after {elapsed}s")
                    return True
        except Exception:
            pass
        time.sleep(10)
        if int(time.time() - t0) % 60 == 0:
            print(f"[{label}]   ...loading ({int(time.time()-t0)}s)")
    print(f"[{label}] FATAL: server didn't become ready within {timeout}s")
    return False


def _run_bench(label: str, port: int) -> Path:
    """Run vllm bench serve, return path to result JSON."""
    tag_dir = OUTDIR / label
    result_file = f"c{C}.json"
    bench_log = tag_dir / f"c{C}.log"
    cmd = [
        "vllm", "bench", "serve",
        "--backend", "openai",
        "--base-url", f"http://localhost:{port}",
        "--endpoint", "/v1/completions",
        "--model", MODEL,
        "--dataset-name", "random",
        "--random-input-len", str(ISL),
        "--random-output-len", str(OSL),
        "--random-range-ratio", "0",
        "--num-prompts", str(N),
        "--max-concurrency", str(C),
        "--request-rate", "inf",
        "--ignore-eos",
        "--num-warmups", str(NWARMUP),
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--metric-percentiles", "50,99",
        "--seed", "42",
        "--save-result",
        "--result-dir", str(tag_dir),
        "--result-filename", result_file,
        "--trust-remote-code",
    ]
    print(f"[{label}] vllm bench serve C={C} N={N} ISL={ISL} OSL={OSL}")
    print(f"[{label}]   bench log -> {bench_log}")
    with open(bench_log, "w") as bf:
        rc = subprocess.call(cmd, stdout=bf, stderr=subprocess.STDOUT, cwd=REPO)
    print(f"[{label}] bench exit={rc}")
    return tag_dir / result_file


def _kill(proc: subprocess.Popen):
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    for _ in range(15):
        if proc.poll() is not None:
            return
        time.sleep(1)
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        return


def _summarize(results: list[tuple[str, Path]]):
    print("\n" + "=" * 120)
    print(f"  Qwen2.5-72B  TP={TP}  ISL={ISL}  OSL={OSL}  C={C}  N={N}   FULL_AND_PIECEWISE cudagraphs")
    print("-" * 120)
    print(f"  {'Run':<26} {'OutTPS':>9} {'TotTPS':>9} {'TTFT_p50':>10} {'TTFT_p99':>10} "
          f"{'TPOT_p50':>10} {'TPOT_p99':>10} {'ITL_p50':>9} {'OK':>9}")
    print("-" * 120)
    rows = []
    for label, p in results:
        if not p.exists():
            print(f"  {label:<26}   -- no result -- ")
            continue
        d = json.load(open(p))
        out_tps  = d.get('output_throughput', 0)
        tot_tps  = d.get('total_token_throughput', 0)
        ttft_p50 = d.get('median_ttft_ms', 0)
        ttft_p99 = d.get('p99_ttft_ms', 0)
        tpot_p50 = d.get('median_tpot_ms', 0)
        tpot_p99 = d.get('p99_tpot_ms', 0)
        itl_p50  = d.get('median_itl_ms', 0)
        ok       = f"{d.get('completed',0)}/{d.get('num_prompts',0)}"
        rows.append((label, d))
        print(f"  {label:<26} {out_tps:>8.1f} {tot_tps:>8.1f} "
              f"{ttft_p50/1000:>7.2f}s {ttft_p99/1000:>7.2f}s "
              f"{tpot_p50:>8.2f}ms {tpot_p99:>8.2f}ms {itl_p50:>7.2f}ms {ok:>9}")
    print("=" * 120)
    if len(rows) == 2:
        (la, da), (lb, db) = rows
        def pct(a, b):
            return 100 * (a - b) / b if b else float("nan")
        print(f"\n  Δ ({lb} vs {la}):")
        print(f"     OutTPS    : {pct(db['output_throughput'], da['output_throughput']):+6.1f}%  "
              f"(bigger=better)")
        print(f"     TPOT p50  : {pct(db['median_tpot_ms'], da['median_tpot_ms']):+6.1f}%  "
              f"(smaller=better)")
        print(f"     TTFT p50  : {pct(db['median_ttft_ms'], da['median_ttft_ms']):+6.1f}%  "
              f"(smaller=better)")


def main():
    configs = [
        # (label, hips, port, kv_dtype, env_extra)
        ("TQ44_V3_SOA",                  "0,1", 9081, "turboquant_4bit_nc", LEG_TQ44_V3_SOA),
        ("fp8_g32_V3_V_DOT_SCALED",      "2,3", 9082, "fp8_kv_g32",         LEG_FP8_G32_V_DOT_SCALED),
    ]
    procs = []
    print(f"=== Qwen2.5-72B 8K/1K C={C} perf A/B ===")
    print(f"Model: {MODEL}   TP={TP}   N={N}   ISL={ISL}   OSL={OSL}\n")

    # Phase 1: launch both servers in parallel
    for label, gpus, port, kv, env_extra in configs:
        p, _ = _start_server(label, gpus, port, kv, env_extra)
        procs.append((p, port, label))

    # Phase 2: wait for both ready (in parallel)
    ready = []
    for proc, port, label in procs:
        ok = _wait_ready(proc, port)
        ready.append((proc, port, label, ok))
    if not all(r[3] for r in ready):
        print("ERROR: one or more servers failed to start; tailing logs:")
        for proc, _, label, ok in ready:
            if not ok:
                with open(proc._server_log) as f:
                    print(f"\n--- {label} server.log tail ---")
                    print("\n".join(f.readlines()[-60:]))
        for proc, _, _, _ in ready:
            _kill(proc)
        sys.exit(1)

    # Phase 3: run benches in parallel (each hits its own server)
    print("\n=== Both servers ready; launching bench serve in parallel ===\n")
    bench_procs = []
    for proc, port, label, _ in ready:
        tag_dir = OUTDIR / label
        cmd = [
            "vllm", "bench", "serve",
            "--backend", "openai",
            "--base-url", f"http://localhost:{port}",
            "--endpoint", "/v1/completions",
            "--model", MODEL,
            "--dataset-name", "random",
            "--random-input-len", str(ISL),
            "--random-output-len", str(OSL),
            "--random-range-ratio", "0",
            "--num-prompts", str(N),
            "--max-concurrency", str(C),
            "--request-rate", "inf",
            "--ignore-eos",
            "--num-warmups", str(NWARMUP),
            "--percentile-metrics", "ttft,tpot,itl,e2el",
            "--metric-percentiles", "50,99",
            "--seed", "42",
            "--save-result",
            "--result-dir", str(tag_dir),
            "--result-filename", f"c{C}.json",
            "--trust-remote-code",
        ]
        bench_log = tag_dir / f"c{C}.log"
        bf = open(bench_log, "w")
        bp = subprocess.Popen(cmd, stdout=bf, stderr=subprocess.STDOUT, cwd=REPO)
        bp._label = label
        bp._bench_log = bench_log
        bp._tag_dir = tag_dir
        bench_procs.append(bp)
        print(f"[{label}] bench started PID={bp.pid} log={bench_log}")

    t0 = time.time()
    for bp in bench_procs:
        bp.wait()
        print(f"[{bp._label}] bench exit={bp.returncode} after {int(time.time()-t0)}s")

    # Phase 4: tear down servers
    print("\n=== Killing servers ===")
    for proc, _, label, _ in ready:
        print(f"[{label}] killing server PID={proc.pid}")
        _kill(proc)

    # Phase 5: summary
    results = [(bp._label, bp._tag_dir / f"c{C}.json") for bp in bench_procs]
    _summarize(results)


if __name__ == "__main__":
    main()
