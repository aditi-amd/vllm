#!/usr/bin/env python3
"""LongCodeBench (LongCodeQA) evaluator that uses the official harness parser
but talks to a local vLLM OpenAI-compatible server.

Why we don't run `long-code-bench/eval.py` directly:
  * eval.py loads the model in-process via OpenSourceVLLMModel — that requires
    re-instantiating LLM(...) per variant and re-applying env vars, neither of
    which is convenient when we want to A/B compare TQ44 vs FP4-g32 quickly.
  * We want correct/wrong/null/oom counts broken out separately so we can tell
    real wrong answers apart from out-of-context failures.

We import LongCodeQAEvaluator only for its parser regex so accounting matches
the harness *exactly*. Everything else (driver loop, error handling, summary)
is local.

Usage
-----
  python3 lcb_local_eval.py \
      --port 9300 --model /shareddata/.../Qwen3-32B \
      --label qwen3_32b_tq44 \
      --output results_lcb/qwen3_32b_tq44.json \
      --concurrency 8

Output JSON shape:
  {
    "label": "...",
    "model": "...",
    "n_total": 113,
    "n_correct": 87,
    "n_wrong": 12,
    "n_null": 11,           # response present but parser found no A/B/C/D
    "n_oom": 3,             # 400-class context_length_exceeded errors
    "n_other_error": 0,     # other transport/HTTP errors (counted as null)
    "accuracy_strict": 0.770,         # correct / total (treats null and oom as wrong)
    "accuracy_answered": 0.879,       # correct / (correct + wrong)
    "duration_s": 1234.5,
    "predictions": [{...}, ...]
  }
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

# Eager import of transformers.AutoTokenizer when available, to avoid the
# lazy-module init race triggered when multiple worker threads do
# `from transformers import AutoTokenizer` concurrently. The race happens
# because transformers.__init__ uses _LazyModule; the first thread starts
# attribute resolution while a second thread reads the partially-populated
# module dict and raises ImportError. Importing on the main thread (where
# only one importer runs) makes it concrete before the pool starts.
try:
    from transformers import AutoTokenizer as _AutoTokenizer
except ImportError:
    _AutoTokenizer = None


# Inlined verbatim from long-code-bench/src/long_code_bench/inference/codeqa_eval.py
# (LongCodeQAEvaluator._parse_final_answer). We can't import the harness module
# directly because its package __init__ pulls in models/gemini.py, which requires
# the optional `google-genai` SDK that isn't installed in our vLLM env.
def _parse_final_answer(response: str):
    match = re.search(r"Final Answer:\s*([ABCD])", response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r"\b([ABCD])\s*$", response.strip(), re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None

DEFAULT_DATASET = (
    "/shareddata/adrana/workspace/long-code-bench/data/LQA/32K.json"
)


def _is_context_overflow(exc: Exception, body: str | None) -> bool:
    """vLLM raises HTTP 400 with 'maximum context length' on overflow."""
    if body and (
        "maximum context length" in body
        or "context_length_exceeded" in body
        or "longer than the maximum" in body
    ):
        return True
    return False


_TOKENIZER_CACHE: dict[str, object] = {}


def _get_tokenizer(model_path: str):
    """Lazily load the HF tokenizer matching the served model.
    Mirrors LongCodeBench's `OpenSourceVLLMModel.__init__` which calls
    `AutoTokenizer.from_pretrained(hf_path)`. The import itself happens
    eagerly at module top to avoid a thread race in transformers.__init__.
    """
    if _AutoTokenizer is None:
        raise RuntimeError(
            "transformers.AutoTokenizer is unavailable; "
            "install transformers>=4.0 to use --prompt-truncate-tokens"
        )
    if model_path not in _TOKENIZER_CACHE:
        _TOKENIZER_CACHE[model_path] = _AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
    return _TOKENIZER_CACHE[model_path]


def _truncate_prompt(prompt: str, model_path: str, max_tokens: int) -> str:
    """LCB-style right-truncation: tokenize, clip to max_tokens, decode back.

    Matches the harness's HF default
    `tokenizer(..., truncation=True, max_length=N)` — keep head, drop tail.
    Used only for the no-YaRN comparison where prompts must fit the model's
    native context window.
    """
    tok = _get_tokenizer(model_path)
    ids = tok.encode(prompt, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return prompt
    ids = ids[:max_tokens]
    return tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def call_chat(port: int, model: str, prompt: str,
              max_tokens: int = 16, timeout: int = 600,
              system_prompt: str | None = None,
              no_think: bool = False) -> tuple[str, dict, str]:
    """Returns (response_text, usage_dict, status_tag).
    status_tag is one of: 'ok', 'oom', 'error'.

    If `no_think` is True, sets `chat_template_kwargs.enable_thinking=false`
    (Qwen3 honors this; ignored by templates that don't read the flag).
    """
    url = f"http://localhost:{port}/v1/chat/completions"
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    if no_think:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    try:
        r = requests.post(url, json=payload, timeout=timeout)
    except requests.RequestException as e:
        return "", {}, f"error:{type(e).__name__}"
    if r.status_code == 200:
        d = r.json()
        return (
            d["choices"][0]["message"]["content"] or "",
            d.get("usage", {}),
            "ok",
        )
    body = r.text
    if r.status_code == 400 and _is_context_overflow(None, body):  # type: ignore[arg-type]
        return "", {}, "oom"
    return "", {}, f"error:http{r.status_code}"


def run_eval(args) -> dict:
    data = json.loads(Path(args.data_path).read_text())
    n = len(data)
    print(f"[lcb] dataset: {args.data_path}  n={n}", flush=True)
    print(f"[lcb] target:  http://localhost:{args.port}  model={args.model}",
          flush=True)
    print(f"[lcb] concurrency={args.concurrency}  max_tokens={args.max_tokens}",
          flush=True)

    sys_prompt = (
        "You are answering a multiple-choice question about a code repository. "
        "Think through it briefly if you need to, but you MUST end your "
        "response with a line of EXACTLY this form: "
        "'Final Answer: X' where X is one of A, B, C, D. "
        "Do not write anything after that line."
    )

    # Pre-warm the tokenizer cache on the main thread when truncation is requested,
    # so the worker pool never races on AutoTokenizer.from_pretrained internals.
    if args.prompt_truncate_tokens is not None:
        _get_tokenizer(args.model)

    results: list[dict] = [None] * n  # type: ignore[list-item]
    t0 = time.time()
    counts = {"correct": 0, "wrong": 0, "null": 0, "oom": 0, "other_error": 0}
    done = 0

    def _one(i: int):
        item = data[i]
        prompt = item["prompt"]
        if args.prompt_truncate_tokens is not None:
            prompt = _truncate_prompt(prompt, args.model, args.prompt_truncate_tokens)
        text, usage, status = call_chat(
            args.port, args.model, prompt,
            max_tokens=args.max_tokens, timeout=args.timeout,
            system_prompt=sys_prompt,
            no_think=args.no_think,
        )
        gold = item["correct_letter"].upper()
        parsed = _parse_final_answer(text) if status == "ok" else None
        if status == "oom":
            kind = "oom"
        elif status.startswith("error"):
            kind = "other_error"
        elif parsed is None:
            kind = "null"
        elif parsed == gold:
            kind = "correct"
        else:
            kind = "wrong"
        return i, {
            "id": str(i),
            "gold": gold,
            "parsed": parsed,
            "kind": kind,
            "status": status,
            "response": text,
            "usage": usage,
        }

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = [pool.submit(_one, i) for i in range(n)]
        for f in as_completed(futs):
            i, rec = f.result()
            results[i] = rec
            counts[rec["kind"]] += 1
            done += 1
            if done % max(1, n // 20) == 0 or done == n:
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1e-3)
                eta = (n - done) / max(rate, 1e-3)
                print(
                    f"[lcb] {done}/{n}  "
                    f"correct={counts['correct']} wrong={counts['wrong']} "
                    f"null={counts['null']} oom={counts['oom']} "
                    f"err={counts['other_error']}  "
                    f"elapsed={elapsed:5.0f}s  ETA={eta:5.0f}s",
                    flush=True,
                )

    duration = time.time() - t0
    n_total = n
    n_correct = counts["correct"]
    n_wrong = counts["wrong"]
    n_null = counts["null"]
    n_oom = counts["oom"]
    n_err = counts["other_error"]
    answered = n_correct + n_wrong

    out = {
        "label": args.label,
        "model": args.model,
        "data_path": args.data_path,
        "n_total": n_total,
        "n_correct": n_correct,
        "n_wrong": n_wrong,
        "n_null": n_null,
        "n_oom": n_oom,
        "n_other_error": n_err,
        "accuracy_strict": n_correct / n_total if n_total else 0.0,
        "accuracy_answered": n_correct / answered if answered else 0.0,
        "duration_s": duration,
        "predictions": results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    print()
    print(f"[lcb] === {args.label} ===")
    print(f"[lcb]   n_total           = {n_total}")
    print(f"[lcb]   correct           = {n_correct}")
    print(f"[lcb]   wrong             = {n_wrong}")
    print(f"[lcb]   null (no parse)   = {n_null}")
    print(f"[lcb]   oom (ctx overflow)= {n_oom}")
    print(f"[lcb]   other errors      = {n_err}")
    print(f"[lcb]   accuracy_strict   = {out['accuracy_strict']*100:5.2f}%   (correct / total)")
    print(f"[lcb]   accuracy_answered = {out['accuracy_answered']*100:5.2f}%   (correct / (correct+wrong))")
    print(f"[lcb]   duration          = {duration:.1f}s")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--model", required=True,
                   help="model id served by vLLM (used as the OpenAI 'model' field)")
    p.add_argument("--label", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--data-path", default=DEFAULT_DATASET)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=4096,
                   help="cap on generated tokens (reasoning models need 1k+); "
                        "harness's own runner uses no cap and lets the model run to EOS")
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--no-think", action="store_true",
                   help="set chat_template_kwargs.enable_thinking=false (Qwen3 only). "
                        "MiniMax-M2.5's chat template hardcodes <think>; flag is a no-op there.")
    p.add_argument("--prompt-truncate-tokens", type=int, default=None,
                   help="If set, tokenize each user prompt with the model tokenizer "
                        "and truncate from the right to N tokens before sending. "
                        "Mirrors LCB harness's HF tokenizer(truncation=True, max_length=N) "
                        "behavior. Use this to feed LCB-128K data to a no-YaRN model "
                        "with native 32K context.")
    args = p.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
