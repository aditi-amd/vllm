#!/usr/bin/env python3
"""Profile a full Gemma-4 decode step kernel mix under the tq_v4_aiter config.

Loads the model offline (small max-len), warms up, then generates a burst of
decode tokens. Run under rocprofv3 --kernel-trace; this script just drives the
work. Aggregation of kernel categories is done by the caller.
"""
import os
os.environ.setdefault("VLLM_TQ_DECODE_V4", "1")
os.environ.setdefault("VLLM_TQ_DECODE_V3", "0")
os.environ.setdefault("VLLM_TQ_SOA_FUSION_STORE", "1")
os.environ.setdefault("VLLM_ROCM_USE_AITER", "1")
os.environ.setdefault("HSA_NO_SCRATCH_RECLAIM", "1")

from vllm import LLM, SamplingParams

def main():
    llm = LLM(
        model="google/gemma-4-31b-it",
        kv_cache_dtype="turboquant_4bit_nc",
        block_size=32,
        max_model_len=2048,
        gpu_memory_utilization=0.75,
        enforce_eager=True,  # eager so kernel names map cleanly per step
        attention_backend="ROCM_AITER_UNIFIED_ATTN",
    )
    # Batch of prompts to reach ~B=32 decode.
    prompts = ["Write a short story about a robot."] * 32
    sp = SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True)
    # warmup
    llm.generate(prompts, sp)
    # profiled region: longer decode burst
    sp2 = SamplingParams(max_tokens=int(os.environ.get("DECODE_TOK", "24")),
                         temperature=0.0, ignore_eos=True)
    import torch
    torch.cuda.synchronize()
    llm.generate(prompts, sp2)
    torch.cuda.synchronize()

if __name__ == "__main__":
    main()
