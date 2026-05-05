# TurboQuant SoA Fusion Path

This package hosts the opt-in full-stack TurboQuant SoA fusion path enabled by
`VLLM_TQ_SOA_FUSION=1`.

## Structure (self-contained)

All Python/Triton SoA kernels and HIP decode implementations are bundled
within this package. No external checkout is required for E2E operation.

### Python/Triton kernels (self-contained)

- `triton_turboquant_store.py`: SoA KV store (FP8 + MSE paths)
- `triton_turboquant_decode.py`: SoA decode attention (stage1 + stage2)
- `triton_turboquant_decode_v2.py`: v2+ decode with bf16 dot, pair LUT
- `triton_turboquant_unified_attention.py`: unified prefill + decode kernel

### HIP kernels (ROCm MI355X / gfx950)

- `hip_v3_scalar.hip`: baseline scalar SoA decode
- `hip_v3_mfma_qk.hip`: MFMA-accelerated Q·K scoring
- `hip_v3_flash_tq.hip`: fused FlashTQ-style decode
- `soa_bf16q_pv_mfma_decode.hip`: bf16 quantized P·V with MFMA (**production**)
- `soa_bf16q_pv_mfma_decode_gqa6.hip`: bf16Q + PV-MFMA tuned for GQA=6

### Glue layer

- `backend_impl.py`: `FusionTurboQuantAttentionImpl` — extends the base
  `TurboQuantAttentionImpl` with SoA store/decode/continuation logic.
- `external_ops.py`: dispatcher that loads internal modules by default;
  supports `VLLM_TQ_SOA_FUSION_SOURCE_ROOT` env override for development.

## Env vars

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_TQ_SOA_FUSION` | `0` | Enable the SoA fusion path |
| `VLLM_TQ_SOA_FUSION_SOURCE_ROOT` | *(none)* | Override: load SoA kernels from external path |
| `VLLM_TQ_SOA_FUSION_DECODE_SCALAR` | `0` | Enable HIP scalar decode |
| `VLLM_TQ_SOA_FUSION_DECODE_MFMA_QK` | `0` | Enable HIP MFMA Q·K decode |
| `VLLM_TQ_SOA_FUSION_DECODE_FLASH_TQ` | `0` | Enable HIP FlashTQ decode |
| `VLLM_TQ_SOA_FUSION_DECODE_BF16Q_PV_MFMA` | `0` | Enable HIP bf16Q/PV-MFMA (**production**) |
| `TQ_DISABLE_HIP_SO` | `0` | Disable all HIP .so loading |

The default vLLM attention backend is intentionally unchanged — this path is
fully opt-in.

## Reproducing paper benchmarks

The TurboQuant paper measured the SoA-fusion path on MI355X (gfx950) with the
**bf16-Q + PV-MFMA** decode variant. To reproduce:

```bash
# 1) Build the HIP .so files once (per machine)
bash vllm/v1/attention/ops/turboquant_soa_fusion/build_hip_kernels.sh

# 2) Top-level enable + production decode kernel selection
export VLLM_TQ_SOA_FUSION=1
export VLLM_TQ_SOA_FUSION_DECODE_BF16Q_PV_MFMA=1

# 3) Launch vLLM with TurboQuant KV cache
vllm serve <model> \
    --kv-cache-dtype turboquant_4bit_nc \
    --attention-backend ROCM_AITER_UNIFIED_ATTN \
    --block-size 32 --gpu-memory-utilization 0.88 \
    --no-enable-prefix-caching \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
```

Both env vars are required — `VLLM_TQ_SOA_FUSION=1` alone enables the
dispatch but does not select a HIP decode variant (the path falls back to
Triton SoA decode in that case). The decode-variant flags are mutually
exclusive in practice; do not set more than one.

### Validated configuration (paper table 5)

| Setting | Value |
|---|---|
| Hardware | AMD MI355X (gfx950) |
| Model | Qwen2.5-72B-Instruct |
| Runtime | vLLM V1, ROCm 7.x |
| KV cache | `turboquant_4bit_nc` |
| `VLLM_TQ_SOA_FUSION` | `1` |
| `VLLM_TQ_SOA_FUSION_DECODE_BF16Q_PV_MFMA` | `1` |
| `VLLM_TQ_SOA_FUSION_SOURCE_ROOT` | *unset* (uses bundled kernels) |
