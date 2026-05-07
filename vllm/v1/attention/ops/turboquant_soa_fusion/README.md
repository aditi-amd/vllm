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

### HIP kernels (ROCm MI300X/MI325X)

- `hip_v3_scalar.hip`: baseline scalar SoA decode
- `hip_v3_mfma_qk.hip`: MFMA-accelerated Q·K scoring
- `hip_v3_flash_tq.hip`: fused FlashTQ-style decode
- `soa_bf16q_pv_mfma_decode.hip`: bf16 quantized P·V with MFMA

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
| `VLLM_TQ_SOA_FUSION_DECODE_BF16Q_PV_MFMA` | `0` | Enable HIP bf16Q/PV-MFMA |
| `TQ_DISABLE_HIP_SO` | `0` | Disable all HIP .so loading |

The default production backend is intentionally unchanged.
