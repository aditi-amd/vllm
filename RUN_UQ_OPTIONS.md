# UltraQuant (fp8_kv_g32) decode options for Qwen3.8 / hd256

UltraQuant (UQ) is the `fp8_kv_g32` KV-cache dtype: **4-bit K/V (FP4 E2M1) + 8-bit Q
(FP8 E4M3)** with UE8M0 group-32 scales, decoded by the FlyDSL/HIP TurboQuant-family
kernel on gfx950 (MI355X) for `HEAD_SIZE=256`, GQA ∈ {8,16}, no sinks/SWA. Ineligible
layers and continuation-prefill fall back to the fp8_g32 Triton path.

Enable it at serve time with:

```bash
vllm serve <MXFP4-ckpt> --tensor-parallel-size 8 \
  --kv-cache-dtype fp8_kv_g32 --max-model-len 262144 \
  --gpu-memory-utilization 0.90 --enable-prefix-caching --trust-remote-code
```

## Default decode path (as of this branch)

The **optimized "v7" decode path is now the built-in default** — no env vars are
required to get it. The following in-tree defaults were flipped so that a plain
`--kv-cache-dtype fp8_kv_g32` run uses the fused, strided v7 kernel:

| Env var | Old default | New default | Effect when on |
| --- | --- | --- | --- |
| `VLLM_FP8_G32_V3` | 0 | **1** | Triton fp8_g32 path (QK/PV via `tl.dot`) for fallback/prefill |
| `VLLM_FP8_G32_DECODE_V4` | 0 | **1** | FlyDSL fp8_g32 v4 decode kernel (bug-free TQ v4 port) |
| `VLLM_FP8_G32_DECODE_V5_FUSED` | 0 | **1** | Fused single-kernel decode: folds partition-combine into the epilogue (no separate reduce kernel / segm HBM round-trip) |
| `VLLM_FP8_G32_DECODE_V4_FUSE_Q_ROT` | 0 | **1** | Q rotation done in-kernel (Walsh–Hadamard butterfly); drops the separate q_rot dispatch |
| `VLLM_FP8_G32_DECODE_V5_FUSE_QROT` | 0 | **1** | In-kernel Q-rot for the fused v5 path (requires qk_scaled, QG%8==0, hd256) |

Already at the v7 value before this change (unchanged):

| Env var | Default | Note |
| --- | --- | --- |
| `VLLM_FP8_G32_DECODE_V4_QK_SCALED` | 1 | scaled QK operand permutation |
| `VLLM_FP8_G32_DECODE_V5_STRIDED_TG` | 1 | strided tile-group→partition mapping (hides HBM latency at server-sized worst-case grid) |
| `VLLM_FP8_G32_DECODE_V4_MAX_PARTITIONS` | (unset) | unset → batch-adaptive cap (512 @ batch≤8, 256 above) |

Inert in-repo (read by the external FlyDSL runtime, not by `vllm/`; set by the
benchmark arm only for reproducibility): `VLLM_FP8_G32_DECODE_V5_REDUCE_BLOCK_D`,
`VLLM_FP8_G32_DECODE_V5_TOK_PER_PART`, `FLYDSL_RUNTIME_ENABLE_CACHE`.

### Opting out / pinning

Every flag keeps its escape hatch — set it to `0` to disable. For example, to fall
back to the pre-v7 (v4-adaptive, non-fused) decode:

```bash
VLLM_FP8_G32_DECODE_V5_FUSED=0 VLLM_FP8_G32_DECODE_V5_FUSE_QROT=0 \
vllm serve ... --kv-cache-dtype fp8_kv_g32
```

> Numerical note: the strided v7 path is **not bit-exact** vs the v6 (blocked)
> mapping — split-K regroups so partition-softmax partials round differently. Both
> match `reference_fp8_g32_attention` at cos ≈ 0.999995 and differ from each other by
> ~1 bf16 ULP (2.4e-4).

## Historical arm presets (for reproducing prior sweeps)

These are the exact env sets used by the SA agentic sweeps (`run_sa_3arm_card.sh`).
With the new defaults, `uq_v7` == running with no env vars.

| Arm | Env set (in addition to `--kv-cache-dtype fp8_kv_g32`) |
| --- | --- |
| `uq_opt` | `V3=1 DECODE_V4=1 DECODE_V4_QK_SCALED=1 DECODE_V4_MAX_PARTITIONS=256 DECODE_V4_FUSE_Q_ROT=1` |
| `uq_adapt` | `uq_opt` minus `MAX_PARTITIONS` (batch-adaptive cap) |
| `uq_v5` | `uq_adapt` + `DECODE_V5_FUSED=1 DECODE_V5_FUSE_QROT=0 DECODE_V5_REDUCE_BLOCK_D=256` |
| `uq_v6` | `uq_v5` + `DECODE_V5_FUSE_QROT=1 DECODE_V5_REDUCE_BLOCK_D=64 DECODE_V5_TOK_PER_PART=192` |
| `uq_v7` | `uq_v6` + `DECODE_V5_STRIDED_TG=1` — **now the default** |

(Prefix `VLLM_FP8_G32_` / `VLLM_FP8_G32_DECODE_` omitted in the table for brevity;
all arms also set `FLYDSL_RUNTIME_ENABLE_CACHE=0` for measurement reproducibility.)

## V4 decode option reference (`VLLM_FP8_G32_DECODE_V4*`)

The v4 FlyDSL decode kernel is the core of the UQ path (the v5 fused/strided
options above sit on top of it). Full knob list, with in-tree defaults verified
from source. Names below omit the `VLLM_FP8_G32_DECODE_` prefix.

| Env var | Default | Effect |
| --- | --- | --- |
| `V4` | **1** | Route the fp8_g32 decode to the FlyDSL v4 kernel (bug-free TQ v4 port). *Now default.* |
| `V4_QK_SCALED` | 1 | Scaled-QK operand permutation (qperm); prerequisite for the in-kernel Q-rot fusions. |
| `V4_QK_FP8` | 0 | Do the QK dot in FP8 instead of the default precision. |
| `V4_FUSE_Q_ROT` | **1** | Fuse Q rotation (`Q @ PiT`) as an in-kernel Walsh–Hadamard butterfly; removes the separate q_rot dispatch. *Now default.* |
| `V4_MAX_PARTITIONS` | (unset) | Fixed split-K partition cap. When **unset**, the batch-adaptive policy below is used. |
| `V4_LOWB_CAP` | 512 | Adaptive partition cap at batch ≤ `LOWB_THRESHOLD`. |
| `V4_HIB_CAP` | 256 | Adaptive partition cap at batch > `LOWB_THRESHOLD`. |
| `V4_LOWB_THRESHOLD` | 8 | Batch size at/below which `LOWB_CAP` applies (else `HIB_CAP`). |
| `V4_DYNAMIC_PARTS` | 0 | Size partitions dynamically from the actual context instead of the fixed cap. |
| `V4_FUSE_BLOCK_D` | 64 | Head-dim reduce-block width for the fused reduce/combine. |
| `V4_NUM_WARPS` | 1 | Kernel warp/wave count (occupancy tuning). |
| `V4_Q_HOIST` | 1 | Hoist the Q load out of the partition loop. |
| `V4_V_CVT` | 1 | Pre-convert V codes to the compute dtype in the decode epilogue. |
| `V4_B_BUCKET` | (unset) | Manual batch-bucket override for kernel selection; auto when unset. |
| `V4_HW_TR` | (unset) | Hardware-transpose path toggle; auto when unset. |

Notes:
- The `uq_opt` arm pinned `V4_MAX_PARTITIONS=256`; `uq_adapt` and later (incl. the
  new default) leave it **unset** to use the `LOWB/HIB/THRESHOLD` adaptive cap.
- `V4_QK_SCALED=1` is required for the `V4_FUSE_Q_ROT` / v5 `FUSE_QROT` in-kernel
  rotations to engage.
