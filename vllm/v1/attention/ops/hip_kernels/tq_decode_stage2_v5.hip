/**
 * TQ Decode Stage2 V5 — Unified: V3 (128-thread) + V4 (32-thread vectorized)
 * 
 * Exports both kernels. Python dispatch selects based on batch size:
 *   B < 80:  V3 variant (128 threads, online softmax, warp broadcast)
 *   B >= 80: V4 variant (32 threads, float4 loads, higher throughput)
 *
 * Both share the same C launch API — selection happens inside the launcher.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

#define HEAD_DIM 128
#define WARP_SIZE 64
#define DIMS_PER_THREAD 4
#define BATCH_CROSSOVER 80

__device__ __forceinline__ hip_bfloat16 f32_to_bf16(float f) {
    return hip_bfloat16(f);
}

// ============================================================
// V3 path: 128 threads, 1 dim/thread, online softmax + shfl
// Best for B < 80 (launch-bound, needs more blocks per CU)
// ============================================================

extern "C" __global__ __launch_bounds__(128, 8)
void tq_decode_stage2_v3_bf16(
    const float* __restrict__ mid_o,
    hip_bfloat16* __restrict__ output,
    const int* __restrict__ seq_lens,
    int stride_mb, int stride_mh, int stride_ms,
    int stride_ob, int stride_oh,
    int num_kv_splits
) {
    const int bid = blockIdx.x;
    const int hid = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;

    const int seq_len = seq_lens[bid];
    if (seq_len <= 0) return;

    const int mid_base = bid * stride_mb + hid * stride_mh;

    float running_max = -1e30f;
    float l_sum = 0.0f;
    float acc = 0.0f;

    for (int s = 0; s < num_kv_splits; s++) {
        const int split_len = (seq_len + num_kv_splits - 1) / num_kv_splits;
        if (split_len * s >= seq_len) break;

        const int s_offset = mid_base + s * stride_ms;
        float lse;
        if (lane_id == 0) lse = mid_o[s_offset + HEAD_DIM];
        lse = __shfl(lse, 0, WARP_SIZE);

        float new_max = fmaxf(running_max, lse);
        float rescale = expf(running_max - new_max);
        float weight = expf(lse - new_max);

        acc = acc * rescale + weight * mid_o[s_offset + tid];
        l_sum = l_sum * rescale + weight;
        running_max = new_max;
    }

    const int out_base = bid * stride_ob + hid * stride_oh;
    float safe_l = l_sum > 0.0f ? l_sum : 1.0f;
    output[out_base + tid] = f32_to_bf16(acc / safe_l);
}

extern "C" __global__ __launch_bounds__(128, 8)
void tq_decode_stage2_v3_f32(
    const float* __restrict__ mid_o,
    float* __restrict__ output,
    const int* __restrict__ seq_lens,
    int stride_mb, int stride_mh, int stride_ms,
    int stride_ob, int stride_oh,
    int num_kv_splits
) {
    const int bid = blockIdx.x;
    const int hid = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;

    const int seq_len = seq_lens[bid];
    if (seq_len <= 0) return;

    const int mid_base = bid * stride_mb + hid * stride_mh;
    float running_max = -1e30f;
    float l_sum = 0.0f;
    float acc = 0.0f;

    for (int s = 0; s < num_kv_splits; s++) {
        const int split_len = (seq_len + num_kv_splits - 1) / num_kv_splits;
        if (split_len * s >= seq_len) break;

        const int s_offset = mid_base + s * stride_ms;
        float lse;
        if (lane_id == 0) lse = mid_o[s_offset + HEAD_DIM];
        lse = __shfl(lse, 0, WARP_SIZE);

        float new_max = fmaxf(running_max, lse);
        float rescale = expf(running_max - new_max);
        float weight = expf(lse - new_max);

        acc = acc * rescale + weight * mid_o[s_offset + tid];
        l_sum = l_sum * rescale + weight;
        running_max = new_max;
    }

    const int out_base = bid * stride_ob + hid * stride_oh;
    float safe_l = l_sum > 0.0f ? l_sum : 1.0f;
    output[out_base + tid] = acc / safe_l;
}

// ============================================================
// V4 path: 32 threads, float4 loads, online softmax
// Best for B >= 80 (compute-bound, vectorized loads win)
// ============================================================

extern "C" __global__ __launch_bounds__(32, 16)
void tq_decode_stage2_v4_bf16(
    const float* __restrict__ mid_o,
    hip_bfloat16* __restrict__ output,
    const int* __restrict__ seq_lens,
    int stride_mb, int stride_mh, int stride_ms,
    int stride_ob, int stride_oh,
    int num_kv_splits
) {
    const int bid = blockIdx.x;
    const int hid = blockIdx.y;
    const int tid = threadIdx.x;

    const int seq_len = seq_lens[bid];
    if (seq_len <= 0) return;

    const int mid_base = bid * stride_mb + hid * stride_mh;
    const int dim_start = tid * DIMS_PER_THREAD;

    float running_max = -1e30f;
    float l_sum = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    for (int s = 0; s < num_kv_splits; s++) {
        const int split_len = (seq_len + num_kv_splits - 1) / num_kv_splits;
        if (split_len * s >= seq_len) break;

        const int s_offset = mid_base + s * stride_ms;
        float lse;
        if (tid == 0) lse = mid_o[s_offset + HEAD_DIM];
        lse = __shfl(lse, 0, WARP_SIZE);

        float new_max = fmaxf(running_max, lse);
        float rescale = expf(running_max - new_max);
        float weight = expf(lse - new_max);

        const float4* src = reinterpret_cast<const float4*>(&mid_o[s_offset + dim_start]);
        float4 vals = *src;

        acc0 = acc0 * rescale + weight * vals.x;
        acc1 = acc1 * rescale + weight * vals.y;
        acc2 = acc2 * rescale + weight * vals.z;
        acc3 = acc3 * rescale + weight * vals.w;
        l_sum = l_sum * rescale + weight;
        running_max = new_max;
    }

    const int out_base = bid * stride_ob + hid * stride_oh;
    float safe_l = l_sum > 0.0f ? l_sum : 1.0f;
    float inv_l = 1.0f / safe_l;

    output[out_base + dim_start + 0] = f32_to_bf16(acc0 * inv_l);
    output[out_base + dim_start + 1] = f32_to_bf16(acc1 * inv_l);
    output[out_base + dim_start + 2] = f32_to_bf16(acc2 * inv_l);
    output[out_base + dim_start + 3] = f32_to_bf16(acc3 * inv_l);
}

extern "C" __global__ __launch_bounds__(32, 16)
void tq_decode_stage2_v4_f32(
    const float* __restrict__ mid_o,
    float* __restrict__ output,
    const int* __restrict__ seq_lens,
    int stride_mb, int stride_mh, int stride_ms,
    int stride_ob, int stride_oh,
    int num_kv_splits
) {
    const int bid = blockIdx.x;
    const int hid = blockIdx.y;
    const int tid = threadIdx.x;

    const int seq_len = seq_lens[bid];
    if (seq_len <= 0) return;

    const int mid_base = bid * stride_mb + hid * stride_mh;
    const int dim_start = tid * DIMS_PER_THREAD;

    float running_max = -1e30f;
    float l_sum = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    for (int s = 0; s < num_kv_splits; s++) {
        const int split_len = (seq_len + num_kv_splits - 1) / num_kv_splits;
        if (split_len * s >= seq_len) break;

        const int s_offset = mid_base + s * stride_ms;
        float lse;
        if (tid == 0) lse = mid_o[s_offset + HEAD_DIM];
        lse = __shfl(lse, 0, WARP_SIZE);

        float new_max = fmaxf(running_max, lse);
        float rescale = expf(running_max - new_max);
        float weight = expf(lse - new_max);

        const float4* src = reinterpret_cast<const float4*>(&mid_o[s_offset + dim_start]);
        float4 vals = *src;

        acc0 = acc0 * rescale + weight * vals.x;
        acc1 = acc1 * rescale + weight * vals.y;
        acc2 = acc2 * rescale + weight * vals.z;
        acc3 = acc3 * rescale + weight * vals.w;
        l_sum = l_sum * rescale + weight;
        running_max = new_max;
    }

    const int out_base = bid * stride_ob + hid * stride_oh;
    float safe_l = l_sum > 0.0f ? l_sum : 1.0f;
    float inv_l = 1.0f / safe_l;

    float4* dst = reinterpret_cast<float4*>(&output[out_base + dim_start]);
    *dst = make_float4(acc0 * inv_l, acc1 * inv_l, acc2 * inv_l, acc3 * inv_l);
}

// ============================================================
// Unified launchers — auto-select V3 or V4 based on batch size
// Same ABI as V2 — drop-in replacement
// ============================================================

extern "C" void launch_tq_decode_stage2_bf16(
    const float* mid_o, hip_bfloat16* output, const int* seq_lens,
    int stride_mb, int stride_mh, int stride_ms,
    int stride_ob, int stride_oh,
    int num_kv_splits, int B, int Hq, hipStream_t stream
) {
    dim3 grid(B, Hq);
    if (B >= BATCH_CROSSOVER) {
        dim3 block(32, 1, 1);
        hipLaunchKernelGGL(tq_decode_stage2_v4_bf16, grid, block, 0, stream,
            mid_o, output, seq_lens, stride_mb, stride_mh, stride_ms,
            stride_ob, stride_oh, num_kv_splits);
    } else {
        dim3 block(128, 1, 1);
        hipLaunchKernelGGL(tq_decode_stage2_v3_bf16, grid, block, 0, stream,
            mid_o, output, seq_lens, stride_mb, stride_mh, stride_ms,
            stride_ob, stride_oh, num_kv_splits);
    }
}

extern "C" void launch_tq_decode_stage2_f32(
    const float* mid_o, float* output, const int* seq_lens,
    int stride_mb, int stride_mh, int stride_ms,
    int stride_ob, int stride_oh,
    int num_kv_splits, int B, int Hq, hipStream_t stream
) {
    dim3 grid(B, Hq);
    if (B >= BATCH_CROSSOVER) {
        dim3 block(32, 1, 1);
        hipLaunchKernelGGL(tq_decode_stage2_v4_f32, grid, block, 0, stream,
            mid_o, output, seq_lens, stride_mb, stride_mh, stride_ms,
            stride_ob, stride_oh, num_kv_splits);
    } else {
        dim3 block(128, 1, 1);
        hipLaunchKernelGGL(tq_decode_stage2_v3_f32, grid, block, 0, stream,
            mid_o, output, seq_lens, stride_mb, stride_mh, stride_ms,
            stride_ob, stride_oh, num_kv_splits);
    }
}
