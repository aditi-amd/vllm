// SPDX-License-Identifier: Apache-2.0
// Scalar HIP decode for TurboQuant fusion v3 SoA cache layout.
//
// This is the first correctness/ABI bridge for the v3 path. It intentionally
// stays scalar: Q rotation is handled by the Python wrapper, while this kernel
// validates the SoA address contract and store-time norm-correction contract.

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#define HEAD_DIM 128
#define MSE_BYTES 64
#define VAL_DATA_BYTES 64
#define DATA_BYTES_PER_SLOT (MSE_BYTES + VAL_DATA_BYTES)
#define NUM_SOA_FIELDS 3
#define SOA_K_NORM 0
#define SOA_V_SCALE 1
#define SOA_V_ZERO 2

#define WARP_SIZE_TQ 32
#define NUM_WARPS 4
#define THREADS (WARP_SIZE_TQ * NUM_WARPS)
#define DIMS_PER_THREAD 4
#define BLOCK_KV 8
#define N_CENTROIDS 16

#define DTYPE_BF16 0
#define DTYPE_FP16 1
#define DTYPE_FP32 2

__device__ __forceinline__ void warp_reduce_sum(float& x) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        x += __shfl_xor(x, off, 32);
    }
}

__device__ __forceinline__ void store_native_4(
    void* out, int base, float v0, float v1, float v2, float v3, int dtype
) {
    if (dtype == DTYPE_BF16) {
        hip_bfloat16* o = reinterpret_cast<hip_bfloat16*>(out);
        o[base] = hip_bfloat16(v0);
        o[base + 1] = hip_bfloat16(v1);
        o[base + 2] = hip_bfloat16(v2);
        o[base + 3] = hip_bfloat16(v3);
    } else {
        __half* o = reinterpret_cast<__half*>(out);
        o[base] = __float2half(v0);
        o[base + 1] = __float2half(v1);
        o[base + 2] = __float2half(v2);
        o[base + 3] = __float2half(v3);
    }
}

__device__ __forceinline__ void load_q_native(
    const void* q, int idx, int dtype, float& q0, float& q1, float& q2, float& q3
) {
    if (dtype == DTYPE_FP32) {
        const float4 v = *reinterpret_cast<const float4*>(
            reinterpret_cast<const float*>(q) + idx);
        q0 = v.x;
        q1 = v.y;
        q2 = v.z;
        q3 = v.w;
    } else if (dtype == DTYPE_FP16) {
        const __half2* p2 = reinterpret_cast<const __half2*>(
            reinterpret_cast<const __half*>(q) + idx);
        const __half2 h01 = p2[0];
        const __half2 h23 = p2[1];
        q0 = __low2float(h01);
        q1 = __high2float(h01);
        q2 = __low2float(h23);
        q3 = __high2float(h23);
    } else {
        const unsigned int* p32 = reinterpret_cast<const unsigned int*>(
            reinterpret_cast<const hip_bfloat16*>(q) + idx);
        const unsigned int w0 = p32[0];
        const unsigned int w1 = p32[1];
        q0 = __uint_as_float((w0 & 0xFFFF) << 16);
        q1 = __uint_as_float((w0 >> 16) << 16);
        q2 = __uint_as_float((w1 & 0xFFFF) << 16);
        q3 = __uint_as_float((w1 >> 16) << 16);
    }
}

__device__ __forceinline__ long long tq_v3_data_base(
    int block_num,
    int page_off,
    int kv_head,
    int stride_cache_block,
    int num_kv_heads
) {
    return (long long)block_num * stride_cache_block
        + (long long)page_off * (num_kv_heads * DATA_BYTES_PER_SLOT)
        + (long long)kv_head * DATA_BYTES_PER_SLOT;
}

__device__ __forceinline__ int tq_v3_meta_u16_addr(
    int block_num,
    int page_off,
    int kv_head,
    int stride_cache_block,
    int block_size,
    int num_kv_heads,
    int field
) {
    const long long block_base = (long long)block_num * stride_cache_block;
    const long long meta_region_offset =
        (long long)block_size * num_kv_heads * DATA_BYTES_PER_SLOT;
    const long long head_meta_u16_base = (block_base + meta_region_offset) / 2
        + (long long)kv_head * (NUM_SOA_FIELDS * block_size);
    return (int)(head_meta_u16_base + field * block_size + page_off);
}

extern "C" __global__ __launch_bounds__(THREADS, 8)
void tq_v3_scalar_stage1(
    const void* __restrict__ q_rot,
    const unsigned char* __restrict__ kv_cache,
    const int* __restrict__ block_table,
    const int* __restrict__ seq_lens,
    const float* __restrict__ centroids,
    float* __restrict__ mid_o,
    int stride_qb,
    int stride_qh,
    int stride_cache_block,
    int stride_bt,
    int stride_mb,
    int stride_mh,
    int stride_ms,
    int num_kv_heads,
    int block_size,
    int num_kv_splits,
    int kv_group_size,
    float attn_scale,
    int q_dtype
) {
    const int bid = blockIdx.x;
    const int hid = blockIdx.y;
    const int sid = blockIdx.z;
    const int kv_head = hid / kv_group_size;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE_TQ;
    const int lane_id = tid % WARP_SIZE_TQ;
    const int dim_base = lane_id * DIMS_PER_THREAD;

    const int seq_len = seq_lens[bid];
    const int split_len = (seq_len + num_kv_splits - 1) / num_kv_splits;
    const int split_start = split_len * sid;
    const int split_end = min(split_start + split_len, seq_len);
    if (split_start >= split_end) return;

    __shared__ float s_c[N_CENTROIDS];
    __shared__ float s_m[NUM_WARPS];
    __shared__ float s_l[NUM_WARPS];
    __shared__ float s_acc[NUM_WARPS][HEAD_DIM];

    if (tid < N_CENTROIDS) s_c[tid] = centroids[tid];
    __syncthreads();

    float q0, q1, q2, q3;
    const int q_base = bid * stride_qb + hid * stride_qh;
    load_q_native(q_rot, q_base + dim_base, q_dtype, q0, q1, q2, q3);

    float m_prev = -1e30f;
    float l_prev = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int bt_base = bid * stride_bt;
    const unsigned short* kv_u16 = reinterpret_cast<const unsigned short*>(kv_cache);
    const int byte_idx = lane_id * 2;

    for (int base_n = split_start; base_n < split_end; base_n += BLOCK_KV * NUM_WARPS) {
        const int warp_base = base_n + warp_id * BLOCK_KV;
        const int actual_kv = min(BLOCK_KV, split_end - warp_base);
        if (actual_kv <= 0) continue;

        float scores[BLOCK_KV];
        long long data_bases[BLOCK_KV];
        int page_offs[BLOCK_KV];
        int block_nums[BLOCK_KV];

        #pragma unroll
        for (int kv = 0; kv < BLOCK_KV; kv++) {
            if (kv < actual_kv) {
                const int kv_pos = warp_base + kv;
                const int page_idx = kv_pos / block_size;
                const int page_off = kv_pos & (block_size - 1);
                const int block_num = block_table[bt_base + page_idx];
                const long long db = tq_v3_data_base(
                    block_num, page_off, kv_head, stride_cache_block, num_kv_heads);
                data_bases[kv] = db;
                page_offs[kv] = page_off;
                block_nums[kv] = block_num;

                const unsigned short mse =
                    *reinterpret_cast<const unsigned short*>(kv_cache + db + byte_idx);
                const float c0 = s_c[mse & 0xF];
                const float c1 = s_c[(mse >> 4) & 0xF];
                const float c2 = s_c[(mse >> 8) & 0xF];
                const float c3 = s_c[(mse >> 12) & 0xF];
                float score = q0 * c0 + q1 * c1 + q2 * c2 + q3 * c3;
                warp_reduce_sum(score);

                const int kn_addr = tq_v3_meta_u16_addr(
                    block_num, page_off, kv_head, stride_cache_block, block_size,
                    num_kv_heads, SOA_K_NORM);
                const unsigned short norm_u16 = kv_u16[kn_addr];
                const float knorm =
                    __half2float(*reinterpret_cast<const __half*>(&norm_u16));
                scores[kv] = knorm * score * attn_scale;
            } else {
                scores[kv] = -1e30f;
                data_bases[kv] = 0;
                page_offs[kv] = 0;
                block_nums[kv] = 0;
            }
        }

        float bmax = scores[0];
        #pragma unroll
        for (int kv = 1; kv < BLOCK_KV; kv++) bmax = fmaxf(bmax, scores[kv]);
        const float new_m = fmaxf(bmax, m_prev);
        const float rescale = expf(m_prev - new_m);
        float new_l = 0.0f;
        float p[BLOCK_KV];
        #pragma unroll
        for (int kv = 0; kv < BLOCK_KV; kv++) {
            p[kv] = expf(scores[kv] - new_m);
            new_l += p[kv];
        }
        acc0 *= rescale;
        acc1 *= rescale;
        acc2 *= rescale;
        acc3 *= rescale;

        #pragma unroll
        for (int kv = 0; kv < BLOCK_KV; kv++) {
            if (kv < actual_kv) {
                const long long vb = data_bases[kv] + MSE_BYTES;
                const int scale_addr = tq_v3_meta_u16_addr(
                    block_nums[kv], page_offs[kv], kv_head, stride_cache_block,
                    block_size, num_kv_heads, SOA_V_SCALE);
                const int zero_addr = tq_v3_meta_u16_addr(
                    block_nums[kv], page_offs[kv], kv_head, stride_cache_block,
                    block_size, num_kv_heads, SOA_V_ZERO);
                const unsigned short scale_u16 = kv_u16[scale_addr];
                const unsigned short zero_u16 = kv_u16[zero_addr];
                const float vs =
                    __half2float(*reinterpret_cast<const __half*>(&scale_u16));
                const float vz =
                    __half2float(*reinterpret_cast<const __half*>(&zero_u16));
                const unsigned short vu =
                    *reinterpret_cast<const unsigned short*>(kv_cache + vb + byte_idx);
                const float pn = p[kv];
                acc0 = __fmaf_rn(pn, (float)(vu & 0xF) * vs + vz, acc0);
                acc1 = __fmaf_rn(pn, (float)((vu >> 4) & 0xF) * vs + vz, acc1);
                acc2 = __fmaf_rn(pn, (float)((vu >> 8) & 0xF) * vs + vz, acc2);
                acc3 = __fmaf_rn(pn, (float)((vu >> 12) & 0xF) * vs + vz, acc3);
            }
        }
        l_prev = __fmaf_rn(l_prev, rescale, new_l);
        m_prev = new_m;
    }

    if (lane_id == 0) {
        s_m[warp_id] = m_prev;
        s_l[warp_id] = l_prev;
    }
    s_acc[warp_id][dim_base] = acc0;
    s_acc[warp_id][dim_base + 1] = acc1;
    s_acc[warp_id][dim_base + 2] = acc2;
    s_acc[warp_id][dim_base + 3] = acc3;
    __syncthreads();

    if (warp_id == 0) {
        float mg = s_m[0];
        #pragma unroll
        for (int w = 1; w < NUM_WARPS; w++) mg = fmaxf(mg, s_m[w]);
        float lt = 0.0f;
        float sc[NUM_WARPS];
        #pragma unroll
        for (int w = 0; w < NUM_WARPS; w++) {
            sc[w] = expf(s_m[w] - mg);
            lt += s_l[w] * sc[w];
        }
        const float inv_l = 1.0f / fmaxf(lt, 1e-16f);
        float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
        #pragma unroll
        for (int w = 0; w < NUM_WARPS; w++) {
            o0 += s_acc[w][dim_base] * sc[w];
            o1 += s_acc[w][dim_base + 1] * sc[w];
            o2 += s_acc[w][dim_base + 2] * sc[w];
            o3 += s_acc[w][dim_base + 3] * sc[w];
        }
        const int ob = bid * stride_mb + hid * stride_mh + sid * stride_ms;
        *reinterpret_cast<float4*>(mid_o + ob + dim_base) =
            make_float4(o0 * inv_l, o1 * inv_l, o2 * inv_l, o3 * inv_l);
        if (lane_id == 0) mid_o[ob + HEAD_DIM] = mg + logf(fmaxf(lt, 1e-16f));
    }
}

extern "C" __global__ __launch_bounds__(THREADS, 8)
void tq_v3_scalar_single(
    const void* __restrict__ q_rot,
    const unsigned char* __restrict__ kv_cache,
    const int* __restrict__ block_table,
    const int* __restrict__ seq_lens,
    const float* __restrict__ centroids,
    void* __restrict__ output,
    int stride_qb,
    int stride_qh,
    int stride_cache_block,
    int stride_bt,
    int stride_ob,
    int stride_oh,
    int num_kv_heads,
    int block_size,
    int kv_group_size,
    float attn_scale,
    int q_dtype,
    int out_dtype
) {
    const int bid = blockIdx.x;
    const int hid = blockIdx.y;
    const int kv_head = hid / kv_group_size;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE_TQ;
    const int lane_id = tid % WARP_SIZE_TQ;
    const int dim_base = lane_id * DIMS_PER_THREAD;

    const int seq_len = seq_lens[bid];
    if (seq_len <= 0) return;

    __shared__ float s_c[N_CENTROIDS];
    __shared__ float s_m[NUM_WARPS];
    __shared__ float s_l[NUM_WARPS];
    __shared__ float s_acc[NUM_WARPS][HEAD_DIM];

    if (tid < N_CENTROIDS) s_c[tid] = centroids[tid];
    __syncthreads();

    float q0, q1, q2, q3;
    const int q_base = bid * stride_qb + hid * stride_qh;
    load_q_native(q_rot, q_base + dim_base, q_dtype, q0, q1, q2, q3);

    float m_prev = -1e30f;
    float l_prev = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int bt_base = bid * stride_bt;
    const unsigned short* kv_u16 = reinterpret_cast<const unsigned short*>(kv_cache);
    const int byte_idx = lane_id * 2;

    for (int base_n = 0; base_n < seq_len; base_n += BLOCK_KV * NUM_WARPS) {
        const int warp_base = base_n + warp_id * BLOCK_KV;
        const int actual_kv = min(BLOCK_KV, max(0, seq_len - warp_base));
        if (actual_kv <= 0) continue;

        float scores[BLOCK_KV];
        long long data_bases[BLOCK_KV];
        int page_offs[BLOCK_KV];
        int block_nums[BLOCK_KV];

        #pragma unroll
        for (int kv = 0; kv < BLOCK_KV; kv++) {
            if (kv < actual_kv) {
                const int kv_pos = warp_base + kv;
                const int page_idx = kv_pos / block_size;
                const int page_off = kv_pos & (block_size - 1);
                const int block_num = block_table[bt_base + page_idx];
                const long long db = tq_v3_data_base(
                    block_num, page_off, kv_head, stride_cache_block, num_kv_heads);
                data_bases[kv] = db;
                page_offs[kv] = page_off;
                block_nums[kv] = block_num;
                const unsigned short mse =
                    *reinterpret_cast<const unsigned short*>(kv_cache + db + byte_idx);
                const float c0 = s_c[mse & 0xF];
                const float c1 = s_c[(mse >> 4) & 0xF];
                const float c2 = s_c[(mse >> 8) & 0xF];
                const float c3 = s_c[(mse >> 12) & 0xF];
                float score = q0 * c0 + q1 * c1 + q2 * c2 + q3 * c3;
                warp_reduce_sum(score);
                const int kn_addr = tq_v3_meta_u16_addr(
                    block_num, page_off, kv_head, stride_cache_block, block_size,
                    num_kv_heads, SOA_K_NORM);
                const unsigned short norm_u16 = kv_u16[kn_addr];
                const float knorm =
                    __half2float(*reinterpret_cast<const __half*>(&norm_u16));
                scores[kv] = knorm * score * attn_scale;
            } else {
                scores[kv] = -1e30f;
                data_bases[kv] = 0;
                page_offs[kv] = 0;
                block_nums[kv] = 0;
            }
        }

        float bmax = scores[0];
        #pragma unroll
        for (int kv = 1; kv < BLOCK_KV; kv++) bmax = fmaxf(bmax, scores[kv]);
        const float new_m = fmaxf(bmax, m_prev);
        const float rescale = expf(m_prev - new_m);
        float new_l = 0.0f;
        float p[BLOCK_KV];
        #pragma unroll
        for (int kv = 0; kv < BLOCK_KV; kv++) {
            p[kv] = expf(scores[kv] - new_m);
            new_l += p[kv];
        }
        acc0 *= rescale;
        acc1 *= rescale;
        acc2 *= rescale;
        acc3 *= rescale;

        #pragma unroll
        for (int kv = 0; kv < BLOCK_KV; kv++) {
            if (kv < actual_kv) {
                const long long vb = data_bases[kv] + MSE_BYTES;
                const int scale_addr = tq_v3_meta_u16_addr(
                    block_nums[kv], page_offs[kv], kv_head, stride_cache_block,
                    block_size, num_kv_heads, SOA_V_SCALE);
                const int zero_addr = tq_v3_meta_u16_addr(
                    block_nums[kv], page_offs[kv], kv_head, stride_cache_block,
                    block_size, num_kv_heads, SOA_V_ZERO);
                const unsigned short scale_u16 = kv_u16[scale_addr];
                const unsigned short zero_u16 = kv_u16[zero_addr];
                const float vs =
                    __half2float(*reinterpret_cast<const __half*>(&scale_u16));
                const float vz =
                    __half2float(*reinterpret_cast<const __half*>(&zero_u16));
                const unsigned short vu =
                    *reinterpret_cast<const unsigned short*>(kv_cache + vb + byte_idx);
                const float pn = p[kv];
                acc0 = __fmaf_rn(pn, (float)(vu & 0xF) * vs + vz, acc0);
                acc1 = __fmaf_rn(pn, (float)((vu >> 4) & 0xF) * vs + vz, acc1);
                acc2 = __fmaf_rn(pn, (float)((vu >> 8) & 0xF) * vs + vz, acc2);
                acc3 = __fmaf_rn(pn, (float)((vu >> 12) & 0xF) * vs + vz, acc3);
            }
        }
        l_prev = __fmaf_rn(l_prev, rescale, new_l);
        m_prev = new_m;
    }

    if (lane_id == 0) {
        s_m[warp_id] = m_prev;
        s_l[warp_id] = l_prev;
    }
    s_acc[warp_id][dim_base] = acc0;
    s_acc[warp_id][dim_base + 1] = acc1;
    s_acc[warp_id][dim_base + 2] = acc2;
    s_acc[warp_id][dim_base + 3] = acc3;
    __syncthreads();

    if (warp_id == 0) {
        float mg = s_m[0];
        #pragma unroll
        for (int w = 1; w < NUM_WARPS; w++) mg = fmaxf(mg, s_m[w]);
        float lt = 0.0f;
        float sc[NUM_WARPS];
        #pragma unroll
        for (int w = 0; w < NUM_WARPS; w++) {
            sc[w] = expf(s_m[w] - mg);
            lt += s_l[w] * sc[w];
        }
        const float inv_l = 1.0f / fmaxf(lt, 1e-16f);
        float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
        #pragma unroll
        for (int w = 0; w < NUM_WARPS; w++) {
            o0 += s_acc[w][dim_base] * sc[w];
            o1 += s_acc[w][dim_base + 1] * sc[w];
            o2 += s_acc[w][dim_base + 2] * sc[w];
            o3 += s_acc[w][dim_base + 3] * sc[w];
        }
        const int ob = bid * stride_ob + hid * stride_oh;
        store_native_4(output, ob + dim_base, o0 * inv_l, o1 * inv_l,
                       o2 * inv_l, o3 * inv_l, out_dtype);
    }
}

extern "C" __global__ __launch_bounds__(THREADS, 8)
void tq_v3_scalar_stage2(
    const float* __restrict__ mid_o,
    void* __restrict__ output,
    const int* __restrict__ seq_lens,
    int stride_mb,
    int stride_mh,
    int stride_ms,
    int stride_ob,
    int stride_oh,
    int num_kv_splits,
    int out_dtype
) {
    const int bid = blockIdx.x;
    const int hid = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane_id = tid % 64;

    const int seq_len = seq_lens[bid];
    if (seq_len <= 0 || tid >= HEAD_DIM) return;

    const int mid_base = bid * stride_mb + hid * stride_mh;
    float running_max = -1e30f;
    float l_sum = 0.0f;
    float acc = 0.0f;

    for (int s = 0; s < num_kv_splits; s++) {
        const int split_len = (seq_len + num_kv_splits - 1) / num_kv_splits;
        if (split_len * s >= seq_len) break;
        const int s_off = mid_base + s * stride_ms;
        float lse;
        if (lane_id == 0) lse = mid_o[s_off + HEAD_DIM];
        lse = __shfl(lse, 0, 64);

        const float new_max = fmaxf(running_max, lse);
        const float rescale = expf(running_max - new_max);
        const float weight = expf(lse - new_max);
        acc = acc * rescale + weight * mid_o[s_off + tid];
        l_sum = l_sum * rescale + weight;
        running_max = new_max;
    }

    const float result = acc / fmaxf(l_sum, 1e-16f);
    const int out_base = bid * stride_ob + hid * stride_oh;
    if (out_dtype == DTYPE_BF16) {
        reinterpret_cast<hip_bfloat16*>(output)[out_base + tid] = hip_bfloat16(result);
    } else {
        reinterpret_cast<__half*>(output)[out_base + tid] = __float2half(result);
    }
}

extern "C" void launch_tq_v3_scalar_decode(
    const void* q_rot,
    const unsigned char* kv_cache,
    const int* block_table,
    const int* seq_lens,
    const float* centroids,
    float* mid_o,
    void* output,
    int stride_qb,
    int stride_qh,
    int stride_cache_block,
    int stride_bt,
    int stride_mb,
    int stride_mh,
    int stride_ms,
    int stride_ob,
    int stride_oh,
    int num_kv_heads,
    int block_size,
    int num_kv_splits,
    int kv_group_size,
    float attn_scale,
    int q_dtype,
    int out_dtype,
    int B,
    int Hq,
    hipStream_t stream
) {
    if (num_kv_splits <= 1) {
        dim3 grid(B, Hq);
        dim3 block(THREADS, 1, 1);
        hipLaunchKernelGGL(
            tq_v3_scalar_single, grid, block, 0, stream,
            q_rot, kv_cache, block_table, seq_lens, centroids, output,
            stride_qb, stride_qh, stride_cache_block, stride_bt, stride_ob,
            stride_oh, num_kv_heads, block_size, kv_group_size, attn_scale,
            q_dtype, out_dtype);
        return;
    }

    dim3 grid1(B, Hq, num_kv_splits);
    dim3 block1(THREADS, 1, 1);
    hipLaunchKernelGGL(
        tq_v3_scalar_stage1, grid1, block1, 0, stream,
        q_rot, kv_cache, block_table, seq_lens, centroids, mid_o,
        stride_qb, stride_qh, stride_cache_block, stride_bt, stride_mb,
        stride_mh, stride_ms, num_kv_heads, block_size, num_kv_splits,
        kv_group_size, attn_scale, q_dtype);

    dim3 grid2(B, Hq);
    dim3 block2(THREADS, 1, 1);
    hipLaunchKernelGGL(
        tq_v3_scalar_stage2, grid2, block2, 0, stream,
        mid_o, output, seq_lens, stride_mb, stride_mh, stride_ms, stride_ob,
        stride_oh, num_kv_splits, out_dtype);
}
