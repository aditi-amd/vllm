// SPDX-License-Identifier: Apache-2.0
// Flash-style TurboQuant HIP decode for fusion v3 SoA cache layout.
//
// This kernel is the next FlashTQ pipeline step after hip_v3_mfma_qk:
// packed V is cached into LDS per KV tile so the 8 GQA query heads share one
// global V load. QK keeps the validated ROCm MFMA operand order (K fragment
// first, Q fragment second), and the online softmax / split LSE contract
// remains identical to hip_v3_mfma_qk.

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

#define BLOCK_N 16
#define KV_GROUP_SIZE 8
#define THREADS 256
#define THREADS_PER_HEAD 32
#define DIMS_PER_THREAD 4

#define DTYPE_BF16 0
#define DTYPE_FP16 1

typedef _Float16 half4_t __attribute__((ext_vector_type(4)));
typedef float float4_t __attribute__((ext_vector_type(4)));

__device__ __forceinline__ long long tq_v3_data_base_flash(
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

__device__ __forceinline__ int tq_v3_meta_u16_addr_flash(
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

__device__ __forceinline__ int stage_tq_v3_flash_tile(
    int base_n,
    int split_end,
    int bid,
    int kv_hid,
    int tid,
    int bt_base,
    int stride_cache_block,
    int stride_bt,
    int block_size,
    int num_kv_heads,
    const unsigned char* __restrict__ kv_cache,
    const int* __restrict__ block_table,
    const unsigned short* __restrict__ kv_u16,
    unsigned short* __restrict__ s_v_packed,
    long long* __restrict__ s_data_base,
    float* __restrict__ s_norms,
    float* __restrict__ s_vscales,
    float* __restrict__ s_vzeros
) {
    const int actual_n = min(BLOCK_N, split_end - base_n);

    if (tid < BLOCK_N) {
        if (tid < actual_n) {
            const int kv_pos = base_n + tid;
            const int page_idx = kv_pos / block_size;
            const int page_off = kv_pos & (block_size - 1);
            const int block_num = block_table[bt_base + page_idx];
            const long long db = tq_v3_data_base_flash(
                block_num, page_off, kv_hid, stride_cache_block, num_kv_heads);
            s_data_base[tid] = db;

            const int kn_addr = tq_v3_meta_u16_addr_flash(
                block_num, page_off, kv_hid, stride_cache_block, block_size,
                num_kv_heads, SOA_K_NORM);
            const int vs_addr = tq_v3_meta_u16_addr_flash(
                block_num, page_off, kv_hid, stride_cache_block, block_size,
                num_kv_heads, SOA_V_SCALE);
            const int vz_addr = tq_v3_meta_u16_addr_flash(
                block_num, page_off, kv_hid, stride_cache_block, block_size,
                num_kv_heads, SOA_V_ZERO);

            const unsigned short kn_u16 = kv_u16[kn_addr];
            const unsigned short vs_u16 = kv_u16[vs_addr];
            const unsigned short vz_u16 = kv_u16[vz_addr];
            s_norms[tid] = __half2float(*reinterpret_cast<const __half*>(&kn_u16));
            s_vscales[tid] = __half2float(*reinterpret_cast<const __half*>(&vs_u16));
            s_vzeros[tid] = __half2float(*reinterpret_cast<const __half*>(&vz_u16));
        } else {
            s_data_base[tid] = 0;
            s_norms[tid] = 1.0f;
            s_vscales[tid] = 0.0f;
            s_vzeros[tid] = 0.0f;
        }
    }
    __syncthreads();

    for (int linear = tid; linear < BLOCK_N * (HEAD_DIM / 4); linear += THREADS) {
        const int tok = linear / (HEAD_DIM / 4);
        const int lane = linear - tok * (HEAD_DIM / 4);
        if (tok < actual_n) {
            const long long db = s_data_base[tok];
            s_v_packed[linear] = *reinterpret_cast<const unsigned short*>(
                kv_cache + db + MSE_BYTES + lane * 2);
        } else {
            s_v_packed[linear] = 0;
        }
    }
    __syncthreads();
    return actual_n;
}

extern "C" __global__ __launch_bounds__(THREADS, 2)
void tq_v3_flash_tq_stage1(
    const float* __restrict__ q_rot,
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
    float attn_scale
) {
    const int bid = blockIdx.x;
    const int kv_hid = blockIdx.y;
    const int sid = blockIdx.z;
    const int tid = threadIdx.x;

    const int seq_len = seq_lens[bid];
    const int split_len = (seq_len + num_kv_splits - 1) / num_kv_splits;
    const int split_start = split_len * sid;
    const int split_end = min(split_start + split_len, seq_len);
    if (split_start >= split_end) return;

    const int my_head = tid / THREADS_PER_HEAD;
    const int my_lane = tid % THREADS_PER_HEAD;
    const int my_dim = my_lane * DIMS_PER_THREAD;
    const int q_head_global = kv_hid * kv_group_size + my_head;

    __shared__ float s_scores[KV_GROUP_SIZE][BLOCK_N];
    __shared__ float s_norms[BLOCK_N];
    __shared__ float s_vscales[BLOCK_N];
    __shared__ float s_vzeros[BLOCK_N];
    __shared__ long long s_data_base[BLOCK_N];
    __shared__ __half2 s_pair_f16[256];

    // Packed V is shared by the 8 GQA query heads. This removes repeated
    // global loads without paying the full K/V dequant-to-LDS cost. A single
    // buffer is enough because this version does synchronous per-tile staging.
    __shared__ unsigned short s_v_packed[BLOCK_N * (HEAD_DIM / 4)];

    if (tid < 256) {
        const int lo = tid & 0xF;
        const int hi = (tid >> 4) & 0xF;
        s_pair_f16[tid] = __halves2half2(
            __float2half(centroids[lo]), __float2half(centroids[hi]));
    }
    __syncthreads();

    float m_prev = -1e30f;
    float l_prev = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    const int bt_base = bid * stride_bt;
    const unsigned short* kv_u16 = reinterpret_cast<const unsigned short*>(kv_cache);

    for (int base_n = split_start; base_n < split_end; base_n += BLOCK_N) {
        const int actual_n = stage_tq_v3_flash_tile(
            base_n, split_end, bid, kv_hid, tid, bt_base, stride_cache_block,
            stride_bt, block_size, num_kv_heads, kv_cache, block_table,
            kv_u16, s_v_packed,
            s_data_base, s_norms, s_vscales, s_vzeros);

        if (tid < 64) {
            float4_t c_acc = {0.0f, 0.0f, 0.0f, 0.0f};

            #pragma unroll
            for (int kb = 0; kb < HEAD_DIM / 16; kb++) {
                const int m = tid % 16;
                const int k_grp = tid / 16;

                half4_t a_frag;
                if (m < KV_GROUP_SIZE) {
                    const int qh = kv_hid * kv_group_size + m;
                    const int qo = bid * stride_qb + qh * stride_qh + kb * 16 + k_grp * 4;
                    a_frag[0] = __float2half(q_rot[qo]);
                    a_frag[1] = __float2half(q_rot[qo + 1]);
                    a_frag[2] = __float2half(q_rot[qo + 2]);
                    a_frag[3] = __float2half(q_rot[qo + 3]);
                } else {
                    a_frag[0] = a_frag[1] = a_frag[2] = a_frag[3] = (_Float16)0;
                }

                const int tok = tid % 16;
                half4_t b_frag;
                if (tok < actual_n) {
                    const long long db = s_data_base[tok];
                    const int byte_off = (kb * 16 + k_grp * 4) / 2;
                    const unsigned char b0 = kv_cache[db + byte_off];
                    const unsigned char b1 = kv_cache[db + byte_off + 1];
                    const __half2 pk0 = s_pair_f16[b0];
                    const __half2 pk1 = s_pair_f16[b1];
                    b_frag[0] = __low2half(pk0);
                    b_frag[1] = __high2half(pk0);
                    b_frag[2] = __low2half(pk1);
                    b_frag[3] = __high2half(pk1);
                } else {
                    b_frag[0] = b_frag[1] = b_frag[2] = b_frag[3] = (_Float16)0;
                }

                c_acc = __builtin_amdgcn_mfma_f32_16x16x16f16(
                    b_frag, a_frag, c_acc, 0, 0, 0);
            }

            const int mfma_head = tid % 16;
            const int mfma_tok_base = (tid / 16) * 4;
            if (mfma_head < KV_GROUP_SIZE) {
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    const int tk = mfma_tok_base + i;
                    s_scores[mfma_head][tk] =
                        (tk < actual_n) ? c_acc[i] * s_norms[tk] * attn_scale : -1e30f;
                }
            }
        }
        __syncthreads();

        float block_max = s_scores[my_head][0];
        #pragma unroll
        for (int i = 1; i < BLOCK_N; i++) {
            block_max = fmaxf(block_max, s_scores[my_head][i]);
        }
        const float new_max = fmaxf(block_max, m_prev);
        const float rescale = expf(m_prev - new_max);
        float new_l = 0.0f;
        float p[BLOCK_N];
        #pragma unroll
        for (int i = 0; i < BLOCK_N; i++) {
            p[i] = expf(s_scores[my_head][i] - new_max);
            new_l += p[i];
        }

        acc0 *= rescale;
        acc1 *= rescale;
        acc2 *= rescale;
        acc3 *= rescale;

        #pragma unroll
        for (int n = 0; n < BLOCK_N; n++) {
            if (n < actual_n) {
                const unsigned short vu =
                    s_v_packed[n * (HEAD_DIM / 4) + my_lane];
                const float vs = s_vscales[n];
                const float vz = s_vzeros[n];
                const float pn = p[n];
                acc0 = __fmaf_rn(pn, (float)(vu & 0xF) * vs + vz, acc0);
                acc1 = __fmaf_rn(pn, (float)((vu >> 4) & 0xF) * vs + vz, acc1);
                acc2 = __fmaf_rn(pn, (float)((vu >> 8) & 0xF) * vs + vz, acc2);
                acc3 = __fmaf_rn(pn, (float)((vu >> 12) & 0xF) * vs + vz, acc3);
            }
        }

        l_prev = l_prev * rescale + new_l;
        m_prev = new_max;
        __syncthreads();
    }

    const float inv_l = 1.0f / fmaxf(l_prev, 1e-16f);
    const int ob = bid * stride_mb + q_head_global * stride_mh + sid * stride_ms;
    mid_o[ob + my_dim] = acc0 * inv_l;
    mid_o[ob + my_dim + 1] = acc1 * inv_l;
    mid_o[ob + my_dim + 2] = acc2 * inv_l;
    mid_o[ob + my_dim + 3] = acc3 * inv_l;
    if (my_lane == 0) {
        mid_o[ob + HEAD_DIM] = m_prev + logf(fmaxf(l_prev, 1e-16f));
    }
}

extern "C" __global__ __launch_bounds__(THREADS, 8)
void tq_v3_flash_tq_stage2(
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

extern "C" void launch_tq_v3_flash_tq_decode(
    const float* q_rot,
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
    int out_dtype,
    int B,
    int Hq,
    hipStream_t stream
) {
    dim3 grid1(B, num_kv_heads, num_kv_splits);
    dim3 block1(THREADS, 1, 1);
    hipLaunchKernelGGL(
        tq_v3_flash_tq_stage1, grid1, block1, 0, stream,
        q_rot, kv_cache, block_table, seq_lens, centroids, mid_o,
        stride_qb, stride_qh, stride_cache_block, stride_bt, stride_mb,
        stride_mh, stride_ms, num_kv_heads, block_size, num_kv_splits,
        kv_group_size, attn_scale);

    dim3 grid2(B, Hq);
    dim3 block2(THREADS, 1, 1);
    hipLaunchKernelGGL(
        tq_v3_flash_tq_stage2, grid2, block2, 0, stream,
        mid_o, output, seq_lens, stride_mb, stride_mh, stride_ms, stride_ob,
        stride_oh, num_kv_splits, out_dtype);
}
