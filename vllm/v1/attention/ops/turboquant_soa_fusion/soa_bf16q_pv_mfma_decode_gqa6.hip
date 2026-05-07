// SPDX-License-Identifier: Apache-2.0
// TurboQuant SoA decode with bf16 Q input and QK/PV MFMA.
//
// This candidate keeps the validated SoA cache addressing while using a
// consistent MFMA lane contract for QK scores and PV accumulation.

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
// gqa=6 variant: KV_GROUP_SIZE=6, THREADS and WAVES updated accordingly.
// HEAD_DIM/16 = 8 dim blocks must be covered by WAVES waves doing
// MFMA_BLOCKS_PER_WAVE blocks each, so we need WAVES * MFMA_BLOCKS_PER_WAVE
// >= 8.  For gqa=8 that was 4*2=8 (exact); for gqa=6 we use 3*3=9 with the
// last wave's trailing iteration skipped via predication (kb/db < 8).
// Original (gqa=8): #define KV_GROUP_SIZE 8
// Original (gqa=8): #define THREADS 256
// Original (gqa=8): #define WAVES 4
// Original (gqa=8): #define MFMA_BLOCKS_PER_WAVE 2
#define KV_GROUP_SIZE 6
#define THREADS 192       // = THREADS_PER_HEAD * KV_GROUP_SIZE = 32 * 6
#define THREADS_PER_HEAD 32
#define DIMS_PER_THREAD 4
#define WAVES 3           // = THREADS / WAVE_SIZE = 192 / 64
#define WAVE_SIZE 64
#define MFMA_BLOCKS_PER_WAVE 3   // ceil(8 / WAVES); last iter predicated
#define DIM_BLOCKS_TOTAL 8       // = HEAD_DIM / 16

#define DTYPE_BF16 0
#define DTYPE_FP16 1

typedef short bf16x4_t __attribute__((ext_vector_type(4)));
typedef float float4_t __attribute__((ext_vector_type(4)));

__device__ __forceinline__ long long tq_soa_data_base(
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

__device__ __forceinline__ int tq_soa_meta_u16_addr(
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

__device__ __forceinline__ short float_to_bf16(float f) {
    union { float fv; unsigned u; } r;
    r.fv = f;
    const unsigned lsb = (r.u >> 16) & 1;
    return (short)((r.u + 0x7FFFu + lsb) >> 16);
}

__device__ __forceinline__ float bf16_to_float(short v) {
    union { float f; unsigned u; } r;
    r.u = ((unsigned)(unsigned short)v) << 16;
    return r.f;
}

__device__ __forceinline__ void store_native_4_soa_fusion(
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

extern "C" __global__ __launch_bounds__(THREADS, 2)
void tq_soa_bf16q_pv_mfma_stage1(
    const short* __restrict__ q_rot_bf16,
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

    const int wave_id = tid / WAVE_SIZE;
    const int lane_id = tid % WAVE_SIZE;
    const int my_head = tid / THREADS_PER_HEAD;
    const int my_lane = tid % THREADS_PER_HEAD;
    const int my_dim = my_lane * DIMS_PER_THREAD;
    const int q_head_global = kv_hid * kv_group_size + my_head;

    __shared__ unsigned int s_pair_lut[256];
    __shared__ short s_q_bf16[KV_GROUP_SIZE * HEAD_DIM];
    __shared__ float s_scores[KV_GROUP_SIZE][BLOCK_N];
    __shared__ float s_partial_scores[WAVES][KV_GROUP_SIZE][BLOCK_N];
    __shared__ float s_norms[BLOCK_N];
    __shared__ float s_vscales[BLOCK_N];
    __shared__ float s_vzeros[BLOCK_N];
    __shared__ long long s_data_base[BLOCK_N];
    __shared__ unsigned char s_key_data[BLOCK_N * MSE_BYTES];
    __shared__ short s_val_bf16[BLOCK_N * HEAD_DIM];
    __shared__ float s_pv_result[KV_GROUP_SIZE * HEAD_DIM];

    // Load all 256 LUT entries. With THREADS=192 we need 2 passes to cover all 256.
    // Original (gqa=8, THREADS=256): single pass was sufficient.
    if (tid < 256) {
        const int lo = tid & 0xF;
        const int hi = (tid >> 4) & 0xF;
        const unsigned short bf16_lo = (unsigned short)float_to_bf16(centroids[lo]);
        const unsigned short bf16_hi = (unsigned short)float_to_bf16(centroids[hi]);
        s_pair_lut[tid] = (unsigned int)bf16_lo | ((unsigned int)bf16_hi << 16);
    }
    // Second pass: load remaining entries [THREADS..255] = [192..255] (64 entries).
    // When THREADS==256 this block is dead (tid < 0 is never true).
    if (tid < (256 - THREADS)) {
        const int lut_idx = THREADS + tid;
        const int lo = lut_idx & 0xF;
        const int hi = (lut_idx >> 4) & 0xF;
        const unsigned short bf16_lo = (unsigned short)float_to_bf16(centroids[lo]);
        const unsigned short bf16_hi = (unsigned short)float_to_bf16(centroids[hi]);
        s_pair_lut[lut_idx] = (unsigned int)bf16_lo | ((unsigned int)bf16_hi << 16);
    }

    {
        const int per_thread = (KV_GROUP_SIZE * HEAD_DIM) / THREADS;
        const int base_idx = tid * per_thread;
        #pragma unroll
        for (int i = 0; i < per_thread; i++) {
            const int idx = base_idx + i;
            const int h = idx / HEAD_DIM;
            const int d = idx - h * HEAD_DIM;
            const int qh = kv_hid * kv_group_size + h;
            const int qo = bid * stride_qb + qh * stride_qh + d;
            s_q_bf16[idx] = q_rot_bf16[qo];
        }
    }
    __syncthreads();

    float m_prev = -1e30f;
    float l_prev = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int bt_base = bid * stride_bt;
    const unsigned short* kv_u16 = reinterpret_cast<const unsigned short*>(kv_cache);

    for (int base_n = split_start; base_n < split_end; base_n += BLOCK_N) {
        const int actual_n = min(BLOCK_N, split_end - base_n);

        if (tid < BLOCK_N) {
            if (tid < actual_n) {
                const int kv_pos = base_n + tid;
                const int page_idx = kv_pos / block_size;
                const int page_off = kv_pos & (block_size - 1);
                const int block_num = block_table[bt_base + page_idx];
                const long long db = tq_soa_data_base(
                    block_num, page_off, kv_hid, stride_cache_block, num_kv_heads);
                s_data_base[tid] = db;

                const int kn_addr = tq_soa_meta_u16_addr(
                    block_num, page_off, kv_hid, stride_cache_block, block_size,
                    num_kv_heads, SOA_K_NORM);
                const int vs_addr = tq_soa_meta_u16_addr(
                    block_num, page_off, kv_hid, stride_cache_block, block_size,
                    num_kv_heads, SOA_V_SCALE);
                const int vz_addr = tq_soa_meta_u16_addr(
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
                s_vscales[tid] = 1.0f;
                s_vzeros[tid] = 0.0f;
            }
        }
        __syncthreads();

        // K data: BLOCK_N tokens × 16 4-byte chunks = 256 thread-loads.
        // With THREADS=192 we need ceil(256/192)=2 passes so all 16 tokens
        // get loaded. The original (gqa=8, THREADS=256) needed only 1 pass.
        #pragma unroll
        for (int kpass = 0; kpass < 2; kpass++) {
            const int gid = kpass * THREADS + tid;
            if (gid < BLOCK_N * 16) {
                const int load_tok = gid / 16;
                const int load_off = gid & 0xF;
                const int byte_off = load_off * 4;
                unsigned int raw = 0;
                if (load_tok < actual_n) {
                    const long long db = s_data_base[load_tok];
                    raw = (unsigned int)kv_cache[db + byte_off]
                        | ((unsigned int)kv_cache[db + byte_off + 1] << 8)
                        | ((unsigned int)kv_cache[db + byte_off + 2] << 16)
                        | ((unsigned int)kv_cache[db + byte_off + 3] << 24);
                }
                *reinterpret_cast<unsigned int*>(
                    &s_key_data[load_tok * MSE_BYTES + byte_off]) = raw;
            }
        }

        // V data: same shape, same need for 2 passes when THREADS=192.
        #pragma unroll
        for (int vpass = 0; vpass < 2; vpass++) {
            const int gid = vpass * THREADS + tid;
            if (gid < BLOCK_N * 16) {
                const int tok = gid / 16;
                const int widx = gid & 0xF;
                const int dim_base = widx * 8;
                const float vs = s_vscales[tok];
                const float vz = s_vzeros[tok];
                unsigned int raw = 0;
                if (tok < actual_n) {
                    const long long vb = s_data_base[tok] + MSE_BYTES;
                    const int byte_off = widx * 4;
                    raw = (unsigned int)kv_cache[vb + byte_off]
                        | ((unsigned int)kv_cache[vb + byte_off + 1] << 8)
                        | ((unsigned int)kv_cache[vb + byte_off + 2] << 16)
                        | ((unsigned int)kv_cache[vb + byte_off + 3] << 24);
                }
                #pragma unroll
                for (int d = 0; d < 8; d++) {
                    const float fv = (tok < actual_n)
                        ? ((float)((raw >> (d * 4)) & 0xF) * vs + vz)
                        : 0.0f;
                    s_val_bf16[tok * HEAD_DIM + dim_base + d] = float_to_bf16(fv);
                }
            }
        }
        __syncthreads();

        {
            float4_t c_acc = {0.0f, 0.0f, 0.0f, 0.0f};
            const int m = lane_id % 16;
            const int k_grp = lane_id / 16;
            const int kb_start = wave_id * MFMA_BLOCKS_PER_WAVE;

            #pragma unroll
            for (int kbi = 0; kbi < MFMA_BLOCKS_PER_WAVE; kbi++) {
                const int kb = kb_start + kbi;
                bf16x4_t a_frag;
                if (kb < DIM_BLOCKS_TOTAL && m < KV_GROUP_SIZE) {
                    const int q_base = m * HEAD_DIM + kb * 16 + k_grp * 4;
                    a_frag[0] = s_q_bf16[q_base];
                    a_frag[1] = s_q_bf16[q_base + 1];
                    a_frag[2] = s_q_bf16[q_base + 2];
                    a_frag[3] = s_q_bf16[q_base + 3];
                } else {
                    a_frag[0] = a_frag[1] = a_frag[2] = a_frag[3] = (short)0;
                }

                const int tok = lane_id % 16;
                bf16x4_t b_frag;
                if (kb < DIM_BLOCKS_TOTAL && tok < actual_n) {
                    const int gd = kb * 16 + k_grp * 4;
                    const int bo = gd / 2;
                    const unsigned char b0 = s_key_data[tok * MSE_BYTES + bo];
                    const unsigned char b1 = s_key_data[tok * MSE_BYTES + bo + 1];
                    const unsigned int pk0 = s_pair_lut[b0];
                    const unsigned int pk1 = s_pair_lut[b1];
                    b_frag[0] = (short)(pk0 & 0xFFFF);
                    b_frag[1] = (short)(pk0 >> 16);
                    b_frag[2] = (short)(pk1 & 0xFFFF);
                    b_frag[3] = (short)(pk1 >> 16);
                } else {
                    b_frag[0] = b_frag[1] = b_frag[2] = b_frag[3] = (short)0;
                }

                c_acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
                    a_frag, b_frag, c_acc, 0, 0, 0);
            }

            const int mfma_tok = lane_id % 16;
            const int mfma_head_base = (lane_id / 16) * 4;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const int hh = mfma_head_base + i;
                if (hh < KV_GROUP_SIZE) {
                    s_partial_scores[wave_id][hh][mfma_tok] = c_acc[i];
                }
            }
        }
        __syncthreads();

        if (tid < KV_GROUP_SIZE * BLOCK_N) {
            const int h = tid / BLOCK_N;
            const int tk = tid - h * BLOCK_N;
            float total_score = 0.0f;
            #pragma unroll
            for (int w = 0; w < WAVES; w++) {
                total_score += s_partial_scores[w][h][tk];
            }
            s_scores[h][tk] =
                (tk < actual_n) ? total_score * s_norms[tk] * attn_scale : -1e30f;
        }
        __syncthreads();

        float block_max = s_scores[my_head][0];
        #pragma unroll
        for (int i = 1; i < BLOCK_N; i++) {
            block_max = fmaxf(block_max, s_scores[my_head][i]);
        }
        const float new_max = fmaxf(block_max, m_prev);
        const float rescale = __expf(m_prev - new_max);
        float new_l = 0.0f;
        #pragma unroll
        for (int i = 0; i < BLOCK_N; i++) {
            const float pi = __expf(s_scores[my_head][i] - new_max);
            new_l += pi;
            s_scores[my_head][i] = pi;
        }

        acc0 *= rescale;
        acc1 *= rescale;
        acc2 *= rescale;
        acc3 *= rescale;

        __syncthreads();

        {
            const int m_lane = lane_id % 16;
            const int k_grp = lane_id / 16;

            bf16x4_t p_frag;
            if (m_lane < KV_GROUP_SIZE) {
                p_frag[0] = float_to_bf16(s_scores[m_lane][k_grp * 4]);
                p_frag[1] = float_to_bf16(s_scores[m_lane][k_grp * 4 + 1]);
                p_frag[2] = float_to_bf16(s_scores[m_lane][k_grp * 4 + 2]);
                p_frag[3] = float_to_bf16(s_scores[m_lane][k_grp * 4 + 3]);
            } else {
                p_frag[0] = p_frag[1] = p_frag[2] = p_frag[3] = (short)0;
            }

            const int db_start = wave_id * MFMA_BLOCKS_PER_WAVE;
            #pragma unroll
            for (int dbi = 0; dbi < MFMA_BLOCKS_PER_WAVE; dbi++) {
                const int db = db_start + dbi;
                if (db >= DIM_BLOCKS_TOTAL) continue;
                float4_t v_acc = {0.0f, 0.0f, 0.0f, 0.0f};
                const int dim_col = (lane_id % 16) + db * 16;
                const int tok0 = k_grp * 4;

                bf16x4_t v_frag;
                v_frag[0] = s_val_bf16[tok0 * HEAD_DIM + dim_col];
                v_frag[1] = s_val_bf16[(tok0 + 1) * HEAD_DIM + dim_col];
                v_frag[2] = s_val_bf16[(tok0 + 2) * HEAD_DIM + dim_col];
                v_frag[3] = s_val_bf16[(tok0 + 3) * HEAD_DIM + dim_col];

                v_acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
                    p_frag, v_frag, v_acc, 0, 0, 0);

                const int head_base = (lane_id / 16) * 4;
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    const int hh = head_base + i;
                    if (hh < KV_GROUP_SIZE) {
                        s_pv_result[hh * HEAD_DIM + dim_col] = v_acc[i];
                    }
                }
            }
        }
        __syncthreads();

        acc0 += s_pv_result[my_head * HEAD_DIM + my_dim];
        acc1 += s_pv_result[my_head * HEAD_DIM + my_dim + 1];
        acc2 += s_pv_result[my_head * HEAD_DIM + my_dim + 2];
        acc3 += s_pv_result[my_head * HEAD_DIM + my_dim + 3];
        l_prev = l_prev * rescale + new_l;
        m_prev = new_max;
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
void tq_soa_bf16q_pv_mfma_stage2(
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

extern "C" void launch_tq_soa_bf16q_pv_mfma_decode(
    const short* q_rot_bf16,
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
        tq_soa_bf16q_pv_mfma_stage1, grid1, block1, 0, stream,
        q_rot_bf16, kv_cache, block_table, seq_lens, centroids, mid_o,
        stride_qb, stride_qh, stride_cache_block, stride_bt, stride_mb,
        stride_mh, stride_ms, num_kv_heads, block_size, num_kv_splits,
        kv_group_size, attn_scale);

    dim3 grid2(B, Hq);
    dim3 block2(THREADS, 1, 1);
    hipLaunchKernelGGL(
        tq_soa_bf16q_pv_mfma_stage2, grid2, block2, 0, stream,
        mid_o, output, seq_lens, stride_mb, stride_mh, stride_ms, stride_ob,
        stride_oh, num_kv_splits, out_dtype);
}
