/**
 * TQ Decode Stage1 — V112: V110 + Higher occupancy + all-waves MFMA
 *
 * Agent 2 optimizations:
 * 1. launch_bounds(256, 4) → allow 4 blocks/CU = 16 waves = 50% occupancy
 * 2. All 4 waves participate in MFMA (each wave handles 2 MFMA blocks)
 *    - Wave 0: kb=0,1  Wave 1: kb=2,3  Wave 2: kb=4,5  Wave 3: kb=6,7
 *    - Then cross-wave reduction via LDS
 * 3. Reduce LDS usage by reusing buffers
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

#define HEAD_DIM 128
#define MSE_BYTES 64
#define N_CENTROIDS 16
#define KPS 68
#define VAL_DATA_BYTES 64
#define BLOCK_N 16
#define KV_GROUP_SIZE 8
#define THREADS 256
#define THREADS_PER_HEAD 32
#define DIMS_PER_THREAD 4
#define WAVES 4
#define WAVE_SIZE 64
#define MFMA_BLOCKS_PER_WAVE 2  // 8 total / 4 waves

typedef short bf16x4_t __attribute__((ext_vector_type(4)));
typedef float float4_t __attribute__((ext_vector_type(4)));

__device__ __forceinline__ float bf16_to_float(short v) {
    union { float f; unsigned u; } r;
    r.u = ((unsigned)(unsigned short)v) << 16;
    return r.f;
}

extern "C" __global__ __launch_bounds__(256, 4)
void tq_decode_stage1_hip(
    const float* __restrict__ q_rot,
    const unsigned char* __restrict__ kv_cache,
    const int* __restrict__ block_table,
    const int* __restrict__ seq_lens,
    const float* __restrict__ centroids,
    float* __restrict__ mid_o,
    int stride_qb, int stride_qh,
    int stride_cb, int stride_cp, int stride_ch,
    int stride_bt,
    int stride_mb, int stride_mh, int stride_ms,
    int num_kv_heads, int block_size,
    int num_kv_splits, int kv_group_size,
    float attn_scale, int norm_correction
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

    const int wave_id = tid / WAVE_SIZE;   // 0-3
    const int lane_id = tid % WAVE_SIZE;   // 0-63
    const int my_head = tid / THREADS_PER_HEAD;
    const int my_lane = tid % THREADS_PER_HEAD;
    const int my_dim = my_lane * DIMS_PER_THREAD;
    const int q_head_global = kv_hid * kv_group_size + my_head;

    // ── LDS ──
    __shared__ unsigned int s_pair_lut[256];
    __shared__ float s_scores[KV_GROUP_SIZE][BLOCK_N];
    __shared__ float s_partial_scores[WAVES][KV_GROUP_SIZE][BLOCK_N]; // per-wave partials
    __shared__ float s_norms[BLOCK_N];
    __shared__ float s_csq[BLOCK_N];
    __shared__ float s_partial_nsq[WAVES][BLOCK_N]; // per-wave norm² partials
    __shared__ long long s_slot_base[BLOCK_N];
    __shared__ float s_v_scale[BLOCK_N];
    __shared__ float s_v_zero[BLOCK_N];

    // Build pairwise bf16 LUT
    if (tid < 256) {
        int lo = tid & 0xF, hi = (tid >> 4) & 0xF;
        union { float f; unsigned u; } clo, chi;
        clo.f = centroids[lo]; chi.f = centroids[hi];
        unsigned short bf16_lo = (unsigned short)(clo.u >> 16);
        unsigned short bf16_hi = (unsigned short)(chi.u >> 16);
        s_pair_lut[tid] = (unsigned int)bf16_lo | ((unsigned int)bf16_hi << 16);
    }
    __syncthreads();

    float m_prev = -1e30f, l_prev = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    const int bt_base = bid * stride_bt;
    const long long kv_head_offset = (long long)kv_hid * stride_ch;

    for (int base_n = split_start; base_n < split_end; base_n += BLOCK_N) {
        const int actual_n = min(BLOCK_N, split_end - base_n);

        // Precompute slot bases + norms + scale/zero
        if (tid < BLOCK_N) {
            if (tid < actual_n) {
                int kv_pos = base_n + tid;
                int pi = kv_pos / block_size;
                int po = kv_pos % block_size;
                int bn = block_table[bt_base + pi];
                long long sb = (long long)bn * stride_cb + (long long)po * stride_cp + kv_head_offset;
                s_slot_base[tid] = sb;
                unsigned short nu = (unsigned short)kv_cache[sb + MSE_BYTES]
                                  | ((unsigned short)kv_cache[sb + MSE_BYTES + 1] << 8);
                s_norms[tid] = __half2float(*reinterpret_cast<const __half*>(&nu));
                // Prefetch scale/zero
                long long scb = sb + KPS + VAL_DATA_BYTES;
                unsigned short su = (unsigned short)kv_cache[scb]
                                  | ((unsigned short)kv_cache[scb + 1] << 8);
                unsigned short zu = (unsigned short)kv_cache[scb + 2]
                                  | ((unsigned short)kv_cache[scb + 3] << 8);
                s_v_scale[tid] = __half2float(*reinterpret_cast<const __half*>(&su));
                s_v_zero[tid] = __half2float(*reinterpret_cast<const __half*>(&zu));
            } else {
                s_slot_base[tid] = 0;
                s_norms[tid] = 1.0f;
                s_v_scale[tid] = 1.0f;
                s_v_zero[tid] = 0.0f;
            }
        }
        __syncthreads();

        // ── Phase 1: ALL waves do MFMA (each wave handles 2 kb blocks) ──
        {
            float4_t c_acc = {0, 0, 0, 0};
            float my_nsq = 0.0f;
            int m = lane_id % 16;
            int k_grp = lane_id / 16;  // 0-3

            int kb_start = wave_id * MFMA_BLOCKS_PER_WAVE;
            int kb_end = kb_start + MFMA_BLOCKS_PER_WAVE;

            for (int kb = kb_start; kb < kb_end; kb++) {
                bf16x4_t a_frag;
                if (m < KV_GROUP_SIZE) {
                    int qh = kv_hid * kv_group_size + m;
                    int qo = bid * stride_qb + qh * stride_qh + kb * 16 + k_grp * 4;
                    union { float f; unsigned u; } cv0,cv1,cv2,cv3;
                    cv0.f = q_rot[qo]; cv1.f = q_rot[qo+1];
                    cv2.f = q_rot[qo+2]; cv3.f = q_rot[qo+3];
                    a_frag[0]=(short)(cv0.u>>16); a_frag[1]=(short)(cv1.u>>16);
                    a_frag[2]=(short)(cv2.u>>16); a_frag[3]=(short)(cv3.u>>16);
                } else {
                    a_frag[0]=a_frag[1]=a_frag[2]=a_frag[3]=(short)0;
                }

                int tok = lane_id % 16;
                bf16x4_t b_frag;
                if (tok < actual_n) {
                    long long sb = s_slot_base[tok];
                    int gd = kb * 16 + k_grp * 4;
                    int bo = gd / 2;
                    unsigned char b0 = kv_cache[sb + bo];
                    unsigned char b1 = kv_cache[sb + bo + 1];
                    unsigned int pk0 = s_pair_lut[b0];
                    unsigned int pk1 = s_pair_lut[b1];
                    b_frag[0] = (short)(pk0 & 0xFFFF);
                    b_frag[1] = (short)(pk0 >> 16);
                    b_frag[2] = (short)(pk1 & 0xFFFF);
                    b_frag[3] = (short)(pk1 >> 16);
                } else {
                    b_frag[0]=b_frag[1]=b_frag[2]=b_frag[3]=(short)0;
                }

                if (norm_correction) {
                    float f0 = bf16_to_float(b_frag[0]);
                    float f1 = bf16_to_float(b_frag[1]);
                    float f2 = bf16_to_float(b_frag[2]);
                    float f3 = bf16_to_float(b_frag[3]);
                    my_nsq += f0*f0 + f1*f1 + f2*f2 + f3*f3;
                }

                c_acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_frag, b_frag, c_acc, 0, 0, 0);
            }

            // Reduce norm² within each wave (across k_grp lanes)
            if (norm_correction) {
                my_nsq += __shfl_xor(my_nsq, 16, 64);
                my_nsq += __shfl_xor(my_nsq, 32, 64);
                int tok = lane_id % 16;
                if (lane_id / 16 == 0) {
                    s_partial_nsq[wave_id][tok] = (tok < actual_n) ? my_nsq : 0.0f;
                }
            }

            // Write partial scores to LDS
            int mfma_tok = lane_id % 16;
            int mfma_head_base = (lane_id / 16) * 4;
            for (int i = 0; i < 4; i++) {
                int hh = mfma_head_base + i;
                if (hh < KV_GROUP_SIZE) {
                    s_partial_scores[wave_id][hh][mfma_tok] = c_acc[i];
                }
            }
        }
        __syncthreads();

        // ── Reduce partial scores across waves ──
        if (tid < KV_GROUP_SIZE * BLOCK_N) {
            int h = tid / BLOCK_N;
            int tk = tid % BLOCK_N;
            float total_score = 0.0f;
            for (int w = 0; w < WAVES; w++)
                total_score += s_partial_scores[w][h][tk];

            if (tk < actual_n) {
                total_score *= attn_scale;
                if (norm_correction) {
                    float total_nsq = 0.0f;
                    for (int w = 0; w < WAVES; w++)
                        total_nsq += s_partial_nsq[w][tk];
                    total_score *= rsqrtf(total_nsq + 1e-16f);
                }
                s_scores[h][tk] = total_score * s_norms[tk];
            } else {
                s_scores[h][tk] = -1e30f;
            }
        }
        __syncthreads();

        // ── Phase 2: Softmax + Value accumulation ──
        float block_max = s_scores[my_head][0];
        #pragma unroll
        for (int i = 1; i < BLOCK_N; i++)
            block_max = fmaxf(block_max, s_scores[my_head][i]);

        float new_max = fmaxf(block_max, m_prev);
        float rescale = __expf(m_prev - new_max);
        float new_l = 0.0f;
        float p[BLOCK_N];
        #pragma unroll
        for (int i = 0; i < BLOCK_N; i++) {
            p[i] = __expf(s_scores[my_head][i] - new_max);
            new_l += p[i];
        }

        acc0 *= rescale; acc1 *= rescale; acc2 *= rescale; acc3 *= rescale;

        const int byte_idx = my_lane * 2;
        for (int n = 0; n < actual_n; n++) {
            long long vb = s_slot_base[n] + KPS;
            float vs = s_v_scale[n];
            float vz = s_v_zero[n];
            unsigned short vu = (unsigned short)kv_cache[vb + byte_idx]
                              | ((unsigned short)kv_cache[vb + byte_idx + 1] << 8);
            float pn = p[n];
            acc0 = __fmaf_rn(pn, (float)(vu & 0xF) * vs + vz, acc0);
            acc1 = __fmaf_rn(pn, (float)((vu >> 4) & 0xF) * vs + vz, acc1);
            acc2 = __fmaf_rn(pn, (float)((vu >> 8) & 0xF) * vs + vz, acc2);
            acc3 = __fmaf_rn(pn, (float)((vu >> 12) & 0xF) * vs + vz, acc3);
        }
        l_prev = l_prev * rescale + new_l;
        m_prev = new_max;
        __syncthreads();
    }

    float il = 1.0f / fmaxf(l_prev, 1e-16f);
    int ob = bid * stride_mb + q_head_global * stride_mh + sid * stride_ms;
    mid_o[ob + my_dim]     = acc0 * il;
    mid_o[ob + my_dim + 1] = acc1 * il;
    mid_o[ob + my_dim + 2] = acc2 * il;
    mid_o[ob + my_dim + 3] = acc3 * il;
    if (my_lane == 0)
        mid_o[ob + HEAD_DIM] = m_prev + logf(fmaxf(l_prev, 1e-16f));
}

extern "C" void launch_tq_decode_stage1(
    const float* q_rot, const unsigned char* kv_cache,
    const int* block_table, const int* seq_lens,
    const float* centroids, float* mid_o,
    int stride_qb, int stride_qh,
    int stride_cb, int stride_cp, int stride_ch,
    int stride_bt, int stride_mb, int stride_mh, int stride_ms,
    int num_kv_heads, int block_size, int num_kv_splits, int kv_group_size,
    float attn_scale, int norm_correction,
    int B, int Hq, hipStream_t stream
) {
    dim3 grid(B, num_kv_heads, num_kv_splits);
    dim3 block(256, 1, 1);
    hipLaunchKernelGGL(tq_decode_stage1_hip, grid, block, 0, stream,
        q_rot, kv_cache, block_table, seq_lens, centroids, mid_o,
        stride_qb, stride_qh, stride_cb, stride_cp, stride_ch, stride_bt,
        stride_mb, stride_mh, stride_ms,
        num_kv_heads, block_size, num_kv_splits, kv_group_size,
        attn_scale, norm_correction);
}
