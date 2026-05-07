/**
 * TQ Decode Unified V136: v132 + bf16 Q input + adaptive splits + direct output
 *
 * Based on v132 (vec loads + Q prefetch, our best Stage1 kernel).
 *
 * Key innovations over v132:
 * 1. NATIVE BF16 Q INPUT:
 *    Accepts bf16 q_rot directly (halves HBM Q traffic: 2B vs 4B per element).
 *    Host computes q_rot = (query.float() @ PiT).bfloat16() — saves the fp32→bf16
 *    conversion that v132 does per-element in the kernel prologue.
 *    (v132 loads fp32, converts to bf16 in LDS — wastes HBM bandwidth)
 *
 * 2. DIRECT OUTPUT PATH (num_kv_splits == 1):
 *    When the Python dispatch sets splits=1 (short sequences), skip mid_o
 *    entirely and write final bf16 output directly. Eliminates:
 *      - mid_o allocation & write (~B*Hq*(D+1)*4 bytes)
 *      - Stage2 kernel launch (~5µs)
 *      - Stage2 mid_o read + output write
 *    Grid = (B, Hk, 1) with each CTA processing full sequence.
 *
 * 3. SPLIT PATH (num_kv_splits > 1):
 *    Identical to v132 — writes partials to mid_o. Separate Stage2 reduces.
 *
 * This is a TRUE unified kernel: same binary handles both paths.
 * The Python dispatch selects splits based on seq_len:
 *   - seq_len <= 512: splits=1 (direct output, single kernel)
 *   - seq_len > 512:  splits=N  (mid_o + Stage2, two kernels)
 *
 * Architecture: 256 threads = 4 wavefronts, Grid=(B, Hk, splits).
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
#define MFMA_BLOCKS_PER_WAVE 2

typedef short bf16x4_t __attribute__((ext_vector_type(4)));
typedef float float4_t __attribute__((ext_vector_type(4)));

__device__ __forceinline__ float bf16_to_float(short v) {
    union { float f; unsigned u; } r;
    r.u = ((unsigned)(unsigned short)v) << 16;
    return r.f;
}

__device__ __forceinline__ short float_to_bf16(float f) {
    union { float fv; unsigned u; } r;
    r.fv = f;
    return (short)(r.u >> 16);
}

extern "C" __global__ __launch_bounds__(256, 3)
void tq_decode_unified_hip(
    const short* __restrict__ q_rot_bf16,  // [B, Hq, D] bf16 (stored as short)
    const unsigned char* __restrict__ kv_cache,
    const int* __restrict__ block_table,
    const int* __restrict__ seq_lens,
    const float* __restrict__ centroids,
    float* __restrict__ mid_o,             // [B, Hq, splits, D+1] — used when splits > 1
    void* __restrict__ output,             // [B, Hq, D] bf16 — used when splits == 1
    int stride_qb, int stride_qh,
    int stride_cb, int stride_cp, int stride_ch,
    int stride_bt,
    int stride_mb, int stride_mh, int stride_ms,
    int stride_ob, int stride_oh,
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

    const int wave_id = tid / WAVE_SIZE;
    const int lane_id = tid % WAVE_SIZE;
    const int my_head = tid / THREADS_PER_HEAD;
    const int my_lane = tid % THREADS_PER_HEAD;
    const int my_dim = my_lane * DIMS_PER_THREAD;
    const int q_head_global = kv_hid * kv_group_size + my_head;

    const int direct_output = (num_kv_splits == 1);

    __shared__ unsigned int s_pair_lut[256];
    __shared__ short s_q_bf16[KV_GROUP_SIZE * HEAD_DIM];
    __shared__ float s_scores[KV_GROUP_SIZE][BLOCK_N];
    __shared__ float s_partial_scores[WAVES][KV_GROUP_SIZE][BLOCK_N];
    __shared__ float s_norms[BLOCK_N];
    __shared__ float s_partial_nsq[WAVES][BLOCK_N];
    __shared__ long long s_slot_base[BLOCK_N];
    __shared__ float s_v_scale[BLOCK_N];
    __shared__ float s_v_zero[BLOCK_N];
    __shared__ unsigned char s_key_data[BLOCK_N * MSE_BYTES];
    __shared__ short s_val_bf16[BLOCK_N * HEAD_DIM];
    __shared__ float s_pv_result[KV_GROUP_SIZE * HEAD_DIM];

    // Build pair LUT
    if (tid < 256) {
        int lo = tid & 0xF, hi = (tid >> 4) & 0xF;
        union { float f; unsigned u; } clo, chi;
        clo.f = centroids[lo]; chi.f = centroids[hi];
        unsigned short bf16_lo = (unsigned short)(clo.u >> 16);
        unsigned short bf16_hi = (unsigned short)(chi.u >> 16);
        s_pair_lut[tid] = (unsigned int)bf16_lo | ((unsigned int)bf16_hi << 16);
    }

    // Pre-load Q bf16 to LDS — DIRECT COPY, no fp32→bf16 conversion needed!
    {
        int q_elements = KV_GROUP_SIZE * HEAD_DIM;  // 1024
        int per_thread = q_elements / THREADS;       // 4
        int base_idx = tid * per_thread;
        for (int i = 0; i < per_thread; i++) {
            int idx = base_idx + i;
            int h = idx / HEAD_DIM;
            int d = idx % HEAD_DIM;
            int qh = kv_hid * kv_group_size + h;
            int qo = bid * stride_qb + qh * stride_qh + d;
            s_q_bf16[idx] = q_rot_bf16[qo];  // Direct bf16 copy — half the bandwidth!
        }
    }
    __syncthreads();

    float m_prev = -1e30f, l_prev = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    const int bt_base = bid * stride_bt;
    const long long kv_head_offset = (long long)kv_hid * stride_ch;

    for (int base_n = split_start; base_n < split_end; base_n += BLOCK_N) {
        const int actual_n = min(BLOCK_N, split_end - base_n);

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
                long long scb = sb + KPS + VAL_DATA_BYTES;
                unsigned int sz_packed = (unsigned int)kv_cache[scb]
                                       | ((unsigned int)kv_cache[scb + 1] << 8)
                                       | ((unsigned int)kv_cache[scb + 2] << 16)
                                       | ((unsigned int)kv_cache[scb + 3] << 24);
                s_v_scale[tid] = __half2float(*reinterpret_cast<const __half*>(&sz_packed));
                unsigned short zu = (unsigned short)(sz_packed >> 16);
                s_v_zero[tid] = __half2float(*reinterpret_cast<const __half*>(&zu));
            } else {
                s_slot_base[tid] = 0;
                s_norms[tid] = 1.0f;
                s_v_scale[tid] = 1.0f;
                s_v_zero[tid] = 0.0f;
            }
        }
        __syncthreads();

        {
            int load_tok = tid / 16;
            int load_off = tid % 16;
            if (load_tok < BLOCK_N) {
                int byte_off = load_off * 4;
                if (load_tok < actual_n) {
                    long long sb = s_slot_base[load_tok];
                    unsigned int raw = (unsigned int)kv_cache[sb + byte_off]
                                     | ((unsigned int)kv_cache[sb + byte_off + 1] << 8)
                                     | ((unsigned int)kv_cache[sb + byte_off + 2] << 16)
                                     | ((unsigned int)kv_cache[sb + byte_off + 3] << 24);
                    *reinterpret_cast<unsigned int*>(&s_key_data[load_tok * MSE_BYTES + byte_off]) = raw;
                } else {
                    *reinterpret_cast<unsigned int*>(&s_key_data[load_tok * MSE_BYTES + byte_off]) = 0;
                }
            }
        }
        {
            int my_tok = tid / 16;
            int widx = tid % 16;
            int my_base = widx * 8;
            if (my_tok < BLOCK_N) {
                float vs, vz;
                unsigned int r;
                if (my_tok < actual_n) {
                    vs = s_v_scale[my_tok]; vz = s_v_zero[my_tok];
                    long long vb = s_slot_base[my_tok] + KPS;
                    int byte_off = widx * 4;
                    r = (unsigned int)kv_cache[vb + byte_off]
                      | ((unsigned int)kv_cache[vb + byte_off + 1] << 8)
                      | ((unsigned int)kv_cache[vb + byte_off + 2] << 16)
                      | ((unsigned int)kv_cache[vb + byte_off + 3] << 24);
                } else {
                    vs = 1.0f; vz = 0.0f; r = 0;
                }
                #pragma unroll
                for (int d = 0; d < 8; d++) {
                    float fv = (my_tok < actual_n) ? ((float)((r >> (d*4)) & 0xF) * vs + vz) : 0.0f;
                    s_val_bf16[my_tok * HEAD_DIM + my_base + d] = float_to_bf16(fv);
                }
            }
        }
        __syncthreads();

        {
            float4_t c_acc = {0, 0, 0, 0};
            float my_nsq = 0.0f;
            int m = lane_id % 16;
            int k_grp = lane_id / 16;
            int kb_start = wave_id * MFMA_BLOCKS_PER_WAVE;
            int kb_end = kb_start + MFMA_BLOCKS_PER_WAVE;
            for (int kb = kb_start; kb < kb_end; kb++) {
                bf16x4_t a_frag;
                if (m < KV_GROUP_SIZE) {
                    int q_base = m * HEAD_DIM + kb * 16 + k_grp * 4;
                    a_frag[0] = s_q_bf16[q_base];
                    a_frag[1] = s_q_bf16[q_base + 1];
                    a_frag[2] = s_q_bf16[q_base + 2];
                    a_frag[3] = s_q_bf16[q_base + 3];
                } else {
                    a_frag[0]=a_frag[1]=a_frag[2]=a_frag[3]=(short)0;
                }
                int tok = lane_id % 16;
                bf16x4_t b_frag;
                if (tok < actual_n) {
                    int gd = kb * 16 + k_grp * 4;
                    int bo = gd / 2;
                    unsigned char b0 = s_key_data[tok * MSE_BYTES + bo];
                    unsigned char b1 = s_key_data[tok * MSE_BYTES + bo + 1];
                    unsigned int pk0 = s_pair_lut[b0]; unsigned int pk1 = s_pair_lut[b1];
                    b_frag[0]=(short)(pk0&0xFFFF); b_frag[1]=(short)(pk0>>16);
                    b_frag[2]=(short)(pk1&0xFFFF); b_frag[3]=(short)(pk1>>16);
                } else {
                    b_frag[0]=b_frag[1]=b_frag[2]=b_frag[3]=(short)0;
                }
                if (norm_correction) {
                    float f0=bf16_to_float(b_frag[0]),f1=bf16_to_float(b_frag[1]);
                    float f2=bf16_to_float(b_frag[2]),f3=bf16_to_float(b_frag[3]);
                    my_nsq += f0*f0+f1*f1+f2*f2+f3*f3;
                }
                c_acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_frag, b_frag, c_acc, 0, 0, 0);
            }
            if (norm_correction) {
                my_nsq += __shfl_xor(my_nsq,16,64); my_nsq += __shfl_xor(my_nsq,32,64);
                int tok = lane_id % 16;
                if (lane_id/16 == 0) s_partial_nsq[wave_id][tok] = (tok<actual_n)?my_nsq:0.0f;
            }
            int mfma_tok = lane_id % 16;
            int mfma_head_base = (lane_id / 16) * 4;
            for (int i = 0; i < 4; i++) {
                int hh = mfma_head_base + i;
                if (hh < KV_GROUP_SIZE) s_partial_scores[wave_id][hh][mfma_tok] = c_acc[i];
            }
        }
        __syncthreads();

        if (tid < KV_GROUP_SIZE * BLOCK_N) {
            int h = tid / BLOCK_N; int tk = tid % BLOCK_N;
            float total_score = 0.0f;
            for (int w = 0; w < WAVES; w++) total_score += s_partial_scores[w][h][tk];
            if (tk < actual_n) {
                total_score *= attn_scale;
                if (norm_correction) {
                    float total_nsq = 0.0f;
                    for (int w = 0; w < WAVES; w++) total_nsq += s_partial_nsq[w][tk];
                    total_score *= rsqrtf(total_nsq + 1e-16f);
                }
                s_scores[h][tk] = total_score * s_norms[tk];
            } else {
                s_scores[h][tk] = -1e30f;
            }
        }
        __syncthreads();

        float block_max = s_scores[my_head][0];
        #pragma unroll
        for (int i = 1; i < BLOCK_N; i++) block_max = fmaxf(block_max, s_scores[my_head][i]);
        float new_max = fmaxf(block_max, m_prev);
        float rescale = __expf(m_prev - new_max);
        float new_l = 0.0f;
        float p[BLOCK_N];
        #pragma unroll
        for (int i = 0; i < BLOCK_N; i++) { p[i] = __expf(s_scores[my_head][i] - new_max); new_l += p[i]; }
        acc0 *= rescale; acc1 *= rescale; acc2 *= rescale; acc3 *= rescale;

        {
            int m = lane_id % 16; int k_grp = lane_id / 16;
            bf16x4_t p_frag;
            if (m < KV_GROUP_SIZE) {
                p_frag[0]=float_to_bf16(p[k_grp*4]);   p_frag[1]=float_to_bf16(p[k_grp*4+1]);
                p_frag[2]=float_to_bf16(p[k_grp*4+2]); p_frag[3]=float_to_bf16(p[k_grp*4+3]);
            } else { p_frag[0]=p_frag[1]=p_frag[2]=p_frag[3]=0; }
            int db_start = wave_id * 2;
            for (int db = db_start; db < db_start + 2; db++) {
                float4_t v_acc = {0,0,0,0};
                int dim_col = lane_id%16 + db*16;
                int tok0 = k_grp*4;
                bf16x4_t v_frag;
                v_frag[0]=s_val_bf16[tok0*HEAD_DIM+dim_col];
                v_frag[1]=s_val_bf16[(tok0+1)*HEAD_DIM+dim_col];
                v_frag[2]=s_val_bf16[(tok0+2)*HEAD_DIM+dim_col];
                v_frag[3]=s_val_bf16[(tok0+3)*HEAD_DIM+dim_col];
                v_acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(p_frag, v_frag, v_acc, 0, 0, 0);
                int head_base = (lane_id/16)*4; int dim_n = lane_id%16 + db*16;
                for (int i = 0; i < 4; i++) {
                    int hh = head_base + i;
                    if (hh < KV_GROUP_SIZE) s_pv_result[hh*HEAD_DIM+dim_n] = v_acc[i];
                }
            }
        }
        __syncthreads();

        acc0 += s_pv_result[my_head*HEAD_DIM+my_dim];
        acc1 += s_pv_result[my_head*HEAD_DIM+my_dim+1];
        acc2 += s_pv_result[my_head*HEAD_DIM+my_dim+2];
        acc3 += s_pv_result[my_head*HEAD_DIM+my_dim+3];
        l_prev = l_prev * rescale + new_l;
        m_prev = new_max;
    }

    // ═══ Output: dual-mode write ═══
    float il = 1.0f / fmaxf(l_prev, 1e-16f);

    if (direct_output) {
        // Direct bf16 output — no Stage2 needed
        float r0 = acc0 * il, r1 = acc1 * il, r2 = acc2 * il, r3 = acc3 * il;
        int out_base = bid * stride_ob + q_head_global * stride_oh;
        hip_bfloat16* out_bf16 = (hip_bfloat16*)output;
        out_bf16[out_base + my_dim]     = hip_bfloat16(r0);
        out_bf16[out_base + my_dim + 1] = hip_bfloat16(r1);
        out_bf16[out_base + my_dim + 2] = hip_bfloat16(r2);
        out_bf16[out_base + my_dim + 3] = hip_bfloat16(r3);
    } else {
        // Split path: write partials to mid_o
        int ob = bid * stride_mb + q_head_global * stride_mh + sid * stride_ms;
        mid_o[ob + my_dim]     = acc0 * il;
        mid_o[ob + my_dim + 1] = acc1 * il;
        mid_o[ob + my_dim + 2] = acc2 * il;
        mid_o[ob + my_dim + 3] = acc3 * il;
        if (my_lane == 0) mid_o[ob + HEAD_DIM] = m_prev + logf(fmaxf(l_prev, 1e-16f));
    }
}

extern "C" void launch_tq_decode_unified(
    const short* q_rot_bf16, const unsigned char* kv_cache,
    const int* block_table, const int* seq_lens,
    const float* centroids, float* mid_o, void* output,
    int stride_qb, int stride_qh,
    int stride_cb, int stride_cp, int stride_ch,
    int stride_bt, int stride_mb, int stride_mh, int stride_ms,
    int stride_ob, int stride_oh,
    int num_kv_heads, int block_size, int num_kv_splits, int kv_group_size,
    float attn_scale, int norm_correction,
    int B, int Hq, hipStream_t stream
) {
    dim3 grid(B, num_kv_heads, num_kv_splits);
    dim3 block(256, 1, 1);
    hipLaunchKernelGGL(tq_decode_unified_hip, grid, block, 0, stream,
        q_rot_bf16, kv_cache, block_table, seq_lens, centroids, mid_o, output,
        stride_qb, stride_qh, stride_cb, stride_cp, stride_ch, stride_bt,
        stride_mb, stride_mh, stride_ms, stride_ob, stride_oh,
        num_kv_heads, block_size, num_kv_splits, kv_group_size,
        attn_scale, norm_correction);
}
