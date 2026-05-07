/**
 * TQ Decode Stage1 — V125: v123 + reduced __syncthreads
 *
 * Optimization: merge sync barriers where possible.
 * v123 has 6 syncs per iteration:
 *   1. After slot-base precompute
 *   2. After K prefetch  
 *   3. After V dequant (implicit, combined with K prefetch)
 *   Actually: after slot_base, then after K+V prefetch+dequant (1 sync),
 *   then after Phase1 MFMA, then after score reduce, then after Value MFMA,
 *   then at end of loop.
 *
 * Can merge: slot_base precompute only uses tid<16, then K prefetch uses all 256.
 *   If we have tid<16 compute slot_base AND also participate in K prefetch,
 *   we can remove one sync. But slot_base is needed by K prefetch...
 *
 * Actually the sync structure is:
 *   sync1: after slot_base (needed before K/V prefetch can read s_slot_base)
 *   sync2: after K prefetch + V dequant (needed before Phase1 reads s_key_data, s_val_bf16)
 *   sync3: after Phase1 MFMA partial writes (needed before reduction)
 *   sync4: after score reduction (needed before softmax reads s_scores)
 *   sync5: after Value MFMA writes to s_pv_result (needed before 256 threads read)
 *   sync6: end of loop (needed before next iter overwrites LDS)
 *
 * Can merge sync5+sync6 by doing the accumulate read immediately after MFMA and 
 * then sync once before next iteration. But sync5 is needed for correctness...
 *
 * Alternative: merge sync3 and sync4. The reduction uses tid<128, so if reduction
 * runs immediately in the same sync region, we just need one sync after both.
 * But Phase1 writes s_partial_scores and reduction reads them — they must be separated.
 *
 * Actually, we can eliminate sync4 if we fold the norm-correction + norms multiplication
 * into Phase 2 softmax computation, using only the head's own scores. Let me check:
 * After sync3, only tid<128 writes s_scores. Then all 256 threads read s_scores in softmax.
 * So sync4 is needed.
 *
 * Best opportunity: merge sync5+sync6. After Value MFMA writes to s_pv_result,
 * we need all threads to read from it. If we do sync, read, then at the end
 * we can go straight to next iteration without another sync, BECAUSE the next
 * iteration's first write is to s_slot_base (tid<16 only), and the reads of
 * s_pv_result are done by all 256 threads. As long as the reader threads finish
 * before the next iteration's writer threads start writing s_slot_base...
 * But s_slot_base doesn't overlap with s_pv_result in LDS, so this is safe!
 * We just need to make sure that all threads have finished reading their
 * accumulators from s_pv_result before s_pv_result gets overwritten.
 * s_pv_result gets overwritten during Value MFMA which happens AFTER sync4.
 * So removing sync6 is safe!
 *
 * Also: can we overlap V dequant with K prefetch? Both write to different LDS
 * regions. V dequant writes s_val_bf16, K prefetch writes s_key_data. They don't
 * conflict. But both read from s_slot_base, which needs sync1 to be done first.
 * So they CAN run concurrently after sync1, and we need only sync2 after both.
 * This is already the case in v123! Let me check...
 *
 * In v123, K prefetch and V dequant are sequential, not concurrent. If I interleave:
 * All 256 threads do K prefetch (1024B = 4B per thread), then same 256 threads
 * do V dequant (1 uint32 load + 8 bf16 writes per thread). These are already
 * sequential within the same sync region (sync1 to sync2). No sync between them.
 * So v123 already has only 1 sync between K+V, not 2.
 *
 * Actual sync count in v123: 
 *   1. After slot_base
 *   2. After K+V prefetch/dequant combined
 *   3. After Phase1 MFMA
 *   4. After score reduction (includes norm_correction)
 *   5. After Value MFMA
 *   6. End of loop → CAN BE REMOVED
 *
 * Let's remove sync6 and verify.
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

    const int wave_id = tid / WAVE_SIZE;
    const int lane_id = tid % WAVE_SIZE;
    const int my_head = tid / THREADS_PER_HEAD;
    const int my_lane = tid % THREADS_PER_HEAD;
    const int my_dim = my_lane * DIMS_PER_THREAD;
    const int q_head_global = kv_hid * kv_group_size + my_head;

    __shared__ unsigned int s_pair_lut[256];
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

        // sync1: slot-base precompute
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
        __syncthreads(); // sync1

        // K prefetch + V dequant (no sync between them — different LDS regions)
        {
            int load_tok = tid / 16;
            int load_off = tid % 16;
            if (load_tok < BLOCK_N) {
                int byte_off = load_off * 4;
                if (load_tok < actual_n) {
                    long long sb = s_slot_base[load_tok];
                    s_key_data[load_tok * MSE_BYTES + byte_off]     = kv_cache[sb + byte_off];
                    s_key_data[load_tok * MSE_BYTES + byte_off + 1] = kv_cache[sb + byte_off + 1];
                    s_key_data[load_tok * MSE_BYTES + byte_off + 2] = kv_cache[sb + byte_off + 2];
                    s_key_data[load_tok * MSE_BYTES + byte_off + 3] = kv_cache[sb + byte_off + 3];
                } else {
                    s_key_data[load_tok * MSE_BYTES + byte_off] = 0;
                    s_key_data[load_tok * MSE_BYTES + byte_off + 1] = 0;
                    s_key_data[load_tok * MSE_BYTES + byte_off + 2] = 0;
                    s_key_data[load_tok * MSE_BYTES + byte_off + 3] = 0;
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
        __syncthreads(); // sync2

        // Phase 1: MFMA scoring
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
        __syncthreads(); // sync3

        // Reduce
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
        __syncthreads(); // sync4

        // Softmax
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

        // Value MFMA
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
        __syncthreads(); // sync5

        acc0 += s_pv_result[my_head*HEAD_DIM+my_dim];
        acc1 += s_pv_result[my_head*HEAD_DIM+my_dim+1];
        acc2 += s_pv_result[my_head*HEAD_DIM+my_dim+2];
        acc3 += s_pv_result[my_head*HEAD_DIM+my_dim+3];
        l_prev = l_prev * rescale + new_l;
        m_prev = new_max;
        // NO sync6 here — next iteration's slot_base precompute (tid<16) doesn't
        // conflict with s_pv_result reads (done by all threads above this point)
    }

    float il = 1.0f / fmaxf(l_prev, 1e-16f);
    int ob = bid * stride_mb + q_head_global * stride_mh + sid * stride_ms;
    mid_o[ob+my_dim]   = acc0*il;
    mid_o[ob+my_dim+1] = acc1*il;
    mid_o[ob+my_dim+2] = acc2*il;
    mid_o[ob+my_dim+3] = acc3*il;
    if (my_lane == 0) mid_o[ob+HEAD_DIM] = m_prev + logf(fmaxf(l_prev, 1e-16f));
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
