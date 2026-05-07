/**
 * TQ Decode Stage1 — V110: V109 + alignment-safe memory reads
 *
 * Fix: gfx950 rounds unaligned uint32 reads to 4-byte boundary.
 * All multi-byte reads from kv_cache (uint8 array) now use byte-level
 * loads to avoid alignment issues with KPS=66 (non-power-of-2 offset).
 *
 * Fixed for current vllm TurboQuant slot format:
 *   turboquant_4bit_nc: KPS=68, MSE_BYTES=64, VAL_DATA_BYTES=64
 *   Layout per slot: [64B MSE indices | 2B vec_norm(fp16) | 2B gamma(fp16) | 64B value_data | 2B scale | 2B zero]
 *
 * Architecture: 256 threads = 4 wavefronts
 *   Wave 0 (tid 0-63):   MFMA score for 8 Q heads × BLOCK_N tokens
 *   All 256 threads:      Value accumulation, each thread handles 1 head
 *
 * Grid: (B, Hk, splits)
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

// ── Layout constants matching vllm turboquant_4bit_nc, head_dim=128 ──
#define HEAD_DIM 128
#define MSE_BYTES 64       // ceil(128 * 4 / 8)
#define N_CENTROIDS 16     // 2^4
#define KPS 68             // MSE_BYTES + 4 (2B vec_norm + 2B gamma)
#define VAL_DATA_BYTES 64  // ceil(128 * 4 / 8)
#define BLOCK_N 16
#define KV_GROUP_SIZE 8
#define THREADS 256
#define THREADS_PER_HEAD 32  // 256 / 8
#define DIMS_PER_THREAD 4    // 128 / 32

typedef short bf16x4_t __attribute__((ext_vector_type(4)));
typedef float float4_t __attribute__((ext_vector_type(4)));

__device__ __forceinline__ float bf16_to_float(short v) {
    union { float f; unsigned u; } r;
    r.u = ((unsigned)(unsigned short)v) << 16;
    return r.f;
}

extern "C" __global__ __launch_bounds__(256, 2)
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

    const int my_head = tid / THREADS_PER_HEAD;  // 0-7
    const int my_lane = tid % THREADS_PER_HEAD;  // 0-31
    const int my_dim = my_lane * DIMS_PER_THREAD; // base dim for this thread
    const int q_head_global = kv_hid * kv_group_size + my_head;

    // ── LDS ──
    __shared__ unsigned int s_pair_lut[256]; // pair bf16 LUT: [lo_idx * 16 + hi_idx]
    __shared__ float s_scores[KV_GROUP_SIZE][BLOCK_N];
    __shared__ float s_norms[BLOCK_N];
    __shared__ float s_csq[BLOCK_N];
    __shared__ long long s_slot_base[BLOCK_N];

    // Build pairwise bf16 LUT in shared memory
    // pair_lut[lo*16+hi] = bf16(centroids[lo]) | (bf16(centroids[hi]) << 16)
    if (tid < 256) {
        int lo = tid & 0xF, hi = (tid >> 4) & 0xF;
        union { float f; unsigned u; } clo, chi;
        clo.f = centroids[lo]; chi.f = centroids[hi];
        unsigned short bf16_lo = (unsigned short)(clo.u >> 16);
        unsigned short bf16_hi = (unsigned short)(chi.u >> 16);
        s_pair_lut[tid] = (unsigned int)bf16_lo | ((unsigned int)bf16_hi << 16);
    }
    __syncthreads();

    // Online softmax accumulators
    float m_prev = -1e30f, l_prev = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    const int bt_base = bid * stride_bt;
    const long long kv_head_offset = (long long)kv_hid * stride_ch;

    // ═══ Main loop ═══
    for (int base_n = split_start; base_n < split_end; base_n += BLOCK_N) {
        const int actual_n = min(BLOCK_N, split_end - base_n);

        // ── Precompute slot bases + load norms ──
        if (tid < BLOCK_N) {
            if (tid < actual_n) {
                int kv_pos = base_n + tid;
                int pi = kv_pos / block_size;
                int po = kv_pos % block_size;
                int bn = block_table[bt_base + pi];
                long long sb = (long long)bn * stride_cb + (long long)po * stride_cp + kv_head_offset;
                s_slot_base[tid] = sb;
                // norm is at offset MSE_BYTES (= 64) from slot base, stored as fp16
                // Byte-safe read (offset 64 is 2-byte aligned, but be safe)
                unsigned short nu = (unsigned short)kv_cache[sb + MSE_BYTES]
                                  | ((unsigned short)kv_cache[sb + MSE_BYTES + 1] << 8);
                s_norms[tid] = __half2float(*reinterpret_cast<const __half*>(&nu));
            } else {
                s_slot_base[tid] = 0;
                s_norms[tid] = 1.0f;
            }
        }

        // ── Phase 1: MFMA score computation (wave 0 only) ──
        if (tid < 64) {
            float4_t c_acc = {0, 0, 0, 0};
            float my_nsq = 0.0f;

            // HEAD_DIM=128, 16 dims per MFMA block → 8 blocks
            for (int kb = 0; kb < HEAD_DIM / 16; kb++) {
                int m = tid % 16;   // row index within MFMA tile (Q head or token)
                int k_grp = tid / 16; // which group of 4 elements within 16-dim block

                // Load Q fragment: [m, kb*16+k_grp*4 .. +3] as bf16
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

                // Load K fragment via pair LUT: dequant 4-bit MSE indices to bf16 centroids
                int tok = tid % 16;
                bf16x4_t b_frag;
                if (tok < actual_n) {
                    long long sb = s_slot_base[tok];
                    // 4-bit MSE indices are packed 2 per byte
                    // For dim range [kb*16+k_grp*4 .. +3], byte offset = dim/2
                    int gd = kb * 16 + k_grp * 4;
                    int bo = gd / 2; // byte offset within MSE region
                    unsigned char b0 = kv_cache[sb + bo];
                    unsigned char b1 = kv_cache[sb + bo + 1];
                    // Pair LUT: b0 = lo_idx | (hi_idx << 4)
                    unsigned int pk0 = s_pair_lut[b0];
                    unsigned int pk1 = s_pair_lut[b1];
                    b_frag[0] = (short)(pk0 & 0xFFFF);        // centroids[lo0] as bf16
                    b_frag[1] = (short)(pk0 >> 16);            // centroids[hi0] as bf16
                    b_frag[2] = (short)(pk1 & 0xFFFF);        // centroids[lo1] as bf16
                    b_frag[3] = (short)(pk1 >> 16);            // centroids[hi1] as bf16
                } else {
                    b_frag[0]=b_frag[1]=b_frag[2]=b_frag[3]=(short)0;
                }

                // Accumulate norm² for norm_correction
                if (norm_correction) {
                    float f0 = bf16_to_float(b_frag[0]);
                    float f1 = bf16_to_float(b_frag[1]);
                    float f2 = bf16_to_float(b_frag[2]);
                    float f3 = bf16_to_float(b_frag[3]);
                    my_nsq += f0*f0 + f1*f1 + f2*f2 + f3*f3;
                }

                // MFMA: C += A × B^T, 16×16×16 bf16 → fp32
                c_acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_frag, b_frag, c_acc, 0, 0, 0);
            }

            // Reduce norm² across k_grp lanes (warp-level)
            if (norm_correction) {
                my_nsq += __shfl_xor(my_nsq, 16, 64);
                my_nsq += __shfl_xor(my_nsq, 32, 64);
                int tok = tid % 16;
                if (tid / 16 == 0) {
                    s_csq[tok] = (tok < actual_n) ? my_nsq : 1.0f;
                }
            }

            // Write scores to shared memory
            // MFMA output layout: c_acc[i] = C[(tid/16)*4+i, tid%16]
            //   where C[head, tok] = dot(Q[head], K[tok])
            int mfma_tok = tid % 16;
            int mfma_head_base = (tid / 16) * 4;
            for (int i = 0; i < 4; i++) {
                int hh = mfma_head_base + i;
                if (hh < KV_GROUP_SIZE) {
                    s_scores[hh][mfma_tok] = (mfma_tok < actual_n) ? c_acc[i] * attn_scale : -1e30f;
                }
            }
        }
        __syncthreads();

        // Apply norms × scores (and norm_correction if enabled)
        // Use first KV_GROUP_SIZE * BLOCK_N threads (= 128 threads ≤ 256)
        if (tid < KV_GROUP_SIZE * BLOCK_N) {
            int h = tid / BLOCK_N;
            int tk = tid % BLOCK_N;
            if (tk < actual_n) {
                float score = s_scores[h][tk];
                if (norm_correction) score *= rsqrtf(s_csq[tk] + 1e-16f);
                s_scores[h][tk] = score * s_norms[tk];
            }
            // padding slots already set to -inf in score write above
        }
        __syncthreads();

        // ── Phase 2: Softmax + Value accumulation (all 256 threads) ──
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

        // Dequantize 4-bit values and accumulate
        // Value layout: slot_base + KPS → [VAL_DATA_BYTES bytes value | 2B scale | 2B zero]
        const int byte_idx = my_lane * 2; // 2 bytes = 4 nibbles = 4 values
        for (int n = 0; n < actual_n; n++) {
            long long sb = s_slot_base[n];
            long long vb = sb + KPS;             // value data start
            long long scb = vb + VAL_DATA_BYTES; // scale+zero start

            // Load scale + zero byte-by-byte (alignment-safe for gfx950)
            unsigned short scale_u16 = (unsigned short)kv_cache[scb]
                                     | ((unsigned short)kv_cache[scb + 1] << 8);
            unsigned short zero_u16  = (unsigned short)kv_cache[scb + 2]
                                     | ((unsigned short)kv_cache[scb + 3] << 8);
            float vs = __half2float(*reinterpret_cast<const __half*>(&scale_u16));
            float vz = __half2float(*reinterpret_cast<const __half*>(&zero_u16));

            // Load 2 bytes of packed 4-bit values → 4 values (byte-safe)
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

    // ── Output ──
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
