/**
 * Fused TQ Store HIP Kernel (v14 - Final optimized)
 *
 * For all NH: Use 4 wavefronts per block (256 threads) with LDS PiT
 * This provides consistent good performance across all sizes
 *
 * Supported configuration (turboquant_4bit_nc):
 *   HEAD_DIM=128, MSE=4bit (64B), VALUE=4bit (64B), KPS=68
 *   block_size: any power-of-2 (16, 32, 64, 128)
 *   key/value: bf16 or fp16 (selected via kv_dtype flag)
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/amd_detail/amd_hip_bf16.h>

#define WARP_SIZE 64

__device__ __forceinline__ float wave_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor(val, offset, WARP_SIZE);
    }
    return val;
}

#define D 128
#define N_CENTROIDS 16
#define N_MIDPOINTS 15
#define NORM_OFFSET 64
#define VAL_OFFSET 68
#define SCALE_OFFSET 132
#define N_WAVES 4
#define TILE_SIZE 32

__global__ __launch_bounds__(256, 2)
void tq_store_fused_kernel(
    const void* __restrict__ key_raw,
    const void* __restrict__ value_raw,
    const float*  __restrict__ PiT,
    const float*  __restrict__ centroids,
    const float*  __restrict__ midpoints,
    unsigned char* __restrict__ kv_cache,
    const long long* __restrict__ slot_map,
    int stride_block,
    int stride_pos,
    int stride_head,
    int H,
    int block_size,
    int NH,
    int kv_dtype    // 0 = bf16, 1 = fp16
) {
    const int wave_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int pid = blockIdx.x * N_WAVES + wave_id;
    
    if (pid >= NH) return;
    
    const int token_idx = pid / H;
    const int head_idx  = pid % H;

    long long slot = slot_map[token_idx];
    if (slot < 0) return;

    int blk = (int)(slot / block_size);
    int off = (int)(slot % block_size);
    long long slot_base = (long long)blk * stride_block 
                        + (long long)off * stride_pos 
                        + (long long)head_idx * stride_head;

    int d0 = lane_id * 2;
    int d1 = lane_id * 2 + 1;

    // Load centroids + midpoints into registers
    float c_reg[N_CENTROIDS];
    float m_reg[N_MIDPOINTS];
    #pragma unroll
    for (int i = 0; i < N_CENTROIDS; i++) {
        c_reg[i] = centroids[i];
    }
    #pragma unroll
    for (int i = 0; i < N_MIDPOINTS; i++) {
        m_reg[i] = midpoints[i];
    }

    int key_base = pid * D;

    float k0, k1;
    if (kv_dtype == 0) {  // bf16
        const __hip_bfloat16* key = (const __hip_bfloat16*)key_raw;
        k0 = __bfloat162float(key[key_base + d0]);
        k1 = __bfloat162float(key[key_base + d1]);
    } else {  // fp16
        const __half* key = (const __half*)key_raw;
        k0 = __half2float(key[key_base + d0]);
        k1 = __half2float(key[key_base + d1]);
    }

    float partial_sq = k0 * k0 + k1 * k1;
    float sq_sum = wave_sum(partial_sq);
    float vec_norm = sqrtf(sq_sum);
    float inv_norm = 1.0f / (vec_norm + 1e-8f);

    float x0 = k0 * inv_norm;
    float x1 = k1 * inv_norm;

    __shared__ float shared_x[N_WAVES][D];
    __shared__ float shared_pit[TILE_SIZE * D];
    
    shared_x[wave_id][d0] = x0;
    shared_x[wave_id][d1] = x1;
    __syncthreads();

    float y0 = 0.0f, y1 = 0.0f;
    
    // Process PiT in 32-row tiles with vectorized loads
    #pragma unroll 1
    for (int tile = 0; tile < D; tile += TILE_SIZE) {
        // Vectorized load: 256 threads, 32*128 = 4096 floats = 1024 float4s
        const float4* PiT_vec = reinterpret_cast<const float4*>(PiT + tile * D);
        float4* shared_pit_vec = reinterpret_cast<float4*>(shared_pit);
        
        #pragma unroll 4
        for (int i = threadIdx.x; i < (TILE_SIZE * D) / 4; i += 256) {
            shared_pit_vec[i] = PiT_vec[i];
        }
        __syncthreads();
        
        // Compute partial GEMV with 4x unrolling
        #pragma unroll 8
        for (int i = 0; i < TILE_SIZE; i += 4) {
            float xi0 = shared_x[wave_id][tile + i];
            float xi1 = shared_x[wave_id][tile + i + 1];
            float xi2 = shared_x[wave_id][tile + i + 2];
            float xi3 = shared_x[wave_id][tile + i + 3];
            
            float p00 = shared_pit[i * D + d0];
            float p10 = shared_pit[(i + 1) * D + d0];
            float p20 = shared_pit[(i + 2) * D + d0];
            float p30 = shared_pit[(i + 3) * D + d0];
            
            float p01 = shared_pit[i * D + d1];
            float p11 = shared_pit[(i + 1) * D + d1];
            float p21 = shared_pit[(i + 2) * D + d1];
            float p31 = shared_pit[(i + 3) * D + d1];
            
            y0 = __fmaf_rn(xi0, p00, y0);
            y0 = __fmaf_rn(xi1, p10, y0);
            y0 = __fmaf_rn(xi2, p20, y0);
            y0 = __fmaf_rn(xi3, p30, y0);
            
            y1 = __fmaf_rn(xi0, p01, y1);
            y1 = __fmaf_rn(xi1, p11, y1);
            y1 = __fmaf_rn(xi2, p21, y1);
            y1 = __fmaf_rn(xi3, p31, y1);
        }
        __syncthreads();
    }

    // Bucketize
    int idx0 = 0, idx1 = 0;
    #pragma unroll
    for (int i = 0; i < N_MIDPOINTS; i++) {
        if (y0 >= m_reg[i]) idx0++;
        if (y1 >= m_reg[i]) idx1++;
    }

    float res0 = y0 - c_reg[idx0];
    float res1 = y1 - c_reg[idx1];
    float partial_res_sq = res0 * res0 + res1 * res1;
    float res_sq_sum = wave_sum(partial_res_sq);
    float gamma = sqrtf(res_sq_sum);

    unsigned char packed_idx = (unsigned char)((idx0 & 0xF) | ((idx1 & 0xF) << 4));
    kv_cache[slot_base + lane_id] = packed_idx;

    if (lane_id == 0) {
        __half vn_h = __float2half(vec_norm);
        unsigned short vn_u = *reinterpret_cast<unsigned short*>(&vn_h);
        kv_cache[slot_base + NORM_OFFSET + 0] = (unsigned char)(vn_u & 0xFF);
        kv_cache[slot_base + NORM_OFFSET + 1] = (unsigned char)((vn_u >> 8) & 0xFF);

        __half gm_h = __float2half(gamma);
        unsigned short gm_u = *reinterpret_cast<unsigned short*>(&gm_h);
        kv_cache[slot_base + NORM_OFFSET + 2] = (unsigned char)(gm_u & 0xFF);
        kv_cache[slot_base + NORM_OFFSET + 3] = (unsigned char)((gm_u >> 8) & 0xFF);
    }

    // VALUE PROCESSING
    float v0, v1;
    if (kv_dtype == 0) {  // bf16
        const __hip_bfloat16* value = (const __hip_bfloat16*)value_raw;
        v0 = __bfloat162float(value[key_base + d0]);
        v1 = __bfloat162float(value[key_base + d1]);
    } else {  // fp16
        const __half* value = (const __half*)value_raw;
        v0 = __half2float(value[key_base + d0]);
        v1 = __half2float(value[key_base + d1]);
    }

    float thread_min = fminf(v0, v1);
    float thread_max = fmaxf(v0, v1);

    float val_min = thread_min;
    float val_max = thread_max;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val_min = fminf(val_min, __shfl_xor(val_min, offset, WARP_SIZE));
        val_max = fmaxf(val_max, __shfl_xor(val_max, offset, WARP_SIZE));
    }

    float v_scale = (val_max - val_min) / 15.0f;
    v_scale = (v_scale > 1e-8f) ? v_scale : 1e-8f;

    int q0 = (int)(((v0 - val_min) / v_scale) + 0.5f);
    int q1 = (int)(((v1 - val_min) / v_scale) + 0.5f);
    q0 = max(0, min(15, q0));
    q1 = max(0, min(15, q1));

    unsigned char packed_val = (unsigned char)((q0 & 0xF) | ((q1 & 0xF) << 4));
    kv_cache[slot_base + VAL_OFFSET + lane_id] = packed_val;

    if (lane_id == 0) {
        __half sc_h = __float2half(v_scale);
        unsigned short sc_u = *reinterpret_cast<unsigned short*>(&sc_h);
        kv_cache[slot_base + SCALE_OFFSET + 0] = (unsigned char)(sc_u & 0xFF);
        kv_cache[slot_base + SCALE_OFFSET + 1] = (unsigned char)((sc_u >> 8) & 0xFF);

        __half zr_h = __float2half(val_min);
        unsigned short zr_u = *reinterpret_cast<unsigned short*>(&zr_h);
        kv_cache[slot_base + SCALE_OFFSET + 2] = (unsigned char)(zr_u & 0xFF);
        kv_cache[slot_base + SCALE_OFFSET + 3] = (unsigned char)((zr_u >> 8) & 0xFF);
    }
}

extern "C" void launch_tq_store_fused(
    const void* key,
    const void* value,
    const float* PiT,
    const float* centroids,
    const float* midpoints,
    unsigned char* kv_cache,
    const long long* slot_map,
    int stride_block,
    int stride_pos,
    int stride_head,
    int H,
    int block_size,
    int NH,
    int kv_dtype,       // 0 = bf16, 1 = fp16
    hipStream_t stream
) {
    dim3 grid((NH + N_WAVES - 1) / N_WAVES);
    dim3 block(256);

    hipLaunchKernelGGL(
        tq_store_fused_kernel,
        grid, block, 0, stream,
        key, value,
        PiT, centroids, midpoints,
        kv_cache, slot_map,
        stride_block, stride_pos, stride_head,
        H, block_size, NH, kv_dtype
    );
}
