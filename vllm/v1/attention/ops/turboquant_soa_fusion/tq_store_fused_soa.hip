/**
 * Fused TQ Store HIP Kernel — SoA Layout
 *
 * Adapted from tq_store_fused_v1.hip (AoS) to SoA cache layout.
 *
 * SoA layout per block (stride_cache_block bytes):
 *   Data region [0, block_size * H * DATA_BYTES_PER_SLOT):
 *     For each (pos, head): [K_MSE_indices(64B) | V_packed(64B)] = 128B
 *     data_base = block_base + pos * (H * 128) + head * 128
 *
 *   Meta region [meta_region_offset, ...):
 *     For each (head, field, pos): one fp16 value
 *     meta_u16_addr = (block_base + meta_offset) / 2
 *                   + head * (NUM_SOA_FIELDS * block_size) + field * block_size + pos
 *     field 0 = K_norm, field 1 = V_scale, field 2 = V_zero
 *
 * Fuses 5 operations into 1 kernel launch:
 *   1. bf16/fp16 -> fp32 cast
 *   2. L2 norm computation
 *   3. GEMV rotation (x_hat @ PiT)
 *   4. MSE bucketize + pack
 *   5. Value quantize + pack
 *   + metadata (K_norm, V_scale, V_zero) written to SoA meta region
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
#define MSE_BYTES 64
#define VAL_DATA_BYTES 64
#define DATA_BYTES_PER_SLOT (MSE_BYTES + VAL_DATA_BYTES)
#define NUM_SOA_FIELDS 3
#define SOA_K_NORM 0
#define SOA_V_SCALE 1
#define SOA_V_ZERO 2
#define N_WAVES 4
#define TILE_SIZE 32

__device__ __forceinline__ long long soa_data_base(
    int block_num, int page_off, int head_idx,
    int stride_cache_block, int num_kv_heads
) {
    return (long long)block_num * stride_cache_block
         + (long long)page_off * (num_kv_heads * DATA_BYTES_PER_SLOT)
         + (long long)head_idx * DATA_BYTES_PER_SLOT;
}

__device__ __forceinline__ long long soa_meta_u16_addr(
    int block_num, int page_off, int head_idx,
    int stride_cache_block, int block_size,
    int num_kv_heads, int field
) {
    const long long block_base = (long long)block_num * stride_cache_block;
    const long long meta_region_offset =
        (long long)block_size * num_kv_heads * DATA_BYTES_PER_SLOT;
    const long long head_meta_base = (block_base + meta_region_offset) / 2
        + (long long)head_idx * (NUM_SOA_FIELDS * block_size);
    return head_meta_base + (long long)field * block_size + page_off;
}

__global__ __launch_bounds__(256, 2)
void tq_store_fused_soa_kernel(
    const void* __restrict__ key_raw,
    const void* __restrict__ value_raw,
    const float*  __restrict__ PiT,
    const float*  __restrict__ centroids,
    const float*  __restrict__ midpoints,
    unsigned char* __restrict__ kv_cache,
    unsigned short* __restrict__ kv_cache_u16,  // u16 aliased view
    const long long* __restrict__ slot_map,
    int stride_cache_block,
    int H,
    int block_size,
    int NH,
    int kv_dtype    // 0 = bf16, 1 = fp16
) {
    const int wave_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int pid = blockIdx.x * N_WAVES + wave_id;

    // Cannot early-return because __syncthreads() below requires all
    // threads in the block. Use an "active" flag instead.
    const bool active = (pid < NH);

    int token_idx = 0, head_idx = 0;
    long long slot = -1;
    int blk = 0, off = 0;
    long long data_base = 0, knorm_addr = 0, vscale_addr = 0, vzero_addr = 0;

    if (active) {
        token_idx = pid / H;
        head_idx  = pid % H;
        slot = slot_map[token_idx];
        if (slot >= 0) {
            blk = (int)(slot / block_size);
            off = (int)(slot % block_size);
            data_base = soa_data_base(blk, off, head_idx, stride_cache_block, H);
            knorm_addr = soa_meta_u16_addr(blk, off, head_idx,
                stride_cache_block, block_size, H, SOA_K_NORM);
            vscale_addr = soa_meta_u16_addr(blk, off, head_idx,
                stride_cache_block, block_size, H, SOA_V_SCALE);
            vzero_addr = soa_meta_u16_addr(blk, off, head_idx,
                stride_cache_block, block_size, H, SOA_V_ZERO);
        }
    }

    const bool do_work = active && (slot >= 0);

    int d0 = lane_id * 2;
    int d1 = lane_id * 2 + 1;

    // Load centroids + midpoints into registers
    float c_reg[N_CENTROIDS];
    float m_reg[N_MIDPOINTS];
    #pragma unroll
    for (int i = 0; i < N_CENTROIDS; i++) c_reg[i] = centroids[i];
    #pragma unroll
    for (int i = 0; i < N_MIDPOINTS; i++) m_reg[i] = midpoints[i];

    int key_base = pid * D;

    float k0 = 0.0f, k1 = 0.0f;
    if (do_work) {
        if (kv_dtype == 0) {  // bf16
            const __hip_bfloat16* key = (const __hip_bfloat16*)key_raw;
            k0 = __bfloat162float(key[key_base + d0]);
            k1 = __bfloat162float(key[key_base + d1]);
        } else {  // fp16
            const __half* key = (const __half*)key_raw;
            k0 = __half2float(key[key_base + d0]);
            k1 = __half2float(key[key_base + d1]);
        }
    }

    // L2 norm
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

    // GEMV: y = x_hat @ PiT
    float y0 = 0.0f, y1 = 0.0f;

    #pragma unroll 1
    for (int tile = 0; tile < D; tile += TILE_SIZE) {
        const float4* PiT_vec = reinterpret_cast<const float4*>(PiT + tile * D);
        float4* shared_pit_vec = reinterpret_cast<float4*>(shared_pit);

        #pragma unroll 4
        for (int i = threadIdx.x; i < (TILE_SIZE * D) / 4; i += 256) {
            shared_pit_vec[i] = PiT_vec[i];
        }
        __syncthreads();

        if (do_work) {
            #pragma unroll 8
            for (int i = 0; i < TILE_SIZE; i += 4) {
                float xi0 = shared_x[wave_id][tile + i];
                float xi1 = shared_x[wave_id][tile + i + 1];
                float xi2 = shared_x[wave_id][tile + i + 2];
                float xi3 = shared_x[wave_id][tile + i + 3];

                y0 = __fmaf_rn(xi0, shared_pit[i * D + d0], y0);
                y0 = __fmaf_rn(xi1, shared_pit[(i+1) * D + d0], y0);
                y0 = __fmaf_rn(xi2, shared_pit[(i+2) * D + d0], y0);
                y0 = __fmaf_rn(xi3, shared_pit[(i+3) * D + d0], y0);

                y1 = __fmaf_rn(xi0, shared_pit[i * D + d1], y1);
                y1 = __fmaf_rn(xi1, shared_pit[(i+1) * D + d1], y1);
                y1 = __fmaf_rn(xi2, shared_pit[(i+2) * D + d1], y1);
                y1 = __fmaf_rn(xi3, shared_pit[(i+3) * D + d1], y1);
            }
        }
        __syncthreads();
    }

    // Everything below only needs active waves with valid slots
    if (!do_work) return;

    // Bucketize
    int idx0 = 0, idx1 = 0;
    #pragma unroll
    for (int i = 0; i < N_MIDPOINTS; i++) {
        if (y0 >= m_reg[i]) idx0++;
        if (y1 >= m_reg[i]) idx1++;
    }

    // Pack MSE indices and store to SoA DATA region
    unsigned char packed_idx = (unsigned char)((idx0 & 0xF) | ((idx1 & 0xF) << 4));
    kv_cache[data_base + lane_id] = packed_idx;

    // Store K_norm to SoA META region (u16 write)
    if (lane_id == 0) {
        __half vn_h = __float2half(vec_norm);
        kv_cache_u16[knorm_addr] = *reinterpret_cast<unsigned short*>(&vn_h);
    }

    // VALUE PROCESSING
    float v0, v1;
    if (kv_dtype == 0) {
        const __hip_bfloat16* value = (const __hip_bfloat16*)value_raw;
        v0 = __bfloat162float(value[key_base + d0]);
        v1 = __bfloat162float(value[key_base + d1]);
    } else {
        const __half* value = (const __half*)value_raw;
        v0 = __half2float(value[key_base + d0]);
        v1 = __half2float(value[key_base + d1]);
    }

    // Min/max for uniform quant
    float val_min = fminf(v0, v1);
    float val_max = fmaxf(v0, v1);
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

    // Pack V data and store to SoA DATA region (offset by MSE_BYTES)
    unsigned char packed_val = (unsigned char)((q0 & 0xF) | ((q1 & 0xF) << 4));
    kv_cache[data_base + MSE_BYTES + lane_id] = packed_val;

    // Store V_scale and V_zero to SoA META region (u16 writes)
    if (lane_id == 0) {
        __half sc_h = __float2half(v_scale);
        kv_cache_u16[vscale_addr] = *reinterpret_cast<unsigned short*>(&sc_h);

        __half zr_h = __float2half(val_min);
        kv_cache_u16[vzero_addr] = *reinterpret_cast<unsigned short*>(&zr_h);
    }
}

extern "C" void launch_tq_store_fused_soa(
    const void* key,
    const void* value,
    const float* PiT,
    const float* centroids,
    const float* midpoints,
    unsigned char* kv_cache,
    unsigned short* kv_cache_u16,
    const long long* slot_map,
    int stride_cache_block,
    int H,
    int block_size,
    int NH,
    int kv_dtype,
    hipStream_t stream
) {
    dim3 grid((NH + N_WAVES - 1) / N_WAVES);
    dim3 block(256);

    hipLaunchKernelGGL(
        tq_store_fused_soa_kernel,
        grid, block, 0, stream,
        key, value,
        PiT, centroids, midpoints,
        kv_cache, kv_cache_u16, slot_map,
        stride_cache_block,
        H, block_size, NH, kv_dtype
    );
}
