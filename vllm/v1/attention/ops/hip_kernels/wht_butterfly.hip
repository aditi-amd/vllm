/**
 * In-place WHT butterfly kernel for TurboQuant q_rot
 *
 * Replaces: q_rot = q @ PiT  (cuBLAS GEMM, ~10µs per layer)
 * With:     q_rot = WHT_butterfly(q, signs)  (in-kernel, ~2µs per layer)
 *
 * For D=128, Sylvester Hadamard = H_2^7 (7 butterfly passes)
 * PiT = diag(signs) @ H / sqrt(D)
 *
 * Algorithm:
 *   1. Load q[d], multiply by signs[d]
 *   2. 7 butterfly passes (stride = 1, 2, 4, 8, 16, 32, 64)
 *   3. Scale by 1/sqrt(128)
 *   4. Store result as bf16
 *
 * Grid: (B*Hq / HEADS_PER_BLOCK, 1, 1)
 * Block: HEADS_PER_BLOCK * (D/DIMS_PER_THREAD)
 *   = e.g., 4 * 32 = 128 threads
 */

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

#define HEAD_DIM 128
#define DIMS_PER_THREAD 4
#define THREADS_PER_HEAD 32   // HEAD_DIM / DIMS_PER_THREAD
#define HEADS_PER_BLOCK 4     // Tune: process 4 heads per block
#define BLOCK_SIZE (HEADS_PER_BLOCK * THREADS_PER_HEAD)  // 128

__device__ __forceinline__ float bf16_to_float_d(hip_bfloat16 v) {
    return (float)v;
}

/**
 * WHT butterfly kernel — in-place on q, outputs q_rot as bf16
 *
 * Input:  q       [M, D] bf16  (M = B*Hq)
 * Output: q_rot   [M, D] bf16
 * Params: signs   [D] bf16 (per-layer, ±1.0)
 */
extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 8)
void wht_butterfly_bf16(
    const hip_bfloat16* __restrict__ q,       // [M, D] bf16
    hip_bfloat16* __restrict__ q_rot,         // [M, D] bf16
    const hip_bfloat16* __restrict__ signs,   // [D] bf16 (±1.0)
    int M,    // = B * Hq
    int D     // = HEAD_DIM = 128
) {
    const int tid = threadIdx.x;
    const int head_in_block = tid / THREADS_PER_HEAD;
    const int lane = tid % THREADS_PER_HEAD;
    const int global_head = blockIdx.x * HEADS_PER_BLOCK + head_in_block;

    if (global_head >= M) return;

    const int base_dim = lane * DIMS_PER_THREAD;  // 0, 4, 8, ..., 124
    const int q_base = global_head * D;

    // ── Step 1: Load q and apply signs ──
    float v0 = bf16_to_float_d(q[q_base + base_dim + 0]) * bf16_to_float_d(signs[base_dim + 0]);
    float v1 = bf16_to_float_d(q[q_base + base_dim + 1]) * bf16_to_float_d(signs[base_dim + 1]);
    float v2 = bf16_to_float_d(q[q_base + base_dim + 2]) * bf16_to_float_d(signs[base_dim + 2]);
    float v3 = bf16_to_float_d(q[q_base + base_dim + 3]) * bf16_to_float_d(signs[base_dim + 3]);

    // ── Step 2: 7 butterfly passes ──
    // Each pass: for pairs separated by 'stride':
    //   a, b = a+b, a-b
    //
    // Thread layout: thread t holds dims [4t, 4t+1, 4t+2, 4t+3]
    //
    // Pass 1 (stride=1): pairs (0,1),(2,3) — within thread
    {
        float a0 = v0 + v1, a1 = v0 - v1;
        float a2 = v2 + v3, a3 = v2 - v3;
        v0 = a0; v1 = a1; v2 = a2; v3 = a3;
    }

    // Pass 2 (stride=2): pairs (0,2),(1,3) — within thread
    {
        float a0 = v0 + v2, a2 = v0 - v2;
        float a1 = v1 + v3, a3 = v1 - v3;
        v0 = a0; v1 = a1; v2 = a2; v3 = a3;
    }

    // Pass 3 (stride=4): thread t ↔ thread t^1
    // Thread t holds dims [4t..4t+3], stride=4 means pair dim d with d+4
    // dim 4t+k pairs with 4(t^1)+k
    // Use __shfl_xor with delta=1 within the 32-thread head group
    {
        // We need to compute within the wavefront.
        // On AMD, wavefront is 64 wide, but our head group is 32 threads.
        // We operate relative to lane within the head group.
        // shfl_xor on the full wavefront: need to XOR with offset that maps
        // to the correct partner.
        //
        // head_in_block * THREADS_PER_HEAD + lane → wavefront lane
        // Partner for stride-4 butterfly: lane ^ 1
        // Wavefront XOR: (head_in_block * THREADS_PER_HEAD + lane) ^ 1
        //   = head_in_block * THREADS_PER_HEAD + (lane ^ 1)  [since THREADS_PER_HEAD=32, bit 0 only in lane]
        // So __shfl_xor with delta=1 across full wavefront works IF
        // head_in_block doesn't affect bit 0 of wavefront lane.
        //
        // wavefront_lane = head_in_block * 32 + lane
        // For HEADS_PER_BLOCK=4: head_in_block ∈ {0,1,2,3}
        // head 0: lanes 0-31, head 1: lanes 32-63 → fits in one wavefront
        // head 2,3: would be lanes 64-127 → second wavefront!
        //
        // Actually BLOCK_SIZE=128, AMD wavefront=64, so we have 2 wavefronts:
        // Wave 0: threads 0-63  (heads 0,1)
        // Wave 1: threads 64-127 (heads 2,3)
        //
        // Within each wavefront: head groups at [0:31] and [32:63]
        // XOR with 1 stays within the same head group ✓

        float p0 = __shfl_xor(v0, 1, 64);
        float p1 = __shfl_xor(v1, 1, 64);
        float p2 = __shfl_xor(v2, 1, 64);
        float p3 = __shfl_xor(v3, 1, 64);

        // If lane is even (lane & 1 == 0): I have lower 4 dims, partner has upper 4
        // I compute: new_v[k] = v[k] + partner_v[k]   (lower half of butterfly)
        // Partner:   new_v[k] = partner_v[k] - v[k]     (upper half — but partner computes too)
        //
        // Actually for Hadamard butterfly:
        // If bit 0 of lane is 0: v[k] = v[k] + p[k]
        // If bit 0 of lane is 1: v[k] = p[k] - v[k]
        if (lane & 1) {
            v0 = p0 - v0; v1 = p1 - v1; v2 = p2 - v2; v3 = p3 - v3;
        } else {
            v0 = v0 + p0; v1 = v1 + p1; v2 = v2 + p2; v3 = v3 + p3;
        }
    }

    // Pass 4 (stride=8): lane ^ 2
    {
        float p0 = __shfl_xor(v0, 2, 64);
        float p1 = __shfl_xor(v1, 2, 64);
        float p2 = __shfl_xor(v2, 2, 64);
        float p3 = __shfl_xor(v3, 2, 64);
        if (lane & 2) {
            v0 = p0 - v0; v1 = p1 - v1; v2 = p2 - v2; v3 = p3 - v3;
        } else {
            v0 = v0 + p0; v1 = v1 + p1; v2 = v2 + p2; v3 = v3 + p3;
        }
    }

    // Pass 5 (stride=16): lane ^ 4
    {
        float p0 = __shfl_xor(v0, 4, 64);
        float p1 = __shfl_xor(v1, 4, 64);
        float p2 = __shfl_xor(v2, 4, 64);
        float p3 = __shfl_xor(v3, 4, 64);
        if (lane & 4) {
            v0 = p0 - v0; v1 = p1 - v1; v2 = p2 - v2; v3 = p3 - v3;
        } else {
            v0 = v0 + p0; v1 = v1 + p1; v2 = v2 + p2; v3 = v3 + p3;
        }
    }

    // Pass 6 (stride=32): lane ^ 8
    {
        float p0 = __shfl_xor(v0, 8, 64);
        float p1 = __shfl_xor(v1, 8, 64);
        float p2 = __shfl_xor(v2, 8, 64);
        float p3 = __shfl_xor(v3, 8, 64);
        if (lane & 8) {
            v0 = p0 - v0; v1 = p1 - v1; v2 = p2 - v2; v3 = p3 - v3;
        } else {
            v0 = v0 + p0; v1 = v1 + p1; v2 = v2 + p2; v3 = v3 + p3;
        }
    }

    // Pass 7 (stride=64): lane ^ 16
    {
        float p0 = __shfl_xor(v0, 16, 64);
        float p1 = __shfl_xor(v1, 16, 64);
        float p2 = __shfl_xor(v2, 16, 64);
        float p3 = __shfl_xor(v3, 16, 64);
        if (lane & 16) {
            v0 = p0 - v0; v1 = p1 - v1; v2 = p2 - v2; v3 = p3 - v3;
        } else {
            v0 = v0 + p0; v1 = v1 + p1; v2 = v2 + p2; v3 = v3 + p3;
        }
    }

    // ── Step 3: Scale by 1/sqrt(128) ──
    // Already included in the Hadamard matrix normalization
    // H_norm = H / sqrt(D), so we need to apply 1/sqrt(128) = 1/11.3137
    const float inv_sqrt_d = 0.08838834764831843f;  // 1/sqrt(128)
    v0 *= inv_sqrt_d;
    v1 *= inv_sqrt_d;
    v2 *= inv_sqrt_d;
    v3 *= inv_sqrt_d;

    // ── Step 4: Store as bf16 ──
    q_rot[q_base + base_dim + 0] = hip_bfloat16(v0);
    q_rot[q_base + base_dim + 1] = hip_bfloat16(v1);
    q_rot[q_base + base_dim + 2] = hip_bfloat16(v2);
    q_rot[q_base + base_dim + 3] = hip_bfloat16(v3);
}

/**
 * Launcher
 */
extern "C" void launch_wht_butterfly(
    const hip_bfloat16* q,
    hip_bfloat16* q_rot,
    const hip_bfloat16* signs,
    int M,   // B * Hq
    int D,   // HEAD_DIM
    hipStream_t stream
) {
    int grid_x = (M + HEADS_PER_BLOCK - 1) / HEADS_PER_BLOCK;
    dim3 grid(grid_x);
    dim3 block(BLOCK_SIZE);
    hipLaunchKernelGGL(wht_butterfly_bf16, grid, block, 0, stream,
        q, q_rot, signs, M, D);
}
