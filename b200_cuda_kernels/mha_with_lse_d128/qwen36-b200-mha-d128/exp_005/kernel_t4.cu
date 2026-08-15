#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define TILE_M 64
#define TILE_N 64
#define HEAD_DIM 128
#define NUM_WARPS (TILE_M / 32)

namespace mha_cuda {

__global__ void flash_attn_fwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S,
    int q_stride_s, int q_stride_h, int q_stride_b,
    int k_stride_s, int k_stride_h, int k_stride_b,
    int v_stride_s, int v_stride_h, int v_stride_b,
    int o_stride_s, int o_stride_h, int o_stride_b,
    int lse_stride_h, int lse_stride_b
) {
    int bx = blockIdx.x;
    int num_q_tiles = (S + TILE_M - 1) / TILE_M;
    int total_tiles = H * num_q_tiles;
    
    int batch_idx = bx / total_tiles;
    int rem = bx % total_tiles;
    int head_idx = rem / num_q_tiles;
    int tile_q = rem % num_q_tiles;

    if (batch_idx >= B || head_idx >= H || tile_q >= num_q_tiles) return;

    int valid_rows = min(TILE_M, S - tile_q * TILE_M);
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    extern __shared__ char smem[];
    __nv_bfloat16* sK = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* sV = sK + TILE_N * HEAD_DIM;
    float* sO = reinterpret_cast<float*>(sV + TILE_N * HEAD_DIM);

    // Initialize shared output buffer to zero
    #pragma unroll
    for (int i = tid; i < TILE_M * HEAD_DIM; i += TILE_M) {
        sO[i] = 0.0f;
    }
    __syncthreads();

    // Each warp owns TILE_M/NUM_WARPS rows = 2 rows
    int rows_per_warp = TILE_M / NUM_WARPS;
    int my_row0 = warp_id * rows_per_warp;
    int my_row1 = my_row0 + 1;
    bool r0_valid = my_row0 < valid_rows;
    bool r1_valid = my_row1 < valid_rows;

    // Load Q rows into registers (2 rows x HEAD_DIM floats)
    float q0[HEAD_DIM];
    float q1[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        q0[d] = 0.0f;
        q1[d] = 0.0f;
    }
    if (r0_valid) {
        const __nv_bfloat16* pq0 = Q + batch_idx * q_stride_b + head_idx * q_stride_h 
                              + (tile_q * TILE_M + my_row0) * q_stride_s;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d += 2) {
            q0[d]   = __bfloat162float(pq0[d]);
            q0[d+1] = __bfloat162float(pq0[d+1]);
        }
    }
    if (r1_valid) {
        const __nv_bfloat16* pq1 = Q + batch_idx * q_stride_b + head_idx * q_stride_h 
                              + (tile_q * TILE_M + my_row1) * q_stride_s;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d += 2) {
            q1[d]   = __bfloat162float(pq1[d]);
            q1[d+1] = __bfloat162float(pq1[d+1]);
        }
    }

    float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F;
    float l0 = 0.0f, l1 = 0.0f;

    float scale = rsqrtf(static_cast<float>(HEAD_DIM));

    // Iterate over KV sequence
    for (int kv_start = 0; kv_start < S; kv_start += TILE_N) {
        // Cooperative load K and V
        int valid_n = min(TILE_N, S - kv_start);
        
        // Load K tile
        #pragma unroll
        for (int i = tid; i < valid_n * HEAD_DIM; i += TILE_M) {
            int rn = i / HEAD_DIM;
            int c = i % HEAD_DIM;
            sK[i] = K + batch_idx * k_stride_b + head_idx * k_stride_h
                 + (kv_start + rn) * k_stride_s + c;
        }
        // Load V tile  
        #pragma unroll
        for (int i = tid; i < valid_n * HEAD_DIM; i += TILE_M) {
            int rn = i / HEAD_DIM;
            int c = i % HEAD_DIM;
            sV[i] = V + batch_idx * v_stride_b + head_idx * v_stride_h
                 + (kv_start + rn) * v_stride_s + c;
        }
        __syncthreads();

        // Compute scores and accumulate
        // Each thread computes contributions for its 2 rows
        #pragma unroll
        for (int nc = 0; nc < valid_n; ++nc) {
            // Compute dot products for rows 0 and 1
            float s0 = 0.0f, s1 = 0.0f;
            const __nv_bfloat16* pk = &sK[nc * HEAD_DIM];
            const __nv_bfloat16* pv = &sV[nc * HEAD_DIM];
            
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d += 4) {
                float k0 = __bfloat162float(pk[d]);
                float k1 = __bfloat162float(pk[d+1]);
                float k2 = __bfloat162float(pk[d+2]);
                float k3 = __bfloat162float(pk[d+3]);
                s0 += q0[d]*k0 + q0[d+1]*k1 + q0[d+2]*k2 + q0[d+3]*k3;
                s1 += q1[d]*k0 + q1[d+1]*k1 + q1[d+2]*k2 + q1[d+3]*k3;
            }
            s0 *= scale;
            s1 *= scale;

            // Softmax numerics
            float n_max0 = fmaxf(m0, s0);
            float n_max1 = fmaxf(m1, s1);
            
            float exp_old0 = expf(m0 - n_max0);
            float exp_old1 = expf(m1 - n_max1);
            float p0 = expf(s0 - n_max0);
            float p1 = expf(s1 - n_max1);

            // Accumulate outputs atomically
            int off0 = my_row0 * HEAD_DIM;
            int off1 = my_row1 * HEAD_DIM;
            if (r0_valid) {
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; d += 2) {
                    float v0 = __bfloat162float(pv[d]);
                    float v1 = __bfloat162float(pv[d+1]);
                    atomicAdd(&sO[off0+d],   exp_old0 * sO[off0+d]   + p0 * v0);
                    atomicAdd(&sO[off0+d+1], exp_old0 * sO[off0+d+1] + p0 * v1);
                }
                atomicAdd(&l0, p0);
                m0 = n_max0;
            }
            if (r1_valid) {
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; d += 2) {
                    float v0 = __bfloat162float(pv[d]);
                    float v1 = __bfloat162float(pv[d+1]);
                    atomicAdd(&sO[off1+d],   exp_old1 * sO[off1+d]   + p1 * v0);
                    atomicAdd(&sO[off1+d+1], exp_old1 * sO[off1+d+1] + p1 * v1);
                }
                atomicAdd(&l1, p1);
                m1 = n_max1;
            }
        }
        __syncthreads();
    }

    // Normalize and store results
    float norm0 = (l0 == 0.0f) ? 0.0f : (1.0f / l0);
    float norm1 = (l1 == 0.0f) ? 0.0f : (1.0f / l1);
    
    __nv_bfloat16* o_base = O + batch_idx * o_stride_b + head_idx * o_stride_h + tile_q * TILE_M * o_stride_s;
    float* lse_base = LSE + batch_idx * lse_stride_b + head_idx * lse_stride_h + tile_q * TILE_M;

    int base_off = my_row0 * o_stride_s;
    if (r0_valid && tid == 0) {
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d += 2) {
            o_base[my_row0 * o_stride_s + d]   = __float2bfloat16(sO[my_row0 * HEAD_DIM + d]   * norm0);
            o_base[my_row0 * o_stride_s + d+1] = __float2bfloat16(sO[my_row0 * HEAD_DIM + d+1] * norm0);
        }
        lse_base[my_row0] = m0 + logf(l0);
    }
    if (r1_valid && tid == 0) {
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d += 2) {
            o_base[my_row1 * o_stride_s + d]   = __float2bfloat16(sO[my_row1 * HEAD_DIM + d]   * norm1);
            o_base[my_row1 * o_stride_s + d+1] = __float2bfloat16(sO[my_row1 * HEAD_DIM + d+1] * norm1);
        }
        lse_base[my_row1] = m1 + logf(l1);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int B = 4, H = 48, D = 128;
    int S = static_cast<int>(Q.size(2));
    
    int q_stride_s = D, q_stride_h = S * D, q_stride_b = H * S * D;
    int k_stride_s = D, k_stride_h = S * D, k_stride_b = H * S * D;
    int v_stride_s = D, v_stride_h = S * D, v_stride_b = H * S * D;
    int o_stride_s = D, o_stride_h = S * D, o_stride_b = H * S * D;
    int lse_stride_h = S, lse_stride_b = H * S;

    int num_tiles_m = (S + TILE_M - 1) / TILE_M;
    dim3 grid(B * H * num_tiles_m);
    dim3 block(TILE_M);
    
    // SMEM: sK(8KB) + sV(8KB) + sO(32KB) = 48KB
    size_t smem_size = 2ULL * TILE_N * HEAD_DIM * sizeof(__nv_bfloat16) + TILE_M * HEAD_DIM * sizeof(float);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    flash_attn_fwd_kernel<<<grid, block, smem_size, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<__nv_bfloat16*>(O.data_ptr()),
        static_cast<float*>(LSE.data_ptr()),
        B, H, S,
        q_stride_s, q_stride_h, q_stride_b,
        k_stride_s, k_stride_h, k_stride_b,
        v_stride_s, v_stride_h, v_stride_b,
        o_stride_s, o_stride_h, o_stride_b,
        lse_stride_h, lse_stride_b
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_cuda::run);

} // namespace mha_cuda