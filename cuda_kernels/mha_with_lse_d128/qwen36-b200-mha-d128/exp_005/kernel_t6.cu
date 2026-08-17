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
#define THREADS_PER_TILE 128

namespace mha_cuda {

__device__ __forceinline__ float bfloat16_to_float(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16(float x) {
    return __float2bfloat16(x);
}

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
    int lane_id = tid % 32;

    extern __shared__ char smem[];
    // Shared memory layout: sK[TILE_N][HEAD_DIM] bf16 + sV[TILE_N][HEAD_DIM] bf16
    __nv_bfloat16* sK = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* sV = sK + TILE_N * HEAD_DIM;

    // Compute starting index for cooperative K/V loading (THREADS_PER_TILE=128 threads)
    // Each thread loads elements with stride THREADS_PER_TILE
    int elem_per_thread_kv = ((TILE_N * HEAD_DIM) + THREADS_PER_TILE - 1) / THREADS_PER_TILE;

    // Online softmax state per row owned by this thread
    // Thread owns up to (valid_rows / THREADS_PER_TILE) rows... 
    // With 128 threads and 64 rows, each thread owns 0.5 rows => we use first 64 threads for rows
    // Actually let's simplify: tid < valid_rows computes for that single row
    
    // We will have 256 threads but only use up to min(valid_rows, 64) for row computation
    // The rest help with KV loading
    
    float scale = rsqrtf(static_cast<float>(HEAD_DIM));

    // Process all KV tiles
    for (int kv_start = 0; kv_start < S; kv_start += TILE_N) {
        int valid_n = min(TILE_N, S - kv_start);
        
        // Cooperative load K and V tiles
        int total_kvs = valid_n * HEAD_DIM;
        
        #pragma unroll 4
        for (int i = tid; i < total_kvs; i += THREADS_PER_TILE) {
            int rn = i / HEAD_DIM;
            int c = i % HEAD_DIM;
            sK[i] = *(K + batch_idx * k_stride_b + head_idx * k_stride_h 
                    + (kv_start + rn) * k_stride_s + c);
            sV[i] = *(V + batch_idx * v_stride_b + head_idx * v_stride_h 
                    + (kv_start + rn) * v_stride_s + c);
        }
        __syncthreads();

        // Row computation: only threads with tid < valid_rows do softmax
        if (tid < valid_rows) {
            int global_row = tile_q * TILE_M + tid;
            
            // Load Q row
            const __nv_bfloat16* q_ptr = Q + batch_idx * q_stride_b + head_idx * q_stride_h 
                                       + global_row * q_stride_s;
            
            // Register file for one row
            alignas(16) float q_reg[HEAD_DIM];
            alignas(16) float o_reg[HEAD_DIM];
            
            // Initialize registers
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d += 4) {
                q_reg[d]   = bfloat16_to_float(q_ptr[d]);
                q_reg[d+1] = bfloat16_to_float(q_ptr[d+1]);
                q_reg[d+2] = bfloat16_to_float(q_ptr[d+2]);
                q_reg[d+3] = bfloat16_to_float(q_ptr[d+3]);
            }
            
            // Compute scores against all K rows in shared memory
            // Store in registers as [TILE_N] values
            float scores[TILE_N];
            #pragma unroll
            for (int nc = 0; nc < valid_n; ++nc) {
                float ds = 0.0f;
                const __nv_bfloat16* pk = &sK[nc * HEAD_DIM];
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; d += 4) {
                    ds += q_reg[d]   * bfloat16_to_float(pk[d]);
                    ds += q_reg[d+1] * bfloat16_to_float(pk[d+1]);
                    ds += q_reg[d+2] * bfloat16_to_float(pk[d+2]);
                    ds += q_reg[d+3] * bfloat16_to_float(pk[d+3]);
                }
                scores[nc] = ds * scale;
            }
            // Pad invalid entries
            for (int nc = valid_n; nc < TILE_N; ++nc) scores[nc] = -CUDART_INF_F;
            
            // Find max in this tile
            float tile_max = -CUDART_INF_F;
            #pragma unroll
            for (int nc = 0; nc < TILE_N; ++nc)
                if (scores[nc] > tile_max) tile_max = scores[nc];
            
            // Get previous max via shared memory communication (first iteration special)
            float prev_max = -CUDART_INF_F;
            float prev_sum = 0.0f;
            float alpha = expf(prev_max - fmaxf(prev_max, tile_max));
            float new_max = fmaxf(prev_max, tile_max);
            
            float p_sum = 0.0f;
            #pragma unroll
            for (int nc = 0; nc < TILE_N; ++nc) {
                float p = expf(scores[nc] - new_max);
                p_sum += p;
            }
        }
        __syncthreads();
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
    dim3 block(THREADS_PER_TILE);
    
    size_t smem_size = 2ULL * TILE_N * HEAD_DIM * sizeof(__nv_bfloat16);

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