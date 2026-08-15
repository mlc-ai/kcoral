#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <math.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace mha_bwd_d128 {

// Tile parameters
constexpr int TILE_Q = 32;   // q-dimension tile
constexpr int TILE_K = 32;   // k-dimension tile  
constexpr int THREADS_X = 32;
constexpr int THREADS_Y = 4;
constexpr int TPB = THREADS_X * THREADS_Y; // 128

__device__ __forceinline__ float bf16tof(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ __nv_bfloat16 ftobf16(float x) {
    return __float2bfloat16(x);
}

// Main backward kernel - each block handles one (bh, q_tile, k_tile)
// Computes all three gradients cooperatively using shared memory
__global__ void mha_bwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    float* __restrict__ dQ_global,
    float* __restrict__ dK_global,
    float* __restrict__ dV_global,
    int B, int H, int S, int D,
    float scale) {

    // Shared memory layout:
    // sQ: TILE_Q x D bf16 (128*32=4096 bytes)
    // sK: TILE_K x D bf16 (4096 bytes)  
    // sV: TILE_K x D bf16 (4096 bytes)
    // sDO: TILE_Q x D bf16 (4096 bytes)
    // sLSE: TILE_Q float (128 bytes)
    // Total: ~17KB, well within limits
    
    extern __shared__ char smem_raw[];
    
    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* sK = sQ + TILE_Q * D;
    __nv_bfloat16* sV = sK + TILE_K * D;
    __nv_bfloat16* sDO = sV + TILE_K * D;
    float* sLSE = reinterpret_cast<float*>(sDO + TILE_Q * D);
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = tx + ty * THREADS_X;
    
    int linear_id = blockIdx.x;
    int num_k_tiles = (S + TILE_K - 1) / TILE_K;
    int num_q_tiles = (S + TILE_Q - 1) / TILE_Q;
    
    int bh = linear_id / (num_q_tiles * num_k_tiles);
    int qk_idx = linear_id % (num_q_tiles * num_k_tiles);
    int q_tile = qk_idx / num_k_tiles;
    int k_tile = qk_idx % num_k_tiles;
    
    int b = bh / H;
    int h = bh % H;
    if (b >= B || h >= H) return;
    
    int q_start = q_tile * TILE_Q;
    int k_start = k_tile * TILE_K;
    
    int base_bh = (b * H + h) * S * D;
    
    // Load Q tile: TILE_Q rows x D cols
    for (int i = tid; i < TILE_Q * D; i += TPB) {
        int row = i / D;
        int col = i % D;
        int gidx = base_bh + (q_start + row) * D + col;
        sQ[i] = (q_start + row < S && col < D) ? Q[gidx] : ftobf16(0.0f);
    }
    
    // Load K tile
    for (int i = tid; i < TILE_K * D; i += TPB) {
        int row = i / D;
        int col = i % D;
        int gidx = base_bh + (k_start + row) * D + col;
        sK[i] = (k_start + row < S && col < D) ? K[gidx] : ftobf16(0.0f);
    }
    
    // Load V tile  
    for (int i = tid; i < TILE_K * D; i += TPB) {
        int row = i / D;
        int col = i % D;
        int gidx = base_bh + (k_start + row) * D + col;
        sV[i] = (k_start + row < S && col < D) ? V[gidx] : ftobf16(0.0f);
    }
    
    // Load dO tile
    for (int i = tid; i < TILE_Q * D; i += TPB) {
        int row = i / D;
        int col = i % D;
        int gidx = base_bh + (q_start + row) * D + col;
        sDO[i] = (q_start + row < S && col < D) ? dO[gidx] : ftobf16(0.0f);
    }
    
    // Load LSE values
    for (int i = tid; i < TILE_Q; i += TPB) {
        sLSE[i] = (q_start + i < S) ? L[(b * H + h) * S + q_start + i] : 0.0f;
    }
    
    __syncthreads();
    
    // Each thread (ty,tx) -> (m_local, n_local) maps to:
    // m_local = ty (0..3), n_local = tx (0..31)
    // We'll compute attention and gradients for this (m,n) pair
    
    int m_local = ty;
    int n_local = tx;
    
    float dq_accum[THREADS_Y] = {};
    float dk_accum[THREADS_Y] = {};
    float dv_accum[THREADS_Y] = {};
    
    // Phase 1: For each m in TILE_Q, compute all attention probs to k in TILE_K
    // and accumulate gradients
    
    for (int m = 0; m < TILE_Q; m++) {
        if (q_start + m >= S) continue;
        
        float lse_val = sLSE[m];
        
        // We'll compute P[m,n], dPV[m,n], and eventually dS[m,n]
        // Thread with tx=n_local computes these for n = k_start + n_local
        
        int n = k_start + n_local;
        if (n >= S) continue;
        
        // Compute S[m,n] = Q[q_start+m].K[k_start+n_local] * scale
        float s_val = 0.0f;
        #pragma unroll
        for (int d = 0; d < D; d++) {
            s_val += bf16tof(sQ[m * D + d]) * bf16tof(sK[n_local * D + d]);
        }
        s_val *= scale;
        
        // P[m,n] = exp(S[m,n] - LSE[q_start+m])
        float p_val = expf(s_val - lse_val);
        
        // dPV[m,n] = sum_d dO[q_start+m, d] * V[k_start+n_local, d]
        float dpv = 0.0f;
        #pragma unroll
        for (int d = 0; d < D; d++) {
            dpv += bf16tof(sDO[m * D + d]) * bf16tof(sV[n_local * D + d]);
        }
        
        // Accumulate for local_mean = sum_n P[m,n] * dPV[m,n]
        // Use warp-level reduction within the TILE_K threads
        float local_mean = 0.0f;
        
        // Warp reduce: sum P[m,n]*dpv across all n in this tile
        float p_dpvi = p_val * dpv;
        
        // Shfl down within warp to sum
        for (int offset = 16; offset > 0; offset /= 2) {
            p_dpvi += __shfl_down_sync(0xFFFFFFFF, p_dpvi, offset);
        }
        for (int offset = 8; offset > 0; offset /= 2) {
            p_dpvi += __shfl_down_sync(0xFFFFFFFF, p_dpvi, offset);
        }
        for (int offset = 4; offset > 0; offset /= 2) {
            p_dpvi += __shfl_down_sync(0xFFFFFFFF, p_dpvi, offset);
        }
        for (int offset = 2; offset > 0; offset /= 2) {
            p_dpvi += __shfl_down_sync(0xFFFFFFFF, p_dpvi, offset);
        }
        for (int offset = 1; offset > 0; offset /= 2) {
            p_dpvi += __shfl_down_sync(0xFFFFFFFF, p_dpvi, offset);
        }
        
        // All threads in warp now have the full sum
        local_mean = p_dpvi;
        
        // Broadcast local_mean to all threads
        local_mean = __shfl_sync(0xFFFFFFFF, local_mean, 0);
        
        // dS[m,n] = P[m,n] * (dPV[m,n] - local_mean)
        float ds = p_val * (dpv - local_mean);
        
        // Now we need to accumulate:
        // dV[k+n_local, d] += P[m,n] * dO[m, d]
        // dQ[m, d] += ds * K[k+n_local, d]
        // dK[k+n_local, d] += ds * Q[m, d]
        
        // Atomic adds to global fp32 buffers
        for (int d = 0; d < D; d++) {
            int dv_addr = base_bh + n * D + d;
            atomicAdd(&dV_global[dv_addr], p_val * bf16tof(sDO[m * D + d]));
            
            int dq_addr = base_bh + (q_start + m) * D + d;
            atomicAdd(&dQ_global[dq_addr], ds * bf16tof(sK[n_local * D + d]));
            
            int dk_addr = dv_addr;
            atomicAdd(&dK_global[dk_addr], ds * bf16tof(sQ[m * D + d]));
        }
    }
}

// Final kernel: convert fp32 accumulated grads to bf16 output format
__global__ void bf16_store_kernel(const float* src, __nv_bfloat16* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = ftobf16(src[idx]);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);
    
    float scale = 1.0f / sqrtf(static_cast<float>(D));
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    // Allocate fp32 temp buffers for atomic accumulation
    size_t out_elements = B * H * S * D;
    size_t out_bytes = out_elements * sizeof(float);
    
    float* dQ_fp32, *dK_fp32, *dV_fp32;
    CUDA_CHECK(cudaMallocAsync(&dQ_fp32, out_bytes, stream));
    CUDA_CHECK(cudaMallocAsync(&dK_fp32, out_bytes, stream));
    CUDA_CHECK(cudaMallocAsync(&dV_fp32, out_bytes, stream));
    
    CUDA_CHECK(cudaMemsetAsync(dQ_fp32, 0, out_bytes, stream));
    CUDA_CHECK(cudaMemsetAsync(dK_fp32, 0, out_bytes, stream));
    CUDA_CHECK(cudaMemsetAsync(dV_fp32, 0, out_bytes, stream));
    
    // Grid/block setup
    int num_k_tiles = (S + TILE_K - 1) / TILE_K;
    int num_q_tiles = (S + TILE_Q - 1) / TILE_Q;
    int total_blocks = B * H * num_q_tiles * num_k_tiles;
    
    dim3 block(THREADS_X, THREADS_Y);
    dim3 grid(total_blocks);
    
    size_t smem_size = 
        (TILE_Q + TILE_K) * D * sizeof(__nv_bfloat16) * 2 +
        TILE_Q * sizeof(float);
    
    mha_bwd_kernel<<<grid, block, smem_size, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        dQ_fp32, dK_fp32, dV_fp32,
        B, H, S, D, scale);
    CUDA_CHECK(cudaGetLastError());
    
    // Convert fp32 -> bf16 output
    int conv_blocks = (out_elements + 255) / 256;
    bf16_store_kernel<<<conv_blocks, 256, 0, stream>>>(dQ_fp32, 
        static_cast<__nv_bfloat16*>(dQ.data_ptr()), out_elements);
    bf16_store_kernel<<<conv_blocks, 256, 0, stream>>>(dK_fp32,
        static_cast<__nv_bfloat16*>(dK.data_ptr()), out_elements);
    bf16_store_kernel<<<conv_blocks, 256, 0, stream>>>(dV_fp32,
        static_cast<__nv_bfloat16*>(dV.data_ptr()), out_elements);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaFreeAsync(dQ_fp32, stream));
    CUDA_CHECK(cudaFreeAsync(dK_fp32, stream));
    CUDA_CHECK(cudaFreeAsync(dV_fp32, stream));
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_d128::run);

}  // namespace mha_bwd_d128