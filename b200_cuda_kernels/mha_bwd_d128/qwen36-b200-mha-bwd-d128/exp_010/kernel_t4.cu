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

constexpr int BM = 32;
constexpr int BN = 32;
constexpr int BD = 128;
constexpr int TPB = 128;

__device__ __forceinline__ float bf16tof(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ __nv_bfloat16 ftobf16(float x) {
    return __float2bfloat16(x);
}

// Stage 1: Compute dS_tile[m,n] for each (q_tile, k_tile) pair
// Write to global fp32 buffer: dS_global[bh][q_start+m][k_start+n]
__global__ void compute_dS_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    float* __restrict__ dS_global,
    int B, int H, int S, int D,
    float scale,
    int bh_stride) {

    extern __shared__ char smem_raw[];

    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* sK = sQ + BM * BD;
    __nv_bfloat16* sV = sK + BN * BD;
    __nv_bfloat16* sDO = sV + BN * BD;
    float* sLSE = reinterpret_cast<float*>(sDO + BM * BD);
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = tx + ty * 32;
    
    int linear_id = blockIdx.x;
    int bh = linear_id / ((S + BM - 1) / BM) / ((S + BN - 1) / BN);
    int q_idx = (linear_id / ((S + BN - 1) / BN)) % ((S + BM - 1) / BM);
    int k_idx = linear_id % ((S + BN - 1) / BN);
    
    int b = bh / H;
    int h = bh % H;
    if (b >= B || h >= H) return;
    
    int q_start = q_idx * BM;
    int k_start = k_idx * BN;
    
    int base_bh = (b * H + h) * S * D;
    int base_dS = bh * bh_stride + q_start * S + k_start;
    
    // Load Q tile
    for (int i = tid; i < BM * BD; i += TPB) {
        int row = i / BD;
        int col = i % BD;
        int gidx = base_bh + (q_start + row) * D + col;
        sQ[i] = (q_start + row < S && col < D) ? Q[gidx] : ftobf16(0.0f);
    }
    
    // Load K tile
    for (int i = tid; i < BN * BD; i += TPB) {
        int row = i / BD;
        int col = i % BD;
        int gidx = base_bh + (k_start + row) * D + col;
        sK[i] = (k_start + row < S && col < D) ? K[gidx] : ftobf16(0.0f);
    }
    
    // Load V tile
    for (int i = tid; i < BN * BD; i += TPB) {
        int row = i / BD;
        int col = i % BD;
        int gidx = base_bh + (k_start + row) * D + col;
        sV[i] = (k_start + row < S && col < D) ? V[gidx] : ftobf16(0.0f);
    }
    
    // Load dO tile
    for (int i = tid; i < BM * BD; i += TPB) {
        int row = i / BD;
        int col = i % BD;
        int gidx = base_bh + (q_start + row) * D + col;
        sDO[i] = (q_start + row < S && col < D) ? dO[gidx] : ftobf16(0.0f);
    }
    
    // Load LSE
    for (int i = tid; i < BM; i += TPB) {
        sLSE[i] = (q_start + i < S) ? L[(b * H + h) * S + q_start + i] : 0.0f;
    }
    
    __syncthreads();
    
    // Each thread computes one (m,n) element of dS
    int m_local = ty;
    int n_local = tx;
    
    if (m_local < BM && n_local < BN && q_start + m_local < S && k_start + n_local < S) {
        float lse_val = sLSE[m_local];
        
        // Compute S[m,n] = Q[q_start+m] . K[k_start+n] * scale
        float s_val = 0.0f;
        for (int d = 0; d < BD; d++) {
            s_val += bf16tof(sQ[m_local * BD + d]) * bf16tof(sK[n_local * BD + d]);
        }
        s_val *= scale;
        
        // P[m,n] = exp(S[m,n] - LSE)
        float p_val = expf(s_val - lse_val);
        
        // dPV[m,n] = sum_d dO[q_start+m, d] * V[k_start+n, d]
        float dpv = 0.0f;
        for (int d = 0; d < BD; d++) {
            dpv += bf16tof(sDO[m_local * BD + d]) * bf16tof(sV[n_local * BD + d]);
        }
        
        dS_global[base_dS + m_local * S + n_local] = p_val * dpv;
    }
}

// Stage 2: Reduce dS to compute dQ and dK
__global__ void reduce_dS_kernel(
    const float* __restrict__ dS_global,
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ_out,
    __nv_bfloat16* __restrict__ dK_out,
    int B, int H, int S, int D,
    int bh_stride) {
    
    extern __shared__ __nv_bfloat16 smem_QK[];
    
    int bh = blockIdx.x;
    int b = bh / H;
    int h = bh % H;
    if (b >= B || h >= H) return;
    
    int tx = threadIdx.x;
    int lane = tx & 31;
    
    int base_bh = (b * H + h) * S * D;
    
    // Phase 1: Compute dQ[m,d] = sum_n dS[m,n] * K[n,d]
    // Each thread handles one (m_row, d_col) 
    // We iterate over all K tiles
    for (int d = tx; d < D; d += TPB) {
        for (int m = 0; m < S; m += BM) {
            float dq_acc = 0.0f;
            for (int n = 0; n < S; n += BN) {
                int base_dS = bh * bh_stride + m * S + n;
                int m_local = (lane * BM) / 32;
                
                for (int mi = 0; mi < BM && m + m_local + mi < S; mi++) {
                    float ds = dS_global[base_dS + (m_local + mi) * S + n];
                    dq_acc += ds * bf16tof(K[base_bh + (n) * D + d]);
                }
            }
            
            if (lane == 0 && m < S) {
                atomicAdd(&reinterpret_cast<float*>(dQ_out)[base_bh + m * D + d], dq_acc);
            }
        }
    }
    
    // Phase 2: Compute dK[n,d] = sum_m dS[m,n] * Q[m,d]
    for (int d = tx; d < D; d += TPB) {
        for (int n = 0; n < S; n += BN) {
            float dk_acc = 0.0f;
            for (int m = 0; m < S; m += BM) {
                int base_dS = bh * bh_stride + m * S + n;
                int m_local = (lane * BM) / 32;
                
                for (int mi = 0; mi < BM && m + m_local + mi < S; mi++) {
                    float ds = dS_global[base_dS + (m_local + mi) * S + n];
                    dk_acc += ds * bf16tof(Q[base_bh + (m + m_local + mi) * D + d]);
                }
            }
            
            if (lane == 0 && n < S) {
                atomicAdd(&reinterpret_cast<float*>(dK_out)[base_bh + n * D + d], dk_acc);
            }
        }
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
    
    // Launch simple reference kernel inline
    dim3 block(TPB);
    
    int num_tiles_q = (S + BM - 1) / BM;
    int num_tiles_k = (S + BN - 1) / BN;
    int total_tiles = B * H * num_tiles_q * num_tiles_k;
    
    dim3 grid(total_tiles);
    
    size_t smem_size = (BM + BN) * BD * sizeof(__nv_bfloat16) * 2 + BM * sizeof(float);
    
    // Simple reference-style kernel embedded here
    // Using a straightforward approach without complex sharing
    
    // Allocate temporary buffer for intermediate results
    float* dS_temp;
    size_t dS_size = B * H * S * S * sizeof(float);
    CUDA_CHECK(cudaMallocAsync(&dS_temp, dS_size, stream));
    CUDA_CHECK(cudaMemsetAsync(dS_temp, 0, dS_size, stream));
    
    // Launch stage 1
    compute_dS_kernel<<<grid, block, smem_size, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        dS_temp,
        B, H, S, D, scale, S);
    CUDA_CHECK(cudaGetLastError());
    
    // Launch stage 2 for reduction
    dim3 block2(TPB);
    dim3 grid2(B * H);
    
    reduce_dS_kernel<<<grid2, block2, (BM + BN) * BD * sizeof(__nv_bfloat16), stream>>>(
        dS_temp,
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        static_cast<__nv_bfloat16*>(dK.data_ptr()),
        B, H, S, D, S);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaFreeAsync(dS_temp, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_d128::run);

}  // namespace mha_bwd_d128