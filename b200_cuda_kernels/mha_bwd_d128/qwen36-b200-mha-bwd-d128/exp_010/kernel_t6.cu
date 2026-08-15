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

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BD = 128;
constexpr int TPB = 256;

__device__ __forceinline__ float bf16tof(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ __nv_bfloat16 ftobf16(float x) {
    return __float2bfloat16(x);
}

__global__ void mha_bwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ_out,
    __nv_bfloat16* __restrict__ dK_out,
    __nv_bfloat16* __restrict__ dV_out,
    int B, int H, int S, int D,
    float scale) {

    extern __shared__ char smem[];
    
    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* sK = sQ + BM * BD;
    __nv_bfloat16* sV = sK + BN * BD;
    __nv_bfloat16* sDO = sV + BN * BD;
    float* sLSE = reinterpret_cast<float*>(sDO + BM * BD);
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = tx + ty * 32;
    
    int linear_id = blockIdx.x;
    int num_bh = B * H;
    int num_qtiles = (S + BM - 1) / BM;
    int num_ktiles = (S + BN - 1) / BN;
    
    int bh = linear_id / (num_qtiles * num_ktiles);
    int qktile = linear_id % (num_qtiles * num_ktiles);
    int qt = qktile / num_ktiles;
    int kt = qktile % num_ktiles;
    
    int b = bh / H;
    int h = bh % H;
    if (b >= B || h >= H) return;
    
    int qs = qt * BM;
    int ks = kt * BN;
    
    int base_bh = (b * H + h) * S * D;
    
    // Load Q tile (BM x BD)
    for (int i = tid; i < BM * BD; i += TPB) {
        int r = i / BD;
        int c = i % BD;
        sQ[i] = (qs + r < S && c < D) ? Q[base_bh + (qs + r) * D + c] : ftobf16(0.0f);
    }
    
    // Load K tile (BN x BD)
    for (int i = tid; i < BN * BD; i += TPB) {
        int r = i / BD;
        int c = i % BD;
        sK[i] = (ks + r < S && c < D) ? K[base_bh + (ks + r) * D + c] : ftobf16(0.0f);
    }
    
    // Load V tile (BN x BD)
    for (int i = tid; i < BN * BD; i += TPB) {
        int r = i / BD;
        int c = i % BD;
        sV[i] = (ks + r < S && c < D) ? V[base_bh + (ks + r) * D + c] : ftobf16(0.0f);
    }
    
    // Load dO tile (BM x BD)
    for (int i = tid; i < BM * BD; i += TPB) {
        int r = i / BD;
        int c = i % BD;
        sDO[i] = (qs + r < S && c < D) ? dO[base_bh + (qs + r) * D + c] : ftobf16(0.0f);
    }
    
    // Load LSE values (BM)
    for (int i = tid; i < BM; i += TPB) {
        sLSE[i] = (qs + i < S) ? L[(b * H + h) * S + qs + i] : 0.0f;
    }
    
    __syncthreads();
    
    // Each thread computes one (m,n) pair: m=ty, n=tx
    // Warp-reduce for local_mean computation
    
    int m_local = ty;
    int n_local = tx;
    
    if (qs + m_local < S && ks + n_local < S) {
        float lse_val = sLSE[m_local];
        
        // Compute S[m,n] = dot(Q[m], K[n]) * scale
        float s_val = 0.0f;
        #pragma unroll
        for (int d = 0; d < BD; d++) {
            s_val += bf16tof(sQ[m_local * BD + d]) * bf16tof(sK[n_local * BD + d]);
        }
        s_val *= scale;
        
        // P[m,n] = exp(S - LSE)
        float p_val = expf(s_val - lse_val);
        
        // dPV[m,n] = dot(dO[m], V[n])
        float dpv = 0.0f;
        #pragma unroll
        for (int d = 0; d < BD; d++) {
            dpv += bf16tof(sDO[m_local * BD + d]) * bf16tof(sV[n_local * BD + d]);
        }
        
        // Warp reduce: compute sum_n P[m,n]*dpv for this m row
        float p_dpvi = p_val * dpv;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            p_dpvi += __shfl_down_sync(0xFFFFFFFF, p_dpvi, offset);
        }
        #pragma unroll
        for (int offset = 8; offset > 0; offset /= 2) {
            p_dpvi += __shfl_down_sync(0xFFFFFFFF, p_dpvi, offset);
        }
        #pragma unroll
        for (int offset = 4; offset > 0; offset /= 2) {
            p_dpvi += __shfl_down_sync(0xFFFFFFFF, p_dpvi, offset);
        }
        #pragma unroll
        for (int offset = 2; offset > 0; offset /= 2) {
            p_dpvi += __shfl_down_sync(0xFFFFFFFF, p_dpvi, offset);
        }
        #pragma unroll
        for (int offset = 1; offset > 0; offset /= 2) {
            p_dpvi += __shfl_down_sync(0xFFFFFFFF, p_dpvi, offset);
        }
        float local_mean = p_dpvi;
        
        // dS[m,n] = P * (dPV - local_mean)
        float ds = p_val * (dpv - local_mean);
        
        // Accumulate dV, dQ, dK via atomic adds
        int kn = ks + n_local;
        int qm = qs + m_local;
        
        for (int d = 0; d < BD; d++) {
            float dOval = bf16tof(sDO[m_local * BD + d]);
            float Kval = bf16tof(sK[n_local * BD + d]);
            float Qval = bf16tof(sQ[m_local * BD + d]);
            
            int dv_idx = base_bh + kn * D + d;
            int dq_idx = base_bh + qm * D + d;
            int dk_idx = base_bh + kn * D + d;
            
            atomicAdd((float*)&dV_out[dv_idx], p_val * dOval);
            atomicAdd((float*)&dQ_out[dq_idx], ds * Kval);
            atomicAdd((float*)&dK_out[dk_idx], ds * Qval);
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
    
    // Initialize outputs to zero
    size_t out_size = B * H * S * D * sizeof(__nv_bfloat16);
    CUDA_CHECK(cudaMemsetAsync(static_cast<__nv_bfloat16*>(dQ.data_ptr()), 0, out_size, stream));
    CUDA_CHECK(cudaMemsetAsync(static_cast<__nv_bfloat16*>(dK.data_ptr()), 0, out_size, stream));
    CUDA_CHECK(cudaMemsetAsync(static_cast<__nv_bfloat16*>(dV.data_ptr()), 0, out_size, stream));
    
    int num_qtiles = (S + BM - 1) / BM;
    int num_ktiles = (S + BN - 1) / BN;
    int total_blocks = B * H * num_qtiles * num_ktiles;
    
    dim3 block(32, 8);
    dim3 grid(total_blocks);
    
    size_t smem_size = 
        BM * BD * sizeof(__nv_bfloat16) +   // sQ
        BN * BD * sizeof(__nv_bfloat16) +   // sK
        BN * BD * sizeof(__nv_bfloat16) +   // sV
        BM * BD * sizeof(__nv_bfloat16) +   // sDO
        BM * sizeof(float);                 // sLSE
    
    mha_bwd_kernel<<<grid, block, smem_size, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        static_cast<__nv_bfloat16*>(dK.data_ptr()),
        static_cast<__nv_bfloat16*>(dV.data_ptr()),
        B, H, S, D, scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_d128::run);

}  // namespace mha_bwd_d128