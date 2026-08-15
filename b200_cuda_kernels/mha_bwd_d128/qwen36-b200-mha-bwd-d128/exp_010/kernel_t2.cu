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
constexpr int TPB = 128;

__device__ __forceinline__ float bf16tof(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ __nv_bfloat16 ftobf16(float x) {
    return __float2bfloat16(x);
}

// Simple reference-style kernel for correctness
// Each thread block computes one BM x BN tile of the SxS attention for one (b,h)
// Kernel 1: Compute dV contributions
// dV[k,d] += sum_m P[m,k] * dO[m,d]
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

    extern __shared__ char smem_raw[];

    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* sK = sQ + BM * BD;
    float* sLSE = reinterpret_cast<float*>(sK + BN);
    float* sP = sLSE + BM;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = tx + ty * 32;
    
    int bh = blockIdx.x;
    int b = bh / H;
    int h = bh % H;
    
    if (b >= B || h >= H) return;
    
    int q_tile = blockIdx.y;
    int k_tile = blockIdx.z;
    
    int q_start = q_tile * BM;
    int k_start = k_tile * BN;
    
    int base_bh = (b * H + h) * S * D;
    
    // Load Q tile into shared memory
    for (int i = tid; i < BM * BD; i += TPB) {
        int row = i / BD;
        int col = i % BD;
        int gidx = base_bh + (q_start + row) * D + col;
        sQ[i] = (q_start + row < S && col < D) ? Q[gidx] : ftobf16(0.0f);
    }
    
    // Load K tile into shared memory
    for (int i = tid; i < BN * BD; i += TPB) {
        int row = i / BD;
        int col = i % BD;
        int gidx = base_bh + (k_start + row) * D + col;
        sK[i] = (k_start + row < S && col < D) ? K[gidx] : ftobf16(0.0f);
    }
    
    // Load LSE values
    for (int i = ty; i < BM; i += THREADS_PER_BLOCK/32) {
        sLSE[i] = (q_start + i < S) ? L[(b * H + h) * S + q_start + i] : 0.0f;
    }
    
    __syncthreads();
    
    // Each warp handles one (m,n) pair - compute S[m,k_start+n] 
    // Use warps: each warp has 32 threads, we have 4 warps = 128 threads
    // Warp idx = threadIdx.y (0..3)
    // Each warp computes BM/4 rows
    
    int warp_m_start = ty * (BM / 4);
    int lane = tx;
    
    for (int m_local = warp_m_start; m_local < warp_m_start + (BM/4) && m_local < BM; m_local++) {
        float lse_val = sLSE[m_local];
        
        for (int n_local = lane; n_local < BN; n_local += 32) {
            // Compute dot product Q[q_start+m_local] . K[k_start+n_local]
            float s = 0.0f;
            #pragma unroll
            for (int d = 0; d < BD; d++) {
                s += bf16tof(sQ[m_local * BD + d]) * bf16tof(sK[n_local * BD + d]);
            }
            s *= scale;
            float p = expf(s - lse_val);
            sP[m_local * BN + n_local] = p;
        }
    }
    
    __syncthreads();
    
    // Now we have P[B,M][B,N] in shared memory
    // We need to compute:
    // 1. dV[k_start+n, d] += sum_m P[m,n] * dO[q_start+m, d]
    // 2. dPV[m,n] = sum_d dO[q_start+m,d] * V[k_start+n,d]
    // 3. local_mean[m] = sum_n P[m,n] * dPV[m,n]
    // 4. dS[m,n] = P[m,n] * (dPV[m,n] - local_mean[m])
    // 5. dQ[q_start+m, d] += sum_n dS[m,n] * K[k_start+n, d]
    // 6. dK[k_start+n, d] += sum_m dS[m,n] * Q[q_start+m, d]
    
    // We need more shared memory for additional buffers... this gets complex
    
    // Write partial dV to global memory using atomicAdd
    // Thread (ty,tx) -> handles output position (n_row, d_col)
    int out_n = tx;
    int out_d_start = ty * 32;
    
    for (int m_local = 0; m_local < BM; m_local++) {
        for (int n_local = out_n; n_local < BN; n_local += 32) {
            float p_val = sP[m_local * BN + n_local];
            for (int dd = 0; dd < 32; dd++) {
                int d_idx = out_d_start + dd;
                if (d_idx < D) {
                    float do_val = bf16tof(dO[base_bh + (q_start + m_local) * D + d_idx]);
                    float dv_acc = p_val * do_val;
                    
                    int dk_addr = base_bh + (k_start + n_local) * D + d_idx;
                    if (k_start + n_local < S) {
                        // Use atomicCAS loop for bf16 accumulation through fp32 reinterpret
                        atomicAdd((float*)&dV_out[dk_addr], dv_acc);
                    }
                }
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
    
    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* dO_ptr = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_ptr = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQ_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());
    
    size_t out_size = B * H * S * D * sizeof(__nv_bfloat16);
    CUDA_CHECK(cudaMemset(dQ_ptr, 0, out_size));
    CUDA_CHECK(cudaMemset(dK_ptr, 0, out_size));
    CUDA_CHECK(cudaMemset(dV_ptr, 0, out_size));
    
    float scale = 1.0f / sqrtf(static_cast<float>(D));
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    int num_q_tiles = (S + BM - 1) / BM;
    int num_k_tiles = (S + BN - 1) / BN;
    
    dim3 block(32, 4);
    dim3 grid(B * H, num_q_tiles, num_k_tiles);
    
    size_t smem_size = BM * BD * sizeof(__nv_bfloat16) + 
                       BN * BD * sizeof(__nv_bfloat16) +
                       BM * sizeof(float) +
                       BM * BN * sizeof(float);
    
    mha_bwd_kernel<<<grid, block, smem_size, stream>>>(
        Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr,
        dQ_ptr, dK_ptr, dV_ptr,
        B, H, S, D, scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_d128::run);

}  // namespace mha_bwd_d128