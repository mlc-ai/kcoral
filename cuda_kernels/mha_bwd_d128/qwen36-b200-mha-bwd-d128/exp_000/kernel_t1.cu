#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace mha_bwd_impl {

// MHA backward kernel using shared memory accumulators
// Each block handles one (batch, head) pair
// Each thread handles one element of the d-dimension
__global__ void mha_backward_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const float* __restrict__ L,
    const __nv_bfloat16* __restrict__ dO,
    __nv_bfloat16* __restrict__ dQ,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    int B, int H, int S, int d) 
{
    const int bh = blockIdx.x;
    const int tid = threadIdx.x;
    const int bi = bh / H;
    const int hi = bh % H;
    
    const float scale = rsqrtf(static_cast<float>(d));
    
    // Offsets for this (batch, head)
    const uint64_t bh_off = static_cast<uint64_t>(bh) * S * d;
    const uint64_t l_off = static_cast<uint64_t>(bi) * H * S + static_cast<uint64_t>(hi) * S;
    
    const __nv_bfloat16* Q_bh = Q + bh_off;
    const __nv_bfloat16* K_bh = K + bh_off;
    const __nv_bfloat16* V_bh = V + bh_off;
    const __nv_bfloat16* dO_bh = dO + bh_off;
    const float* L_bh = L + l_off;
    __nv_bfloat16* dQ_bh = dQ + bh_off;
    __nv_bfloat16* dK_bh = dK + bh_off;
    __nv_bfloat16* dV_bh = dV + bh_off;
    
    // Bounds check for d-dimension
    if (tid >= d) return;
    
    // Shared memory layout:
    // [0:S-1]:       dV accumulators
    // [S:2S-1]:      dK part1 accumulators (P*dS*Q)
    // [2S:3S-1]:     dK part2 accumulators (corr*P*Q)  
    // [3S:4S-1]:     corr[q] cache
    extern __shared__ char smem[];
    float* __restrict__ dV_acc = reinterpret_cast<float*>(smem);
    float* __restrict__ dK1_acc = dV_acc + S;
    float* __restrict__ dK2_acc = dK1_acc + S;
    float* __restrict__ corr_cache = dK2_acc + S;
    
    // Initialize shared memory accumulators
    for (int i = tid; i < S; i += blockDim.x) {
        dV_acc[i] = 0.f;
        dK1_acc[i] = 0.f;
        dK2_acc[i] = 0.f;
        corr_cache[i] = 0.f;
    }
    __syncthreads();
    
    // ========================================
    // For each query position q:
    //   1. Compute corr[q] = sum_k P[q][k] * dS[q][k]
    //   2. Compute dQ[q] = sum_k P[q][k]*(dS[q][k]-corr[q])*K[k]
    //   3. Accumulate dV[k] += P[q][k]*dO[q]
    //   4. Accumulate dK components
    // ========================================
    
    for (int q = 0; q < S; q++) {
        const float Q_val = __bfloat162float(Q_bh[q * d + tid]);
        const float dO_val = __bfloat162float(dO_bh[q * d + tid]);
        const float L_val = L_bh[q];
        
        // --- Pass 1: Compute softmax correction corr[q] ---
        // corr[q] = sum_k P[q][k] * dS[q][k]
        // where P[q][k] = exp(Q[q]*K[k]/sqrt(d) - L[q])
        //       dS[q][k] = dO[q] * V[k]
        float corr = 0.f;
        for (int k = 0; k < S; k++) {
            const float K_val = __bfloat162float(K_bh[k * d + tid]);
            const float V_val = __bfloat162float(V_bh[k * d + tid]);
            const float score = Q_val * K_val * scale;
            const float P_val = expf(score - L_val);
            const float dS_val = dO_val * V_val;
            corr += P_val * dS_val;
        }
        
        // Cache corr for later use
        corr_cache[q] = corr;
        
        // --- Pass 2: Compute dQ and accumulate dV, dK ---
        // dQ[q] = sum_k P[q][k] * (dS[q][k] - corr[q]) * K[k]
        float dq = 0.f;
        for (int k = 0; k < S; k++) {
            const float K_val = __bfloat162float(K_bh[k * d + tid]);
            const float V_val = __bfloat162float(V_bh[k * d + tid]);
            const float score = Q_val * K_val * scale;
            const float P_val = expf(score - L_val);
            const float dS_val = dO_val * V_val;
            
            // dQ contribution
            dq += P_val * (dS_val - corr) * K_val;
            
            // dV[k] += P[q][k] * dO[q]
            atomicAdd(&dV_acc[k], P_val * dO_val);
            
            // dK[k] = sum_q P[q][k] * (dS[q][k] - corr[q]) * Q[q]
            // Split into: dK1[k] += P*q*k * dS*q*k * Q[q]
            //             dK2[k] += corr[q] * P[q][k] * Q[q]
            // Then dK[k] = dK1[k] - dK2[k]
            atomicAdd(&dK1_acc[k], P_val * dS_val * Q_val);
            atomicAdd(&dK2_acc[k], corr * P_val * Q_val);
        }
        
        // Write dQ
        dQ_bh[q * d + tid] = __float2bfloat16(dq);
    }
    
    // ========================================
    // Finalize dK and dV
    // ========================================
    __syncthreads();
    
    for (int k = tid; k < S; k += blockDim.x) {
        const float dk = dK1_acc[k] - dK2_acc[k];
        dK_bh[k * d + tid] = __float2bfloat16(dk);
        dV_bh[k * d + tid] = __float2bfloat16(dV_acc[k]);
    }
}

}  // namespace mha_bwd_impl

extern "C" {

void run(tvm::ffi::TensorView Q,
         tvm::ffi::TensorView K,
         tvm::ffi::TensorView V,
         tvm::ffi::TensorView O,
         tvm::ffi::TensorView dO,
         tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ,
         tvm::ffi::TensorView dK,
         tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    // Extract dimensions
    const int64_t B = Q.size(0);
    const int64_t H = Q.size(1);
    const int64_t S = Q.size(2);
    const int64_t d = Q.size(3);
    
    // Get data pointers
    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* O_ptr = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dO_ptr = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_ptr = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQ_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());
    
    // Launch configuration:
    // One block per (batch, head) pair
    // blockDim.x = d (128 threads for d=128)
    const int num_blocks = static_cast<int>(B * H);
    const int block_size = static_cast<int>(d);
    
    // Shared memory: 4 arrays of S floats each
    const int shmem_bytes = 4 * static_cast<int>(S) * sizeof(float);
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    dim3 grid(num_blocks);
    dim3 block(block_size);
    
    mha_bwd_impl::mha_backward_kernel<<<grid, block, shmem_bytes, stream>>>(
        Q_ptr, K_ptr, V_ptr, L_ptr, dO_ptr,
        dQ_ptr, dK_ptr, dV_ptr,
        static_cast<int>(B), static_cast<int>(H),
        static_cast<int>(S), static_cast<int>(d));
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

}  // extern "C"