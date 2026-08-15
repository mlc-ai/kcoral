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

// MHA backward kernel
// Each block handles one (batch, head) pair
// Each thread handles one element of the d-dimension
// No shared memory needed - each thread owns disjoint output locations
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
    
    if (tid >= d) return;
    
    const int bi = bh / H;
    const int hi = bh % H;
    
    const float scale = rsqrtf(static_cast<float>(d));
    
    // Offsets for this (batch, head)
    const uint64_t bh_off = static_cast<uint64_t>(bh) * S * d;
    const uint64_t l_off = static_cast<uint64_t>(bi) * H * S + static_cast<uint64_t>(hi) * S;
    
    // Input/output pointers for this (batch, head) slice
    const __nv_bfloat16* Q_bh = Q + bh_off;
    const __nv_bfloat16* K_bh = K + bh_off;
    const __nv_bfloat16* V_bh = V + bh_off;
    const __nv_bfloat16* dO_bh = dO + bh_off;
    const float* L_bh = L + l_off;
    __nv_bfloat16* dQ_bh = dQ + bh_off;
    __nv_bfloat16* dK_bh = dK + bh_off;
    __nv_bfloat16* dV_bh = dV + bh_off;
    
    // Initialize dK[tid] and dV[tid] for this thread's d-element positions
    // Thread tid owns positions: tid, tid+d, tid+2d, ...
    for (int k = 0; k < S; k++) {
        dK_bh[k * d + tid] = __float2bfloat16(0.f);
        dV_bh[k * d + tid] = __float2bfloat16(0.f);
    }
    
    // Main loop over query positions
    for (int q = 0; q < S; q++) {
        const float Q_val = __bfloat162float(Q_bh[q * d + tid]);
        const float dO_val = __bfloat162float(dO_bh[q * d + tid]);
        const float L_val = L_bh[q];
        
        // Phase 1: Compute softmax correction corr[q] = sum_k P[q][k] * dS[q][k]
        float corr = 0.f;
        for (int k = 0; k < S; k++) {
            const float K_val = __bfloat162float(K_bh[k * d + tid]);
            const float V_val = __bfloat162float(V_bh[k * d + tid]);
            const float score = Q_val * K_val * scale;
            const float P_val = expf(score - L_val);
            const float dS_val = dO_val * V_val;
            corr += P_val * dS_val;
        }
        
        // Phase 2: Compute dQ[q][tid] and accumulate dK, dV
        float dq = 0.f;
        for (int k = 0; k < S; k++) {
            const float K_val = __bfloat162float(K_bh[k * d + tid]);
            const float V_val = __bfloat162float(V_bh[k * d + tid]);
            const float score = Q_val * K_val * scale;
            const float P_val = expf(score - L_val);
            const float dS_val = dO_val * V_val;
            
            // dQ contribution: P[q][k] * (dS[q][k] - corr[q]) * K[k][tid]
            dq += P_val * (dS_val - corr) * K_val;
            
            // dV[k][tid] += P[q][k] * dO[q][tid]
            // Since each thread owns unique d-position, no race condition
            float dv_cur = __bfloat162float(dV_bh[k * d + tid]);
            dV_bh[k * d + tid] = __float2bfloat16(dv_cur + P_val * dO_val);
            
            // dK[k][tid] += P[q][k] * (dS[q][k] - corr[q]) * Q[q][tid]
            float dk_cur = __bfloat162float(dK_bh[k * d + tid]);
            dK_bh[k * d + tid] = __float2bfloat16(dk_cur + P_val * (dS_val - corr) * Q_val);
        }
        dQ_bh[q * d + tid] = __float2bfloat16(dq);
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
    
    const int64_t B = Q.size(0);
    const int64_t H = Q.size(1);
    const int64_t S = Q.size(2);
    const int64_t d = Q.size(3);
    
    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* dO_ptr = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_ptr = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQ_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());
    
    const int num_blocks = static_cast<int>(B * H);
    const int block_size = static_cast<int>(d);
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    dim3 grid(num_blocks);
    dim3 block(block_size);
    
    mha_bwd_impl::mha_backward_kernel<<<grid, block, 0, stream>>>(
        Q_ptr, K_ptr, V_ptr, L_ptr, dO_ptr,
        dQ_ptr, dK_ptr, dV_ptr,
        static_cast<int>(B), static_cast<int>(H),
        static_cast<int>(S), static_cast<int>(d));
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

}  // extern "C"