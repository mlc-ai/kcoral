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

// MHA backward kernel with zero shared memory
// Each block handles one (batch, head) pair
// Each thread handles one element of the d-dimension
// Uses atomicAdd on global memory for dK and dV accumulation
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
    
    // Bounds check
    if (tid >= d) return;
    
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
    
    // For dQ, write directly (no reduction needed across q)
    // For dK and dV, we accumulate via atomicAdd
    
    // Initialize dK[tid] and dV[tid] for this thread's d-element positions
    // Each thread owns positions: tid, tid+blockDim.x, tid+2*blockDim.x, ...
    for (int k = tid; k < S; k += blockDim.x) {
        const int idx = k * d + tid;
        dK_bh[idx] = __float2bfloat16(0.f);
        dV_bh[idx] = __float2bfloat16(0.f);
    }
    
    // Main loop over query positions
    for (int q = 0; q < S; q++) {
        const float Q_val = __bfloat162float(Q_bh[q * d + tid]);
        const float dO_val = __bfloat162float(dO_bh[q * d + tid]);
        const float L_val = L_bh[q];
        
        // --- Phase 1: Compute softmax correction corr[q] ---
        float corr = 0.f;
        for (int k = 0; k < S; k++) {
            const float K_val = __bfloat162float(K_bh[k * d + tid]);
            const float V_val = __bfloat162float(V_bh[k * d + tid]);
            const float score = Q_val * K_val * scale;
            const float P_val = expf(score - L_val);
            const float dS_val = dO_val * V_val;
            corr += P_val * dS_val;
        }
        
        // --- Phase 2: Compute dQ[q][tid] ---
        float dq = 0.f;
        for (int k = 0; k < S; k++) {
            const float K_val = __bfloat162float(K_bh[k * d + tid]);
            const float V_val = __bfloat162float(V_bh[k * d + tid]);
            const float score = Q_val * K_val * scale;
            const float P_val = expf(score - L_val);
            const float dS_val = dO_val * V_val;
            dq += P_val * (dS_val - corr) * K_val;
        }
        dQ_bh[q * d + tid] = __float2bfloat16(dq);
        
        // --- Phase 3: Accumulate dK and dV ---
        for (int k = 0; k < S; k++) {
            const float K_val = __bfloat162float(K_bh[k * d + tid]);
            const float V_val = __bfloat162float(V_bh[k * d + tid]);
            const float score = Q_val * K_val * scale;
            const float P_val = expf(score - L_val);
            const float dS_val = dO_val * V_val;
            
            // dV[k][tid] += P[q][k] * dO[q][tid]
            const float dv_add = P_val * dO_val;
            const int dv_idx = k * d + tid;
            atomicAdd((float*)&reinterpret_cast<const nv_float*>(dV_bh)[dv_idx], dv_add);
            
            // dK[k][tid] += P[q][k] * (dS[q][k] - corr[q]) * Q[q][tid]
            const float dk_add = P_val * (dS_val - corr) * Q_val;
            atomicAdd((float*)&reinterpret_cast<const nv_float*>(dK_bh)[dv_idx], dk_add);
        }
    }
}

// Simpler atomicAdd for bf16 stored at global memory location
template<>
__device__ __forceinline__ void atomicAdd<nv_float>(nv_float* address, nv_float val) {
    unsigned int* address_as_ui = (unsigned int*)((char*)address - ((size_t)address & 3));
    unsigned int old = *address_as_ui;
    unsigned int assumed;
    do {
        assumed = old;
        __half half_old = __short_as_half(old & 0xFFFF);
        __half half_val = __short_as_half(*reinterpret_cast<__half*>(&val));
        __half half_sum = __hadd(half_old, half_val);
        old = (old & 0xFFFF0000) | __half_as_short(half_sum);
        old = atomicCAS(address_as_ui, assumed, old);
    } while (assumed != old);
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
    
    mha_bwd_impl::mha_backward_kernel<<<num_blocks, block_size, 0, stream>>>(
        Q_ptr, K_ptr, V_ptr, L_ptr, dO_ptr,
        dQ_ptr, dK_ptr, dV_ptr,
        static_cast<int>(B), static_cast<int>(H),
        static_cast<int>(S), static_cast<int>(d));
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

}  // extern "C"