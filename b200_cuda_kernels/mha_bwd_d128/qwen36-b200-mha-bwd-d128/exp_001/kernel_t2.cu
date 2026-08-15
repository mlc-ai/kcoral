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

namespace tvm_ffi_mha_bwd {

__forceinline__ __device__ float bf162f(__nv_bfloat16 h) {
    return __bfloat162float(h);
}

__forceinline__ __device__ __nv_bfloat16 f2bf16(float f) {
    return __float2bfloat16(f);
}

// Atomic add for bf16
__forceinline__ __device__ void atomic_add_bf16(__nv_bfloat16* address, __nv_bfloat16 val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    atomicAdd(address, val);
#else
    unsigned int* address_as_ui = reinterpret_cast<unsigned int*>(
        reinterpret_cast<char*>(address));
    unsigned int old = *address_as_ui, assumed;
    do {
        assumed = old;
        __nv_bfloat16 new_val = val + *reinterpret_cast<__nv_bfloat16*>(&assumed);
        old = atomicCAS(address_as_ui, assumed, *(unsigned int*)&new_val);
    } while (old != assumed);
#endif
}

template<int BDIM, int BLOCK_SIZE>
__global__ void mha_bwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    int B, int H, int S)
{
    // Each block handles one (b,h) pair
    int bh = blockIdx.x;
    int b = bh / H;
    int h = bh % H;
    int tid = threadIdx.x;
    
    if (b >= B || h >= H) return;
    
    // Base pointers for this (b,h)
    size_t bh_offset = (size_t)(b * H + h) * S * BDIM;
    size_t l_offset  = (size_t)(b * H + h) * S;
    
    const __nv_bfloat16* Q_bh = Q + bh_offset;
    const __nv_bfloat16* K_bh = K + bh_offset;
    const __nv_bfloat16* V_bh = V + bh_offset;
    const __nv_bfloat16* dO_bh = dO + bh_offset;
    const float* L_bh = L + l_offset;
    __nv_bfloat16* dQ_bh = dQ + bh_offset;
    __nv_bfloat16* dK_bh = dK + bh_offset;
    __nv_bfloat16* dV_bh = dV + bh_offset;
    
    float inv_sqrt_d = rsqrtf((float)BDIM);
    
    // Initialize outputs to zero
    int total_elements = S * BDIM;
    for (int i = tid; i < total_elements; i += BLOCK_SIZE) {
        dQ_bh[i] = f2bf16(0.f);
        dK_bh[i] = f2bf16(0.f);
        dV_bh[i] = f2bf16(0.f);
    }
    __syncthreads();
    
    // Phase 1: Compute dV[n] = sum_m attn[m,n] * dO[m]
    // Each thread handles one output position n (strided over S)
    for (int n = tid; n < S; n += BLOCK_SIZE) {
        // Load K[n] fully
        float k_vals[BDIM];
        #pragma unroll
        for (int fd = 0; fd < BDIM; fd++) {
            k_vals[fd] = bf162f(K_bh[n * BDIM + fd]);
        }
        
        // Loop over all m positions
        for (int m = 0; m < S; m++) {
            // Compute dot product Q[m] · K[n]
            float score = 0.f;
            #pragma unroll
            for (int fd = 0; fd < BDIM; fd += 4) {
                float q0 = bf162f(Q_bh[m * BDIM + fd]);
                float q1 = bf162f(Q_bh[m * BDIM + fd + 1]);
                float q2 = bf162f(Q_bh[m * BDIM + fd + 2]);
                float q3 = bf162f(Q_bh[m * BDIM + fd + 3]);
                score += q0*k_vals[fd] + q1*k_vals[fd+1] + q2*k_vals[fd+2] + q3*k_vals[fd+3];
            }
            score *= inv_sqrt_d;
            
            float attn = expf(score - L_bh[m]);
            
            // Accumulate attn * dO[m] into dV[n] using atomics
            #pragma unroll
            for (int fd = 0; fd < BDIM; fd += 2) {
                atomic_add_bf16(&dV_bh[n * BDIM + fd], 
                    f2bf16(attn * bf162f(dO_bh[m * BDIM + fd])));
                atomic_add_bf16(&dV_bh[n * BDIM + fd + 1], 
                    f2bf16(attn * bf162f(dO_bh[m * BDIM + fd + 1])));
            }
        }
    }
    __syncthreads();
    
    // Phase 2: Compute corr[m] = sum_n attn[m,n] * (dO[m]·V[n])
    // Store corr values in shared memory
    extern __shared__ char smem[];
    float* s_corr = reinterpret_cast<float*>(smem);
    
    // Each thread computes corr[m] for its assigned m positions
    for (int m = tid; m < S; m += BLOCK_SIZE) {
        float corr = 0.f;
        float lm = L_bh[m];
        
        // Load Q[m] and dO[m] once
        float qm[BDIM];
        float dom[BDIM];
        #pragma unroll
        for (int fd = 0; fd < BDIM; fd++) {
            qm[fd] = bf162f(Q_bh[m * BDIM + fd]);
            dom[fd] = bf162f(dO_bh[m * BDIM + fd]);
        }
        
        for (int n = 0; n < S; n++) {
            // score = Q[m] · K[n] / sqrt(d)
            float score = 0.f;
            #pragma unroll
            for (int fd = 0; fd < BDIM; fd += 4) {
                float k0 = bf162f(K_bh[n * BDIM + fd]);
                float k1 = bf162f(K_bh[n * BDIM + fd + 1]);
                float k2 = bf162f(K_bh[n * BDIM + fd + 2]);
                float k3 = bf162f(K_bh[n * BDIM + fd + 3]);
                score += qm[fd]*k0 + qm[fd+1]*k1 + qm[fd+2]*k2 + qm[fd+3]*k3;
            }
            score *= inv_sqrt_d;
            
            // dov = dO[m] · V[n]
            float dov = 0.f;
            #pragma unroll
            for (int fd = 0; fd < BDIM; fd += 4) {
                float v0 = bf162f(V_bh[n * BDIM + fd]);
                float v1 = bf162f(V_bh[n * BDIM + fd + 1]);
                float v2 = bf162f(V_bh[n * BDIM + fd + 2]);
                float v3 = bf162f(V_bh[n * BDIM + fd + 3]);
                dov += dom[fd]*v0 + dom[fd+1]*v1 + dom[fd+2]*v2 + dom[fd+3]*v3;
            }
            
            float attn = expf(score - lm);
            corr += attn * dov;
        }
        s_corr[m] = corr;
    }
    __syncthreads();
    
    // Phase 3 & 4: Compute dQ[m] and dK[n] using corr
    for (int m = tid; m < S; m += BLOCK_SIZE) {
        float corr = s_corr[m];
        float lm = L_bh[m];
        
        float qm[BDIM];
        float dom[BDIM];
        #pragma unroll
        for (int fd = 0; fd < BDIM; fd++) {
            qm[fd] = bf162f(Q_bh[m * BDIM + fd]);
            dom[fd] = bf162f(dO_bh[m * BDIM + fd]);
        }
        
        for (int n = 0; n < S; n++) {
            // Recompute score and dov
            float score = 0.f;
            float dov = 0.f;
            #pragma unroll
            for (int fd = 0; fd < BDIM; fd += 4) {
                float k0 = bf162f(K_bh[n * BDIM + fd]);
                float k1 = bf162f(K_bh[n * BDIM + fd + 1]);
                float k2 = bf162f(K_bh[n * BDIM + fd + 2]);
                float k3 = bf162f(K_bh[n * BDIM + fd + 3]);
                
                float v0 = bf162f(V_bh[n * BDIM + fd]);
                float v1 = bf162f(V_bh[n * BDIM + fd + 1]);
                float v2 = bf162f(V_bh[n * BDIM + fd + 2]);
                float v3 = bf162f(V_bh[n * BDIM + fd + 3]);
                
                score += qm[fd]*k0 + qm[fd+1]*k1 + qm[fd+2]*k2 + qm[fd+3]*k3;
                dov   += dom[fd]*v0 + dom[fd+1]*v1 + dom[fd+2]*v2 + dom[fd+3]*v3;
            }
            score *= inv_sqrt_d;
            
            float attn = expf(score - lm);
            float dscore = attn * (dov - corr);
            
            // dQ[m] += dscore * K[n] / sqrt(d)
            #pragma unroll
            for (int fd = 0; fd < BDIM; fd += 2) {
                float kfd = bf162f(K_bh[n * BDIM + fd]);
                float kfd1 = bf162f(K_bh[n * BDIM + fd + 1]);
                atomic_add_bf16(&dQ_bh[m * BDIM + fd], 
                    f2bf16(dscore * kfd * inv_sqrt_d));
                atomic_add_bf16(&dQ_bh[m * BDIM + fd + 1], 
                    f2bf16(dscore * kfd1 * inv_sqrt_d));
            }
            
            // dK[n] += dscore * Q[m] / sqrt(d)
            #pragma unroll
            for (int fd = 0; fd < BDIM; fd += 2) {
                atomic_add_bf16(&dK_bh[n * BDIM + fd], 
                    f2bf16(dscore * qm[fd] * inv_sqrt_d));
                atomic_add_bf16(&dK_bh[n * BDIM + fd + 1], 
                    f2bf16(dscore * qm[fd+1] * inv_sqrt_d));
            }
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    constexpr int B = 4;
    constexpr int H = 48;
    constexpr int d = 128;
    int64_t S = Q.size(2);  // Sequence length
    
    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* dO_data = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_data = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQ_data = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_data = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_data = static_cast<__nv_bfloat16*>(dV.data_ptr());
    
    int num_blocks = B * H;
    constexpr int threads = 256;
    int smem_size = (int)S * sizeof(float);  // Shared memory for corr values
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    mha_bwd_kernel<128, 256><<<num_blocks, threads, smem_size, stream>>>(
        Q_data, K_data, V_data, dO_data, L_data,
        dQ_data, dK_data, dV_data, B, H, (int)S);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_bwd::run);

}  // namespace tvm_ffi_mha_bwd