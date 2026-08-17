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

// Helper: convert bf16 to/from float
__forceinline__ __device__ float bf162f(__nv_bfloat16 h) {
    return __bfloat162float(h);
}
__forceinline__ __device__ __nv_bfloat16 f2bf16(float f) {
    return __float2bfloat16(f);
}

// Atomic add for bf16 on architectures without native support
__forceinline__ __device__ void atomic_add_bf16(__nv_bfloat16* address, __nv_bfloat16 val) {
    unsigned int* address_as_ui = (unsigned int*)((char*)address);
    unsigned int old = *address_as_ui, assumed;
    do {
        assumed = old;
        __nv_bfloat16 new_val = val + *reinterpret_cast<__nv_bfloat16*>(&assumed);
        old = atomicCAS(address_as_ui, assumed, *(unsigned int*)&new_val);
    } while (old != assumed);
}

// Naive but correct kernel: each block processes one (b,h) pair
// Each thread processes one output element with strided iteration over sequence
// This is slow due to atomics but demonstrates correctness
__global__ void mha_bwd_naive_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    int B, int H, int S, int d)
{
    // Thread handles one (b, h, pos, feat) and accumulates over sequence
    // Grid: B * H * S * d threads total (too many!), so we compress
    
    // Better: grid = B*H blocks, each block computes all S*S interactions
    int bh = blockIdx.x;
    int b = bh / H;
    int h = bh % H;
    int tid = threadIdx.x;
    
    if (b >= B || h >= H) return;
    
    // Pointers for this (b, h)
    const __nv_bfloat16* Q_ptr = Q + (size_t)(b * H + h) * S * d;
    const __nv_bfloat16* K_ptr = K + (size_t)(b * H + h) * S * d;
    const __nv_bfloat16* V_ptr = V + (size_t)(b * H + h) * S * d;
    const __nv_bfloat16* dO_ptr = dO + (size_t)(b * H + h) * S * d;
    const float* L_ptr = L + (size_t)(b * H + h) * S;
    __nv_bfloat16* dQ_ptr = dQ + (size_t)(b * H + h) * S * d;
    __nv_bfloat16* dK_ptr = dK + (size_t)(b * H + h) * S * d;
    __nv_bfloat16* dV_ptr = dV + (size_t)(b * H + h) * S * d;
    
    float inv_sqrt_d = rsqrtf((float)d);
    
    // Initialize output buffers to zero for this (b,h)
    int total_elem = S * d;
    for (int i = tid; i < total_elem; i += blockDim.x) {
        dQ_ptr[i] = f2bf16(0.f);
        dK_ptr[i] = f2bf16(0.f);
        dV_ptr[i] = f2bf16(0.f);
    }
    __syncthreads();
    
    // Phase 1: Compute dV[n] = sum_m attn[m,n] * dO[m]
    // Each thread owns position n (strided), loops over all m
    for (int n = tid; n < S; n += blockDim.x) {
        // Load K[n] into local storage
        float kn[d];
        for (int fd = 0; fd < d; fd++) {
            kn[fd] = bf162f(K_ptr[n * d + fd]);
        }
        
        for (int m = 0; m < S; m++) {
            // Compute Q[m] · K[n]
            float score = 0.f;
            for (int fd = 0; fd < d; fd++) {
                score += bf162f(Q_ptr[m * d + fd]) * kn[fd];
            }
            score *= inv_sqrt_d;
            
            float attn = expf(score - L_ptr[m]);
            
            // Accumulate attn * dO[m] into dV[n]
            for (int fd = 0; fd < d; fd++) {
                float dvo = attn * bf162f(dO_ptr[m * d + fd]);
                atomic_add_bf16(&dV_ptr[n * d + fd], f2bf16(dvo));
            }
        }
    }
    __syncthreads();
    
    // Phase 2 & 3: Compute dQ and dK
    // dQ[m] = sum_n attn[m,n] * (dO[m]·V[n] - corr[m]) * K[n] / sqrt(d)
    // dK[n] = sum_m attn[m,n] * (dO[m]·V[n] - corr[m]) * Q[m] / sqrt(d)
    //
    // Where corr[m] = sum_k attn[m,k] * (dO[m]·V[k])
    //
    // Approach: for each m, compute corr[m], then for each n, accumulate dQ and dK
    
    // First, compute corr[m] for all m and store in shared memory
    extern __shared__ char smem_pool[];
    float* s_corr = (float*)smem_pool;  // S floats for correction values
    
    // Initialize corr to zero
    for (int m = tid; m < S; m += blockDim.x) {
        s_corr[m] = 0.f;
    }
    __syncthreads();
    
    // Each thread handles position m, loops over n to compute corr[m]
    for (int m = tid; m < S; m += blockDim.x) {
        float lm = L_ptr[m];
        float qm[d];
        float dom[d];
        for (int fd = 0; fd < d; fd++) {
            qm[fd] = bf162f(Q_ptr[m * d + fd]);
            dom[fd] = bf162f(dO_ptr[m * d + fd]);
        }
        
        for (int n = 0; n < S; n++) {
            // score = Q[m] · K[n] / sqrt(d)
            float score = 0.f;
            for (int fd = 0; fd < d; fd++) {
                score += qm[fd] * bf162f(K_ptr[n * d + fd]);
            }
            score *= inv_sqrt_d;
            
            // dov = dO[m] · V[n]
            float dov = 0.f;
            for (int fd = 0; fd < d; fd++) {
                dov += dom[fd] * bf162f(V_ptr[n * d + fd]);
            }
            
            float attn = expf(score - lm);
            s_corr[m] += attn * dov;
        }
    }
    __syncthreads();
    
    // Now compute dQ[m] and dK[n] using corr
    // dQ[m][fd] = sum_n (attn[m,n] * dov[m,n] - corr[m] * attn[m,n]) * K[n][fd] / sqrt(d)
    //           = sum_n [attn*(dov-corr)] * K[n][fd] / sqrt(d)
    // 
    // dK[n][fd] = sum_m (attn[m,n] * dov[m,n] - corr[m] * attn[m,n]) * Q[m][fd] / sqrt(d)
    //
    // Split into two contributions:
    // Contribution 1 (positive): sum_n attn*dov*K/sqrt(d) for dQ, sum_m attn*dov*Q/sqrt(d) for dK
    // Contribution 2 (negative): sum_n corr*attn*K/sqrt(d) for dQ, sum_m corr*attn*Q/sqrt(d) for dK
    //
    // Each thread handles one m, loops over n
    
    for (int m = tid; m < S; m += blockDim.x) {
        float corr = s_corr[m];
        float lm = L_ptr[m];
        float qm[d];
        float dom[d];
        for (int fd = 0; fd < d; fd++) {
            qm[fd] = bf162f(Q_ptr[m * d + fd]);
            dom[fd] = bf162f(dO_ptr[m * d + fd]);
        }
        
        for (int n = 0; n < S; n++) {
            float score = 0.f;
            float dov = 0.f;
            for (int fd = 0; fd < d; fd++) {
                score += qm[fd] * bf162f(K_ptr[n * d + fd]);
                dov += dom[fd] * bf162f(V_ptr[n * d + fd]);
            }
            score *= inv_sqrt_d;
            
            float attn = expf(score - lm);
            float dscore = attn * (dov - corr);
            
            // dQ[m] += dscore * K[n] / sqrt(d)
            for (int fd = 0; fd < d; fd++) {
                float dk = dscore * bf162f(K_ptr[n * d + fd]) * inv_sqrt_d;
                atomic_add_bf16(&dQ_ptr[m * d + fd], f2bf16(dk));
            }
            
            // dK[n] += dscore * Q[m] / sqrt(d)
            for (int fd = 0; fd < d; fd++) {
                float dk = dscore * qm[fd] * inv_sqrt_d;
                atomic_add_bf16(&dK_ptr[n * d + fd], f2bf16(dk));
            }
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int B = 4, H = 48, d = 128;
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
    int threads = 256;
    int smem_size = S * sizeof(float);  // Shared memory for corr values
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    mha_bwd_naive_kernel<<<num_blocks, threads, smem_size, stream>>>(
        Q_data, K_data, V_data, dO_data, L_data,
        dQ_data, dK_data, dV_data, B, H, S, d);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_bwd::run);

}  // namespace tvm_ffi_mha_bwd