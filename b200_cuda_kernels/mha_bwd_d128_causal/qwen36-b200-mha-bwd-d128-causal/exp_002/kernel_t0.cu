#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cstdio>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

#define CUDA_CHECK(call) do {                                         \
    cudaError_t _e = (call);                                          \
    if (_e != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                  \
                cudaGetErrorString(_e), __FILE__, __LINE__);          \
        exit(1);                                                       \
    }                                                                  \
} while(0)

namespace mha_bwd_impl {

// Helper to atomically add to bfloat16 value (using atomicCAS loop)
__device__ __forceinline__ void atomicAddBF16(__nv_bfloat16* addr, float val) {
    unsigned int* addr_as_ui = (unsigned int*)((char*)addr - ((size_t)addr & 2));
    unsigned int old = *addr_as_ui;
    unsigned int assumed;
    do {
        assumed = old;
        __nv_bfloat16 old_val;
        old_val.__x = assumed & 0xFFFF;
        float f_old = __bfloat162float(old_val);
        float f_new = f_old + val;
        __nv_bfloat16 new_val = __float2bfloat16(f_new);
        old = (assumed & 0xFFFF0000) | new_val.__x;
        old = atomicCAS(addr_as_ui, assumed, old);
    } while (assumed != old);
}

__device__ __forceinline__ void atomicAddBF16Lo(__nv_bfloat16* addr, float val) {
    unsigned int* addr_as_ui = (unsigned int*)((char*)addr - ((size_t)addr & 2));
    unsigned int old = *addr_as_ui;
    unsigned int assumed;
    do {
        assumed = old;
        __nv_bfloat16 old_val;
        old_val.__x = (assumed >> 16) & 0xFFFF;
        float f_old = __bfloat162float(old_val);
        float f_new = f_old + val;
        __nv_bfloat16 new_val = __float2bfloat16(f_new);
        old = (assumed & 0x0000FFFF) | ((unsigned int)new_val.__x << 16);
        old = atomicCAS(addr_as_ui, assumed, old);
    } while (assumed != old);
}

// Kernel to clear output buffers to zero
__global__ void mha_bwd_clear_kernel(
    __nv_bfloat16* dQ, __nv_bfloat16* dK, __nv_bfloat16* dV,
    int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total_elements; i += stride) {
        dQ[i] = __float2bfloat16(0.f);
        dK[i] = __float2bfloat16(0.f);
        dV[i] = __float2bfloat16(0.f);
    }
}

// Main backward kernel: compute dQ, dK, dV
// Each block handles one (batch, head) pair
__global__ void mha_bwd_main_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    int B, int H, int S, int D)
{
    int bh = blockIdx.x * blockDim.x + threadIdx.x;
    int total_bh = B * H;
    
    for (; bh < total_bh; bh += blockDim.x * gridDim.x) {
        int b = bh / H;
        int h = bh % H;
        
        size_t base = ((size_t)b * H + h) * S * D;
        float inv_sqrt_d = rsqrtf((float)D);
        int tid = threadIdx.x;
        int nthreads = blockDim.x;
        
        // Pointer to L for this (b, h)
        const float* L_bh = L + (size_t)b * H * S + h * S;
        
        // ---- Phase 1: Compute dQ ----
        // dQ[q, d] = sum_{k<=q} P[q,k] * K[k, d] / sqrt(d)
        // Each thread handles multiple q indices in grid-stride
        for (int q_idx = tid; q_idx < S; q_idx += nthreads) {
            // Load Q[q_idx, :] as float
            const __nv_bfloat16* q_src = Q + base + (size_t)q_idx * D;
            float q_f[D];
            #pragma unroll
            for (int d = 0; d < D; d++) {
                q_f[d] = __bfloat162float(q_src[d]);
            }
            
            float lse_q = L_bh[q_idx];
            
            // Accumulate dQ[q_idx, :] by iterating over all k <= q_idx
            for (int k_idx = 0; k_idx <= q_idx && k_idx < S; k_idx++) {
                // Load K[k_idx, :] as float
                const __nv_bfloat16* k_src = K + base + (size_t)k_idx * D;
                float score = 0.f;
                
                // Compute dot product Q[q] · K[k]
                for (int d = 0; d < D; d += 4) {
                    float k0 = __bfloat162float(k_src[d]);
                    float k1 = __bfloat162float(k_src[d+1]);
                    float k2 = __bfloat162float(k_src[d+2]);
                    float k3 = __bfloat162float(k_src[d+3]);
                    score += q_f[d]*k0 + q_f[d+1]*k1 + q_f[d+2]*k2 + q_f[d+3]*k3;
                }
                
                // Scale and compute attention weight
                score *= inv_sqrt_d;
                float attn = expf(score - lse_q);
                
                // Add contribution: dQ[q, d] += attn * K[k, d]
                __nv_bfloat16* dq_dst = dQ + base + (size_t)q_idx * D;
                for (int d = 0; d < D; d += 2) {
                    float k0 = __bfloat162float(k_src[d]);
                    float k1 = __bfloat162float(k_src[d+1]);
                    if ((size_t)&dq_dst[d] & 2) {
                        atomicAddBF16Lo(&dq_dst[d], attn * k0);
                        atomicAddBF16Lo(&dq_dst[d+1], attn * k1);
                    } else {
                        atomicAddBF16(&dq_dst[d], attn * k0);
                        atomicAddBF16(&dq_dst[d+1], attn * k1);
                    }
                }
            }
        }
        
        // ---- Phase 2: Compute dK ----
        // dK[k, d] = sum_{q>=k} P[q,k] * Q[q, d] / sqrt(d)
        for (int k_idx = tid; k_idx < S; k_idx += nthreads) {
            const __nv_bfloat16* k_src = K + base + (size_t)k_idx * D;
            float k_f[D];
            #pragma unroll
            for (int d = 0; d < D; d++) {
                k_f[d] = __bfloat162float(k_src[d]);
            }
            
            // Iterate over all q >= k_idx
            for (int q_idx = k_idx; q_idx < S; q_idx++) {
                // Load Q[q_idx, :]
                const __nv_bfloat16* q_src = Q + base + (size_t)q_idx * D;
                float score = 0.f;
                
                for (int d = 0; d < D; d += 4) {
                    float q0 = __bfloat162float(q_src[d]);
                    float q1 = __bfloat162float(q_src[d+1]);
                    float q2 = __bfloat162float(q_src[d+2]);
                    float q3 = __bfloat162float(q_src[d+3]);
                    score += q0*k_f[d] + q1*k_f[d+1] + q2*k_f[d+2] + q3*k_f[d+3];
                }
                
                score *= inv_sqrt_d;
                float attn = expf(score - L_bh[q_idx]);
                
                // Add contribution: dK[k, d] += attn * Q[q, d]
                __nv_bfloat16* dk_dst = dK + base + (size_t)k_idx * D;
                for (int d = 0; d < D; d += 2) {
                    float q0 = __bfloat162float(q_src[d]);
                    float q1 = __bfloat162float(q_src[d+1]);
                    if ((size_t)&dk_dst[d] & 2) {
                        atomicAddBF16Lo(&dk_dst[d], attn * q0);
                        atomicAddBF16Lo(&dk_dst[d+1], attn * q1);
                    } else {
                        atomicAddBF16(&dk_dst[d], attn * q0);
                        atomicAddBF16(&dk_dst[d+1], attn * q1);
                    }
                }
            }
        }
        
        // ---- Phase 3: Compute dV ----
        // dV[k, d] = sum_{q>=k} P[q,k] * dO[q, d]
        // Each thread handles different k indices - no overlap!
        for (int k_idx = tid; k_idx < S; k_idx += nthreads) {
            // Local accumulator for this k_idx
            float dV_local[D];
            #pragma unroll
            for (int d = 0; d < D; d++) {
                dV_local[d] = 0.f;
            }
            
            // Iterate over all q >= k_idx
            for (int q_idx = k_idx; q_idx < S; q_idx++) {
                const __nv_bfloat16* q_src = Q + base + (size_t)q_idx * D;
                const __nv_bfloat16* k_src = K + base + (size_t)k_idx * D;
                
                float score = 0.f;
                for (int d = 0; d < D; d += 4) {
                    float q0 = __bfloat162float(q_src[d]);
                    float q1 = __bfloat162float(q_src[d+1]);
                    float q2 = __bfloat162float(q_src[d+2]);
                    float q3 = __bfloat162float(q_src[d+3]);
                    float k0 = __bfloat162float(k_src[d]);
                    float k1 = __bfloat162float(k_src[d+1]);
                    float k2 = __bfloat162float(k_src[d+2]);
                    float k3 = __bfloat162float(k_src[d+3]);
                    score += q0*k0 + q1*k1 + q2*k2 + q3*k3;
                }
                
                score *= inv_sqrt_d;
                float attn = expf(score - L_bh[q_idx]);
                
                // Accumulate dV[k, d] += attn * dO[q, d]
                const __nv_bfloat16* do_src = dO + base + (size_t)q_idx * D;
                for (int d = 0; d < D; d++) {
                    dV_local[d] += attn * __bfloat162float(do_src[d]);
                }
            }
            
            // Write dV[k_idx, :] - no contention since each thread owns unique k_idx
            __nv_bfloat16* dv_dst = dV + base + (size_t)k_idx * D;
            for (int d = 0; d < D; d++) {
                dv_dst[d] = __float2bfloat16(dV_local[d]);
            }
        }
    }
}

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
    
    // Extract shapes: all have shape [B, H, S, d] except L which is [B, H, S]
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);
    
    // Cast pointers
    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* O_ptr = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dO_ptr = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_ptr = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQ_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());
    
    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    // Total number of bf16 elements in each output
    int64_t total_elements = B * H * S * D;
    
    // Step 1: Clear output buffers to zero
    int clear_threads = 512;
    int clear_blocks = std::min((int)((total_elements + clear_threads - 1) / clear_threads), 65535);
    mha_bwd_clear_kernel<<<clear_blocks, clear_threads, 0, stream>>>(
        dQ_ptr, dK_ptr, dV_ptr, (int)total_elements);
    CUDA_CHECK(cudaGetLastError());
    
    // Step 2: Main backward kernel
    // Launch: one block per (B, H) pair, with grid-stride for excess
    int threads_per_block = 256;
    int num_bh = (int)(B * H);
    int main_blocks = std::min(num_bh, 65535);
    
    // Configure cluster launch for better SM occupancy on Blackwell
    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(main_blocks, 1, 1);
    config.blockDim = dim3(threads_per_block, 1, 1);
    config.sharedMemBytes = 0;
    config.stream = stream;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    cudaLaunchKernelEx(&config, mha_bwd_main_kernel,
        Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
        dQ_ptr, dK_ptr, dV_ptr,
        (int)B, (int)H, (int)S, (int)D);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_impl::run);

}  // namespace mha_bwd_impl