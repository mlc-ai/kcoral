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

__device__ __forceinline__ void bf16_atomic_add_aligned(__nv_bfloat16* addr, float val) {
    unsigned int old = *(reinterpret_cast<unsigned int*>(addr));
    unsigned int assumed;
    do {
        assumed = old;
        __nv_bfloat16 old_val;
        memcpy(&old_val, &assumed, sizeof(old_val));
        float f_new = __bfloat162float(old_val) + val;
        __nv_bfloat16 new_val = __float2bfloat16(f_new);
        unsigned int new_bits;
        memcpy(&new_bits, &new_val, sizeof(new_bits));
        old = atomicCAS((unsigned int*)addr, assumed, new_bits);
    } while (assumed != old);
}

// Clear all output buffers to zero
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

// Kernel for dQ: dQ[b,h,q,d] = sum_{k<=q} P[b,h,q,k] * K[b,h,k,d] / sqrt(d)
// Each thread computes all elements for one (b,h,q) row, no write conflicts
__global__ void mha_bwd_dq_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    int B, int H, int S, int D)
{
    int bh = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_bh = blockDim.x * gridDim.x;
    
    for (; bh < B * H; bh += stride_bh) {
        int b = bh / H;
        int h = bh % H;
        
        size_t base_QK = ((size_t)b * H + h) * S * D;
        float inv_sqrt_d = rsqrtf((float)D);
        
        // Pointer to L for this (b,h): shape [B, H, S] -> offset = b*H*S + h*S
        const float* L_bh = L + (size_t)b * H * S + h * S;
        
        // Process each q in this (b,h) pair
        // Each thread does one q, grid-stride for excess
        for (int q = 0; q < S; q++) {
            float lse_q = L_bh[q];
            const __nv_bfloat16* q_row = Q + base_QK + (size_t)q * D;
            __nv_bfloat16* dq_row = dQ + base_QK + (size_t)q * D;
            
            // Accumulate dQ[q,d] for each k <= q
            for (int k = 0; k <= q; k++) {
                const __nv_bfloat16* k_row = K + base_QK + (size_t)k * D;
                
                // Compute score = Q[q] . K[k]
                float score = 0.f;
                for (int d = 0; d < D; d += 4) {
                    float q0 = __bfloat162float(q_row[d]);
                    float q1 = __bfloat162float(q_row[d+1]);
                    float q2 = __bfloat162float(q_row[d+2]);
                    float q3 = __bfloat162float(q_row[d+3]);
                    float k0 = __bfloat162float(k_row[d]);
                    float k1 = __bfloat162float(k_row[d+1]);
                    float k2 = __bfloat162float(k_row[d+2]);
                    float k3 = __bfloat162float(k_row[d+3]);
                    score += q0*k0 + q1*k1 + q2*k2 + q3*k3;
                }
                
                float attn = expf(score * inv_sqrt_d - lse_q);
                
                // dQ[q,d] += attn * K[k,d]
                for (int d = 0; d < D; d += 2) {
                    float k0 = __bfloat162float(k_row[d]);
                    float k1 = __bfloat162float(k_row[d+1]);
                    // Direct BF16 atomic add (we're aligned to 2-byte boundaries)
                    float cur = __bfloat162float(dq_row[d]) + attn * k0;
                    dq_row[d] = __float2bfloat16(cur);
                    cur = __bfloat162float(dq_row[d+1]) + attn * k1;
                    dq_row[d+1] = __float2bfloat16(cur);
                }
            }
        }
    }
}

// Kernel for dK: dK[b,h,k,d] = sum_{q>=k} P[b,h,q,k] * Q[b,h,q,d] / sqrt(d)
// Each thread computes all elements for one (b,h,k) row
__global__ void mha_bwd_dk_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dK,
    int B, int H, int S, int D)
{
    int bh = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_bh = blockDim.x * gridDim.x;
    
    for (; bh < B * H; bh += stride_bh) {
        int b = bh / H;
        int h = bh % H;
        
        size_t base_QK = ((size_t)b * H + h) * S * D;
        float inv_sqrt_d = rsqrtf((float)D);
        
        const float* L_bh = L + (size_t)b * H * S + h * S;
        
        for (int k = 0; k < S; k++) {
            const __nv_bfloat16* k_row = K + base_QK + (size_t)k * D;
            __nv_bfloat16* dk_row = dK + base_QK + (size_t)k * D;
            
            for (int q = k; q < S; q++) {
                const __nv_bfloat16* q_row = Q + base_QK + (size_t)q * D;
                
                float score = 0.f;
                for (int d = 0; d < D; d += 4) {
                    float q0 = __bfloat162float(q_row[d]);
                    float q1 = __bfloat162float(q_row[d+1]);
                    float q2 = __bfloat162float(q_row[d+2]);
                    float q3 = __bfloat162float(q_row[d+3]);
                    float k0 = __bfloat162float(k_row[d]);
                    float k1 = __bfloat162float(k_row[d+1]);
                    float k2 = __bfloat162float(k_row[d+2]);
                    float k3 = __bfloat162float(k_row[d+3]);
                    score += q0*k0 + q1*k1 + q2*k2 + q3*k3;
                }
                
                float attn = expf(score * inv_sqrt_d - L_bh[q]);
                
                for (int d = 0; d < D; d += 2) {
                    float q0 = __bfloat162float(q_row[d]);
                    float q1 = __bfloat162float(q_row[d+1]);
                    float cur = __bfloat162float(dk_row[d]) + attn * q0;
                    dk_row[d] = __float2bfloat16(cur);
                    cur = __bfloat162float(dk_row[d+1]) + attn * q1;
                    dk_row[d+1] = __float2bfloat16(cur);
                }
            }
        }
    }
}

// Kernel for dV: dV[b,h,k,d] = sum_{q>=k} P[b,h,q,k] * dO[b,h,q,d]
// Each thread computes all elements for one (b,h,k) row
__global__ void mha_bwd_dv_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dV,
    int B, int H, int S, int D)
{
    int bh = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_bh = blockDim.x * gridDim.x;
    
    for (; bh < B * H; bh += stride_bh) {
        int b = bh / H;
        int h = bh % H;
        
        size_t base = ((size_t)b * H + h) * S * D;
        float inv_sqrt_d = rsqrtf((float)D);
        
        const float* L_bh = L + (size_t)b * H * S + h * S;
        
        for (int k = 0; k < S; k++) {
            const __nv_bfloat16* k_row = K + base + (size_t)k * D;
            __nv_bfloat16* dv_row = dV + base + (size_t)k * D;
            
            for (int q = k; q < S; q++) {
                const __nv_bfloat16* q_row = Q + base + (size_t)q * D;
                const __nv_bfloat16* do_row = dO + base + (size_t)q * D;
                
                float score = 0.f;
                for (int d = 0; d < D; d += 4) {
                    float q0 = __bfloat162float(q_row[d]);
                    float q1 = __bfloat162float(q_row[d+1]);
                    float q2 = __bfloat162float(q_row[d+2]);
                    float q3 = __bfloat162float(q_row[d+3]);
                    float k0 = __bfloat162float(k_row[d]);
                    float k1 = __bfloat162float(k_row[d+1]);
                    float k2 = __bfloat162float(k_row[d+2]);
                    float k3 = __bfloat162float(k_row[d+3]);
                    score += q0*k0 + q1*k1 + q2*k2 + q3*k3;
                }
                
                float attn = expf(score * inv_sqrt_d - L_bh[q]);
                
                for (int d = 0; d < D; d++) {
                    float cur = __bfloat162float(dv_row[d]) + attn * __bfloat162float(do_row[d]);
                    dv_row[d] = __float2bfloat16(cur);
                }
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
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);
    
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
    
    int64_t total_elements = B * H * S * D;
    
    // Step 1: Clear outputs
    int clear_threads = 512;
    int clear_blocks = std::min((int)((total_elements + clear_threads - 1) / clear_threads), 65535);
    mha_bwd_clear_kernel<<<clear_blocks, clear_threads, 0, stream>>>(
        dQ_ptr, dK_ptr, dV_ptr, (int)total_elements);
    CUDA_CHECK(cudaGetLastError());
    
    // Step 2: Launch gradient kernels
    // One thread per (b,h) pair; grid-stride loops handle >num_threads (b,h) pairs
    int num_bh = (int)(B * H);
    int threads_per_block = 256;
    int main_blocks = std::min(num_bh, 65535);
    
    dim3 grid(main_blocks, 1, 1);
    dim3 block(threads_per_block, 1, 1);
    
    mha_bwd_dq_kernel<<<grid, block, 0, stream>>>(
        Q_ptr, K_ptr, L_ptr, dQ_ptr,
        (int)B, (int)H, (int)S, (int)D);
    CUDA_CHECK(cudaGetLastError());
    
    mha_bwd_dk_kernel<<<grid, block, 0, stream>>>(
        Q_ptr, K_ptr, L_ptr, dK_ptr,
        (int)B, (int)H, (int)S, (int)D);
    CUDA_CHECK(cudaGetLastError());
    
    mha_bwd_dv_kernel<<<grid, block, 0, stream>>>(
        Q_ptr, K_ptr, dO_ptr, L_ptr, dV_ptr,
        (int)B, (int)H, (int)S, (int)D);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_impl::run);

}  // namespace mha_bwd_impl