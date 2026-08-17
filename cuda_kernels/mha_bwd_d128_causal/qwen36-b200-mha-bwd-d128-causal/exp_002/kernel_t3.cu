#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cstdio>
#include <cassert>
#include <algorithm>
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

// Clear output buffers to zero using vectorized stores
__global__ void mha_clear_kernel(__nv_bfloat16* dQ, __nv_bfloat16* dK, __nv_bfloat16* dV, int64_t total) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total; i += stride) {
        dQ[i] = __float2bfloat16(0.f);
        dK[i] = __float2bfloat16(0.f);
        dV[i] = __float2bfloat16(0.f);
    }
}

template<int D>
__global__ __launch_bounds__(256)
void dQ_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ_out,
    int B, int H, int S)
{
    // Each thread computes dQ[b,h,q,:] for one (b,h,q) position.
    // No contention: each thread owns unique output location.
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float inv_sqrt_d = rsqrtf((float)D);
    int total_positions = B * H * S;
    
    for (; gid < total_positions; gid += stride) {
        int bh = gid / S;
        int q = gid % S;
        int b = bh / H;
        int h = bh % H;
        size_t base = ((size_t)b * H + h) * (size_t)S * D;
        
        const float* L_bh = L + (size_t)bh * S;
        const __nv_bfloat16* q_row = Q + base + (size_t)q * D;
        float lse_q = L_bh[q];
        
        // Process D in two chunks to reduce register pressure
        // Chunk 0: d=0..63, Chunk 1: d=64..127
        float acc0[D/2];
        float acc1[D/2];
        #pragma unroll
        for (int d = 0; d < D/2; d++) { acc0[d] = 0.f; acc1[d] = 0.f; }
        
        // Load Q once into registers for dot product
        float qf[D];
        #pragma unroll
        for (int d = 0; d < D; d++) qf[d] = __bfloat162float(q_row[d]);
        
        // Accumulate dQ[q,d] = sum_{k<=q} P[q,k] * K[k,d] / sqrt(d)
        for (int k = 0; k <= q; k++) {
            const __nv_bfloat16* k_row = K + base + (size_t)k * D;
            
            // Compute dot product score = Q[q] . K[k]
            float score = 0.f;
            #pragma unroll
            for (int d = 0; d < D; d += 4) {
                score += qf[d]     * __bfloat162float(k_row[d]);
                score += qf[d + 1] * __bfloat162float(k_row[d + 1]);
                score += qf[d + 2] * __bfloat162float(k_row[d + 2]);
                score += qf[d + 3] * __bfloat162float(k_row[d + 3]);
            }
            
            float attn = expf(score * inv_sqrt_d - lse_q);
            
            // Accumulate dQ[q,d] += attn * K[k,d]
            #pragma unroll
            for (int d = 0; d < D/2; d++) {
                acc0[d] += attn * __bfloat162float(k_row[d]);
                acc1[d] += attn * __bfloat162float(k_row[d + D/2]);
            }
        }
        
        // Write results
        __nv_bfloat16* dst = dQ_out + base + (size_t)q * D;
        #pragma unroll
        for (int d = 0; d < D/2; d++) {
            dst[d]       = __float2bfloat16(acc0[d]);
            dst[d + D/2] = __float2bfloat16(acc1[d]);
        }
    }
}

template<int D>
__global__ __launch_bounds__(256)
void dK_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dK_out,
    int B, int H, int S)
{
    // Each thread computes dK[b,h,k,:] for one (b,h,k) position.
    // dK[k,d] = sum_{q>=k} P[q,k] * Q[q,d] / sqrt(d)
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float inv_sqrt_d = rsqrtf((float)D);
    int total_positions = B * H * S;
    
    for (; gid < total_positions; gid += stride) {
        int bh = gid / S;
        int k = gid % S;
        int b = bh / H;
        int h = bh % H;
        size_t base = ((size_t)b * H + h) * (size_t)S * D;
        
        const float* L_bh = L + (size_t)bh * S;
        const __nv_bfloat16* k_row = K + base + (size_t)k * D;
        
        float acc0[D/2];
        float acc1[D/2];
        #pragma unroll
        for (int d = 0; d < D/2; d++) { acc0[d] = 0.f; acc1[d] = 0.f; }
        
        // Load K[k,:] into registers
        float kf[D];
        #pragma unroll
        for (int d = 0; d < D; d++) kf[d] = __bfloat162float(k_row[d]);
        
        // Accumulate dK[k,d] = sum_{q>=k} P[q,k] * Q[q,d]
        for (int q = k; q < S; q++) {
            const __nv_bfloat16* q_row = Q + base + (size_t)q * D;
            
            float score = 0.f;
            #pragma unroll
            for (int d = 0; d < D; d += 4) {
                score += __bfloat162float(q_row[d])     * kf[d];
                score += __bfloat162float(q_row[d + 1]) * kf[d + 1];
                score += __bfloat162float(q_row[d + 2]) * kf[d + 2];
                score += __bfloat162float(q_row[d + 3]) * kf[d + 3];
            }
            
            float attn = expf(score * inv_sqrt_d - L_bh[q]);
            
            #pragma unroll
            for (int d = 0; d < D/2; d++) {
                acc0[d] += attn * __bfloat162float(q_row[d]);
                acc1[d] += attn * __bfloat162float(q_row[d + D/2]);
            }
        }
        
        __nv_bfloat16* dst = dK_out + base + (size_t)k * D;
        #pragma unroll
        for (int d = 0; d < D/2; d++) {
            dst[d]       = __float2bfloat16(acc0[d]);
            dst[d + D/2] = __float2bfloat16(acc1[d]);
        }
    }
}

template<int D>
__global__ __launch_bounds__(256)
void dV_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dV_out,
    int B, int H, int S)
{
    // Each thread computes dV[b,h,k,:] for one (b,h,k) position.
    // dV[k,d] = sum_{q>=k} P[q,k] * dO[q,d]
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float inv_sqrt_d = rsqrtf((float)D);
    int total_positions = B * H * S;
    
    for (; gid < total_positions; gid += stride) {
        int bh = gid / S;
        int k = gid % S;
        int b = bh / H;
        int h = bh % H;
        size_t base = ((size_t)b * H + h) * (size_t)S * D;
        
        const float* L_bh = L + (size_t)bh * S;
        const __nv_bfloat16* k_row = K + base + (size_t)k * D;
        
        float acc0[D/2];
        float acc1[D/2];
        #pragma unroll
        for (int d = 0; d < D/2; d++) { acc0[d] = 0.f; acc1[d] = 0.f; }
        
        // Load K[k,:] into registers
        float kf[D];
        #pragma unroll
        for (int d = 0; d < D; d++) kf[d] = __bfloat162float(k_row[d]);
        
        // Accumulate dV[k,d] = sum_{q>=k} P[q,k] * dO[q,d]
        for (int q = k; q < S; q++) {
            const __nv_bfloat16* q_row = Q + base + (size_t)q * D;
            const __nv_bfloat16* do_row = dO + base + (size_t)q * D;
            
            float score = 0.f;
            #pragma unroll
            for (int d = 0; d < D; d += 4) {
                score += __bfloat162float(q_row[d])     * kf[d];
                score += __bfloat162float(q_row[d + 1]) * kf[d + 1];
                score += __bfloat162float(q_row[d + 2]) * kf[d + 2];
                score += __bfloat162float(q_row[d + 3]) * kf[d + 3];
            }
            
            float attn = expf(score * inv_sqrt_d - L_bh[q]);
            
            #pragma unroll
            for (int d = 0; d < D/2; d++) {
                acc0[d] += attn * __bfloat162float(do_row[d]);
                acc1[d] += attn * __bfloat162float(do_row[d + D/2]);
            }
        }
        
        __nv_bfloat16* dst = dV_out + base + (size_t)k * D;
        #pragma unroll
        for (int d = 0; d < D/2; d++) {
            dst[d]       = __float2bfloat16(acc0[d]);
            dst[d + D/2] = __float2bfloat16(acc1[d]);
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
    int64_t D_val = Q.size(3);
    int D = static_cast<int>(D_val);
    
    assert(D == 128 && "D must be 128 for this kernel");
    
    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* dO_ptr = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_ptr = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQ_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());
    
    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    int64_t total_elems = B * H * S * D;
    int total_positions = static_cast<int>(B * H * S);
    
    // Step 1: Clear outputs (may not strictly be needed since each thread writes, but safe)
    int clear_threads = 512;
    int clear_blocks = std::min(static_cast<int>((total_elems + clear_threads - 1) / clear_threads), 65535);
    mha_clear_kernel<<<clear_blocks, clear_threads, 0, stream>>>(dQ_ptr, dK_ptr, dV_ptr, total_elems);
    CUDA_CHECK(cudaGetLastError());
    
    // Step 2: Launch gradient kernels
    // Each block handles 128 unique (b,h,position) triples.
    // Grid strides handle cases where total_positions > blocks*threads.
    int threads = 128;
    int blocks = std::min((total_positions + threads - 1) / threads, 65535);
    
    dim3 grid(blocks, 1, 1);
    dim3 blk(threads, 1, 1);
    
    dQ_kernel<128><<<grid, blk, 0, stream>>>(Q_ptr, K_ptr, L_ptr, dQ_ptr,
        static_cast<int>(B), static_cast<int>(H), static_cast<int>(S));
    CUDA_CHECK(cudaGetLastError());
    
    dK_kernel<128><<<grid, blk, 0, stream>>>(Q_ptr, K_ptr, L_ptr, dK_ptr,
        static_cast<int>(B), static_cast<int>(H), static_cast<int>(S));
    CUDA_CHECK(cudaGetLastError());
    
    dV_kernel<128><<<grid, blk, 0, stream>>>(Q_ptr, K_ptr, dO_ptr, L_ptr, dV_ptr,
        static_cast<int>(B), static_cast<int>(H), static_cast<int>(S));
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_impl::run);

}  // namespace mha_bwd_impl