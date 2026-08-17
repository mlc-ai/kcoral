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

// Elementwise clear kernel
__global__ void mha_clear_kernel(__nv_bfloat16* out, int64_t total) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total; i += stride) {
        out[i] = __float2bfloat16(0.f);
    }
}

// Kernel: compute dQ element-by-element.
// Each thread computes ONE dQ[b,h,q,d] value.
// Global index maps to (output_index = elem_id / 3 + offset_for_grad_type)
// For simplicity, dQ threads get indices [0, BHSD)
template<int D>
__global__ __launch_bounds__(512)
void dQ_single_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const float* __restrict__ L_in,
    __nv_bfloat16* __restrict__ dQ_out,
    int B, int H, int S)
{
    // Each thread computes ONE dQ element.
    // Thread ID = b*H*S*D + h*S*D + q*D + d
    int64_t tid = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    int64_t total = (int64_t)B * H * S * D;
    
    float inv_sqrt_d = rsqrtf((float)D);
    
    for (int64_t eid = tid; eid < total; eid += stride) {
        int d = static_cast<int>(eid % D);
        int64_t rem = eid / D;
        int q = static_cast<int>(rem % S);
        int bh = static_cast<int>(rem / S);
        int b = bh / H;
        int h = bh % H;
        
        size_t base = ((size_t)b * H + h) * (size_t)S * D;
        float lse_q = L_in[(size_t)bh * S + q];
        
        const __nv_bfloat16* q_row = Q + base + (size_t)q * D;
        
        // Load Q[q,:] into registers
        float q_reg[D];
        #pragma unroll
        for (int dd = 0; dd < D; dd++) {
            q_reg[dd] = __bfloat162float(q_row[dd]);
        }
        
        // Accumulate dQ[q,d] = sum_{k=0}^{q} P[q,k] * K[k,d]
        float acc = 0.f;
        float kd_val;
        
        for (int k = 0; k <= q; k++) {
            const __nv_bfloat16* k_row = K + base + (size_t)k * D;
            
            // Compute score = Q[q] . K[k]
            float score = 0.f;
            #pragma unroll
            for (int dd = 0; dd < D; dd += 4) {
                score += q_reg[dd]     * __bfloat162float(k_row[dd]);
                score += q_reg[dd + 1] * __bfloat162float(k_row[dd + 1]);
                score += q_reg[dd + 2] * __bfloat162float(k_row[dd + 2]);
                score += q_reg[dd + 3] * __bfloat162float(k_row[dd + 3]);
            }
            
            float attn = expf(score * inv_sqrt_d - lse_q);
            
            // Get K[k,d] and accumulate
            kd_val = __bfloat162float(k_row[d]);
            acc += attn * kd_val;
        }
        
        dQ_out[eid] = __float2bfloat16(acc);
    }
}

// Kernel: compute dK element-by-element.
// dK[b,h,k,d] = sum_{q=k}^{S-1} P[q,k] * Q[q,d] / sqrt(d)
template<int D>
__global__ __launch_bounds__(512)
void dK_single_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const float* __restrict__ L_in,
    __nv_bfloat16* __restrict__ dK_out,
    int B, int H, int S)
{
    int64_t tid = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    int64_t total = (int64_t)B * H * S * D;
    
    float inv_sqrt_d = rsqrtf((float)D);
    
    for (int64_t eid = tid; eid < total; eid += stride) {
        int d = static_cast<int>(eid % D);
        int64_t rem = eid / D;
        int k = static_cast<int>(rem % S);
        int bh = static_cast<int>(rem / S);
        int b = bh / H;
        int h = bh % H;
        
        size_t base = ((size_t)b * H + h) * (size_t)S * D;
        
        const __nv_bfloat16* k_row = K + base + (size_t)k * D;
        
        // Load K[k,:] into registers
        float k_reg[D];
        #pragma unroll
        for (int dd = 0; dd < D; dd++) {
            k_reg[dd] = __bfloat162float(k_row[dd]);
        }
        
        float qd_val;
        float acc = 0.f;
        
        for (int q = k; q < S; q++) {
            const __nv_bfloat16* q_row = Q + base + (size_t)q * D;
            
            float score = 0.f;
            #pragma unroll
            for (int dd = 0; dd < D; dd += 4) {
                score += __bfloat162float(q_row[dd])     * k_reg[dd];
                score += __bfloat162float(q_row[dd + 1]) * k_reg[dd + 1];
                score += __bfloat162float(q_row[dd + 2]) * k_reg[dd + 2];
                score += __bfloat162float(q_row[dd + 3]) * k_reg[dd + 3];
            }
            
            float attn = expf(score * inv_sqrt_d - L_in[(size_t)bh * S + q]);
            
            qd_val = __bfloat162float(q_row[d]);
            acc += attn * qd_val;
        }
        
        dK_out[eid] = __float2bfloat16(acc);
    }
}

// Kernel: compute dV element-by-element.
// dV[b,h,k,d] = sum_{q=k}^{S-1} P[q,k] * dO[q,d]
template<int D>
__global__ __launch_bounds__(512)
void dV_single_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L_in,
    __nv_bfloat16* __restrict__ dV_out,
    int B, int H, int S)
{
    int64_t tid = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    int64_t total = (int64_t)B * H * S * D;
    
    float inv_sqrt_d = rsqrtf((float)D);
    
    for (int64_t eid = tid; eid < total; eid += stride) {
        int d = static_cast<int>(eid % D);
        int64_t rem = eid / D;
        int k = static_cast<int>(rem % S);
        int bh = static_cast<int>(rem / S);
        int b = bh / H;
        int h = bh % H;
        
        size_t base = ((size_t)b * H + h) * (size_t)S * D;
        
        const __nv_bfloat16* k_row = K + base + (size_t)k * D;
        
        // Load K[k,:] into registers
        float k_reg[D];
        #pragma unroll
        for (int dd = 0; dd < D; dd++) {
            k_reg[dd] = __bfloat162float(k_row[dd]);
        }
        
        float dov_val;
        float acc = 0.f;
        
        for (int q = k; q < S; q++) {
            const __nv_bfloat16* q_row = Q + base + (size_t)q * D;
            const __nv_bfloat16* do_row = dO + base + (size_t)q * D;
            
            float score = 0.f;
            #pragma unroll
            for (int dd = 0; dd < D; dd += 4) {
                score += __bfloat162float(q_row[dd])     * k_reg[dd];
                score += __bfloat162float(q_row[dd + 1]) * k_reg[dd + 1];
                score += __bfloat162float(q_row[dd + 2]) * k_reg[dd + 2];
                score += __bfloat162float(q_row[dd + 3]) * k_reg[dd + 3];
            }
            
            float attn = expf(score * inv_sqrt_d - L_in[(size_t)bh * S + q]);
            
            dov_val = __bfloat162float(do_row[d]);
            acc += attn * dov_val;
        }
        
        dV_out[eid] = __float2bfloat16(acc);
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
    assert(D_val == 128 && "Expected D=128");
    int D = static_cast<int>(D_val);
    
    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* dO_ptr = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_ptr = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQ_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());
    
    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    int64_t total_elem = B * H * S * D;
    
    // Step 1: Clear outputs
    int ct = 1024;
    int cb = std::min(static_cast<int>((total_elem + ct - 1) / ct), 65535);
    mha_clear_kernel<<<cb, ct, 0, stream>>>(dQ_ptr, total_elem);
    CUDA_CHECK(cudaGetLastError());
    mha_clear_kernel<<<cb, ct, 0, stream>>>(dK_ptr, total_elem);
    CUDA_CHECK(cudaGetLastError());
    mha_clear_kernel<<<cb, ct, 0, stream>>>(dV_ptr, total_elem);
    CUDA_CHECK(cudaGetLastError());
    
    // Step 2: Launch gradient kernels with MAXIMUM parallelism
    // Each thread handles ONE output element
    int threads_per_block = 256;
    int max_blocks = 65535;
    
    dim3 grid(max_blocks, 1, 1);
    dim3 blk(threads_per_block, 1, 1);
    
    dQ_single_kernel<128><<<grid, blk, 0, stream>>>(
        Q_ptr, K_ptr, L_ptr, dQ_ptr,
        static_cast<int>(B), static_cast<int>(H), static_cast<int>(S));
    CUDA_CHECK(cudaGetLastError());
    
    dK_single_kernel<128><<<grid, blk, 0, stream>>>(
        Q_ptr, K_ptr, L_ptr, dK_ptr,
        static_cast<int>(B), static_cast<int>(H), static_cast<int>(S));
    CUDA_CHECK(cudaGetLastError());
    
    dV_single_kernel<128><<<grid, blk, 0, stream>>>(
        Q_ptr, K_ptr, dO_ptr, L_ptr, dV_ptr,
        static_cast<int>(B), static_cast<int>(H), static_cast<int>(S));
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_impl::run);

}  // namespace mha_bwd_impl