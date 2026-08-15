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

// Clear output to zero
__global__ void mha_clear_kernel(__nv_bfloat16* out, int64_t total) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total; i += stride) {
        out[i] = __float2bfloat16(0.f);
    }
}

// Compute dQ[b,h,q,d] for one output element.
// dQ[q,d] = sum_{k=0}^{q} P[q,k] * K[k,d] / sqrt(d)
// where P[q,k] = exp(Q[q].K[k]/sqrt(d) - L[q])
template<int D>
__global__ __launch_bounds__(256)
void dQ_kernel_simple(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const float* __restrict__ L_in,
    __nv_bfloat16* __restrict__ dQ_out,
    int B, int H, int S)
{
    // Each thread computes ONE dQ element.
    int64_t eid = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    int64_t total = (int64_t)B * H * S * D;
    
    float inv_sqrt_d = rsqrtf((float)D);
    
    for (int64_t elem = eid; elem < total; elem += stride) {
        int d = static_cast<int>(elem % D);
        int64_t rem = elem / D;
        int q = static_cast<int>(rem % S);
        int bh = static_cast<int>(rem / S);
        
        size_t base = (size_t)bh * S * D;
        float lse_q = L_in[(size_t)bh * S + q];
        
        // We need Q[q,:] loaded for dot products. Use register array.
        float qf[D];
        const __nv_bfloat16* q_row = Q + base + (size_t)q * D;
        #pragma unroll
        for (int dd = 0; dd < D; dd++) {
            qf[dd] = __bfloat162float(q_row[dd]);
        }
        
        float acc = 0.f;
        for (int k = 0; k <= q; k++) {
            // score = Q[q] . K[k]
            const __nv_bfloat16* k_row = K + base + (size_t)k * D;
            float score = 0.f;
            
            // Unrolled vectorized dot product
            #pragma unroll
            for (int dd = 0; dd < D; dd += 8) {
                float q0=qf[dd], q1=qf[dd+1], q2=qf[dd+2], q3=qf[dd+3];
                float q4=qf[dd+4], q5=qf[dd+5], q6=qf[dd+6], q7=qf[dd+7];
                float k0=__bfloat162float(k_row[dd]), k1=__bfloat162float(k_row[dd+1]);
                float k2=__bfloat162float(k_row[dd+2]), k3=__bfloat162float(k_row[dd+3]);
                float k4=__bfloat162float(k_row[dd+4]), k5=__bfloat162float(k_row[dd+5]);
                float k6=__bfloat162float(k_row[dd+6]), k7=__bfloat162float(k_row[dd+7]);
                score += q0*k0 + q1*k1 + q2*k2 + q3*k3 + q4*k4 + q5*k5 + q6*k6 + q7*k7;
            }
            
            float attn = expf(score * inv_sqrt_d - lse_q);
            acc += attn * __bfloat162float(k_row[d]);
        }
        
        dQ_out[elem] = __float2bfloat16(acc);
    }
}

// Compute dK[b,h,k,d] for one output element.
// dK[k,d] = sum_{q=k}^{S-1} P[q,k] * Q[q,d] / sqrt(d)
template<int D>
__global__ __launch_bounds__(256)
void dK_kernel_simple(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const float* __restrict__ L_in,
    __nv_bfloat16* __restrict__ dK_out,
    int B, int H, int S)
{
    int64_t eid = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    int64_t total = (int64_t)B * H * S * D;
    
    float inv_sqrt_d = rsqrtf((float)D);
    
    for (int64_t elem = eid; elem < total; elem += stride) {
        int d = static_cast<int>(elem % D);
        int64_t rem = elem / D;
        int k = static_cast<int>(rem % S);
        int bh = static_cast<int>(rem / S);
        
        size_t base = (size_t)bh * S * D;
        
        // Load K[k,:] for repeated dot products
        float kf[D];
        const __nv_bfloat16* k_row = K + base + (size_t)k * D;
        #pragma unroll
        for (int dd = 0; dd < D; dd++) {
            kf[dd] = __bfloat162float(k_row[dd]);
        }
        
        float acc = 0.f;
        for (int q = k; q < S; q++) {
            const __nv_bfloat16* q_row = Q + base + (size_t)q * D;
            float score = 0.f;
            
            #pragma unroll
            for (int dd = 0; dd < D; dd += 8) {
                float q0=__bfloat162float(q_row[dd]), q1=__bfloat162float(q_row[dd+1]);
                float q2=__bfloat162float(q_row[dd+2]), q3=__bfloat162float(q_row[dd+3]);
                float q4=__bfloat162float(q_row[dd+4]), q5=__bfloat162float(q_row[dd+5]);
                float q6=__bfloat162float(q_row[dd+6]), q7=__bfloat162float(q_row[dd+7]);
                score += q0*kf[dd] + q1*kf[dd+1] + q2*kf[dd+2] + q3*kf[dd+3] 
                       + q4*kf[dd+4] + q5*kf[dd+5] + q6*kf[dd+6] + q7*kf[dd+7];
            }
            
            float attn = expf(score * inv_sqrt_d - L_in[(size_t)bh * S + q]);
            acc += attn * __bfloat162float(q_row[d]);
        }
        
        dK_out[elem] = __float2bfloat16(acc);
    }
}

// Compute dV[b,h,k,d] for one output element.
// dV[k,d] = sum_{q=k}^{S-1} P[q,k] * dO[q,d]
template<int D>
__global__ __launch_bounds__(256)
void dV_kernel_simple(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L_in,
    __nv_bfloat16* __restrict__ dV_out,
    int B, int H, int S)
{
    int64_t eid = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    int64_t total = (int64_t)B * H * S * D;
    
    float inv_sqrt_d = rsqrtf((float)D);
    
    for (int64_t elem = eid; elem < total; elem += stride) {
        int d = static_cast<int>(elem % D);
        int64_t rem = elem / D;
        int k = static_cast<int>(rem % S);
        int bh = static_cast<int>(rem / S);
        
        size_t base = (size_t)bh * S * D;
        
        // Load K[k,:] for repeated dot products
        float kf[D];
        const __nv_bfloat16* k_row = K + base + (size_t)k * D;
        #pragma unroll
        for (int dd = 0; dd < D; dd++) {
            kf[dd] = __bfloat162float(k_row[dd]);
        }
        
        float acc = 0.f;
        for (int q = k; q < S; q++) {
            const __nv_bfloat16* q_row = Q + base + (size_t)q * D;
            const __nv_bfloat16* do_row = dO + base + (size_t)q * D;
            
            float score = 0.f;
            #pragma unroll
            for (int dd = 0; dd < D; dd += 8) {
                float q0=__bfloat162float(q_row[dd]), q1=__bfloat162float(q_row[dd+1]);
                float q2=__bfloat162float(q_row[dd+2]), q3=__bfloat162float(q_row[dd+3]);
                float q4=__bfloat162float(q_row[dd+4]), q5=__bfloat162float(q_row[dd+5]);
                float q6=__bfloat162float(q_row[dd+6]), q7=__bfloat162float(q_row[dd+7]);
                score += q0*kf[dd] + q1*kf[dd+1] + q2*kf[dd+2] + q3*kf[dd+3]
                       + q4*kf[dd+4] + q5*kf[dd+5] + q6*kf[dd+6] + q7*kf[dd+7];
            }
            
            float attn = expf(score * inv_sqrt_d - L_in[(size_t)bh * S + q]);
            acc += attn * __bfloat162float(do_row[d]);
        }
        
        dV_out[elem] = __float2bfloat16(acc);
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
    
    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* dO_ptr = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_ptr = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQ_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());
    
    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    int64_t total_elem = B * H * S * D_val;
    
    // Clear outputs
    int ct = 1024;
    int cb = std::min(static_cast<int>((total_elem + ct - 1) / ct), 65535);
    mha_clear_kernel<<<cb, ct, 0, stream>>>(dQ_ptr, total_elem);
    CUDA_CHECK(cudaGetLastError());
    mha_clear_kernel<<<cb, ct, 0, stream>>>(dK_ptr, total_elem);
    CUDA_CHECK(cudaGetLastError());
    mha_clear_kernel<<<cb, ct, 0, stream>>>(dV_ptr, total_elem);
    CUDA_CHECK(cudaGetLastError());
    
    // Launch gradient kernels with maximum occupancy
    // Each thread computes exactly ONE output element.
    // Total elements = B*H*S*D = 4*48*4096*128 = 100M elements
    // With 256 threads/block and 65535 blocks, we get 16.7M threads.
    // Grid-stride loop iterates ~6x.
    int threads_per_block = 256;
    int max_blocks = 65535;
    
    dim3 grid(max_blocks, 1, 1);
    dim3 blk(threads_per_block, 1, 1);
    
    dQ_kernel_simple<128><<<grid, blk, 0, stream>>>(
        Q_ptr, K_ptr, L_ptr, dQ_ptr,
        static_cast<int>(B), static_cast<int>(H), static_cast<int>(S));
    CUDA_CHECK(cudaGetLastError());
    
    dK_kernel_simple<128><<<grid, blk, 0, stream>>>(
        Q_ptr, K_ptr, L_ptr, dK_ptr,
        static_cast<int>(B), static_cast<int>(H), static_cast<int>(S));
    CUDA_CHECK(cudaGetLastError());
    
    dV_kernel_simple<128><<<grid, blk, 0, stream>>>(
        Q_ptr, K_ptr, dO_ptr, L_ptr, dV_ptr,
        static_cast<int>(B), static_cast<int>(H), static_cast<int>(S));
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_impl::run);

}  // namespace mha_bwd_impl