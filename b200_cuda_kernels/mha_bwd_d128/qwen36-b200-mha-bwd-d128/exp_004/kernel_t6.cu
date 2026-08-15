#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

#define CUDA_CHECK(call) do {                                          \
    cudaError_t _e = (call);                                           \
    if (_e != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                   \
                cudaGetErrorString(_e), __FILE__, __LINE__);           \
        exit(1);                                                       \
    }                                                                  \
} while(0)

namespace mha_bwd_d128 {

static constexpr int NT = 128;

__global__ void mha_bwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O_fwd,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    int B, int H, int S, int d,
    float inv_sqrt_d) {
    
    int bh = blockIdx.x;
    if (bh >= B * H) return;
    
    int b = bh / H;
    int h = bh % H;
    
    uint64_t off_bh = ((uint64_t)b * H + h) * (uint64_t)S * d;
    uint64_t off_lse = ((uint64_t)b * H + h) * (uint64_t)S;
    
    int tid = threadIdx.x;
    
    // Initialize dK, dV to zero for this (b,h)
    for (int idx = tid; idx < S * d; idx += NT) {
        dK[off_bh + idx] = __float2bfloat16(0.0f);
        dV[off_bh + idx] = __float2bfloat16(0.0f);
    }
    __syncthreads();
    
    for (int q = 0; q < S; q++) {
        // Load Q[q,:], dO[q,:], O_fwd[q,:] -> fp32
        float q_reg[d];
        float do_reg[d];
        float of_reg[d];
        
        for (int dd = tid; dd < d; dd += NT) {
            q_reg[dd] = __bfloat162float(Q[off_bh + (uint64_t)q*d + dd]);
            do_reg[dd] = __bfloat162float(dO[off_bh + (uint64_t)q*d + dd]);
            of_reg[dd] = __bfloat162float(O_fwd[off_bh + (uint64_t)q*d + dd]);
        }
        __syncthreads();
        
        // Compute delta[q] = dot(dO[q], O_fwd[q])
        float delta_q = 0.0f;
        for (int dd = tid; dd < d; dd += NT) {
            delta_q += do_reg[dd] * of_reg[dd];
        }
        
        // Warp-reduce delta_q
        #pragma unroll
        for (int mask = NT/2; mask > 0; mask >>= 1) {
            delta_q += __shfl_down_sync(0xFFFFFFFF, delta_q, mask);
        }
        
        float l_val = L[off_lse + q];
        
        // Clear dQ accumulator for this q
        float dq_acc[d];
        #pragma unroll
        for (int c = 0; c < d; c++) dq_acc[c] = 0.0f;
        
        for (int k = 0; k < S; k++) {
            float s_qk = 0.0f;
            float dp_qk = 0.0f;
            
            // Vectorized dot products
            for (int dd = 0; dd < d; dd += 4) {
                float kv[4], vvv[4];
                #pragma unroll
                for (int e = 0; e < 4; e++) {
                    kv[e] = __bfloat162float(K[off_bh + (uint64_t)k*d + dd+e]);
                    vvv[e] = __bfloat162float(V[off_bh + (uint64_t)k*d + dd+e]);
                }
                
                s_qk += q_reg[dd+0]*kv[0] + q_reg[dd+1]*kv[1] + q_reg[dd+2]*kv[2] + q_reg[dd+3]*kv[3];
                dp_qk += do_reg[dd+0]*vvv[0] + do_reg[dd+1]*vvv[1] + do_reg[dd+2]*vvv[2] + do_reg[dd+3]*vvv[3];
            }
            
            s_qk *= inv_sqrt_d;
            float p_qk = expf(s_qk - l_val);
            float ds_qk = p_qk * (dp_qk - delta_q);
            
            // Accumulate dQ
            for (int dd = 0; dd < d; dd += 4) {
                float kv[4];
                #pragma unroll
                for (int e = 0; e < 4; e++) {
                    kv[e] = __bfloat162float(K[off_bh + (uint64_t)k*d + dd+e]);
                }
                dq_acc[dd+0] += ds_qk * kv[0];
                dq_acc[dd+1] += ds_qk * kv[1];
                dq_acc[dd+2] += ds_qk * kv[2];
                dq_acc[dd+3] += ds_qk * kv[3];
            }
            
            // Accumulate dK and dV globally (atomic-ish with R-M-W per unique thread)
            // Each thread writes a unique position based on (k, dd) pattern
            // To avoid conflicts, each thread writes to dK/dV[k, :stride]
            uint64_t base_k = off_bh + (uint64_t)k * d;
            for (int dd = tid; dd < d; dd += NT) {
                float dk_add = ds_qk * q_reg[dd];
                float dv_add = p_qk * do_reg[dd];
                
                dK[base_k + dd] = __float2bfloat16(__bfloat162float(dK[base_k + dd]) + dk_add);
                dV[base_k + dd] = __float2bfloat16(__bfloat162float(dV[base_k + dd]) + dv_add);
            }
        }
        
        // Write dQ[q,:]
        for (int dd = tid; dd < d; dd += NT) {
            dQ[off_bh + (uint64_t)q*d + dd] = __float2bfloat16(dq_acc[dd]);
        }
        __syncthreads();
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t d_dim = Q.size(3);
    
    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* O_data = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dO_data = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_data = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQ_data = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_data = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_data = static_cast<__nv_bfloat16*>(dV.data_ptr());
    
    float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(d_dim));
    int total_heads = static_cast<int>(B * H);
    
    dim3 grid(total_heads);
    dim3 block(NT);
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    auto kernel_fn = &mha_bwd_kernel;
    kernel_fn<<<grid, block, 0, stream>>>(
        Q_data, K_data, V_data, O_data, dO_data, L_data,
        dQ_data, dK_data, dV_data,
        static_cast<int>(B), static_cast<int>(H), static_cast<int>(S), static_cast<int>(d_dim),
        inv_sqrt_d);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_d128::run);

}  // namespace mha_bwd_d128