#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <stdio.h>
#include <math.h>
#include <cfloat>
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

namespace mha_impl {

static constexpr int BLOCK_THREADS = 128;

// Simple per-thread-per-query kernel with numerical-stable fused softmax
__global__ void mha_forward_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE_out,
    int B, int H, int S, int D,
    float inv_sqrt_d) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_queries = B * H * S;
    if (idx >= total_queries) return;
    
    int s_q = idx % S;
    int h   = (idx / S) % H;
    int b   = idx / (S * H);
    
    int bh_s_offset = (b * H + h) * S;
    
    const __nv_bfloat16* q_row = Q + bh_s_offset * D + s_q * D;
    const __nv_bfloat16* k_base = K + bh_s_offset * D;
    const __nv_bfloat16* v_base = V + bh_s_offset * D;
    __nv_bfloat16* o_row = O + bh_s_offset * D + s_q * D;
    
    float max_val = -FLT_MAX;
    float sum_val = 0.0f;
    float z[D];
    
    for (int d = 0; d < D; d++) z[d] = 0.0f;
    
    for (int s_kv = 0; s_kv < S; s_kv++) {
        // Dot product: Q[s_q] · K[s_kv]
        float score = 0.0f;
        const __nv_bfloat16* k_row = k_base + s_kv * D;
        for (int d = 0; d < D; d++) {
            score += static_cast<float>(q_row[d]) * static_cast<float>(k_row[d]);
        }
        score *= inv_sqrt_d;
        
        float p;
        if (score > max_val) {
            float new_max = score;
            sum_val = (sum_val == 0.0f) ? 1.0f : (sum_val * expf(max_val - new_max) + 1.0f);
            for (int d = 0; d < D; d++) {
                z[d] *= expf(max_val - new_max);
            }
            max_val = new_max;
            p = 1.0f;
        } else {
            p = expf(score - max_val);
            sum_val += p;
        }
        
        const __nv_bfloat16* v_row = v_base + s_kv * D;
        for (int d = 0; d < D; d++) {
            z[d] += p * static_cast<float>(v_row[d]);
        }
    }
    
    // Write output
    if (sum_val > 0.0f) {
        float inv_sum = 1.0f / sum_val;
        for (int d = 0; d < D; d++) {
            o_row[d] = __float2bfloat16(z[d] * inv_sum);
        }
    } else {
        // Empty attention — write zeros / inf LSE
        for (int d = 0; d < D; d++) {
            o_row[d] = __float2bfloat16(0.0f);
        }
    }
    
    LSE_out[bh_s_offset + s_q] = (sum_val > 0.0f) ? (max_val + logf(sum_val)) : INFINITY;
}

void run(tvm::ffi::TensorView Q_in, tvm::ffi::TensorView K_in, tvm::ffi::TensorView V_in,
         tvm::ffi::TensorView O_out, tvm::ffi::TensorView LSE_out) {
    CUDA_CHECK(cudaSetDevice(Q_in.device().device_id));
    
    int B = static_cast<int>(Q_in.size(0));
    int H = static_cast<int>(Q_in.size(1));
    int S = static_cast<int>(Q_in.size(2));
    int D = static_cast<int>(Q_in.size(3));
    
    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q_in.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K_in.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V_in.data_ptr());
    __nv_bfloat16* O_ptr = static_cast<__nv_bfloat16*>(O_out.data_ptr());
    float* LSE_ptr = static_cast<float*>(LSE_out.data_ptr());
    
    // Validate shapes match
    if (K_in.size(0)!=B || K_in.size(1)!=H || K_in.size(2)!=S || K_in.size(3)!=D) {
        fprintf(stderr, "Shape mismatch K\n"); exit(1);
    }
    if (V_in.size(0)!=B || V_in.size(1)!=H || V_in.size(2)!=S || V_in.size(3)!=D) {
        fprintf(stderr, "Shape mismatch V\n"); exit(1);
    }
    if (O_out.size(0)!=B || O_out.size(1)!=H || O_out.size(2)!=S || O_out.size(3)!=D) {
        fprintf(stderr, "Shape mismatch O\n"); exit(1);
    }
    if (LSE_out.size(0)!=B || LSE_out.size(1)!=H || LSE_out.size(2)!=S) {
        fprintf(stderr, "Shape mismatch LSE\n"); exit(1);
    }
    
    float inv_sqrt_d = 1.0f / sqrtf(static_cast<float>(D));
    
    // One thread per query token → total_queries = B*H*S
    int total_threads = B * H * S;
    int blocks = (total_threads + BLOCK_THREADS - 1) / BLOCK_THREADS;
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q_in.device().device_type, Q_in.device().device_id));
    
    mha_forward_kernel<<<blocks, BLOCK_THREADS, 0, stream>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr, B, H, S, D, inv_sqrt_d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_impl::run);

}  // namespace mha_impl