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

// Tiled Multi-Head Attention Kernel
// Processes BLOCK_M query tokens per block, tiling over KV sequence with BLOCK_N.
template <int D, int BLOCK_M, int BLOCK_N>
__global__ void mha_forward_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE_out,
    int B, int H, int S,
    float inv_sqrt_d) 
{
    extern __shared__ __nv_bfloat16 smem[];
    
    // Shared memory layout: [BLOCK_M * D] + [BLOCK_N * D] + [BLOCK_N * D]
    __nv_bfloat16* Q_s = smem;
    __nv_bfloat16* K_s = smem + BLOCK_M * D;
    __nv_bfloat16* V_s = smem + (BLOCK_M + BLOCK_N) * D;

    int b         = blockIdx.z;
    int h         = blockIdx.y;
    int q_start   = blockIdx.x * BLOCK_M;
    int m         = threadIdx.x; // Maps to query row within tile [0, BLOCK_M-1]

    int actual_M = min(BLOCK_M, S - q_start);
    if (m >= actual_M) return;

    // Base linear index for this batch/head
    int base_idx = (b * H + h) * S * D;
    
    // Pointers for current query row and KV bases
    const __nv_bfloat16* Q_row = Q + base_idx + (q_start + m) * D;
    const __nv_bfloat16* K_base = K + base_idx;
    const __nv_bfloat16* V_base = V + base_idx;
    __nv_bfloat16* O_row = O + base_idx + (q_start + m) * D;
    
    // Per-thread running online-softmax state and accumulator
    float max_val = -FLT_MAX;
    float sum_val = 0.0f;
    float acc[D];
    #pragma unroll
    for (int d = 0; d < D; d++) acc[d] = 0.0f;

    // Load Q tile into shared memory (each thread loads its own row)
    {
        __nv_bfloat16* q_dst = Q_s + m * D;
        for (int d = 0; d < D; d++) {
            q_dst[d] = Q_row[d];
        }
    }
    __syncthreads();

    // Iterate over Key/Value tiles
    for (int n_start = 0; n_start < S; n_start += BLOCK_N) {
        int n_actual = min(BLOCK_N, S - n_start);
        
        // Cooperative load of K and V tiles into shared memory
        if (threadIdx.x == 0) {
            int kv_size = n_actual * D;
            for (int i = 0; i < kv_size; ++i) {
                K_s[i] = K_base[n_start * D + i];
                V_s[i] = V_base[n_start * D + i];
            }
        }
        __syncthreads();

        // Compute QK^T scores and fuse online softmax accumulation
        const __nv_bfloat16* q_local = Q_s + m * D;
        for (int k = 0; k < n_actual; ++k) {
            const __nv_bfloat16* k_local = K_s + k * D;
            const __nv_bfloat16* v_local = V_s + k * D;
            
            float score = 0.0f;
            #pragma unroll
            for (int d = 0; d < D; ++d) {
                score += static_cast<float>(q_local[d]) * static_cast<float>(k_local[d]);
            }
            score *= inv_sqrt_d;

            if (score > max_val) {
                float ratio = expf(max_val - score);
                sum_val *= ratio;
                #pragma unroll
                for (int d = 0; d < D; ++d) acc[d] *= ratio;
                
                max_val = score;
                sum_val += 1.0f;
                #pragma unroll
                for (int d = 0; d < D; ++d) {
                    acc[d] += static_cast<float>(v_local[d]);
                }
            } else {
                float p = expf(score - max_val);
                sum_val += p;
                #pragma unroll
                for (int d = 0; d < D; ++d) {
                    acc[d] += p * static_cast<float>(v_local[d]);
                }
            }
        }
        __syncthreads();
    }

    // Final normalization and write-out
    float inv_sum = (sum_val > 0.0f) ? (1.0f / sum_val) : 0.0f;
    #pragma unroll
    for (int d = 0; d < D; ++d) {
        O_row[d] = __float2bfloat16(acc[d] * inv_sum);
    }
    
    float lse = (sum_val > 0.0f) ? (max_val + logf(sum_val)) : INFINITY;
    LSE_out[(b * H + h) * S + q_start + m] = lse;
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
    __nv_bfloat16* O_ptr       = static_cast<__nv_bfloat16*>(O_out.data_ptr());
    float* LSE_ptr             = static_cast<float*>(LSE_out.data_ptr());
    
    // Shape validation
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
    
    constexpr int BLOCK_M = 64;
    constexpr int BLOCK_N = 64;
    int smem_bytes = (BLOCK_M + 2 * BLOCK_N) * D * sizeof(__nv_bfloat16);
    
    dim3 grid((S + BLOCK_M - 1) / BLOCK_M, H, B);
    dim3 block(BLOCK_M);
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q_in.device().device_type, Q_in.device().device_id));
        
    // Dispatch based on compile-time head dimension
    switch(D) {
        case 32:  mha_forward_kernel<32, 64, 64><<<grid, block, smem_bytes, stream>>>(Q_ptr,K_ptr,V_ptr,O_ptr,LSE_ptr,B,H,S,inv_sqrt_d); break;
        case 64:  mha_forward_kernel<64, 64, 64><<<grid, block, smem_bytes, stream>>>(Q_ptr,K_ptr,V_ptr,O_ptr,LSE_ptr,B,H,S,inv_sqrt_d); break;
        case 128: mha_forward_kernel<128, 64, 64><<<grid, block, smem_bytes, stream>>>(Q_ptr,K_ptr,V_ptr,O_ptr,LSE_ptr,B,H,S,inv_sqrt_d); break;
        default:  fprintf(stderr,"Unsupported head dim D=%d\n",D); exit(1);
    }
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_impl::run);

}  // namespace mha_impl