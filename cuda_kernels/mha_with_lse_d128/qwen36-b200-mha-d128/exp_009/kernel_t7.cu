#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <stdio.h>
#include <math.h>
#include <cfloat>
#include <algorithm>
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
// Processes TM query tokens per block, tiling over KV sequence with TN.
// Uses FP32 accumulators in shared memory to avoid register spilling.
template <int D, int TM, int TN, int TPB>
__global__ void mha_forward_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE_out,
    int B, int H, int S,
    float inv_sqrt_d) 
{
    extern __shared__ char smem[];
    
    // Shared memory layout: [TM*D BF16] + [TN*D BF16] + [TN*D BF16] + [TM*D FP32]
    __nv_bfloat16* Q_s = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* K_s = Q_s + TM * D;
    __nv_bfloat16* V_s = K_s + TN * D;
    float* Acc_s = reinterpret_cast<float*>(smem + (TM + 2 * TN) * D * sizeof(__nv_bfloat16));

    int b         = blockIdx.z;
    int h         = blockIdx.y;
    int q_start   = blockIdx.x * TM;
    int tid       = threadIdx.x;
    
    int qs = q_start + tid;
    if (qs >= S) return;

    int base_idx = (b * H + h) * S * D;
    
    // Zero-initialize accumulators cooperatively
    for (int i = tid; i < TM * D; i += TPB) {
        Acc_s[i] = 0.0f;
    }
    __syncthreads();

    // Cooperative load of Q tile into shared memory
    const __nv_bfloat16* q_base = Q + base_idx;
    for (int i = tid; i < TM * D; i += TPB) {
        int r = i / D;
        int d = i % D;
        int row_q = q_start + r;
        Q_s[i] = (row_q < S) ? q_base[row_q * D + d] : __float2bfloat16(0.0f);
    }
    __syncthreads();

    // Per-thread online softmax state & accumulator pointer
    float max_val = -FLT_MAX;
    float sum_val = 0.0f;
    float* my_acc = Acc_s + tid * D;

    // Iterate over Key/Value sequence in tiles
    for (int n_start = 0; n_start < S; n_start += TN) {
        int n_actual = (TN < S - n_start) ? TN : (S - n_start);
        
        // Cooperative load of K & V tiles into shared memory
        const __nv_bfloat16* k_base_tile = K + base_idx + n_start * D;
        const __nv_bfloat16* v_base_tile = V + base_idx + n_start * D;
        for (int i = tid; i < n_actual * D; i += TPB) {
            K_s[i] = k_base_tile[i];
            V_s[i] = v_base_tile[i];
        }
        __syncthreads();

        // Compute QK^T scores and fuse online softmax accumulation
        const __nv_bfloat16* q_local = Q_s + tid * D;
        for (int k = 0; k < n_actual; ++k) {
            const __nv_bfloat16* k_local = K_s + k * D;
            const __nv_bfloat16* v_local = V_s + k * D;
            
            float score = 0.0f;
            for (int d = 0; d < D; ++d) {
                score += static_cast<float>(q_local[d]) * static_cast<float>(k_local[d]);
            }
            score *= inv_sqrt_d;

            if (score > max_val) {
                float ratio = expf(max_val - score);
                sum_val *= ratio;
                for (int d = 0; d < D; ++d) my_acc[d] *= ratio;
                
                max_val = score;
                sum_val += 1.0f;
                for (int d = 0; d < D; ++d) {
                    my_acc[d] += static_cast<float>(v_local[d]);
                }
            } else {
                float p = expf(score - max_val);
                sum_val += p;
                for (int d = 0; d < D; ++d) {
                    my_acc[d] += p * static_cast<float>(v_local[d]);
                }
            }
        }
        __syncthreads();
    }

    // Final normalization and write-out to global memory
    float inv_sum = (sum_val > 0.0f) ? (1.0f / sum_val) : 0.0f;
    __nv_bfloat16* o_ptr = O + base_idx + qs * D;
    for (int d = 0; d < D; ++d) {
        o_ptr[d] = __float2bfloat16(my_acc[d] * inv_sum);
    }
    
    // Write Log-Sum-Exp
    LSE_out[(b * H + h) * S + qs] = (sum_val > 0.0f) ? (max_val + logf(sum_val)) : INFINITY;
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
    auto shape_check = [](auto& T, int eB, int eH, int eS, int eD) {
        if(T.size(0)!=eB||T.size(1)!=eH||T.size(2)!=eS||T.size(3)!=eD){fprintf(stderr,"Shape mismatch\n");exit(1);}
    };
    shape_check(K_in, B,H,S,D);
    shape_check(V_in, B,H,S,D);
    shape_check(O_out,B,H,S,D);
    if(LSE_out.size(0)!=B||LSE_out.size(1)!=H||LSE_out.size(2)!=S){fprintf(stderr,"LSE mismatch\n");exit(1);}
    
    float inv_sqrt_d = 1.0f / sqrtf(static_cast<float>(D));
    
    // Tuned tile parameters
    const int TM = 64;
    const int TN = 64;
    const int TPB = 64;
    
    // Dynamic shared memory size calculation
    int smem_bytes = (TM + 2 * TN) * D * sizeof(__nv_bfloat16) + TM * D * sizeof(float);
    
    dim3 grid((S + TM - 1) / TM, H, B);
    dim3 block(TPB);
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q_in.device().device_type, Q_in.device().device_id));
        
    // Dispatch based on compile-time head dimension
    if(D == 32)  mha_forward_kernel<32, 64, 64, 64><<<grid, block, smem_bytes, stream>>>(Q_ptr,K_ptr,V_ptr,O_ptr,LSE_ptr,B,H,S,inv_sqrt_d);
    else if(D == 64) mha_forward_kernel<64, 64, 64, 64><<<grid, block, smem_bytes, stream>>>(Q_ptr,K_ptr,V_ptr,O_ptr,LSE_ptr,B,H,S,inv_sqrt_d);
    else if(D == 128) mha_forward_kernel<128,64, 64, 64><<<grid, block, smem_bytes, stream>>>(Q_ptr,K_ptr,V_ptr,O_ptr,LSE_ptr,B,H,S,inv_sqrt_d);
    else { fprintf(stderr,"Unsupported head dim D=%d\n",D); exit(1); }
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_impl::run);

}  // namespace mha_impl