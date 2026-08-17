#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
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

namespace tvm_ffi_example_cuda {

__global__ __launch_bounds__(64) void AttentionKernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int S)
{
    int b_h = blockIdx.y;
    int s_offset_q = blockIdx.x * 64;
    int row = threadIdx.x;
    int num_S_blocks = (S + 63) / 64;
    float scale = 1.0f / sqrtf(128.0f);
    int s_q = s_offset_q + row;

    extern __shared__ __align__(16) char smem_pool[];
    uint4* smem_Q = (uint4*)smem_pool;                
    uint4* smem_K = smem_Q + 1024;                   
    uint4* smem_V = smem_K + 1024;                    

    float4* Q_row = (float4*)&Q[b_h * S * 128 + s_q * 128];
    for (int i = 0; i < 16; i++) {
        if (s_q < S) {
            smem_Q[row * 16 + i] = *(uint4*)&Q_row[i];
        } else {
            smem_Q[row * 16 + i] = {0, 0, 0, 0};
        }
    }
    
    float max_val = -1e20f;
    float sum_val = 0.0f;
    float O_f[128];
    for (int i = 0; i < 128; i++) O_f[i] = 0.0f;

    __syncthreads();

    for (int j = 0; j < num_S_blocks; j++) {
        int s_kv = j * 64;
        uint4* K_ptr = (uint4*)&K[b_h * S * 128 + (s_kv + row) * 128];
        uint4* V_ptr = (uint4*)&V[b_h * S * 128 + (s_kv + row) * 128];
        for (int i = 0; i < 16; i++) {
            if (s_kv + row < S) {
                smem_K[row * 16 + i] = *(uint4*)&K_ptr[i];
                smem_V[row * 16 + i] = *(uint4*)&V_ptr[i];
            } else {
                smem_K[row * 16 + i] = {0, 0, 0, 0};
                smem_V[row * 16 + i] = {0, 0, 0, 0};
            }
        }
        __syncthreads();

        float local_max = -1e20f;
        float S_local[64];

        for (int col = 0; col < 64; col++) {
            float s = 0;
            uint64_t zero = 0;
            for(int i = 0; i < 16; i++) {
                uint4 q_vec = smem_Q[row * 16 + i];
                uint4 k_vec = smem_K[col * 16 + i];
                uint32_t* qw = (uint32_t*)&q_vec;
                uint32_t* kw = (uint32_t*)&k_vec;
                
                uint64_t q0 = ((uint64_t)qw[1] << 32) | qw[0];
                uint64_t k0 = ((uint64_t)kw[1] << 32) | kw[0];
                asm volatile("dot16x4.b16.f32 %0, %1, %2, %3, %4;\n" : "+f"(s) : "l"(q0), "l"(k0), "l"(zero), "l"(zero));
                
                uint64_t q1 = ((uint64_t)qw[3] << 32) | qw[2];
                uint64_t k1 = ((uint64_t)kw[3] << 32) | kw[2];
                asm volatile("dot16x4.b16.f32 %0, %1, %2, %3, %4;\n" : "+f"(s) : "l"(q1), "l"(k1), "l"(zero), "l"(zero));
            }
            int key_idx = j * 64 + col;
            if (key_idx >= S) {
                S_local[col] = -1e20f;
            } else {
                S_local[col] = s * scale;
            }
            if (S_local[col] > local_max) local_max = S_local[col];
        }

        float local_sum = 0;
        float P_local[64];
        for (int col = 0; col < 64; col++) {
            float p = expf(S_local[col] - local_max);
            local_sum += p;
            P_local[col] = p;
        }

        __syncthreads();
        
        float new_max = max_val;
        if (local_max > new_max) new_max = local_max;
        float alpha = expf(max_val - new_max);
        
        sum_val *= alpha;
        for (int d = 0; d < 128; d++) {
            O_f[d] *= alpha;
        }
        
        float prev_sum = sum_val;
        float P_scaled[64];
        for (int k = 0; k < 64; k++) {
            P_scaled[k] = P_local[k] * expf(local_max - new_max);
            sum_val += P_scaled[k];
        }
        
        max_val = new_max;

        for (int k = 0; k < 64; k++) {
            for (int d = 0; d < 128; d++) {
                O_f[d] += P_scaled[k] * __bfloat162float(((uint16_t*)smem_V)[k * 128 + d]);
            }
        }
        __syncthreads();
    }

    if (s_q < S) {
        float4* O_row = (float4*)&O[b_h * S * 128 + s_q * 128];
        for (int i = 0; i < 16; i++) {
            float4 out_val;
            for (int j = 0; j < 4; j++) {
                int d = i * 8 + j * 2;
                __nv_bfloat16 bf0 = __float2bfloat16(O_f[d]);
                __nv_bfloat16 bf1 = __float2bfloat16(O_f[d+1]);
                out_val.uint32s[j] = ((uint32_t)*(uint16_t*)&bf1 << 16) | *(uint16_t*)&bf0;
            }
            O_row[i] = out_val;
        }
        LSE[b_h * S + s_q] = max_val + logf(sum_val);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id)); 
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    
    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_ptr = static_cast<float*>(LSE.data_ptr());
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    int num_S_blocks = (S + 63) / 64;
    dim3 grid(num_S_blocks, B * H);
    dim3 block(64);
    
    CUDA_CHECK(cudaFuncSetAttribute(AttentionKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 49152));
    
    AttentionKernel<<<grid, block, 49152, stream>>>(Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr, S);
    CUDA_CHECK(cudaGetLastError());
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

}  // namespace tvm_ffi_example_cuda