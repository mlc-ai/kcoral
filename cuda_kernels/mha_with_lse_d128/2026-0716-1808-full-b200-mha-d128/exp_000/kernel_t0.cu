#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
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

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void mbarrier_arrive_fn(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])) : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(tx_bytes) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(phase));
}

__device__ __forceinline__ void tma_copy_1d_g2s_fn(void const* gmem, uint64_t* mbar, void* smem, int32_t bytes) {
    uint32_t smem_mbar = (uint32_t)__cvta_generic_to_shared(mbar);
    uint32_t smem_ptr  = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
        :: "r"(smem_ptr), "l"(gmem), "r"(bytes), "r"(smem_mbar) : "memory");
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

extern __shared__ __align__(128) uint8_t smem_pool[];

__global__ void __launch_bounds__(128) attention_kernel(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* O, float* LSE, int S)
{
    int q_start = blockIdx.x * 128;
    int bh = blockIdx.y;
    
    if (q_start >= S) return;
    
    int tid = threadIdx.x; // 0..127
    
    __nv_bfloat16* smem_Q = (__nv_bfloat16*)(smem_pool + 0);               // 32KB
    __nv_bfloat16* smem_K = (__nv_bfloat16*)(smem_pool + 32768);            // 32KB
    __nv_bfloat16* smem_V = (__nv_bfloat16*)(smem_pool + 65536);            // 32KB
    __nv_bfloat16* smem_S = (__nv_bfloat16*)(smem_pool + 98304);            // 32KB (scores)
    __nv_bfloat16* smem_P = (__nv_bfloat16*)(smem_pool + 131072);           // 32KB (probabilities)
    float* smem_O = (float*)(smem_pool + 163840);                           // 64KB (accumulation)
    float* smem_m_prev = (float*)(smem_pool + 229376);                      // 512B
    float* smem_l_prev = (float*)(smem_pool + 229888);                      // 512B
    uint64_t* mbar_Q = (uint64_t*)(smem_pool + 230400);                     // 8B
    uint64_t* mbar_KV = (uint64_t*)(smem_pool + 230408);                    // 8B
    
    if (tid == 0) {
        init_smem_barrier_fn(mbar_Q, 128);
        init_smem_barrier_fn(mbar_KV, 128);
    }
    __syncthreads();
    
    uint64_t offset_Q = (bh * S + q_start) * 128;
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_Q, 32768);
        tma_copy_1d_g2s_fn(Q + offset_Q, mbar_Q, smem_Q, 32768);
    } else {
        mbarrier_arrive_fn(mbar_Q);
    }
    mbarrier_wait_fn(mbar_Q, 0);
    
    if (tid < 128) {
        smem_m_prev[tid] = -1e20f;
        smem_l_prev[tid] = 0.0f;
    }
    __syncthreads();
    
    float temp_S[128] = {0};
    __nv_bfloat16* temp_Q_fast_ptr = smem_Q + tid * 128;
    __nv_bfloat16* temp_K_fast[128];
    for(int j = 0; j < 128; ++j) {
        temp_K_fast[j] = smem_K + j * 128;
    }
    
    float inv_sqrt_D = 1.0f / sqrtf(128.0f);
    
    uint64_t num_kv_blocks = (S + 127) / 128;
    uint32_t phase_KV = 0;
    
    for (uint64_t kv_start = 0; kv_start < num_kv_blocks * 128; kv_start += 128) {
        uint64_t offset_K = (bh * S + kv_start) * 128;
        uint64_t offset_V = (bh * S + kv_start) * 128;
        
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_KV, 65536);
            tma_copy_1d_g2s_fn(K + offset_K, mbar_KV, smem_K, 32768);
            tma_copy_1d_g2s_fn(V + offset_V, mbar_KV, smem_V, 32768);
        } else {
            mbarrier_arrive_fn(mbar_KV);
        }
        mbarrier_wait_fn(mbar_KV, phase_KV);
        phase_KV ^= 1;
        
        for (int i = 0; i < 128; i += 2) {
            float temp_Q_i = temp_Q_fast_ptr[i];
            float temp_Q_i1 = temp_Q_fast_ptr[i + 1];
            for (int j = 0; j < 128; j += 2) {
                __nv_bfloat16* temp_K_fast_ptr = (__nv_bfloat16*)&temp_K_fast[j][0];
                uint2 temp_k2 = *(uint2*)&temp_K_fast_ptr[j];
                float2 temp_K2;
                temp_K2.x = temp_k2.x;
                temp_K2.y = temp_k2.y;
                temp_S[j] += temp_Q_i * temp_K2.x * inv_sqrt_D;
                temp_S[j + 1] += temp_Q_i1 * temp_K2.y * inv_sqrt_D;
            }
        }
        
        for (int j = 0; j < 128; ++j) {
            int kv_idx = kv_start + j;
            if (kv_idx >= S) {
                temp_S[j] = -1e20f;
            }
        }
        
        float max_val = -1e20f;
        for (int j = 0; j < 128; ++j) {
            if (temp_S[j] > max_val) {
                max_val = temp_S[j];
            }
        }
        
        float temp_p[128];
        float sum_p = 0.0f;
        for (int j = 0; j < 128; ++j) {
            float p = fast_exp2f_fn((temp_S[j] - max_val) * 1.44269504f);
            temp_p[j] = p;
            sum_p += p;
        }
        
        if (smem_row_sum[tid] > 0) {
            float m_prev = smem_m_prev[tid];
            float m_new = fmaxf(m_prev, max_val);
            float rescale = fast_exp2f_fn((m_prev - m_new) * 1.44269504f);
            
            smem_l_prev[tid] *= rescale;
            smem_l_prev[tid] += smem_row_sum[tid] * fast_exp2f_fn((row_max - m_new) * 1.44269504f);
            
            for (int i = 0; i < 128; ++i) {
                O[i] *= rescale;
            }
            
            float exp_row_max_to_mnew = fast_exp2f_fn((row_max - m_new) * 1.44269504f);
            for (int j = 0; j < 128; ++j) {
                smem_P[tid * 128 + j] = __float2bfloat16(p * exp_row_max_to_mnew);
            }
            
            __syncthreads(); 
            
            float temp_O[128];
            for (int j = 0; j < 128; j += 2) {
                temp_O[j] = smem_O[tid * 128 + j];
                temp_O[j + 1] = smem_O[tid * 128 + j + 1];
            }
            
            __nv_bfloat16* temp_V_fast[128];
            for(int i = 0; i < 128; ++i) {
                temp_V_fast[i] = smem_V + i * 128;
            }
            
            for (int i = 0; i < 128; ++i) {
                __nv_bfloat16* temp_V_fast_ptr = (__nv_bfloat16*)&temp_V_fast[i][0];
                for (int j = 0; j < 128; j += 2) {
                    uint2 temp_v2 = *(uint2*)&temp_V_fast_ptr[j];
                    float2 temp_V2;
                    temp_V2.x = temp_v2.x;
                    temp_V2.y = temp_v2.y;
                    
                    temp_O[j] += smem_P[tid * 128 + i] * temp_V2.x;
                    temp_O[j + 1] += smem_P[tid * 128 + i] * temp_V2.y;
                }
            }
            
            for (int j = 0; j < 128; j += 2) {
                smem_O[tid * 128 + j] = temp_O[j];
                smem_O[tid * 128 + j + 1] = temp_O[j + 1];
            }
            
            smem_m_prev[tid] = m_new;
        }
    }
    
    uint64_t offset_O = (bh * S + q_start) * 128;
    
    float m_prev = smem_m_prev[tid];
    float l_prev = smem_l_prev[tid];
    
    for (int i = 0; i < 128; i += 2) {
        if (q_start + tid < S) {
            float out_x = smem_O[tid * 128 + i];
            float out_y = smem_O[tid * 128 + i + 1];
            
            if (l_prev > 0) {
                out_x /= l_prev;
                out_y /= l_prev;
            }
            
            float2 temp2;
            temp2.x = __float2bfloat16(out_x);
            temp2.y = __float2bfloat16(out_y);
            
            *(uint2*)&g_O[offset_O + tid * 128 + i] = *(uint2*)&temp2;
        }
    }
    
    if (q_start + tid < S) {
        LSE[bh * S + q_start + tid] = m_prev[tid] + __logf(smem_l_prev[tid]);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  
  int64_t B = Q.size(0);
  int64_t H = Q.size(1);
  int64_t S = Q.size(2);
  int64_t D = Q.size(3);
  
  if (D != 128) {
    fprintf(stderr, "Expected D=128, got D=%ld\n", D);
    exit(1);
  }
  
  const __nv_bfloat16* g_Q = static_cast<const __nv_bfloat16*>(Q.data_ptr());
  const __nv_bfloat16* g_K = static_cast<const __nv_bfloat16*>(K.data_ptr());
  const __nv_bfloat16* g_V = static_cast<const __nv_bfloat16*>(V.data_ptr());
  
  __nv_bfloat16* g_O = static_cast<__nv_bfloat16*>(O.data_ptr());
  float* g_LSE = static_cast<float*>(LSE.data_ptr());
  
  cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
  
  dim3 grid((S + 127) / 128, B * H);
  dim3 block(128);
  
  int smem_size = 231168;
  CUDA_CHECK(cudaFuncSetAttribute(attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  
  attention_kernel<<<grid, block, smem_size, stream>>>(g_Q, g_K, g_V, g_O, g_LSE, S);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda