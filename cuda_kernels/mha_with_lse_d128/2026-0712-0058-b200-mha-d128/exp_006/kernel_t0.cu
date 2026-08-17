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

#define CU_CHECK(call) do {                                      \
    CUresult _e = (call);                                        \
    if (_e != CUDA_SUCCESS) {                                    \
        fprintf(stderr, "CU error %d at %s:%d\n", (int)_e,         \
                __FILE__, __LINE__);                             \
        exit(1);                                                 \
    }                                                            \
} while(0)

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_kernel {

__global__ void attention_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int S, float scale)
{
    int b_idx = blockIdx.z;
    int h_idx = blockIdx.y;
    int s_offset = blockIdx.x * 64;
    
    if (s_offset >= S) return;
    
    size_t bh_off = (b_idx * 48 + h_idx) * S;
    const __nv_bfloat16* Q_bh = Q + bh_off * 128;
    const __nv_bfloat16* K_bh = K + bh_off * 128;
    const __nv_bfloat16* V_bh = V + bh_off * 128;
    __nv_bfloat16* O_bh = O + bh_off * 128;
    float* LSE_bh = LSE + bh_off;
    
    extern __shared__ __align__(128) uint8_t smem_buf[];
    __nv_bfloat16* q_shared = (__nv_bfloat16*)smem_buf;
    __nv_bfloat16* k_shared = (__nv_bfloat16*)(smem_buf + 16384);
    __nv_bfloat16* v_shared = (__nv_bfloat16*)(smem_buf + 32768);
    __nv_bfloat16* p_shared = (__nv_bfloat16*)(smem_buf + 49152);
    
    int row = threadIdx.x;
    
    // Load Q tile data into shared memory
    if (row < 64) {
        int s_idx = s_offset + row;
        if (s_idx < S) {
            for (int i = 0; i < 64; ++i) {
                *(uint32_t*)&q_shared[row * 128 + i * 2] = *(const uint32_t*)&Q_bh[s_idx * 128 + i * 2];
            }
        } else {
            for (int i = 0; i < 64; ++i) {
                *(uint32_t*)&q_shared[row * 128 + i * 2] = 0;
            }
        }
    }
    __syncthreads();
    
    // Output accumulator state and Online Softmax tracking variables
    float o_acc[128];
    for (int i = 0; i < 128; ++i) o_acc[i] = 0.0f;
    float m_prev = -1e20f;
    float l_prev = 0.0f;
    
    // Iterate through KV tiles
    for (int kv_offset = 0; kv_offset < S; kv_offset += 64) {
        int r = row % 64;
        int c = (row / 64) * 64;
        int load_kv = kv_offset + r;
        
        // Cooperative load of K and V tiles
        if (load_kv < S) {
            for (int i = 0; i < 32; ++i) {
                *(uint32_t*)&k_shared[r * 128 + c + i * 2] = *(const uint32_t*)&K_bh[load_kv * 128 + c + i * 2];
                *(uint32_t*)&v_shared[r * 128 + c + i * 2] = *(const uint32_t*)&V_bh[load_kv * 128 + c + i * 2];
            }
        } else {
            for (int i = 0; i < 32; ++i) {
                *(uint32_t*)&k_shared[r * 128 + c + i * 2] = 0;
                *(uint32_t*)&v_shared[r * 128 + c + i * 2] = 0;
            }
        }
        __syncthreads();
        
        // Compute QK^T locally via packed vector math
        float s_val[64];
        if (row < 64) {
            for (int key = 0; key < 64; ++key) {
                float sum = 0.0f;
                for (int i = 0; i < 64; ++i) {
                    uint32_t q_bits = *(uint32_t*)&q_shared[row * 128 + i * 2];
                    uint32_t k_bits = *(uint32_t*)&k_shared[key * 128 + i * 2];
                    __nv_bfloat162 q_b2 = *(__nv_bfloat162*)&q_bits;
                    __nv_bfloat162 k_b2 = *(__nv_bfloat162*)&k_bits;
                    float2 q_f2 = __bfloat1622float2(q_b2);
                    float2 k_f2 = __bfloat1622float2(k_b2);
                    sum += q_f2.x * k_f2.x + q_f2.y * k_f2.y;
                }
                if (kv_offset + key < S) {
                    s_val[key] = sum * scale;
                } else {
                    s_val[key] = -1e20f;
                }
            }
        }
        __syncthreads();
        
        float row_max = -1e20f;
        if (row < 64) {
            for (int key = 0; key < 64; ++key) {
                row_max = max(row_max, s_val[key]);
            }
        }
        
        // Mask out rows completely out of bounds acting safely in softmax progression
        if (s_offset + row >= S) {
            row_max = -1e20f;
        }
        
        float m_new = max(m_prev, row_max);
        float alpha = expf(m_prev - m_new);
        
        // Rescale historic O contributions dynamically
        if (row < 64) {
            for (int d = 0; d < 128; ++d) {
                o_acc[d] *= alpha;
            }
        }
        l_prev *= alpha;
        
        float row_sum = 0.0f;
        if (row < 64) {
            for (int key = 0; key < 64; ++key) {
                s_val[key] = expf(s_val[key] - m_new);
                row_sum += s_val[key];
                p_shared[row * 64 + key] = __float2bfloat16(s_val[key]);
            }
        }
        l_prev += row_sum;
        m_prev = m_new;
        __syncthreads();
        
        // Compute P @ V
        if (row < 64) {
            for (int key = 0; key < 64; ++key) {
                __nv_bfloat16 p = p_shared[row * 64 + key];
                float p_f = __low2float(p);
                for (int i = 0; i < 64; ++i) {
                    uint32_t v_bits = *(uint32_t*)&v_shared[key * 128 + i * 2];
                    __nv_bfloat162 v_b2 = *(__nv_bfloat162*)&v_bits;
                    float2 v_f2 = __bfloat1622float2(v_b2);
                    o_acc[i * 2] += p_f * v_f2.x;
                    o_acc[i * 2 + 1] += p_f * v_f2.y;
                }
            }
        }
        __syncthreads();
    }
    
    // Final output normalization and store
    if (row < 64) {
        int s_idx = s_offset + row;
        if (s_idx < S && l_prev > 0.0f) {
            for (int d = 0; d < 128; d += 2) {
                float2 o_f2 = {o_acc[d] / l_prev, o_acc[d+1] / l_prev};
                __nv_bfloat162 o_b2 = __float22bfloat162(o_f2);
                *(uint32_t*)&O_bh[s_idx * 128 + d] = *(uint32_t*)&o_b2;
            }
            
            // Log-Sum-Exp evaluated natively matching reference formatting expectations
            LSE_bh[s_idx] = m_prev + logf(l_prev);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id)); // This is necessary to ensure the correct GPU is used, especially in multi-GPU setups.
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    float scale = 1.0f / sqrtf(128);
    
    dim3 grid((S + 63) / 64, H, B);
    dim3 block(128);
    
    int smem_size = 56 * 1024;
    CUDA_CHECK(cudaFuncSetAttribute(attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    attention_kernel<<<grid, block, smem_size, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<__nv_bfloat16*>(O.data_ptr()),
        static_cast<float*>(LSE.data_ptr()),
        S, scale);
        
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_kernel