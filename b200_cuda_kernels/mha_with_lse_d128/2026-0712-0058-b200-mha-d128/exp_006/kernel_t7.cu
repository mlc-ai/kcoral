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

__device__ __forceinline__ uint32_t pack_bf16_fn(float a, float b) {
    __nv_bfloat162 ba = __float22bfloat162({a, b});
    return *reinterpret_cast<uint32_t*>(&ba);
}

__device__ __forceinline__ int get_swizzled_col(int row, int col) {
    int chunk_idx = col / 8;
    int swizzled_chunk = chunk_idx ^ (row % 8);
    return swizzled_chunk * 8 + (col % 8);
}

__global__ __launch_bounds__(64) void attention_kernel(
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
    int tid = threadIdx.x;
    
    extern __shared__ __align__(128) uint8_t smem_buf[];
    uintptr_t smem_addr = (uintptr_t)smem_buf;
    uintptr_t padded_addr = (smem_addr + 127) & ~127;
    
    __nv_bfloat16* Q_shared = (__nv_bfloat16*)padded_addr;
    __nv_bfloat16* K_shared = Q_shared + 8192; 
    __nv_bfloat16* V_shared = K_shared + 8192; 
    __nv_bfloat16* P_shared = V_shared + 8192; 
    
    size_t bh_off = (b_idx * 48 + h_idx);
    const __nv_bfloat16* Q_bh = Q + bh_off * S * 128;
    const __nv_bfloat16* K_bh = K + bh_off * S * 128;
    const __nv_bfloat16* V_bh = V + bh_off * S * 128;
    __nv_bfloat16* O_bh = O + bh_off * S * 128;
    float* LSE_bh = LSE + bh_off * S;
    
    for(int i = 0; i < 16; i++) {
        int gmem_idx = tid * 128 + i * 8;
        ulonglong2 val = {0, 0};
        if (s_offset + tid < S) {
            val = *(const ulonglong2*)(Q_bh + gmem_idx);
        }
        int sc = get_swizzled_col(tid, i * 8);
        *(ulonglong2*)&Q_shared[tid * 128 + sc] = val;
    }
    
    __syncthreads();
    
    float m_prev_h = -1e20f;
    float l_prev_h = 0.0f;
    
    float o_acc[128];
    for(int i = 0; i < 128; i++) o_acc[i] = 0.0f;

    for (int kv_offset = 0; kv_offset < S; kv_offset += 64) {
        for(int i = 0; i < 16; i++) {
            int gmem_idx = tid * 128 + i * 8;
            
            ulonglong2 kval = {0, 0};
            if (kv_offset + tid < S) {
                kval = *(const ulonglong2*)(K_bh + kv_offset * 128 + gmem_idx);
            }
            ulonglong2 vval = {0, 0};
            if (kv_offset + tid < S) {
                vval = *(const ulonglong2*)(V_bh + kv_offset * 128 + gmem_idx);
            }
            
            int sc = get_swizzled_col(tid, i * 8);
            *(ulonglong2*)&K_shared[tid * 128 + sc] = kval;
            *(ulonglong2*)&V_shared[tid * 128 + sc] = vval;
        }
        
        __syncthreads(); 
        
        float row_max_h = -1e20f;
        float val[64];
        
        for (int k = 0; k < 64; k++) {
            float sum = 0;
            for(int i = 0; i < 128; i+=2) {
                uint32_t q_bits = *(uint32_t*)&Q_shared[tid * 128 + get_swizzled_col(tid, i)];
                uint32_t k_bits = *(uint32_t*)&K_shared[k * 128 + get_swizzled_col(k, i)];
                
                __nv_bfloat162 q_b2 = *(__nv_bfloat162*)&q_bits;
                __nv_bfloat162 k_b2 = *(__nv_bfloat162*)&k_bits;
                
                float2 q_f2 = __bfloat1622float2(q_b2);
                float2 k_f2 = __bfloat1622float2(k_b2);
                
                sum += q_f2.x * k_f2.x + q_f2.y * k_f2.y;
            }
            float v = sum * scale;
            if (kv_offset + k >= S) v = -1e20f;
            val[k] = v;
            row_max_h = max(row_max_h, v);
        }
        
        float m_new_h = max(m_prev_h, row_max_h);
        float alpha_h = expf(m_prev_h - m_new_h);
        l_prev_h *= alpha_h;
        
        for(int i = 0; i < 128; i++) {
            o_acc[i] *= alpha_h;
        }
        
        float row_sum_h = 0;
        if (tid < 64) {
            for (int i = 0; i < 64; i += 2) {
                float2 p_f2 = {expf(val[i] - m_new_h), expf(val[i+1] - m_new_h)};
                row_sum_h += p_f2.x + p_f2.y;
                
                __nv_bfloat162 p_b2 = __float22bfloat162(p_f2);
                int sc = get_swizzled_col(tid, i);
                *(uint32_t*)&P_shared[tid * 64 + sc] = *(uint32_t*)&p_b2;
            }
        }
        
        l_prev_h += row_sum_h;
        m_prev_h = m_new_h;
        
        __syncthreads();
        
        for (int k = 0; k < 64; k++) {
            int sc = get_swizzled_col(tid, k);
            __nv_bfloat16 p = P_shared[tid * 64 + sc];
            float p_f = __bfloat162float(p);
            for(int i = 0; i < 128; i+=2) {
                uint32_t v_bits = *(uint32_t*)&V_shared[k * 128 + get_swizzled_col(k, i)];
                __nv_bfloat162 v_b2 = *(__nv_bfloat162*)&v_bits;
                float2 v_f2 = __bfloat1622float2(v_b2);
                o_acc[i] += p_f * v_f2.x;
                o_acc[i + 1] += p_f * v_f2.y;
            }
        }
        __syncthreads();
    }
    
    int s_idx = s_offset + tid;
    if (tid < 64) {
        if (s_idx < S) {
            if (l_prev_h > 0.0f) {
                for (int d = 0; d < 128; d += 2) {
                    float2 of = {o_acc[d] / l_prev_h, o_acc[d+1] / l_prev_h};
                    __nv_bfloat162 ob = __float22bfloat162(of);
                    *(uint32_t*)&O_bh[s_idx * 128 + d] = *(uint32_t*)&ob;
                }
            } else {
                for (int d = 0; d < 128; d += 2) {
                    *(uint32_t*)&O_bh[s_idx * 128 + d] = 0;
                }
            }
            LSE_bh[s_idx] = m_prev_h + logf(l_prev_h);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    float scale = 1.0f / sqrtf(128);
    
    dim3 grid((S + 63) / 64, H, B);
    dim3 block(64);
    
    int smem_size = 64 * 1024;
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
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_kernel