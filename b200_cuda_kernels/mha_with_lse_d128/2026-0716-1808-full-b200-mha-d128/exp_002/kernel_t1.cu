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

namespace mha_with_lse_d128 {

#define bf16x uint32_t

__device__ __forceinline__ float __low(float2 x) {
    asm volatile("cvt.f32.b16 %0, %1;" : "=f"(x) : "h"(*(uint16_t*)&x));
    return x;
}

__device__ __forceinline__ float __high(float2 x) {
    asm volatile("cvt.f32.b16 %0, %1;" : "=f"(x) : "h"(*(uint16_t*)&((uint32_t&)x)[1]));
    return x;
}

__device__ __forceinline__ float2 __bfloat1622mul(bf16x a, bf16x b) {
    float2 res;
    uint32_t* a_ptr = (uint32_t*)&a;
    uint32_t* b_ptr = (uint32_t*)&b;
    asm volatile("mma.sync.aligned.16x8x8.f32.bf16 {%0, %1}, %2, %3, 0;\n"
                 : "=f"(*((float*)&res)), "=f"(*((float*)&res + 1))
                 : "r"(*a_ptr), "r"(*b_ptr));
    return res;
}

__device__ __forceinline__ void write_swizzled_128B(__nv_bfloat16* smem, int row, int col, __nv_bfloat16 val) {
    int chunk_x = col / 8;
    int swizzled_chunk_x = chunk_x ^ (row % 8);
    smem[row * 128 + swizzled_chunk_x * 8 + (col % 8)] = val;
}

__global__ __launch_bounds__(128, 1) void mha_kernel(const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V, __nv_bfloat16* O, float* LSE, int S) {
    int bh = blockIdx.x;
    int q_tile = blockIdx.y;
    int tid = threadIdx.x;
    
    const __nv_bfloat16* Q_bh = Q + bh * S * 128;
    const __nv_bfloat16* K_bh = K + bh * S * 128;
    const __nv_bfloat16* V_bh = V + bh * S * 128;
    __nv_bfloat16* O_bh = O + bh * S * 128;
    float* LSE_bh = LSE + bh * S;

    if (q_tile * 128 >= S) return;
    
    int num_k_tiles = (S + 127) / 128;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* smem_Q = (__nv_bfloat16*)(smem_raw + 0);
    __nv_bfloat16* smem_K = (__nv_bfloat16*)(smem_raw + 32768);
    __nv_bfloat16* smem_V = (__nv_bfloat16*)(smem_raw + 65536);
    __nv_bfloat16* smem_P = (__nv_bfloat16*)(smem_raw + 98304);
    __nv_bfloat16* smem_O = (__nv_bfloat16*)(smem_raw + 131072);

    const __nv_bfloat16* global_Q = Q_bh + q_tile * 128 * 128;
    
    for (int col = 0; col < 128; col += 4) {
        int m_idx = tid;
        if (q_tile * 128 + m_idx < S) {
            int chunk_x = col / 8;
            int swizzled_chunk_x = chunk_x ^ (m_idx % 8);
            uint32_t smem_offset = m_idx * 128 + swizzled_chunk_x * 8 + (col % 8);
            *(uint2*)&smem_Q[smem_offset] = *(uint2*)&global_Q[m_idx * 128 + col];
        } else {
            int chunk_x = col / 8;
            int swizzled_chunk_x = chunk_x ^ (m_idx % 8);
            uint32_t smem_offset = m_idx * 128 + swizzled_chunk_x * 8 + (col % 8);
            *(uint2*)&smem_Q[smem_offset] = make_uint2(0, 0);
        }
    }
    
    float m_prev = -INFINITY;
    float l_prev = 0.0f;

    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        const __nv_bfloat16* global_K = K_bh + k_tile * 128 * 128;
        const __nv_bfloat16* global_V = V_bh + k_tile * 128 * 128;
        
        for (int col = 0; col < 128; col += 4) {
            int s_idx = tid;
            if (k_tile * 128 + s_idx < S) {
                int chunk_x = col / 8;
                int swizzled_chunk_x = chunk_x ^ (s_idx % 8);
                uint32_t smem_offset = s_idx * 128 + swizzled_chunk_x * 8 + (col % 8);
                *(uint2*)&smem_K[smem_offset] = *(uint2*)&global_K[s_idx * 128 + col];
                *(uint2*)&smem_V[smem_offset] = *(uint2*)&global_V[s_idx * 128 + col];
            } else {
                int chunk_x = col / 8;
                int swizzled_chunk_x = chunk_x ^ (s_idx % 8);
                uint32_t smem_offset = s_idx * 128 + swizzled_chunk_x * 8 + (col % 8);
                *(uint2*)&smem_K[smem_offset] = make_uint2(0, 0);
                *(uint2*)&smem_V[smem_offset] = make_uint2(0, 0);
            }
        }
        
        __syncthreads();
        
        int m_idx = tid;
        float S_acc[128] = {0};
        
        for (int n_block = 0; n_block < 128; n_block += 16) {
            for (int d_block = 0; d_block < 128; d_block += 8) {
                bf16x Q_vec[4];
                uint32_t q_offset_base = m_idx * 128 + ((d_block / 8) ^ (m_idx % 8)) * 8 + (d_block % 8);
                *(uint2*)&Q_vec[0] = *(uint2*)&smem_Q[q_offset_base];
                *(uint2*)&Q_vec[2] = *(uint2*)&smem_Q[q_offset_base + 4];
                
                for (int n_idx = 0; n_idx < 8; ++n_idx) {
                    int n = n_block + n_idx * 2;
                    uint32_t k_offset_base = n * 128 + ((d_block / 8) ^ (n % 8)) * 8 + (d_block % 8);
                    
                    bf16x K_vec_flat[4];
                    *(uint2*)&K_vec_flat[0] = *(uint2*)&smem_K[k_offset_base];
                    *(uint2*)&K_vec_flat[2] = *(uint2*)&smem_K[k_offset_base + 4];
                    
                    for(int j = 0; j < 4; j++) {
                        float2 res = __bfloat1622mul(Q_vec[j], K_vec_flat[j]);
                        S_acc[n_block + n_idx * 2] += __low(res) + __high(res);
                    }
                }
            }
        }
        
        float m_curr = -INFINITY;
        for(int i = 0; i < 128; i++) {
            if (k_tile * 128 + i >= S) {
                S_acc[i] = -INFINITY;
            } else {
                S_acc[i] *= (1.0f / sqrtf(128.0f));
            }
            m_curr = fmaxf(m_curr, S_acc[i]);
        }
        
        float m_new = fmaxf(m_prev, m_curr);
        float l_new = 0.0f;
        
        for(int i = 0; i < 128; i++) {
            float p = 0.0f;
            if (k_tile * 128 + i < S) {
                p = expf(S_acc[i] - m_new);
            }
            l_new += p;
            write_swizzled_128B(smem_P, m_idx, i, __float2bfloat16(p));
        }
        
        float scale_o = expf(m_prev - m_new);
        l_prev = l_prev * scale_o + l_new;
        m_prev = m_new;
        
        __syncthreads(); 
        
        for(int i = 0; i < 128; i++) {
            float o_val = smem_O[m_idx * 128 + i];
            smem_O[m_idx * 128 + i] = o_val * scale_o;
        }
        
        for (int d_block = 0; d_block < 128; d_block += 8) {
            for (int n_block = 0; n_block < 128; n_block += 16) {
                bf16x P_vec_flat[4];
                uint32_t p_offset_base = m_idx * 128 + ((n_block / 8) ^ (m_idx % 8)) * 8 + (n_block % 8);
                *(uint2*)&P_vec_flat[0] = *(uint2*)&smem_P[p_offset_base];
                *(uint2*)&P_vec_flat[2] = *(uint2*)&smem_P[p_offset_base + 4];
                
                for (int n_idx = 0; n_idx < 8; ++n_idx) {
                    int n = n_block + n_idx * 2;
                    uint32_t v_offset_base = n * 128 + ((d_block / 8) ^ (n % 8)) * 8 + (d_block % 8);
                    
                    bf16x V_vec_flat[4];
                    *(uint2*)&V_vec_flat[0] = *(uint2*)&smem_V[v_offset_base];
                    *(uint2*)&V_vec_flat[2] = *(uint2*)&smem_V[v_offset_base + 4];
                    
                    for(int j = 0; j < 4; j++) {
                        float2 res = __bfloat1622mul(P_vec_flat[j], V_vec_flat[j]);
                        smem_O[m_idx * 128 + d_block + j * 2] += __low(res);
                        smem_O[m_idx * 128 + d_block + j * 2 + 1] += __high(res);
                    }
                }
            }
        }
        
        __syncthreads();
    }
    
    for(int d = 0; d < 128; d += 4) {
        if (q_tile * 128 + m_idx < S) {
            __nv_bfloat16* out = O_bh + (q_tile * 128 + m_idx) * 128 + d;
            __nv_bfloat16 o0 = __float2bfloat16(smem_O[m_idx * 128 + d] / l_prev);
            __nv_bfloat16 o1 = __float2bfloat16(smem_O[m_idx * 128 + d + 1] / l_prev);
            *(uint2*)out = make_uint2(*((uint16_t*)&o1), *((uint16_t*)&o0));
        }
    }
    
    if (tid == 0 && q_tile * 128 < S) {
        LSE_bh[q_tile * 128 + tid] = m_prev + logf(l_prev);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int64_t S = Q.size(2);
    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_data = static_cast<float*>(LSE.data_ptr());
    
    int num_k_tiles = (S + 127) / 128;
    dim3 grid(num_k_tiles, Q.size(0) * Q.size(1));
    dim3 block(128);
    
    int smem_size = 163840;
    CUDA_CHECK(cudaFuncSetAttribute(
        mha_with_lse_d128::mha_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size));
        
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    mha_kernel<<<grid, block, smem_size, stream>>>(Q_data, K_data, V_data, O_data, LSE_data, S);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

}  // namespace mha_with_lse_d128