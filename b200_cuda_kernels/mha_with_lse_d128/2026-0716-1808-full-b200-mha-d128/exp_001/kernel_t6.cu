#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>
#include <math.h>

#define CU_CHECK(call) do {                                      \
    CUresult _e = (call);                                        \
    if (_e != CUDA_SUCCESS) {                                    \
        fprintf(stderr, "CU error %d at %s:%d\n",               \
                _e, __FILE__, __LINE__);                         \
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

__device__ __forceinline__ int swizzle(int row, int col) {
    return ((row & 7) ^ (col / 8)) * 8 + (col % 8);
}

__device__ __forceinline__ __nv_bfloat16 read_swizzled(const __nv_bfloat16* smem, int row, int col) {
    return smem[row * 64 + swizzle(row, col)];
}

__device__ __forceinline__ void write_swizzled(__nv_bfloat16* smem, int row, int col, __nv_bfloat16 val) {
    smem[row * 64 + swizzle(row, col)] = val;
}

struct SharedStorage {
    alignas(1024) __nv_bfloat16 Q_0[4096]; 
    alignas(1024) __nv_bfloat16 Q_1[4096]; 
    alignas(1024) __nv_bfloat16 Q_2[4096]; 
    alignas(1024) __nv_bfloat16 Q_3[4096]; 
    
    alignas(1024) __nv_bfloat16 K_0[4096];     
    alignas(1024) __nv_bfloat16 K_1[4096];     
    alignas(1024) __nv_bfloat16 V_0[4096];     
    alignas(1024) __nv_bfloat16 V_1[4096];     
    
    alignas(1024) __nv_bfloat16 O_0[4096];     
    alignas(1024) __nv_bfloat16 O_1[4096];     
    alignas(1024) __nv_bfloat16 O_2[4096];     
    alignas(1024) __nv_bfloat16 O_3[4096];     
    
    alignas(1024) __nv_bfloat16 P[4096];       
};

__global__ __launch_bounds__(128) void flash_attention_kernel(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* O, float* LSE,
    uint32_t S_len, uint32_t num_heads) 
{
    uint32_t bh_idx = blockIdx.y;
    uint32_t q_start = blockIdx.x * 128;
    
    if (q_start + 128 > S_len) return;
    
    extern __shared__ __align__(1024) char smem[];
    SharedStorage* s = reinterpret_cast<SharedStorage*>(smem);
    uint32_t tid = threadIdx.x;
    
    const __nv_bfloat16* Q_ptr_bh = Q + (uint64_t)bh_idx * S_len * 128;
    const __nv_bfloat16* K_ptr_bh = K + (uint64_t)bh_idx * S_len * 128;
    const __nv_bfloat16* V_ptr_bh = V + (uint64_t)bh_idx * S_len * 128;
    
    __shared__ float running_max_top[64];
    __shared__ float running_sum_top[64];
    __shared__ float running_max_bot[64];
    __shared__ float running_sum_bot[64];
    
    if (tid < 64) {
        running_max_top[tid] = -1e20f;
        running_sum_top[tid] = 0.0f;
        running_max_bot[tid] = -1e20f;
        running_sum_bot[tid] = 0.0f;
    }
    __syncthreads();
    
    for (int i = 0; i < 4096; ++i) {
        int idx = tid + i * 128;
        s->O_0[idx] = __float2bfloat16(0.0f);
        s->O_1[idx] = __float2bfloat16(0.0f);
        s->O_2[idx] = __float2bfloat16(0.0f);
        s->O_3[idx] = __float2bfloat16(0.0f);
    }
    __syncthreads();
    
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 8; ++j) {
            for (int k = 0; k < 4; ++k) {
                int idx = tid + (i * 8 + j) * 32 + k * 8;
                if (idx < 512) {
                    int r_idx = (idx / 8) % 64;
                    int c_idx = idx % 8;
                    
                    uint4 val0 = {0, 0, 0, 0};
                    int gr_idx = q_start + r_idx;
                    if (gr_idx < S_len) {
                        val0 = *(const uint4*)(Q_ptr_bh + (uint64_t)gr_idx * 128 + 0 + c_idx * 8);
                    }
                    *(uint4*)&s->Q_0[r_idx * 64 + swizzle(r_idx, c_idx * 8)] = val0;
                    
                    uint4 val1 = {0, 0, 0, 0};
                    if (gr_idx < S_len) {
                        val1 = *(const uint4*)(Q_ptr_bh + (uint64_t)gr_idx * 128 + 64 + c_idx * 8);
                    }
                    *(uint4*)&s->Q_1[r_idx * 64 + swizzle(r_idx, c_idx * 8)] = val1;
                    
                    uint4 val2 = {0, 0, 0, 0};
                    gr_idx = q_start + 64 + r_idx;
                    if (gr_idx < S_len) {
                        val2 = *(const uint4*)(Q_ptr_bh + (uint64_t)gr_idx * 128 + 0 + c_idx * 8);
                    }
                    *(uint4*)&s->Q_2[r_idx * 64 + swizzle(r_idx, c_idx * 8)] = val2;
                    
                    uint4 val3 = {0, 0, 0, 0};
                    if (gr_idx < S_len) {
                        val3 = *(const uint4*)(Q_ptr_bh + (uint64_t)gr_idx * 128 + 64 + c_idx * 8);
                    }
                    *(uint4*)&s->Q_3[r_idx * 64 + swizzle(r_idx, c_idx * 8)] = val3;
                }
            }
        }
    }
    __syncthreads();
    
    float scale = 1.0f / sqrtf(128.0f);
    
    for (int c_start = 0; c_start < S_len; c_start += 64) {
        __syncthreads();
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 8; ++j) {
                for (int k = 0; k < 4; ++k) {
                    int idx = tid + (i * 8 + j) * 32 + k * 8;
                    if (idx < 512) {
                        int r_idx = (idx / 8) % 64;
                        int c_idx = idx % 8;
                        
                        uint4 val0 = {0, 0, 0, 0};
                        int gr_idx = c_start + r_idx;
                        if (gr_idx < S_len) {
                            val0 = *(const uint4*)(K_ptr_bh + (uint64_t)gr_idx * 128 + 0 + c_idx * 8);
                        }
                        *(uint4*)&s->K_0[r_idx * 64 + swizzle(r_idx, c_idx * 8)] = val0;
                        
                        uint4 val1 = {0, 0, 0, 0};
                        if (gr_idx < S_len) {
                            val1 = *(const uint4*)(K_ptr_bh + (uint64_t)gr_idx * 128 + 64 + c_idx * 8);
                        }
                        *(uint4*)&s->K_1[r_idx * 64 + swizzle(r_idx, c_idx * 8)] = val1;
                        
                        uint4 val2 = {0, 0, 0, 0};
                        if (gr_idx < S_len) {
                            val2 = *(const uint4*)(V_ptr_bh + (uint64_t)gr_idx * 128 + 0 + c_idx * 8);
                        }
                        *(uint4*)&s->V_0[r_idx * 64 + swizzle(r_idx, c_idx * 8)] = val2;
                        
                        uint4 val3 = {0, 0, 0, 0};
                        if (gr_idx < S_len) {
                            val3 = *(const uint4*)(V_ptr_bh + (uint64_t)gr_idx * 128 + 64 + c_idx * 8);
                        }
                        *(uint4*)&s->V_1[r_idx * 64 + swizzle(r_idx, c_idx * 8)] = val3;
                    }
                }
            }
        }
        
        __syncthreads(); 
        
        float S_partial[32];
        for(int i=0; i<32; ++i) S_partial[i] = 0.0f;
        
        for (int k_iter = 0; k_iter < 64; k_iter += 2) {
            int idx = tid + k_iter * 128;
            int row = idx / 64;
            int col = idx % 64;
            
            if (row < 64) {
                S_partial[col] += __low2float(__bfloat1622low(read_swizzled(s->Q_0, row, k_iter))) 
                                 * __low2float(__bfloat1622low(read_swizzled(s->K_0, k_iter, col)));
                S_partial[col] += __low2float(__bfloat1622high(read_swizzled(s->Q_0, row, k_iter))) 
                                 * __low2float(__bfloat1622high(read_swizzled(s->K_0, k_iter, col)));
                
                S_partial[col] += __low2float(__bfloat1622low(read_swizzled(s->Q_1, row, k_iter))) 
                                 * __low2float(__bfloat1622low(read_swizzled(s->K_1, k_iter, col)));
                S_partial[col] += __low2float(__bfloat1622high(read_swizzled(s->Q_1, row, k_iter))) 
                                 * __low2float(__bfloat1622high(read_swizzled(s->K_1, k_iter, col)));
            } else {
                S_partial[col] += __low2float(__bfloat1622low(read_swizzled(s->Q_2, row - 64, k_iter))) 
                                 * __low2float(__bfloat1622low(read_swizzled(s->K_0, k_iter, col)));
                S_partial[col] += __low2float(__bfloat1622high(read_swizzled(s->Q_2, row - 64, k_iter))) 
                                 * __low2float(__bfloat1622high(read_swizzled(s->K_0, k_iter, col)));
                
                S_partial[col] += __low2float(__bfloat1622low(read_swizzled(s->Q_3, row - 64, k_iter))) 
                                 * __low2float(__bfloat1622low(read_swizzled(s->K_1, k_iter, col)));
                S_partial[col] += __low2float(__bfloat1622high(read_swizzled(s->Q_3, row - 64, k_iter))) 
                                 * __low2float(__bfloat1622high(read_swizzled(s->K_1, k_iter, col)));
            }
        }
        
        int row = tid / 2;
        int col = (tid % 2) * 32;
        
        if (tid < 64) {
            if (q_start + tid >= S_len) {
                for(int i=0; i<32; ++i) S_partial[i] = -1e20f;
            } else {
                for(int i=0; i<32; ++i) {
                    if (c_start + col + i >= S_len) S_partial[i] = -1e20f;
                }
            }
        } else {
            if (q_start + 64 + tid - 64 >= S_len) {
                for(int i=0; i<32; ++i) S_partial[i] = -1e20f;
            } else {
                for(int i=0; i<32; ++i) {
                    if (c_start + col + i >= S_len) S_partial[i] = -1e20f;
                }
            }
        }
        
        float l_max = -1e20f;
        for(int i=0; i<32; ++i) {
            l_max = fmaxf(l_max, S_partial[i] * scale);
        }
        l_max = fmaxf(l_max, __shfl_xor_sync(0xffffffff, l_max, 1));
        
        float r_max = (tid < 64) ? running_max_top[row] : running_max_bot[row];
        float new_max = fmaxf(r_max, l_max);
        
        float l_sum = 0.0f;
        for(int i=0; i<32; ++i) {
            float v = __expf(S_partial[i] * scale - new_max);
            l_sum += v;
            S_partial[i] = v;
        }
        l_sum += __shfl_xor_sync(0xffffffff, l_sum, 1);
        
        float r_sum = (tid < 64) ? running_sum_top[row] : running_sum_bot[row];
        if (r_max == -1e20f) {
            r_sum = l_sum;
        } else {
            r_sum = r_sum * __expf(r_max - new_max);
        }
        
        if (tid % 2 == 0) {
            if (tid < 64) {
                running_max_top[row] = new_max;
                running_sum_top[row] = r_sum;
            } else {
                running_max_bot[row] = new_max;
                running_sum_bot[row] = r_sum;
            }
        }
        
        float scale_factor = (r_max == -1e20f) ? 0.0f : __expf(r_max - new_max);
        
        if (scale_factor != 1.0f) {
            __nv_bfloat16* my_s_O_0 = (tid < 64) ? s->O_0 : s->O_2;
            __nv_bfloat16* my_s_O_1 = (tid < 64) ? s->O_1 : s->O_3;
            for (int i = 0; i < 32; ++i) {
                float f0 = __low2float(__bfloat1622low(read_swizzled(my_s_O_0, row, col + i)));
                float f1 = __low2float(__bfloat1622low(read_swizzled(my_s_O_1, row, col + i)));
                write_swizzled(my_s_O_0, row, col + i, __float2bfloat16(f0 * scale_factor));
                write_swizzled(my_s_O_1, row, col + i, __float2bfloat16(f1 * scale_factor));
            }
        }
        
        for(int i=0; i<32; ++i) {
            __nv_bfloat16 val = __float2bfloat16(S_partial[i]);
            write_swizzled(s->P, row, col + i, val);
        }
        
        __syncthreads(); 
        
        uint32_t O_reg[2][4]; 
        for(int i=0; i<2; ++i) {
            for(int j=0; j<4; ++j) {
                O_reg[i][j] = __float_as_uint(0.0f);
            }
        }
        
        for (int c_iter = 0; c_iter < 128; c_iter += 2) {
            int idx = tid + c_iter * 128;
            int row = idx / 64;
            int col = idx % 64;
            
            float p = __low2float(__bfloat1622low(read_swizzled(s->P, row, c_iter)));
            
            float v0 = __low2float(__bfloat1622low(read_swizzled(s->V_0, c_iter, col)));
            float v1 = __low2float(__bfloat1622low(read_swizzled(s->V_1, c_iter, col)));
            
            O_reg[0][col / 8] += p * v0;
            O_reg[1][col / 8] += p * v1;
        }
        
        __syncthreads(); 
        
        __nv_bfloat16* my_s_O_0_out = (tid < 64) ? s->O_0 : s->O_2;
        __nv_bfloat16* my_s_O_1_out = (tid < 64) ? s->O_1 : s->O_3;
        
        for (int i = 0; i < 4; ++i) {
            int idx = tid + i * 128;
            int row = idx / 64;
            int col = idx % 64;
            
            float f0 = __low2float(O_reg[0][i]);
            float f1 = __low2float(O_reg[1][i]);
            
            float current_o0 = __low2float(__bfloat1622low(read_swizzled(my_s_O_0_out, row, col)));
            float current_o1 = __low2float(__bfloat1622low(read_swizzled(my_s_O_1_out, row, col)));
            
            write_swizzled(my_s_O_0_out, row, col, __float2bfloat16(current_o0 + f0));
            write_swizzled(my_s_O_1_out, row, col, __float2bfloat16(current_o1 + f1));
        }
        
        __syncthreads();
    }
    
    float inv_sum_top[64];
    float inv_sum_bot[64];
    if (tid < 64) {
        inv_sum_top[tid] = (running_sum_top[tid] > 0.0f) ? 1.0f / running_sum_top[tid] : 0.0f;
        inv_sum_bot[tid] = (running_sum_bot[tid] > 0.0f) ? 1.0f / running_sum_bot[tid] : 0.0f;
    }
    __syncthreads();
    
    for (int i = 0; i < 4096; ++i) {
        int idx = tid + i * 128;
        int row = idx / 64;
        int col = idx % 64;
        
        float inv_sum = (tid < 64) ? inv_sum_top[row] : inv_sum_bot[row];
        
        float f0 = __low2float(__bfloat1622low(read_swizzled(s->O_0, row, col)));
        float f1 = __low2float(__bfloat1622low(read_swizzled(s->O_1, row, col)));
        write_swizzled(s->O_0, row, col, __float2bfloat16(f0 * inv_sum));
        write_swizzled(s->O_1, row, col, __float2bfloat16(f1 * inv_sum));
        
        f0 = __low2float(__bfloat1622low(read_swizzled(s->O_2, row, col)));
        f1 = __low2float(__bfloat1622low(read_swizzled(s->O_3, row, col)));
        write_swizzled(s->O_2, row, col, __float2bfloat16(f0 * inv_sum));
        write_swizzled(s->O_3, row, col, __float2bfloat16(f1 * inv_sum));
    }
    
    __syncthreads(); 
    
    __nv_bfloat16* my_O = O + (uint64_t)bh_idx * S_len * 128;
    
    for(int i = 0; i < 4096; ++i) {
        int idx = tid + i * 128;
        int row = idx / 64;
        int col = idx % 64;
        
        __nv_bfloat16 val0 = read_swizzled(s->O_0, row, col);
        __nv_bfloat16 val1 = read_swizzled(s->O_1, row, col);
        
        uint32_t s_idx = q_start + row;
        uint32_t d_idx = col;
        
        if (s_idx < S_len) {
            uint64_t g_idx = (uint64_t)s_idx * 128 + d_idx;
            *reinterpret_cast<uint2*>(&my_O[g_idx]) = *reinterpret_cast<uint2*>(&val0);
            d_idx += 64;
            if (d_idx < 128) {
                *reinterpret_cast<__nv_bfloat16*>(&my_O[g_idx + 64]) = val1;
            }
        }
    }
    
    for(int i = 0; i < 4096; ++i) {
        int idx = tid + i * 128;
        int row = idx / 64;
        int col = idx % 64;
        
        __nv_bfloat16 val0 = read_swizzled(s->O_2, row, col);
        __nv_bfloat16 val1 = read_swizzled(s->O_3, row, col);
        
        uint32_t s_idx = q_start + row + 64;
        uint32_t d_idx = col;
        
        if (s_idx < S_len) {
            uint64_t g_idx = (uint64_t)s_idx * 128 + d_idx;
            *reinterpret_cast<uint2*>(&my_O[g_idx]) = *reinterpret_cast<uint2*>(&val0);
            d_idx += 64;
            if (d_idx < 128) {
                *reinterpret_cast<__nv_bfloat16*>(&my_O[g_idx + 64]) = val1;
            }
        }
    }
    
    if (tid < 64) {
        uint32_t s_idx_0 = q_start + tid;
        if (s_idx_0 < S_len) {
            LSE[(uint64_t)bh_idx * S_len + s_idx_0] = running_max_top[tid] + logf(running_sum_top[tid]);
        }
        
        uint32_t s_idx_1 = q_start + tid + 64;
        if (s_idx_1 < S_len) {
            LSE[(uint64_t)bh_idx * S_len + s_idx_1] = running_max_bot[tid] + logf(running_sum_bot[tid]);
        }
    }
}

namespace tvm_ffi_mha_lse {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    uint32_t B = Q.size(0);
    uint32_t H = Q.size(1);
    uint32_t S_len = Q.size(2);
    uint32_t D = Q.size(3); 
    
    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_ptr = static_cast<float*>(LSE.data_ptr());
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    uint32_t num_blocks_x = (S_len + 127) / 128;
    dim3 grid(num_blocks_x, B * H);
    dim3 block(128);
    
    uint32_t smem_size = sizeof(SharedStorage) + 1024;
    CUDA_CHECK(cudaFuncSetAttribute(
        flash_attention_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size
    ));
    
    flash_attention_kernel<<<grid, block, smem_size, stream>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr, S_len, H
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_mha_lse