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

__device__ __forceinline__ uint32_t swizzle_128B(uint32_t row, uint32_t col) {
    uint32_t span_idx = col >> 3;
    uint32_t offset = col & 7;
    uint32_t swizzled_span = (row & 7) ^ span_idx;
    return ((row << 6) + (swizzled_span << 3)) + offset;
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= (0u << 15);   
    d |= (0u << 16);   
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ void umma_f16_cg1_fn(
    void* d_smem, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(d_smem)), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

struct SharedStorage {
    __align__(1024) __nv_bfloat16 smem_Q_0_0[4096];
    __align__(1024) __nv_bfloat16 smem_Q_0_1[4096];
    __align__(1024) __nv_bfloat16 smem_Q_1_0[4096];
    __align__(1024) __nv_bfloat16 smem_Q_1_1[4096];
    
    __align__(1024) __nv_bfloat16 smem_K_0[4096];
    __align__(1024) __nv_bfloat16 smem_K_1[4096];
    __align__(1024) __nv_bfloat16 smem_V_0[4096];
    __align__(1024) __nv_bfloat16 smem_V_1[4096];
    
    __align__(1024) __nv_bfloat16 smem_P_0[4096];
    __align__(1024) __nv_bfloat16 smem_P_1[4096];
    __align__(1024) __nv_bfloat16 smem_S_0[4096];
    __align__(1024) __nv_bfloat16 smem_S_1[4096];
    
    __align__(1024) __nv_bfloat16 smem_O_0[4096]; 
    __align__(1024) __nv_bfloat16 smem_O_1[4096]; 
};

__device__ __forceinline__ void load_Q_vec(
    const __nv_bfloat16* Q, uint32_t S_len, uint32_t q_start, 
    uint32_t batch_head, uint32_t num_heads, uint32_t tid,
    __nv_bfloat16* s00, __nv_bfloat16* s01, __nv_bfloat16* s10, __nv_bfloat16* s11) 
{
    const __nv_bfloat16* my_Q = Q + (uint64_t)batch_head * S_len * 128;
    
    for(int i = 0; i < 128 * 128 / 2; ++i) {
        uint32_t elem_idx = tid + i * 128;
        uint32_t row = elem_idx / 128;
        uint32_t col = elem_idx % 128;
        
        uint32_t s_idx = q_start + row;
        uint32_t d_idx = col;
        
        uint2 val = {0, 0};
        if (s_idx < S_len && d_idx + 1 < 128) {
            uint64_t g_idx = (uint64_t)s_idx * 128 + d_idx;
            val = *reinterpret_cast<const uint2*>(&my_Q[g_idx]);
        }
        
        if (row < 64) {
            if (d_idx < 64) {
                *reinterpret_cast<uint2*>(&s00[swizzle_128B(row, d_idx)]) = val;
            } else {
                *reinterpret_cast<uint2*>(&s01[swizzle_128B(row, d_idx - 64)]) = val;
            }
        } else {
            if (d_idx < 64) {
                *reinterpret_cast<uint2*>(&s10[swizzle_128B(row - 64, d_idx)]) = val;
            } else {
                *reinterpret_cast<uint2*>(&s11[swizzle_128B(row - 64, d_idx - 64)]) = val;
            }
        }
    }
}

__device__ __forceinline__ void load_K_vec(
    const __nv_bfloat16* K, uint32_t S_len, uint32_t c_start, 
    uint32_t batch_head, uint32_t tid,
    __nv_bfloat16* sk0, __nv_bfloat16* sk1) 
{
    const __nv_bfloat16* my_K = K + (uint64_t)batch_head * S_len * 128;
    
    for(int i = 0; i < 64 * 128 / 2; ++i) {
        uint32_t elem_idx = tid + i * 128;
        uint32_t row = elem_idx / 128;
        uint32_t col = elem_idx % 128;
        
        uint32_t s_idx = c_start + row;
        uint32_t d_idx = col;
        
        uint2 val = {0, 0};
        if (s_idx < S_len && d_idx + 1 < 128) {
            uint64_t g_idx = (uint64_t)s_idx * 128 + d_idx;
            val = *reinterpret_cast<const uint2*>(&my_K[g_idx]);
        }
        
        if (d_idx < 64) {
            *reinterpret_cast<uint2*>(&sk0[swizzle_128B(row, d_idx)]) = val;
        } else {
            *reinterpret_cast<uint2*>(&sk1[swizzle_128B(row, d_idx - 64)]) = val;
        }
    }
}

__device__ __forceinline__ void load_V_vec(
    const __nv_bfloat16* V, uint32_t S_len, uint32_t c_start, 
    uint32_t batch_head, uint32_t tid,
    __nv_bfloat16* sv0, __nv_bfloat16* sv1) 
{
    const __nv_bfloat16* my_V = V + (uint64_t)batch_head * S_len * 128;

    for(int i = 0; i < 64 * 128 / 2; ++i) {
        uint32_t elem_idx = tid + i * 128;
        uint32_t row = elem_idx / 128;
        uint32_t col = elem_idx % 128;
        
        uint32_t s_idx = c_start + row;
        uint32_t d_idx = col;
        
        uint2 val = {0, 0};
        if (s_idx < S_len && d_idx + 1 < 128) {
            uint64_t g_idx = (uint64_t)s_idx * 128 + d_idx;
            val = *reinterpret_cast<const uint2*>(&my_V[g_idx]);
        }
        
        if (d_idx < 64) {
            *reinterpret_cast<uint2*>(&sv0[swizzle_128B(row, d_idx)]) = val;
        } else {
            *reinterpret_cast<uint2*>(&sv1[swizzle_128B(row, d_idx - 64)]) = val;
        }
    }
}

__device__ __forceinline__ void store_O_vec(
    __nv_bfloat16* O, uint32_t S_len, uint32_t q_start, 
    uint32_t batch_head, uint32_t num_heads, uint32_t tid,
    __nv_bfloat16* s0, __nv_bfloat16* s1) 
{
    __nv_bfloat16* my_O = O + (uint64_t)batch_head * S_len * 128;
    
    for(int i = 0; i < 64 * 128 / 2; ++i) {
        uint32_t elem_idx = tid + i * 128;
        uint32_t row = elem_idx / 128;
        uint32_t col = elem_idx % 128;
        
        __nv_bfloat16 val0 = s0[swizzle_128B(row, col)];
        __nv_bfloat16 val1 = s1[swizzle_128B(row, col)];
        
        uint32_t s_idx = q_start + row;
        uint32_t d_idx = col;
        
        if (s_idx < S_len) {
            uint64_t g_idx = (uint64_t)s_idx * 128 + d_idx;
            *reinterpret_cast<uint2*>(&my_O[g_idx]) = *reinterpret_cast<uint2*>(&val0);
            d_idx++;
            if (d_idx < 128) {
                *reinterpret_cast<__nv_bfloat16*>(&my_O[g_idx + 1]) = val1;
            }
        }
    }
}

__global__ __launch_bounds__(128) void flash_attention_kernel(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* O, float* LSE,
    uint32_t S_len, uint32_t num_heads) 
{
    uint32_t batch_head = blockIdx.y;
    uint32_t q_start = blockIdx.x * 128;
    
    if (q_start >= S_len) return;
    
    extern __shared__ char smem[];
    SharedStorage& s = *reinterpret_cast<SharedStorage*>(smem);
    uint32_t tid = threadIdx.x;
    
    float scale = 1.0f / sqrtf(128.0f); 
    
    // Initialize O to 0
    for(int i = 0; i < 4096; ++i) {
        uint32_t idx = tid + i * 128;
        s.smem_O_0[idx] = __float2bfloat16(0.0f);
        s.smem_O_1[idx] = __float2bfloat16(0.0f);
    }
    __syncthreads();
    
    // Load Q tile
    load_Q_vec(Q, S_len, q_start, batch_head, num_heads, tid, 
               s.smem_Q_0_0, s.smem_Q_0_1, s.smem_Q_1_0, s.smem_Q_1_1);
    
    __shared__ float running_max_0[128];
    __shared__ float running_max_1[128];
    __shared__ float running_sum_0[128];
    __shared__ float running_sum_1[128];
    if (tid < 128) {
        running_max_0[tid] = -1e20f;
        running_max_1[tid] = -1e20f;
        running_sum_0[tid] = 0.0f;
        running_sum_1[tid] = 0.0f;
    }
    __syncthreads();
    
    for (int c_start = 0; c_start < S_len; c_start += 64) {
        
        load_K_vec(K, S_len, c_start, batch_head, tid, s.smem_K_0, s.smem_K_1);
        load_V_vec(V, S_len, c_start, batch_head, tid, s.smem_V_0, s.smem_V_1);
        
        __syncthreads(); 
        
        uint32_t idesc = make_instr_desc_fn(64, 64);
        
        for (int k_iter = 0; k_iter < 128 / 64; ++k_iter) {
            int idx = k_iter * 64 + (tid / 2);
            
            uint64_t desc_A0 = make_smem_desc_sm100_fn(s.smem_Q_0_0 + (idx < 64 ? 0 : (idx - 64) << 6), 8192, 1024);
            uint64_t desc_A1 = make_smem_desc_sm100_fn(s.smem_Q_0_1 + (idx < 64 ? 0 : (idx - 64) << 6), 8192, 1024);
            uint64_t desc_B0 = make_smem_desc_sm100_fn(s.smem_K_0 + (k_iter << 7), 8192, 1024);
            uint64_t desc_B1 = make_smem_desc_sm100_fn(s.smem_K_1 + (k_iter << 7), 8192, 1024);
            
            for (int k = 0; k < 64; k += 16) {
                int accum = 1;
                if (k == 0) accum = 0;
                uint64_t desc_S0 = make_smem_desc_sm100_fn(s.smem_S_0 + (idx < 64 ? 0 : (idx - 64) << 6), 8192, 1024);
                uint64_t desc_S1 = make_smem_desc_sm100_fn(s.smem_S_1 + (idx < 64 ? 0 : (idx - 64) << 6), 8192, 1024);
                
                umma_f16_cg1_fn(desc_S0, desc_A0 + (k << 1), desc_B0 + (k << 1), idesc, accum);
                umma_f16_cg1_fn(desc_S1, desc_A1 + (k << 1), desc_B1 + (k << 1), idesc, accum);
            }
        }
        
        __syncthreads(); 
        
        int idx = tid / 2;
        int half = tid % 2;
        
        float max_val_0 = -1e20f;
        float max_val_1 = -1e20f;
        
        for(int i=0; i<32; ++i) {
            max_val_0 = fmaxf(max_val_0, __uint_as_float(s.smem_S_0[swizzle_128B(idx, half*32 + i)]) * scale);
            max_val_1 = fmaxf(max_val_1, __uint_as_float(s.smem_S_1[swizzle_128B(idx, half*32 + i)]) * scale);
        }
        
        max_val_0 = fmaxf(max_val_0, __shfl_xor_sync(0xffffffff, max_val_0, 1));
        max_val_1 = fmaxf(max_val_1, __shfl_xor_sync(0xffffffff, max_val_1, 1));
        
        float r_max_0 = running_max_0[idx];
        float r_max_1 = running_max_1[idx];
        
        r_max_0 = fmaxf(r_max_0, max_val_0);
        r_max_1 = fmaxf(r_max_1, max_val_1);
        
        float sum_val_0 = 0.0f;
        float sum_val_1 = 0.0f;
        
        for(int i=0; i<32; ++i) {
            float v0 = __expf(__uint_as_float(s.smem_S_0[swizzle_128B(idx, half*32 + i)]) * scale - max_val_0);
            float v1 = __expf(__uint_as_float(s.smem_S_1[swizzle_128B(idx, half*32 + i)]) * scale - max_val_1);
            sum_val_0 += v0;
            sum_val_1 += v1;
            
            s.smem_S_0[swizzle_128B(idx, half*32 + i)] = __float2bfloat16(v0);
            s.smem_S_1[swizzle_128B(idx, half*32 + i)] = __float2bfloat16(v1);
        }
        
        sum_val_0 += __shfl_xor_sync(0xffffffff, sum_val_0, 1);
        sum_val_1 += __shfl_xor_sync(0xffffffff, sum_val_1, 1);
        
        float r_sum_0 = running_sum_0[idx];
        float r_sum_1 = running_sum_1[idx];
        
        if (c_start + 64 > S_len) {
            r_sum_0 += sum_val_0;
            r_sum_1 += sum_val_1;
        } else {
            r_sum_0 = r_sum_0 * __expf(r_max_0 - r_max_0) + sum_val_0;
            r_sum_1 = r_sum_1 * __expf(r_max_1 - r_max_1) + sum_val_1;
        }
        
        if (tid % 2 == 0) {
            running_max_0[idx] = r_max_0;
            running_max_1[idx] = r_max_1;
            running_sum_0[idx] = r_sum_0;
            running_sum_1[idx] = r_sum_1;
        }
        
        __syncthreads();
        
        float scale_factor_0 = __expf(running_max_0[idx] - r_max_0);
        float scale_factor_1 = __expf(running_max_1[idx] - r_max_1);
        
        if (c_start > 0 && tid < 128) {
            int row = tid;
            int col = (tid % 2) * 32 + (tid / 4) * 2;
            if (col < 64) {
                float f0 = __uint_as_float(s.smem_O_0[swizzle_128B(row % 64, col)]);
                float f1 = __uint_as_float(s.smem_O_0[swizzle_128B(row % 64, col + 1)]);
                s.smem_O_0[swizzle_128B(row % 64, col)] = __float2bfloat16(f0 * scale_factor_0);
                s.smem_O_0[swizzle_128B(row % 64, col + 1)] = __float2bfloat16(f1 * scale_factor_0);
                
                float g0 = __uint_as_float(s.smem_O_1[swizzle_128B(row % 64, col)]);
                float g1 = __uint_as_float(s.smem_O_1[swizzle_128B(row % 64, col + 1)]);
                s.smem_O_1[swizzle_128B(row % 64, col)] = __float2bfloat16(g0 * scale_factor_1);
                s.smem_O_1[swizzle_128B(row % 64, col + 1)] = __float2bfloat16(g1 * scale_factor_1);
            }
        }
        
        __syncthreads();
        
        // P @ V
        __syncthreads();

        for (int k = 0; k < 64; k += 16) {
            int accum = 1;
            if (k == 0) accum = 0;
            uint64_t desc_P0 = make_smem_desc_sm100_fn(s.smem_P_0 + (idx < 64 ? 0 : (idx - 64) << 6), 8192, 1024);
            uint64_t desc_V0 = make_smem_desc_sm100_fn(s.smem_V_0 + (k << 6), 8192, 1024);
            
            uint64_t desc_O0 = make_smem_desc_sm100_fn(s.smem_O_0 + (idx < 64 ? 0 : (idx - 64) << 6), 8192, 1024);
            umma_f16_cg1_fn(desc_O0, desc_P0, desc_V0, idesc, accum);
        }
        for (int k = 0; k < 64; k += 16) {
            int accum = 1;
            if (k == 0) accum = 0;
            uint64_t desc_P1 = make_smem_desc_sm100_fn(s.smem_P_1 + (idx < 64 ? 0 : (idx - 64) << 6), 8192, 1024);
            uint64_t desc_V1 = make_smem_desc_sm100_fn(s.smem_V_1 + (k << 6), 8192, 1024);
            
            uint64_t desc_O1 = make_smem_desc_sm100_fn(s.smem_O_1 + (idx < 64 ? 0 : (idx - 64) << 6), 8192, 1024);
            umma_f16_cg1_fn(desc_O1, desc_P1, desc_V1, idesc, accum);
        }

        __syncthreads();
    }
    
    // Final Normalization
    __syncthreads();
    for(int i = 0; i < 4096; ++i) {
        uint32_t idx = tid + i * 128;
        uint32_t row = idx / 64;
        
        float final_sum_0 = running_sum_0[row];
        float final_sum_1 = running_sum_1[row];
        
        float inv_sum_0 = (final_sum_0 > 0.0f) ? 1.0f / final_sum_0 : 0.0f;
        float inv_sum_1 = (final_sum_1 > 0.0f) ? 1.0f / final_sum_1 : 0.0f;
        
        float f0 = __uint_as_float(s.smem_O_0[idx]);
        float f1 = __uint_as_float(s.smem_O_1[idx]);
        
        s.smem_O_0[idx] = __float2bfloat16(f0 * inv_sum_0);
        s.smem_O_1[idx] = __float2bfloat16(f1 * inv_sum_1);
    }
    __syncthreads();
    
    store_O_vec(O, S_len, q_start, batch_head, num_heads, tid, s.smem_O_0, s.smem_O_1);
    
    // Compute and write LogSumExp (LSE)
    if (tid < 128) {
        int row = tid;
        int half = 0; 
        
        float r_max_0 = running_max_0[row];
        float r_sum_0 = running_sum_0[row];
        float lse_0 = r_max_0 + logf(r_sum_0);
        
        uint32_t s_idx_0 = q_start + row;
        if (s_idx_0 < S_len) {
            LSE[(uint64_t)batch_head * S_len + s_idx_0] = lse_0;
        }
        
        float r_max_1 = running_max_1[row];
        float r_sum_1 = running_sum_1[row];
        float lse_1 = r_max_1 + logf(r_sum_1);
        
        uint32_t s_idx_1 = q_start + row + 64;
        if (s_idx_1 < S_len) {
            LSE[(uint64_t)batch_head * S_len + s_idx_1] = lse_1;
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
    
    CUDA_CHECK(cudaFuncSetAttribute(
        flash_attention_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        sizeof(SharedStorage)
    ));
    
    flash_attention_kernel<<<grid, block, sizeof(SharedStorage), stream>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr, S_len, H
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_mha_lse