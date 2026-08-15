#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <math.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

using namespace nvcuda;

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_mha {

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;" ::: "memory");
}

template<int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ void copy_tile_128b(uint8_t* smem, const uint8_t* gmem, uint64_t g_offset, uint32_t tid) {
    for (int i = 0; i < 2; ++i) {
        uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem)) + tid * 256 + i * 128;
        uint64_t gmem_addr = (uint64_t)gmem + g_offset + tid * 256 + i * 128;
        asm volatile("cp.async.cached.shared.global [%0], [%1], 128;" 
                     :: "r"(smem_addr), "l"(gmem_addr), "n"(128) : "memory");
    }
}


__global__ __launch_bounds__(128, 1)
void mha_kernel(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* O, float* LSE, uint32_t S, uint32_t H) 
{
    extern __shared__ __align__(1024) uint8_t smem_pool[];
    
    // Requested size setup dynamically via Host (expecting 228KB)
    __nv_bfloat16* q_smem = (__nv_bfloat16*)smem_pool;                
    __nv_bfloat16* k_smem = q_smem + 16384;                           
    __nv_bfloat16* v_smem = k_smem + 16384;                            
    __nv_bfloat16* p_smem = v_smem + 16384;                            
    float* s_fp32_smem = (float*)(p_smem + 16384);                    
    float* o_fp32_smem = s_fp32_smem;                                 

    uint32_t total_q_blks = (S + 127) / 128;
    uint32_t q_blk_idx = blockIdx.x % total_q_blks;
    uint32_t total_head_idx = blockIdx.x / total_q_blks;
    uint32_t b_idx = total_head_idx / H;
    uint32_t h_idx = total_head_idx % H;
    uint32_t head_offset = b_idx * H * S + h_idx * S;

    uint64_t q_offset = (head_offset + q_blk_idx * 128) * 128 * sizeof(__nv_bfloat16);
    copy_tile_128b(smem_pool, (const uint8_t*)Q, q_offset, threadIdx.x);
    cp_async_commit_group();
    cp_async_wait_group<0>();
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> o_frag[2][8];
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 8; ++j) {
            wmma::fill_fragment(o_frag[i][j], 0.0f);
        }
    }

    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    float scale_factor = 1.0f / sqrtf(128.0f);
    
    int warp_id = threadIdx.x / 32;
    int m_base_warp = warp_id * 2 * 16;

    #pragma unroll 1
    for (uint32_t k_blk_idx = 0; k_blk_idx <= q_blk_idx; ++k_blk_idx) {
        uint64_t k_offset = (head_offset + k_blk_idx * 128) * 128 * sizeof(__nv_bfloat16);
        copy_tile_128b(smem_pool + 32768, (const uint8_t*)K, k_offset, threadIdx.x);
        cp_async_commit_group();
        cp_async_wait_group<0>();
        __syncthreads();

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> s_frag[2][8];
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 8; ++j) {
                wmma::fill_fragment(s_frag[i][j], 0.0f);
            }
        }

        for (int k = 0; k < 8; ++k) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16> q_frag[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16> k_frag[8];
            
            for (int i = 0; i < 2; ++i) {
                int m_base = m_base_warp + i * 16;
                wmma::load_matrix_sync(q_frag[i], q_smem + m_base * 128 + k * 16, 128);
            }
            for (int j = 0; j < 8; ++j) {
                int n_base = j * 16;
                wmma::load_matrix_sync(k_frag[j], k_smem + n_base * 128 + k * 16, 128);
            }
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 8; ++j) {
                    wmma::ma_sync(s_frag[i][j], q_frag[i], k_frag[j], s_frag[i][j]);
                }
            }
        }

        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 8; ++j) {
                int m_base = m_base_warp + i * 16;
                int n_base = j * 16;
                wmma::store_matrix_sync(s_fp32_smem + m_base * 128 + n_base, s_frag[i][j], 128, wmma::mem_row_major);
            }
        }
        __syncthreads();

        uint32_t global_q_idx = q_blk_idx * 128 + threadIdx.x;
        float m_curr = -INFINITY;
        float* my_s_row = s_fp32_smem + threadIdx.x * 128;
        
        for (int c = 0; c < 128; c += 4) {
            float4 vals = *(float4*)&my_s_row[c];
            uint32_t global_k_idx = k_blk_idx * 128 + c;
            if (global_q_idx >= S || global_k_idx >= S || global_k_idx > global_q_idx) {
                vals.x = -INFINITY; vals.y = -INFINITY; vals.z = -INFINITY; vals.w = -INFINITY;
            } else {
                vals.x *= scale_factor; vals.y *= scale_factor; vals.z *= scale_factor; vals.w *= scale_factor;
            }
            m_curr = fmaxf(m_curr, fmaxf(fmaxf(vals.x, vals.y), fmaxf(vals.z, vals.w)));
            *(float4*)&my_s_row[c] = vals;
        }

        float m_new = fmaxf(m_prev, m_curr);
        float l_curr = 0.0f;
        
        for (int c = 0; c < 128; c += 4) {
            float4 vals = *(float4*)&my_s_row[c];
            vals.x = fast_exp2f_fn((vals.x - m_new) * 1.4426950f);
            vals.y = fast_exp2f_fn((vals.y - m_new) * 1.4426950f);
            vals.z = fast_exp2f_fn((vals.z - m_new) * 1.4426950f);
            vals.w = fast_exp2f_fn((vals.w - m_new) * 1.4426950f);
            l_curr += vals.x + vals.y + vals.z + vals.w;
            
            __nv_bfloat16* my_p_row = p_smem + threadIdx.x * 128;
            my_p_row[c] = __float2bfloat16(vals.x);
            my_p_row[c+1] = __float2bfloat16(vals.y);
            my_p_row[c+2] = __float2bfloat16(vals.z);
            my_p_row[c+3] = __float2bfloat16(vals.w);
        }

        float l_new = l_prev * fast_exp2f_fn((m_prev - m_new) * 1.4426950f) + l_curr;
        
        if (l_prev > 0.0f) {
            float scale = fast_exp2f_fn((m_prev - m_new) * 1.4426950f);
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 8; ++j) {
                    int m_base = m_base_warp + i * 16;
                    int n_base = j * 16;
                    wmma::store_matrix_sync(o_fp32_smem + m_base * 128 + n_base, o_frag[i][j], 128, wmma::mem_row_major);
                }
            }
            __syncthreads();

            float* my_o_row = o_fp32_smem + threadIdx.x * 128;
            for (int c = 0; c < 128; c += 4) {
                float4 vals = *(float4*)&my_o_row[c];
                vals.x *= scale; vals.y *= scale; vals.z *= scale; vals.w *= scale;
                *(float4*)&my_o_row[c] = vals;
            }
            __syncthreads();

            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 8; ++j) {
                    int m_base = m_base_warp + i * 16;
                    int n_base = j * 16;
                    wmma::load_matrix_sync(o_frag[i][j], o_fp32_smem + m_base * 128 + n_base, 128);
                }
            }
        }

        m_prev = m_new;
        l_prev = l_new;
        
        uint64_t v_offset = (head_offset + k_blk_idx * 128) * 128 * sizeof(__nv_bfloat16);
        copy_tile_128b(smem_pool + 65536, (const uint8_t*)V, v_offset, threadIdx.x);
        cp_async_commit_group();
        cp_async_wait_group<0>();
        __syncthreads();

        for (int k = 0; k < 8; ++k) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16> p_frag[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16> v_frag[8];
            
            for (int i = 0; i < 2; ++i) {
                int m_base = m_base_warp + i * 16;
                wmma::load_matrix_sync(p_frag[i], p_smem + m_base * 128 + k * 16, 128);
            }
            for (int j = 0; j < 8; ++j) {
                int n_base = j * 16;
                wmma::load_matrix_sync(v_frag[j], v_smem + k * 16 * 128 + n_base, 128);
            }
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 8; ++j) {
                    wmma::ma_sync(o_frag[i][j], p_frag[i], v_frag[j], o_frag[i][j]);
                }
            }
        }

        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 8; ++j) {
                int m_base = m_base_warp + i * 16;
                int n_base = j * 16;
                wmma::store_matrix_sync(o_fp32_smem + m_base * 128 + n_base, o_frag[i][j], 128, wmma::mem_row_major);
            }
        }
        __syncthreads();
    }

    __syncthreads();
    
    uint32_t global_row = q_blk_idx * 128 + threadIdx.x;
    float* my_final_o_row = o_fp32_smem + threadIdx.x * 128;
    for (int c = 0; c < 128; c += 4) {
        float4 vals = *(float4*)&my_final_o_row[c];
        float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
        vals.x *= inv_l; vals.y *= inv_l; vals.z *= inv_l; vals.w *= inv_l;
        
        if (global_row < S) {
            uint32_t base_idx = (head_offset + global_row) * 128 + c;
            O[base_idx] = __float2bfloat16(vals.x);
            O[base_idx + 1] = __float2bfloat16(vals.y);
            O[base_idx + 2] = __float2bfloat16(vals.z);
            O[base_idx + 3] = __float2bfloat16(vals.w);
        }
    }

    if (threadIdx.x < 128 && global_row < S) {
        LSE[head_offset + global_row] = m_prev + logf(l_prev);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);

    int64_t total_q_blks = (S + 127) / 128;
    dim3 grid(total_q_blks * B * H, 1, 1);
    dim3 block(128, 1, 1);

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    CUDA_CHECK(cudaFuncSetAttribute(mha_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 228 * 1024));
    mha_kernel<<<grid, block, 228 * 1024, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()), 
        static_cast<const __nv_bfloat16*>(K.data_ptr()), 
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<__nv_bfloat16*>(O.data_ptr()), 
        static_cast<float*>(LSE.data_ptr()), 
        S, H);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha::run);

}  // namespace tvm_ffi_mha