#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <math.h>
#include <mma.h>
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

using namespace nvcuda;

namespace flash_attn_causal {

union TempSmem {
    struct {
        union {
            __nv_bfloat16 K_smem[64][128]; 
            float P_f32_smem[64][64];      
        } k_p;
        __nv_bfloat16 V_smem[64][128];     
    } kv;
    float O_update_smem[64][128];          
};

extern __shared__ __align__(16) char dynamic_smem[];

__global__ void flash_attn_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S) 
{
    int block_q = blockIdx.x;
    int b = blockIdx.y / H;
    int h = blockIdx.y % H;

    __nv_bfloat16 (*Q_smem)[128] = reinterpret_cast<__nv_bfloat16(*)[128]>(dynamic_smem);
    float (*O_smem)[128] = reinterpret_cast<float(*)[128]>(dynamic_smem + 16384);
    __nv_bfloat16 (*P_bf16_smem)[64] = reinterpret_cast<__nv_bfloat16(*)[64]>(dynamic_smem + 49152);
    TempSmem* temp_smem = reinterpret_cast<TempSmem*>(dynamic_smem + 57344);
    float* m_smem = reinterpret_cast<float*>(dynamic_smem + 90112);
    float* l_smem = reinterpret_cast<float*>(dynamic_smem + 90368);
    float* scale_smem = reinterpret_cast<float*>(dynamic_smem + 90624);

    int warp_id = threadIdx.x / 32;

    // Initialize O_smem, m_smem, l_smem
    for (int idx = threadIdx.x; idx < 64 * 32; idx += blockDim.x) {
        reinterpret_cast<float4*>(O_smem[idx / 32])[idx % 32] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    for (int idx = threadIdx.x; idx < 64; idx += blockDim.x) {
        m_smem[idx] = -INFINITY;
        l_smem[idx] = 0.0f;
    }
    __syncthreads();

    // Load Q to SMEM
    for (int idx = threadIdx.x; idx < 64 * 16; idx += blockDim.x) {
        int r = idx / 16;
        int c_f4 = idx % 16;
        int global_r = block_q * 64 + r;
        if (global_r < S) {
            int64_t g_idx = (int64_t(b) * H * S + int64_t(h) * S + global_r) * 16 + c_f4;
            reinterpret_cast<float4*>(Q_smem[r])[c_f4] = reinterpret_cast<const float4*>(Q)[g_idx];
        } else {
            reinterpret_cast<float4*>(Q_smem[r])[c_f4] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }
    __syncthreads();

    int max_kv_block = block_q; // Causal mask constraint

    for (int kv_idx = 0; kv_idx <= max_kv_block; ++kv_idx) {
        // 1. Load K to temp_smem
        for (int idx = threadIdx.x; idx < 64 * 16; idx += blockDim.x) { 
            int r = idx / 16;
            int c_f4 = idx % 16;
            int global_r = kv_idx * 64 + r;
            if (global_r < S) {
                int64_t g_idx = (int64_t(b) * H * S + int64_t(h) * S + global_r) * 16 + c_f4;
                reinterpret_cast<float4*>(temp_smem->kv.k_p.K_smem[r])[c_f4] = reinterpret_cast<const float4*>(K)[g_idx];
            } else {
                reinterpret_cast<float4*>(temp_smem->kv.k_p.K_smem[r])[c_f4] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        __syncthreads();

        // 2. Compute S = Q @ K^T
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> s_frag[4];
        for(int i = 0; i < 4; ++i) wmma::fill_fragment(s_frag[i], 0.0f);

        for (int k = 0; k < 128; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> q_frag;
            wmma::load_matrix_sync(q_frag, &Q_smem[warp_id * 16][k], 128);
            for (int i = 0; i < 4; ++i) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> k_frag;
                wmma::load_matrix_sync(k_frag, &temp_smem->kv.k_p.K_smem[i * 16][k], 128);
                wmma::mma_sync(s_frag[i], q_frag, k_frag, s_frag[i]);
            }
        }
        __syncthreads();

        // 3. Store S to P_f32_smem
        for (int i = 0; i < 4; ++i) {
            wmma::store_matrix_sync(&temp_smem->kv.k_p.P_f32_smem[warp_id * 16][i * 16], s_frag[i], 64, wmma::mem_row_major);
        }
        __syncthreads();

        // 4. Softmax per row
        if (threadIdx.x < 64) {
            int r = threadIdx.x;
            int global_q = block_q * 64 + r;
            
            if (global_q >= S) {
                for (int c = 0; c < 64; ++c) {
                    temp_smem->kv.k_p.P_f32_smem[r][c] = 0.0f;
                }
                scale_smem[r] = 1.0f;
            } else {
                float m_prev = m_smem[r];
                float m_block = -INFINITY;
                
                for (int c = 0; c < 64; ++c) {
                    int global_k = kv_idx * 64 + c;
                    if (global_k <= global_q && global_k < S) {
                        float val = temp_smem->kv.k_p.P_f32_smem[r][c] * 0.08838834764f;
                        temp_smem->kv.k_p.P_f32_smem[r][c] = val; 
                        m_block = fmaxf(m_block, val);
                    } else {
                        temp_smem->kv.k_p.P_f32_smem[r][c] = -INFINITY;
                    }
                }

                float m_new = fmaxf(m_prev, m_block);
                m_smem[r] = m_new;

                float l_prev = l_smem[r];
                float scale = expf(m_prev - m_new);
                float l_block = 0.0f;
                
                for (int c = 0; c < 64; ++c) {
                    float val = temp_smem->kv.k_p.P_f32_smem[r][c];
                    float p = expf(val - m_new);
                    if (val == -INFINITY) p = 0.0f; 
                    temp_smem->kv.k_p.P_f32_smem[r][c] = p;
                    l_block += p;
                }

                l_smem[r] = l_prev * scale + l_block;
                scale_smem[r] = scale;
            }
        }
        __syncthreads();

        // Rescale O_smem
        for (int idx = threadIdx.x; idx < 64 * 128; idx += blockDim.x) {
            int r = idx / 128;
            int c = idx % 128;
            int global_q = block_q * 64 + r;
            if (global_q < S) {
                O_smem[r][c] *= scale_smem[r];
            }
        }
        __syncthreads();

        // 5. Convert P_f32 to P_bf16
        for (int idx = threadIdx.x; idx < 64 * 64; idx += blockDim.x) {
            int r = idx / 64;
            int c = idx % 64;
            P_bf16_smem[r][c] = __float2bfloat16(temp_smem->kv.k_p.P_f32_smem[r][c]);
        }
        
        // 6. Load V to V_smem
        for (int idx = threadIdx.x; idx < 64 * 16; idx += blockDim.x) {
            int r = idx / 16;
            int c_f4 = idx % 16;
            int global_r = kv_idx * 64 + r;
            if (global_r < S) {
                int64_t g_idx = (int64_t(b) * H * S + int64_t(h) * S + global_r) * 16 + c_f4;
                reinterpret_cast<float4*>(temp_smem->kv.V_smem[r])[c_f4] = reinterpret_cast<const float4*>(V)[g_idx];
            } else {
                reinterpret_cast<float4*>(temp_smem->kv.V_smem[r])[c_f4] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        __syncthreads();

        // 7. Compute P @ V
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> o_frag[8];
        for(int j = 0; j < 8; ++j) wmma::fill_fragment(o_frag[j], 0.0f);

        for (int k = 0; k < 64; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> p_frag;
            wmma::load_matrix_sync(p_frag, &P_bf16_smem[warp_id * 16][k], 64);
            for (int j = 0; j < 8; ++j) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> v_frag; 
                wmma::load_matrix_sync(v_frag, &temp_smem->kv.V_smem[k][j * 16], 128);
                wmma::mma_sync(o_frag[j], p_frag, v_frag, o_frag[j]);
            }
        }
        __syncthreads();

        // 8. Store o_frag to O_update_smem
        for (int j = 0; j < 8; ++j) {
            wmma::store_matrix_sync(&temp_smem->O_update_smem[warp_id * 16][j * 16], o_frag[j], 128, wmma::mem_row_major);
        }
        __syncthreads();

        // 9. Add O_update_smem to O_smem
        for (int idx = threadIdx.x; idx < 64 * 128; idx += blockDim.x) {
            int r = idx / 128;
            int c = idx % 128;
            O_smem[r][c] += temp_smem->O_update_smem[r][c];
        }
        __syncthreads();
    }

    // Write Output and LSE
    for (int idx = threadIdx.x; idx < 64 * 128; idx += blockDim.x) {
        int r = idx / 128;
        int c = idx % 128;
        int global_r = block_q * 64 + r;
        if (global_r < S) {
            float out_val = O_smem[r][c] / l_smem[r];
            int64_t out_idx = (int64_t(b) * H * S + int64_t(h) * S + global_r) * 128 + c;
            O[out_idx] = __float2bfloat16(out_val);
        }
    }
    
    for (int r = threadIdx.x; r < 64; r += blockDim.x) {
        int global_r = block_q * 64 + r;
        if (global_r < S) {
            int64_t lse_idx = int64_t(b) * H * S + int64_t(h) * S + global_r;
            LSE[lse_idx] = m_smem[r] + logf(l_smem[r]);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) 
{
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    // int64_t D = Q.size(3); // Known to be 128

    if (S == 0) return;

    const __nv_bfloat16* q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* k_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* v_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* o_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* lse_ptr = static_cast<float*>(LSE.data_ptr());

    int64_t num_q_blocks = (S + 63) / 64;
    dim3 grid(num_q_blocks, B * H);
    dim3 block(128);

    int smem_size = 90880;

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute(flash_attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    flash_attn_kernel<<<grid, block, smem_size, stream>>>(q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr, B, H, S);
    CUDA_CHECK(cudaGetLastError());
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace flash_attn_causal