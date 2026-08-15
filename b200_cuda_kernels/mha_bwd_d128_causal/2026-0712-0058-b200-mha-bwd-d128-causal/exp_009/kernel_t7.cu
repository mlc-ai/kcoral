#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdio.h>
#include <math.h>
#include <mma.h>
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

namespace tvm_ffi_flash_attention_bwd {

__device__ void load_tile(__nv_bfloat16* smem, const __nv_bfloat16* gmem, uint32_t row_start, uint32_t col_start, uint32_t S, uint32_t d, uint32_t batch_offset) {
    for (uint32_t idx = threadIdx.x; idx < 64 * 64; idx += blockDim.x) {
        uint32_t row = idx / 64;
        uint32_t col = idx % 64;
        if (row_start + row < S && col_start + col < d) {
            smem[idx] = gmem[batch_offset + (row_start + row) * d + (col_start + col)];
        } else {
            smem[idx] = __float2bfloat16(0.0f);
        }
    }
}

__device__ void load_lse(float* smem_LSE, const float* gmem_L, uint32_t row_start, uint32_t S, uint32_t batch_offset) {
    for (uint32_t idx = threadIdx.x; idx < 64; idx += blockDim.x) {
        if (row_start + idx < S) {
            smem_LSE[idx] = gmem_L[batch_offset + row_start + idx];
        } else {
            smem_LSE[idx] = 0.0f;
        }
    }
}

__device__ __forceinline__ __nv_bfloat16 read_smem(const __nv_bfloat16* smem, uint32_t row, uint32_t col) {
    uint32_t chunk = col / 8;
    uint32_t offset = col % 8;
    uint32_t swizzled_chunk = chunk ^ (row % 8);
    uint32_t swizzled_col = swizzled_chunk * 8 + offset;
    return smem[row * 64 + swizzled_col];
}

__device__ __forceinline__ void write_smem(__nv_bfloat16* smem, uint32_t row, uint32_t col, __nv_bfloat16 val) {
    uint32_t chunk = col / 8;
    uint32_t offset = col % 8;
    uint32_t swizzled_chunk = chunk ^ (row % 8);
    uint32_t swizzled_col = swizzled_chunk * 8 + offset;
    smem[row * 64 + swizzled_col] = val;
}

__global__ void __launch_bounds__(128) flash_attention_bwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    uint32_t S)
{
    uint32_t my_tile = blockIdx.x;
    uint32_t batch_head = blockIdx.y;
    float scale = 1.0f / sqrtf(128.0f);

    extern __shared__ char smem_raw[];
    __nv_bfloat16* smem_Q0 = (__nv_bfloat16*)smem_raw;
    __nv_bfloat16* smem_Q1 = smem_Q0 + 64*64;
    __nv_bfloat16* smem_K0 = smem_Q1 + 64*64;
    __nv_bfloat16* smem_K1 = smem_K0 + 64*64;
    __nv_bfloat16* smem_V0 = smem_K1 + 64*64;
    __nv_bfloat16* smem_V1 = smem_V0 + 64*64;
    __nv_bfloat16* smem_dO0 = smem_V1 + 64*64;
    __nv_bfloat16* smem_dO1 = smem_dO0 + 64*64;
    __nv_bfloat16* smem_O0 = smem_dO1 + 64*64;
    __nv_bfloat16* smem_O1 = smem_O0 + 64*64;
    __nv_bfloat16* smem_dS_T = smem_O1 + 64*64;
    __nv_bfloat16* smem_P_T = smem_dS_T + 64*64;
    float* smem_D = (float*)(smem_P_T + 64*64);
    float* smem_LSE = smem_D + 64;
    
    uint32_t tmem_base;
    if (threadIdx.x == 0) {
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
            :: "r"((uint32_t)__cvta_generic_to_shared(&tmem_base)), "r"(512));
    }
    __syncthreads();
    
    uint64_t b_off = (uint64_t)batch_head * S * 128;
    uint32_t query_start = my_tile * 64;
    uint32_t key_start = my_tile * 64;
    uint32_t last_q_tile = (S + 63) / 64 - 1;

    // ==========================================
    // PASS 1: Compute dQ
    // ==========================================
    float my_dQ_acc0[64] = {0}, my_dQ_acc1[64] = {0};

    load_tile(smem_Q0, Q, query_start, 0, S, 128, b_off);
    load_tile(smem_Q1, Q, query_start, 64, S, 128, b_off);
    load_tile(smem_dO0, dO, query_start, 0, S, 128, b_off);
    load_tile(smem_dO1, dO, query_start, 64, S, 128, b_off);
    load_tile(smem_O0, O, query_start, 0, S, 128, b_off);
    load_tile(smem_O1, O, query_start, 64, S, 128, b_off);
    load_lse(smem_LSE, L, query_start, S, (uint64_t)batch_head * S);
    __syncthreads();

    if (threadIdx.x < 64) {
        float d_val = 0;
        for(uint32_t d = 0; d < 64; ++d) {
            d_val += __bfloat162float(read_smem(smem_O0, threadIdx.x, d)) * __bfloat162float(read_smem(smem_dO0, threadIdx.x, d));
            d_val += __bfloat162float(read_smem(smem_O1, threadIdx.x, d)) * __bfloat162float(read_smem(smem_dO1, threadIdx.x, d));
        }
        smem_D[threadIdx.x] = d_val;
    }
    __syncthreads();
    
    for (uint32_t k_tile = 0; k_tile <= my_tile; ++k_tile) {
        load_tile(smem_K0, K, k_tile * 64, 0, S, 128, b_off);
        load_tile(smem_K1, K, k_tile * 64, 64, S, 128, b_off);
        load_tile(smem_V0, V, k_tile * 64, 0, S, 128, b_off);
        load_tile(smem_V1, V, k_tile * 64, 64, S, 128, b_off);
        __syncthreads();

        for (uint32_t row = threadIdx.x; row < 64; row += 128) {
            for (uint32_t col = 0; col < 64; col += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
                
                wmma::fill_fragment(acc, 0);
                for (uint32_t k = 0; k < 64; k += 16) {
                    wmma::load_matrix_sync(a, smem_Q0 + row * 64 + k, 64);
                    wmma::load_matrix_sync(b, smem_K0 + col * 64 + k, 64);
                    wmma::mma_sync(acc, a, b, acc);
                }
                for (uint32_t k = 0; k < 64; k += 16) {
                    wmma::load_matrix_sync(a, smem_Q1 + row * 64 + k, 64);
                    wmma::load_matrix_sync(b, smem_K1 + col * 64 + k, 64);
                    wmma::mma_sync(acc, a, b, acc);
                }
                
                float s[2];
                wmma::store_matrix_sync(s, acc, 16, wmma::mem_row_major);
                s[0] *= scale;
                s[1] *= scale;
                
                if (query_start + row >= key_start + col && query_start + row < S && key_start + col < S) {
                    float lse = (query_start + row < S) ? L[batch_head * S + query_start + row] : 0.0f;
                    s[0] = expf(s[0] - lse);
                    s[1] = expf(s[1] - lse);
                } else {
                    s[0] = 0;
                    s[1] = 0;
                }
                smem_P_T[col * 64 + row] = __float2bfloat16(s[0]);
                smem_P_T[col * 64 + row + 16] = __float2bfloat16(s[1]);
            }
        }
        __syncthreads();

        for (uint32_t row = threadIdx.x; row < 64; row += 128) {
            for (uint32_t col = 0; col < 64; col += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
                
                wmma::fill_fragment(acc, 0);
                // dP row-major = dO * V^T
                for (uint32_t k = 0; k < 64; k += 16) {
                    wmma::load_matrix_sync(a, smem_dO0 + row * 64 + k, 64);
                    wmma::load_matrix_sync(b, smem_V0 + k * 64 + col, 64);
                    wmma::mma_sync(acc, a, b, acc);
                }
                for (uint32_t k = 0; k < 64; k += 16) {
                    wmma::load_matrix_sync(a, smem_dO1 + row * 64 + k, 64);
                    wmma::load_matrix_sync(b, smem_V1 + k * 64 + col, 64);
                    wmma::mma_sync(acc, a, b, acc);
                }
                
                float dp[2];
                wmma::store_matrix_sync(dp, acc, 16, wmma::mem_row_major);
                
                float p0 = __bfloat162float(read_smem(smem_P_T, col, row));
                float ds0 = p0 * (dp[0] - smem_D[row]);
                write_smem(smem_dS_T, col, row, __float2bfloat16(ds0));
                
                float p1 = __bfloat162float(read_smem(smem_P_T, col + 16, row));
                float ds1 = p1 * (dp[1] - smem_D[row]);
                write_smem(smem_dS_T, col + 16, row, __float2bfloat16(ds1));
            }
        }
        __syncthreads();

        uint32_t row = threadIdx.x % 64;
        uint32_t head_half = threadIdx.x / 64;
        
        if (head_half == 0) {
            for (uint32_t k = 0; k < 64; k += 8) {
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_dQ0;
                wmma::fill_fragment(acc_dQ0, 0);
                
                for (uint32_t s_idx = 0; s_idx < 64; s_idx += 16) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b;
                    
                    wmma::load_matrix_sync(a, smem_dS_T + s_idx * 64 + k, 64);
                    wmma::load_matrix_sync(b, smem_K0 + k * 64 + s_idx, 64);
                    
                    wmma::mma_sync(acc_dQ0, a, b, acc_dQ0);
                }
                
                float dq[2];
                wmma::store_matrix_sync(dq, acc_dQ0, 16, wmma::mem_row_major);
                if (query_start + row < S && k < 64) {
                    dQ[b_off + (query_start + row) * 128 + k] = __float2bfloat16(dq[0]);
                }
                if (query_start + row < S && k + 16 < 128) {
                    dQ[b_off + (query_start + row) * 128 + k + 16] = __float2bfloat16(dq[1]);
                }
            }
        } else {
            for (uint32_t k = 0; k < 64; k += 8) {
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_dQ1;
                wmma::fill_fragment(acc_dQ1, 0);
                
                for (uint32_t s_idx = 0; s_idx < 64; s_idx += 16) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b;
                    
                    wmma::load_matrix_sync(a, smem_dS_T + s_idx * 64 + k, 64);
                    wmma::load_matrix_sync(b, smem_K1 + k * 64 + s_idx, 64);
                    
                    wmma::mma_sync(acc_dQ1, a, b, acc_dQ1);
                }
                
                float dq[2];
                wmma::store_matrix_sync(dq, acc_dQ1, 16, wmma::mem_row_major);
                if (query_start + row < S && 64 + k < 128) {
                    dQ[b_off + (query_start + row) * 128 + 64 + k] = __float2bfloat16(dq[0]);
                }
                if (query_start + row < S && 64 + k + 16 < 128) {
                    dQ[b_off + (query_start + row) * 128 + 64 + k + 16] = __float2bfloat16(dq[1]);
                }
            }
        }
        __syncthreads();
    }

    // ==========================================
    // PASS 2: Compute dK and dV
    // ==========================================
    
    float dK_acc0[64] = {0}, dK_acc1[64] = {0};
    float dV_acc0[64] = {0}, dV_acc1[64] = {0};

    load_tile(smem_K0, K, key_start, 0, S, 128, b_off);
    load_tile(smem_K1, K, key_start, 64, S, 128, b_off);
    load_tile(smem_V0, V, key_start, 0, S, 128, b_off);
    load_tile(smem_V1, V, key_start, 64, S, 128, b_off);
    __syncthreads();
    
    for (uint32_t q_tile = my_tile; q_tile <= last_q_tile; ++q_tile) {
        load_tile(smem_Q0, Q, q_tile * 64, 0, S, 128, b_off);
        load_tile(smem_Q1, Q, q_tile * 64, 64, S, 128, b_off);
        load_tile(smem_dO0, dO, q_tile * 64, 0, S, 128, b_off);
        load_tile(smem_dO1, dO, q_tile * 64, 64, S, 128, b_off);
        load_tile(smem_O0, O, q_tile * 64, 0, S, 128, b_off);
        load_tile(smem_O1, O, q_tile * 64, 64, S, 128, b_off);
        load_lse(smem_LSE, L, q_tile * 64, S, (uint64_t)batch_head * S);
        __syncthreads();
        
        if (threadIdx.x < 64) {
            float d_val = 0;
            for(uint32_t d = 0; d < 64; ++d) {
                d_val += __bfloat162float(read_smem(smem_O0, threadIdx.x, d)) * __bfloat162float(read_smem(smem_dO0, threadIdx.x, d));
                d_val += __bfloat162float(read_smem(smem_O1, threadIdx.x, d)) * __bfloat162float(read_smem(smem_dO1, threadIdx.x, d));
            }
            smem_D[threadIdx.x] = d_val;
        }
        __syncthreads();

        for (uint32_t row = threadIdx.x; row < 64; row += 128) {
            for (uint32_t col = 0; col < 64; col += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
                
                wmma::fill_fragment(acc, 0);
                for (uint32_t k = 0; k < 64; k += 16) {
                    wmma::load_matrix_sync(a, smem_Q0 + row * 64 + k, 64);
                    wmma::load_matrix_sync(b, smem_K0 + col * 64 + k, 64);
                    wmma::mma_sync(acc, a, b, acc);
                }
                for (uint32_t k = 0; k < 64; k += 16) {
                    wmma::load_matrix_sync(a, smem_Q1 + row * 64 + k, 64);
                    wmma::load_matrix_sync(b, smem_K1 + col * 64 + k, 64);
                    wmma::mma_sync(acc, a, b, acc);
                }
                
                float s[2];
                wmma::store_matrix_sync(s, acc, 16, wmma::mem_row_major);
                s[0] *= scale;
                s[1] *= scale;
                
                if (q_tile * 64 + row >= key_start + col && q_tile * 64 + row < S && key_start + col < S) {
                    float lse = (q_tile * 64 + row < S) ? L[batch_head * S + q_tile * 64 + row] : 0.0f;
                    s[0] = expf(s[0] - lse);
                    s[1] = expf(s[1] - lse);
                } else {
                    s[0] = 0;
                    s[1] = 0;
                }
                smem_P_T[col * 64 + row] = __float2bfloat16(s[0]);
                smem_P_T[col * 64 + row + 16] = __float2bfloat16(s[1]);
            }
        }
        __syncthreads();

        for (uint32_t row = threadIdx.x; row < 64; row += 128) {
            for (uint32_t col = 0; col < 64; col += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
                
                wmma::fill_fragment(acc, 0);
                for (uint32_t k = 0; k < 64; k += 16) {
                    wmma::load_matrix_sync(a, smem_dO0 + row * 64 + k, 64);
                    wmma::load_matrix_sync(b, smem_V0 + k * 64 + col, 64);
                    wmma::mma_sync(acc, a, b, acc);
                }
                for (uint32_t k = 0; k < 64; k += 16) {
                    wmma::load_matrix_sync(a, smem_dO1 + row * 64 + k, 64);
                    wmma::load_matrix_sync(b, smem_V1 + k * 64 + col, 64);
                    wmma::mma_sync(acc, a, b, acc);
                }
                
                float dp[2];
                wmma::store_matrix_sync(dp, acc, 16, wmma::mem_row_major);
                
                float p0 = __bfloat162float(read_smem(smem_P_T, col, row));
                float ds0 = p0 * (dp[0] - smem_D[row]);
                write_smem(smem_dS_T, col, row, __float2bfloat16(ds0));
                
                float p1 = __bfloat162float(read_smem(smem_P_T, col + 16, row));
                float ds1 = p1 * (dp[1] - smem_D[row]);
                write_smem(smem_dS_T, col + 16, row, __float2bfloat16(ds1));
            }
        }
        __syncthreads();

        uint32_t my_j = threadIdx.x % 64;
        uint32_t head_half = threadIdx.x / 64;
        
        for (uint32_t col = 0; col < 64; ++col) {
            float dk = 0, dv = 0;
            for (uint32_t i = 0; i < 64; ++i) {
                float ds_val = __bfloat162float(read_smem(smem_dS_T, my_j, i));
                float p_val = __bfloat162float(read_smem(smem_P_T, my_j, i));
                
                dk += ds_val * __bfloat162float(head_half == 0 ? read_smem(smem_Q0, i, col) : read_smem(smem_Q1, i, col));
                dv += p_val * __bfloat162float(head_half == 0 ? read_smem(smem_dO0, i, col) : read_smem(smem_dO1, i, col));
            }
            
            if (head_half == 0) {
                dK_acc0[col] += dk;
                dV_acc0[col] += dv;
            } else {
                dK_acc1[col] += dk;
                dV_acc1[col] += dv;
            }
        }
        
        __syncthreads();
    }
    
    uint32_t my_j_store = threadIdx.x % 64;
    uint32_t head_half_store = threadIdx.x / 64;
    
    if (key_start + my_j_store < S) {
        for (uint32_t col = 0; col < 64; ++col) {
            if (head_half_store == 0) {
                dK[b_off + (key_start + my_j_store) * 128 + col] = __float2bfloat16(dK_acc0[col]);
                dV[b_off + (key_start + my_j_store) * 128 + col] = __float2bfloat16(dV_acc0[col]);
            } else {
                dK[b_off + (key_start + my_j_store) * 128 + 64 + col] = __float2bfloat16(dK_acc1[col]);
                dV[b_off + (key_start + my_j_store) * 128 + 64 + col] = __float2bfloat16(dV_acc1[col]);
            }
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    uint32_t B = Q.size(0);
    uint32_t H = Q.size(1);
    uint32_t S = Q.size(2);
    
    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* O_ptr = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dO_ptr = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_ptr = static_cast<const float*>(L.data_ptr());
    
    __nv_bfloat16* dQ_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());
    
    uint32_t num_tiles = (S + 63) / 64;
    dim3 grid(num_tiles, B * H);
    dim3 block(128);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    cudaFuncSetAttribute(flash_attention_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 102400);
    
    flash_attention_bwd_kernel<<<grid, block, 102400, stream>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
        dQ_ptr, dK_ptr, dV_ptr, S);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

}  // namespace tvm_ffi_flash_attention_bwd