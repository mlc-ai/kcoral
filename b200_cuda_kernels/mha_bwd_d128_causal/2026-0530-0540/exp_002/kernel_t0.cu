#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>
#include <mma.h>

using namespace nvcuda;

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_mha_bwd {

__global__ void precompute_D_kernel(const __nv_bfloat16* __restrict__ dO, 
                                    const __nv_bfloat16* __restrict__ O, 
                                    float* __restrict__ D, 
                                    int B, int H, int S) 
{
    int seq_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;
    if (seq_idx < S) {
        float sum = 0.0f;
        int offset = (b * H * S + h * S + seq_idx) * 128;
        for (int d = 0; d < 128; d += 2) {
            __nv_bfloat162 do_val = *reinterpret_cast<const __nv_bfloat162*>(&dO[offset + d]);
            __nv_bfloat162 o_val = *reinterpret_cast<const __nv_bfloat162*>(&O[offset + d]);
            sum += __bfloat162float(do_val.x) * __bfloat162float(o_val.x);
            sum += __bfloat162float(do_val.y) * __bfloat162float(o_val.y);
        }
        D[b * H * S + h * S + seq_idx] = sum;
    }
}

__device__ __forceinline__ void load_tile_bf16_d128(__nv_bfloat16* dst, const __nv_bfloat16* src, int valid_rows) {
    int tid = threadIdx.x;
    for (int i = tid; i < 64 * 128 / 8; i += 128) {
        int row = i / 16;
        int col = (i % 16) * 8;
        if (row < valid_rows) {
            float4 val = *reinterpret_cast<const float4*>(&src[row * 128 + col]);
            *reinterpret_cast<float4*>(&dst[row * 128 + col]) = val;
        } else {
            *reinterpret_cast<float4*>(&dst[row * 128 + col]) = {0.0f, 0.0f, 0.0f, 0.0f};
        }
    }
}

__device__ __forceinline__ void load_vec_float(float* dst, const float* src, int valid_rows) {
    int tid = threadIdx.x;
    if (tid < 64) {
        if (tid < valid_rows) {
            dst[tid] = src[tid];
        } else {
            dst[tid] = 0.0f;
        }
    }
}

__global__ void mha_bwd_d128_causal_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const float* __restrict__ L,
    const float* __restrict__ D,
    const __nv_bfloat16* __restrict__ dO,
    __nv_bfloat16* __restrict__ dQ,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    int B, int H, int S, float scale) 
{
    int b = blockIdx.z;
    int h = blockIdx.y;
    int i_chunk = blockIdx.x; 
    
    int i_base = i_chunk * 64;
    int valid_rows_q = min(64, S - i_base);
    if (valid_rows_q <= 0) return;
    
    int head_offset = (b * H + h) * S * 128;
    int vec_offset = (b * H + h) * S;
    
    const __nv_bfloat16* q_ptr = Q + head_offset + i_base * 128;
    const __nv_bfloat16* do_ptr = dO + head_offset + i_base * 128;
    __nv_bfloat16* dq_ptr = dQ + head_offset + i_base * 128;
    const float* l_ptr = L + vec_offset + i_base;
    const float* d_ptr = D + vec_offset + i_base;
    
    // Dynamic shared memory layout
    extern __shared__ char shared_mem[];
    __nv_bfloat16* s_Q = (__nv_bfloat16*)shared_mem;                             // 16 KB
    __nv_bfloat16* s_dO = s_Q + 64 * 128;                                        // 16 KB
    __nv_bfloat16* s_K = s_dO + 64 * 128;                                        // 16 KB
    __nv_bfloat16* s_V = s_K + 64 * 128;                                         // 16 KB
    float* s_cast = (float*)(s_V + 64 * 128);                                    // 32 KB (Alias for S and dP)
    float* s_S = s_cast;                                                         // 16 KB
    float* s_dP = s_cast + 64 * 64;                                              // 16 KB
    __nv_bfloat16* s_P = (__nv_bfloat16*)(s_cast + 64 * 128);                    // 8 KB
    __nv_bfloat16* s_dS = s_P + 64 * 64;                                         // 8 KB
    float* s_L = (float*)(s_dS + 64 * 64);                                       // 256 B
    float* s_D = s_L + 64;                                                       // 256 B

    load_tile_bf16_d128(s_Q, q_ptr, valid_rows_q);
    load_tile_bf16_d128(s_dO, do_ptr, valid_rows_q);
    load_vec_float(s_L, l_ptr, valid_rows_q);
    load_vec_float(s_D, d_ptr, valid_rows_q);
    __syncthreads();
    
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dq_acc[8];
    for (int i = 0; i < 8; i++) {
        wmma::fill_fragment(dq_acc[i], 0.0f);
    }
    
    int warp_id = threadIdx.x / 32;
    int tid = threadIdx.x;
    
    for (int j_chunk = 0; j_chunk <= i_chunk; j_chunk++) {
        int j_base = j_chunk * 64;
        int valid_rows_kv = min(64, S - j_base);
        
        const __nv_bfloat16* k_ptr = K + head_offset + j_base * 128;
        const __nv_bfloat16* v_ptr = V + head_offset + j_base * 128;
        
        load_tile_bf16_d128(s_K, k_ptr, valid_rows_kv);
        load_tile_bf16_d128(s_V, v_ptr, valid_rows_kv);
        __syncthreads();
        
        int row_s = (warp_id / 2) * 32;
        int col_s = (warp_id % 2) * 32;
        
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> s_acc[4];
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> dp_acc[4];
        for (int i = 0; i < 4; i++) {
            wmma::fill_fragment(s_acc[i], 0.0f);
            wmma::fill_fragment(dp_acc[i], 0.0f);
        }
        
        for (int k = 0; k < 128; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> q_frag[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> k_frag[2]; 
            
            wmma::load_matrix_sync(q_frag[0], &s_Q[row_s * 128 + k], 128);
            wmma::load_matrix_sync(q_frag[1], &s_Q[(row_s + 16) * 128 + k], 128);
            
            wmma::load_matrix_sync(k_frag[0], &s_K[col_s * 128 + k], 128);
            wmma::load_matrix_sync(k_frag[1], &s_K[(col_s + 16) * 128 + k], 128);
            
            wmma::mma_sync(s_acc[0], q_frag[0], k_frag[0], s_acc[0]);
            wmma::mma_sync(s_acc[1], q_frag[0], k_frag[1], s_acc[1]);
            wmma::mma_sync(s_acc[2], q_frag[1], k_frag[0], s_acc[2]);
            wmma::mma_sync(s_acc[3], q_frag[1], k_frag[1], s_acc[3]);
            
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> do_frag[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> v_frag[2];
            
            wmma::load_matrix_sync(do_frag[0], &s_dO[row_s * 128 + k], 128);
            wmma::load_matrix_sync(do_frag[1], &s_dO[(row_s + 16) * 128 + k], 128);
            
            wmma::load_matrix_sync(v_frag[0], &s_V[col_s * 128 + k], 128);
            wmma::load_matrix_sync(v_frag[1], &s_V[(col_s + 16) * 128 + k], 128);
            
            wmma::mma_sync(dp_acc[0], do_frag[0], v_frag[0], dp_acc[0]);
            wmma::mma_sync(dp_acc[1], do_frag[0], v_frag[1], dp_acc[1]);
            wmma::mma_sync(dp_acc[2], do_frag[1], v_frag[0], dp_acc[2]);
            wmma::mma_sync(dp_acc[3], do_frag[1], v_frag[1], dp_acc[3]);
        }
        
        for (int i = 0; i < 4; i++) {
            for (int t = 0; t < s_acc[i].num_elements; t++) {
                s_acc[i].x[t] *= scale;
            }
        }
        
        wmma::store_matrix_sync(&s_S[row_s * 64 + col_s], s_acc[0], 64, wmma::mem_row_major);
        wmma::store_matrix_sync(&s_S[row_s * 64 + col_s + 16], s_acc[1], 64, wmma::mem_row_major);
        wmma::store_matrix_sync(&s_S[(row_s + 16) * 64 + col_s], s_acc[2], 64, wmma::mem_row_major);
        wmma::store_matrix_sync(&s_S[(row_s + 16) * 64 + col_s + 16], s_acc[3], 64, wmma::mem_row_major);
        
        wmma::store_matrix_sync(&s_dP[row_s * 64 + col_s], dp_acc[0], 64, wmma::mem_row_major);
        wmma::store_matrix_sync(&s_dP[row_s * 64 + col_s + 16], dp_acc[1], 64, wmma::mem_row_major);
        wmma::store_matrix_sync(&s_dP[(row_s + 16) * 64 + col_s], dp_acc[2], 64, wmma::mem_row_major);
        wmma::store_matrix_sync(&s_dP[(row_s + 16) * 64 + col_s + 16], dp_acc[3], 64, wmma::mem_row_major);
        
        __syncthreads();
        
        for (int i = tid; i < 64 * 64; i += 128) {
            int r = i / 64;
            int c = i % 64;
            float p_val = 0.0f;
            float ds_val = 0.0f;
            if (r < valid_rows_q && c < valid_rows_kv) {
                int q_idx = i_base + r;
                int k_idx = j_base + c;
                if (k_idx <= q_idx) {
                    float s_val = s_S[r * 64 + c];
                    p_val = expf(s_val - s_L[r]);
                    float dp_val = s_dP[r * 64 + c];
                    ds_val = p_val * (dp_val - s_D[r]) * scale;
                }
            }
            s_P[r * 64 + c] = __float2bfloat16(p_val);
            s_dS[r * 64 + c] = __float2bfloat16(ds_val);
        }
        __syncthreads();
        
        int row_dq = warp_id * 16;
        for (int k = 0; k < 64; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> ds_frag;
            wmma::load_matrix_sync(ds_frag, &s_dS[row_dq * 64 + k], 64);
            
            for (int c_step = 0; c_step < 8; c_step++) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> k_frag;
                wmma::load_matrix_sync(k_frag, &s_K[k * 128 + c_step * 16], 128);
                wmma::mma_sync(dq_acc[c_step], ds_frag, k_frag, dq_acc[c_step]);
            }
        }
        
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> dk_acc[8];
        for (int i = 0; i < 8; i++) wmma::fill_fragment(dk_acc[i], 0.0f);
        
        int row_dk = warp_id * 16; 
        for (int k = 0; k < 64; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> ds_t_frag;
            wmma::load_matrix_sync(ds_t_frag, &s_dS[k * 64 + row_dk], 64);
            
            for (int c_step = 0; c_step < 8; c_step++) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> q_frag;
                wmma::load_matrix_sync(q_frag, &s_Q[k * 128 + c_step * 16], 128);
                wmma::mma_sync(dk_acc[c_step], ds_t_frag, q_frag, dk_acc[c_step]);
            }
        }
        
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> dv_acc[8];
        for (int i = 0; i < 8; i++) wmma::fill_fragment(dv_acc[i], 0.0f);
        
        for (int k = 0; k < 64; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> p_t_frag;
            wmma::load_matrix_sync(p_t_frag, &s_P[k * 64 + row_dk], 64);
            
            for (int c_step = 0; c_step < 8; c_step++) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> do_frag;
                wmma::load_matrix_sync(do_frag, &s_dO[k * 128 + c_step * 16], 128);
                wmma::mma_sync(dv_acc[c_step], p_t_frag, do_frag, dv_acc[c_step]);
            }
        }
        
        __syncthreads(); 
        
        for (int c_step = 0; c_step < 8; c_step++) {
            wmma::store_matrix_sync(&s_cast[row_dk * 128 + c_step * 16], dk_acc[c_step], 128, wmma::mem_row_major);
        }
        __syncthreads();
        
        __nv_bfloat16* dk_ptr = dK + head_offset + j_base * 128;
        for (int i = tid; i < 64 * 128 / 2; i += 128) {
            int row = i / 64;
            int col = (i % 64) * 2;
            if (row < valid_rows_kv) {
                float f0 = s_cast[row * 128 + col];
                float f1 = s_cast[row * 128 + col + 1];
                __nv_bfloat162 val2;
                val2.x = __float2bfloat16(f0);
                val2.y = __float2bfloat16(f1);
                atomicAdd(reinterpret_cast<__nv_bfloat162*>(&dk_ptr[row * 128 + col]), val2);
            }
        }
        __syncthreads();
        
        for (int c_step = 0; c_step < 8; c_step++) {
            wmma::store_matrix_sync(&s_cast[row_dk * 128 + c_step * 16], dv_acc[c_step], 128, wmma::mem_row_major);
        }
        __syncthreads();
        
        __nv_bfloat16* dv_ptr = dV + head_offset + j_base * 128;
        for (int i = tid; i < 64 * 128 / 2; i += 128) {
            int row = i / 64;
            int col = (i % 64) * 2;
            if (row < valid_rows_kv) {
                float f0 = s_cast[row * 128 + col];
                float f1 = s_cast[row * 128 + col + 1];
                __nv_bfloat162 val2;
                val2.x = __float2bfloat16(f0);
                val2.y = __float2bfloat16(f1);
                atomicAdd(reinterpret_cast<__nv_bfloat162*>(&dv_ptr[row * 128 + col]), val2);
            }
        }
        __syncthreads();
    }
    
    for (int c_step = 0; c_step < 8; c_step++) {
        wmma::store_matrix_sync(&s_cast[warp_id * 16 * 128 + c_step * 16], dq_acc[c_step], 128, wmma::mem_row_major);
    }
    __syncthreads();
    
    for (int i = tid; i < 64 * 128 / 2; i += 128) {
        int row = i / 64;
        int col = (i % 64) * 2;
        if (row < valid_rows_q) {
            float f0 = s_cast[row * 128 + col];
            float f1 = s_cast[row * 128 + col + 1];
            __nv_bfloat162 val2;
            val2.x = __float2bfloat16(f0);
            val2.y = __float2bfloat16(f1);
            *reinterpret_cast<__nv_bfloat162*>(&dq_ptr[row * 128 + col]) = val2;
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) 
{
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int B = static_cast<int>(Q.size(0));
    int H = static_cast<int>(Q.size(1));
    int S = static_cast<int>(Q.size(2));
    
    if (S == 0) return;
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    __nv_bfloat16* q_ptr = static_cast<__nv_bfloat16*>(Q.data_ptr());
    __nv_bfloat16* k_ptr = static_cast<__nv_bfloat16*>(K.data_ptr());
    __nv_bfloat16* v_ptr = static_cast<__nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* o_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    __nv_bfloat16* do_ptr = static_cast<__nv_bfloat16*>(dO.data_ptr());
    float* l_ptr = static_cast<float*>(L.data_ptr());
    __nv_bfloat16* dq_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dk_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dv_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());

    int64_t total_elements = static_cast<int64_t>(B) * H * S * 128;
    CUDA_CHECK(cudaMemsetAsync(dk_ptr, 0, total_elements * sizeof(__nv_bfloat16), stream));
    CUDA_CHECK(cudaMemsetAsync(dv_ptr, 0, total_elements * sizeof(__nv_bfloat16), stream));

    float* d_ptr;
    CUDA_CHECK(cudaMallocAsync(&d_ptr, static_cast<int64_t>(B) * H * S * sizeof(float), stream));
    
    dim3 d_block(128);
    dim3 d_grid((S + 127) / 128, H, B);
    precompute_D_kernel<<<d_grid, d_block, 0, stream>>>(do_ptr, o_ptr, d_ptr, B, H, S);
    
    int smem_size = 16384 * 4 + 32768 + 8192 * 2 + 256 * 2; // ~112.5 KB
    CUDA_CHECK(cudaFuncSetAttribute(mha_bwd_d128_causal_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    dim3 bwd_block(128);
    dim3 bwd_grid((S + 63) / 64, H, B);
    float scale = 1.0f / sqrtf(128.0f);
    
    mha_bwd_d128_causal_kernel<<<bwd_grid, bwd_block, smem_size, stream>>>(
        q_ptr, k_ptr, v_ptr, l_ptr, d_ptr, do_ptr, dq_ptr, dk_ptr, dv_ptr, B, H, S, scale);
        
    CUDA_CHECK(cudaFreeAsync(d_ptr, stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_bwd::run);

} // namespace tvm_ffi_mha_bwd