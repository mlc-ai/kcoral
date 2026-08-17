#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
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

__device__ __forceinline__ void atomicAddAvg(__nv_bfloat16* address, __nv_bfloat16 val) {
    int* address_i = (int*)(((uintptr_t)address >> 1) & ~1);
    __nv_bfloat162 val2;
    asm volatile("mov.b32 {%0, %1}, %2;" : "=h"(val2.x), "=h"(val2.y) : "r"(*address_i));
    bool is_even = (((uintptr_t)address) >> 1) & ~1;
    while (1) {
        int old = *address_i;
        __nv_bfloat162 old2;
        asm volatile("mov.b32 {%0, %1}, %2;" : "=h"(old2.x), "=h"(old2.y) : "r"(old));
        if (is_even) {
            val2.x = __float2bfloat16(__bfloat162float(old2.x) + __bfloat162float(val));
        } else {
            val2.y = __float2bfloat16(__bfloat162float(old2.y) + __bfloat162float(val));
        }
        int newval;
        asm volatile("mov.b32 %0, {%1, %2};" : "=r"(newval) : "h"(val2.x), "h"(val2.y));
        int replaced = atomicCAS(address_i, old, newval);
        if (replaced == old) {
            break;
        }
    }
}

__device__ __forceinline__ void load_tile(const __nv_bfloat16* gmem_base, const __nv_bfloat16* gmem, __nv_bfloat16* smem, int S, int stride, int global_i) {
    float4 gmem_vals[16];
    uint32_t base_byte = ((const char*)gmem) - ((const char*)gmem_base);
    for (int i = 0; i < 16; ++i) {
        uint32_t byte_offset = base_byte + (threadIdx.x * 256 + i * 16);
        uint32_t row = byte_offset / (128 * sizeof(__nv_bfloat16));
        uint32_t col = (byte_offset % (128 * sizeof(__nv_bfloat16))) / sizeof(__nv_bfloat16);
        if (global_i + row < S && col < 128) {
            gmem_vals[i] = *(const float4*)&gmem[row * stride + col];
        } else {
            float4 zero = {0, 0, 0, 0};
            gmem_vals[i] = zero;
        }
    }
    for (int i = 0; i < 16; ++i) {
        uint32_t byte_offset = base_byte + (threadIdx.x * 256 + i * 16);
        uint32_t row = byte_offset / (128 * sizeof(__nv_bfloat16));
        uint32_t col = (byte_offset % (128 * sizeof(__nv_bfloat16))) / sizeof(__nv_bfloat16);
        if (global_i + row < S && col < 128) {
            *(float4*)&smem[row * 128 + col] = gmem_vals[i];
        } else {
            *(float4*)&smem[row * 128 + col] = {0, 0, 0, 0};
        }
    }
}

__device__ __forceinline__ void transpose_128x128(__nv_bfloat16* smem) {
    for (int col = threadIdx.x; col < 128; col += 128) {
        for (int row = 0; row < col; row++) {
            __nv_bfloat16 tmp = smem[row * 128 + col];
            smem[row * 128 + col] = smem[col * 128 + row];
            smem[col * 128 + row] = tmp;
        }
    }
}

__device__ __forceinline__ void gemm_128x128(const __nv_bfloat16* smem_A, const __nv_bfloat16* smem_B, __nv_bfloat16* out_shared) {
    int warp_id = threadIdx.x / 32;
    int row = warp_id * 32;
    
    wmma::accumulator<16,16,float> acc_B[8];
    for (int i = 0; i < 8; ++i) { wmma::fill_fragment(acc_B[i]); }
    
    for (int i = 0; i < 8; ++i) {
        wmma::fragment<wmma::matrix_b, 16,16,16,__nv_bfloat16> B[i];
        uint32_t b_addr = (0 * 128 + i * 16 * 128) * sizeof(__nv_bfloat16);
        wmma::load_matrix_sync(B[i], &smem_B[b_addr]);
    }
    for (int k = 1; k < 8; ++k) {
        for (int i = 0; i < 8; ++i) {
            uint32_t b_addr = (k * 128 + i * 16 * 128) * sizeof(__nv_bfloat16);
            wmma::load_matrix_sync(B[i], &smem_B[b_addr]);
        }
        
        wmma::fragment<wmma::matrix_a, 16,16,16,__nv_bfloat16> A[8];
        for (int i = 0; i < 8; ++i) {
            uint32_t a_addr = (row * 128 + k * 16 + i * 16 * 128) * sizeof(__nv_bfloat16);
            wmma::load_matrix_sync(A[i], &smem_A[a_addr]);
            wmma::mma_sync(acc_B[i], A[i], B[i]);
        }
    }
    
    for (int i = 0; i < 8; ++i) {
        float local_out[16];
        wmma::store_matrix_sync(local_out, acc_B[i], 16, wmma::mem_row_major);
        uint32_t out_base = (row + i * 16) * 128 * sizeof(float);
        *(float(*)[16])(&out_shared[out_base]) = *(float(*)[16])(&local_out[0]);
    }
}

__global__ __launch_bounds__(128)
void bwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    int S,
    float attn_scale,
    uint64_t batch_head_stride)
{
    int num_j = (S + 127) / 128;
    int j_idx = blockIdx.x / num_j;
    int global_j = (blockIdx.x % num_j) * 128;

    if (global_j >= S) return;

    const __nv_bfloat16* q_ptr = Q + j_idx * S * 128;
    const __nv_bfloat16* k_ptr = K + j_idx * S * 128;
    const __nv_bfloat16* v_ptr = V + j_idx * S * 128;
    const __nv_bfloat16* o_ptr = O + j_idx * S * 128;
    const __nv_bfloat16* do_ptr = dO + j_idx * S * 128;
    
    __nv_bfloat16* dq_ptr = dQ + j_idx * S * 128;
    __nv_bfloat16* dk_ptr = dK + j_idx * S * 128;
    __nv_bfloat16* dv_ptr = dV + j_idx * S * 128;
    
    const float* l_ptr = L + j_idx * batch_head_stride;

    __shared__ __nv_bfloat16 s_k[128*128];
    __shared__ __nv_bfloat16 s_v[128*128];
    __shared__ __nv_bfloat16 s_q[128*128];
    __shared__ __nv_bfloat16 s_o[128*128];
    __shared__ __nv_bfloat16 s_do[128*128];
    __shared__ __nv_bfloat16 s_s[128*128];
    __shared__ __nv_bfloat16 s_p[128*128];
    __shared__ __nv_bfloat16 s_dp[128*128];
    __shared__ __nv_bfloat16 s_ds[128*128];
    __shared__ float s_d[128];
    __shared__ float s_l[128];

    load_tile(k_ptr, k_ptr + global_j * 128, s_k, S, 128, global_j);
    load_tile(v_ptr, v_ptr + global_j * 128, s_v, S, 128, global_j);
    
    for(int i = threadIdx.x; i < 128*128; i += 128) {
        s_dk[i] = __float2bfloat16(0.0f);
        s_dv[i] = __float2bfloat16(0.0f);
    }
    __syncthreads();

    for (int global_i = global_j; global_i < S; global_i += 128) {
        load_tile(q_ptr, q_ptr + global_i * 128, s_q, S, 128, global_i);
        load_tile(o_ptr, o_ptr + global_i * 128, s_o, S, 128, global_i);
        load_tile(do_ptr, do_ptr + global_i * 128, s_do, S, 128, global_i);
        
        if (global_i + threadIdx.x < S) {
            s_l[threadIdx.x] = l_ptr[global_i + threadIdx.x];
        } else {
            s_l[threadIdx.x] = 0.0f;
        }
        __syncthreads(); 
        
        float d_val = 0;
        for (int col = 0; col < 128; col++) {
            d_val += __bfloat162float(s_o[threadIdx.x * 128 + col]) * 
                     __bfloat162float(s_do[threadIdx.x * 128 + col]);
        }
        s_d[threadIdx.x] = d_val;
        __syncthreads();
        
        transpose_128x128(s_q);
        __syncthreads();
        
        gemm_128x128(s_k, s_q, s_s);
        __syncthreads();
        
        for (int col = 0; col < 128; col++) {
            int g_i = global_i + threadIdx.x;
            int g_j = global_j + col;
            if (g_j > g_i || g_j >= S) {
                s_p[threadIdx.x * 128 + col] = __float2bfloat16(0.0f);
            } else {
                float s_val = __bfloat162float(s_s[threadIdx.x * 128 + col]);
                float p_val = expf(s_val * attn_scale - s_l[threadIdx.x]);
                s_p[threadIdx.x * 128 + col] = __float2bfloat16(p_val);
            }
        }
        __syncthreads();
        
        transpose_128x128(s_do);
        __syncthreads();
        
        gemm_128x128(s_v, s_do, s_dp);
        __syncthreads();
        
        for (int col = 0; col < 128; col++) {
            float dp_val = __bfloat162float(s_dp[threadIdx.x * 128 + col]);
            float ds_val = __bfloat162float(s_p[threadIdx.x * 128 + col]) * (dp_val - s_d[threadIdx.x]);
            s_ds[threadIdx.x * 128 + col] = __float2bfloat16(ds_val);
        }
        __syncthreads();
        
        transpose_128x128(s_do);
        __syncthreads();
        
        gemm_128x128(s_ds, s_k, s_s); 
        __syncthreads();
        
        for (int col = 0; col < 128; col++) {
            int g_i = global_i + threadIdx.x;
            if (g_i < S && col < 128) {
                atomicAddAvg(&dq_ptr[g_i * 128 + col], s_s[threadIdx.x * 128 + col]);
            }
        }
        
        transpose_128x128(s_ds);
        __syncthreads();
        
        gemm_128x128(s_ds, s_q, s_dp); 
        __syncthreads();
        
        for(int idx = threadIdx.x; idx < 128*128; idx += 128) {
            float cur = __bfloat162float(s_dk[idx]);
            float contrib = __bfloat162float(s_dp[idx]);
            s_dk[idx] = __float2bfloat16(cur + contrib);
        }
        __syncthreads();
        
        transpose_128x128(s_p);
        __syncthreads();
        
        gemm_128x128(s_p, s_do, s_ds); 
        __syncthreads();
        
        for(int idx = threadIdx.x; idx < 128*128; idx += 128) {
            float cur = __bfloat162float(s_dv[idx]);
            float contrib = __bfloat162float(s_ds[idx]);
            s_dv[idx] = __float2bfloat16(cur + contrib);
        }
        __syncthreads();
        
        transpose_128x128(s_q);
        __syncthreads();
    } 
    
    for(int idx = threadIdx.x; idx < 128*128; idx += 128) {
        int row = idx / 128;
        int col = idx % 128;
        int g_j = global_j + row;
        if (g_j < S && col < 128) {
            dk_ptr[g_j * 128 + col] = s_dk[idx];
            dv_ptr[g_j * 128 + col] = s_dv[idx];
        }
    }
}

namespace tvm_ffi_mha_bwd {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t d = Q.size(3);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    CUDA_CHECK(cudaMemsetAsync(dQ.data_ptr(), 0, B * H * S * d * sizeof(__nv_bfloat16), stream));
    
    int num_j = (S + 127) / 128;
    dim3 grid(B * H * num_j);
    dim3 block(128);
    
    float attn_scale = 1.0f / sqrtf((float)d);
    
    CUDA_CHECK(cudaFuncSetAttribute(bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 228 * 1024));
    
    bwd_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<const __nv_bfloat16*>(O.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        static_cast<__nv_bfloat16*>(dK.data_ptr()),
        static_cast<__nv_bfloat16*>(dV.data_ptr()),
        S, attn_scale, L.stride(1)
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_bwd::run);

} // namespace tvm_ffi_mha_bwd