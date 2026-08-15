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

using namespace nvcuda;

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

#define CU_CHECK(call) do {                                        \
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        fprintf(stderr, "CU error %d at %s:%d\n",                  \
                _e, __FILE__, __LINE__);                           \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_example_cuda {

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, globalAddress, globalDim, globalStrides,
        boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, l2Promotion, oobFill
    );
}

__device__ __forceinline__ uint32_t fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return __float_as_uint(y);
}

__device__ __forceinline__ uint32_t pack_bf16_fn(uint32_t fp32_a, uint32_t fp32_b) {
    __nv_bfloat16 a = __float2bfloat16(__uint_as_float(fp32_a));
    __nv_bfloat16 b = __float2bfloat16(__uint_as_float(fp32_b));
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};"
        : "=r"(result)
        : "h"(*reinterpret_cast<uint16_t*>(&a)),
          "h"(*reinterpret_cast<uint16_t*>(&b)));
    return result;
}

// ---------------- KERNEL DEFINITION ----------------
__global__ void bwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q, 
    const __grid_constant__ CUtensorMap tma_K, 
    const __grid_constant__ CUtensorMap tma_V, 
    const __grid_constant__ CUtensorMap tma_dO, 
    const __grid_constant__ CUtensorMap tma_O, 
    __nv_bfloat16* dQ, __nv_bfloat16* dK, __nv_bfloat16* dV, 
    const float* L, int num_blocks, int M_global, int N_global,
    const __nv_bfloat16* Q_ptr, const __nv_bfloat16* K_ptr, 
    const __nv_bfloat16* V_ptr, const __nv_bfloat16* dO_ptr, 
    const __nv_bfloat16* O_ptr, int S_len, int bh,
    float* temp_dK, float* temp_dV) 
{
    extern __shared__ char smem[];
    char* smem_aligned = smem;
    uintptr_t smem_addr = (uintptr_t)smem;
    if (smem_addr % 1024 != 0) {
        smem_aligned = smem + (1024 - (smem_addr % 1024));
    }

    __nv_bfloat16* smem_Q_head = (__nv_bfloat16*)smem_aligned;
    __nv_bfloat16* smem_Q_tail = smem_Q_head + 64 * 64;
    __nv_bfloat16* smem_O_head = smem_Q_tail + 64 * 64;
    __nv_bfloat16* smem_O_tail = smem_O_head + 64 * 64;
    __nv_bfloat16* smem_dO_head = smem_O_tail + 64 * 64;
    __nv_bfloat16* smem_dO_tail = smem_dO_head + 64 * 64;
    __nv_bfloat16* smem_K_head = smem_dO_tail + 64 * 64;
    __nv_bfloat16* smem_K_tail = smem_K_head + 64 * 64;
    __nv_bfloat16* smem_V_head = smem_K_tail + 64 * 64;
    __nv_bfloat16* smem_V_tail = smem_V_head + 64 * 64;
    float* smem_S_float = (float*)(smem_V_tail + 64 * 64);
    __nv_bfloat16* smem_P_bf16 = (__nv_bfloat16*)(smem_S_float + 64 * 64);
    float* smem_dP_float = (float*)(smem_P_bf16 + 64 * 64);
    __nv_bfloat16* smem_dS_bf16 = (__nv_bfloat16*)(smem_dP_float + 64 * 64);
    float* smem_D_local = (float*)(smem_dS_bf16 + 64 * 64);
    float* smem_LSE_local = smem_D_local + 64;

    int q_idx = blockIdx.x;
    const float* L_ptr = L;
    float attn_scale = 1.0f / sqrtf(128.0f);

    // Load Q, O, dO safely and correctly via float4 vectors 
    float4* smem_Q_h_f4 = (float4*)smem_Q_head;
    float4* gmem_Q_h_f4 = (float4*)(Q_ptr + bh * S_len * 128 + q_idx * 64 * 128);
    for(int idx = threadIdx.x; idx < 512; idx += blockDim.x) {
        smem_Q_h_f4[idx] = gmem_Q_h_f4[idx];
    }
    float4* smem_Q_t_f4 = (float4*)smem_Q_tail;
    float4* gmem_Q_t_f4 = (float4*)(Q_ptr + bh * S_len * 128 + q_idx * 64 * 128 + 64);
    for(int idx = threadIdx.x; idx < 512; idx += blockDim.x) {
        smem_Q_t_f4[idx] = gmem_Q_t_f4[idx];
    }
    float4* smem_O_h_f4 = (float4*)smem_O_head;
    float4* gmem_O_h_f4 = (float4*)(O_ptr + bh * S_len * 128 + q_idx * 64 * 128);
    for(int idx = threadIdx.x; idx < 512; idx += blockDim.x) {
        smem_O_h_f4[idx] = gmem_O_h_f4[idx];
    }
    float4* smem_O_t_f4 = (float4*)smem_O_tail;
    float4* gmem_O_t_f4 = (float4*)(O_ptr + bh * S_len * 128 + q_idx * 64 * 128 + 64);
    for(int idx = threadIdx.x; idx < 512; idx += blockDim.x) {
        smem_O_t_f4[idx] = gmem_O_t_f4[idx];
    }
    float4* smem_dO_h_f4 = (float4*)smem_dO_head;
    float4* gmem_dO_h_f4 = (float4*)(dO_ptr + bh * S_len * 128 + q_idx * 64 * 128);
    for(int idx = threadIdx.x; idx < 512; idx += blockDim.x) {
        smem_dO_h_f4[idx] = gmem_dO_h_f4[idx];
    }
    float4* smem_dO_t_f4 = (float4*)smem_dO_tail;
    float4* gmem_dO_t_f4 = (float4*)(dO_ptr + bh * S_len * 128 + q_idx * 64 * 128 + 64);
    for(int idx = threadIdx.x; idx < 512; idx += blockDim.x) {
        smem_dO_t_f4[idx] = gmem_dO_t_f4[idx];
    }

    __syncthreads();

    float d_val = 0;
    int row_idx = (threadIdx.x / 4) * 2 + (threadIdx.x % 4) / 2;
    int col_idx = (threadIdx.x % 4) * 2 + (threadIdx.x % 2);
    
    if (col_idx < 64) {
        d_val += __bfloat162float(smem_O_head[row_idx * 64 + col_idx]) * __bfloat162float(smem_dO_head[row_idx * 64 + col_idx]);
    }
    d_val += __bfloat162float(smem_O_tail[row_idx * 64 + (col_idx - 64)]) * __bfloat162float(smem_dO_tail[row_idx * 64 + (col_idx - 64)]);
    smem_D_local[row_idx] = d_val;
    
    if (threadIdx.x < 64) {
        smem_LSE_local[threadIdx.x] = L_ptr[bh * S_len + q_idx * 64 + threadIdx.x];
    }

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dK_head_frag[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dK_tail_frag[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dV_head_frag[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dV_tail_frag[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dQ_head_frag[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dQ_tail_frag[2];

    for (int i = 0; i < 2; ++i) {
        wmma::fill_fragment(dK_head_frag[i], 0.0f);
        wmma::fill_fragment(dK_tail_frag[i], 0.0f);
        wmma::fill_fragment(dV_head_frag[i], 0.0f);
        wmma::fill_fragment(dV_tail_frag[i], 0.0f);
        wmma::fill_fragment(dQ_head_frag[i], 0.0f);
        wmma::fill_fragment(dQ_tail_frag[i], 0.0f);
    }

    int WarpID = threadIdx.x / 32;
    int row_warp = WarpID / 4;
    int col_warp = WarpID % 4;

    for (int kv_idx = 0; kv_idx <= q_idx && kv_idx < num_blocks; ++kv_idx) {
        
        float4* smem_K_h_f4 = (float4*)smem_K_head;
        float4* gmem_K_h_f4 = (float4*)(K_ptr + bh * S_len * 128 + kv_idx * 64 * 128);
        for(int idx = threadIdx.x; idx < 512; idx += blockDim.x) {
            smem_K_h_f4[idx] = gmem_K_h_f4[idx];
        }
        float4* smem_K_t_f4 = (float4*)smem_K_tail;
        float4* gmem_K_t_f4 = (float4*)(K_ptr + bh * S_len * 128 + kv_idx * 64 * 128 + 64);
        for(int idx = threadIdx.x; idx < 512; idx += blockDim.x) {
            smem_K_t_f4[idx] = gmem_K_t_f4[idx];
        }
        float4* smem_V_h_f4 = (float4*)smem_V_head;
        float4* gmem_V_h_f4 = (float4*)(V_ptr + bh * S_len * 128 + kv_idx * 64 * 128);
        for(int idx = threadIdx.x; idx < 512; idx += blockDim.x) {
            smem_V_h_f4[idx] = gmem_V_h_f4[idx];
        }
        float4* smem_V_t_f4 = (float4*)smem_V_tail;
        float4* gmem_V_t_f4 = (float4*)(V_ptr + bh * S_len * 128 + kv_idx * 64 * 128 + 64);
        for(int idx = threadIdx.x; idx < 512; idx += blockDim.x) {
            smem_V_t_f4[idx] = gmem_V_t_f4[idx];
        }
        __syncthreads();

        for (int row_iter = 0; row_iter < 2; ++row_iter) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> ma_S;
            wmma::fill_fragment(ma_S, 0.0f);

            for (int k = 0; k < 2; ++k) {
                for (int i = 0; i < 4; ++i) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16x2> ma_Q;
                    __nv_bfloat16* Q_h_ptr = (k == 0) ? smem_Q_head : smem_Q_tail;
                    wmma::load_matrix_sync(ma_Q, Q_h_ptr + (row_warp * 32 + row_iter * 16) * 64 + i * 16