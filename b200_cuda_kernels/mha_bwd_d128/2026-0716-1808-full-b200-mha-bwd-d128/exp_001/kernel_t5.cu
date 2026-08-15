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

#define DRV_CHECK(call) do {                                       \
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        const char* err_str;                                       \
        cuGetErrorString(_e, &err_str);                            \
        fprintf(stderr, "CUDA Driver error %s at %s:%d\n",         \
                err_str, __FILE__, __LINE__);                      \
        exit(1);                                                   \
    }                                                              \
} while(0)

__device__ __forceinline__ bool elect_one_sync_fn() {
    uint32_t pred;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "elect.sync _|p, 0xFFFFFFFF;\n"
        "selp.b32 %0, 1, 0, p;\n"
        "}\n"
        : "=r"(pred));
    return pred != 0;
}

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(tx_bytes) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(phase));
}

__device__ __forceinline__ void fence_proxy_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tma_load_3d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
}

__device__ __forceinline__ int swizzle_128B(int row, int col) {
    int chunk_x = col / 8;
    int chunk_x_swizzled = chunk_x ^ (row % 8);
    return chunk_x_swizzled * 8 + (col % 8);
}

CUresult create_tma_3d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t dim0, uint64_t dim1, uint64_t dim2, uint32_t box0, uint32_t box1, uint32_t box2, CUtensorMapSwizzle swizzle) {
    cuuint64_t globalDim[3] = {dim0, dim1, dim2};
    cuuint64_t globalStrides[2] = {dim0 * 2, dim0 * dim1 * 2};
    cuuint32_t boxDim[3] = {box0, box1, box2};
    cuuint32_t elementStrides[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, globalAddress, globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

namespace tvm_ffi_example_cuda {

__device__ __forceinline__ void gemm_64x64x64(
    float* D, float* D_init,
    const __nv_bfloat16* A, const __nv_bfloat16* B,
    int stride_A, int stride_B, int stride_D)
{
    for (int i = tid; i < 4096; i += blockDim.x) {
        int row = i / 64;
        int col = i % 64;
        float sum = D_init ? D_init[row * stride_D + col] : 0.0f;
        for (int k = 0; k < 64; ++k) {
            sum += __bfloat162float(A[row * stride_A + k]) * 
                   __bfloat162float(B[k * stride_B + col]);
        }
        D[row * stride_D + col] = sum;
    }
}

__device__ __forceinline__ void gemm_64x64x64_trans_b(
    float* D, float* D_init,
    const __nv_bfloat16* A, const __nv_bfloat16* B,
    int stride_A, int stride_B, int stride_D)
{
    for (int i = tid; i < 4096; i += blockDim.x) {
        int row = i / 64;
        int col = i % 64;
        float sum = D_init ? D_init[row * stride_D + col] : 0.0f;
        for (int k = 0; k < 64; ++k) {
            sum += __bfloat162float(A[row * stride_A + k]) * 
                   __bfloat162float(B[col * stride_B + k]);
        }
        D[row * stride_D + col] = sum;
    }
}

__device__ __forceinline__ void gemm_64x64x64_trans_a(
    float* D, float* D_init,
    const __nv_bfloat16* A, const __nv_bfloat16* B,
    int stride_A, int stride_B, int stride_D)
{
    for (int i = tid; i < 4096; i += blockDim.x) {
        int row = i / 64;
        int col = i % 64;
        float sum = D_init ? D_init[row * stride_D + col] : 0.0f;
        for (int k = 0; k < 64; ++k) {
            sum += __bfloat162float(A[k * stride_A + row]) * 
                   __bfloat162float(B[k * stride_B + col]);
        }
        D[row * stride_D + col] = sum;
    }
}

__device__ __forceinline__ void load_tile(__nv_bfloat16* s_mem, const __nv_bfloat16* gmem, int bh, int r_start, int c_start, int S, int d) {
    for (int i = tid; i < 4096; i += blockDim.x) {
        int row = i / 64;
        int col = i % 64;
        int g_idx = bh * S * d + (r_start + row) * d + (c_start + col);
        if (r_start + row < S && c_start + col < d)
            s_mem[row * 64 + swizzle_128B(row, col)] = gmem[g_idx];
        else
            s_mem[row * 64 + swizzle_128B(row, col)] = __float2bfloat16(0.0f);
    }
}

__device__ __forceinline__ void softmax_to_smem(__nv_bfloat16* s_P, const float* s_S, const float* L, int bh, int k_blk, int j_blk, int S, float scale) {
    for (int i = tid; i < 4096; i += blockDim.x) {
        int row = i / 64;
        int col = i % 64;
        float s = s_S[row * 64 + col] * scale;
        float l = L[bh * S + (k_blk * 64 + row)];
        float p = __isnanf(l) ? 0.0f : expf(s - l);
        
        int global_row = k_blk * 64 + row;
        int global_col = j_blk * 64 + col;
        if (global_row >= S || global_col >= S) {
            p = 0.0f;
        }
        
        s_P[row * 64 + swizzle_128B(row, col)] = __float2bfloat16(p);
    }
}

__device__ __forceinline__ void compute_Ds(__nv_bfloat16* s_D_s, const float* s_dP, const __nv_bfloat16* s_P, int k_blk, int j_blk, int S, float scale) {
    for (int i = tid; i < 4096; i += blockDim.x) {
        int row = i / 64;
        int col = i % 64;
        float dp = s_dP[row * 64 + col];
        float p = __bfloat162float(s_P[row * 64 + swizzle_128B(row, col)]);
        
        float ds = dp * p * scale;
        
        int global_row = k_blk * 64 + row;
        int global_col = j_blk * 64 + col;
        if (global_row >= S || global_col >= S) ds = 0.0f;
        
        s_D_s[row * 64 + swizzle_128B(row, col)] = __float2bfloat16(ds);
    }
}

__global__ __launch_bounds__(128) void mha_bwd_dq_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_dO,
    const float* dQ_gmem, const float* L, int32_t S, int32_t d) 
{
    int32_t k_blk = blockIdx.x;
    int32_t bh = blockIdx.y;
    int32_t num_blocks = (S + 63) / 64;
    float scale = 1.0f / sqrtf((float)d);
    int tid = threadIdx.x;

    extern __shared__ __align__(1024) char smem_buf[];
    char* s_Q_h0  = smem_buf + 0 * 8192;
    char* s_Q_h1  = smem_buf + 1 * 8192;
    char* s_K_h0  = smem_buf + 2 * 8192;
    char* s_K_h1  = smem_buf + 3 * 8192;
    char* s_V_h0  = smem_buf + 4 * 8192;
    char* s_V_h1  = smem_buf + 5 * 8192;
    char* s_dO_h0 = smem_buf + 6 * 8192;
    char* s_dO_h1 = smem_buf + 7 * 8192;
    char* s_P     = smem_buf + 8 * 8192;
    char* s_D_s   = smem_buf + 9 * 8192;
    char* s_S_acc = smem_buf + 10 * 8192;
    char* s_dP_acc= smem_buf + 11 * 8192;
    
    uint64_t* bar_outer = (uint64_t*)(smem_buf + 12 * 8192);
    uint64_t* bar_inner = (uint64_t*)(smem_buf + 12 * 8192 + sizeof(uint64_t));

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(bar_outer, 1);
        init_smem_barrier_fn(bar_inner, 1);
    }
    __syncthreads();

    if (elect_one_sync_fn()) {
        mbarrier_arrive_and_expect_tx_fn(bar_outer, 4 * 8192);
        tma_load_3d_fn(&tma_Q, bar_outer, s_Q_h0, 0, k_blk * 64, bh);
        tma_load_3d_fn(&tma_Q, bar_outer, s_Q_h1, 64, k_blk * 64, bh);
        tma_load_3d_fn(&tma_dO, bar_outer, s_dO_h0, 0, k_blk * 64, bh);
        tma_load_3d_fn(&tma_dO, bar_outer, s_dO_h1, 64, k_blk * 64, bh);
    }

    uint32_t phase_outer = 0;
    mbarrier_wait_fn(bar_outer, phase_outer);
    __syncthreads();
    fence_proxy_async_shared_fn();
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar_inner, 0);
    }
    mbarrier_wait_fn(bar_inner, 0);

    float dQ_h0_acc[4] = {0}, dQ_h1_acc[4] = {0};

    uint32_t phase_inner = 0;
    for (int32_t j_blk = 0; j_blk < num_blocks; ++j_blk) {
        if (elect_one_sync_fn()) {
            mbarrier_arrive_and_expect_tx_fn(bar_inner, 4 * 8192);
            tma_load_3d_fn(&tma_K, bar_inner, s_K_h0, 0, j_blk * 64, bh);
            tma_load_3d_fn(&tma_K, bar_inner, s_K_h1, 64, j_blk * 64, bh);
            tma_load_3d_fn(&tma_V, bar_inner, s_V_h0, 0, j_blk * 64, bh);
            tma_load_3d_fn(&tma_V, bar_inner, s_V_h1, 64, j_blk * 64, bh);
        }
        mbarrier_wait_fn(bar_inner, phase_inner);
        __syncthreads();
        fence_proxy_async_shared_fn();

        for (int i = tid; i < 4096; i += blockDim.x) *(float*)((char*)s_S_acc + i * sizeof(float)) = 0.0f;
        
        gemm_64x64x64_trans_b((float*)s_S_acc, (float*)s_S_acc, (__nv_bfloat16*)s_Q_h0, (__nv_bfloat16*)s_K_h0, 64, 64, 64);
        gemm_64x64x64_trans_b((float*)s_S_acc, (float*)s_S_acc, (__nv_bfloat16*)s_Q_h1, (__nv_bfloat16*)s_K_h1, 64, 64, 64);
        
        __syncthreads();

        softmax_to_smem((__nv_bfloat16*)s_P, (float*)s_S_acc, L, bh, k_blk, j_blk, S, scale);
        __syncthreads();
        fence_proxy_async_shared_fn();

        for (int i = tid; i < 4096; i += blockDim.x) *(float*)((char*)s_dP_acc + i * sizeof(float)) = 0.0f;
        
        gemm_64x64x64_trans_b((float*)s_dP_acc, (float*)s_dP_acc, (__nv_bfloat16*)s_dO_h0, (__nv_bfloat16*)s_V_h0, 64, 64, 64);
        gemm_64x64x64_trans_b((float*)s_dP_acc, (float*)s_dP_acc, (__nv_bfloat16*)s_dO_h1, (__nv_bfloat16*)s_V_h1, 64, 64, 64);
        __syncthreads();
        fence_proxy_async_shared_fn();

        compute_Ds((__nv_bfloat16*)s_D_s, (float*)s_dP_acc, (__nv_bfloat16*)s_P, k_blk, j_blk, S, scale);
        __syncthreads();
        fence_proxy_async_shared_fn();

        // Accumulate correctly over j_blk loops
        gemm_64x64x64_trans_b((float*)s_Q_h0, dQ_h0_acc, (__nv_bfloat16*)s_D_s, (__nv_bfloat16*)s_K_h0, 64, 64, 64);
        gemm_64x64x64_trans_b((float*)s_Q_h1, dQ_h1_acc, (__nv_bfloat16*)s_D_s, (__nv_bfloat16*)s_K_h1, 64, 64, 64);
        
        __syncthreads();
        phase_inner ^= 1;
    }

    for (int i = 0; i < 4; ++i) {
        int row = tid / 4;
        int lane_c = (tid % 4) * 4 + i;
        
        *reinterpret_cast<float*>((char*)s_Q_h0 + row * 16 * sizeof(float) + lane_c * sizeof(float)) = dQ_h0_acc[i];
        *reinterpret_cast<float*>((char*)s_Q_h1 + row * 16 * sizeof(float) + lane_c * sizeof(float)) = dQ_h1_acc[i];
    }

    __syncthreads();
    fence_proxy_async_shared_fn();

    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(dQ_gmem);
    for (int i = tid; i < 4096; i += blockDim.x) {
        int row = i / 64;
        int col = i % 64;
        int g_row = k_blk * 64 + row;
        if (g_row < S) {
            int swizzled_col = swizzle_128B(row, col);
            __nv_bfloat16 val_h0 = *reinterpret_cast<__nv_bfloat16*>((char*)s_Q_h0 + row * 64 + swizzled_col);
            __nv_bfloat16 val_h1 = *reinterpret_cast<__nv_bfloat16*>((char*)s_Q_h1 + row * 64 + swizzled_col);
            
            int g_col_h0 = col;
            int g_col_h1 = col + 64;
            
            if (g_col_h0 < d) Q_ptr[bh * S * d + g_row * d + g_col_h0] = val_h0;
            if (g_col_h1 < d) Q_ptr[bh * S * d + g_row * d + g_col_h1] = val_h1;
        }
    }
    __syncthreads();
}

__global__ __launch_bounds__(128) void mha_bwd_dkv_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_dO,
    const float* dK_gmem, const float* dV_gmem, const float* L, int32_t S, int32_t d) 
{
    int32_t j_blk = blockIdx.x;
    int32_t bh = blockIdx.y;
    int32_t num_blocks = (S + 63) / 64;
    float scale = 1.0f / sqrtf((float)d);
    int tid = threadIdx.x;

    extern __shared__ __align__(1024) char smem_buf[];
    char* s_Q_h0  = smem_buf + 0 * 8192;
    char* s_Q_h1  = smem_buf + 1 * 8192;
    char* s_K_h0  = smem_buf + 2 * 8192;
    char* s_K_h1  = smem_buf + 3 * 8192;
    char* s_V_h0  = smem_buf + 4 * 8192;
    char* s_V_h1  = smem_buf + 5 * 8192;
    char* s_dO_h0 = smem_buf + 6 * 8192;
    char* s_dO_h1 = smem_buf + 7 * 8192;
    char* s_P     = smem_buf + 8 * 8192;
    char* s_D_s   = smem_buf + 9 * 8192;
    char* s_S_acc = smem_buf + 10 * 8192;
    char* s_dP_acc= smem_buf + 11 * 8192;
    
    uint64_t* bar_outer = (uint64_t*)(smem_buf + 12 * 8192);
    uint64_t* bar_inner = (uint64_t*)(smem_buf + 12 * 8192 + sizeof(uint64_t));

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(bar_outer, 1);
        init_smem_barrier_fn(bar_inner, 1);
    }
    __syncthreads();

    if (elect_one_sync_fn()) {
        mbarrier_arrive_and_expect_tx_fn(bar_outer, 4 * 8192);
        tma_load_3d_fn(&tma_K, bar_outer, s_K_h0, 0, j_blk * 64, bh);
        tma_load_3d_fn(&tma_K, bar_outer, s_K_h1, 64, j_blk * 64, bh);
        tma_load_3d_fn(&tma_V, bar_outer, s_V_h0, 0, j_blk * 64, bh);
        tma_load_3d_fn(&tma_V, bar_outer, s_V_h1, 64, j_blk * 64, bh);
    }

    uint32_t phase_outer = 0;
    mbarrier_wait_fn(bar_outer, phase_outer);
    __syncthreads();
    fence_proxy_async_shared_fn();
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar_inner, 0);
    }
    mbarrier_wait_fn(bar_inner, 0);

    float dK_h0_acc[4] = {0}, dK_h1_acc[4] = {0};
    float dV_h0_acc[4] = {0}, dV_h1_acc[4] = {0};

    uint32_t phase_inner = 0;
    for (int32_t k_blk = 0; k_blk < num_blocks; ++k_blk) {
        if (elect_one_sync_fn()) {
            mbarrier_arrive_and_expect_tx_fn(bar_inner, 4 * 8192);
            tma_load_3d_fn(&tma_Q, bar_inner, s_Q_h0, 0, k_blk * 64, bh);
            tma_load_3d_fn(&tma_Q, bar_inner, s_Q_h1, 64, k_blk * 64, bh);
            tma_load_3d_fn(&tma_dO, bar_inner, s_dO_h0, 0, k_blk * 64, bh);
            tma_load_3d_fn(&tma_dO, bar_inner, s_dO_h1, 64, k_blk * 64, bh);
        }
        mbarrier_wait_fn(bar_inner, phase_inner);
        __syncthreads();
        fence_proxy_async_shared_fn();

        for (int i = tid; i < 4096; i += blockDim.x) *(float*)((char*)s_S_acc + i * sizeof(float)) = 0.0f;

        gemm_64x64x64_trans_b((float*)s_S_acc, (float*)s_S_acc, (__nv_bfloat16*)s_Q_h0, (__nv_bfloat16*)s_K_h0, 64, 64, 64);
        gemm_64x64x64_trans_b((float*)s_S_acc, (float*)s_S_acc, (__nv_bfloat16*)s_Q_h1, (__nv_bfloat16*)s_K_h1, 64, 64, 64);
        
        __syncthreads();

        softmax_to_smem((__nv_bfloat16*)s_P, (float*)s_S_acc, L, bh, k_blk, j_blk, S, scale);
        __syncthreads();
        fence_proxy_async_shared_fn();

        for (int i = tid; i < 4096; i += blockDim.x) *(float*)((char*)s_dP_acc + i * sizeof(float)) = 0.0f;

        gemm_64x64x64_trans_b((float*)s_dP_acc, (float*)s_dP_acc, (__nv_bfloat16*)s_dO_h0, (__nv_bfloat16*)s_V_h0, 64, 64, 64);
        gemm_64x64x64_trans_b((float*)s_dP_acc, (float*)s_dP_acc, (__nv_bfloat16*)s_dO_h1, (__nv_bfloat16*)s_V_h1, 64, 64, 64);
        __syncthreads();
        fence_proxy_async_shared_fn();

        compute_Ds((__nv_bfloat16*)s_D_s, (float*)s_dP_acc, (__nv_bfloat16*)s_P, k_blk, j_blk, S, scale);
        __syncthreads();
        fence_proxy_async_shared_fn();

        gemm_64x64x64_trans_a((float*)s_K_h0, dK_h0_acc, (__nv_bfloat16*)s_D_s, (__nv_bfloat16*)s_Q_h0, 64, 64, 64);
        gemm_64x64x64_trans_a((float*)s_K_h1, dK_h1_acc, (__nv_bfloat16*)s_D_s, (__nv_bfloat16*)s_Q_h1, 64, 64, 64);

        gemm_64x64x64_trans_a((float*)s_V_h0, dV_h0_acc, (__nv_bfloat16*)s_P, (__nv_bfloat16*)s_dO_h0, 64, 64, 64);
        gemm_64x64x64_trans_a((float*)s_V_h1, dV_h1_acc, (__nv_bfloat16*)s_P, (__nv_bfloat16*)s_dO_h1, 64, 64, 64);
        
        __syncthreads();
        phase_inner ^= 1;
    }

    for (int i = 0; i < 4; ++i) {
        int row = tid / 4;
        int lane_c = (tid % 4) * 4 + i;
        
        *reinterpret_cast<float*>((char*)s_K_h0 + row * 16 * sizeof(float) + lane_c * sizeof(float)) = dK_h0_acc[i];
        *reinterpret_cast<float*>((char*)s_K_h1 + row * 16 * sizeof(float) + lane_c * sizeof(float)) = dK_h1_acc[i];
        
        *reinterpret_cast<float*>((char*)s_V_h0 + row * 16 * sizeof(float) + lane_c * sizeof(float)) = dV_h0_acc[i];
        *reinterpret_cast<float*>((char*)s_V_h1 + row * 16 * sizeof(float) + lane_c * sizeof(float)) = dV_h1_acc[i];
    }

    __syncthreads();
    fence_proxy_async_shared_fn();

    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(dK_gmem);
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(dV_gmem);
    for (int i = tid; i < 4096; i += blockDim.x) {
        int row = i / 64;
        int col = i % 64;
        int g_row = j_blk * 64 + row;
        if (g_row < S) {
            int swizzled_col = swizzle_128B(row, col);
            __nv_bfloat16 dk0 = *reinterpret_cast<__nv_bfloat16*>((char*)s_K_h0 + row * 64 + swizzled_col);
            __nv_bfloat16 dk1 = *reinterpret_cast<__nv_bfloat16*>((char*)s_K_h1 + row * 64 + swizzled_col);
            __nv_bfloat16 dv0 = *reinterpret_cast<__nv_bfloat16*>((char*)s_V_h0 + row * 64 + swizzled_col);
            __nv_bfloat16 dv1 = *reinterpret_cast<__nv_bfloat16*>((char*)s_V_h1 + row * 64 + swizzled_col);
            
            int g_col_h0 = col;
            int g_col_h1 = col + 64;
            
            if (g_col_h0 < d) {
                K_ptr[bh * S * d + g_row * d + g_col_h0] = dk0;
                V_ptr[bh * S * d + g_row * d + g_col_h0] = dv0;
            }
            if (g_col_h1 < d) {
                K_ptr[bh * S * d + g_row * d + g_col_h1] = dk1;
                V_ptr[bh * S * d + g_row * d + g_col_h1] = dv1;
            }
        }
    }
    __syncthreads();
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t d = Q.size(3);
    
    (void)O;
    
    float* L_data = static_cast<float*>(L.data_ptr());
    
    __nv_bfloat16* Q_ptr = static_cast<__nv_bfloat16*>(Q.data_ptr());
    __nv_bfloat16* K_ptr = static_cast<__nv_bfloat16*>(K.data_ptr());
    __nv_bfloat16* V_ptr = static_cast<__nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* dO_ptr = static_cast<__nv_bfloat16*>(dO.data_ptr());
    __nv_bfloat16* dQ_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_dO;
    
    DRV_CHECK(create_tma_3d_descriptor_2B(&tma_Q, Q_ptr, d, S, B * H, 64, 64, 1, CU_TENSOR_MAP_SWIZZLE_128B));
    DRV_CHECK(create_tma_3d_descriptor_2B(&tma_K, K_ptr, d, S, B * H, 64, 64, 1, CU_TENSOR_MAP_SWIZZLE_128B));
    DRV_CHECK(create_tma_3d_descriptor_2B(&tma_V, V_ptr, d, S, B * H, 64, 64, 1, CU_TENSOR_MAP_SWIZZLE_128B));
    DRV_CHECK(create_tma_3d_descriptor_2B(&tma_dO, dO_ptr, d, S, B * H, 64, 64, 1, CU_TENSOR_MAP_SWIZZLE_128B));
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    int64_t threads = 128;
    int64_t blocks_x = (S + 63) / 64;
    int64_t blocks_y = B * H;
    dim3 grid(blocks_x, blocks_y);
    
    int smem_size = 120 * 1024;
    CUDA_CHECK(cudaFuncSetAttribute((const void*)mha_bwd_dq_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    CUDA_CHECK(cudaFuncSetAttribute((const void*)mha_bwd_dkv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    mha_bwd_dq_kernel<<<grid, threads, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, tma_dO, reinterpret_cast<const float*>(dQ_ptr), L_data, S, d
    );
    CUDA_CHECK(cudaGetLastError());
    
    mha_bwd_dkv_kernel<<<grid, threads, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, tma_dO, reinterpret_cast<const float*>(dK_ptr), reinterpret_cast<const float*>(dV_ptr), L_data, S, d
    );
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

} // namespace tvm_ffi_example_cuda