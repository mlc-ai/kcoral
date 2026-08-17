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

__device__ __forceinline__ void tma_store_3d_fn(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group [%0, {%2, %3, %4}], [%1];"
        :: "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
}

__device__ __forceinline__ void tma_store_commit_fn() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void tma_store_wait_fn() {
    asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory");
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

__global__ __launch_bounds__(128) void mha_bwd_dq_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_dO,
    const __grid_constant__ CUtensorMap tma_dQ,
    const float* L,
    int32_t S) 
{
    int32_t k_blk = blockIdx.x;
    int32_t bh = blockIdx.y;
    int32_t num_blocks = (S + 63) / 64;
    float scale = 1.0f / sqrtf(128.0f);
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
        
        for (int k_block = 0; k_block < 4; ++k_block) {
            __nv_bfloat16* q_ptr = (k_block < 4) ? (__nv_bfloat16*)s_Q_h0 : (__nv_bfloat16*)s_Q_h1;
            __nv_bfloat16* k_ptr = (k_block < 4) ? (__nv_bfloat16*)s_K_h0 : (__nv_bfloat16*)s_K_h1;
            
            // S += Q @ K^T
            // Q (M x K) and K (N x K) -> K^T (K x N). Both stored with K contiguous.
            // Thus row_major for A and row_major for B performs Q @ K^T seamlessly
            typename GEMM_64x64x16_cg::Gemm gemm1{s_S_acc, q_ptr, k_ptr, 
                cutlass::gemm::GemmCoord(64, 64, 16), 64, 64, 64};
            gemm1(k_block * 16, 0, k_block * 16);
        }
        
        for (int k_block = 0; k_block < 4; ++k_block) {
            __nv_bfloat16* q_ptr = (k_block < 4) ? (__nv_bfloat16*)s_Q_h1 : (__nv_bfloat16*)s_Q_h0;
            __nv_bfloat16* k_ptr = (k_block < 4) ? (__nv_bfloat16*)s_K_h1 : (__nv_bfloat16*)s_K_h0;
            
            typename GEMM_64x64x16_cg::Gemm gemm1{s_S_acc, q_ptr, k_ptr, 
                cutlass::gemm::GemmCoord(64, 64, 16), 64, 64, 64};
            gemm1(k_block * 16 + 64, 0, k_block * 16);
        }
        
        __syncthreads();

        __nv_bfloat16* s_P_bf16 = (__nv_bfloat16*)s_P;

        for (int i = 0; i < 4; ++i) {
            int row = tid / 4;
            int lane_c = (tid % 4) * 4 + i;
            float s = ((float*)s_S_acc)[row * 16 + lane_c] * scale;
            
            int idx = ((tid * 2) + i) % 64;
            int idy = (((tid * 2) + i) / 64) * 2 + (i >= 2 ? 1 : 0);
            
            float l_val = L[bh * S + (j_blk * 64) + idx];
            float p = __isnanf(l_val) ? 0.0f : expf(s - l_val);
            
            int global_row = k_blk * 64 + idy;
            int global_col = j_blk * 64 + idx;
            if (global_row >= S || global_col >= S) {
                p = 0.0f;
            }
            
            s_P_bf16[row * 64 + swizzle_128B(row, lane_c)] = __float2bfloat16(p);
        }
        __syncthreads();
        fence_proxy_async_shared_fn();

        for (int i = tid; i < 4096; i += blockDim.x) *(float*)((char*)s_dP_acc + i * sizeof(float)) = 0.0f;
        
        // dP = dO @ V^T
        for (int k_block = 0; k_block < 4; ++k_block) {
            __nv_bfloat16* do_ptr = (k_block < 4) ? (__nv_bfloat16*)s_dO_h0 : (__nv_bfloat16*)s_dO_h1;
            __nv_bfloat16* v_ptr = (k_block < 4) ? (__nv_bfloat16*)s_V_h0 : (__nv_bfloat16*)s_V_h1;
            
            typename GEMM_64x64x16_cg::Gemm gemm2{s_dP_acc, do_ptr, v_ptr, 
                cutlass::gemm::GemmCoord(64, 64, 16), 64, 64, 64};
            gemm2(k_block * 16, 0, k_block * 16);
        }
        for (int k_block = 0; k_block < 4; ++k_block) {
            __nv_bfloat16* do_ptr = (k_block < 4) ? (__nv_bfloat16*)s_dO_h1 : (__nv_bfloat16*)s_dO_h0;
            __nv_bfloat16* v_ptr = (k_block < 4) ? (__nv_bfloat16*)s_V_h1 : (__nv_bfloat16*)s_V_h0;
            
            typename GEMM_64x64x16_cg::Gemm gemm2{s_dP_acc, do_ptr, v_ptr, 
                cutlass::gemm::GemmCoord(64, 64, 16), 64, 64, 64};
            gemm2(k_block * 16 + 64, 0, k_block * 16);
        }
        __syncthreads();

        for (int i = 0; i < 4; ++i) {
            int row = tid / 4;
            int lane_c = (tid % 4) * 4 + i;
            float dp = ((float*)s_dP_acc)[row * 16 + lane_c];
            
            int idx = ((tid * 2) + i) % 64;
            int idy = (((tid * 2) + i) / 64) * 2 + (i >= 2 ? 1 : 0);
            int global_row = k_blk * 64 + idy;
            int global_col = j_blk * 64 + idx;
            if (global_row >= S || global_col >= S) dp = 0.0f;
            
            float p = __bfloat162float(s_P_bf16[row * 64 + swizzle_128B(row, lane_c)]);
            
            float ds = dp * p * scale;
            *(__nv_bfloat16*)((char*)s_D_s + row * 128 + swizzle_128B(row, lane_c) * sizeof(__nv_bfloat16)) = __float2bfloat16(ds);
        }
        __syncthreads();
        fence_proxy_async_shared_fn();

        // dQ_h0 += D_s @ K_h0
        for (int k_block = 0; k_block < 4; ++k_block) {
            __nv_bfloat16* ds_ptr = (__nv_bfloat16*)s_D_s;
            __nv_bfloat16* k_ptr = (__nv_bfloat16*)s_K_h0;
            
            typename GEMM_64x64x64_cg::Gemm gemm3_0{dQ_h0_acc, ds_ptr, k_ptr};
            gemm3_0(0, k_block * 16, 0, k_block * 16);
        }
        
        // dQ_h1 += D_s @ K_h1
        for (int k_block = 0; k_block < 4; ++k_block) {
            __nv_bfloat16* ds_ptr = (__nv_bfloat16*)s_D_s;
            __nv_bfloat16* k_ptr = (__nv_bfloat16*)s_K_h1;
            
            typename GEMM_64x64x64_cg::Gemm gemm3_1{dQ_h1_acc, ds_ptr, k_ptr};
            gemm3_1(0, k_block * 16, 0, k_block * 16);
        }
        
        __syncthreads();
        phase_inner ^= 1;
    }

    for (int i = 0; i < 4; ++i) {
        int row = tid / 4;
        int lane_c = (tid % 4) * 4 + i;
        
        *(float*)((char*)s_Q_h0 + row * 16 * sizeof(float) + lane_c * sizeof(float)) = dQ_h0_acc[i];
        *(float*)((char*)s_Q_h1 + row * 16 * sizeof(float) + lane_c * sizeof(float)) = dQ_h1_acc[i];
    }

    __syncthreads();
    fence_proxy_async_shared_fn();

    if (elect_one_sync_fn()) {
        tma_store_3d_fn(&tma_dQ, s_Q_h0, 0, k_blk * 64, bh);
        tma_store_3d_fn(&tma_dQ, s_Q_h1, 64, k_blk * 64, bh);
        tma_store_commit_fn();
    }
    tma_store_wait_fn<0>();
    __syncthreads();
}

__global__ __launch_bounds__(128) void mha_bwd_dkv_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_dO,
    const __grid_constant__ CUtensorMap tma_dK,
    const __grid_constant__ CUtensorMap tma_dV,
    const float* L,
    int32_t S) 
{
    int32_t j_blk = blockIdx.x;
    int32_t bh = blockIdx.y;
    int32_t num_blocks = (S + 63) / 64;
    float scale = 1.0f / sqrtf(128.0f);
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

        for (int k_block = 0; k_block < 4; ++k_block) {
            __nv_bfloat16* q_ptr = (k_block < 4) ? (__nv_bfloat16*)s_Q_h0 : (__nv_bfloat16*)s_Q_h1;
            __nv_bfloat16* k_ptr = (k_block < 4) ? (__nv_bfloat16*)s_K_h0 : (__nv_bfloat16*)s_K_h1;
            
            typename GEMM_64x64x16_cg::Gemm gemm1{s_S_acc, q_ptr, k_ptr, 
                cutlass::gemm::GemmCoord(64, 64, 16), 64, 64, 64};
            gemm1(k_block * 16, 0, k_block * 16);
        }
        
        for (int k_block = 0; k_block < 4; ++k_block) {
            __nv_bfloat16* q_ptr = (k_block < 4) ? (__nv_bfloat16*)s_Q_h1 : (__nv_bfloat16*)s_Q_h0;
            __nv_bfloat16* k_ptr = (k_block < 4) ? (__nv_bfloat16*)s_K_h1 : (__nv_bfloat16*)s_K_h0;
            
            typename GEMM_64x64x16_cg::Gemm gemm1{s_S_acc, q_ptr, k_ptr, 
                cutlass::gemm::GemmCoord(64, 64, 16), 64, 64, 64};
            gemm1(k_block * 16 + 64, 0, k_block * 16);
        }
        
        __syncthreads();

        __nv_bfloat16* s_P_bf16 = (__nv_bfloat16*)s_P;

        for (int i = 0; i < 4; ++i) {
            int row = tid / 4;
            int lane_c = (tid % 4) * 4 + i;
            float s = ((float*)s_S_acc)[row * 16 + lane_c] * scale;
            
            int idx = ((tid * 2) + i) % 64;
            int idy = (((tid * 2) + i) / 64) * 2 + (i >= 2 ? 1 : 0);
            
            float l_val = L[bh * S + (k_blk * 64) + idx];
            float p = __isnanf(l_val) ? 0.0f : expf(s - l_val);
            
            int global_row = k_blk * 64 + idy;
            int global_col = j_blk * 64 + idx;
            if (global_row >= S || global_col >= S) {
                p = 0.0f;
            }
            
            s_P_bf16[row * 64 + swizzle_128B(row, lane_c)] = __float2bfloat16(p);
        }
        __syncthreads();
        fence_proxy_async_shared_fn();

        for (int i = tid; i < 4096; i += blockDim.x) *(float*)((char*)s_dP_acc + i * sizeof(float)) = 0.0f;

        for (int k_block = 0; k_block < 4; ++k_block) {
            __nv_bfloat16* do_ptr = (k_block < 4) ? (__nv_bfloat16*)s_dO_h0 : (__nv_bfloat16*)s_dO_h1;
            __nv_bfloat16* v_ptr = (k_block < 4) ? (__nv_bfloat16*)s_V_h0 : (__nv_bfloat16*)s_V_h1;
            
            typename GEMM_64x64x16_cg::Gemm gemm2{s_dP_acc, do_ptr, v_ptr, 
                cutlass::gemm::GemmCoord(64, 64, 16), 64, 64, 64};
            gemm2(k_block * 16, 0, k_block * 16);
        }
        for (int k_block = 0; k_block < 4; ++k_block) {
            __nv_bfloat16* do_ptr = (k_block < 4) ? (__nv_bfloat16*)s_dO_h1 : (__nv_bfloat16*)s_dO_h0;
            __nv_bfloat16* v_ptr = (k_block < 4) ? (__nv_bfloat16*)s_V_h1 : (__nv_bfloat16*)s_V_h0;
            
            typename GEMM_64x64x16_cg::Gemm gemm2{s_dP_acc, do_ptr, v_ptr, 
                cutlass::gemm::GemmCoord(64, 64, 16), 64, 64, 64};
            gemm2(k_block * 16 + 64, 0, k_block * 16);
        }
        __syncthreads();

        for (int i = 0; i < 4; ++i) {
            int row = tid / 4;
            int lane_c = (tid % 4) * 4 + i;
            float dp = ((float*)s_dP_acc)[row * 16 + lane_c];
            
            int idx = ((tid * 2) + i) % 64;
            int idy = (((tid * 2) + i) / 64) * 2 + (i >= 2 ? 1 : 0);
            int global_row = k_blk * 64 + idy;
            int global_col = j_blk * 64 + idx;
            if (global_row >= S || global_col >= S) dp = 0.0f;
            
            float p = __bfloat162float(s_P_bf16[row * 64 + swizzle_128B(row, lane_c)]);
            
            float ds = dp * p * scale;
            *(__nv_bfloat16*)((char*)s_D_s + row * 128 + swizzle_128B(row, lane_c) * sizeof(__nv_bfloat16)) = __float2bfloat16(ds);
        }
        __syncthreads();
        fence_proxy_async_shared_fn();

        for (int k_block = 0; k_block < 4; ++k_block) {
            __nv_bfloat16* ds_ptr = (__nv_bfloat16*)s_D_s;
            __nv_bfloat16* q_ptr = (k_block < 4) ? (__nv_bfloat16*)s_Q_h0 : (__nv_bfloat16*)s_Q_h1;
            
            typename GEMM_64x64x64_ct::Gemm gemm4_0{dK_h0_acc, ds_ptr, q_ptr};
            gemm4_0(k_block * 16, 0, k_block * 16, 0);
            
            typename GEMM_64x64x64_ct::Gemm gemm4_1{dK_h1_acc, ds_ptr, q_ptr};
            gemm4_1(k_block * 16, 64, k_block * 16, 0);
        }

        for (int k_block = 0; k_block < 4; ++k_block) {
            __nv_bfloat16* p_ptr = (__nv_bfloat16*)s_P;
            __nv_bfloat16* do_ptr = (k_block < 4) ? (__nv_bfloat16*)s_dO_h0 : (__nv_bfloat16*)s_dO_h1;
            
            typename GEMM_64x64x64_ct::Gemm gemm5_0{dV_h0_acc, p_ptr, do_ptr};
            gemm5_0(k_block * 16, 0, k_block * 16, 0);
            
            typename GEMM_64x64x64_ct::Gemm gemm5_1{dV_h1_acc, p_ptr, do_ptr};
            gemm5_1(k_block * 16, 64, k_block * 16, 0);
        }
        
        __syncthreads();
        phase_inner ^= 1;
    }

    for (int i = 0; i < 4; ++i) {
        int row = tid / 4;
        int lane_c = (tid % 4) * 4 + i;
        
        *(float*)((char*)s_K_h0 + row * 16 * sizeof(float) + lane_c * sizeof(float)) = dK_h0_acc[i];
        *(float*)((char*)s_K_h1 + row * 16 * sizeof(float) + lane_c * sizeof(float)) = dK_h1_acc[i];
        
        *(float*)((char*)s_V_h0 + row * 16 * sizeof(float) + lane_c * sizeof(float)) = dV_h0_acc[i];
        *(float*)((char*)s_V_h1 + row * 16 * sizeof(float) + lane_c * sizeof(float)) = dV_h1_acc[i];
    }

    __syncthreads();
    fence_proxy_async_shared_fn();

    if (elect_one_sync_fn()) {
        tma_store_3d_fn(&tma_dK, s_K_h0, 0, j_blk * 64, bh);
        tma_store_3d_fn(&tma_dK, s_K_h1, 64, j_blk * 64, bh);
        tma_store_3d_fn(&tma_dV, s_V_h0, 0, j_blk * 64, bh);
        tma_store_3d_fn(&tma_dV, s_V_h1, 64, j_blk * 64, bh);
        tma_store_commit_fn();
    }
    tma_store_wait_fn<0>();
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
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_dO, tma_dQ, tma_dK, tma_dV;
    
    DRV_CHECK(create_tma_3d_descriptor_2B(&tma_Q, Q_ptr, d, S, B * H, 64, 64, 1, CU_TENSOR_MAP_SWIZZLE_128B));
    DRV_CHECK(create_tma_3d_descriptor_2B(&tma_K, K_ptr, d, S, B * H, 64, 64, 1, CU_TENSOR_MAP_SWIZZLE_128B));
    DRV_CHECK(create_tma_3d_descriptor_2B(&tma_V, V_ptr, d, S, B * H, 64, 64, 1, CU_TENSOR_MAP_SWIZZLE_128B));
    DRV_CHECK(create_tma_3d_descriptor_2B(&tma_dO, dO_ptr, d, S, B * H, 64, 64, 1, CU_TENSOR_MAP_SWIZZLE_128B));
    DRV_CHECK(create_tma_3d_descriptor_2B(&tma_dQ, dQ_ptr, d, S, B * H, 64, 64, 1, CU_TENSOR_MAP_SWIZZLE_128B));
    DRV_CHECK(create_tma_3d_descriptor_2B(&tma_dK, dK_ptr, d, S, B * H, 64, 64, 1, CU_TENSOR_MAP_SWIZZLE_128B));
    DRV_CHECK(create_tma_3d_descriptor_2B(&tma_dV, dV_ptr, d, S, B * H, 64, 64, 1, CU_TENSOR_MAP_SWIZZLE_128B));
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    int64_t threads = 128;
    int64_t blocks_x = (S + 63) / 64;
    int64_t blocks_y = B * H;
    dim3 grid(blocks_x, blocks_y);
    
    int smem_size = 90 * 1024;
    CUDA_CHECK(cudaFuncSetAttribute((const void*)mha_bwd_dq_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    CUDA_CHECK(cudaFuncSetAttribute((const void*)mha_bwd_dkv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    mha_bwd_dq_kernel<<<grid, threads, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, tma_dO, tma_dQ, L_data, S
    );
    CUDA_CHECK(cudaGetLastError());
    
    mha_bwd_dkv_kernel<<<grid, threads, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, tma_dO, tma_dK, tma_dV, L_data, S
    );
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

} // namespace tvm_ffi_example_cuda