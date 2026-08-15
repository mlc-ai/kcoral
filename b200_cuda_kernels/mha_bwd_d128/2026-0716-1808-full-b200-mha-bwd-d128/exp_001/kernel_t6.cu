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

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
   :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void tmma_64x64x16_accum(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t accum) {
    
    uint32_t idesc_val = 0;
    idesc_val |= (1u << 4);
    idesc_val |= (1u << 7);
    idesc_val |= (1u << 10);
    idesc_val |= (0u << 15);
    idesc_val |= (0u << 16); 
    idesc_val |= ((uint32_t)(64 / 8) << 17);
    idesc_val |= ((uint32_t)(64 / 16) << 24);

    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        : : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc_val), "r"(accum));
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

__device__ __forceinline__ uint64_t make_desc_k_major(void* smem_ptr) {
    return make_smem_desc_sm100_fn(smem_ptr, 1, 1024);
}
__device__ __forceinline__ uint64_t make_desc_mn_major(void* smem_ptr) {
    return make_smem_desc_sm100_fn(smem_ptr, 8192, 1024);
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
    __nv_bfloat16* dQ_gmem, const float* L, int32_t S, int32_t d) 
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
    
    uint64_t* bar_outer = (uint64_t*)(smem_buf + 10 * 8192);
    uint64_t* bar_inner = (uint64_t*)(smem_buf + 10 * 8192 + sizeof(uint64_t));

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(bar_outer, 1);
        init_smem_barrier_fn(bar_inner, 1);
    }
    __syncthreads();

    uint32_t tmem_S;
    if (threadIdx.x == 0) {
        tmem_alloc_fn(&tmem_S, 64);
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

    uint64_t desc_Q_h0 = make_desc_k_major(s_Q_h0);
    uint64_t desc_Q_h1 = make_desc_k_major(s_Q_h1);
    uint64_t desc_dO_h0 = make_desc_k_major(s_dO_h0);
    uint64_t desc_dO_h1 = make_desc_k_major(s_dO_h1);

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

        uint64_t desc_K_h0 = make_desc_k_major(s_K_h0);
        uint64_t desc_K_h1 = make_desc_k_major(s_K_h1);
        uint64_t desc_V_h0 = make_desc_k_major(s_V_h0);
        uint64_t desc_V_h1 = make_desc_k_major(s_V_h1);

        if (threadIdx.x == 0) {
            tmma_64x64x16_accum(tmem_S, desc_Q_h0, desc_K_h0, 0);
            tmma_64x64x16_accum(tmem_S + 2, desc_Q_h1, desc_K_h1, 1);
        }
        
        if (threadIdx.x == 0) {
            uint32_t tx = 4 * sizeof(float);
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared.b64 [%0];"
                :: "r"((uint32_t)__cvta_generic_to_shared(bar_inner)));
        }
        mbarrier_wait_fn(bar_inner, phase_inner);
        phase_inner ^= 1;
        __syncthreads();

        uint32_t r_S_raw[4];
        for (int i = 0; i < 4; ++i) {
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 %0, [%1];" 
                : "=r"(r_S_raw[i]) : "r"(tmem_S + tid + i * 4096));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        }
        float r_S[4];
        for (int i = 0; i < 4; ++i) r_S[i] = __uint_as_float(r_S_raw[i]);

        float r_dP[4];
        uint32_t off_P[4];
        for (int i = 0; i < 4; ++i) {
            int idx = ((tid * 2) + i) % 64;
            int idy = (((tid * 2) + i) / 64) * 2 + (i >= 2 ? 1 : 0);
            off_P[i] = (uint32_t)(idy * 64 + swizzle_128B(idy, idx));
            
            float l_val = L[bh * S + (j_blk * 64) + idx];
            float val = __isnanf(l_val) ? 0.0f : expf(r_S[i] * scale - l_val);
            
            int global_row = k_blk * 64 + idy;
            int global_col = j_blk * 64 + idx;
            if (global_row >= S || global_col >= S) val = 0.0f;
            
            *(__nv_bfloat16*)((char*)s_P + off_P[i] * sizeof(__nv_bfloat16)) = __float2bfloat16(val);
            r_dP[i] = val;
        }

        char* s_dP_acc = s_P;
        for (uint32_t i = threadIdx.x; i < 4096; i += blockDim.x) *(float*)((char*)s_dP_acc + i * sizeof(float)) = 0.0f;

        if (threadIdx.x == 0) {
            tmma_64x64x16_accum(tmem_S, desc_dO_h0, desc_V_h0, 0);
            tmma_64x64x16_accum(tmem_S + 2, desc_dO_h1, desc_V_h1, 1);
        }
        
        if (threadIdx.x == 0) {
            uint32_t tx = 4 * sizeof(float);
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared.b64 [%0];"
                :: "r"((uint32_t)__cvta_generic_to_shared(bar_inner)));
        }
        mbarrier_wait_fn(bar_inner, phase_inner);
        phase_inner ^= 1;
        __syncthreads();

        for (int i = 0; i < 4; ++i) {
            int idx = ((tid * 2) + i) % 64;
            int idy = (((tid * 2) + i) / 64) * 2 + (i >= 2 ? 1 : 0);
            int global_row = k_blk * 64 + idy;
            int global_col = j_blk * 64 + idx;
            if (global_row >= S || global_col >= S) r_dP[i] = 0.0f;
            
            float p = __bfloat162float(*(__nv_bfloat16*)((char*)s_P + off_P[i] * sizeof(__nv_bfloat16)));
            float ds = r_dP[i] * p * scale;
            
            uint32_t off_Ds = (uint32_t)(idy * 64 + swizzle_128B(idy, idx));
            *(__nv_bfloat16*)((char*)s_D_s + off_Ds * sizeof(__nv_bfloat16)) = __float2bfloat16(ds);
        }
        __syncthreads();
        fence_proxy_async_shared_fn();

        uint64_t desc_Ds = make_desc_k_major(s_D_s);
        uint64_t desc_K_h0_mn = make_desc_mn_major(s_K_h0);
        uint64_t desc_K_h1_mn = make_desc_mn_major(s_K_h1);

        if (threadIdx.x == 0) {
            tmma_64x64x16_accum(tmem_S, desc_Ds, desc_K_h0_mn, 0);
            tmma_64x64x16_accum(tmem_S + 2, desc_Ds, desc_K_h1_mn, 1);
        }
        
        if (threadIdx.x == 0) {
            uint32_t tx = 4 * sizeof(float);
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared.b64 [%0];"
                :: "r"((uint32_t)__cvta_generic_to_shared(bar_inner)));
        }
        mbarrier_wait_fn(bar_inner, phase_inner);
        phase_inner ^= 1;
        __syncthreads();
        
        char* s_dQ_h0 = s_Q_h0;
        char* s_dQ_h1 = s_Q_h1;
        for (int i = 0; i < 4; ++i) {
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 %0, [%1];" 
                : "=r"(r_S_raw[i]) : "r"(tmem_S + tid + i * 4096));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            float val = __uint_as_float(r_S_raw[i]);
            int idx = ((tid * 2) + i) % 64;
            int idy = (((tid * 2) + i) / 64) * 2 + (i >= 2 ? 1 : 0);
            int swizzled_col = swizzle_128B(idy, idx);
            *(__nv_bfloat16*)((char*)s_dQ_h0 + idy * 64 + swizzled_col) = __float2bfloat16(val);
            *(__nv_bfloat16*)((char*)s_dQ_h1 + idy * 64 + swizzled_col) = __float2bfloat16(val);
        }
    }

    __syncthreads();
    fence_proxy_async_shared_fn();

    for (int i = tid; i < 4096; i += blockDim.x) {
        int row = i / 64;
        int col = i % 64;
        int g_row = k_blk * 64 + row;
        if (g_row < S) {
            int swizzled_col = swizzle_128B(row, col);
            __nv_bfloat16 val_h0 = *(__nv_bfloat16*)((char*)s_Q_h0 + row * 64 + swizzled_col);
            __nv_bfloat16 val_h1 = *(__nv_bfloat16*)((char*)s_Q_h1 + row * 64 + swizzled_col);
            
            int g_col_h0 = col;
            int g_col_h1 = col + 64;
            
            if (g_col_h0 < d) dQ_gmem[bh * S * d + g_row * d + g_col_h0] = val_h0;
            if (g_col_h1 < d) dQ_gmem[bh * S * d + g_row * d + g_col_h1] = val_h1;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        tmem_dealloc_fn(tmem_S, 64);
    }
}

__global__ __launch_bounds__(128) void mha_bwd_dkv_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_dO,
    __nv_bfloat16* dK_gmem, __nv_bfloat16* dV_gmem, const float* L, int32_t S, int32_t d) 
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
    
    uint64_t* bar_outer = (uint64_t*)(smem_buf + 10 * 8192);
    uint64_t* bar_inner = (uint64_t*)(smem_buf + 10 * 8192 + sizeof(uint64_t));

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(bar_outer, 1);
        init_smem_barrier_fn(bar_inner, 1);
    }
    __syncthreads();

    uint32_t tmem_S;
    if (threadIdx.x == 0) {
        tmem_alloc_fn(&tmem_S, 64);
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

    uint64_t desc_K_h0 = make_desc_k_major(s_K_h0);
    uint64_t desc_K_h1 = make_desc_k_major(s_K_h1);
    uint64_t desc_V_h0 = make_desc_k_major(s_V_h0);
    uint64_t desc_V_h1 = make_desc_k_major(s_V_h1);

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

        uint64_t desc_Q_h0 = make_desc_k_major(s_Q_h0);
        uint64_t desc_Q_h1 = make_desc_k_major(s_Q_h1);
        uint64_t desc_dO_h0 = make_desc_k_major(s_dO_h0);
        uint64_t desc_dO_h1 = make_desc_k_major(s_dO_h1);

        if (threadIdx.x == 0) {
            tmma_64x64x16_accum(tmem_S, desc_Q_h0, desc_K_h0, 0);
            tmma_64x64x16_accum(tmem_S + 2, desc_Q_h1, desc_K_h1, 1);
        }
        
        if (threadIdx.x == 0) {
            uint32_t tx = 4 * sizeof(float);
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared.b64 [%0];"
                :: "r"((uint32_t)__cvta_generic_to_shared(bar_inner)));
        }
        mbarrier_wait_fn(bar_inner, phase_inner);
        phase_inner ^= 1;
        __syncthreads();

        uint32_t r_S_raw[4];
        for (int i = 0; i < 4; ++i) {
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 %0, [%1];" 
                : "=r"(r_S_raw[i]) : "r"(tmem_S + tid + i * 4096));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        }
        float r_S[4];
        for (int i = 0; i < 4; ++i) r_S[i] = __uint_as_float(r_S_raw[i]);

        float r_dP[4];
        uint32_t off_P[4];
        for (int i = 0; i < 4; ++i) {
            int idx = ((tid * 2) + i) % 64;
            int idy = (((tid * 2) + i) / 64) * 2 + (i >= 2 ? 1 : 0);
            off_P[i] = (uint32_t)(idy * 64 + swizzle_128B(idy, idx));
            
            float l_val = L[bh * S + (k_blk * 64) + idx];
            float val = __isnanf(l_val) ? 0.0f : expf(r_S[i] * scale - l_val);
            
            int global_row = k_blk * 64 + idy;
            int global_col = j_blk * 64 + idx;
            if (global_row >= S || global_col >= S) val = 0.0f;
            
            *(__nv_bfloat16*)((char*)s_P + off_P[i] * sizeof(__nv_bfloat16)) = __float2bfloat16(val);
            r_dP[i] = val;
        }

        char* s_dP_acc = s_P;
        for (uint32_t i = threadIdx.x; i < 4096; i += blockDim.x) *(float*)((char*)s_dP_acc + i * sizeof(float)) = 0.0f;

        if (threadIdx.x == 0) {
            tmma_64x64x16_accum(tmem_S, desc_dO_h0, desc_V_h0, 0);
            tmma_64x64x16_accum(tmem_S + 2, desc_dO_h1, desc_V_h1, 1);
        }
        
        if (threadIdx.x == 0) {
            uint32_t tx = 4 * sizeof(float);
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared.b64 [%0];"
                :: "r"((uint32_t)__cvta_generic_to_shared(bar_inner)));
        }
        mbarrier_wait_fn(bar_inner, phase_inner);
        phase_inner ^= 1;
        __syncthreads();

        for (int i = 0; i < 4; ++i) {
            int idx = ((tid * 2) + i) % 64;
            int idy = (((tid * 2) + i) / 64) * 2 + (i >= 2 ? 1 : 0);
            int global_row = k_blk * 64 + idy;
            int global_col = j_blk * 64 + idx;
            if (global_row >= S || global_col >= S) r_dP[i] = 0.0f;
            
            float p = __bfloat162float(*(__nv_bfloat16*)((char*)s_P + off_P[i] * sizeof(__nv_bfloat16)));
            float ds = r_dP[i] * p * scale;
            
            uint32_t off_Ds = (uint32_t)(idy * 64 + swizzle_128B(idy, idx));
            *(__nv_bfloat16*)((char*)s_D_s + off_Ds * sizeof(__nv_bfloat16)) = __float2bfloat16(ds);
        }
        __syncthreads();
        fence_proxy_async_shared_fn();

        uint64_t desc_Ds_mn = make_desc_mn_major(s_D_s);
        uint64_t desc_Q_h0_mn = make_desc_mn_major(s_Q_h0);
        uint64_t desc_Q_h1_mn = make_desc_mn_major(s_Q_h1);
        uint64_t desc_P_mn = make_desc_mn_major(s_P);
        uint64_t desc_dO_h0_mn = make_desc_mn_major(s_dO_h0);
        uint64_t desc_dO_h1_mn = make_desc_mn_major(s_dO_h1);

        if (threadIdx.x == 0) {
            tmma_64x64x16_accum(tmem_S, desc_Ds_mn, desc_Q_h0_mn, 0);
            tmma_64x64x16_accum(tmem_S + 2, desc_Ds_mn, desc_Q_h1_mn, 1);
        }
        if (threadIdx.x == 0) {
            tmma_64x64x16_accum(tmem_S + 4096, desc_P_mn, desc_dO_h0_mn, 0);
            tmma_64x64x16_accum(tmem_S + 4096 + 2, desc_P_mn, desc_dO_h1_mn, 1);
        }
        
        if (threadIdx.x == 0) {
            uint32_t tx = 8 * sizeof(float);
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared.b64 [%0];"
                :: "r"((uint32_t)__cvta_generic_to_shared(bar_inner)));
        }
        mbarrier_wait_fn(bar_inner, phase_inner);
        phase_inner ^= 1;
        __syncthreads();

        char* s_dK_h0 = s_K_h0;
        char* s_dK_h1 = s_K_h1;
        char* s_dV_h0 = s_V_h0;
        char* s_dV_h1 = s_V_h1;

        for (int i = 0; i < 4; ++i) {
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 %0, [%1];" 
                : "=r"(r_S_raw[i]) : "r"(tmem_S + tid + i * 4096));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            float val = __uint_as_float(r_S_raw[i]);
            int idx = ((tid * 2) + i) % 64;
            int idy = (((tid * 2) + i) / 64) * 2 + (i >= 2 ? 1 : 0);
            int swizzled_col = swizzle_128B(idy, idx);
            *(__nv_bfloat16*)((char*)s_dK_h0 + idy * 64 + swizzled_col) = __float2bfloat16(val);
            *(__nv_bfloat16*)((char*)s_dK_h1 + idy * 64 + swizzled_col) = __float2bfloat16(val);
        }
        
        for (int i = 0; i < 4; ++i) {
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 %0, [%1];" 
                : "=r"(r_S_raw[i]) : "r"(tmem_S + 4096 + tid + i * 4096));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            float val = __uint_as_float(r_S_raw[i]);
            int idx = ((tid * 2) + i) % 64;
            int idy = (((tid * 2) + i) / 64) * 2 + (i >= 2 ? 1 : 0);
            int swizzled_col = swizzle_128B(idy, idx);
            *(__nv_bfloat16*)((char*)s_dV_h0 + idy * 64 + swizzled_col) = __float2bfloat16(val);
            *(__nv_bfloat16*)((char*)s_dV_h1 + idy * 64 + swizzled_col) = __float2bfloat16(val);
        }
    }

    __syncthreads();
    fence_proxy_async_shared_fn();

    for (int i = tid; i < 4096; i += blockDim.x) {
        int row = i / 64;
        int col = i % 64;
        int g_row = j_blk * 64 + row;
        if (g_row < S) {
            int swizzled_col = swizzle_128B(row, col);
            __nv_bfloat16 dk0 = *(__nv_bfloat16*)((char*)s_K_h0 + row * 64 + swizzled_col);
            __nv_bfloat16 dk1 = *(__nv_bfloat16*)((char*)s_K_h1 + row * 64 + swizzled_col);
            __nv_bfloat16 dv0 = *(__nv_bfloat16*)((char*)s_V_h0 + row * 64 + swizzled_col);
            __nv_bfloat16 dv1 = *(__nv_bfloat16*)((char*)s_V_h1 + row * 64 + swizzled_col);
            
            int g_col_h0 = col;
            int g_col_h1 = col + 64;
            
            if (g_col_h0 < d) {
                dK_gmem[bh * S * d + g_row * d + g_col_h0] = dk0;
                dV_gmem[bh * S * d + g_row * d + g_col_h0] = dv0;
            }
            if (g_col_h1 < d) {
                dK_gmem[bh * S * d + g_row * d + g_col_h1] = dk1;
                dV_gmem[bh * S * d + g_row * d + g_col_h1] = dv1;
            }
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        tmem_dealloc_fn(tmem_S, 64);
    }
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
    
    int smem_size = 96 * 1024;
    CUDA_CHECK(cudaFuncSetAttribute((const void*)mha_bwd_dq_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    CUDA_CHECK(cudaFuncSetAttribute((const void*)mha_bwd_dkv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    mha_bwd_dq_kernel<<<grid, threads, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, tma_dO, dQ_ptr, L_data, S, d
    );
    CUDA_CHECK(cudaGetLastError());
    
    mha_bwd_dkv_kernel<<<grid, threads, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, tma_dO, dK_ptr, dV_ptr, L_data, S, d
    );
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

} // namespace tvm_ffi_example_cuda