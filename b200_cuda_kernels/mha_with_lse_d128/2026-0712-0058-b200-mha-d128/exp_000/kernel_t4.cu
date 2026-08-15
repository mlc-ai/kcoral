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

#define CU_CHECK(call) do {                                      \
    CUresult _e = (call);                                        \
    if (_e != CUDA_SUCCESS) {                                    \
        fprintf(stderr, "CU error %d at %s:%d\n",               \
                (int)_e, __FILE__, __LINE__);                    \
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

__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

__device__ __forceinline__ void cluster_sync_fn() {
    asm volatile("barrier.cluster.arrive;\nbarrier.cluster.wait;\n" ::: "memory");
}

__device__ __forceinline__ float fast_exp2f_fn(float x, float log2e) {
    x *= log2e;
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
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

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
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

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tma_load_4d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
       :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
       :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void umma_f16_cg2_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_f16_cg2_scaled_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, float scale) {
    asm volatile(
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, 1, %4;\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "f"(scale));
}

__device__ __forceinline__ void umma_commit_2sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::2"
        ".mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
        " [%0], %1;"
        :: "r"(a), "h"((uint16_t)0x3));
}

__device__ __forceinline__ uint64_t make_smem_desc_swizzled(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;   
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_swizzled_k_slice(void* smem_ptr, uint32_t slice) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr) + slice * 32;
    uint64_t d = (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((1 & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((1024 & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_swizzled_n_slice(void* smem_ptr, uint32_t slice) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr) + slice * 1024;
    uint64_t d = (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((8192 & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((1024 & 0x3FFFF) >> 4) << 32;
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

__device__ __forceinline__ uint32_t make_instr_desc_transposed(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);     
    d |= (1u << 7);     
    d |= (1u << 10);    
    d |= (0u << 15);    
    d |= (1u << 16);    
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

CUresult create_tma_4d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t dim0, uint64_t dim1, uint64_t dim2, uint64_t dim3, uint32_t box0, uint32_t box1, uint32_t box2, uint32_t box3, CUtensorMapSwizzle swizzle) {
    cuuint64_t globalDim[4] = {dim0, dim1, dim2, dim3};
    cuuint64_t globalStrides[3] = {dim0 * 2, dim0 * dim1 * 2, dim0 * dim1 * dim2 * 2};
    cuuint32_t boxDim[4] = {box0, box1, box2, box3};
    cuuint32_t elementStrides[4] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, globalAddress, globalDim,
        globalStrides, boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

__global__ void __launch_bounds__(128) attention_kernel_swizzled(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    uint32_t S, uint32_t H, uint32_t B, 
    __nv_bfloat16* O, float* LSE) 
{
    uint32_t bh = blockIdx.y;         
    uint32_t sq_block = blockIdx.x;   
    
    extern __shared__ __align__(1024) uint8_t smem_pool[];
    uint8_t* s_Q0 = smem_pool;              
    uint8_t* s_Q1 = smem_pool + 8192;       
    uint8_t* s_K0 = smem_pool + 16384;      
    uint8_t* s_K1 = smem_pool + 24576;      
    uint8_t* s_V0 = smem_pool + 32768;      
    uint8_t* s_V1 = smem_pool + 40960;      
    uint8_t* s_P  = smem_pool + 49152;      
    uint8_t* s_P_0 = smem_pool + 57344;     
    uint8_t* s_P_1 = smem_pool + 65536;     
    
    uint64_t* bar_Q = (uint64_t*)(smem_pool + 65600);
    uint64_t* bar_K = (uint64_t*)(smem_pool + 65608);
    uint64_t* bar_V = (uint64_t*)(smem_pool + 65616);
    uint32_t* tmem_S = (uint32_t*)(smem_pool + 65624);
    uint32_t* tmem_P = (uint32_t*)(smem_pool + 65628);
    uint32_t* tmem_O_0 = (uint32_t*)(smem_pool + 65632);
    uint32_t* tmem_O_1 = (uint32_t*)(smem_pool + 65636);

    const float log2e = 1.4426950408889634f;
    const float HEAD_SCALE = 1.0f / sqrtf(128.0f);

    uint32_t cta_offset = (cluster_rank_fn() % 2) * 64;

    if (threadIdx.x == 0) {
        tmem_alloc_fn(tmem_S, 128);
        tmem_alloc_fn(tmem_P, 128);
        tmem_alloc_fn(tmem_O_0, 128);
        tmem_alloc_fn(tmem_O_1, 128);
        tmem_alloc_fn((uint32_t*)(smem_pool + 65640), 128);
        tmem_alloc_fn((uint32_t*)(smem_pool + 65644), 128);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(bar_Q, 1);
        init_smem_barrier_fn(bar_K, 1);
        init_smem_barrier_fn(bar_V, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    uint32_t tmem_S_base = *tmem_S;
    uint32_t tmem_P_base = *tmem_P;
    uint32_t tmem_O_0_base = *tmem_O_0;
    uint32_t tmem_O_1_base = *tmem_O_1;
    uint32_t tmem_P_0_base = *(uint32_t*)(smem_pool + 65640);
    uint32_t tmem_P_1_base = *(uint32_t*)(smem_pool + 65644);

    int h_idx = bh % H;
    int b_idx = bh / H;

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar_Q, 16384);
        tma_load_4d_fn(&tma_Q, bar_Q, s_Q0, 0, sq_block * 128 + cta_offset, h_idx, b_idx);
        tma_load_4d_fn(&tma_Q, bar_Q, s_Q1, 64, sq_block * 128 + cta_offset, h_idx, b_idx);
    }
    
    mbarrier_wait_fn(bar_Q, 0);

    float l_sum[2] = {0, 0};
    float m_global_fast[2] = {-1e20f, -1e20f};
    uint32_t phase_K = 0, phase_V = 0;

    uint32_t idesc_QK_0 = make_instr_desc_fn(128, 64);
    uint32_t idesc_QK_1 = make_instr_desc_fn(128, 64);
    uint32_t idesc_PV_0 = make_instr_desc_transposed(128, 64);
    uint32_t idesc_PV_1 = make_instr_desc_transposed(128, 64);

    for (uint32_t kv_iter = 0; kv_iter < (S + 63) / 64; ++kv_iter) {
        uint32_t kv_block = kv_iter * 64;
        
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_K, 16384);
            tma_load_4d_fn(&tma_K, bar_K, s_K0, 0, kv_block, h_idx, b_idx);
            tma_load_4d_fn(&tma_K, bar_K, s_K1, 64, kv_block, h_idx, b_idx);

            mbarrier_arrive_and_expect_tx_fn(bar_V, 16384);
            tma_load_4d_fn(&tma_V, bar_V, s_V0, 0, kv_block, h_idx, b_idx);
            tma_load_4d_fn(&tma_V, bar_V, s_V1, 64, kv_block, h_idx, b_idx);
        }
        
        mbarrier_wait_fn(bar_K, phase_K);
        mbarrier_wait_fn(bar_V, phase_V);
        
        if (threadIdx.x < 32) {
            asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {0,0,0,0};" ::: "r"(tmem_S_base), "r"(tmem_S_base + 4), "r"(tmem_S_base + 8), "r"(tmem_S_base + 12));
            asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {0,0,0,0};" ::: "r"(tmem_S_base + 16), "r"(tmem_S_base + 20), "r"(tmem_S_base + 24), "r"(tmem_S_base + 28));
            asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {0,0,0,0};" ::: "r"(tmem_S_base + 32), "r"(tmem_S_base + 36), "r"(tmem_S_base + 40), "r"(tmem_S_base + 44));
            asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {0,0,0,0};" ::: "r"(tmem_S_base + 48), "r"(tmem_S_base + 52), "r"(tmem_S_base + 56), "r"(tmem_S_base + 60));
            asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {0,0,0,0};" ::: "r"(tmem_S_base + 64), "r"(tmem_S_base + 68), "r"(tmem_S_base + 72), "r"(tmem_S_base + 76));
            asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {0,0,0,0};" ::: "r"(tmem_S_base + 80), "r"(tmem_S_base + 84), "r"(tmem_S_base + 88), "r"(tmem_S_base + 92));
            asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {0,0,0,0};" ::: "r"(tmem_S_base + 96), "r"(tmem_S_base + 100), "r"(tmem_S_base + 104), "r"(tmem_S_base + 108));
            asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {0,0,0,0};" ::: "r"(tmem_S_base + 112), "r"(tmem_S_base + 116), "r"(tmem_S_base + 120), "r"(tmem_S_base + 124));
        }
        fence_proxy_async_shared_fn();
        
        for (uint32_t k_chunk = 0; k_chunk < 8; ++k_chunk) {
            uint64_t desc_A = make_smem_desc_swizzled_k_slice(s_Q0, k_chunk);
            uint64_t desc_B = make_smem_desc_swizzled_k_slice(s_K0, k_chunk);
            if (k_chunk == 0) { umma_f16_cg2_fn(tmem_S_base, desc_A, desc_B, idesc_QK_0, 0); }
            else              { umma_f16_cg2_fn(tmem_S_base, desc_A, desc_B, idesc_QK_0, 1); }
        }
        for (uint32_t k_chunk = 0; k_chunk < 8; ++k_chunk) {
            uint64_t desc_A = make_smem_desc_swizzled_k_slice(s_Q1, k_chunk);
            uint64_t desc_B = make_smem_desc_swizzled_k_slice(s_K1, k_chunk);
            umma_f16_cg2_fn(tmem_S_base, desc_A, desc_B, idesc_QK_1, 1);
        }
        umma_commit_2sm_fn(bar_K);
        mbarrier_wait_fn(bar_K, phase_K);
        
        float my_max[2] = {-1e20f, -1e20f};
        float regs_p[8];
        for (uint32_t col = 0; col < 64; col += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(tmem_S_base + col));
            tmem_load_fence_fn();
            
            float vals[4] = {__uint_as_float(r0), __uint_as_float(r1), __uint_as_float(r2), __uint_as_float(r3)};
            for(int i = 0; i < 4; i++) {
                uint32_t global_sq_idx = sq_block * 128 + cta_offset + threadIdx.x;
                uint32_t global_kv_idx = kv_block + col + i;
                if (global_kv_idx < S && global_sq_idx < S) {
                    vals[i] *= HEAD_SCALE;
                    my_max[i/2] = fmaxf(my_max[i/2], vals[i]);
                } else {
                    vals[i] = -1e20f;
                }
            }
            regs_p[col] = vals[0]; regs_p[col+1] = vals[1];
            regs_p[col+2] = vals[2]; regs_p[col+3] = vals[3];
        }
        
        for (uint32_t offset = 1; offset < 32; offset *= 2) {
            my_max[0] = fmaxf(my_max[0], __shfl_xor_sync(0xFFFFFFFF, my_max[0], offset));
            my_max[1] = fmaxf(my_max[1], __shfl_xor_sync(0xFFFFFFFF, my_max[1], offset));
        }
        
        float my_sum[2] = {0, 0};
        for(int i = 0; i < 4; i++) {
            float val_f = fast_exp2f_fn(regs_p[i] - my_max[i/2], log2e);
            my_sum[i/2] += val_f;
            regs_p[i] = val_f;
        }
        for(int i = 0; i < 4; i++) {
            float val_f = fast_exp2f_fn(regs_p[i+4] - my_max[i/2], log2e);
            my_sum[i/2] += val_f;
            regs_p[i+4] = val_f;
        }
        
        for (uint32_t offset = 1; offset < 32; offset *= 2) {
            my_sum[0] += __shfl_xor_sync(0xFFFFFFFF, my_sum[0], offset);
            my_sum[1] += __shfl_xor_sync(0xFFFFFFFF, my_sum[1], offset);
        }
        
        float m_new[2] = {fmaxf(m_global_fast[0], my_max[0]), fmaxf(m_global_fast[1], my_max[1])};
        float m_scale2[2] = {fast_exp2f_fn(m_global_fast[0] - m_new[0], log2e), fast_exp2f_fn(m_global_fast[1] - m_new[1], log2e)};
        
        l_sum[0] = l_sum[0] * m_scale2[0] + my_sum[0];
        l_sum[1] = l_sum[1] * m_scale2[1] + my_sum[1];
        
        m_global_fast[0] = m_new[0];
        m_global_fast[1] = m_new[1];
        
        for(int i = 0; i < 4; i++) {
            regs_p[i] *= m_scale2[i/2];
        }
        for(int i = 0; i < 4; i++) {
            regs_p[i+4] *= m_scale2[i/2];
        }

        __syncthreads();
        fence_proxy_async_shared_fn();

        for (int i = 0; i < 4; i+=2) {
            uint32_t p0 = pack_bf16_fn(__float_as_uint(regs_p[i]), __float_as_uint(regs_p[i+1]));
            uint32_t row = threadIdx.x;
            uint32_t chunk = (col + i) / 8;
            uint32_t rem = (col + i) % 8;
            uint32_t swizzled_col = (chunk ^ (row % 8)) * 8 + rem;
            ((__nv_bfloat16*)s_P_0)[row * 64 + swizzled_col] = *reinterpret_cast<uint16_t*>(&p0);
        }
        for (int i = 0; i < 4; i+=2) {
            uint32_t p0 = pack_bf16_fn(__float_as_uint(regs_p[i+4]), __float_as_uint(regs_p[i+5]));
            uint32_t row = threadIdx.x;
            uint32_t chunk = (col + i) / 8;
            uint32_t rem = (col + i) % 8;
            uint32_t swizzled_col = (chunk ^ (row % 8)) * 8 + rem;
            ((__nv_bfloat16*)s_P_1)[row * 64 + swizzled_col] = *reinterpret_cast<uint16_t*>(&p0);
        }
        
        __syncthreads(); 
        
        for (uint32_t i = 0; i < 8; ++i) {
            uint64_t desc_A_0 = make_smem_desc_swizzled_k_slice(s_P_0, i);
            uint64_t desc_B_0 = make_smem_desc_swizzled_n_slice(s_V0, i);
            
            if (i == 0) { umma_f16_cg2_scaled_fn(tmem_O_0_base, desc_A_0, desc_B_0, idesc_PV_0, m_scale2[0]); }
            else        { umma_f16_cg2_fn(tmem_O_0_base, desc_A_0, desc_B_0, idesc_PV_0, 1); }
        }
        for (uint32_t i = 0; i < 8; ++i) {
            uint64_t desc_A_1 = make_smem_desc_swizzled_k_slice(s_P_1, i);
            uint64_t desc_B_1 = make_smem_desc_swizzled_n_slice(s_V1, i);
            
            if (i == 0) { umma_f16_cg2_scaled_fn(tmem_O_1_base, desc_A_1, desc_B_1, idesc_PV_1, m_scale2[1]); }
            else        { umma_f16_cg2_fn(tmem_O_1_base, desc_A_1, desc_B_1, idesc_PV_1, 1); }
        }
        umma_commit_2sm_fn(bar_V);
        mbarrier_wait_fn(bar_V, phase_V);
        
        phase_K ^= 1;
        phase_V ^= 1;
    }
    
    tmem_load_fence_fn();
    
    float final_l_sum[2];
    final_l_sum[0] = l_sum[0];
    final_l_sum[1] = l_sum[1];
    
    uint32_t b_off = (bh * S + (sq_block * 128 + cta_offset + threadIdx.x)) * 128;
    
    float O_reg_0[8], O_reg_1[8];
    for (uint32_t col = 0; col < 64; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(tmem_O_0_base + col));
        O_reg_0[col] = __uint_as_float(r0); O_reg_0[col+1] = __uint_as_float(r1);
        O_reg_0[col+2] = __uint_as_float(r2); O_reg_0[col+3] = __uint_as_float(r3);
        
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(tmem_O_1_base + col));
        O_reg_1[col] = __uint_as_float(r0); O_reg_1[col+1] = __uint_as_float(r1);
        O_reg_1[col+2] = __uint_as_float(r2); O_reg_1[col+3] = __uint_as_float(r3);
    }
    
    for (int i = 0; i < 8; i += 4) {
        uint32_t col = (i / 4) * 64 + (i % 4); 
        uint32_t global_row = sq_block * 128 + cta_offset + threadIdx.x;
        int r_idx = (i % 4) / 2; 
        
        float final_val_0 = O_reg_0[i] / final_l_sum[r_idx];
        float final_val_1 = O_reg_1[i] / final_l_sum[r_idx];
        
        if (global_row < S && col < 128) {
            O[b_off + global_row * 128 + col] = __float2bfloat16(final_val_0);
            O[b_off + global_row * 128 + col + 64] = __float2bfloat16(final_val_1);
        }
    }
    
    if (threadIdx.x < 2) {
        uint32_t global_row = sq_block * 128 + cta_offset + threadIdx.x;
        if (global_row < S) {
            int r_idx = threadIdx.x;
            LSE[bh * S + global_row] = m_global_fast[r_idx] + logf(final_l_sum[r_idx]);
        }
    }
    
    if (threadIdx.x == 0) {
        tmem_dealloc_fn(*tmem_S, 128);
        tmem_dealloc_fn(*tmem_P, 128);
        tmem_dealloc_fn(*tmem_O_0, 128);
        tmem_dealloc_fn(*tmem_O_1, 128);
        tmem_dealloc_fn(*(uint32_t*)(smem_pool + 65640), 128);
        tmem_dealloc_fn(*(uint32_t*)(smem_pool + 65644), 128);
    }
    cluster_sync_fn();
}

namespace tvm_ffi {

void run(TensorView Q, TensorView K, TensorView V, TensorView O, TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    uint32_t B = Q.size(0);
    uint32_t H = Q.size(1);
    uint32_t S = Q.size(2);
    uint32_t D = Q.size(3);

    CUtensorMap tma_Q, tma_K, tma_V;
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_Q, Q.data_ptr(), D, S, H, B, 64, 64, 1, 1, CU_TENSOR_MAP_SWIZZLE_128B));
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_K, K.data_ptr(), D, S, H, B, 64, 64, 1, 1, CU_TENSOR_MAP_SWIZZLE_128B));
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_V, V.data_ptr(), D, S, H, B, 64, 64, 1, 1, CU_TENSOR_MAP_SWIZZLE_128B));

    uint32_t num_sq_blocks = (S + 127) / 128;
    dim3 grid(num_sq_blocks, B * H);
    dim3 block(128);

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 65648;
    config.stream = stream;

    cudaLaunchAttribute attrs[2];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    
    attrs[1].id = cudaLaunchAttributeMaxDynamicSharedMemorySize;
    attrs[1].val.maxDynamicSharedMemorySize = 65648;
    
    config.attrs = attrs;
    config.numAttrs = 2;

    CUDA_CHECK(cudaLaunchKernelEx(&config, attention_kernel_swizzled,
        tma_Q, tma_K, tma_V, S, H, B, 
        static_cast<__nv_bfloat16*>(O.data_ptr()), static_cast<float*>(LSE.data_ptr())));
        
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi