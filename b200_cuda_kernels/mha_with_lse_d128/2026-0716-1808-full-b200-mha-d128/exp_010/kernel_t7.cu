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

#define CU_CHECK(call) do {                                        \
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        fprintf(stderr, "CU error %d at %s:%d\n",                 \
                _e, __FILE__, __LINE__);                           \
        exit(1);                                                   \
    }                                                              \
} while(0)


namespace tvm_ffi_optimized_cuda {

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

__device__ __forceinline__ void tma_load_2d_cta_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void add_base_offset_bytes(uint64_t* desc, uint32_t offset_bytes) {
    uint64_t d = *desc;
    uint32_t base = d & 0x3FFF;
    base += offset_bytes / 16;
    if ((base >> 14) != 0) {
        base &= 0x3FFF;
    }
    d &= ~0x3FFF;
    d |= base;
    *desc = d;
}

__device__ __forceinline__ uint64_t make_smem_desc_128B(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16; 
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32; 
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_mnmaj_128B(void* smem_ptr, void* pattern_start_ptr, uint32_t sbo, uint32_t lbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint32_t pattern_addr = (uint32_t)__cvta_generic_to_shared(pattern_start_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16; 
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32; 
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    uint32_t base_offset = (pattern_addr >> 7) & 0x7;
    d |= (uint64_t)base_offset << 49;
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= (0u << 15);   
    d |= (0u << 16);   
    d |= ((N >> 3) & 0x3F);     
    d |= ((M >> 4) & 0x1F) << 24;    
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn_trans_B(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= (0u << 15);   
    d |= (1u << 16);   
    d |= ((N >> 3) & 0x3F);     
    d |= ((M >> 4) & 0x1F) << 24;    
    return d;
}

__device__ __forceinline__ void umma_f16_cg2(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_2sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::2"
        ".mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
        " [%0], %1;"
        :: "r"(a), "h"((uint16_t)0x3));
}

__device__ __forceinline__ int swizzle_128B(int row, int col) {
    int c = col * 2;
    int x_int4 = c / 16;
    int swizzled_x = (row % 8) ^ x_int4;
    return (swizzled_x * 8) | (c % 8);
}

__device__ __forceinline__ __nv_bfloat16* get_swizzled_ptr(__nv_bfloat16* base, int row, int col) {
    return &base[(row * 64) + swizzle_128B(row, col)];
}

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        globalAddress,
        globalDim,
        globalStrides,
        boxDim, 
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle,
        l2Promotion,
        oobFill
    );
}

__global__ void __launch_bounds__(128) mha_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* gmem_O,
    float* gmem_LSE,
    uint32_t S)
{
    extern __shared__ __align__(16) char smem_pool[];
    
    __nv_bfloat16* smem_Q0 = (__nv_bfloat16*)smem_pool;                
    __nv_bfloat16* smem_Q1 = (__nv_bfloat16*)(smem_pool + 16384);  
    
    uintptr_t aligned_pool = ((uintptr_t)smem_pool + 1023) & ~1023;
    __nv_bfloat16* smem_K0_0 = (__nv_bfloat16*)aligned_pool;                
    __nv_bfloat16* smem_K0_1 = (__nv_bfloat16*)(aligned_pool + 8192);      
    __nv_bfloat16* smem_K1_0 = (__nv_bfloat16*)(aligned_pool + 16384);     
    __nv_bfloat16* smem_K1_1 = (__nv_bfloat16*)(aligned_pool + 24576);     
    __nv_bfloat16* smem_V0_0 = (__nv_bfloat16*)(aligned_pool + 32768);     
    __nv_bfloat16* smem_V0_1 = (__nv_bfloat16*)(aligned_pool + 40960);     
    __nv_bfloat16* smem_V1_0 = (__nv_bfloat16*)(aligned_pool + 49152);     
    __nv_bfloat16* smem_V1_1 = (__nv_bfloat16*)(aligned_pool + 57344);     
    
    __nv_bfloat16* smem_S = (__nv_bfloat16*)(smem_pool + 98304);
    
    uint64_t* mbar_Q = (uint64_t*)(smem_pool + 98304 + 8192);
    uint64_t* mbar_KV = (uint64_t*)(smem_pool + 98304 + 8192 + 8);

    uint32_t tid = threadIdx.x;
    uint32_t s_blk = blockIdx.y;
    uint32_t bh = blockIdx.x;
    uint32_t row_base = s_blk * 128;

    if (tid == 0) {
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(&mbar_KV[0], 1);
        init_smem_barrier_fn(&mbar_KV[1], 1);
    }
    __syncthreads();
    
    uint32_t tmem_base;
    if (tid == 0) {
        tmem_alloc_fn(&tmem_base, 512); 
    }
    __syncthreads();
    
    uint32_t tmem_P = tmem_base + (cluster_rank_fn() & 1) * 4096;
    uint32_t tmem_O0 = tmem_P + 4096;
    uint32_t tmem_O1 = tmem_P + 8192;
    
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_Q, 32768);
        tma_load_2d_cta_fn(&tma_Q, mbar_Q, smem_Q0, 0, bh * S + row_base);
        tma_load_2d_cta_fn(&tma_Q, mbar_Q, smem_Q1, 64, bh * S + row_base);
        
        if (S > 0) {
            mbarrier_arrive_and_expect_tx_fn(&mbar_KV[0], 32768);
            tma_load_2d_cta_fn(&tma_K, &mbar_KV[0], smem_K0_0, 0, bh * S);
            tma_load_2d_cta_fn(&tma_K, &mbar_KV[0], smem_K1_0, 64, bh * S);
            tma_load_2d_cta_fn(&tma_V, &mbar_KV[0], smem_V0_0, 0, bh * S);
            tma_load_2d_cta_fn(&tma_V, &mbar_KV[0], smem_V1_0, 64, bh * S);
        }
    }
    mbarrier_wait_fn(mbar_Q, 0);
    fence_proxy_async_shared_fn();

    float global_max = -INFINITY;
    float global_sum = 0.0f;
    
    uint32_t phase[2] = {0, 0};
    uint32_t idx = 0;
    int cur_kv = 0;

    uint64_t desc_Q0 = make_smem_desc_128B(smem_Q0, 1, 1024);
    uint64_t desc_Q1 = make_smem_desc_128B(smem_Q1, 1, 1024);
    uint64_t desc_S = make_smem_desc_128B(smem_S, 1, 1024);
    
    uint32_t idesc_QK = make_instr_desc_fn(64, 64);
    uint32_t idesc_SV = make_instr_desc_fn_trans_B(64, 64);

    for (; idx < S/64; idx++) {
        mbarrier_wait_fn(&mbar_KV[cur_kv], phase[cur_kv]);
        fence_proxy_async_shared_fn();
        phase[cur_kv] ^= 1;
        
        if (idx + 1 < S/64) {
            if (tid == 0) {
                mbarrier_arrive_and_expect_tx_fn(&mbar_KV[cur_kv ^ 1], 32768);
                tma_load_2d_cta_fn(&tma_K, &mbar_KV[cur_kv ^ 1], (cur_kv == 0 ? smem_K0_1 : smem_K0_0), 0, bh * S + (idx + 1) * 64);
                tma_load_2d_cta_fn(&tma_K, &mbar_KV[cur_kv ^ 1], (cur_kv == 0 ? smem_K1_1 : smem_K1_0), 64, bh * S + (idx + 1) * 64);
                tma_load_2d_cta_fn(&tma_V, &mbar_KV[cur_kv ^ 1], (cur_kv == 0 ? smem_V0_1 : smem_V0_0), 0, bh * S + (idx + 1) * 64);
                tma_load_2d_cta_fn(&tma_V, &mbar_KV[cur_kv ^ 1], (cur_kv == 0 ? smem_V1_1 : smem_V1_0), 64, bh * S + (idx + 1) * 64);
            }
        }
        cur_kv ^= 1;
        
        __syncthreads(); 
        
        float local_max = -INFINITY;
        uint32_t r_idx = tid + ((cluster_rank_fn() & 1) << 6);
        
        uint64_t desc_K0 = make_smem_desc_128B((cur_kv == 0 ? smem_K0_0 : smem_K0_1), 1, 1024);
        uint64_t desc_K1 = make_smem_desc_128B((cur_kv == 0 ? smem_K1_0 : smem_K1_1), 1, 1024);

        for (uint32_t k_step = 0; k_step < 4; k_step++) {
            add_base_offset_bytes(&desc_Q0, k_step * 32);
            add_base_offset_bytes(&desc_K0, k_step * 32);
            if (k_step == 0) {
                umma_f16_cg2(tmem_P, desc_Q0, desc_K0, idesc_QK, 0);
            } else {
                umma_f16_cg2(tmem_P, desc_Q0, desc_K0, idesc_QK, 1);
            }
        }
        for (uint32_t k_step = 0; k_step < 4; k_step++) {
            add_base_offset_bytes(&desc_Q1, k_step * 32);
            add_base_offset_bytes(&desc_K1, k_step * 32);
            umma_f16_cg2(tmem_P, desc_Q1, desc_K1, idesc_QK, 1);
        }
        
        umma_commit_2sm_fn(mbar_Q);
        mbarrier_wait_fn(mbar_Q, 0); 
        
        for (uint32_t col = 0; col < 64; col += 4) {
            uint32_t val = tmem_P + (r_idx << 16) + col;
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(val));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            float f0 = __uint_as_float(r0) * (1.0f / __sqrtf(128.0f));
            float f1 = __uint_as_float(r1) * (1.0f / __sqrtf(128.0f));
            float f2 = __uint_as_float(r2) * (1.0f / __sqrtf(128.0f));
            float f3 = __uint_as_float(r3) * (1.0f / __sqrtf(128.0f));
            
            int global_c = idx * 64 + col;
            if (global_c >= S) {
                f0 = -INFINITY;
            }
            
            local_max = fmaxf(local_max, fmaxf(fmaxf(f0, f1), fmaxf(f2, f3)));
        }
        
        local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, 1));
        
        float prev_max = global_max;
        float new_max = fmaxf(prev_max, local_max);
        float scale = __expf(prev_max - new_max);
        global_sum *= scale;
        global_max = new_max;
        
        float local_sum = 0;
        for (uint32_t col = 0; col < 64; col += 4) {
            uint32_t val = tmem_P + (r_idx << 16) + col;
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(val));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            float f0 = __uint_as_float(r0) * (1.0f / __sqrtf(128.0f));
            float f1 = __uint_as_float(r1) * (1.0f / __sqrtf(128.0f));
            float f2 = __uint_as_float(r2) * (1.0f / __sqrtf(128.0f));
            float f3 = __uint_as_float(r3) * (1.0f / __sqrtf(128.0f));
            
            f0 = __expf(f0 - new_max);
            f1 = __expf(f1 - new_max);
            f2 = __expf(f2 - new_max);
            f3 = __expf(f3 - new_max);
            
            local_sum += (f0 + f1 + f2 + f3);
            
            int x_int4 = col >> 3;
            int swizzled_x = (r_idx & 7) ^ x_int4;
            int swizzled_col = (swizzled_x << 3) | (col & 7);
            __nv_bfloat16* s_ptr = &smem_S[(r_idx * 64) + swizzled_col];
            s_ptr[0] = __float2bfloat16(f0);
            s_ptr[1] = __float2bfloat16(f1);
            s_ptr[2] = __float2bfloat16(f2);
            s_ptr[3] = __float2bfloat16(f3);
        }
        local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, 1);
        
        global_sum += local_sum;
        
        for (uint32_t col = 0; col < 64; col += 4) {
            uint32_t val_O0 = tmem_O0 + (r_idx << 16) + col;
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(val_O0));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            r0 = __float_as_uint(__bfloat162float(__float2bfloat16(__uint_as_float(r0) * scale)));
            r1 = __float_as_uint(__bfloat162float(__float2bfloat16(__uint_as_float(r1) * scale)));
            r2 = __float_as_uint(__bfloat162float(__float2bfloat16(__uint_as_float(r2) * scale)));
            r3 = __float_as_uint(__bfloat162float(__float2bfloat16(__uint_as_float(r3) * scale)));
            
            asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};"
                   :: "r"(val_O0), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
                   
            uint32_t val_O1 = tmem_O1 + (r_idx << 16) + col;
            asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};"
                   :: "r"(val_O1), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
        }
        
        __syncthreads();
        
        uint64_t desc_V0_mn = make_smem_desc_mnmaj_128B((cur_kv == 0 ? smem_V0_0 : smem_V0_1), (cur_kv == 0 ? smem_V0_0 : smem_V0_1), 1024, 8192);
        uint64_t desc_V1_mn = make_smem_desc_mnmaj_128B((cur_kv == 0 ? smem_V1_0 : smem_V1_1), (cur_kv == 0 ? smem_V1_0 : smem_V1_1), 1024, 8192);

        for (uint32_t k_step = 0; k_step < 4; k_step++) {
            add_base_offset_bytes(&desc_S, k_step * 32);
            add_base_offset_bytes(&desc_V0_mn, k_step * 2048);
            
            if (k_step == 0) {
                umma_f16_cg2(tmem_O0, desc_S, desc_V0_mn, idesc_SV, 0);
            } else {
                umma_f16_cg2(tmem_O0, desc_S, desc_V0_mn, idesc_SV, 1);
            }
        }
        
        for (uint32_t k_step = 0; k_step < 4; k_step++) {
            add_base_offset_bytes(&desc_S, k_step * 32);
            add_base_offset_bytes(&desc_V1_mn, k_step * 2048);
            
            if (k_step == 0) {
                umma_f16_cg2(tmem_O1, desc_S, desc_V1_mn, idesc_SV, 0);
            } else {
                umma_f16_cg2(tmem_O1, desc_S, desc_V1_mn, idesc_SV, 1);
            }
        }
        
        umma_commit_2sm_fn(&mbar_KV[cur_kv]);
        mbarrier_wait_fn(&mbar_KV[cur_kv], phase[cur_kv]);
        
        __syncthreads();
    }
    
    for (int half = 0; half < 2; ++half) {
        for (uint32_t col = 0; col < 64; col += 4) {
            uint32_t val_O = (half == 0) ? (tmem_O0 + (tid << 16) + col) : (tmem_O1 + (tid << 16) + col);
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(val_O));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            float f0 = __uint_as_float(r0) / global_sum;
            float f1 = __uint_as_float(r1) / global_sum;
            float f2 = __uint_as_float(r2) / global_sum;
            float f3 = __uint_as_float(r3) / global_sum;
            
            uint32_t nc = col;
            if (half == 1) nc += 64;
            __nv_bfloat16* out = gmem_O + bh * S * 128 + row_base * 128 + nc;
            if (row_base < S && nc < 128) out[tid * 128 + 0] = __float2bfloat16(f0);
            if (row_base < S && nc + 1 < 128) out[tid * 128 + 1] = __float2bfloat16(f1);
            if (row_base < S && nc + 2 < 128) out[tid * 128 + 2] = __float2bfloat16(f2);
            if (row_base < S && nc + 3 < 128) out[tid * 128 + 3] = __float2bfloat16(f3);
        }
    }
    
    if (tid < 128) {
        if (row_base + tid < S) {
            *(gmem_LSE + bh * S + row_base + tid) = global_max + __logf(global_sum);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    uint32_t B = Q.size(0);
    uint32_t H = Q.size(1);
    uint32_t S = Q.size(2);
    uint32_t D = Q.size(3); 

    CUtensorMap tma_Q, tma_K, tma_V;
    __nv_bfloat16* q_ptr = static_cast<__nv_bfloat16*>(Q.data_ptr());
    __nv_bfloat16* k_ptr = static_cast<__nv_bfloat16*>(K.data_ptr());
    __nv_bfloat16* v_ptr = static_cast<__nv_bfloat16*>(V.data_ptr());

    CU_CHECK(create_tma_2d_descriptor_2B(&tma_Q, q_ptr, D, B * H * S, 64, 128, 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    CU_CHECK(create_tma_2d_descriptor_2B(&tma_K, k_ptr, D, B * H * S, 64, 64, 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    CU_CHECK(create_tma_2d_descriptor_2B(&tma_V, v_ptr, D, B * H * S, 64, 64, 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    __nv_bfloat16* o_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* lse_data = static_cast<float*>(LSE.data_ptr());

    dim3 grid(B * H, S / 128);
    dim3 block(128); 

    uint32_t smem_bytes = 100 * 1024;
    CUDA_CHECK(cudaFuncSetAttribute(mha_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem_bytes;
    config.stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    CUDA_CHECK(cudaLaunchKernelEx(&config, mha_kernel, tma_Q, tma_K, tma_V, o_data, lse_data, S));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id))));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_optimized_cuda::run);

} // namespace tvm_ffi_optimized_cuda