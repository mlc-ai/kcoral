#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>
#include <stdio.h>
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

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_before_fn() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
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

__device__ __forceinline__ void umma_f16_cg1_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_1sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1"
        ".mbarrier::arrive::one.shared::cluster.b64"
        " [%0];"
        :: "r"(a));
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    uint64_t base_offset = (addr >> 7) & 0x7;
    d |= (base_offset << 49);
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_cg1(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_transpose_A(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= (1u << 15);   
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_transpose_B(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= (1u << 16);   
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_transpose_A_B(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= (1u << 15);   
    d |= (1u << 16);   
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ void tmem_load_4x(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
}

__device__ __forceinline__ uint32_t rotate_tm_rows(uint32_t addr, uint32_t offset) {
    uint32_t rotate = ((offset >> 7) & 2) ^ ((offset >> 4) & 1);
    return (addr & ~3) | (rotate << 2);
}

__device__ __forceinline__ void tmem_load_4x_rotated(uint32_t col, uint32_t row_idx, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    uint32_t addr = col;
    addr = rotate_tm_rows(addr, row_idx);
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(addr));
}

__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

__device__ __forceinline__ uint32_t get_smid() {
    uint32_t smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

__global__ void __launch_bounds__(128, 1) mha_bwd_dq_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_dO,
    __nv_bfloat16* g_dQ,
    const float* L_raw, int S, float scale)
{
    extern __shared__ char smem_pool[];
    uint64_t* mbar_load = (uint64_t*)smem_pool;
    uint64_t* mbar_smem = (uint64_t*)(smem_pool + 8);
    
    uint8_t* buf_ptr = (uint8_t*)(((((uintptr_t)smem_pool) + 1023) & ~1023));
    
    uint8_t* s_Q0 = buf_ptr;
    uint8_t* s_Q1 = s_Q0 + 8192;
    uint8_t* s_K0 = s_Q1 + 8192;
    uint8_t* s_K1 = s_K0 + 8192;
    uint8_t* s_V0 = s_K1 + 8192;
    uint8_t* s_V1 = s_V0 + 8192;
    uint8_t* s_dO0 = s_V1 + 8192;
    uint8_t* s_dO1 = s_dO0 + 8192;
    uint8_t* s_dS = s_dO1 + 8192;
    uint8_t* s_D = s_dS + 8192;
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_load, 1);
        init_smem_barrier_fn(mbar_smem, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    uint32_t tmem_p[1], tmem_dp[1], tmem_ds[1], tmem_dq0[1], tmem_dq1[1];
    if (threadIdx.x == 0) {
        tmem_alloc_fn(tmem_p, 128);
        tmem_alloc_fn(tmem_dp, 128);
        tmem_alloc_fn(tmem_ds, 128);
        tmem_alloc_fn(tmem_dq0, 128);
        tmem_alloc_fn(tmem_dq1, 128);
    }
    __syncthreads();

    uint32_t s_block = blockIdx.x * 64;
    uint32_t batch_head = blockIdx.y;
    int32_t c1 = batch_head * S + s_block;
    uint32_t tid = threadIdx.x;
    uint32_t phase = 0;

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_load, 6 * 8192);
        
        tma_load_2d_fn(&tma_Q, mbar_load, s_Q0, 0, c1);
        tma_load_2d_fn(&tma_Q, mbar_load, s_Q1, 64, c1);
        
        tma_load_2d_fn(&tma_O, mbar_load, s_K0, 0, c1);
        tma_load_2d_fn(&tma_O, mbar_load, s_K1, 64, c1);
        
        tma_load_2d_fn(&tma_dO, mbar_load, s_dO0, 0, c1);
        tma_load_2d_fn(&tma_dO, mbar_load, s_dO1, 64, c1);
    }
    mbarrier_wait_fn(mbar_load, phase);
    __syncthreads();
    phase ^= 1;

    float sd = scale; // 1.0f / sqrtf(128)

    for (uint32_t iter = 0; iter < S; iter += 64) {
        int32_t c1_iter = batch_head * S + iter;
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_load, 4 * 8192); 
            
            tma_load_2d_fn(&tma_K, mbar_load, s_K0, 0, c1_iter);
            tma_load_2d_fn(&tma_K, mbar_load, s_K1, 64, c1_iter);
            
            tma_load_2d_fn(&tma_V, mbar_load, s_V0, 0, c1_iter);
            tma_load_2d_fn(&tma_V, mbar_load, s_V1, 64, c1_iter);
        }
        mbarrier_wait_fn(mbar_load, phase);
        __syncthreads();
        phase ^= 1;
        
        if (threadIdx.x == 0) {
            uint32_t accum = (iter == 0) ? 0 : 1;
            uint64_t desc_Q0_k = make_smem_desc_sm100_fn(s_Q0, 1, 1024);
            uint64_t desc_K0_k = make_smem_desc_sm100_fn(s_K0, 1, 1024);
            uint32_t id = make_instr_desc_cg1(64, 64);
            umma_f16_cg1_fn(tmem_p[0], desc_Q0_k, desc_K0_k, id, accum);
            
            uint64_t desc_Q1_k = make_smem_desc_sm100_fn(s_Q1, 1, 1024);
            uint64_t desc_K1_k = make_smem_desc_sm100_fn(s_K1, 1, 1024);
            umma_f16_cg1_fn(tmem_p[0], desc_Q1_k, desc_K1_k, id, 1);
        }
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        for (uint32_t row_idx = 0; row_idx < 128; row_idx += 32) {
            uint32_t r_P[8];
            tmem_load_4x_rotated(tmem_p[0], row_idx, &r_P[0], &r_P[1], &r_P[2], &r_P[3]);
            tmem_load_4x_rotated(tmem_p[0] + 64, row_idx, &r_P[4], &r_P[5], &r_P[6], &r_P[7]);
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            uint32_t l_q_idx = batch_head * S + s_block + tid;
            float l_q = (s_block + tid < S) ? L_raw[l_q_idx] : 0.0f;
            uint32_t l_k_idx = batch_head * S + iter + tid;
            float l_k = (iter + tid < S) ? L_raw[l_k_idx] : 0.0f;
            
            for(int i = 0; i < 8; ++i) {
                float p = __uint_as_float(r_P[i]);
                float diff = l_k - l_q;
                p = p * expf(diff);
                
                uint32_t col = (row_idx == 0 || row_idx == 32) ? (i / 4) * 64 + (i % 4) : (i / 4) * 64 + (i % 4) + 64;
                uint32_t x = col / 8;
                uint32_t swizzled_x = (tid % 8) ^ x;
                uint32_t swizzled_col = x * 8 + (col % 8);
                uint32_t addr = tid * 64 + swizzled_col;
                *(uint32_t*)(s_dS + addr) = __float_as_uint(__nv_bfloat162(p, p));
            }
        }
        __syncthreads();
        
        if (threadIdx.x == 0) {
            uint32_t accum = 0;
            uint64_t desc_dO0_k = make_smem_desc_sm100_fn(s_dO0, 1, 1024);
            uint64_t desc_V0_k = make_smem_desc_sm100_fn(s_V0, 1, 1024);
            uint32_t id = make_instr_desc_cg1(64, 64);
            umma_f16_cg1_fn(tmem_dp[0], desc_dO0_k, desc_V0_k, id, accum);
            
            uint64_t desc_dO1_k = make_smem_desc_sm100_fn(s_dO1, 1, 1024);
            uint64_t desc_V1_k = make_smem_desc_sm100_fn(s_V1, 1, 1024);
            umma_f16_cg1_fn(tmem_dp[0], desc_dO1_k, desc_V1_k, id, 1);
        }
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        for (uint32_t row_idx = 0; row_idx < 128; row_idx += 32) {
            uint32_t r_dP[8];
            tmem_load_4x_rotated(tmem_dp[0], row_idx, &r_dP[0], &r_dP[1], &r_dP[2], &r_dP[3]);
            tmem_load_4x_rotated(tmem_dp[0] + 64, row_idx, &r_dP[4], &r_dP[5], &r_dP[6], &r_dP[7]);
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            for(int i = 0; i < 8; ++i) {
                float dp = __uint_as_float(r_dP[i]);
                uint32_t col = (row_idx == 0 || row_idx == 32) ? (i / 4) * 64 + (i % 4) : (i / 4) * 64 + (i % 4) + 64;
                uint32_t x = col / 8;
                uint32_t swizzled_x = (tid % 8) ^ x;
                uint32_t swizzled_col = x * 8 + (col % 8);
                uint32_t addr = tid * 64 + swizzled_col;
                *(uint32_t*)(s_dS + addr) = __float_as_uint(__nv_bfloat162(dp, dp));
            }
        }
        __syncthreads();
        
        if (threadIdx.x == 0) {
            uint32_t accum = 1;
            uint64_t desc_dS_k = make_smem_desc_sm100_fn(s_dS, 1, 1024);
            uint64_t desc_K0_k = make_smem_desc_sm100_fn(s_K0, 1, 1024);
            uint32_t id = make_instr_desc_cg1(64, 64);
            umma_f16_cg1_fn(tmem_dq0[0], desc_dS_k, desc_K0_k, id, accum);
            
            uint64_t desc_K1_k = make_smem_desc_sm100_fn(s_K1, 1, 1024);
            umma_f16_cg1_fn(tmem_dq1[0], desc_dS_k, desc_K1_k, id, accum);
        }
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    }

    asm volatile("barrier.sync.aligned 2, 128;" ::: "memory");
    for (uint32_t col = 0; col < 64; col += 4) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4x(tmem_dq0[0] + col, &r0, &r1, &r2, &r3);
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        uint32_t idx = (batch_head * S + s_block + tid) * 128 + col;
        if (idx < (uint64_t)S * 128) {
            *(uint32_t*)&g_dQ[idx] = __float_as_uint(__nv_bfloat162(__uint_as_float(r0), __uint_as_float(r1)));
            *(uint32_t*)&g_dQ[idx+2] = __float_as_uint(__nv_bfloat162(__uint_as_float(r2), __uint_as_float(r3)));
        }
    }
    for (uint32_t col = 0; col < 64; col += 4) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4x(tmem_dq1[0] + col, &r0, &r1, &r2, &r3);
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        uint32_t idx = (batch_head * S + s_block + tid) * 128 + 64 + col;
        if (idx < (uint64_t)S * 128) {
            *(uint32_t*)&g_dQ[idx] = __float_as_uint(__nv_bfloat162(__uint_as_float(r0), __uint_as_float(r1)));
            *(uint32_t*)&g_dQ[idx+2] = __float_as_uint(__nv_bfloat162(__uint_as_float(r2), __uint_as_float(r3)));
        }
    }
    asm volatile("barrier.sync.aligned 2, 128;" ::: "memory");

    if (threadIdx.x == 0) {
        tmem_dealloc_fn(tmem_p[0], 128);
        tmem_dealloc_fn(tmem_dp[0], 128);
        tmem_dealloc_fn(tmem_ds[0], 128);
        tmem_dealloc_fn(tmem_dq0[0], 128);
        tmem_dealloc_fn(tmem_dq1[0], 128);
    }
}

__global__ void __launch_bounds__(128, 1) mha_bwd_dkv_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_dO,
    __nv_bfloat16* g_dK, __nv_bfloat16* g_dV,
    const float* L_raw, int S, float scale)
{
    extern __shared__ char smem_pool[];
    uint64_t* mbar_load = (uint64_t*)smem_pool;
    uint64_t* mbar_smem = (uint64_t*)(smem_pool + 8);
    
    uint8_t* buf_ptr = (uint8_t*)(((((uintptr_t)smem_pool) + 1023) & ~1023));
    
    uint8_t* s_Q0 = buf_ptr;
    uint8_t* s_Q1 = s_Q0 + 8192;
    uint8_t* s_K0 = s_Q1 + 8192;
    uint8_t* s_K1 = s_K0 + 8192;
    uint8_t* s_V0 = s_K1 + 8192;
    uint8_t* s_V1 = s_V0 + 8192;
    uint8_t* s_dO0 = s_V1 + 8192;
    uint8_t* s_dO1 = s_dO0 + 8192;
    uint8_t* s_P = s_dO1 + 8192;
    uint8_t* s_dS = s_P + 8192;
    uint8_t* s_D = s_dS + 8192;
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_load, 1);
        init_smem_barrier_fn(mbar_smem, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    uint32_t tmem_p[1], tmem_dp[1], tmem_ds[1], tmem_dk0[1], tmem_dk1[1], tmem_dv0[1], tmem_dv1[1];
    if (threadIdx.x == 0) {
        tmem_alloc_fn(tmem_p, 128);
        tmem_alloc_fn(tmem_dp, 128);
        tmem_alloc_fn(tmem_ds, 128);
        tmem_alloc_fn(tmem_dk0, 128);
        tmem_alloc_fn(tmem_dk1, 128);
        tmem_alloc_fn(tmem_dv0, 128);
        tmem_alloc_fn(tmem_dv1, 128);
    }
    __syncthreads();

    uint32_t s_block = blockIdx.x * 64;
    uint32_t batch_head = blockIdx.y;
    int32_t c1 = batch_head * S + s_block;
    uint32_t tid = threadIdx.x;
    uint32_t phase = 0;

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_load, 4 * 8192);
        
        tma_load_2d_fn(&tma_K, mbar_load, s_K0, 0, c1);
        tma_load_2d_fn(&tma_K, mbar_load, s_K1, 64, c1);
        tma_load_2d_fn(&tma_V, mbar_load, s_V0, 0, c1);
        tma_load_2d_fn(&tma_V, mbar_load, s_V1, 64, c1);
    }
    mbarrier_wait_fn(mbar_load, phase);
    __syncthreads();
    phase ^= 1;

    float sd = scale;

    for (uint32_t iter = 0; iter < S; iter += 64) {
        int32_t c1_iter = batch_head * S + iter;
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_load, 6 * 8192);
            
            tma_load_2d_fn(&tma_Q, mbar_load, s_Q0, 0, c1_iter);
            tma_load_2d_fn(&tma_Q, mbar_load, s_Q1, 64, c1_iter);
            tma_load_2d_fn(&tma_O, mbar_load, s_dO0, 0, c1_iter);
            tma_load_2d_fn(&tma_O, mbar_load, s_dO1, 64, c1_iter);
            tma_load_2d_fn(&tma_dO, mbar_load, s_P, 0, c1_iter);
            tma_load_2d_fn(&tma_dO, mbar_load, s_dS, 64, c1_iter);
        }
        mbarrier_wait_fn(mbar_load, phase);
        __syncthreads();
        phase ^= 1;
        
        if (threadIdx.x == 0) {
            uint32_t accum = 0;
            uint64_t desc_Q0_k = make_smem_desc_sm100_fn(s_Q0, 1, 1024);
            uint64_t desc_K0_k = make_smem_desc_sm100_fn(s_K0, 1, 1024);
            uint32_t id = make_instr_desc_cg1(64, 64);
            umma_f16_cg1_fn(tmem_p[0], desc_Q0_k, desc_K0_k, id, accum);
            
            uint64_t desc_Q1_k = make_smem_desc_sm100_fn(s_Q1, 1, 1024);
            uint64_t desc_K1_k = make_smem_desc_sm100_fn(s_K1, 1, 1024);
            umma_f16_cg1_fn(tmem_p[0], desc_Q1_k, desc_K1_k, id, 1);
        }
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        for (uint32_t row_idx = 0; row_idx < 128; row_idx += 32) {
            uint32_t r_P[8];
            tmem_load_4x_rotated(tmem_p[0], row_idx, &r_P[0], &r_P[1], &r_P[2], &r_P[3]);
            tmem_load_4x_rotated(tmem_p[0] + 64, row_idx, &r_P[4], &r_P[5], &r_P[6], &r_P[7]);
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            uint32_t l_q_idx = batch_head * S + iter + tid;
            float l_q = (iter + tid < S) ? L_raw[l_q_idx] : 0.0f;
            uint32_t l_k_idx = batch_head * S + s_block + tid;
            float l_k = (s_block + tid < S) ? L_raw[l_k_idx] : 0.0f;
            
            for(int i = 0; i < 8; ++i) {
                float p = __uint_as_float(r_P[i]);
                float diff = l_k - l_q;
                p = p * expf(diff);
                
                uint32_t col = (row_idx == 0 || row_idx == 32) ? (i / 4) * 64 + (i % 4) : (i / 4) * 64 + (i % 4) + 64;
                uint32_t x = col / 8;
                uint32_t swizzled_x = (tid % 8) ^ x;
                uint32_t swizzled_col = x * 8 + (col % 8);
                uint32_t addr = tid * 64 + swizzled_col;
                *(uint32_t*)(s_P + addr) = __float_as_uint(__nv_bfloat162(p, p));
            }
        }
        __syncthreads();
        
        if (threadIdx.x == 0) {
            uint32_t accum = 0;
            uint64_t desc_dO0_k = make_smem_desc_sm100_fn(s_dO0, 1, 1024);
            uint64_t desc_V0_k = make_smem_desc_sm100_fn(s_V0, 1, 1024);
            uint32_t id = make_instr_desc_cg1(64, 64);
            umma_f16_cg1_fn(tmem_dp[0], desc_dO0_k, desc_V0_k, id, accum);
            
            uint64_t desc_dO1_k = make_smem_desc_sm100_fn(s_dO1, 1, 1024);
            uint64_t desc_V1_k = make_smem_desc_sm100_fn(s_V1, 1, 1024);
            umma_f16_cg1_fn(tmem_dp[0], desc_dO1_k, desc_V1_k, id, 1);
        }
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        for (uint32_t row_idx = 0; row_idx < 128; row_idx += 32) {
            uint32_t r_dP[8];
            tmem_load_4x_rotated(tmem_dp[0], row_idx, &r_dP[0], &r_dP[1], &r_dP[2], &r_dP[3]);
            tmem_load_4x_rotated(tmem_dp[0] + 64, row_idx, &r_dP[4], &r_dP[5], &r_dP[6], &r_dP[7]);
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            for(int i = 0; i < 8; ++i) {
                float dp = __uint_as_float(r_dP[i]);
                uint32_t col = (row_idx == 0 || row_idx == 32) ? (i / 4) * 64 + (i % 4) : (i / 4) * 64 + (i % 4) + 64;
                uint32_t x = col / 8;
                uint32_t swizzled_x = (tid % 8) ^ x;
                uint32_t swizzled_col = x * 8 + (col % 8);
                uint32_t addr = tid * 64 + swizzled_col;
                *(uint32_t*)(s_dS + addr) = __float_as_uint(__nv_bfloat162(dp, dp));
            }
        }
        __syncthreads();
        
        if (threadIdx.x == 0) {
            uint32_t accum_dk0 = (iter == 0) ? 0 : 1;
            uint32_t accum_dk1 = (iter == 0) ? 0 : 1;
            uint64_t desc_dS_T = make_smem_desc_sm100_fn(s_dS, (64 / 8) * 1024, 1024);
            uint64_t desc_Q0 = make_smem_desc_sm100_fn(s_Q0, 1, 1024);
            uint64_t desc_Q1 = make_smem_desc_sm100_fn(s_Q1, 1, 1024);
            uint32_t id_dS_T_Q = make_instr_desc_transpose_A_B(64, 64);
            umma_f16_cg1_fn(tmem_dk0[0], desc_dS_T, desc_Q0, id_dS_T_Q, accum_dk0);
            umma_f16_cg1_fn(tmem_dk1[0], desc_dS_T, desc_Q1, id_dS_T_Q, accum_dk1);
        }
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        if (threadIdx.x == 0) {
            uint32_t accum_dv0 = (iter == 0) ? 0 : 1;
            uint32_t accum_dv1 = (iter == 0) ? 0 : 1;
            uint64_t desc_P_T = make_smem_desc_sm100_fn(s_P, (64 / 8) * 1024, 1024);
            uint64_t desc_dO0 = make_smem_desc_sm100_fn(s_dO0, 1, 1024);
            uint64_t desc_dO1 = make_smem_desc_sm100_fn(s_dO1, 1, 1024);
            uint32_t id_P_T_dO = make_instr_desc_transpose_A_B(64, 64);
            umma_f16_cg1_fn(tmem_dv0[0], desc_P_T, desc_dO0, id_P_T_dO, accum_dv0);
            umma_f16_cg1_fn(tmem_dv1[0], desc_P_T, desc_dO1, id_P_T_dO, accum_dv1);
        }
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    }

    asm volatile("barrier.sync.aligned 2, 128;" ::: "memory");
    for (uint32_t col = 0; col < 64; col += 4) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4x(tmem_dk0[0] + col, &r0, &r1, &r2, &r3);
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        uint32_t idx = (batch_head * S + s_block + tid) * 128 + col;
        if (idx < (uint64_t)S * 128) {
            *(uint32_t*)&g_dK[idx] = __float_as_uint(__nv_bfloat162(__uint_as_float(r0), __uint_as_float(r1)));
            *(uint32_t*)&g_dK[idx+2] = __float_as_uint(__nv_bfloat162(__uint_as_float(r2), __uint_as_float(r3)));
        }
    }
    for (uint32_t col = 0; col < 64; col += 4) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4x(tmem_dk1[0] + col, &r0, &r1, &r2, &r3);
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        uint32_t idx = (batch_head * S + s_block + tid) * 128 + 64 + col;
        if (idx < (uint64_t)S * 128) {
            *(uint32_t*)&g_dK[idx] = __float_as_uint(__nv_bfloat162(__uint_as_float(r0), __uint_as_float(r1)));
            *(uint32_t*)&g_dK[idx+2] = __float_as_uint(__nv_bfloat162(__uint_as_float(r2), __uint_as_float(r3)));
        }
    }
    for (uint32_t col = 0; col < 64; col += 4) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4x(tmem_dv0[0] + col, &r0, &r1, &r2, &r3);
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        uint32_t idx = (batch_head * S + s_block + tid) * 128 + col;
        if (idx < (uint64_t)S * 128) {
            *(uint32_t*)&g_dV[idx] = __float_as_uint(__nv_bfloat162(__uint_as_float(r0), __uint_as_float(r1)));
            *(uint32_t*)&g_dV[idx+2] = __float_as_uint(__nv_bfloat162(__uint_as_float(r2), __uint_as_float(r3)));
        }
    }
    for (uint32_t col = 0; col < 64; col += 4) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4x(tmem_dv1[0] + col, &r0, &r1, &r2, &r3);
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        uint32_t idx = (batch_head * S + s_block + tid) * 128 + 64 + col;
        if (idx < (uint64_t)S * 128) {
            *(uint32_t*)&g_dV[idx] = __float_as_uint(__nv_bfloat162(__uint_as_float(r0), __uint_as_float(r1)));
            *(uint32_t*)&g_dV[idx+2] = __float_as_uint(__nv_bfloat162(__uint_as_float(r2), __uint_as_float(r3)));
        }
    }
    asm volatile("barrier.sync.aligned 2, 128;" ::: "memory");

    if (threadIdx.x == 0) {
        tmem_dealloc_fn(tmem_p[0], 128);
        tmem_dealloc_fn(tmem_dp[0], 128);
        tmem_dealloc_fn(tmem_ds[0], 128);
        tmem_dealloc_fn(tmem_dk0[0], 128);
        tmem_dealloc_fn(tmem_dk1[0], 128);
        tmem_dealloc_fn(tmem_dv0[0], 128);
        tmem_dealloc_fn(tmem_dv1[0], 128);
    }
}

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapDataType dataType, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d, dataType, 2, globalAddress, globalDim, globalStrides,
        boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle, l2Promotion, oobFill
    );
}

namespace tvm_ffi_kernel {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L, tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t d = Q.size(3); 
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_O, tma_dO;
    
    auto create = [&](CUtensorMap* tma, void* ptr) {
        CUresult res = create_tma_2d_descriptor_2B(
            tma, ptr, 128, B * H * S, 64, 64,
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (res != CUDA_SUCCESS) {
            fprintf(stderr, "TMA error %d\n", res);
            exit(1);
        }
    };
    
    create(&tma_Q, Q.data_ptr());
    create(&tma_K, K.data_ptr());
    create(&tma_V, V.data_ptr());
    create(&tma_O, O.data_ptr());
    create(&tma_dO, dO.data_ptr());
    
    dim3 grid((S + 63) / 64, B * H);
    dim3 block(128);
    
    int smem_size = 196608; 
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem_size;
    config.stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    CUDA_CHECK(cudaFuncSetAttribute(mha_bwd_dq_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    CUDA_CHECK(cudaFuncSetAttribute(mha_bwd_dkv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    CUDA_CHECK(cudaLaunchKernelEx(&config, mha_bwd_dq_kernel, tma_Q, tma_K, tma_V, tma_O, tma_dO, static_cast<__nv_bfloat16*>(dQ.data_ptr()), static_cast<const float*>(L.data_ptr()), S, 1.0f / sqrtf(128)));
    CUDA_CHECK(cudaLaunchKernelEx(&config, mha_bwd_dkv_kernel, tma_Q, tma_K, tma_V, tma_O, tma_dO, static_cast<__nv_bfloat16*>(dK.data_ptr()), static_cast<__nv_bfloat16*>(dV.data_ptr()), static_cast<const float*>(L.data_ptr()), S, 1.0f / sqrtf(128)));
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(config.stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_kernel