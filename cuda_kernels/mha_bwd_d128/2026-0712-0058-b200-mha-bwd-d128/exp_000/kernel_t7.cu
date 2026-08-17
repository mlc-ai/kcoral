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

__device__ __forceinline__ void umma_commit_2sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::2"
        ".mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
        " [%0], %1;"
        :: "r"(a), "h"((uint16_t)0x3));
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

__device__ __forceinline__ uint64_t make_smem_desc_k_major(void* smem_ptr) {
    return make_smem_desc_sm100_fn(smem_ptr, 1, 1024);
}

__device__ __forceinline__ uint64_t make_smem_desc_mn_major_64(void* smem_ptr) {
    return make_smem_desc_sm100_fn(smem_ptr, 8192, 1024);
}

__device__ __forceinline__ uint32_t make_instr_desc_cg2(uint32_t M, uint32_t N) {
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

__device__ __forceinline__ void tmem_load_4x(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
}

__device__ __forceinline__ uint32_t get_swizzled_offset_mn_major(uint32_t row, uint32_t col) {
    uint32_t x = col / 8;
    uint32_t rem = col % 8;
    uint32_t swizzled_x = (row % 8) ^ x;
    return row * 64 + swizzled_x * 8 + rem;
}

__device__ __forceinline__ uint32_t pack_bf16_to_u32(float a, float b) {
    __nv_bfloat16 ba = __float2bfloat16(a);
    __nv_bfloat16 bb = __float2bfloat16(b);
    uint32_t res;
    asm volatile("mov.b32 %0, {%1, %2};" : "=r"(res) : "h"(*(uint16_t*)&ba), "h"(*(uint16_t*)&bb));
    return res;
}

__global__ void __launch_bounds__(128, 1) mha_bwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_dO,
    __nv_bfloat16* g_dQ, __nv_bfloat16* g_dK, __nv_bfloat16* g_dV,
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
    uint8_t* s_O0 = s_V1 + 8192;
    uint8_t* s_O1 = s_O0 + 8192;
    uint8_t* s_dO0 = s_O1 + 8192;
    uint8_t* s_dO1 = s_dO0 + 8192;
    uint8_t* s_P = s_dO1 + 8192;
    uint8_t* s_dP = s_P + 16384;
    uint8_t* s_dS = s_dP + 16384;
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_load, 1);
        init_smem_barrier_fn(mbar_smem, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    uint32_t tmem_P[1], tmem_dP[1], tmem_dS[1], tmem_dQ0[1], tmem_dQ1[1], tmem_dK0[1], tmem_dK1[1], tmem_dV0[1], tmem_dV1[1];
    if (elect_one_sync_fn()) {
        tmem_alloc_fn(tmem_P, 128);
        tmem_alloc_fn(tmem_dP, 128);
        tmem_alloc_fn(tmem_dS, 128);
        tmem_alloc_fn(tmem_dQ0, 64);
        tmem_alloc_fn(tmem_dQ1, 64);
        tmem_alloc_fn(tmem_dK0, 64);
        tmem_alloc_fn(tmem_dK1, 64);
        tmem_alloc_fn(tmem_dV0, 64);
        tmem_alloc_fn(tmem_dV1, 64);
    }
    __syncthreads();

    uint32_t s_block = blockIdx.x * 64;
    uint32_t batch_head = blockIdx.y;
    int32_t c1 = batch_head * S + s_block;
    uint32_t tid = threadIdx.x;
    uint32_t phase = 0;

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_load, 8 * 8192);
        
        tma_load_2d_fn(&tma_Q, mbar_load, s_Q0, 0, c1);
        tma_load_2d_fn(&tma_Q, mbar_load, s_Q1, 64, c1);
        
        tma_load_2d_fn(&tma_O, mbar_load, s_O0, 0, c1);
        tma_load_2d_fn(&tma_O, mbar_load, s_O1, 64, c1);
        
        tma_load_2d_fn(&tma_dO, mbar_load, s_dO0, 0, c1);
        tma_load_2d_fn(&tma_dO, mbar_load, s_dO1, 64, c1);
    }
    mbarrier_wait_fn(mbar_load, phase);
    __syncthreads();
    phase ^= 1;

    uint32_t phase_smem = 0;
    float sd = 1.0f / sqrtf(128.0f);

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
            uint32_t accum_P = (iter == 0) ? 0 : 1;
            uint32_t accum_dP = (iter == 0) ? 0 : 1;
            
            uint64_t desc_O0_mn = make_smem_desc_mn_major_64(s_O0);
            uint64_t desc_V0_k = make_smem_desc_k_major(s_V0);
            uint32_t id_P0 = make_instr_desc_transpose_B(128, 128);
            umma_f16_cg2_fn(tmem_P[0], desc_O0_mn, desc_V0_k, id_P0, accum_P);
            
            uint64_t desc_O1_mn = make_smem_desc_mn_major_64(s_O1);
            uint64_t desc_V1_k = make_smem_desc_k_major(s_V1);
            umma_f16_cg2_fn(tmem_P[0] + 64, desc_O1_mn, desc_V1_k, id_P0, accum_P);
            
            uint64_t desc_dO0_mn = make_smem_desc_mn_major_64(s_dO0);
            uint32_t id_dP0 = make_instr_desc_transpose_B(128, 128);
            umma_f16_cg2_fn(tmem_dP[0], desc_dO0_mn, desc_V0_k, id_dP0, accum_dP);
            
            uint64_t desc_dO1_mn = make_smem_desc_mn_major_64(s_dO1);
            umma_f16_cg2_fn(tmem_dP[0] + 64, desc_dO1_mn, desc_V1_k, id_dP0, accum_dP);
        }
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        float d_val = 0.0f;
        for(int i = 0; i < 4; ++i) {
            uint32_t col_offset = (tid / 64) * 64 + i * 16;
            uint32_t p0, p1, p2, p3;
            uint32_t dp0, dp1, dp2, dp3;
            
            tmem_load_4x(tmem_P[0] + col_offset, &p0, &p1, &p2, &p3);
            tmem_load_4x(tmem_dP[0] + col_offset, &dp0, &dp1, &dp2, &dp3);
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            d_val += __uint_as_float(p0) * __uint_as_float(dp0);
            d_val += __uint_as_float(p1) * __uint_as_float(dp1);
            d_val += __uint_as_float(p2) * __uint_as_float(dp2);
            d_val += __uint_as_float(p3) * __uint_as_float(dp3);
            
            float ds0 = __uint_as_float(p0) * (__uint_as_float(dp0) - d_val) * sd;
            float ds1 = __uint_as_float(p1) * (__uint_as_float(dp1) - d_val) * sd;
            float ds2 = __uint_as_float(p2) * (__uint_as_float(dp2) - d_val) * sd;
            float ds3 = __uint_as_float(p3) * (__uint_as_float(dp3) - d_val) * sd;
            
            *(uint32_t*)(s_dS + get_swizzled_offset_mn_major(tid, col_offset) * 2) = pack_bf16_to_u32(ds0, ds1);
            *(uint32_t*)(s_dS + get_swizzled_offset_mn_major(tid, col_offset + 2) * 2) = pack_bf16_to_u32(ds2, ds3);
            
            *(uint32_t*)(s_P + get_swizzled_offset_mn_major(tid, col_offset) * 2) = pack_bf16_to_u32(__uint_as_float(p0), __uint_as_float(p1));
            *(uint32_t*)(s_P + get_swizzled_offset_mn_major(tid, col_offset + 2) * 2) = pack_bf16_to_u32(__uint_as_float(p2), __uint_as_float(p3));
        }
        
        __syncthreads();
        if (threadIdx.x == 0) {
            uint32_t accum_dQ0 = (iter == 0) ? 0 : 1;
            uint32_t accum_dQ1 = (iter == 0) ? 0 : 1;
            uint32_t accum_dK0 = (iter == 0) ? 0 : 1;
            uint32_t accum_dK1 = (iter == 0) ? 0 : 1;
            uint32_t accum_dV0 = (iter == 0) ? 0 : 1;
            uint32_t accum_dV1 = (iter == 0) ? 0 : 1;
            
            uint64_t desc_dS_k = make_smem_desc_k_major(s_dS);
            uint64_t desc_K0_k = make_smem_desc_k_major(s_K0);
            uint32_t id_dQ = make_instr_desc_cg2(128, 128);
            umma_f16_cg2_fn(tmem_dQ0[0], desc_dS_k, desc_K0_k, id_dQ, accum_dQ0);
            
            uint64_t desc_K1_k = make_smem_desc_k_major(s_K1);
            umma_f16_cg2_fn(tmem_dQ1[0], desc_dS_k, desc_K1_k, id_dQ, accum_dQ1);
            
            uint64_t desc_dS_T = make_smem_desc_mn_major_64(s_dS);
            uint64_t desc_Q0 = make_smem_desc_k_major(s_Q0);
            uint32_t id_dS_T_Q = make_instr_desc_transpose_A_B(128, 128);
            umma_f16_cg2_fn(tmem_dK0[0], desc_dS_T, desc_Q0, id_dS_T_Q, accum_dK0);
            
            uint64_t desc_Q1 = make_smem_desc_k_major(s_Q1);
            umma_f16_cg2_fn(tmem_dK1[0], desc_dS_T, desc_Q1, id_dS_T_Q, accum_dK1);
            
            uint64_t desc_P_T = make_smem_desc_mn_major_64(s_P);
            uint64_t desc_dO0 = make_smem_desc_k_major(s_dO0);
            uint32_t id_P_T_dO = make_instr_desc_transpose_A_B(128, 128);
            umma_f16_cg2_fn(tmem_dV0[0], desc_P_T, desc_dO0, id_P_T_dO, accum_dV0);
            
            uint64_t desc_dO1 = make_smem_desc_k_major(s_dO1);
            umma_f16_cg2_fn(tmem_dV1[0], desc_P_T, desc_dO1, id_P_T_dO, accum_dV1);
            
            umma_commit_2sm_fn(mbar_smem);
        }
        mbarrier_wait_fn(mbar_smem, phase_smem);
        phase_smem ^= 1;
        __syncthreads();
    }
    
    for (uint32_t col = 0; col < 64; col += 4) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4x(tmem_dQ0[0] + col, &r0, &r1, &r2, &r3);
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        uint32_t row_idx = (tid / 64) * 64 + (tid % 64);
        uint32_t idx = (batch_head * S * 128) + (s_block + row_idx) * 128 + col;
        if (idx < (uint64_t)(S * 128)) {
            *(uint32_t*)&g_dQ[idx] = pack_bf16_to_u32(__uint_as_float(r0), __uint_as_float(r1));
            *(uint32_t*)&g_dQ[idx+2] = pack_bf16_to_u32(__uint_as_float(r2), __uint_as_float(r3));
        }
    }
    for (uint32_t col = 0; col < 64; col += 4) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4x(tmem_dQ1[0] + col, &r0, &r1, &r2, &r3);
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        uint32_t row_idx = (tid / 64) * 64 + (tid % 64);
        uint32_t idx = (batch_head * S * 128) + (s_block + row_idx) * 128 + 64 + col;
        if (idx < (uint64_t)(S * 128)) {
            *(uint32_t*)&g_dQ[idx] = pack_bf16_to_u32(__uint_as_float(r0), __uint_as_float(r1));
            *(uint32_t*)&g_dQ[idx+2] = pack_bf16_to_u32(__uint_as_float(r2), __uint_as_float(r3));
        }
    }
    for (uint32_t col = 0; col < 64; col += 4) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4x(tmem_dK0[0] + col, &r0, &r1, &r2, &r3);
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        uint32_t row_idx = (tid / 64) * 64 + (tid % 64);
        uint32_t idx = (batch_head * S * 128) + (s_block + row_idx) * 128 + col;
        if (idx < (uint64_t)(S * 128)) {
            *(uint32_t*)&g_dK[idx] = pack_bf16_to_u32(__uint_as_float(r0), __uint_as_float(r1));
            *(uint32_t*)&g_dK[idx+2] = pack_bf16_to_u32(__uint_as_float(r2), __uint_as_float(r3));
        }
    }
    for (uint32_t col = 0; col < 64; col += 4) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4x(tmem_dK1[0] + col, &r0, &r1, &r2, &r3);
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        uint32_t row_idx = (tid / 64) * 64 + (tid % 64);
        uint32_t idx = (batch_head * S * 128) + (s_block + row_idx) * 128 + 64 + col;
        if (idx < (uint64_t)(S * 128)) {
            *(uint32_t*)&g_dK[idx] = pack_bf16_to_u32(__uint_as_float(r0), __uint_as_float(r1));
            *(uint32_t*)&g_dK[idx+2] = pack_bf16_to_u32(__uint_as_float(r2), __uint_as_float(r3));
        }
    }
    for (uint32_t col = 0; col < 64; col += 4) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4x(tmem_dV0[0] + col, &r0, &r1, &r2, &r3);
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        uint32_t row_idx = (tid / 64) * 64 + (tid % 64);
        uint32_t idx = (batch_head * S * 128) + (s_block + row_idx) * 128 + col;
        if (idx < (uint64_t)(S * 128)) {
            *(uint32_t*)&g_dV[idx] = pack_bf16_to_u32(__uint_as_float(r0), __uint_as_float(r1));
            *(uint32_t*)&g_dV[idx+2] = pack_bf16_to_u32(__uint_as_float(r2), __uint_as_float(r3));
        }
    }
    for (uint32_t col = 0; col < 64; col += 4) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4x(tmem_dV1[0] + col, &r0, &r1, &r2, &r3);
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        uint32_t row_idx = (tid / 64) * 64 + (tid % 64);
        uint32_t idx = (batch_head * S * 128) + (s_block + row_idx) * 128 + 64 + col;
        if (idx < (uint64_t)(S * 128)) {
            *(uint32_t*)&g_dV[idx] = pack_bf16_to_u32(__uint_as_float(r0), __uint_as_float(r1));
            *(uint32_t*)&g_dV[idx+2] = pack_bf16_to_u32(__uint_as_float(r2), __uint_as_float(r3));
        }
    }

    if (elect_one_sync_fn()) {
        tmem_dealloc_fn(tmem_P[0], 128);
        tmem_dealloc_fn(tmem_dP[0], 128);
        tmem_dealloc_fn(tmem_dS[0], 128);
        tmem_dealloc_fn(tmem_dQ0[0], 64);
        tmem_dealloc_fn(tmem_dQ1[0], 64);
        tmem_dealloc_fn(tmem_dK0[0], 64);
        tmem_dealloc_fn(tmem_dK1[0], 64);
        tmem_dealloc_fn(tmem_dV0[0], 64);
        tmem_dealloc_fn(tmem_dV1[0], 64);
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

    CUDA_CHECK(cudaFuncSetAttribute(mha_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    CUDA_CHECK(cudaLaunchKernelEx(&config, mha_bwd_kernel, tma_Q, tma_K, tma_V, tma_O, tma_dO, static_cast<__nv_bfloat16*>(dQ.data_ptr()), static_cast<__nv_bfloat16*>(dK.data_ptr()), static_cast<__nv_bfloat16*>(dV.data_ptr()), static_cast<const float*>(L.data_ptr()), S, 1.0f / sqrtf(128)));
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(config.stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_kernel