#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>
#include <algorithm>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_mha_bwd {

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

__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
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

__device__ __forceinline__ void tma_load_2d_cg2_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    uint32_t sa = (uint32_t)__cvta_generic_to_shared(smem);
    uint32_t ba = (uint32_t)__cvta_generic_to_shared(&bar[0]) & 0xFEFFFFFF;
    asm volatile(
        "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(sa), "l"((uint64_t)d), "r"(c0), "r"(c1), "r"(ba) : "memory");
}

__device__ __forceinline__ void load_tile_tma(uint32_t c0, uint64_t* bar, void* smem, const CUtensorMap* tma, int32_t inner_offset, int32_t outer_offset) {
    uint32_t smem_mbar = (uint32_t)__cvta_generic_to_shared(bar);
    uint32_t smem_ptr  = (uint32_t)__cvta_generic_to_shared(smem);
    uint64_t cache_hint = 0; // L2 promotion NONE
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, %4}], [%2], %5, %6;\n"
        :: "r"(smem_ptr), "l"((uint64_t)tma), "r"(smem_mbar), "r"(inner_offset), "r"(outer_offset), "r"(c0), "l"(cache_hint) : "memory");
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

__device__ __forceinline__ void copy_to_tmem_swizzled_128B(uint32_t tmem_col, void* smem_ptr) {
    uint32_t smem_a = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "tcgen05.cta_group::2.128x128b [%0], [%1];"
        :: "r"(tmem_col), "r"(smem_a));
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

__device__ __forceinline__ uint64_t make_smem_desc_K_major_128B(void* smem_ptr) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint32_t lbo = 16;
    uint32_t sbo = 1024;
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_N_major_128B(void* smem_ptr) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint32_t lbo = 8192;
    uint32_t sbo = 1024;
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;
    return d;
}

template <uint32_t M, uint32_t N>
__device__ __forceinline__ uint32_t make_instr_desc_B_BF16() {
    uint32_t d = 0;
    d |= (1u << 4);
    d |= (1u << 7);
    d |= (1u << 10);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__device__ void gemm_S(uint32_t tmem_S_T, void* A_desc0, void* A_desc1, void* B_desc0, void* B_desc1, bool initial_accum) {
    uint32_t base_addr = tmem_S_T;
    uint32_t offset = 0;
    uint64_t a_desc = *(uint64_t*)A_desc0;
    uint64_t b_desc = *(uint64_t*)B_desc0;
    uint32_t d = make_instr_desc_B_BF16<64, 64>();
    d |= (1u << 16);
    
    uint32_t accum = initial_accum ? 1 : 0;
    for (uint32_t i = 0; i < 4; ++i) {
        umma_f16_cg2_fn(base_addr + offset, a_desc, b_desc, d, accum);
        offset += (i % 2 == 0) ? 0 : 8;
        a_desc += (i % 2 == 0) ? 16 : 0;
        b_desc += (i % 2 == 0) ? 16 : 0;
        accum = 1;
    }
    
    offset = 0;
    a_desc = *(uint64_t*)A_desc1;
    b_desc = *(uint64_t*)B_desc1;
    for (uint32_t i = 0; i < 4; ++i) {
        umma_f16_cg2_fn(base_addr + offset, a_desc, b_desc, d, accum);
        offset += (i % 2 == 0) ? 0 : 8;
        a_desc += (i % 2 == 0) ? 16 : 0;
        b_desc += (i % 2 == 0) ? 16 : 0;
        accum = 1;
    }
}

__device__ void gemm_dV_T(uint32_t tmem_dV, void* A_desc, void* B_desc, bool initial_accum) {
    uint32_t base_addr = tmem_dV;
    uint32_t offset = 0;
    uint64_t a_desc = *(uint64_t*)A_desc;
    uint64_t b_desc = *(uint64_t*)B_desc;
    uint32_t d = make_instr_desc_B_BF16<64, 64>();
    
    uint32_t accum = initial_accum ? 1 : 0;
    for (uint32_t i = 0; i < 4; ++i) {
        umma_f16_cg2_fn(base_addr + offset, a_desc, b_desc, d, accum);
        offset += (i % 2 == 0) ? 0 : 8;
        a_desc += (i % 2 == 0) ? 16 : 0;
        b_desc += (i % 2 == 0) ? 16 : 0;
        accum = 1;
    }
}

__device__ void gemm_dK(uint32_t tmem_dK, void* A_desc, void* B_desc, bool initial_accum) {
    uint32_t base_addr = tmem_dK;
    uint32_t offset = 0;
    uint64_t a_desc = *(uint64_t*)A_desc;
    uint64_t b_desc = *(uint64_t*)B_desc;
    uint32_t d = make_instr_desc_B_BF16<64, 64>();
    d |= (1u << 16);
    
    uint32_t accum = initial_accum ? 1 : 0;
    for (uint32_t i = 0; i < 4; ++i) {
        umma_f16_cg2_fn(base_addr + offset, a_desc, b_desc, d, accum);
        offset += (i % 2 == 0) ? 0 : 8;
        a_desc += (i % 2 == 0) ? 16 : 0;
        b_desc += (i % 2 == 0) ? 16 : 0;
        accum = 1;
    }
}

__device__ void gemm_dQ(uint32_t tmem_dQ, void* A_desc, void* B_desc) {
    uint32_t base_addr = tmem_dQ;
    uint32_t offset = 0;
    uint64_t a_desc = *(uint64_t*)A_desc;
    uint64_t b_desc = *(uint64_t*)B_desc;
    uint32_t d = make_instr_desc_B_BF16<64, 64>();
    uint32_t accum = 0;
    for (uint32_t i = 0; i < 4; ++i) {
        umma_f16_cg2_fn(base_addr + offset, a_desc, b_desc, d, accum);
        offset += (i % 2 == 0) ? 0 : 8;
        a_desc += (i % 2 == 0) ? 16 : 0;
        b_desc += (i % 2 == 0) ? 16 : 0;
        accum = 1;
    }
}

__device__ void gemm_dP_T(uint32_t tmem_dP_T, void* A_desc, void* B_desc) {
    uint32_t base_addr = tmem_dP_T;
    uint32_t offset = 0;
    uint64_t a_desc = *(uint64_t*)A_desc;
    uint64_t b_desc = *(uint64_t*)B_desc;
    uint32_t d = make_instr_desc_B_BF16<64, 64>();
    d |= (1u << 16);
    uint32_t accum = 0;
    for (uint32_t i = 0; i < 4; ++i) {
        umma_f16_cg2_fn(base_addr + offset, a_desc, b_desc, d, accum);
        offset += (i % 2 == 0) ? 0 : 8;
        a_desc += (i % 2 == 0) ? 16 : 0;
        b_desc += (i % 2 == 0) ? 16 : 0;
        accum = 1;
    }
}

__device__ __forceinline__ void store_gmem_from_tmem(int b_h, int n_base, int tid, uint32_t tmem_ptr, __nv_bfloat16* gmem) {
    for (uint32_t col = 0; col < 128; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(tmem_ptr + col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        float f0 = __uint_as_float(r0);
        float f1 = __uint_as_float(r1);
        float f2 = __uint_as_float(r2);
        float f3 = __uint_as_float(r3);
        
        int d_row = n_base + tid;
        if (d_row < S_len && col < 128) {
            gmem[(b_h * S_len + d_row) * 128 + col] = __float2bfloat16(f0);
            gmem[(b_h * S_len + d_row) * 128 + col + 1] = __float2bfloat16(f1);
            gmem[(b_h * S_len + d_row) * 128 + col + 2] = __float2bfloat16(f2);
            gmem[(b_h * S_len + d_row) * 128 + col + 3] = __float2bfloat16(f3);
        }
    }
}

__device__ __forceinline__ void atomic_add_gmem_from_tmem(int b_h, int q_base, int tid, uint32_t tmem_ptr, __nv_bfloat16* gmem) {
    for (uint32_t col = 0; col < 128; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(tmem_ptr + col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        float f0 = __uint_as_float(r0);
        float f1 = __uint_as_float(r1);
        float f2 = __uint_as_float(r2);
        float f3 = __uint_as_float(r3);
        
        int q_row = q_base + tid;
        if (q_row < S_len && col < 128) {
            atomicAdd_bf16(&gmem[(b_h * S_len + q_row) * 128 + col], __float2bfloat16(f0));
            atomicAdd_bf16(&gmem[(b_h * S_len + q_row) * 128 + col + 1], __float2bfloat16(f1));
            atomicAdd_bf16(&gmem[(b_h * S_len + q_row) * 128 + col + 2], __float2bfloat16(f2));
            atomicAdd_bf16(&gmem[(b_h * S_len + q_row) * 128 + col + 3], __float2bfloat16(f3));
        }
    }
}

__device__ __forceinline__ __nv_bfloat16 atomicAdd_bf16(__nv_bfloat16* address, __nv_bfloat16 val) {
    uint32_t* addr32 = (uint32_t*)(((uintptr_t)address) & ~3);
    int offset = (((uintptr_t)address) & 3) / 2;
    uint32_t old = *addr32;
    uint32_t assumed;
    do {
        assumed = old;
        uint16_t old_bf16 = (offset == 0) ? (assumed & 0xFFFF) : (assumed >> 16);
        float f_val = __bfloat162float(val);
        float f_old = __bfloat162float(*reinterpret_cast<float*>(&old_bf16));
        float f_sum = f_old + f_val;
        __nv_bfloat16 sum_bf16 = __float2bfloat16(f_sum);
        uint16_t sum_u16 = *reinterpret_cast<uint16_t*>(&sum_bf16);
        uint32_t new_val = (offset == 0) ? (sum_u16 | (assumed & 0xFFFF0000)) : ((sum_u16 << 16) | (assumed & 0x0000FFFF));
        old = atomicCAS(addr32, assumed, new_val);
    } while (assumed != old);
    return val;
}

__global__ void mha_bwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_dO,
    const float* L,
    __nv_bfloat16* dQ,
    __nv_bfloat16* dK,
    __nv_bfloat16* dV,
    int S_len)
{
    int b_h = blockIdx.y;
    int n_blk = blockIdx.x;
    int n_base = n_blk * 128;

    if (n_base >= S_len) return;

    int tid = threadIdx.x % 128;
    int wg_id = threadIdx.x / 128;
    float attn_scale = 1.0f / sqrtf(128.0f);

    extern __shared__ __align__(128) char smem_raw[];
    __nv_bfloat16* smem_pool = (__nv_bfloat16*)smem_raw;
    
    __nv_bfloat16* smem_Q0 = smem_pool;                
    __nv_bfloat16* smem_Q1 = smem_pool + 4096;         
    __nv_bfloat16* smem_K0 = smem_pool + 8192;         
    __nv_bfloat16* smem_K1 = smem_pool + 12288;        
    __nv_bfloat16* smem_V0 = smem_pool + 16384;        
    __nv_bfloat16* smem_V1 = smem_pool + 20480;        
    __nv_bfloat16* smem_dO = smem_pool + 24576;        
    __nv_bfloat16* smem_O = smem_pool + 32768;         
    __nv_bfloat16* smem_dS = smem_pool + 40960;        
    __nv_bfloat16* smem_P_T = smem_pool + 49152;       
    __nv_bfloat16* smem_dP_T = smem_pool + 57344;      

    __shared__ float smem_D[128];
    __shared__ float smem_LSE[128];

    __shared__ alignas(16) uint64_t mbar_load[2];
    __shared__ alignas(16) uint64_t mbar_S[2];
    __shared__ alignas(16) uint64_t mbar_dP[2];
    __shared__ alignas(16) uint64_t mbar_dV[2];
    __shared__ alignas(16) uint64_t mbar_dK_dQ[2];

    __shared__ uint32_t tmem_base[1];

    if (elect_one_sync_fn()) {
        init_smem_barrier_fn(&mbar_load[0], 1);
        init_smem_barrier_fn(&mbar_S[0], 1);
        init_smem_barrier_fn(&mbar_dP[0], 1);
        init_smem_barrier_fn(&mbar_dV[0], 1);
        init_smem_barrier_fn(&mbar_dK_dQ[0], 1);
        
        tmem_alloc_fn(tmem_base, 512);
    }
    __syncthreads();
    fence_smem_barrier_init_fn();
    __syncthreads();

    uint32_t tmem_S_T = tmem_base[0] + 0;
    uint32_t tmem_P_T = tmem_base[0] + 0;
    uint32_t tmem_dV = tmem_base[0] + 128;
    uint32_t tmem_dP_T = tmem_base[0] + 256;
    uint32_t tmem_dS_T = tmem_base[0] + 256;
    uint32_t tmem_dQ = tmem_base[0] + 256;
    uint32_t tmem_dK = tmem_base[0] + 384;
    uint32_t tmem_dS = tmem_base[0] + 256;

    uint32_t c0 = cluster_rank_fn();
    int phase_load = 0;

    int q_base = 0;
    
    mbarrier_arrive_and_expect_tx_fn(&mbar_load[c0], 16384);
    if (elect_one_sync_fn()) {
        if (c0 == 0) {
            load_tile_tma(c0, mbar_load, smem_Q0, &tma_Q, 0, b_h * S_len + q_base);
            load_tile_tma(c0, mbar_load, smem_Q1, &tma_Q, 64, b_h * S_len + q_base);
            load_tile_tma(c0, mbar_load, smem_K0, &tma_K, 0, b_h * S_len + n_base);
            load_tile_tma(c0, mbar_load, smem_K1, &tma_K, 64, b_h * S_len + n_base);
            load_tile_tma(c0, mbar_load, smem_V0, &tma_V, 0, b_h * S_len + n_base);
            load_tile_tma(c0, mbar_load, smem_V1, &tma_V, 64, b_h * S_len + n_base);
            load_tile_tma(c0, mbar_load, smem_dO, &tma_dO, 0, b_h * S_len + q_base);
            load_tile_tma(c0, mbar_load, smem_O, &tma_O, 0, b_h * S_len + q_base);
        } else {
            load_tile_tma(c0, mbar_load, smem_Q0, &tma_Q, 0, b_h * S_len + q_base);
            load_tile_tma(c0, mbar_load, smem_Q1, &tma_Q, 64, b_h * S_len + q_base);
            load_tile_tma(c0, mbar_load, smem_K0, &tma_K, 0, b_h * S_len + n_base + 64);
            load_tile_tma(c0, mbar_load, smem_K1, &tma_K, 64, b_h * S_len + n_base + 64);
            load_tile_tma(c0, mbar_load, smem_V0, &tma_V, 0, b_h * S_len + n_base + 64);
            load_tile_tma(c0, mbar_load, smem_V1, &tma_V, 64, b_h * S_len + n_base + 64);
            load_tile_tma(c0, mbar_load, smem_dO, &tma_dO, 0, b_h * S_len + q_base);
            load_tile_tma(c0, mbar_load, smem_O, &tma_O, 0, b_h * S_len + q_base);
        }
    }
    
    if (tid < 128) {
        smem_D[tid] = 0;
        smem_LSE[tid] = (q_base + tid < S_len) ? L[b_h * S_len + q_base + tid] : 0;
        int o_row = q_base + tid;
        if (o_row < S_len) {
            for (int col = 0; col < 128; col += 2) {
                uint32_t do_u32 = *(const uint32_t*)(smem_dO + tid * 128 + col);
                uint32_t o_u32  = *(const uint32_t*)(smem_O + tid * 128 + col);
                __nv_bfloat16* do_p = (__nv_bfloat16*)&do_u32;
                __nv_bfloat16* o_p  = (__nv_bfloat16*)&o_u32;
                smem_D[tid] += __bfloat162float(do_p[0]) * __bfloat162float(o_p[0]);
                smem_D[tid] += __bfloat162float(do_p[1]) * __bfloat162float(o_p[1]);
            }
        }
    }
    __syncthreads();
    mbarrier_wait_fn(&mbar_load[c0], phase_load);
    phase_load ^= 1;
    
    mbarrier_arrive_and_expect_tx_fn(&mbar_S[c0], 8192);
    if (elect_one_sync_fn()) {
        uint64_t desc_Q0_K = make_smem_desc_K_major_128B(smem_Q0);
        uint64_t desc_Q1_K = make_smem_desc_K_major_128B(smem_Q1);
        uint64_t desc_K0_N = make_smem_desc_N_major_128B(smem_K0);
        uint64_t desc_K1_N = make_smem_desc_N_major_128B(smem_K1);
        gemm_S(tmem_S_T, &desc_Q0_K, &desc_Q1_K, &desc_K0_N, &desc_K1_N, true);
        umma_commit_2sm_fn(&mbar_S[c0]);
    }
    mbarrier_wait_fn(&mbar_S[c0], 0);
    
    int n_base_c0 = n_base + (c0 == 0 ? 0 : 64);
    for (uint32_t col = 0; col < 128; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(tmem_S_T + col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        float s0 = __uint_as_float(r0);
        float s1 = __uint_as_float(r1);
        float s2 = __uint_as_float(r2);
        float s3 = __uint_as_float(r3);
        
        float p0 = (q_base + tid < S_len) ? expf(s0 * attn_scale - smem_LSE[tid]) : 0;
        float p1 = (q_base + tid < S_len) ? expf(s1 * attn_scale - smem_LSE[tid]) : 0;
        float p2 = (q_base + tid < S_len) ? expf(s2 * attn_scale - smem_LSE[tid]) : 0;
        float p3 = (q_base + tid < S_len) ? expf(s3 * attn_scale - smem_LSE[tid]) : 0;
        
        *(uint32_t*)(smem_P_T + tid * 128 + col) = *(uint32_t*)&__float2bfloat16(p0);
        *(uint32_t*)(smem_P_T + tid * 128 + col + 2) = *(uint32_t*)&__float2bfloat16(p1); 
        *(uint32_t*)(smem_P_T + tid * 128 + col + 4) = *(uint32_t*)&__float2bfloat16(p2);
        *(uint32_t*)(smem_P_T + tid * 128 + col + 6) = *(uint32_t*)&__float2bfloat16(p3);
    }
    
    __syncthreads(); 
    if (elect_one_sync_fn()) {
        copy_to_tmem_swizzled_128B(tmem_P_T, smem_P_T);
    }
    __syncthreads();
    
    mbarrier_arrive_and_expect_tx_fn(&mbar_dP[c0], 8192);
    if (elect_one_sync_fn()) {
        uint64_t desc_V0_N = make_smem_desc_N_major_128B(smem_V0);
        uint64_t desc_V1_N = make_smem_desc_N_major_128B(smem_V1);
        uint64_t desc_dO0_N = make_smem_desc_N_major_128B(smem_dO);
        uint64_t desc_dO1_N = make_smem_desc_N_major_128B(smem_dO + 4096);
        gemm_dP_T(tmem_dP_T, &desc_V0_N, &desc_dO0_N);
        gemm_dP_T(tmem_dP_T + 64, &desc_V1_N, &desc_dO1_N);
        umma_commit_2sm_fn(&mbar_dP[c0]);
    }
    
    mbarrier_arrive_and_expect_tx_fn(&mbar_dV[c0], 8192);
    if (elect_one_sync_fn()) {
        uint64_t desc_P_T_A = make_smem_desc_N_major_128B(smem_P_T);
        uint64_t desc_dO_B = make_smem_desc_N_major_128B(smem_dO);
        gemm_dV_T(tmem_dV, &desc_P_T_A, &desc_dO_B);
        umma_commit_2sm_fn(&mbar_dV[c0]);
    }
    mbarrier_wait_fn(&mbar_dP[c0], 0);
    
    for (uint32_t col = 0; col < 128; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(tmem_dP_T + col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        float dp0 = __uint_as_float(r0);
        float dp1 = __uint_as_float(r1);
        float dp2 = __uint_as_float(r2);
        float dp3 = __uint_as_float(r3);
        
        uint32_t p_u32_0 = *(const uint32_t*)(smem_P_T + col * 128 + tid);
        uint32_t p_u32_1 = *(const uint32_t*)(smem_P_T + col * 128 + tid + 2);
        __nv_bfloat16* p_ptr_0 = (__nv_bfloat16*)&p_u32_0;
        __nv_bfloat16* p_ptr_1 = (__nv_bfloat16*)&p_u32_1;
        
        float p0 = __bfloat162float(p_ptr_0[0]);
        float p1 = __bfloat162float(p_ptr_0[1]);
        float p2 = __bfloat162float(p_ptr_1[0]);
        float p3 = __bfloat162float(p_ptr_1[1]);
        
        float ds0 = p0 * (dp0 - smem_D[col]) * attn_scale;
        float ds1 = p1 * (dp1 - smem_D[col]) * attn_scale;
        float ds2 = p2 * (dp2 - smem_D[col]) * attn_scale;
        float ds3 = p3 * (dp3 - smem_D[col]) * attn_scale;
        
        *(uint32_t*)(smem_dS + tid * 128 + col) = *(uint32_t*)&__float2bfloat16(ds0);
        *(uint32_t*)(smem_dS + tid * 128 + col + 2) = *(uint32_t*)&__float2bfloat16(ds1);
        *(uint32_t*)(smem_dS + tid * 128 + col + 4) = *(uint32_t*)&__float2bfloat16(ds2);
        *(uint32_t*)(smem_dS + tid * 128 + col + 6) = *(uint32_t*)&__float2bfloat16(ds3);
    }
    
    __syncthreads();
    if (elect_one_sync_fn()) {
        copy_to_tmem_swizzled_128B(tmem_dS, smem_dS);
    }
    __syncthreads();
    
    int phase_dK_dQ = 0;
    int phase_dV = 0;
    
    for (int q_blk = 1; q_blk < (S_len + 127) / 128; ++q_blk) {
        q_base = q_blk * 64;
        if (q_base >= S_len) break;
        
        mbarrier_arrive_and_expect_tx_fn(&mbar_load[c0], 16384);
        if (elect_one_sync_fn()) {
            if (c0 == 0) {
                load_tile_tma(c0, mbar_load, smem_Q0, &tma_Q, 0, b_h * S_len + q_base);
                load_tile_tma(c0, mbar_load, smem_Q1, &tma_Q, 64, b_h * S_len + q_base);
                load_tile_tma(c0, mbar_load, smem_dO, &tma_dO, 0, b_h * S_len + q_base);
                load_tile_tma(c0, mbar_load, smem_O, &tma_O, 0, b_h * S_len + q_base);
            } else {
                load_tile_tma(c0, mbar_load, smem_Q0, &tma_Q, 0, b_h * S_len + q_base);
                load_tile_tma(c0, mbar_load, smem_Q1, &tma_Q, 64, b_h * S_len + q_base);
                load_tile_tma(c0, mbar_load, smem_dO, &tma_dO, 0, b_h * S_len + q_base);
                load_tile_tma(c0, mbar_load, smem_O, &tma_O, 0, b_h * S_len + q_base);
            }
        }
        
        if (tid < 128) {
            smem_D[tid] = 0;
            smem_LSE[tid] = (q_base + tid < S_len) ? L[b_h * S_len + q_base + tid] : 0;
            int o_row = q_base + tid;
            if (o_row < S_len) {
                for (int col = 0; col < 128; col += 2) {
                    uint32_t do_u32 = *(const uint32_t*)(smem_dO + tid * 128 + col);
                    uint32_t o_u32  = *(const uint32_t*)(smem_O + tid * 128 + col);
                    __nv_bfloat16* do_p = (__nv_bfloat16*)&do_u32;
                    __nv_bfloat16* o_p  = (__nv_bfloat16*)&o_u32;
                    smem_D[tid] += __bfloat162float(do_p[0]) * __bfloat162float(o_p[0]);
                    smem_D[tid] += __bfloat162float(do_p[1]) * __bfloat162float(o_p[1]);
                }
            }
        }
        __syncthreads();
        
        mbarrier_arrive_and_expect_tx_fn(&mbar_dK_dQ[c0], 8192);
        if (elect_one_sync_fn()) {
            uint64_t desc_dS_T_A = make_smem_desc_N_major_128B(smem_dS);
            uint64_t desc_Q0_B = make_smem_desc_K_major_128B(smem_Q0);
            gemm_dK(tmem_dK, &desc_dS_T_A, &desc_Q0_B);
            
            uint64_t desc_dS_A = make_smem_desc_K_major_128B(smem_dS);
            uint64_t desc_K0_B = make_smem_desc_K_major_128B(smem_K0);
            gemm_dQ(tmem_dQ, &desc_dS_A, &desc_K0_B);
            
            umma_commit_2sm_fn(&mbar_dK_dQ[c0]);
        }
        
        if (q_blk >= 2) {
            mbarrier_wait_fn(&mbar_dK_dQ[c0], phase_dK_dQ);
            phase_dK_dQ ^= 1;
            store_gmem_from_tmem(b_h, n_base_c0, tid, tmem_dK, dK);
            atomic_add_gmem_from_tmem(b_h, q_base - 64, tid, tmem_dQ, dQ);
        }
        
        mbarrier_wait_fn(&mbar_load[c0], phase_load);
        phase_load ^= 1;
        
        mbarrier_arrive_and_expect_tx_fn(&mbar_S[c0], 8192);
        if (elect_one_sync_fn()) {
            uint64_t desc_Q0_K = make_smem_desc_K_major_128B(smem_Q0);
            uint64_t desc_Q1_K = make_smem_desc_K_major_128B(smem_Q1);
            uint64_t desc_K0_N = make_smem_desc_N_major_128B(smem_K0);
            uint64_t desc_K1_N = make_smem_desc_N_major_128B(smem_K1);
            gemm_S(tmem_S_T, &desc_Q0_K, &desc_Q1_K, &desc_K0_N, &desc_K1_N, false);
            umma_commit_2sm_fn(&mbar_S[c0]);
        }
        mbarrier_wait_fn(&mbar_S[c0], phase_dK_dQ);
        
        for (uint32_t col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(tmem_S_T + col));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            float s0 = __uint_as_float(r0);
            float s1 = __uint_as_float(r1);
            float s2 = __uint_as_float(r2);
            float s3 = __uint_as_float(r3);
            
            float p0 = (q_base + tid < S_len) ? expf(s0 * attn_scale - smem_LSE[tid]) : 0;
            float p1 = (q_base + tid < S_len) ? expf(s1 * attn_scale - smem_LSE[tid]) : 0;
            float p2 = (q_base + tid < S_len) ? expf(s2 * attn_scale - smem_LSE[tid]) : 0;
            float p3 = (q_base + tid < S_len) ? expf(s3 * attn_scale - smem_LSE[tid]) : 0;
            
            *(uint32_t*)(smem_P_T + tid * 128 + col) = *(uint32_t*)&__float2bfloat16(p0);
            *(uint32_t*)(smem_P_T + tid * 128 + col + 2) = *(uint32_t*)&__float2bfloat16(p1);
            *(uint32_t*)(smem_P_T + tid * 128 + col + 4) = *(uint32_t*)&__float2bfloat16(p2);
            *(uint32_t*)(smem_P_T + tid * 128 + col + 6) = *(uint32_t*)&__float2bfloat16(p3);
        }
        
        __syncthreads();
        if (elect_one_sync_fn()) {
            copy_to_tmem_swizzled_128B(tmem_P_T, smem_P_T);
        }
        __syncthreads();
        
        mbarrier_arrive_and_expect_tx_fn(&mbar_dP[c0], 8192);
        if (elect_one_sync_fn()) {
            uint64_t desc_V0_N = make_smem_desc_N_major_128B(smem_V0);
            uint64_t desc_V1_N = make_smem_desc_N_major_128B(smem_V1);
            uint64_t desc_dO0_N = make_smem_desc_N_major_128B(smem_dO);
            uint64_t desc_dO1_N = make_smem_desc_N_major_128B(smem_dO + 4096);
            gemm_dP_T(tmem_dP_T, &desc_V0_N, &desc_dO0_N);
            gemm_dP_T(tmem_dP_T + 64, &desc_V1_N, &desc_dO1_N);
            umma_commit_2sm_fn(&mbar_dP[c0]);
        }
        
        mbarrier_arrive_and_expect_tx_fn(&mbar_dV[c0], 8192);
        if (elect_one_sync_fn()) {
            uint64_t desc_P_T_A = make_smem_desc_N_major_128B(smem_P_T);
            uint64_t desc_dO_B = make_smem_desc_N_major_128B(smem_dO);
            gemm_dV_T(tmem_dV, &desc_P_T_A, &desc_dO_B);
            umma_commit_2sm_fn(&mbar_dV[c0]);
        }
        mbarrier_wait_fn(&mbar_dP[c0], phase_dV);
        phase_dV ^= 1;
        
        for (uint32_t col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(tmem_dP_T + col));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            float dp0 = __uint_as_float(r0);
            float dp1 = __uint_as_float(r1);
            float dp2 = __uint_as_float(r2);
            float dp3 = __uint_as_float(r3);
            
            uint32_t p_u32_0 = *(const uint32_t*)(smem_P_T + col * 128 + tid);
            uint32_t p_u32_1 = *(const uint32_t*)(smem_P_T + col * 128 + tid + 2);
            __nv_bfloat16* p_ptr_0 = (__nv_bfloat16*)&p_u32_0;
            __nv_bfloat16* p_ptr_1 = (__nv_bfloat16*)&p_u32_1;
            
            float p0 = __bfloat162float(p_ptr_0[0]);
            float p1 = __bfloat162float(p_ptr_0[1]);
            float p2 = __bfloat162float(p_ptr_1[0]);
            float p3 = __bfloat162float(p_ptr_1[1]);
            
            float ds0 = p0 * (dp0 - smem_D[col]) * attn_scale;
            float ds1 = p1 * (dp1 - smem_D[col]) * attn_scale;
            float ds2 = p2 * (dp2 - smem_D[col]) * attn_scale;
            float ds3 = p3 * (dp3 - smem_D[col]) * attn_scale;
            
            *(uint32_t*)(smem_dS + tid * 128 + col) = *(uint32_t*)&__float2bfloat16(ds0);
            *(uint32_t*)(smem_dS + tid * 128 + col + 2) = *(uint32_t*)&__float2bfloat16(ds1);
            *(uint32_t*)(smem_dS + tid * 128 + col + 4) = *(uint32_t*)&__float2bfloat16(ds2);
            *(uint32_t*)(smem_dS + tid * 128 + col + 6) = *(uint32_t*)&__float2bfloat16(ds3);
        }
        
        __syncthreads();
        if (elect_one_sync_fn()) {
            copy_to_tmem_swizzled_128B(tmem_dS, smem_dS);
        }
        __syncthreads();
    }
    
    mbarrier_arrive_and_expect_tx_fn(&mbar_dK_dQ[c0], 8192);
    if (elect_one_sync_fn()) {
        uint64_t desc_dS_T_A = make_smem_desc_N_major_128B(smem_dS);
        uint64_t desc_Q0_B = make_smem_desc_K_major_128B(smem_Q0);
        gemm_dK(tmem_dK, &desc_dS_T_A, &desc_Q0_B);
        
        uint64_t desc_dS_A = make_smem_desc_K_major_128B(smem_dS);
        uint64_t desc_K0_B = make_smem_desc_K_major_128B(smem_K0);
        gemm_dQ(tmem_dQ, &desc_dS_A, &desc_K0_B);
        
        umma_commit_2sm_fn(&mbar_dK_dQ[c0]);
    }
    mbarrier_wait_fn(&mbar_dK_dQ[c0], phase_dK_dQ);
    
    store_gmem_from_tmem(b_h, n_base_c0, tid, tmem_dK, dK);
    atomic_add_gmem_from_tmem(b_h, q_base, tid, tmem_dQ, dQ);
    store_gmem_from_tmem(b_h, n_base_c0, tid, tmem_dV, dV);

    if (elect_one_sync_fn()) {
        tmem_dealloc_fn(tmem_base[0], 512);
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

    if (d != 128) {
        fprintf(stderr, "Expected head dim 128, got %ld\n", d);
        exit(1);
    }

    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* O_data = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dO_data = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_data = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQ_data = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_data = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_data = static_cast<__nv_bfloat16*>(dV.data_ptr());

    CUDA_CHECK(cudaMemsetAsync(dQ_data, 0, B * H * S * 128 * sizeof(__nv_bfloat16)));

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUtensorMap tma_Q, tma_K, tma_V, tma_O, tma_dO;

    auto create_tma = [&](CUtensorMap* tma, const void* ptr) {
        cuuint64_t globalDim[2] = {128, (cuuint64_t)(B * H * S)};
        cuuint64_t globalStrides[1] = {128 * 2};
        cuuint32_t boxDim[2] = {64, 64};
        cuuint32_t elementStrides[2] = {1, 1};
        CUresult res = cuTensorMapEncodeTiled(
            tma, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)ptr, globalDim, globalStrides, boxDim, elementStrides,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );
        if (res != CUDA_SUCCESS) {
            fprintf(stderr, "cuTensorMapEncodeTiled failed with %d\n", res);
            exit(1);
        }
    };

    create_tma(&tma_Q, Q_data);
    create_tma(&tma_K, K_data);
    create_tma(&tma_V, V_data);
    create_tma(&tma_O, O_data);
    create_tma(&tma_dO, dO_data);

    int num_tiles = (S + 127) / 128;
    dim3 grid(num_tiles, B * H);
    dim3 block(128);
    
    int smem_size = 104448;
    CUDA_CHECK(cudaFuncSetAttribute(tvm_ffi_mha_bwd::mha_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    CUDA_CHECK(cudaLaunchKernelEx(&config, tvm_ffi_mha_bwd::mha_bwd_kernel,
        tma_Q, tma_K, tma_V, tma_O, tma_dO, L_data,
        dQ_data, dK_data, dV_data, S
    ));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_bwd::run);

} // namespace tvm_ffi_mha_bwd