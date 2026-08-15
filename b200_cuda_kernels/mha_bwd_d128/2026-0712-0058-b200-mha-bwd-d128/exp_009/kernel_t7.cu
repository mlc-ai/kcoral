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

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(tx_bytes) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(phase));
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    uint32_t sa = (uint32_t)__cvta_generic_to_shared(smem);
    uint32_t ba = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(sa), "l"((uint64_t)d), "r"(c0), "r"(c1), "r"(ba) : "memory");
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

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
       : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void umma_f16_ctg2(uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void commit_umma(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile(
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64 [%0];"
        :: "r"(a));
}

__device__ __forceinline__ uint64_t make_smem_desc(void* smem_ptr) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint64_t d = 0;
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((16 & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((1024 & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;
    return d;
}

template <uint32_t M, uint32_t N>
__device__ __forceinline__ uint32_t make_instr_desc() {
    uint32_t d = 0;
    d |= (1u << 4);
    d |= (1u << 7);
    d |= (1u << 10);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__device__ void gemm_S(uint32_t tmem_D, void* A0, void* A1, void* B0, void* B1, bool initial_accum) {
    uint32_t d = make_instr_desc<64, 64>();
    uint32_t accum = initial_accum ? 1 : 0;
    for (int k = 0; k < 64; k += 16) {
        uint64_t desc_A = make_smem_desc((char*)A0 + k * 2);
        uint64_t desc_B = make_smem_desc((char*)B0 + k * 2);
        umma_f16_ctg2(tmem_D, desc_A, desc_B, d, accum);
        accum = 1;
    }
    for (int k = 0; k < 64; k += 16) {
        uint64_t desc_A = make_smem_desc((char*)A1 + k * 2);
        uint64_t desc_B = make_smem_desc((char*)B1 + k * 2);
        umma_f16_ctg2(tmem_D, desc_A, desc_B, d, accum);
        accum = 1;
    }
}

__device__ void gemm_dP(uint32_t tmem_D, void* A0, void* A1, void* B0, void* B1, bool initial_accum) {
    uint32_t d = make_instr_desc<64, 64>();
    d |= (1u << 16);
    uint32_t accum = initial_accum ? 1 : 0;
    for (int k = 0; k < 64; k += 16) {
        uint64_t desc_A = make_smem_desc((char*)A0 + k * 2);
        uint64_t desc_B = make_smem_desc((char*)B0 + k * 2);
        umma_f16_ctg2(tmem_D, desc_A, desc_B, d, accum);
        accum = 1;
    }
    for (int k = 0; k < 64; k += 16) {
        uint64_t desc_A = make_smem_desc((char*)A1 + k * 2);
        uint64_t desc_B = make_smem_desc((char*)B1 + k * 2);
        umma_f16_ctg2(tmem_D, desc_A, desc_B, d, accum);
        accum = 1;
    }
}

__device__ void gemm_dV(uint32_t tmem_D0, uint32_t tmem_D1, void* A, void* B0, void* B1, bool initial_accum) {
    uint32_t d = make_instr_desc<64, 64>();
    d |= (1u << 16);
    
    uint32_t accum = initial_accum ? 1 : 0;
    for (int k = 0; k < 64; k += 16) {
        uint64_t desc_A = make_smem_desc((char*)A + k * 2);
        uint64_t desc_B = make_smem_desc((char*)B0 + k * 2);
        umma_f16_ctg2(tmem_D0, desc_A, desc_B, d, accum);
        accum = 1;
    }
    
    accum = initial_accum ? 1 : 0;
    for (int k = 0; k < 64; k += 16) {
        uint64_t desc_A = make_smem_desc((char*)A + k * 2);
        uint64_t desc_B = make_smem_desc((char*)B1 + k * 2);
        umma_f16_ctg2(tmem_D1, desc_A, desc_B, d, accum);
        accum = 1;
    }
}

__device__ void gemm_dK(uint32_t tmem_D0, uint32_t tmem_D1, void* A, void* B0, void* B1, bool initial_accum) {
    uint32_t d = make_instr_desc<64, 64>();
    d |= (1u << 16);
    
    uint32_t accum = initial_accum ? 1 : 0;
    for (int k = 0; k < 64; k += 16) {
        uint64_t desc_A = make_smem_desc((char*)A + k * 2);
        uint64_t desc_B = make_smem_desc((char*)B0 + k * 2);
        umma_f16_ctg2(tmem_D0, desc_A, desc_B, d, accum);
        accum = 1;
    }
    
    accum = initial_accum ? 1 : 0;
    for (int k = 0; k < 64; k += 16) {
        uint64_t desc_A = make_smem_desc((char*)A + k * 2);
        uint64_t desc_B = make_smem_desc((char*)B1 + k * 2);
        umma_f16_ctg2(tmem_D1, desc_A, desc_B, d, accum);
        accum = 1;
    }
}

__device__ void gemm_dQ(uint32_t tmem_D0, uint32_t tmem_D1, void* A, void* B0, void* B1, bool initial_accum) {
    uint32_t d = make_instr_desc<64, 64>();
    d |= (1u << 16);
    
    uint32_t accum = initial_accum ? 1 : 0;
    for (int k = 0; k < 64; k += 16) {
        uint64_t desc_A = make_smem_desc((char*)A + k * 2);
        uint64_t desc_B = make_smem_desc((char*)B0 + k * 2);
        umma_f16_ctg2(tmem_D0, desc_A, desc_B, d, accum);
        accum = 1;
    }
    
    accum = initial_accum ? 1 : 0;
    for (int k = 0; k < 64; k += 16) {
        uint64_t desc_A = make_smem_desc((char*)A + k * 2);
        uint64_t desc_B = make_smem_desc((char*)B1 + k * 2);
        umma_f16_ctg2(tmem_D1, desc_A, desc_B, d, accum);
        accum = 1;
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
        float f_old = __bfloat162float(old_bf16);
        float f_sum = f_old + f_val;
        __nv_bfloat16 sum_bf16 = __float2bfloat16(f_sum);
        uint16_t sum_u16 = *reinterpret_cast<uint16_t*>(&sum_bf16);
        uint32_t new_val = (offset == 0) ? (sum_u16 | (assumed & 0xFFFF0000)) : ((sum_u16 << 16) | (assumed & 0x0000FFFF));
        old = atomicCAS((uint32_t*)addr32, assumed, new_val);
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
    int n_base = n_blk * 64;
    int tid = threadIdx.x;

    if (n_base >= S_len) return;

    extern __shared__ __align__(1024) __nv_bfloat16 smem_pool[];
    __nv_bfloat16* smem_Q0 = smem_pool;                
    __nv_bfloat16* smem_Q1 = smem_pool + 4096;         
    __nv_bfloat16* smem_K0 = smem_pool + 8192;         
    __nv_bfloat16* smem_K1 = smem_pool + 12288;        
    __nv_bfloat16* smem_V0 = smem_pool + 16384;        
    __nv_bfloat16* smem_V1 = smem_pool + 20480;        
    __nv_bfloat16* smem_dO0 = smem_pool + 24576;       
    __nv_bfloat16* smem_dO1 = smem_pool + 28672;       
    __nv_bfloat16* smem_O0 = smem_pool + 32768;        
    __nv_bfloat16* smem_O1 = smem_pool + 36864;        
    __nv_bfloat16* smem_PT = smem_pool + 40960;        
    __nv_bfloat16* smem_dPT = smem_pool + 45056;       
    __nv_bfloat16* smem_dST = smem_pool + 49152;       
    __nv_bfloat16* smem_dS = smem_pool + 53248;        

    __shared__ alignas(16) uint64_t mbar_load[2];
    __shared__ alignas(16) uint64_t tmbar[2];

    if (elect_one_sync_fn()) {
        init_smem_barrier_fn(&mbar_load[0], 1);
        init_smem_barrier_fn(&mbar_load[1], 1);
        init_smem_barrier_fn(&tmbar[0], 1);
        init_smem_barrier_fn(&tmbar[1], 1);
        __shared__ uint32_t tmem_base[1];
        tmem_alloc_fn(tmem_base, 512);
    }
    __syncthreads();
    fence_smem_barrier_init_fn();
    __syncthreads();
    
    __shared__ uint32_t tmem_base[1];
    uint32_t tmem_S = tmem_base[0] + 0;
    uint32_t tmem_dP = tmem_base[0] + 64;
    uint32_t tmem_dV0 = tmem_base[0] + 128;
    uint32_t tmem_dV1 = tmem_base[0] + 192;
    uint32_t tmem_dK0 = tmem_base[0] + 256;
    uint32_t tmem_dK1 = tmem_base[0] + 320;
    uint32_t tmem_dQ0 = tmem_base[0] + 384;
    uint32_t tmem_dQ1 = tmem_base[0] + 448;

    int phase_load = 0;
    int phase_tmbar = 0;

    mbarrier_arrive_and_expect_tx_fn(&mbar_load[0], 16384);
    if (tid == 0) {
        tma_load_2d_fn(&tma_K, &mbar_load[0], smem_K0, 0, b_h * S_len + n_base);
        tma_load_2d_fn(&tma_K, &mbar_load[0], smem_K1, 64, b_h * S_len + n_base);
        tma_load_2d_fn(&tma_V, &tma_V, &mbar_load[0], smem_V0, 0, b_h * S_len + n_base);
        tma_load_2d_fn(&tma_V, &mbar_load[0], smem_V1, 64, b_h * S_len + n_base);
    }
    mbarrier_wait_fn(&mbar_load[0], phase_load);
    phase_load ^= 1;
    __syncthreads();
    
    float attn_scale = 1.0f / sqrtf(128.0f);

    for (int q_blk = 0; q_blk < (S_len + 63) / 64; ++q_blk) {
        int q_base = q_blk * 64;
        if (q_base >= S_len) break;
        
        mbarrier_arrive_and_expect_tx_fn(&mbar_load[0], 16384);
        if (tid == 0) {
            tma_load_2d_fn(&tma_Q, &mbar_load[0], smem_Q0, 0, b_h * S_len + q_base);
            tma_load_2d_fn(&tma_Q, &mbar_load[0], smem_Q1, 64, b_h * S_len + q_base);
            tma_load_2d_fn(&tma_O, &mbar_load[0], smem_O0, 0, b_h * S_len + q_base);
            tma_load_2d_fn(&tma_O, &mbar_load[0], smem_O1, 64, b_h * S_len + q_base);
            tma_load_2d_fn(&tma_dO, &mbar_load[0], smem_dO0, 0, b_h * S_len + q_base);
            tma_load_2d_fn(&tma_dO, &mbar_load[0], smem_dO1, 64, b_h * S_len + q_base);
        }
        
        __shared__ float D_shared[64];
        __shared__ float LSE_shared[64];
        
        if (tid < 64) {
            D_shared[tid] = 0;
            LSE_shared[tid] = (q_base + tid < S_len) ? L[b_h * S_len + q_base + tid] : 0;
            for(int col = 0; col < 64; col++) {
                int s_col = ((col / 8) ^ (tid % 8)) * 8 + (col % 8);
                D_shared[tid] += __bfloat162float(smem_dO0[tid * 64 + s_col]) * __bfloat162float(smem_O0[tid * 64 + s_col]);
                D_shared[tid] += __bfloat162float(smem_dO1[tid * 64 + s_col]) * __bfloat162float(smem_O1[tid * 64 + s_col]);
            }
        }
        __syncthreads();
        
        mbarrier_wait_fn(&mbar_load[0], phase_load);
        phase_load ^= 1;
        __syncthreads();
        
        mbarrier_arrive_and_expect_tx_fn(&tmbar[0], 8192);
        if (tid == 0) {
            gemm_S(tmem_S, smem_Q0, smem_Q1, smem_K0, smem_K1, true);
            commit_umma(&tmbar[0]);
        }
        mbarrier_wait_fn(&tmbar[0], phase_tmbar);
        phase_tmbar ^= 1;
        __syncthreads();
        
        uint32_t r0, r1, r2, r3;
        tmem_load_4x_fn(tmem_S + tid * 4, &r0, &r1, &r2, &r3);
        tmem_load_fence_fn();
        
        float s0 = __uint_as_float(r0);
        float s1 = __uint_as_float(r1);
        float s2 = __uint_as_float(r2);
        float s3 = __uint_as_float(r3);
        
        float p0 = (q_base + tid < S_len) ? expf(s0 * attn_scale - LSE_shared[tid]) : 0;
        float p1 = (q_base + tid < S_len) ? expf(s1 * attn_scale - LSE_shared[tid]) : 0;
        float p2 = (q_base + tid < S_len) ? expf(s2 * attn_scale - LSE_shared[tid]) : 0;
        float p3 = (q_base + tid < S_len) ? expf(s3 * attn_scale - LSE_shared[tid]) : 0;
        
        for (int col = 0; col < 64; col += 4) {
            int s_col = ((col / 8) ^ (tid % 8)) * 8 + (col % 8);
            __nv_bfloat16 bp0 = __float2bfloat16(p0);
            __nv_bfloat16 bp1 = __float2bfloat16(p1);
            __nv_bfloat16 bp2 = __float2bfloat16(p2);
            __nv_bfloat16 bp3 = __float2bfloat16(p3);
            *(uint32_t*)(smem_PT + tid * 64 + s_col) = *(uint32_t*)&bp0;
            *(uint32_t*)(smem_PT + tid * 64 + s_col + 2) = *(uint32_t*)&bp1;
            *(uint32_t*)(smem_PT + tid * 64 + s_col + 4) = *(uint32_t*)&bp2;
            *(uint32_t*)(smem_PT + tid * 64 + s_col + 6) = *(uint32_t*)&bp3;
        }
        
        mbarrier_arrive_and_expect_tx_fn(&tmbar[0], 8192);
        if (tid == 0) {
            gemm_dP(tmem_dP, smem_dO0, smem_dO1, smem_V0, smem_V1, true);
            commit_umma(&tmbar[0]);
        }
        mbarrier_wait_fn(&tmbar[0], phase_tmbar);
        phase_tmbar ^= 1;
        __syncthreads();
        
        tmem_load_4x_fn(tmem_dP + tid * 4, &r0, &r1, &r2, &r3);
        tmem_load_fence_fn();
        
        float dp0 = __uint_as_float(r0);
        float dp1 = __uint_as_float(r1);
        float dp2 = __uint_as_float(r2);
        float dp3 = __uint_as_float(r3);
        
        float ds0 = p0 * (dp0 - D_shared[tid]) * attn_scale;
        float ds1 = p1 * (dp1 - D_shared[tid]) * attn_scale;
        float ds2 = p2 * (dp2 - D_shared[tid]) * attn_scale;
        float ds3 = p3 * (dp3 - D_shared[tid]) * attn_scale;
        
        for (int col = 0; col < 64; col += 4) {
            int s_col = ((col / 8) ^ (tid % 8)) * 8 + (col % 8);
            __nv_bfloat16 bds0 = __float2bfloat16(ds0);
            __nv_bfloat16 bds1 = __float2bfloat16(ds1);
            __nv_bfloat16 bds2 = __float2bfloat16(ds2);
            __nv_bfloat16 bds3 = __float2bfloat16(ds3);
            
            *(uint32_t*)(smem_dST + tid * 64 + s_col) = *(uint32_t*)&bds0;
            *(uint32_t*)(smem_dST + tid * 64 + s_col + 2) = *(uint32_t*)&bds1;
            *(uint32_t*)(smem_dST + tid * 64 + s_col + 4) = *(uint32_t*)&bds2;
            *(uint32_t*)(smem_dST + tid * 64 + s_col + 6) = *(uint32_t*)&bds3;
            
            *(uint32_t*)(smem_dS + col * 64 + s_col) = *(uint32_t*)&bds0;
            *(uint32_t*)(smem_dS + col * 64 + s_col + 2) = *(uint32_t*)&bds1;
            *(uint32_t*)(smem_dS + col * 64 + s_col + 4) = *(uint32_t*)&bds2;
            *(uint32_t*)(smem_dS + col * 64 + s_col + 6) = *(uint32_t*)&bds3;
        }
        __syncthreads();
        
        mbarrier_arrive_and_expect_tx_fn(&tmbar[0], 8192);
        if (tid == 0) {
            gemm_dV(tmem_dV0, tmem_dV1, smem_PT, smem_dO0, smem_dO1, q_blk == 0);
            commit_umma(&tmbar[0]);
        }
        
        mbarrier_arrive_and_expect_tx_fn(&tmbar[1], 8192);
        if (tid == 0) {
            gemm_dK(tmem_dK0, tmem_dK1, smem_dST, smem_Q0, smem_Q1, q_blk == 0);
            commit_umma(&tmbar[1]);
        }
        
        mbarrier_arrive_and_expect_tx_fn(&tmbar[0], 8192);
        if (tid == 0) {
            gemm_dQ(tmem_dQ0, tmem_dQ1, smem_dS, smem_K0, smem_K1, false);
            commit_umma(&tmbar[0]);
        }
        
        mbarrier_wait_fn(&tmbar[0], phase_tmbar);
        mbarrier_wait_fn(&tmbar[1], phase_tmbar);
        phase_tmbar ^= 1;
        __syncthreads();
        
        for (int col = 0; col < 64; col += 4) {
            tmem_load_4x_fn(tmem_dQ0 + col, &r0, &r1, &r2, &r3);
            tmem_load_fence_fn();
            int q_row = q_base + tid;
            if (q_row < S_len && col < 64) {
                atomicAdd_bf16(&dQ[(b_h * S_len + q_row) * 128 + col], __float2bfloat16(__uint_as_float(r0)));
                atomicAdd_bf16(&dQ[(b_h * S_len + q_row) * 128 + col + 1], __float2bfloat16(__uint_as_float(r1)));
                atomicAdd_bf16(&dQ[(b_h * S_len + q_row) * 128 + col + 2], __float2bfloat16(__uint_as_float(r2)));
                atomicAdd_bf16(&dQ[(b_h * S_len + q_row) * 128 + col + 3], __float2bfloat16(__uint_as_float(r3)));
            }
        }
        
        for (int col = 0