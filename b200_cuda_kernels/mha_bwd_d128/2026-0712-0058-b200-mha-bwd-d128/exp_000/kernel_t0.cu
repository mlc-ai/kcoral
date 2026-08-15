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

__device__ __forceinline__ void tmem_store_bf16_row_fn(
    __nv_bfloat16* D, uint32_t tid, uint32_t M, uint32_t N,
    uint32_t m_base, uint32_t n_base, uint32_t BN) {
    uint32_t m_idx = m_base + tid;
    if (m_idx >= M) return;
    for (uint32_t col = 0; col < BN; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        float f0 = __uint_as_float(r0);
        float f1 = __uint_as_float(r1);
        float f2 = __uint_as_float(r2);
        float f3 = __uint_as_float(r3);
        uint32_t nc = n_base + col;
        __nv_bfloat16* out = D + (uint64_t)m_idx * N + nc;
        if (nc     < N) out[0] = __float2bfloat16(f0);
        if (nc + 1 < N) out[1] = __float2bfloat16(f1);
        if (nc + 2 < N) out[2] = __float2bfloat16(f2);
        if (nc + 3 < N) out[3] = __float2bfloat16(f3);
    }
}

__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

__device__ __forceinline__ void gemm_64x64x16_cg2(uint32_t tmem_addr, uint64_t desc_a, uint64_t desc_b, uint32_t idesc) {
    if (threadIdx.x == 0) {
        uint32_t accum = 1; 
        umma_f16_cg2_fn(tmem_addr, desc_a, desc_b, idesc, accum);
    }
}

__global__ void mha_bwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_dO,
    const __grid_constant__ CUtensorMap tma_dQ,
    const __grid_constant__ CUtensorMap tma_dK,
    const __grid_constant__ CUtensorMap tma_dV,
    const float* L_raw, int S, float scale)
{
    extern __shared__ char smem_pool[];
    uint64_t* mbar_load = (uint64_t*)smem_pool;
    uint64_t* mbar_smem = (uint64_t*)(smem_pool + 8);
    
    char* buf_ptr = (char*)((uintptr_t)(smem_pool + 127) & ~127);
    
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
    uint8_t* s_dP = s_V0; 
    uint8_t* s_dS = s_V1; 
    uint8_t* s_D = s_P + 8192; 
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_load, 1);
        init_smem_barrier_fn(mbar_smem, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    uint32_t tmem_c[1], tmem_dp[1], tmem_dv0[1], tmem_dv1[1], tmem_dq0[1], tmem_dq1[1], tmem_dk0[1], tmem_dk1[1];
    if (threadIdx.x == 0) {
        tmem_alloc_fn(tmem_c, 64);
        tmem_alloc_fn(tmem_dp, 64);
        tmem_alloc_fn(tmem_dv0, 64);
        tmem_alloc_fn(tmem_dv1, 64);
        tmem_alloc_fn(tmem_dq0, 64);
        tmem_alloc_fn(tmem_dq1, 64);
        tmem_alloc_fn(tmem_dk0, 64);
        tmem_alloc_fn(tmem_dk1, 64);
    }
    __syncthreads();

    uint32_t s_block = blockIdx.x * 96;
    uint32_t batch_head = blockIdx.y;
    int32_t c1 = batch_head * S + s_block;
    
    uint32_t phase = 0;
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_load, 10 * 8192); 
        
        tma_load_2d_fn(&tma_Q, mbar_load, s_Q0, 0, c1);
        tma_load_2d_fn(&tma_Q, mbar_load, s_Q1, 64, c1);
        tma_load_2d_fn(&tma_K, mbar_load, s_K0, 0, c1);
        tma_load_2d_fn(&tma_K, mbar_load, s_K1, 64, c1);
        
        tma_load_2d_fn(&tma_V, mbar_load, s_V0, 0, c1);
        tma_load_2d_fn(&tma_V, mbar_load, s_V1, 64, c1);
        
        tma_load_2d_fn(&tma_O, mbar_load, s_O0, 0, c1);
        tma_load_2d_fn(&tma_O, mbar_load, s_O1, 64, c1);
        
        tma_load_2d_fn(&tma_dO, mbar_load, s_dO0, 0, c1);
        tma_load_2d_fn(&tma_dO, mbar_load, s_dO1, 64, c1);
    }
    mbarrier_wait_fn(mbar_load, phase);
    __syncthreads();
    phase ^= 1;

    float sd = 1.0f / sqrtf(S);

    // ---------------- Compute P = Q @ K^T ----------------
    if (threadIdx.x == 0) {
        uint32_t accum_c = 0;
        if (cluster_rank_fn() == 0) {
            uint64_t desc_Q0 = make_smem_desc_sm100_fn(s_Q0, 1, 1024);
            uint64_t desc_K0 = make_smem_desc_sm100_fn(s_K0, 1, 1024);
            uint32_t id_Q0_K0 = make_instr_desc_cg2(64, 64);
            gemm_64x64x16_cg2(tmem_c[0], desc_Q0, desc_K0, id_Q0_K0);
        } else {
            uint64_t desc_Q1 = make_smem_desc_sm100_fn(s_Q1, 1, 1024);
            uint64_t desc_K1 = make_smem_desc_sm100_fn(s_K1, 1, 1024);
            uint32_t id_Q1_K1 = make_instr_desc_cg2(64, 64);
            gemm_64x64x16_cg2(tmem_c[0], desc_Q1, desc_K1, id_Q1_K1);
        }
    }
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    asm volatile("barrier.sync.aligned 2, 128;" ::: "memory");
    tmem_store_bf16_row_fn((__nv_bfloat16*)s_P, threadIdx.x, S, S, s_block, s_block, S);
    asm volatile("barrier.sync.aligned 2, 128;" ::: "memory");

    // ---------------- Correct P with LSE ----------------
    uint32_t tid = threadIdx.x % 64;
    uint32_t s_idx_base = batch_head * S + s_block;
    float l_q = (tid < S) ? L_raw[s_idx_base + tid] : 0.0f;
    float l_k = (tid < S) ? L_raw[s_idx_base + tid] : 0.0f;

    for (int c = 0; c < 64; c += 2) {
        uint32_t addr = (tid * 64 + c) * 2;
        __nv_bfloat16 p0 = *(__nv_bfloat16*)(s_P + addr);
        __nv_bfloat16 p1 = *(__nv_bfloat16*)(s_P + addr + 2);
        float val0 = __bfloat162float(p0) * sd;
        float val1 = __bfloat162float(p1) * sd;
        float lse_diff = l_k - l_q; 
        *(uint32_t*)(s_P + addr) = *(uint32_t*)&pack_bf16_fn(__float_as_uint(val0 * expf(lse_diff)), __float_as_uint(val1 * expf(lse_diff)));
    }
    __syncthreads();

    // ---------------- Compute dP = dO @ V^T ----------------
    if (threadIdx.x == 0) {
        uint32_t accum_dp = 0;
        if (cluster_rank_fn() == 0) {
            uint64_t desc_dO = make_smem_desc_sm100_fn(s_dO0, 1, 1024);
            uint64_t desc_V  = make_smem_desc_sm100_fn(s_V0, 1, 1024);
            uint32_t id_dP = make_instr_desc_cg2(64, 64);
            gemm_64x64x16_cg2(tmem_dp[0], desc_dO, desc_V, id_dP);
        } else {
            uint64_t desc_dO = make_smem_desc_sm100_fn(s_dO1, 1, 1024);
            uint64_t desc_V  = make_smem_desc_sm100_fn(s_V1, 1, 1024);
            uint32_t id_dP = make_instr_desc_cg2(64, 64);
            gemm_64x64x16_cg2(tmem_dp[0], desc_dO, desc_V, id_dP);
        }
    }
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    asm volatile("barrier.sync.aligned 2, 128;" ::: "memory");
    tmem_store_bf16_row_fn((__nv_bfloat16*)s_dP, threadIdx.x, S, S, s_block, s_block, S);
    asm volatile("barrier.sync.aligned 2, 128;" ::: "memory");

    // ---------------- Compute D = row_sum(P * dP) ----------------
    for (int c = 0; c < 64; c += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(c));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        float sum = __uint_as_float(r0) * __uint_as_float(r0) + 
                    __uint_as_float(r1) * __uint_as_float(r1) +
                    __uint_as_float(r2) * __uint_as_float(r2) + 
                    __uint_as_float(r3) * __uint_as_float(r3);
        
        *(uint32_t*)(s_D + tid * 2) = __float_as_uint(sum);
    }

    // ---------------- Compute dS = P * (dP - D) * scale ----------------
    for (int c = 0; c < 64; c += 2) {
        uint32_t addr = (tid * 64 + c) * 2;
        __nv_bfloat16 p = *(__nv_bfloat16*)(s_P + addr);
        __nv_bfloat16 dp = *(__nv_bfloat16*)(s_dP + addr);
        float d_val = *(float*)(s_D + tid * 2);
        float ds_val = __bfloat162float(p) * (__bfloat162float(dp) - d_val) * sd;
        *(uint32_t*)(s_dS + addr) = __float_as_uint(ds_val);
    }
    __syncthreads();

    // ---------------- Compute dV = P^T @ dO ----------------
    if (threadIdx.x == 0) {
        uint32_t accum_dv0 = 0;
        uint32_t accum_dv1 = 0;
        uint64_t desc_P_T = make_smem_desc_sm100_fn(s_P, (64 / 8) * 1024, 1024);
        for(int i=0; i<4; ++i) {
            uint32_t id_s_P_T = make_instr_desc_transpose_A(64, 64);
            asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
            asm volatile("barrier.sync.aligned 2, 128;" ::: "memory");
            
            uint64_t desc_dO0 = make_smem_desc_sm100_fn(s_dO0 + i*16, 1, 1024);
            uint64_t desc_dO1 = make_smem_desc_sm100_fn(s_dO1 + i*16, 1, 1024);
            
            asm volatile("barrier.sync.aligned 2, 128;" ::: "memory");
            fence_async_shared_fn();
            
            if (cluster_rank_fn() == 0) {
                uint32_t accum_dv0 = (i == 0) ? 0 : 1;
                umma_f16_cg2_fn(tmem_dv0[0], desc_P_T, desc_dO0, id_s_P_T, accum_dv0);
            } else {
                uint32_t accum_dv1 = (i == 0) ? 0 : 1;
                umma_f16_cg2_fn(tmem_dv1[0], desc_P_T, desc_dO1, id_s_P_T, accum_dv1);
            }
        }
    }
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    asm volatile("barrier.sync.aligned 2, 128;" ::: "memory");

    // ---------------- Compute dQ = dS @ K ----------------
    if (threadIdx.x == 0) {
        uint32_t accum_dq0 = 0;
        uint32_t accum_dq1 = 0;
        uint64_t desc_dS = make_smem_desc_sm100_fn(s_dS, (64 / 8) * 1024, 1024);
        for(int i=0; i<4; ++i) {
            uint32_t id_dS_K = make_instr_desc_cg2(64, 64);
            asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
            asm volatile("barrier.sync.aligned 2, 128;" ::: "memory");
            
            uint64_t desc_K0 = make_smem_desc_sm100_fn(s_K0 + i*16, 1, 1024);
            uint64_t desc_K1 = make_smem_desc_sm100_fn(s_K1 + i*16, 1, 1024);
            
            asm volatile("barrier.sync.aligned 2, 128;" ::: "memory");
            fence_async_shared_fn();
            
            if (cluster_rank_fn() == 0) {
                uint32_t accum_dq0 = (i == 0) ? 0 : 1;
                umma_f16_cg2_fn(tmem_dq0[0], desc_dS, desc_K0, id_dS_K, accum_dq0);
            } else {
                uint32_t accum_dq1 = (i == 0) ? 0 : 1;
                umma_f16_cg2_fn(tmem_dq1[0], desc_dS, desc_K1, id_dS_K, accum_dq1);
            }
        }
    }
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    asm volatile("barrier.sync.aligned 2, 128;" ::: "memory");

    // ---------------- Compute dK = dS^T @ Q ----------------
    if (threadIdx.x == 0) {
        uint32_t accum_dk0 = 0;
        uint32_t accum_dk1 = 0;
        uint64_t desc_dS_T = make_smem_desc_sm100_fn(s_dS, (64 / 8) * 1024, 1024);
        for(int i=0; i<4; ++i) {
            uint32_t id_dS_T_Q = make_instr_desc_transpose_A(64, 64);
            asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
            asm volatile("barrier.sync.aligned 2, 128;" ::: "memory");
            
            uint64_t desc_Q0 = make_smem_desc_sm100_fn(s_Q0 + i*16, 1, 1024);
            uint64_t desc_Q1 = make_smem_desc_sm100_fn(s_Q1 + i*16, 1, 1024);
            
            asm volatile("barrier.sync.aligned 2, 128;" ::: "memory");
            fence_async_shared_fn();
            
            if (cluster_rank_fn() == 0) {
                uint32_t accum_dk0 = (i == 0) ? 0 : 1;
                umma_f16_cg2_fn(tmem_dk0[0], desc_dS_T, desc_Q0, id_dS_T_Q, accum_dk0);
            } else {
                uint32_t accum_dk1 = (i == 0) ? 0 : 1;
                umma_f16_cg2_fn(tmem_dk1[0], desc_dS_T, desc_Q1, id_dS_T_Q, accum_dk1);
            }
        }
    }
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    asm volatile("barrier.sync.aligned 2, 128;" ::: "memory");

    // ---------------- Output dV, dQ, dK ----------------
    if (cluster_rank_fn() == 0) {
        tmem_store_bf16_row_fn((__nv_bfloat16*)(((uintptr_t)tma_dV) + (batch_head * S + s_block) * 128), threadIdx.x, S, 128, s_block, 0, 64);
        tmem_store_bf16_row_fn((__nv_bfloat16*)(((uintptr_t)tma_dQ) + (batch_head * S + s_block) * 128), threadIdx.x, S, 128, s_block, 0, 64);
        tmem_store_bf16_row_fn((__nv_bfloat16*)(((uintptr_t)tma_dK) + (batch_head * S + s_block) * 128), threadIdx.x, S, 128, s_block, 0, 64);
    } else {
        tmem_store_bf16_row_fn((__nv_bfloat16*)(((uintptr_t)tma_dV) + (batch_head * S + s_block) * 128 + 64), threadIdx.x, S, 128, s_block, 64, 64);
        tmem_store_bf16_row_fn((__nv_bfloat16*)(((uintptr_t)tma_dQ) + (batch_head * S + s_block) * 128 + 64), threadIdx.x, S, 128, s_block, 64, 64);
        tmem_store_bf16_row_fn((__nv_bfloat16*)(((uintptr_t)tma_dK) + (batch_head * S + s_block) * 128 + 64), threadIdx.x, S, 128, s_block, 64, 64);
    }

    if (threadIdx.x == 0) {
        tmem_dealloc_fn(tmem_c[0], 64);
        tmem_dealloc_fn(tmem_dp[0], 64);
        tmem_dealloc_fn(tmem_dv0[0], 64);
        tmem_dealloc_fn(tmem_dv1[0], 64);
        tmem_dealloc_fn(tmem_dq0[0], 64);
        tmem_dealloc_fn(tmem_dq1[0], 64);
        tmem_dealloc_fn(tmem_dk0[0], 64);
        tmem_dealloc_fn(tmem_dk1[0], 64);
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
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_O, tma_dO, tma_dQ, tma_dK, tma_dV;
    
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
    create(&tma_dQ, dQ.data_ptr());
    create(&tma_dK, dK.data_ptr());
    create(&tma_dV, dV.data_ptr());
    
    dim3 grid((S + 95) / 96, B * H);
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

    CUDA_CHECK(cudaLaunchKernelEx(&config, mha_bwd_kernel, tma_Q, tma_K, tma_V, tma_O, tma_dO, tma_dQ, tma_dK, tma_dV, static_cast<const float*>(L.data_ptr()), S, 1.0f));
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(config.stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_kernel