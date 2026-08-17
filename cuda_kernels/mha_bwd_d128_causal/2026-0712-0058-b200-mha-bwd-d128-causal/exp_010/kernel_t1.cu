#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
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
                (int)_e, __FILE__, __LINE__);                      \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_kernel {

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

__device__ __forceinline__ void tma_load_3d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
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

__device__ __forceinline__ uint64_t make_smem_desc(void* smem_ptr, bool is_mn_major) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint32_t sbo = 1024;
    uint32_t lbo = is_mn_major ? 8192 : 1;
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    if (is_mn_major) {
        uint32_t base_offset = (addr >> 7) & 0x7;
        d |= (uint64_t)base_offset << 49;
    }
    d |= (uint64_t)2 << 61;
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc(uint32_t M, uint32_t N, bool a_trans, bool b_trans) {
    uint32_t d = 0;
    d |= (1u << 4);
    d |= (1u << 7);
    d |= (1u << 10);
    d |= ((a_trans ? 1 : 0) << 15);
    d |= ((b_trans ? 1 : 0) << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__global__ void bwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_dO,
    const float* L,
    __nv_bfloat16* dQ,
    __nv_bfloat16* dK,
    __nv_bfloat16* dV,
    int64_t S_len,
    float scale)
{
    int bh = blockIdx.y;
    int q_block = blockIdx.x;
    int cluster_rank = (blockIdx.x % 2); 
    
    extern __shared__ __align__(1024) uint8_t smem[];
    __nv_bfloat16* smem_Q0 = (__nv_bfloat16*)(smem + 0);
    __nv_bfloat16* smem_Q1 = (__nv_bfloat16*)(smem + 8192);
    __nv_bfloat16* smem_K0_prev = (__nv_bfloat16*)(smem + 16384);
    __nv_bfloat16* smem_K0_next = (__nv_bfloat16*)(smem + 24576);
    __nv_bfloat16* smem_K1_prev = (__nv_bfloat16*)(smem + 32768);
    __nv_bfloat16* smem_K1_next = (__nv_bfloat16*)(smem + 40960);
    __nv_bfloat16* smem_V0_prev = (__nv_bfloat16*)(smem + 49152);
    __nv_bfloat16* smem_V0_next = (__nv_bfloat16*)(smem + 57344);
    __nv_bfloat16* smem_V1_prev = (__nv_bfloat16*)(smem + 65536);
    __nv_bfloat16* smem_V1_next = (__nv_bfloat16*)(smem + 73728);
    __nv_bfloat16* smem_dO0 = (__nv_bfloat16*)(smem + 81920);
    __nv_bfloat16* smem_dO1 = (__nv_bfloat16*)(smem + 90112);
    __nv_bfloat16* smem_O0 = (__nv_bfloat16*)(smem + 98304);
    __nv_bfloat16* smem_O1 = (__nv_bfloat16*)(smem + 106496);
    __nv_bfloat16* smem_PT = (__nv_bfloat16*)(smem + 114688);
    __nv_bfloat16* smem_dST = (__nv_bfloat16*)(smem + 122880);
    float* smem_D = (float*)(smem + 131072);
    float* smem_L = (float*)(smem + 131328);
    uint64_t* mbar_tma = (uint64_t*)(smem + 131584);
    uint64_t* mbar_mma = (uint64_t*)(smem + 131592);

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_tma, 1);
        init_smem_barrier_fn(mbar_mma, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    uint32_t tmem_S, tmem_dP, tmem_P, tmem_dS;
    if ((threadIdx.x % 32) == 0) {
        tmem_alloc_fn(&tmem_S, 128);
        tmem_alloc_fn(&tmem_dP, 128);
        tmem_alloc_fn(&tmem_P, 128);
        tmem_alloc_fn(&tmem_dS, 128);
    }
    __syncthreads();

    uint32_t phase = 0;
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_tma, 6 * 8192);
        tma_load_3d_fn(&tma_Q, mbar_tma, smem_Q0, 0, q_block * 64, bh);
        tma_load_3d_fn(&tma_Q, mbar_tma, smem_Q1, 64, q_block * 64, bh);
        tma_load_3d_fn(&tma_dO, mbar_tma, smem_dO0, 0, q_block * 64, bh);
        tma_load_3d_fn(&tma_dO, mbar_tma, smem_dO1, 64, q_block * 64, bh);
        tma_load_3d_fn(&tma_O, mbar_tma, smem_O0, 0, q_block * 64, bh);
        tma_load_3d_fn(&tma_O, mbar_tma, smem_O1, 64, q_block * 64, bh);
    }
    mbarrier_wait_fn(mbar_tma, phase);
    phase ^= 1;
    __syncthreads();

    if (threadIdx.x < 64) {
        smem_L[threadIdx.x] = (q_block * 64 + threadIdx.x < S_len) ? L[bh * S_len + q_block * 64 + threadIdx.x] : 0.0f;
        float sum = 0;
        for(int i=0; i<64; ++i) {
            int chunk = i / 8;
            int chunk_phys = (threadIdx.x % 8) ^ chunk;
            __nv_bfloat16 do0 = smem_dO0[threadIdx.x * 64 + chunk_phys * 8 + (i % 8)];
            __nv_bfloat16 o0 = smem_O0[threadIdx.x * 64 + chunk_phys * 8 + (i % 8)];
            sum += __bfloat162float(do0) * __bfloat162float(o0);
            
            __nv_bfloat16 do1 = smem_dO1[threadIdx.x * 64 + chunk_phys * 8 + (i % 8)];
            __nv_bfloat16 o1 = smem_O1[threadIdx.x * 64 + chunk_phys * 8 + (i % 8)];
            sum += __bfloat162float(do1) * __bfloat162float(o1);
        }
        smem_D[threadIdx.x] = sum;
    }
    __syncthreads();

    uint32_t phase_kv = 0;
    uint32_t phase_mma = 0;
    
    uint32_t idesc_full = make_instr_desc(128, 128, 0, 0);

    uint64_t desc_Q0 = make_smem_desc(smem_Q0, false);
    uint64_t desc_Q1 = make_smem_desc(smem_Q1, false);
    uint64_t desc_dO0 = make_smem_desc(smem_dO0, false);
    uint64_t desc_dO1 = make_smem_desc(smem_dO1, false);

    uint64_t desc_PT = make_smem_desc(smem_PT, false);
    uint64_t desc_dST = make_smem_desc(smem_dST, false);

    uint64_t desc_Q0_mn = make_smem_desc(smem_Q0, true);
    uint64_t desc_Q1_mn = make_smem_desc(smem_Q1, true);

    uint32_t first_kv = 1;
    
    __nv_bfloat16* dk_ptr = dK + bh * S_len * 128;
    __nv_bfloat16* dv_ptr = dV + bh * S_len * 128;

    for (int k_block = 0; k_block <= q_block && k_block < (S_len + 63)/64; ++k_block) {
        
        __nv_bfloat16* smem_K0_curr = (k_block == 0) ? smem_K0_prev : smem_K0_next;
        __nv_bfloat16* smem_K0_next = (k_block == 0) ? smem_K0_prev : smem_K0_curr;
        __nv_bfloat16* smem_K1_curr = (k_block == 0) ? smem_K1_prev : smem_K1_next;
        __nv_bfloat16* smem_K1_next = (k_block == 0) ? smem_K1_prev : smem_K1_curr;
        
        __nv_bfloat16* smem_V0_curr = (k_block == 0) ? smem_V0_prev : smem_V0_next;
        __nv_bfloat16* smem_V0_next = (k_block == 0) ? smem_V0_prev : smem_V0_curr;
        __nv_bfloat16* smem_V1_curr = (k_block == 0) ? smem_V1_prev : smem_V1_next;
        __nv_bfloat16* smem_V1_next = (k_block == 0) ? smem_V1_prev : smem_V1_curr;

        if (threadIdx.x == 0) {
            uint32_t next_k = k_block + 1;
            bool load_next = (next_k <= q_block && next_k < (S_len + 63)/64);
            
            if (load_next) {
                mbarrier_arrive_and_expect_tx_fn(mbar_tma, 4 * 8192);
                tma_load_3d_fn(&tma_K, mbar_tma, smem_K0_next, 0, next_k * 64, bh);
                tma_load_3d_fn(&tma_K, mbar_tma, smem_K1_next, 64, next_k * 64, bh);
                tma_load_3d_fn(&tma_V, mbar_tma, smem_V0_next, 0, next_k * 64, bh);
                tma_load_3d_fn(&tma_V, mbar_tma, smem_V1_next, 64, next_k * 64, bh);
            }
            
            if (first_kv) {
                mbarrier_arrive_and_expect_tx_fn(mbar_tma, 4 * 8192);
                tma_load_3d_fn(&tma_K, mbar_tma, smem_K0_curr, 0, k_block * 64, bh);
                tma_load_3d_fn(&tma_K, mbar_tma, smem_K1_curr, 64, k_block * 64, bh);
                tma_load_3d_fn(&tma_V, mbar_tma, smem_V0_curr, 0, k_block * 64, bh);
                tma_load_3d_fn(&tma_V, mbar_tma, smem_V1_curr, 64, k_block * 64, bh);
            }
        }
        mbarrier_wait_fn(mbar_tma, phase_kv);
        phase_kv ^= 1;
        __syncthreads();
        
        uint64_t desc_K0_A = make_smem_desc(smem_K0_curr, false);
        uint64_t desc_K1_A = make_smem_desc(smem_K1_curr, false);
        uint64_t desc_V0_A = make_smem_desc(smem_V0_curr, false);
        uint64_t desc_V1_A = make_smem_desc(smem_V1_curr, false);
        
        uint64_t desc_Q0_B = make_smem_desc(smem_Q0, false);
        uint64_t desc_Q1_B = make_smem_desc(smem_Q1, false);
        
        if (threadIdx.x == 0) {
            umma_f16_cg2_fn(tmem_S, desc_K0_A, desc_Q0_B, idesc_full, 0);
            umma_f16_cg2_fn(tmem_S, desc_K1_A, desc_Q1_B, idesc_full, 1);
            umma_commit_2sm_fn(mbar_mma);
        }
        mbarrier_wait_fn(mbar_mma, phase_mma);
        phase_mma ^= 1;
        __syncthreads();

        int tid = threadIdx.x;
        for(int col_base = 0; col_base < 64; col_base += 4) {
            uint32_t r_s[4], r_dp[4];
            tmem_load_4x_fn(tmem_S + col_base, &r_s[0], &r_s[1], &r_s[2], &r_s[3]);
            tmem_load_4x_fn(tmem_dP + col_base, &r_dp[0], &r_dp[1], &r_dp[2], &r_dp[3]);
            tmem_load_fence_fn();
            
            int row0 = tid;
            int row1 = tid + 64;
            
            int q_idx0 = q_block * 64 + row0;
            int q_idx1 = q_block * 64 + row1;
            
            float p0[4], p1[4];
            float ds0[4], ds1[4];
            for(int i=0; i<4; ++i) {
                int col = col_base + i;
                int k_idx = k_block * 64 + col;
                
                p0[i] = 0; p1[i] = 0;
                ds0[i] = 0; ds1[i] = 0;
                
                if (q_idx0 < S_len && k_idx <= q_idx0) {
                    float s = __uint_as_float(r_s[i]);
                    p0[i] = fast_exp2f_fn((s * scale - smem_L[row0]) * 1.44269504f);
                    ds0[i] = p0[i] * (__uint_as_float(r_dp[i]) - smem_D[row0]);
                }
                if (q_idx1 < S_len && k_idx <= q_idx1) {
                    float s = __uint_as_float(r_s[i]);
                    p1[i] = fast_exp2f_fn((s * scale - smem_L[row1]) * 1.44269504f);
                    ds1[i] = p1[i] * (__uint_as_float(r_dp[i]) - smem_D[row1]);
                }
                
                int chunk = col / 8;
                int chunk_phys0 = (row0 % 8) ^ chunk;
                smem_PT[row0 * 64 + chunk_phys0 * 8 + (col % 8)] = __float2bfloat16(p0[i]);
                smem_dST[row0 * 64 + chunk_phys0 * 8 + (col % 8)] = __float2bfloat16(ds0[i]);
                
                int chunk_phys1 = (row1 % 8) ^ chunk;
                smem_PT[row1 * 64 + chunk_phys1 * 8 + (col % 8)] = __float2bfloat16(p1[i]);
                smem_dST[row1 * 64 + chunk_phys1 * 8 + (col % 8)] = __float2bfloat16(ds1[i]);
            }
        }
        fence_async_shared_fn();
        __syncthreads();

        uint64_t desc_P_A = make_smem_desc(smem_PT, false);
        uint64_t desc_dS_A = make_smem_desc(smem_dST, false);

        uint64_t desc_K0_B_mn = make_smem_desc(smem_K0_curr, true);
        uint64_t desc_K1_B_mn = make_smem_desc(smem_K1_curr, true);

        if (threadIdx.x == 0) {
            umma_f16_cg2_fn(tmem_dP, desc_V0_A, desc_dO0, idesc_full, 0);
            umma_f16_cg2_fn(tmem_dP, desc_V1_A, desc_dO1, idesc_full, 1);
            umma_commit_2sm_fn(mbar_mma);
        }
        mbarrier_wait_fn(mbar_mma, phase_mma);
        phase_mma ^= 1;
        __syncthreads();

        uint32_t first_k = (k_block == 0);
        if (threadIdx.x == 0) {
            if (first_k) {
                umma_f16_cg2_fn(tmem_dV, desc_P_A, desc_dO0_B, idesc_full, 0);
                umma_f16_cg2_fn(tmem_dK, desc_dST_A, desc_Q0_B_mn, idesc_full, 0);
            } else {
                umma_f16_cg2_fn(tmem_dV, desc_P_A, desc_dO0_B, idesc_full, 1);
                umma_f16_cg2_fn(tmem_dK, desc_dST_A, desc_Q0_B_mn, idesc_full, 1);
            }
            umma_f16_cg2_fn(tmem_dV, desc_P_A, desc_dO1_B, idesc_full, 1);
            umma_f16_cg2_fn(tmem_dK, desc_dST_A, desc_Q1_B_mn, idesc_full, 1);
            umma_commit_2sm_fn(mbar_mma);
        }
        mbarrier_wait_fn(mbar_mma, phase_mma);
        phase_mma ^= 1;
        __syncthreads();
        
        if (threadIdx.x == 0) {
            if (first_k) {
                umma_f16_cg2_fn(tmem_dP, desc_dST_A, desc_K0_B_mn, idesc_full, 0);
                umma_f16_cg2_fn(tmem_dS, desc_dST_A, desc_K1_B_mn, idesc_full, 1);
            } else {
                umma_f16_cg2_fn(tmem_dP, desc_dST_A, desc_K0_B_mn, idesc_full, 1);
                umma_f16_cg2_fn(tmem_dS, desc_dST_A, desc_K1_B_mn, idesc_full, 1);
            }
            umma_commit_2sm_fn(mbar_mma);
        }
        mbarrier_wait_fn(mbar_mma, phase_mma);
        phase_mma ^= 1;
        __syncthreads();

        for(int col_base = 0; col_base < 64; col_base += 4) {
            uint32_t r[4];
            tmem_load_4x_fn(tmem_dK + col_base, &r[0], &r[1], &r[2], &r[3]);
            tmem_load_fence_fn();
            if (threadIdx.x < 64) {
                int row = threadIdx.x;
                int k_idx = k_block * 64 + row;
                if (k_idx < S_len) {
                    for(int i=0; i<4; ++i) {
                        int col = col_base + i;
                        dk_ptr[k_idx * 128 + col] = __float2bfloat16(__uint_as_float(r[i]));
                    }
                }
            }
            tmem_load_4x_fn(tmem_dK + 64 + col_base, &r[0], &r[1], &r[2], &r[3]);
            tmem_load_fence_fn();
            if (threadIdx.x < 64) {
                int row = threadIdx.x;
                int k_idx = k_block * 64 + row;
                if (k_idx < S_len) {
                    for(int i=0; i<4; ++i) {
                        int col = col_base + i;
                        dk_ptr[k_idx * 128 + 64 + col] = __float2bfloat16(__uint_as_float(r[i]));
                    }
                }
            }
        }

        for(int col_base = 0; col_base < 64; col_base += 4) {
            uint32_t r[4];
            tmem_load_4x_fn(tmem_dV + col_base, &r[0], &r[1], &r[2], &r[3]);
            tmem_load_fence_fn();
            if (threadIdx.x < 64) {
                int row = threadIdx.x;
                int k_idx = k_block * 64 + row;
                if (k_idx < S_len) {
                    for(int i=0; i<4; ++i) {
                        int col = col_base + i;
                        dv_ptr[k_idx * 128 + col] = __float2bfloat16(__uint_as_float(r[i]));
                    }
                }
            }
            tmem_load_4x_fn(tmem_dV + 64 + col_base, &r[0], &r[1], &r[2], &r[3]);
            tmem_load_fence_fn();
            if (threadIdx.x < 64) {
                int row = threadIdx.x;
                int k_idx = k_block * 64 + row;
                if (k_idx < S_len) {
                    for(int i=0; i<4; ++i) {
                        int col = col_base + i;
                        dv_ptr[k_idx * 128 + 64 + col] = __float2bfloat16(__uint_as_float(r[i]));
                    }
                }
            }
        }
        
        __syncthreads();
    }

    __nv_bfloat16* dq_ptr = dQ + bh * S_len * 128;
    for(int col_base = 0; col_base < 64; col_base += 4) {
        uint32_t r0[4], r1[4];
        tmem_load_4x_fn(tmem_dP + col_base, &r0[0], &r0[1], &r0[2], &r0[3]);
        tmem_load_4x_fn(tmem_dS + col_base, &r1[0], &r1[1], &r1[2], &r1[3]);
        tmem_load_fence_fn();
        if (threadIdx.x < 64) {
            int row = threadIdx.x;
            int q_idx = q_block * 64 + row;
            if (q_idx < S_len) {
                for(int i=0; i<4; ++i) {
                    int col = col_base + i;
                    dq_ptr[q_idx * 128 + col] = __float2bfloat16(__uint_as_float(r0[i]));
                    dq_ptr[q_idx * 128 + 64 + col] = __float2bfloat16(__uint_as_float(r1[i]));
                }
            }
        }
    }

    if ((threadIdx.x % 32) == 0) {
        tmem_dealloc_fn(tmem_S, 128);
        tmem_dealloc_fn(tmem_dP, 128);
        tmem_dealloc_fn(tmem_P, 128);
        tmem_dealloc_fn(tmem_dS, 128);
    }
}

CUresult create_tma_3d_descriptor_2B(CUtensorMap* d, void* globalAddress, 
    uint64_t gmem_dim0, uint64_t gmem_dim1, uint64_t gmem_dim2,
    uint32_t smem_dim0, uint32_t smem_dim1, uint32_t smem_dim2) 
{
    cuuint64_t globalDim[3] = {gmem_dim0, gmem_dim1, gmem_dim2};
    cuuint64_t globalStrides[2] = {gmem_dim0 * 2, gmem_dim0 * gmem_dim1 * 2};
    cuuint32_t boxDim[3] = {smem_dim0, smem_dim1, smem_dim2};
    cuuint32_t elementStrides[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, globalAddress, globalDim, globalStrides,
        boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) 
{
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t d = Q.size(3);
    
    float scale = 1.0f / sqrtf((float)d);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_O, tma_dO;
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_Q, Q.data_ptr(), d, S, B*H, 64, 64, 1));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_K, K.data_ptr(), d, S, B*H, 64, 64, 1));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_V, V.data_ptr(), d, S, B*H, 64, 64, 1));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_O, O.data_ptr(), d, S, B*H, 64, 64, 1));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_dO, dO.data_ptr(), d, S, B*H, 64, 64, 1));
    
    int smem_size = 131600;
    CUDA_CHECK(cudaFuncSetAttribute(bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    dim3 grid((S + 63)/64, B*H);
    dim3 block(128);
    
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
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, bwd_kernel,
        tma_Q, tma_K, tma_V, tma_O, tma_dO, 
        static_cast<const float*>(L.data_ptr()), 
        static_cast<__nv_bfloat16*>(dQ.data_ptr()), 
        static_cast<__nv_bfloat16*>(dK.data_ptr()), 
        static_cast<__nv_bfloat16*>(dV.data_ptr()), 
        S, scale));
        
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_kernel