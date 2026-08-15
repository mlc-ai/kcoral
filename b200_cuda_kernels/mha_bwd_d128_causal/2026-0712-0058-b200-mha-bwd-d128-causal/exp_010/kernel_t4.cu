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
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
       :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
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
        ".mbarrier::arrive::one.shared::cta.b64"
        " [%0];"
        :: "r"(a));
}

__device__ __forceinline__ uint64_t make_smem_desc_k_major(void* smem_ptr) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((1 & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((1024 & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_mn_major(void* smem_ptr) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((8192 & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((1024 & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    uint32_t base_offset = (addr >> 7) & 0x7;
    d |= (uint64_t)base_offset << 49;
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

__device__ __forceinline__ __nv_bfloat16 read_swizzled(const __nv_bfloat16* smem, int row, int col) {
    int chunk = col / 8;
    int chunk_phys = (row % 8) ^ chunk;
    return smem[row * 64 + chunk_phys * 8 + (col % 8)];
}

__device__ __forceinline__ void write_swizzled(__nv_bfloat16* smem, int row, int col, __nv_bfloat16 val) {
    int chunk = col / 8;
    int chunk_phys = (row % 8) ^ chunk;
    smem[row * 64 + chunk_phys * 8 + (col % 8)] = val;
}

// ---------------------- Kernel 1: Compute dQ ----------------------
__global__ void bwd_dQ_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_dO,
    const float* L,
    __nv_bfloat16* dQ,
    int64_t S_len,
    float scale)
{
    int bh = blockIdx.y;
    int q_block = blockIdx.x;
    
    extern __shared__ __align__(1024) uint8_t smem[];
    __nv_bfloat16* smem_Q0 = (__nv_bfloat16*)(smem + 0);
    __nv_bfloat16* smem_Q1 = (__nv_bfloat16*)(smem + 8192);
    __nv_bfloat16* smem_K0 = (__nv_bfloat16*)(smem + 16384);
    __nv_bfloat16* smem_K1 = (__nv_bfloat16*)(smem + 24576);
    __nv_bfloat16* smem_V0 = (__nv_bfloat16*)(smem + 32768);
    __nv_bfloat16* smem_V1 = (__nv_bfloat16*)(smem + 40960);
    __nv_bfloat16* smem_dO0 = (__nv_bfloat16*)(smem + 49152);
    __nv_bfloat16* smem_dO1 = (__nv_bfloat16*)(smem + 57344);
    __nv_bfloat16* smem_O0 = (__nv_bfloat16*)(smem + 65536);
    __nv_bfloat16* smem_O1 = (__nv_bfloat16*)(smem + 73728);
    __nv_bfloat16* smem_PT = (__nv_bfloat16*)(smem + 81920);
    __nv_bfloat16* smem_dST = (__nv_bfloat16*)(smem + 90112);
    float* smem_D = (float*)(smem + 98304);
    float* smem_L = (float*)(smem + 98560);
    uint64_t* mbar_tma = (uint64_t*)(smem + 98816);
    uint64_t* mbar_mma = (uint64_t*)(smem + 98824);

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_tma, 1);
        init_smem_barrier_fn(mbar_mma, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    uint32_t tmem_S, tmem_dP_T, tmem_dS, tmem_P_T, tmem_dQ0, tmem_dQ1;
    if (threadIdx.x < 32) {
        tmem_alloc_fn(&tmem_S, 64);
        tmem_alloc_fn(&tmem_dP_T, 64);
        tmem_alloc_fn(&tmem_dS, 64);
        tmem_alloc_fn(&tmem_P_T, 64);
        tmem_alloc_fn(&tmem_dQ0, 64);
        tmem_alloc_fn(&tmem_dQ1, 64);
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
        int row = threadIdx.x;
        smem_L[row] = (q_block * 64 + row < S_len) ? L[bh * S_len + q_block * 64 + row] : 0.0f;
        float sum = 0;
        for(int i=0; i<64; ++i) {
            sum += __bfloat162float(read_swizzled(smem_dO0, row, i)) * __bfloat162float(read_swizzled(smem_O0, row, i));
            sum += __bfloat162float(read_swizzled(smem_dO1, row, i)) * __bfloat162float(read_swizzled(smem_O1, row, i));
        }
        smem_D[row] = sum;
    }
    __syncthreads();

    uint32_t phase_kv = 0;
    uint32_t phase_mma = 0;
    uint32_t idesc_S = make_instr_desc(64, 64, 0, 0);
    uint32_t idesc_dQ = make_instr_desc(64, 64, 1, 1);
    
    __nv_bfloat16* dq_ptr = dQ + bh * S_len * 128;
    uint32_t first_k = 1;
    
    for (int k_block = 0; k_block <= q_block && k_block < (S_len + 63)/64; ++k_block) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_tma, 4 * 8192);
            tma_load_3d_fn(&tma_K, mbar_tma, smem_K0, 0, k_block * 64, bh);
            tma_load_3d_fn(&tma_K, mbar_tma, smem_K1, 64, k_block * 64, bh);
            tma_load_3d_fn(&tma_V, mbar_tma, smem_V0, 0, k_block * 64, bh);
            tma_load_3d_fn(&tma_V, mbar_tma, smem_V1, 64, k_block * 64, bh);
        }
        mbarrier_wait_fn(mbar_tma, phase_kv);
        phase_kv ^= 1;
        __syncthreads();

        if (threadIdx.x == 0) {
            umma_f16_cg1_fn(tmem_S, make_smem_desc_k_major(smem_Q0), make_smem_desc_k_major(smem_K0), idesc_S, 0);
            umma_f16_cg1_fn(tmem_S, make_smem_desc_k_major(smem_Q1), make_smem_desc_k_major(smem_K1), idesc_S, 1);
            
            umma_f16_cg1_fn(tmem_dP_T, make_smem_desc_k_major(smem_V0), make_smem_desc_k_major(smem_dO0), idesc_S, 0);
            umma_f16_cg1_fn(tmem_dP_T, make_smem_desc_k_major(smem_V1), make_smem_desc_k_major(smem_dO1), idesc_S, 1);
            
            umma_commit_1sm_fn(mbar_mma);
        }
        mbarrier_wait_fn(mbar_mma, phase_mma);
        phase_mma ^= 1;
        __syncthreads();

        for(int col_base = 0; col_base < 64; col_base += 4) {
            uint32_t r_s[4], r_dp[4];
            tmem_load_4x_fn(tmem_S + col_base, &r_s[0], &r_s[1], &r_s[2], &r_s[3]);
            tmem_load_4x_fn(tmem_dP_T + col_base, &r_dp[0], &r_dp[1], &r_dp[2], &r_dp[3]);
            tmem_load_fence_fn();
            
            if (threadIdx.x < 64) {
                int row = threadIdx.x;
                int query_idx = q_block * 64 + row;
                for(int i=0; i<4; ++i) {
                    int col = col_base + i;
                    int key_idx = k_block * 64 + col;
                    float p = 0, ds = 0;
                    if (query_idx < S_len && key_idx < S_len && key_idx <= query_idx) {
                        float s = __uint_as_float(r_s[i]);
                        p = fast_exp2f_fn((s * scale - smem_L[row]) * 1.44269504f);
                        float dp = __uint_as_float(r_dp[i]);
                        ds = p * (dp - smem_D[row]);
                    }
                    write_swizzled(smem_PT, col, row, __float2bfloat16(p));
                    write_swizzled(smem_dST, col, row, __float2bfloat16(ds));
                }
            }
        }
        fence_async_shared_fn();
        __syncthreads();

        if (threadIdx.x == 0) {
            if (first_k) {
                umma_f16_cg1_fn(tmem_dQ0, make_smem_desc_k_major(smem_dST), make_smem_desc_mn_major(smem_K0), idesc_dQ, 0);
            } else {
                umma_f16_cg1_fn(tmem_dQ0, make_smem_desc_k_major(smem_dST), make_smem_desc_mn_major(smem_K0), idesc_dQ, 1);
            }
            umma_f16_cg1_fn(tmem_dQ0, make_smem_desc_k_major(smem_dST), make_smem_desc_mn_major(smem_K0 + 16*128), idesc_dQ, 1);
            umma_f16_cg1_fn(tmem_dQ0, make_smem_desc_k_major(smem_dST), make_smem_desc_mn_major(smem_K0 + 32*128), idesc_dQ, 1);
            umma_f16_cg1_fn(tmem_dQ0, make_smem_desc_k_major(smem_dST), make_smem_desc_mn_major(smem_K0 + 48*128), idesc_dQ, 1);
            
            if (first_k) {
                umma_f16_cg1_fn(tmem_dQ1, make_smem_desc_k_major(smem_dST), make_smem_desc_mn_major(smem_K1), idesc_dQ, 0);
            } else {
                umma_f16_cg1_fn(tmem_dQ1, make_smem_desc_k_major(smem_dST), make_smem_desc_mn_major(smem_K1), idesc_dQ, 1);
            }
            umma_f16_cg1_fn(tmem_dQ1, make_smem_desc_k_major(smem_dST), make_smem_desc_mn_major(smem_K1 + 16*128), idesc_dQ, 1);
            umma_f16_cg1_fn(tmem_dQ1, make_smem_desc_k_major(smem_dST), make_smem_desc_mn_major(smem_K1 + 32*128), idesc_dQ, 1);
            umma_f16_cg1_fn(tmem_dQ1, make_smem_desc_k_major(smem_dST), make_smem_desc_mn_major(smem_K1 + 48*128), idesc_dQ, 1);
            
            umma_commit_1sm_fn(mbar_mma);
        }
        mbarrier_wait_fn(mbar_mma, phase_mma);
        phase_mma ^= 1;
        __syncthreads();
        
        first_k = 0;
        __syncthreads();
    }

    for(int col_base = 0; col_base < 64; col_base += 4) {
        uint32_t r0[4], r1[4];
        tmem_load_4x_fn(tmem_dQ0 + col_base, &r0[0], &r0[1], &r0[2], &r0[3]);
        tmem_load_4x_fn(tmem_dQ1 + col_base, &r1[0], &r1[1], &r1[2], &r1[3]);
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

    if (threadIdx.x < 32) {
        tmem_dealloc_fn(tmem_S, 64);
        tmem_dealloc_fn(tmem_dP_T, 64);
        tmem_dealloc_fn(tmem_dS, 64);
        tmem_dealloc_fn(tmem_P_T, 64);
        tmem_dealloc_fn(tmem_dQ0, 64);
        tmem_dealloc_fn(tmem_dQ1, 64);
    }
}

// ---------------------- Kernel 2: Compute dK, dV ----------------------
__global__ void bwd_dK_dV_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_dO,
    const float* L,
    __nv_bfloat16* dK,
    __nv_bfloat16* dV,
    int64_t S_len,
    float scale)
{
    int bh = blockIdx.y;
    int k_block = blockIdx.x;
    
    extern __shared__ __align__(1024) uint8_t smem[];
    __nv_bfloat16* smem_Q0 = (__nv_bfloat16*)(smem + 0);
    __nv_bfloat16* smem_Q1 = (__nv_bfloat16*)(smem + 8192);
    __nv_bfloat16* smem_K0 = (__nv_bfloat16*)(smem + 16384);
    __nv_bfloat16* smem_K1 = (__nv_bfloat16*)(smem + 24576);
    __nv_bfloat16* smem_V0 = (__nv_bfloat16*)(smem + 32768);
    __nv_bfloat16* smem_V1 = (__nv_bfloat16*)(smem + 40960);
    __nv_bfloat16* smem_dO0 = (__nv_bfloat16*)(smem + 49152);
    __nv_bfloat16* smem_dO1 = (__nv_bfloat16*)(smem + 57344);
    __nv_bfloat16* smem_O0 = (__nv_bfloat16*)(smem + 65536);
    __nv_bfloat16* smem_O1 = (__nv_bfloat16*)(smem + 73728);
    __nv_bfloat16* smem_PT = (__nv_bfloat16*)(smem + 81920);
    __nv_bfloat16* smem_dST = (__nv_bfloat16*)(smem + 90112);
    float* smem_D = (float*)(smem + 98304);
    float* smem_L = (float*)(smem + 98560);
    uint64_t* mbar_tma = (uint64_t*)(smem + 98816);
    uint64_t* mbar_mma = (uint64_t*)(smem + 98824);

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_tma, 1);
        init_smem_barrier_fn(mbar_mma, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    uint32_t tmem_S, tmem_dP, tmem_dS, tmem_P_T, tmem_dK0, tmem_dK1, tmem_dV0, tmem_dV1;
    if (threadIdx.x < 32) {
        tmem_alloc_fn(&tmem_S, 64);
        tmem_alloc_fn(&tmem_dP, 64);
        tmem_alloc_fn(&tmem_dS, 64);
        tmem_alloc_fn(&tmem_P_T, 64);
        tmem_alloc_fn(&tmem_dK0, 64);
        tmem_alloc_fn(&tmem_dK1, 64);
        tmem_alloc_fn(&tmem_dV0, 64);
        tmem_alloc_fn(&tmem_dV1, 64);
    }
    __syncthreads();

    uint32_t phase = 0;
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_tma, 4 * 8192);
        tma_load_3d_fn(&tma_K, mbar_tma, smem_K0, 0, k_block * 64, bh);
        tma_load_3d_fn(&tma_K, mbar_tma, smem_K1, 64, k_block * 64, bh);
        tma_load_3d_fn(&tma_V, mbar_tma, smem_V0, 0, k_block * 64, bh);
        tma_load_3d_fn(&tma_V, mbar_tma, smem_V1, 64, k_block * 64, bh);
    }
    mbarrier_wait_fn(mbar_tma, phase);
    phase ^= 1;
    __syncthreads();

    uint32_t phase_q = 0;
    uint32_t phase_mma = 0;
    uint32_t idesc_S = make_instr_desc(64, 64, 0, 0);
    uint32_t idesc_dV = make_instr_desc(64, 64, 0, 1);
    uint32_t idesc_dK = make_instr_desc(64, 64, 0, 1);
    
    __nv_bfloat16* dk_ptr = dK + bh * S_len * 128;
    __nv_bfloat16* dv_ptr = dV + bh * S_len * 128;

    int max_q_block = (S_len + 63)/64 - 1;
    uint32_t first_q = 1;
    
    for (int q_block = k_block; q_block <= max_q_block; ++q_block) {
        uint32_t tx_bytes = 6 * 8192;
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_tma, tx_bytes);
            tma_load_3d_fn(&tma_Q, mbar_tma, smem_Q0, 0, q_block * 64, bh);
            tma_load_3d_fn(&tma_Q, mbar_tma, smem_Q1, 64, q_block * 64, bh);
            tma_load_3d_fn(&tma_dO, mbar_tma, smem_dO0, 0, q_block * 64, bh);
            tma_load_3d_fn(&tma_dO, mbar_tma, smem_dO1, 64, q_block * 64, bh);
            tma_load_3d_fn(&tma_O, mbar_tma, smem_O0, 0, q_block * 64, bh);
            tma_load_3d_fn(&tma_O, mbar_tma, smem_O1, 64, q_block * 64, bh);
        }
        mbarrier_wait_fn(mbar_tma, phase_q);
        phase_q ^= 1;
        __syncthreads();

        if (threadIdx.x < 64) {
            int row = threadIdx.x;
            int query_idx = q_block * 64 + row;
            smem_L[row] = (query_idx < S_len) ? L[bh * S_len + query_idx] : 0.0f;
            float sum = 0;
            for(int i=0; i<64; ++i) {
                sum += __bfloat162float(read_swizzled(smem_dO0, row, i)) * __bfloat162float(read_swizzled(smem_O0, row, i));
                sum += __bfloat162float(read_swizzled(smem_dO1, row, i)) * __bfloat162float(read_swizzled(smem_O1, row, i));
            }
            smem_D[row] = sum;
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            umma_f16_cg1_fn(tmem_S, make_smem_desc_k_major(smem_K0), make_smem_desc_k_major(smem_Q0), idesc_S, 0);
            umma_f16_cg1_fn(tmem_S, make_smem_desc_k_major(smem_K1), make_smem_desc_k_major(smem_Q1), idesc_S, 1);
            
            umma_f16_cg1_fn(tmem_dP, make_smem_desc_k_major(smem_dO0), make_smem_desc_k_major(smem_V0), idesc_S, 0);
            umma_f16_cg1_fn(tmem_dP, make_smem_desc_k_major(smem_dO1), make_smem_desc_k_major(smem_V1), idesc_S, 1);
            
            umma_commit_1sm_fn(mbar_mma);
        }
        mbarrier_wait_fn(mbar_mma, phase_mma);
        phase_mma ^= 1;
        __syncthreads();

        for(int col_base = 0; col_base < 64; col_base += 4) {
            uint32_t r_s[4], r_dp[4];
            tmem_load_4x_fn(tmem_S + col_base, &r_s[0], &r_s[1], &r_s[2], &r_s[3]);
            tmem_load_4x_fn(tmem_dP + col_base, &r_dp[0], &r_dp[1], &r_dp[2], &r_dp[3]);
            tmem_load_fence_fn();
            
            if (threadIdx.x < 64) {
                int row = threadIdx.x;
                int query_idx = q_block * 64 + row;
                for(int i=0; i<4; ++i) {
                    int col = col_base + i;
                    int key_idx = k_block * 64 + col;
                    float p = 0, ds = 0;
                    if (query_idx < S_len && key_idx < S_len && key_idx <= query_idx) {
                        float s = __uint_as_float(r_s[i]);
                        p = fast_exp2f_fn((s * scale - smem_L[row]) * 1.44269504f);
                        float dp = __uint_as_float(r_dp[i]);
                        ds = p * (dp - smem_D[row]);
                    }
                    write_swizzled(smem_PT, col, row, __float2bfloat16(p));
                    write_swizzled(smem_dST, col, row, __float2bfloat16(ds));
                }
            }
        }
        fence_async_shared_fn();
        __syncthreads();

        if (threadIdx.x == 0) {
            if (first_q) {
                umma_f16_cg1_fn(tmem_dV0, make_smem_desc_k_major(smem_PT), make_smem_desc_mn_major(smem_dO0), idesc_dV, 0);
                umma_f16_cg1_fn(tmem_dV1, make_smem_desc_k_major(smem_PT), make_smem_desc_mn_major(smem_dO1), idesc_dV, 0);
                umma_f16_cg1_fn(tmem_dK0, make_smem_desc_k_major(smem_dST), make_smem_desc_mn_major(smem_Q0), idesc_dK, 0);
                umma_f16_cg1_fn(tmem_dK1, make_smem_desc_k_major(smem_dST), make_smem_desc_mn_major(smem_Q1), idesc_dK, 0);
            } else {
                umma_f16_cg1_fn(tmem_dV0, make_smem_desc_k_major(smem_PT), make_smem_desc_mn_major(smem_dO0), idesc_dV, 1);
                umma_f16_cg1_fn(tmem_dV1, make_smem_desc_k_major(smem_PT), make_smem_desc_mn_major(smem_dO1), idesc_dV, 1);
                umma_f16_cg1_fn(tmem_dK0, make_smem_desc_k_major(smem_dST), make_smem_desc_mn_major(smem_Q0), idesc_dK, 1);
                umma_f16_cg1_fn(tmem_dK1, make_smem_desc_k_major(smem_dST), make_smem_desc_mn_major(smem_Q1), idesc_dK, 1);
            }
            umma_commit_1sm_fn(mbar_mma);
        }
        mbarrier_wait_fn(mbar_mma, phase_mma);
        phase_mma ^= 1;
        __syncthreads();
        
        first_q = 0;
        __syncthreads();
    }

    for(int col_base = 0; col_base < 64; col_base += 4) {
        uint32_t r[4];
        tmem_load_4x_fn(tmem_dK0 + col_base, &r[0], &r[1], &r[2], &r[3]);
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
        tmem_load_4x_fn(tmem_dK1 + col_base, &r[0], &r[1], &r[2], &r[3]);
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
        tmem_load_4x_fn(tmem_dV0 + col_base, &r[0], &r[1], &r[2], &r[3]);
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
        tmem_load_4x_fn(tmem_dV1 + col_base, &r[0], &r[1], &r[2], &r[3]);
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

    if (threadIdx.x < 32) {
        tmem_dealloc_fn(tmem_S, 64);
        tmem_dealloc_fn(tmem_dP, 64);
        tmem_dealloc_fn(tmem_dS, 64);
        tmem_dealloc_fn(tmem_P_T, 64);
        tmem_dealloc_fn(tmem_dK0, 64);
        tmem_dealloc_fn(tmem_dK1, 64);
        tmem_dealloc_fn(tmem_dV0, 64);
        tmem_dealloc_fn(tmem_dV1, 64);
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
    
    int smem_size = 90112;
    CUDA_CHECK(cudaFuncSetAttribute(bwd_dQ_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    CUDA_CHECK(cudaFuncSetAttribute(bwd_dK_dV_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    dim3 grid((S + 63)/64, B*H);
    dim3 block(128);
    
    bwd_dQ_kernel<<<grid, block, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, tma_O, tma_dO, 
        static_cast<const float*>(L.data_ptr()), 
        static_cast<__nv_bfloat16*>(dQ.data_ptr()), 
        S, scale);
        
    bwd_dK_dV_kernel<<<grid, block, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, tma_O, tma_dO, 
        static_cast<const float*>(L.data_ptr()), 
        static_cast<__nv_bfloat16*>(dK.data_ptr()), 
        static_cast<__nv_bfloat16*>(dV.data_ptr()), 
        S, scale);
        
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_kernel