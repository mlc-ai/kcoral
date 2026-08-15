#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
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

namespace tvm_ffi_mha_bwd {

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

__device__ __forceinline__ void tcgen05_fence_after_fn() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
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
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ uint32_t tmem_addr(uint32_t base, uint16_t row, uint16_t col) {
    return base + (((uint32_t)row << 16) | col);
}

__device__ __forceinline__ uint64_t make_smem_desc_k_major(void* ptr) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    uint32_t sbo = 1024;
    uint32_t lbo = 1;
    uint64_t d = 0;
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)((addr >> 7) & 0x7) << 49;
    d |= (uint64_t)2 << 61;
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_mn_major(void* ptr, uint32_t k_dim) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    uint32_t sbo = 1024;
    uint32_t lbo = (k_dim / 8) * sbo;
    uint64_t d = 0;
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)((addr >> 7) & 0x7) << 49;
    d |= (uint64_t)2 << 61;
    return d;
}

__device__ __forceinline__ uint64_t advance_desc(uint64_t desc, uint32_t bytes) {
    uint32_t old_addr_shifted = desc & 0x3FFF;
    uint32_t new_addr = (old_addr_shifted << 4) + bytes;
    desc &= ~0x3FFFull;
    desc &= ~(0x7ull << 49);
    desc |= ((new_addr >> 4) & 0x3FFF);
    desc |= (uint64_t)((new_addr >> 7) & 0x7) << 49;
    return desc;
}

__device__ __forceinline__ uint32_t make_idesc(uint32_t M, uint32_t N, int trans_a, int trans_b) {
    uint32_t d = 0;
    d |= (1u << 4);
    d |= (1u << 7);
    d |= (1u << 10);
    d |= (trans_a << 15);
    d |= (trans_b << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__device__ __forceinline__ void umma_f16_cg1_fn(uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_f16_cg1_tmem_A_fn(uint32_t tmem_c, uint32_t tmem_a, uint64_t desc_b, uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n}\n"
        :: "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void st_shared_128_fn(uint32_t addr, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" :: "r"(addr), "r"(v0), "r"(v1), "r"(v2), "r"(v3) : "memory");
}

__device__ __forceinline__ uint32_t pack_bf16_fn(uint32_t fp32_a, uint32_t fp32_b) {
    __nv_bfloat16 a = __float2bfloat16(__uint_as_float(fp32_a));
    __nv_bfloat16 b = __float2bfloat16(__uint_as_float(fp32_b));
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};" : "=r"(result) : "h"(*reinterpret_cast<uint16_t*>(&a)), "h"(*reinterpret_cast<uint16_t*>(&b)));
    return result;
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

struct SharedStorage {
    __align__(1024) __nv_bfloat16 K_L[128 * 64];
    __align__(1024) __nv_bfloat16 K_R[128 * 64];
    __align__(1024) __nv_bfloat16 V_L[128 * 64];
    __align__(1024) __nv_bfloat16 V_R[128 * 64];
    __align__(1024) __nv_bfloat16 Q_L[128 * 64];
    __align__(1024) __nv_bfloat16 Q_R[128 * 64];
    __align__(1024) __nv_bfloat16 dO_L[128 * 64];
    __align__(1024) __nv_bfloat16 dO_R[128 * 64];
    __align__(1024) __nv_bfloat16 dS_L[128 * 64];
    __align__(1024) __nv_bfloat16 dS_R[128 * 64];
    __align__(1024) float LSE[128];
    __align__(1024) float D[128];
    uint64_t bar_KV;
    uint64_t bar_Q;
    uint64_t bar_mma;
    uint32_t tmem_ptr;
};

__global__ void precompute_D_kernel(const __nv_bfloat16* O, const __nv_bfloat16* dO, float* D, int B, int H, int S, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * S;
    if (idx < total) {
        float sum = 0;
        for (int i = 0; i < d; ++i) {
            sum += __bfloat162float(O[idx * d + i]) * __bfloat162float(dO[idx * d + i]);
        }
        D[idx] = sum;
    }
}

__global__ void mha_bwd_kernel(
    const __grid_constant__ CUtensorMap tma_K, const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_Q, const __grid_constant__ CUtensorMap tma_dO,
    const float* __restrict__ g_LSE, const float* __restrict__ g_D,
    __nv_bfloat16* __restrict__ dQ, __nv_bfloat16* __restrict__ dK, __nv_bfloat16* __restrict__ dV,
    int B, int H, int S) 
{
    extern __shared__ SharedStorage smem[];

    int n_block = blockIdx.x;
    int bh_offset = blockIdx.y;
    int kv_start = n_block * 128;
    if (kv_start >= S) return;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&smem->bar_KV, 1);
        init_smem_barrier_fn(&smem->bar_Q, 1);
        init_smem_barrier_fn(&smem->bar_mma, 1);
    }
    if (threadIdx.x < 32) {
        tmem_alloc_fn(&smem->tmem_ptr, 512);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    uint32_t base_t = smem->tmem_ptr;
    int global_kv_start = bh_offset * S + kv_start;

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(&smem->bar_KV, 4 * 128 * 64 * 2);
        tma_load_2d_fn(&tma_K, &smem->bar_KV, smem->K_L, 0, global_kv_start);
        tma_load_2d_fn(&tma_K, &smem->bar_KV, smem->K_R, 64, global_kv_start);
        tma_load_2d_fn(&tma_V, &smem->bar_KV, smem->V_L, 0, global_kv_start);
        tma_load_2d_fn(&tma_V, &smem->bar_KV, smem->V_R, 64, global_kv_start);
    }

    uint64_t desc_K_L_K = make_smem_desc_k_major(smem->K_L);
    uint64_t desc_K_R_K = make_smem_desc_k_major(smem->K_R);
    uint64_t desc_V_L_K = make_smem_desc_k_major(smem->V_L);
    uint64_t desc_V_R_K = make_smem_desc_k_major(smem->V_R);
    uint64_t desc_Q_L_K = make_smem_desc_k_major(smem->Q_L);
    uint64_t desc_Q_R_K = make_smem_desc_k_major(smem->Q_R);
    uint64_t desc_dO_L_K = make_smem_desc_k_major(smem->dO_L);
    uint64_t desc_dO_R_K = make_smem_desc_k_major(smem->dO_R);

    uint64_t desc_Q_L_N = make_smem_desc_mn_major(smem->Q_L, 64);
    uint64_t desc_Q_R_N = make_smem_desc_mn_major(smem->Q_R, 64);
    uint64_t desc_dO_L_N = make_smem_desc_mn_major(smem->dO_L, 64);
    uint64_t desc_dO_R_N = make_smem_desc_mn_major(smem->dO_R, 64);
    uint64_t desc_K_L_N = make_smem_desc_mn_major(smem->K_L, 128);
    uint64_t desc_K_R_N = make_smem_desc_mn_major(smem->K_R, 128);
    uint64_t desc_dS_L_M = make_smem_desc_mn_major(smem->dS_L, 128);
    uint64_t desc_dS_R_M = make_smem_desc_mn_major(smem->dS_R, 128);

    uint32_t idesc_128x128_K_N = make_idesc(128, 128, 0, 1);
    uint32_t idesc_128x128_K_K = make_idesc(128, 128, 0, 0);
    uint32_t idesc_128x64_K_N  = make_idesc(128, 64, 0, 1);
    uint32_t idesc_128x64_K_K  = make_idesc(128, 64, 0, 0);
    uint32_t idesc_64x64_M_N   = make_idesc(64, 64, 1, 1);
    uint32_t idesc_64x64_M_K   = make_idesc(64, 64, 1, 0);

    // Initialize dV and dK in TMEM (128 columns each)
    for (int c = 0; c < 128; c += 8) {
        uint32_t zeros[8] = {0};
        asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%8], {%0,%1,%2,%3,%4,%5,%6,%7};"
            :: "r"(zeros[0]), "r"(zeros[1]), "r"(zeros[2]), "r"(zeros[3]),
               "r"(zeros[4]), "r"(zeros[5]), "r"(zeros[6]), "r"(zeros[7]), "r"(tmem_addr(base_t, 0, 128 + c)) : "memory");
        asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%8], {%0,%1,%2,%3,%4,%5,%6,%7};"
            :: "r"(zeros[0]), "r"(zeros[1]), "r"(zeros[2]), "r"(zeros[3]),
               "r"(zeros[4]), "r"(zeros[5]), "r"(zeros[6]), "r"(zeros[7]), "r"(tmem_addr(base_t, 0, 384 + c)) : "memory");
    }
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");

    mbarrier_wait_fn(&smem->bar_KV, 0);
    
    int num_q_blocks = (S + 127) / 128;
    uint32_t q_phase = 0;
    uint32_t mma_phase = 0;

    for (int q_block = n_block; q_block < num_q_blocks; ++q_block) {
        int q_start = q_block * 128;
        int global_q_start = bh_offset * S + q_start;

        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(&smem->bar_Q, 4 * 128 * 64 * 2);
            tma_load_2d_fn(&tma_Q, &smem->bar_Q, smem->Q_L, 0, global_q_start);
            tma_load_2d_fn(&tma_Q, &smem->bar_Q, smem->Q_R, 64, global_q_start);
            tma_load_2d_fn(&tma_dO, &smem->bar_Q, smem->dO_L, 0, global_q_start);
            tma_load_2d_fn(&tma_dO, &smem->bar_Q, smem->dO_R, 64, global_q_start);
        }

        if (q_start + threadIdx.x < S) {
            smem->LSE[threadIdx.x] = g_LSE[bh_offset * S + q_start + threadIdx.x];
            smem->D[threadIdx.x]   = g_D[bh_offset * S + q_start + threadIdx.x];
        }

        mbarrier_wait_fn(&smem->bar_Q, q_phase);
        q_phase ^= 1;

        if (threadIdx.x == 127) {
            // 1. S^T = K @ Q^T
            for (int k = 0; k < 4; ++k) umma_f16_cg1_fn(tmem_addr(base_t, 0, 0), advance_desc(desc_K_L_K, k*32), advance_desc(desc_Q_L_K, k*32), idesc_128x128_K_K, 0);
            for (int k = 4; k < 8; ++k) umma_f16_cg1_fn(tmem_addr(base_t, 0, 0), advance_desc(desc_K_R_K, (k-4)*32), advance_desc(desc_Q_R_K, (k-4)*32), idesc_128x128_K_K, 1);

            // 2. dP^T = V @ dO^T
            for (int k = 0; k < 4; ++k) umma_f16_cg1_fn(tmem_addr(base_t, 0, 256), advance_desc(desc_V_L_K, k*32), advance_desc(desc_dO_L_K, k*32), idesc_128x128_K_K, 0);
            for (int k = 4; k < 8; ++k) umma_f16_cg1_fn(tmem_addr(base_t, 0, 256), advance_desc(desc_V_R_K, (k-4)*32), advance_desc(desc_dO_R_K, (k-4)*32), idesc_128x128_K_K, 1);

            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"((uint32_t)__cvta_generic_to_shared(&smem->bar_mma)));
        }
        mbarrier_wait_fn(&smem->bar_mma, mma_phase);
        mma_phase ^= 1;

        // 3. Element-wise
        for (int c = 0; c < 128; c += 8) {
            uint32_t s_regs[8], dp_regs[8];
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                : "=r"(s_regs[0]),"=r"(s_regs[1]),"=r"(s_regs[2]),"=r"(s_regs[3]),
                  "=r"(s_regs[4]),"=r"(s_regs[5]),"=r"(s_regs[6]),"=r"(s_regs[7]) : "r"(tmem_addr(base_t, 0, 0 + c)));
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                : "=r"(dp_regs[0]),"=r"(dp_regs[1]),"=r"(dp_regs[2]),"=r"(dp_regs[3]),
                  "=r"(dp_regs[4]),"=r"(dp_regs[5]),"=r"(dp_regs[6]),"=r"(dp_regs[7]) : "r"(tmem_addr(base_t, 0, 256 + c)));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");

            uint32_t p_packed[4], ds_packed[4];
            int kv_idx = kv_start + threadIdx.x;
            float scale = 0.08838834764831844f;
            
            for (int i = 0; i < 4; ++i) {
                float s0 = __uint_as_float(s_regs[i*2]) * scale;
                float dp0 = __uint_as_float(dp_regs[i*2]);
                float lse0 = smem->LSE[c + i*2];
                float d0 = smem->D[c + i*2];
                float p0 = fast_exp2f_fn((s0 - lse0) * 1.44269504f);
                float ds0 = p0 * (dp0 - d0) * scale;
                int q_idx0 = q_start + c + i*2;
                if (kv_idx > q_idx0 || q_idx0 >= S || kv_idx >= S) { p0 = 0; ds0 = 0; }

                float s1 = __uint_as_float(s_regs[i*2+1]) * scale;
                float dp1 = __uint_as_float(dp_regs[i*2+1]);
                float lse1 = smem->LSE[c + i*2 + 1];
                float d1 = smem->D[c + i*2 + 1];
                float p1 = fast_exp2f_fn((s1 - lse1) * 1.44269504f);
                float ds1 = p1 * (dp1 - d1) * scale;
                int q_idx1 = q_start + c + i*2 + 1;
                if (kv_idx > q_idx1 || q_idx1 >= S || kv_idx >= S) { p1 = 0; ds1 = 0; }

                p_packed[i] = pack_bf16_fn(__float_as_uint(p0), __float_as_uint(p1));
                ds_packed[i] = pack_bf16_fn(__float_as_uint(ds0), __float_as_uint(ds1));
            }
            asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {%0,%1,%2,%3};"
                :: "r"(p_packed[0]), "r"(p_packed[1]), "r"(p_packed[2]), "r"(p_packed[3]), "r"(tmem_addr(base_t, 0, 0 + c/2)) : "memory");
            asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {%0,%1,%2,%3};"
                :: "r"(ds_packed[0]), "r"(ds_packed[1]), "r"(ds_packed[2]), "r"(ds_packed[3]), "r"(tmem_addr(base_t, 0, 256 + c/2)) : "memory");
        }
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");

        // 4. dV and dK
        if (threadIdx.x == 127) {
            for (int k = 0; k < 8; ++k) {
                int accum_flag = (q_block != n_block || k > 0) ? 1 : 0;
                umma_f16_cg1_tmem_A_fn(tmem_addr(base_t, 0, 128), tmem_addr(base_t, 0, 0 + k*8), advance_desc(desc_dO_L_N, k*2048), idesc_128x64_K_N, accum_flag);
                umma_f16_cg1_tmem_A_fn(tmem_addr(base_t, 0, 192), tmem_addr(base_t, 0, 0 + k*8), advance_desc(desc_dO_R_N, k*2048), idesc_128x64_K_N, accum_flag);
                
                umma_f16_cg1_tmem_A_fn(tmem_addr(base_t, 0, 384), tmem_addr(base_t, 0, 256 + k*8), advance_desc(desc_Q_L_N, k*2048), idesc_128x64_K_N, accum_flag);
                umma_f16_cg1_tmem_A_fn(tmem_addr(base_t, 0, 448), tmem_addr(base_t, 0, 256 + k*8), advance_desc(desc_Q_R_N, k*2048), idesc_128x64_K_N, accum_flag);
            }
        }

        // Extract dS_T to SMEM
        for (int c = 0; c < 32; c += 4) {
            uint32_t regs[4];
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3]) : "r"(tmem_addr(base_t, 0, 256 + c)));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            int x = c / 4;
            int swizzled_x = (threadIdx.x % 8) ^ x;
            uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(&smem->dS_L[threadIdx.x * 64 + swizzled_x * 8]);
            st_shared_128_fn(smem_addr, regs[0], regs[1], regs[2], regs[3]);
        }
        for (int c = 0; c < 32; c += 4) {
            uint32_t regs[4];
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3]) : "r"(tmem_addr(base_t, 0, 256 + 32 + c)));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            int x = c / 4;
            int swizzled_x = (threadIdx.x % 8) ^ x;
            uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(&smem->dS_R[threadIdx.x * 64 + swizzled_x * 8]);
            st_shared_128_fn(smem_addr, regs[0], regs[1], regs[2], regs[3]);
        }
        __syncthreads();
        fence_async_shared_fn();

        // 5. dQ
        if (threadIdx.x == 127) {
            for (int k = 0; k < 8; ++k) {
                int acc = k > 0 ? 1 : 0;
                umma_f16_cg1_fn(tmem_addr(base_t, 0, 256), advance_desc(desc_dS_L_M, k*2048), advance_desc(desc_K_L_N, k*2048), idesc_64x64_M_N, acc);
                umma_f16_cg1_fn(tmem_addr(base_t, 0, 320), advance_desc(desc_dS_L_M, k*2048), advance_desc(desc_K_R_N, k*2048), idesc_64x64_M_N, acc);
            }
            for (int k = 0; k < 8; ++k) {
                int acc = k > 0 ? 1 : 0;
                umma_f16_cg1_fn(tmem_addr(base_t, 64, 256), advance_desc(desc_dS_R_M, k*2048), advance_desc(desc_K_L_N, k*2048), idesc_64x64_M_N, acc);
                umma_f16_cg1_fn(tmem_addr(base_t, 64, 320), advance_desc(desc_dS_R_M, k*2048), advance_desc(desc_K_R_N, k*2048), idesc_64x64_M_N, acc);
            }
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"((uint32_t)__cvta_generic_to_shared(&smem->bar_mma)));
        }
        mbarrier_wait_fn(&smem->bar_mma, mma_phase);
        mma_phase ^= 1;

        // AtomicAdd dQ
        for (int c = 0; c < 128; c += 8) {
            uint32_t regs[8];
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                : "=r"(regs[0]),"=r"(regs[1]),"=r"(regs[2]),"=r"(regs[3]),
                  "=r"(regs[4]),"=r"(regs[5]),"=r"(regs[6]),"=r"(regs[7]) : "r"(tmem_addr(base_t, 0, 256 + c)));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            int row = threadIdx.x;
            int q_idx = q_block * 128 + row;
            if (q_idx < S) {
                for (int i = 0; i < 4; ++i) {
                    float f0 = __uint_as_float(regs[i*2]);
                    float f1 = __uint_as_float(regs[i*2+1]);
                    __nv_bfloat162 bf2 = __halves2bfloat162(__float2bfloat16(f0), __float2bfloat16(f1));
                    if (c + i*2 < 128) {
                        atomicAdd((__nv_bfloat162*)&dQ[bh_offset * S * 128 + q_idx * 128 + c + i*2], bf2);
                    }
                }
            }
        }
        tcgen05_fence_after_fn();
        __syncthreads();
    }

    // Write dV and dK
    for (int c = 0; c < 128; c += 8) {
        uint32_t regs[8];
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=r"(regs[0]),"=r"(regs[1]),"=r"(regs[2]),"=r"(regs[3]),
              "=r"(regs[4]),"=r"(regs[5]),"=r"(regs[6]),"=r"(regs[7]) : "r"(tmem_addr(base_t, 0, 128 + c)));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        int row = threadIdx.x;
        int kv_idx = kv_start + row;
        if (kv_idx < S) {
            for (int i = 0; i < 4; ++i) {
                float f0 = __uint_as_float(regs[i*2]);
                float f1 = __uint_as_float(regs[i*2+1]);
                *(__nv_bfloat162*)&dV[bh_offset * S * 128 + kv_idx * 128 + c + i*2] = __halves2bfloat162(__float2bfloat16(f0), __float2bfloat16(f1));
            }
        }
    }
    for (int c = 0; c < 128; c += 8) {
        uint32_t regs[8];
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=r"(regs[0]),"=r"(regs[1]),"=r"(regs[2]),"=r"(regs[3]),
              "=r"(regs[4]),"=r"(regs[5]),"=r"(regs[6]),"=r"(regs[7]) : "r"(tmem_addr(base_t, 0, 384 + c)));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        int row = threadIdx.x;
        int kv_idx = kv_start + row;
        if (kv_idx < S) {
            for (int i = 0; i < 4; ++i) {
                float f0 = __uint_as_float(regs[i*2]);
                float f1 = __uint_as_float(regs[i*2+1]);
                *(__nv_bfloat162*)&dK[bh_offset * S * 128 + kv_idx * 128 + c + i*2] = __halves2bfloat162(__float2bfloat16(f0), __float2bfloat16(f1));
            }
        }
    }

    if (threadIdx.x < 32) {
        tmem_dealloc_fn(base_t, 512);
    }
}

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle) {
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
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L, tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    int d = Q.size(3);

    float* D_buf;
    CUDA_CHECK(cudaMallocAsync(&D_buf, B * H * S * sizeof(float), stream));
    int pre_blocks = (B * H * S + 255) / 256;
    precompute_D_kernel<<<pre_blocks, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(O.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        D_buf, B, H, S, d
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemsetAsync(dQ.data_ptr(), 0, B * H * S * d * sizeof(__nv_bfloat16), stream));

    CUtensorMap tma_K, tma_V, tma_Q, tma_dO;
    create_tma_2d_descriptor_2B(&tma_K, K.data_ptr(), d, B * H * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B);
    create_tma_2d_descriptor_2B(&tma_V, V.data_ptr(), d, B * H * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B);
    create_tma_2d_descriptor_2B(&tma_Q, Q.data_ptr(), d, B * H * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B);
    create_tma_2d_descriptor_2B(&tma_dO, dO.data_ptr(), d, B * H * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B);

    int num_kv_blocks = (S + 127) / 128;
    dim3 grid(num_kv_blocks, B * H);
    dim3 block(128);
    
    int smem_size = sizeof(SharedStorage);
    CUDA_CHECK(cudaFuncSetAttribute(mha_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    mha_bwd_kernel<<<grid, block, smem_size, stream>>>(
        tma_K, tma_V, tma_Q, tma_dO,
        static_cast<const float*>(L.data_ptr()), D_buf,
        static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        static_cast<__nv_bfloat16*>(dK.data_ptr()),
        static_cast<__nv_bfloat16*>(dV.data_ptr()),
        B, H, S
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFreeAsync(D_buf, stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

}  // namespace tvm_ffi_mha_bwd