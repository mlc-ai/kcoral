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

namespace tvm_ffi_mha {

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

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_after_fn() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_before_fn() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
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

__device__ __forceinline__ void st_shared_128_fn(uint32_t addr, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                 :: "r"(addr), "r"(v0), "r"(v1), "r"(v2), "r"(v3) : "memory");
}

__device__ __forceinline__ void tma_load_4d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}

__device__ __forceinline__ void tmem_alloc_cg1_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_cg1_fn(uint32_t addr, int ncols) {
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

__device__ __forceinline__ void umma_commit_cg1_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
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
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, uint32_t a_major, uint32_t b_major) {
    uint32_t d = 0;
    d |= (1u << 4);
    d |= (1u << 7);
    d |= (1u << 10);
    d |= (a_major << 15);
    d |= (b_major << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__device__ __forceinline__ uint64_t advance_smem_desc(uint64_t desc, uint32_t bytes) {
    uint32_t addr = (desc & 0x3FFFull) << 4;
    addr += bytes;
    desc &= ~0x3FFFull;
    desc |= ((addr >> 4) & 0x3FFFull);
    uint32_t base_offset = (addr >> 7) & 0x7;
    desc &= ~(0x7ull << 49);
    desc |= ((uint64_t)base_offset << 49);
    return desc;
}

__device__ __forceinline__ void tmem_scale_O(uint32_t tmem_O, float exp_diff) {
    for (int chunk = 0; chunk < 4; ++chunk) {
        uint32_t r[8][4];
        for (int i = 0; i < 8; ++i) {
            uint32_t col = (chunk * 8 + i) * 4;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r[i][0]),"=r"(r[i][1]),"=r"(r[i][2]),"=r"(r[i][3]) : "r"(tmem_O + col));
        }
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        for (int i = 0; i < 8; ++i) {
            uint32_t col = (chunk * 8 + i) * 4;
            r[i][0] = __float_as_uint(__uint_as_float(r[i][0]) * exp_diff);
            r[i][1] = __float_as_uint(__uint_as_float(r[i][1]) * exp_diff);
            r[i][2] = __float_as_uint(__uint_as_float(r[i][2]) * exp_diff);
            r[i][3] = __float_as_uint(__uint_as_float(r[i][3]) * exp_diff);
            asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {%0,%1,%2,%3};"
                :: "r"(r[i][0]),"r"(r[i][1]),"r"(r[i][2]),"r"(r[i][3]), "r"(tmem_O + col) : "memory");
        }
    }
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

extern __shared__ uint8_t dynamic_smem[];

__global__ void mha_forward_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* out, float* lse_out, int S) {

    uint8_t* smem_Q = dynamic_smem;
    uint8_t* smem_K[2] = {smem_Q + 32768, smem_Q + 65536};
    uint8_t* smem_V[2] = {smem_K[1] + 32768, smem_K[1] + 65536};
    uint8_t* smem_P[2] = {smem_V[1] + 32768, smem_V[1] + 65536};
    
    uint64_t* mbar_Q = (uint64_t*)(smem_P[1] + 32768);
    uint64_t* mbar_K = mbar_Q + 1;
    uint64_t* mbar_V = mbar_K + 2;
    uint64_t* mbar_S = mbar_V + 2;
    uint64_t* mbar_O = mbar_S + 1;
    uint32_t* smem_tmem_base = (uint32_t*)(mbar_O + 2);

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(&mbar_K[0], 1);
        init_smem_barrier_fn(&mbar_K[1], 1);
        init_smem_barrier_fn(&mbar_V[0], 1);
        init_smem_barrier_fn(&mbar_V[1], 1);
        init_smem_barrier_fn(mbar_S, 1);
        init_smem_barrier_fn(&mbar_O[0], 1);
        init_smem_barrier_fn(&mbar_O[1], 1);
        tmem_alloc_cg1_fn(smem_tmem_base, 256);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    uint32_t tmem_base = *smem_tmem_base;
    uint32_t tmem_O = tmem_base;
    uint32_t tmem_S = tmem_base + 128;

    int m_idx = blockIdx.x;
    int h_idx = blockIdx.y;
    int b_idx = blockIdx.z;
    uint32_t H = gridDim.y;

    int num_blocks = (S + 127) / 128;

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_Q, 32768);
        tma_load_4d_fn(&tma_Q, mbar_Q, smem_Q, 0, m_idx * 128, h_idx, b_idx);
        tma_load_4d_fn(&tma_Q, mbar_Q, smem_Q + 16384, 64, m_idx * 128, h_idx, b_idx);
        
        mbarrier_arrive_and_expect_tx_fn(&mbar_K[0], 32768);
        tma_load_4d_fn(&tma_K, &mbar_K[0], smem_K[0], 0, 0, h_idx, b_idx);
        tma_load_4d_fn(&tma_K, &mbar_K[0], smem_K[0] + 16384, 64, 0, h_idx, b_idx);
        
        mbarrier_arrive_and_expect_tx_fn(&mbar_V[0], 32768);
        tma_load_4d_fn(&tma_V, &mbar_V[0], smem_V[0], 0, 0, h_idx, b_idx);
        tma_load_4d_fn(&tma_V, &mbar_V[0], smem_V[0] + 16384, 64, 0, h_idx, b_idx);
    }

    mbarrier_wait_fn(mbar_Q, 0);

    float m_i = -INFINITY;
    float d_i = 0.0f;
    uint32_t idesc_S = make_instr_desc_fn(128, 128, 0, 0);
    uint32_t idesc_O = make_instr_desc_fn(128, 64, 0, 1);

    for (int n = 0; n < num_blocks; ++n) {
        int pp = n % 2;
        int phase_buf = (n / 2) & 1;
        int phase_S = n & 1;
        
        mbarrier_wait_fn(&mbar_K[pp], phase_buf);
        mbarrier_wait_fn(&mbar_V[pp], phase_buf);
        
        tcgen05_fence_before_fn();
        if (threadIdx.x == 0) {
            uint64_t desc_Q = make_smem_desc_sm100_fn(smem_Q, 1, 1024);
            uint64_t desc_K = make_smem_desc_sm100_fn(smem_K[pp], 1, 1024);
            for (int k = 0; k < 4; ++k) {
                umma_f16_cg1_fn(tmem_S, desc_Q, desc_K, idesc_S, (k == 0) ? 0 : 1);
                desc_Q = advance_smem_desc(desc_Q, 32);
                desc_K = advance_smem_desc(desc_K, 32);
            }
            desc_Q = make_smem_desc_sm100_fn(smem_Q + 16384, 1, 1024);
            desc_K = make_smem_desc_sm100_fn(smem_K[pp] + 16384, 1, 1024);
            for (int k = 4; k < 8; ++k) {
                umma_f16_cg1_fn(tmem_S, desc_Q, desc_K, idesc_S, 1);
                desc_Q = advance_smem_desc(desc_Q, 32);
                desc_K = advance_smem_desc(desc_K, 32);
            }
            umma_commit_cg1_fn(mbar_S);
        }
        
        mbarrier_wait_fn(mbar_S, phase_S);
        fence_proxy_async_fn();
        
        float row_S[128];
        float m_ij = -INFINITY;
        for (int chunk = 0; chunk < 4; ++chunk) {
            uint32_t r[8][4];
            for (int i = 0; i < 8; ++i) {
                uint32_t col = (chunk * 8 + i) * 4;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(r[i][0]),"=r"(r[i][1]),"=r"(r[i][2]),"=r"(r[i][3]) : "r"(tmem_S + col));
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            for (int i = 0; i < 8; ++i) {
                uint32_t col = (chunk * 8 + i) * 4;
                float s0 = __uint_as_float(r[i][0]) * 0.08838834764f;
                float s1 = __uint_as_float(r[i][1]) * 0.08838834764f;
                float s2 = __uint_as_float(r[i][2]) * 0.08838834764f;
                float s3 = __uint_as_float(r[i][3]) * 0.08838834764f;
                
                uint32_t s_idx = m_idx * 128 + threadIdx.x;
                uint32_t k_idx = n * 128 + col;
                if (s_idx >= S || k_idx + 0 >= S) s0 = -INFINITY;
                if (s_idx >= S || k_idx + 1 >= S) s1 = -INFINITY;
                if (s_idx >= S || k_idx + 2 >= S) s2 = -INFINITY;
                if (s_idx >= S || k_idx + 3 >= S) s3 = -INFINITY;
                
                row_S[col+0] = s0; row_S[col+1] = s1; row_S[col+2] = s2; row_S[col+3] = s3;
                m_ij = max(m_ij, max(max(s0, s1), max(s2, s3)));
            }
        }
        
        float m_i_new = max(m_i, m_ij);
        float exp_diff = fast_exp2f_fn((m_i - m_i_new) * 1.44269504f);
        d_i = d_i * exp_diff;
        
        if (n > 0 && exp_diff != 1.0f) {
            tmem_scale_O(tmem_O, exp_diff);
        }
        m_i = m_i_new;
        
        float d_ij = 0.0f;
        for (int i = 0; i < 16; ++i) {
            uint32_t col = i * 8;
            float p0 = fast_exp2f_fn((row_S[col+0] - m_i_new) * 1.44269504f);
            float p1 = fast_exp2f_fn((row_S[col+1] - m_i_new) * 1.44269504f);
            float p2 = fast_exp2f_fn((row_S[col+2] - m_i_new) * 1.44269504f);
            float p3 = fast_exp2f_fn((row_S[col+3] - m_i_new) * 1.44269504f);
            float p4 = fast_exp2f_fn((row_S[col+4] - m_i_new) * 1.44269504f);
            float p5 = fast_exp2f_fn((row_S[col+5] - m_i_new) * 1.44269504f);
            float p6 = fast_exp2f_fn((row_S[col+6] - m_i_new) * 1.44269504f);
            float p7 = fast_exp2f_fn((row_S[col+7] - m_i_new) * 1.44269504f);
            
            if (row_S[col+0] == -INFINITY) p0 = 0.0f;
            if (row_S[col+1] == -INFINITY) p1 = 0.0f;
            if (row_S[col+2] == -INFINITY) p2 = 0.0f;
            if (row_S[col+3] == -INFINITY) p3 = 0.0f;
            if (row_S[col+4] == -INFINITY) p4 = 0.0f;
            if (row_S[col+5] == -INFINITY) p5 = 0.0f;
            if (row_S[col+6] == -INFINITY) p6 = 0.0f;
            if (row_S[col+7] == -INFINITY) p7 = 0.0f;
            
            d_ij += p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7;
            
            uint32_t p01 = pack_bf16_fn(__float_as_uint(p0), __float_as_uint(p1));
            uint32_t p23 = pack_bf16_fn(__float_as_uint(p2), __float_as_uint(p3));
            uint32_t p45 = pack_bf16_fn(__float_as_uint(p4), __float_as_uint(p5));
            uint32_t p67 = pack_bf16_fn(__float_as_uint(p6), __float_as_uint(p7));
            
            uint32_t y = threadIdx.x;
            if (i < 8) {
                uint32_t x = i;
                uint32_t swizzled_x = (x & ~7) | ((y % 8) ^ (x & 7));
                uint32_t offset = y * 128 + swizzled_x * 16;
                st_shared_128_fn((uint32_t)__cvta_generic_to_shared(smem_P[pp] + offset), p01, p23, p45, p67);
            } else {
                uint32_t x = i - 8;
                uint32_t swizzled_x = (x & ~7) | ((y % 8) ^ (x & 7));
                uint32_t offset = 16384 + y * 128 + swizzled_x * 16;
                st_shared_128_fn((uint32_t)__cvta_generic_to_shared(smem_P[pp] + offset), p01, p23, p45, p67);
            }
        }
        d_i += d_ij;
        
        fence_async_shared_fn();
        tcgen05_fence_before_fn();
        
        if (threadIdx.x == 0) {
            uint64_t desc_P = make_smem_desc_sm100_fn(smem_P[pp], 1, 1024);
            uint64_t desc_V = make_smem_desc_sm100_fn(smem_V[pp], 16384, 1024);
            for (int k = 0; k < 4; ++k) {
                umma_f16_cg1_fn(tmem_O, desc_P, desc_V, idesc_O, (n == 0 && k == 0) ? 0 : 1);
                desc_P = advance_smem_desc(desc_P, 32);
                desc_V = advance_smem_desc(desc_V, 2048);
            }
            desc_P = make_smem_desc_sm100_fn(smem_P[pp] + 16384, 1, 1024);
            desc_V = make_smem_desc_sm100_fn(smem_V[pp] + 8192, 16384, 1024);
            for (int k = 4; k < 8; ++k) {
                umma_f16_cg1_fn(tmem_O, desc_P, desc_V, idesc_O, 1);
                desc_P = advance_smem_desc(desc_P, 32);
                desc_V = advance_smem_desc(desc_V, 2048);
            }
            
            desc_P = make_smem_desc_sm100_fn(smem_P[pp], 1, 1024);
            desc_V = make_smem_desc_sm100_fn(smem_V[pp] + 16384, 16384, 1024);
            for (int k = 0; k < 4; ++k) {
                umma_f16_cg1_fn(tmem_O + 64, desc_P, desc_V, idesc_O, (n == 0 && k == 0) ? 0 : 1);
                desc_P = advance_smem_desc(desc_P, 32);
                desc_V = advance_smem_desc(desc_V, 2048);
            }
            desc_P = make_smem_desc_sm100_fn(smem_P[pp] + 16384, 1, 1024);
            desc_V = make_smem_desc_sm100_fn(smem_V[pp] + 16384 + 8192, 16384, 1024);
            for (int k = 4; k < 8; ++k) {
                umma_f16_cg1_fn(tmem_O + 64, desc_P, desc_V, idesc_O, 1);
                desc_P = advance_smem_desc(desc_P, 32);
                desc_V = advance_smem_desc(desc_V, 2048);
            }
            umma_commit_cg1_fn(&mbar_O[pp]);
        }
        
        if (n + 1 < num_blocks) {
            int next_pp = (n + 1) % 2;
            if (n + 1 >= 2) {
                mbarrier_wait_fn(&mbar_O[next_pp], ((n + 1) / 2 - 1) & 1); 
            }
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(&mbar_K[next_pp], 32768);
                tma_load_4d_fn(&tma_K, &mbar_K[next_pp], smem_K[next_pp], 0, (n + 1) * 128, h_idx, b_idx);
                tma_load_4d_fn(&tma_K, &mbar_K[next_pp], smem_K[next_pp] + 16384, 64, (n + 1) * 128, h_idx, b_idx);
                
                mbarrier_arrive_and_expect_tx_fn(&mbar_V[next_pp], 32768);
                tma_load_4d_fn(&tma_V, &mbar_V[next_pp], smem_V[next_pp], 0, (n + 1) * 128, h_idx, b_idx);
                tma_load_4d_fn(&tma_V, &mbar_V[next_pp], smem_V[next_pp] + 16384, 64, (n + 1) * 128, h_idx, b_idx);
            }
        }
    }
    
    mbarrier_wait_fn(&mbar_O[(num_blocks - 1) % 2], ((num_blocks - 1) / 2) & 1);
    
    float inv_d = 1.0f / d_i;
    uint32_t s_idx = m_idx * 128 + threadIdx.x;
    
    for (int chunk = 0; chunk < 4; ++chunk) {
        uint32_t r[8][4];
        for (int i = 0; i < 8; ++i) {
            uint32_t col = (chunk * 8 + i) * 4;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r[i][0]),"=r"(r[i][1]),"=r"(r[i][2]),"=r"(r[i][3]) : "r"(tmem_O + col));
        }
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        for (int i = 0; i < 8; ++i) {
            uint32_t col = (chunk * 8 + i) * 4;
            float s0 = __uint_as_float(r[i][0]) * inv_d;
            float s1 = __uint_as_float(r[i][1]) * inv_d;
            float s2 = __uint_as_float(r[i][2]) * inv_d;
            float s3 = __uint_as_float(r[i][3]) * inv_d;
            
            uint32_t p01 = pack_bf16_fn(__float_as_uint(s0), __float_as_uint(s1));
            uint32_t p23 = pack_bf16_fn(__float_as_uint(s2), __float_as_uint(s3));
            
            if (s_idx < S) {
                uint32_t d_idx = col;
                uint2 p_vec = make_uint2(p01, p23);
                *(uint2*)&out[b_idx * H * S * 128 + h_idx * S * 128 + s_idx * 128 + d_idx] = p_vec;
            }
        }
    }
    
    if (s_idx < S) {
        lse_out[b_idx * H * S + h_idx * S + s_idx] = m_i + logf(d_i);
    }
    
    if (threadIdx.x == 0) {
        tmem_dealloc_cg1_fn(tmem_base, 256);
    }
}

CUresult create_tma_4d_descriptor_2B(CUtensorMap* d, void* globalAddress, 
    uint64_t dim0, uint64_t dim1, uint64_t dim2, uint64_t dim3,
    uint32_t box0, uint32_t box1, uint32_t box2, uint32_t box3,
    CUtensorMapSwizzle swizzle) {
    cuuint64_t globalDim[4] = {dim0, dim1, dim2, dim3};
    cuuint64_t globalStrides[3] = {dim0 * 2, dim0 * dim1 * 2, dim0 * dim1 * dim2 * 2};
    cuuint32_t boxDim[4] = {box0, box1, box2, box3};
    cuuint32_t elementStrides[4] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, globalAddress, globalDim, globalStrides,
        boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);

    CUtensorMap tma_Q, tma_K, tma_V;
    
    CUresult res = create_tma_4d_descriptor_2B(&tma_Q, Q.data_ptr(), D, S, H, B, 64, 128, 1, 1, CU_TENSOR_MAP_SWIZZLE_128B);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA Q failed %d\n", res); exit(1); }
    
    res = create_tma_4d_descriptor_2B(&tma_K, K.data_ptr(), D, S, H, B, 64, 128, 1, 1, CU_TENSOR_MAP_SWIZZLE_128B);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA K failed %d\n", res); exit(1); }
    
    res = create_tma_4d_descriptor_2B(&tma_V, V.data_ptr(), D, S, H, B, 64, 128, 1, 1, CU_TENSOR_MAP_SWIZZLE_128B);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA V failed %d\n", res); exit(1); }

    int grid_x = (S + 127) / 128;
    dim3 grid(grid_x, H, B);
    dim3 block(128);
    int smem_size = 230000;
    
    CUDA_CHECK(cudaFuncSetAttribute((void*)mha_forward_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    mha_forward_kernel<<<grid, block, smem_size, stream>>>(tma_Q, tma_K, tma_V, 
        static_cast<__nv_bfloat16*>(O.data_ptr()), 
        static_cast<float*>(LSE.data_ptr()), S);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha::run);

}  // namespace tvm_ffi_mha