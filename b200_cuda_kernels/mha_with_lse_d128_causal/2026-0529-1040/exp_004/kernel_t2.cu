#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
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

// ------------------------------------------------------------------
// SM100 specific helper functions
// ------------------------------------------------------------------

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

__device__ __forceinline__ void st_shared_128_fn(uint32_t addr, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                 :: "r"(addr), "r"(v0), "r"(v1), "r"(v2), "r"(v3) : "memory");
}

__device__ __forceinline__ void tma_load_4d_cg1_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
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

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo, uint32_t swizzle = 0) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   // version = 1 (SM100)
    uint32_t base_offset = (addr >> 7) & 0x7;
    d |= (uint64_t)base_offset << 49;
    d |= (uint64_t)swizzle << 61;
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, uint32_t a_major, uint32_t b_major) {
    uint32_t d = 0;
    d |= (1u << 4);    // FP32 Out
    d |= (1u << 7);    // BF16 A
    d |= (1u << 10);   // BF16 B
    d |= (a_major << 15);
    d |= (b_major << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
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

__device__ __forceinline__ void tmem_load_8x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3, uint32_t* r4, uint32_t* r5, uint32_t* r6, uint32_t* r7) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3),
     "=r"(*r4),"=r"(*r5),"=r"(*r6),"=r"(*r7) : "r"(col));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ uint32_t pack_bf16_f32_fn(float a_f, float b_f) {
    __nv_bfloat16 a = __float2bfloat16(a_f);
    __nv_bfloat16 b = __float2bfloat16(b_f);
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};"
        : "=r"(result)
        : "h"(*reinterpret_cast<uint16_t*>(&a)),
          "h"(*reinterpret_cast<uint16_t*>(&b)));
    return result;
}


// ------------------------------------------------------------------
// Main Attention Kernel
// ------------------------------------------------------------------

__global__ __launch_bounds__(256, 1) void mha_causal_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int S, int H, int B
) {
    extern __shared__ __align__(128) char smem[];
    __nv_bfloat16* smem_Q0 = (__nv_bfloat16*)smem;
    __nv_bfloat16* smem_Q1 = smem_Q0 + 128 * 128;
    __nv_bfloat16* smem_K  = smem_Q1 + 128 * 128; // double buffered
    __nv_bfloat16* smem_V  = smem_K + 64 * 128 * 2; // double buffered
    __nv_bfloat16* smem_P0 = smem_V + 64 * 128 * 2;
    __nv_bfloat16* smem_P1 = smem_P0 + 128 * 64;
    
    uint64_t* mbar_Q    = (uint64_t*)(smem_P1 + 128 * 64);
    uint64_t* mbar_K    = mbar_Q + 1;
    uint64_t* mbar_V    = mbar_K + 2;
    uint64_t* mbar_mma0 = mbar_V + 2;
    uint64_t* mbar_mma1 = mbar_mma0 + 1;
    uint32_t* tmem_addr_smem = (uint32_t*)(mbar_mma1 + 1);

    int b_idx = blockIdx.z;
    int h_idx = blockIdx.y;
    int q_base = blockIdx.x * 256;
    
    int wg_id = threadIdx.x / 128;
    int tid_in_wg = threadIdx.x % 128;
    int q_wg_base = q_base + wg_id * 128;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(&mbar_K[0], 1);
        init_smem_barrier_fn(&mbar_K[1], 1);
        init_smem_barrier_fn(&mbar_V[0], 1);
        init_smem_barrier_fn(&mbar_V[1], 1);
        init_smem_barrier_fn(mbar_mma0, 1);
        init_smem_barrier_fn(mbar_mma1, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    if (threadIdx.x == 0) {
        tmem_alloc_cg1_fn(tmem_addr_smem, 256);
    }
    __syncthreads();
    
    uint32_t tmem_addr = *tmem_addr_smem;
    uint32_t TMEM_QK = tmem_addr + wg_id * 128;
    uint32_t TMEM_PV = tmem_addr + wg_id * 128 + 64;

    uint32_t q_phase = 0;
    uint32_t k_phase[2] = {0, 0};
    uint32_t v_phase[2] = {0, 0};
    uint32_t mma_phase = 0;

    if (threadIdx.x == 0) {
        int expect_bytes = 0;
        if (q_base < S) expect_bytes += 128 * 128 * 2;
        if (q_base + 128 < S) expect_bytes += 128 * 128 * 2;
        if (expect_bytes > 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_Q, expect_bytes);
            if (q_base < S) {
                tma_load_4d_cg1_fn(&tma_Q, mbar_Q, smem_Q0, 0, q_base, h_idx, b_idx);
                tma_load_4d_cg1_fn(&tma_Q, mbar_Q, smem_Q0 + 128*64, 64, q_base, h_idx, b_idx);
            }
            if (q_base + 128 < S) {
                tma_load_4d_cg1_fn(&tma_Q, mbar_Q, smem_Q1, 0, q_base + 128, h_idx, b_idx);
                tma_load_4d_cg1_fn(&tma_Q, mbar_Q, smem_Q1 + 128*64, 64, q_base + 128, h_idx, b_idx);
            }
        }
    }
    
    if (q_wg_base < S) {
        mbarrier_wait_fn(mbar_Q, q_phase); 
    }
    
    float m_i = -INFINITY;
    float l_i = 0.0f;
    float O_i[128];
    for (int i = 0; i < 128; ++i) {
        O_i[i] = 0.0f;
    }

    float scale = 1.0f / sqrtf(128.0f);
    int global_q = q_wg_base + tid_in_wg;

    __nv_bfloat16* my_smem_Q = (wg_id == 0) ? smem_Q0 : smem_Q1;
    __nv_bfloat16* my_smem_P = (wg_id == 0) ? smem_P0 : smem_P1;
    uint64_t* my_mbar_mma = (wg_id == 0) ? mbar_mma0 : mbar_mma1;

    uint32_t idesc_QK = make_instr_desc_fn(128, 64, 0, 0); // K-Maj, K-Maj
    uint32_t idesc_PV = make_instr_desc_fn(128, 64, 0, 1); // K-Maj, N-Maj

    if (threadIdx.x == 0 && S > 0) {
        mbarrier_arrive_and_expect_tx_fn(&mbar_K[0], 64 * 128 * 2);
        tma_load_4d_cg1_fn(&tma_K, &mbar_K[0], smem_K, 0, 0, h_idx, b_idx);
        tma_load_4d_cg1_fn(&tma_K, &mbar_K[0], smem_K + 64*64, 64, 0, h_idx, b_idx);
        
        mbarrier_arrive_and_expect_tx_fn(&mbar_V[0], 64 * 128 * 2);
        tma_load_4d_cg1_fn(&tma_V, &mbar_V[0], smem_V, 0, 0, h_idx, b_idx);
        tma_load_4d_cg1_fn(&tma_V, &mbar_V[0], smem_V + 64*64, 64, 0, h_idx, b_idx);
    }

    int k_end = q_base + 256;
    if (k_end > S) k_end = S;
    float chunk[64];

    for (int k_base = 0; k_base < k_end; k_base += 64) {
        int buf = (k_base / 64) % 2;
        int next_buf = 1 - buf;
        
        if (k_base + 64 < k_end) {
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(&mbar_K[next_buf], 64 * 128 * 2);
                tma_load_4d_cg1_fn(&tma_K, &mbar_K[next_buf], smem_K + next_buf * 64 * 128, 0, k_base + 64, h_idx, b_idx);
                tma_load_4d_cg1_fn(&tma_K, &mbar_K[next_buf], smem_K + next_buf * 64 * 128 + 64*64, 64, k_base + 64, h_idx, b_idx);
                
                mbarrier_arrive_and_expect_tx_fn(&mbar_V[next_buf], 64 * 128 * 2);
                tma_load_4d_cg1_fn(&tma_V, &mbar_V[next_buf], smem_V + next_buf * 64 * 128, 0, k_base + 64, h_idx, b_idx);
                tma_load_4d_cg1_fn(&tma_V, &mbar_V[next_buf], smem_V + next_buf * 64 * 128 + 64*64, 64, k_base + 64, h_idx, b_idx);
            }
        }

        bool wg_active = (q_wg_base < S) && (k_base < q_wg_base + 128);

        mbarrier_wait_fn(&mbar_K[buf], k_phase[buf]);

        if (wg_active) {
            __nv_bfloat16* cur_K = smem_K + buf * 64 * 128;
            if (tid_in_wg == 0) {
                #pragma unroll
                for (int k_mma = 0; k_mma < 64; k_mma += 16) {
                    uint64_t d_Q = make_smem_desc_sm100_fn((char*)my_smem_Q + k_mma * 2, 1, 1024, 2);
                    uint64_t d_K = make_smem_desc_sm100_fn((char*)cur_K + k_mma * 2, 1, 1024, 2);
                    uint32_t accum = (k_mma == 0) ? 0 : 1;
                    umma_f16_cg1_fn(TMEM_QK, d_Q, d_K, idesc_QK, accum);
                }
                #pragma unroll
                for (int k_mma = 0; k_mma < 64; k_mma += 16) {
                    uint64_t d_Q = make_smem_desc_sm100_fn((char*)my_smem_Q + 128*64*2 + k_mma * 2, 1, 1024, 2);
                    uint64_t d_K = make_smem_desc_sm100_fn((char*)cur_K + 64*64*2 + k_mma * 2, 1, 1024, 2);
                    umma_f16_cg1_fn(TMEM_QK, d_Q, d_K, idesc_QK, 1);
                }
                umma_commit_cg1_fn(my_mbar_mma);
            }
            mbarrier_wait_fn(my_mbar_mma, mma_phase);

            #pragma unroll
            for (int c = 0; c < 64; c += 8) {
                tmem_load_8x_fn(TMEM_QK + c, (uint32_t*)&chunk[c], (uint32_t*)&chunk[c+1], 
                    (uint32_t*)&chunk[c+2], (uint32_t*)&chunk[c+3], 
                    (uint32_t*)&chunk[c+4], (uint32_t*)&chunk[c+5], 
                    (uint32_t*)&chunk[c+6], (uint32_t*)&chunk[c+7]);
            }
            tmem_load_fence_fn();

            float row_max = m_i;
            #pragma unroll
            for (int c = 0; c < 64; ++c) {
                int global_k = k_base + c;
                if (global_k > global_q) {
                    chunk[c] = -INFINITY;
                } else {
                    chunk[c] *= scale;
                    row_max = fmaxf(row_max, chunk[c]);
                }
            }
            
            // To prevent MUFU unit contention, WG0 computes exp first.
            if (wg_id == 0) {
                float exp_diff = fast_exp2f_fn((m_i - row_max) * 1.44269504f);
                l_i *= exp_diff;
                #pragma unroll
                for (int i = 0; i < 128; ++i) O_i[i] *= exp_diff;
                m_i = row_max;

                float row_sum = 0.0f;
                #pragma unroll
                for (int c = 0; c < 64; ++c) {
                    if (chunk[c] != -INFINITY) {
                        chunk[c] = fast_exp2f_fn((chunk[c] - row_max) * 1.44269504f);
                        row_sum += chunk[c];
                    } else {
                        chunk[c] = 0.0f;
                    }
                }
                l_i += row_sum;
            }
        }
        
        __syncthreads(); // Sync 1
        
        if (wg_active && wg_id == 1) {
            float exp_diff = fast_exp2f_fn((m_i - row_max) * 1.44269504f);
            l_i *= exp_diff;
            #pragma unroll
            for (int i = 0; i < 128; ++i) O_i[i] *= exp_diff;
            m_i = row_max;

            float row_sum = 0.0f;
            #pragma unroll
            for (int c = 0; c < 64; ++c) {
                if (chunk[c] != -INFINITY) {
                    chunk[c] = fast_exp2f_fn((chunk[c] - row_max) * 1.44269504f);
                    row_sum += chunk[c];
                } else {
                    chunk[c] = 0.0f;
                }
            }
            l_i += row_sum;
        }
        
        __syncthreads(); // Sync 2
        
        if (wg_active) {
            #pragma unroll
            for (int c = 0; c < 64; c += 8) {
                int x = c / 8;
                int y = tid_in_wg;
                int x_swizzled = (y % 8) ^ x;
                uint32_t p0 = pack_bf16_f32_fn(chunk[c], chunk[c+1]);
                uint32_t p1 = pack_bf16_f32_fn(chunk[c+2], chunk[c+3]);
                uint32_t p2 = pack_bf16_f32_fn(chunk[c+4], chunk[c+5]);
                uint32_t p3 = pack_bf16_f32_fn(chunk[c+6], chunk[c+7]);
                uint32_t addr = (uint32_t)__cvta_generic_to_shared(my_smem_P) + (y * 8 + x_swizzled) * 16;
                st_shared_128_fn(addr, p0, p1, p2, p3);
            }
            fence_async_shared_fn();
        }

        mbarrier_wait_fn(&mbar_V[buf], v_phase[buf]);

        if (wg_active) {
            __nv_bfloat16* cur_V = smem_V + buf * 64 * 128;
            if (tid_in_wg == 0) {
                #pragma unroll
                for (int k_mma = 0; k_mma < 64; k_mma += 16) {
                    uint64_t d_P = make_smem_desc_sm100_fn((char*)my_smem_P + k_mma * 2, 1, 1024, 2);
                    uint64_t d_V = make_smem_desc_sm100_fn((char*)cur_V + k_mma * 128, 8192, 1024, 2);
                    uint32_t accum = (k_mma == 0) ? 0 : 1;
                    umma_f16_cg1_fn(TMEM_PV, d_P, d_V, idesc_PV, accum);
                }
                umma_commit_cg1_fn(my_mbar_mma);
            }
            mma_phase ^= 1;
            mbarrier_wait_fn(my_mbar_mma, mma_phase);
            
            #pragma unroll
            for (int c = 0; c < 64; c += 8) {
                tmem_load_8x_fn(TMEM_PV + c, (uint32_t*)&chunk[c], (uint32_t*)&chunk[c+1], 
                    (uint32_t*)&chunk[c+2], (uint32_t*)&chunk[c+3], 
                    (uint32_t*)&chunk[c+4], (uint32_t*)&chunk[c+5], 
                    (uint32_t*)&chunk[c+6], (uint32_t*)&chunk[c+7]);
            }
            tmem_load_fence_fn();
            
            #pragma unroll
            for (int c = 0; c < 64; ++c) O_i[c] += chunk[c];
            
            if (tid_in_wg == 0) {
                #pragma unroll
                for (int k_mma = 0; k_mma < 64; k_mma += 16) {
                    uint64_t d_P = make_smem_desc_sm100_fn((char*)my_smem_P + k_mma * 2, 1, 1024, 2);
                    uint64_t d_V = make_smem_desc_sm100_fn((char*)cur_V + 64*64*2 + k_mma * 128, 8192, 1024, 2);
                    uint32_t accum = (k_mma == 0) ? 0 : 1;
                    umma_f16_cg1_fn(TMEM_PV, d_P, d_V, idesc_PV, accum);
                }
                umma_commit_cg1_fn(my_mbar_mma);
            }
            mma_phase ^= 1;
            mbarrier_wait_fn(my_mbar_mma, mma_phase);
            
            #pragma unroll
            for (int c = 0; c < 64; c += 8) {
                tmem_load_8x_fn(TMEM_PV + c, (uint32_t*)&chunk[c], (uint32_t*)&chunk[c+1], 
                    (uint32_t*)&chunk[c+2], (uint32_t*)&chunk[c+3], 
                    (uint32_t*)&chunk[c+4], (uint32_t*)&chunk[c+5], 
                    (uint32_t*)&chunk[c+6], (uint32_t*)&chunk[c+7]);
            }
            tmem_load_fence_fn();
            
            #pragma unroll
            for (int c = 0; c < 64; ++c) O_i[c + 64] += chunk[c];
        }

        __syncthreads(); // Complete double buffer phase
        
        k_phase[buf] ^= 1;
        v_phase[buf] ^= 1;
    }

    if (q_wg_base < S && global_q < S) {
        float inv_sum = 1.0f / l_i;
        uint64_t o_offset = (uint64_t)b_idx * H * S * 128 + (uint64_t)h_idx * S * 128 + (uint64_t)global_q * 128;
        __nv_bfloat16* O_ptr = O + o_offset;
        
        #pragma unroll
        for (int c = 0; c < 128; c += 8) {
            uint32_t p0 = pack_bf16_f32_fn(O_i[c] * inv_sum, O_i[c+1] * inv_sum);
            uint32_t p1 = pack_bf16_f32_fn(O_i[c+2] * inv_sum, O_i[c+3] * inv_sum);
            uint32_t p2 = pack_bf16_f32_fn(O_i[c+4] * inv_sum, O_i[c+5] * inv_sum);
            uint32_t p3 = pack_bf16_f32_fn(O_i[c+6] * inv_sum, O_i[c+7] * inv_sum);
            *reinterpret_cast<uint4*>(&O_ptr[c]) = make_uint4(p0, p1, p2, p3);
        }
        
        uint64_t lse_offset = (uint64_t)b_idx * H * S + (uint64_t)h_idx * S + (uint64_t)global_q;
        LSE[lse_offset] = m_i + logf(l_i);
    }

    if (threadIdx.x == 0) {
        tmem_dealloc_cg1_fn(tmem_addr, 256);
    }
}

// ------------------------------------------------------------------
// Host Run Function
// ------------------------------------------------------------------

CUresult create_tma_4d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t D, uint64_t S, uint64_t H, uint64_t B, uint32_t smem_D, uint32_t smem_S) {
    cuuint64_t globalDim[4] = {D, S, H, B};
    cuuint64_t globalStrides[3] = {D * 2, S * D * 2, H * S * D * 2};
    cuuint32_t boxDim[4] = {smem_D, smem_S, 1, 1};
    cuuint32_t elementStrides[4] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(
        d,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        4,
        globalAddress,
        globalDim,
        globalStrides,
        boxDim,
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

namespace tvm_ffi_example_cuda {

void run(
    tvm::ffi::TensorView Q, 
    tvm::ffi::TensorView K, 
    tvm::ffi::TensorView V,
    tvm::ffi::TensorView O,
    tvm::ffi::TensorView LSE
) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    int D = Q.size(3);

    CUtensorMap tma_Q, tma_K, tma_V;
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_Q, Q.data_ptr(), D, S, H, B, 64, 128));
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_K, K.data_ptr(), D, S, H, B, 64, 64));
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_V, V.data_ptr(), D, S, H, B, 64, 64));

    dim3 grid((S + 255) / 256, H, B);
    dim3 block(256);
    int smem_bytes = 164000;
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    CUDA_CHECK(cudaFuncSetAttribute(mha_causal_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

    mha_causal_kernel<<<grid, block, smem_bytes, stream>>>(
        tma_Q, tma_K, tma_V, 
        static_cast<__nv_bfloat16*>(O.data_ptr()), 
        static_cast<float*>(LSE.data_ptr()), 
        S, H, B
    );
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda