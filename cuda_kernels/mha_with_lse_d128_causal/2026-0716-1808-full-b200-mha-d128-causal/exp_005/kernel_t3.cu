#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
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
                _e, __FILE__, __LINE__);                           \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_mha {

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

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(count));
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
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(bar)),
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

__device__ __forceinline__ uint64_t make_smem_desc_sm100_k_major(void* smem_ptr) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (addr & 0x3FFFF) >> 4;
    d |= ((uint64_t)16 >> 4) << 16;
    d |= ((uint64_t)1024 >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61; // SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_n_major(void* smem_ptr) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint32_t sbo = 128;
    uint32_t lbo = 8192;
    d |= (addr & 0x3FFFF) >> 4;
    d |= (lbo & 0x3FFFF) >> 4;
    d |= ((sbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61; // SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= (0u << 15);   
    d |= (0u << 16);   
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn_pv(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= (0u << 15);   
    d |= (1u << 16);   
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

__device__ __forceinline__ void umma_commit_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared.b64 [%0];"
        :: "r"(a) : "memory");
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ uint32_t pack_fp32_to_bf16(float a, float b) {
    __nv_bfloat16 ba = __float2bfloat16(a);
    __nv_bfloat16 bb = __float2bfloat16(b);
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};"
        : "=r"(result)
        : "h"(*reinterpret_cast<uint16_t*>(&ba)),
          "h"(*reinterpret_cast<uint16_t*>(&bb)));
    return result;
}

__device__ __forceinline__ void read_S_from_tmem(
    float* S_val, int row, int global_row, int global_col_start, int S_len,
    uint32_t tmem_S, float* m_val, float* l_val, float* m_new, float* l_new) 
{
    float temp_m = -1e20f;
    float temp_l = 0.0f;
    float scale = 1.0f / sqrtf(128.0f);

    for (int col = 0; col < 128; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_S + col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        float val_0 = __uint_as_float(r0);
        float val_1 = __uint_as_float(r1);
        float val_2 = __uint_as_float(r2);
        float val_3 = __uint_as_float(r3);

        int global_col = global_col_start + col;
        if (global_col > global_row || global_col >= S_len) {
            val_0 = -1e20f;
            val_1 = -1e20f;
            val_2 = -1e20f;
            val_3 = -1e20f;
        } else {
            val_0 *= scale;
            val_1 *= scale;
            val_2 *= scale;
            val_3 *= scale;
        }
        
        temp_m = max(temp_m, max(max(val_0, val_1), max(val_2, val_3)));
        S_val[col] = val_0;
        S_val[col+1] = val_1;
        S_val[col+2] = val_2;
        S_val[col+3] = val_3;
    }
    
    float m_curr = *m_val;
    float m_n = max(m_curr, temp_m);
    float factor = fast_exp2f_fn((m_curr - m_n) * 1.44269504f);
    *l_val *= factor;
    *m_new = m_n;
    temp_l *= factor;
    
    for (int col = 0; col < 128; ++col) {
        float exp_f = fast_exp2f_fn((S_val[col] - m_n) * 1.44269504f);
        temp_l += exp_f;
        S_val[col] = exp_f;
    }
    *l_new = temp_l;
}

__global__ __launch_bounds__(128) void mha_with_lse_d128_causal(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O, float* LSE, int S, int num_heads)
{
    int b_idx = blockIdx.y / H;
    int h_idx = blockIdx.y % H;
    int i = blockIdx.x * 128;
    int tid = threadIdx.x;

    extern __shared__ __align__(1024) char smem_buf[];
    __nv_bfloat16* smem_Q_cur = (__nv_bfloat16*)smem_buf;                
    __nv_bfloat16* smem_Q_next = (__nv_bfloat16*)(smem_buf + 32768);     
    __nv_bfloat16* smem_K = (__nv_bfloat16*)(smem_buf + 65536);          
    __nv_bfloat16* smem_V = (__nv_bfloat16*)(smem_buf + 98304);          
    __nv_bfloat16* smem_P_0 = (__nv_bfloat16*)(smem_buf + 131072);       
    __nv_bfloat16* smem_P_1 = (__nv_bfloat16*)(smem_buf + 147456);       

    extern __shared__ __align__(8) uint64_t bar_Q_cur_dyn[];
    uint64_t* bar_Q_cur = bar_Q_cur_dyn;
    uint64_t* bar_Q_next = bar_Q_cur_dyn + 1;
    uint64_t* bar_K = bar_Q_cur_dyn + 2;
    uint64_t* bar_V = bar_Q_cur_dyn + 3;
    uint64_t* bar_QK = bar_Q_cur_dyn + 4;
    uint64_t* bar_PV = bar_Q_cur_dyn + 5;

    extern __shared__ __align__(4) uint32_t tmem_S_buf[];
    uint32_t* tmem_S = tmem_S_buf;
    uint32_t* tmem_O_0 = tmem_S_buf + 1;
    uint32_t* tmem_O_1 = tmem_S_buf + 2;
    uint32_t* tmem_P = tmem_S_buf + 3;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(bar_Q_cur, 1);
        init_smem_barrier_fn(bar_Q_next, 1);
        init_smem_barrier_fn(bar_K, 1);
        init_smem_barrier_fn(bar_V, 1);
        init_smem_barrier_fn(bar_QK, 1);
        init_smem_barrier_fn(bar_PV, 1);
        
        tmem_alloc_fn(tmem_S, 128);
        tmem_alloc_fn(tmem_O_0, 64);
        tmem_alloc_fn(tmem_O_1, 64);
        tmem_alloc_fn(tmem_P, 128);
    }
    __syncthreads();

    int phase_Q_cur = 0;
    int phase_Q_next = 0;
    int phase_K = 0;
    int phase_V = 0;
    int phase_QK = 0;
    int phase_PV = 0;

    int s_base = (b_idx * num_heads + h_idx) * S;

    if (threadIdx.x == 0 && i < S) {
        mbarrier_arrive_and_expect_tx_fn(bar_Q_cur, 32768);
        tma_load_2d_fn(&tma_Q, bar_Q_cur, smem_Q_cur, 0, s_base + i);
        tma_load_2d_fn(&tma_Q, bar_Q_cur, smem_Q_cur + 8192, 64, s_base + i);
    }
    mbarrier_wait_fn(bar_Q_cur, phase_Q_cur);
    phase_Q_cur ^= 1;

    float m_val_0 = -1e20f;
    float l_val_0 = 0.0f;

    uint32_t idesc_qk = make_instr_desc_fn(128, 128);
    uint32_t idesc_pv = make_instr_desc_fn_pv(128, 64);

    for (; i < S; i += 128) {
        if (threadIdx.x == 0 && i + 128 < S) {
            mbarrier_arrive_and_expect_tx_fn(bar_Q_next, 32768);
            tma_load_2d_fn(&tma_Q, bar_Q_next, smem_Q_next, 0, s_base + i + 128);
            tma_load_2d_fn(&tma_Q, bar_Q_next, smem_Q_next + 8192, 64, s_base + i + 128);
        }
        
        // Zero tmem accumulator for fresh O computation
        if (threadIdx.x == 0) {
            for(int col = 0; col < 64; col += 4) {
                umma_f16_cg1_fn(*tmem_O_0 + col * 2, 0, 0, idesc_pv, 0);
                umma_f16_cg1_fn(*tmem_O_1 + col * 2, 0, 0, idesc_pv, 0);
            }
        }

        for (int j = 0; j <= i; j += 128) {
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(bar_K, 32768);
                tma_load_2d_fn(&tma_K, bar_K, smem_K, 0, s_base + j);
                tma_load_2d_fn(&tma_K, bar_K, smem_K + 8192, 64, s_base + j);
                
                mbarrier_arrive_and_expect_tx_fn(bar_V, 32768);
                tma_load_2d_fn(&tma_V, bar_V, smem_V, 0, s_base + j);
                tma_load_2d_fn(&tma_V, bar_V, smem_V + 8192, 64, s_base + j);
            }
            mbarrier_wait_fn(bar_K, phase_K);
            mbarrier_wait_fn(bar_V, phase_V);
            phase_K ^= 1;
            phase_V ^= 1;

            if (threadIdx.x == 0) {
                for (int k = 0; k < 128; k += 16) {
                    uint64_t desc_A, desc_B;
                    if (k < 64) {
                        desc_A = make_smem_desc_sm100_k_major(smem_Q_cur + k * 2);
                        desc_B = make_smem_desc_sm100_k_major(smem_K + k * 2);
                    } else {
                        desc_A = make_smem_desc_sm100_k_major(smem_Q_cur + 8192 + (k - 64) * 2);
                        desc_B = make_smem_desc_sm100_k_major(smem_K + 8192 + (k - 64) * 2);
                    }
                    if (k == 0) {
                        umma_f16_cg1_fn(*tmem_S, desc_A, desc_B, idesc_qk, 0);
                    } else {
                        umma_f16_cg1_fn(*tmem_S, desc_A, desc_B, idesc_qk, 1);
                    }
                }
                umma_commit_fn(bar_QK);
            }
            mbarrier_wait_fn(bar_QK, phase_QK);
            phase_QK ^= 1;

            float S_val[128];
            int global_row = s_base + i + tid;
            read_S_from_tmem(S_val, tid, global_row, j * 128, S, *tmem_S, &m_val_0, &l_val_0, &m_val_0, &l_val_0);

            float p_val_0[64];
            float p_val_1[64];
            for (int c = 0; c < 64; ++c) {
                p_val_0[c] = S_val[c];
                p_val_1[c] = S_val[c + 64];
            }

            for (int c = 0; c < 64; ++c) {
                uint32_t swizzled_idx_0 = ((c / 8) ^ (tid % 8)) * 8 + (c % 8);
                smem_P_0[tid * 64 + swizzled_idx_0] = __float2bfloat16(p_val_0[c]);
                
                uint32_t swizzled_idx_1 = ((c / 8) ^ (tid % 8)) * 8 + (c % 8);
                smem_P_1[tid * 64 + swizzled_idx_1] = __float2bfloat16(p_val_1[c]);
            }
            __syncthreads();

            if (threadIdx.x == 0) {
                uint32_t swizzled_smem_P_0 = (uint32_t)__cvta_generic_to_shared(smem_P_0);
                uint32_t swizzled_smem_P_1 = (uint32_t)__cvta_generic_to_shared(smem_P_1);
                uint32_t row_offset_0 = 0;
                uint32_t row_offset_1 = 8192;
                asm volatile("tcgen05.cp.cta_group::1.128x128b [%0], [%1];" 
                             :: "r"(*tmem_P + row_offset_0 * 4), "r"(swizzled_smem_P_0) : "memory");
                asm volatile("tcgen05.cp.cta_group::1.128x128b [%0], [%1];" 
                             :: "r"(*tmem_P + row_offset_1 * 4), "r"(swizzled_smem_P_1) : "memory");
                asm volatile("cp.async.bulk.commit_group;" ::: "memory");
                asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
            }
            __syncthreads();

            if (threadIdx.x == 0) {
                for (int k_pv = 0; k_pv < 128; k_pv += 16) {
                    uint64_t desc_V_0, desc_V_1;
                    
                    if (k_pv < 64) {
                        desc_V_0 = make_smem_desc_sm100_n_major(smem_V + k_pv * 128);
                        desc_V_1 = make_smem_desc_sm100_n_major(smem_V + 8192 + k_pv * 128);
                    } else {
                        desc_V_0 = make_smem_desc_sm100_n_major(smem_V + 8192 + (k_pv - 64) * 128);
                        desc_V_1 = make_smem_desc_sm100_n_major(smem_V + 16384 + (k_pv - 64) * 128);
                    }

                    if (k_pv == 0) {
                        asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n"
                                     :: "r"(*tmem_O_0), "r"(*tmem_P), "l"(desc_V_0), "r"(idesc_pv), "r"(0) : : "memory");
                        asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n"
                                     :: "r"(*tmem_O_1), "r"(*tmem_P + 4096), "l"(desc_V_1), "r"(idesc_pv), "r"(0) : : "memory");
                    } else {
                        asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n"
                                     :: "r"(*tmem_O_0 + k_pv * 2), "r"(*tmem_P + k_pv * 2), "l"(desc_V_0), "r"(idesc_pv), "r"(1) : : "memory");
                        asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n"
                                     :: "r"(*tmem_O_1 + k_pv * 2), "r"(*tmem_P + 4096 + k_pv * 2), "l"(desc_V_1), "r"(idesc_pv), "r"(1) : : "memory");
                    }
                }
                umma_commit_fn(bar_PV);
            }
            mbarrier_wait_fn(bar_PV, phase_PV);
            phase_PV ^= 1;
        }

        // Normalized epilogue from TMEM to O
        for (uint32_t col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(*tmem_O_0 + col * 4));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            float f0 = __uint_as_float(r0);
            float f1 = __uint_as_float(r1);
            float f2 = __uint_as_float(r2);
            float f3 = __uint_as_float(r3);
            
            uint32_t packed_0 = pack_fp32_to_bf16(f0 / l_val_0, f1 / l_val_0);
            uint32_t packed_1 = pack_fp32_to_bf16(f2 / l_val_0, f3 / l_val_0);
            
            uint32_t global_c = col;
            uint32_t global_row = s_base + i + tid;
            
            if (global_row < S) {
                *(reinterpret_cast<uint32_t*>(&O[global_row * 128 + global_c])) = packed_0;
                *(reinterpret_cast<uint32_t*>(&O[global_row * 128 + global_c + 2])) = packed_1;
            }
        }

        for (uint32_t col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(*tmem_O_1 + col * 4));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            float f0 = __uint_as_float(r0);
            float f1 = __uint_as_float(r1);
            float f2 = __uint_as_float(r2);
            float f3 = __uint_as_float(r3);
            
            uint32_t packed_0 = pack_fp32_to_bf16(f0 / l_val_0, f1 / l_val_0);
            uint32_t packed_1 = pack_fp32_to_bf16(f2 / l_val_0, f3 / l_val_0);
            
            uint32_t global_c = col + 64;
            uint32_t global_row = s_base + i + tid;
            
            if (global_row < S) {
                *(reinterpret_cast<uint32_t*>(&O[global_row * 128 + global_c])) = packed_0;
                *(reinterpret_cast<uint32_t*>(&O[global_row * 128 + global_c + 2])) = packed_1;
            }
        }

        // Output Log-Sum-Exp (LSE)
        if (tid < 128) {
            uint32_t global_row = s_base + i + tid;
            if (global_row < S) {
                int lse_idx = s_base + global_row;
                LSE[lse_idx] = m_val_0 + logf(l_val_0);
            }
        }
        
        __syncthreads();

        // Setup looping structures for next dual iteration block
        __nv_bfloat16* tmp_smem = smem_Q_cur;
        smem_Q_cur = smem_Q_next;
        smem_Q_next = tmp_smem;
        
        uint64_t* tmp_bar = bar_Q_cur;
        bar_Q_cur = bar_Q_next;
        bar_Q_next = tmp_bar;
        
        int tmp_phase = phase_Q_cur;
        phase_Q_cur = phase_Q_next;
        phase_Q_next = tmp_phase;
    }
    
    if (threadIdx.x == 0) {
        tmem_dealloc_fn(*tmem_S, 128);
        tmem_dealloc_fn(*tmem_O_0, 64);
        tmem_dealloc_fn(*tmem_O_1, 64);
        tmem_dealloc_fn(*tmem_P, 64);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);
    
    if (S == 0) return;

    int num_heads = B * H;
    int num_blocks = (S + 127) / 128;
    int smem_size = 160000; // ~156 KB comfortably resolves our 154 KB requirement.

    CUDA_CHECK(cudaFuncSetAttribute(
        mha_with_lse_d128_causal,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size));

    CUtensorMap tma_Q, tma_K, tma_V;
    uint64_t total_S = B * H * S;
    
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_Q, Q.data_ptr(), 128, total_S, 128, 128, CU_TENSOR_MAP_SWIZZLE_128B));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_K, K.data_ptr(), 128, total_S, 128, 128, CU_TENSOR_MAP_SWIZZLE_128B));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_V, V.data_ptr(), 128, total_S, 128, 128, CU_TENSOR_MAP_SWIZZLE_128B));

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    dim3 grid(num_blocks, num_heads);
    
    mha_with_lse_d128_causal<<<dim3(num_blocks, num_heads), 128, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, 
        static_cast<__nv_bfloat16*>(O.data_ptr()), 
        static_cast<float*>(LSE.data_ptr()), 
        S, num_heads);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha::run);

} // namespace tvm_ffi_mha