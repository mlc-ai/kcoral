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
    d |= ((uint64_t)1 >> 4) << 16; // LBO=1 (unused for K-major SWIZZLE_128B)
    d |= ((uint64_t)1024 >> 4) << 32; // SBO=1024 bytes -> 8 spans along M-mode
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61; // SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_n_major(void* smem_ptr) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint32_t sbo = 1024; // SBO = 8 * 128 = 1024 bytes
    uint32_t lbo = 8192; // LBO = (64 / 8) * 1024 = 8192 bytes
    d |= (addr & 0x3FFFF) >> 4;
    d |= (lbo & 0x3FFFF) >> 4;
    d |= ((sbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61; // SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn_cg2(uint32_t M, uint32_t N) {
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

__device__ __forceinline__ uint32_t make_instr_desc_fn_pv_cg2(uint32_t M, uint32_t N) {
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
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile(
        "tcgen05.commit.cta_group::2"
        ".mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
        " [%0], %1;"
        :: "r"(a), "h"((uint16_t)0x3));
}

__device__ __forceinline__ float exp2_fast(float x) {
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

__device__ __forceinline__ void zero_tmem_64x64(uint32_t tmem_base, uint32_t row_offset) {
    uint32_t tmem_col = (row_offset % 8) * 8;
    uint32_t addr = (row_offset << 16) | tmem_col;
    uint32_t zero = 0;
    asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {%0,%1,%2,%3};"
                 :: "r"(zero),"r"(zero),"r"(zero),"r"(zero) : "r"(addr));
}

__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
}

__global__ __launch_bounds__(128, 2) void mha_with_lse_d128_causal(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O, float* LSE, int S, int H, int B)
{
    setmaxnreg_inc_sync_fn<248>();
    
    int b_idx = blockIdx.z;
    int h_idx = blockIdx.y;
    int i = blockIdx.x * 128;
    int tid = threadIdx.x;

    extern __shared__ __align__(1024) char smem_buf[];
    __nv_bfloat16* smem_Q_cur = (__nv_bfloat16*)smem_buf;                
    __nv_bfloat16* smem_Q_next = (__nv_bfloat16*)(smem_buf + 32768);     
    __nv_bfloat16* smem_K = (__nv_bfloat16*)(smem_buf + 65536);          
    __nv_bfloat16* smem_V = (__nv_bfloat16*)(smem_buf + 98304);          
    __nv_bfloat16* smem_P = (__nv_bfloat16*)(smem_buf + 131072);         

    extern __shared__ __align__(8) uint64_t bar_Q_cur_dyn[];
    uint64_t* bar_Q_cur = bar_Q_cur_dyn;
    uint64_t* bar_Q_next = bar_Q_cur_dyn + 1;
    uint64_t* bar_K = bar_Q_cur_dyn + 2;
    uint64_t* bar_V = bar_Q_cur_dyn + 3;
    uint64_t* bar_QK = bar_Q_cur_dyn + 4;
    uint64_t* bar_PV = bar_Q_cur_dyn + 5;

    extern __shared__ __align__(4) uint32_t tmem_S_addr[];
    uint32_t* tmem_O_0_addr = tmem_S_addr + 1;
    uint32_t* tmem_O_1_addr = tmem_S_addr + 2;
    uint32_t* tmem_P_addr = tmem_S_addr + 3;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(bar_Q_cur, 1);
        init_smem_barrier_fn(bar_Q_next, 1);
        init_smem_barrier_fn(bar_K, 1);
        init_smem_barrier_fn(bar_V, 1);
        init_smem_barrier_fn(bar_QK, 1);
        init_smem_barrier_fn(bar_PV, 1);
        
        tmem_alloc_fn(tmem_S_addr, 128);
        tmem_alloc_fn(tmem_O_0_addr, 64);
        tmem_alloc_fn(tmem_O_1_addr, 64);
        tmem_alloc_fn(tmem_P_addr, 128);
    }
    __syncthreads();

    int phase_Q_cur = 0;
    int phase_Q_next = 0;
    int phase_K = 0;
    int phase_V = 0;
    int phase_QK = 0;
    int phase_PV = 0;

    int s_base = (b_idx * H + h_idx) * S;

    if (threadIdx.x == 0 && i < S) {
        mbarrier_arrive_and_expect_tx_fn(bar_Q_cur, 32768);
        tma_load_2d_fn(&tma_Q, bar_Q_cur, smem_Q_cur, 0, s_base + i);
        tma_load_2d_fn(&tma_Q, bar_Q_cur, smem_Q_cur + 8192, 64, s_base + i);
    }
    mbarrier_wait_fn(bar_Q_cur, phase_Q_cur);
    phase_Q_cur ^= 1;

    float m_val_0[128];
    float l_val_0[128];
    #pragma unroll 4
    for (int r = 0; r < 128; ++r) {
        m_val_0[r] = -1e20f;
        l_val_0[r] = 0.0f;
    }

    uint32_t idesc_qk = make_instr_desc_fn_cg2(128, 128);
    uint32_t idesc_pv = make_instr_desc_fn_pv_cg2(128, 64);

    float scale_factor = 1.0f / sqrtf(128.0f);

    // Zero O accumulators
    if (threadIdx.x == 0) {
        uint64_t zero_desc_0 = make_smem_desc_sm100_k_major(smem_Q_cur);
        uint64_t zero_desc_1 = make_smem_desc_sm100_k_major(smem_Q_cur + 1);
        
        for(int n = 0; n < 64; n += 16) {
            umma_f16_cg2_fn(tmem_O_0_addr[0] + n * 2, zero_desc_0, zero_desc_1, idesc_pv, 0);
            umma_f16_cg2_fn(tmem_O_1_addr[0] + n * 2, zero_desc_0, zero_desc_1, idesc_pv, 0);
        }
        umma_commit_2sm_fn(bar_PV);
    }
    mbarrier_wait_fn(bar_PV, phase_PV);
    phase_PV ^= 1;

    for (; i < S; i += 128) {
        if (threadIdx.x == 0 && i + 128 < S) {
            mbarrier_arrive_and_expect_tx_fn(bar_Q_next, 32768);
            tma_load_2d_fn(&tma_Q, bar_Q_next, smem_Q_next, 0, s_base + i + 128);
            tma_load_2d_fn(&tma_Q, bar_Q_next, smem_Q_next + 8192, 64, s_base + i + 128);
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
                mbarrier_arrive_and_expect_tx_fn(bar_QK, 0);
                
                for (int k = 0; k < 64; k += 16) {
                    uint64_t desc_A = make_smem_desc_sm100_k_major(smem_Q_cur + k * 2);
                    uint64_t desc_B = make_smem_desc_sm100_k_major(smem_K + k * 2);
                    if (k == 0) {
                        umma_f16_cg2_fn(tmem_S_addr[0], desc_A, desc_B, idesc_qk, 0);
                    } else {
                        umma_f16_cg2_fn(tmem_S_addr[0] + k * 2, desc_A, desc_B, idesc_qk, 1);
                    }
                }
                for (int k = 0; k < 64; k += 16) {
                    uint64_t desc_A = make_smem_desc_sm100_k_major(smem_Q_cur + 8192 + k * 2);
                    uint64_t desc_B = make_smem_desc_sm100_k_major(smem_K + 8192 + k * 2);
                    umma_f16_cg2_fn(tmem_S_addr[0] + (64 + k) * 2, desc_A, desc_B, idesc_qk, 1);
                }
                umma_commit_2sm_fn(bar_QK);
            }
            mbarrier_wait_fn(bar_QK, phase_QK);
            phase_QK ^= 1;

            float temp_m = -1e20f;
            float S_val[128];

            for (int col = 0; col < 128; col += 4) {
                uint32_t tmem_col = ((col / 8) ^ (tid % 8)) * 8 + (col % 8);
                uint32_t addr = (tid << 16) | tmem_col;
                uint32_t r0, r1, r2, r3;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(addr));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                
                float val_0 = __uint_as_float(r0);
                float val_1 = __uint_as_float(r1);
                float val_2 = __uint_as_float(r2);
                float val_3 = __uint_as_float(r3);

                int global_col = j * 128 + col;
                int query_row = i + tid;
                if (global_col > query_row || global_col >= S) {
                    val_0 = -1e20f;
                    val_1 = -1e20f;
                    val_2 = -1e20f;
                    val_3 = -1e20f;
                } else {
                    val_0 *= scale_factor;
                    val_1 *= scale_factor;
                    val_2 *= scale_factor;
                    val_3 *= scale_factor;
                }
                
                temp_m = max(temp_m, max(max(val_0, val_1), max(val_2, val_3)));
                S_val[col] = val_0;
                S_val[col+1] = val_1;
                S_val[col+2] = val_2;
                S_val[col+3] = val_3;
            }
            
            float m_prev = m_val_0[tid];
            float m_curr = max(m_prev, temp_m);
            float factor = exp2_fast((m_prev - m_curr) * 1.44269504f);
            l_val_0[tid] *= factor;
            
            float temp_l = 0.0f;
            for (int col = 0; col < 128; ++col) {
                float exp_f = exp2_fast((S_val[col] - m_curr) * 1.44269504f);
                temp_l += exp_f;
                S_val[col] = exp_f;
            }
            l_val_0[tid] += temp_l;
            m_val_0[tid] = m_curr;
            
            // Write P to SMEM with Swizzle
            for (int col = 0; col < 128; ++col) {
                int chunk = col / 64;
                int col_in_chunk = col % 64;
                uint32_t swizzled_col = ((col_in_chunk / 8) ^ (tid % 8)) * 8 + (col_in_chunk % 8);
                __nv_bfloat16 bf_exp = __float2bfloat16(S_val[col]);
                smem_P[tid * 128 + chunk * 64 + swizzled_col] = bf_exp;
            }
            
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

            if (threadIdx.x == 0) {
                uint32_t smem_P_phys = (uint32_t)__cvta_generic_to_shared(smem_P);
                asm volatile("tcgen05.cp.cta_group::2.128x128b [%0], [%1];" 
                             :: "r"(tmem_P_addr[0]), "r"(smem_P_phys) : "memory");
                asm volatile("cp.async.bulk.commit_group;" ::: "memory");
                asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
            }
            __syncthreads();

            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(bar_PV, 0);
                
                for (int k_pv = 0; k_pv < 64; k_pv += 16) {
                    uint64_t desc_P = make_smem_desc_sm100_k_major(smem_P + k_pv * 2);
                    uint64_t desc_V_0 = make_smem_desc_sm100_n_major(smem_V + k_pv * 128);
                    uint64_t desc_V_1 = make_smem_desc_sm100_n_major(smem_V + 8192 + k_pv * 128);
                    
                    if (k_pv == 0) {
                        umma_f16_cg2_fn(tmem_O_0_addr[0], desc_P, desc_V_0, idesc_pv, 0);
                        umma_f16_cg2_fn(tmem_O_1_addr[0], desc_P, desc_V_1, idesc_pv, 0);
                    } else {
                        umma_f16_cg2_fn(tmem_O_0_addr[0] + k_pv * 2, desc_P, desc_V_0, idesc_pv, 1);
                        umma_f16_cg2_fn(tmem_O_1_addr[0] + k_pv * 2, desc_P, desc_V_1, idesc_pv, 1);
                    }
                }
                
                for (int k_pv = 0; k_pv < 64; k_pv += 16) {
                    uint64_t desc_P = make_smem_desc_sm100_k_major(smem_P + 8192 + k_pv * 2);
                    uint64_t desc_V_0 = make_smem_desc_sm100_n_major(smem_V + 64 + k_pv * 128);
                    uint64_t desc_V_1 = make_smem_desc_sm100_n_major(smem_V + 8192 + 64 + k_pv * 128);
                    
                    umma_f16_cg2_fn(tmem_O_0_addr[0] + k_pv * 2, desc_P, desc_V_0, idesc_pv, 1);
                    umma_f16_cg2_fn(tmem_O_1_addr[0] + k_pv * 2, desc_P, desc_V_1, idesc_pv, 1);
                }
                umma_commit_2sm_fn(bar_PV);
            }
            mbarrier_wait_fn(bar_PV, phase_PV);
            phase_PV ^= 1;
        }

        // Normalized epilogue from TMEM to O
        uint32_t c_base = (cluster_rank_fn() % 2) * 64;
        uint32_t r_base = (cluster_rank_fn() % 2) * 64;
        
        for (uint32_t col = 0; col < 64; col += 4) {
            uint32_t tmem_col = ((col / 8) ^ (((tid / 2) % 64) % 8)) * 8 + (col % 8);
            uint32_t addr = ((tid / 2) % 64 + r_base) << 16 | tmem_col;
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(addr));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            float f0 = __uint_as_float(r0);
            float f1 = __uint_as_float(r1);
            float f2 = __uint_as_float(r2);
            float f3 = __uint_as_float(r3);
            
            uint32_t packed_0 = pack_fp32_to_bf16(f0 / l_val_0[tid], f1 / l_val_0[tid]);
            uint32_t packed_1 = pack_fp32_to_bf16(f2 / l_val_0[tid], f3 / l_val_0[tid]);
            
            uint32_t global_row = s_base + i + r_base + (tid / 2) % 64;
            
            if (global_row < (uint32_t)(B * H * S)) {
                *(reinterpret_cast<uint32_t*>(&O[global_row * 128 + c_base + col])) = packed_0;
                *(reinterpret_cast<uint32_t*>(&O[global_row * 128 + c_base + col + 2])) = packed_1;
            }
        }

        for (uint32_t col = 0; col < 64; col += 4) {
            uint32_t tmem_col = ((col / 8) ^ (((tid / 2) % 64) % 8)) * 8 + (col % 8);
            uint32_t addr = ((tid / 2) % 64 + r_base) << 16 | tmem_col;
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(addr));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            float f0 = __uint_as_float(r0);
            float f1 = __uint_as_float(r1);
            float f2 = __uint_as_float(r2);
            float f3 = __uint_as_float(r3);
            
            uint32_t packed_0 = pack_fp32_to_bf16(f0 / l_val_0[tid], f1 / l_val_0[tid]);
            uint32_t packed_1 = pack_fp32_to_bf16(f2 / l_val_0[tid], f3 / l_val_0[tid]);
            
            uint32_t global_row = s_base + i + r_base + (tid / 2) % 64;
            
            if (global_row < (uint32_t)(B * H * S)) {
                *(reinterpret_cast<uint32_t*>(&O[global_row * 128 + c_base + 64 + col])) = packed_0;
                *(reinterpret_cast<uint32_t*>(&O[global_row * 128 + c_base + col + 2])) = packed_1;
            }
        }

        // Output Log-Sum-Exp (LSE)
        if (tid < 128) {
            uint32_t global_row = s_base + i + tid;
            if (global_row < (uint32_t)(B * H * S)) {
                LSE[global_row] = m_val_0[tid] + logf(l_val_0[tid]);
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
        tmem_dealloc_fn(tmem_S_addr[0], 128);
        tmem_dealloc_fn(tmem_O_0_addr[0], 64);
        tmem_dealloc_fn(tmem_O_1_addr[0], 64);
        tmem_dealloc_fn(tmem_P_addr[0], 64);
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

    int num_blocks = (S + 127) / 128;
    int smem_size = 196608; // 192 KB

    CUDA_CHECK(cudaFuncSetAttribute(
        mha_with_lse_d128_causal,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size));

    CUtensorMap tma_Q, tma_K, tma_V;
    uint64_t total_S = B * H * S;
    
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_Q, Q.data_ptr(), 128, total_S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_K, K.data_ptr(), 128, total_S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_V, V.data_ptr(), 128, total_S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B));

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    dim3 grid(num_blocks, H, B);
    
    mha_with_lse_d128_causal<<<dim3(num_blocks, H, B), 128, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, 
        static_cast<__nv_bfloat16*>(O.data_ptr()), 
        static_cast<float*>(LSE.data_ptr()), 
        S, H, B);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha::run);

} // namespace tvm_ffi_mha