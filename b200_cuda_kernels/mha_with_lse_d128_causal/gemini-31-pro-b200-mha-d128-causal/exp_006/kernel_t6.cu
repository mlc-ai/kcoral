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
        const char* errStr;                                        \
        cuGetErrorName(_e, &errStr);                               \
        fprintf(stderr, "CU error %s at %s:%d\n", errStr,          \
                __FILE__, __LINE__);                               \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace flash_attn_causal_v7 {

struct __align__(1024) SharedStorage {
    __align__(1024) __nv_bfloat16 Q0[128 * 128]; 
    __align__(1024) __nv_bfloat16 Q1[128 * 128]; 
    __align__(1024) __nv_bfloat16 K[128 * 128];  
    __align__(1024) __nv_bfloat16 V[128 * 128];  
    __align__(1024) __nv_bfloat16 P0[128 * 128]; 
    __align__(1024) __nv_bfloat16 P1[128 * 128]; 
    
    float m0[128];
    float l0[128];
    float m1[128];
    float l1[128];

    uint64_t mbar_q;
    uint64_t mbar_kv;
    uint64_t mbar_umma0;
    uint64_t mbar_umma1;
    uint64_t mbar_umma_pv;
    
    uint32_t tmem_S0;
    uint32_t tmem_S1;
    uint32_t tmem_O0;
    uint32_t tmem_O1;
};

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
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

__device__ __forceinline__ void tma_load_4d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
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

__device__ __forceinline__ uint64_t make_smem_desc_linear_K_major(void* ptr, uint32_t pitch) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    uint32_t lbo = 16;
    uint32_t sbo = 8 * pitch;
    uint64_t d = 0;
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)0 << 61; 
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_linear_MN_major(void* ptr, uint32_t pitch) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    uint32_t lbo = 8 * pitch;
    uint32_t sbo = 16;
    uint64_t d = 0;
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)0 << 61; 
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, int a_major, int b_major) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= ((uint32_t)a_major << 15);   
    d |= ((uint32_t)b_major << 16);   
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
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(a));
}

__device__ __forceinline__ void tmem_alloc_cg1_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_cg1_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(addr), "r"(ncols));
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
    asm("mov.b32 %0, {%1, %2};" : "=r"(result) : "h"(*reinterpret_cast<uint16_t*>(&a)), "h"(*reinterpret_cast<uint16_t*>(&b)));
    return result;
}

extern __shared__ __align__(1024) char smem_buf[];

__global__ void __launch_bounds__(384) flash_attn_causal_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* __restrict__ O_ptr,
    float* __restrict__ LSE_ptr,
    int B, int H, int S) 
{
    setmaxnreg_inc_sync_fn<256>();

    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(smem_buf);
    
    int b = blockIdx.y / H;
    int h = blockIdx.y % H;
    int block_q = blockIdx.x; 
    
    int global_q0_base = block_q * 256;
    int global_q1_base = block_q * 256 + 128;
    
    if (global_q0_base >= S) return;

    int tid = threadIdx.x;
    int wg = tid / 128;
    int wg_tid = tid % 128;
    
    if (tid == 0) {
        init_smem_barrier_fn(&smem.mbar_q, 1);
        init_smem_barrier_fn(&smem.mbar_kv, 1);
        init_smem_barrier_fn(&smem.mbar_umma0, 1);
        init_smem_barrier_fn(&smem.mbar_umma1, 1);
        init_smem_barrier_fn(&smem.mbar_umma_pv, 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();
    
    if (wg == 0) {
        smem.m0[wg_tid] = -INFINITY;
        smem.l0[wg_tid] = 0.0f;
    } else if (wg == 1) {
        smem.m1[wg_tid] = -INFINITY;
        smem.l1[wg_tid] = 0.0f;
    }
    __syncthreads();
    
    if (tid < 32) {
        tmem_alloc_cg1_fn(&smem.tmem_S0, 128);
        tmem_alloc_cg1_fn(&smem.tmem_S1, 128);
        tmem_alloc_cg1_fn(&smem.tmem_O0, 128);
        tmem_alloc_cg1_fn(&smem.tmem_O1, 128);
    }
    __syncthreads();
    
    if (tid == 256) {
        mbarrier_arrive_and_expect_tx_fn(&smem.mbar_q, 32768 * 2);
        tma_load_4d_fn(&tma_Q, &smem.mbar_q, smem.Q0, 0, global_q0_base, h, b);
        tma_load_4d_fn(&tma_Q, &smem.mbar_q, smem.Q1, 0, global_q1_base, h, b);
    }
    
    if (wg == 0) {
        for (int col = 0; col < 128; col += 4) {
            asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};"
                :: "r"(smem.tmem_O0 + col), "r"(0), "r"(0), "r"(0), "r"(0));
        }
    } else if (wg == 1) {
        for (int col = 0; col < 128; col += 4) {
            asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};"
                :: "r"(smem.tmem_O1 + col), "r"(0), "r"(0), "r"(0), "r"(0));
        }
    }
    
    if (wg == 0 || wg == 1) {
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    }
    
    int max_q = (S - 1 < global_q1_base + 127) ? S - 1 : global_q1_base + 127;
    int max_kv = max_q / 128;
    
    uint32_t q_phase = 0;
    uint32_t kv_phase = 0;
    uint32_t umma0_phase = 0;
    uint32_t umma1_phase = 0;
    uint32_t pv_phase = 0;
    
    for (int kv_idx = 0; kv_idx <= max_kv; ++kv_idx) {
        if (tid == 256) {
            if (kv_idx > 0) {
                mbarrier_wait_fn(&smem.mbar_umma_pv, pv_phase);
                pv_phase ^= 1;
            }
            
            mbarrier_arrive_and_expect_tx_fn(&smem.mbar_kv, 32768 * 2);
            tma_load_4d_fn(&tma_K, &smem.mbar_kv, smem.K, 0, kv_idx * 128, h, b);
            tma_load_4d_fn(&tma_V, &smem.mbar_kv, smem.V, 0, kv_idx * 128, h, b);
            
            if (kv_idx == 0) {
                mbarrier_wait_fn(&smem.mbar_q, q_phase);
                q_phase ^= 1;
            }
            
            mbarrier_wait_fn(&smem.mbar_kv, kv_phase);
            kv_phase ^= 1;
            
            fence_async_shared_fn();
            
            uint32_t idesc_qk = make_instr_desc_fn(128, 128, 0, 0); 
            for (int k = 0; k < 128; k += 16) {
                uint64_t desc_q0 = make_smem_desc_linear_K_major((char*)smem.Q0 + k * 2, 256);
                uint64_t desc_k  = make_smem_desc_linear_K_major((char*)smem.K + k * 2, 256);
                uint32_t accum   = (k == 0) ? 0 : 1;
                umma_f16_cg1_fn(smem.tmem_S0, desc_q0, desc_k, idesc_qk, accum);
            }
            umma_commit_cg1_fn(&smem.mbar_umma0);
            
            for (int k = 0; k < 128; k += 16) {
                uint64_t desc_q1 = make_smem_desc_linear_K_major((char*)smem.Q1 + k * 2, 256);
                uint64_t desc_k  = make_smem_desc_linear_K_major((char*)smem.K + k * 2, 256);
                uint32_t accum   = (k == 0) ? 0 : 1;
                umma_f16_cg1_fn(smem.tmem_S1, desc_q1, desc_k, idesc_qk, accum);
            }
            umma_commit_cg1_fn(&smem.mbar_umma1);
        }
        
        if (wg == 0) {
            mbarrier_wait_fn(&smem.mbar_umma0, umma0_phase);
            umma0_phase ^= 1;
            
            uint32_t r_S[32][4];
            for (uint32_t col = 0; col < 128; col += 4) {
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(r_S[col/4][0]),"=r"(r_S[col/4][1]),"=r"(r_S[col/4][2]),"=r"(r_S[col/4][3]) : "r"(smem.tmem_S0 + col));
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            float m_block = -INFINITY;
            int global_q = global_q0_base + wg_tid;
            for (uint32_t col = 0; col < 128; col += 4) {
                int global_k0 = kv_idx * 128 + col;
                
                float f0 = __uint_as_float(r_S[col/4][0]) * 0.08838834764f; 
                float f1 = __uint_as_float(r_S[col/4][1]) * 0.08838834764f;
                float f2 = __uint_as_float(r_S[col/4][2]) * 0.08838834764f;
                float f3 = __uint_as_float(r_S[col/4][3]) * 0.08838834764f;
                
                if (global_q >= S || global_k0 + 0 > global_q) f0 = -INFINITY;
                if (global_q >= S || global_k0 + 1 > global_q) f1 = -INFINITY;
                if (global_q >= S || global_k0 + 2 > global_q) f2 = -INFINITY;
                if (global_q >= S || global_k0 + 3 > global_q) f3 = -INFINITY;
                
                m_block = fmaxf(m_block, fmaxf(f0, fmaxf(f1, fmaxf(f2, f3))));
                
                r_S[col/4][0] = __float_as_uint(f0);
                r_S[col/4][1] = __float_as_uint(f1);
                r_S[col/4][2] = __float_as_uint(f2);
                r_S[col/4][3] = __float_as_uint(f3);
            }
            
            float m_old = smem.m0[wg_tid];
            float m_new = fmaxf(m_old, m_block);
            float scale = (m_old == -INFINITY && m_new == -INFINITY) ? 1.0f : fast_exp2f_fn((m_old - m_new) * 1.44269504089f);
            
            int needs_rescale = __ballot_sync(0xFFFFFFFF, scale < 1.0f);
            if (needs_rescale != 0 && kv_idx > 0) {
                for (uint32_t col = 0; col < 128; col += 4) {
                    uint32_t r_O[4];
                    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(r_O[0]), "=r"(r_O[1]), "=r"(r_O[2]), "=r"(r_O[3])
                        : "r"(smem.tmem_O0 + col));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    
                    r_O[0] = __float_as_uint(__uint_as_float(r_O[0]) * scale);
                    r_O[1] = __float_as_uint(__uint_as_float(r_O[1]) * scale);
                    r_O[2] = __float_as_uint(__uint_as_float(r_O[2]) * scale);
                    r_O[3] = __float_as_uint(__uint_as_float(r_O[3]) * scale);
                    
                    asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};"
                        :: "r"(smem.tmem_O0 + col), "r"(r_O[0]), "r"(r_O[1]), "r"(r_O[2]), "r"(r_O[3]));
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            }
            smem.m0[wg_tid] = m_new;
            
            asm volatile("barrier.sync.aligned 2, 256;");
            
            float row_sum = 0.0f;
            for (uint32_t col = 0; col < 128; col += 4) {
                float f0 = __uint_as_float(r_S[col/4][0]);
                float f1 = __uint_as_float(r_S[col/4][1]);
                float f2 = __uint_as_float(r_S[col/4][2]);
                float f3 = __uint_as_float(r_S[col/4][3]);
                
                float p0 = (f0 == -INFINITY) ? 0.0f : fast_exp2f_fn((f0 - m_new) * 1.44269504089f);
                float p1 = (f1 == -INFINITY) ? 0.0f : fast_exp2f_fn((f1 - m_new) * 1.44269504089f);
                float p2 = (f2 == -INFINITY) ? 0.0f : fast_exp2f_fn((f2 - m_new) * 1.44269504089f);
                float p3 = (f3 == -INFINITY) ? 0.0f : fast_exp2f_fn((f3 - m_new) * 1.44269504089f);
                
                row_sum += p0 + p1 + p2 + p3;
                
                uint32_t p01 = pack_bf16_fn(__float_as_uint(p0), __float_as_uint(p1));
                uint32_t p23 = pack_bf16_fn(__float_as_uint(p2), __float_as_uint(p3));
                
                int byte_offset = wg_tid * 256 + col * 2;
                *reinterpret_cast<uint2*>((char*)smem.P0 + byte_offset) = make_uint2(p01, p23);
            }
            smem.l0[wg_tid] = smem.l0[wg_tid] * scale + row_sum;
            
            asm volatile("barrier.sync.aligned 3, 256;");
        }
        
        if (wg == 1) {
            mbarrier_wait_fn(&smem.mbar_umma1, umma1_phase);
            umma1_phase ^= 1;
            
            uint32_t r_S[32][4];
            for (uint32_t col = 0; col < 128; col += 4) {
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(r_S[col/4][0]),"=r"(r_S[col/4][1]),"=r"(r_S[col/4][2]),"=r"(r_S[col/4][3]) : "r"(smem.tmem_S1 + col));
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            float m_block = -INFINITY;
            int global_q = global_q1_base + wg_tid;
            for (uint32_t col = 0; col < 128; col += 4) {
                int global_k0 = kv_idx * 128 + col;
                
                float f0 = __uint_as_float(r_S[col/4][0]) * 0.08838834764f; 
                float f1 = __uint_as_float(r_S[col/4][1]) * 0.08838834764f;
                float f2 = __uint_as_float(r_S[col/4][2]) * 0.08838834764f;
                float f3 = __uint_as_float(r_S[col/4][3]) * 0.08838834764f;
                
                if (global_q >= S || global_k0 + 0 > global_q) f0 = -INFINITY;
                if (global_q >= S || global_k0 + 1 > global_q) f1 = -INFINITY;
                if (global_q >= S || global_k0 + 2 > global_q) f2 = -INFINITY;
                if (global_q >= S || global_k0 + 3 > global_q) f3 = -INFINITY;
                
                m_block = fmaxf(m_block, fmaxf(f0, fmaxf(f1, fmaxf(f2, f3))));
                
                r_S[col/4][0] = __float_as_uint(f0);
                r_S[col/4][1] = __float_as_uint(f1);
                r_S[col/4][2] = __float_as_uint(f2);
                r_S[col/4][3] = __float_as_uint(f3);
            }
            
            float m_old = smem.m1[wg_tid];
            float m_new = fmaxf(m_old, m_block);
            float scale = (m_old == -INFINITY && m_new == -INFINITY) ? 1.0f : fast_exp2f_fn((m_old - m_new) * 1.44269504089f);
            
            int needs_rescale = __ballot_sync(0xFFFFFFFF, scale < 1.0f);
            if (needs_rescale != 0 && kv_idx > 0) {
                for (uint32_t col = 0; col < 128; col += 4) {
                    uint32_t r_O[4];
                    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(r_O[0]), "=r"(r_O[1]), "=r"(r_O[2]), "=r"(r_O[3])
                        : "r"(smem.tmem_O1 + col));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    
                    r_O[0] = __float_as_uint(__uint_as_float(r_O[0]) * scale);
                    r_O[1] = __float_as_uint(__uint_as_float(r_O[1]) * scale);
                    r_O[2] = __float_as_uint(__uint_as_float(r_O[2]) * scale);
                    r_O[3] = __float_as_uint(__uint_as_float(r_O[3]) * scale);
                    
                    asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};"
                        :: "r"(smem.tmem_O1 + col), "r"(r_O[0]), "r"(r_O[1]), "r"(r_O[2]), "r"(r_O[3]));
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            }
            smem.m1[wg_tid] = m_new;
            
            asm volatile("barrier.sync.aligned 2, 256;");
            asm volatile("barrier.sync.aligned 3, 256;");
            
            float row_sum = 0.0f;
            for (uint32_t col = 0; col < 128; col += 4) {
                float f0 = __uint_as_float(r_S[col/4][0]);
                float f1 = __uint_as_float(r_S[col/4][1]);
                float f2 = __uint_as_float(r_S[col/4][2]);
                float f3 = __uint_as_float(r_S[col/4][3]);
                
                float p0 = (f0 == -INFINITY) ? 0.0f : fast_exp2f_fn((f0 - m_new) * 1.44269504089f);
                float p1 = (f1 == -INFINITY) ? 0.0f : fast_exp2f_fn((f1 - m_new) * 1.44269504089f);
                float p2 = (f2 == -INFINITY) ? 0.0f : fast_exp2f_fn((f2 - m_new) * 1.44269504089f);
                float p3 = (f3 == -INFINITY) ? 0.0f : fast_exp2f_fn((f3 - m_new) * 1.44269504089f);
                
                row_sum += p0 + p1 + p2 + p3;
                
                uint32_t p01 = pack_bf16_fn(__float_as_uint(p0), __float_as_uint(p1));
                uint32_t p23 = pack_bf16_fn(__float_as_uint(p2), __float_as_uint(p3));
                
                int byte_offset = wg_tid * 256 + col * 2;
                *reinterpret_cast<uint2*>((char*)smem.P1 + byte_offset) = make_uint2(p01, p23);
            }
            smem.l1[wg_tid] = smem.l1[wg_tid] * scale + row_sum;
        }
        
        tcgen05_fence_before_fn();
        asm volatile("barrier.sync.aligned 1, 384;");
        tcgen05_fence_after_fn();
        
        if (tid == 256) {
            uint32_t idesc_pv = make_instr_desc_fn(128, 128, 0, 1);
            
            for (int k = 0; k < 128; k += 16) {
                uint64_t desc_p0 = make_smem_desc_linear_K_major((char*)smem.P0 + k * 2, 256);
                uint64_t desc_v  = make_smem_desc_linear_MN_major((char*)smem.V + k * 256, 256);
                umma_f16_cg1_fn(smem.tmem_O0, desc_p0, desc_v, idesc_pv, 1);
            }
            for (int k = 0; k < 128; k += 16) {
                uint64_t desc_p1 = make_smem_desc_linear_K_major((char*)smem.P1 + k * 2, 256);
                uint64_t desc_v  = make_smem_desc_linear_MN_major((char*)smem.V + k * 256, 256);
                umma_f16_cg1_fn(smem.tmem_O1, desc_p1, desc_v, idesc_pv, 1);
            }
            umma_commit_cg1_fn(&smem.mbar_umma_pv);
        }
        
        if (wg == 0 || wg == 1) {
            pv_phase ^= 1;
        }
    }
    
    if (tid == 256) {
        mbarrier_wait_fn(&smem.mbar_umma_pv, pv_phase);
    }
    
    tcgen05_fence_before_fn();
    asm volatile("barrier.sync.aligned 1, 384;");
    tcgen05_fence_after_fn();
    
    if (wg == 0) {
        int global_q = global_q0_base + wg_tid;
        if (global_q < S) {
            float inv_l = 1.0f / smem.l0[wg_tid];
            for (uint32_t col = 0; col < 128; col += 8) {
                uint32_t r_O[8];
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                    : "=r"(r_O[0]),"=r"(r_O[1]),"=r"(r_O[2]),"=r"(r_O[3]),
                      "=r"(r_O[4]),"=r"(r_O[5]),"=r"(r_O[6]),"=r"(r_O[7]) : "r"(smem.tmem_O0 + col));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                
                uint32_t p0 = pack_bf16_fn(__float_as_uint(__uint_as_float(r_O[0]) * inv_l), __float_as_uint(__uint_as_float(r_O[1]) * inv_l));
                uint32_t p1 = pack_bf16_fn(__float_as_uint(__uint_as_float(r_O[2]) * inv_l), __float_as_uint(__uint_as_float(r_O[3]) * inv_l));
                uint32_t p2 = pack_bf16_fn(__float_as_uint(__uint_as_float(r_O[4]) * inv_l), __float_as_uint(__uint_as_float(r_O[5]) * inv_l));
                uint32_t p3 = pack_bf16_fn(__float_as_uint(__uint_as_float(r_O[6]) * inv_l), __float_as_uint(__uint_as_float(r_O[7]) * inv_l));
                
                int64_t out_idx = (int64_t(b) * H * S + int64_t(h) * S + global_q) * 128 + col;
                *reinterpret_cast<uint4*>(&O_ptr[out_idx]) = make_uint4(p0, p1, p2, p3);
            }
            int64_t lse_idx = int64_t(b) * H * S + int64_t(h) * S + global_q;
            LSE_ptr[lse_idx] = smem.m0[wg_tid] + logf(smem.l0[wg_tid]);
        }
    }
    else if (wg == 1) {
        int global_q = global_q1_base + wg_tid;
        if (global_q < S) {
            float inv_l = 1.0f / smem.l1[wg_tid];
            for (uint32_t col = 0; col < 128; col += 8) {
                uint32_t r_O[8];
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                    : "=r"(r_O[0]),"=r"(r_O[1]),"=r"(r_O[2]),"=r"(r_O[3]),
                      "=r"(r_O[4]),"=r"(r_O[5]),"=r"(r_O[6]),"=r"(r_O[7]) : "r"(smem.tmem_O1 + col));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                
                uint32_t p0 = pack_bf16_fn(__float_as_uint(__uint_as_float(r_O[0]) * inv_l), __float_as_uint(__uint_as_float(r_O[1]) * inv_l));
                uint32_t p1 = pack_bf16_fn(__float_as_uint(__uint_as_float(r_O[2]) * inv_l), __float_as_uint(__uint_as_float(r_O[3]) * inv_l));
                uint32_t p2 = pack_bf16_fn(__float_as_uint(__uint_as_float(r_O[4]) * inv_l), __float_as_uint(__uint_as_float(r_O[5]) * inv_l));
                uint32_t p3 = pack_bf16_fn(__float_as_uint(__uint_as_float(r_O[6]) * inv_l), __float_as_uint(__uint_as_float(r_O[7]) * inv_l));
                
                int64_t out_idx = (int64_t(b) * H * S + int64_t(h) * S + global_q) * 128 + col;
                *reinterpret_cast<uint4*>(&O_ptr[out_idx]) = make_uint4(p0, p1, p2, p3);
            }
            int64_t lse_idx = int64_t(b) * H * S + int64_t(h) * S + global_q;
            LSE_ptr[lse_idx] = smem.m1[wg_tid] + logf(smem.l1[wg_tid]);
        }
    }
    
    if (tid < 32) {
        tmem_dealloc_cg1_fn(smem.tmem_S0, 128);
        tmem_dealloc_cg1_fn(smem.tmem_S1, 128);
        tmem_dealloc_cg1_fn(smem.tmem_O0, 128);
        tmem_dealloc_cg1_fn(smem.tmem_O1, 128);
    }
}

CUresult create_tma_4d_descriptor_none(CUtensorMap* d, void* globalAddress, 
                                  uint64_t dim0, uint64_t dim1, uint64_t dim2, uint64_t dim3,
                                  uint32_t box0, uint32_t box1, uint32_t box2, uint32_t box3) {
    cuuint64_t globalDim[4] = {dim0, dim1, dim2, dim3};
    cuuint64_t globalStrides[3] = {dim0 * 2, dim1 * dim0 * 2, dim2 * dim1 * dim0 * 2};
    cuuint32_t boxDim[4] = {box0, box1, box2, box3};
    cuuint32_t elementStrides[4] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, globalAddress, globalDim, globalStrides,
        boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) 
{
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);

    if (S == 0) return;

    const __nv_bfloat16* q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* k_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* v_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* o_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* lse_ptr = static_cast<float*>(LSE.data_ptr());

    CUtensorMap tma_Q, tma_K, tma_V;
    CU_CHECK(create_tma_4d_descriptor_none(&tma_Q, (void*)q_ptr, 128, S, H, B, 128, 128, 1, 1));
    CU_CHECK(create_tma_4d_descriptor_none(&tma_K, (void*)k_ptr, 128, S, H, B, 128, 128, 1, 1));
    CU_CHECK(create_tma_4d_descriptor_none(&tma_V, (void*)v_ptr, 128, S, H, B, 128, 128, 1, 1));

    int64_t num_q_blocks = (S + 255) / 256;
    dim3 grid(num_q_blocks, B * H);
    dim3 block(384); 

    int smem_size = sizeof(SharedStorage);
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute(flash_attn_causal_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    flash_attn_causal_kernel<<<grid, block, smem_size, stream>>>(tma_Q, tma_K, tma_V, o_ptr, lse_ptr, B, H, S);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

}