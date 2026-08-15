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

#define CU_CHECK(call) do {                                        \
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        const char* err_str;                                       \
        cuGetErrorString(_e, &err_str);                            \
        fprintf(stderr, "CU error %s at %s:%d\n",                 \
                err_str, __FILE__, __LINE__);                      \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_example_cuda {

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

__device__ __forceinline__ void tma_load_3d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(bar)),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
}

__device__ __forceinline__ void tma_store_3d_fn(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group [%0, {%2, %3, %4}], [%1];"
        :: "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
}

__device__ __forceinline__ void tcgen05_fence_after_fn() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_wait_st_fn() {
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc_k_major(void* ptr) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    uint32_t lbo = 1;
    uint32_t sbo = 1024;
    uint64_t d = 0;
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)0 << 49;
    d |= (uint64_t)2 << 61; // SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_mn_major(void* ptr) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    uint32_t lbo = 16384; 
    uint32_t sbo = 1024;
    uint64_t d = 0;
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)0 << 49;
    d |= (uint64_t)2 << 61;
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc(uint32_t M, uint32_t N, int a_major, int b_major) {
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

__device__ __forceinline__ void umma_f16(uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc, int accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
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

__device__ __forceinline__ void store_p_smem_swizzled(__nv_bfloat16* p_smem, float* p_row, int offset) {
    int y = threadIdx.x;
    int y_mod_8 = y % 8;
    int4* base = (int4*)p_smem;
    for (int i = 0; i < 64; i += 8) {
        int x = i / 8;
        int swizzled_x = x ^ y_mod_8;
        
        uint32_t p0 = pack_bf16_fn(__float_as_uint(p_row[offset + i + 0]), __float_as_uint(p_row[offset + i + 1]));
        uint32_t p1 = pack_bf16_fn(__float_as_uint(p_row[offset + i + 2]), __float_as_uint(p_row[offset + i + 3]));
        uint32_t p2 = pack_bf16_fn(__float_as_uint(p_row[offset + i + 4]), __float_as_uint(p_row[offset + i + 5]));
        uint32_t p3 = pack_bf16_fn(__float_as_uint(p_row[offset + i + 6]), __float_as_uint(p_row[offset + i + 7]));
        
        base[y * 8 + swizzled_x] = make_int4(p0, p1, p2, p3);
    }
}


__global__ void mha_fwd_kernel(
    const __grid_constant__ CUtensorMap tma_q,
    const __grid_constant__ CUtensorMap tma_k,
    const __grid_constant__ CUtensorMap tma_v,
    const __grid_constant__ CUtensorMap tma_o,
    float* lse_out,
    int S
) {
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int q_step = blockIdx.x;
    int start_S = q_step * 128;
    int c2 = batch_idx * gridDim.y + head_idx;
    
    __shared__ __align__(1024) __nv_bfloat16 q_smem_0[128 * 64];
    __shared__ __align__(1024) __nv_bfloat16 q_smem_1[128 * 64];
    __shared__ __align__(1024) __nv_bfloat16 k_smem_0[128 * 64];
    __shared__ __align__(1024) __nv_bfloat16 k_smem_1[128 * 64];
    __shared__ __align__(1024) __nv_bfloat16 v_smem_0[128 * 64];
    __shared__ __align__(1024) __nv_bfloat16 v_smem_1[128 * 64];
    __shared__ __align__(1024) __nv_bfloat16 p_smem_0[128 * 64];
    __shared__ __align__(1024) __nv_bfloat16 p_smem_1[128 * 64];
    
    __shared__ uint64_t mbar_q;
    __shared__ uint64_t mbar_k;
    __shared__ uint64_t mbar_v;
    __shared__ uint64_t mbar_umma;
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&mbar_q, 1);
        init_smem_barrier_fn(&mbar_k, 1);
        init_smem_barrier_fn(&mbar_v, 1);
        init_smem_barrier_fn(&mbar_umma, 1);
    }
    __syncthreads();

    uint32_t tmem_base;
    if (threadIdx.x < 32) {
        uint32_t addr = (uint32_t)__cvta_generic_to_shared(&tmem_base);
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(addr), "r"(256));
    }
    __syncthreads();
    
    uint32_t tmem_S = tmem_base;
    uint32_t tmem_O_0 = tmem_base + 128;
    uint32_t tmem_O_1 = tmem_base + 128 + 64;

    for (int i = 0; i < 64; i += 4) {
        asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {%0,%1,%2,%3};"
            :: "r"(0), "r"(0), "r"(0), "r"(0), "r"(tmem_O_0 + i));
        asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {%0,%1,%2,%3};"
            :: "r"(0), "r"(0), "r"(0), "r"(0), "r"(tmem_O_1 + i));
    }
    tcgen05_wait_st_fn();

    int phase_q = 0, phase_k = 0, phase_v = 0, phase_umma = 0;

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(&mbar_q, 32768);
        tma_load_3d_fn(&tma_q, &mbar_q, q_smem_0, 0, start_S, c2);
        tma_load_3d_fn(&tma_q, &mbar_q, q_smem_1, 64, start_S, c2);
    }
    mbarrier_wait_fn(&mbar_q, phase_q);
    phase_q ^= 1;
    
    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    
    int num_steps = (S + 127) / 128;
    
    for (int step = 0; step < num_steps; ++step) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(&mbar_k, 32768);
            tma_load_3d_fn(&tma_k, &mbar_k, k_smem_0, 0, step * 128, c2);
            tma_load_3d_fn(&tma_k, &mbar_k, k_smem_1, 64, step * 128, c2);
            
            mbarrier_arrive_and_expect_tx_fn(&mbar_v, 32768);
            tma_load_3d_fn(&tma_v, &mbar_v, v_smem_0, 0, step * 128, c2);
            tma_load_3d_fn(&tma_v, &mbar_v, v_smem_1, 64, step * 128, c2);
        }
        
        mbarrier_wait_fn(&mbar_k, phase_k);
        mbarrier_wait_fn(&mbar_v, phase_v);
        
        if (threadIdx.x == 0) {
            for (int k_idx = 0; k_idx < 4; ++k_idx) {
                uint64_t q_desc = make_smem_desc_k_major(q_smem_0 + k_idx * 16);
                uint64_t k_desc = make_smem_desc_k_major(k_smem_0 + k_idx * 16);
                uint32_t idesc = make_instr_desc(128, 128, 0, 0);
                int accum = (k_idx == 0) ? 0 : 1;
                umma_f16(tmem_S, q_desc, k_desc, idesc, accum);
            }
            for (int k_idx = 0; k_idx < 4; ++k_idx) {
                uint64_t q_desc = make_smem_desc_k_major(q_smem_1 + k_idx * 16);
                uint64_t k_desc = make_smem_desc_k_major(k_smem_1 + k_idx * 16);
                uint32_t idesc = make_instr_desc(128, 128, 0, 0);
                umma_f16(tmem_S, q_desc, k_desc, idesc, 1);
            }
            tcgen05_fence_after_fn();
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"((uint32_t)__cvta_generic_to_shared(&mbar_umma)));
        }
        mbarrier_wait_fn(&mbar_umma, phase_umma);
        phase_umma ^= 1;
        
        float s_row[128];
        for (int i = 0; i < 128; i += 8) {
            uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) : "r"(tmem_S + i));
            s_row[i+0] = __uint_as_float(r0); s_row[i+1] = __uint_as_float(r1);
            s_row[i+2] = __uint_as_float(r2); s_row[i+3] = __uint_as_float(r3);
            s_row[i+4] = __uint_as_float(r4); s_row[i+5] = __uint_as_float(r5);
            s_row[i+6] = __uint_as_float(r6); s_row[i+7] = __uint_as_float(r7);
        }
        tmem_load_fence_fn();
        
        float m_curr = -INFINITY;
        for (int i = 0; i < 128; ++i) {
            int global_k = step * 128 + i;
            if (global_k >= S) {
                s_row[i] = -INFINITY;
            } else {
                s_row[i] *= 0.0883883476f;
                m_curr = fmaxf(m_curr, s_row[i]);
            }
        }
        
        float m_new = fmaxf(m_prev, m_curr);
        float exp_diff = fast_exp2f_fn((m_prev - m_new) * 1.44269504f);
        
        float l_curr = 0.0f;
        for (int i = 0; i < 128; ++i) {
            float p = fast_exp2f_fn((s_row[i] - m_new) * 1.44269504f);
            s_row[i] = p;
            l_curr += p;
        }
        float l_new = exp_diff * l_prev + l_curr;
        
        store_p_smem_swizzled(p_smem_0, s_row, 0);
        store_p_smem_swizzled(p_smem_1, s_row, 64);
        
        float o_row_0[64];
        for (int i = 0; i < 64; i += 8) {
            uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) : "r"(tmem_O_0 + i));
            o_row_0[i+0] = __uint_as_float(r0); o_row_0[i+1] = __uint_as_float(r1);
            o_row_0[i+2] = __uint_as_float(r2); o_row_0[i+3] = __uint_as_float(r3);
            o_row_0[i+4] = __uint_as_float(r4); o_row_0[i+5] = __uint_as_float(r5);
            o_row_0[i+6] = __uint_as_float(r6); o_row_0[i+7] = __uint_as_float(r7);
        }
        float o_row_1[64];
        for (int i = 0; i < 64; i += 8) {
            uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) : "r"(tmem_O_1 + i));
            o_row_1[i+0] = __uint_as_float(r0); o_row_1[i+1] = __uint_as_float(r1);
            o_row_1[i+2] = __uint_as_float(r2); o_row_1[i+3] = __uint_as_float(r3);
            o_row_1[i+4] = __uint_as_float(r4); o_row_1[i+5] = __uint_as_float(r5);
            o_row_1[i+6] = __uint_as_float(r6); o_row_1[i+7] = __uint_as_float(r7);
        }
        tmem_load_fence_fn();
        
        for (int i = 0; i < 64; ++i) {
            o_row_0[i] *= exp_diff;
            o_row_1[i] *= exp_diff;
        }
        
        for (int i = 0; i < 64; i += 4) {
            asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {%0,%1,%2,%3};"
                :: "r"(__float_as_uint(o_row_0[i+0])), "r"(__float_as_uint(o_row_0[i+1])),
                   "r"(__float_as_uint(o_row_0[i+2])), "r"(__float_as_uint(o_row_0[i+3])), "r"(tmem_O_0 + i));
            asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {%0,%1,%2,%3};"
                :: "r"(__float_as_uint(o_row_1[i+0])), "r"(__float_as_uint(o_row_1[i+1])),
                   "r"(__float_as_uint(o_row_1[i+2])), "r"(__float_as_uint(o_row_1[i+3])), "r"(tmem_O_1 + i));
        }
        tcgen05_wait_st_fn();
        
        __syncthreads();
        
        if (threadIdx.x == 0) {
            for (int k_idx = 0; k_idx < 8; ++k_idx) {
                __nv_bfloat16* p_ptr = (k_idx < 4) ? p_smem_0 : p_smem_1;
                uint64_t p_desc = make_smem_desc_k_major(p_ptr + (k_idx % 4) * 16);
                uint64_t v_desc = make_smem_desc_mn_major(v_smem_0 + k_idx * 16 * 64);
                uint32_t idesc = make_instr_desc(128, 64, 0, 1);
                int accum = 1;
                umma_f16(tmem_O_0, p_desc, v_desc, idesc, accum);
            }
            
            for (int k_idx = 0; k_idx < 8; ++k_idx) {
                __nv_bfloat16* p_ptr = (k_idx < 4) ? p_smem_0 : p_smem_1;
                uint64_t p_desc = make_smem_desc_k_major(p_ptr + (k_idx % 4) * 16);
                uint64_t v_desc = make_smem_desc_mn_major(v_smem_1 + k_idx * 16 * 64);
                uint32_t idesc = make_instr_desc(128, 64, 0, 1);
                int accum = 1;
                umma_f16(tmem_O_1, p_desc, v_desc, idesc, accum);
            }
            
            tcgen05_fence_after_fn();
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"((uint32_t)__cvta_generic_to_shared(&mbar_umma)));
        }
        mbarrier_wait_fn(&mbar_umma, phase_umma);
        phase_umma ^= 1;
        
        m_prev = m_new;
        l_prev = l_new;
        
        phase_k ^= 1;
        phase_v ^= 1;
        __syncthreads();
    }
    
    float o_row_0[64];
    for (int i = 0; i < 64; i += 8) {
        uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) : "r"(tmem_O_0 + i));
        o_row_0[i+0] = __uint_as_float(r0); o_row_0[i+1] = __uint_as_float(r1);
        o_row_0[i+2] = __uint_as_float(r2); o_row_0[i+3] = __uint_as_float(r3);
        o_row_0[i+4] = __uint_as_float(r4); o_row_0[i+5] = __uint_as_float(r5);
        o_row_0[i+6] = __uint_as_float(r6); o_row_0[i+7] = __uint_as_float(r7);
    }
    float o_row_1[64];
    for (int i = 0; i < 64; i += 8) {
        uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) : "r"(tmem_O_1 + i));
        o_row_1[i+0] = __uint_as_float(r0); o_row_1[i+1] = __uint_as_float(r1);
        o_row_1[i+2] = __uint_as_float(r2); o_row_1[i+3] = __uint_as_float(r3);
        o_row_1[i+4] = __uint_as_float(r4); o_row_1[i+5] = __uint_as_float(r5);
        o_row_1[i+6] = __uint_as_float(r6); o_row_1[i+7] = __uint_as_float(r7);
    }
    tmem_load_fence_fn();
    
    for (int i = 0; i < 64; ++i) {
        o_row_0[i] /= l_prev;
        o_row_1[i] /= l_prev;
    }
    
    store_p_smem_swizzled(q_smem_0, o_row_0, 0);
    store_p_smem_swizzled(q_smem_1, o_row_1, 0);
    
    __syncthreads();
    
    if (threadIdx.x == 0) {
        tma_store_3d_fn(&tma_o, q_smem_0, 0, start_S, c2);
        tma_store_3d_fn(&tma_o, q_smem_1, 64, start_S, c2);
        asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
        asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory");
    }
    
    int seq_idx = start_S + threadIdx.x;
    if (seq_idx < S) {
        lse_out[batch_idx * gridDim.y * S + head_idx * S + seq_idx] = m_prev + logf(l_prev);
    }
    
    __syncthreads();
    if (threadIdx.x < 32) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_base), "r"(256));
    }
}

CUresult create_tma_3d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t dim0, uint64_t dim1, uint64_t dim2, uint32_t box0, uint32_t box1) {
    cuuint64_t globalDim[3] = {dim0, dim1, dim2};
    cuuint64_t globalStrides[2] = {dim0 * 2, dim0 * dim1 * 2};
    cuuint32_t boxDim[3] = {box0, box1, 1};
    cuuint32_t elementStrides[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, globalAddress, globalDim, globalStrides, boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3); // 128
    
    void* q_ptr = Q.data_ptr();
    void* k_ptr = K.data_ptr();
    void* v_ptr = V.data_ptr();
    void* o_ptr = O.data_ptr();
    float* lse_ptr = static_cast<float*>(LSE.data_ptr());
    
    CUtensorMap tma_q, tma_k, tma_v, tma_o;
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_q, q_ptr, 128, S, B * H, 64, 128));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_k, k_ptr, 128, S, B * H, 64, 128));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_v, v_ptr, 128, S, B * H, 64, 128));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_o, o_ptr, 128, S, B * H, 64, 128));
    
    int64_t threads = 128;
    int64_t blocks_x = (S + 127) / 128;
    dim3 blocks(blocks_x, H, B);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    cudaFuncSetAttribute((const void*)mha_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 227 * 1024);
    mha_fwd_kernel<<<blocks, threads, 0, stream>>>(tma_q, tma_k, tma_v, tma_o, lse_ptr, S);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}