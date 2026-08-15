#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <algorithm>
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
        fprintf(stderr, "CUDA Driver error %d at %s:%d\n",         \
                _e, __FILE__, __LINE__);                           \
        exit(1);                                                   \
    }                                                              \
} while(0)

struct StaticSharedStorage {
    __align__(1024) __nv_bfloat16 sm_Q[16384];
    __align__(1024) __nv_bfloat16 sm_K0[16384];
    __align__(1024) __nv_bfloat16 sm_K1[16384];
    __align__(1024) __nv_bfloat16 sm_V0_0[8192];
    __align__(1024) __nv_bfloat16 sm_V0_1[8192];
    __align__(1024) __nv_bfloat16 sm_V1_0[8192];
    __align__(1024) __nv_bfloat16 sm_V1_1[8192];
    
    uint64_t bar_Q;
    uint64_t bar_K0;
    uint64_t bar_K1;
    uint64_t bar_V0;
    uint64_t bar_V1;
    __nv_bfloat16* smem_P;
};

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

__device__ __forceinline__ void tma_load_3d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
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
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
        :: "r"(a) : "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc_swizzle(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_QK_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= (1u << 15);   // a_major = 1 (M-major)
    d |= (1u << 16);   // b_major = 1 (N-major)
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_PV_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= (1u << 15);   // a_major = 1 (M-major)
    d |= (1u << 16);   // b_major = 1 (N-major)
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tmem_store_fence_fn() {
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tmem_ld_32x32b_x4(uint32_t tmem_addr, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*r0), "=r"(*r1), "=r"(*r2), "=r"(*r3) : "r"(tmem_addr) : "memory");
}

__device__ __forceinline__ void tmem_st_32x32b_x4(uint32_t tmem_addr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};"
        :: "r"(tmem_addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
}

__device__ __forceinline__ uint32_t pack_bf16_fn(float a, float b) {
    __nv_bfloat16 ba = __float2bfloat16(a);
    __nv_bfloat16 bb = __float2bfloat16(b);
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};"
        : "=r"(result)
        : "h"(*reinterpret_cast<uint16_t*>(&ba)),
          "h"(*reinterpret_cast<uint16_t*>(&bb)));
    return result;
}

__global__ void fa_kernel(
    __grid_constant__ const CUtensorMap tma_Q,
    __grid_constant__ const CUtensorMap tma_K,
    __grid_constant__ const CUtensorMap tma_V,
    __nv_bfloat16* O_gmem,
    float* LSE_gmem,
    uint32_t S_seq,
    float scale)
{
    uint32_t m_block = blockIdx.x;
    uint32_t head_idx = blockIdx.y;
    uint32_t m_row = m_block * 128;

    extern __shared__ char smem_raw[];
    char* smem = (char*)(((uintptr_t)smem_raw + 1023) & ~1023);
    struct StaticSharedStorage* s_mem = reinterpret_cast<struct StaticSharedStorage*>(smem);
    
    uintptr_t p_addr = (uintptr_t)&s_mem->bar_V1 + 8;
    p_addr = (p_addr + 127) & ~127;
    s_mem->smem_P = (__nv_bfloat16*)p_addr;
    
    uint64_t* bar_Q = &s_mem->bar_Q;
    uint64_t* bar_K0 = &s_mem->bar_K0;
    uint64_t* bar_K1 = &s_mem->bar_K1;
    uint64_t* bar_V0 = &s_mem->bar_V0;
    uint64_t* bar_V1 = &s_mem->bar_V1;
    uint64_t* bar_cp = &s_mem->bar_V1 + 1;
    
    uint64_t* bar_K[2] = {bar_K0, bar_K1};
    uint64_t* bar_V[2] = {bar_V0, bar_V1};

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(bar_Q, 1);
        init_smem_barrier_fn(bar_K0, 1);
        init_smem_barrier_fn(bar_K1, 1);
        init_smem_barrier_fn(bar_V0, 1);
        init_smem_barrier_fn(bar_V1, 1);
        init_smem_barrier_fn(bar_cp, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    uint32_t* tmem_addrs = (uint32_t*)(bar_cp + 1);
    if (threadIdx.x == 0) {
        tmem_alloc_fn(&tmem_addrs[0], 256); 
    }
    __syncthreads();

    uint32_t tmem_S = tmem_addrs[0];
    uint32_t tmem_O = tmem_S + 128;
    
    if (threadIdx.x < 128) {
        for (uint32_t c = 0; c < 128; c += 4) {
            tmem_st_32x32b_x4(tmem_O + c, 0, 0, 0, 0);
        }
    }
    tmem_store_fence_fn();

    __nv_bfloat16* sm_K[2] = {s_mem->sm_K0, s_mem->sm_K1};
    __nv_bfloat16* sm_V[2][2] = {
        {s_mem->sm_V0_0, s_mem->sm_V0_1},
        {s_mem->sm_V1_0, s_mem->sm_V1_1}
    };

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar_Q, 32768);
        tma_load_3d_fn(&tma_Q, bar_Q, s_mem->sm_Q, 0, m_row, head_idx);
        tma_load_3d_fn(&tma_Q, bar_Q, s_mem->sm_Q + 8192, 64, m_row, head_idx);
    }
    mbarrier_wait_fn(bar_Q, 0);

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar_K[0], 32768);
        tma_load_3d_fn(&tma_K, bar_K[0], sm_K[0], 0, 0, head_idx);
        tma_load_3d_fn(&tma_K, bar_K[0], sm_K[0] + 8192, 64, 0, head_idx);
        
        mbarrier_arrive_and_expect_tx_fn(bar_V[0], 65536);
        tma_load_3d_fn(&tma_V, bar_V[0], sm_V[0][0], 0, 0, head_idx);
        tma_load_3d_fn(&tma_V, bar_V[0], sm_V[0][0] + 4096, 64, 0, head_idx);
        tma_load_3d_fn(&tma_V, bar_V[0], sm_V[0][1], 0, 0, head_idx);
        tma_load_3d_fn(&tma_V, bar_V[0], sm_V[0][1] + 4096, 64, 0, head_idx);
    }
    
    uint64_t q_desc = make_smem_desc_swizzle(s_mem->sm_Q, 1, 1024);
    uint64_t* k_desc_buf = (uint64_t*)(sm_V[1][1] + 8192);
    k_desc_buf[0] = make_smem_desc_swizzle(sm_K[0], 1, 1024);
    k_desc_buf[1] = make_smem_desc_swizzle(sm_K[1], 1, 1024);
    
    uint64_t* v0_desc_buf = (uint64_t*)((char*)k_desc_buf + 16);
    v0_desc_buf[0] = make_smem_desc_swizzle(sm_V[0][0], 1024, 1024);
    v0_desc_buf[1] = make_smem_desc_swizzle(sm_V[1][0], 1024, 1024);
    
    uint64_t* v1_desc_buf = (uint64_t*)((char*)v0_desc_buf + 16);
    v1_desc_buf[0] = make_smem_desc_swizzle(sm_V[0][1], 1024, 1024);
    v1_desc_buf[1] = make_smem_desc_swizzle(sm_V[1][1], 1024, 1024);

    uint64_t p_desc = make_smem_desc_swizzle(s_mem->smem_P, 1, 1024);

    uint32_t idesc_QK = make_instr_desc_QK_fn(128, 128);
    uint32_t idesc_PV = make_instr_desc_PV_fn(128, 64);

    float running_max = -INFINITY;
    float running_sum = 0.0f;

    uint32_t phase_K[2] = {0, 0};
    uint32_t phase_V[2] = {0, 0};
    uint32_t phase_cp = 0;
    
    uint32_t num_iters = (S_seq + 127) / 128;

    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        uint32_t n_block = iter * 128;
        uint32_t buf_idx = iter % 2;
        uint32_t next_buf_idx = (iter + 1) % 2;
        
        if (iter + 1 < num_iters) {
            uint32_t next_n_block = (iter + 1) * 128;
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(bar_K[next_buf_idx], 32768);
                tma_load_3d_fn(&tma_K, bar_K[next_buf_idx], sm_K[next_buf_idx], 0, next_n_block, head_idx);
                tma_load_3d_fn(&tma_K, bar_K[next_buf_idx], sm_K[next_buf_idx] + 8192, 64, next_n_block, head_idx);
                
                mbarrier_arrive_and_expect_tx_fn(bar_V[next_buf_idx], 65536);
                tma_load_3d_fn(&tma_V, bar_V[next_buf_idx], sm_V[next_buf_idx][0], 0, next_n_block, head_idx);
                tma_load_3d_fn(&tma_V, bar_V[next_buf_idx], sm_V[next_buf_idx][0] + 4096, 64, next_n_block, head_idx);
                tma_load_3d_fn(&tma_V, bar_V[next_buf_idx], sm_V[next_buf_idx][1], 0, next_n_block, head_idx);
                tma_load_3d_fn(&tma_V, bar_V[next_buf_idx], sm_V[next_buf_idx][1] + 4096, 64, next_n_block, head_idx);
            }
        }
        
        mbarrier_wait_fn(bar_K[buf_idx], phase_K[buf_idx]); phase_K[buf_idx] ^= 1;
        mbarrier_wait_fn(bar_V[buf_idx], phase_V[buf_idx]); phase_V[buf_idx] ^= 1;
        fence_proxy_async_fn();

        if (threadIdx.x < 128) {
            for (uint32_t c = 0; c < 128; c += 4) {
                tmem_st_32x32b_x4(tmem_S + c, 0, 0, 0, 0);
            }
            tmem_store_fence_fn();
        }

        if (threadIdx.x == 0) {
            for (int k_step = 0; k_step < 8; ++k_step) {
                uint32_t accum = (k_step == 0) ? 0 : 1; 
                umma_f16_cg1_fn(tmem_S + (k_step << 1), q_desc + (k_step << 1), k_desc_buf[buf_idx] + (k_step << 1), idesc_QK, accum);
            }
            umma_commit_1sm_fn(bar_cp);
        }
        mbarrier_wait_fn(bar_cp, phase_cp); phase_cp ^= 1;

        float local_max = -INFINITY;
        for (uint32_t n_step = 0; n_step < 4; ++n_step) {
            uint32_t r0, r1, r2, r3;
            tmem_ld_32x32b_x4(tmem_S + (n_step * 32), &r0, &r1, &r2, &r3);
            float f0 = __uint_as_float(r0) * scale;
            float f1 = __uint_as_float(r1) * scale;
            float f2 = __uint_as_float(r2) * scale;
            float f3 = __uint_as_float(r3) * scale;
            
            uint32_t col_base = n_step * 32;
            if (iter * 128 + col_base >= S_seq) f0 = -INFINITY;
            if (iter * 128 + col_base + 1 >= S_seq) f1 = -INFINITY;
            if (iter * 128 + col_base + 2 >= S_seq) f2 = -INFINITY;
            if (iter * 128 + col_base + 3 >= S_seq) f3 = -INFINITY;
            
            local_max = fmaxf(local_max, fmaxf(fmaxf(f0, f1), fmaxf(f2, f3)));
        }
        tmem_load_fence_fn();

        float old_max = running_max;
        float new_max = fmaxf(old_max, local_max);
        
        bool should_rescale = (new_max - old_max) > 0.5f;
        
        if (should_rescale) {
            float factor = __expf(old_max - new_max);
            for (uint32_t n_step = 0; n_step < 4; ++n_step) {
                uint32_t o0, o1, o2, o3;
                tmem_ld_32x32b_x4(tmem_O + (n_step * 32), &o0, &o1, &o2, &o3);
                o0 = __float_as_uint(__uint_as_float(o0) * factor);
                o1 = __float_as_uint(__uint_as_float(o1) * factor);
                o2 = __float_as_uint(__uint_as_float(o2) * factor);
                o3 = __float_as_uint(__uint_as_float(o3) * factor);
                tmem_st_32x32b_x4(tmem_O + (n_step * 32), o0, o1, o2, o3);
            }
            tmem_store_fence_fn();
            running_sum *= factor;
        }
        running_max = should_rescale ? new_max : old_max;

        float local_sum = 0;
        for (uint32_t n_step = 0; n_step < 4; ++n_step) {
            uint32_t r0, r1, r2, r3;
            tmem_ld_32x32b_x4(tmem_S + (n_step * 32), &r0, &r1, &r2, &r3);
            
            float f0 = __uint_as_float(r0) * scale;
            float f1 = __uint_as_float(r1) * scale;
            float f2 = __uint_as_float(r2) * scale;
            float f3 = __uint_as_float(r3) * scale;
            
            float p0 = (iter * 128 + n_step * 32 >= S_seq) ? 0 : __expf(f0 - running_max);
            float p1 = (iter * 128 + n_step * 32 + 1 >= S_seq) ? 0 : __expf(f1 - running_max);
            float p2 = (iter * 128 + n_step * 32 + 2 >= S_seq) ? 0 : __expf(f2 - running_max);
            float p3 = (iter * 128 + n_step * 32 + 3 >= S_seq) ? 0 : __expf(f3 - running_max);
            
            local_sum += p0 + p1 + p2 + p3;
            
            if (threadIdx.x < 128) {
                int row = threadIdx.x;
                int col = n_step * 32;
                int x_chunk = col / 8;
                int x_rem = col % 8;
                int swizzled_col = (x_chunk ^ (row % 8)) * 8 + x_rem;
                
                uint32_t packed0 = pack_bf16_fn(p0, p1);
                uint32_t packed1 = pack_bf16_fn(p2, p3);
                
                *(uint32_t*)(&s_mem->smem_P[row * 128 + swizzled_col]) = packed0;
                *(uint32_t*)(&s_mem->smem_P[row * 128 + swizzled_col + 2]) = packed1;
            }
        }
        tmem_load_fence_fn();
        running_sum += local_sum;
        
        __syncthreads(); 
        fence_proxy_async_fn(); 
        
        if (threadIdx.x == 0) {
            for (int k_step = 0; k_step < 8; ++k_step) {
                uint32_t pv_accum = (iter == 0 && k_step == 0) ? 0 : 1;
                umma_f16_cg1_fn(tmem_O + (k_step << 1), q_desc + (k_step << 1), k_desc_buf[buf_idx] + (k_step << 1), idesc_QK, pv_accum);
            }
            umma_commit_1sm_fn(bar_cp);
        }
        mbarrier_wait_fn(bar_cp, phase_cp); phase_cp ^= 1;
        
        if (threadIdx.x == 0) {
            for (int k_step = 0; k_step < 8; ++k_step) {
                uint32_t pv_accum = (iter == 0 && k_step == 0) ? 0 : 1;
                umma_f16_cg1_fn(tmem_O + 64 + (k_step << 1), q_desc + (k_step << 1), k_desc_buf[buf_idx] + (k_step << 1), idesc_QK, pv_accum);
            }
            umma_commit_1sm_fn(bar_cp);
        }
        mbarrier_wait_fn(bar_cp, phase_cp); phase_cp ^= 1;
    }

    for (uint32_t n_step = 0; n_step < 4; ++n_step) {
        uint32_t o0, o1, o2, o3;
        tmem_ld_32x32b_x4(tmem_O + (n_step * 32), &o0, &o1, &o2, &o3);
        tmem_load_fence_fn();
        
        if (running_sum > 0) {
            o0 = __float_as_uint(__uint_as_float(o0) / running_sum);
            o1 = __float_as_uint(__uint_as_float(o1) / running_sum);
            o2 = __float_as_uint(__uint_as_float(o2) / running_sum);
            o3 = __float_as_uint(__uint_as_float(o3) / running_sum);
        }
        
        uint32_t g_row = m_row + threadIdx.x;
        if (g_row < S_seq) {
            __nv_bfloat16* out = O_gmem + (head_idx * S_seq + g_row) * 128 + (n_step * 32);
            *reinterpret_cast<uint32_t*>(out) = pack_bf16_fn(__uint_as_float(o0), __uint_as_float(o1));
            *reinterpret_cast<uint32_t*>(out + 2) = pack_bf16_fn(__uint_as_float(o2), __uint_as_float(o3));
        }
    }

    uint32_t g_row_lse = m_row + threadIdx.x;
    if (threadIdx.x < 128 && g_row_lse < S_seq) {
        LSE_gmem[head_idx * S_seq + g_row_lse] = running_max + __logf(running_sum);
    }

    if (threadIdx.x == 0) {
        tmem_dealloc_fn(tmem_S, 256);
    }
}

CUresult create_tma_3d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t dim0, uint64_t dim1, uint64_t dim2, uint32_t box0, uint32_t box1, uint32_t box2, CUtensorMapDataType dataType, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[3] = {dim0, dim1, dim2};
    cuuint64_t globalStrides[2] = {dim0 * 2, dim0 * dim1 * 2};
    cuuint32_t boxDim[3] = {box0, box1, box2};
    cuuint32_t elementStrides[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, dataType, 3, globalAddress, globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, l2Promotion, oobFill
    );
}

namespace tvm_ffi_fa {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);

    uint32_t S_seq = static_cast<uint32_t>(S);
    float scale = 1.f / sqrtf(static_cast<float>(D));

    uint32_t smem_size = sizeof(StaticSharedStorage) + 16384 * sizeof(__nv_bfloat16) + 1024;
    CUDA_CHECK(cudaFuncSetAttribute(fa_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    CUtensorMap tma_Q, tma_K, tma_V;

    CU_CHECK(create_tma_3d_descriptor_2B(&tma_Q, Q.data_ptr(), D, S, B * H, 64, 128, 1, 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_K, K.data_ptr(), D, S, B * H, 64, 128, 1, 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_V, V.data_ptr(), D, S, B * H, 64, 128, 1, 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    uint32_t num_blocks_x = (S_seq + 127) / 128;
    if (num_blocks_x > 1024) num_blocks_x = 1024;
    dim3 grid(num_blocks_x, B * H);
    dim3 block(256);

    __nv_bfloat16* O_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_ptr = static_cast<float*>(LSE.data_ptr());

    fa_kernel<<<grid, block, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, O_ptr, LSE_ptr, S_seq, scale
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_fa