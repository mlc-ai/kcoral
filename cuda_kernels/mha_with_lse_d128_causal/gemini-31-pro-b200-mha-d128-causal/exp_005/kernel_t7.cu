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
        fprintf(stderr, "CUDA error %s at %s:%d\n",                \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

#define CU_CHECK(call) do {                                        \
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        const char* errStr;                                        \
        cuGetErrorString(_e, &errStr);                             \
        fprintf(stderr, "CU error %s at %s:%d\n", errStr, __FILE__, __LINE__); \
        exit(1);                                                   \
    }                                                              \
} while(0)

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

__device__ __forceinline__ void tma_load_4d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc_swizzle_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    
    uint32_t base_offset = (addr >> 7) & 0x7;
    d |= (uint64_t)base_offset << 49;
    
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

__global__ void flash_fwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int S, int D)
{
    int m_start = blockIdx.x * 128;
    int head_idx = blockIdx.y;
    int batch_idx = blockIdx.z;
    size_t H_size = gridDim.y;

    extern __shared__ __align__(1024) uint8_t smem_pool[];
    
    __nv_bfloat16* smem_Q_0 = (__nv_bfloat16*)(smem_pool + 0); 
    __nv_bfloat16* smem_Q_1 = (__nv_bfloat16*)(smem_pool + 16384); 
    
    __nv_bfloat16* smem_K_0[2] = {(__nv_bfloat16*)(smem_pool + 32768), (__nv_bfloat16*)(smem_pool + 49152)}; 
    __nv_bfloat16* smem_K_1[2] = {(__nv_bfloat16*)(smem_pool + 65536), (__nv_bfloat16*)(smem_pool + 81920)}; 
    
    __nv_bfloat16* smem_V_0[2] = {(__nv_bfloat16*)(smem_pool + 98304), (__nv_bfloat16*)(smem_pool + 114688)}; 
    __nv_bfloat16* smem_V_1[2] = {(__nv_bfloat16*)(smem_pool + 131072), (__nv_bfloat16*)(smem_pool + 147456)}; 
    
    __nv_bfloat16* smem_P_0 = (__nv_bfloat16*)(smem_pool + 163840); 
    __nv_bfloat16* smem_P_1 = (__nv_bfloat16*)(smem_pool + 180224); 
    
    uint64_t* mbar_Q = (uint64_t*)(smem_pool + 196608); 
    uint64_t* mbar_K = (uint64_t*)(smem_pool + 196616);  
    uint64_t* mbar_V = (uint64_t*)(smem_pool + 196632);  
    uint64_t* mbar_mma = (uint64_t*)(smem_pool + 196648); 
    uint32_t* tmem_alloc_dst = (uint32_t*)(smem_pool + 196656); 
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(&mbar_K[0], 1);
        init_smem_barrier_fn(&mbar_K[1], 1);
        init_smem_barrier_fn(&mbar_V[0], 1);
        init_smem_barrier_fn(&mbar_V[1], 1);
        init_smem_barrier_fn(mbar_mma, 1);
    }
    __syncthreads();
    fence_smem_barrier_init_fn();
    
    if (threadIdx.x < 32) {
        // Allocate 512 columns max-per-CTA to allow double-buffering of S-TMEM 
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" :: "r"((uint32_t)__cvta_generic_to_shared(tmem_alloc_dst)));
    }
    __syncthreads();
    
    uint32_t tmem_base_addr = *tmem_alloc_dst;
    int O_tmem = tmem_base_addr;
    int S_tmem[2] = {(int)tmem_base_addr + 128, (int)tmem_base_addr + 256};

    for (int c = 0; c < 128; c += 8) {
        uint32_t z = 0;
        asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
                     :: "r"(O_tmem + c), "r"(z),"r"(z),"r"(z),"r"(z),"r"(z),"r"(z),"r"(z),"r"(z));
    }
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_Q, 32768);
        tma_load_4d_fn(&tma_Q, mbar_Q, smem_Q_0, 0, m_start, head_idx, batch_idx);
        tma_load_4d_fn(&tma_Q, mbar_Q, smem_Q_1, 64, m_start, head_idx, batch_idx);

        mbarrier_arrive_and_expect_tx_fn(&mbar_K[0], 32768);
        tma_load_4d_fn(&tma_K, &mbar_K[0], smem_K_0[0], 0, 0, head_idx, batch_idx);
        tma_load_4d_fn(&tma_K, &mbar_K[0], smem_K_1[0], 64, 0, head_idx, batch_idx);

        mbarrier_arrive_and_expect_tx_fn(&mbar_V[0], 32768);
        tma_load_4d_fn(&tma_V, &mbar_V[0], smem_V_0[0], 0, 0, head_idx, batch_idx);
        tma_load_4d_fn(&tma_V, &mbar_V[0], smem_V_1[0], 64, 0, head_idx, batch_idx);
        
        if (128 <= m_start) {
            mbarrier_arrive_and_expect_tx_fn(&mbar_K[1], 32768);
            tma_load_4d_fn(&tma_K, &mbar_K[1], smem_K_0[1], 0, 128, head_idx, batch_idx);
            tma_load_4d_fn(&tma_K, &mbar_K[1], smem_K_1[1], 64, 128, head_idx, batch_idx);

            mbarrier_arrive_and_expect_tx_fn(&mbar_V[1], 32768);
            tma_load_4d_fn(&tma_V, &mbar_V[1], smem_V_0[1], 0, 128, head_idx, batch_idx);
            tma_load_4d_fn(&tma_V, &mbar_V[1], smem_V_1[1], 64, 128, head_idx, batch_idx);
        }
    }
    
    mbarrier_wait_fn(mbar_Q, 0);
    mbarrier_wait_fn(&mbar_K[0], 0);

    tcgen05_fence_after_fn();
    uint32_t idesc = make_instr_desc_fn(128, 128);
    for (int k = 0; k < 64; k += 16) {
        uint64_t desc_Q = make_smem_desc_swizzle_fn(smem_Q_0 + k, 1, 1024);
        uint64_t desc_K = make_smem_desc_swizzle_fn(smem_K_0[0] + k, 1, 1024);
        int accumulate = (k == 0) ? 0 : 1;
        if (threadIdx.x == 0) {
            asm volatile(
                "{\n.reg .pred p;\n"
                "setp.ne.b32 p, %4, 0;\n"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                :: "r"(S_tmem[0]), "l"(desc_Q), "l"(desc_K), "r"(idesc), "r"(accumulate));
        }
    }
    for (int k = 0; k < 64; k += 16) {
        uint64_t desc_Q = make_smem_desc_swizzle_fn(smem_Q_1 + k, 1, 1024);
        uint64_t desc_K = make_smem_desc_swizzle_fn(smem_K_1[0] + k, 1, 1024);
        if (threadIdx.x == 0) {
            asm volatile(
                "{\n.reg .pred p;\n"
                "setp.ne.b32 p, 1, 0;\n"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                :: "r"(S_tmem[0]), "l"(desc_Q), "l"(desc_K), "r"(idesc));
        }
    }
    
    uint32_t mma_bar_ptr = (uint32_t)__cvta_generic_to_shared(mbar_mma);
    if (threadIdx.x == 0) {
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(mma_bar_ptr));
    }
    int mma_phase = 0;
    mbarrier_wait_fn(mbar_mma, mma_phase);
    mma_phase ^= 1;

    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    float scale_log2_e = (1.0f / sqrtf(128.0f)) * 1.4426950408889634f;

    for (int i = 0; i <= m_start / 128; ++i) {
        int n_idx = i * 128;
        int buf = i % 2;
        int next_buf = (i + 1) % 2;
        int next2_buf = (i + 2) % 2;
        
        float m_curr = -INFINITY;
        uint32_t s_r[128];
        
        // Massive batch TMEM read unroll to hide latencies
        #pragma unroll
        for (int c = 0; c < 128; c += 8) {
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                         : "=r"(s_r[c+0]),"=r"(s_r[c+1]),"=r"(s_r[c+2]),"=r"(s_r[c+3]),
                           "=r"(s_r[c+4]),"=r"(s_r[c+5]),"=r"(s_r[c+6]),"=r"(s_r[c+7]) 
                         : "r"(S_tmem[buf] + c));
        }
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        #pragma unroll
        for (int c = 0; c < 128; ++c) {
            float val = __uint_as_float(s_r[c]);
            int global_n = n_idx + c;
            int global_m = m_start + threadIdx.x;
            if (global_n > global_m || global_n >= S || global_m >= S) {
                val = -INFINITY;
            } else {
                val *= scale_log2_e;
            }
            s_r[c] = __float_as_uint(val);
            m_curr = fmaxf(m_curr, val);
        }
        
        float m_new = fmaxf(m_prev, m_curr);
        float rescale_factor = 1.0f;
        
        if (m_prev != -INFINITY && m_new != -INFINITY) {
            rescale_factor = fast_exp2f_fn(m_prev - m_new);
        } else if (m_prev == -INFINITY && m_new != -INFINITY) {
            rescale_factor = 0.0f;
        }
        
        if (i > 0 && rescale_factor < 1.0f) {
            uint32_t o_r[128];
            #pragma unroll
            for (int c = 0; c < 128; c += 8) {
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                             : "=r"(o_r[c+0]),"=r"(o_r[c+1]),"=r"(o_r[c+2]),"=r"(o_r[c+3]),
                               "=r"(o_r[c+4]),"=r"(o_r[c+5]),"=r"(o_r[c+6]),"=r"(o_r[c+7]) 
                             : "r"(O_tmem + c));
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            #pragma unroll
            for (int c = 0; c < 128; ++c) {
                float val = __uint_as_float(o_r[c]) * rescale_factor;
                o_r[c] = __float_as_uint(val);
            }
            
            #pragma unroll
            for (int c = 0; c < 128; c += 8) {
                asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
                             :: "r"(O_tmem + c), 
                                "r"(o_r[c+0]),"r"(o_r[c+1]),"r"(o_r[c+2]),"r"(o_r[c+3]),
                                "r"(o_r[c+4]),"r"(o_r[c+5]),"r"(o_r[c+6]),"r"(o_r[c+7]));
            }
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        }
        
        l_prev *= rescale_factor;
        
        #pragma unroll
        for (int c = 0; c < 128; c += 8) {
            float p[8];
            for (int j = 0; j < 8; ++j) {
                float val = __uint_as_float(s_r[c+j]);
                if (val == -INFINITY) {
                    p[j] = 0.0f;
                } else {
                    p[j] = fast_exp2f_fn(val - m_new);
                }
            }
            l_prev += p[0] + p[1] + p[2] + p[3] + p[4] + p[5] + p[6] + p[7];
            
            uint32_t p01 = pack_bf16_fn(__float_as_uint(p[0]), __float_as_uint(p[1]));
            uint32_t p23 = pack_bf16_fn(__float_as_uint(p[2]), __float_as_uint(p[3]));
            uint32_t p45 = pack_bf16_fn(__float_as_uint(p[4]), __float_as_uint(p[5]));
            uint32_t p67 = pack_bf16_fn(__float_as_uint(p[6]), __float_as_uint(p[7]));
            
            int y = threadIdx.x;
            int chunk_idx = (c % 64) / 8;
            int swizzled_chunk = (y % 8) ^ chunk_idx;
            
            if (c < 64) {
                uint32_t offset = (y * 64 + swizzled_chunk * 8) * sizeof(__nv_bfloat16);
                uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_P_0) + offset;
                asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" :: "r"(addr), "r"(p01), "r"(p23), "r"(p45), "r"(p67));
            } else {
                uint32_t offset = (y * 64 + swizzled_chunk * 8) * sizeof(__nv_bfloat16);
                uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_P_1) + offset;
                asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" :: "r"(addr), "r"(p01), "r"(p23), "r"(p45), "r"(p67));
            }
        }
        
        m_prev = m_new;
        fence_async_shared_fn();
        __syncthreads();
        tcgen05_fence_after_fn();
        
        // Asynchronously pipeline P * V(i) and Q * K(i+1) simultaneously 
        int v_phase = i / 2;
        mbarrier_wait_fn(&mbar_V[buf], v_phase % 2);
        
        uint32_t idesc_V0 = make_instr_desc_fn(128, 128);
        idesc_V0 |= (1u << 16); 
        for (int k = 0; k < 128; k += 16) {
            uint64_t desc_P = (k < 64) ? make_smem_desc_swizzle_fn(smem_P_0 + k, 1, 1024) 
                                       : make_smem_desc_swizzle_fn(smem_P_1 + (k - 64), 1, 1024);
            uint64_t desc_V = make_smem_desc_swizzle_fn(smem_V_0[buf] + k * 64, 16384, 1024);
            
            if (threadIdx.x == 0) {
                asm volatile(
                    "{\n.reg .pred p;\n"
                    "setp.ne.b32 p, 1, 0;\n"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                    :: "r"(O_tmem), "l"(desc_P), "l"(desc_V), "r"(idesc_V0));
            }
        }
        for (int k = 0; k < 128; k += 16) {
            uint64_t desc_P = (k < 64) ? make_smem_desc_swizzle_fn(smem_P_0 + k, 1, 1024) 
                                       : make_smem_desc_swizzle_fn(smem_P_1 + (k - 64), 1, 1024);
            uint64_t desc_V = make_smem_desc_swizzle_fn(smem_V_1[buf] + k * 64, 16384, 1024);
            if (threadIdx.x == 0) {
                asm volatile(
                    "{\n.reg .pred p;\n"
                    "setp.ne.b32 p, 1, 0;\n"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                    :: "r"(O_tmem + 64), "l"(desc_P), "l"(desc_V), "r"(idesc_V0));
            }
        }

        if (i + 1 <= m_start / 128) {
            int k_phase = (i + 1) / 2;
            mbarrier_wait_fn(&mbar_K[next_buf], k_phase % 2);
            
            for (int k = 0; k < 64; k += 16) {
                uint64_t desc_Q = make_smem_desc_swizzle_fn(smem_Q_0 + k, 1, 1024);
                uint64_t desc_K = make_smem_desc_swizzle_fn(smem_K_0[next_buf] + k, 1, 1024);
                int accumulate = (k == 0) ? 0 : 1;
                if (threadIdx.x == 0) {
                    asm volatile(
                        "{\n.reg .pred p;\n"
                        "setp.ne.b32 p, %4, 0;\n"
                        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                        :: "r"(S_tmem[next_buf]), "l"(desc_Q), "l"(desc_K), "r"(idesc), "r"(accumulate));
                }
            }
            for (int k = 0; k < 64; k += 16) {
                uint64_t desc_Q = make_smem_desc_swizzle_fn(smem_Q_1 + k, 1, 1024);
                uint64_t desc_K = make_smem_desc_swizzle_fn(smem_K_1[next_buf] + k, 1, 1024);
                if (threadIdx.x == 0) {
                    asm volatile(
                        "{\n.reg .pred p;\n"
                        "setp.ne.b32 p, 1, 0;\n"
                        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                        :: "r"(S_tmem[next_buf]), "l"(desc_Q), "l"(desc_K), "r"(idesc));
                }
            }
            
            if (i + 2 <= m_start / 128) {
                if (threadIdx.x == 0) {
                    int load_n_idx = (i + 2) * 128;
                    mbarrier_arrive_and_expect_tx_fn(&mbar_K[next2_buf], 32768);
                    tma_load_4d_fn(&tma_K, &mbar_K[next2_buf], smem_K_0[next2_buf], 0, load_n_idx, head_idx, batch_idx);
                    tma_load_4d_fn(&tma_K, &mbar_K[next2_buf], smem_K_1[next2_buf], 64, load_n_idx, head_idx, batch_idx);

                    mbarrier_arrive_and_expect_tx_fn(&mbar_V[next2_buf], 32768);
                    tma_load_4d_fn(&tma_V, &mbar_V[next2_buf], smem_V_0[next2_buf], 0, load_n_idx, head_idx, batch_idx);
                    tma_load_4d_fn(&tma_V, &mbar_V[next2_buf], smem_V_1[next2_buf], 64, load_n_idx, head_idx, batch_idx);
                }
            }
        }
        
        if (threadIdx.x == 0) {
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(mma_bar_ptr));
        }
        mbarrier_wait_fn(mbar_mma, mma_phase);
        mma_phase ^= 1;
    }

    float out_scale = 1.0f / fmaxf(l_prev, 1e-12f);
    __nv_bfloat16* D_ptr = O + batch_idx * (H_size * S * D) + head_idx * (S * D);
    
    for (uint32_t col = 0; col < 128; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(O_tmem + col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        uint32_t base = threadIdx.x * 64 + (col % 64);
        __nv_bfloat16* dst = (col < 64) ? smem_P_0 : smem_P_1;
        dst[base + 0] = __float2bfloat16(__uint_as_float(r0) * out_scale);
        dst[base + 1] = __float2bfloat16(__uint_as_float(r1) * out_scale);
        dst[base + 2] = __float2bfloat16(__uint_as_float(r2) * out_scale);
        dst[base + 3] = __float2bfloat16(__uint_as_float(r3) * out_scale);
    }
    __syncthreads();
    
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t num_steps = (128 + 3) / 4; 
    for (uint32_t step = 0; step < num_steps; ++step) {
        uint32_t row = step * 4 + warp_id;
        if (row >= 128) continue;
        uint32_t global_row = m_start + row;
        uint32_t col_start = lane_id * 4;
        if (global_row < S && col_start < D) {
            uint2 data;
            if (col_start < 64) {
                data = *reinterpret_cast<uint2*>(&smem_P_0[row * 64 + col_start]);
            } else {
                data = *reinterpret_cast<uint2*>(&smem_P_1[row * 64 + (col_start - 64)]);
            }
            *reinterpret_cast<uint2*>(D_ptr + (size_t)global_row * D + col_start) = data;
        }
    }
    
    int global_m = m_start + threadIdx.x;
    if (global_m < S) {
        float lse_val = -INFINITY;
        if (m_prev != -INFINITY) {
            lse_val = (m_prev + log2f(l_prev)) * 0.6931471805599453f;
        }
        LSE[batch_idx * H_size * S + head_idx * S + global_m] = lse_val;
    }

    if (threadIdx.x < 32) {
        uint32_t taddr = *tmem_alloc_dst;
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" :: "r"(taddr));
    }
}

namespace tvm_ffi_example_cuda {

CUresult create_tma_4d_descriptor(CUtensorMap* d, void* globalAddress, uint64_t D, uint64_t S, uint64_t H, uint64_t B) {
    cuuint64_t globalDim[4] = {D, S, H, B};
    cuuint64_t globalStrides[3] = {D * 2, D * S * 2, D * S * H * 2};
    cuuint32_t boxDim[4] = {64, 128, 1, 1};
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
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);
    
    const __nv_bfloat16* q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* k_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* v_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* o_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* lse_ptr = static_cast<float*>(LSE.data_ptr());
    
    CUtensorMap tma_Q, tma_K, tma_V;
    CU_CHECK(create_tma_4d_descriptor(&tma_Q, (void*)q_ptr, D, S, H, B));
    CU_CHECK(create_tma_4d_descriptor(&tma_K, (void*)k_ptr, D, S, H, B));
    CU_CHECK(create_tma_4d_descriptor(&tma_V, (void*)v_ptr, D, S, H, B));
    
    int64_t blocks_m = (S + 127) / 128;
    dim3 grid(blocks_m, H, B);
    dim3 block(128);
    int smem_bytes = 196864; 
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    cudaFuncSetAttribute(flash_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    
    flash_fwd_kernel<<<grid, block, smem_bytes, stream>>>(
        tma_Q, tma_K, tma_V,
        o_ptr, lse_ptr,
        S, D
    );
    
    CUDA_CHECK(cudaGetLastError());
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda