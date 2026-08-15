#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
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

__device__ __forceinline__ void tma_store_3d_fn(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group [%0, {%2, %3, %4}], [%1];"
        :: "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
}

__device__ __forceinline__ void tma_store_commit_fn() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void tma_store_wait_fn() {
    asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory");
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

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo, uint32_t swizzle) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint32_t base_offset = (addr >> 7) & 0x7;
    d |= (uint64_t)((addr & 0x3FFFF) >> 4);
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)base_offset << 49;
    d |= (uint64_t)swizzle << 61;
    return d;
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

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_after_fn() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_before_fn() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
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

__device__ __forceinline__ uint64_t advance_desc_k_kmajor(uint64_t desc, uint32_t k_elements) {
    uint32_t k_bytes = k_elements * 2;
    uint32_t addr16 = desc & 0x3FFF;
    uint32_t addr = addr16 << 4;
    addr += k_bytes;
    
    uint32_t base_offset = (addr >> 7) & 0x7;
    
    desc &= ~0x3FFFULL;
    desc |= (addr >> 4) & 0x3FFF;
    
    desc &= ~(0x7ULL << 49);
    desc |= ((uint64_t)base_offset << 49);
    
    return desc;
}

__device__ __forceinline__ uint64_t advance_desc_k_mnmajor(uint64_t desc, uint32_t k_elements, uint32_t stride_bytes) {
    uint32_t k_bytes = k_elements * stride_bytes;
    uint32_t addr16 = desc & 0x3FFF;
    uint32_t addr = addr16 << 4;
    addr += k_bytes;
    
    uint32_t base_offset = (addr >> 7) & 0x7;
    
    desc &= ~0x3FFFULL;
    desc |= (addr >> 4) & 0x3FFF;
    
    desc &= ~(0x7ULL << 49);
    desc |= ((uint64_t)base_offset << 49);
    
    return desc;
}

extern __shared__ __align__(128) char smem[];

__global__ __launch_bounds__(128) void mha_forward_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    float* lse_ptr,
    int S
) {
    setmaxnreg_inc_sync_fn<248>();

    int b = blockIdx.z;
    int h = blockIdx.y;
    int s_q = blockIdx.x * 128;
    int bh = b * gridDim.y + h;
    
    if (s_q >= S) return;

    char* smem_Q = smem;                                      // 32 KB
    char* smem_K[2] = {smem + 32768, smem + 32768 + 32768};   // 64 KB total
    char* smem_V[2] = {smem + 98304, smem + 98304 + 32768};   // 64 KB total
    char* smem_P = smem + 163840;                             // 32 KB
    
    uint64_t* mbar_Q = (uint64_t*)(smem_P + 32768);
    uint64_t* mbar_KV = mbar_Q + 1;
    uint64_t* mbar_mma = mbar_KV + 1;
    
    uint32_t* tmem_base_ptr = (uint32_t*)(mbar_mma + 1);

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(mbar_KV, 1);
        init_smem_barrier_fn(mbar_mma, 1);
        fence_smem_barrier_init_fn();
        
        mbarrier_arrive_and_expect_tx_fn(mbar_Q, 2 * 16384);
        tma_load_3d_fn(&tma_Q, mbar_Q, smem_Q, 0, s_q, bh);
        tma_load_3d_fn(&tma_Q, mbar_Q, smem_Q + 16384, 64, s_q, bh);
    }
    
    tcgen05_fence_before_fn();
    __syncthreads();
    tcgen05_fence_after_fn();

    if (threadIdx.x / 32 == 0) {
        tmem_alloc_fn(tmem_base_ptr, 256);
    }
    
    tcgen05_fence_before_fn();
    __syncthreads();
    tcgen05_fence_after_fn();
    
    uint32_t tmem_base = *tmem_base_ptr;
    uint32_t tmem_O = tmem_base;
    uint32_t tmem_S = tmem_base + 128;

    mbarrier_wait_fn(mbar_Q, 0);
    
    // K-Major descriptors for Q and K
    uint64_t desc_Q_left = make_smem_desc_sm100_fn(smem_Q, 1, 1024, 2);
    uint64_t desc_Q_right = make_smem_desc_sm100_fn(smem_Q + 16384, 1, 1024, 2);
    uint64_t desc_K_left[2] = {
        make_smem_desc_sm100_fn(smem_K[0], 1, 1024, 2),
        make_smem_desc_sm100_fn(smem_K[1], 1, 1024, 2)
    };
    uint64_t desc_K_right[2] = {
        make_smem_desc_sm100_fn(smem_K[0] + 16384, 1, 1024, 2),
        make_smem_desc_sm100_fn(smem_K[1] + 16384, 1, 1024, 2)
    };
    
    // MN-Major descriptors for V
    uint64_t desc_V_left[2] = {
        make_smem_desc_sm100_fn(smem_V[0], 16384, 1024, 2),
        make_smem_desc_sm100_fn(smem_V[1], 16384, 1024, 2)
    };
    uint64_t desc_V_right[2] = {
        make_smem_desc_sm100_fn(smem_V[0] + 16384, 16384, 1024, 2),
        make_smem_desc_sm100_fn(smem_V[1] + 16384, 16384, 1024, 2)
    };
    
    // K-Major descriptors for P
    uint64_t desc_P_left = make_smem_desc_sm100_fn(smem_P, 1, 1024, 2);
    uint64_t desc_P_right = make_smem_desc_sm100_fn(smem_P + 16384, 1, 1024, 2);

    uint32_t idesc_QK = 0;
    idesc_QK |= (1u << 4) | (1u << 7) | (1u << 10);
    idesc_QK |= (0u << 15) | (0u << 16);
    idesc_QK |= (16u << 17) | (8u << 24); // N=128, M=128

    uint32_t idesc_PV = 0;
    idesc_PV |= (1u << 4) | (1u << 7) | (1u << 10);
    idesc_PV |= (0u << 15) | (1u << 16);
    idesc_PV |= (8u << 17) | (8u << 24); // N=64, M=128

    uint32_t phase_KV = 0;
    uint32_t phase_mma = 0;
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_KV, 4 * 16384);
        tma_load_3d_fn(&tma_K, mbar_KV, smem_K[0], 0, 0, bh);
        tma_load_3d_fn(&tma_K, mbar_KV, smem_K[0] + 16384, 64, 0, bh);
        tma_load_3d_fn(&tma_V, mbar_KV, smem_V[0], 0, 0, bh);
        tma_load_3d_fn(&tma_V, mbar_KV, smem_V[0] + 16384, 64, 0, bh);
    }
    
    float m_i = -1e20f;
    float l_i = 0.0f;
    int tid = threadIdx.x;
    
    for (int j = 0; j < S; j += 128) {
        int buf = (j / 128) % 2;
        int next_buf = (buf + 1) % 2;
        
        mbarrier_wait_fn(mbar_KV, phase_KV);
        
        if (j + 128 < S) {
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(mbar_KV, 4 * 16384);
                tma_load_3d_fn(&tma_K, mbar_KV, smem_K[next_buf], 0, j + 128, bh);
                tma_load_3d_fn(&tma_K, mbar_KV, smem_K[next_buf] + 16384, 64, j + 128, bh);
                tma_load_3d_fn(&tma_V, mbar_KV, smem_V[next_buf], 0, j + 128, bh);
                tma_load_3d_fn(&tma_V, mbar_KV, smem_V[next_buf] + 16384, 64, j + 128, bh);
            }
        }
        
        if (threadIdx.x == 0) {
            uint64_t cur_desc_Q = desc_Q_left;
            uint64_t cur_desc_K = desc_K_left[buf];
            for (int k = 0; k < 64; k += 16) {
                int accum = (k == 0) ? 0 : 1;
                umma_f16_cg1_fn(tmem_S, cur_desc_Q, cur_desc_K, idesc_QK, accum);
                cur_desc_Q = advance_desc_k_kmajor(cur_desc_Q, 16);
                cur_desc_K = advance_desc_k_kmajor(cur_desc_K, 16);
            }
            
            cur_desc_Q = desc_Q_right;
            cur_desc_K = desc_K_right[buf];
            for (int k = 0; k < 64; k += 16) {
                umma_f16_cg1_fn(tmem_S, cur_desc_Q, cur_desc_K, idesc_QK, 1);
                cur_desc_Q = advance_desc_k_kmajor(cur_desc_Q, 16);
                cur_desc_K = advance_desc_k_kmajor(cur_desc_K, 16);
            }
            
            uint32_t a = (uint32_t)__cvta_generic_to_shared(mbar_mma);
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(a));
        }
        
        mbarrier_wait_fn(mbar_mma, phase_mma);
        phase_mma ^= 1;
        
        float row[128];
        for (int c = 0; c < 128; c += 4) {
            uint32_t* r = (uint32_t*)&row[c];
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" 
                : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]) : "r"(tmem_S + c));
        }
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        float rowmax = -1e20f;
        for (int c = 0; c < 128; ++c) {
            uint32_t bits = ((uint32_t*)row)[c];
            float f = (s_q + tid >= S || j + c >= S) ? -1e20f : __uint_as_float(bits) * 0.08838834764f;
            rowmax = max(rowmax, f);
            row[c] = f;
        }
        
        float m_prev = m_i;
        float m_new = max(m_prev, rowmax);
        
        float sum = 0.0f;
        for (int c = 0; c < 128; ++c) {
            float p = (s_q + tid >= S || j + c >= S) ? 0.0f : fast_exp2f_fn((row[c] - m_new) * 1.44269504089f);
            row[c] = p;
            sum += p;
        }
        
        float scale = fast_exp2f_fn((m_prev - m_new) * 1.44269504089f);
        l_i = l_i * scale + sum;
        m_i = m_new;
        
        for (int c = 0; c < 64; c += 8) {
            uint32_t p0 = pack_bf16_fn(__float_as_uint(row[c+0]), __float_as_uint(row[c+1]));
            uint32_t p1 = pack_bf16_fn(__float_as_uint(row[c+2]), __float_as_uint(row[c+3]));
            uint32_t p2 = pack_bf16_fn(__float_as_uint(row[c+4]), __float_as_uint(row[c+5]));
            uint32_t p3 = pack_bf16_fn(__float_as_uint(row[c+6]), __float_as_uint(row[c+7]));
            int x = c / 8;
            int swizzled_x = (tid % 8) ^ x;
            int offset = (tid * 8 + swizzled_x) * 16;
            st_shared_128_fn((uint32_t)__cvta_generic_to_shared(smem_P + offset), p0, p1, p2, p3);
        }
        for (int c = 64; c < 128; c += 8) {
            uint32_t p0 = pack_bf16_fn(__float_as_uint(row[c+0]), __float_as_uint(row[c+1]));
            uint32_t p1 = pack_bf16_fn(__float_as_uint(row[c+2]), __float_as_uint(row[c+3]));
            uint32_t p2 = pack_bf16_fn(__float_as_uint(row[c+4]), __float_as_uint(row[c+5]));
            uint32_t p3 = pack_bf16_fn(__float_as_uint(row[c+6]), __float_as_uint(row[c+7]));
            int x = (c - 64) / 8;
            int swizzled_x = (tid % 8) ^ x;
            int offset = (tid * 8 + swizzled_x) * 16;
            st_shared_128_fn((uint32_t)__cvta_generic_to_shared(smem_P + 16384 + offset), p0, p1, p2, p3);
        }
        
        fence_async_shared_fn(); 
        tcgen05_fence_before_fn();
        __syncthreads();
        tcgen05_fence_after_fn();
        
        if (j > 0 && scale != 1.0f) {
            for (int c = 0; c < 128; c += 8) {
                uint32_t* r = (uint32_t*)&row[c];
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];" 
                    : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7])
                    : "r"(tmem_O + c));
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            for (int c = 0; c < 128; ++c) {
                float f = __uint_as_float(((uint32_t*)row)[c]) * scale;
                ((uint32_t*)row)[c] = __float_as_uint(f);
            }
            for (int c = 0; c < 128; c += 4) {
                uint32_t* r = (uint32_t*)&row[c];
                asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};" 
                    :: "r"(tmem_O + c), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]));
            }
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        }
        
        tcgen05_fence_before_fn();
        __syncthreads();
        tcgen05_fence_after_fn();
        
        if (threadIdx.x == 0) {
            uint64_t cur_desc_P = desc_P_left;
            uint64_t cur_desc_V = desc_V_left[buf];
            for (int k = 0; k < 64; k += 16) {
                int accum = (j == 0 && k == 0) ? 0 : 1;
                umma_f16_cg1_fn(tmem_O + 0, cur_desc_P, cur_desc_V, idesc_PV, accum);
                cur_desc_P = advance_desc_k_kmajor(cur_desc_P, 16);
                cur_desc_V = advance_desc_k_mnmajor(cur_desc_V, 16, 128);
            }
            
            cur_desc_P = desc_P_right;
            for (int k = 64; k < 128; k += 16) {
                umma_f16_cg1_fn(tmem_O + 0, cur_desc_P, cur_desc_V, idesc_PV, 1);
                cur_desc_P = advance_desc_k_kmajor(cur_desc_P, 16);
                cur_desc_V = advance_desc_k_mnmajor(cur_desc_V, 16, 128);
            }
            
            cur_desc_P = desc_P_left;
            cur_desc_V = desc_V_right[buf];
            for (int k = 0; k < 64; k += 16) {
                int accum = (j == 0 && k == 0) ? 0 : 1;
                umma_f16_cg1_fn(tmem_O + 64, cur_desc_P, cur_desc_V, idesc_PV, accum);
                cur_desc_P = advance_desc_k_kmajor(cur_desc_P, 16);
                cur_desc_V = advance_desc_k_mnmajor(cur_desc_V, 16, 128);
            }
            
            cur_desc_P = desc_P_right;
            for (int k = 64; k < 128; k += 16) {
                umma_f16_cg1_fn(tmem_O + 64, cur_desc_P, cur_desc_V, idesc_PV, 1);
                cur_desc_P = advance_desc_k_kmajor(cur_desc_P, 16);
                cur_desc_V = advance_desc_k_mnmajor(cur_desc_V, 16, 128);
            }
            
            uint32_t a = (uint32_t)__cvta_generic_to_shared(mbar_mma);
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(a));
        }
        
        mbarrier_wait_fn(mbar_mma, phase_mma);
        phase_mma ^= 1;
        phase_KV ^= 1;
    }
    
    float inv_l = 1.0f / l_i;
    float row[128];
    for (int c = 0; c < 128; c += 8) {
        uint32_t* r = (uint32_t*)&row[c];
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];" 
            : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]) 
            : "r"(tmem_O + c));
    }
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    
    for (int c = 0; c < 64; c += 8) {
        float f0 = __uint_as_float(((uint32_t*)row)[c+0]) * inv_l;
        float f1 = __uint_as_float(((uint32_t*)row)[c+1]) * inv_l;
        uint32_t p0 = pack_bf16_fn(__float_as_uint(f0), __float_as_uint(f1));
        
        float f2 = __uint_as_float(((uint32_t*)row)[c+2]) * inv_l;
        float f3 = __uint_as_float(((uint32_t*)row)[c+3]) * inv_l;
        uint32_t p1 = pack_bf16_fn(__float_as_uint(f2), __float_as_uint(f3));
        
        float f4 = __uint_as_float(((uint32_t*)row)[c+4]) * inv_l;
        float f5 = __uint_as_float(((uint32_t*)row)[c+5]) * inv_l;
        uint32_t p2 = pack_bf16_fn(__float_as_uint(f4), __float_as_uint(f5));
        
        float f6 = __uint_as_float(((uint32_t*)row)[c+6]) * inv_l;
        float f7 = __uint_as_float(((uint32_t*)row)[c+7]) * inv_l;
        uint32_t p3 = pack_bf16_fn(__float_as_uint(f6), __float_as_uint(f7));
        
        int x = c / 8;
        int swizzled_x = (tid % 8) ^ x;
        int offset = (tid * 8 + swizzled_x) * 16;
        st_shared_128_fn((uint32_t)__cvta_generic_to_shared(smem_P + offset), p0, p1, p2, p3);
    }
    
    for (int c = 64; c < 128; c += 8) {
        float f0 = __uint_as_float(((uint32_t*)row)[c+0]) * inv_l;
        float f1 = __uint_as_float(((uint32_t*)row)[c+1]) * inv_l;
        uint32_t p0 = pack_bf16_fn(__float_as_uint(f0), __float_as_uint(f1));
        
        float f2 = __uint_as_float(((uint32_t*)row)[c+2]) * inv_l;
        float f3 = __uint_as_float(((uint32_t*)row)[c+3]) * inv_l;
        uint32_t p1 = pack_bf16_fn(__float_as_uint(f2), __float_as_uint(f3));
        
        float f4 = __uint_as_float(((uint32_t*)row)[c+4]) * inv_l;
        float f5 = __uint_as_float(((uint32_t*)row)[c+5]) * inv_l;
        uint32_t p2 = pack_bf16_fn(__float_as_uint(f4), __float_as_uint(f5));
        
        float f6 = __uint_as_float(((uint32_t*)row)[c+6]) * inv_l;
        float f7 = __uint_as_float(((uint32_t*)row)[c+7]) * inv_l;
        uint32_t p3 = pack_bf16_fn(__float_as_uint(f6), __float_as_uint(f7));
        
        int x = (c - 64) / 8;
        int swizzled_x = (tid % 8) ^ x;
        int offset = (tid * 8 + swizzled_x) * 16;
        st_shared_128_fn((uint32_t)__cvta_generic_to_shared(smem_P + 16384 + offset), p0, p1, p2, p3);
    }
    
    fence_async_shared_fn(); 
    tcgen05_fence_before_fn();
    __syncthreads();
    tcgen05_fence_after_fn();
    
    if (threadIdx.x == 0) {
        tma_store_3d_fn(&tma_O, smem_P, 0, s_q, bh);
        tma_store_3d_fn(&tma_O, smem_P + 16384, 64, s_q, bh);
        tma_store_commit_fn();
        tma_store_wait_fn<0>();
    }
    
    if (s_q + tid < S) {
        lse_ptr[bh * S + s_q + tid] = m_i + __logf(l_i);
    }
    
    tcgen05_fence_before_fn();
    __syncthreads();
    tcgen05_fence_after_fn();
    if (threadIdx.x / 32 == 0) {
        tmem_dealloc_fn(tmem_base, 256);
    }
}

CUresult create_tma_3d_descriptor_2B(
    CUtensorMap* d, void* globalAddress, 
    uint64_t dim0, uint64_t dim1, uint64_t dim2, 
    uint32_t box0, uint32_t box1, uint32_t box2, 
    CUtensorMapSwizzle swizzle) 
{
    cuuint64_t globalDim[3] = {dim0, dim1, dim2};
    cuuint64_t globalStrides[2] = {dim0 * 2, dim0 * dim1 * 2};
    cuuint32_t boxDim[3] = {box0, box1, box2};
    cuuint32_t elementStrides[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, globalAddress,
        globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

namespace tvm_ffi_example_cuda {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);

    void* q_ptr = Q.data_ptr();
    void* k_ptr = K.data_ptr();
    void* v_ptr = V.data_ptr();
    void* o_ptr = O.data_ptr();
    float* lse_ptr = static_cast<float*>(LSE.data_ptr());

    CUtensorMap tma_Q, tma_K, tma_V, tma_O;
    if (create_tma_3d_descriptor_2B(&tma_Q, q_ptr, 128, S, B * H, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B) != CUDA_SUCCESS ||
        create_tma_3d_descriptor_2B(&tma_K, k_ptr, 128, S, B * H, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B) != CUDA_SUCCESS ||
        create_tma_3d_descriptor_2B(&tma_V, v_ptr, 128, S, B * H, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B) != CUDA_SUCCESS ||
        create_tma_3d_descriptor_2B(&tma_O, o_ptr, 128, S, B * H, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B) != CUDA_SUCCESS) {
        fprintf(stderr, "Failed to create TMA descriptors\n");
        exit(1);
    }

    int blocks_s = (S + 127) / 128;
    dim3 grid(blocks_s, H, B);
    dim3 block(128); 
    
    int smem_bytes = 196864; 
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    CUDA_CHECK(cudaFuncSetAttribute((void*)mha_forward_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    
    mha_forward_kernel<<<grid, block, smem_bytes, stream>>>(
        tma_Q, tma_K, tma_V, tma_O, lse_ptr, S
    );
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

}  // namespace tvm_ffi_example_cuda