#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>
#include <math.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
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
        l2Promotion,
        oobFill
    );
}

namespace tvm_ffi_flash_attn_sm100 {

__device__ __forceinline__ void tmem_alloc_cg1_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_cg1_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(addr), "r"(ncols));
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

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tma_store_2d_fn(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0, {%2, %3}], [%1];"
        :: "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tma_store_commit_fn() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void tma_store_wait_fn() {
    asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ uint32_t pack_bf16_fn(float a_f, float b_f) {
    __nv_bfloat16 a = __float2bfloat16(a_f);
    __nv_bfloat16 b = __float2bfloat16(b_f);
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};"
        : "=r"(result)
        : "h"(*reinterpret_cast<uint16_t*>(&a)),
          "h"(*reinterpret_cast<uint16_t*>(&b)));
    return result;
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61; // SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint64_t get_desc_Q(void* smem_Q, int k) {
    uint32_t block_idx = k / 4;
    uint32_t k_offset = (k % 4) * 32;
    char* ptr = (char*)smem_Q + block_idx * 16384 + k_offset;
    return make_smem_desc_sm100_fn(ptr, 1, 1024);
}

__device__ __forceinline__ uint64_t get_desc_K(void* smem_K, int k) {
    uint32_t block_idx = k / 4;
    uint32_t k_offset = (k % 4) * 32;
    char* ptr = (char*)smem_K + block_idx * 16384 + k_offset;
    return make_smem_desc_sm100_fn(ptr, 1, 1024);
}

__device__ __forceinline__ uint64_t get_desc_P(void* smem_P, int k) {
    uint32_t block_idx = k / 4;
    uint32_t k_offset = (k % 4) * 32;
    char* ptr = (char*)smem_P + block_idx * 16384 + k_offset;
    return make_smem_desc_sm100_fn(ptr, 1, 1024);
}

__device__ __forceinline__ uint64_t get_desc_V(void* smem_V, int k) {
    char* ptr = (char*)smem_V + k * 2048; 
    return make_smem_desc_sm100_fn(ptr, 2048, 1024);
}

__device__ __forceinline__ void umma_f16_cg1_fn(uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc, uint32_t accum) {
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
        :: "r"(a) : "memory");
}

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void store_swizzled(uint32_t row, uint32_t c, float f0, float f1, float f2, float f3, __nv_bfloat16* smem) {
    uint32_t bank_offset = ((row & 7) ^ (c >> 3)) << 3;
    uint32_t base = row * 64 + bank_offset + (c & 7);
    uint32_t out0 = pack_bf16_fn(f0, f1);
    uint32_t out1 = pack_bf16_fn(f2, f3);
    *reinterpret_cast<uint2*>(&smem[base]) = make_uint2(out0, out1);
}

__global__ void mha_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    float* __restrict__ LSE,
    int S, int H, int B
) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int m_block = blockIdx.x;

    __shared__ __align__(1024) __nv_bfloat16 smem_Q[2][128 * 64];
    __shared__ __align__(1024) __nv_bfloat16 smem_K[2][2][128 * 64];
    __shared__ __align__(1024) __nv_bfloat16 smem_V[2][2][128 * 64];
    __shared__ __align__(1024) __nv_bfloat16 smem_P[2][128 * 64];

    __shared__ __align__(16) uint64_t mbar_Q;
    __shared__ __align__(16) uint64_t mbar_K[2];
    __shared__ __align__(16) uint64_t mbar_V[2];
    __shared__ __align__(16) uint64_t mbar_UMMA_S[2];
    __shared__ __align__(16) uint64_t mbar_UMMA_O;
    __shared__ uint32_t smem_tmem_addr;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&mbar_Q, 1);
        init_smem_barrier_fn(&mbar_K[0], 1);
        init_smem_barrier_fn(&mbar_K[1], 1);
        init_smem_barrier_fn(&mbar_V[0], 1);
        init_smem_barrier_fn(&mbar_V[1], 1);
        init_smem_barrier_fn(&mbar_UMMA_S[0], 1);
        init_smem_barrier_fn(&mbar_UMMA_S[1], 1);
        init_smem_barrier_fn(&mbar_UMMA_O, 1);
    }
    if (threadIdx.x < 32) {
        tmem_alloc_cg1_fn(&smem_tmem_addr, 512);
    }
    __syncthreads();
    fence_smem_barrier_init_fn();

    uint32_t tmem_base = smem_tmem_addr;
    uint32_t tmem_O_left = tmem_base;
    uint32_t tmem_O_right = tmem_base + 64;
    uint32_t tmem_S[2] = { tmem_base + 128, tmem_base + 256 };

    uint32_t idesc_QK = (1u << 4) | (1u << 7) | (1u << 10) | (0u << 15) | (0u << 16) | ((128 / 8) << 17) | ((128 / 16) << 24);
    uint32_t idesc_PV_half = (1u << 4) | (1u << 7) | (1u << 10) | (0u << 15) | (1u << 16) | (8u << 17) | ((128 / 16) << 24);

    int row_offset_Q = (b * H + h) * S + m_block * 128;
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(&mbar_Q, 32768);
        tma_load_2d_fn(&tma_Q, &mbar_Q, smem_Q[0], 0, row_offset_Q);
        tma_load_2d_fn(&tma_Q, &mbar_Q, smem_Q[1], 64, row_offset_Q);
    }
    mbarrier_wait_fn(&mbar_Q, 0);

    int total_kv_blocks = (S + 127) / 128;
    if (total_kv_blocks > 0) {
        if (threadIdx.x == 0) {
            int row_offset_K = (b * H + h) * S + 0;
            mbarrier_arrive_and_expect_tx_fn(&mbar_K[0], 32768);
            tma_load_2d_fn(&tma_K, &mbar_K[0], smem_K[0][0], 0, row_offset_K);
            tma_load_2d_fn(&tma_K, &mbar_K[0], smem_K[0][1], 64, row_offset_K);
            
            mbarrier_arrive_and_expect_tx_fn(&mbar_V[0], 32768);
            tma_load_2d_fn(&tma_V, &mbar_V[0], smem_V[0][0], 0, row_offset_K);
            tma_load_2d_fn(&tma_V, &mbar_V[0], smem_V[0][1], 64, row_offset_K);
        }
    }

    if (total_kv_blocks > 0) {
        mbarrier_wait_fn(&mbar_K[0], 0);
        fence_proxy_async_fn();
        if (threadIdx.x == 0) {
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            for (int k = 0; k < 8; ++k) {
                uint64_t desc_A = get_desc_Q(smem_Q[0], k);
                uint64_t desc_B = get_desc_K((void*)smem_K[0][0], k);
                umma_f16_cg1_fn(tmem_S[0], desc_A, desc_B, idesc_QK, (k > 0));
            }
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            umma_commit_cg1_fn(&mbar_UMMA_S[0]);
        }
    }

    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    int umma_phase_S[2] = {0, 0};
    int umma_phase_O = 0;

    for (int kv = 0; kv < total_kv_blocks; ++kv) {
        int buf = kv % 2;
        int next_buf = (kv + 1) % 2;
        
        if (kv + 1 < total_kv_blocks) {
            int row_offset_K = (b * H + h) * S + (kv + 1) * 128;
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(&mbar_K[next_buf], 32768);
                tma_load_2d_fn(&tma_K, &mbar_K[next_buf], smem_K[next_buf][0], 0, row_offset_K);
                tma_load_2d_fn(&tma_K, &mbar_K[next_buf], smem_K[next_buf][1], 64, row_offset_K);
                
                mbarrier_arrive_and_expect_tx_fn(&mbar_V[next_buf], 32768);
                tma_load_2d_fn(&tma_V, &mbar_V[next_buf], smem_V[next_buf][0], 0, row_offset_K);
                tma_load_2d_fn(&tma_V, &mbar_V[next_buf], smem_V[next_buf][1], 64, row_offset_K);
            }
        }
        
        mbarrier_wait_fn(&mbar_UMMA_S[buf], umma_phase_S[buf]);
        umma_phase_S[buf] ^= 1;
        
        if (kv > 0) {
            mbarrier_wait_fn(&mbar_UMMA_O, umma_phase_O);
            umma_phase_O ^= 1;
        }

        uint32_t r_S[128];
        #pragma unroll
        for (int c = 0; c < 128; c += 8) {
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                : "=r"(r_S[c+0]),"=r"(r_S[c+1]),"=r"(r_S[c+2]),"=r"(r_S[c+3]),
                  "=r"(r_S[c+4]),"=r"(r_S[c+5]),"=r"(r_S[c+6]),"=r"(r_S[c+7]) : "r"(tmem_S[buf] + c));
        }
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        float row_max = -INFINITY;
        #pragma unroll
        for (int c = 0; c < 128; ++c) {
            float f = __uint_as_float(r_S[c]) * 0.08838834764f; 
            int global_k = kv * 128 + c;
            if (global_k >= S) f = -INFINITY;
            row_max = fmaxf(row_max, f);
            r_S[c] = __float_as_uint(f);
        }

        float m_new = fmaxf(m_prev, row_max);
        float scale_O = (m_prev == -INFINITY) ? 0.0f : fast_exp2f_fn((m_prev - m_new) * 1.4426950408889634f);

        int need_scale = __any_sync(0xFFFFFFFF, scale_O != 1.0f);
        if (need_scale && kv > 0) {
            #pragma unroll
            for (int c = 0; c < 128; c += 8) {
                uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                    : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),
                      "=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) : "r"(tmem_O_left + c));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                
                float f0 = __uint_as_float(r0) * scale_O;
                float f1 = __uint_as_float(r1) * scale_O;
                float f2 = __uint_as_float(r2) * scale_O;
                float f3 = __uint_as_float(r3) * scale_O;
                float f4 = __uint_as_float(r4) * scale_O;
                float f5 = __uint_as_float(r5) * scale_O;
                float f6 = __uint_as_float(r6) * scale_O;
                float f7 = __uint_as_float(r7) * scale_O;
                
                r0 = __float_as_uint(f0); r1 = __float_as_uint(f1); r2 = __float_as_uint(f2); r3 = __float_as_uint(f3);
                r4 = __float_as_uint(f4); r5 = __float_as_uint(f5); r6 = __float_as_uint(f6); r7 = __float_as_uint(f7);
                
                asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
                    :: "r"(tmem_O_left + c),
                       "r"(r0), "r"(r1), "r"(r2), "r"(r3),
                       "r"(r4), "r"(r5), "r"(r6), "r"(r7) : "memory");
            }
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        }

        float row_sum = 0.0f;
        #pragma unroll
        for (int c = 0; c < 128; c += 8) {
            float f0 = __uint_as_float(r_S[c+0]);
            float f1 = __uint_as_float(r_S[c+1]);
            float f2 = __uint_as_float(r_S[c+2]);
            float f3 = __uint_as_float(r_S[c+3]);
            float f4 = __uint_as_float(r_S[c+4]);
            float f5 = __uint_as_float(r_S[c+5]);
            float f6 = __uint_as_float(r_S[c+6]);
            float f7 = __uint_as_float(r_S[c+7]);
            
            int global_k = kv * 128 + c;
            f0 = (global_k + 0 >= S) ? 0.0f : fast_exp2f_fn((f0 - m_new) * 1.4426950408889634f);
            f1 = (global_k + 1 >= S) ? 0.0f : fast_exp2f_fn((f1 - m_new) * 1.4426950408889634f);
            f2 = (global_k + 2 >= S) ? 0.0f : fast_exp2f_fn((f2 - m_new) * 1.4426950408889634f);
            f3 = (global_k + 3 >= S) ? 0.0f : fast_exp2f_fn((f3 - m_new) * 1.4426950408889634f);
            f4 = (global_k + 4 >= S) ? 0.0f : fast_exp2f_fn((f4 - m_new) * 1.4426950408889634f);
            f5 = (global_k + 5 >= S) ? 0.0f : fast_exp2f_fn((f5 - m_new) * 1.4426950408889634f);
            f6 = (global_k + 6 >= S) ? 0.0f : fast_exp2f_fn((f6 - m_new) * 1.4426950408889634f);
            f7 = (global_k + 7 >= S) ? 0.0f : fast_exp2f_fn((f7 - m_new) * 1.4426950408889634f);
            
            row_sum += f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7;
            
            if (c < 64) {
                store_swizzled(threadIdx.x, c, f0, f1, f2, f3, smem_P[0]);
                store_swizzled(threadIdx.x, c+4, f4, f5, f6, f7, smem_P[0]);
            } else {
                store_swizzled(threadIdx.x, c-64, f0, f1, f2, f3, smem_P[1]);
                store_swizzled(threadIdx.x, c-60, f4, f5, f6, f7, smem_P[1]);
            }
        }
        
        l_prev = l_prev * scale_O + row_sum;
        m_prev = m_new;

        fence_proxy_async_fn();

        if (kv + 1 < total_kv_blocks) {
            mbarrier_wait_fn(&mbar_K[next_buf], ((kv + 1) / 2) & 1);
            if (threadIdx.x == 0) {
                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                for (int k = 0; k < 8; ++k) {
                    uint64_t desc_A = get_desc_Q(smem_Q[0], k);
                    uint64_t desc_B = get_desc_K((void*)smem_K[next_buf][0], k);
                    umma_f16_cg1_fn(tmem_S[next_buf], desc_A, desc_B, idesc_QK, (k > 0));
                }
                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                umma_commit_cg1_fn(&mbar_UMMA_S[next_buf]);
            }
        }

        mbarrier_wait_fn(&mbar_V[buf], (kv / 2) & 1);
        if (threadIdx.x == 0) {
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            for (int k = 0; k < 8; ++k) {
                uint64_t desc_A = get_desc_P(smem_P[0], k);
                uint64_t desc_B_left = get_desc_V((void*)smem_V[buf][0], k);
                uint64_t desc_B_right = get_desc_V((void*)smem_V[buf][1], k);
                
                umma_f16_cg1_fn(tmem_O_left, desc_A, desc_B_left, idesc_PV_half, (kv > 0 || k > 0));
                umma_f16_cg1_fn(tmem_O_right, desc_A, desc_B_right, idesc_PV_half, (kv > 0 || k > 0));
            }
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            umma_commit_cg1_fn(&mbar_UMMA_O);
        }
    }

    if (total_kv_blocks > 0) {
        mbarrier_wait_fn(&mbar_UMMA_O, umma_phase_O);
    }

    float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
    #pragma unroll
    for (int c = 0; c < 128; c += 8) {
        uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),
              "=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) : "r"(tmem_O_left + c));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        float f0 = __uint_as_float(r0) * inv_l;
        float f1 = __uint_as_float(r1) * inv_l;
        float f2 = __uint_as_float(r2) * inv_l;
        float f3 = __uint_as_float(r3) * inv_l;
        float f4 = __uint_as_float(r4) * inv_l;
        float f5 = __uint_as_float(r5) * inv_l;
        float f6 = __uint_as_float(r6) * inv_l;
        float f7 = __uint_as_float(r7) * inv_l;
        
        if (c < 64) {
            store_swizzled(threadIdx.x, c, f0, f1, f2, f3, smem_Q[0]);
            store_swizzled(threadIdx.x, c+4, f4, f5, f6, f7, smem_Q[0]);
        } else {
            store_swizzled(threadIdx.x, c-64, f0, f1, f2, f3, smem_Q[1]);
            store_swizzled(threadIdx.x, c-60, f4, f5, f6, f7, smem_Q[1]);
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        tma_store_2d_fn(&tma_O, smem_Q[0], 0, row_offset_Q);
        tma_store_2d_fn(&tma_O, smem_Q[1], 64, row_offset_Q);
        tma_store_commit_fn();
    }

    if (threadIdx.x < 128) {
        int row = threadIdx.x;
        int global_row = m_block * 128 + row;
        if (global_row < S) {
            float final_lse = (m_prev == -INFINITY) ? -INFINITY : (m_prev + logf(l_prev));
            LSE[(b * H + h) * S + global_row] = final_lse;
        }
    }

    if (threadIdx.x == 0) {
        tma_store_wait_fn<0>();
    }
    __syncthreads();
    
    if (threadIdx.x < 32) {
        tmem_dealloc_cg1_fn(tmem_base, 512);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_O;
    uint64_t rows = B * H * S;
    uint64_t cols = D;
    
    CUresult res;
    res = create_tma_2d_descriptor_2B(&tma_Q, Q.data_ptr(), cols, rows, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA Q failed\n"); exit(1); }
    
    res = create_tma_2d_descriptor_2B(&tma_K, K.data_ptr(), cols, rows, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA K failed\n"); exit(1); }
    
    res = create_tma_2d_descriptor_2B(&tma_V, V.data_ptr(), cols, rows, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA V failed\n"); exit(1); }
    
    res = create_tma_2d_descriptor_2B(&tma_O, O.data_ptr(), cols, rows, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA O failed\n"); exit(1); }
    
    int64_t m_blocks = (S + 127) / 128;
    dim3 grid(m_blocks, H, B);
    dim3 block(128);
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    mha_kernel<<<grid, block, 0, stream>>>(tma_Q, tma_K, tma_V, tma_O, static_cast<float*>(LSE.data_ptr()), S, H, B);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_flash_attn_sm100::run);

}