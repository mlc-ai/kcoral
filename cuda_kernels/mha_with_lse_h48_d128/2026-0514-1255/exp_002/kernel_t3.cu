#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>
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

#define CU_CHECK(call) do {                                        \
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        fprintf(stderr, "CU error %d at %s:%d\n",                 \
                _e, __FILE__, __LINE__);                           \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_mha {

CUresult create_tma_3d_descriptor_2B(CUtensorMap* d, void* globalAddress, 
                                     uint64_t dim0, uint64_t dim1, uint64_t dim2, 
                                     uint32_t box0, uint32_t box1, uint32_t box2) {
    cuuint64_t globalDim[3] = {dim0, dim1, dim2};
    cuuint64_t globalStrides[2] = {dim0 * 2, dim0 * dim1 * 2};
    cuuint32_t boxDim[3] = {box0, box1, box2};
    cuuint32_t elementStrides[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(
        d,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        3, 
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

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ void st_shared_128_fn(uint32_t addr, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                 :: "r"(addr), "r"(v0), "r"(v1), "r"(v2), "r"(v3) : "memory");
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

__device__ __forceinline__ void tma_store_fence_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tma_store_commit_fn() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void tma_store_wait_fn() {
    asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory");
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

__device__ __forceinline__ void tmem_load_8x_fn(uint32_t col,
    uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3,
    uint32_t* r4, uint32_t* r5, uint32_t* r6, uint32_t* r7) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3),
                   "=r"(*r4),"=r"(*r5),"=r"(*r6),"=r"(*r7) : "r"(col));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_after_fn() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
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
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
        :: "r"(a) : "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t sbo, uint32_t lbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr) >> 4;
    d |= (uint64_t)(addr & 0x3FFF);
    d |= (uint64_t)((lbo >> 4) & 0x3FFF) << 16;
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32; 
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   // SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, bool trans_a=false, bool trans_b=false) {
    uint32_t d = 0;
    d |= (1u << 4);           
    d |= (1u << 7);           
    d |= (1u << 10);          
    if (trans_a) d |= (1u << 15);
    if (trans_b) d |= (1u << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__global__ void __launch_bounds__(128, 2) cta_gemm_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    float* lse_ptr,
    int S)
{
    int b = blockIdx.z;
    int h = blockIdx.y;
    int m_base = blockIdx.x * 128;
    int bh = b * gridDim.y + h;

    extern __shared__ uint8_t smem[];
    uint8_t* smem_Q0 = smem; 
    uint8_t* smem_Q1 = smem_Q0 + 16384; 
    uint8_t* smem_K0 = smem_Q1 + 16384; 
    uint8_t* smem_K1 = smem_K0 + 16384; 
    uint8_t* smem_V_left = smem_K1 + 16384; 
    uint8_t* smem_V_right = smem_V_left + 16384; 
    uint8_t* smem_P0 = smem_V_right + 16384; 
    uint8_t* smem_P1 = smem_P0 + 16384; 
    uint8_t* smem_O0 = smem_P0; 
    uint8_t* smem_O1 = smem_P1; 

    uint64_t* mbar_Q = (uint64_t*)(smem_P1 + 16384);
    uint64_t* mbar_K = mbar_Q + 1;
    uint64_t* mbar_V = mbar_K + 1;
    uint64_t* mbar_mma = mbar_V + 1;
    uint32_t* smem_tmem_base = (uint32_t*)(mbar_mma + 1);

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(mbar_K, 1);
        init_smem_barrier_fn(mbar_V, 1);
        init_smem_barrier_fn(mbar_mma, 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads(); 

    if (threadIdx.x < 32) {
        tmem_alloc_cg1_fn(smem_tmem_base, 256);
    }
    __syncthreads();
    
    uint32_t tmem_base = *smem_tmem_base;
    uint32_t tmem_QK = tmem_base;
    uint32_t tmem_O_left = tmem_base + 128;
    uint32_t tmem_O_right = tmem_base + 192;

    uint64_t smem_desc_Q0 = make_smem_desc_sm100_fn(smem_Q0, 1024, 16); 
    uint64_t smem_desc_Q1 = make_smem_desc_sm100_fn(smem_Q1, 1024, 16); 
    uint64_t smem_desc_K0 = make_smem_desc_sm100_fn(smem_K0, 1024, 16); 
    uint64_t smem_desc_K1 = make_smem_desc_sm100_fn(smem_K1, 1024, 16); 
    uint64_t smem_desc_V_left = make_smem_desc_sm100_fn(smem_V_left, 1024, 16384); 
    uint64_t smem_desc_V_right = make_smem_desc_sm100_fn(smem_V_right, 1024, 16384); 
    uint64_t smem_desc_P0 = make_smem_desc_sm100_fn(smem_P0, 1024, 16); 
    uint64_t smem_desc_P1 = make_smem_desc_sm100_fn(smem_P1, 1024, 16); 

    uint32_t idesc_QK = make_instr_desc_fn(128, 128, false, false);
    uint32_t idesc_O = make_instr_desc_fn(128, 64, false, true);

    uint32_t phase_Q = 0, phase_K = 0, phase_V = 0, phase_mma = 0;

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_Q, 32768);
        tma_load_3d_fn(&tma_Q, mbar_Q, smem_Q0, 0, m_base, bh);
        tma_load_3d_fn(&tma_Q, mbar_Q, smem_Q1, 64, m_base, bh);
    }
    mbarrier_wait_fn(mbar_Q, phase_Q);
    phase_Q ^= 1;

    float O_reg[128] = {0};
    float m_prev = -INFINITY;
    float d_prev = 0;

    int num_n_blocks = (S + 127) / 128;

    for (int n_idx = 0; n_idx < num_n_blocks; n_idx++) {
        int n_base = n_idx * 128;

        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_K, 32768);
            tma_load_3d_fn(&tma_K, mbar_K, smem_K0, 0, n_base, bh);
            tma_load_3d_fn(&tma_K, mbar_K, smem_K1, 64, n_base, bh);

            mbarrier_arrive_and_expect_tx_fn(mbar_V, 32768);
            tma_load_3d_fn(&tma_V, mbar_V, smem_V_left, 0, n_base, bh);
            tma_load_3d_fn(&tma_V, mbar_V, smem_V_right, 64, n_base, bh);
        }

        mbarrier_wait_fn(mbar_K, phase_K);
        phase_K ^= 1;
        fence_async_shared_fn();

        if (threadIdx.x == 0) {
            #pragma unroll
            for (int step = 0; step < 8; step++) {
                uint64_t desc_Q = (step < 4) ? smem_desc_Q0 : smem_desc_Q1;
                uint32_t offset_Q = (step % 4) * 32;
                desc_Q += (offset_Q >> 4);

                uint64_t desc_K = (step < 4) ? smem_desc_K0 : smem_desc_K1;
                uint32_t offset_K = (step % 4) * 32;
                desc_K += (offset_K >> 4);

                uint32_t accum = (step == 0) ? 0 : 1;
                umma_f16_cg1_fn(tmem_QK, desc_Q, desc_K, idesc_QK, accum);
            }
            tcgen05_fence_after_fn();
            umma_commit_cg1_fn(mbar_mma);
        }
        
        mbarrier_wait_fn(mbar_mma, phase_mma);
        phase_mma ^= 1;

        float m_curr = -INFINITY;
        int y = threadIdx.x;
        
        #pragma unroll
        for (int c = 0; c < 128; c += 8) {
            float S_chunk[8];
            tmem_load_8x_fn(tmem_QK + c, 
                (uint32_t*)&S_chunk[0], (uint32_t*)&S_chunk[1], (uint32_t*)&S_chunk[2], (uint32_t*)&S_chunk[3], 
                (uint32_t*)&S_chunk[4], (uint32_t*)&S_chunk[5], (uint32_t*)&S_chunk[6], (uint32_t*)&S_chunk[7]);
            tmem_load_fence_fn();
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                if (m_base + y < S && n_base + c + i < S) {
                    S_chunk[i] *= 0.088388347648f; 
                    m_curr = fmaxf(m_curr, S_chunk[i]);
                }
            }
        }
        
        float m_new = fmaxf(m_prev, m_curr);
        float d_curr = 0;

        #pragma unroll
        for (int c = 0; c < 64; c += 8) {
            float S_chunk[8];
            tmem_load_8x_fn(tmem_QK + c, 
                (uint32_t*)&S_chunk[0], (uint32_t*)&S_chunk[1], (uint32_t*)&S_chunk[2], (uint32_t*)&S_chunk[3], 
                (uint32_t*)&S_chunk[4], (uint32_t*)&S_chunk[5], (uint32_t*)&S_chunk[6], (uint32_t*)&S_chunk[7]);
            tmem_load_fence_fn();
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                if (m_base + y < S && n_base + c + i < S && m_new != -INFINITY) {
                    S_chunk[i] = fast_exp2f_fn((S_chunk[i] * 0.088388347648f - m_new) * 1.4426950408889634f); 
                } else {
                    S_chunk[i] = 0.0f;
                }
                d_curr += S_chunk[i];
            }
            
            uint32_t b01 = pack_bf16_fn(__float_as_uint(S_chunk[0]), __float_as_uint(S_chunk[1]));
            uint32_t b23 = pack_bf16_fn(__float_as_uint(S_chunk[2]), __float_as_uint(S_chunk[3]));
            uint32_t b45 = pack_bf16_fn(__float_as_uint(S_chunk[4]), __float_as_uint(S_chunk[5]));
            uint32_t b67 = pack_bf16_fn(__float_as_uint(S_chunk[6]), __float_as_uint(S_chunk[7]));
            
            int chunk = c / 8;
            int swizzled_chunk = (y % 8) ^ chunk;
            int offset = y * 128 + swizzled_chunk * 16;
            st_shared_128_fn((uint32_t)__cvta_generic_to_shared(smem_P0 + offset), b01, b23, b45, b67);
        }

        #pragma unroll
        for (int c = 64; c < 128; c += 8) {
            float S_chunk[8];
            tmem_load_8x_fn(tmem_QK + c, 
                (uint32_t*)&S_chunk[0], (uint32_t*)&S_chunk[1], (uint32_t*)&S_chunk[2], (uint32_t*)&S_chunk[3], 
                (uint32_t*)&S_chunk[4], (uint32_t*)&S_chunk[5], (uint32_t*)&S_chunk[6], (uint32_t*)&S_chunk[7]);
            tmem_load_fence_fn();
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                if (m_base + y < S && n_base + c + i < S && m_new != -INFINITY) {
                    S_chunk[i] = fast_exp2f_fn((S_chunk[i] * 0.088388347648f - m_new) * 1.4426950408889634f); 
                } else {
                    S_chunk[i] = 0.0f;
                }
                d_curr += S_chunk[i];
            }
            
            uint32_t b01 = pack_bf16_fn(__float_as_uint(S_chunk[0]), __float_as_uint(S_chunk[1]));
            uint32_t b23 = pack_bf16_fn(__float_as_uint(S_chunk[2]), __float_as_uint(S_chunk[3]));
            uint32_t b45 = pack_bf16_fn(__float_as_uint(S_chunk[4]), __float_as_uint(S_chunk[5]));
            uint32_t b67 = pack_bf16_fn(__float_as_uint(S_chunk[6]), __float_as_uint(S_chunk[7]));
            
            int chunk = (c - 64) / 8;
            int swizzled_chunk = (y % 8) ^ chunk;
            int offset = y * 128 + swizzled_chunk * 16;
            st_shared_128_fn((uint32_t)__cvta_generic_to_shared(smem_P1 + offset), b01, b23, b45, b67);
        }

        float rescale = (m_prev == -INFINITY) ? 0.0f : fast_exp2f_fn((m_prev - m_new) * 1.4426950408889634f);
        d_prev = d_prev * rescale + d_curr;
        m_prev = m_new;

        fence_async_shared_fn();
        __syncthreads(); 

        mbarrier_wait_fn(mbar_V, phase_V);
        phase_V ^= 1;
        fence_async_shared_fn();

        if (threadIdx.x == 0) {
            #pragma unroll
            for (int step = 0; step < 8; step++) {
                uint64_t desc_P = (step < 4) ? smem_desc_P0 : smem_desc_P1;
                uint32_t offset_P = (step % 4) * 32;
                desc_P += (offset_P >> 4);

                uint64_t desc_V_l = smem_desc_V_left;
                uint32_t offset_V = step * 2048;
                desc_V_l += (offset_V >> 4);

                uint32_t accum = (step == 0) ? 0 : 1;
                umma_f16_cg1_fn(tmem_O_left, desc_P, desc_V_l, idesc_O, accum);
            }

            #pragma unroll
            for (int step = 0; step < 8; step++) {
                uint64_t desc_P = (step < 4) ? smem_desc_P0 : smem_desc_P1;
                uint32_t offset_P = (step % 4) * 32;
                desc_P += (offset_P >> 4);

                uint64_t desc_V_r = smem_desc_V_right;
                uint32_t offset_V = step * 2048;
                desc_V_r += (offset_V >> 4);

                uint32_t accum = (step == 0) ? 0 : 1;
                umma_f16_cg1_fn(tmem_O_right, desc_P, desc_V_r, idesc_O, accum);
            }

            tcgen05_fence_after_fn();
            umma_commit_cg1_fn(mbar_mma);
        }

        mbarrier_wait_fn(mbar_mma, phase_mma);
        phase_mma ^= 1;

        #pragma unroll
        for (int c = 0; c < 64; c += 8) {
            float PV_chunk[8];
            tmem_load_8x_fn(tmem_O_left + c, 
                (uint32_t*)&PV_chunk[0], (uint32_t*)&PV_chunk[1], (uint32_t*)&PV_chunk[2], (uint32_t*)&PV_chunk[3], 
                (uint32_t*)&PV_chunk[4], (uint32_t*)&PV_chunk[5], (uint32_t*)&PV_chunk[6], (uint32_t*)&PV_chunk[7]);
            tmem_load_fence_fn();
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                O_reg[c + i] = O_reg[c + i] * rescale + PV_chunk[i];
            }
        }
        
        #pragma unroll
        for (int c = 0; c < 64; c += 8) {
            float PV_chunk[8];
            tmem_load_8x_fn(tmem_O_right + c, 
                (uint32_t*)&PV_chunk[0], (uint32_t*)&PV_chunk[1], (uint32_t*)&PV_chunk[2], (uint32_t*)&PV_chunk[3], 
                (uint32_t*)&PV_chunk[4], (uint32_t*)&PV_chunk[5], (uint32_t*)&PV_chunk[6], (uint32_t*)&PV_chunk[7]);
            tmem_load_fence_fn();
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                O_reg[64 + c + i] = O_reg[64 + c + i] * rescale + PV_chunk[i];
            }
        }

        __syncthreads();
    }

    float inv_d = (d_prev > 0.0f) ? (1.0f / d_prev) : 0.0f;
    int y = threadIdx.x;
    
    #pragma unroll
    for (int c = 0; c < 64; c += 8) {
        uint32_t b01 = pack_bf16_fn(__float_as_uint(O_reg[c+0] * inv_d), __float_as_uint(O_reg[c+1] * inv_d));
        uint32_t b23 = pack_bf16_fn(__float_as_uint(O_reg[c+2] * inv_d), __float_as_uint(O_reg[c+3] * inv_d));
        uint32_t b45 = pack_bf16_fn(__float_as_uint(O_reg[c+4] * inv_d), __float_as_uint(O_reg[c+5] * inv_d));
        uint32_t b67 = pack_bf16_fn(__float_as_uint(O_reg[c+6] * inv_d), __float_as_uint(O_reg[c+7] * inv_d));
        
        int chunk = c / 8;
        int swizzled_chunk = (y % 8) ^ chunk;
        int offset = y * 128 + swizzled_chunk * 16;
        st_shared_128_fn((uint32_t)__cvta_generic_to_shared(smem_O0 + offset), b01, b23, b45, b67);
    }
    
    #pragma unroll
    for (int c = 64; c < 128; c += 8) {
        uint32_t b01 = pack_bf16_fn(__float_as_uint(O_reg[c+0] * inv_d), __float_as_uint(O_reg[c+1] * inv_d));
        uint32_t b23 = pack_bf16_fn(__float_as_uint(O_reg[c+2] * inv_d), __float_as_uint(O_reg[c+3] * inv_d));
        uint32_t b45 = pack_bf16_fn(__float_as_uint(O_reg[c+4] * inv_d), __float_as_uint(O_reg[c+5] * inv_d));
        uint32_t b67 = pack_bf16_fn(__float_as_uint(O_reg[c+6] * inv_d), __float_as_uint(O_reg[c+7] * inv_d));
        
        int chunk = (c - 64) / 8;
        int swizzled_chunk = (y % 8) ^ chunk;
        int offset = y * 128 + swizzled_chunk * 16;
        st_shared_128_fn((uint32_t)__cvta_generic_to_shared(smem_O1 + offset), b01, b23, b45, b67);
    }
    
    fence_async_shared_fn();
    tma_store_fence_fn();

    __syncthreads();
    
    if (threadIdx.x == 0) {
        tma_store_3d_fn(&tma_O, smem_O0, 0, m_base, bh);
        tma_store_3d_fn(&tma_O, smem_O1, 64, m_base, bh);
        tma_store_commit_fn();
        tma_store_wait_fn<0>();
    }
    __syncthreads();

    if (m_base + y < S) {
        int lse_idx = bh * S + m_base + y;
        lse_ptr[lse_idx] = m_prev + logf(d_prev);
    }
    
    if (threadIdx.x < 32) {
        tmem_dealloc_cg1_fn(tmem_base, 256);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3); 

    CUtensorMap tma_Q, tma_K, tma_V, tma_O;
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_Q, Q.data_ptr(), D, S, B * H, 64, 128, 1));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_K, K.data_ptr(), D, S, B * H, 64, 128, 1));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_V, V.data_ptr(), D, S, B * H, 64, 128, 1));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_O, O.data_ptr(), D, S, B * H, 64, 128, 1));

    int m_blocks = (S + 127) / 128;
    dim3 grid(m_blocks, H, B);
    dim3 block(128); 

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute(cta_gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 132096));

    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 132096; 
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 1;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    CUDA_CHECK(cudaLaunchKernelEx(&config, cta_gemm_kernel, tma_Q, tma_K, tma_V, tma_O, static_cast<float*>(LSE.data_ptr()), static_cast<int>(S)));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha::run);

}  // namespace tvm_ffi_mha