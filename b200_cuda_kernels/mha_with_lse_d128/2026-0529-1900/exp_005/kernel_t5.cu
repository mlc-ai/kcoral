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

namespace mha_with_lse_d128 {

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
}

__device__ __forceinline__ void st_shared_128_fn(uint32_t addr, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                 :: "r"(addr), "r"(v0), "r"(v1), "r"(v2), "r"(v3) : "memory");
}

__device__ __forceinline__ uint32_t pack_bf16_from_float_fn(float fp32_a, float fp32_b) {
    __nv_bfloat16 a = __float2bfloat16(fp32_a);
    __nv_bfloat16 b = __float2bfloat16(fp32_b);
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};"
        : "=r"(result)
        : "h"(*reinterpret_cast<uint16_t*>(&a)),
          "h"(*reinterpret_cast<uint16_t*>(&b)));
    return result;
}

__device__ __forceinline__ float fast_exp_fn(float x) {
    float y;
    x *= 1.4426950408889634f; // 1 / ln(2)
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
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

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

CUresult create_tma_4d_descriptor_2B(CUtensorMap* d, void* globalAddress, 
    uint64_t D, uint64_t S, uint64_t H, uint64_t B,
    uint32_t smem_inner_dim, uint32_t smem_outer_dim, 
    CUtensorMapSwizzle swizzle) {
    cuuint64_t globalDim[4] = {D, S, H, B};
    cuuint64_t globalStrides[3] = {D * 2, D * S * 2, D * S * H * 2};
    cuuint32_t boxDim[4] = {smem_inner_dim, smem_outer_dim, 1, 1};
    cuuint32_t elementStrides[4] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
        globalAddress, globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

__device__ __forceinline__ void tma_load_4d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    uint64_t cache_hint = 0;
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, %4, %5, %6}], [%2], %7;"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3), "l"(cache_hint) : "memory");
}

__device__ __forceinline__ void tmem_alloc_cg1_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_cg1_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(addr), "r"(ncols));
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

__device__ __forceinline__ void tcgen05_commit_1sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(a));
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    uint32_t base_offset = (addr >> 7) & 0x7;
    d |= (uint64_t)base_offset << 49;
    d |= (uint64_t)2 << 61;
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, uint32_t a_major, uint32_t b_major) {
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

__device__ __forceinline__ void st_shared_128_swizzled(uint8_t* base, uint32_t row, uint32_t col, uint32_t p01, uint32_t p23, uint32_t p45, uint32_t p67) {
    uint32_t chunk_idx = col / 8;
    uint32_t swizzled_chunk = (row % 8) ^ chunk_idx;
    uint32_t byte_offset = row * 128 + swizzled_chunk * 16;
    st_shared_128_fn((uint32_t)__cvta_generic_to_shared(base + byte_offset), p01, p23, p45, p67);
}

extern __shared__ __align__(1024) uint8_t smem_pool[];

__global__ void __launch_bounds__(256, 1) fwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int S, int H) {
    
    setmaxnreg_inc_sync_fn<248>();

    int b_idx = blockIdx.z;
    int h_idx = blockIdx.y;
    int m = blockIdx.x;

    __nv_bfloat16* smem_Q = (__nv_bfloat16*)(smem_pool);
    __nv_bfloat16* smem_K[2];
    __nv_bfloat16* smem_V[2];
    smem_K[0] = (__nv_bfloat16*)(smem_pool + 65536);
    smem_V[0] = (__nv_bfloat16*)(smem_pool + 81920);
    smem_K[1] = (__nv_bfloat16*)(smem_pool + 98304);
    smem_V[1] = (__nv_bfloat16*)(smem_pool + 114688);
    __nv_bfloat16* smem_P = (__nv_bfloat16*)(smem_pool + 131072);
    
    uint64_t* mbar_tma_Q = (uint64_t*)(smem_pool + 163840);
    uint64_t* mbar_tma_KV = mbar_tma_Q + 1;
    uint64_t* mbar_umma_QK = mbar_tma_KV + 2;
    uint64_t* mbar_umma_PV = mbar_umma_QK + 2;
    uint32_t* smem_tmem_S = (uint32_t*)(mbar_umma_PV + 1);
    uint32_t* smem_tmem_dP = smem_tmem_S + 4;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&mbar_tma_Q[0], 1);
        init_smem_barrier_fn(&mbar_tma_KV[0], 1);
        init_smem_barrier_fn(&mbar_tma_KV[1], 1);
        init_smem_barrier_fn(&mbar_umma_QK[0], 1);
        init_smem_barrier_fn(&mbar_umma_QK[1], 1);
        init_smem_barrier_fn(&mbar_umma_PV[0], 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        tmem_alloc_cg1_fn(&smem_tmem_S[0], 64);
        tmem_alloc_cg1_fn(&smem_tmem_S[1], 64);
        tmem_alloc_cg1_fn(&smem_tmem_S[2], 64);
        tmem_alloc_cg1_fn(&smem_tmem_S[3], 64);
        tmem_alloc_cg1_fn(&smem_tmem_dP[0], 128);
        tmem_alloc_cg1_fn(&smem_tmem_dP[1], 128);
    }
    __syncthreads();
    
    uint32_t tmem_S_b0_0 = smem_tmem_S[0];
    uint32_t tmem_S_b0_1 = smem_tmem_S[1];
    uint32_t tmem_S_b1_0 = smem_tmem_S[2];
    uint32_t tmem_S_b1_1 = smem_tmem_S[3];
    uint32_t tmem_dP0 = smem_tmem_dP[0];
    uint32_t tmem_dP1 = smem_tmem_dP[1];

    uint32_t tma_phase_Q = 0;
    uint32_t tma_phase_KV[2] = {0, 0};
    uint32_t umma_phase_QK[2] = {0, 0};
    uint32_t umma_phase_PV = 0;

    if (threadIdx.x == 0) {
        uint32_t expected_tx = (m * 256 + 128 < S) ? 65536 : 32768;
        mbarrier_arrive_and_expect_tx_fn(&mbar_tma_Q[0], expected_tx);
        tma_load_4d_fn(&tma_Q, &mbar_tma_Q[0], smem_Q, 0, m * 256, h_idx, b_idx);
        tma_load_4d_fn(&tma_Q, &mbar_tma_Q[0], smem_Q + 8192, 64, m * 256, h_idx, b_idx);
        if (m * 256 + 128 < S) {
            tma_load_4d_fn(&tma_Q, &mbar_tma_Q[0], smem_Q + 16384, 0, m * 256 + 128, h_idx, b_idx);
            tma_load_4d_fn(&tma_Q, &mbar_tma_Q[0], smem_Q + 24576, 64, m * 256 + 128, h_idx, b_idx);
        }
    }
    
    if (S > 0) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(&mbar_tma_KV[0], 32768);
            tma_load_4d_fn(&tma_K, &mbar_tma_KV[0], smem_K[0], 0, 0, h_idx, b_idx);
            tma_load_4d_fn(&tma_K, &mbar_tma_KV[0], smem_K[0] + 4096, 64, 0, h_idx, b_idx);
            tma_load_4d_fn(&tma_V, &mbar_tma_KV[0], smem_V[0], 0, 0, h_idx, b_idx);
            tma_load_4d_fn(&tma_V, &mbar_tma_KV[0], smem_V[0] + 4096, 64, 0, h_idx, b_idx);
        }
    }
    if (S > 64) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(&mbar_tma_KV[1], 32768);
            tma_load_4d_fn(&tma_K, &mbar_tma_KV[1], smem_K[1], 0, 64, h_idx, b_idx);
            tma_load_4d_fn(&tma_K, &mbar_tma_KV[1], smem_K[1] + 4096, 64, 64, h_idx, b_idx);
            tma_load_4d_fn(&tma_V, &mbar_tma_KV[1], smem_V[1], 0, 64, h_idx, b_idx);
            tma_load_4d_fn(&tma_V, &mbar_tma_KV[1], smem_V[1] + 4096, 64, 64, h_idx, b_idx);
        }
    }
    
    mbarrier_wait_fn(&mbar_tma_Q[0], tma_phase_Q);
    tma_phase_Q ^= 1;
    fence_async_shared_fn();

    float rowmax = -INFINITY;
    float rowsum = 0.0f;
    float O_m[128];
    #pragma unroll
    for(int i=0; i<128; i++) O_m[i] = 0.0f;

    uint32_t idesc_QK = make_instr_desc_fn(128, 64, 0, 0);
    uint32_t idesc_PV = make_instr_desc_fn(128, 64, 0, 1);

    int q_idx = threadIdx.x;
    int q_half = q_idx / 128;
    int q_row = q_idx % 128;
    bool q_valid = (m * 256 + q_idx < S);

    if (S > 0) {
        mbarrier_wait_fn(&mbar_tma_KV[0], tma_phase_KV[0]);
        tma_phase_KV[0] ^= 1;
        fence_async_shared_fn();
        
        tcgen05_fence_after_fn();
        #pragma unroll
        for (int k = 0; k < 128; k += 16) {
            uint32_t panel = k / 64;
            uint32_t k_in = k % 64;
            uint32_t offset_Q = panel * 16384 + k_in * 2;
            uint32_t offset_K = panel * 8192 + k_in * 2;
            uint64_t desc_Q0 = make_smem_desc_sm100_fn((uint8_t*)smem_Q + offset_Q, 1, 1024);
            uint64_t desc_Q1 = make_smem_desc_sm100_fn((uint8_t*)smem_Q + 32768 + offset_Q, 1, 1024);
            uint64_t desc_K = make_smem_desc_sm100_fn((uint8_t*)smem_K[0] + offset_K, 1, 1024);
            uint32_t accum = (k == 0) ? 0 : 1;
            if (threadIdx.x == 0) {
                umma_f16_cg1_fn(tmem_S_b0_0, desc_Q0, desc_K, idesc_QK, accum);
                if (m * 256 + 128 < S) umma_f16_cg1_fn(tmem_S_b0_1, desc_Q1, desc_K, idesc_QK, accum);
            }
        }
        if (threadIdx.x == 0) tcgen05_commit_1sm_fn(&mbar_umma_QK[0]);
        mbarrier_wait_fn(&mbar_umma_QK[0], umma_phase_QK[0]);
        umma_phase_QK[0] ^= 1;
    }

    for (int n = 0; n < S; n += 64) {
        int stage = (n / 64) % 2;
        int next_stage = (stage + 1) % 2;
        int n_next = n + 64;
        
        uint32_t tmem_s_ptr = (q_half == 0) ? ((stage == 0) ? tmem_S_b0_0 : tmem_S_b1_0) : ((stage == 0) ? tmem_S_b0_1 : tmem_S_b1_1);
        uint32_t tmem_s_next_ptr0 = (next_stage == 0) ? tmem_S_b0_0 : tmem_S_b1_0;
        uint32_t tmem_s_next_ptr1 = (next_stage == 0) ? tmem_S_b0_1 : tmem_S_b1_1;
        
        if (n_next < S) {
            mbarrier_wait_fn(&mbar_tma_KV[next_stage], tma_phase_KV[next_stage]);
            tma_phase_KV[next_stage] ^= 1;
            fence_async_shared_fn();
            
            tcgen05_fence_after_fn();
            #pragma unroll
            for (int k = 0; k < 128; k += 16) {
                uint32_t panel = k / 64;
                uint32_t k_in = k % 64;
                uint32_t offset_Q = panel * 16384 + k_in * 2;
                uint32_t offset_K = panel * 8192 + k_in * 2;
                uint64_t desc_Q0 = make_smem_desc_sm100_fn((uint8_t*)smem_Q + offset_Q, 1, 1024);
                uint64_t desc_Q1 = make_smem_desc_sm100_fn((uint8_t*)smem_Q + 32768 + offset_Q, 1, 1024);
                uint64_t desc_K = make_smem_desc_sm100_fn((uint8_t*)smem_K[next_stage] + offset_K, 1, 1024);
                uint32_t accum = (k == 0) ? 0 : 1;
                if (threadIdx.x == 0) {
                    umma_f16_cg1_fn(tmem_s_next_ptr0, desc_Q0, desc_K, idesc_QK, accum);
                    if (m * 256 + 128 < S) umma_f16_cg1_fn(tmem_s_next_ptr1, desc_Q1, desc_K, idesc_QK, accum);
                }
            }
            if (threadIdx.x == 0) tcgen05_commit_1sm_fn(&mbar_umma_QK[next_stage]);
        }

        float s_max = -INFINITY;
        if (q_valid) {
            uint32_t r[64];
            tmem_load_8x_fn(tmem_s_ptr + 0,  &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
            tmem_load_8x_fn(tmem_s_ptr + 8,  &r[8], &r[9], &r[10], &r[11], &r[12], &r[13], &r[14], &r[15]);
            tmem_load_8x_fn(tmem_s_ptr + 16, &r[16], &r[17], &r[18], &r[19], &r[20], &r[21], &r[22], &r[23]);
            tmem_load_8x_fn(tmem_s_ptr + 24, &r[24], &r[25], &r[26], &r[27], &r[28], &r[29], &r[30], &r[31]);
            tmem_load_8x_fn(tmem_s_ptr + 32, &r[32], &r[33], &r[34], &r[35], &r[36], &r[37], &r[38], &r[39]);
            tmem_load_8x_fn(tmem_s_ptr + 40, &r[40], &r[41], &r[42], &r[43], &r[44], &r[45], &r[46], &r[47]);
            tmem_load_8x_fn(tmem_s_ptr + 48, &r[48], &r[49], &r[50], &r[51], &r[52], &r[53], &r[54], &r[55]);
            tmem_load_8x_fn(tmem_s_ptr + 56, &r[56], &r[57], &r[58], &r[59], &r[60], &r[61], &r[62], &r[63]);
            tmem_load_fence_fn();
            
            #pragma unroll
            for (int i=0; i<64; i++) {
                if (n + i < S) {
                    s_max = max(s_max, __uint_as_float(r[i]) * 0.0883883476f);
                }
            }
            
            float m_prev = rowmax;
            float m_new = max(m_prev, s_max);
            
            if (m_prev != m_new) {
                float rescale = fast_exp_fn(m_prev - m_new);
                rowsum *= rescale;
                #pragma unroll
                for(int i=0; i<128; i++) O_m[i] *= rescale;
            }
            rowmax = m_new;
            
            #pragma unroll
            for (int i=0; i<64; i+=8) {
                float p0 = (n + i + 0 < S) ? fast_exp_fn(__uint_as_float(r[i+0]) * 0.0883883476f - m_new) : 0.0f;
                float p1 = (n + i + 1 < S) ? fast_exp_fn(__uint_as_float(r[i+1]) * 0.0883883476f - m_new) : 0.0f;
                float p2 = (n + i + 2 < S) ? fast_exp_fn(__uint_as_float(r[i+2]) * 0.0883883476f - m_new) : 0.0f;
                float p3 = (n + i + 3 < S) ? fast_exp_fn(__uint_as_float(r[i+3]) * 0.0883883476f - m_new) : 0.0f;
                float p4 = (n + i + 4 < S) ? fast_exp_fn(__uint_as_float(r[i+4]) * 0.0883883476f - m_new) : 0.0f;
                float p5 = (n + i + 5 < S) ? fast_exp_fn(__uint_as_float(r[i+5]) * 0.0883883476f - m_new) : 0.0f;
                float p6 = (n + i + 6 < S) ? fast_exp_fn(__uint_as_float(r[i+6]) * 0.0883883476f - m_new) : 0.0f;
                float p7 = (n + i + 7 < S) ? fast_exp_fn(__uint_as_float(r[i+7]) * 0.0883883476f - m_new) : 0.0f;
                rowsum += p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7;
                
                uint32_t p01 = pack_bf16_from_float_fn(p0, p1);
                uint32_t p23 = pack_bf16_from_float_fn(p2, p3);
                uint32_t p45 = pack_bf16_from_float_fn(p4, p5);
                uint32_t p67 = pack_bf16_from_float_fn(p6, p7);
                
                st_shared_128_swizzled((uint8_t*)smem_P + q_half * 16384, q_row, i, p01, p23, p45, p67);
            }
        }
        
        fence_async_shared_fn();
        __syncthreads(); 
        
        tcgen05_fence_after_fn();
        #pragma unroll
        for (int k = 0; k < 64; k += 16) {
            uint32_t offset_P = k * 2;
            uint32_t offset_V_left = (k / 8) * 1024;
            uint32_t offset_V_right = 8192 + (k / 8) * 1024;
            uint64_t desc_P0 = make_smem_desc_sm100_fn((uint8_t*)smem_P + offset_P, 1, 1024);
            uint64_t desc_P1 = make_smem_desc_sm100_fn((uint8_t*)smem_P + 16384 + offset_P, 1, 1024);
            uint64_t desc_V_left = make_smem_desc_sm100_fn((uint8_t*)smem_V[stage] + offset_V_left, 8192, 1024);
            uint64_t desc_V_right = make_smem_desc_sm100_fn((uint8_t*)smem_V[stage] + offset_V_right, 8192, 1024);
            
            uint32_t accum = (k == 0) ? 0 : 1;
            if (threadIdx.x == 0) {
                umma_f16_cg1_fn(tmem_dP0, desc_P0, desc_V_left, idesc_PV, accum);
                if (m * 256 + 128 < S) umma_f16_cg1_fn(tmem_dP1, desc_P1, desc_V_left, idesc_PV, accum);
                
                umma_f16_cg1_fn(tmem_dP0 + 64, desc_P0, desc_V_right, idesc_PV, accum);
                if (m * 256 + 128 < S) umma_f16_cg1_fn(tmem_dP1 + 64, desc_P1, desc_V_right, idesc_PV, accum);
            }
        }
        if (threadIdx.x == 0) tcgen05_commit_1sm_fn(&mbar_umma_PV[0]);
        mbarrier_wait_fn(&mbar_umma_PV[0], umma_phase_PV);
        umma_phase_PV ^= 1;
        
        int n_next2 = n + 128;
        if (n_next2 < S) {
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(&mbar_tma_KV[stage], 32768);
                tma_load_4d_fn(&tma_K, &mbar_tma_KV[stage], smem_K[stage], 0, n_next2, h_idx, b_idx);
                tma_load_4d_fn(&tma_K, &mbar_tma_KV[stage], smem_K[stage] + 4096, 64, n_next2, h_idx, b_idx);
                tma_load_4d_fn(&tma_V, &mbar_tma_KV[stage], smem_V[stage], 0, n_next2, h_idx, b_idx);
                tma_load_4d_fn(&tma_V, &mbar_tma_KV[stage], smem_V[stage] + 4096, 64, n_next2, h_idx, b_idx);
            }
        }
        
        if (q_valid) {
            uint32_t tmem_dp_ptr = (q_half == 0) ? tmem_dP0 : tmem_dP1;
            for (int c = 0; c < 128; c += 64) {
                uint32_t r[64];
                tmem_load_8x_fn(tmem_dp_ptr + c + 0,  &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
                tmem_load_8x_fn(tmem_dp_ptr + c + 8,  &r[8], &r[9], &r[10], &r[11], &r[12], &r[13], &r[14], &r[15]);
                tmem_load_8x_fn(tmem_dp_ptr + c + 16, &r[16], &r[17], &r[18], &r[19], &r[20], &r[21], &r[22], &r[23]);
                tmem_load_8x_fn(tmem_dp_ptr + c + 24, &r[24], &r[25], &r[26], &r[27], &r[28], &r[29], &r[30], &r[31]);
                tmem_load_8x_fn(tmem_dp_ptr + c + 32, &r[32], &r[33], &r[34], &r[35], &r[36], &r[37], &r[38], &r[39]);
                tmem_load_8x_fn(tmem_dp_ptr + c + 40, &r[40], &r[41], &r[42], &r[43], &r[44], &r[45], &r[46], &r[47]);
                tmem_load_8x_fn(tmem_dp_ptr + c + 48, &r[48], &r[49], &r[50], &r[51], &r[52], &r[53], &r[54], &r[55]);
                tmem_load_8x_fn(tmem_dp_ptr + c + 56, &r[56], &r[57], &r[58], &r[59], &r[60], &r[61], &r[62], &r[63]);
                tmem_load_fence_fn();
                
                #pragma unroll
                for(int i=0; i<64; i++) {
                    O_m[c + i] += __uint_as_float(r[i]);
                }
            }
        }
        
        if (n_next < S) {
            mbarrier_wait_fn(&mbar_umma_QK[next_stage], umma_phase_QK[next_stage]);
            umma_phase_QK[next_stage] ^= 1;
        }
    }

    if (q_valid) {
        float lse = rowmax + logf(rowsum);
        #pragma unroll
        for (int i=0; i<128; i++) {
            O_m[i] /= rowsum;
        }
        
        uint64_t LSE_offset = (uint64_t)b_idx * H * S + (uint64_t)h_idx * S + m * 256 + q_idx;
        LSE[LSE_offset] = lse;
    }

    __syncthreads();
    
    __nv_bfloat16* smem_O = smem_Q; 
    if (q_valid) {
        for (int c = 0; c < 128; c += 8) {
            uint32_t p01 = pack_bf16_from_float_fn(O_m[c+0], O_m[c+1]);
            uint32_t p23 = pack_bf16_from_float_fn(O_m[c+2], O_m[c+3]);
            uint32_t p45 = pack_bf16_from_float_fn(O_m[c+4], O_m[c+5]);
            uint32_t p67 = pack_bf16_from_float_fn(O_m[c+6], O_m[c+7]);
            uint4 vec; vec.x = p01; vec.y = p23; vec.z = p45; vec.w = p67;
            *reinterpret_cast<uint4*>(&smem_O[q_idx * 128 + c]) = vec;
        }
    }
    
    __syncthreads();
    
    uint64_t base_O_offset = ((uint64_t)b_idx * H * S + (uint64_t)h_idx * S + m * 256) * 128;
    for (int i = threadIdx.x; i < 256 * 16; i += 256) {
        int r = i / 16;
        int c_uint4 = i % 16;
        if (m * 256 + r < S) {
            uint4 vec = *reinterpret_cast<uint4*>(&smem_O[r * 128 + c_uint4 * 8]);
            *reinterpret_cast<uint4*>(O + base_O_offset + r * 128 + c_uint4 * 8) = vec;
        }
    }

    if (threadIdx.x < 32) {
        tmem_dealloc_cg1_fn(tmem_S_b0_0, 64);
        tmem_dealloc_cg1_fn(tmem_S_b0_1, 64);
        tmem_dealloc_cg1_fn(tmem_S_b1_0, 64);
        tmem_dealloc_cg1_fn(tmem_S_b1_1, 64);
        tmem_dealloc_cg1_fn(tmem_dP0, 128);
        tmem_dealloc_cg1_fn(tmem_dP1, 128);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);

    CUtensorMap tma_Q, tma_K, tma_V;
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_Q, Q.data_ptr(), D, S, H, B, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B));
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_K, K.data_ptr(), D, S, H, B, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B));
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_V, V.data_ptr(), D, S, H, B, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B));

    int m_blocks = (S + 255) / 256;
    dim3 grid(m_blocks, H, B);
    dim3 block(256);
    
    int smem_size = 164000; 
    CUDA_CHECK(cudaFuncSetAttribute((void*)fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    fwd_kernel<<<grid, block, smem_size, stream>>>(tma_Q, tma_K, tma_V, 
        static_cast<__nv_bfloat16*>(O.data_ptr()), 
        static_cast<float*>(LSE.data_ptr()), 
        S, H);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

} // namespace mha_with_lse_d128

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_with_lse_d128::run);