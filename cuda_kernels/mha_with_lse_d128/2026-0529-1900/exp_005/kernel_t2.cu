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

__global__ void __launch_bounds__(128) fwd_kernel(
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
    smem_K[0] = (__nv_bfloat16*)(smem_pool + 32768);
    smem_V[0] = (__nv_bfloat16*)(smem_pool + 65536);
    smem_K[1] = (__nv_bfloat16*)(smem_pool + 98304);
    smem_V[1] = (__nv_bfloat16*)(smem_pool + 131072);
    __nv_bfloat16* smem_P = (__nv_bfloat16*)(smem_pool + 163840);
    
    uint64_t* mbar_tma_Q = (uint64_t*)(smem_pool + 196608);
    uint64_t* mbar_tma_KV = mbar_tma_Q + 1;
    uint64_t* mbar_umma = mbar_tma_KV + 2;
    uint32_t* smem_tmem_S = (uint32_t*)(mbar_umma + 1);
    uint32_t* smem_tmem_dP = smem_tmem_S + 1;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&mbar_tma_Q[0], 1);
        init_smem_barrier_fn(&mbar_tma_KV[0], 1);
        init_smem_barrier_fn(&mbar_tma_KV[1], 1);
        init_smem_barrier_fn(&mbar_umma[0], 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        tmem_alloc_cg1_fn(smem_tmem_S, 128);
        tmem_alloc_cg1_fn(smem_tmem_dP, 128);
    }
    __syncthreads();
    
    uint32_t tmem_S = *smem_tmem_S;
    uint32_t tmem_dP = *smem_tmem_dP;

    uint32_t tma_phase_Q = 0;
    uint32_t tma_phase_KV[2] = {0, 0};
    uint32_t umma_phase = 0;

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(&mbar_tma_Q[0], 32768);
        tma_load_4d_fn(&tma_Q, &mbar_tma_Q[0], smem_Q, 0, m * 128, h_idx, b_idx);
        tma_load_4d_fn(&tma_Q, &mbar_tma_Q[0], smem_Q + 8192, 64, m * 128, h_idx, b_idx);
    }
    
    if (S > 0) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(&mbar_tma_KV[0], 65536);
            tma_load_4d_fn(&tma_K, &mbar_tma_KV[0], smem_K[0], 0, 0, h_idx, b_idx);
            tma_load_4d_fn(&tma_K, &mbar_tma_KV[0], smem_K[0] + 8192, 64, 0, h_idx, b_idx);
            tma_load_4d_fn(&tma_V, &mbar_tma_KV[0], smem_V[0], 0, 0, h_idx, b_idx);
            tma_load_4d_fn(&tma_V, &mbar_tma_KV[0], smem_V[0] + 8192, 64, 0, h_idx, b_idx);
        }
    }
    
    mbarrier_wait_fn(&mbar_tma_Q[0], tma_phase_Q);
    tma_phase_Q ^= 1;

    float rowmax = -INFINITY;
    float rowsum = 0.0f;
    float O_m[128];
    for(int i=0; i<128; i++) O_m[i] = 0.0f;

    uint32_t idesc_QK = make_instr_desc_fn(128, 128, 0, 0);
    uint32_t idesc_PV = make_instr_desc_fn(128, 128, 0, 1);

    for (int n = 0; n < S; n += 128) {
        int stage = (n / 128) % 2;
        int next_stage = (stage + 1) % 2;
        
        mbarrier_wait_fn(&mbar_tma_KV[stage], tma_phase_KV[stage]);
        tma_phase_KV[stage] ^= 1;
        
        if (n + 128 < S) {
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(&mbar_tma_KV[next_stage], 65536);
                tma_load_4d_fn(&tma_K, &mbar_tma_KV[next_stage], smem_K[next_stage], 0, n + 128, h_idx, b_idx);
                tma_load_4d_fn(&tma_K, &mbar_tma_KV[next_stage], smem_K[next_stage] + 8192, 64, n + 128, h_idx, b_idx);
                tma_load_4d_fn(&tma_V, &mbar_tma_KV[next_stage], smem_V[next_stage], 0, n + 128, h_idx, b_idx);
                tma_load_4d_fn(&tma_V, &mbar_tma_KV[next_stage], smem_V[next_stage] + 8192, 64, n + 128, h_idx, b_idx);
            }
        }

        tcgen05_fence_after_fn();
        for (int k = 0; k < 128; k += 16) {
            uint32_t panel = k / 64;
            uint32_t k_in = k % 64;
            uint32_t offset = panel * 16384 + k_in * 2;
            uint64_t desc_Q = make_smem_desc_sm100_fn((uint8_t*)smem_Q + offset, 16, 1024);
            uint64_t desc_K = make_smem_desc_sm100_fn((uint8_t*)smem_K[stage] + offset, 16, 1024);
            uint32_t accum = (k == 0) ? 0 : 1;
            if (threadIdx.x == 0) umma_f16_cg1_fn(tmem_S, desc_Q, desc_K, idesc_QK, accum);
        }
        if (threadIdx.x == 0) tcgen05_commit_1sm_fn(&mbar_umma[0]);
        mbarrier_wait_fn(&mbar_umma[0], umma_phase);
        umma_phase ^= 1;

        float s_max = -INFINITY;
        for (int c = 0; c < 128; c += 16) {
            uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
            uint32_t r8, r9, r10, r11, r12, r13, r14, r15;
            tmem_load_8x_fn(tmem_S + c, &r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7);
            tmem_load_8x_fn(tmem_S + c + 8, &r8, &r9, &r10, &r11, &r12, &r13, &r14, &r15);
            tmem_load_fence_fn();
            
            if (n + c + 0 < S) s_max = max(s_max, __uint_as_float(r0) * 0.0883883476f);
            if (n + c + 1 < S) s_max = max(s_max, __uint_as_float(r1) * 0.0883883476f);
            if (n + c + 2 < S) s_max = max(s_max, __uint_as_float(r2) * 0.0883883476f);
            if (n + c + 3 < S) s_max = max(s_max, __uint_as_float(r3) * 0.0883883476f);
            if (n + c + 4 < S) s_max = max(s_max, __uint_as_float(r4) * 0.0883883476f);
            if (n + c + 5 < S) s_max = max(s_max, __uint_as_float(r5) * 0.0883883476f);
            if (n + c + 6 < S) s_max = max(s_max, __uint_as_float(r6) * 0.0883883476f);
            if (n + c + 7 < S) s_max = max(s_max, __uint_as_float(r7) * 0.0883883476f);
            
            if (n + c + 8 < S) s_max = max(s_max, __uint_as_float(r8) * 0.0883883476f);
            if (n + c + 9 < S) s_max = max(s_max, __uint_as_float(r9) * 0.0883883476f);
            if (n + c + 10 < S) s_max = max(s_max, __uint_as_float(r10) * 0.0883883476f);
            if (n + c + 11 < S) s_max = max(s_max, __uint_as_float(r11) * 0.0883883476f);
            if (n + c + 12 < S) s_max = max(s_max, __uint_as_float(r12) * 0.0883883476f);
            if (n + c + 13 < S) s_max = max(s_max, __uint_as_float(r13) * 0.0883883476f);
            if (n + c + 14 < S) s_max = max(s_max, __uint_as_float(r14) * 0.0883883476f);
            if (n + c + 15 < S) s_max = max(s_max, __uint_as_float(r15) * 0.0883883476f);
        }
        
        float m_prev = rowmax;
        float m_new = max(m_prev, s_max);
        float rescale = fast_exp_fn(m_prev - m_new);
        rowmax = m_new;
        rowsum *= rescale;
        for(int i=0; i<128; i++) O_m[i] *= rescale;
        
        for (int k = 0; k < 128; k += 16) {
            uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
            uint32_t r8, r9, r10, r11, r12, r13, r14, r15;
            tmem_load_8x_fn(tmem_S + k, &r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7);
            tmem_load_8x_fn(tmem_S + k + 8, &r8, &r9, &r10, &r11, &r12, &r13, &r14, &r15);
            tmem_load_fence_fn();
            
            float p_vals[16];
            uint32_t r_arr[16] = {r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15};
            for(int i=0; i<16; i++) {
                if (n + k + i < S) {
                    float val = __uint_as_float(r_arr[i]) * 0.0883883476f;
                    float p = fast_exp_fn(val - m_new);
                    rowsum += p;
                    p_vals[i] = p;
                } else {
                    p_vals[i] = 0.0f;
                }
            }
            
            uint32_t p01 = pack_bf16_from_float_fn(p_vals[0], p_vals[1]);
            uint32_t p23 = pack_bf16_from_float_fn(p_vals[2], p_vals[3]);
            uint32_t p45 = pack_bf16_from_float_fn(p_vals[4], p_vals[5]);
            uint32_t p67 = pack_bf16_from_float_fn(p_vals[6], p_vals[7]);
            
            if (k < 64) {
                st_shared_128_swizzled((uint8_t*)smem_P, threadIdx.x, k, p01, p23, p45, p67);
            } else {
                st_shared_128_swizzled((uint8_t*)smem_P + 16384, threadIdx.x, k - 64, p01, p23, p45, p67);
            }
            
            uint32_t p89 = pack_bf16_from_float_fn(p_vals[8], p_vals[9]);
            uint32_t p1011 = pack_bf16_from_float_fn(p_vals[10], p_vals[11]);
            uint32_t p1213 = pack_bf16_from_float_fn(p_vals[12], p_vals[13]);
            uint32_t p1415 = pack_bf16_from_float_fn(p_vals[14], p_vals[15]);
            
            if (k + 8 < 64) {
                st_shared_128_swizzled((uint8_t*)smem_P, threadIdx.x, k + 8, p89, p1011, p1213, p1415);
            } else {
                st_shared_128_swizzled((uint8_t*)smem_P + 16384, threadIdx.x, k + 8 - 64, p89, p1011, p1213, p1415);
            }
            
            fence_async_shared_fn();
            __syncthreads();
            
            tcgen05_fence_after_fn();
            uint32_t panel_P = k / 64;
            uint32_t k_in_P = k % 64;
            uint32_t offset_P = panel_P * 16384 + k_in_P * 2;
            uint32_t offset_V = (k / 8) * 1024;
            uint64_t desc_P = make_smem_desc_sm100_fn((uint8_t*)smem_P + offset_P, 16, 1024);
            uint64_t desc_V = make_smem_desc_sm100_fn((uint8_t*)smem_V[stage] + offset_V, 16384, 1024);
            uint32_t accum = (k == 0) ? 0 : 1;
            if (threadIdx.x == 0) umma_f16_cg1_fn(tmem_dP, desc_P, desc_V, idesc_PV, accum);
        }
        if (threadIdx.x == 0) tcgen05_commit_1sm_fn(&mbar_umma[0]);
        mbarrier_wait_fn(&mbar_umma[0], umma_phase);
        umma_phase ^= 1;
        
        for (int c = 0; c < 128; c += 16) {
            uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
            uint32_t r8, r9, r10, r11, r12, r13, r14, r15;
            tmem_load_8x_fn(tmem_dP + c, &r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7);
            tmem_load_8x_fn(tmem_dP + c + 8, &r8, &r9, &r10, &r11, &r12, &r13, &r14, &r15);
            tmem_load_fence_fn();
            
            O_m[c + 0] += __uint_as_float(r0);
            O_m[c + 1] += __uint_as_float(r1);
            O_m[c + 2] += __uint_as_float(r2);
            O_m[c + 3] += __uint_as_float(r3);
            O_m[c + 4] += __uint_as_float(r4);
            O_m[c + 5] += __uint_as_float(r5);
            O_m[c + 6] += __uint_as_float(r6);
            O_m[c + 7] += __uint_as_float(r7);
            
            O_m[c + 8] += __uint_as_float(r8);
            O_m[c + 9] += __uint_as_float(r9);
            O_m[c + 10] += __uint_as_float(r10);
            O_m[c + 11] += __uint_as_float(r11);
            O_m[c + 12] += __uint_as_float(r12);
            O_m[c + 13] += __uint_as_float(r13);
            O_m[c + 14] += __uint_as_float(r14);
            O_m[c + 15] += __uint_as_float(r15);
        }
    }

    float lse = rowmax + logf(rowsum);
    for (int i=0; i<128; i++) {
        O_m[i] /= rowsum;
    }

    __syncthreads();
    
    __nv_bfloat16* smem_O = smem_P; 
    for (int c = 0; c < 128; c += 8) {
        uint32_t p01 = pack_bf16_from_float_fn(O_m[c+0], O_m[c+1]);
        uint32_t p23 = pack_bf16_from_float_fn(O_m[c+2], O_m[c+3]);
        uint32_t p45 = pack_bf16_from_float_fn(O_m[c+4], O_m[c+5]);
        uint32_t p67 = pack_bf16_from_float_fn(O_m[c+6], O_m[c+7]);
        uint4 vec; vec.x = p01; vec.y = p23; vec.z = p45; vec.w = p67;
        *reinterpret_cast<uint4*>(&smem_O[threadIdx.x * 128 + c]) = vec;
    }
    
    __syncthreads();
    
    uint64_t base_O_offset = ((uint64_t)b_idx * H * S + (uint64_t)h_idx * S + m * 128) * 128;
    for (int i = threadIdx.x; i < 2048; i += 128) {
        int r = i / 16;
        int c_uint4 = i % 16;
        if (m * 128 + r < S) {
            uint4 vec = *reinterpret_cast<uint4*>(&smem_O[i * 8]);
            *reinterpret_cast<uint4*>(O + base_O_offset + i * 8) = vec;
        }
    }

    uint32_t row_idx = m * 128 + threadIdx.x;
    if (row_idx < S) {
        uint64_t LSE_offset = (uint64_t)b_idx * H * S + (uint64_t)h_idx * S + row_idx;
        LSE[LSE_offset] = lse;
    }

    if (threadIdx.x < 32) {
        tmem_dealloc_cg1_fn(tmem_S, 128);
        tmem_dealloc_cg1_fn(tmem_dP, 128);
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
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_K, K.data_ptr(), D, S, H, B, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B));
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_V, V.data_ptr(), D, S, H, B, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B));

    int m_blocks = (S + 127) / 128;
    dim3 grid(m_blocks, H, B);
    dim3 block(128);
    
    int smem_size = 197140; 
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