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

namespace tvm_ffi_example_cuda {

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

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
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

__device__ __forceinline__ void st_smem_swizzle_128b(__nv_bfloat16* smem, int y, int c, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
    int swizzled_c = (y % 8) ^ c;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem) + (y * 128) + (swizzled_c * 16);
    st_shared_128_fn(addr, v0, v1, v2, v3);
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
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(phase) : "memory");
}

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_after_thread_sync_fn() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tmem_alloc_cg1_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols) : "memory");
}

__device__ __forceinline__ void tmem_dealloc_cg1_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
   :: "r"(addr), "r"(ncols) : "memory");
}

__device__ __forceinline__ void tmem_load_8x_fn(uint32_t col,
    uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3,
    uint32_t* r4, uint32_t* r5, uint32_t* r6, uint32_t* r7) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3),
     "=r"(*r4),"=r"(*r5),"=r"(*r6),"=r"(*r7) : "r"(col) : "memory");
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
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
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
        :: "r"(a) : "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   // SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_cg1_fn(uint32_t M, uint32_t N, int a_major, int b_major) {
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

__device__ __forceinline__ uint64_t advance_desc(uint64_t desc, uint32_t bytes) {
    uint32_t addr = (uint32_t)(desc & 0x3FFF);
    addr = (addr + (bytes >> 4)) & 0x3FFF;
    return (desc & ~0x3FFFull) | addr;
}

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapDataType dataType, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d,
        dataType,
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

extern __shared__ __align__(128) uint8_t smem_pool[];

__global__ __launch_bounds__(128, 1) void mha_d128_causal_kernel(
    const __grid_constant__ CUtensorMap tma_q,
    const __grid_constant__ CUtensorMap tma_k,
    const __grid_constant__ CUtensorMap tma_v,
    void* __restrict__ O,
    void* __restrict__ LSE,
    int B, int H, int S, float scale) 
{
    setmaxnreg_inc_sync_fn<256>();

    __nv_bfloat16* smem_q0 = (__nv_bfloat16*)smem_pool;
    __nv_bfloat16* smem_q1 = smem_q0 + 128 * 64;
    __nv_bfloat16* smem_k0 = smem_q1 + 128 * 64;
    __nv_bfloat16* smem_k1 = smem_k0 + 128 * 64;
    __nv_bfloat16* smem_v0 = smem_k1 + 128 * 64;
    __nv_bfloat16* smem_v1 = smem_v0 + 128 * 64;
    __nv_bfloat16* smem_p0 = smem_v1 + 128 * 64;
    __nv_bfloat16* smem_p1 = smem_p0 + 128 * 64;

    uint64_t* mbar_q = (uint64_t*)(smem_p1 + 128 * 64);
    uint64_t* mbar_kv = mbar_q + 1;
    uint64_t* mbar_umma1 = mbar_kv + 1;
    uint64_t* mbar_umma2 = mbar_umma1 + 1;

    __shared__ uint32_t tmem_qkt;
    __shared__ uint32_t tmem_o0;
    __shared__ uint32_t tmem_o1;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_q, 1);
        init_smem_barrier_fn(mbar_kv, 1);
        init_smem_barrier_fn(mbar_umma1, 1);
        init_smem_barrier_fn(mbar_umma2, 1);
    }
    
    if (threadIdx.x < 32) {
        tmem_alloc_cg1_fn(&tmem_qkt, 128);
        tmem_alloc_cg1_fn(&tmem_o0, 64);
        tmem_alloc_cg1_fn(&tmem_o1, 64);
    }
    
    fence_smem_barrier_init_fn();
    __syncthreads();

    uint32_t base_qkt = tmem_qkt;
    uint32_t base_o0 = tmem_o0;
    uint32_t base_o1 = tmem_o1;

    int bh_idx = blockIdx.x;
    int q_step = blockIdx.y;

    int global_q_start = bh_idx * S + q_step * 128;
    if (threadIdx.x == 0) {
        tma_load_2d_fn(&tma_q, mbar_q, smem_q0, 0, global_q_start);
        tma_load_2d_fn(&tma_q, mbar_q, smem_q1, 64, global_q_start);
        mbarrier_arrive_and_expect_tx_fn(mbar_q, 32768);
    }
    mbarrier_wait_fn(mbar_q, 0);
    __syncthreads();

    uint32_t idesc1 = make_instr_desc_cg1_fn(128, 128, 0, 0);
    uint32_t idesc_o0 = make_instr_desc_cg1_fn(128, 64, 0, 1);

    uint64_t desc_q0 = make_smem_desc_sm100_fn(smem_q0, 1, 1024);
    uint64_t desc_q1 = make_smem_desc_sm100_fn(smem_q1, 1, 1024);
    uint64_t desc_k0 = make_smem_desc_sm100_fn(smem_k0, 1, 1024);
    uint64_t desc_k1 = make_smem_desc_sm100_fn(smem_k1, 1, 1024);

    uint64_t desc_v0 = make_smem_desc_sm100_fn(smem_v0, 16384, 1024);
    uint64_t desc_v1 = make_smem_desc_sm100_fn(smem_v1, 16384, 1024);

    uint64_t desc_p0 = make_smem_desc_sm100_fn(smem_p0, 1, 1024);
    uint64_t desc_p1 = make_smem_desc_sm100_fn(smem_p1, 1, 1024);

    float O_total[128];
    for (int i = 0; i < 128; i++) O_total[i] = 0.0f;
    float m_prev = __int_as_float(0xff800000);
    float l_prev = 0.0f;

    uint32_t phase_kv = 0;
    uint32_t phase_umma1 = 0;
    uint32_t phase_umma2 = 0;

    int num_k_steps = min(q_step + 1, (S + 127) / 128);

    for (int k_step = 0; k_step < num_k_steps; ++k_step) {
        int global_k_start = bh_idx * S + k_step * 128;
        if (threadIdx.x == 0) {
            tma_load_2d_fn(&tma_k, mbar_kv, smem_k0, 0, global_k_start);
            tma_load_2d_fn(&tma_k, mbar_kv, smem_k1, 64, global_k_start);
            tma_load_2d_fn(&tma_v, mbar_kv, smem_v0, 0, global_k_start);
            tma_load_2d_fn(&tma_v, mbar_kv, smem_v1, 64, global_k_start);
            mbarrier_arrive_and_expect_tx_fn(mbar_kv, 65536);
        }
        mbarrier_wait_fn(mbar_kv, phase_kv);
        phase_kv ^= 1;
        __syncthreads();

        if (threadIdx.x == 0) {
            tcgen05_fence_after_thread_sync_fn();
            uint64_t cur_desc_q0 = desc_q0;
            uint64_t cur_desc_k0 = desc_k0;
            uint64_t cur_desc_q1 = desc_q1;
            uint64_t cur_desc_k1 = desc_k1;
            for (int k = 0; k < 8; ++k) {
                uint32_t accum = (k == 0) ? 0 : 1;
                if (k < 4) {
                    umma_f16_cg1_fn(base_qkt, cur_desc_q0, cur_desc_k0, idesc1, accum);
                    cur_desc_q0 = advance_desc(cur_desc_q0, 32);
                    cur_desc_k0 = advance_desc(cur_desc_k0, 32);
                } else {
                    umma_f16_cg1_fn(base_qkt, cur_desc_q1, cur_desc_k1, idesc1, accum);
                    cur_desc_q1 = advance_desc(cur_desc_q1, 32);
                    cur_desc_k1 = advance_desc(cur_desc_k1, 32);
                }
            }
            umma_commit_cg1_fn(mbar_umma1);
        }
        mbarrier_wait_fn(mbar_umma1, phase_umma1);
        phase_umma1 ^= 1;
        __syncthreads();

        float row_max = __int_as_float(0xff800000);
        for (int col = 0; col < 128; col += 8) {
            uint32_t r[8];
            tmem_load_8x_fn(base_qkt + col, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
            tmem_load_fence_fn();
            for (int i = 0; i < 8; i++) {
                float val = __uint_as_float(r[i]) * scale;
                int g_q = q_step * 128 + threadIdx.x;
                int g_k = k_step * 128 + col + i;
                if (g_k > g_q || g_k >= S) {
                    val = __int_as_float(0xff800000);
                }
                row_max = fmaxf(row_max, val);
            }
        }

        float m_new = fmaxf(m_prev, row_max);
        float exp_scale = (m_prev == __int_as_float(0xff800000)) ? 0.0f : fast_exp2f_fn((m_prev - m_new) * 1.4426950408889634f);

        for (int i = 0; i < 128; i++) {
            O_total[i] *= exp_scale;
        }

        float row_sum = 0.0f;
        for (int col = 0; col < 128; col += 8) {
            uint32_t r[8];
            tmem_load_8x_fn(base_qkt + col, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
            tmem_load_fence_fn();
            uint32_t p_bf16[4];
            for (int i = 0; i < 8; i += 2) {
                float v0 = __uint_as_float(r[i]) * scale;
                float v1 = __uint_as_float(r[i + 1]) * scale;
                int g_q = q_step * 128 + threadIdx.x;
                int g_k0 = k_step * 128 + col + i;
                int g_k1 = g_k0 + 1;
                
                v0 = (g_k0 <= g_q && g_k0 < S) ? fast_exp2f_fn((v0 - m_new) * 1.4426950408889634f) : 0.0f;
                v1 = (g_k1 <= g_q && g_k1 < S) ? fast_exp2f_fn((v1 - m_new) * 1.4426950408889634f) : 0.0f;
                
                row_sum += v0 + v1;
                p_bf16[i/2] = pack_bf16_fn(__float_as_uint(v0), __float_as_uint(v1));
            }
            __nv_bfloat16* target_smem = (col < 64) ? smem_p0 : smem_p1;
            int local_col = col % 64; 
            st_smem_swizzle_128b(target_smem, threadIdx.x, local_col / 8, p_bf16[0], p_bf16[1], p_bf16[2], p_bf16[3]);
        }

        l_prev = l_prev * exp_scale + row_sum;
        m_prev = m_new;

        fence_async_shared_fn();
        __syncthreads();

        if (threadIdx.x == 0) {
            tcgen05_fence_after_thread_sync_fn();
            uint64_t cur_desc_p0 = desc_p0;
            uint64_t cur_desc_p1 = desc_p1;
            uint64_t cur_desc_v0 = desc_v0;
            uint64_t cur_desc_v1 = desc_v1;
            for (int k = 0; k < 8; ++k) {
                uint32_t accum = (k == 0) ? 0 : 1;
                if (k < 4) {
                    umma_f16_cg1_fn(base_o0, cur_desc_p0, cur_desc_v0, idesc_o0, accum);
                    umma_f16_cg1_fn(base_o1, cur_desc_p0, cur_desc_v1, idesc_o0, accum);
                    cur_desc_p0 = advance_desc(cur_desc_p0, 32);
                } else {
                    umma_f16_cg1_fn(base_o0, cur_desc_p1, cur_desc_v0, idesc_o0, accum);
                    umma_f16_cg1_fn(base_o1, cur_desc_p1, cur_desc_v1, idesc_o0, accum);
                    cur_desc_p1 = advance_desc(cur_desc_p1, 32);
                }
                cur_desc_v0 = advance_desc(cur_desc_v0, 2048);
                cur_desc_v1 = advance_desc(cur_desc_v1, 2048);
            }
            umma_commit_cg1_fn(mbar_umma2);
        }
        mbarrier_wait_fn(mbar_umma2, phase_umma2);
        phase_umma2 ^= 1;
        __syncthreads();

        for (int col = 0; col < 64; col += 8) {
            uint32_t r[8];
            tmem_load_8x_fn(base_o0 + col, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
            tmem_load_fence_fn();
            for (int i = 0; i < 8; i++) {
                O_total[col + i] += __uint_as_float(r[i]);
            }
        }
        for (int col = 0; col < 64; col += 8) {
            uint32_t r[8];
            tmem_load_8x_fn(base_o1 + col, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
            tmem_load_fence_fn();
            for (int i = 0; i < 8; i++) {
                O_total[64 + col + i] += __uint_as_float(r[i]);
            }
        }
        
        __syncthreads();
    }

    int global_q = q_step * 128 + threadIdx.x;
    if (global_q < S) {
        float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
        uint4* O_out = (uint4*)((__nv_bfloat16*)O + bh_idx * S * 128 + global_q * 128);
        for (int col = 0; col < 128; col += 8) {
            uint32_t out[4];
            for (int i = 0; i < 8; i += 2) {
                float v0 = O_total[col + i] * inv_l;
                float v1 = O_total[col + i + 1] * inv_l;
                out[i/2] = pack_bf16_fn(__float_as_uint(v0), __float_as_uint(v1));
            }
            O_out[col/8] = make_uint4(out[0], out[1], out[2], out[3]);
        }
        
        float lse = m_prev + logf(l_prev);
        ((float*)LSE)[bh_idx * S + global_q] = lse;
    }

    if (threadIdx.x < 32) {
        tmem_dealloc_cg1_fn(base_qkt, 128);
        tmem_dealloc_cg1_fn(base_o0, 64);
        tmem_dealloc_cg1_fn(base_o1, 64);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    int D = Q.size(3);
    
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    CUtensorMap tma_q, tma_k, tma_v;
    uint64_t B_H_S = B * H * S;
    
    CUresult res;
    res = create_tma_2d_descriptor_2B(&tma_q, Q.data_ptr(), D, B_H_S, 64, 128, 
                                      CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, 
                                      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if(res != CUDA_SUCCESS) { fprintf(stderr, "TMA Q fail\n"); exit(1); }
    
    res = create_tma_2d_descriptor_2B(&tma_k, K.data_ptr(), D, B_H_S, 64, 128, 
                                      CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, 
                                      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if(res != CUDA_SUCCESS) { fprintf(stderr, "TMA K fail\n"); exit(1); }

    res = create_tma_2d_descriptor_2B(&tma_v, V.data_ptr(), D, B_H_S, 64, 128, 
                                      CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, 
                                      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if(res != CUDA_SUCCESS) { fprintf(stderr, "TMA V fail\n"); exit(1); }

    int num_blocks_s = (S + 127) / 128;
    dim3 grid(B * H, num_blocks_s, 1);
    dim3 block(128, 1, 1);
    
    int smem_size = 131200; 
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 1;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    float scale = 1.0f / sqrtf((float)D);
    
    CUDA_CHECK(cudaFuncSetAttribute((void*)mha_d128_causal_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    cudaLaunchKernelEx(&config, mha_d128_causal_kernel, tma_q, tma_k, tma_v, 
                       O.data_ptr(), LSE.data_ptr(), B, H, S, scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda