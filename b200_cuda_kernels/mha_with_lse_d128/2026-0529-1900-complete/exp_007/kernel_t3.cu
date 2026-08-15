#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <math.h>
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
        fprintf(stderr, "CU error %s at %s:%d\n",                  \
                err_str, __FILE__, __LINE__);                      \
        exit(1);                                                   \
    }                                                              \
} while(0)

#define M_LOG2E_F 1.4426950408889634f

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

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
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
                 :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_cg1_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void tmem_load_8x_fn(uint32_t taddr,
    uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3,
    uint32_t* r4, uint32_t* r5, uint32_t* r6, uint32_t* r7) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3),
     "=r"(*r4),"=r"(*r5),"=r"(*r6),"=r"(*r7) : "r"(taddr));
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
    asm volatile(
        "tcgen05.commit.cta_group::1"
        ".mbarrier::arrive::one.b64"
        " [%0];"
        :: "l"((uint64_t)bar) : "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   // version = 1
    d |= (uint64_t)0 << 61;   // layout_type = SWIZZLE_NONE
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, uint32_t a_major, uint32_t b_major) {
    uint32_t d = 0;
    d |= (1u << 4);    // c_format = FP32
    d |= (1u << 7);    // a_format = BF16
    d |= (1u << 10);   // b_format = BF16
    d |= (a_major << 15); // 0 = K-Major, 1 = MN-Major
    d |= (b_major << 16);
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__global__ void __launch_bounds__(128) mha_fwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* __restrict__ O_ptr,
    float* __restrict__ LSE_ptr,
    int B, int H, int S, int D) 
{
    setmaxnreg_inc_sync_fn<256>();

    int b_idx = blockIdx.z;
    int h_idx = blockIdx.y;
    int q_idx = blockIdx.x * 128;

    __shared__ __align__(1024) __nv_bfloat16 smem_Q[128][128];
    __shared__ __align__(1024) __nv_bfloat16 smem_K[128][128];
    __shared__ __align__(1024) __nv_bfloat16 smem_V[128][128];
    __shared__ __align__(1024) __nv_bfloat16 smem_P[128][128];

    __shared__ uint64_t smem_bar_Q;
    __shared__ uint64_t smem_bar_K;
    __shared__ uint64_t smem_bar_V;
    __shared__ uint64_t smem_bar_MMA;

    __shared__ uint32_t tmem_S_ptr;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&smem_bar_Q, 1);
        init_smem_barrier_fn(&smem_bar_K, 1);
        init_smem_barrier_fn(&smem_bar_V, 1);
        init_smem_barrier_fn(&smem_bar_MMA, 1);
        fence_smem_barrier_init_fn();
    }
    if (threadIdx.x < 32) {
        tmem_alloc_cg1_fn(&tmem_S_ptr, 128);
    }
    __syncthreads();

    uint32_t tmem_S = tmem_S_ptr;
    uint32_t phase_Q = 0, phase_K = 0, phase_V = 0, phase_MMA = 0;

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(&smem_bar_Q, 128 * 128 * 2);
        tma_load_2d_fn(&tma_Q, &smem_bar_Q, &smem_Q[0][0], 0, b_idx * H * S + h_idx * S + q_idx);
    }

    float O_reg[128];
    #pragma unroll
    for (int i = 0; i < 128; ++i) O_reg[i] = 0.0f;
    float m_max = -INFINITY;
    float l_sum = 0.0f;

    if (threadIdx.x == 0) {
        mbarrier_wait_fn(&smem_bar_Q, phase_Q);
        phase_Q ^= 1;
    }
    __syncthreads();

    for (int n_idx = 0; n_idx < S; n_idx += 128) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(&smem_bar_K, 128 * 128 * 2);
            tma_load_2d_fn(&tma_K, &smem_bar_K, &smem_K[0][0], 0, b_idx * H * S + h_idx * S + n_idx);

            mbarrier_arrive_and_expect_tx_fn(&smem_bar_V, 128 * 128 * 2);
            tma_load_2d_fn(&tma_V, &smem_bar_V, &smem_V[0][0], 0, b_idx * H * S + h_idx * S + n_idx);
            
            mbarrier_wait_fn(&smem_bar_K, phase_K);
            phase_K ^= 1;

            uint32_t idesc = make_instr_desc_fn(128, 128, 0, 0); // Q is K-major, K^T is K-major
            for (int k = 0; k < 8; ++k) {
                // Row-major data mapped to K-Major descriptor implies swapping classical LBO & SBO behavior
                uint64_t desc_Q = make_smem_desc_sm100_fn(&smem_Q[0][k * 16], 16, 2048);
                uint64_t desc_K = make_smem_desc_sm100_fn(&smem_K[0][k * 16], 16, 2048);
                uint32_t accum = (k == 0) ? 0 : 1;
                umma_f16_cg1_fn(tmem_S, desc_Q, desc_K, idesc, accum);
            }
            umma_commit_cg1_fn(&smem_bar_MMA);
        }

        mbarrier_wait_fn(&smem_bar_MMA, phase_MMA);
        phase_MMA ^= 1;

        float row_max = -INFINITY;
        for (uint32_t c = 0; c < 128; c += 8) {
            uint32_t r[8];
            tmem_load_8x_fn(tmem_S + c, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
            tmem_load_fence_fn();
            for (int i = 0; i < 8; ++i) {
                float val = __uint_as_float(r[i]);
                val *= 0.08838834764f; // 1.0f / sqrtf(128.0f)
                if (n_idx + c + i >= S) val = -INFINITY;
                if (q_idx + threadIdx.x < S) {
                    row_max = fmaxf(row_max, val);
                }
            }
        }

        float new_m_max = fmaxf(m_max, row_max);
        float scale_old = 1.0f;
        if (new_m_max > -1e30f) {
            scale_old = fast_exp2f_fn((m_max - new_m_max) * M_LOG2E_F);
        }
        l_sum *= scale_old;

        for (int i = 0; i < 128; ++i) {
            O_reg[i] *= scale_old;
        }
        m_max = new_m_max;

        float row_sum = 0;
        for (uint32_t c = 0; c < 128; c += 8) {
            uint32_t r[8];
            tmem_load_8x_fn(tmem_S + c, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
            tmem_load_fence_fn();
            uint32_t p_u32[4];
            for (int i = 0; i < 8; i += 2) {
                float v0 = __uint_as_float(r[i]) * 0.08838834764f;
                float v1 = __uint_as_float(r[i+1]) * 0.08838834764f;
                float p0 = 0.0f;
                float p1 = 0.0f;
                if (m_max > -1e30f) {
                    p0 = fast_exp2f_fn((v0 - m_max) * M_LOG2E_F);
                    p1 = fast_exp2f_fn((v1 - m_max) * M_LOG2E_F);
                }
                if (n_idx + c + i >= S) p0 = 0.0f;
                if (n_idx + c + i + 1 >= S) p1 = 0.0f;
                row_sum += p0 + p1;
                p_u32[i / 2] = pack_bf16_fn(__float_as_uint(p0), __float_as_uint(p1));
            }
            uint32_t addr = (uint32_t)__cvta_generic_to_shared(&smem_P[threadIdx.x][c]);
            st_shared_128_fn(addr, p_u32[0], p_u32[1], p_u32[2], p_u32[3]);
        }
        l_sum += row_sum;

        // Essential: Fence local generic memory writes (smem_P) to make them visible to the async proxy 
        // across all participating warps BEFORE thread synchronizing & issuing UMMA
        fence_async_shared_fn(); 
        __syncthreads();

        if (threadIdx.x == 0) {
            mbarrier_wait_fn(&smem_bar_V, phase_V);
            phase_V ^= 1;

            uint32_t idesc_PV = make_instr_desc_fn(128, 128, 0, 1); // P is K-major, V is MN-major
            for (int k = 0; k < 8; ++k) {
                uint64_t desc_P = make_smem_desc_sm100_fn(&smem_P[0][k * 16], 16, 2048);
                uint64_t desc_V = make_smem_desc_sm100_fn(&smem_V[k * 16][0], 2048, 16); // MN-Major correctly shifts LBO and SBO behavior
                uint32_t accum = (k == 0) ? 0 : 1;
                umma_f16_cg1_fn(tmem_S, desc_P, desc_V, idesc_PV, accum);
            }
            umma_commit_cg1_fn(&smem_bar_MMA);
        }

        mbarrier_wait_fn(&smem_bar_MMA, phase_MMA);
        phase_MMA ^= 1;

        for (uint32_t c = 0; c < 128; c += 8) {
            uint32_t r[8];
            tmem_load_8x_fn(tmem_S + c, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
            tmem_load_fence_fn();
            for (int i = 0; i < 8; ++i) {
                float pv = __uint_as_float(r[i]);
                O_reg[c + i] += pv;
            }
        }
        __syncthreads();
    }

    float inv_l_sum = 1.0f / l_sum;
    if (l_sum == 0.0f) inv_l_sum = 0.0f;
    for (int i = 0; i < 128; ++i) {
        O_reg[i] *= inv_l_sum;
    }

    for (uint32_t c = 0; c < 128; c += 8) {
        uint32_t o_u32[4];
        for (int i = 0; i < 8; i += 2) {
            o_u32[i / 2] = pack_bf16_fn(__float_as_uint(O_reg[c + i]), __float_as_uint(O_reg[c + i + 1]));
        }
        uint32_t addr = (uint32_t)__cvta_generic_to_shared(&smem_P[threadIdx.x][c]);
        st_shared_128_fn(addr, o_u32[0], o_u32[1], o_u32[2], o_u32[3]);
    }
    __syncthreads();

    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    for (uint32_t step = 0; step < 32; ++step) {
        uint32_t row = step * 4 + warp_id;
        uint32_t global_row = q_idx + row;
        uint32_t col_start = lane_id * 4;
        
        if (global_row < S && col_start < 128) {
            uint2 data = *reinterpret_cast<uint2*>(&smem_P[row][col_start]);
            uint64_t out_idx = (uint64_t)b_idx * H * S * 128 + (uint64_t)h_idx * S * 128 + (uint64_t)global_row * 128 + col_start;
            *reinterpret_cast<uint2*>(O_ptr + out_idx) = data;
        }
    }

    if (q_idx + threadIdx.x < S) {
        float lse = m_max + logf(l_sum);
        if (l_sum == 0.0f) lse = -INFINITY;
        uint64_t lse_idx = (uint64_t)b_idx * H * S + (uint64_t)h_idx * S + q_idx + threadIdx.x;
        LSE_ptr[lse_idx] = lse;
    }

    __syncthreads();
    if (threadIdx.x < 32) {
        tmem_dealloc_cg1_fn(tmem_S_ptr, 128);
    }
}

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

namespace tvm_ffi_mha {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);

    CUtensorMap tma_Q, tma_K, tma_V;
    uint64_t gmem_inner_dim = D;
    uint64_t gmem_outer_dim = B * H * S;
    
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_Q, Q.data_ptr(), gmem_inner_dim, gmem_outer_dim, 128, 128, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_K, K.data_ptr(), gmem_inner_dim, gmem_outer_dim, 128, 128, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_V, V.data_ptr(), gmem_inner_dim, gmem_outer_dim, 128, 128, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    __nv_bfloat16* O_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_ptr = static_cast<float*>(LSE.data_ptr());

    dim3 block(128);
    dim3 grid((S + 127) / 128, H, B);

    mha_fwd_kernel<<<grid, block, 0, stream>>>(tma_Q, tma_K, tma_V, O_ptr, LSE_ptr, B, H, S, D);
    CUDA_CHECK(cudaGetLastError());
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha::run);

} // namespace tvm_ffi_mha