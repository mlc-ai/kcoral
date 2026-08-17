#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
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

#define CU_CHECK(call) do {                                        \
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        fprintf(stderr, "CU error %d at %s:%d\n",                 \
                _e, __FILE__, __LINE__);                           \
        exit(1);                                                   \
    }                                                              \
} while(0)

// ------------------------------------------------------------------
// SM100 specific helper functions
// ------------------------------------------------------------------

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

__device__ __forceinline__ void tma_load_4d_cg1_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
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

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo, uint32_t swizzle = 0) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   // version = 1 (SM100)
    d |= (uint64_t)swizzle << 61;
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, uint32_t a_major, uint32_t b_major) {
    uint32_t d = 0;
    d |= (1u << 4);    // FP32 Out
    d |= (1u << 7);    // BF16 A
    d |= (1u << 10);   // BF16 B
    d |= (a_major << 15);
    d |= (b_major << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
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
        :: "r"(a));
}

__device__ __forceinline__ void tmem_load_8x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3, uint32_t* r4, uint32_t* r5, uint32_t* r6, uint32_t* r7) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3),
     "=r"(*r4),"=r"(*r5),"=r"(*r6),"=r"(*r7) : "r"(col));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
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

// ------------------------------------------------------------------
// Main Attention Kernel
// ------------------------------------------------------------------

__global__ __launch_bounds__(128, 1) void mha_causal_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int S, int H, int B
) {
    extern __shared__ char smem[];
    __nv_bfloat16* smem_Q = (__nv_bfloat16*)smem;
    __nv_bfloat16* smem_K = smem_Q + 128 * 128;
    __nv_bfloat16* smem_V = smem_K + 64 * 128;
    __nv_bfloat16* smem_P = smem_V + 64 * 128;
    
    uint64_t* mbar_Q   = (uint64_t*)(smem_P + 128 * 64);
    uint64_t* mbar_K   = mbar_Q + 1;
    uint64_t* mbar_V   = mbar_K + 1;
    uint64_t* mbar_mma = mbar_V + 1;
    uint32_t* tmem_addr_smem = (uint32_t*)(mbar_mma + 1);

    int b_idx = blockIdx.z;
    int h_idx = blockIdx.y;
    int q_base = blockIdx.x * 128;
    int tid = threadIdx.x;

    if (tid == 0) {
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(mbar_K, 1);
        init_smem_barrier_fn(mbar_V, 1);
        init_smem_barrier_fn(mbar_mma, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    if (tid / 32 == 0) {
        tmem_alloc_cg1_fn(tmem_addr_smem, 256);
    }
    __syncthreads();
    
    uint32_t tmem_addr = *tmem_addr_smem;
    uint32_t TMEM_QK = tmem_addr;
    uint32_t TMEM_PV = tmem_addr + 64;

    uint32_t q_phase = 0, k_phase = 0, v_phase = 0, mma_phase = 0;

    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_Q, 128 * 128 * 2);
        tma_load_4d_cg1_fn(&tma_Q, mbar_Q, smem_Q, 0, q_base, h_idx, b_idx);
    }
    mbarrier_wait_fn(mbar_Q, q_phase); q_phase ^= 1;

    float m_i = -INFINITY;
    float l_i = 0.0f;
    float O_i[128];
    for (int i = 0; i < 128; ++i) {
        O_i[i] = 0.0f;
    }

    int row = tid;
    float scale = 1.0f / sqrtf(128.0f);
    int global_q = q_base + row;

    uint32_t idesc_QK = make_instr_desc_fn(128, 64, 0, 0);
    uint32_t idesc_PV = make_instr_desc_fn(128, 128, 0, 1);

    for (int k_base = 0; k_base < q_base + 128 && k_base < S; k_base += 64) {
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_K, 64 * 128 * 2);
            tma_load_4d_cg1_fn(&tma_K, mbar_K, smem_K, 0, k_base, h_idx, b_idx);
            
            mbarrier_arrive_and_expect_tx_fn(mbar_V, 64 * 128 * 2);
            tma_load_4d_cg1_fn(&tma_V, mbar_V, smem_V, 0, k_base, h_idx, b_idx);
        }

        mbarrier_wait_fn(mbar_K, k_phase); k_phase ^= 1;

        if (tid == 0) {
            for (int k_mma = 0; k_mma < 128; k_mma += 16) {
                uint64_t d_Q = make_smem_desc_sm100_fn((char*)smem_Q + k_mma * 2, 2048, 128, 0);
                uint64_t d_K = make_smem_desc_sm100_fn((char*)smem_K + k_mma * 2, 1024, 128, 0);
                uint32_t accum = (k_mma == 0) ? 0 : 1;
                umma_f16_cg1_fn(TMEM_QK, d_Q, d_K, idesc_QK, accum);
            }
            umma_commit_cg1_fn(mbar_mma);
        }
        mbarrier_wait_fn(mbar_mma, mma_phase); mma_phase ^= 1;

        float chunk[64];
        for (int c = 0; c < 64; c += 8) {
            tmem_load_8x_fn(TMEM_QK + c, (uint32_t*)&chunk[c], (uint32_t*)&chunk[c+1], (uint32_t*)&chunk[c+2], (uint32_t*)&chunk[c+3], (uint32_t*)&chunk[c+4], (uint32_t*)&chunk[c+5], (uint32_t*)&chunk[c+6], (uint32_t*)&chunk[c+7]);
        }
        tmem_load_fence_fn();

        float row_max = m_i;
        for (int c = 0; c < 64; ++c) {
            int global_k = k_base + c;
            if (global_q >= S || global_k >= S || global_k > global_q) {
                chunk[c] = -INFINITY;
            } else {
                chunk[c] *= scale;
                row_max = fmaxf(row_max, chunk[c]);
            }
        }

        float exp_diff = fast_exp2f_fn((m_i - row_max) * 1.44269504f);
        l_i = l_i * exp_diff;
        for (int i = 0; i < 128; ++i) {
            O_i[i] *= exp_diff;
        }
        m_i = row_max;

        float row_sum = 0.0f;
        for (int c = 0; c < 64; ++c) {
            if (chunk[c] != -INFINITY) {
                chunk[c] = fast_exp2f_fn((chunk[c] - row_max) * 1.44269504f);
                row_sum += chunk[c];
            } else {
                chunk[c] = 0.0f;
            }
        }
        l_i += row_sum;

        for (int c = 0; c < 64; c += 2) {
            uint32_t packed = pack_bf16_fn(*(uint32_t*)&chunk[c], *(uint32_t*)&chunk[c+1]);
            uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(smem_P) + (row * 64 + c) * 2;
            *(uint32_t*)smem_addr = packed;
        }
        fence_async_shared_fn();
        __syncthreads();

        mbarrier_wait_fn(mbar_V, v_phase); v_phase ^= 1;

        if (tid == 0) {
            for (int k_mma = 0; k_mma < 64; k_mma += 16) {
                uint64_t d_P = make_smem_desc_sm100_fn((char*)smem_P + k_mma * 2, 2048, 128, 0);
                uint64_t d_V = make_smem_desc_sm100_fn((char*)smem_V + k_mma * 256, 128, 1024, 0);
                uint32_t accum = (k_mma == 0) ? 0 : 1;
                umma_f16_cg1_fn(TMEM_PV, d_P, d_V, idesc_PV, accum);
            }
            umma_commit_cg1_fn(mbar_mma);
        }
        mbarrier_wait_fn(mbar_mma, mma_phase); mma_phase ^= 1;

        for (int c = 0; c < 64; c += 8) {
            tmem_load_8x_fn(TMEM_PV + c, (uint32_t*)&chunk[c], (uint32_t*)&chunk[c+1], (uint32_t*)&chunk[c+2], (uint32_t*)&chunk[c+3], (uint32_t*)&chunk[c+4], (uint32_t*)&chunk[c+5], (uint32_t*)&chunk[c+6], (uint32_t*)&chunk[c+7]);
        }
        tmem_load_fence_fn();
        for (int c = 0; c < 64; ++c) {
            O_i[c] += chunk[c];
        }

        for (int c = 64; c < 128; c += 8) {
            tmem_load_8x_fn(TMEM_PV + c, (uint32_t*)&chunk[c - 64], (uint32_t*)&chunk[c - 63], (uint32_t*)&chunk[c - 62], (uint32_t*)&chunk[c - 61], (uint32_t*)&chunk[c - 60], (uint32_t*)&chunk[c - 59], (uint32_t*)&chunk[c - 58], (uint32_t*)&chunk[c - 57]);
        }
        tmem_load_fence_fn();
        for (int c = 64; c < 128; ++c) {
            O_i[c] += chunk[c - 64];
        }

        __syncthreads();
    }

    if (global_q < S) {
        float inv_sum = 1.0f / l_i;
        __nv_bfloat16* O_ptr = O + b_idx * H * S * 128 + h_idx * S * 128 + global_q * 128;
        for (int c = 0; c < 128; ++c) {
            O_i[c] *= inv_sum;
            O_ptr[c] = __float2bfloat16(O_i[c]);
        }
        LSE[b_idx * H * S + h_idx * S + global_q] = m_i + logf(l_i);
    }

    if (tid / 32 == 0) {
        tmem_dealloc_cg1_fn(tmem_addr, 256);
    }
}

// ------------------------------------------------------------------
// Host Run Function
// ------------------------------------------------------------------

CUresult create_tma_4d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t D, uint64_t S, uint64_t H, uint64_t B, uint32_t smem_D, uint32_t smem_S) {
    cuuint64_t globalDim[4] = {D, S, H, B};
    cuuint64_t globalStrides[3] = {D * 2, S * D * 2, H * S * D * 2};
    cuuint32_t boxDim[4] = {smem_D, smem_S, 1, 1};
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
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

namespace tvm_ffi_example_cuda {

void run(
    tvm::ffi::TensorView Q, 
    tvm::ffi::TensorView K, 
    tvm::ffi::TensorView V,
    tvm::ffi::TensorView O,
    tvm::ffi::TensorView LSE
) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    int D = Q.size(3);

    CUtensorMap tma_Q, tma_K, tma_V;
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_Q, Q.data_ptr(), D, S, H, B, 128, 128));
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_K, K.data_ptr(), D, S, H, B, 128, 64));
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_V, V.data_ptr(), D, S, H, B, 128, 64));

    dim3 grid((S + 127) / 128, H, B);
    dim3 block(128);
    int smem_bytes = 82000;
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    CUDA_CHECK(cudaFuncSetAttribute(mha_causal_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

    mha_causal_kernel<<<grid, block, smem_bytes, stream>>>(
        tma_Q, tma_K, tma_V, 
        static_cast<__nv_bfloat16*>(O.data_ptr()), 
        static_cast<float*>(LSE.data_ptr()), 
        S, H, B
    );
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda