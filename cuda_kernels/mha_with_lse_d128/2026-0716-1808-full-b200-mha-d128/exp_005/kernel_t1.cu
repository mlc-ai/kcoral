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

// Device Helper Functions
__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
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

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
   :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void umma_f16_cg1_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    // Accum = 1 acts as standard accumulation. Accum = 0 clears the output matrix D.
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

__device__ __forceinline__ void cp_tmem_128x128b(uint32_t tmem_addr, void* smem_ptr) {
    uint64_t desc = make_smem_desc_sm100_fn(smem_ptr, 1, 1024);
    asm volatile("tcgen05.cp.cta_group::1.128x128b [%0], %1;\n"
        :: "r"(tmem_addr), "l"(desc) : "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   // SM100 version (1)
    d |= (uint64_t)2 << 61;   // SWIZZLE_128B mode (2)
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    // c_format = FP32
    d |= (1u << 7);    // a_format = BF16
    d |= (1u << 10);   // b_format = BF16
    d |= (0u << 15);   // a_major = 0 (A is K-Major)
    d |= (0u << 16);   // b_major = 0 (B is K-Major)
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_PV_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    // c_format = FP32
    d |= (1u << 7);    // a_format = BF16
    d |= (1u << 10);   // b_format = BF16
    d |= (0u << 15);   // a_major = 0 (A is K-Major)
    d |= (1u << 16);   // b_major = 1 (B is MN-Major)
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tmem_ld_32x32b_x4(uint32_t tmem_addr, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*r0), "=r"(*r1), "=r"(*r2), "=r"(*r3) : "r"(tmem_addr) : "memory");
}

__device__ __forceinline__ void tmem_st_32x32b_x4(uint32_t tmem_addr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};"
        :: "r"(tmem_addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
}

// FlashAttention Device Kernel
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

    extern __shared__ char smem[];
    __nv_bfloat16* sm_Q = (__nv_bfloat16*)smem;
    __nv_bfloat16* sm_K = sm_Q + 16384;
    __nv_bfloat16* sm_V = sm_K + 16384;
    __nv_bfloat16* smem_P = sm_V + 16384;
    uint64_t* bar_Q = (uint64_t*)(smem_P + 16384);
    uint64_t* bar_K = bar_Q + 1;
    uint64_t* bar_V = bar_K + 1;
    uint64_t* bar_cp = bar_V + 1;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(bar_Q, 1);
        init_smem_barrier_fn(bar_K, 1);
        init_smem_barrier_fn(bar_V, 1);
        init_smem_barrier_fn(bar_cp, 1);
    }
    __syncthreads();

    uint32_t tmem_Q, tmem_O, tmem_K_or_P, tmem_V_or_S;
    if (threadIdx.x == 0) {
        tmem_alloc_fn(&tmem_Q, 128);
        tmem_alloc_fn(&tmem_O, 128);
        tmem_alloc_fn(&tmem_K_or_P, 128);
        tmem_alloc_fn(&tmem_V_or_S, 128);
    }
    __syncthreads();

    uint32_t s_coord = head_idx * S_seq + m_row;
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar_Q, 32768);
        tma_load_2d_fn(&tma_Q, bar_Q, sm_Q, 0, s_coord);
        tma_load_2d_fn(&tma_Q, bar_Q, sm_Q + 8192, 64, s_coord);
    }
    mbarrier_wait_fn(bar_Q, 0);
    
    uint64_t q_desc = make_smem_desc_sm100_fn(sm_Q, 1, 1024);
    uint64_t k_desc = make_smem_desc_sm100_fn(sm_K, 1, 1024);
    uint64_t v_desc = make_smem_desc_sm100_fn(sm_V, 1, 1024);
    uint64_t p_desc = make_smem_desc_sm100_fn(smem_P, 1, 1024);

    uint32_t idesc_QK = make_instr_desc_fn(128, 128);
    uint32_t idesc_PV = make_instr_desc_PV_fn(128, 128);

    float running_max = -INFINITY;
    float running_sum = 0.0f;

    uint32_t phase_K = 0, phase_V = 0, phase_cp = 0;
    uint32_t num_iters = (S_seq + 127) / 128;

    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        uint32_t n_block = iter * 128;
        uint32_t kv_coord = head_idx * S_seq + n_block;

        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_K, 32768);
            tma_load_2d_fn(&tma_K, bar_K, sm_K, 0, kv_coord);
            tma_load_2d_fn(&tma_K, bar_K, sm_K + 8192, 64, kv_coord);
        }
        mbarrier_wait_fn(bar_K, phase_K); phase_K ^= 1;

        if (threadIdx.x == 0) {
            cp_tmem_128x128b(tmem_K_or_P + 0, sm_K);
            cp_tmem_128x128b(tmem_K_or_P + 64, sm_K + 8192);
            umma_commit_1sm_fn(bar_cp);
        }
        mbarrier_wait_fn(bar_cp, phase_cp); phase_cp ^= 1;

        uint32_t s_addr = tmem_V_or_S;
        if (threadIdx.x == 0) {
            for (int k = 0; k < 8; ++k) {
                umma_f16_cg1_fn(s_addr + k * 2, q_desc + k * 2, k_desc + k * 2, idesc_QK, 1);
            }
            umma_commit_1sm_fn(bar_cp);
        }
        mbarrier_wait_fn(bar_cp, phase_cp); phase_cp ^= 1;

        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_V, 32768);
            tma_load_2d_fn(&tma_V, bar_V, sm_V, 0, kv_coord);
            tma_load_2d_fn(&tma_V, bar_V, sm_V + 8192, 64, kv_coord);
        }

        float local_max = -INFINITY;
        for (uint32_t col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_ld_32x32b_x4(s_addr + col, &r0, &r1, &r2, &r3);
            float f0 = __uint_as_float(r0);
            float f1 = __uint_as_float(r1);
            float f2 = __uint_as_float(r2);
            float f3 = __uint_as_float(r3);
            if (iter * 128 + col >= S_seq) f0 = -INFINITY;
            if (iter * 128 + col + 1 >= S_seq) f1 = -INFINITY;
            if (iter * 128 + col + 2 >= S_seq) f2 = -INFINITY;
            if (iter * 128 + col + 3 >= S_seq) f3 = -INFINITY;
            f0 *= scale; f1 *= scale; f2 *= scale; f3 *= scale;
            local_max = fmaxf(local_max, fmaxf(fmaxf(f0, f1), fmaxf(f2, f3)));
        }
        tmem_load_fence_fn();

        float old_max = running_max;
        float new_max = fmaxf(old_max, local_max);
        
        if (new_max > old_max) {
            float factor = __expf(old_max - new_max);
            uint32_t o_addr = tmem_O;
            for (uint32_t col = 0; col < 128; col += 4) {
                uint32_t o0, o1, o2, o3;
                tmem_ld_32x32b_x4(o_addr + col, &o0, &o1, &o2, &o3);
                o0 = __float_as_uint(__uint_as_float(o0) * factor);
                o1 = __float_as_uint(__uint_as_float(o1) * factor);
                o2 = __float_as_uint(__uint_as_float(o2) * factor);
                o3 = __float_as_uint(__uint_as_float(o3) * factor);
                tmem_st_32x32b_x4(o_addr + col, o0, o1, o2, o3);
            }
            tmem_load_fence_fn();
            tmem_store_fence_fn();
            running_sum *= factor;
        }
        running_max = new_max;

        float local_sum = 0;
        for (uint32_t col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_ld_32x32b_x4(s_addr + col, &r0, &r1, &r2, &r3);
            float f0 = __uint_as_float(r0) * scale;
            float f1 = __uint_as_float(r1) * scale;
            float f2 = __uint_as_float(r2) * scale;
            float f3 = __uint_as_float(r3) * scale;
            
            float p0 = __expf(f0 - new_max);
            float p1 = __expf(f1 - new_max);
            float p2 = __expf(f2 - new_max);
            float p3 = __expf(f3 - new_max);
            if (iter * 128 + col >= S_seq) p0 = 0;
            if (iter * 128 + col + 1 >= S_seq) p1 = 0;
            if (iter * 128 + col + 2 >= S_seq) p2 = 0;
            if (iter * 128 + col + 3 >= S_seq) p3 = 0;
            
            local_sum += p0 + p1 + p2 + p3;
            
            int row = threadIdx.x;
            int x_chunk = col / 8;
            int x_rem = col % 8;
            int swizzled_x_chunk = (row % 8) ^ x_chunk;
            int phys_col = swizzled_x_chunk * 8 + x_rem;
            smem_P[row * 128 + phys_col] = __float2bfloat16(p0);
            smem_P[row * 128 + phys_col + 1] = __float2bfloat16(p1);
            smem_P[row * 128 + phys_col + 2] = __float2bfloat16(p2);
            smem_P[row * 128 + phys_col + 3] = __float2bfloat16(p3);
        }
        tmem_load_fence_fn();

        running_sum += local_sum;

        __syncthreads();
        
        if (threadIdx.x == 0) {
            cp_tmem_128x128b(tmem_K_or_P + 0, smem_P);
            cp_tmem_128x128b(tmem_K_or_P + 64, smem_P + 8192);
            umma_commit_1sm_fn(bar_cp);
        }
        mbarrier_wait_fn(bar_cp, phase_cp); phase_cp ^= 1;

        mbarrier_wait_fn(bar_V, phase_V); phase_V ^= 1;
        
        if (threadIdx.x == 0) {
            cp_tmem_128x128b(tmem_V_or_S + 0, sm_V);
            cp_tmem_128x128b(tmem_V_or_S + 64, sm_V + 8192);
            umma_commit_1sm_fn(bar_cp);
        }
        mbarrier_wait_fn(bar_cp, phase_cp); phase_cp ^= 1;

        uint32_t o_addr = tmem_O;
        if (threadIdx.x == 0) {
            for (int k = 0; k < 8; ++k) {
                umma_f16_cg1_fn(o_addr + k * 2, p_desc + k * 2, v_desc + k * 256, idesc_PV, 1);
            }
            umma_commit_1sm_fn(bar_cp);
        }
        mbarrier_wait_fn(bar_cp, phase_cp); phase_cp ^= 1;
    }

    uint32_t o_addr = tmem_O;
    for (uint32_t col = 0; col < 128; col += 4) {
        uint32_t o0, o1, o2, o3;
        tmem_ld_32x32b_x4(o_addr + col, &o0, &o1, &o2, &o3);
        tmem_load_fence_fn();
        
        if (running_sum > 0) {
            o0 = __float_as_uint(__uint_as_float(o0) / running_sum);
            o1 = __float_as_uint(__uint_as_float(o1) / running_sum);
            o2 = __float_as_uint(__uint_as_float(o2) / running_sum);
            o3 = __float_as_uint(__uint_as_float(o3) / running_sum);
        }
        
        uint32_t g_row = m_row + threadIdx.x;
        if (g_row < S_seq) {
            __nv_bfloat16* out = O_gmem + head_idx * S_seq * 128 + g_row * 128 + col;
            out[0] = __float2bfloat16(__uint_as_float(o0));
            out[1] = __float2bfloat16(__uint_as_float(o1));
            out[2] = __float2bfloat16(__uint_as_float(o2));
            out[3] = __float2bfloat16(__uint_as_float(o3));
        }
    }

    uint32_t g_row_lse = m_row + threadIdx.x;
    if (g_row_lse < S_seq) {
        LSE_gmem[head_idx * S_seq + g_row_lse] = running_max + __logf(running_sum);
    }

    if (threadIdx.x == 0) {
        tmem_dealloc_fn(tmem_Q, 128);
        tmem_dealloc_fn(tmem_O, 128);
        tmem_dealloc_fn(tmem_K_or_P, 128);
        tmem_dealloc_fn(tmem_V_or_S, 128);
    }
}

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapDataType dataType, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d, dataType, 2, globalAddress, globalDim, globalStrides, boxDim, elementStrides,
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

    uint32_t smem_size = 135168;
    CUDA_CHECK(cudaFuncSetAttribute(fa_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    CUtensorMap tma_Q, tma_K, tma_V;
    constexpr uint32_t ATOM_KMODE_DIM = 64;
    constexpr uint32_t ATOM_MMODE_DIM = 128;

    CU_CHECK(create_tma_2d_descriptor_2B(&tma_Q, Q.data_ptr(), D, B * H * S, ATOM_KMODE_DIM, ATOM_MMODE_DIM, 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_K, K.data_ptr(), D, B * H * S, ATOM_KMODE_DIM, ATOM_MMODE_DIM, 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_V, V.data_ptr(), D, B * H * S, ATOM_KMODE_DIM, ATOM_MMODE_DIM, 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    uint32_t num_blocks_x = (S_seq + 127) / 128;
    if (num_blocks_x > 1024) num_blocks_x = 1024;
    dim3 grid(num_blocks_x, B * H);
    dim3 block(128);

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