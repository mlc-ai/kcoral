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
        cuGetErrorName(_e, &err_str);                              \
        fprintf(stderr, "CU error %s at %s:%d\n", err_str,         \
                __FILE__, __LINE__);                               \
        exit(1);                                                   \
    }                                                              \
} while(0)

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float fast_log2f_fn(float x) {
    float y;
    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
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

__device__ __forceinline__ void tma_load_4d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}

__device__ __forceinline__ void tma_store_4d_fn(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    asm volatile(
        "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group [%0, {%2, %3, %4, %5}], [%1];"
        :: "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}

__device__ __forceinline__ void tma_store_commit_fn() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void tma_store_wait_fn() {
    asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
}

__device__ __forceinline__ void tmem_store_4x_fn(uint32_t col, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {%0,%1,%2,%3};"
        :: "r"(r0),"r"(r1),"r"(r2),"r"(r3), "r"(col));
}

__device__ __forceinline__ void tcgen05_wait_ld_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_wait_st_fn() {
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
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
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(a));
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo, uint32_t swizzle) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)swizzle << 61;
    return d;
}

__device__ __forceinline__ uint32_t make_idesc_fn(uint32_t M, uint32_t N, uint32_t a_major, uint32_t b_major) {
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

__device__ __forceinline__ uint64_t advance_desc_base(uint64_t desc, uint32_t byte_offset) {
    uint32_t current_encoded = desc & 0x3FFF;
    uint32_t offset_encoded = byte_offset >> 4;
    uint32_t new_encoded = (current_encoded + offset_encoded) & 0x3FFF;
    return (desc & ~0x3FFFull) | new_encoded;
}

__global__ __launch_bounds__(128, 1) void mha_fwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    float* LSE,
    int seqlen)
{
    int m = blockIdx.x;
    int head = blockIdx.y;
    int batch = blockIdx.z;
    int row_idx = threadIdx.x;

    if (m * 128 >= seqlen) return;

    setmaxnreg_inc_sync_fn<248>();

    extern __shared__ __align__(128) uint8_t smem_pool[];
    uint8_t* smem_Q = smem_pool;
    uint8_t* smem_K = smem_pool + 32768;
    uint8_t* smem_V = smem_pool + 65536;
    uint8_t* smem_P = smem_pool + 98304;
    uint8_t* smem_O = smem_pool + 131072;

    // Zero-initialize SMEM to prevent NaN propagation from out-of-bounds TMA memory regions
    for (int i = threadIdx.x; i < 163840 / 4; i += 128) {
        ((uint32_t*)smem_pool)[i] = 0;
    }
    __syncthreads();

    __shared__ __align__(8) uint64_t mbar_Q[1];
    __shared__ __align__(8) uint64_t mbar_K[1];
    __shared__ __align__(8) uint64_t mbar_V[1];
    __shared__ __align__(8) uint64_t mbar_umma_S[1];
    __shared__ __align__(8) uint64_t mbar_umma_O[1];

    __shared__ uint32_t tmem_addr_S;
    __shared__ uint32_t tmem_addr_O;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&mbar_Q[0], 1);
        init_smem_barrier_fn(&mbar_K[0], 1);
        init_smem_barrier_fn(&mbar_V[0], 1);
        init_smem_barrier_fn(&mbar_umma_S[0], 1);
        init_smem_barrier_fn(&mbar_umma_O[0], 1);
    }
    fence_smem_barrier_init_fn();
    
    if (threadIdx.x < 32) {
        uint32_t a1 = (uint32_t)__cvta_generic_to_shared(&tmem_addr_S);
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 128;" :: "r"(a1));
        uint32_t a2 = (uint32_t)__cvta_generic_to_shared(&tmem_addr_O);
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 128;" :: "r"(a2));
    }
    __syncthreads();

    uint32_t S_tmem_addr = tmem_addr_S;
    uint32_t O_tmem_addr = tmem_addr_O;

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(&mbar_Q[0], 128 * 128 * 2);
        tma_load_4d_fn(&tma_Q, &mbar_Q[0], smem_Q, 0, m * 128, head, batch);
    }
    mbarrier_wait_fn(&mbar_Q[0], 0);

    // K-Major NO_SWIZZLE linear layout: LBO = 2048, SBO = 128
    uint64_t desc_Q = make_smem_desc_sm100_fn(smem_Q, 2048, 128, 0);
    uint64_t desc_K = make_smem_desc_sm100_fn(smem_K, 2048, 128, 0);
    uint64_t desc_P = make_smem_desc_sm100_fn(smem_P, 2048, 128, 0);

    // MN-Major NO_SWIZZLE linear layout: LBO = 128, SBO = 2048
    uint64_t desc_V = make_smem_desc_sm100_fn(smem_V, 128, 2048, 0);

    uint32_t idesc_QK = make_idesc_fn(128, 128, 0, 0); // Q(K-major) x K(K-major)
    uint32_t idesc_PV = make_idesc_fn(128, 128, 0, 1); // P(K-major) x V(N-major)

    float m_i = -INFINITY;
    float l_i = 0.0f;
    int phase_K = 0, phase_V = 0;
    int phase_umma_S = 0, phase_umma_O = 0;

    for (int n = 0; n <= m; ++n) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(&mbar_K[0], 128 * 128 * 2);
            tma_load_4d_fn(&tma_K, &mbar_K[0], smem_K, 0, n * 128, head, batch);
            mbarrier_arrive_and_expect_tx_fn(&mbar_V[0], 128 * 128 * 2);
            tma_load_4d_fn(&tma_V, &mbar_V[0], smem_V, 0, n * 128, head, batch);
        }
        mbarrier_wait_fn(&mbar_K[0], phase_K);
        mbarrier_wait_fn(&mbar_V[0], phase_V);
        phase_K ^= 1;
        phase_V ^= 1;

        if (threadIdx.x == 0) {
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            for (int k_step = 0; k_step < 8; ++k_step) {
                uint64_t d_Q = advance_desc_base(desc_Q, k_step * 32);
                uint64_t d_K = advance_desc_base(desc_K, k_step * 32);
                uint32_t accum = (k_step == 0) ? 0 : 1;
                umma_f16_cg1_fn(S_tmem_addr, d_Q, d_K, idesc_QK, accum);
            }
            umma_commit_cg1_fn(&mbar_umma_S[0]);
        }
        mbarrier_wait_fn(&mbar_umma_S[0], phase_umma_S);
        phase_umma_S ^= 1;

        uint32_t S_regs[128];
        for (int col = 0; col < 128; col += 4) {
            tmem_load_4x_fn(S_tmem_addr + col, &S_regs[col], &S_regs[col+1], &S_regs[col+2], &S_regs[col+3]);
        }
        tcgen05_wait_ld_fn();

        float row_max = -INFINITY;
        int global_row = m * 128 + row_idx;

        for (int col = 0; col < 128; ++col) {
            int global_col = n * 128 + col;
            if (global_col > global_row || global_col >= seqlen) {
                S_regs[col] = __float_as_uint(-INFINITY);
            } else {
                float val = __uint_as_float(S_regs[col]) * 0.08838834764f;
                S_regs[col] = __float_as_uint(val);
                if (val > row_max) {
                    row_max = val;
                }
            }
        }

        float new_m_i = fmaxf(m_i, row_max);
        float scale_o = 1.0f;
        if (new_m_i != -INFINITY) {
            scale_o = fast_exp2f_fn((m_i - new_m_i) * 1.44269504089f);
        }

        float row_sum = 0.0f;
        for (int col = 0; col < 128; ++col) {
            float val = __uint_as_float(S_regs[col]);
            float p = 0.0f;
            if (val != -INFINITY) {
                p = fast_exp2f_fn((val - new_m_i) * 1.44269504089f);
            }
            S_regs[col] = __float_as_uint(p);
            row_sum += p;
        }

        float new_l_i = l_i * scale_o + row_sum;

        if (n > 0) {
            for (int chunk = 0; chunk < 128; chunk += 32) {
                uint32_t O_regs[32];
                for (int col = 0; col < 32; col += 4) {
                    tmem_load_4x_fn(O_tmem_addr + chunk + col, &O_regs[col], &O_regs[col+1], &O_regs[col+2], &O_regs[col+3]);
                }
                tcgen05_wait_ld_fn();
                for (int col = 0; col < 32; col += 4) {
                    float o0 = __uint_as_float(O_regs[col]) * scale_o;
                    float o1 = __uint_as_float(O_regs[col+1]) * scale_o;
                    float o2 = __uint_as_float(O_regs[col+2]) * scale_o;
                    float o3 = __uint_as_float(O_regs[col+3]) * scale_o;
                    tmem_store_4x_fn(O_tmem_addr + chunk + col, __float_as_uint(o0), __float_as_uint(o1), __float_as_uint(o2), __float_as_uint(o3));
                }
            }
            tcgen05_wait_st_fn();
        }

        for (int col = 0; col < 128; col += 2) {
            uint32_t packed = pack_bf16_fn(__float_as_uint(S_regs[col]), __float_as_uint(S_regs[col+1]));
            ((uint32_t*)smem_P)[row_idx * 64 + col/2] = packed;
        }
        
        fence_async_shared_fn();
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
        __syncthreads();

        if (threadIdx.x == 0) {
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            for (int k_step = 0; k_step < 8; ++k_step) {
                uint64_t d_P = advance_desc_base(desc_P, k_step * 32);
                uint64_t d_V = advance_desc_base(desc_V, k_step * 4096);
                uint32_t accum = (n == 0 && k_step == 0) ? 0 : 1;
                umma_f16_cg1_fn(O_tmem_addr, d_P, d_V, idesc_PV, accum);
            }
            umma_commit_cg1_fn(&mbar_umma_O[0]);
        }
        mbarrier_wait_fn(&mbar_umma_O[0], phase_umma_O);
        phase_umma_O ^= 1;

        m_i = new_m_i;
        l_i = new_l_i;
    }

    float inv_l_i = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;

    for (int chunk = 0; chunk < 128; chunk += 32) {
        uint32_t O_regs[32];
        for (int col = 0; col < 32; col += 4) {
            tmem_load_4x_fn(O_tmem_addr + chunk + col, &O_regs[col], &O_regs[col+1], &O_regs[col+2], &O_regs[col+3]);
        }
        tcgen05_wait_ld_fn();
        for (int col = 0; col < 32; col += 4) {
            float o0 = __uint_as_float(O_regs[col]) * inv_l_i;
            float o1 = __uint_as_float(O_regs[col+1]) * inv_l_i;
            float o2 = __uint_as_float(O_regs[col+2]) * inv_l_i;
            float o3 = __uint_as_float(O_regs[col+3]) * inv_l_i;
            
            uint32_t p0 = pack_bf16_fn(__float_as_uint(o0), __float_as_uint(o1));
            uint32_t p1 = pack_bf16_fn(__float_as_uint(o2), __float_as_uint(o3));
            
            ((uint32_t*)smem_O)[row_idx * 64 + (chunk + col)/2] = p0;
            ((uint32_t*)smem_O)[row_idx * 64 + (chunk + col)/2 + 1] = p1;
        }
    }

    fence_async_shared_fn();
    __syncthreads();

    if (threadIdx.x == 0) {
        tma_store_4d_fn(&tma_O, smem_O, 0, m * 128, head, batch);
        tma_store_commit_fn();
        tma_store_wait_fn<0>();
    }
    __syncthreads();

    int global_row = m * 128 + row_idx;
    if (global_row < seqlen) {
        float lse = -INFINITY;
        if (l_i > 0.0f) {
            lse = m_i + fast_log2f_fn(l_i) * 0.69314718056f;
        }
        LSE[batch * (gridDim.y * seqlen) + head * seqlen + global_row] = lse;
    }

    if (threadIdx.x < 32) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 128;" :: "r"(S_tmem_addr));
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 128;" :: "r"(O_tmem_addr));
    }
}

CUresult create_tma_4d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t D, uint64_t S, uint64_t H, uint64_t B) {
    cuuint64_t globalDim[4] = {D, S, H, B};
    cuuint64_t globalStrides[3] = {D * 2, S * D * 2, H * S * D * 2};
    cuuint32_t boxDim[4] = {128, 128, 1, 1};
    cuuint32_t elementStrides[4] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
        globalAddress, globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

namespace tvm_ffi_mha {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B_dim = Q.size(0);
    int64_t H_dim = Q.size(1);
    int64_t S_dim = Q.size(2);
    int64_t D_dim = Q.size(3);
    
    alignas(64) CUtensorMap tma_Q, tma_K, tma_V, tma_O;
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_Q, Q.data_ptr(), D_dim, S_dim, H_dim, B_dim));
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_K, K.data_ptr(), D_dim, S_dim, H_dim, B_dim));
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_V, V.data_ptr(), D_dim, S_dim, H_dim, B_dim));
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_O, O.data_ptr(), D_dim, S_dim, H_dim, B_dim));
    
    int num_blocks_m = (S_dim + 127) / 128;
    dim3 grid(num_blocks_m, H_dim, B_dim);
    dim3 block(128);
    
    int smem_size = 5 * 32768; // 160KB
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
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
    
    CUDA_CHECK(cudaFuncSetAttribute((void*)mha_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, mha_fwd_kernel, tma_Q, tma_K, tma_V, tma_O, static_cast<float*>(LSE.data_ptr()), (int)S_dim));
}

}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha::run);