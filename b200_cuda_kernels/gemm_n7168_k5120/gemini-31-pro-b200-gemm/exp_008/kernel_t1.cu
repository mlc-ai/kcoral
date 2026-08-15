#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
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
        fprintf(stderr, "CU error %d at %s:%d\n",                 \
                (int)_e, __FILE__, __LINE__);                      \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_example_cuda {

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

__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t phase_parity) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(phase_parity));
}

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void named_barrier_sync_fn(int bar_id, int count) {
    asm volatile("barrier.sync.aligned %0, %1;" :: "r"(bar_id), "r"(count));
}

__device__ __forceinline__ void prefetch_tma_descriptor_fn(const CUtensorMap* d) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(d) : "memory");
}

__device__ __forceinline__ void tma_load_2d_cta_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
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
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
        :: "l"(bar) : "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);
    d |= (1u << 7);
    d |= (1u << 10);
    d |= (0u << 15);
    d |= (0u << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__device__ __forceinline__ void tmem_epilogue_tma_store(
    uint32_t tmem_c,
    __nv_bfloat16* smem_out,
    const CUtensorMap* tma_C,
    uint32_t m_idx, uint32_t n_idx,
    uint32_t BM, uint32_t BN) {
    
    for (uint32_t col = 0; col < BN; col += 4) {
        uint32_t r0, r1, r2, r3;
        uint32_t addr = tmem_c + col;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(addr));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        uint32_t base = threadIdx.x * BN + col;
        smem_out[base + 0] = __float2bfloat16(__uint_as_float(r0));
        smem_out[base + 1] = __float2bfloat16(__uint_as_float(r1));
        smem_out[base + 2] = __float2bfloat16(__uint_as_float(r2));
        smem_out[base + 3] = __float2bfloat16(__uint_as_float(r3));
    }
    
    named_barrier_sync_fn(0, 128);
    
    if (threadIdx.x == 0) {
        tma_store_fence_fn();
        tma_store_2d_fn(tma_C, smem_out, n_idx * BN, m_idx * BM);
        tma_store_commit_fn();
        tma_store_wait_fn<0>();
    }
    
    named_barrier_sync_fn(1, 128);
}

extern __shared__ __align__(1024) uint8_t smem_pool[];

__global__ void gemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    const __grid_constant__ CUtensorMap tma_C,
    int K
) {
    int m_idx = blockIdx.y;
    int n_idx = blockIdx.x;
    int k_steps = K / 64;

    uint8_t* smem_A = smem_pool;
    uint8_t* smem_B = smem_pool + 65536;
    __nv_bfloat16* smem_C = (__nv_bfloat16*)smem_pool;

    __shared__ uint64_t bar[4];
    __shared__ uint64_t umma_bar[4];
    __shared__ uint32_t tmem_c_smem;

    if (threadIdx.x == 0) {
        for (int i = 0; i < 4; i++) {
            init_smem_barrier_fn(&bar[i], 1);
            init_smem_barrier_fn(&umma_bar[i], 1);
        }
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        fence_smem_barrier_init_fn();
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        tmem_alloc_cg1_fn(&tmem_c_smem, 128);
    }
    __syncthreads();
    uint32_t tmem_c = tmem_c_smem;

    if (threadIdx.x == 0) {
        prefetch_tma_descriptor_fn(&tma_A);
        prefetch_tma_descriptor_fn(&tma_B);
        prefetch_tma_descriptor_fn(&tma_C);

        for (int i = 0; i < 4 && i < k_steps; i++) {
            mbarrier_arrive_and_expect_tx_fn(&bar[i], 32768);
            tma_load_2d_cta_fn(&tma_A, &bar[i], smem_A + i * 16384, i * 64, m_idx * 128);
            tma_load_2d_cta_fn(&tma_B, &bar[i], smem_B + i * 16384, i * 64, n_idx * 128);
        }

        int accum = 0;
        uint32_t idesc = make_instr_desc_fn(128, 128);

        for (int k = 0; k < k_steps; k++) {
            int stage = k % 4;

            mbarrier_wait_fn(&bar[stage], (k / 4) & 1);
            fence_proxy_async_fn();

            for (int step = 0; step < 4; step++) {
                uint32_t addr_A = (uint32_t)__cvta_generic_to_shared(smem_A + stage * 16384 + step * 32);
                uint32_t addr_B = (uint32_t)__cvta_generic_to_shared(smem_B + stage * 16384 + step * 32);
                uint64_t desc_A = make_smem_desc_sm100_fn((void*)addr_A, 1, 1024);
                uint64_t desc_B = make_smem_desc_sm100_fn((void*)addr_B, 1, 1024);

                umma_f16_cg1_fn(tmem_c, desc_A, desc_B, idesc, accum);
                accum = 1;
            }

            umma_commit_cg1_fn(&umma_bar[stage]);

            if (k >= 1 && (k - 1) + 4 < k_steps) {
                int prev_k = k - 1;
                int prev_stage = prev_k % 4;
                mbarrier_wait_fn(&umma_bar[prev_stage], (prev_k / 4) & 1);
                
                int next_k = prev_k + 4;
                mbarrier_arrive_and_expect_tx_fn(&bar[prev_stage], 32768);
                tma_load_2d_cta_fn(&tma_A, &bar[prev_stage], smem_A + prev_stage * 16384, next_k * 64, m_idx * 128);
                tma_load_2d_cta_fn(&tma_B, &bar[prev_stage], smem_B + prev_stage * 16384, next_k * 64, n_idx * 128);
            }
        }

        if (k_steps > 0) {
            int last_k = k_steps - 1;
            mbarrier_wait_fn(&umma_bar[last_k % 4], (last_k / 4) & 1);
        }
    }

    __syncthreads();
    tcgen05_fence_after_fn();

    tmem_epilogue_tma_store(tmem_c, smem_C, &tma_C, m_idx, n_idx, 128, 128);

    __syncthreads();
    if (threadIdx.x < 32) {
        tmem_dealloc_cg1_fn(tmem_c, 128);
    }
}

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    uint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    uint64_t globalStrides[1] = {gmem_inner_dim * 2};
    uint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    uint32_t elementStrides[2] = {1, 1};
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

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(0);
    
    if (M == 0 || N == 0 || K == 0) return;
    
    __nv_bfloat16* a_ptr = static_cast<__nv_bfloat16*>(A.data_ptr());
    __nv_bfloat16* b_ptr = static_cast<__nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* c_ptr = static_cast<__nv_bfloat16*>(C.data_ptr());
    
    CUtensorMap tma_A, tma_B, tma_C;
    
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_A, a_ptr, K, M, 64, 128, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
        
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_B, b_ptr, K, N, 64, 128, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
        
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_C, c_ptr, N, M, 128, 128, 
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    
    dim3 block(128);
    dim3 grid((N + 127) / 128, (M + 127) / 128);
    
    int smem_size = 131072;
    CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
    
    gemm_kernel<<<grid, block, smem_size, stream>>>(tma_A, tma_B, tma_C, K);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}