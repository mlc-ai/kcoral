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
        fprintf(stderr, "CUDA Driver error %d at %s:%d\n",          \
                _e, __FILE__, __LINE__);                           \
        exit(1);                                                   \
    }                                                              \
} while(0)


__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
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

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
   :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
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

__device__ __forceinline__ void umma_f16_cg2_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_2sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::2"
        ".mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
        " [%0], %1;"
        :: "r"(a), "h"((uint16_t)0x3));
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

__global__ void __launch_bounds__(128) gemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    __nv_bfloat16* C, uint32_t M, uint32_t N, uint32_t K) 
{
    uint32_t m_block = blockIdx.y * 128;
    uint32_t n_block = (blockIdx.x / 2) * 256;
    uint32_t cta_id = cluster_rank_fn();

    if (m_block >= M) return;

    extern __shared__ __align__(1024) uint8_t smem_pool[];
    __nv_bfloat16* smem_A = (__nv_bfloat16*)smem_pool;                   // 32KB
    __nv_bfloat16* smem_B = (__nv_bfloat16*)(smem_A + 8192);              // 32KB
    
    __shared__ __align__(8) uint64_t bar_A[2], bar_B[2], bar_umma[1];
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&bar_A[0], 1);
        init_smem_barrier_fn(&bar_A[1], 1);
        init_smem_barrier_fn(&bar_B[0], 1);
        init_smem_barrier_fn(&bar_B[1], 1);
        init_smem_barrier_fn(bar_umma, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    __shared__ uint32_t tmem_C_addr;
    if (threadIdx.x < 32) { 
        tmem_alloc_fn(&tmem_C_addr, 128); 
    }
    __syncthreads();

    uint32_t idesc = make_instr_desc_fn(128, 256); 
    
    uint32_t my_n_block = n_block + cta_id * 128;
    uint32_t phase_A[2] = {0, 0}, phase_B[2] = {0, 0}, phase_umma = 0;

    // Prologue load
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(&bar_A[0], 16384);
        tma_load_2d_fn(&tma_A, &bar_A[0], smem_A, 0, m_block);
        
        mbarrier_arrive_and_expect_tx_fn(&bar_B[0], 16384);
        tma_load_2d_fn(&tma_B, &bar_B[0], smem_B, 0, my_n_block);
    }
    
    for (uint32_t j = 0; j < K / 64; ++j) {
        uint32_t cur_buf = j % 2;
        uint32_t next_buf = (j + 1) % 2;

        mbarrier_wait_fn(&bar_A[cur_buf], phase_A[cur_buf]);
        mbarrier_wait_fn(&bar_B[cur_buf], phase_B[cur_buf]);
        fence_proxy_async_fn();

        if (threadIdx.x == 0) {
            uint32_t accum = (j == 0) ? 0 : 1;
            for (uint32_t k = 0; k < 4; ++k) {
                uint64_t desc_A = make_smem_desc_sm100_fn(smem_A + cur_buf * 8192 + k * 16, 1024, 1024);
                uint64_t desc_B = make_smem_desc_sm100_fn(smem_B + cur_buf * 8192 + k * 16, 1024, 1024);
                umma_f16_cg2_fn(tmem_C_addr, desc_A, desc_B, idesc, accum);
                accum = 1;
            }
            umma_commit_2sm_fn(bar_umma);
        }

        // Double buffer preload
        if (j + 1 < K / 64) {
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(&bar_A[next_buf], 16384);
                tma_load_2d_fn(&tma_A, &bar_A[next_buf], smem_A + next_buf * 8192, (j + 1) * 64, m_block);
                
                mbarrier_arrive_and_expect_tx_fn(&bar_B[next_buf], 16384);
                tma_load_2d_fn(&tma_B, &bar_B[next_buf], smem_B + next_buf * 8192, (j + 1) * 64, my_n_block);
            }
        }

        mbarrier_wait_fn(bar_umma, phase_umma);
        
        phase_A[cur_buf] ^= 1;
        phase_B[cur_buf] ^= 1;
        phase_umma ^= 1;
    }

    __syncthreads();

    // Epilogue
    uint32_t lane_id = threadIdx.x;
    uint32_t m_idx = m_block + lane_id;
    
    for (uint32_t col = 0; col < 128; col += 4) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4x_fn(col, &r0, &r1, &r2, &r3);
        tmem_load_fence_fn();

        if (m_idx < M) {
            uint32_t n_idx = my_n_block + col;
            if (n_idx + 0 < N) C[(uint64_t)m_idx * N + n_idx + 0] = __float2bfloat16(__uint_as_float(r0));
            if (n_idx + 1 < N) C[(uint64_t)m_idx * N + n_idx + 1] = __float2bfloat16(__uint_as_float(r1));
            if (n_idx + 2 < N) C[(uint64_t)m_idx * N + n_idx + 2] = __float2bfloat16(__uint_as_float(r2));
            if (n_idx + 3 < N) C[(uint64_t)m_idx * N + n_idx + 3] = __float2bfloat16(__uint_as_float(r3));
        }
    }

    __syncthreads();
    
    if (threadIdx.x < 32) {
        tmem_dealloc_fn(tmem_C_addr, 128);
    }
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    
    uint32_t M = A.size(0);
    uint32_t N = 7168;
    uint32_t K = 5120;
    
    const __nv_bfloat16* A_data = static_cast<const __nv_bfloat16*>(A.data_ptr());
    const __nv_bfloat16* B_data = static_cast<const __nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* C_data = static_cast<__nv_bfloat16*>(C.data_ptr());
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
    
    CUtensorMap tma_A, tma_B;
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_A, (void*)A_data, K, M, 64, 128, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_B, (void*)B_data, K, N, 64, 128, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
        
    dim3 grid((N / 256) * 2, (M + 127) / 128);
    dim3 block(128);
    
    size_t smem_size = 65536;
    CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, gemm_kernel, tma_A, tma_B, C_data, M, N, K));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);