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

__device__ __forceinline__ void named_barrier_sync_fn(int bar_id, int count) {
    asm volatile("barrier.sync.aligned %0, %1;" :: "r"(bar_id), "r"(count));
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
    uint32_t n_block = blockIdx.x * 128;

    if (m_block >= M) return;

    extern __shared__ __align__(1024) uint8_t smem_pool[];
    __nv_bfloat16* smem_A = (__nv_bfloat16*)smem_pool;                      // 32KB
    __nv_bfloat16* smem_B = (__nv_bfloat16*)(smem_A + 8192);                 // 16KB
    
    __shared__ __align__(8) uint64_t bar_A[2], bar_B[1], bar_umma[1];
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&bar_A[0], 1);
        init_smem_barrier_fn(&bar_A[1], 1);
        init_smem_barrier_fn(bar_B, 1);
        init_smem_barrier_fn(bar_umma, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    __shared__ uint32_t tmem_C_addr;
    if (threadIdx.x == 0) {
        tmem_alloc_fn(&tmem_C_addr, 128); 
    }
    __syncthreads();

    uint32_t idesc = make_instr_desc_fn(128, 128);

    uint32_t my_m_block = m_block + cluster_rank_fn() * 64;
    uint32_t my_n_block = n_block + cluster_rank_fn() * 64;

    uint32_t phase_A[2] = {0, 0}, phase_B = 0, phase_umma = 0;

    // Prologue load
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(&bar_A[0], 8192);
        tma_load_2d_fn(&tma_A, &bar_A[0], smem_A, 0, my_m_block);
        
        mbarrier_arrive_and_expect_tx_fn(bar_B, 8192);
        tma_load_2d_fn(&tma_B, bar_B, smem_B, 0, my_n_block);
    }
    
    // Software transpose of B to fix geometry for WGMMA
    mbarrier_wait_fn(bar_B, phase_B);
    __syncthreads();
    uint32_t* smem_B_u32 = (uint32_t*)smem_B;
    for (uint32_t i = threadIdx.x; i < 512; i += 128) {
        uint32_t row = i / 8;
        uint32_t col = i % 8;
        uint32_t swizzled_col = col ^ (row % 8);
        uint32_t read_idx = row * 8 + swizzled_col;
        
        uint32_t write_swizzled_col = col ^ (swizzled_col % 8);
        uint32_t write_idx = swizzled_col * 8 + write_swizzled_col;
        
        smem_B_u32[write_idx] = smem_B_u32[read_idx];
    }
    __syncthreads();
    if (threadIdx.x == 0) mbarrier_wait_fn(&bar_A[0], phase_A[0]);

    for (uint32_t j = 0; j < K / 64; ++j) {
        // Double buffer preload
        if (threadIdx.x == 0 && j + 1 < K / 64) {
            uint32_t next_buf = (j + 1) % 2;
            mbarrier_arrive_and_expect_tx_fn(&bar_A[next_buf], 8192);
            tma_load_2d_fn(&tma_A, &bar_A[next_buf], smem_A + next_buf * 4096, (j + 1) * 64, my_m_block);
        }

        if (threadIdx.x == 0 && j + 1 < K / 64) {
            uint32_t next_buf = (j + 1) % 2;
            mbarrier_arrive_and_expect_tx_fn(&bar_A[next_buf], 8192);
            tma_load_2d_fn(&tma_A, &bar_A[next_buf], smem_A + next_buf * 4096, (j + 1) * 64, my_m_block);
        }

        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_B, 8192);
            tma_load_2d_fn(&tma_B, bar_B, smem_B, (j + 1) * 64, my_n_block);
        }

        uint32_t buf_idx = j % 2;
        mbarrier_wait_fn(&bar_A[buf_idx], phase_A[buf_idx]);
        mbarrier_wait_fn(bar_B, phase_B);
        fence_proxy_async_fn();

        // Software transpose inline
        __syncthreads();
        for (uint32_t i = threadIdx.x; i < 512; i += 128) {
            uint32_t row = i / 8;
            uint32_t col = i % 8;
            uint32_t swizzled_col = col ^ (row % 8);
            uint32_t read_idx = row * 8 + swizzled_col;
            
            uint32_t write_swizzled_col = col ^ (swizzled_col % 8);
            uint32_t write_idx = swizzled_col * 8 + write_swizzled_col;
            
            smem_B_u32[write_idx] = smem_B_u32[read_idx];
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            uint32_t accum = (j == 0) ? 0 : 1;
            for (uint32_t k = 0; k < 4; ++k) {
                uint64_t desc_A = make_smem_desc_sm100_fn(smem_A + buf_idx * 4096 + k * 16, 1024, 1024);
                uint64_t desc_B = make_smem_desc_sm100_fn(smem_B + k * 16, 1024, 1024);
                umma_f16_cg2_fn(tmem_C_addr, desc_A, desc_B, idesc, accum);
                accum = 1;
            }
            umma_commit_2sm_fn(bar_umma);
        }
        
        mbarrier_wait_fn(bar_umma, phase_umma);
        
        phase_A[buf_idx] ^= 1;
        phase_B ^= 1;
        phase_umma ^= 1;
    }

    __syncthreads();

    // Epilogue
    uint32_t* smem_out_u32 = (uint32_t*)smem_B;
    uint32_t lane_id = threadIdx.x;
    uint32_t m_idx = my_m_block + lane_id;
    
    for (uint32_t col = 0; col < 128; col += 4) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4x_fn(col, &r0, &r1, &r2, &r3);
        tmem_load_fence_fn();

        uint32_t base = lane_id * 128 + col;
        smem_out_u32[base / 2] = ((uint32_t)(__float2bfloat16(__uint_as_float(r1))) << 16) | (uint32_t)(__float2bfloat16(__uint_as_float(r0)));
        smem_out_u32[base / 2 + 1] = ((uint32_t)(__float2bfloat16(__uint_as_float(r3))) << 16) | (uint32_t)(__float2bfloat16(__uint_as_float(r2)));
    }
    
    named_barrier_sync_fn(1, 128);

    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id2 = threadIdx.x % 32;
    uint32_t num_steps = (64 + 3) / 4;
    for (uint32_t step = 0; step < num_steps; ++step) {
        uint32_t row = step * 4 + warp_id;
        if (row >= 64) continue;
        uint32_t global_row = my_m_block * 1 + row; // Scale my_m_block artificially to prevent OOB writes scaling issues
        uint32_t global_col = n_block + lane_id2 * 4;
        if (global_row < M && global_col + 3 < N) {
             uint2 data = *reinterpret_cast<uint2*>(&smem_B[row * 128 + lane_id2 * 4]);
             *reinterpret_cast<uint2*>(C + (uint64_t)global_row * N + global_col) = data;
        }
    }

    __syncthreads();
    
    if (threadIdx.x == 0) {
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
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_A, (void*)A_data, K, M, 64, 64, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_B, (void*)B_data, K, N, 64, 64, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
        
    dim3 grid((N + 127) / 128, (M + 127) / 128);
    dim3 block(128);
    
    size_t smem_size = 49152; // 48 KB
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