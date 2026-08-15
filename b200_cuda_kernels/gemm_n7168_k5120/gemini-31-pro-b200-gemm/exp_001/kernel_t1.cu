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

// Helper functions for SM100 architecture
template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
}

__device__ __forceinline__ void cluster_sync_fn() {
    asm volatile("barrier.cluster.arrive;\nbarrier.cluster.wait;\n" ::: "memory");
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

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void tma_load_2d_cg2_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    uint32_t sa = (uint32_t)__cvta_generic_to_shared(smem);
    uint32_t ba = (uint32_t)__cvta_generic_to_shared(&bar[0]) & 0xFEFFFFFF; 
    asm volatile(
        "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(sa), "l"((uint64_t)d), "r"(c0), "r"(c1), "r"(ba) : "memory");
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

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
   :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void tmem_load_8x_fn(uint32_t col,
    uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3,
    uint32_t* r4, uint32_t* r5, uint32_t* r6, uint32_t* r7) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3),
     "=r"(*r4),"=r"(*r5),"=r"(*r6),"=r"(*r7) : "r"(col));
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

__device__ __forceinline__ void advance_desc_k(uint64_t& desc, uint32_t k_bytes) {
    uint32_t current_addr_encoded = desc & 0x3FFF;
    uint32_t new_addr_encoded = current_addr_encoded + (k_bytes >> 4);
    desc = (desc & ~0x3FFFull) | (new_addr_encoded & 0x3FFF);
}

CUresult create_tma_2d_descriptor(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapDataType dataType, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
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

struct SharedStorage {
    alignas(128) __nv_bfloat16 A[3][128][64]; // 48 KB
    alignas(128) __nv_bfloat16 B[3][128][64]; // 48 KB
    alignas(128) uint64_t tma_mbar[3];        // 24 B
    alignas(128) uint64_t umma_mbar[3];       // 24 B
};

__global__ void __launch_bounds__(128, 2) gemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    const __grid_constant__ CUtensorMap tma_C,
    uint32_t M, uint32_t N, uint32_t K
) {
    setmaxnreg_inc_sync_fn<240>();
    
    extern __shared__ __align__(128) char smem_buf[];
    SharedStorage* smem = reinterpret_cast<SharedStorage*>(smem_buf);
    
    uint32_t pair_idx = blockIdx.x / 2;
    uint32_t rank_in_pair = blockIdx.x % 2;
    uint32_t M_TILES = (M + 255) / 256;
    
    uint32_t pair_m = pair_idx % M_TILES;
    uint32_t pair_n = pair_idx / M_TILES;
    
    uint32_t global_m_start_0 = pair_m * 256;
    uint32_t global_m_start_1 = pair_m * 256 + 128;
    uint32_t global_n_start_0 = pair_n * 256;
    uint32_t global_n_start_1 = pair_n * 256 + 128;
    
    uint32_t tma_m_start = (rank_in_pair == 0) ? global_m_start_0 : global_m_start_1;
    uint32_t tma_n_start = (rank_in_pair == 0) ? global_n_start_0 : global_n_start_1;
    
    uint32_t my_m_start = tma_m_start;
    uint32_t my_n_start = global_n_start_0; 
    
    __shared__ uint32_t smem_tmem_c;
    if (threadIdx.x < 32) {
        tmem_alloc_fn(&smem_tmem_c, 256);
    }
    
    if (rank_in_pair == 0 && threadIdx.x == 0) {
        for (int s = 0; s < 3; s++) {
            init_smem_barrier_fn(&smem->tma_mbar[s], 1);
        }
    }
    if (threadIdx.x == 0) {
        for (int s = 0; s < 3; s++) {
            init_smem_barrier_fn(&smem->umma_mbar[s], 1);
        }
    }
    
    __syncthreads();
    if (threadIdx.x == 0) {
        fence_smem_barrier_init_fn();
    }
    cluster_sync_fn(); 
    
    uint32_t tmem_c = smem_tmem_c;
    int k_iters = K / 64;
    int tma_phases[3] = {0, 0, 0};
    int umma_phases[3] = {0, 0, 0};
    uint32_t idesc = make_instr_desc_fn(256, 256);
    
    // Prologue: Fill the 3 stages
    for (int s = 0; s < 3 && s < k_iters; s++) {
        if (rank_in_pair == 0 && threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(&smem->tma_mbar[s], 65536);
        }
        if (threadIdx.x == 0) {
            tma_load_2d_cg2_fn(&tma_A, &smem->tma_mbar[s], smem->A[s], s * 64, tma_m_start);
            tma_load_2d_cg2_fn(&tma_B, &smem->tma_mbar[s], smem->B[s], s * 64, tma_n_start);
        }
    }
    
    // Main loop
    for (int i = 0; i < k_iters; i++) {
        int s = i % 3;
        
        if (rank_in_pair == 0 && threadIdx.x == 0) {
            mbarrier_wait_fn(&smem->tma_mbar[s], tma_phases[s]);
            tma_phases[s] ^= 1;
            
            fence_proxy_async_fn();
            
            uint64_t desc_A_s = make_smem_desc_sm100_fn(smem->A[s], 1, 1024); 
            uint64_t desc_B_s = make_smem_desc_sm100_fn(smem->B[s], 1, 1024);
            
            for (int k_step = 0; k_step < 4; k_step++) {
                uint64_t d_A = desc_A_s;
                uint64_t d_B = desc_B_s;
                advance_desc_k(d_A, k_step * 32);
                advance_desc_k(d_B, k_step * 32);
                
                uint32_t accum = (i == 0 && k_step == 0) ? 0 : 1;
                umma_f16_cg2_fn(tmem_c, d_A, d_B, idesc, accum);
            }
            
            umma_commit_2sm_fn(&smem->umma_mbar[s]);
        }
        
        if (i + 3 < k_iters) {
            int next_s = (i + 3) % 3;
            
            if (threadIdx.x == 0) {
                mbarrier_wait_fn(&smem->umma_mbar[next_s], umma_phases[next_s]);
                umma_phases[next_s] ^= 1;
            }
            
            if (rank_in_pair == 0 && threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(&smem->tma_mbar[next_s], 65536);
            }
            if (threadIdx.x == 0) {
                tma_load_2d_cg2_fn(&tma_A, &smem->tma_mbar[next_s], smem->A[next_s], (i + 3) * 64, tma_m_start);
                tma_load_2d_cg2_fn(&tma_B, &smem->tma_mbar[next_s], smem->B[next_s], (i + 3) * 64, tma_n_start);
            }
        }
    }
    
    // Epilogue wait
    if (threadIdx.x == 0) {
        for (int i = max(0, k_iters - 3); i < k_iters; i++) {
            int s = i % 3;
            mbarrier_wait_fn(&smem->umma_mbar[s], umma_phases[s]);
        }
    }
    __syncthreads();
    
    // Epilogue store: TMEM -> SMEM -> Global via TMA
    __nv_bfloat16* smem_C = reinterpret_cast<__nv_bfloat16*>(smem);
    
    for (uint32_t c_step = 0; c_step < 256; c_step += 8) {
        uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
        tmem_load_8x_fn(tmem_c + c_step, &r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7);
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        uint32_t base = threadIdx.x * 256 + c_step;
        uint32_t p0 = pack_bf16_fn(r0, r1);
        uint32_t p1 = pack_bf16_fn(r2, r3);
        uint32_t p2 = pack_bf16_fn(r4, r5);
        uint32_t p3 = pack_bf16_fn(r6, r7);
        *reinterpret_cast<uint4*>(&smem_C[base]) = make_uint4(p0, p1, p2, p3);
    }
    
    __syncthreads();
    
    tma_store_fence_fn();
    if (threadIdx.x == 0) {
        tma_store_2d_fn(&tma_C, smem_C, my_n_start, my_m_start);
        tma_store_commit_fn();
        tma_store_wait_fn<0>();
    }
    
    __syncthreads();
    
    if (threadIdx.x < 32) {
        tmem_dealloc_fn(tmem_c, 256);
    }
}

namespace tvm_ffi_example_cuda {

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(0);
    
    CUtensorMap tma_A, tma_B, tma_C;
    CU_CHECK(create_tma_2d_descriptor(&tma_A, A.data_ptr(), K, M, 64, 128, 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor(&tma_B, B.data_ptr(), K, N, 64, 128, 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor(&tma_C, C.data_ptr(), N, M, 256, 128, 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    int M_TILES = (M + 255) / 256;
    int N_TILES = (N + 255) / 256;
    int grid_x = M_TILES * N_TILES * 2;

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute((void*)gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(SharedStorage)));

    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(grid_x, 1, 1);
    config.blockDim = dim3(128, 1, 1);
    config.dynamicSmemBytes = sizeof(SharedStorage);
    config.stream = stream;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    CUDA_CHECK(cudaLaunchKernelEx(&config, gemm_kernel, tma_A, tma_B, tma_C, M, N, K));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda