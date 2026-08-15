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

// Helper functions for SM100 TMA and TMEM operations

__device__ __forceinline__ uint32_t get_smid() {
    uint32_t smid;
    asm ("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
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

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tma_load_multicast_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, uint16_t mask) {
    uint64_t cache_hint = 0;
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint [%0], [%1, {%4, %5}], [%2], %3, %6;"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "h"(mask), "r"(c0), "r"(c1), "l"(cache_hint) : "memory");
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

__device__ __forceinline__ void tcgen05_fence_after_fn() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
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

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr) >> 4;
    d |= (uint64_t)(addr & 0x3FFF);
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);           
    d |= (1u << 7);           
    d |= (1u << 10);          
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ void epilogue(
    __nv_bfloat16* D, __nv_bfloat16* smem_out,
    uint32_t tmem_c,
    uint32_t M, uint32_t N, uint32_t m_block, uint32_t n_block,
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
    __syncthreads();
    
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t num_steps = (BM + 3) / 4;
    
    for (uint32_t step = 0; step < num_steps; ++step) {
        uint32_t row = step * 4 + warp_id;
        if (row >= BM) continue;
        uint32_t global_row = m_block * BM + row;
        uint32_t col_start = lane_id * 4;
        
        for (uint32_t pass = 0; pass < BN / 128; ++pass) {
            uint32_t actual_col_start = col_start + pass * 128;
            uint32_t global_col = n_block * BN + actual_col_start;
            if (global_row < M && global_col + 3 < N) {
                uint2 data = *reinterpret_cast<uint2*>(&smem_out[row * BN + actual_col_start]);
                *reinterpret_cast<uint2*>(D + (uint64_t)global_row * N + global_col) = data;
            } else if (global_row < M) {
                for (int i = 0; i < 4; ++i) {
                    if (global_col + i < N) {
                        D[(uint64_t)global_row * N + global_col + i] = smem_out[row * BN + actual_col_start + i];
                    }
                }
            }
        }
    }
}

CUresult create_tma_2d_descriptor_2B(
    CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, 
    uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, 
    CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    
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

struct SharedStorage {
    alignas(128) __nv_bfloat16 A[3][128][64];
    alignas(128) __nv_bfloat16 B[3][256][64];
    alignas(16) uint64_t mbar[3];
    alignas(16) uint64_t mbar_commit[3];
};

extern __shared__ char dynamic_smem[];

__global__ void gemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    __nv_bfloat16* C_ptr,
    uint32_t M, uint32_t N, uint32_t K
) {
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(dynamic_smem);
    
    int rank = cluster_rank_fn();
    uint32_t M_start = blockIdx.y * 256 + rank * 128;
    uint32_t N_start = blockIdx.x * 256;
    
    __shared__ uint32_t smem_tmem_addr;
    if (threadIdx.x < 32) {
        tmem_alloc_fn(&smem_tmem_addr, 256);
    }
    __syncthreads();
    uint32_t tmem_addr = smem_tmem_addr;
    
    if (threadIdx.x == 0) {
        for (int i = 0; i < 3; ++i) {
            init_smem_barrier_fn(&smem.mbar[i], 1);
            init_smem_barrier_fn(&smem.mbar_commit[i], 1);
        }
    }
    fence_smem_barrier_init_fn();
    __syncthreads();
    
    uint32_t K_tiles = (K + 63) / 64;
    
    for (int s = 0; s < min(3, (int)K_tiles); ++s) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(&smem.mbar[s], 48 * 1024);
        }
        cluster_sync_fn();
        
        if (threadIdx.x == 0) {
            if (rank == 0) {
                tma_load_multicast_2d_fn(&tma_B, &smem.mbar[s], smem.B[s], s * 64, N_start, 0x3);
                tma_load_2d_fn(&tma_A, &smem.mbar[s], smem.A[s], s * 64, M_start);
            } else {
                tma_load_2d_fn(&tma_A, &smem.mbar[s], smem.A[s], s * 64, M_start);
            }
        }
    }
    
    for (int k = 0; k < K_tiles; ++k) {
        int s = k % 3;
        mbarrier_wait_fn(&smem.mbar[s], (k / 3) & 1);
        
        if (threadIdx.x == 0 && rank == 0) {
            uint64_t desc_a = make_smem_desc_sm100_fn(smem.A[s], 1024);
            uint64_t desc_b = make_smem_desc_sm100_fn(smem.B[s], 1024);
            uint32_t idesc = make_instr_desc_fn(256, 256);
            uint32_t accum = (k > 0) ? 1 : 0;
            
            tcgen05_fence_after_fn();
            umma_f16_cg2_fn(tmem_addr, desc_a, desc_b, idesc, accum);
            umma_commit_2sm_fn(&smem.mbar_commit[s]);
        }
        
        int next_k = k + 3;
        if (next_k < K_tiles) {
            int next_s = next_k % 3;
            mbarrier_wait_fn(&smem.mbar_commit[next_s], ((next_k - 3) / 3) & 1);
            
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(&smem.mbar[next_s], 48 * 1024);
            }
            cluster_sync_fn();
            
            if (threadIdx.x == 0) {
                if (rank == 0) {
                    tma_load_multicast_2d_fn(&tma_B, &smem.mbar[next_s], smem.B[next_s], next_k * 64, N_start, 0x3);
                    tma_load_2d_fn(&tma_A, &smem.mbar[next_s], smem.A[next_s], next_k * 64, M_start);
                } else {
                    tma_load_2d_fn(&tma_A, &smem.mbar[next_s], smem.A[next_s], next_k * 64, M_start);
                }
            }
        }
    }
    
    if (K_tiles > 0) {
        mbarrier_wait_fn(&smem.mbar_commit[(K_tiles - 1) % 3], ((K_tiles - 1) / 3) & 1);
    }
    tcgen05_fence_after_fn();
    __syncthreads();
    
    __nv_bfloat16* smem_out = (__nv_bfloat16*)&smem;
    epilogue(C_ptr, smem_out, tmem_addr, M, N, blockIdx.y * 2 + rank, blockIdx.x, 128, 256);
    
    __syncthreads();
    cluster_sync_fn();
    if (threadIdx.x < 32) {
        tmem_dealloc_fn(tmem_addr, 256);
    }
}

namespace tvm_ffi_gemm_sm100 {

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(0);

    if (M == 0 || N == 0 || K == 0) return;

    CUtensorMap tma_A, tma_B;
    void* A_ptr = A.data_ptr();
    void* B_ptr = B.data_ptr();
    void* C_ptr = C.data_ptr();

    CUresult res_A = create_tma_2d_descriptor_2B(
        &tma_A, A_ptr, K, M, 64, 128,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    if (res_A != CUDA_SUCCESS) {
        fprintf(stderr, "TMA A descriptor creation failed: %d\n", res_A);
        exit(1);
    }

    CUresult res_B = create_tma_2d_descriptor_2B(
        &tma_B, B_ptr, K, N, 64, 256,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    if (res_B != CUDA_SUCCESS) {
        fprintf(stderr, "TMA B descriptor creation failed: %d\n", res_B);
        exit(1);
    }

    int threads = 128;
    dim3 block(threads);
    dim3 grid((N + 255) / 256, (M + 255) / 256, 1);

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = sizeof(SharedStorage);
    config.stream = stream;

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    CUDA_CHECK(cudaLaunchKernelEx(&config, gemm_kernel, tma_A, tma_B, (__nv_bfloat16*)C_ptr, (uint32_t)M, (uint32_t)N, (uint32_t)K));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_gemm_sm100