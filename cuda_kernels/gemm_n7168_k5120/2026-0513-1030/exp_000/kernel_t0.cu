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

namespace tvm_ffi_example_cuda {

struct SharedStorage {
    alignas(1024) __nv_bfloat16 A[4][128 * 64]; 
    alignas(1024) __nv_bfloat16 B[4][128 * 64];
    alignas(8) uint64_t mbar_tma[4];
    alignas(8) uint64_t mbar_umma[4];
};

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

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
                 :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void umma_f16_cg2_fn(uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc, uint32_t accum) {
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
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)((addr >> 4) & 0x3FFF);
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    uint32_t base_offset = (addr >> 7) & 0x7;
    d |= (uint64_t)base_offset << 49;
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

__device__ __forceinline__ void my_tmem_epilogue(
    __nv_bfloat16* D, __nv_bfloat16* smem_out, uint32_t tmem_addr,
    uint32_t M, uint32_t N, uint32_t m_idx, uint32_t n_idx,
    uint32_t BM, uint32_t BN) {
    
    // Phase 1: TMEM -> SMEM
    for (uint32_t col = 0; col < BN; col += 4) {
        uint32_t r0, r1, r2, r3;
        uint32_t taddr = tmem_addr + col;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(taddr));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        uint32_t base = threadIdx.x * BN + col;
        smem_out[base + 0] = __float2bfloat16(__uint_as_float(r0));
        smem_out[base + 1] = __float2bfloat16(__uint_as_float(r1));
        smem_out[base + 2] = __float2bfloat16(__uint_as_float(r2));
        smem_out[base + 3] = __float2bfloat16(__uint_as_float(r3));
    }
    __syncthreads();
    
    // Phase 2: SMEM -> Global
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t num_steps = (BM + 3) / 4;
    
    for (uint32_t step = 0; step < num_steps; ++step) {
        uint32_t row = step * 4 + warp_id;
        if (row >= BM) continue;
        uint32_t global_row = m_idx + row;
        uint32_t col_start = lane_id * 8;
        uint32_t global_col = n_idx + col_start;
        
        if (global_row < M) {
            if (global_col + 7 < N) {
                float4 data = *reinterpret_cast<float4*>(&smem_out[row * BN + col_start]);
                *reinterpret_cast<float4*>(D + (uint64_t)global_row * N + global_col) = data;
            } else {
                for (int i = 0; i < 8; ++i) {
                    if (global_col + i < N) {
                        D[(uint64_t)global_row * N + global_col + i] = smem_out[row * BN + col_start + i];
                    }
                }
            }
        }
    }
}

__global__ void gemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    __nv_bfloat16* D,
    uint32_t M, uint32_t N, uint32_t K
) {
    uint32_t pair_idx = blockIdx.x / 2;
    uint32_t m_idx = pair_idx * 256 + (cluster_rank_fn() % 2) * 128;
    uint32_t n_idx = blockIdx.y * 256;
    
    extern __shared__ uint8_t raw_smem[];
    uintptr_t smem_ptr = (uintptr_t)raw_smem;
    smem_ptr = (smem_ptr + 1023) & ~1023;
    SharedStorage* smem = (SharedStorage*)smem_ptr;
    
    __shared__ uint32_t tmem_addr;
    if (threadIdx.x == 0) {
        tmem_alloc_fn(&tmem_addr, 256);
        for (int i = 0; i < 4; ++i) {
            init_smem_barrier_fn(&smem->mbar_tma[i], 1);
            init_smem_barrier_fn(&smem->mbar_umma[i], 1);
        }
    }
    fence_smem_barrier_init_fn();
    __syncthreads();
    
    int phase_tma[4] = {0, 0, 0, 0};
    int phase_umma[4] = {0, 0, 0, 0};
    
    uint32_t local_n_idx = n_idx + (cluster_rank_fn() % 2) * 128;
    
    // Prologue
    for (int i = 0; i < 3; ++i) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(&smem->mbar_tma[i], 32768);
            tma_load_2d_fn(&tma_A, &smem->mbar_tma[i], &smem->A[i], i * 64, m_idx);
            tma_load_2d_fn(&tma_B, &smem->mbar_tma[i], &smem->B[i], i * 64, local_n_idx);
        }
    }
    
    int s_load = 3;
    int s_compute = 0;
    
    uint32_t K_steps = (K + 63) / 64;
    
    for (int step = 0; step < K_steps; ++step) {
        if (step + 3 < K_steps) {
            if (step > 0) {
                if (threadIdx.x == 0) {
                    mbarrier_wait_fn(&smem->mbar_umma[s_load], phase_umma[s_load]);
                    phase_umma[s_load] ^= 1;
                }
            }
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(&smem->mbar_tma[s_load], 32768);
                tma_load_2d_fn(&tma_A, &smem->mbar_tma[s_load], &smem->A[s_load], (step + 3) * 64, m_idx);
                tma_load_2d_fn(&tma_B, &smem->mbar_tma[s_load], &smem->B[s_load], (step + 3) * 64, local_n_idx);
            }
            s_load = (s_load + 1) % 4;
        }
        
        if (threadIdx.x == 0) {
            mbarrier_wait_fn(&smem->mbar_tma[s_compute], phase_tma[s_compute]);
            phase_tma[s_compute] ^= 1;
        }
        
        cluster_sync_fn();
        
        if (cluster_rank_fn() == 0 && threadIdx.x == 0) {
            for (int k_iter = 0; k_iter < 4; ++k_iter) {
                uint32_t idesc = make_instr_desc_fn(256, 256);
                uint64_t desc_a = make_smem_desc_sm100_fn((uint8_t*)&smem->A[s_compute] + k_iter * 32, 1024);
                uint64_t desc_b = make_smem_desc_sm100_fn((uint8_t*)&smem->B[s_compute] + k_iter * 32, 1024);
                uint32_t accum = (step == 0 && k_iter == 0) ? 0 : 1;
                umma_f16_cg2_fn(tmem_addr, desc_a, desc_b, idesc, accum);
            }
            umma_commit_2sm_fn(&smem->mbar_umma[s_compute]);
        }
        
        s_compute = (s_compute + 1) % 4;
    }
    
    if (threadIdx.x == 0) {
        int last_compute = (s_compute + 3) % 4;
        mbarrier_wait_fn(&smem->mbar_umma[last_compute], phase_umma[last_compute]);
    }
    __syncthreads();
    
    my_tmem_epilogue(D, (__nv_bfloat16*)smem, tmem_addr, M, N, m_idx, n_idx, 128, 256);
    
    __syncthreads();
    if (threadIdx.x == 0) {
        tmem_dealloc_fn(tmem_addr, 256);
    }
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

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(0);
    
    __nv_bfloat16* a_ptr = static_cast<__nv_bfloat16*>(A.data_ptr());
    __nv_bfloat16* b_ptr = static_cast<__nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* c_ptr = static_cast<__nv_bfloat16*>(C.data_ptr());
    
    CUtensorMap tma_A, tma_B;
    
    CUresult resA = create_tma_2d_descriptor_2B(&tma_A, a_ptr, K, M, 64, 128, 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, 
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (resA != CUDA_SUCCESS) {
        fprintf(stderr, "TMA A failed\n");
        exit(1);
    }
    
    CUresult resB = create_tma_2d_descriptor_2B(&tma_B, b_ptr, K, N, 64, 128, 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, 
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (resB != CUDA_SUCCESS) {
        fprintf(stderr, "TMA B failed\n");
        exit(1);
    }
    
    int grid_x = (M + 255) / 256 * 2;
    int grid_y = (N + 255) / 256;
    dim3 grid(grid_x, grid_y, 1);
    dim3 block(128, 1, 1);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = sizeof(SharedStorage) + 1024;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, gemm_kernel, tma_A, tma_B, c_ptr, M, N, K));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda