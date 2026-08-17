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

#define K_BLOCK 64
#define STAGES 4

struct SharedStorage {
    alignas(1024) __nv_bfloat16 A[STAGES][128 * 64]; 
    alignas(1024) __nv_bfloat16 B[STAGES][128 * 64];
    alignas(16) uint64_t mbar_tma[STAGES];
    alignas(16) uint64_t mbar_umma[STAGES];
};

__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r");
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

__device__ __forceinline__ void tma_load_2d_cg2_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    uint32_t sa = (uint32_t)__cvta_generic_to_shared(smem);
    uint32_t local_ba = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    uint32_t target_cta = cluster_rank_fn() & ~1; // Force peer 0
    uint32_t remote_ba;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote_ba) : "r"(local_ba), "r"(target_cta));
    asm volatile(
        "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(sa), "l"((uint64_t)d), "r"(c0), "r"(c1), "r"(remote_ba) : "memory");
}

__device__ __forceinline__ void tma_store_2d_fn(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0, {%2, %3}], [%1];"
        :: "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "r"(c0), "r"(c1) : "memory");
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
    d |= (uint64_t)2 << 61;   // SWIZZLE_128B
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

__device__ __forceinline__ void my_tmem_epilogue(
    __nv_bfloat16* smem_out, uint32_t tmem_addr,
    uint32_t BN) {
    
    uint32_t wg_id = threadIdx.x / 128; 
    uint32_t tid_in_wg = threadIdx.x % 128; 
    
    uint32_t col_start = wg_id * 128;
    uint32_t col_end = col_start + 128;
    
    for (uint32_t col = col_start; col < col_end; col += 8) {
        uint32_t r[8];
        uint32_t taddr = tmem_addr + col;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                     : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),
                       "=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]) : "r"(taddr));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        uint32_t p0 = pack_bf16_fn(r[0], r[1]);
        uint32_t p1 = pack_bf16_fn(r[2], r[3]);
        uint32_t p2 = pack_bf16_fn(r[4], r[5]);
        uint32_t p3 = pack_bf16_fn(r[6], r[7]);
        
        uint32_t x_chunk = col / 8;
        uint32_t swizzled_x_chunk = x_chunk ^ (tid_in_wg % 8);
        uint32_t swizzled_col = swizzled_x_chunk * 8;
        uint32_t swizzled_base = tid_in_wg * BN + swizzled_col;
        
        *reinterpret_cast<uint4*>(&smem_out[swizzled_base]) = make_uint4(p0, p1, p2, p3);
    }
}

__global__ void gemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    const __grid_constant__ CUtensorMap tma_C,
    uint32_t M, uint32_t N, uint32_t K
) {
    uint32_t pair_idx = blockIdx.x / 2;
    uint32_t m_idx = pair_idx * 256 + (cluster_rank_fn() % 2) * 128;
    uint32_t n_idx = blockIdx.y * 256;
    
    extern __shared__ uint8_t raw_smem[];
    uintptr_t smem_ptr = (uintptr_t)raw_smem;
    smem_ptr = (smem_ptr + 1023) & ~1023;
    SharedStorage* smem = (SharedStorage*)smem_ptr;
    
    __shared__ uint32_t tmem_addr_smem;
    if (threadIdx.x < 32) {
        tmem_alloc_fn(&tmem_addr_smem, 256);
    }
    
    if (threadIdx.x == 0) {
        for (int i = 0; i < STAGES; ++i) {
            init_smem_barrier_fn(&smem->mbar_tma[i], 1);
            init_smem_barrier_fn(&smem->mbar_umma[i], 1);
        }
    }
    __syncthreads();
    
    fence_smem_barrier_init_fn();
    cluster_sync_fn(); 
    
    uint32_t tmem_addr = tmem_addr_smem;
    int phase_tma[STAGES] = {0};
    int phase_umma[STAGES] = {0};
    uint32_t local_n_idx = n_idx + (cluster_rank_fn() % 2) * 128;
    
    // Prologue
    for (int i = 0; i < STAGES - 1; ++i) {
        if (threadIdx.x == 0) {
            if (cluster_rank_fn() == 0) mbarrier_arrive_and_expect_tx_fn(&smem->mbar_tma[i], 65536);
            tma_load_2d_cg2_fn(&tma_A, &smem->mbar_tma[i], &smem->A[i][0], i * K_BLOCK, m_idx);
        } else if (threadIdx.x == 1) {
            tma_load_2d_cg2_fn(&tma_B, &smem->mbar_tma[i], &smem->B[i][0], i * K_BLOCK, local_n_idx);
        }
    }
    
    int s_load = STAGES - 1;
    int s_compute = 0;
    uint32_t K_steps = (K + K_BLOCK - 1) / K_BLOCK;
    
    for (int step = 0; step < K_steps; ++step) {
        if (step + STAGES - 1 < K_steps) {
            if (step > 0) {
                if (threadIdx.x == 0) {
                    mbarrier_wait_fn(&smem->mbar_umma[s_load], phase_umma[s_load]);
                    phase_umma[s_load] ^= 1;
                }
                __syncthreads();
            }
            if (threadIdx.x == 0) {
                if (cluster_rank_fn() == 0) mbarrier_arrive_and_expect_tx_fn(&smem->mbar_tma[s_load], 65536);
                tma_load_2d_cg2_fn(&tma_A, &smem->mbar_tma[s_load], &smem->A[s_load][0], (step + STAGES - 1) * K_BLOCK, m_idx);
            } else if (threadIdx.x == 1) {
                tma_load_2d_cg2_fn(&tma_B, &smem->mbar_tma[s_load], &smem->B[s_load][0], (step + STAGES - 1) * K_BLOCK, local_n_idx);
            }
            s_load = (s_load + 1) % STAGES;
        }
        
        if (threadIdx.x == 0) {
            if (cluster_rank_fn() == 0) {
                mbarrier_wait_fn(&smem->mbar_tma[s_compute], phase_tma[s_compute]);
                phase_tma[s_compute] ^= 1;
                
                uint32_t idesc = make_instr_desc_fn(256, 256);
                uint64_t desc_a_base = make_smem_desc_sm100_fn((uint8_t*)&smem->A[s_compute][0], 1024);
                uint64_t desc_b_base = make_smem_desc_sm100_fn((uint8_t*)&smem->B[s_compute][0], 1024);
                
                for (int k_iter = 0; k_iter < 4; ++k_iter) {
                    uint64_t desc_a = desc_a_base + k_iter * 2;
                    uint64_t desc_b = desc_b_base + k_iter * 2;
                    uint32_t accum = (step == 0 && k_iter == 0) ? 0 : 1;
                    umma_f16_cg2_fn(tmem_addr, desc_a, desc_b, idesc, accum);
                }
                umma_commit_2sm_fn(&smem->mbar_umma[s_compute]);
            }
        }
        
        s_compute = (s_compute + 1) % STAGES;
    }
    
    if (threadIdx.x == 0) {
        int last_compute = (s_compute + STAGES - 1) % STAGES;
        mbarrier_wait_fn(&smem->mbar_umma[last_compute], phase_umma[last_compute]);
    }
    __syncthreads();
    
    my_tmem_epilogue((__nv_bfloat16*)smem, tmem_addr, 256);
    __syncthreads();
    
    cluster_sync_fn();
    if (threadIdx.x < 32) {
        tmem_dealloc_fn(tmem_addr, 256);
    }
    
    if (threadIdx.x == 0) {
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        tma_store_2d_fn(&tma_C, smem, n_idx, m_idx);
        tma_store_commit_fn();
        tma_store_wait_fn<0>();
    }
    __syncthreads();
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
    
    CUtensorMap tma_A, tma_B, tma_C;
    
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
    
    CUresult resC = create_tma_2d_descriptor_2B(&tma_C, c_ptr, N, M, 256, 128, 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, 
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (resC != CUDA_SUCCESS) {
        fprintf(stderr, "TMA C failed\n");
        exit(1);
    }
    
    int grid_x = (M + 255) / 256 * 2;
    int grid_y = (N + 255) / 256;
    dim3 grid(grid_x, grid_y, 1);
    dim3 block(256, 1, 1);
    
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
    
    CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, config.dynamicSmemBytes));
    CUDA_CHECK(cudaLaunchKernelEx(&config, gemm_kernel, tma_A, tma_B, tma_C, M, N, K));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda