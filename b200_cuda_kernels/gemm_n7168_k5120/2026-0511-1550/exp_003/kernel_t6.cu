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

// ---------------------------------------------------------------------------------
// PTX Helper Functions
// ---------------------------------------------------------------------------------

__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

__device__ __forceinline__ void cluster_sync_fn() {
    asm volatile("barrier.cluster.arrive;\nbarrier.cluster.wait;\n" ::: "memory");
}

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
}

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_arrive_fn(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])) : "memory");
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

inline CUresult my_create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
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

__device__ __forceinline__ void prefetch_tma_descriptor_fn(const CUtensorMap* d) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(d) : "memory");
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
    // Stripped L2 cache hint to avoid undefined behavior
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%4, %5}], [%2], %3;"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "h"(mask), "r"(c0), "r"(c1) : "memory");
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
    d |= (1u << 4);           // dtype = F32
    d |= (1u << 7);           // atype = BF16
    d |= (1u << 10);          // btype = BF16
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

// ---------------------------------------------------------------------------------

constexpr uint32_t BM = 128; 
constexpr uint32_t BN = 256; 
constexpr uint32_t BK = 64;
constexpr uint32_t STAGES = 2;

struct SharedStorage {
    alignas(1024) __nv_bfloat16 A[STAGES][BM * BK];
    alignas(1024) __nv_bfloat16 B[STAGES][BN * BK];
    alignas(1024) uint64_t mbar_tma[STAGES];
    alignas(1024) uint64_t mbar_umma[STAGES];
    uint32_t tmem_addr_smem;
};

__global__ __launch_bounds__(128) void gemm_sm100_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    __nv_bfloat16* D,
    uint32_t M, uint32_t N, uint32_t K) 
{
    setmaxnreg_inc_sync_fn<256>();

    extern __shared__ uint8_t smem_bytes[];
    uintptr_t smem_ptr = reinterpret_cast<uintptr_t>(smem_bytes);
    smem_ptr = (smem_ptr + 1023) & ~1023ULL; 
    SharedStorage* smem = reinterpret_cast<SharedStorage*>(smem_ptr);

    uint32_t cluster_rank = cluster_rank_fn();
    
    // Explicit chunk mapping avoids runtime blockIdx variance mapping bugs
    uint32_t cluster_group_x = blockIdx.x / 2;
    uint32_t my_m_offset = (cluster_group_x * 2 + cluster_rank) * BM;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&smem->mbar_tma[0], 1);
        init_smem_barrier_fn(&smem->mbar_tma[1], 1);
        init_smem_barrier_fn(&smem->mbar_umma[0], 1);
        init_smem_barrier_fn(&smem->mbar_umma[1], 1);
        
        fence_smem_barrier_init_fn();

        mbarrier_arrive_fn(&smem->mbar_umma[0]);
        mbarrier_arrive_fn(&smem->mbar_umma[1]);
    }
    cluster_sync_fn();

    if (threadIdx.x < 32) {
        tmem_alloc_fn(&smem->tmem_addr_smem, BN);
    }
    __syncthreads();
    uint32_t tmem_addr = smem->tmem_addr_smem;

    uint32_t tx_bytes = (BM * BK * 2) + (BN * BK * 2);

    if (threadIdx.x == 0) {
        prefetch_tma_descriptor_fn(&tma_A);
        prefetch_tma_descriptor_fn(&tma_B);
    }

    int num_stages = K / BK; 
    int phase_tma[STAGES] = {0};
    int phase_umma[STAGES] = {0};

    // Prologue: Load Stage 0
    if (threadIdx.x == 0) {
        mbarrier_wait_fn(&smem->mbar_umma[0], phase_umma[0]);
        mbarrier_arrive_and_expect_tx_fn(&smem->mbar_tma[0], tx_bytes);
    }
    cluster_sync_fn();

    if (threadIdx.x == 0) {
        tma_load_2d_fn(&tma_A, &smem->mbar_tma[0], smem->A[0], 0, my_m_offset);
        if (cluster_rank == 0) {
            tma_load_multicast_2d_fn(&tma_B, &smem->mbar_tma[0], smem->B[0], 0, blockIdx.y * BN, 3);
        }
    }

    uint32_t idesc = make_instr_desc_fn(BM * 2, BN); 

    for (int k = 0; k < num_stages; ++k) {
        int stage = k % STAGES;
        int next_stage = (k + 1) % STAGES;

        mbarrier_wait_fn(&smem->mbar_tma[stage], phase_tma[stage]);
        cluster_sync_fn();

        if (k + 1 < num_stages) {
            if (threadIdx.x == 0) {
                mbarrier_wait_fn(&smem->mbar_umma[next_stage], phase_umma[next_stage]);
                mbarrier_arrive_and_expect_tx_fn(&smem->mbar_tma[next_stage], tx_bytes);
            }
            cluster_sync_fn();
            
            if (threadIdx.x == 0) {
                tma_load_2d_fn(&tma_A, &smem->mbar_tma[next_stage], smem->A[next_stage], (k + 1) * BK, my_m_offset);
                if (cluster_rank == 0) {
                    tma_load_multicast_2d_fn(&tma_B, &smem->mbar_tma[next_stage], smem->B[next_stage], (k + 1) * BK, blockIdx.y * BN, 3);
                }
            }
        }

        fence_async_shared_fn();

        uint64_t desc_a = make_smem_desc_sm100_fn(smem->A[stage], 1024);
        uint64_t desc_b = make_smem_desc_sm100_fn(smem->B[stage], 1024);

        if (threadIdx.x == 0 && cluster_rank == 0) {
            uint32_t accum = (k == 0) ? 0 : 1;
            tcgen05_fence_after_fn();
            umma_f16_cg2_fn(tmem_addr, desc_a, desc_b, idesc, accum);
            umma_commit_2sm_fn(&smem->mbar_umma[stage]);
        }

        phase_tma[stage] ^= 1;
        phase_umma[stage] ^= 1;
    }

    int last_stage = (num_stages - 1) % STAGES;
    mbarrier_wait_fn(&smem->mbar_umma[last_stage], phase_umma[last_stage]); 
    __syncthreads();

    // Epilogue
    __nv_bfloat16* smem_out = reinterpret_cast<__nv_bfloat16*>(smem);
    
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

    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t num_steps = (BM + 3) / 4;
    for (uint32_t step = 0; step < num_steps; ++step) {
        uint32_t row = step * 4 + warp_id;
        if (row >= BM) continue;
        uint32_t global_row = my_m_offset + row;
        if (global_row >= M) continue;

        for (uint32_t col_chunk = 0; col_chunk < BN; col_chunk += 128) {
            uint32_t col_start = col_chunk + lane_id * 4;
            uint32_t global_col = blockIdx.y * BN + col_start;
            if (global_col + 3 < N) {
                uint2 data = *reinterpret_cast<uint2*>(&smem_out[row * BN + col_start]);
                *reinterpret_cast<uint2*>(D + (uint64_t)global_row * N + global_col) = data;
            } else {
                for(int i = 0; i < 4; ++i) {
                    if (global_col + i < N) {
                        D[(uint64_t)global_row * N + global_col + i] = smem_out[row * BN + col_start + i];
                    }
                }
            }
        }
    }

    __syncthreads();
    cluster_sync_fn();
    
    if (threadIdx.x < 32) {
        tmem_dealloc_fn(tmem_addr, BN);
    }
}

namespace tvm_ffi_gemm_sm100 {

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(0); 

    __nv_bfloat16* a_ptr = static_cast<__nv_bfloat16*>(A.data_ptr());
    __nv_bfloat16* b_ptr = static_cast<__nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* c_ptr = static_cast<__nv_bfloat16*>(C.data_ptr());

    CUtensorMap tma_A;
    CUresult res = my_create_tma_2d_descriptor_2B(
        &tma_A, a_ptr, K, M, BK, BM, 
        CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, 
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "TMA A descriptor creation failed\n");
    }

    CUtensorMap tma_B;
    res = my_create_tma_2d_descriptor_2B(
        &tma_B, b_ptr, K, N, BK, BN, 
        CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, 
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "TMA B descriptor creation failed\n");
    }

    dim3 block(128);
    dim3 cluster(2, 1, 1);
    dim3 grid(((M + BM * 2 - 1) / (BM * 2)) * 2, (N + BN - 1) / BN, 1);

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute((const void*)gemm_sm100_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(SharedStorage) + 1024));

    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = sizeof(SharedStorage) + 1024;
    config.stream = stream;

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = cluster.x;
    attrs[0].val.clusterDim.y = cluster.y;
    attrs[0].val.clusterDim.z = cluster.z;
    config.attrs = attrs;
    config.numAttrs = 1;

    CUDA_CHECK(cudaLaunchKernelEx(&config, gemm_sm100_kernel, tma_A, tma_B, c_ptr, (uint32_t)M, (uint32_t)N, (uint32_t)K));
}

} // namespace tvm_ffi_gemm_sm100

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_gemm_sm100::run);