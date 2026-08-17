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

// -------------------------------------------------------------------------
// Architecture Specific Inline PTX Helpers
// -------------------------------------------------------------------------

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

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tma_load_multicast_2d_cg2_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, uint16_t mask) {
    uint64_t cache_hint = 0;
    uint32_t ba = (uint32_t)__cvta_generic_to_shared(&bar[0]) & 0xFEFFFFFF; 
    asm volatile(
        "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint [%0], [%1, {%4, %5}], [%2], %3, %6;"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"(ba),
        "h"(mask), "r"(c0), "r"(c1), "l"(cache_hint) : "memory");
}

__device__ __forceinline__ void tma_load_2d_cg2_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    uint32_t sa = (uint32_t)__cvta_generic_to_shared(smem);
    uint32_t ba = (uint32_t)__cvta_generic_to_shared(&bar[0]) & 0xFEFFFFFF; 
    asm volatile(
        "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(sa), "l"((uint64_t)d), "r"(c0), "r"(c1), "r"(ba) : "memory");
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

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t sbo, uint32_t lbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)((addr >> 4) & 0x3FFF);
    d |= (uint64_t)((lbo >> 4) & 0x3FFF) << 16; 
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32; 
    d |= (uint64_t)1 << 46;   
    // SWIZZLE_NONE is 0 (bits 61-63)
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);           // c_format = FP32
    d |= (1u << 7);           // a_format = BF16
    d |= (1u << 10);          // b_format = BF16
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

// -------------------------------------------------------------------------
// Epilogue
// -------------------------------------------------------------------------

__device__ __forceinline__ void my_tmem_epilogue(
    uint32_t tmem_base,
    __nv_bfloat16* D, uint8_t* smem_out_ptr,
    uint32_t M, uint32_t N, uint32_t m_block, uint32_t n_block,
    uint32_t BM, uint32_t BN) {
    
    __nv_bfloat16* smem_out = (__nv_bfloat16*)smem_out_ptr;
    // Pad to 136 elements to gracefully bypass massive 32-way bank conflicts 
    uint32_t stride = BN + 8; 
    
    // TMEM -> SMEM
    for (uint32_t col = 0; col < BN; col += 4) {
        uint32_t r0, r1, r2, r3;
        uint32_t addr = tmem_base + col; 
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(addr));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        uint32_t base = threadIdx.x * stride + col;
        smem_out[base + 0] = __float2bfloat16(__uint_as_float(r0));
        smem_out[base + 1] = __float2bfloat16(__float2bfloat16(__uint_as_float(r1))); // Cast wrapper fallback
        smem_out[base + 1] = __float2bfloat16(__uint_as_float(r1));
        smem_out[base + 2] = __float2bfloat16(__uint_as_float(r2));
        smem_out[base + 3] = __float2bfloat16(__uint_as_float(r3));
    }
    __syncthreads();
    
    // SMEM -> GMEM (Vectorized & coalesced writes)
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t num_steps = (BM + 3) / 4;
    for (uint32_t step = 0; step < num_steps; ++step) {
        uint32_t row = step * 4 + warp_id;
        if (row >= BM) continue;
        uint32_t global_row = m_block * BM + row;
        uint32_t col_start = lane_id * 4;
        uint32_t global_col = n_block * BN + col_start;
        if (global_row < M && global_col + 3 < N) {
            uint2 data = *reinterpret_cast<uint2*>(&smem_out[row * stride + col_start]);
            *reinterpret_cast<uint2*>(D + (uint64_t)global_row * N + global_col) = data;
        }
    }
}

// -------------------------------------------------------------------------
// Main CUDA Kernel
// -------------------------------------------------------------------------

__global__ void gemm_kernel(const __grid_constant__ CUtensorMap tma_A, 
                            const __grid_constant__ CUtensorMap tma_B, 
                            __nv_bfloat16* C, int M, int N, int K) {
    extern __shared__ uint8_t smem_pool[];
    
    uintptr_t smem_base = (uintptr_t)smem_pool;
    uintptr_t aligned_base = (smem_base + 1023) & ~1023ULL;
    uint8_t* smem_A_ptr = (uint8_t*)aligned_base;
    uint8_t* smem_B_ptr = smem_A_ptr + 32768;
    uint64_t* mbar_tma = (uint64_t*)(smem_B_ptr + 32768);
    uint64_t* mbar_umma = mbar_tma + 2;
    uint32_t* tmem_c_ptr = (uint32_t*)(mbar_umma + 2);

    uint8_t* smem_A[2] = {smem_A_ptr, smem_A_ptr + 16384};
    uint8_t* smem_B[2] = {smem_B_ptr, smem_B_ptr + 16384};

    int cta_rank = cluster_rank_fn();
    int M_block = blockIdx.y;
    int N_cluster_block = blockIdx.x / 2;
    int N_block = N_cluster_block * 2 + cta_rank;
    
    int M_start = M_block * 128;
    int N_local = N_block * 128;

    cluster_sync_fn();
    if (threadIdx.x < 32) {
        tmem_alloc_fn(tmem_c_ptr, 128); 
    }
    cluster_sync_fn();
    uint32_t tmem_c = *tmem_c_ptr;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&mbar_tma[0], 1);
        init_smem_barrier_fn(&mbar_tma[1], 1);
        init_smem_barrier_fn(&mbar_umma[0], 1);
        init_smem_barrier_fn(&mbar_umma[1], 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    uint32_t tma_phase[2] = {0, 0};
    uint32_t umma_phase[2] = {0, 0};
    uint32_t A_bytes = 16384; 
    uint32_t B_bytes = 16384;
    uint32_t expected_tx = A_bytes * 2 + B_bytes * 2; 
    uint16_t mask = 0x3;

    for (int k = 0; k < K; k += 64) {
        int stage = (k / 64) % 2;

        if (k >= 128) {
            mbarrier_wait_fn(&mbar_umma[stage], umma_phase[stage]);
            umma_phase[stage] ^= 1;
        }

        if (threadIdx.x == 0) {
            if (cta_rank == 0) {
                mbarrier_arrive_and_expect_tx_fn(&mbar_tma[stage], expected_tx);
            }
        }
        
        cluster_sync_fn();

        if (threadIdx.x == 0) {
            if (cta_rank == 0) {
                tma_load_multicast_2d_cg2_fn(&tma_A, &mbar_tma[stage], smem_A[stage], k, M_start, mask);
                tma_load_2d_cg2_fn(&tma_B, &mbar_tma[stage], smem_B[stage], k, N_local);
            } else {
                tma_load_2d_cg2_fn(&tma_B, &mbar_tma[stage], smem_B[stage], k, N_local);
            }
        }

        if (cta_rank == 0) {
            mbarrier_wait_fn(&mbar_tma[stage], tma_phase[stage]);
            tma_phase[stage] ^= 1;

            fence_async_shared_fn();

            for (int k_step = 0; k_step < 4; ++k_step) {
                // Correctly passing SBO = 1024 and LBO = 16
                uint64_t desc_A = make_smem_desc_sm100_fn(smem_A[stage] + k_step * 32, 1024, 16);
                uint64_t desc_B = make_smem_desc_sm100_fn(smem_B[stage] + k_step * 32, 1024, 16);
                uint32_t idesc = make_instr_desc_fn(128, 256); 
                uint32_t accum = (k == 0 && k_step == 0) ? 0 : 1;
                
                if (threadIdx.x == 0) {
                    umma_f16_cg2_fn(tmem_c, desc_A, desc_B, idesc, accum);
                }
            }

            if (threadIdx.x == 0) {
                umma_commit_2sm_fn(&mbar_umma[stage]);
            }
        }
    }

    int last_k = K - 64;
    if (last_k >= 0) {
        int last_stage = (last_k / 64) % 2;
        mbarrier_wait_fn(&mbar_umma[last_stage], umma_phase[last_stage]);
        umma_phase[last_stage] ^= 1;
    }

    if (K >= 128) {
        int prev_k = K - 128;
        int prev_stage = (prev_k / 64) % 2;
        mbarrier_wait_fn(&mbar_umma[prev_stage], umma_phase[prev_stage]);
        umma_phase[prev_stage] ^= 1;
    }

    tcgen05_fence_after_fn();
    __syncthreads();

    my_tmem_epilogue(tmem_c, C, smem_A_ptr, M, N, M_block, N_block, 128, 128);
    __syncthreads();

    cluster_sync_fn();
    if (threadIdx.x < 32) {
        tmem_dealloc_fn(tmem_c, 128);
    }
}

// -------------------------------------------------------------------------
// Host Run Wrapper
// -------------------------------------------------------------------------

CUresult my_create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
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

namespace tvm_ffi_gemm_n7168_k5120 {

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(0);

    __nv_bfloat16* a_ptr = static_cast<__nv_bfloat16*>(A.data_ptr());
    __nv_bfloat16* b_ptr = static_cast<__nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* c_ptr = static_cast<__nv_bfloat16*>(C.data_ptr());

    CUtensorMap tma_A, tma_B;
    
    // Explicitly disabling Swizzle to safely iterate K pointers while adhering to exact 16B LBO offsets.
    if (my_create_tma_2d_descriptor_2B(
        &tma_A, a_ptr, K, M, 64, 128, 
        CU_TENSOR_MAP_SWIZZLE_NONE, 
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, 
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS) {
        fprintf(stderr, "TMA A descriptor failed\n"); exit(1);
    }
    
    if (my_create_tma_2d_descriptor_2B(
        &tma_B, b_ptr, K, N, 64, 128, 
        CU_TENSOR_MAP_SWIZZLE_NONE, 
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, 
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS) {
        fprintf(stderr, "TMA B descriptor failed\n"); exit(1);
    }

    dim3 block(128); 
    dim3 grid(N / 128, (M + 127) / 128);

    cudaLaunchConfig_t config = {};
    config.gridDim = grid;       
    config.blockDim = block;
    config.dynamicSmemBytes = 67000; 
    config.stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2; 
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 67000));
    CUDA_CHECK(cudaLaunchKernelEx(&config, gemm_kernel, tma_A, tma_B, c_ptr, (int)M, (int)N, (int)K));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_gemm_n7168_k5120::run);

} // namespace tvm_ffi_gemm_n7168_k5120