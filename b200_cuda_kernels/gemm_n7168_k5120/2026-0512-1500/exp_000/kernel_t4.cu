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

__device__ __forceinline__ void prefetch_tma_descriptor_fn(const CUtensorMap* d) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(d) : "memory");
}

__device__ __forceinline__ void tcgen05_fence_after_fn() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_before_fn() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
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
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)((addr >> 4) & 0x3FFF);
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32; 
    d |= (uint64_t)1 << 46;   
    
    uint64_t base_offset = (addr >> 7) & 0x7;
    d |= (base_offset << 49);
    
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
    __nv_bfloat16* D, __nv_bfloat16* smem_out, uint32_t tmem_addr,
    uint32_t M, uint32_t N, uint32_t m_base, uint32_t n_base,
    uint32_t BM, uint32_t BN) {
    
    uint32_t PADDED_BN = BN + 8; // Pad rows to avoid SMEM bank conflicts
    
    // Phase 1: TMEM -> SMEM
    for (uint32_t col = 0; col < BN; col += 4) {
        uint32_t r0, r1, r2, r3;
        uint32_t addr = tmem_addr + col;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(addr));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        uint32_t base = threadIdx.x * PADDED_BN + col;
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
        uint32_t global_row = m_base + row;
        if (global_row >= M) continue;
        
        for (int c_iter = 0; c_iter < BN; c_iter += 128) {
            uint32_t col = c_iter + lane_id * 4;
            uint32_t global_col = n_base + col;
            for(int i = 0; i < 4; ++i) {
                if (global_col + i < N) {
                    D[(uint64_t)global_row * N + global_col + i] = smem_out[row * PADDED_BN + col + i];
                }
            }
        }
    }
}

__global__ void __launch_bounds__(128) gemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    __nv_bfloat16* D,
    uint32_t M, uint32_t N, uint32_t K) {
    
    extern __shared__ __align__(128) uint8_t raw_smem[];
    uint64_t smem_ptr = (uint64_t)raw_smem;
    uint64_t offset = (1024 - (smem_ptr % 1024)) % 1024;
    uint8_t* smem_buf = raw_smem + offset;

    // Both A and B are fully staged in SMEM for both CTAs (256x64 tiles)
    __nv_bfloat16* A_smem = (__nv_bfloat16*)smem_buf; // 98304 bytes (3 stages)
    __nv_bfloat16* B_smem = (__nv_bfloat16*)(smem_buf + 98304); // 98304 bytes (3 stages)
    uint64_t* full_barriers = (uint64_t*)(smem_buf + 196608);
    uint64_t* empty_barriers = (uint64_t*)(smem_buf + 196632);
    uint64_t* epilogue_barrier = (uint64_t*)(smem_buf + 196656);

    int cta_rank = cluster_rank_fn();
    
    uint32_t cluster_idx_x = blockIdx.x / 2;
    uint32_t n_block = blockIdx.y;
    
    uint32_t cluster_m_base = cluster_idx_x * 256;
    uint32_t cta_m_base = cluster_m_base + cta_rank * 128;
    uint32_t n_base = n_block * 256;

    if (threadIdx.x == 0) {
        for (int i = 0; i < 3; ++i) {
            init_smem_barrier_fn(&full_barriers[i], 1);
            init_smem_barrier_fn(&empty_barriers[i], 1);
        }
        init_smem_barrier_fn(&epilogue_barrier[0], 1);
        fence_smem_barrier_init_fn();
    }
    cluster_sync_fn();

    int num_k_tiles = K / 64;

    if (threadIdx.x == 0) {
        prefetch_tma_descriptor_fn(&tma_A);
        prefetch_tma_descriptor_fn(&tma_B);
    }

    int producer_stage = 0;
    int empty_phase = 0;

    // A single stage expects 4 loads to complete (CTA0 A, CTA1 A, CTA0 B, CTA1 B)
    // 128 * 64 * 2 (bytes) * 4 = 65536 bytes expected on full_barriers[i]
    for (int i = 0; i < 3; ++i) {
        if (i < num_k_tiles) {
            if (threadIdx.x == 0) {
                uint32_t k_coord = i * 64;
                tma_load_multicast_2d_fn(&tma_A, &full_barriers[i], 
                    A_smem + i * 16384 + cta_rank * 8192, 
                    k_coord, cta_m_base, 0x3);
                tma_load_multicast_2d_fn(&tma_B, &full_barriers[i], 
                    B_smem + i * 16384 + cta_rank * 8192, 
                    k_coord, n_base + cta_rank * 128, 0x3);
                mbarrier_arrive_and_expect_tx_fn(&full_barriers[i], 65536);
            }
        }
    }

    __shared__ uint32_t tmem_addr;
    if (threadIdx.x < 32) {
        tmem_alloc_fn(&tmem_addr, 256);
    }
    __syncthreads();

    uint32_t idesc = make_instr_desc_fn(256, 256);
    int consumer_stage = 0;
    int consumer_phase = 0;

    for (int step = 0; step < num_k_tiles; ++step) {
        if (threadIdx.x == 0) {
            mbarrier_wait_fn(&full_barriers[consumer_stage], consumer_phase);
        }
        __syncthreads();

        // Only one CTA triggers the shared TC operations
        if (cta_rank == 0 && threadIdx.x == 0) {
            uint64_t desc_a = make_smem_desc_sm100_fn(A_smem + consumer_stage * 16384, 1024);
            uint64_t desc_b = make_smem_desc_sm100_fn(B_smem + consumer_stage * 16384, 1024);
            
            tcgen05_fence_after_fn();
            for (int k_step = 0; k_step < 4; ++k_step) {
                uint32_t is_accum = (step > 0 || k_step > 0) ? 1 : 0;
                umma_f16_cg2_fn(tmem_addr, desc_a, desc_b, idesc, is_accum);
                desc_a += 2; // Advance by 16 elements along K 
                desc_b += 2;
            }
            tcgen05_fence_before_fn();

            umma_commit_2sm_fn(&empty_barriers[consumer_stage]);
        }

        int next_step = step + 3;
        if (next_step < num_k_tiles) {
            if (threadIdx.x == 0) {
                mbarrier_wait_fn(&empty_barriers[producer_stage], empty_phase);
                
                uint32_t k_coord = next_step * 64;
                tma_load_multicast_2d_fn(&tma_A, &full_barriers[producer_stage], 
                    A_smem + producer_stage * 16384 + cta_rank * 8192, 
                    k_coord, cta_m_base, 0x3);
                tma_load_multicast_2d_fn(&tma_B, &full_barriers[producer_stage], 
                    B_smem + producer_stage * 16384 + cta_rank * 8192, 
                    k_coord, n_base + cta_rank * 128, 0x3);
                mbarrier_arrive_and_expect_tx_fn(&full_barriers[producer_stage], 65536);
            }
        }

        if (threadIdx.x == 0) {
            consumer_stage++;
            if (consumer_stage == 3) {
                consumer_stage = 0;
                consumer_phase ^= 1;
            }

            if (next_step < num_k_tiles) {
                producer_stage++;
                if (producer_stage == 3) {
                    producer_stage = 0;
                    empty_phase ^= 1;
                }
            }
        }
        __syncthreads();
    }

    if (cta_rank == 0 && threadIdx.x == 0) {
        umma_commit_2sm_fn(&epilogue_barrier[0]);
    }
    if (threadIdx.x == 0) {
        mbarrier_wait_fn(&epilogue_barrier[0], 0);
    }
    __syncthreads();

    epilogue(D, B_smem, tmem_addr, M, N, cta_m_base, n_base, 128, 256);

    __syncthreads();
    cluster_sync_fn();
    if (threadIdx.x < 32) {
        tmem_dealloc_fn(tmem_addr, 256);
    }
    __syncthreads();
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

namespace tvm_ffi_example_cuda {

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
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    CUresult resB = create_tma_2d_descriptor_2B(&tma_B, b_ptr, K, N, 64, 128, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    
    if (resA != CUDA_SUCCESS || resB != CUDA_SUCCESS) {
        fprintf(stderr, "TMA descriptor creation failed\n");
        exit(1);
    }

    int grid_x = ((M + 255) / 256) * 2;
    dim3 blocks(grid_x, (N + 255) / 256, 1);
    int threads = 128;
    int smem_bytes = 198000;

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

    cudaLaunchConfig_t config = {};
    config.gridDim = blocks;
    config.blockDim = dim3(threads);
    config.dynamicSmemBytes = smem_bytes;
    config.stream = stream;

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    CUDA_CHECK(cudaLaunchKernelEx(&config, gemm_kernel, tma_A, tma_B, c_ptr, (uint32_t)M, (uint32_t)N, (uint32_t)K));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

}  // namespace tvm_ffi_example_cuda