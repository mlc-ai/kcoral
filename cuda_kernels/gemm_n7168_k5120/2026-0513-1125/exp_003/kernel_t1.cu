#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <stdio.h>
#include <cstdint>
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

// Helper functions for SM100
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

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
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
    d |= (uint64_t)1 << 46;   // version = 1
    d |= (uint64_t)2 << 61;   // layout_type = SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);           // c_format = FP32
    d |= (1u << 7);           // a_format = BF16
    d |= (1u << 10);          // b_format = BF16
    d |= ((N / 8) << 17);     // n_dim
    d |= ((M / 16) << 24);    // m_dim
    return d;
}

__device__ __forceinline__ void my_tmem_epilogue_coalesced(
    __nv_bfloat16* D, __nv_bfloat16* smem_out,
    uint32_t M, uint32_t N, uint32_t m_block_start, uint32_t n_block_start,
    uint32_t BM, uint32_t BN, uint32_t tmem_c) {
    
    uint32_t BN_stride = BN + 8; // padding to avoid bank conflicts

    // Phase 1: TMEM -> SMEM
    for (uint32_t col = 0; col < BN; col += 4) {
        uint32_t taddr = tmem_c + col;
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(taddr));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        uint32_t base = threadIdx.x * BN_stride + col;
        smem_out[base + 0] = __float2bfloat16(__uint_as_float(r0));
        smem_out[base + 1] = __float2bfloat16(__uint_as_float(r1));
        smem_out[base + 2] = __float2bfloat16(__uint_as_float(r2));
        smem_out[base + 3] = __float2bfloat16(__uint_as_float(r3));
    }
    __syncthreads();
    
    // Phase 2: SMEM -> Global (coalesced)
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t num_steps = (BM + 3) / 4;
    for (uint32_t step = 0; step < num_steps; ++step) {
        uint32_t row = step * 4 + warp_id;
        if (row >= BM) continue;
        uint32_t global_row = m_block_start + row;
        
        for (uint32_t c_step = 0; c_step < BN; c_step += 128) {
            uint32_t col_start = c_step + lane_id * 4;
            uint32_t global_col = n_block_start + col_start;
            if (global_row < M && global_col < N) {
                if (global_col + 3 < N) {
                    uint2 data = *reinterpret_cast<uint2*>(&smem_out[row * BN_stride + col_start]);
                    *reinterpret_cast<uint2*>(D + (uint64_t)global_row * N + global_col) = data;
                } else {
                    for(int i = 0; i < 4; ++i) {
                        if (global_col + i < N) {
                            D[(uint64_t)global_row * N + global_col + i] = smem_out[row * BN_stride + col_start + i];
                        }
                    }
                }
            }
        }
    }
}

__global__ void __launch_bounds__(128, 1) gemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    __nv_bfloat16* C,
    uint32_t M, uint32_t N, uint32_t K)
{
    setmaxnreg_inc_sync_fn<256>();

    uint32_t rank = cluster_rank_fn();
    uint32_t cluster_idx_x = blockIdx.x / 2;
    uint32_t cluster_idx_y = blockIdx.y;
    
    uint32_t n_block_start = cluster_idx_x * 256;
    uint32_t m_block_start = cluster_idx_y * 256;
    
    uint32_t my_m = m_block_start + rank * 128;
    uint32_t my_n_b = n_block_start + rank * 128;

    uint32_t safe_my_m = (my_m < M) ? my_m : (M > 0 ? M - 1 : 0);
    uint32_t safe_my_n_b = (my_n_b < N) ? my_n_b : (N > 0 ? N - 1 : 0);

    __shared__ uint32_t tmem_c;
    if (threadIdx.x < 32) {
        tmem_alloc_fn(&tmem_c, 256);
    }
    __syncthreads();

    // SMEM Allocation (~70KB total for data, separate for mbarriers)
    __shared__ __align__(1024) uint8_t smem_out_bytes[128 * 272 * 2];
    __shared__ __align__(8) uint64_t mbar_tma[2];
    __shared__ __align__(8) uint64_t mbar_umma[2];

    uint8_t* smem_A0 = smem_out_bytes;
    uint8_t* smem_B0 = smem_out_bytes + 16384;
    uint8_t* smem_A1 = smem_out_bytes + 32768;
    uint8_t* smem_B1 = smem_out_bytes + 49152;

    uint32_t idesc = make_instr_desc_fn(256, 256);

    if (threadIdx.x == 0) {
        for (int p = 0; p < 2; ++p) {
            init_smem_barrier_fn(&mbar_tma[p], 1);
            // Initialize umma barrier with 1 because only CTA 0 will issue the commit
            init_smem_barrier_fn(&mbar_umma[p], 1); 
        }
        fence_smem_barrier_init_fn();
    }
    cluster_sync_fn();

    int num_steps = K / 64;
    int phase_tma[2] = {0, 0};
    int phase_umma[2] = {0, 0};

    if (threadIdx.x == 0 && num_steps > 0) {
        mbarrier_arrive_and_expect_tx_fn(&mbar_tma[0], 16384 * 2);
        tma_load_2d_fn(&tma_A, &mbar_tma[0], smem_A0, 0, safe_my_m);
        tma_load_2d_fn(&tma_B, &mbar_tma[0], smem_B0, 0, safe_my_n_b);
    }

    for (int step = 0; step < num_steps; ++step) {
        int buf = step % 2;
        int next_buf = (step + 1) % 2;

        if (threadIdx.x == 0) {
            mbarrier_wait_fn(&mbar_tma[buf], phase_tma[buf]);
            phase_tma[buf] ^= 1;
        }
        // Sync cluster to ensure both CTAs have their data loaded
        cluster_sync_fn();

        // Only the even CTA (rank 0) issues the UMMA and Commit to avoid conflicts
        if (rank == 0 && threadIdx.x == 0) {
            for (int k_mma = 0; k_mma < 4; ++k_mma) {
                uint32_t offset = k_mma * 32;
                uint8_t* curr_A = (buf == 0) ? smem_A0 : smem_A1;
                uint8_t* curr_B = (buf == 0) ? smem_B0 : smem_B1;
                
                uint64_t desc_a = make_smem_desc_sm100_fn(curr_A + offset, 1024);
                uint64_t desc_b = make_smem_desc_sm100_fn(curr_B + offset, 1024);
                
                uint32_t accum = (step == 0 && k_mma == 0) ? 0 : 1;
                umma_f16_cg2_fn(tmem_c, desc_a, desc_b, idesc, accum);
            }
            // Commits and multicasts arrival signal to both CTA 0 and CTA 1
            umma_commit_2sm_fn(&mbar_umma[buf]);
        }

        if (step + 1 < num_steps) {
            if (threadIdx.x == 0) {
                if (step > 0) {
                    mbarrier_wait_fn(&mbar_umma[next_buf], phase_umma[next_buf]);
                    phase_umma[next_buf] ^= 1;
                }
                mbarrier_arrive_and_expect_tx_fn(&mbar_tma[next_buf], 16384 * 2);
                uint8_t* next_A = (next_buf == 0) ? smem_A0 : smem_A1;
                uint8_t* next_B = (next_buf == 0) ? smem_B0 : smem_B1;
                tma_load_2d_fn(&tma_A, &mbar_tma[next_buf], next_A, (step + 1) * 64, safe_my_m);
                tma_load_2d_fn(&tma_B, &mbar_tma[next_buf], next_B, (step + 1) * 64, safe_my_n_b);
            }
        }
    }

    if (num_steps > 0) {
        if (threadIdx.x == 0) {
            mbarrier_wait_fn(&mbar_umma[(num_steps - 1) % 2], phase_umma[(num_steps - 1) % 2]);
        }
    }
    // Final sync before reading TMEM
    cluster_sync_fn();

    // Epilogue
    __nv_bfloat16* smem_out = reinterpret_cast<__nv_bfloat16*>(smem_out_bytes);
    my_tmem_epilogue_coalesced(C, smem_out, M, N, my_m, n_block_start, 128, 256, tmem_c);

    // Sync cluster before deallocation
    cluster_sync_fn(); 
    if (threadIdx.x < 32) {
        tmem_dealloc_fn(tmem_c, 256);
    }
}

namespace tvm_ffi_example_cuda {

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

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(0);
    
    __nv_bfloat16* A_ptr = static_cast<__nv_bfloat16*>(A.data_ptr());
    __nv_bfloat16* B_ptr = static_cast<__nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* C_ptr = static_cast<__nv_bfloat16*>(C.data_ptr());
    
    CUtensorMap tma_A, tma_B;
    
    CUresult resA = create_tma_2d_descriptor_2B(
        &tma_A, A_ptr, K, M, 64, 128,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (resA != CUDA_SUCCESS) {
        fprintf(stderr, "Failed to create TMA A: %d\n", resA); exit(1);
    }
    
    CUresult resB = create_tma_2d_descriptor_2B(
        &tma_B, B_ptr, K, N, 64, 128,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (resB != CUDA_SUCCESS) {
        fprintf(stderr, "Failed to create TMA B: %d\n", resB); exit(1);
    }
    
    int clusters_x = (N + 255) / 256;
    int clusters_y = (M + 255) / 256;
    dim3 grid(clusters_x * 2, clusters_y, 1);
    dim3 block(128, 1, 1); 
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, gemm_kernel, tma_A, tma_B, C_ptr, (uint32_t)M, (uint32_t)N, (uint32_t)K));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda