#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <stdio.h>
#include <stdint.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

#define CU_CHECK(call) do {                                        \
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        const char* err_str;                                       \
        cuGetErrorString(_e, &err_str);                            \
        fprintf(stderr, "CU error %s at %s:%d\n",                  \
                err_str, __FILE__, __LINE__);                      \
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

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
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
    d |= (uint64_t)2 << 61;
    uint32_t base_offset = (addr >> 7) & 0x7;
    d |= (uint64_t)base_offset << 49;
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);           // FP32 output
    d |= (1u << 7);           // BF16 input A
    d |= (1u << 10);          // BF16 input B
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

__device__ __forceinline__ void tmem_epilogue_fn(
    __nv_bfloat16* D, __nv_bfloat16* smem_out, uint32_t tmem_base,
    uint32_t M, uint32_t N, uint32_t m_block, uint32_t n_block,
    uint32_t BM, uint32_t BN) {
    // Phase 1: TMEM -> SMEM 
    for (uint32_t col = 0; col < BN; col += 4) {
        uint32_t r0, r1, r2, r3;
        uint32_t src_col = tmem_base + col;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(src_col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        uint32_t base = threadIdx.x * BN + col;
        
        uint32_t packed01 = pack_bf16_fn(r0, r1);
        uint32_t packed23 = pack_bf16_fn(r2, r3);
        
        *reinterpret_cast<uint32_t*>(&smem_out[base + 0]) = packed01;
        *reinterpret_cast<uint32_t*>(&smem_out[base + 2]) = packed23;
    }
    __syncthreads();
    
    // Phase 2: SMEM -> Global (coalesced 128-thread parallel write-out)
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t num_steps = (BM + 3) / 4;
    for (uint32_t step = 0; step < num_steps; ++step) {
        uint32_t row = step * 4 + warp_id;
        if (row >= BM) continue;
        uint32_t global_row = m_block * BM + row;
        uint32_t col_start = lane_id * 4;
        uint32_t global_col = n_block * BN + col_start;
        // Verify rigorous horizontal bounds
        if (col_start < BN && global_row < M && global_col + 3 < N) {
            uint2 data = *reinterpret_cast<uint2*>(&smem_out[row * BN + col_start]);
            *reinterpret_cast<uint2*>(D + (uint64_t)global_row * N + global_col) = data;
        }
    }
}

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, 
                                     uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, 
                                     uint32_t smem_inner_dim, uint32_t smem_outer_dim, 
                                     CUtensorMapDataType dataType, CUtensorMapSwizzle swizzle, 
                                     CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d, dataType, 2, globalAddress, globalDim, globalStrides, boxDim,
        elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, l2Promotion, oobFill
    );
}

struct SharedStorage {
    alignas(1024) __nv_bfloat16 A[4][128][64]; 
    alignas(1024) __nv_bfloat16 B[4][64][64];  
    alignas(16) uint64_t tma_bar[4];           
    alignas(16) uint64_t umma_bar[4];          
};

__global__ __launch_bounds__(128, 2) void gemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    __nv_bfloat16* C_ptr,
    int M, int N, int K) 
{
    // Provide explicit guaranteed 1024 bytes alignment for dynamic buffer, 
    // shielding us entirely from CUDA runtime 16 byte unaligned offsets.
    extern __shared__ char smem_buf_raw[];
    uintptr_t raw_ptr = reinterpret_cast<uintptr_t>(smem_buf_raw);
    uintptr_t aligned_ptr = (raw_ptr + 1023) & ~1023;
    SharedStorage* smem = reinterpret_cast<SharedStorage*>(aligned_ptr);

    int rank = cluster_rank_fn();
    int cluster_x = blockIdx.x / 2;
    int m_idx = cluster_x * 256 + rank * 128;
    int n_idx = blockIdx.y * 128 + rank * 64;

    if (threadIdx.x == 0) {
        if (rank == 0) {
            for (int i = 0; i < 4; ++i) {
                init_smem_barrier_fn(&smem->tma_bar[i], 1);
            }
        }
        for (int i = 0; i < 4; ++i) {
            init_smem_barrier_fn(&smem->umma_bar[i], 1);
        }
        fence_smem_barrier_init_fn();
    }
    cluster_sync_fn();

    __shared__ uint32_t tmem_addr;
    if (threadIdx.x < 32) {
        tmem_alloc_fn(&tmem_addr, 128);
    }
    __syncthreads();

    int K_ITERS = K / 64;
    int tma_phase[4] = {0, 0, 0, 0};
    int umma_phase[4] = {0, 0, 0, 0};

    // Prologue: Prefill pipelines
    for (int iter = 0; iter < 3; ++iter) {
        int s = iter % 4;
        if (threadIdx.x == 0) {
            if (rank == 0) mbarrier_arrive_and_expect_tx_fn(&smem->tma_bar[s], 49152);
            tma_load_2d_cg2_fn(&tma_A, &smem->tma_bar[s], &smem->A[s][0][0], iter * 64, m_idx);
            tma_load_2d_cg2_fn(&tma_B, &smem->tma_bar[s], &smem->B[s][0][0], iter * 64, n_idx);
        }
    }

    for (int iter = 0; iter < K_ITERS; ++iter) {
        int s_load = (iter + 3) % 4;
        int s_comp = iter % 4;

        if (iter + 3 < K_ITERS) {
            if (iter >= 1) {
                if (threadIdx.x == 0) {
                    mbarrier_wait_fn(&smem->umma_bar[s_load], umma_phase[s_load]);
                }
                umma_phase[s_load] ^= 1;
            }
            if (threadIdx.x == 0) {
                if (rank == 0) {
                    mbarrier_arrive_and_expect_tx_fn(&smem->tma_bar[s_load], 49152);
                }
                tma_load_2d_cg2_fn(&tma_A, &smem->tma_bar[s_load], &smem->A[s_load][0][0], (iter + 3) * 64, m_idx);
                tma_load_2d_cg2_fn(&tma_B, &smem->tma_bar[s_load], &smem->B[s_load][0][0], (iter + 3) * 64, n_idx);
            }
        }

        if (rank == 0) {
            if (threadIdx.x == 0) {
                mbarrier_wait_fn(&smem->tma_bar[s_comp], tma_phase[s_comp]);
                fence_async_shared_fn();

                uint64_t desc_a = make_smem_desc_sm100_fn(&smem->A[s_comp][0][0], 1024);
                uint64_t desc_b = make_smem_desc_sm100_fn(&smem->B[s_comp][0][0], 1024);
                uint32_t idesc = make_instr_desc_fn(256, 128);

                for (int step = 0; step < 4; ++step) {
                    uint32_t accum = (iter == 0 && step == 0) ? 0 : 1;
                    umma_f16_cg2_fn(tmem_addr, desc_a, desc_b, idesc, accum);
                    desc_a += 2; // Step K gracefully 
                    desc_b += 2;
                }
                umma_commit_2sm_fn(&smem->umma_bar[s_comp]);
            }
            tma_phase[s_comp] ^= 1;
        }
    }

    // Await conclusion of the last UMMA compute phase
    int s_last = (K_ITERS - 1) % 4;
    if (threadIdx.x == 0) {
        mbarrier_wait_fn(&smem->umma_bar[s_last], umma_phase[s_last]);
    }
    umma_phase[s_last] ^= 1;
    
    // Explicit sync barrier ensuring everything is ready for the re-use writeback cycle
    __syncthreads();

    // Re-use smem->A region for the collective memory epilogue 
    __nv_bfloat16* smem_out = (__nv_bfloat16*)smem->A;
    // BN size is 128 dynamically expanded here resulting from distributed combined `B` tile handling 
    tmem_epilogue_fn(C_ptr, smem_out, tmem_addr, M, N, cluster_x * 2 + rank, blockIdx.y, 128, 128);

    __syncthreads();
    if (threadIdx.x < 32) {
        tmem_dealloc_fn(tmem_addr, 128);
    }
}

namespace tvm_ffi_gemm {

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(0);
    
    if (M == 0) return;

    CUtensorMap tma_A, tma_B;
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_A, A.data_ptr(), K, M, 64, 128, 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_NONE, 
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    CU_CHECK(create_tma_2d_descriptor_2B(&tma_B, B.data_ptr(), K, N, 64, 64, 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_NONE, 
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    int grid_x = (M + 255) / 256 * 2;
    int grid_y = (N + 127) / 128;
    dim3 grid(grid_x, grid_y, 1);
    dim3 block(128, 1, 1);

    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    // Oversize memory dynamically, safeguarding standard 128B alignment
    config.dynamicSmemBytes = sizeof(SharedStorage) + 1024;
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    CUDA_CHECK(cudaFuncSetAttribute((void*)gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(SharedStorage) + 1024));
    CUDA_CHECK(cudaLaunchKernelEx(&config, gemm_kernel, tma_A, tma_B, static_cast<__nv_bfloat16*>(C.data_ptr()), M, N, K));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_gemm