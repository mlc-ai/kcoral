#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", \
                cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define CU_CHECK(call) do { \
    CUresult _e = (call); \
    if (_e != CUDA_SUCCESS) { \
        const char* errStr; \
        cuGetErrorString(_e, &errStr); \
        fprintf(stderr, "CU error %s at %s:%d\n", \
                errStr ? errStr : "unknown", __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// === Helper functions ===

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

__device__ __forceinline__ uint64_t make_smem_desc_32b_fn(void* smem_ptr) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    // SBO = 256 (8 rows * 32B per row for 32B swizzle), encoded as 256>>4 = 16
    d |= (uint64_t)16 << 32;
    d |= (uint64_t)1 << 46;   // version = 1 (SM100)
    d |= (uint64_t)6 << 61;   // layout_type = 32B swizzling
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    // c_format = FP32
    d |= (1u << 7);    // a_format = BF16
    d |= (1u << 10);   // b_format = BF16
    d |= (0u << 15);   // a_major = 0 (K-Major)
    d |= (0u << 16);   // b_major = 0 (K-Major)
    d |= ((N / 8) << 17);     // n_dim
    d |= ((M / 16) << 24);    // m_dim
    return d;
}

__device__ __forceinline__ void tmem_epilogue_coalesced_4w_fn(
    __nv_bfloat16* D, __nv_bfloat16* smem_out,
    uint32_t M, uint32_t N, uint32_t m_block, uint32_t n_block,
    uint32_t BM, uint32_t BN) {
    // Phase 1: TMEM -> SMEM
    for (uint32_t col = 0; col < BN; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        uint32_t base = threadIdx.x * BN + col;
        smem_out[base + 0] = __float2bfloat16(__uint_as_float(r0));
        smem_out[base + 1] = __float2bfloat16(__uint_as_float(r1));
        smem_out[base + 2] = __float2bfloat16(__uint_as_float(r2));
        smem_out[base + 3] = __float2bfloat16(__uint_as_float(r3));
    }
    __syncthreads();
    // Phase 2: SMEM -> Global (coalesced 8-byte writes)
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
            uint2 data = *reinterpret_cast<uint2*>(&smem_out[row * BN + col_start]);
            *reinterpret_cast<uint2*>(D + (uint64_t)global_row * N + global_col) = data;
        }
    }
}

// === Constants ===
constexpr int BM_PER_CTA = 128;
constexpr int BN_HALF = 128;
constexpr int BN_COMBINED = 256;
constexpr int K_TILE = 16;
constexpr int NUM_STAGES = 2;
constexpr int A_SIZE = BM_PER_CTA * K_TILE * 2;
constexpr int B_SIZE = BN_HALF * K_TILE * 2;
constexpr int STAGE_SIZE = A_SIZE + B_SIZE;
constexpr int TMA_BYTES = 2 * STAGE_SIZE;
constexpr int SMEM_OUT_SIZE = BM_PER_CTA * BN_COMBINED * 2;
constexpr int TOTAL_SMEM = NUM_STAGES * STAGE_SIZE + 4 * 8 + SMEM_OUT_SIZE;

// === Kernel ===
__global__ __cluster_dims__(2, 1, 1) __launch_bounds__(128, 1) void gemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    __nv_bfloat16* C,
    int M, int N, int K) {

    int cta_rank = cluster_rank_fn();
    int n_block = blockIdx.x / 2;
    int m_block = blockIdx.y;

    int m_offset = m_block * 2 * BM_PER_CTA + cta_rank * BM_PER_CTA;
    int n_offset = n_block * BN_COMBINED + cta_rank * BN_HALF;

    extern __shared__ __align__(1024) uint8_t smem[];

    __nv_bfloat16* A_smem[2];
    __nv_bfloat16* B_smem[2];
    A_smem[0] = (__nv_bfloat16*)(smem + 0);
    B_smem[0] = (__nv_bfloat16*)(smem + A_SIZE);
    A_smem[1] = (__nv_bfloat16*)(smem + STAGE_SIZE);
    B_smem[1] = (__nv_bfloat16*)(smem + STAGE_SIZE + A_SIZE);

    uint64_t* full_bar = (uint64_t*)(smem + NUM_STAGES * STAGE_SIZE);
    uint64_t* mma_bar = full_bar + NUM_STAGES;

    __nv_bfloat16* smem_out = (__nv_bfloat16*)(mma_bar + NUM_STAGES);

    __shared__ uint32_t tmem_addr_shmem;

    int tid = threadIdx.x;
    int warp = tid / 32;

    // Init barriers
    if (tid == 0) {
        for (int s = 0; s < NUM_STAGES; s++) {
            init_smem_barrier_fn(&full_bar[s], 1);
            init_smem_barrier_fn(&mma_bar[s], 1);
        }
        fence_smem_barrier_init_fn();
    }
    __syncthreads();

    // TMEM allocation (cta_group::2, both CTAs participate)
    if (warp == 0) {
        tmem_alloc_fn(&tmem_addr_shmem, 256);
    }
    __syncthreads();
    uint32_t tmem_addr = tmem_addr_shmem;

    int k_iters = K / K_TILE;

    // Prologue: issue TMA for stage 0
    if (tid == 0) {
        tma_load_2d_cg2_fn(&tma_A, &full_bar[0], A_smem[0], 0, m_offset);
        tma_load_2d_cg2_fn(&tma_B, &full_bar[0], B_smem[0], 0, n_offset);
    }
    if (cta_rank == 0 && tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(&full_bar[0], TMA_BYTES);
    }

    for (int k = 0; k < k_iters; k++) {
        int s = k % NUM_STAGES;
        int next_s = (k + 1) % NUM_STAGES;

        // CTA 0: wait for TMA completion (all CTAs' TMAs)
        if (cta_rank == 0 && tid == 0) {
            mbarrier_wait_fn(&full_bar[s], (k / NUM_STAGES) % 2);
        }
        __syncthreads();

        // CTA 0: issue UMMA + commit
        if (cta_rank == 0 && tid == 0) {
            tcgen05_fence_after_fn();
            uint64_t desc_a = make_smem_desc_32b_fn(A_smem[s]);
            uint64_t desc_b = make_smem_desc_32b_fn(B_smem[s]);
            uint32_t idesc = make_instr_desc_fn(256, 256);
            uint32_t accum = (k == 0) ? 0 : 1;
            umma_f16_cg2_fn(tmem_addr, desc_a, desc_b, idesc, accum);
            umma_commit_2sm_fn(&mma_bar[s]);
        }
        __syncthreads();

        // Issue TMA for next iteration (if exists)
        if (k + 1 < k_iters) {
            // Wait for UMMA on next_s to free SMEM (if stage was used before)
            if (k + 1 >= NUM_STAGES) {
                if (tid == 0) {
                    mbarrier_wait_fn(&mma_bar[next_s], ((k + 1 - NUM_STAGES) / NUM_STAGES) % 2);
                }
                __syncthreads();
            }
            if (tid == 0) {
                tma_load_2d_cg2_fn(&tma_A, &full_bar[next_s], A_smem[next_s], (k + 1) * K_TILE, m_offset);
                tma_load_2d_cg2_fn(&tma_B, &full_bar[next_s], B_smem[next_s], (k + 1) * K_TILE, n_offset);
            }
            if (cta_rank == 0 && tid == 0) {
                mbarrier_arrive_and_expect_tx_fn(&full_bar[next_s], TMA_BYTES);
            }
        }

        // Wait for UMMA[k] (both CTAs)
        if (tid == 0) {
            mbarrier_wait_fn(&mma_bar[s], (k / NUM_STAGES) % 2);
        }
        __syncthreads();
    }

    // Epilogue: read TMEM -> convert to BF16 -> store to global
    tmem_epilogue_coalesced_4w_fn(
        C, smem_out, (uint32_t)M, (uint32_t)N,
        (uint32_t)(m_offset / BM_PER_CTA), (uint32_t)n_block,
        (uint32_t)BM_PER_CTA, (uint32_t)BN_COMBINED);

    __syncthreads();

    // Dealloc TMEM
    cluster_sync_fn();
    if (warp == 0) {
        tmem_dealloc_fn(tmem_addr, 256);
    }
}

// === TMA descriptor creation ===
static CUresult create_tma_2d_descriptor_bf16(
    CUtensorMap* d, void* globalAddress,
    uint64_t gmem_inner_dim, uint64_t gmem_outer_dim,
    uint32_t smem_inner_dim, uint32_t smem_outer_dim,
    CUtensorMapSwizzle swizzle,
    CUtensorMapL2promotion l2Promotion,
    CUtensorMapFloatOOBfill oobFill) {

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

// === Host function ===
namespace gemm_cuda {

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));

    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(0);

    __nv_bfloat16* A_ptr = static_cast<__nv_bfloat16*>(A.data_ptr());
    __nv_bfloat16* B_ptr = static_cast<__nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* C_ptr = static_cast<__nv_bfloat16*>(C.data_ptr());

    // Create TMA descriptors
    CUtensorMap tma_A, tma_B;
    CU_CHECK(create_tma_2d_descriptor_bf16(
        &tma_A, A_ptr, (uint64_t)K, (uint64_t)M, K_TILE, BM_PER_CTA,
        CU_TENSOR_MAP_SWIZZLE_32B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA));

    CU_CHECK(create_tma_2d_descriptor_bf16(
        &tma_B, B_ptr, (uint64_t)K, (uint64_t)N, K_TILE, BN_HALF,
        CU_TENSOR_MAP_SWIZZLE_32B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA));

    // Grid configuration: cluster of 2 CTAs, each cluster handles 256x256 output tile
    int grid_x = (int)(N / BN_COMBINED) * 2;
    int grid_y = (int)((M + 2 * BM_PER_CTA - 1) / (2 * BM_PER_CTA));

    dim3 grid(grid_x, grid_y, 1);
    dim3 block(128, 1, 1);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, TOTAL_SMEM));

    gemm_kernel<<<grid, block, TOTAL_SMEM, stream>>>(tma_A, tma_B, C_ptr, (int)M, (int)N, (int)K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_cuda::run);

}  // namespace gemm_cuda