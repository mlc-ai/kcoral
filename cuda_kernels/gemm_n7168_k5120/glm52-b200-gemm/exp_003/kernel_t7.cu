#include <cuda_bf16.h>
#include <cuda_runtime.h>
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

// ===================== Device Helpers =====================

__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

__device__ __forceinline__ void cluster_sync_fn() {
    asm volatile("barrier.cluster.arrive;\nbarrier.cluster.wait;\n" ::: "memory");
}

__device__ __forceinline__ void cluster_arrive_fn() {
    asm volatile("barrier.cluster.arrive;\n" ::: "memory");
}

__device__ __forceinline__ void cluster_wait_fn() {
    asm volatile("barrier.cluster.wait;\n" ::: "memory");
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
    d |= (1u << 4);    // c_format = FP32
    d |= (1u << 7);    // a_format = BF16
    d |= (1u << 10);   // b_format = BF16
    d |= (0u << 15);   // a_major = 0 (K-Major)
    d |= (0u << 16);   // b_major = 0 (K-Major)
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
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

// ===================== TMA Descriptor Creation =====================

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress,
    uint64_t gmem_inner_dim, uint64_t gmem_outer_dim,
    uint32_t smem_inner_dim, uint32_t smem_outer_dim,
    CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion,
    CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
        globalAddress, globalDim, globalStrides,
        boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle, l2Promotion, oobFill);
}

// ===================== Kernel =====================

namespace gemm_cuda {

// Each CTA: A tile = 128x64, B tile = 256x64 (FULL B for cta_group::2)
// Cluster: 2 CTAs compute 256x256 output tile
constexpr int BM = 128;        // A rows per CTA
constexpr int BN = 256;        // B columns (full, loaded by each CTA)
constexpr int BK = 64;         // K dimension per tile
constexpr int TOTAL_M = 256;   // Combined M across cluster
constexpr int TOTAL_N = 256;   // Combined N (= BN, since each CTA loads full B)
constexpr int NUM_THREADS = 128;
// SMEM: 2 * (A + B) = 2 * (128*64 + 256*64) * 2 = 98304 bytes + barriers
constexpr int SMEM_SIZE = 2 * (BM * BK + BN * BK) * 2 + 64;

__launch_bounds__(NUM_THREADS, 2)
__global__ void gemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    const __grid_constant__ CUtensorMap tma_C,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K,
    int num_m_blocks
) {
    uint32_t rank = cluster_rank_fn();
    int linear_block = blockIdx.x;
    int cluster_block = linear_block / 2;
    int n_block = cluster_block / num_m_blocks;
    int m_block = cluster_block % num_m_blocks;
    int m_base = m_block * TOTAL_M;
    int n_base = n_block * TOTAL_N;
    int m_offset = m_base + rank * BM;  // Each CTA's A rows
    // Both CTAs load the SAME full B tile (n_base to n_base + BN)

    extern __shared__ __align__(1024) char smem_buf[];
    __nv_bfloat16* smem_A[2];
    __nv_bfloat16* smem_B[2];
    smem_A[0] = reinterpret_cast<__nv_bfloat16*>(smem_buf);
    smem_A[1] = smem_A[0] + BM * BK;
    smem_B[0] = reinterpret_cast<__nv_bfloat16*>(smem_buf + 2 * BM * BK * sizeof(__nv_bfloat16));
    smem_B[1] = smem_B[0] + BN * BK;
    uint64_t* mbar = reinterpret_cast<uint64_t*>(
        smem_buf + 2 * (BM * BK + BN * BK) * sizeof(__nv_bfloat16));
    __shared__ uint32_t tmem_addr_smem;

    // Init barriers: 2 for TMA double-buffer, 1 for MMA
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&mbar[0], 1);
        init_smem_barrier_fn(&mbar[1], 1);
        init_smem_barrier_fn(&mbar[2], 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();
    cluster_sync_fn();

    // Alloc TMEM: 256 columns for BN=256 FP32 output
    if (threadIdx.x < 32) {
        tmem_alloc_fn(&tmem_addr_smem, 256);
    }
    __syncthreads();
    cluster_sync_fn();
    uint32_t tmem_c = tmem_addr_smem;

    prefetch_tma_descriptor_fn(&tma_A);
    prefetch_tma_descriptor_fn(&tma_B);
    prefetch_tma_descriptor_fn(&tma_C);

    // Instruction descriptor: M=128 (per CTA), N=256 (full B), K=16
    uint32_t idesc = make_instr_desc_fn(BM, BN);
    const int num_k_tiles = K / BK;
    int phase_load[2] = {0, 0};
    int phase_mma = 0;

    constexpr int TMA_BYTES = (BM * BK + BN * BK) * 2;  // 49152

    // Prologue: issue TMA load k=0 (each CTA loads its own A + full B)
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(&mbar[0], TMA_BYTES);
        tma_load_2d_fn(&tma_A, &mbar[0], smem_A[0], 0, m_offset);
        tma_load_2d_fn(&tma_B, &mbar[0], smem_B[0], 0, n_base);
    }

    #pragma unroll 1
    for (int k = 0; k < num_k_tiles; k++) {
        int buf = k % 2;
        int next_buf = (k + 1) % 2;

        // Issue next TMA load EARLY (before waiting for current) for max overlap
        if (k + 1 < num_k_tiles && threadIdx.x == 0) {
            int k_coord = (k + 1) * BK;
            mbarrier_arrive_and_expect_tx_fn(&mbar[next_buf], TMA_BYTES);
            tma_load_2d_fn(&tma_A, &mbar[next_buf], smem_A[next_buf], k_coord, m_offset);
            tma_load_2d_fn(&tma_B, &mbar[next_buf], smem_B[next_buf], k_coord, n_base);
        }

        // Wait for TMA load of current buffer
        mbarrier_wait_fn(&mbar[buf], phase_load[buf]);
        phase_load[buf] ^= 1;

        // Fence: ensure async proxy SMEM writes are visible for MMA reads
        fence_async_shared_fn();

        // CTA0: issue cta_group::2 UMMA (4 sub-MMAs, K=16 each)
        // Both CTAs' TMEM get results: CTA0 gets rows [m_offset], CTA1 gets rows [m_offset+128]
        if (rank == 0 && threadIdx.x == 0) {
            #pragma unroll
            for (int sub = 0; sub < 4; sub++) {
                uint64_t desc_a = make_smem_desc_sm100_fn(
                    smem_A[buf] + sub * 16, 1, 1024);
                uint64_t desc_b = make_smem_desc_sm100_fn(
                    smem_B[buf] + sub * 16, 1, 1024);
                uint32_t accum = (k == 0 && sub == 0) ? 0 : 1;
                umma_f16_cg2_fn(tmem_c, desc_a, desc_b, idesc, accum);
            }
        }
    }

    // Commit all MMAs and wait for completion
    if (rank == 0 && threadIdx.x == 0) {
        umma_commit_2sm_fn(&mbar[2]);
    }
    mbarrier_wait_fn(&mbar[2], phase_mma);

    // ===================== Epilogue: TMEM -> SMEM -> Global =====================
    __nv_bfloat16* smem_out = reinterpret_cast<__nv_bfloat16*>(smem_buf);

    // Phase 1: TMEM -> SMEM (each thread reads its row, converts FP32->BF16)
    // Each CTA has 128 rows (threadIdx.x 0..127) x 256 columns in TMEM
    #pragma unroll 4
    for (uint32_t col = 0; col < (uint32_t)BN; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
            : "r"(tmem_c + col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        uint32_t base = threadIdx.x * BN + col;
        smem_out[base + 0] = __float2bfloat16(__uint_as_float(r0));
        smem_out[base + 1] = __float2bfloat16(__uint_as_float(r1));
        smem_out[base + 2] = __float2bfloat16(__uint_as_float(r2));
        smem_out[base + 3] = __float2bfloat16(__uint_as_float(r3));
    }
    __syncthreads();

    // Phase 2: SMEM -> Global
    bool full_tile = (m_offset + BM <= M) && (n_base + BN <= N);
    if (full_tile) {
        // Fast path: TMA store
        fence_async_shared_fn();
        if (threadIdx.x == 0) {
            tma_store_2d_fn(&tma_C, smem_out, n_base, m_offset);
            tma_store_commit_fn();
            tma_store_wait_fn<0>();
        }
    } else {
        // Slow path: thread-level stores with bounds checking
        uint32_t warp_id = threadIdx.x / 32;
        uint32_t lane_id = threadIdx.x % 32;
        for (uint32_t step = 0; step < 32; step++) {
            uint32_t row = warp_id * 32 + step;
            uint32_t global_row = m_offset + row;
            if (global_row >= (uint32_t)M) continue;
            uint32_t col_start = lane_id * 8;
            uint32_t global_col = n_base + col_start;
            if (global_col + 7 < (uint32_t)N) {
                float4 data = *reinterpret_cast<float4*>(&smem_out[row * BN + col_start]);
                *reinterpret_cast<float4*>(C + (uint64_t)global_row * N + global_col) = data;
            } else {
                for (int c = 0; c < 8 && global_col + c < (uint32_t)N; c++) {
                    C[(uint64_t)global_row * N + global_col + c] =
                        smem_out[row * BN + col_start + c];
                }
            }
        }
    }
    __syncthreads();

    // Dealloc TMEM
    cluster_sync_fn();
    if (threadIdx.x < 32) {
        tmem_dealloc_fn(tmem_c, 256);
    }
}

// ===================== Host Function =====================

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));

    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(0);

    __nv_bfloat16* A_ptr = static_cast<__nv_bfloat16*>(A.data_ptr());
    __nv_bfloat16* B_ptr = static_cast<__nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* C_ptr = static_cast<__nv_bfloat16*>(C.data_ptr());

    CUtensorMap tma_A, tma_B, tma_C;
    CUresult res;

    // A: (M, K) -> globalDim={K, M}, boxDim={BK, BM}
    res = create_tma_2d_descriptor_2B(&tma_A, A_ptr, (uint64_t)K, (uint64_t)M, BK, BM,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "TMA A descriptor creation failed: %d\n", res);
        exit(1);
    }

    // B: (N, K) -> globalDim={K, N}, boxDim={BK, BN} (full 256 columns)
    res = create_tma_2d_descriptor_2B(&tma_B, B_ptr, (uint64_t)K, (uint64_t)N, BK, BN,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "TMA B descriptor creation failed: %d\n", res);
        exit(1);
    }

    // C: (M, N) -> globalDim={N, M}, boxDim={BN, BM}
    res = create_tma_2d_descriptor_2B(&tma_C, C_ptr, (uint64_t)N, (uint64_t)M, BN, BM,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "TMA C descriptor creation failed: %d\n", res);
        exit(1);
    }

    int num_m_blocks = (M + TOTAL_M - 1) / TOTAL_M;
    int num_n_blocks = (N + TOTAL_N - 1) / TOTAL_N;
    int total_clusters = num_m_blocks * num_n_blocks;
    int total_blocks = total_clusters * 2;

    dim3 grid(total_blocks);
    dim3 block(NUM_THREADS);

    cudaFuncSetAttribute(gemm_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = SMEM_SIZE;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    cudaLaunchKernelEx(&config, gemm_kernel,
        tma_A, tma_B, tma_C, C_ptr, (int)M, (int)N, (int)K, num_m_blocks);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_cuda::run);

}  // namespace gemm_cuda