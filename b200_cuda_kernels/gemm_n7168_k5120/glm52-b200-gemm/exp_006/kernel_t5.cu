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

constexpr int BM_PER_CTA = 128;
constexpr int BN_PER_CTA = 128;
constexpr int BN_COMBINED = 256;
constexpr int K_TILE = 16;
constexpr int NUM_STAGES = 4;
constexpr int A_TILE_BYTES = BM_PER_CTA * K_TILE * 2;
constexpr int B_TILE_BYTES = BN_PER_CTA * K_TILE * 2;
constexpr int STAGE_BYTES = A_TILE_BYTES + B_TILE_BYTES;
constexpr int TMA_BYTES = A_TILE_BYTES + B_TILE_BYTES;
constexpr int SMEM_OUT_BYTES = BM_PER_CTA * BN_COMBINED * 2;
constexpr int BARRIER_BYTES = 3 * NUM_STAGES * 8;
constexpr int TOTAL_SMEM = NUM_STAGES * STAGE_BYTES + BARRIER_BYTES + SMEM_OUT_BYTES;
constexpr int TMEM_COLS = 256;

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

__device__ __forceinline__ void mbarrier_arrive_fn(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])) : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_remote_fn(uint64_t* bar, uint32_t target_cta) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    uint32_t remote_a;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(remote_a) : "r"(a), "r"(target_cta));
    asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];"
                 :: "r"(remote_a) : "memory");
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
    d |= (uint64_t)16 << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)6 << 61;
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

__device__ __forceinline__ void tmem_epilogue_fn(
    __nv_bfloat16* D, __nv_bfloat16* smem_out,
    uint32_t M, uint32_t N, uint32_t m_base, uint32_t n_base,
    uint32_t BM_val, uint32_t BN_val) {

    // Phase 1: TMEM -> SMEM (batch 8 x4 loads = 32 cols per batch, 8 batches)
    for (uint32_t col = 0; col < BN_val; col += 32) {
        uint32_t r[8][4];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r[i][0]),"=r"(r[i][1]),"=r"(r[i][2]),"=r"(r[i][3]) : "r"(col + i*4));
        }
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            uint32_t base = threadIdx.x * BN_val + col + i*4;
            smem_out[base + 0] = __float2bfloat16(__uint_as_float(r[i][0]));
            smem_out[base + 1] = __float2bfloat16(__uint_as_float(r[i][1]));
            smem_out[base + 2] = __float2bfloat16(__uint_as_float(r[i][2]));
            smem_out[base + 3] = __float2bfloat16(__uint_as_float(r[i][3]));
        }
    }
    __syncthreads();

    // Phase 2: SMEM -> Global (coalesced 16-byte uint4 writes)
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t num_steps = BM_val / 4;
    for (uint32_t step = 0; step < num_steps; ++step) {
        uint32_t row = step * 4 + warp_id;
        uint32_t global_row = m_base + row;
        uint32_t col = lane_id * 8;
        uint32_t global_col = n_base + col;
        if (global_row < M && global_col + 7 < N) {
            uint4 data = *reinterpret_cast<uint4*>(&smem_out[row * BN_val + col]);
            *reinterpret_cast<uint4*>(D + (uint64_t)global_row * N + global_col) = data;
        } else if (global_row < M) {
            for (int c = 0; c < 8 && global_col + c < N; c++) {
                D[(uint64_t)global_row * N + global_col + c] = smem_out[row * BN_val + col + c];
            }
        }
    }
}

__global__ __cluster_dims__(2, 1, 1) __launch_bounds__(128, 1) void gemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    __nv_bfloat16* C,
    int M, int N, int K) {

    int cta = cluster_rank_fn();
    int n_cluster = blockIdx.x / 2;
    int m_cluster = blockIdx.y;
    int m_offset = m_cluster * 2 * BM_PER_CTA + cta * BM_PER_CTA;
    int n_offset = n_cluster * BN_COMBINED + cta * BN_PER_CTA;

    extern __shared__ __align__(1024) uint8_t smem[];

    __nv_bfloat16* A_smem[NUM_STAGES];
    __nv_bfloat16* B_smem[NUM_STAGES];
    #pragma unroll
    for (int i = 0; i < NUM_STAGES; i++) {
        A_smem[i] = (__nv_bfloat16*)(smem + i * STAGE_BYTES);
        B_smem[i] = (__nv_bfloat16*)(smem + i * STAGE_BYTES + A_TILE_BYTES);
    }

    uint64_t* full_bar = (uint64_t*)(smem + NUM_STAGES * STAGE_BYTES);
    uint64_t* ready_bar = full_bar + NUM_STAGES;
    uint64_t* mma_bar = ready_bar + NUM_STAGES;
    __nv_bfloat16* smem_out = (__nv_bfloat16*)(mma_bar + NUM_STAGES);

    __shared__ uint32_t tmem_addr_shmem;
    int tid = threadIdx.x;
    int warp = tid / 32;

    // Init barriers
    if (tid == 0) {
        for (int s = 0; s < NUM_STAGES; s++) {
            init_smem_barrier_fn(&full_bar[s], 1);
            init_smem_barrier_fn(&mma_bar[s], 1);
            if (cta == 0) {
                init_smem_barrier_fn(&ready_bar[s], 2);
            }
        }
        fence_smem_barrier_init_fn();
    }
    __syncthreads();
    cluster_sync_fn();

    if (warp == 0) {
        tmem_alloc_fn(&tmem_addr_shmem, TMEM_COLS);
    }
    __syncthreads();
    cluster_sync_fn();
    uint32_t tmem_addr = tmem_addr_shmem;

    int k_iters = K / K_TILE;

    // Prologue: issue TMA for first NUM_STAGES iterations
    if (tid == 0) {
        for (int s = 0; s < NUM_STAGES && s < k_iters; s++) {
            tma_load_2d_fn(&tma_A, &full_bar[s], A_smem[s], s * K_TILE, m_offset);
            tma_load_2d_fn(&tma_B, &full_bar[s], B_smem[s], s * K_TILE, n_offset);
            mbarrier_arrive_and_expect_tx_fn(&full_bar[s], TMA_BYTES);
        }
    }

    // Main loop - only thread 0
    if (tid == 0) {
        uint32_t full_phase[NUM_STAGES] = {0, 0, 0, 0};
        uint32_t ready_phase[NUM_STAGES] = {0, 0, 0, 0};
        uint32_t mma_phase[NUM_STAGES] = {0, 0, 0, 0};
        uint32_t idesc = make_instr_desc_fn(256, 256);

        for (int k = 0; k < k_iters; k++) {
            int s = k % NUM_STAGES;
            int next_s = (k + 1) % NUM_STAGES;

            // Wait for local TMA[s]
            mbarrier_wait_fn(&full_bar[s], full_phase[s]);
            full_phase[s] ^= 1;

            if (cta == 0) {
                // CTA0: sync with CTA1, then issue UMMA
                mbarrier_arrive_fn(&ready_bar[s]);
                mbarrier_wait_fn(&ready_bar[s], ready_phase[s]);
                ready_phase[s] ^= 1;

                tcgen05_fence_after_fn();
                uint64_t desc_a = make_smem_desc_32b_fn(A_smem[s]);
                uint64_t desc_b = make_smem_desc_32b_fn(B_smem[s]);
                uint32_t accum = (k == 0) ? 0 : 1;
                umma_f16_cg2_fn(tmem_addr, desc_a, desc_b, idesc, accum);
                umma_commit_2sm_fn(&mma_bar[s]);
            } else {
                // CTA1: signal CTA0 that our TMA is done
                mbarrier_arrive_remote_fn(&ready_bar[s], 0);
            }

            // Prefetch TMA for next iteration (both CTAs)
            if (k + 1 < k_iters) {
                if (k + 1 >= NUM_STAGES) {
                    mbarrier_wait_fn(&mma_bar[next_s], mma_phase[next_s]);
                    mma_phase[next_s] ^= 1;
                }
                tma_load_2d_fn(&tma_A, &full_bar[next_s], A_smem[next_s], (k + 1) * K_TILE, m_offset);
                tma_load_2d_fn(&tma_B, &full_bar[next_s], B_smem[next_s], (k + 1) * K_TILE, n_offset);
                mbarrier_arrive_and_expect_tx_fn(&full_bar[next_s], TMA_BYTES);
            }
        }

        // Wait for remaining UMMAs
        for (int i = 0; i < NUM_STAGES && i < k_iters; i++) {
            int k = k_iters - NUM_STAGES + i;
            if (k < 0) continue;
            int s = k % NUM_STAGES;
            mbarrier_wait_fn(&mma_bar[s], mma_phase[s]);
            mma_phase[s] ^= 1;
        }
    }
    __syncthreads();

    // Epilogue
    tmem_epilogue_fn(C, smem_out, (uint32_t)M, (uint32_t)N,
        (uint32_t)m_offset, (uint32_t)(n_cluster * BN_COMBINED),
        (uint32_t)BM_PER_CTA, (uint32_t)BN_COMBINED);

    __syncthreads();

    cluster_sync_fn();
    if (warp == 0) {
        tmem_dealloc_fn(tmem_addr, TMEM_COLS);
    }
}

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
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
        globalAddress, globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, l2Promotion, oobFill);
}

namespace gemm_cuda {

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));

    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(0);

    __nv_bfloat16* A_ptr = static_cast<__nv_bfloat16*>(A.data_ptr());
    __nv_bfloat16* B_ptr = static_cast<__nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* C_ptr = static_cast<__nv_bfloat16*>(C.data_ptr());

    CUtensorMap tma_A, tma_B;
    CU_CHECK(create_tma_2d_descriptor_bf16(
        &tma_A, A_ptr, (uint64_t)K, (uint64_t)M, K_TILE, BM_PER_CTA,
        CU_TENSOR_MAP_SWIZZLE_32B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA));

    CU_CHECK(create_tma_2d_descriptor_bf16(
        &tma_B, B_ptr, (uint64_t)K, (uint64_t)N, K_TILE, BN_PER_CTA,
        CU_TENSOR_MAP_SWIZZLE_32B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA));

    int n_clusters = (int)((N + BN_COMBINED - 1) / BN_COMBINED);
    int m_clusters = (int)((M + 2 * BM_PER_CTA - 1) / (2 * BM_PER_CTA));

    dim3 grid(n_clusters * 2, m_clusters, 1);
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