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

#define CU_CHECK(call) do {                                        \
    CUresult _r = (call);                                          \
    if (_r != CUDA_SUCCESS) {                                      \
        const char* errStr;                                        \
        cuGetErrorString(_r, &errStr);                             \
        fprintf(stderr, "CU error %s at %s:%d\n",                  \
                errStr ? errStr : "unknown", __FILE__, __LINE__);  \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace gemm_cuda {

// ======================== Helper Functions ========================

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

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void prefetch_tma_descriptor_fn(const CUtensorMap* d) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(d) : "memory");
}

// TMEM alloc/dealloc for cta_group::1
__device__ __forceinline__ void tmem_alloc_cg1_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_cg1_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
        :: "r"(addr), "r"(ncols));
}

// UMMA for cta_group::1, kind::f16 (BF16 x BF16 -> FP32)
__device__ __forceinline__ void umma_f16_cg1_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_cg1_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
        :: "r"(a) : "memory");
}

// SMEM descriptor for SM100 UMMA (K-major, 128B swizzle)
__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   // version = 1 (SM100)
    d |= (uint64_t)2 << 61;   // SWIZZLE_128B
    return d;
}

// Instruction descriptor for BF16 x BF16 -> FP32, K-major A and B
__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    // dtype = FP32
    d |= (1u << 7);    // atype = BF16
    d |= (1u << 10);   // btype = BF16
    d |= (0u << 15);   // a_major = 0 (K-Major)
    d |= (0u << 16);   // b_major = 0 (K-Major)
    d |= ((N / 8) << 17);   // n_dim
    d |= ((M / 16) << 24);  // m_dim
    return d;
}

// TMA descriptor creation for BF16
CUresult create_tma_2d_descriptor_bf16(CUtensorMap* d, void* globalAddress,
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
        globalAddress, globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, l2Promotion, oobFill);
}

// ======================== Kernel ========================

constexpr int BM = 128;
constexpr int BN = 256;
constexpr int BK = 64;
constexpr int NUM_THREADS = 128;

__global__ __launch_bounds__(NUM_THREADS, 1) void gemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    __nv_bfloat16* __restrict__ C, int M, int N, int K) {

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int m_block = blockIdx.x;
    const int n_block = blockIdx.y;
    const int m_start = m_block * BM;
    const int n_start = n_block * BN;

    // Shared memory layout
    extern __shared__ __align__(1024) char smem[];
    __nv_bfloat16* A_smem[2];
    __nv_bfloat16* B_smem[2];
    A_smem[0] = reinterpret_cast<__nv_bfloat16*>(smem);
    A_smem[1] = A_smem[0] + BM * BK;
    B_smem[0] = A_smem[1] + BM * BK;
    B_smem[1] = B_smem[0] + BN * BK;
    uint64_t* full_bar = reinterpret_cast<uint64_t*>(B_smem[1] + BN * BK);
    uint64_t* empty_bar = full_bar + 2;

    __shared__ uint32_t tmem_alloc_result;

    // Initialize barriers
    if (tid == 0) {
        init_smem_barrier_fn(&full_bar[0], 1);
        init_smem_barrier_fn(&full_bar[1], 1);
        init_smem_barrier_fn(&empty_bar[0], 1);
        init_smem_barrier_fn(&empty_bar[1], 1);
        // Pre-arrive empty barriers to mark buffers as free
        mbarrier_arrive_fn(&empty_bar[0]);
        mbarrier_arrive_fn(&empty_bar[1]);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();

    // Allocate TMEM: 256 columns for 128 rows x 256 cols FP32
    if (warp_id == 0) {
        tmem_alloc_cg1_fn(&tmem_alloc_result, 256);
    }
    __syncthreads();
    const uint32_t tmem_addr = tmem_alloc_result;

    // Prefetch TMA descriptors
    if (tid == 0) {
        prefetch_tma_descriptor_fn(&tma_A);
        prefetch_tma_descriptor_fn(&tma_B);
    }

    const int num_k_tiles = K / BK;  // 80
    const uint32_t idesc = make_instr_desc_fn(BM, BN);
    constexpr uint32_t tx_bytes = BM * BK * 2 + BN * BK * 2;  // 49152

    // Prologue: TMA load stage 0
    if (tid == 0) {
        tma_load_2d_fn(&tma_A, &full_bar[0], A_smem[0], 0, m_start);
        tma_load_2d_fn(&tma_B, &full_bar[0], B_smem[0], 0, n_start);
        mbarrier_arrive_and_expect_tx_fn(&full_bar[0], tx_bytes);
    }

    uint32_t phase_full[2] = {0, 0};
    uint32_t phase_empty[2] = {1, 1};
    int stage = 0;
    int prev_stage = 0;

    for (int k = 0; k < num_k_tiles; k++) {
        // Wait for previous stage's UMMA to complete (accumulation ordering)
        if (k > 0) {
            mbarrier_wait_fn(&empty_bar[prev_stage], phase_empty[prev_stage]);
            phase_empty[prev_stage] ^= 1;
        }

        // Wait for TMA data of current stage
        mbarrier_wait_fn(&full_bar[stage], phase_full[stage]);
        phase_full[stage] ^= 1;

        // Issue 4 MMAs (each K=16, total BK=64 per stage)
        for (int kk = 0; kk < 4; kk++) {
            uint64_t desc_a = make_smem_desc_sm100_fn(A_smem[stage] + kk * 16, 1, 1024);
            uint64_t desc_b = make_smem_desc_sm100_fn(B_smem[stage] + kk * 16, 1, 1024);
            uint32_t accum = (k == 0 && kk == 0) ? 0 : 1;
            if (tid == 0) {
                umma_f16_cg1_fn(tmem_addr, desc_a, desc_b, idesc, accum);
            }
        }
        // Commit all MMAs for this stage
        if (tid == 0) {
            umma_commit_cg1_fn(&empty_bar[stage]);
        }

        // Pre-fetch next stage (no explicit wait needed — buffer freed by UMMA wait above)
        int next_stage = 1 - stage;
        if (k + 1 < num_k_tiles) {
            if (tid == 0) {
                tma_load_2d_fn(&tma_A, &full_bar[next_stage], A_smem[next_stage],
                               (k + 1) * BK, m_start);
                tma_load_2d_fn(&tma_B, &full_bar[next_stage], B_smem[next_stage],
                               (k + 1) * BK, n_start);
                mbarrier_arrive_and_expect_tx_fn(&full_bar[next_stage], tx_bytes);
            }
        }

        prev_stage = stage;
        stage = next_stage;
    }

    // Wait for last UMMA to complete
    mbarrier_wait_fn(&empty_bar[prev_stage], phase_empty[prev_stage]);

    // ======================== Epilogue ========================
    // Reuse B_smem space for output staging (128 x 256 BF16 = 64KB)
    __nv_bfloat16* smem_out = B_smem[0];

    // Phase 1: TMEM (FP32) -> SMEM (BF16)
    // Each of 128 threads loads its own row (tid -> row tid)
    for (uint32_t col = 0; col < (uint32_t)BN; col += 4) {
        uint32_t r0, r1, r2, r3;
        uint32_t addr = tmem_addr + col;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        uint32_t base = (uint32_t)tid * BN + col;
        smem_out[base + 0] = __float2bfloat16(__uint_as_float(r0));
        smem_out[base + 1] = __float2bfloat16(__uint_as_float(r1));
        smem_out[base + 2] = __float2bfloat16(__uint_as_float(r2));
        smem_out[base + 3] = __float2bfloat16(__uint_as_float(r3));
    }
    __syncthreads();

    // Phase 2: SMEM -> Global (coalesced 128-bit stores)
    // 4 warps, each step processes 4 rows (one per warp), 32 steps total
    for (uint32_t step = 0; step < (uint32_t)(BM / 4); step++) {
        uint32_t row = step * 4 + warp_id;
        uint32_t global_row = m_start + row;
        if (global_row >= (uint32_t)M) continue;
        uint32_t col_start = lane_id * 8;
        uint32_t global_col = n_start + col_start;
        // 128-bit store (8 BF16 values)
        uint4 data = *reinterpret_cast<uint4*>(&smem_out[row * BN + col_start]);
        *reinterpret_cast<uint4*>(C + (uint64_t)global_row * N + global_col) = data;
    }

    // Deallocate TMEM
    __syncthreads();
    if (warp_id == 0) {
        tmem_dealloc_cg1_fn(tmem_addr, 256);
    }
}

// ======================== Host Function ========================

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));

    const int M = (int)A.size(0);
    const int N = 7168;
    const int K = 5120;

    __nv_bfloat16* A_ptr = static_cast<__nv_bfloat16*>(A.data_ptr());
    __nv_bfloat16* B_ptr = static_cast<__nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* C_ptr = static_cast<__nv_bfloat16*>(C.data_ptr());

    // Create TMA descriptors
    CUtensorMap tma_A_desc, tma_B_desc;

    // A: (M, K) BF16, K-major. TMA inner=K, outer=M. Box=(BK, BM)=(64, 128)
    CU_CHECK(create_tma_2d_descriptor_bf16(&tma_A_desc, A_ptr, K, M, BK, BM,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA));

    // B: (N, K) BF16, K-major. TMA inner=K, outer=N. Box=(BK, BN)=(64, 256)
    CU_CHECK(create_tma_2d_descriptor_bf16(&tma_B_desc, B_ptr, K, N, BK, BN,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA));

    // Grid and block configuration
    const int m_tiles = (M + BM - 1) / BM;
    const int n_tiles = N / BN;  // 28
    dim3 grid(m_tiles, n_tiles, 1);
    dim3 block(NUM_THREADS, 1, 1);

    // Shared memory: 2 * (A_tile + B_tile) + barriers
    // A_tile = 128*64*2 = 16384, B_tile = 256*64*2 = 32768
    // Total = 2 * (16384 + 32768) + 64 = 98368
    const int smem_bytes = 2 * (BM * BK * 2 + BN * BK * 2) + 64;

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(C.device().device_type, C.device().device_id));

    // Set max dynamic shared memory
    CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

    gemm_kernel<<<grid, block, smem_bytes, stream>>>(
        tma_A_desc, tma_B_desc, C_ptr, M, N, K);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_cuda::run);

}  // namespace gemm_cuda