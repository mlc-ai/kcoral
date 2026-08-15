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

// ============ Configuration ============
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 16;        // matches MMA K dimension
constexpr int NUM_STAGES = 8;
constexpr int NUM_THREADS = 128;

// ============ Helper functions from reference ============

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

__device__ __forceinline__ void tcgen05_fence_before_fn() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

// ============ Custom functions for cta_group::1 ============

__device__ __forceinline__ void tma_load_2d_cta_fn(
    const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc_32b_fn(void* smem_ptr, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    // LBO not used for K-major swizzled, leave as 0
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;    // version = 1 (SM100)
    d |= (uint64_t)6 << 61;    // SWIZZLE_32B
    return d;
}

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
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
        :: "r"(a));
}

__device__ __forceinline__ void tmem_alloc_cg1_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_cg1_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
        :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);             // c_format = FP32
    d |= (1u << 7);             // a_format = BF16
    d |= (1u << 10);            // b_format = BF16
    d |= (0u << 15);            // a_major = 0 (K-Major, No Transpose)
    d |= (0u << 16);            // b_major = 0 (K-Major, No Transpose)
    d |= ((N / 8) << 17);       // n_dim
    d |= ((M / 16) << 24);      // m_dim
    return d;
}

// ============ Kernel ============

__global__ __launch_bounds__(NUM_THREADS) void gemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    __nv_bfloat16* C,
    int M, int N, int K) {

    // ---- SMEM Layout ----
    constexpr int A_STAGE_BYTES = BM * BK * 2;   // 4096
    constexpr int B_STAGE_BYTES = BN * BK * 2;   // 4096
    constexpr int EPI_BYTES = BM * BN * 2;        // 32768

    extern __shared__ __align__(256) char smem_raw[];

    // Align base to 256 bytes
    char* base = (char*)(((uintptr_t)smem_raw + 255) & ~255);

    // A stages
    __nv_bfloat16* smem_A[NUM_STAGES];
    for (int i = 0; i < NUM_STAGES; i++) {
        smem_A[i] = reinterpret_cast<__nv_bfloat16*>(base + i * A_STAGE_BYTES);
    }
    char* ptr = base + NUM_STAGES * A_STAGE_BYTES;

    // B stages
    __nv_bfloat16* smem_B[NUM_STAGES];
    for (int i = 0; i < NUM_STAGES; i++) {
        smem_B[i] = reinterpret_cast<__nv_bfloat16*>(ptr + i * B_STAGE_BYTES);
    }
    ptr += NUM_STAGES * B_STAGE_BYTES;

    // Barriers
    uint64_t* tma_bar = reinterpret_cast<uint64_t*>(ptr);
    ptr += NUM_STAGES * 8;
    uint64_t* umma_bar = reinterpret_cast<uint64_t*>(ptr);
    ptr += 8;
    uint32_t* tmem_addr_smem = reinterpret_cast<uint32_t*>(ptr);
    ptr += 4;

    // Epilogue buffer (align to 256)
    ptr = (char*)(((uintptr_t)ptr + 255) & ~255);
    __nv_bfloat16* smem_epi = reinterpret_cast<__nv_bfloat16*>(ptr);

    // ---- Tile indices ----
    int m_block = blockIdx.x;
    int n_block = blockIdx.y;
    int m_start = m_block * BM;
    int n_start = n_block * BN;
    int num_k_tiles = K / BK;  // 5120 / 16 = 320

    // ---- Initialize barriers ----
    if (threadIdx.x == 0) {
        for (int i = 0; i < NUM_STAGES; i++) {
            init_smem_barrier_fn(&tma_bar[i], 1);
        }
        init_smem_barrier_fn(umma_bar, 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();

    // ---- Allocate TMEM (warp 0) ----
    if (threadIdx.x < 32) {
        tmem_alloc_cg1_fn(tmem_addr_smem, 128);
    }
    __syncthreads();

    uint32_t tmem_addr = *tmem_addr_smem;

    // ---- Build descriptors ----
    uint64_t desc_A[NUM_STAGES];
    uint64_t desc_B[NUM_STAGES];
    // K-major, 32B swizzle: SBO = 8 * 32 = 256
    for (int i = 0; i < NUM_STAGES; i++) {
        desc_A[i] = make_smem_desc_32b_fn(smem_A[i], 256);
        desc_B[i] = make_smem_desc_32b_fn(smem_B[i], 256);
    }
    uint32_t idesc = make_instr_desc_fn(BM, BN);
    uint32_t tma_bytes = A_STAGE_BYTES + B_STAGE_BYTES;  // 8192

    // ---- Prologue: issue first NUM_STAGES TMA loads ----
    if (threadIdx.x == 0) {
        for (int i = 0; i < NUM_STAGES && i < num_k_tiles; i++) {
            mbarrier_arrive_and_expect_tx_fn(&tma_bar[i], tma_bytes);
            tma_load_2d_cta_fn(&tma_A, &tma_bar[i], smem_A[i], i * BK, m_start);
            tma_load_2d_cta_fn(&tma_B, &tma_bar[i], smem_B[i], i * BK, n_start);
        }
    }

    // ---- Main loop ----
    uint32_t umma_phase = 0;
    uint32_t tma_phase[NUM_STAGES];
    for (int i = 0; i < NUM_STAGES; i++) tma_phase[i] = 0;

    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        int stage = k_tile % NUM_STAGES;

        if (threadIdx.x == 0) {
            // Wait for TMA data
            mbarrier_wait_fn(&tma_bar[stage], tma_phase[stage]);

            // Issue UMMA
            uint32_t accum = (k_tile == 0) ? 0 : 1;
            umma_f16_cg1_fn(tmem_addr, desc_A[stage], desc_B[stage], idesc, accum);

            // Commit and wait for UMMA completion
            umma_commit_cg1_fn(umma_bar);
            mbarrier_wait_fn(umma_bar, umma_phase);
            umma_phase ^= 1;

            // Issue next TMA load if available
            int next_k = k_tile + NUM_STAGES;
            if (next_k < num_k_tiles) {
                mbarrier_arrive_and_expect_tx_fn(&tma_bar[stage], tma_bytes);
                tma_load_2d_cta_fn(&tma_A, &tma_bar[stage], smem_A[stage], next_k * BK, m_start);
                tma_load_2d_cta_fn(&tma_B, &tma_bar[stage], smem_B[stage], next_k * BK, n_start);
            }

            tma_phase[stage] ^= 1;
        }
    }

    // Fence tcgen05 ops before thread sync
    if (threadIdx.x == 0) {
        tcgen05_fence_before_fn();
    }
    __syncthreads();

    // ============ Epilogue: TMEM -> SMEM -> Global ============

    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t tmem_row = warp_id * 32 + lane_id;  // each warp accesses 32 lanes

    // Phase 1: TMEM -> SMEM (each warp loads its 32 rows, 8 cols at a time)
    for (uint32_t col = 0; col < (uint32_t)BN; col += 8) {
        uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3),
              "=r"(r4), "=r"(r5), "=r"(r6), "=r"(r7)
            : "r"(tmem_addr + col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");

        uint32_t smem_off = tmem_row * BN + col;
        smem_epi[smem_off + 0] = __float2bfloat16(__uint_as_float(r0));
        smem_epi[smem_off + 1] = __float2bfloat16(__uint_as_float(r1));
        smem_epi[smem_off + 2] = __float2bfloat16(__uint_as_float(r2));
        smem_epi[smem_off + 3] = __float2bfloat16(__uint_as_float(r3));
        smem_epi[smem_off + 4] = __float2bfloat16(__uint_as_float(r4));
        smem_epi[smem_off + 5] = __float2bfloat16(__uint_as_float(r5));
        smem_epi[smem_off + 6] = __float2bfloat16(__uint_as_float(r6));
        smem_epi[smem_off + 7] = __float2bfloat16(__uint_as_float(r7));
    }
    __syncthreads();

    // Phase 2: SMEM -> Global (coalesced, each warp writes multiple rows)
    // 32 threads per warp, each writes 4 BF16 (8 bytes) -> 32*4=128 cols per row
    // BN=128, so one pass covers the full row
    for (uint32_t row = warp_id; row < (uint32_t)BM; row += 4) {
        uint32_t global_row = m_start + row;
        if (global_row >= (uint32_t)M) continue;

        uint32_t smem_off = row * BN + lane_id * 4;
        uint32_t global_col = n_start + lane_id * 4;

        if (global_col + 3 < (uint32_t)N) {
            uint2 data = *reinterpret_cast<uint2*>(&smem_epi[smem_off]);
            *reinterpret_cast<uint2*>(C + (uint64_t)global_row * N + global_col) = data;
        } else {
            for (int j = 0; j < 4 && global_col + j < (uint32_t)N; j++) {
                C[(uint64_t)global_row * N + global_col + j] = smem_epi[smem_off + j];
            }
        }
    }

    __syncthreads();

    // ---- Deallocate TMEM (warp 0) ----
    if (threadIdx.x < 32) {
        tmem_dealloc_cg1_fn(tmem_addr, 128);
    }
}

// ============ Host function ============

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));

    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(0);  // B is [N, K]

    __nv_bfloat16* A_ptr = static_cast<__nv_bfloat16*>(A.data_ptr());
    __nv_bfloat16* B_ptr = static_cast<__nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* C_ptr = static_cast<__nv_bfloat16*>(C.data_ptr());

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

    // ---- Create TMA descriptors ----
    CUtensorMap tma_A, tma_B;

    // A: [M, K] bf16, K is inner (fast) dimension
    {
        cuuint64_t globalDim[2] = {(cuuint64_t)K, (cuuint64_t)M};
        cuuint64_t globalStrides[1] = {(cuuint64_t)K * 2};
        cuuint32_t boxDim[2] = {(cuuint32_t)BK, (cuuint32_t)BM};
        cuuint32_t elementStrides[2] = {1, 1};
        CU_CHECK(cuTensorMapEncodeTiled(
            &tma_A,
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            2,
            A_ptr,
            globalDim,
            globalStrides,
            boxDim,
            elementStrides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_32B,
            CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    }

    // B: [N, K] bf16, K is inner (fast) dimension
    {
        cuuint64_t globalDim[2] = {(cuuint64_t)K, (cuuint64_t)N};
        cuuint64_t globalStrides[1] = {(cuuint64_t)K * 2};
        cuuint32_t boxDim[2] = {(cuuint32_t)BK, (cuuint32_t)BN};
        cuuint32_t elementStrides[2] = {1, 1};
        CU_CHECK(cuTensorMapEncodeTiled(
            &tma_B,
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            2,
            B_ptr,
            globalDim,
            globalStrides,
            boxDim,
            elementStrides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_32B,
            CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    }

    // ---- Grid and block ----
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN, 1);
    dim3 block(NUM_THREADS, 1, 1);

    // ---- SMEM size ----
    // alignment(256) + A(8*4096) + B(8*4096) + barriers(72) + tmem_addr(4) + align(256) + epilogue(32768)
    int smem_bytes = 256
                   + NUM_STAGES * BM * BK * 2
                   + NUM_STAGES * BN * BK * 2
                   + (NUM_STAGES + 1) * 8
                   + 4
                   + 256
                   + BM * BN * 2;

    // Set dynamic SMEM limit
    CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

    gemm_kernel<<<grid, block, smem_bytes, stream>>>(
        tma_A, tma_B, C_ptr, (int)M, (int)N, (int)K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_cuda::run);

}  // namespace gemm_cuda