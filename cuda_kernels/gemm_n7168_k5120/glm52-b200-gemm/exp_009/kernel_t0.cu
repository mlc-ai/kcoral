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
    CUresult _r = (call); \
    if (_r != CUDA_SUCCESS) { \
        const char* errStr; \
        cuGetErrorString(_r, &errStr); \
        fprintf(stderr, "CU error %s at %s:%d\n", \
                errStr ? errStr : "unknown", __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

namespace gemm_cuda {

constexpr int BM = 128;
constexpr int BN = 256;
constexpr int BK_CHUNK = 64;
constexpr int BK = 16;
constexpr int NUM_THREADS = 128;
constexpr int SMEM_SIZE = 163968;

// ---- TMA descriptor creation ----
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

// ---- MBarrier operations ----
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

// ---- TMA load/store ----
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

__device__ __forceinline__ void tma_store_fence_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tma_store_commit_fn() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void tma_store_wait_fn() {
    asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ void prefetch_tma_descriptor_fn(const CUtensorMap* d) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(d) : "memory");
}

// ---- SM100 UMMA descriptor helpers ----
__device__ __forceinline__ uint64_t make_smem_desc_128b_swizzle_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;  // SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    // c_format = FP32
    d |= (1u << 7);    // a_format = BF16
    d |= (1u << 10);   // b_format = BF16
    d |= (0u << 15);   // a K-Major
    d |= (0u << 16);   // b K-Major
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

// ---- TMEM allocation (cta_group::1) ----
__device__ __forceinline__ void tmem_alloc_cg1_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_cg1_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
        :: "r"(addr), "r"(ncols));
}

// ---- UMMA (cta_group::1) ----
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
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cta.b64 [%0];"
        :: "r"(a));
}

// ---- Kernel ----
__global__ void gemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    const __grid_constant__ CUtensorMap tma_C,
    int M, int N, int K) {

    extern __shared__ __align__(128) uint8_t smem_raw[];

    // Shared memory layout:
    // smem_A: 2 stages * 128 * 64 * 2 = 32768 bytes (offset 0)
    // smem_B: 2 stages * 256 * 64 * 2 = 65536 bytes (offset 32768)
    // mbar:   3 * 8 = 24 bytes (offset 98304)
    // tmem:   4 bytes (offset 98328)
    // pad to 98432 (128-byte aligned)
    // smem_out: 128 * 256 * 2 = 65536 bytes (offset 98432)
    __nv_bfloat16* smem_A_base = (__nv_bfloat16*)smem_raw;
    __nv_bfloat16* smem_B_base = (__nv_bfloat16*)(smem_raw + 32768);
    uint64_t* mbar = (uint64_t*)(smem_raw + 98304);
    uint32_t* tmem_addr_ptr = (uint32_t*)(smem_raw + 98328);
    __nv_bfloat16* smem_out = (__nv_bfloat16*)(smem_raw + 98432);

    int tid = threadIdx.x;
    int bm_idx = blockIdx.y;
    int bn_idx = blockIdx.x;
    int m_offset = bm_idx * BM;
    int n_offset = bn_idx * BN;

    // Prefetch TMA descriptors
    if (tid == 0) {
        prefetch_tma_descriptor_fn(&tma_A);
        prefetch_tma_descriptor_fn(&tma_B);
        prefetch_tma_descriptor_fn(&tma_C);
    }

    // Allocate TMEM (warp 0 only) - 256 columns for BN=256 FP32 accumulator
    if (tid < 32) {
        tmem_alloc_cg1_fn(tmem_addr_ptr, 256);
    }
    __syncthreads();
    uint32_t tmem_addr = *tmem_addr_ptr;

    // Initialize mbarriers
    if (tid == 0) {
        init_smem_barrier_fn(&mbar[0], 1);
        init_smem_barrier_fn(&mbar[1], 1);
        init_smem_barrier_fn(&mbar[2], 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();

    int num_k_chunks = K / BK_CHUNK;  // 5120/64 = 80

    constexpr uint32_t tile_bytes_A = BM * BK_CHUNK * 2;   // 16384
    constexpr uint32_t tile_bytes_B = BN * BK_CHUNK * 2;   // 32768
    constexpr uint32_t tile_bytes_total = tile_bytes_A + tile_bytes_B; // 49152

    // Prefetch stage 0
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(&mbar[0], tile_bytes_total);
        tma_load_2d_fn(&tma_A, &mbar[0], smem_A_base, 0, m_offset);
        tma_load_2d_fn(&tma_B, &mbar[0], smem_B_base, 0, n_offset);
    }

    uint32_t idesc = make_instr_desc_fn(BM, BN);

    // SMEM descriptor constants for K-major, 128B swizzle:
    // SBO = 8 * 128 = 1024, LBO = 1 (not used for swizzle)
    constexpr uint32_t SBO = 1024;
    constexpr uint32_t LBO = 1;

    for (int kc = 0; kc < num_k_chunks; kc++) {
        int stage = kc % 2;
        uint32_t tma_phase = (kc / 2) % 2;

        __nv_bfloat16* smem_A = smem_A_base + stage * (BM * BK_CHUNK);
        __nv_bfloat16* smem_B = smem_B_base + stage * (BN * BK_CHUNK);

        // Wait for TMA
        mbarrier_wait_fn(&mbar[stage], tma_phase);

        // Prefetch next chunk
        if (kc + 1 < num_k_chunks) {
            int next_stage = (kc + 1) % 2;
            __nv_bfloat16* next_A = smem_A_base + next_stage * (BM * BK_CHUNK);
            __nv_bfloat16* next_B = smem_B_base + next_stage * (BN * BK_CHUNK);
            if (tid == 0) {
                mbarrier_arrive_and_expect_tx_fn(&mbar[next_stage], tile_bytes_total);
                tma_load_2d_fn(&tma_A, &mbar[next_stage], next_A, (kc + 1) * BK_CHUNK, m_offset);
                tma_load_2d_fn(&tma_B, &mbar[next_stage], next_B, (kc + 1) * BK_CHUNK, n_offset);
            }
        }

        // Issue 4 MMAs (K=16 each) for this K_CHUNK=64 tile
        if (tid == 0) {
            for (int ks = 0; ks < 4; ks++) {
                // Advance base address by ks*16 BF16 elements = ks*32 bytes in K direction
                uint64_t desc_a = make_smem_desc_128b_swizzle_fn(smem_A + ks * 16, LBO, SBO);
                uint64_t desc_b = make_smem_desc_128b_swizzle_fn(smem_B + ks * 16, LBO, SBO);
                int k_global = kc * 4 + ks;
                umma_f16_cg1_fn(tmem_addr, desc_a, desc_b, idesc, k_global > 0 ? 1 : 0);
            }
            umma_commit_cg1_fn(&mbar[2]);
        }

        // Wait for all 4 MMAs to complete
        mbarrier_wait_fn(&mbar[2], kc % 2);
    }

    // ---- Epilogue: TMEM -> SMEM -> TMA store ----
    // Phase 1: Load FP32 from TMEM, convert to BF16, write to SMEM
    // Each warp (32 threads) loads 32 lanes. 4 warps cover 128 lanes (M=128).
    // Load 4 columns at a time, batch 4 loads before waiting.
    for (uint32_t col = 0; col < (uint32_t)BN; col += 16) {
        uint32_t r[4][4];
        for (int i = 0; i < 4; i++) {
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r[i][0]), "=r"(r[i][1]), "=r"(r[i][2]), "=r"(r[i][3])
                : "r"(tmem_addr + col + i * 4));
        }
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");

        for (int i = 0; i < 4; i++) {
            uint32_t base = tid * BN + col + i * 4;
            smem_out[base + 0] = __float2bfloat16(__uint_as_float(r[i][0]));
            smem_out[base + 1] = __float2bfloat16(__uint_as_float(r[i][1]));
            smem_out[base + 2] = __float2bfloat16(__uint_as_float(r[i][2]));
            smem_out[base + 3] = __float2bfloat16(__uint_as_float(r[i][3]));
        }
    }
    __syncthreads();

    // Phase 2: TMA store from SMEM to global C
    if (tid == 0) {
        tma_store_fence_fn();
        tma_store_2d_fn(&tma_C, smem_out, n_offset, m_offset);
        tma_store_commit_fn();
        tma_store_wait_fn<0>();
    }
    __syncthreads();

    // Deallocate TMEM (warp 0)
    if (tid < 32) {
        tmem_dealloc_cg1_fn(tmem_addr, 256);
    }
}

// ---- Host function ----
void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));

    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(0);

    __nv_bfloat16* A_ptr = static_cast<__nv_bfloat16*>(A.data_ptr());
    __nv_bfloat16* B_ptr = static_cast<__nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* C_ptr = static_cast<__nv_bfloat16*>(C.data_ptr());

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

    // Create TMA descriptors
    CUtensorMap tma_A, tma_B, tma_C;

    // A: [M, K] BF16, K contiguous. globalDim=[K, M], boxDim=[64, 128], 128B swizzle
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_A, A_ptr, K, M, BK_CHUNK, BM,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    // B: [N, K] BF16, K contiguous. globalDim=[K, N], boxDim=[64, 256], 128B swizzle
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_B, B_ptr, K, N, BK_CHUNK, BN,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    // C: [M, N] BF16, N contiguous. globalDim=[N, M], boxDim=[256, 128], no swizzle
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_C, C_ptr, N, M, BN, BM,
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    int grid_x = (N + BN - 1) / BN;   // 7168/256 = 28
    int grid_y = (M + BM - 1) / BM;
    dim3 grid(grid_x, grid_y, 1);
    dim3 block(NUM_THREADS, 1, 1);

    CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));

    gemm_kernel<<<grid, block, SMEM_SIZE, stream>>>(tma_A, tma_B, tma_C, (int)M, (int)N, (int)K);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_cuda::run);

}  // namespace gemm_cuda