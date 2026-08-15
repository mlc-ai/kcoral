#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <cstdio>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                    \
    cudaError_t _e = (call);                                     \
    if (_e != cudaSuccess) {                                     \
        fprintf(stderr, "CUDA error %s at %s:%d\n",              \
                cudaGetErrorString(_e), __FILE__, __LINE__);     \
        exit(1);                                                 \
    }                                                            \
} while(0)

#define CU_CHECK(call) do {                                      \
    CUresult _r = (call);                                        \
    if (_r != CUDA_SUCCESS) {                                    \
        const char* err = nullptr;                               \
        cuGetErrorString(_r, &err);                              \
        fprintf(stderr, "CU error %s at %s:%d\n",                \
                err ? err : "unknown", __FILE__, __LINE__);      \
        exit(1);                                                 \
    }                                                            \
} while(0)

namespace gemm_n7168_k5120 {

constexpr int BM = 128;
constexpr int BN = 256;
constexpr int BK = 16;
constexpr int STAGES = 3;
constexpr int THREADS = 128;

struct GemmSmem {
    alignas(256) __nv_bfloat16 A[STAGES][BM][BK];
    alignas(256) __nv_bfloat16 B[STAGES][BN][BK];
    alignas(256) __nv_bfloat16 C[BM][BN];
    uint64_t full_bar[STAGES];
    uint64_t umma_bar;
    uint32_t tmem_addr;
};

// ---------------- TMA helpers ----------------
__device__ __forceinline__ void tma_load_2d_local(
    const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
           "l"((uint64_t)d),
           "r"((uint32_t)__cvta_generic_to_shared(bar)),
           "r"(c0), "r"(c1) : "memory");
}

// ---------------- mbarrier helpers ----------------
__device__ __forceinline__ void init_barrier(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(count));
}

__device__ __forceinline__ void fence_barrier_init() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* bar, uint32_t tx) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(tx) : "memory");
}

__device__ __forceinline__ void mbarrier_wait(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(phase));
}

// ---------------- TMEM helpers ----------------
__device__ __forceinline__ void tmem_alloc(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
        :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void tmem_wait_ld() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

// ---------------- UMMA helpers ----------------
__device__ __forceinline__ uint64_t make_smem_desc(void* ptr, uint32_t lbo, uint32_t sbo, uint32_t swizzle_mode) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    d |= (uint64_t)((addr & 0x3FFFF) >> 4);
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;       // version = 1

    uint32_t pattern_size = (swizzle_mode == 6) ? 256 : 0;
    uint32_t base_offset = (pattern_size > 0) ? ((addr % pattern_size) >> 7) : 0;
    d |= (uint64_t)base_offset << 49;

    d |= (uint64_t)swizzle_mode << 61;
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);               // D type = FP32
    d |= (1u << 7);               // A type = BF16
    d |= (1u << 10);              // B type = BF16
    d |= (0u << 15);              // A K-major (no transpose)
    d |= (0u << 16);              // B K-major (no transpose)
    d |= ((N / 8) << 17);         // N
    d |= ((M / 16) << 24);        // M
    return d;
}

__device__ __forceinline__ void umma_cg1(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.u32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n"
        "}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_cg1(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
        :: "r"(a));
}

// ---------------- Epilogue ----------------
__device__ __forceinline__ void epilogue(
    GemmSmem* smem, __nv_bfloat16* C,
    uint32_t M, uint32_t N, uint32_t m_base, uint32_t n_base) {
    uint32_t tmem = smem->tmem_addr;
    uint32_t tid = threadIdx.x;

    // Phase 1: TMEM (FP32) -> SMEM (BF16)
    for (uint32_t col = 0; col < BN; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
            : "r"(tmem + col));
        tmem_wait_ld();
        __nv_bfloat16* out = &smem->C[tid][col];
        out[0] = __float2bfloat16(__uint_as_float(r0));
        out[1] = __float2bfloat16(__uint_as_float(r1));
        out[2] = __float2bfloat16(__uint_as_float(r2));
        out[3] = __float2bfloat16(__uint_as_float(r3));
    }
    __syncthreads();

    // Phase 2: SMEM -> Global (coalesced 8-byte stores)
    uint32_t warp = tid / 32;
    uint32_t lane = tid % 32;
    for (uint32_t j = 0; j < 2; ++j) {
        for (uint32_t step = 0; step < BM / 4; ++step) {
            uint32_t row = step * 4 + warp;
            uint32_t col_start = lane * 4 + j * 128;
            uint32_t gr = m_base + row;
            uint32_t gc = n_base + col_start;
            if (gr < M && gc + 3 < N) {
                uint2 data = *reinterpret_cast<uint2*>(&smem->C[row][col_start]);
                *reinterpret_cast<uint2*>(C + (uint64_t)gr * N + gc) = data;
            } else if (gr < M) {
                for (int c = 0; c < 4; ++c) {
                    if (gc + c < N) {
                        C[(uint64_t)gr * N + gc + c] = smem->C[row][col_start + c];
                    }
                }
            }
        }
    }
}

// ---------------- Kernel ----------------
__global__ void gemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    __nv_bfloat16* C,
    uint32_t M, uint32_t N, uint32_t K) {

    extern __shared__ char smem_buf[];
    GemmSmem* smem = reinterpret_cast<GemmSmem*>(smem_buf);

    uint32_t tid = threadIdx.x;
    uint32_t warp = tid / 32;

    constexpr uint32_t N_tiles = 7168 / BN;
    uint32_t m_tile = blockIdx.x / N_tiles;
    uint32_t n_tile = blockIdx.x % N_tiles;
    uint32_t m_base = m_tile * BM;
    uint32_t n_base = n_tile * BN;

    // Init barriers
    if (tid == 0) {
        for (int s = 0; s < STAGES; ++s) {
            init_barrier(&smem->full_bar[s], 1);
        }
        init_barrier(&smem->umma_bar, 1);
        fence_barrier_init();
    }
    __syncthreads();

    // Allocate TMEM (256 columns for 128x256 FP32 accumulator)
    if (warp == 0) {
        tmem_alloc(&smem->tmem_addr, 256);
    }
    __syncthreads();
    uint32_t tmem = smem->tmem_addr;

    uint32_t Ksteps = K / BK;
    constexpr uint32_t load_bytes = BM * BK * 2 + BN * BK * 2;

    // Preload first STAGES tiles
    if (tid == 0) {
        for (int s = 0; s < STAGES && s < (int)Ksteps; ++s) {
            mbarrier_arrive_expect_tx(&smem->full_bar[s], load_bytes);
            tma_load_2d_local(&tma_A, &smem->full_bar[s], &smem->A[s][0][0], s * BK, m_base);
            tma_load_2d_local(&tma_B, &smem->full_bar[s], &smem->B[s][0][0], s * BK, n_base);
        }
    }

    uint32_t idesc = make_instr_desc(BM, BN);
    // 32B swizzle mode = 6, LBO = 16 (not used, assumed 1), SBO = 256
    constexpr uint32_t swizzle_32b = 6;
    constexpr uint32_t lbo_val = 16;
    constexpr uint32_t sbo_val = 256;

    for (uint32_t k = 0; k < Ksteps; ++k) {
        int s = k % STAGES;
        uint32_t fphase = (k / STAGES) & 1u;

        mbarrier_wait(&smem->full_bar[s], fphase);

        uint64_t desc_a = make_smem_desc(&smem->A[s][0][0], lbo_val, sbo_val, swizzle_32b);
        uint64_t desc_b = make_smem_desc(&smem->B[s][0][0], lbo_val, sbo_val, swizzle_32b);

        if (tid == 0) {
            umma_cg1(tmem, desc_a, desc_b, idesc, k > 0 ? 1 : 0);
            umma_commit_cg1(&smem->umma_bar);
        }

        uint32_t uphase = k & 1u;
        mbarrier_wait(&smem->umma_bar, uphase);

        // Issue load for k+STAGES
        if (tid == 0 && k + STAGES < Ksteps) {
            int s2 = (k + STAGES) % STAGES;
            mbarrier_arrive_expect_tx(&smem->full_bar[s2], load_bytes);
            tma_load_2d_local(&tma_A, &smem->full_bar[s2], &smem->A[s2][0][0], (k + STAGES) * BK, m_base);
            tma_load_2d_local(&tma_B, &smem->full_bar[s2], &smem->B[s2][0][0], (k + STAGES) * BK, n_base);
        }
    }

    __syncthreads();
    epilogue(smem, C, M, N, m_base, n_base);
    __syncthreads();

    if (warp == 0) {
        tmem_dealloc(tmem, 256);
    }
}

// ---------------- Host ----------------
CUresult create_tma_desc_2d(
    CUtensorMap* d, void* ptr,
    uint64_t gmem_inner, uint64_t gmem_outer,
    uint32_t smem_inner, uint32_t smem_outer,
    CUtensorMapSwizzle swizzle,
    CUtensorMapL2promotion l2prom,
    CUtensorMapFloatOOBfill oob) {
    cuuint64_t globalDim[2] = {gmem_inner, gmem_outer};
    cuuint64_t globalStrides[1] = {gmem_inner * 2};
    cuuint32_t boxDim[2] = {smem_inner, smem_outer};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        ptr,
        globalDim,
        globalStrides,
        boxDim,
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle,
        l2prom,
        oob);
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
    CU_CHECK(create_tma_desc_2d(&tma_A, A_ptr, (uint64_t)K, (uint64_t)M,
                                BK, BM,
                                CU_TENSOR_MAP_SWIZZLE_32B,
                                CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_desc_2d(&tma_B, B_ptr, (uint64_t)K, (uint64_t)N,
                                BK, BN,
                                CU_TENSOR_MAP_SWIZZLE_32B,
                                CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    int smem_size = (int)sizeof(GemmSmem);
    CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));

    uint32_t M_tiles = (uint32_t)((M + BM - 1) / BM);
    uint32_t N_tiles = (uint32_t)(N / BN);
    dim3 grid(M_tiles * N_tiles, 1, 1);
    dim3 block(THREADS, 1, 1);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

    gemm_kernel<<<grid, block, smem_size, stream>>>(
        tma_A, tma_B, C_ptr, (uint32_t)M, (uint32_t)N, (uint32_t)K);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_n7168_k5120::run);

}  // namespace gemm_n7168_k5120