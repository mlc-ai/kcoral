#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_RT_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                          \
    if (_e != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                  \
                cudaGetErrorString(_e), __FILE__, __LINE__);          \
        exit(1);                                                      \
    }                                                                 \
} while(0)

#define CUDA_DRV_CHECK(call) do {                                     \
    CUresult _r = (call);                                             \
    if (_r != CUDA_SUCCESS) {                                         \
        const char *_msg;                                             \
        cuGetErrorName(_r, &_msg);                                    \
        fprintf(stderr, "CUDA Driver error %s at %s:%d\n",           \
                _msg, __FILE__, __LINE__);                            \
        exit(1);                                                      \
    }                                                                 \
} while(0)

namespace tvm_ffi_gemm_blackwell {

CUresult create_tma_2d_descriptor(
    CUtensorMap* d, void* globalAddress,
    uint64_t gmem_inner_dim, uint64_t gmem_outer_dim,
    uint32_t smem_inner_dim, uint32_t smem_outer_dim,
    CUtensorMapDataType dataType,
    CUtensorMapSwizzle swizzle,
    CUtensorMapL2promotion l2Promotion,
    CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d, dataType, 2, globalAddress,
        globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, l2Promotion, oobFill);
}

constexpr uint32_t BLOCK_M = 64;
constexpr uint32_t BLOCK_N = 128;
constexpr uint32_t BLOCK_K = 16;
constexpr uint32_t NUM_THREADS = 128;

constexpr uint32_t SMEM_A_BYTES = BLOCK_M * BLOCK_K * 2;
constexpr uint32_t SMEM_B_BYTES = BLOCK_N * BLOCK_K * 2;
constexpr uint32_t TOTAL_SMEM_BYTES = 128 + SMEM_A_BYTES + SMEM_B_BYTES;

// ---- Barrier helpers ----

__device__ __forceinline__ void init_mbarrier_shared(void* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
                 :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(count));
}

__device__ __forceinline__ void fence_mbarrier_init_release() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_try_wait_parity(void* bar, uint32_t parity) {
    asm volatile(
        "{\n\t"
        ".reg .pred P_%=;\n\t"
        "1%=: mbarrier.try_wait.parity.shared.b64 P_%=, [%0], %1;\n\t"
        "@!P_%= bra 1%=;\n\t"
        "}\n\t"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(parity) : "memory");
}

// ---- TMA load ----

__device__ __forceinline__ void tma_load_2d(const CUtensorMap* desc, void* bar, void* smem_dst,
                                            int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem_dst)),
           "l"(desc),
           "r"((uint32_t)__cvta_generic_to_shared(bar)),
           "r"(c0), "r"(c1) : "memory");
}

// ---- Descriptor builders ----

__device__ __forceinline__ uint64_t build_smem_desc(void* ptr, uint32_t lbo_bytes, uint32_t sbo_bytes) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    d |= ((addr & 0x3FFFF) >> 4);
    d |= ((uint64_t)((lbo_bytes & 0x3FFFF) >> 4)) << 16;
    d |= ((uint64_t)((sbo_bytes & 0x3FFFF) >> 4)) << 32;
    d |= 1ULL << 46;       // version = 1 (SM100)
    d |= 2ULL << 61;       // swizzle = 128B
    return d;
}

__device__ __forceinline__ uint32_t build_idesc(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= 1u << 4;                   // dtype = FP32
    d |= 1u << 7;                   // atype = BF16
    d |= 1u << 10;                  // btype = BF16
    // transpose A = 0 (K-major), transpose B = 0 (K-major)
    d |= ((N >> 3) & 0x3Fu) << 17;  // N dim (lower 3 bits implied as zero)
    d |= ((M >> 4) & 0x1Fu) << 24;  // M dim (lower 4 bits implied as zero)
    return d;
}

// =====================================================================
// Kernel
// =====================================================================

__global__ void gemm_kernel(const __nv_bfloat16* __restrict__ A,
                             const __nv_bfloat16* __restrict__ B,
                             __nv_bfloat16* __restrict__ C,
                             const __grid_constant__ CUtensorMap tma_a,
                             const __grid_constant__ CUtensorMap tma_b,
                             uint32_t M, uint32_t N, uint32_t K) {

    extern __shared__ unsigned char smem[];

    // Barrier is first 8 bytes of shared memory (aligned to 8B)
    uint64_t* bar = reinterpret_cast<uint64_t*>(smem);
    // Data buffers after 128-byte aligned region
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem + 128);
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem + 128 + SMEM_A_BYTES);

    // Initialize barrier (thread 0 only)
    if (threadIdx.x == 0) {
        init_mbarrier_shared(bar, 2);   // expect 2 TMA arrivals
        fence_mbarrier_init_release();
    }
    __syncthreads();

    // Block grid mapping
    uint32_t bx = blockIdx.x;
    uint32_t n_blocks_x = (N + BLOCK_N - 1) / BLOCK_N;
    uint32_t mb = bx / n_blocks_x;
    uint32_t nb = bx % n_blocks_x;

    uint32_t a_row_off = mb * BLOCK_M;
    uint32_t b_col_off = nb * BLOCK_N;
    uint32_t k_iters = K / BLOCK_K;

    // Build shared-memory descriptors (K-major, 128B swizzle)
    // SBO = 8 spans * 128B = 1024 for both matrices
    // LBO = (ATOM_MMODE_DIM / 8) * SBO where ATOM_MMODE_DIM = BM or BN
    uint32_t sbo = 1024;
    uint64_t desc_a = build_smem_desc(smem_a, (BLOCK_M / 8) * sbo, sbo);
    uint64_t desc_b = build_smem_desc(smem_b, (BLOCK_N / 8) * sbo, sbo);

    // UMMA instruction descriptor: BF16×BF16→FP32, no transpose
    uint32_t idesc = build_idesc(BLOCK_M, BLOCK_N);

    // Allocate Tensor Memory (BLOCK_N columns of FP32)
    uint32_t tmem_c_addr;
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 : "=r"(tmem_c_addr) : "r"(BLOCK_N));

    // Clear accumulator: D = A*B with enable-input-d = false
    asm volatile("{\n\t"
        ".reg .pred p_clr;\n\t"
        "setp.eq.b32 p_clr, 0, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p_clr;\n\t"
        "}" :: "r"(tmem_c_addr), "l"(desc_a), "l"(desc_b), "r"(idesc));
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
                 :: "r"((uint32_t)__cvta_generic_to_shared(bar)));
    if (threadIdx.x == 0) { mbarrier_try_wait_parity(bar, 0); }
    __syncthreads();

    // Prefetch TMA descriptors
    if (threadIdx.x == 0) {
        asm volatile("prefetch.tensormap [%0];" :: "l"(&tma_a));
        asm volatile("prefetch.tensormap [%0];" :: "l"(&tma_b));
    }

    // ===== Main K-loop =====
    for (uint32_t ki = 0; ki < k_iters; ki++) {
        uint32_t ko = ki * BLOCK_K;

        // Issue async TMA loads (each completes 1 tx on barrier)
        tma_load_2d(&tma_a, bar, smem_a, static_cast<int32_t>(ko), static_cast<int32_t>(a_row_off));
        tma_load_2d(&tma_b, bar, smem_b, static_cast<int32_t>(ko), static_cast<int32_t>(b_col_off));

        // Wait for both TMA loads
        if (threadIdx.x == 0) { mbarrier_try_wait_parity(bar, 0); }
        __syncthreads();

        // Execute UMMA: C += A @ B  (enable-input-d = true via predicate)
        asm volatile("{\n\t"
            ".reg .pred p_acc;\n\t"
            "setp.ne.b32 p_acc, 1, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p_acc;\n\t"
            "}" :: "r"(tmem_c_addr), "l"(desc_a), "l"(desc_b), "r"(idesc));

        // Commit MMA result to barrier
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
                     :: "r"((uint32_t)__cvta_generic_to_shared(bar)));

        // Wait before next iteration
        if (threadIdx.x == 0) { mbarrier_try_wait_parity(bar, 0); }
        __syncthreads();
    }

    // ===== Epilogue: TMEM (FP32) → BF16 → Global C =====
    // TMEM layout: 128 lanes × cols; warp r accesses lanes r*32 .. r*32+31
    // We have BLOCK_M=64 rows → 2 warps needed (warp 0: rows 0-31, warp 1: rows 32-63)
    uint32_t tid = threadIdx.x;
    uint32_t warp_r = tid / 32;  // 0..3
    uint32_t lane   = tid % 32;  // 0..31

    // Each thread handles one row within its accessible lane range
    uint32_t local_row = warp_r * 32 + lane;  // maps to lane index in TMEM

    if (local_row < BLOCK_M) {
        uint32_t global_row = a_row_off + local_row;
        if (global_row < M) {
            for (uint32_t col = 0; col < BLOCK_N; col += 4) {
                uint32_t r0, r1, r2, r3;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                    : "r"(static_cast<uint32_t>(col)));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");

                float f0 = __uint_as_float(r0);
                float f1 = __uint_as_float(r1);
                float f2 = __uint_as_float(r2);
                float f3 = __uint_as_float(r3);

                uint32_t nc = b_col_off + col;
                __nv_bfloat16* out = C + (uint64_t)global_row * N + nc;

                if (nc + 3 < N) {
                    out[0] = __float2bfloat16(f0);
                    out[1] = __float2bfloat16(f1);
                    out[2] = __float2bfloat16(f2);
                    out[3] = __float2bfloat16(f3);
                } else {
                    if (nc < N)     out[0] = __float2bfloat16(f0);
                    if (nc + 1 < N) out[1] = __float2bfloat16(f1);
                    if (nc + 2 < N) out[2] = __float2bfloat16(f2);
                }
            }
        }
    }

    // Deallocate Tensor Memory
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(tmem_c_addr), "r"(BLOCK_N));
}

// =====================================================================
// Host runner
// =====================================================================

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_RT_CHECK(cudaSetDevice(A.device().device_id));

    uint32_t M = static_cast<uint32_t>(A.size(0));
    uint32_t N = 7168u;
    uint32_t K = 5120u;

    const __nv_bfloat16* A_data = static_cast<const __nv_bfloat16*>(A.data_ptr());
    const __nv_bfloat16* B_data = static_cast<const __nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* C_data = static_cast<__nv_bfloat16*>(C.data_ptr());

    // Build TMA descriptors (host-side)
    CUtensorMap h_tma_a, h_tma_b;

    CUDA_DRV_CHECK(create_tma_2d_descriptor(
        &h_tma_a, const_cast<void*>(static_cast<const void*>(A_data)),
        K, M, BLOCK_K, BLOCK_M,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    CUDA_DRV_CHECK(create_tma_2d_descriptor(
        &h_tma_b, const_cast<void*>(static_cast<const void*>(B_data)),
        K, N, BLOCK_K, BLOCK_N,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    // Copy descriptors to device
    CUtensorMap *d_ta = nullptr, *d_tb = nullptr;
    CUDA_DRV_CHECK(cuMemAlloc(reinterpret_cast<CUdeviceptr*>(&d_ta), sizeof(CUtensorMap)));
    CUDA_DRV_CHECK(cuMemAlloc(reinterpret_cast<CUdeviceptr*>(&d_tb), sizeof(CUtensorMap)));

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

    CUDA_DRV_CHECK(cuMemcpyHtoDAsync(
        reinterpret_cast<CUdeviceptr>(d_ta), &h_tma_a, sizeof(CUtensorMap), stream));
    CUDA_DRV_CHECK(cuMemcpyHtoDAsync(
        reinterpret_cast<CUdeviceptr>(d_tb), &h_tma_b, sizeof(CUtensorMap), stream));

    // Launch
    dim3 blk(NUM_THREADS);
    uint32_t gm = (M + BLOCK_M - 1) / BLOCK_M;
    uint32_t gn = (N + BLOCK_N - 1) / BLOCK_N;
    dim3 grd(gm * gn);

    gemm_kernel<<<grd, blk, TOTAL_SMEM_BYTES, stream>>>(
        A_data, B_data, C_data, *d_ta, *d_tb, M, N, K);

    CUDA_RT_CHECK(cudaGetLastError());
    CUDA_RT_CHECK(cudaStreamSynchronize(stream));

    CUDA_DRV_CHECK(cuMemFree(reinterpret_cast<CUdeviceptr>(d_ta)));
    CUDA_DRV_CHECK(cuMemFree(reinterpret_cast<CUdeviceptr>(d_tb)));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_gemm_blackwell::run);

}  // namespace tvm_ffi_gemm_blackwell