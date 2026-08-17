#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
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

namespace gemm_blackwell {

template <int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ void gemm_kernel(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    __nv_bfloat16* C,
    int M, int N, int K) 
{
    // Thread mapping: each thread handles one row of the BM x BN output tile
    // Plus 2 adjacent N columns for coalescing
    constexpr int THREADS_PER_ROW = 2;
    constexpr int TOTAL_THREADS = BLOCK_M * (BLOCK_N / THREADS_PER_ROW);

    int tid = threadIdx.x;
    int local_m = tid / (BLOCK_N / THREADS_PER_ROW);
    int local_n_base = (tid % (BLOCK_N / THREADS_PER_ROW)) * THREADS_PER_ROW;

    // Shared memory: As[BLOCK_M][BLOCK_K] + Bs[BLOCK_N][BLOCK_K] + Cs[BLOCK_M][BLOCK_N] (fp32)
    extern __shared__ char smem_bytes[];

    __nv_bfloat16* __restrict__ As =
        reinterpret_cast<__nv_bfloat16*>(smem_bytes);

    __nv_bfloat16* __restrict__ Bs =
        reinterpret_cast<__nv_bfloat16*>(smem_bytes + BLOCK_M * BLOCK_K * sizeof(__nv_bfloat16));

    float* __restrict__ Cs =
        reinterpret_cast<float*>(smem_bytes + BLOCK_M * BLOCK_K * sizeof(__nv_bfloat16) +
                                 BLOCK_N * BLOCK_K * sizeof(__nv_bfloat16));

    int m_tile_start = blockIdx.x * BLOCK_M;
    int n_tile_start = blockIdx.y * BLOCK_N;

    // Build instruction descriptor: BF16*BF16->FP32, K-major for both A and B
    // Bits 4-5: dtype=F32(1), Bits 7-9: a_format=BF16(1), Bits 10-12: b_format=BF16(1)
    // Bits 15: TransposeA=0(K-major), Bits 16: TransposeB=0(K-major)
    // Bits 17-22: N>>3, Bits 24-28: M>>4
    uint32_t idesc = (1u << 4) | (1u << 7) | (1u << 10) |
                     ((BLOCK_N >> 3) & 0x3Fu) << 17 |
                     ((BLOCK_M >> 4) & 0x1Fu) << 24;

    // Descriptor layout for K-major 128B swizzle:
    // SBO (stride byte offset) = 8 * BLOCK_K * sizeof(bf16) = offset between groups of 8 rows
    // LBO = 1 (not used for K-major with swizzle)
    uint32_t sbo = 8u * BLOCK_K * sizeof(__nv_bfloat16);
    uint32_t lbo = 1u;
    uint64_t ver_swz = (1ULL << 46) | (2ULL << 61);  // version=1, SWIZZLE_128B

    uint32_t c_addr = (uint32_t)__cvta_generic_to_shared(Cs);

    int num_k_tiles = K / BLOCK_K;

    for (int kt = 0; kt < num_k_tiles; kt++) {
        int k_global_base = kt * BLOCK_K;

        // ---- Phase 1: Cooperatively load A tile [BLOCK_M][BLOCK_K] ----
        // K-major layout: As[m][k] at As[m * BLOCK_K + k]
        if (m_tile_start + local_m < M) {
            for (int ko = 0; ko < BLOCK_K; ko += 2) {
                int kg = k_global_base + ko;
                As[local_m * BLOCK_K + ko]     = A[(size_t)(m_tile_start + local_m) * K + kg];
                As[local_m * BLOCK_K + ko + 1] = A[(size_t)(m_tile_start + local_m) * K + kg + 1];
            }
        }

        // ---- Phase 2: Cooperatively load B tile [BLOCK_N][BLOCK_K] ----
        // K-major layout: Bs[n][k] at Bs[n * BLOCK_K + k]
        for (int th = 0; th < THREADS_PER_ROW; th++) {
            int ln = local_n_base + th;
            if (n_tile_start + ln < N) {
                for (int ko = 0; ko < BLOCK_K; ko += 2) {
                    int kg = k_global_base + ko;
                    Bs[ln * BLOCK_K + ko]     = B[(size_t)(n_tile_start + ln) * K + kg];
                    Bs[ln * BLOCK_K + ko + 1] = B[(size_t)(n_tile_start + ln) * K + kg + 1];
                }
            }
        }

        __syncthreads();

        // ---- Phase 3: Issue UMMA instructions for this K-tile ----
        // Each UMMA contracts K=16, so BLOCK_K/16 sub-iterations
        bool first_k_iter = (kt == 0);

        #pragma unroll
        for (int sub = 0; sub < BLOCK_K / 16; sub++) {
            int k_off = sub * 16;

            if (tid == 0) {
                // Base pointers for the current K-slice within shared memory
                __nv_bfloat16* a_base = As + k_off * BLOCK_M;
                __nv_bfloat16* b_base = Bs + k_off * BLOCK_N;

                uint32_t a_saddr = (uint32_t)__cvta_generic_to_shared(a_base);
                uint32_t b_saddr = (uint32_t)__cvta_generic_to_shared(b_base);

                // Build SMEM descriptors
                uint64_t desc_a = ((uint64_t)(a_saddr & 0x3FFFFu) >> 4) |
                                  ((uint64_t)((lbo & 0x3FFFFu) >> 4) << 16) |
                                  ((uint64_t)((sbo & 0x3FFFFu) >> 4) << 32) |
                                  ver_swz;

                uint64_t desc_b = ((uint64_t)(b_saddr & 0x3FFFFu) >> 4) |
                                  ((uint64_t)((lbo & 0x3FFFFu) >> 4) << 16) |
                                  ((uint64_t)((sbo & 0x3FFFFu) >> 4) << 32) |
                                  ver_swz;

                // enable-input-d predicate: 0=c-clear accumulator, non-zero=accumulate
                uint32_t accum_val = (first_k_iter && sub == 0) ? 0u : 1u;

                asm volatile(
                    "{\n\t"
                    ".reg .pred p;\n\t"
                    "setp.ne.s32 p, %4, 0;\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t"
                    "}\n\t"
                    : : "r"(c_addr), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum_val));
            }
        }

        // Synchronize to ensure UMMA completes before next tile
        __syncthreads();
    }

    // ---- Epilogue: FP32 accumulator -> BF16, write to global memory ----
    __syncthreads();

    if (m_tile_start + local_m < M) {
        for (int th = 0; th < THREADS_PER_ROW; th++) {
            int ln = local_n_base + th;
            if (n_tile_start + ln < N) {
                int g_m = m_tile_start + local_m;
                int g_n = n_tile_start + ln;
                C[(size_t)g_m * N + g_n] = __float2bfloat16(Cs[local_m * BLOCK_N + ln]);
            }
        }
    }
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));

    int64_t M = A.size(0);
    constexpr int64_t N = 7168;
    constexpr int64_t K = 5120;

    constexpr int BLOCK_M = 64;
    constexpr int BLOCK_N = 128;
    constexpr int BLOCK_K = 16;
    constexpr int THREADS_PER_ROW = 2;
    constexpr int TOTAL_THREADS = BLOCK_M * (BLOCK_N / THREADS_PER_ROW);  // 64*64=4096

    dim3 block(TOTAL_THREADS);
    dim3 grid((M + BLOCK_M - 1) / BLOCK_M, (int64_t)(N + BLOCK_N - 1) / BLOCK_N);

    // Shared memory: As + Bs + Cs (fp32 accumulator)
    size_t smem_size = BLOCK_M * BLOCK_K * sizeof(__nv_bfloat16) +
                       BLOCK_N * BLOCK_K * sizeof(__nv_bfloat16) +
                       BLOCK_M * BLOCK_N * sizeof(float);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

    gemm_kernel<BLOCK_M, BLOCK_N, BLOCK_K><<<grid, block, smem_size, stream>>>(
        static_cast<const __nv_bfloat16*>(A.data_ptr()),
        static_cast<const __nv_bfloat16*>(B.data_ptr()),
        static_cast<__nv_bfloat16*>(C.data_ptr()),
        static_cast<int>(M), static_cast<int>(N), static_cast<int>(K)
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_blackwell::run);

} // namespace gemm_blackwell