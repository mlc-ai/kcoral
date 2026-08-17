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
    constexpr int NUM_THREADS = 256;
    int tid = threadIdx.x;

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

    // Instruction descriptor: BF16*BF16->FP32, K-major for both A and B, NO SWIZZLE
    //   Bits 4-5: dtype=F32(1)
    //   Bits 7-9: a_format=BF16(1)
    //   Bits 10-12: b_format=BF16(1)
    //   Bit 15: TransposeA=0 => K-major
    //   Bit 16: TransposeB=0 => K-major
    //   Bits 17-22: N>>3
    //   Bits 24-28: M>>4
    uint32_t idesc = (1u << 4) | (1u << 7) | (1u << 10) |
                     ((BLOCK_N >> 3) & 0x3Fu) << 17 |
                     ((BLOCK_M >> 4) & 0x1Fu) << 24;

    // Descriptor layout for K-major NON-SWIZZLED:
    //   ATOM_MMODE_DIM = BLOCK_M, ATOM_KMODE_DIM = 16/sizeof(bf16) = 8
    //   SBO = 8 * 16 = 128
    //   LBO = (ATOM_MMODE_DIM / 8) * SBO = (64/8)*128 = 1024
    //   Swizzle bits = 0 (no swizzle)
    // NOTE: BLOCK_K MUST equal 8 for a single UMMA to consume the full tile.
    uint32_t sbo = 128u;
    uint32_t lbo = (uint32_t)((BLOCK_M >> 3) * 128u);
    uint64_t ver_swz = (1ULL << 46);  // version=1, swizzle=0

    uint32_t c_addr = (uint32_t)__cvta_generic_to_shared(Cs);

    int num_k_tiles = K / BLOCK_K;

    for (int kt = 0; kt < num_k_tiles; kt++) {
        int k_global_base = kt * BLOCK_K;

        // ---- Load A tile [BLOCK_M][BLOCK_K] ----
        if (tid < BLOCK_M) {
            int m = tid;
            if (m_tile_start + m < M) {
                const __nv_bfloat16* row_ptr = A + (size_t)(m_tile_start + m) * K + k_global_base;
                #pragma unroll
                for (int ko = 0; ko < BLOCK_K; ko += 4) {
                    As[m * BLOCK_K + ko]     = row_ptr[ko];
                    As[m * BLOCK_K + ko + 1] = row_ptr[ko + 1];
                    As[m * BLOCK_K + ko + 2] = row_ptr[ko + 2];
                    As[m * BLOCK_K + ko + 3] = row_ptr[ko + 3];
                }
            }
        }

        // ---- Load B tile [BLOCK_N][BLOCK_K] ----
        // Threads BLOCK_M..NUM_THREADS-1 do strided access over BLOCK_N*BLOCK_K elements
        if (tid >= BLOCK_M) {
            int b_tid = tid - BLOCK_M;
            int num_b_threads = NUM_THREADS - BLOCK_M;
            int total_b_elems = BLOCK_N * BLOCK_K;
            
            #pragma unroll
            for (int ei = b_tid; ei < total_b_elems; ei += num_b_threads) {
                int bn = ei / BLOCK_K;
                int bk = ei % BLOCK_K;
                if (n_tile_start + bn < N) {
                    Bs[bn * BLOCK_K + bk] = 
                        B[(size_t)(n_tile_start + bn) * K + k_global_base + bk];
                }
            }
        }

        __syncthreads();

        // ---- Issue UMMA for this K-tile ----
        if (tid == 0) {
            // Base pointers for A and B tiles
            __nv_bfloat16* a_base = As;
            __nv_bfloat16* b_base = Bs;

            uint32_t a_saddr = (uint32_t)__cvta_generic_to_shared(a_base);
            uint32_t b_saddr = (uint32_t)__cvta_generic_to_shared(b_base);

            // Build SMEM descriptors (K-major, no swizzle)
            uint64_t desc_a = ((uint64_t)(a_saddr & 0x3FFFFu) >> 4) |
                              ((uint64_t)((lbo & 0x3FFFFu) >> 4) << 16) |
                              ((uint64_t)((sbo & 0x3FFFFu) >> 4) << 32) |
                              ver_swz;

            uint64_t desc_b = ((uint64_t)(b_saddr & 0x3FFFFu) >> 4) |
                              ((uint64_t)((lbo & 0x3FFFFu) >> 4) << 16) |
                              ((uint64_t)((sbo & 0x3FFFFu) >> 4) << 32) |
                              ver_swz;

            // enable-input-d predicate: 0=clear accumulator, non-zero=accumulate
            uint32_t accum_val = (kt == 0) ? 0u : 1u;

            asm volatile(
                "{\n\t"
                ".reg .pred p;\n\t"
                "setp.ne.s32 p, %4, 0;\n\t"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t"
                "}\n\t"
                : : "r"(c_addr), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum_val));
        }

        __syncthreads();
    }

    // ---- Epilogue: FP32 -> BF16, write to global ----
    __syncthreads();

    int total_out = BLOCK_M * BLOCK_N;
    #pragma unroll
    for (int idx = tid; idx < total_out; idx += NUM_THREADS) {
        int local_m = idx / BLOCK_N;
        int local_n = idx % BLOCK_N;
        int g_m = m_tile_start + local_m;
        int g_n = n_tile_start + local_n;
        if (g_m < M && g_n < N) {
            C[(size_t)g_m * N + g_n] = __float2bfloat16(Cs[idx]);
        }
    }
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));

    int64_t M = A.size(0);
    constexpr int64_t N = 7168;
    constexpr int64_t K = 5120;

    // BLOCK_K must equal 8 to match the UMMA atom size for K-major no-swizzle layout
    constexpr int BLOCK_M = 64;
    constexpr int BLOCK_N = 128;
    constexpr int BLOCK_K = 8;

    dim3 block(256);
    dim3 grid((M + BLOCK_M - 1) / BLOCK_M, (int64_t)(N + BLOCK_N - 1) / BLOCK_N);

    // Shared memory: As[64][8] + Bs[128][8] + Cs[64][128](fp32)
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