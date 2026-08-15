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
    // Use 256 threads per block (well under 1024 max for SM100)
    // Phase 1 - Load A: threads 0..BLOCK_M-1
    // Phase 1 - Load B: threads BLOCK_M..BLOCK_M+(BLOCK_N*BLOCK_K/THREAD_LOAD_B)-1
    // Phase 2 - UMMA:  thread 0 only
    // Phase 3 - Store: all threads
    
    constexpr int NUM_THREADS = 256;
    constexpr int LOAD_B_ELEM_PER_THREAD = BLOCK_N * BLOCK_K / (NUM_THREADS - BLOCK_M); // 2048/192 ≈ not even
    // Actually use: threads 0..63 load A, threads 64..191 load B
    // 192 threads for B -> 2048/192 = 10.67 elements each - not clean
    
    // Better: threads 0..63 load A (each 2 cols), threads 64..191 load B (each ~10.67)
    // Even cleaner: split B differently. Let's use:
    //   Threads 0..BLOCK_M-1  : load A  (each handles 1 row of BM x BK)
    //   Threads 64..255       : load B  (192 threads, each handles BLOCK_N/192*N_cols...)
    
    // Simplest correct scheme:
    //   Threads 0..63:  load A[m][k] for their row m
    //   Threads 64..191: load B for 128 distinct n-values, each handles BLOCK_K elems
    //                    128 threads => need 128*16=2048 from 192 threads => ~10.67 each - messy
    //   Let's use threads 64..191 with strided access
    
    int tid = threadIdx.x;
    
    // Shared memory: As[BLOCK_M][BLOCK_K] + Bs[BLOCK_N][BLOCK_K] + Cs[BLOCK_M][BLOCK_N](fp32)
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
    uint32_t idesc = (1u << 4) | (1u << 7) | (1u << 10) |
                     ((BLOCK_N >> 3) & 0x3Fu) << 17 |
                     ((BLOCK_M >> 4) & 0x1Fu) << 24;

    // Descriptor layout for K-major 128B swizzle
    uint32_t sbo = 8u * BLOCK_K * sizeof(__nv_bfloat16);
    uint32_t lbo = 1u;
    uint64_t ver_swz = (1ULL << 46) | (2ULL << 61);

    uint32_t c_addr = (uint32_t)__cvta_generic_to_shared(Cs);

    int num_k_tiles = K / BLOCK_K;

    for (int kt = 0; kt < num_k_tiles; kt++) {
        int k_global_base = kt * BLOCK_K;

        // ---- Load A tile [BLOCK_M][BLOCK_K] ----
        // Threads 0..BLOCK_M-1 each load their row
        if (tid < BLOCK_M) {
            int m = tid;
            if (m_tile_start + m < M) {
                for (int ko = 0; ko < BLOCK_K; ko += 4) {
                    int kg = k_global_base + ko;
                    As[m * BLOCK_K + ko]     = A[(size_t)(m_tile_start + m) * K + kg];
                    As[m * BLOCK_K + ko + 1] = A[(size_t)(m_tile_start + m) * K + kg + 1];
                    As[m * BLOCK_K + ko + 2] = A[(size_t)(m_tile_start + m) * K + kg + 2];
                    As[m * BLOCK_K + ko + 3] = A[(size_t)(m_tile_start + m) * K + kg + 3];
                }
            }
        }

        // ---- Load B tile [BLOCK_N][BLOCK_K] ----
        // Threads BLOCK_M..NUM_THREADS-1 load B with striped assignment
        // Total B elements = BLOCK_N * BLOCK_K = 128*16 = 2048
        // Available threads = NUM_THREADS - BLOCK_M = 192
        // Each thread loads ceil(2048/192) ~= 11 elements via strided access
        if (tid >= BLOCK_M) {
            int b_tid = tid - BLOCK_M;
            int total_b_elems = BLOCK_N * BLOCK_K;
            int num_b_threads = NUM_THREADS - BLOCK_M;
            
            #pragma unroll
            for (int ei = b_tid; ei < total_b_elems; ei += num_b_threads) {
                int bn = ei / BLOCK_K;
                int bk = ei % BLOCK_K;
                if (n_tile_start + bn < N) {
                    Bs[bn * BLOCK_K + bk] = B[(size_t)(n_tile_start + bn) * K + k_global_base + bk];
                }
            }
        }

        __syncthreads();

        // ---- Issue UMMA for this K-tile ----
        bool first_k_iter = (kt == 0);

        #pragma unroll
        for (int sub = 0; sub < BLOCK_K / 16; sub++) {
            int k_off = sub * 16;

            if (tid == 0) {
                __nv_bfloat16* a_base = As + k_off * BLOCK_M;
                __nv_bfloat16* b_base = Bs + k_off * BLOCK_N;

                uint32_t a_saddr = (uint32_t)__cvta_generic_to_shared(a_base);
                uint32_t b_saddr = (uint32_t)__cvta_generic_to_shared(b_base);

                uint64_t desc_a = ((uint64_t)(a_saddr & 0x3FFFFu) >> 4) |
                                  ((uint64_t)((lbo & 0x3FFFFu) >> 4) << 16) |
                                  ((uint64_t)((sbo & 0x3FFFFu) >> 4) << 32) |
                                  ver_swz;

                uint64_t desc_b = ((uint64_t)(b_saddr & 0x3FFFFu) >> 4) |
                                  ((uint64_t)((lbo & 0x3FFFFu) >> 4) << 16) |
                                  ((uint64_t)((sbo & 0x3FFFFu) >> 4) << 32) |
                                  ver_swz;

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

        __syncthreads();
    }

    // ---- Epilogue: FP32 -> BF16, write to global ----
    __syncthreads();

    // All threads contribute to writing output
    // Total output elements per tile: BLOCK_M * BLOCK_N = 64*128 = 8192
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

    constexpr int BLOCK_M = 64;
    constexpr int BLOCK_N = 128;
    constexpr int BLOCK_K = 16;

    dim3 block(256);  // Under 1024 max
    dim3 grid((M + BLOCK_M - 1) / BLOCK_M, (int64_t)(N + BLOCK_N - 1) / BLOCK_N);

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