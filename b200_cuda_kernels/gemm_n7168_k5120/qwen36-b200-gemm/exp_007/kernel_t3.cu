#include <cuda_bf16.h>
#include <cuda_fp16.h>
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

namespace tvm_ffi_example_cuda {

constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 64;
constexpr int NUM_THREADS = 128;
constexpr int TILE_A = BLOCK_M * BLOCK_K;
constexpr int TILE_B = BLOCK_N * BLOCK_K;

// Shared memory: single tile buffers + no pipeline (simple version first)
constexpr int SMEM_BYTES = (TILE_A + TILE_B) * sizeof(__nv_bfloat16);

__global__ void gemm_bf16_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K,
    int lda, int ldb, int ldc) 
{
    // Shared memory tile buffers
    __shared__ __nv_bfloat16 sA[TILE_A];
    __shared__ __nv_bfloat16 sB[TILE_B];

    const int tid = threadIdx.x;
    const int bm = blockIdx.x * BLOCK_M;
    const int bn = blockIdx.y * BLOCK_N;

    const int nk = K / BLOCK_K;

    // Each thread owns outputs: my_row x [my_col0, my_col1]
    const int my_row = tid % BLOCK_M;
    const int my_col0 = (tid / BLOCK_M) * 2;
    const int my_col1 = my_col0 + 1;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (int stage = 0; stage < nk; ++stage) {
        const int ko = stage * BLOCK_K;

        // === LOAD A tile into sA[brow * BLOCK_K + bcol] ===
        const int elems_per_thread_a = TILE_A / NUM_THREADS;

        #pragma unroll
        for (int e = 0; e < elems_per_thread_a; ++e) {
            const int glob_idx = tid * elems_per_thread_a + e;
            const int brow = glob_idx / BLOCK_K;
            const int bcol = glob_idx % BLOCK_K;

            const int gr = bm + brow;
            const int gc = ko + bcol;

            if (gr < M && gc < K) {
                sA[brow * BLOCK_K + bcol] = A[gr * lda + gc];
            } else {
                sA[brow * BLOCK_K + bcol] = __float2bfloat16(0.0f);
            }
        }

        // === LOAD B tile into sB[brow * BLOCK_K + bcol] ===
        const int elems_per_thread_b = TILE_B / NUM_THREADS;

        #pragma unroll
        for (int e = 0; e < elems_per_thread_b; ++e) {
            const int glob_idx = tid * elems_per_thread_b + e;
            const int brow = glob_idx / BLOCK_K;
            const int bcol = glob_idx % BLOCK_K;

            const int gr = bn + brow;
            const int gc = ko + bcol;

            if (gr < N && gc < K) {
                sB[brow * BLOCK_K + bcol] = B[gr * ldb + gc];
            } else {
                sB[brow * BLOCK_K + bcol] = __float2bfloat16(0.0f);
            }
        }

        __syncthreads();

        // === COMPUTE partial dot product over this K tile ===
        const __nv_bfloat16* pA = sA + my_row * BLOCK_K;
        const __nv_bfloat16* pB0 = sB + my_col0 * BLOCK_K;
        const __nv_bfloat16* pB1 = sB + my_col1 * BLOCK_K;

        #pragma unroll
        for (int k = 0; k < BLOCK_K; ++k) {
            float a_val = __bfloat162float(pA[k]);
            float b0_val = __bfloat162float(pB0[k]);
            float b1_val = __bfloat162float(pB1[k]);
            acc0 += a_val * b0_val;
            acc1 += a_val * b1_val;
        }

        __syncthreads();
    }

    // === STORE output ===
    const int gr = bm + my_row;
    const int gc0 = bn + my_col0;
    const int gc1 = bn + my_col1;

    if (gr < M && gc0 < N) {
        C[gr * ldc + gc0] = __float2bfloat16(acc0);
    }
    if (gr < M && gc1 < N) {
        C[gr * ldc + gc1] = __float2bfloat16(acc1);
    }
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(0);
    
    const __nv_bfloat16* A_d = static_cast<const __nv_bfloat16*>(A.data_ptr());
    const __nv_bfloat16* B_d = static_cast<const __nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* C_d = static_cast<__nv_bfloat16*>(C.data_ptr());
    
    dim3 blk(NUM_THREADS);
    dim3 grd((M + BLOCK_M - 1) / BLOCK_M, (N + BLOCK_N - 1) / BLOCK_N);
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
    
    // Shared mem is statically allocated inside the kernel
    gemm_bf16_kernel<<<grd, blk, 0, stream>>>(
        A_d, B_d, C_d, 
        static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
        static_cast<int>(K), static_cast<int>(K), static_cast<int>(N));
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda