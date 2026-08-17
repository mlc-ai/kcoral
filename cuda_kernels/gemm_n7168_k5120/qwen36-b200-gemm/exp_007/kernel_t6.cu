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
constexpr int NUM_THREADS = 256;
constexpr int TILE_A = BLOCK_M * BLOCK_K;
constexpr int TILE_B = BLOCK_N * BLOCK_K;

__global__ void gemm_bf16_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K,
    int lda, int ldb, int ldc) 
{
    __shared__ __nv_bfloat16 sA[TILE_A];
    __shared__ __nv_bfloat16 sB[TILE_B];

    const int tid = threadIdx.x;
    const int bm = blockIdx.x * BLOCK_M;
    const int bn = blockIdx.y * BLOCK_N;
    const int nk = K / BLOCK_K;

    // 256 threads -> each owns 2 columns on a single row
    // thread idx: row = idx % 64, col = (idx / 64) * 2 + offset
    const int om = tid % BLOCK_M;        // row in block: 0..63
    const int oc = (tid / BLOCK_M) * 2;  // base column: 0,2,4,...,62

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    const int elems_per_thread_a = TILE_A / NUM_THREADS;   // 16
    const int elems_per_thread_b = TILE_B / NUM_THREADS;   // 16

    for (int stage = 0; stage < nk; ++stage) {
        const int ko = stage * BLOCK_K;

        // === LOAD A tile cooperatively ===
        #pragma unroll 4
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

        // === LOAD B tile cooperatively ===
        #pragma unroll 4
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

        // === COMPUTE: dot product for our row against two columns ===
        const __nv_bfloat16* pA_row = &sA[om * BLOCK_K];
        const __nv_bfloat16* pB_c0  = &sB[oc * BLOCK_K];
        const __nv_bfloat16* pB_c1  = &sB[(oc + 1) * BLOCK_K];

        #pragma unroll
        for (int k = 0; k < BLOCK_K; k += 2) {
            float a0 = __bfloat162float(pA_row[k]);
            float a1 = __bfloat162float(pA_row[k + 1]);
            float b00 = __bfloat162float(pB_c0[k]);
            float b01 = __bfloat162float(pB_c0[k + 1]);
            float b10 = __bfloat162float(pB_c1[k]);
            float b11 = __bfloat162float(pB_c1[k + 1]);
            acc0 += a0 * b00 + a1 * b01;
            acc1 += a0 * b10 + a1 * b11;
        }

        __syncthreads();
    }

    // === STORE outputs ===
    const int gr = bm + om;
    const int gc0 = bn + oc;
    const int gc1 = bn + oc + 1;

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
    
    gemm_bf16_kernel<<<grd, blk, 0, stream>>>(
        A_d, B_d, C_d, 
        static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
        static_cast<int>(K), static_cast<int>(K), static_cast<int>(N));
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda