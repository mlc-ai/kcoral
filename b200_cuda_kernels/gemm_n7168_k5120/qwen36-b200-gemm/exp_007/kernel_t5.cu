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
constexpr int OUTPUTS_PER_THREAD = (BLOCK_M * BLOCK_N) / NUM_THREADS;

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

    // Each thread owns 32 outputs spread across rows
    // Map: thread tid computes C[row][col] for 32 unique (row,col) pairs
    const int elem_per_thread = OUTPUTS_PER_THREAD;

    // We store 32 fp32 accumulators - but to reduce registers, use shared mem as temp
    // Actually, let's just keep it simple with reduced per-thread work
    // Each thread computes only 2 outputs, but we iterate over 2 groups
    
    // Better scheme: 128 threads, each computes 1 row x 2 cols portion
    // Thread idx: row = idx % 64, col_pair = idx / 64
    // This gives 128 threads -> 128*2 = 256 outputs... not enough for 64x64=4096
    
    // Use 256 threads instead for full coverage
    // Fall back to ORIGINAL simple scheme but with better register management
    // 128 threads, each computes 32 outputs, but STORE intermediates to smem
    
    float local_acc[OUTPUTS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < OUTPUTS_PER_THREAD; ++i) {
        local_acc[i] = 0.0f;
    }

    for (int stage = 0; stage < nk; ++stage) {
        const int ko = stage * BLOCK_K;

        // === LOAD A tile cooperatively ===
        const int elems_per_thread_a = TILE_A / NUM_THREADS;

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
        const int elems_per_thread_b = TILE_B / NUM_THREADS;

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

        // === COMPUTE: accumulate for each of this thread's 32 outputs ===
        #pragma unroll
        for (int oi = 0; oi < OUTPUTS_PER_THREAD; ++oi) {
            const int flat_out = tid * OUTPUTS_PER_THREAD + oi;
            const int om = flat_out / BLOCK_N;
            const int on = flat_out % BLOCK_N;

            const __nv_bfloat16* pA_row = &sA[om * BLOCK_K];
            const __nv_bfloat16* pB_row = &sB[on * BLOCK_K];

            float tmp = 0.0f;
            #pragma unroll
            for (int k = 0; k < BLOCK_K; k += 2) {
                tmp += __bfloat162float(pA_row[k])     * __bfloat162float(pB_row[k]);
                tmp += __bfloat162float(pA_row[k + 1]) * __bfloat162float(pB_row[k + 1]);
            }
            local_acc[oi] += tmp;
        }

        __syncthreads();
    }

    // === STORE outputs ===
    #pragma unroll
    for (int oi = 0; oi < OUTPUTS_PER_THREAD; ++oi) {
        const int flat_out = tid * OUTPUTS_PER_THREAD + oi;
        const int om = flat_out / BLOCK_N;
        const int on = flat_out % BLOCK_N;

        const int gr = bm + om;
        const int gc = bn + on;

        if (gr < M && gc < N) {
            C[gr * ldc + gc] = __float2bfloat16(local_acc[oi]);
        }
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