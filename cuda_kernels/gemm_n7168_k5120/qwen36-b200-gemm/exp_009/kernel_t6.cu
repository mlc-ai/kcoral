#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cstdio>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t e = (call);                                        \
    if (e != cudaSuccess) {                                        \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(e), __FILE__, __LINE__);        \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_gemm_bf16 {

static constexpr int BLOCK_M = 64;
static constexpr int BLOCK_N = 64;
static constexpr int K_STRIDE = 16;
// Each thread handles TILE_M rows and TILE_N cols within the block
static constexpr int TILE_M = 4;
static constexpr int TILE_N = 4;
// Threads per block = (BLOCK_M/TILE_M) * (BLOCK_N/TILE_N) = 16 * 16 = 256
static constexpr int THREADS_PER_BLOCK = 256;

__global__ void gemm_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K)
{
    __shared__ __nv_bfloat16 sA[BLOCK_M][K_STRIDE];
    __shared__ __nv_bfloat16 sB[BLOCK_N][K_STRIDE];

    int tid = threadIdx.x;
    
    // Map linear tid to (tile_y, tile_x) within block
    int ty_blocks = BLOCK_M / TILE_M;  // 16
    int tx_blocks = BLOCK_N / TILE_N;  // 16
    int tile_row_idx = tid / tx_blocks;  // 0..15
    int tile_col_idx = tid % tx_blocks;  // 0..15
    
    // This thread is responsible for output sub-tile starting at:
    int block_m_start = blockIdx.y * BLOCK_M + tile_row_idx * TILE_M;
    int block_n_start = blockIdx.x * BLOCK_N + tile_col_idx * TILE_N;

    // Allocate per-thread accumulators for TILE_M x TILE_N
    float acc[TILE_M * TILE_N] = {};

    for (int kb = 0; kb < K; kb += K_STRIDE) {
        int ks = min(K_STRIDE, K - kb);

        // Load A: each thread loads TILE_M rows x K_STRIDE cols
        // But actually, we want to load the entire BLOCK_M x K_STRIDE tile cooperatively
        // Each of 256 threads loads 1 element. Total needed = 64*16 = 1024. 
        // So each thread needs to load 4 elements.
        
        // Simpler approach: just have each thread load its portion
        for (int i = 0; i < TILE_M; ++i) {
            int global_row = block_m_start + i;
            if (global_row < M) {
                const __nv_bfloat16* src = A + static_cast<int64_t>(global_row) * K + kb;
                for (int j = 0; j < K_STRIDE; ++j) {
                    sA[global_row - blockIdx.y * BLOCK_M][j] = 
                        (j < ks) ? src[j] : __float2bfloat16(0.0f);
                }
            } else {
                for (int j = 0; j < K_STRIDE; ++j) {
                    sA[global_row - blockIdx.y * BLOCK_M][j] = __float2bfloat16(0.0f);
                }
            }
        }

        // Load B: similar
        for (int i = 0; i < TILE_N; ++i) {
            int global_col = block_n_start + i;
            if (global_col < N) {
                const __nv_bfloat16* src = B + static_cast<int64_t>(global_col) * K + kb;
                for (int j = 0; j < K_STRIDE; ++j) {
                    sB[global_col - blockIdx.x * BLOCK_N][j] = 
                        (j < ks) ? src[j] : __float2bfloat16(0.0f);
                }
            } else {
                for (int j = 0; j < K_STRIDE; ++j) {
                    sB[global_col - blockIdx.x * BLOCK_N][j] = __float2bfloat16(0.0f);
                }
            }
        }

        __syncthreads();

        // Compute
        for (int ii = 0; ii < TILE_M; ++ii) {
            for (int jj = 0; jj < TILE_N; ++jj) {
                int local_row = tile_row_idx * TILE_M + ii;
                int local_col = tile_col_idx * TILE_N + jj;
                for (int kk = 0; kk < K_STRIDE; ++kk) {
                    float a = __bfloat162float(sA[local_row][kk]);
                    float b = __bfloat162float(sB[local_col][kk]);
                    acc[ii * TILE_N + jj] += a * b;
                }
            }
        }

        __syncthreads();
    }

    // Store
    for (int ii = 0; ii < TILE_M; ++ii) {
        for (int jj = 0; jj < TILE_N; ++jj) {
            int g_row = block_m_start + ii;
            int g_col = block_n_start + jj;
            if (g_row < M && g_col < N) {
                C[static_cast<int64_t>(g_row) * N + g_col] = 
                    __float2bfloat16(acc[ii * TILE_N + jj]);
            }
        }
    }
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));

    int64_t M = A.size(0);
    int64_t N = 7168;
    int64_t K = 5120;

    const __nv_bfloat16* A_ptr = static_cast<const __nv_bfloat16*>(A.data_ptr());
    const __nv_bfloat16* B_ptr = static_cast<const __nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16*       C_ptr = static_cast<__nv_bfloat16*>(C.data_ptr());

    dim3 block(THREADS_PER_BLOCK, 1, 1);
    int grid_x = static_cast<int>((N + BLOCK_N - 1) / BLOCK_N);
    int grid_y = static_cast<int>((M + BLOCK_M - 1) / BLOCK_M);
    dim3 grid(grid_x, grid_y, 1);

    size_t smem_bytes = sizeof(__nv_bfloat16) * (BLOCK_M * K_STRIDE + BLOCK_N * K_STRIDE);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

    gemm_kernel<<<grid, block, smem_bytes, stream>>>(
        A_ptr, B_ptr, C_ptr, 
        static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_gemm_bf16::run);

}  // namespace tvm_ffi_gemm_bf16