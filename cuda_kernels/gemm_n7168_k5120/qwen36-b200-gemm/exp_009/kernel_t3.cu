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
static constexpr int TM = 8;
static constexpr int TN = 8;
static constexpr int K_STRIDE = 16;

__global__ void gemm_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K)
{
    __shared__ __nv_bfloat16 shared_A[BLOCK_M][K_STRIDE];
    __shared__ __nv_bfloat16 shared_BT[BLOCK_N][K_STRIDE];

    int tx = threadIdx.x;  // 0..7
    int ty = threadIdx.y;  // 0..7

    int block_row = blockIdx.y * BLOCK_M;
    int block_col = blockIdx.x * BLOCK_N;

    float acc[TM][TN] = {};

    for (int kb = 0; kb < K; kb += K_STRIDE) {
        int ks = min(K_STRIDE, K - kb);

        // Load A tile into shared memory
        for (int i = ty; i < TM; i += blockDim.y) {
            int g_row = block_row + i;
            if (g_row < M) {
                const __nv_bfloat16* src = A + static_cast<int64_t>(g_row) * K + kb;
                for (int j = 0; j < K_STRIDE; ++j) {
                    shared_A[i][j] = (j < ks) ? src[j] : __float2bfloat16(0.0f);
                }
            } else {
                for (int j = 0; j < K_STRIDE; ++j) {
                    shared_A[i][j] = __float2bfloat16(0.0f);
                }
            }
        }

        // Load B tile into shared memory (B is stored [N][K], we want same layout for BT)
        for (int i = tx; i < TN; i += blockDim.x) {
            int g_col = block_col + i;
            if (g_col < N) {
                const __nv_bfloat16* src = B + static_cast<int64_t>(g_col) * K + kb;
                for (int j = 0; j < K_STRIDE; ++j) {
                    shared_BT[i][j] = (j < ks) ? src[j] : __float2bfloat16(0.0f);
                }
            } else {
                for (int j = 0; j < K_STRIDE; ++j) {
                    shared_BT[i][j] = __float2bfloat16(0.0f);
                }
            }
        }

        __syncthreads();

        // Compute sub-tile
        for (int ii = 0; ii < TM; ++ii) {
            for (int jj = 0; jj < TN; ++jj) {
                for (int kk = 0; kk < K_STRIDE; ++kk) {
                    float a = __bfloat162float(shared_A[ii][kk]);
                    float b = __bfloat162float(shared_BT[jj][kk]);
                    acc[ii][jj] += a * b;
                }
            }
        }

        __syncthreads();
    }

    // Write result back
    for (int ii = 0; ii < TM; ++ii) {
        for (int jj = 0; jj < TN; ++jj) {
            int g_row = block_row + ii;
            int g_col = block_col + jj;
            if (g_row < M && g_col < N) {
                C[static_cast<int64_t>(g_row) * N + g_col] = __float2bfloat16(acc[ii][jj]);
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

    dim3 block(TM, TN, 1);

    int grid_x = static_cast<int>((N + BLOCK_N - 1) / BLOCK_N);
    int grid_y = static_cast<int>((M + BLOCK_M - 1) / BLOCK_M);
    dim3 grid(grid_x, grid_y, 1);

    // Explicit shared memory size: 2 * 64 * 16 * 2 bytes = 4KB
    size_t smem_bytes = sizeof(__nv_bfloat16) * (BLOCK_M * K_STRIDE + BLOCK_N * K_STRIDE);

    cudaStream_t stream = nullptr;
    {
        auto dev = A.device();
        int dtype_code = TVMDLDevice2VDevice(dev).v_int64 & 0xFFFF;
        int device_id  = TVMDLDevice2VDevice(dev).v_int64 >> 32;
        stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(dtype_code, device_id));
    }

    gemm_kernel<<<grid, block, smem_bytes, stream>>>(
        A_ptr, B_ptr, C_ptr, 
        static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_gemm_bf16::run);

}  // namespace tvm_ffi_gemm_bf16