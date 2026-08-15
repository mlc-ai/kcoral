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

// Simple correctness-first kernel: each thread does one C element
__global__ void gemm_naive_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int64_t M, int64_t N, int64_t K)
{
    int64_t row = blockIdx.y * blockDim.y + threadIdx.y;
    int64_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int64_t k = 0; k < K; ++k) {
            float a = __bfloat162float(A[row * K + k]);
            float b = __bfloat162float(B[col * K + k]);
            acc += a * b;
        }
        C[row * N + col] = __float2bfloat16(acc);
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

    dim3 block(32, 16, 1);  // 512 threads per block

    int grid_x = static_cast<int>((N + block.x - 1) / block.x);
    int grid_y = static_cast<int>((M + block.y - 1) / block.y);
    dim3 grid(grid_x, grid_y, 1);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

    gemm_naive_kernel<<<grid, block, 0, stream>>>(
        A_ptr, B_ptr, C_ptr, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_gemm_bf16::run);

}  // namespace tvm_ffi_gemm_bf16