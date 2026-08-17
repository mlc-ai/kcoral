#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t _e = (call);                                                  \
    if (_e != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e),     \
              __FILE__, __LINE__);                                            \
      std::abort();                                                           \
    }                                                                         \
  } while (0)

namespace gemm_n7168_k5120_cuda {

static constexpr int N_CONST = 7168;
static constexpr int K_CONST = 5120;

// Simple, correct tiled GEMM for C = A @ B^T
// A: [M, K], row-major bf16
// B: [N, K], row-major bf16
// C: [M, N], row-major bf16
//
// Each block computes a BM x BN tile of C.
// Accumulation is in FP32, output converted to BF16.
template <int BM, int BN, int BK>
__global__ void GemmBF16Kernel(const __nv_bfloat16* __restrict__ A,
                               const __nv_bfloat16* __restrict__ B,
                               __nv_bfloat16* __restrict__ C,
                               int M) {
  __shared__ __nv_bfloat16 As[BM][BK];
  __shared__ __nv_bfloat16 Bs[BN][BK];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  const int row = blockIdx.y * BM + ty;
  const int col = blockIdx.x * BN + tx;

  float acc = 0.0f;

  // Cooperative load is organized over BM*BN threads.
  const int tid = ty * blockDim.x + tx;
  const int num_threads = blockDim.x * blockDim.y;

  for (int k0 = 0; k0 < K_CONST; k0 += BK) {
    // Load A tile: [BM, BK]
    for (int idx = tid; idx < BM * BK; idx += num_threads) {
      int r = idx / BK;
      int k = idx % BK;
      int gr = blockIdx.y * BM + r;
      int gk = k0 + k;
      if (gr < M) {
        As[r][k] = A[(int64_t)gr * K_CONST + gk];
      } else {
        As[r][k] = __float2bfloat16(0.0f);
      }
    }

    // Load B tile: [BN, BK], since B is [N, K] row-major and we need B[col, k]
    for (int idx = tid; idx < BN * BK; idx += num_threads) {
      int c = idx / BK;
      int k = idx % BK;
      int gc = blockIdx.x * BN + c;
      int gk = k0 + k;
      if (gc < N_CONST) {
        Bs[c][k] = B[(int64_t)gc * K_CONST + gk];
      } else {
        Bs[c][k] = __float2bfloat16(0.0f);
      }
    }

    __syncthreads();

    if (row < M && col < N_CONST) {
#pragma unroll
      for (int k = 0; k < BK; ++k) {
        acc += __bfloat162float(As[ty][k]) * __bfloat162float(Bs[tx][k]);
      }
    }

    __syncthreads();
  }

  if (row < M && col < N_CONST) {
    C[(int64_t)row * N_CONST + col] = __float2bfloat16(acc);
  }
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
  CUDA_CHECK(cudaSetDevice(A.device().device_id));

  const int64_t M = A.size(0);

  const __nv_bfloat16* A_ptr =
      static_cast<const __nv_bfloat16*>(A.data_ptr());
  const __nv_bfloat16* B_ptr =
      static_cast<const __nv_bfloat16*>(B.data_ptr());
  __nv_bfloat16* C_ptr =
      static_cast<__nv_bfloat16*>(C.data_ptr());

  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

  constexpr int BM = 16;
  constexpr int BN = 16;
  constexpr int BK = 16;

  dim3 block(BN, BM);
  dim3 grid((N_CONST + BN - 1) / BN, (static_cast<int>(M) + BM - 1) / BM);

  GemmBF16Kernel<BM, BN, BK><<<grid, block, 0, stream>>>(
      A_ptr, B_ptr, C_ptr, static_cast<int>(M));

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_n7168_k5120_cuda::run);

}  // namespace gemm_n7168_k5120_cuda