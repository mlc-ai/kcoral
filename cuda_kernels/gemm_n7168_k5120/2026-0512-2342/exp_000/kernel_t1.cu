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

// 2D block tiled GEMM specialized for:
//   C[M, 7168] = A[M, 5120] @ B[7168, 5120]^T
//
// Mapping:
// - block tile: BM x BN
// - each thread computes TM rows x TN cols
//
// B is stored as [N, K] row-major, so B[n, k] is contiguous in K.
// Since we compute A @ B^T, output element is:
//   C[m, n] = sum_k A[m, k] * B[n, k]
template <int BM, int BN, int BK, int TM, int TN>
__global__ void GemmBF16Kernel(const __nv_bfloat16* __restrict__ A,
                               const __nv_bfloat16* __restrict__ B,
                               __nv_bfloat16* __restrict__ C,
                               int M) {
  constexpr int THREADS_X = BN / TN;
  constexpr int THREADS_Y = BM / TM;
  constexpr int NUM_THREADS = THREADS_X * THREADS_Y;

  __shared__ __nv_bfloat16 As[BM][BK];
  __shared__ __nv_bfloat16 Bs[BN][BK];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;

  const int block_m = blockIdx.y * BM;
  const int block_n = blockIdx.x * BN;

  const int row0 = block_m + ty * TM;
  const int col0 = block_n + tx * TN;

  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i) {
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      acc[i][j] = 0.0f;
    }
  }

  for (int k0 = 0; k0 < K_CONST; k0 += BK) {
    // Cooperative load A tile [BM, BK]
    for (int idx = tid; idx < BM * BK; idx += NUM_THREADS) {
      const int r = idx / BK;
      const int k = idx - r * BK;
      const int gr = block_m + r;
      As[r][k] = (gr < M) ? A[(int64_t)gr * K_CONST + (k0 + k)]
                          : __float2bfloat16(0.0f);
    }

    // Cooperative load B tile [BN, BK]
    for (int idx = tid; idx < BN * BK; idx += NUM_THREADS) {
      const int n = idx / BK;
      const int k = idx - n * BK;
      const int gn = block_n + n;
      Bs[n][k] = (gn < N_CONST) ? B[(int64_t)gn * K_CONST + (k0 + k)]
                                : __float2bfloat16(0.0f);
    }

    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
      float a_frag[TM];
      float b_frag[TN];

#pragma unroll
      for (int i = 0; i < TM; ++i) {
        a_frag[i] = __bfloat162float(As[ty * TM + i][kk]);
      }

#pragma unroll
      for (int j = 0; j < TN; ++j) {
        b_frag[j] = __bfloat162float(Bs[tx * TN + j][kk]);
      }

#pragma unroll
      for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          acc[i][j] += a_frag[i] * b_frag[j];
        }
      }
    }

    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int row = row0 + i;
    if (row < M) {
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        const int col = col0 + j;
        if (col < N_CONST) {
          C[(int64_t)row * N_CONST + col] = __float2bfloat16(acc[i][j]);
        }
      }
    }
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

  // 256 threads/block, 64x64 output tile, 32-depth K tile.
  // Each thread computes 4x4 outputs.
  constexpr int BM = 64;
  constexpr int BN = 64;
  constexpr int BK = 32;
  constexpr int TM = 4;
  constexpr int TN = 4;

  static_assert(BM % TM == 0, "BM must be divisible by TM");
  static_assert(BN % TN == 0, "BN must be divisible by TN");

  dim3 block(BN / TN, BM / TM);
  dim3 grid((N_CONST + BN - 1) / BN, (static_cast<int>(M) + BM - 1) / BM);

  GemmBF16Kernel<BM, BN, BK, TM, TN><<<grid, block, 0, stream>>>(
      A_ptr, B_ptr, C_ptr, static_cast<int>(M));

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_n7168_k5120_cuda::run);

}  // namespace gemm_n7168_k5120_cuda