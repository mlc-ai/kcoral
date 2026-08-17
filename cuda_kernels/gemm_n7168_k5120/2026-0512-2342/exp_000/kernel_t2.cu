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

// Vectorized tiled GEMM.
// A: [M, K]
// B: [N, K]
// C: [M, N]
// Computes C = A @ B^T
//
// Design goals for speed vs previous versions:
// - 128x128x16 tile
// - 256 threads/block
// - each thread computes 8x8 outputs
// - shared-memory tiled reuse of A and B
// - vectorized bf16x2 global/shared loads
//
// This remains a conventional CUDA-core kernel for portability/correctness.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void GemmBF16Kernel(const __nv_bfloat16* __restrict__ A,
                               const __nv_bfloat16* __restrict__ B,
                               __nv_bfloat16* __restrict__ C,
                               int M) {
  constexpr int THREADS_X = BN / TN;
  constexpr int THREADS_Y = BM / TM;
  constexpr int NUM_THREADS = THREADS_X * THREADS_Y;
  constexpr int VEC = 2;  // bf16x2 = 4 bytes

  __shared__ __align__(16) __nv_bfloat16 As[BM][BK];
  __shared__ __align__(16) __nv_bfloat16 Bs[BN][BK];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;

  const int block_m = blockIdx.y * BM;
  const int block_n = blockIdx.x * BN;

  const int row_base = block_m + ty * TM;
  const int col_base = block_n + tx * TN;

  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i) {
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      acc[i][j] = 0.0f;
    }
  }

  for (int k0 = 0; k0 < K_CONST; k0 += BK) {
    // Load A tile: BM x BK elements
    for (int idx2 = tid; idx2 < (BM * BK) / VEC; idx2 += NUM_THREADS) {
      const int elem = idx2 * VEC;
      const int r = elem / BK;
      const int k = elem - r * BK;
      const int gr = block_m + r;

      if (gr < M) {
        reinterpret_cast<uint32_t*>(&As[r][k])[0] =
            reinterpret_cast<const uint32_t*>(
                &A[(int64_t)gr * K_CONST + (k0 + k)])[0];
      } else {
        reinterpret_cast<uint32_t*>(&As[r][k])[0] = 0u;
      }
    }

    // Load B tile: BN x BK elements
    for (int idx2 = tid; idx2 < (BN * BK) / VEC; idx2 += NUM_THREADS) {
      const int elem = idx2 * VEC;
      const int n = elem / BK;
      const int k = elem - n * BK;
      const int gn = block_n + n;

      if (gn < N_CONST) {
        reinterpret_cast<uint32_t*>(&Bs[n][k])[0] =
            reinterpret_cast<const uint32_t*>(
                &B[(int64_t)gn * K_CONST + (k0 + k)])[0];
      } else {
        reinterpret_cast<uint32_t*>(&Bs[n][k])[0] = 0u;
      }
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
    const int row = row_base + i;
    if (row < M) {
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        const int col = col_base + j;
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

  // 256-thread block: 16x16 threads
  // Tile: 128x128, each thread computes 8x8 outputs
  // BK=16 to keep shared memory modest and improve occupancy.
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 16;
  constexpr int TM = 8;
  constexpr int TN = 8;

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