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

__device__ __forceinline__ float bf16_to_f32(const __nv_bfloat16 x) {
  return __bfloat162float(x);
}

// Double-buffered, vectorized shared-memory GEMM.
// C[M, N] = A[M, K] @ B[N, K]^T
//
// Tile choice tries to balance:
// - lower register pressure than 8x8/thread
// - more reuse than tiny tiles
// - enough threads to saturate SMs
//
// Block tile: 128 x 128
// K tile:     32
// Threads:    16 x 16 = 256
// Per-thread: 8 x 8 outputs
//
// This version pipelines global->shared traffic with ping-pong shared tiles.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void GemmBF16Kernel(const __nv_bfloat16* __restrict__ A,
                               const __nv_bfloat16* __restrict__ B,
                               __nv_bfloat16* __restrict__ C,
                               int M) {
  constexpr int THREADS_X = BN / TN;
  constexpr int THREADS_Y = BM / TM;
  constexpr int NUM_THREADS = THREADS_X * THREADS_Y;

  __shared__ __align__(16) __nv_bfloat16 As[2][BM][BK];
  __shared__ __align__(16) __nv_bfloat16 Bs[2][BN][BK];

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

  auto load_tiles = [&](int stage, int k0) {
    // Vectorized bf16x2 loads for A
    for (int idx2 = tid; idx2 < (BM * BK) / 2; idx2 += NUM_THREADS) {
      const int elem = idx2 * 2;
      const int r = elem / BK;
      const int k = elem - r * BK;
      const int gr = block_m + r;
      uint32_t v = 0;
      if (gr < M) {
        v = reinterpret_cast<const uint32_t*>(
            &A[(int64_t)gr * K_CONST + (k0 + k)])[0];
      }
      reinterpret_cast<uint32_t*>(&As[stage][r][k])[0] = v;
    }

    // Vectorized bf16x2 loads for B
    for (int idx2 = tid; idx2 < (BN * BK) / 2; idx2 += NUM_THREADS) {
      const int elem = idx2 * 2;
      const int n = elem / BK;
      const int k = elem - n * BK;
      const int gn = block_n + n;
      uint32_t v = 0;
      if (gn < N_CONST) {
        v = reinterpret_cast<const uint32_t*>(
            &B[(int64_t)gn * K_CONST + (k0 + k)])[0];
      }
      reinterpret_cast<uint32_t*>(&Bs[stage][n][k])[0] = v;
    }
  };

  int stage = 0;
  load_tiles(stage, 0);
  __syncthreads();

  for (int k0 = 0; k0 < K_CONST; k0 += BK) {
    const int next_k0 = k0 + BK;
    const int next_stage = stage ^ 1;

    if (next_k0 < K_CONST) {
      load_tiles(next_stage, next_k0);
    }

#pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
      float a_frag[TM];
      float b_frag[TN];

#pragma unroll
      for (int i = 0; i < TM; ++i) {
        a_frag[i] = bf16_to_f32(As[stage][ty * TM + i][kk]);
      }

#pragma unroll
      for (int j = 0; j < TN; ++j) {
        b_frag[j] = bf16_to_f32(Bs[stage][tx * TN + j][kk]);
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
    stage = next_stage;
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

// Lower-register variant for small M, often better occupancy.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void GemmBF16KernelSmall(const __nv_bfloat16* __restrict__ A,
                                    const __nv_bfloat16* __restrict__ B,
                                    __nv_bfloat16* __restrict__ C,
                                    int M) {
  constexpr int THREADS_X = BN / TN;
  constexpr int THREADS_Y = BM / TM;
  constexpr int NUM_THREADS = THREADS_X * THREADS_Y;

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
    for (int idx2 = tid; idx2 < (BM * BK) / 2; idx2 += NUM_THREADS) {
      const int elem = idx2 * 2;
      const int r = elem / BK;
      const int k = elem - r * BK;
      const int gr = block_m + r;
      uint32_t v = 0;
      if (gr < M) {
        v = reinterpret_cast<const uint32_t*>(
            &A[(int64_t)gr * K_CONST + (k0 + k)])[0];
      }
      reinterpret_cast<uint32_t*>(&As[r][k])[0] = v;
    }

    for (int idx2 = tid; idx2 < (BN * BK) / 2; idx2 += NUM_THREADS) {
      const int elem = idx2 * 2;
      const int n = elem / BK;
      const int k = elem - n * BK;
      const int gn = block_n + n;
      uint32_t v = 0;
      if (gn < N_CONST) {
        v = reinterpret_cast<const uint32_t*>(
            &B[(int64_t)gn * K_CONST + (k0 + k)])[0];
      }
      reinterpret_cast<uint32_t*>(&Bs[n][k])[0] = v;
    }

    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
      float a_frag[TM];
      float b_frag[TN];

#pragma unroll
      for (int i = 0; i < TM; ++i) {
        a_frag[i] = bf16_to_f32(As[ty * TM + i][kk]);
      }

#pragma unroll
      for (int j = 0; j < TN; ++j) {
        b_frag[j] = bf16_to_f32(Bs[tx * TN + j][kk]);
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

  const int M = static_cast<int>(A.size(0));
  const __nv_bfloat16* A_ptr =
      static_cast<const __nv_bfloat16*>(A.data_ptr());
  const __nv_bfloat16* B_ptr =
      static_cast<const __nv_bfloat16*>(B.data_ptr());
  __nv_bfloat16* C_ptr =
      static_cast<__nv_bfloat16*>(C.data_ptr());

  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

  // Heuristic dispatch.
  // Large-M case: larger tile + double buffering.
  // Small-M case: smaller register footprint.
  if (M >= 2048) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 32;
    constexpr int TM = 8;
    constexpr int TN = 8;
    dim3 block(BN / TN, BM / TM);  // 16 x 16 = 256 threads
    dim3 grid((N_CONST + BN - 1) / BN, (M + BM - 1) / BM);
    GemmBF16Kernel<BM, BN, BK, TM, TN><<<grid, block, 0, stream>>>(
        A_ptr, B_ptr, C_ptr, M);
  } else {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 32;
    constexpr int TM = 4;
    constexpr int TN = 4;
    dim3 block(BN / TN, BM / TM);  // 16 x 16 = 256 threads
    dim3 grid((N_CONST + BN - 1) / BN, (M + BM - 1) / BM);
    GemmBF16KernelSmall<BM, BN, BK, TM, TN><<<grid, block, 0, stream>>>(
        A_ptr, B_ptr, C_ptr, M);
  }

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_n7168_k5120_cuda::run);

}  // namespace gemm_n7168_k5120_cuda