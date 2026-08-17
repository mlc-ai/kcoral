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

__device__ __forceinline__ float bf16_to_f32(__nv_bfloat16 x) {
  return __bfloat162float(x);
}

// Corrected main kernel.
// 128x128 CTA, 256 threads.
// 2x4 warps over CTA tile.
// Each warp computes 64x32.
// Each thread computes 8x4.
// Mapping inside warp:
//   lane_row in [0,7], lane_col in [0,3]
//   row_base = warp_row*64 + lane_row*8
//   col_base = warp_col*32 + lane_col*4
template <int BK>
__global__ void GemmMainKernel(const __nv_bfloat16* __restrict__ A,
                               const __nv_bfloat16* __restrict__ B,
                               __nv_bfloat16* __restrict__ C,
                               int M) {
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int WARP_M = 64;
  constexpr int WARP_N = 32;
  constexpr int TM = 8;
  constexpr int TN = 4;

  __shared__ __align__(16) __nv_bfloat16 As[BM][BK];
  __shared__ __align__(16) __nv_bfloat16 Bs[BN][BK];

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;

  const int block_m = blockIdx.y * BM;
  const int block_n = blockIdx.x * BN;

  const int warp_m_idx = warp_id >> 2;  // 0..1
  const int warp_n_idx = warp_id & 3;   // 0..3

  const int lane_row = lane >> 2;       // 0..7
  const int lane_col = lane & 3;        // 0..3

  const int row_base = block_m + warp_m_idx * WARP_M + lane_row * TM;
  const int col_base = block_n + warp_n_idx * WARP_N + lane_col * TN;

  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i) {
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      acc[i][j] = 0.0f;
    }
  }

  for (int k0 = 0; k0 < K_CONST; k0 += BK) {
    for (int idx2 = tid; idx2 < (BM * BK) / 2; idx2 += 256) {
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

    for (int idx2 = tid; idx2 < (BN * BK) / 2; idx2 += 256) {
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
        a_frag[i] = bf16_to_f32(As[warp_m_idx * WARP_M + lane_row * TM + i][kk]);
      }
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        b_frag[j] = bf16_to_f32(Bs[warp_n_idx * WARP_N + lane_col * TN + j][kk]);
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

// Reliable smaller-tile kernel used as fallback / default.
// 64x64 CTA, 256 threads organized as 16x16, 4x4 outputs/thread.
template <int BK>
__global__ void GemmSmallKernel(const __nv_bfloat16* __restrict__ A,
                                const __nv_bfloat16* __restrict__ B,
                                __nv_bfloat16* __restrict__ C,
                                int M) {
  constexpr int BM = 64;
  constexpr int BN = 64;
  constexpr int TM = 4;
  constexpr int TN = 4;

  __shared__ __align__(16) __nv_bfloat16 As[BM][BK];
  __shared__ __align__(16) __nv_bfloat16 Bs[BN][BK];

  const int tid = threadIdx.x;
  const int tx = tid & 15;
  const int ty = tid >> 4;

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
    for (int idx2 = tid; idx2 < (BM * BK) / 2; idx2 += 256) {
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

    for (int idx2 = tid; idx2 < (BN * BK) / 2; idx2 += 256) {
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
      float a0 = bf16_to_f32(As[ty * TM + 0][kk]);
      float a1 = bf16_to_f32(As[ty * TM + 1][kk]);
      float a2 = bf16_to_f32(As[ty * TM + 2][kk]);
      float a3 = bf16_to_f32(As[ty * TM + 3][kk]);

      float b0 = bf16_to_f32(Bs[tx * TN + 0][kk]);
      float b1 = bf16_to_f32(Bs[tx * TN + 1][kk]);
      float b2 = bf16_to_f32(Bs[tx * TN + 2][kk]);
      float b3 = bf16_to_f32(Bs[tx * TN + 3][kk]);

      acc[0][0] += a0 * b0; acc[0][1] += a0 * b1; acc[0][2] += a0 * b2; acc[0][3] += a0 * b3;
      acc[1][0] += a1 * b0; acc[1][1] += a1 * b1; acc[1][2] += a1 * b2; acc[1][3] += a1 * b3;
      acc[2][0] += a2 * b0; acc[2][1] += a2 * b1; acc[2][2] += a2 * b2; acc[2][3] += a2 * b3;
      acc[3][0] += a3 * b0; acc[3][1] += a3 * b1; acc[3][2] += a3 * b2; acc[3][3] += a3 * b3;
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

  // Use the proven-correct 64x64 kernel as the default path.
  // Keep the 128x128 kernel available for future tuning but avoid risking correctness.
  if (false && M >= 4096) {
    constexpr int BK = 32;
    dim3 block(256);
    dim3 grid((N_CONST + 127) / 128, (M + 127) / 128);
    GemmMainKernel<BK><<<grid, block, 0, stream>>>(A_ptr, B_ptr, C_ptr, M);
  } else {
    constexpr int BK = 32;
    dim3 block(256);
    dim3 grid((N_CONST + 63) / 64, (M + 63) / 64);
    GemmSmallKernel<BK><<<grid, block, 0, stream>>>(A_ptr, B_ptr, C_ptr, M);
  }

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_n7168_k5120_cuda::run);

}  // namespace gemm_n7168_k5120_cuda