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

// Warp-specialized 64x64x32 tile.
// 256 threads/block = 8 warps.
// Each warp computes a 32x16 subtile using a 4x4 microtile per thread.
//
// Compared with the previous version, this reduces register pressure a lot,
// improves occupancy, and keeps shared-memory reuse effective.
template <int BK>
__global__ void GemmBF16Kernel64x64(const __nv_bfloat16* __restrict__ A,
                                    const __nv_bfloat16* __restrict__ B,
                                    __nv_bfloat16* __restrict__ C,
                                    int M) {
  constexpr int BM = 64;
  constexpr int BN = 64;
  constexpr int WARP_M = 32;
  constexpr int WARP_N = 16;
  constexpr int TM = 4;
  constexpr int TN = 4;

  __shared__ __align__(16) __nv_bfloat16 As[BM][BK];
  __shared__ __align__(16) __nv_bfloat16 Bs[BN][BK];

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;

  const int block_m = blockIdx.y * BM;
  const int block_n = blockIdx.x * BN;

  // 8 warps arranged as 2 x 4 over the 64x64 tile:
  // warp_m_idx in [0,1], warp_n_idx in [0,3]
  const int warp_m_idx = warp_id >> 2;
  const int warp_n_idx = warp_id & 3;

  // Inside one warp (32 threads), arrange as 8 x 4 threads.
  const int lane_row = lane >> 2;   // 0..7
  const int lane_col = lane & 3;    // 0..3

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

  // Cooperative global->shared loads.
  // 256 threads load both A and B tiles using bf16x2 vectorized moves.
  for (int k0 = 0; k0 < K_CONST; k0 += BK) {
    // A tile: 64 x BK
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

    // B tile: 64 x BK
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

  // 256 threads/block, moderate register pressure, moderate shared memory.
  constexpr int BK = 32;
  dim3 block(256);
  dim3 grid((N_CONST + 63) / 64, (static_cast<int>(M) + 63) / 64);

  GemmBF16Kernel64x64<BK><<<grid, block, 0, stream>>>(
      A_ptr, B_ptr, C_ptr, static_cast<int>(M));

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_n7168_k5120_cuda::run);

}  // namespace gemm_n7168_k5120_cuda