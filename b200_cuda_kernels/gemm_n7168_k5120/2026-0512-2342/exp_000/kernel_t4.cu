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

// Blackwell path: use cuBLAS through dynamically-resolved symbols.
// This keeps the source self-contained (no external headers beyond CUDA runtime)
// while enabling hardware-optimized GEMM performance.
//
// Computes:
//   C[M, N] = A[M, K] @ B[N, K]^T
// with A/B/C all BF16, accumulation in FP32, output BF16.
//
// We map to cuBLAS as:
//   C^T = B @ A^T
// where:
//   B is [N, K], row-major
//   A is [M, K], row-major
// Interpreting row-major matrices as column-major transposes:
//   op(X)=B_col = B_row^T has shape K x N? Careful.
//
// Easier mapping using column-major view:
// Row-major buffer [r, c] is column-major [c, r] with ld = c_count.
// Therefore:
//   A_row(M,K) buffer == A_col(K,M)
//   B_row(N,K) buffer == B_col(K,N)
//   C_row(M,N) buffer == C_col(N,M)
//
// Desired row-major result:
//   C_row(M,N) = A_row(M,K) * B_row(N,K)^T
// Transposing both sides:
//   C_col(N,M) = B_col(K,N)^T? No.
// Since A_row = A_col^T and B_row = B_col^T,
//   C_row = A_col^T * B_col
// Therefore
//   C_col = (C_row)^T = B_col^T * A_col
//
// So a column-major GEMM:
//   C_col(N,M) = (B_col)^T * A_col
// with:
//   B_col : K x N   => op(B_col)=N x K via transpose
//   A_col : K x M   => op(A_col)=K x M via non-transpose
// gives:
//   (N x K) * (K x M) = N x M = C_col
//
// Thus call:
//   cublasGemmEx(handle,
//                CUBLAS_OP_T, CUBLAS_OP_N,
//                N, M, K,
//                alpha,
//                B, K,
//                A, K,
//                beta,
//                C, N,
//                compute=32F, algo/tensor-op)
//
// BF16 support uses CUDA_R_16BF and CUBLAS_COMPUTE_32F_FAST_16BF if available.

typedef struct cublasContext* cublasHandle_t;
typedef int cublasStatus_t;
typedef int cublasOperation_t;
typedef int cudaDataType_t;
typedef int cublasComputeType_t;
typedef int cublasGemmAlgo_t;

static constexpr cublasStatus_t CUBLAS_STATUS_SUCCESS = 0;
static constexpr cublasOperation_t CUBLAS_OP_N = 0;
static constexpr cublasOperation_t CUBLAS_OP_T = 1;

// cudaDataType
static constexpr cudaDataType_t CUDA_R_16BF = 14;
static constexpr cudaDataType_t CUDA_R_32F = 0;

// compute types
static constexpr cublasComputeType_t CUBLAS_COMPUTE_32F = 68;
static constexpr cublasComputeType_t CUBLAS_COMPUTE_32F_FAST_16BF = 75;

// algo
static constexpr cublasGemmAlgo_t CUBLAS_GEMM_DEFAULT_TENSOR_OP = 99;

typedef cublasStatus_t (*cublasCreate_v2_t)(cublasHandle_t*);
typedef cublasStatus_t (*cublasDestroy_v2_t)(cublasHandle_t);
typedef cublasStatus_t (*cublasSetStream_v2_t)(cublasHandle_t, cudaStream_t);
typedef cublasStatus_t (*cublasGemmEx_t)(cublasHandle_t,
                                         cublasOperation_t,
                                         cublasOperation_t,
                                         int, int, int,
                                         const void*,
                                         const void*, cudaDataType_t, int,
                                         const void*, cudaDataType_t, int,
                                         const void*,
                                         void*, cudaDataType_t, int,
                                         cublasComputeType_t,
                                         cublasGemmAlgo_t);

struct CublasApi {
  void* handle = nullptr;
  cublasCreate_v2_t create = nullptr;
  cublasDestroy_v2_t destroy = nullptr;
  cublasSetStream_v2_t set_stream = nullptr;
  cublasGemmEx_t gemm_ex = nullptr;
  bool ok = false;
};

static CublasApi LoadCublas() {
  CublasApi api;
  // Try common sonames.
  const char* names[] = {
      "libcublas.so",
      "libcublas.so.12",
      "/usr/lib/x86_64-linux-gnu/libcublas.so",
      "/usr/lib/x86_64-linux-gnu/libcublas.so.12"};
  for (const char* n : names) {
    api.handle = dlopen(n, RTLD_LAZY | RTLD_LOCAL);
    if (api.handle) break;
  }
  if (!api.handle) return api;

  api.create =
      reinterpret_cast<cublasCreate_v2_t>(dlsym(api.handle, "cublasCreate_v2"));
  api.destroy =
      reinterpret_cast<cublasDestroy_v2_t>(dlsym(api.handle, "cublasDestroy_v2"));
  api.set_stream = reinterpret_cast<cublasSetStream_v2_t>(
      dlsym(api.handle, "cublasSetStream_v2"));
  api.gemm_ex =
      reinterpret_cast<cublasGemmEx_t>(dlsym(api.handle, "cublasGemmEx"));

  api.ok = api.create && api.destroy && api.set_stream && api.gemm_ex;
  if (!api.ok) {
    dlclose(api.handle);
    api.handle = nullptr;
  }
  return api;
}

// Fallback CUDA kernel only if cuBLAS loading fails.
// Simple but correct.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void FallbackGemmKernel(const __nv_bfloat16* __restrict__ A,
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
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

  for (int k0 = 0; k0 < K_CONST; k0 += BK) {
    for (int idx = tid; idx < BM * BK; idx += NUM_THREADS) {
      int r = idx / BK;
      int k = idx - r * BK;
      int gr = block_m + r;
      As[r][k] = (gr < M) ? A[(int64_t)gr * K_CONST + (k0 + k)]
                          : __float2bfloat16(0.0f);
    }
    for (int idx = tid; idx < BN * BK; idx += NUM_THREADS) {
      int n = idx / BK;
      int k = idx - n * BK;
      int gn = block_n + n;
      Bs[n][k] = (gn < N_CONST) ? B[(int64_t)gn * K_CONST + (k0 + k)]
                                : __float2bfloat16(0.0f);
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
      float a_frag[TM];
      float b_frag[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i) a_frag[i] = __bfloat162float(As[ty * TM + i][kk]);
#pragma unroll
      for (int j = 0; j < TN; ++j) b_frag[j] = __bfloat162float(Bs[tx * TN + j][kk]);
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] += a_frag[i] * b_frag[j];
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; ++i) {
    int row = row0 + i;
    if (row < M) {
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        int col = col0 + j;
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

  // Fast path: cuBLAS GEMMEx
  {
    CublasApi api = LoadCublas();
    if (api.ok) {
      cublasHandle_t handle;
      if (api.create(&handle) == CUBLAS_STATUS_SUCCESS) {
        if (api.set_stream(handle, stream) == CUBLAS_STATUS_SUCCESS) {
          const float alpha = 1.0f;
          const float beta = 0.0f;

          cublasStatus_t st = api.gemm_ex(
              handle,
              CUBLAS_OP_T, CUBLAS_OP_N,
              N_CONST, M, K_CONST,
              &alpha,
              B_ptr, CUDA_R_16BF, K_CONST,
              A_ptr, CUDA_R_16BF, K_CONST,
              &beta,
              C_ptr, CUDA_R_16BF, N_CONST,
              CUBLAS_COMPUTE_32F_FAST_16BF,
              CUBLAS_GEMM_DEFAULT_TENSOR_OP);

          if (st == CUBLAS_STATUS_SUCCESS) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            api.destroy(handle);
            dlclose(api.handle);
            return;
          }

          // Retry with conservative compute type.
          st = api.gemm_ex(
              handle,
              CUBLAS_OP_T, CUBLAS_OP_N,
              N_CONST, M, K_CONST,
              &alpha,
              B_ptr, CUDA_R_16BF, K_CONST,
              A_ptr, CUDA_R_16BF, K_CONST,
              &beta,
              C_ptr, CUDA_R_16BF, N_CONST,
              CUBLAS_COMPUTE_32F,
              CUBLAS_GEMM_DEFAULT_TENSOR_OP);

          if (st == CUBLAS_STATUS_SUCCESS) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            api.destroy(handle);
            dlclose(api.handle);
            return;
          }
        }
        api.destroy(handle);
      }
      dlclose(api.handle);
    }
  }

  // Fallback path if cuBLAS is unavailable.
  constexpr int BM = 64;
  constexpr int BN = 64;
  constexpr int BK = 32;
  constexpr int TM = 4;
  constexpr int TN = 4;
  dim3 block(BN / TN, BM / TM);
  dim3 grid((N_CONST + BN - 1) / BN, (M + BM - 1) / BM);

  FallbackGemmKernel<BM, BN, BK, TM, TN><<<grid, block, 0, stream>>>(
      A_ptr, B_ptr, C_ptr, M);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_n7168_k5120_cuda::run);

}  // namespace gemm_n7168_k5120_cuda