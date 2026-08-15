#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <cmath>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

using namespace nvcuda;

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace mha_bwd {

typedef __nv_bfloat16 bf16;

constexpr int BK = 64;
constexpr int BQ = 64;
constexpr int HD = 128;
constexpr int NT = 256;
constexpr int LDH = HD + 8;   // padded leading dim for [*][HD] tiles
constexpr int LDS = BQ + 8;   // padded leading dim for score tiles

// C[64,64] : row_major A[.,lda] @ (col_major B[.,ldb]) -> C[.,ldc]
__device__ __forceinline__ void score_gemm(const bf16* A, int lda, const bf16* B, int ldb,
                                            float* C, int ldc, int warpId) {
  #pragma unroll
  for (int tt = 0; tt < 2; tt++) {
    int t = warpId + tt * 8;
    int tm = t >> 2, tn = t & 3;
    wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::row_major> fa[8];
    wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::col_major> fb[8];
    #pragma unroll
    for (int k = 0; k < 8; k++) {
      wmma::load_matrix_sync(fa[k], A + tm*16*lda + k*16, lda);
      wmma::load_matrix_sync(fb[k], B + tn*16*ldb + k*16, ldb);
    }
    wmma::fragment<wmma::accumulator,16,16,16,float> ac;
    wmma::fill_fragment(ac, 0.0f);
    #pragma unroll
    for (int k = 0; k < 8; k++) wmma::mma_sync(ac, fa[k], fb[k], ac);
    wmma::store_matrix_sync(C + tm*16*ldc + tn*16, ac, ldc, wmma::mem_row_major);
  }
}

__global__ void delta_kernel(const bf16* O, const bf16* dO, float* Delta) {
  int row = blockIdx.x;
  int t = threadIdx.x;
  float v = __bfloat162float(O[(size_t)row*HD + t]) *
            __bfloat162float(dO[(size_t)row*HD + t]);
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
  __shared__ float ws[4];
  int lane = t & 31, wid = t >> 5;
  if (lane == 0) ws[wid] = v;
  __syncthreads();
  if (t == 0) Delta[row] = ws[0] + ws[1] + ws[2] + ws[3];
}

__global__ __launch_bounds__(NT, 2) void bwd_dkdv_kernel(
    const bf16* Q, const bf16* K, const bf16* V, const bf16* dO,
    const float* L, const float* Delta,
    bf16* dK, bf16* dV, int B, int H, int S, float scale) {
  int b = blockIdx.z, h = blockIdx.y, kb = blockIdx.x;
  int kv_start = kb * BK;
  size_t bh = (size_t)(b*H + h);
  const bf16* Kbh = K + bh*S*HD;
  const bf16* Vbh = V + bh*S*HD;
  const bf16* Qbh = Q + bh*S*HD;
  const bf16* dObh = dO + bh*S*HD;
  const float* Lbh = L + bh*S;
  const float* Dbh = Delta + bh*S;
  bf16* dKbh = dK + bh*S*HD;
  bf16* dVbh = dV + bh*S*HD;

  extern __shared__ char smem[];
  bf16* sK  = (bf16*)smem;              // BK*LDH
  bf16* sV  = sK  + BK*LDH;
  bf16* sQ  = sV  + BK*LDH;             // BQ*LDH
  bf16* sdO = sQ  + BQ*LDH;
  bf16* sP  = sdO + BQ*LDH;             // BK*LDS
  bf16* sDS = sP  + BK*LDS;             // BK*LDS
  float* sScore = (float*)(sDS + BK*LDS); // BK*LDS
  float* sL = sScore + BK*LDS;
  float* sD = sL + BQ;
  float* stage = (float*)sK;            // BK*HD floats (aliases sK+sV)

  int tid = threadIdx.x, warpId = tid >> 5;
  bf16 zero = __float2bfloat16(0.f);

  for (int idx = tid; idx < BK*HD; idx += NT) {
    int j = idx / HD, c = idx % HD; int gr = kv_start + j;
    sK[j*LDH + c] = (gr < S) ? Kbh[(size_t)gr*HD + c] : zero;
    sV[j*LDH + c] = (gr < S) ? Vbh[(size_t)gr*HD + c] : zero;
  }
  __syncthreads();

  bool isV = warpId < 4;
  int mi = warpId & 3;
  wmma::fragment<wmma::accumulator,16,16,16,float> acc[8];
  #pragma unroll
  for (int i = 0; i < 8; i++) wmma::fill_fragment(acc[i], 0.0f);

  int numQ = (S + BQ - 1) / BQ;
  for (int qb = 0; qb < numQ; qb++) {
    int q_start = qb * BQ;
    for (int idx = tid; idx < BQ*HD; idx += NT) {
      int i = idx / HD, c = idx % HD; int gr = q_start + i;
      sQ[i*LDH + c]  = (gr < S) ? Qbh[(size_t)gr*HD + c] : zero;
      sdO[i*LDH + c] = (gr < S) ? dObh[(size_t)gr*HD + c] : zero;
    }
    if (tid < BQ) { int gr = q_start + tid; sL[tid] = (gr<S)?Lbh[gr]:0.f; sD[tid] = (gr<S)?Dbh[gr]:0.f; }
    __syncthreads();

    score_gemm(sK, LDH, sQ, LDH, sScore, LDS, warpId);   // S^T = K @ Q^T
    __syncthreads();
    for (int p = tid; p < BK*BQ; p += NT) {
      int i = p / BQ, q = p % BQ;
      sP[i*LDS + q] = __float2bfloat16(__expf(scale * sScore[i*LDS + q] - sL[q]));
    }
    __syncthreads();
    score_gemm(sV, LDH, sdO, LDH, sScore, LDS, warpId);  // dP^T = V @ dO^T
    __syncthreads();
    for (int p = tid; p < BK*BQ; p += NT) {
      int i = p / BQ, q = p % BQ;
      float pf = __bfloat162float(sP[i*LDS + q]);
      sDS[i*LDS + q] = __float2bfloat16(pf * (sScore[i*LDS + q] - sD[q]));
    }
    __syncthreads();

    const bf16* Aacc = isV ? sP : sDS;   // [BK][LDS]
    const bf16* Bacc = isV ? sdO : sQ;   // [BQ][LDH]
    wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::row_major> fa[4];
    #pragma unroll
    for (int k = 0; k < 4; k++)
      wmma::load_matrix_sync(fa[k], Aacc + mi*16*LDS + k*16, LDS);
    #pragma unroll
    for (int k = 0; k < 4; k++) {
      wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::row_major> fb[8];
      #pragma unroll
      for (int ni = 0; ni < 8; ni++)
        wmma::load_matrix_sync(fb[ni], Bacc + (k*16)*LDH + ni*16, LDH);
      #pragma unroll
      for (int ni = 0; ni < 8; ni++)
        wmma::mma_sync(acc[ni], fa[k], fb[ni], acc[ni]);
    }
    __syncthreads();
  }

  if (isV) {
    #pragma unroll
    for (int ni = 0; ni < 8; ni++)
      wmma::store_matrix_sync(stage + mi*16*HD + ni*16, acc[ni], HD, wmma::mem_row_major);
  }
  __syncthreads();
  for (int idx = tid; idx < BK*HD; idx += NT) {
    int j = idx / HD, c = idx % HD; int gr = kv_start + j;
    if (gr < S) dVbh[(size_t)gr*HD + c] = __float2bfloat16(stage[j*HD + c]);
  }
  __syncthreads();
  if (!isV) {
    int mk = warpId - 4;
    #pragma unroll
    for (int ni = 0; ni < 8; ni++)
      wmma::store_matrix_sync(stage + mk*16*HD + ni*16, acc[ni], HD, wmma::mem_row_major);
  }
  __syncthreads();
  for (int idx = tid; idx < BK*HD; idx += NT) {
    int j = idx / HD, c = idx % HD; int gr = kv_start + j;
    if (gr < S) dKbh[(size_t)gr*HD + c] = __float2bfloat16(scale * stage[j*HD + c]);
  }
}

__global__ __launch_bounds__(NT, 2) void bwd_dq_kernel(
    const bf16* Q, const bf16* K, const bf16* V, const bf16* dO,
    const float* L, const float* Delta,
    bf16* dQ, int B, int H, int S, float scale) {
  int b = blockIdx.z, h = blockIdx.y, qb = blockIdx.x;
  int q_start = qb * BQ;
  size_t bh = (size_t)(b*H + h);
  const bf16* Kbh = K + bh*S*HD;
  const bf16* Vbh = V + bh*S*HD;
  const bf16* Qbh = Q + bh*S*HD;
  const bf16* dObh = dO + bh*S*HD;
  const float* Lbh = L + bh*S;
  const float* Dbh = Delta + bh*S;
  bf16* dQbh = dQ + bh*S*HD;

  extern __shared__ char smem[];
  bf16* sQ  = (bf16*)smem;              // BQ*LDH
  bf16* sdO = sQ  + BQ*LDH;
  bf16* sK  = sdO + BQ*LDH;             // BK*LDH
  bf16* sV  = sK  + BK*LDH;
  bf16* sP  = sV  + BK*LDH;             // BQ*LDS
  bf16* sDS = sP  + BQ*LDS;             // BQ*LDS
  float* sScore = (float*)(sDS + BQ*LDS);
  float* sL = sScore + BQ*LDS;
  float* sD = sL + BQ;
  float* stage = (float*)sQ;

  int tid = threadIdx.x, warpId = tid >> 5;
  bf16 zero = __float2bfloat16(0.f);

  for (int idx = tid; idx < BQ*HD; idx += NT) {
    int i = idx / HD, c = idx % HD; int gr = q_start + i;
    sQ[i*LDH + c]  = (gr < S) ? Qbh[(size_t)gr*HD + c] : zero;
    sdO[i*LDH + c] = (gr < S) ? dObh[(size_t)gr*HD + c] : zero;
  }
  if (tid < BQ) { int gr = q_start + tid; sL[tid] = (gr<S)?Lbh[gr]:0.f; sD[tid] = (gr<S)?Dbh[gr]:0.f; }
  __syncthreads();

  int mi = warpId >> 1;
  int ng = warpId & 1;
  wmma::fragment<wmma::accumulator,16,16,16,float> acc[4];
  #pragma unroll
  for (int i = 0; i < 4; i++) wmma::fill_fragment(acc[i], 0.0f);

  int numK = (S + BK - 1) / BK;
  for (int kb = 0; kb < numK; kb++) {
    int kv_start = kb * BK;
    for (int idx = tid; idx < BK*HD; idx += NT) {
      int j = idx / HD, c = idx % HD; int gr = kv_start + j;
      sK[j*LDH + c] = (gr < S) ? Kbh[(size_t)gr*HD + c] : zero;
      sV[j*LDH + c] = (gr < S) ? Vbh[(size_t)gr*HD + c] : zero;
    }
    __syncthreads();
    score_gemm(sQ, LDH, sK, LDH, sScore, LDS, warpId);   // S = Q @ K^T [BQ][BK]
    __syncthreads();
    for (int p = tid; p < BQ*BK; p += NT) {
      int q = p / BK, kk = p % BK;
      sP[q*LDS + kk] = __float2bfloat16(__expf(scale * sScore[q*LDS + kk] - sL[q]));
    }
    __syncthreads();
    score_gemm(sdO, LDH, sV, LDH, sScore, LDS, warpId);  // dP = dO @ V^T
    __syncthreads();
    for (int p = tid; p < BQ*BK; p += NT) {
      int q = p / BK, kk = p % BK;
      float pf = __bfloat162float(sP[q*LDS + kk]);
      sDS[q*LDS + kk] = __float2bfloat16(pf * (sScore[q*LDS + kk] - sD[q]));
    }
    __syncthreads();
    // dQ += dS @ K
    wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::row_major> fa[4];
    #pragma unroll
    for (int k = 0; k < 4; k++)
      wmma::load_matrix_sync(fa[k], sDS + mi*16*LDS + k*16, LDS);
    #pragma unroll
    for (int k = 0; k < 4; k++) {
      wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::row_major> fb[4];
      #pragma unroll
      for (int nn = 0; nn < 4; nn++)
        wmma::load_matrix_sync(fb[nn], sK + (k*16)*LDH + (ng*4+nn)*16, LDH);
      #pragma unroll
      for (int nn = 0; nn < 4; nn++)
        wmma::mma_sync(acc[nn], fa[k], fb[nn], acc[nn]);
    }
    __syncthreads();
  }

  #pragma unroll
  for (int nn = 0; nn < 4; nn++)
    wmma::store_matrix_sync(stage + mi*16*HD + (ng*4+nn)*16, acc[nn], HD, wmma::mem_row_major);
  __syncthreads();
  for (int idx = tid; idx < BQ*HD; idx += NT) {
    int i = idx / HD, c = idx % HD; int gr = q_start + i;
    if (gr < S) dQbh[(size_t)gr*HD + c] = __float2bfloat16(scale * stage[i*HD + c]);
  }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int B = Q.size(0), H = Q.size(1), S = Q.size(2), d = Q.size(3);
  float scale = 1.0f / sqrtf((float)d);

  const bf16* Qp = (const bf16*)Q.data_ptr();
  const bf16* Kp = (const bf16*)K.data_ptr();
  const bf16* Vp = (const bf16*)V.data_ptr();
  const bf16* Op = (const bf16*)O.data_ptr();
  const bf16* dOp = (const bf16*)dO.data_ptr();
  const float* Lp = (const float*)L.data_ptr();
  bf16* dQp = (bf16*)dQ.data_ptr();
  bf16* dKp = (bf16*)dK.data_ptr();
  bf16* dVp = (bf16*)dV.data_ptr();

  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

  if (S == 0) { CUDA_CHECK(cudaStreamSynchronize(stream)); return; }

  size_t totalRows = (size_t)B * H * S;
  float* Delta = nullptr;
  CUDA_CHECK(cudaMallocAsync(&Delta, totalRows * sizeof(float), stream));

  delta_kernel<<<(unsigned)totalRows, HD, 0, stream>>>(Op, dOp, Delta);
  CUDA_CHECK(cudaGetLastError());

  int sizeA = (int)((4*BK*LDH + 2*BK*LDS) * sizeof(bf16) + (BK*LDS + 2*BQ) * sizeof(float));
  int sizeB = (int)((4*BQ*LDH + 2*BQ*LDS) * sizeof(bf16) + (BQ*LDS + 2*BQ) * sizeof(float));

  CUDA_CHECK(cudaFuncSetAttribute(bwd_dkdv_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, sizeA));
  CUDA_CHECK(cudaFuncSetAttribute(bwd_dq_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, sizeB));

  int numK = (S + BK - 1) / BK;
  int numQ = (S + BQ - 1) / BQ;
  dim3 gA(numK, H, B);
  dim3 gB(numQ, H, B);

  bwd_dkdv_kernel<<<gA, NT, sizeA, stream>>>(
      Qp, Kp, Vp, dOp, Lp, Delta, dKp, dVp, B, H, S, scale);
  CUDA_CHECK(cudaGetLastError());

  bwd_dq_kernel<<<gB, NT, sizeB, stream>>>(
      Qp, Kp, Vp, dOp, Lp, Delta, dQp, B, H, S, scale);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaFreeAsync(Delta, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd