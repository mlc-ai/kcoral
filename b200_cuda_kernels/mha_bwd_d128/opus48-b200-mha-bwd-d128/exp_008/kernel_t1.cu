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

// Score gemm: C[64,64] = A[64,HD] @ (B[64,HD] as col-major -> B^T), i.e. C[m,n]=sum_d A[m,d]B[n,d]
__device__ __forceinline__ void score_gemm(const bf16* A, const bf16* B, float* C, int warpId) {
  #pragma unroll
  for (int tt = 0; tt < 2; tt++) {
    int t = warpId + tt * 8;      // 0..15
    int tm = t >> 2, tn = t & 3;  // 4x4 tiles
    wmma::fragment<wmma::accumulator,16,16,16,float> ac;
    wmma::fill_fragment(ac, 0.0f);
    #pragma unroll
    for (int k = 0; k < HD; k += 16) {
      wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::row_major> fa;
      wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::col_major> fb;
      wmma::load_matrix_sync(fa, A + tm*16*HD + k, HD);
      wmma::load_matrix_sync(fb, B + tn*16*HD + k, HD);
      wmma::mma_sync(ac, fa, fb, ac);
    }
    wmma::store_matrix_sync(C + tm*16*BQ + tn*16, ac, BQ, wmma::mem_row_major);
  }
}

__global__ void delta_kernel(const bf16* O, const bf16* dO, float* Delta) {
  int row = blockIdx.x;
  int t = threadIdx.x; // 0..127 == HD
  float v = __bfloat162float(O[(size_t)row*HD + t]) *
            __bfloat162float(dO[(size_t)row*HD + t]);
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
  __shared__ float ws[4];
  int lane = t & 31, wid = t >> 5;
  if (lane == 0) ws[wid] = v;
  __syncthreads();
  if (t == 0) Delta[row] = ws[0] + ws[1] + ws[2] + ws[3];
}

// dK, dV : grid over key-blocks, loop over query-blocks. Register-resident accumulators.
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
  bf16* sK  = (bf16*)smem;          // BK*HD
  bf16* sV  = sK  + BK*HD;
  bf16* sQ  = sV  + BK*HD;          // BQ*HD
  bf16* sdO = sQ  + BQ*HD;
  bf16* sP  = sdO + BQ*HD;          // BK*BQ
  bf16* sDS = sP  + BK*BQ;          // BK*BQ
  float* sScore  = (float*)(sDS + BK*BQ); // BK*BQ
  float* sScore2 = sScore + BK*BQ;         // BK*BQ  (contiguous -> stage[64][128])
  float* sL = sScore2 + BK*BQ;             // BQ
  float* sD = sL + BQ;                     // BQ
  float* stage = sScore;                   // [BK][HD]

  int tid = threadIdx.x, warpId = tid >> 5;
  bf16 zero = __float2bfloat16(0.f);

  for (int idx = tid; idx < BK*HD; idx += NT) {
    int j = idx / HD, c = idx % HD; int gr = kv_start + j;
    sK[idx] = (gr < S) ? Kbh[(size_t)gr*HD + c] : zero;
    sV[idx] = (gr < S) ? Vbh[(size_t)gr*HD + c] : zero;
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
      sQ[idx]  = (gr < S) ? Qbh[(size_t)gr*HD + c] : zero;
      sdO[idx] = (gr < S) ? dObh[(size_t)gr*HD + c] : zero;
    }
    if (tid < BQ) { int gr = q_start + tid; sL[tid] = (gr<S)?Lbh[gr]:0.f; sD[tid] = (gr<S)?Dbh[gr]:0.f; }
    __syncthreads();

    // S^T = K @ Q^T
    score_gemm(sK, sQ, sScore, warpId);
    __syncthreads();
    // P^T = exp(scale*S^T - L[q]) ; store float(P) in sScore, bf16 in sP
    for (int idx = tid; idx < BK*BQ; idx += NT) {
      int q = idx % BQ;
      float p = __expf(scale * sScore[idx] - sL[q]);
      sScore[idx] = p;
      sP[idx] = __float2bfloat16(p);
    }
    __syncthreads();
    // dP^T = V @ dO^T -> sScore2
    score_gemm(sV, sdO, sScore2, warpId);
    __syncthreads();
    // dS^T = P^T * (dP^T - D[q]) -> sDS (bf16)
    for (int idx = tid; idx < BK*BQ; idx += NT) {
      int q = idx % BQ;
      float ds = sScore[idx] * (sScore2[idx] - sD[q]);
      sDS[idx] = __float2bfloat16(ds);
    }
    __syncthreads();

    // accumulate dV (warps 0-3) and dK (warps 4-7)
    const bf16* Aacc = isV ? sP : sDS;   // [BK,BQ] row-major
    const bf16* Bacc = isV ? sdO : sQ;   // [BQ,HD] row-major
    wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::row_major> fa[4];
    #pragma unroll
    for (int k = 0; k < 4; k++)
      wmma::load_matrix_sync(fa[k], Aacc + mi*16*BQ + k*16, BQ);
    #pragma unroll
    for (int ni = 0; ni < 8; ni++) {
      #pragma unroll
      for (int k = 0; k < 4; k++) {
        wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::row_major> fb;
        wmma::load_matrix_sync(fb, Bacc + (k*16)*HD + ni*16, HD);
        wmma::mma_sync(acc[ni], fa[k], fb, acc[ni]);
      }
    }
    __syncthreads();
  }

  // Output dV
  if (isV) {
    #pragma unroll
    for (int ni = 0; ni < 8; ni++)
      wmma::store_matrix_sync(stage + mi*16*HD + ni*16, acc[ni], HD, wmma::mem_row_major);
  }
  __syncthreads();
  for (int idx = tid; idx < BK*HD; idx += NT) {
    int j = idx / HD, c = idx % HD; int gr = kv_start + j;
    if (gr < S) dVbh[(size_t)gr*HD + c] = __float2bfloat16(stage[idx]);
  }
  __syncthreads();
  // Output dK (scaled)
  if (!isV) {
    int mk = warpId - 4;
    #pragma unroll
    for (int ni = 0; ni < 8; ni++)
      wmma::store_matrix_sync(stage + mk*16*HD + ni*16, acc[ni], HD, wmma::mem_row_major);
  }
  __syncthreads();
  for (int idx = tid; idx < BK*HD; idx += NT) {
    int j = idx / HD, c = idx % HD; int gr = kv_start + j;
    if (gr < S) dKbh[(size_t)gr*HD + c] = __float2bfloat16(scale * stage[idx]);
  }
}

// dQ : grid over query-blocks, loop over key-blocks.
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
  bf16* sQ  = (bf16*)smem;          // BQ*HD
  bf16* sdO = sQ  + BQ*HD;
  bf16* sK  = sdO + BQ*HD;          // BK*HD
  bf16* sV  = sK  + BK*HD;
  bf16* sDS = sV  + BK*HD;          // BQ*BK
  float* sScore  = (float*)(sDS + BQ*BK); // BQ*BK
  float* sScore2 = sScore + BQ*BK;         // BQ*BK  (contiguous -> stage[64][128])
  float* sL = sScore2 + BQ*BK;
  float* sD = sL + BQ;
  float* stage = sScore;             // [BQ][HD]

  int tid = threadIdx.x, warpId = tid >> 5;
  bf16 zero = __float2bfloat16(0.f);

  for (int idx = tid; idx < BQ*HD; idx += NT) {
    int i = idx / HD, c = idx % HD; int gr = q_start + i;
    sQ[idx]  = (gr < S) ? Qbh[(size_t)gr*HD + c] : zero;
    sdO[idx] = (gr < S) ? dObh[(size_t)gr*HD + c] : zero;
  }
  if (tid < BQ) { int gr = q_start + tid; sL[tid] = (gr<S)?Lbh[gr]:0.f; sD[tid] = (gr<S)?Dbh[gr]:0.f; }
  __syncthreads();

  int mi = warpId >> 1;   // 0..3 row block
  int ng = warpId & 1;    // col group (0/1) -> ni in [ng*4, ng*4+4)
  wmma::fragment<wmma::accumulator,16,16,16,float> acc[4];
  #pragma unroll
  for (int i = 0; i < 4; i++) wmma::fill_fragment(acc[i], 0.0f);

  int numK = (S + BK - 1) / BK;
  for (int kb = 0; kb < numK; kb++) {
    int kv_start = kb * BK;
    for (int idx = tid; idx < BK*HD; idx += NT) {
      int j = idx / HD, c = idx % HD; int gr = kv_start + j;
      sK[idx] = (gr < S) ? Kbh[(size_t)gr*HD + c] : zero;
      sV[idx] = (gr < S) ? Vbh[(size_t)gr*HD + c] : zero;
    }
    __syncthreads();
    // S = Q @ K^T -> sScore  (rows=query, cols=key)
    score_gemm(sQ, sK, sScore, warpId);
    __syncthreads();
    // P = exp(scale*S - L[q]) ; keep float in sScore
    for (int idx = tid; idx < BQ*BK; idx += NT) {
      int q = idx / BK;
      sScore[idx] = __expf(scale * sScore[idx] - sL[q]);
    }
    __syncthreads();
    // dP = dO @ V^T -> sScore2
    score_gemm(sdO, sV, sScore2, warpId);
    __syncthreads();
    // dS = P * (dP - D[q]) -> sDS bf16
    for (int idx = tid; idx < BQ*BK; idx += NT) {
      int q = idx / BK;
      float ds = sScore[idx] * (sScore2[idx] - sD[q]);
      sDS[idx] = __float2bfloat16(ds);
    }
    __syncthreads();
    // dQ += dS @ K
    wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::row_major> fa[4];
    #pragma unroll
    for (int k = 0; k < 4; k++)
      wmma::load_matrix_sync(fa[k], sDS + mi*16*BK + k*16, BK);
    #pragma unroll
    for (int nn = 0; nn < 4; nn++) {
      int ni = ng*4 + nn;
      #pragma unroll
      for (int k = 0; k < 4; k++) {
        wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::row_major> fb;
        wmma::load_matrix_sync(fb, sK + (k*16)*HD + ni*16, HD);
        wmma::mma_sync(acc[nn], fa[k], fb, acc[nn]);
      }
    }
    __syncthreads();
  }

  // output dQ (scaled)
  #pragma unroll
  for (int nn = 0; nn < 4; nn++) {
    int ni = ng*4 + nn;
    wmma::store_matrix_sync(stage + mi*16*HD + ni*16, acc[nn], HD, wmma::mem_row_major);
  }
  __syncthreads();
  for (int idx = tid; idx < BQ*HD; idx += NT) {
    int i = idx / HD, c = idx % HD; int gr = q_start + i;
    if (gr < S) dQbh[(size_t)gr*HD + c] = __float2bfloat16(scale * stage[idx]);
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

  int sizeA = (int)((4*BK*HD + 2*BK*BQ) * sizeof(bf16) +
                    (2*BK*BQ + 2*BQ) * sizeof(float));
  int sizeB = (int)((2*BQ*HD + 2*BK*HD + BQ*BK) * sizeof(bf16) +
                    (2*BQ*BK + 2*BQ) * sizeof(float));

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