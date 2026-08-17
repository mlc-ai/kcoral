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
constexpr int NWARP = 8;
constexpr int NTHREAD = 256;

// Generic WMMA GEMM: C[M,N] (+)= A@B, with configurable operand layouts.
template<typename LayA, typename LayB, bool COLA, bool COLB, bool ACC>
__device__ __forceinline__ void wmma_gemm(
    const bf16* A, int lda, const bf16* B, int ldb,
    float* C, int ldc, int M, int N, int Kd, int warpId) {
  const int tilesN = N / 16;
  const int numTiles = (M / 16) * tilesN;
  for (int t = warpId; t < numTiles; t += NWARP) {
    int tm = t / tilesN;
    int tn = t % tilesN;
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;
    if (ACC) wmma::load_matrix_sync(acc, C + (tm*16)*ldc + tn*16, ldc, wmma::mem_row_major);
    else     wmma::fill_fragment(acc, 0.0f);
    for (int k = 0; k < Kd; k += 16) {
      wmma::fragment<wmma::matrix_a,16,16,16,bf16,LayA> fa;
      wmma::fragment<wmma::matrix_b,16,16,16,bf16,LayB> fb;
      const bf16* pa = COLA ? (A + tm*16 + k*lda) : (A + (tm*16)*lda + k);
      const bf16* pb = COLB ? (B + k + (tn*16)*ldb) : (B + k*ldb + tn*16);
      wmma::load_matrix_sync(fa, pa, lda);
      wmma::load_matrix_sync(fb, pb, ldb);
      wmma::mma_sync(acc, fa, fb, acc);
    }
    wmma::store_matrix_sync(C + (tm*16)*ldc + tn*16, acc, ldc, wmma::mem_row_major);
  }
}

// D_i = sum_c O[i,c]*dO[i,c]
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

// dK, dV : grid over key-blocks, loop over query-blocks.
__global__ void bwd_dkdv_kernel(
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
  float* sdV = (float*)smem;          // BK*HD
  float* sdK = sdV + BK*HD;           // BK*HD
  float* sSf = sdK + BK*HD;           // BK*BQ
  float* sDPf = sSf + BK*BQ;          // BK*BQ
  float* sL = sDPf + BK*BQ;           // BQ
  float* sD = sL + BQ;                // BQ
  bf16* sK = (bf16*)(sD + BQ);        // BK*HD
  bf16* sV = sK + BK*HD;              // BK*HD
  bf16* sQ = sV + BK*HD;              // BQ*HD
  bf16* sdO = sQ + BQ*HD;             // BQ*HD
  bf16* sP = sdO + BQ*HD;             // BK*BQ
  bf16* sDS = sP + BK*BQ;             // BK*BQ

  int tid = threadIdx.x;
  int warpId = tid >> 5;
  bf16 zero = __float2bfloat16(0.f);

  for (int idx = tid; idx < BK*HD; idx += NTHREAD) {
    int j = idx / HD, c = idx % HD; int gr = kv_start + j;
    sK[idx] = (gr < S) ? Kbh[(size_t)gr*HD + c] : zero;
    sV[idx] = (gr < S) ? Vbh[(size_t)gr*HD + c] : zero;
    sdV[idx] = 0.f; sdK[idx] = 0.f;
  }
  __syncthreads();

  int numQ = (S + BQ - 1) / BQ;
  for (int qb = 0; qb < numQ; qb++) {
    int q_start = qb * BQ;
    for (int idx = tid; idx < BQ*HD; idx += NTHREAD) {
      int i = idx / HD, c = idx % HD; int gr = q_start + i;
      sQ[idx]  = (gr < S) ? Qbh[(size_t)gr*HD + c] : zero;
      sdO[idx] = (gr < S) ? dObh[(size_t)gr*HD + c] : zero;
    }
    if (tid < BQ) { int gr = q_start + tid; sL[tid] = (gr<S)?Lbh[gr]:0.f; sD[tid] = (gr<S)?Dbh[gr]:0.f; }
    __syncthreads();

    // S^T = K @ Q^T   [BK,BQ]
    wmma_gemm<wmma::row_major, wmma::col_major, false, true, false>(
        sK, HD, sQ, HD, sSf, BQ, BK, BQ, HD, warpId);
    __syncthreads();
    // P^T = exp(scale*S^T - L)
    for (int idx = tid; idx < BK*BQ; idx += NTHREAD) {
      int i = idx % BQ;
      float p = __expf(scale * sSf[idx] - sL[i]);
      sSf[idx] = p; sP[idx] = __float2bfloat16(p);
    }
    __syncthreads();
    // dP^T = V @ dO^T   [BK,BQ]
    wmma_gemm<wmma::row_major, wmma::col_major, false, true, false>(
        sV, HD, sdO, HD, sDPf, BQ, BK, BQ, HD, warpId);
    __syncthreads();
    // dS^T = P^T * (dP^T - D)
    for (int idx = tid; idx < BK*BQ; idx += NTHREAD) {
      int i = idx % BQ;
      float ds = sSf[idx] * (sDPf[idx] - sD[i]);
      sDS[idx] = __float2bfloat16(ds);
    }
    __syncthreads();
    // dV += P^T @ dO   [BK,HD]
    wmma_gemm<wmma::row_major, wmma::row_major, false, false, true>(
        sP, BQ, sdO, HD, sdV, HD, BK, HD, BQ, warpId);
    // dK += dS^T @ Q   [BK,HD]
    wmma_gemm<wmma::row_major, wmma::row_major, false, false, true>(
        sDS, BQ, sQ, HD, sdK, HD, BK, HD, BQ, warpId);
    __syncthreads();
  }

  for (int idx = tid; idx < BK*HD; idx += NTHREAD) {
    int j = idx / HD, c = idx % HD; int gr = kv_start + j;
    if (gr < S) {
      dVbh[(size_t)gr*HD + c] = __float2bfloat16(sdV[idx]);
      dKbh[(size_t)gr*HD + c] = __float2bfloat16(scale * sdK[idx]);
    }
  }
}

// dQ : grid over query-blocks, loop over key-blocks.
__global__ void bwd_dq_kernel(
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
  float* sdQ = (float*)smem;          // BQ*HD
  float* sSf = sdQ + BQ*HD;           // BK*BQ
  float* sDPf = sSf + BK*BQ;          // BK*BQ
  float* sL = sDPf + BK*BQ;           // BQ
  float* sD = sL + BQ;                // BQ
  bf16* sQ = (bf16*)(sD + BQ);        // BQ*HD
  bf16* sdO = sQ + BQ*HD;             // BQ*HD
  bf16* sK = sdO + BQ*HD;             // BK*HD
  bf16* sV = sK + BK*HD;              // BK*HD
  bf16* sDS = sV + BK*HD;             // BK*BQ

  int tid = threadIdx.x;
  int warpId = tid >> 5;
  bf16 zero = __float2bfloat16(0.f);

  for (int idx = tid; idx < BQ*HD; idx += NTHREAD) {
    int i = idx / HD, c = idx % HD; int gr = q_start + i;
    sQ[idx]  = (gr < S) ? Qbh[(size_t)gr*HD + c] : zero;
    sdO[idx] = (gr < S) ? dObh[(size_t)gr*HD + c] : zero;
    sdQ[idx] = 0.f;
  }
  if (tid < BQ) { int gr = q_start + tid; sL[tid] = (gr<S)?Lbh[gr]:0.f; sD[tid] = (gr<S)?Dbh[gr]:0.f; }
  __syncthreads();

  int numK = (S + BK - 1) / BK;
  for (int kb = 0; kb < numK; kb++) {
    int kv_start = kb * BK;
    for (int idx = tid; idx < BK*HD; idx += NTHREAD) {
      int j = idx / HD, c = idx % HD; int gr = kv_start + j;
      sK[idx] = (gr < S) ? Kbh[(size_t)gr*HD + c] : zero;
      sV[idx] = (gr < S) ? Vbh[(size_t)gr*HD + c] : zero;
    }
    __syncthreads();
    // S^T = K @ Q^T
    wmma_gemm<wmma::row_major, wmma::col_major, false, true, false>(
        sK, HD, sQ, HD, sSf, BQ, BK, BQ, HD, warpId);
    __syncthreads();
    // P^T
    for (int idx = tid; idx < BK*BQ; idx += NTHREAD) {
      int i = idx % BQ;
      sSf[idx] = __expf(scale * sSf[idx] - sL[i]);
    }
    __syncthreads();
    // dP^T = V @ dO^T
    wmma_gemm<wmma::row_major, wmma::col_major, false, true, false>(
        sV, HD, sdO, HD, sDPf, BQ, BK, BQ, HD, warpId);
    __syncthreads();
    // dS^T = P^T * (dP^T - D)
    for (int idx = tid; idx < BK*BQ; idx += NTHREAD) {
      int i = idx % BQ;
      float ds = sSf[idx] * (sDPf[idx] - sD[i]);
      sDS[idx] = __float2bfloat16(ds);
    }
    __syncthreads();
    // dQ += dS @ K   (dS = (dS^T)^T via col-major matrix_a)   [BQ,HD]
    wmma_gemm<wmma::col_major, wmma::row_major, true, false, true>(
        sDS, BQ, sK, HD, sdQ, HD, BQ, HD, BK, warpId);
    __syncthreads();
  }

  for (int idx = tid; idx < BQ*HD; idx += NTHREAD) {
    int i = idx / HD, c = idx % HD; int gr = q_start + i;
    if (gr < S) dQbh[(size_t)gr*HD + c] = __float2bfloat16(scale * sdQ[idx]);
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

  int sizeA = (int)((2*BK*HD + 2*BK*BQ + 2*BQ) * sizeof(float) +
                    (2*BK*HD + 2*BQ*HD + 2*BK*BQ) * sizeof(bf16));
  int sizeB = (int)((BQ*HD + 2*BK*BQ + 2*BQ) * sizeof(float) +
                    (2*BQ*HD + 2*BK*HD + BK*BQ) * sizeof(bf16));

  CUDA_CHECK(cudaFuncSetAttribute(bwd_dkdv_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, sizeA));
  CUDA_CHECK(cudaFuncSetAttribute(bwd_dq_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, sizeB));

  int numK = (S + BK - 1) / BK;
  int numQ = (S + BQ - 1) / BQ;
  dim3 gA(numK, H, B);
  dim3 gB(numQ, H, B);

  bwd_dkdv_kernel<<<gA, NTHREAD, sizeA, stream>>>(
      Qp, Kp, Vp, dOp, Lp, Delta, dKp, dVp, B, H, S, scale);
  CUDA_CHECK(cudaGetLastError());

  bwd_dq_kernel<<<gB, NTHREAD, sizeB, stream>>>(
      Qp, Kp, Vp, dOp, Lp, Delta, dQp, B, H, S, scale);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaFreeAsync(Delta, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd