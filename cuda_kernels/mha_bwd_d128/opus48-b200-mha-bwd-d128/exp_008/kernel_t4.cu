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
constexpr int HDV = HD / 8;   // uint4 units per row

__device__ __forceinline__ uint4 pack8_scaled(const float* s, float sc) {
  __nv_bfloat162 a = __floats2bfloat162_rn(s[0]*sc, s[1]*sc);
  __nv_bfloat162 b = __floats2bfloat162_rn(s[2]*sc, s[3]*sc);
  __nv_bfloat162 c = __floats2bfloat162_rn(s[4]*sc, s[5]*sc);
  __nv_bfloat162 d = __floats2bfloat162_rn(s[6]*sc, s[7]*sc);
  uint4 o;
  o.x = *reinterpret_cast<uint32_t*>(&a);
  o.y = *reinterpret_cast<uint32_t*>(&b);
  o.z = *reinterpret_cast<uint32_t*>(&c);
  o.w = *reinterpret_cast<uint32_t*>(&d);
  return o;
}

// Vectorized load a [rows x HD] bf16 tile from global (with row-bound) into smem.
template<int ROWS>
__device__ __forceinline__ void vload_tile(const bf16* G, bf16* Sm, int row0, int S, int tid) {
  const uint4 z = make_uint4(0,0,0,0);
  #pragma unroll
  for (int u = tid; u < ROWS*HDV; u += NT) {
    int row = u / HDV, c8 = (u % HDV) * 8;
    int gr = row0 + row;
    uint4 v = (gr < S) ? *reinterpret_cast<const uint4*>(&G[(size_t)gr*HD + c8]) : z;
    *reinterpret_cast<uint4*>(&Sm[row*HD + c8]) = v;
  }
}

// C[64,64] : C[m,n] = sum_d A[m,d]*B[n,d]; A,B are [64,HD] row-major
__device__ __forceinline__ void score_gemm(const bf16* A, const bf16* B, float* C, int warpId) {
  #pragma unroll
  for (int tt = 0; tt < 2; tt++) {
    int t = warpId + tt * 8;
    int tm = t >> 2, tn = t & 3;
    wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::row_major> fa[8];
    wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::col_major> fb[8];
    #pragma unroll
    for (int k = 0; k < 8; k++) {
      wmma::load_matrix_sync(fa[k], A + tm*16*HD + k*16, HD);
      wmma::load_matrix_sync(fb[k], B + tn*16*HD + k*16, HD);
    }
    wmma::fragment<wmma::accumulator,16,16,16,float> ac;
    wmma::fill_fragment(ac, 0.0f);
    #pragma unroll
    for (int k = 0; k < 8; k++) wmma::mma_sync(ac, fa[k], fb[k], ac);
    wmma::store_matrix_sync(C + tm*16*BQ + tn*16, ac, BQ, wmma::mem_row_major);
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
  bf16* sK  = (bf16*)smem;          // BK*HD
  bf16* sV  = sK  + BK*HD;
  bf16* sQ  = sV  + BK*HD;          // BQ*HD
  bf16* sdO = sQ  + BQ*HD;
  bf16* sP  = sdO + BQ*HD;          // BK*BQ
  bf16* sDS = sP  + BK*BQ;          // BK*BQ
  float* sScore = (float*)(sDS + BK*BQ); // BK*BQ
  float* sL = sScore + BK*BQ;
  float* sD = sL + BQ;
  float* stage = (float*)sK;        // BK*HD floats

  int tid = threadIdx.x, warpId = tid >> 5;

  vload_tile<BK>(Kbh, sK, kv_start, S, tid);
  vload_tile<BK>(Vbh, sV, kv_start, S, tid);
  __syncthreads();

  bool isV = warpId < 4;
  int mi = warpId & 3;
  wmma::fragment<wmma::accumulator,16,16,16,float> acc[8];
  #pragma unroll
  for (int i = 0; i < 8; i++) wmma::fill_fragment(acc[i], 0.0f);

  int numQ = (S + BQ - 1) / BQ;
  for (int qb = 0; qb < numQ; qb++) {
    int q_start = qb * BQ;
    vload_tile<BQ>(Qbh, sQ, q_start, S, tid);
    vload_tile<BQ>(dObh, sdO, q_start, S, tid);
    if (tid < BQ) { int gr = q_start + tid; sL[tid] = (gr<S)?Lbh[gr]:0.f; sD[tid] = (gr<S)?Dbh[gr]:0.f; }
    __syncthreads();

    score_gemm(sK, sQ, sScore, warpId);       // S^T = K @ Q^T
    __syncthreads();
    for (int idx = tid; idx < BK*BQ; idx += NT) {
      int q = idx & (BQ-1);
      sP[idx] = __float2bfloat16(__expf(scale * sScore[idx] - sL[q]));
    }
    __syncthreads();
    score_gemm(sV, sdO, sScore, warpId);      // dP^T = V @ dO^T
    __syncthreads();
    for (int idx = tid; idx < BK*BQ; idx += NT) {
      int q = idx & (BQ-1);
      float pf = __bfloat162float(sP[idx]);
      sDS[idx] = __float2bfloat16(pf * (sScore[idx] - sD[q]));
    }
    __syncthreads();

    const bf16* Aacc = isV ? sP : sDS;    // [BK,BQ]
    const bf16* Bacc = isV ? sdO : sQ;    // [BQ,HD]
    wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::row_major> fa[4];
    #pragma unroll
    for (int k = 0; k < 4; k++)
      wmma::load_matrix_sync(fa[k], Aacc + mi*16*BQ + k*16, BQ);
    #pragma unroll
    for (int k = 0; k < 4; k++) {
      wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::row_major> fb[8];
      #pragma unroll
      for (int ni = 0; ni < 8; ni++)
        wmma::load_matrix_sync(fb[ni], Bacc + (k*16)*HD + ni*16, HD);
      #pragma unroll
      for (int ni = 0; ni < 8; ni++)
        wmma::mma_sync(acc[ni], fa[k], fb[ni], acc[ni]);
    }
    __syncthreads();
  }

  // dV out
  if (isV) {
    #pragma unroll
    for (int ni = 0; ni < 8; ni++)
      wmma::store_matrix_sync(stage + mi*16*HD + ni*16, acc[ni], HD, wmma::mem_row_major);
  }
  __syncthreads();
  #pragma unroll
  for (int u = tid; u < BK*HDV; u += NT) {
    int row = u / HDV, c8 = (u % HDV)*8; int gr = kv_start + row;
    if (gr < S) *reinterpret_cast<uint4*>(&dVbh[(size_t)gr*HD + c8]) = pack8_scaled(&stage[row*HD+c8], 1.0f);
  }
  __syncthreads();
  // dK out (scaled)
  if (!isV) {
    int mk = warpId - 4;
    #pragma unroll
    for (int ni = 0; ni < 8; ni++)
      wmma::store_matrix_sync(stage + mk*16*HD + ni*16, acc[ni], HD, wmma::mem_row_major);
  }
  __syncthreads();
  #pragma unroll
  for (int u = tid; u < BK*HDV; u += NT) {
    int row = u / HDV, c8 = (u % HDV)*8; int gr = kv_start + row;
    if (gr < S) *reinterpret_cast<uint4*>(&dKbh[(size_t)gr*HD + c8]) = pack8_scaled(&stage[row*HD+c8], scale);
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
  bf16* sQ  = (bf16*)smem;          // BQ*HD
  bf16* sdO = sQ  + BQ*HD;
  bf16* sK  = sdO + BQ*HD;          // BK*HD
  bf16* sV  = sK  + BK*HD;
  bf16* sP  = sV  + BK*HD;          // BQ*BK
  bf16* sDS = sP  + BQ*BK;          // BQ*BK
  float* sScore = (float*)(sDS + BQ*BK);
  float* sL = sScore + BQ*BK;
  float* sD = sL + BQ;
  float* stage = (float*)sQ;

  int tid = threadIdx.x, warpId = tid >> 5;

  vload_tile<BQ>(Qbh, sQ, q_start, S, tid);
  vload_tile<BQ>(dObh, sdO, q_start, S, tid);
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
    vload_tile<BK>(Kbh, sK, kv_start, S, tid);
    vload_tile<BK>(Vbh, sV, kv_start, S, tid);
    __syncthreads();
    score_gemm(sQ, sK, sScore, warpId);      // S = Q @ K^T [BQ,BK]
    __syncthreads();
    for (int idx = tid; idx < BQ*BK; idx += NT) {
      int q = idx / BK;
      sP[idx] = __float2bfloat16(__expf(scale * sScore[idx] - sL[q]));
    }
    __syncthreads();
    score_gemm(sdO, sV, sScore, warpId);     // dP = dO @ V^T
    __syncthreads();
    for (int idx = tid; idx < BQ*BK; idx += NT) {
      int q = idx / BK;
      float pf = __bfloat162float(sP[idx]);
      sDS[idx] = __float2bfloat16(pf * (sScore[idx] - sD[q]));
    }
    __syncthreads();
    // dQ += dS @ K
    wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::row_major> fa[4];
    #pragma unroll
    for (int k = 0; k < 4; k++)
      wmma::load_matrix_sync(fa[k], sDS + mi*16*BK + k*16, BK);
    #pragma unroll
    for (int k = 0; k < 4; k++) {
      wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::row_major> fb[4];
      #pragma unroll
      for (int nn = 0; nn < 4; nn++)
        wmma::load_matrix_sync(fb[nn], sK + (k*16)*HD + (ng*4+nn)*16, HD);
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
  #pragma unroll
  for (int u = tid; u < BQ*HDV; u += NT) {
    int row = u / HDV, c8 = (u % HDV)*8; int gr = q_start + row;
    if (gr < S) *reinterpret_cast<uint4*>(&dQbh[(size_t)gr*HD + c8]) = pack8_scaled(&stage[row*HD+c8], scale);
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

  int sizeA = (int)((4*BK*HD + 2*BK*BQ) * sizeof(bf16) + (BK*BQ + 2*BQ) * sizeof(float));
  int sizeB = (int)((4*BQ*HD + 2*BQ*BK) * sizeof(bf16) + (BQ*BK + 2*BQ) * sizeof(float));

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