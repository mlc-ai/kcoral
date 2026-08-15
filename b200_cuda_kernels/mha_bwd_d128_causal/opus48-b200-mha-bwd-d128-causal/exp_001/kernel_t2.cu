#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <mma.h>
#include <cstdint>
#include <math.h>
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

#define HD 128
#define BM 64
#define BN 64

// ---------------- delta kernel: D_i = sum_e dO_ie * O_ie ----------------
__global__ void delta_kernel(const __nv_bfloat16* dO, const __nv_bfloat16* O,
                             float* D, int total) {
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const __nv_bfloat16* a = dO + idx * HD;
  const __nv_bfloat16* b = O + idx * HD;
  float acc = 0.f;
  #pragma unroll
  for (int c = 0; c < HD; c += 8) {
    float4 va = *reinterpret_cast<const float4*>(a + c);
    float4 vb = *reinterpret_cast<const float4*>(b + c);
    const __nv_bfloat16* pa = reinterpret_cast<const __nv_bfloat16*>(&va);
    const __nv_bfloat16* pb = reinterpret_cast<const __nv_bfloat16*>(&vb);
    #pragma unroll
    for (int i = 0; i < 8; i++) acc += __bfloat162float(pa[i]) * __bfloat162float(pb[i]);
  }
  D[idx] = acc;
}

// ---------------- device helpers ----------------
__device__ __forceinline__ void load_tile(__nv_bfloat16* dst, const __nv_bfloat16* src,
                                           int row0, int nrows, int S) {
  const int perRow = HD / 8;
  int total = nrows * perRow;
  for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
    int r = idx / perRow, c = idx % perRow;
    int gr = row0 + r;
    float4 v;
    if (gr < S) v = *reinterpret_cast<const float4*>(src + (long)gr * HD + c * 8);
    else { v.x = v.y = v.z = v.w = 0.f; }
    *reinterpret_cast<float4*>(dst + r * HD + c * 8) = v;
  }
}

__device__ __forceinline__ void load_vec(float* dst, const float* src,
                                          int row0, int nrows, int S) {
  for (int i = threadIdx.x; i < nrows; i += blockDim.x) {
    int gr = row0 + i;
    dst[i] = (gr < S) ? src[gr] : 0.f;
  }
}

// C[M=BM][N=BN] = A[BM][HD] @ B[BN][HD]^T   (fp32 out into sC)
__device__ __forceinline__ void mm_S(const __nv_bfloat16* sA, const __nv_bfloat16* sB,
                                     float* sC, int w) {
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bf;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf[BN / 16];
  #pragma unroll
  for (int n = 0; n < BN / 16; n++) wmma::fill_fragment(cf[n], 0.f);
  #pragma unroll
  for (int k = 0; k < HD / 16; k++) {
    wmma::load_matrix_sync(af, sA + (w * 16) * HD + k * 16, HD);
    #pragma unroll
    for (int n = 0; n < BN / 16; n++) {
      wmma::load_matrix_sync(bf, sB + (n * 16) * HD + k * 16, HD);
      wmma::mma_sync(cf[n], af, bf, cf[n]);
    }
  }
  #pragma unroll
  for (int n = 0; n < BN / 16; n++)
    wmma::store_matrix_sync(sC + (w * 16) * BN + n * 16, cf[n], BN, wmma::mem_row_major);
}

__device__ __forceinline__ void split_bf16(float x, __nv_bfloat16& hi, __nv_bfloat16& lo) {
  hi = __float2bfloat16(x);
  lo = __float2bfloat16(x - __bfloat162float(hi));
}

// ---------------- dK / dV kernel ----------------
__global__ __launch_bounds__(128) void dkdv_kernel(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* dO, const float* L, const float* D,
    __nv_bfloat16* dK, __nv_bfloat16* dV, int S, int nBlocks, float scale) {
  int bh = blockIdx.x, bj = blockIdx.y;
  int j0 = bj * BN;
  if (j0 >= S) return;

  const __nv_bfloat16* Qb = Q + (long)bh * S * HD;
  const __nv_bfloat16* Kb = K + (long)bh * S * HD;
  const __nv_bfloat16* Vb = V + (long)bh * S * HD;
  const __nv_bfloat16* dOb = dO + (long)bh * S * HD;
  const float* Lb = L + (long)bh * S;
  const float* Db = D + (long)bh * S;
  __nv_bfloat16* dKb = dK + (long)bh * S * HD;
  __nv_bfloat16* dVb = dV + (long)bh * S * HD;

  extern __shared__ char smem[];
  __nv_bfloat16* sK  = (__nv_bfloat16*)smem;
  __nv_bfloat16* sV  = sK + BN * HD;
  __nv_bfloat16* sQ  = sV + BN * HD;
  __nv_bfloat16* sdO = sQ + BM * HD;
  __nv_bfloat16* sPh = sdO + BM * HD;
  __nv_bfloat16* sPl = sPh + BM * BN;
  __nv_bfloat16* sDh = sPl + BM * BN;
  __nv_bfloat16* sDl = sDh + BM * BN;
  float* sScr = (float*)(sDl + BM * BN);
  float* sPf  = sScr + BM * BN;
  float* sL   = sPf + BM * BN;
  float* sD   = sL + BM;
  float* sdV  = sD + BM;          // [BN][HD]
  float* sdK  = sdV + BN * HD;    // [BN][HD]

  int w = threadIdx.x / 32;

  load_tile(sK, Kb, j0, BN, S);
  load_tile(sV, Vb, j0, BN, S);
  for (int idx = threadIdx.x; idx < BN * HD; idx += blockDim.x) { sdV[idx] = 0.f; sdK[idx] = 0.f; }
  __syncthreads();

  for (int bi = bj; bi < nBlocks; bi++) {
    int qi0 = bi * BM;
    load_tile(sQ, Qb, qi0, BM, S);
    load_tile(sdO, dOb, qi0, BM, S);
    load_vec(sL, Lb, qi0, BM, S);
    load_vec(sD, Db, qi0, BM, S);
    __syncthreads();

    mm_S(sQ, sK, sScr, w);        // S = Q @ K^T
    __syncthreads();

    // P (fp32 + split bf16), causal masked
    for (int idx = threadIdx.x; idx < BM * BN; idx += blockDim.x) {
      int ii = idx / BN, jj = idx % BN;
      int ig = qi0 + ii, jg = j0 + jj;
      float p = 0.f;
      if (ig < S && jg < S && jg <= ig) p = expf(scale * sScr[idx] - sL[ii]);
      sPf[idx] = p;
      __nv_bfloat16 hi, lo; split_bf16(p, hi, lo);
      sPh[idx] = hi; sPl[idx] = lo;
    }
    __syncthreads();

    mm_S(sdO, sV, sScr, w);       // dP = dO @ V^T
    __syncthreads();

    // dS = P * (dP - D), split bf16
    for (int idx = threadIdx.x; idx < BM * BN; idx += blockDim.x) {
      int ii = idx / BN;
      float ds = sPf[idx] * (sScr[idx] - sD[ii]);
      __nv_bfloat16 hi, lo; split_bf16(ds, hi, lo);
      sDh[idx] = hi; sDl[idx] = lo;
    }
    __syncthreads();

    // dV += P^T @ dO ; dK += dS^T @ Q  (warp w owns KV rows [w*16, w*16+16))
    #pragma unroll
    for (int n = 0; n < HD / 16; n++) {
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> fv, fk;
      wmma::load_matrix_sync(fv, sdV + (w * 16) * HD + n * 16, HD, wmma::mem_row_major);
      wmma::load_matrix_sync(fk, sdK + (w * 16) * HD + n * 16, HD, wmma::mem_row_major);
      #pragma unroll
      for (int k = 0; k < BM / 16; k++) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> aph, apl, adh, adl;
        wmma::load_matrix_sync(aph, sPh + (k * 16) * BN + w * 16, BN);
        wmma::load_matrix_sync(apl, sPl + (k * 16) * BN + w * 16, BN);
        wmma::load_matrix_sync(adh, sDh + (k * 16) * BN + w * 16, BN);
        wmma::load_matrix_sync(adl, sDl + (k * 16) * BN + w * 16, BN);
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b_do, b_q;
        wmma::load_matrix_sync(b_do, sdO + (k * 16) * HD + n * 16, HD);
        wmma::load_matrix_sync(b_q,  sQ  + (k * 16) * HD + n * 16, HD);
        wmma::mma_sync(fv, aph, b_do, fv);
        wmma::mma_sync(fv, apl, b_do, fv);
        wmma::mma_sync(fk, adh, b_q, fk);
        wmma::mma_sync(fk, adl, b_q, fk);
      }
      wmma::store_matrix_sync(sdV + (w * 16) * HD + n * 16, fv, HD, wmma::mem_row_major);
      wmma::store_matrix_sync(sdK + (w * 16) * HD + n * 16, fk, HD, wmma::mem_row_major);
    }
    __syncthreads();
  }

  for (int idx = threadIdx.x; idx < BN * HD; idx += blockDim.x) {
    int jj = idx / HD, e = idx % HD;
    int jg = j0 + jj;
    if (jg < S) {
      dVb[(long)jg * HD + e] = __float2bfloat16(sdV[idx]);
      dKb[(long)jg * HD + e] = __float2bfloat16(scale * sdK[idx]);
    }
  }
}

// ---------------- dQ kernel ----------------
__global__ __launch_bounds__(128) void dq_kernel(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* dO, const float* L, const float* D,
    __nv_bfloat16* dQ, int S, int nBlocks, float scale) {
  int bh = blockIdx.x, bi = blockIdx.y;
  int i0 = bi * BM;
  if (i0 >= S) return;

  const __nv_bfloat16* Qb = Q + (long)bh * S * HD;
  const __nv_bfloat16* Kb = K + (long)bh * S * HD;
  const __nv_bfloat16* Vb = V + (long)bh * S * HD;
  const __nv_bfloat16* dOb = dO + (long)bh * S * HD;
  const float* Lb = L + (long)bh * S;
  const float* Db = D + (long)bh * S;
  __nv_bfloat16* dQb = dQ + (long)bh * S * HD;

  extern __shared__ char smem[];
  __nv_bfloat16* sK  = (__nv_bfloat16*)smem;
  __nv_bfloat16* sV  = sK + BN * HD;
  __nv_bfloat16* sQ  = sV + BN * HD;
  __nv_bfloat16* sdO = sQ + BM * HD;
  __nv_bfloat16* sDh = sdO + BM * HD;
  __nv_bfloat16* sDl = sDh + BM * BN;
  float* sScr = (float*)(sDl + BM * BN);
  float* sPf  = sScr + BM * BN;
  float* sL   = sPf + BM * BN;
  float* sD   = sL + BM;
  float* sdQ  = sD + BM;   // [BM][HD]

  int w = threadIdx.x / 32;

  load_tile(sQ, Qb, i0, BM, S);
  load_tile(sdO, dOb, i0, BM, S);
  load_vec(sL, Lb, i0, BM, S);
  load_vec(sD, Db, i0, BM, S);
  for (int idx = threadIdx.x; idx < BM * HD; idx += blockDim.x) sdQ[idx] = 0.f;
  __syncthreads();

  for (int bj = 0; bj <= bi; bj++) {
    int j0 = bj * BN;
    load_tile(sK, Kb, j0, BN, S);
    load_tile(sV, Vb, j0, BN, S);
    __syncthreads();

    mm_S(sQ, sK, sScr, w);        // S = Q @ K^T
    __syncthreads();

    for (int idx = threadIdx.x; idx < BM * BN; idx += blockDim.x) {
      int ii = idx / BN, jj = idx % BN;
      int ig = i0 + ii, jg = j0 + jj;
      float p = 0.f;
      if (ig < S && jg < S && jg <= ig) p = expf(scale * sScr[idx] - sL[ii]);
      sPf[idx] = p;
    }
    __syncthreads();

    mm_S(sdO, sV, sScr, w);       // dP = dO @ V^T
    __syncthreads();

    for (int idx = threadIdx.x; idx < BM * BN; idx += blockDim.x) {
      int ii = idx / BN;
      float ds = sPf[idx] * (sScr[idx] - sD[ii]);
      __nv_bfloat16 hi, lo; split_bf16(ds, hi, lo);
      sDh[idx] = hi; sDl[idx] = lo;
    }
    __syncthreads();

    // dQ += dS @ K  (warp w owns Q rows [w*16, w*16+16))
    #pragma unroll
    for (int n = 0; n < HD / 16; n++) {
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> fq;
      wmma::load_matrix_sync(fq, sdQ + (w * 16) * HD + n * 16, HD, wmma::mem_row_major);
      #pragma unroll
      for (int k = 0; k < BN / 16; k++) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> ah, al;
        wmma::load_matrix_sync(ah, sDh + (w * 16) * BN + k * 16, BN);
        wmma::load_matrix_sync(al, sDl + (w * 16) * BN + k * 16, BN);
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b_k;
        wmma::load_matrix_sync(b_k, sK + (k * 16) * HD + n * 16, HD);
        wmma::mma_sync(fq, ah, b_k, fq);
        wmma::mma_sync(fq, al, b_k, fq);
      }
      wmma::store_matrix_sync(sdQ + (w * 16) * HD + n * 16, fq, HD, wmma::mem_row_major);
    }
    __syncthreads();
  }

  for (int idx = threadIdx.x; idx < BM * HD; idx += blockDim.x) {
    int ii = idx / HD, e = idx % HD;
    int ig = i0 + ii;
    if (ig < S) dQb[(long)ig * HD + e] = __float2bfloat16(scale * sdQ[idx]);
  }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int64_t B = Q.size(0), H = Q.size(1), S = Q.size(2), d = Q.size(3);
  int BH = (int)(B * H);

  const __nv_bfloat16* Qp = static_cast<const __nv_bfloat16*>(Q.data_ptr());
  const __nv_bfloat16* Kp = static_cast<const __nv_bfloat16*>(K.data_ptr());
  const __nv_bfloat16* Vp = static_cast<const __nv_bfloat16*>(V.data_ptr());
  const __nv_bfloat16* Op = static_cast<const __nv_bfloat16*>(O.data_ptr());
  const __nv_bfloat16* dOp = static_cast<const __nv_bfloat16*>(dO.data_ptr());
  const float* Lp = static_cast<const float*>(L.data_ptr());
  __nv_bfloat16* dQp = static_cast<__nv_bfloat16*>(dQ.data_ptr());
  __nv_bfloat16* dKp = static_cast<__nv_bfloat16*>(dK.data_ptr());
  __nv_bfloat16* dVp = static_cast<__nv_bfloat16*>(dV.data_ptr());

  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

  float* dD = nullptr;
  CUDA_CHECK(cudaMalloc(&dD, sizeof(float) * (size_t)BH * (size_t)S));

  int total = BH * (int)S;
  int tpb = 256;
  delta_kernel<<<(total + tpb - 1) / tpb, tpb, 0, stream>>>(dOp, Op, dD, total);
  CUDA_CHECK(cudaGetLastError());

  int nBlocks = ((int)S + BN - 1) / BN;
  float scale = 1.0f / sqrtf((float)d);

  // dkdv smem: bf16(sK,sV,sQ,sdO,sPh,sPl,sDh,sDl) + fp32(sScr,sPf,sL,sD,sdV,sdK)
  size_t bf16_dkdv = (size_t)(2 * BN * HD + 2 * BM * HD + 4 * BM * BN) * 2;
  size_t fp32_dkdv = (size_t)(2 * BM * BN + 2 * BM + 2 * BN * HD) * 4;
  size_t smem_dkdv = bf16_dkdv + fp32_dkdv;

  // dq smem: bf16(sK,sV,sQ,sdO,sDh,sDl) + fp32(sScr,sPf,sL,sD,sdQ)
  size_t bf16_dq = (size_t)(2 * BN * HD + 2 * BM * HD + 2 * BM * BN) * 2;
  size_t fp32_dq = (size_t)(2 * BM * BN + 2 * BM + BM * HD) * 4;
  size_t smem_dq = bf16_dq + fp32_dq;

  CUDA_CHECK(cudaFuncSetAttribute(dkdv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_dkdv));
  CUDA_CHECK(cudaFuncSetAttribute(dq_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_dq));

  dim3 grid(BH, nBlocks);
  dkdv_kernel<<<grid, 128, smem_dkdv, stream>>>(Qp, Kp, Vp, dOp, Lp, dD, dKp, dVp, (int)S, nBlocks, scale);
  CUDA_CHECK(cudaGetLastError());
  dq_kernel<<<grid, 128, smem_dq, stream>>>(Qp, Kp, Vp, dOp, Lp, dD, dQp, (int)S, nBlocks, scale);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaFree(dD));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd