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
#define NWARP 8
#define NTHREAD (NWARP*32)

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
__device__ __forceinline__ void load_tile(half* dst, const __nv_bfloat16* src,
                                           int row0, int nrows, int S) {
  const int perRow = HD / 8;
  int total = nrows * perRow;
  for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
    int r = idx / perRow, c = idx % perRow;
    int gr = row0 + r;
    half* pd = dst + r * HD + c * 8;
    if (gr < S) {
      float4 v = *reinterpret_cast<const float4*>(src + (long)gr * HD + c * 8);
      const __nv_bfloat16* pb = reinterpret_cast<const __nv_bfloat16*>(&v);
      #pragma unroll
      for (int i = 0; i < 8; i++) pd[i] = __float2half(__bfloat162float(pb[i]));
    } else {
      #pragma unroll
      for (int i = 0; i < 8; i++) pd[i] = __float2half(0.f);
    }
  }
}

__device__ __forceinline__ void load_vec(float* dst, const float* src,
                                          int row0, int nrows, int S) {
  for (int i = threadIdx.x; i < nrows; i += blockDim.x) {
    int gr = row0 + i;
    dst[i] = (gr < S) ? src[gr] : 0.f;
  }
}

// C[BM x BN] = A[BM x HD] @ B[BN x HD]^T  into sC (fp32), 8-warp tiling
__device__ __forceinline__ void qk_matmul(const half* sA, const half* sB, float* sC, int w) {
  int wr = w & 3;    // BM row tile (0..3)
  int wc = w >> 2;   // BN col group (0..1), covers 2 tiles
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bf;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf[2];
  #pragma unroll
  for (int nn = 0; nn < 2; nn++) wmma::fill_fragment(cf[nn], 0.f);
  #pragma unroll
  for (int k = 0; k < HD / 16; k++) {
    wmma::load_matrix_sync(af, sA + (wr * 16) * HD + k * 16, HD);
    #pragma unroll
    for (int nn = 0; nn < 2; nn++) {
      int n = wc * 2 + nn;
      wmma::load_matrix_sync(bf, sB + (n * 16) * HD + k * 16, HD);
      wmma::mma_sync(cf[nn], af, bf, cf[nn]);
    }
  }
  #pragma unroll
  for (int nn = 0; nn < 2; nn++) {
    int n = wc * 2 + nn;
    wmma::store_matrix_sync(sC + (wr * 16) * BN + n * 16, cf[nn], BN, wmma::mem_row_major);
  }
}

// ---------------- dK / dV kernel ----------------
__global__ __launch_bounds__(NTHREAD) void dkdv_kernel(
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
  half* sK  = (half*)smem;
  half* sV  = sK + BN * HD;
  half* sQ  = sV + BN * HD;
  half* sdO = sQ + BM * HD;
  half* sP  = sdO + BM * HD;
  half* sDS = sP + BM * BN;
  float* sScr = (float*)(sDS + BM * BN);
  float* sL   = sScr + BM * BN;
  float* sD   = sL + BM;

  int tid = threadIdx.x;
  int w = tid >> 5;
  int br = w & 3;    // BN row tile (0..3)
  int bc = w >> 2;   // HD half (0..1)

  load_tile(sK, Kb, j0, BN, S);
  load_tile(sV, Vb, j0, BN, S);

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> accV[4], accK[4];
  #pragma unroll
  for (int nt = 0; nt < 4; nt++) { wmma::fill_fragment(accV[nt], 0.f); wmma::fill_fragment(accK[nt], 0.f); }
  __syncthreads();

  for (int bi = bj; bi < nBlocks; bi++) {
    int qi0 = bi * BM;
    load_tile(sQ, Qb, qi0, BM, S);
    load_tile(sdO, dOb, qi0, BM, S);
    load_vec(sL, Lb, qi0, BM, S);
    load_vec(sD, Db, qi0, BM, S);
    __syncthreads();

    qk_matmul(sQ, sK, sScr, w);   // S = Q @ K^T
    __syncthreads();

    for (int idx = tid; idx < BM * BN; idx += blockDim.x) {
      int ii = idx / BN, jj = idx % BN;
      int ig = qi0 + ii, jg = j0 + jj;
      float p = 0.f;
      if (ig < S && jg < S && jg <= ig) p = __expf(scale * sScr[idx] - sL[ii]);
      sP[idx] = __float2half(p);
    }
    __syncthreads();

    qk_matmul(sdO, sV, sScr, w);  // dP = dO @ V^T
    __syncthreads();

    for (int idx = tid; idx < BM * BN; idx += blockDim.x) {
      int ii = idx / BN;
      float p = __half2float(sP[idx]);
      float ds = p * (sScr[idx] - sD[ii]);
      sDS[idx] = __float2half(ds);
    }
    __syncthreads();

    // dV += P^T @ dO ; dK += dS^T @ Q
    #pragma unroll
    for (int kk = 0; kk < BM / 16; kk++) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> aP, aDS;
      wmma::load_matrix_sync(aP,  sP  + br * 16 + kk * 16 * BN, BN);
      wmma::load_matrix_sync(aDS, sDS + br * 16 + kk * 16 * BN, BN);
      #pragma unroll
      for (int nt = 0; nt < 4; nt++) {
        int n = bc * 64 + nt * 16;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bdO, bQ;
        wmma::load_matrix_sync(bdO, sdO + (kk * 16) * HD + n, HD);
        wmma::load_matrix_sync(bQ,  sQ  + (kk * 16) * HD + n, HD);
        wmma::mma_sync(accV[nt], aP, bdO, accV[nt]);
        wmma::mma_sync(accK[nt], aDS, bQ, accK[nt]);
      }
    }
    __syncthreads();
  }

  // store via staging in dead sK/sV region ([BN x HD] fp32)
  float* stage = reinterpret_cast<float*>(sK);
  #pragma unroll
  for (int nt = 0; nt < 4; nt++) {
    int n = bc * 64 + nt * 16;
    wmma::store_matrix_sync(stage + (br * 16) * HD + n, accV[nt], HD, wmma::mem_row_major);
  }
  __syncthreads();
  for (int idx = tid; idx < BN * HD; idx += blockDim.x) {
    int jj = idx / HD, e = idx % HD; int jg = j0 + jj;
    if (jg < S) dVb[(long)jg * HD + e] = __float2bfloat16(stage[idx]);
  }
  __syncthreads();
  #pragma unroll
  for (int nt = 0; nt < 4; nt++) {
    int n = bc * 64 + nt * 16;
    wmma::store_matrix_sync(stage + (br * 16) * HD + n, accK[nt], HD, wmma::mem_row_major);
  }
  __syncthreads();
  for (int idx = tid; idx < BN * HD; idx += blockDim.x) {
    int jj = idx / HD, e = idx % HD; int jg = j0 + jj;
    if (jg < S) dKb[(long)jg * HD + e] = __float2bfloat16(scale * stage[idx]);
  }
}

// ---------------- dQ kernel ----------------
__global__ __launch_bounds__(NTHREAD) void dq_kernel(
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
  half* sK  = (half*)smem;
  half* sV  = sK + BN * HD;
  half* sQ  = sV + BN * HD;
  half* sdO = sQ + BM * HD;
  half* sP  = sdO + BM * HD;
  half* sDS = sP + BM * BN;
  float* sScr = (float*)(sDS + BM * BN);
  float* sL   = sScr + BM * BN;
  float* sD   = sL + BM;

  int tid = threadIdx.x;
  int w = tid >> 5;
  int mr = w & 3;    // BM row tile (0..3)
  int mc = w >> 2;   // HD half (0..1)

  load_tile(sQ, Qb, i0, BM, S);
  load_tile(sdO, dOb, i0, BM, S);
  load_vec(sL, Lb, i0, BM, S);
  load_vec(sD, Db, i0, BM, S);

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> accQ[4];
  #pragma unroll
  for (int nt = 0; nt < 4; nt++) wmma::fill_fragment(accQ[nt], 0.f);
  __syncthreads();

  for (int bj = 0; bj <= bi; bj++) {
    int j0 = bj * BN;
    load_tile(sK, Kb, j0, BN, S);
    load_tile(sV, Vb, j0, BN, S);
    __syncthreads();

    qk_matmul(sQ, sK, sScr, w);   // S = Q @ K^T
    __syncthreads();

    for (int idx = tid; idx < BM * BN; idx += blockDim.x) {
      int ii = idx / BN, jj = idx % BN;
      int ig = i0 + ii, jg = j0 + jj;
      float p = 0.f;
      if (ig < S && jg < S && jg <= ig) p = __expf(scale * sScr[idx] - sL[ii]);
      sP[idx] = __float2half(p);
    }
    __syncthreads();

    qk_matmul(sdO, sV, sScr, w);  // dP = dO @ V^T
    __syncthreads();

    for (int idx = tid; idx < BM * BN; idx += blockDim.x) {
      int ii = idx / BN;
      float p = __half2float(sP[idx]);
      float ds = p * (sScr[idx] - sD[ii]);
      sDS[idx] = __float2half(ds);
    }
    __syncthreads();

    // dQ += dS @ K
    #pragma unroll
    for (int kk = 0; kk < BN / 16; kk++) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> aDS;
      wmma::load_matrix_sync(aDS, sDS + (mr * 16) * BN + kk * 16, BN);
      #pragma unroll
      for (int nt = 0; nt < 4; nt++) {
        int n = mc * 64 + nt * 16;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bK;
        wmma::load_matrix_sync(bK, sK + (kk * 16) * HD + n, HD);
        wmma::mma_sync(accQ[nt], aDS, bK, accQ[nt]);
      }
    }
    __syncthreads();
  }

  float* stage = reinterpret_cast<float*>(sK);
  #pragma unroll
  for (int nt = 0; nt < 4; nt++) {
    int n = mc * 64 + nt * 16;
    wmma::store_matrix_sync(stage + (mr * 16) * HD + n, accQ[nt], HD, wmma::mem_row_major);
  }
  __syncthreads();
  for (int idx = tid; idx < BM * HD; idx += blockDim.x) {
    int ii = idx / HD, e = idx % HD; int ig = i0 + ii;
    if (ig < S) dQb[(long)ig * HD + e] = __float2bfloat16(scale * stage[idx]);
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

  size_t smem = (size_t)(2 * BN * HD + 2 * BM * HD) * sizeof(half) +
                (size_t)(2 * BM * BN) * sizeof(half) +
                (size_t)(BM * BN) * sizeof(float) +
                (size_t)(2 * BM) * sizeof(float);

  CUDA_CHECK(cudaFuncSetAttribute(dkdv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  CUDA_CHECK(cudaFuncSetAttribute(dq_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));

  dim3 grid(BH, nBlocks);
  dkdv_kernel<<<grid, NTHREAD, smem, stream>>>(Qp, Kp, Vp, dOp, Lp, dD, dKp, dVp, (int)S, nBlocks, scale);
  CUDA_CHECK(cudaGetLastError());
  dq_kernel<<<grid, NTHREAD, smem, stream>>>(Qp, Kp, Vp, dOp, Lp, dD, dQp, (int)S, nBlocks, scale);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaFree(dD));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd