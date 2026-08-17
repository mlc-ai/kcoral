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
#define LDT 136
#define LDP 72
#define LDC 72

// cached scratch
static void*  g_ds = nullptr; static size_t g_ds_sz = 0;
static float* g_dd = nullptr; static size_t g_dd_sz = 0;

// ---------------- delta kernel ----------------
__global__ void delta_kernel(const __nv_bfloat16* __restrict__ dO,
                             const __nv_bfloat16* __restrict__ O,
                             float* __restrict__ D, int total) {
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

__device__ __forceinline__ void load_tile(half* dst, const __nv_bfloat16* src,
                                           int row0, int nrows, int S) {
  const int perRow = HD / 8;
  int total = nrows * perRow;
  for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
    int r = idx / perRow, c = idx % perRow;
    int gr = row0 + r;
    half* pd = dst + r * LDT + c * 8;
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

__device__ __forceinline__ void qk_matmul(const half* sA, const half* sB, float* sC, int w) {
  int wr = w & 3, wc = w >> 2;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bf;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf[2];
  #pragma unroll
  for (int nn = 0; nn < 2; nn++) wmma::fill_fragment(cf[nn], 0.f);
  #pragma unroll
  for (int k = 0; k < HD / 16; k++) {
    wmma::load_matrix_sync(af, sA + (wr * 16) * LDT + k * 16, LDT);
    #pragma unroll
    for (int nn = 0; nn < 2; nn++) {
      int n = wc * 2 + nn;
      wmma::load_matrix_sync(bf, sB + (n * 16) * LDT + k * 16, LDT);
      wmma::mma_sync(cf[nn], af, bf, cf[nn]);
    }
  }
  #pragma unroll
  for (int nn = 0; nn < 2; nn++) {
    int n = wc * 2 + nn;
    wmma::store_matrix_sync(sC + (wr * 16) * LDC + n * 16, cf[nn], LDC, wmma::mem_row_major);
  }
}

// ---------------- dK / dV kernel (also writes dS scratch) ----------------
__global__ __launch_bounds__(NTHREAD, 2) void dkdv_kernel(
    const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V, const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L, const float* __restrict__ D,
    __nv_bfloat16* __restrict__ dK, __nv_bfloat16* __restrict__ dV,
    half* __restrict__ dSg, int S, int nBlocks, float scale) {
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
  long dsBase_bh = (long)bh * nBlocks * nBlocks * BM * BN;

  extern __shared__ char smem[];
  half* sK  = (half*)smem;
  half* sV  = sK + BN * LDT;
  half* sQ  = sV + BN * LDT;
  half* sdO = sQ + BM * LDT;
  half* sP  = sdO + BM * LDT;
  half* sDS = sP + BM * LDP;
  float* sScr = (float*)(sDS + BM * LDP);
  float* sL   = sScr + BM * LDC;
  float* sD   = sL + BM;

  int tid = threadIdx.x;
  int w = tid >> 5;
  int lane = tid & 31;
  int wr = w & 3, wc = w >> 2;
  int br = w & 3, bc = w >> 2;

  load_tile(sK, Kb, j0, BN, S);
  load_tile(sV, Vb, j0, BN, S);

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> accV[4], accK[4];
  #pragma unroll
  for (int nt = 0; nt < 4; nt++) { wmma::fill_fragment(accV[nt], 0.f); wmma::fill_fragment(accK[nt], 0.f); }
  __syncthreads();

  int lastBlock = (S + BM - 1) / BM;
  for (int bi = bj; bi < lastBlock; bi++) {
    int qi0 = bi * BM;
    bool edge = (qi0 + BM > S) || (bi == bj);
    load_tile(sQ, Qb, qi0, BM, S);
    load_tile(sdO, dOb, qi0, BM, S);
    load_vec(sL, Lb, qi0, BM, S);
    load_vec(sD, Db, qi0, BM, S);
    __syncthreads();

    qk_matmul(sQ, sK, sScr, w);
    __syncwarp();
    if (edge) {
      for (int e = lane; e < 16 * 32; e += 32) {
        int ii = wr * 16 + (e >> 5), jj = wc * 32 + (e & 31);
        int ig = qi0 + ii, jg = j0 + jj;
        float p = 0.f;
        if (ig < S && jg < S && jg <= ig) p = __expf(scale * sScr[ii * LDC + jj] - sL[ii]);
        sP[ii * LDP + jj] = __float2half(p);
      }
    } else {
      for (int e = lane; e < 16 * 32; e += 32) {
        int ii = wr * 16 + (e >> 5), jj = wc * 32 + (e & 31);
        float p = __expf(scale * sScr[ii * LDC + jj] - sL[ii]);
        sP[ii * LDP + jj] = __float2half(p);
      }
    }
    __syncwarp();

    qk_matmul(sdO, sV, sScr, w);
    __syncwarp();
    for (int e = lane; e < 16 * 32; e += 32) {
      int ii = wr * 16 + (e >> 5), jj = wc * 32 + (e & 31);
      float p = __half2float(sP[ii * LDP + jj]);
      float ds = p * (sScr[ii * LDC + jj] - sD[ii]);
      sDS[ii * LDP + jj] = __float2half(ds);
    }
    __syncwarp();
    __syncthreads();

    // write dS scratch (contiguous [BM][BN])
    long dsBase = dsBase_bh + ((long)bi * nBlocks + bj) * BM * BN;
    for (int pos = tid; pos < BM * BN; pos += blockDim.x) {
      int ii = pos / BN, jj = pos % BN;
      dSg[dsBase + ii * BN + jj] = sDS[ii * LDP + jj];
    }

    // dV += P^T @ dO ; dK += dS^T @ Q
    #pragma unroll
    for (int kk = 0; kk < BM / 16; kk++) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> aP, aDS;
      wmma::load_matrix_sync(aP,  sP  + kk * 16 * LDP + br * 16, LDP);
      wmma::load_matrix_sync(aDS, sDS + kk * 16 * LDP + br * 16, LDP);
      #pragma unroll
      for (int nt = 0; nt < 4; nt++) {
        int n = bc * 64 + nt * 16;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bdO, bQ;
        wmma::load_matrix_sync(bdO, sdO + (kk * 16) * LDT + n, LDT);
        wmma::load_matrix_sync(bQ,  sQ  + (kk * 16) * LDT + n, LDT);
        wmma::mma_sync(accV[nt], aP, bdO, accV[nt]);
        wmma::mma_sync(accK[nt], aDS, bQ, accK[nt]);
      }
    }
    __syncthreads();
  }

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

// ---------------- dQ kernel (reads precomputed dS) ----------------
__global__ __launch_bounds__(NTHREAD, 3) void dq_kernel(
    const __nv_bfloat16* __restrict__ K, const half* __restrict__ dSg,
    __nv_bfloat16* __restrict__ dQ, int S, int nBlocks, float scale) {
  int bh = blockIdx.x, bi = blockIdx.y;
  int i0 = bi * BM;
  if (i0 >= S) return;

  const __nv_bfloat16* Kb = K + (long)bh * S * HD;
  __nv_bfloat16* dQb = dQ + (long)bh * S * HD;
  long dsBase_bh = (long)bh * nBlocks * nBlocks * BM * BN;

  extern __shared__ char smem[];
  half* sK  = (half*)smem;
  half* sDS = sK + BN * LDT;
  float* stage = (float*)(sDS + BM * LDP);

  int tid = threadIdx.x;
  int w = tid >> 5;
  int mr = w & 3, mc = w >> 2;

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> accQ[4];
  #pragma unroll
  for (int nt = 0; nt < 4; nt++) wmma::fill_fragment(accQ[nt], 0.f);

  for (int bj = 0; bj <= bi; bj++) {
    int j0 = bj * BN;
    load_tile(sK, Kb, j0, BN, S);
    long dsBase = dsBase_bh + ((long)bi * nBlocks + bj) * BM * BN;
    for (int pos = tid; pos < BM * BN; pos += blockDim.x) {
      int ii = pos / BN, jj = pos % BN;
      sDS[ii * LDP + jj] = dSg[dsBase + ii * BN + jj];
    }
    __syncthreads();

    #pragma unroll
    for (int kk = 0; kk < BN / 16; kk++) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> aDS;
      wmma::load_matrix_sync(aDS, sDS + (mr * 16) * LDP + kk * 16, LDP);
      #pragma unroll
      for (int nt = 0; nt < 4; nt++) {
        int n = mc * 64 + nt * 16;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bK;
        wmma::load_matrix_sync(bK, sK + (kk * 16) * LDT + n, LDT);
        wmma::mma_sync(accQ[nt], aDS, bK, accQ[nt]);
      }
    }
    __syncthreads();
  }

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

  int nBlocks = ((int)S + BN - 1) / BN;
  float scale = 1.0f / sqrtf((float)d);

  size_t dd_sz = sizeof(float) * (size_t)BH * (size_t)S;
  if (dd_sz > g_dd_sz) {
    if (g_dd) CUDA_CHECK(cudaFree(g_dd));
    CUDA_CHECK(cudaMalloc(&g_dd, dd_sz));
    g_dd_sz = dd_sz;
  }
  float* dD = g_dd;

  size_t ds_sz = (size_t)BH * nBlocks * nBlocks * BM * BN * sizeof(half);
  if (ds_sz > g_ds_sz) {
    if (g_ds) CUDA_CHECK(cudaFree(g_ds));
    CUDA_CHECK(cudaMalloc(&g_ds, ds_sz));
    g_ds_sz = ds_sz;
  }
  half* dSg = (half*)g_ds;

  int total = BH * (int)S;
  int tpb = 256;
  delta_kernel<<<(total + tpb - 1) / tpb, tpb, 0, stream>>>(dOp, Op, dD, total);
  CUDA_CHECK(cudaGetLastError());

  size_t smem_dkdv = (size_t)(2 * BN * LDT + 2 * BM * LDT) * sizeof(half) +
                     (size_t)(2 * BM * LDP) * sizeof(half) +
                     (size_t)(BM * LDC) * sizeof(float) +
                     (size_t)(2 * BM) * sizeof(float);
  size_t smem_dq = (size_t)(BN * LDT + BM * LDP) * sizeof(half) +
                   (size_t)(BM * HD) * sizeof(float);

  CUDA_CHECK(cudaFuncSetAttribute(dkdv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_dkdv));
  CUDA_CHECK(cudaFuncSetAttribute(dq_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_dq));

  dim3 grid(BH, nBlocks);
  dkdv_kernel<<<grid, NTHREAD, smem_dkdv, stream>>>(Qp, Kp, Vp, dOp, Lp, dD, dKp, dVp, dSg, (int)S, nBlocks, scale);
  CUDA_CHECK(cudaGetLastError());
  dq_kernel<<<grid, NTHREAD, smem_dq, stream>>>(Kp, dSg, dQp, (int)S, nBlocks, scale);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd