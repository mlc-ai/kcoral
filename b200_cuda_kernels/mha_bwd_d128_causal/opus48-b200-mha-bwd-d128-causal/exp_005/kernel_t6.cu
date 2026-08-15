#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <type_traits>
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

constexpr int BQ = 64;
constexpr int BK = 64;
constexpr int DH = 128;
constexpr int WARPS = 8;
constexpr int NTHREADS = WARPS * 32; // 256
constexpr int SMEM_BYTES = 115200;

typedef wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> AH_R;
typedef wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::col_major> AH_C;
typedef wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> BH_R;
typedef wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> BH_C;
typedef wmma::fragment<wmma::accumulator,16,16,16,float> FragC;

__device__ __forceinline__ half b2h(const __nv_bfloat16 x) {
  return __float2half(__bfloat162float(x));
}

// C[M,N] = A @ B  (results stored to smem row-major, ld=N), all warps cooperate.
template<bool AisCol, bool BisCol, int M, int N, int Kd>
__device__ __forceinline__ void gemm_to_smem(const half* A, int lda,
                                              const half* B, int ldb,
                                              float* C, int warp) {
  using FragA = typename std::conditional<AisCol, AH_C, AH_R>::type;
  using FragB = typename std::conditional<BisCol, BH_C, BH_R>::type;
  constexpr int MT = M/16, NT = N/16, KT = Kd/16;
  for (int t = warp; t < MT*NT; t += WARPS) {
    int tr = t / NT, tc = t % NT;
    FragC acc; wmma::fill_fragment(acc, 0.0f);
    #pragma unroll
    for (int kk = 0; kk < KT; kk++) {
      FragA a; FragB b;
      const half* aptr = AisCol ? (A + (kk*16)*lda + tr*16) : (A + (tr*16)*lda + kk*16);
      const half* bptr = BisCol ? (B + (kk*16) + (tc*16)*ldb) : (B + (kk*16)*ldb + tc*16);
      wmma::load_matrix_sync(a, aptr, lda);
      wmma::load_matrix_sync(b, bptr, ldb);
      wmma::mma_sync(acc, a, b, acc);
    }
    wmma::store_matrix_sync(C + tr*16*N + tc*16, acc, N, wmma::mem_row_major);
  }
}

// 8 rows per block (256 threads = 8 warps, one warp per row), coalesced vectorized reads
__global__ void delta_kernel(const __nv_bfloat16* O, const __nv_bfloat16* dO,
                             float* Delta, long total_rows) {
  int warp = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  long row = (long)blockIdx.x * 8 + warp;
  if (row >= total_rows) return;
  // each lane reads 4 consecutive bf16 (8 bytes) -> covers 128 elems
  const __nv_bfloat16* o = O + row*DH + lane*4;
  const __nv_bfloat16* g = dO + row*DH + lane*4;
  int2 vo = *reinterpret_cast<const int2*>(o);
  int2 vg = *reinterpret_cast<const int2*>(g);
  const __nv_bfloat162* po = reinterpret_cast<const __nv_bfloat162*>(&vo);
  const __nv_bfloat162* pg = reinterpret_cast<const __nv_bfloat162*>(&vg);
  float v = 0.f;
  #pragma unroll
  for (int i = 0; i < 2; i++) {
    float2 fo = __bfloat1622float2(po[i]);
    float2 fg = __bfloat1622float2(pg[i]);
    v += fo.x*fg.x + fo.y*fg.y;
  }
  #pragma unroll
  for (int o2 = 16; o2 > 0; o2 >>= 1) v += __shfl_down_sync(0xffffffffu, v, o2);
  if (lane == 0) Delta[row] = v;
}

// ---- Kernel 1: dK, dV (parallel over KV blocks) ----
__global__ void __launch_bounds__(256) bwd_kv_kernel(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* dO, const float* L, const float* Delta,
    __nv_bfloat16* dK, __nv_bfloat16* dV, int S, float scale) {
  int kb = blockIdx.x;
  int bh = blockIdx.y;
  int tid = threadIdx.x;
  int warp = tid >> 5;
  int nqb = (S + BQ - 1) / BQ;

  const long headoff = (long)bh * S * DH;
  const __nv_bfloat16* Qh = Q + headoff;
  const __nv_bfloat16* Kh = K + headoff;
  const __nv_bfloat16* Vh = V + headoff;
  const __nv_bfloat16* dOh = dO + headoff;
  const float* Lh = L + (long)bh * S;
  const float* Dh = Delta + (long)bh * S;
  __nv_bfloat16* dKh = dK + headoff;
  __nv_bfloat16* dVh = dV + headoff;

  extern __shared__ char smem[];
  half* sK  = (half*)(smem + 0);
  half* sV  = (half*)(smem + 16384);
  half* sQ  = (half*)(smem + 32768);
  half* sdO = (half*)(smem + 49152);
  float* sS  = (float*)(smem + 65536);
  float* sdP = (float*)(smem + 81920);
  half* sP  = (half*)(smem + 98304);
  half* sdS = (half*)(smem + 106496);
  float* sL  = (float*)(smem + 114688);
  float* sD  = (float*)(smem + 114944);
  float* sStage = (float*)(smem + 0);

  half zero = __float2half(0.0f);

  for (int idx = tid; idx < BK*DH; idx += NTHREADS) {
    int r = idx / DH, c = idx % DH;
    int gk = kb*BK + r;
    sK[idx] = (gk < S) ? b2h(Kh[(long)gk*DH + c]) : zero;
    sV[idx] = (gk < S) ? b2h(Vh[(long)gk*DH + c]) : zero;
  }

  FragC dV_acc[4], dK_acc[4];
  #pragma unroll
  for (int i = 0; i < 4; i++) { wmma::fill_fragment(dV_acc[i], 0.f); wmma::fill_fragment(dK_acc[i], 0.f); }
  __syncthreads();

  for (int qb = kb; qb < nqb; qb++) {
    for (int idx = tid; idx < BQ*DH; idx += NTHREADS) {
      int r = idx / DH, c = idx % DH;
      int gq = qb*BQ + r;
      sQ[idx]  = (gq < S) ? b2h(Qh[(long)gq*DH + c]) : zero;
      sdO[idx] = (gq < S) ? b2h(dOh[(long)gq*DH + c]) : zero;
    }
    for (int r = tid; r < BQ; r += NTHREADS) {
      int gq = qb*BQ + r;
      sL[r] = (gq < S) ? Lh[gq] : 0.f;
      sD[r] = (gq < S) ? Dh[gq] : 0.f;
    }
    __syncthreads();

    gemm_to_smem<false,true,BQ,BK,DH>(sQ, DH, sK, DH, sS, warp);
    gemm_to_smem<false,true,BQ,BK,DH>(sdO, DH, sV, DH, sdP, warp);
    __syncthreads();

    for (int idx = tid; idx < BQ*BK; idx += NTHREADS) {
      int r = idx >> 6, c = idx & 63;
      int gq = qb*BQ + r, gk = kb*BK + c;
      float p = 0.f, ds = 0.f;
      if (gq < S && gk < S && gk <= gq) {
        p = __expf(sS[idx]*scale - sL[r]);
        ds = p * (sdP[idx] - sD[r]);
      }
      sP[idx]  = __float2half(p);
      sdS[idx] = __float2half(ds);
    }
    __syncthreads();

    #pragma unroll
    for (int li = 0; li < 4; li++) {
      int t = warp + li*WARPS;
      int tr = t / (DH/16);
      int tc = t % (DH/16);
      #pragma unroll
      for (int kk = 0; kk < BQ/16; kk++) {
        AH_C aP, aS; BH_R bO, bQ;
        wmma::load_matrix_sync(aP, sP  + tr*16 + kk*16*BK, BK);
        wmma::load_matrix_sync(bO, sdO + kk*16*DH + tc*16, DH);
        wmma::mma_sync(dV_acc[li], aP, bO, dV_acc[li]);
        wmma::load_matrix_sync(aS, sdS + tr*16 + kk*16*BK, BK);
        wmma::load_matrix_sync(bQ, sQ  + kk*16*DH + tc*16, DH);
        wmma::mma_sync(dK_acc[li], aS, bQ, dK_acc[li]);
      }
    }
    __syncthreads();
  }

  // write dV
  #pragma unroll
  for (int li = 0; li < 4; li++) {
    int t = warp + li*WARPS; int tr = t/(DH/16); int tc = t%(DH/16);
    wmma::store_matrix_sync(sStage + tr*16*DH + tc*16, dV_acc[li], DH, wmma::mem_row_major);
  }
  __syncthreads();
  for (int idx = tid; idx < BK*DH; idx += NTHREADS) {
    int r = idx / DH, c = idx % DH; int gk = kb*BK + r;
    if (gk < S) dVh[(long)gk*DH + c] = __float2bfloat16(sStage[idx]);
  }
  __syncthreads();
  // write dK (scaled)
  #pragma unroll
  for (int li = 0; li < 4; li++) {
    int t = warp + li*WARPS; int tr = t/(DH/16); int tc = t%(DH/16);
    wmma::store_matrix_sync(sStage + tr*16*DH + tc*16, dK_acc[li], DH, wmma::mem_row_major);
  }
  __syncthreads();
  for (int idx = tid; idx < BK*DH; idx += NTHREADS) {
    int r = idx / DH, c = idx % DH; int gk = kb*BK + r;
    if (gk < S) dKh[(long)gk*DH + c] = __float2bfloat16(sStage[idx]*scale);
  }
}

// ---- Kernel 2: dQ (parallel over Q blocks) ----
__global__ void __launch_bounds__(256) bwd_q_kernel(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* dO, const float* L, const float* Delta,
    __nv_bfloat16* dQ, int S, float scale) {
  int qb = blockIdx.x;
  int bh = blockIdx.y;
  int tid = threadIdx.x;
  int warp = tid >> 5;

  const long headoff = (long)bh * S * DH;
  const __nv_bfloat16* Qh = Q + headoff;
  const __nv_bfloat16* Kh = K + headoff;
  const __nv_bfloat16* Vh = V + headoff;
  const __nv_bfloat16* dOh = dO + headoff;
  const float* Lh = L + (long)bh * S;
  const float* Dh = Delta + (long)bh * S;
  __nv_bfloat16* dQh = dQ + headoff;

  extern __shared__ char smem[];
  half* sK  = (half*)(smem + 0);
  half* sV  = (half*)(smem + 16384);
  half* sQ  = (half*)(smem + 32768);
  half* sdO = (half*)(smem + 49152);
  float* sS  = (float*)(smem + 65536);
  float* sdP = (float*)(smem + 81920);
  half* sdS = (half*)(smem + 98304);
  float* sL  = (float*)(smem + 114688);
  float* sD  = (float*)(smem + 114944);
  float* sStage = (float*)(smem + 0);

  half zero = __float2half(0.0f);

  for (int idx = tid; idx < BQ*DH; idx += NTHREADS) {
    int r = idx / DH, c = idx % DH; int gq = qb*BQ + r;
    sQ[idx]  = (gq < S) ? b2h(Qh[(long)gq*DH + c]) : zero;
    sdO[idx] = (gq < S) ? b2h(dOh[(long)gq*DH + c]) : zero;
  }
  for (int r = tid; r < BQ; r += NTHREADS) {
    int gq = qb*BQ + r;
    sL[r] = (gq < S) ? Lh[gq] : 0.f;
    sD[r] = (gq < S) ? Dh[gq] : 0.f;
  }

  FragC dQ_acc[4];
  #pragma unroll
  for (int i = 0; i < 4; i++) wmma::fill_fragment(dQ_acc[i], 0.f);
  __syncthreads();

  for (int kb = 0; kb <= qb; kb++) {
    for (int idx = tid; idx < BK*DH; idx += NTHREADS) {
      int r = idx / DH, c = idx % DH; int gk = kb*BK + r;
      sK[idx] = (gk < S) ? b2h(Kh[(long)gk*DH + c]) : zero;
      sV[idx] = (gk < S) ? b2h(Vh[(long)gk*DH + c]) : zero;
    }
    __syncthreads();

    gemm_to_smem<false,true,BQ,BK,DH>(sQ, DH, sK, DH, sS, warp);
    gemm_to_smem<false,true,BQ,BK,DH>(sdO, DH, sV, DH, sdP, warp);
    __syncthreads();

    for (int idx = tid; idx < BQ*BK; idx += NTHREADS) {
      int r = idx >> 6, c = idx & 63;
      int gq = qb*BQ + r, gk = kb*BK + c;
      float ds = 0.f;
      if (gq < S && gk < S && gk <= gq) {
        float p = __expf(sS[idx]*scale - sL[r]);
        ds = p * (sdP[idx] - sD[r]);
      }
      sdS[idx] = __float2half(ds);
    }
    __syncthreads();

    #pragma unroll
    for (int li = 0; li < 4; li++) {
      int t = warp + li*WARPS;
      int tr = t / (DH/16);
      int tc = t % (DH/16);
      #pragma unroll
      for (int kk = 0; kk < BK/16; kk++) {
        AH_R aS; BH_R bK;
        wmma::load_matrix_sync(aS, sdS + tr*16*BK + kk*16, BK);
        wmma::load_matrix_sync(bK, sK  + kk*16*DH + tc*16, DH);
        wmma::mma_sync(dQ_acc[li], aS, bK, dQ_acc[li]);
      }
    }
    __syncthreads();
  }

  #pragma unroll
  for (int li = 0; li < 4; li++) {
    int t = warp + li*WARPS; int tr = t/(DH/16); int tc = t%(DH/16);
    wmma::store_matrix_sync(sStage + tr*16*DH + tc*16, dQ_acc[li], DH, wmma::mem_row_major);
  }
  __syncthreads();
  for (int idx = tid; idx < BQ*DH; idx += NTHREADS) {
    int r = idx / DH, c = idx % DH; int gq = qb*BQ + r;
    if (gq < S) dQh[(long)gq*DH + c] = __float2bfloat16(sStage[idx]*scale);
  }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int64_t B = Q.size(0), H = Q.size(1), S = Q.size(2), Dd = Q.size(3);
  int64_t BH = B * H;
  (void)Dd;

  const __nv_bfloat16* Qp  = static_cast<const __nv_bfloat16*>(Q.data_ptr());
  const __nv_bfloat16* Kp  = static_cast<const __nv_bfloat16*>(K.data_ptr());
  const __nv_bfloat16* Vp  = static_cast<const __nv_bfloat16*>(V.data_ptr());
  const __nv_bfloat16* Op  = static_cast<const __nv_bfloat16*>(O.data_ptr());
  const __nv_bfloat16* dOp = static_cast<const __nv_bfloat16*>(dO.data_ptr());
  const float* Lp = static_cast<const float*>(L.data_ptr());
  __nv_bfloat16* dQp = static_cast<__nv_bfloat16*>(dQ.data_ptr());
  __nv_bfloat16* dKp = static_cast<__nv_bfloat16*>(dK.data_ptr());
  __nv_bfloat16* dVp = static_cast<__nv_bfloat16*>(dV.data_ptr());

  float scale = 1.0f / sqrtf((float)DH);
  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

  float* Delta = nullptr;
  CUDA_CHECK(cudaMallocAsync(&Delta, sizeof(float) * BH * S, stream));

  long total_rows = (long)BH * S;
  int dblocks = (int)((total_rows + 7) / 8);
  delta_kernel<<<(unsigned)dblocks, 256, 0, stream>>>(Op, dOp, Delta, total_rows);
  CUDA_CHECK(cudaGetLastError());

  int nkb = (int)((S + BK - 1) / BK);
  int nqb = (int)((S + BQ - 1) / BQ);

  cudaFuncSetAttribute(bwd_kv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
  cudaFuncSetAttribute(bwd_q_kernel,  cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);

  dim3 g1((unsigned)nkb, (unsigned)BH);
  bwd_kv_kernel<<<g1, NTHREADS, SMEM_BYTES, stream>>>(
      Qp, Kp, Vp, dOp, Lp, Delta, dKp, dVp, (int)S, scale);
  CUDA_CHECK(cudaGetLastError());

  dim3 g2((unsigned)nqb, (unsigned)BH);
  bwd_q_kernel<<<g2, NTHREADS, SMEM_BYTES, stream>>>(
      Qp, Kp, Vp, dOp, Lp, Delta, dQp, (int)S, scale);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaFreeAsync(Delta, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd