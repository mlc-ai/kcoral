#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
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

constexpr int DH = 128;
constexpr int BM = 128;   // query tile
constexpr int BN = 64;    // kv tile
constexpr int THREADS = 256;
constexpr int NWARPS = THREADS / 32;   // 8
constexpr int MT = BM / 16;            // 8
constexpr int NT = BN / 16;            // 4
constexpr int KT = DH / 16;            // 8
constexpr int NLOC = 4;                // nt-tiles per warp for dV/dK
constexpr int TfLD = BN + 8;           // padded fp32 stride

constexpr int SMEM_BF16 = (BM*DH + BN*DH + BN*DH + BM*DH + BM*BN + BM*BN);
constexpr int SMEM_F32  = (BM*TfLD + NWARPS*256 + BM + BM);
constexpr int SMEM_BYTES = SMEM_BF16*2 + SMEM_F32*4;

__device__ __forceinline__ void load_tile(__nv_bfloat16* smem,
                                           const __nv_bfloat16* gbase,
                                           int r0, int S, int ROWS) {
  const int VECB = 8;
  const int chunksPerRow = DH / VECB;
  int total = ROWS * chunksPerRow;
  for (int i = threadIdx.x; i < total; i += blockDim.x) {
    int row = i / chunksPerRow;
    int chunk = i % chunksPerRow;
    int4* dst = reinterpret_cast<int4*>(smem + row * DH + chunk * VECB);
    int gs = r0 + row;
    if (gs < S) {
      const int4* src = reinterpret_cast<const int4*>(gbase + (int64_t)gs * DH + chunk * VECB);
      *dst = __ldg(src);
    } else {
      *dst = make_int4(0, 0, 0, 0);
    }
  }
}

__global__ void compute_D_kernel(const __nv_bfloat16* O, const __nv_bfloat16* dOg,
                                 float* Dg, int64_t nrows) {
  int64_t row = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= nrows) return;
  const int4* o4 = reinterpret_cast<const int4*>(O + row * DH);
  const int4* g4 = reinterpret_cast<const int4*>(dOg + row * DH);
  float acc = 0.f;
  #pragma unroll
  for (int c = 0; c < DH / 8; c++) {
    int4 ov = __ldg(o4 + c);
    int4 gv = __ldg(g4 + c);
    const __nv_bfloat16* op = reinterpret_cast<const __nv_bfloat16*>(&ov);
    const __nv_bfloat16* gp = reinterpret_cast<const __nv_bfloat16*>(&gv);
    #pragma unroll
    for (int j = 0; j < 8; j++)
      acc += __bfloat162float(op[j]) * __bfloat162float(gp[j]);
  }
  Dg[row] = acc;
}

__global__ void convert_kernel(const float* dQf, __nv_bfloat16* dQ, int64_t n) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dQ[i] = __float2bfloat16(dQf[i]);
}

__global__ void __launch_bounds__(THREADS) attn_bwd_kernel(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* dOg, const float* Lg, const float* Dg,
    float* dQf, __nv_bfloat16* dK, __nv_bfloat16* dV, int S) {

  extern __shared__ char smem_raw[];
  __nv_bfloat16* Qs  = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  __nv_bfloat16* Ks  = Qs  + BM * DH;
  __nv_bfloat16* Vs  = Ks  + BN * DH;
  __nv_bfloat16* dOs = Vs  + BN * DH;
  __nv_bfloat16* Ps  = dOs + BM * DH;
  __nv_bfloat16* dSs = Ps  + BM * BN;
  float* Tf    = reinterpret_cast<float*>(dSs + BM * BN);
  float* stg   = Tf + BM * TfLD;
  float* Lsh   = stg + NWARPS * 256;
  float* Dsh   = Lsh + BM;

  int bh = blockIdx.z * gridDim.y + blockIdx.y;
  int n0 = blockIdx.x * BN;
  if (n0 >= S) return;

  int64_t head_off = (int64_t)bh * S * DH;
  const __nv_bfloat16* Qh  = Q   + head_off;
  const __nv_bfloat16* Kh  = K   + head_off;
  const __nv_bfloat16* Vh  = V   + head_off;
  const __nv_bfloat16* dOh = dOg + head_off;
  const float* Lh = Lg + (int64_t)bh * S;
  const float* Dh = Dg + (int64_t)bh * S;

  const float scale = 1.0f / sqrtf((float)DH);
  int warp_id = threadIdx.x / 32;
  int lane = threadIdx.x % 32;

  load_tile(Ks, Kh, n0, S, BN);
  load_tile(Vs, Vh, n0, S, BN);

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> dv_acc[NLOC];
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> dk_acc[NLOC];
  #pragma unroll
  for (int li = 0; li < NLOC; li++) {
    wmma::fill_fragment(dv_acc[li], 0.0f);
    wmma::fill_fragment(dk_acc[li], 0.0f);
  }

  int numQ = (S + BM - 1) / BM;
  for (int qb = 0; qb < numQ; qb++) {
    int m0 = qb * BM;
    load_tile(Qs, Qh, m0, S, BM);
    load_tile(dOs, dOh, m0, S, BM);
    for (int i = threadIdx.x; i < BM; i += blockDim.x) {
      int s = m0 + i;
      Lsh[i] = (s < S) ? Lh[s] : 0.0f;
      Dsh[i] = (s < S) ? Dh[s] : 0.0f;
    }
    __syncthreads();

    // GEMM1: S = Q @ K^T -> Tf  (each warp: nt fixed, 4 mt accumulators, reuse B)
    {
      int nt = warp_id % NT;
      int mt_base = (warp_id / NT) * 4;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> c0, c1, c2, c3;
      wmma::fill_fragment(c0, 0.0f); wmma::fill_fragment(c1, 0.0f);
      wmma::fill_fragment(c2, 0.0f); wmma::fill_fragment(c3, 0.0f);
      #pragma unroll
      for (int kt = 0; kt < KT; kt++) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
        wmma::load_matrix_sync(b, Ks + nt * 16 * DH + kt * 16, DH);
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a0, a1, a2, a3;
        wmma::load_matrix_sync(a0, Qs + (mt_base+0) * 16 * DH + kt * 16, DH);
        wmma::load_matrix_sync(a1, Qs + (mt_base+1) * 16 * DH + kt * 16, DH);
        wmma::load_matrix_sync(a2, Qs + (mt_base+2) * 16 * DH + kt * 16, DH);
        wmma::load_matrix_sync(a3, Qs + (mt_base+3) * 16 * DH + kt * 16, DH);
        wmma::mma_sync(c0, a0, b, c0);
        wmma::mma_sync(c1, a1, b, c1);
        wmma::mma_sync(c2, a2, b, c2);
        wmma::mma_sync(c3, a3, b, c3);
      }
      wmma::store_matrix_sync(Tf + (mt_base+0) * 16 * TfLD + nt * 16, c0, TfLD, wmma::mem_row_major);
      wmma::store_matrix_sync(Tf + (mt_base+1) * 16 * TfLD + nt * 16, c1, TfLD, wmma::mem_row_major);
      wmma::store_matrix_sync(Tf + (mt_base+2) * 16 * TfLD + nt * 16, c2, TfLD, wmma::mem_row_major);
      wmma::store_matrix_sync(Tf + (mt_base+3) * 16 * TfLD + nt * 16, c3, TfLD, wmma::mem_row_major);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < BM * BN; i += blockDim.x) {
      int m = i / BN, n = i % BN;
      int gn = n0 + n;
      float p = (gn < S) ? __expf(scale * Tf[m * TfLD + n] - Lsh[m]) : 0.0f;
      Ps[m * BN + n] = __float2bfloat16(p);
    }
    __syncthreads();

    // GEMM2: dP = dO @ V^T -> Tf
    {
      int nt = warp_id % NT;
      int mt_base = (warp_id / NT) * 4;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> c0, c1, c2, c3;
      wmma::fill_fragment(c0, 0.0f); wmma::fill_fragment(c1, 0.0f);
      wmma::fill_fragment(c2, 0.0f); wmma::fill_fragment(c3, 0.0f);
      #pragma unroll
      for (int kt = 0; kt < KT; kt++) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
        wmma::load_matrix_sync(b, Vs + nt * 16 * DH + kt * 16, DH);
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a0, a1, a2, a3;
        wmma::load_matrix_sync(a0, dOs + (mt_base+0) * 16 * DH + kt * 16, DH);
        wmma::load_matrix_sync(a1, dOs + (mt_base+1) * 16 * DH + kt * 16, DH);
        wmma::load_matrix_sync(a2, dOs + (mt_base+2) * 16 * DH + kt * 16, DH);
        wmma::load_matrix_sync(a3, dOs + (mt_base+3) * 16 * DH + kt * 16, DH);
        wmma::mma_sync(c0, a0, b, c0);
        wmma::mma_sync(c1, a1, b, c1);
        wmma::mma_sync(c2, a2, b, c2);
        wmma::mma_sync(c3, a3, b, c3);
      }
      wmma::store_matrix_sync(Tf + (mt_base+0) * 16 * TfLD + nt * 16, c0, TfLD, wmma::mem_row_major);
      wmma::store_matrix_sync(Tf + (mt_base+1) * 16 * TfLD + nt * 16, c1, TfLD, wmma::mem_row_major);
      wmma::store_matrix_sync(Tf + (mt_base+2) * 16 * TfLD + nt * 16, c2, TfLD, wmma::mem_row_major);
      wmma::store_matrix_sync(Tf + (mt_base+3) * 16 * TfLD + nt * 16, c3, TfLD, wmma::mem_row_major);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < BM * BN; i += blockDim.x) {
      int m = i / BN, n = i % BN;
      float dp = Tf[m * TfLD + n];
      float p = __bfloat162float(Ps[m * BN + n]);
      float ds = scale * p * (dp - Dsh[m]);
      dSs[m * BN + n] = __float2bfloat16(ds);
    }
    __syncthreads();

    // GEMM3: dV += P^T @ dO ; GEMM4: dK += dS^T @ Q
    {
      int dt = warp_id;  // k-tile 0..7
      #pragma unroll
      for (int mt = 0; mt < MT; mt++) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bdo, bq;
        wmma::load_matrix_sync(bdo, dOs + mt * 16 * DH + dt * 16, DH);
        wmma::load_matrix_sync(bq,  Qs  + mt * 16 * DH + dt * 16, DH);
        #pragma unroll
        for (int nt = 0; nt < NLOC; nt++) {
          wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> ap, ads;
          wmma::load_matrix_sync(ap,  Ps  + mt * 16 * BN + nt * 16, BN);
          wmma::load_matrix_sync(ads, dSs + mt * 16 * BN + nt * 16, BN);
          wmma::mma_sync(dv_acc[nt], ap,  bdo, dv_acc[nt]);
          wmma::mma_sync(dk_acc[nt], ads, bq,  dk_acc[nt]);
        }
      }
    }

    // GEMM5: dQ = dS @ K -> atomic accumulate. dt=warp_id (k-tile), 2 groups of 4 mt.
    {
      int dt = warp_id;
      #pragma unroll
      for (int grp = 0; grp < 2; grp++) {
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> q0, q1, q2, q3;
        wmma::fill_fragment(q0, 0.0f); wmma::fill_fragment(q1, 0.0f);
        wmma::fill_fragment(q2, 0.0f); wmma::fill_fragment(q3, 0.0f);
        #pragma unroll
        for (int nt = 0; nt < NT; nt++) {
          wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b;
          wmma::load_matrix_sync(b, Ks + nt * 16 * DH + dt * 16, DH);
          wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a0, a1, a2, a3;
          wmma::load_matrix_sync(a0, dSs + (grp*4+0) * 16 * BN + nt * 16, BN);
          wmma::load_matrix_sync(a1, dSs + (grp*4+1) * 16 * BN + nt * 16, BN);
          wmma::load_matrix_sync(a2, dSs + (grp*4+2) * 16 * BN + nt * 16, BN);
          wmma::load_matrix_sync(a3, dSs + (grp*4+3) * 16 * BN + nt * 16, BN);
          wmma::mma_sync(q0, a0, b, q0);
          wmma::mma_sync(q1, a1, b, q1);
          wmma::mma_sync(q2, a2, b, q2);
          wmma::mma_sync(q3, a3, b, q3);
        }
        wmma::fragment<wmma::accumulator, 16, 16, 16, float>* qs[4] = {&q0, &q1, &q2, &q3};
        #pragma unroll
        for (int j = 0; j < 4; j++) {
          int mt = grp * 4 + j;
          wmma::store_matrix_sync(stg + warp_id * 256, *qs[j], 16, wmma::mem_row_major);
          __syncwarp();
          for (int e = lane; e < 256; e += 32) {
            int r = e / 16, cc = e % 16;
            int gm = m0 + mt * 16 + r;
            int gc = dt * 16 + cc;
            if (gm < S)
              atomicAdd(&dQf[head_off + (int64_t)gm * DH + gc], stg[warp_id * 256 + e]);
          }
          __syncwarp();
        }
      }
    }
    __syncthreads();
  }

  // Store dV, dK
  {
    int dt = warp_id;
    #pragma unroll
    for (int nt = 0; nt < NLOC; nt++) {
      wmma::store_matrix_sync(stg + warp_id * 256, dv_acc[nt], 16, wmma::mem_row_major);
      __syncwarp();
      for (int e = lane; e < 256; e += 32) {
        int r = e / 16, cc = e % 16;
        int gn = n0 + nt * 16 + r;
        int gc = dt * 16 + cc;
        if (gn < S)
          dV[head_off + (int64_t)gn * DH + gc] = __float2bfloat16(stg[warp_id * 256 + e]);
      }
      __syncwarp();
      wmma::store_matrix_sync(stg + warp_id * 256, dk_acc[nt], 16, wmma::mem_row_major);
      __syncwarp();
      for (int e = lane; e < 256; e += 32) {
        int r = e / 16, cc = e % 16;
        int gn = n0 + nt * 16 + r;
        int gc = dt * 16 + cc;
        if (gn < S)
          dK[head_off + (int64_t)gn * DH + gc] = __float2bfloat16(stg[warp_id * 256 + e]);
      }
      __syncwarp();
    }
  }
}

static void* g_dQf = nullptr;   static size_t g_dQf_bytes = 0;
static void* g_Dbuf = nullptr;  static size_t g_Dbuf_bytes = 0;
static bool g_attr_set = false;

static void ensure_buf(void** ptr, size_t* cur, size_t need) {
  if (need > *cur) {
    if (*ptr) cudaFree(*ptr);
    CUDA_CHECK(cudaMalloc(ptr, need));
    *cur = need;
  }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));

  int64_t B = Q.size(0), H = Q.size(1), S = Q.size(2);
  int64_t BH = B * H;

  const __nv_bfloat16* Qp  = static_cast<const __nv_bfloat16*>(Q.data_ptr());
  const __nv_bfloat16* Kp  = static_cast<const __nv_bfloat16*>(K.data_ptr());
  const __nv_bfloat16* Vp  = static_cast<const __nv_bfloat16*>(V.data_ptr());
  const __nv_bfloat16* Op  = static_cast<const __nv_bfloat16*>(O.data_ptr());
  const __nv_bfloat16* dOp = static_cast<const __nv_bfloat16*>(dO.data_ptr());
  const float* Lp = static_cast<const float*>(L.data_ptr());
  __nv_bfloat16* dQp = static_cast<__nv_bfloat16*>(dQ.data_ptr());
  __nv_bfloat16* dKp = static_cast<__nv_bfloat16*>(dK.data_ptr());
  __nv_bfloat16* dVp = static_cast<__nv_bfloat16*>(dV.data_ptr());

  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

  size_t dqf_bytes  = (size_t)BH * S * DH * sizeof(float);
  size_t dbuf_bytes = (size_t)BH * S * sizeof(float);
  ensure_buf(&g_dQf, &g_dQf_bytes, dqf_bytes);
  ensure_buf(&g_Dbuf, &g_Dbuf_bytes, dbuf_bytes);
  float* dQf  = static_cast<float*>(g_dQf);
  float* Dbuf = static_cast<float*>(g_Dbuf);

  CUDA_CHECK(cudaMemsetAsync(dQf, 0, dqf_bytes, stream));

  int64_t nrows = BH * S;
  compute_D_kernel<<<(unsigned)((nrows + 255) / 256), 256, 0, stream>>>(Op, dOp, Dbuf, nrows);
  CUDA_CHECK(cudaGetLastError());

  if (!g_attr_set) {
    CUDA_CHECK(cudaFuncSetAttribute(attn_bwd_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
    g_attr_set = true;
  }

  int numKV = (int)((S + BN - 1) / BN);
  dim3 grid((unsigned)numKV, (unsigned)H, (unsigned)B);
  attn_bwd_kernel<<<grid, THREADS, SMEM_BYTES, stream>>>(
      Qp, Kp, Vp, dOp, Lp, Dbuf, dQf, dKp, dVp, (int)S);
  CUDA_CHECK(cudaGetLastError());

  int64_t total = BH * S * DH;
  convert_kernel<<<(unsigned)((total + 255) / 256), 256, 0, stream>>>(dQf, dQp, total);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd