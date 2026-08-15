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
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int THREADS = 256;
constexpr int NWARPS = THREADS / 32;   // 8
constexpr int MT = BM / 16;            // 4
constexpr int NT = BN / 16;            // 4
constexpr int KT = DH / 16;            // 8
constexpr int NLOC = (NT * 8) / NWARPS; // 4

constexpr int SMEM_BF16 = (BM*DH + BN*DH + BN*DH + BM*DH + BM*BN + BM*BN);
constexpr int SMEM_F32  = (BM*BN + NWARPS*256 + BM + BM);
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

// -------- Kernel A: dK, dV (parallel over KV blocks) --------
__global__ void __launch_bounds__(THREADS) attn_dkv_kernel(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* dOg, const float* Lg, const float* Dg,
    __nv_bfloat16* dK, __nv_bfloat16* dV, int S) {

  extern __shared__ char smem_raw[];
  __nv_bfloat16* Qs  = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  __nv_bfloat16* Ks  = Qs  + BM * DH;
  __nv_bfloat16* Vs  = Ks  + BN * DH;
  __nv_bfloat16* dOs = Vs  + BN * DH;
  __nv_bfloat16* Ps  = dOs + BM * DH;
  __nv_bfloat16* dSs = Ps  + BM * BN;
  float* Tf    = reinterpret_cast<float*>(dSs + BM * BN);
  float* stg   = Tf + BM * BN;
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
  __syncthreads();

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

    // GEMM1: S = Q @ K^T -> Tf
    for (int t = warp_id; t < MT * NT; t += NWARPS) {
      int mt = t / NT, nt = t % NT;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
      wmma::fill_fragment(c, 0.0f);
      for (int kt = 0; kt < KT; kt++) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
        wmma::load_matrix_sync(a, Qs + mt * 16 * DH + kt * 16, DH);
        wmma::load_matrix_sync(b, Ks + nt * 16 * DH + kt * 16, DH);
        wmma::mma_sync(c, a, b, c);
      }
      wmma::store_matrix_sync(Tf + mt * 16 * BN + nt * 16, c, BN, wmma::mem_row_major);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < BM * BN; i += blockDim.x) {
      int m = i / BN, n = i % BN;
      int gn = n0 + n;
      float p = (gn < S) ? __expf(scale * Tf[m * BN + n] - Lsh[m]) : 0.0f;
      Ps[m * BN + n] = __float2bfloat16(p);
    }
    __syncthreads();

    // GEMM2: dP = dO @ V^T -> Tf
    for (int t = warp_id; t < MT * NT; t += NWARPS) {
      int mt = t / NT, nt = t % NT;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
      wmma::fill_fragment(c, 0.0f);
      for (int kt = 0; kt < KT; kt++) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
        wmma::load_matrix_sync(a, dOs + mt * 16 * DH + kt * 16, DH);
        wmma::load_matrix_sync(b, Vs + nt * 16 * DH + kt * 16, DH);
        wmma::mma_sync(c, a, b, c);
      }
      wmma::store_matrix_sync(Tf + mt * 16 * BN + nt * 16, c, BN, wmma::mem_row_major);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < BM * BN; i += blockDim.x) {
      int m = i / BN, n = i % BN;
      float dp = Tf[m * BN + n];
      float p = __bfloat162float(Ps[m * BN + n]);
      float ds = scale * p * (dp - Dsh[m]);
      dSs[m * BN + n] = __float2bfloat16(ds);
    }
    __syncthreads();

    // GEMM3: dV += P^T @ dO ; GEMM4: dK += dS^T @ Q
    {
      int li = 0;
      for (int t = warp_id; t < NT * 8; t += NWARPS, li++) {
        int nt = t / 8, dt = t % 8;
        for (int mt = 0; mt < MT; mt++) {
          wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> a;
          wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b;
          wmma::load_matrix_sync(a, Ps + mt * 16 * BN + nt * 16, BN);
          wmma::load_matrix_sync(b, dOs + mt * 16 * DH + dt * 16, DH);
          wmma::mma_sync(dv_acc[li], a, b, dv_acc[li]);
        }
      }
    }
    {
      int li = 0;
      for (int t = warp_id; t < NT * 8; t += NWARPS, li++) {
        int nt = t / 8, dt = t % 8;
        for (int mt = 0; mt < MT; mt++) {
          wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> a;
          wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b;
          wmma::load_matrix_sync(a, dSs + mt * 16 * BN + nt * 16, BN);
          wmma::load_matrix_sync(b, Qs + mt * 16 * DH + dt * 16, DH);
          wmma::mma_sync(dk_acc[li], a, b, dk_acc[li]);
        }
      }
    }
    __syncthreads();
  }

  // Store dV, dK
  {
    int li = 0;
    for (int t = warp_id; t < NT * 8; t += NWARPS, li++) {
      int nt = t / 8, dt = t % 8;
      wmma::store_matrix_sync(stg + warp_id * 256, dv_acc[li], 16, wmma::mem_row_major);
      __syncwarp();
      for (int e = lane; e < 256; e += 32) {
        int r = e / 16, cc = e % 16;
        int gn = n0 + nt * 16 + r;
        int gc = dt * 16 + cc;
        if (gn < S)
          dV[head_off + (int64_t)gn * DH + gc] = __float2bfloat16(stg[warp_id * 256 + e]);
      }
      __syncwarp();
      wmma::store_matrix_sync(stg + warp_id * 256, dk_acc[li], 16, wmma::mem_row_major);
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

// -------- Kernel B: dQ (parallel over query blocks) --------
__global__ void __launch_bounds__(THREADS) attn_dq_kernel(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* dOg, const float* Lg, const float* Dg,
    __nv_bfloat16* dQ, int S) {

  extern __shared__ char smem_raw[];
  __nv_bfloat16* Qs  = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  __nv_bfloat16* Ks  = Qs  + BM * DH;
  __nv_bfloat16* Vs  = Ks  + BN * DH;
  __nv_bfloat16* dOs = Vs  + BN * DH;
  __nv_bfloat16* Ps  = dOs + BM * DH;
  __nv_bfloat16* dSs = Ps  + BM * BN;
  float* Tf    = reinterpret_cast<float*>(dSs + BM * BN);
  float* stg   = Tf + BM * BN;
  float* Lsh   = stg + NWARPS * 256;
  float* Dsh   = Lsh + BM;

  int bh = blockIdx.z * gridDim.y + blockIdx.y;
  int m0 = blockIdx.x * BM;
  if (m0 >= S) return;

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

  load_tile(Qs, Qh, m0, S, BM);
  load_tile(dOs, dOh, m0, S, BM);
  for (int i = threadIdx.x; i < BM; i += blockDim.x) {
    int s = m0 + i;
    Lsh[i] = (s < S) ? Lh[s] : 0.0f;
    Dsh[i] = (s < S) ? Dh[s] : 0.0f;
  }
  __syncthreads();

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> dq_acc[MT];
  #pragma unroll
  for (int li = 0; li < MT; li++) wmma::fill_fragment(dq_acc[li], 0.0f);

  int numKV = (S + BN - 1) / BN;
  for (int kb = 0; kb < numKV; kb++) {
    int n0 = kb * BN;
    load_tile(Ks, Kh, n0, S, BN);
    load_tile(Vs, Vh, n0, S, BN);
    __syncthreads();

    // GEMM1: S = Q @ K^T -> Tf
    for (int t = warp_id; t < MT * NT; t += NWARPS) {
      int mt = t / NT, nt = t % NT;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
      wmma::fill_fragment(c, 0.0f);
      for (int kt = 0; kt < KT; kt++) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
        wmma::load_matrix_sync(a, Qs + mt * 16 * DH + kt * 16, DH);
        wmma::load_matrix_sync(b, Ks + nt * 16 * DH + kt * 16, DH);
        wmma::mma_sync(c, a, b, c);
      }
      wmma::store_matrix_sync(Tf + mt * 16 * BN + nt * 16, c, BN, wmma::mem_row_major);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < BM * BN; i += blockDim.x) {
      int m = i / BN, n = i % BN;
      int gn = n0 + n;
      float p = (gn < S) ? __expf(scale * Tf[m * BN + n] - Lsh[m]) : 0.0f;
      Ps[m * BN + n] = __float2bfloat16(p);
    }
    __syncthreads();

    // GEMM2: dP = dO @ V^T -> Tf
    for (int t = warp_id; t < MT * NT; t += NWARPS) {
      int mt = t / NT, nt = t % NT;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
      wmma::fill_fragment(c, 0.0f);
      for (int kt = 0; kt < KT; kt++) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
        wmma::load_matrix_sync(a, dOs + mt * 16 * DH + kt * 16, DH);
        wmma::load_matrix_sync(b, Vs + nt * 16 * DH + kt * 16, DH);
        wmma::mma_sync(c, a, b, c);
      }
      wmma::store_matrix_sync(Tf + mt * 16 * BN + nt * 16, c, BN, wmma::mem_row_major);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < BM * BN; i += blockDim.x) {
      int m = i / BN, n = i % BN;
      float dp = Tf[m * BN + n];
      float p = __bfloat162float(Ps[m * BN + n]);
      float ds = scale * p * (dp - Dsh[m]);
      dSs[m * BN + n] = __float2bfloat16(ds);
    }
    __syncthreads();

    // GEMM5: dQ += dS @ K
    for (int li = 0; li < MT; li++) {
      int mt = li, et = warp_id;
      for (int nt = 0; nt < NT; nt++) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b;
        wmma::load_matrix_sync(a, dSs + mt * 16 * BN + nt * 16, BN);
        wmma::load_matrix_sync(b, Ks + nt * 16 * DH + et * 16, DH);
        wmma::mma_sync(dq_acc[li], a, b, dq_acc[li]);
      }
    }
    __syncthreads();
  }

  // Store dQ
  for (int li = 0; li < MT; li++) {
    wmma::store_matrix_sync(stg + warp_id * 256, dq_acc[li], 16, wmma::mem_row_major);
    __syncwarp();
    for (int e = lane; e < 256; e += 32) {
      int r = e / 16, cc = e % 16;
      int gm = m0 + li * 16 + r;
      int gc = warp_id * 16 + cc;
      if (gm < S)
        dQ[head_off + (int64_t)gm * DH + gc] = __float2bfloat16(stg[warp_id * 256 + e]);
    }
    __syncwarp();
  }
}

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

  size_t dbuf_bytes = (size_t)BH * S * sizeof(float);
  ensure_buf(&g_Dbuf, &g_Dbuf_bytes, dbuf_bytes);
  float* Dbuf = static_cast<float*>(g_Dbuf);

  int64_t nrows = BH * S;
  compute_D_kernel<<<(unsigned)((nrows + 255) / 256), 256, 0, stream>>>(Op, dOp, Dbuf, nrows);
  CUDA_CHECK(cudaGetLastError());

  if (!g_attr_set) {
    CUDA_CHECK(cudaFuncSetAttribute(attn_dkv_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
    CUDA_CHECK(cudaFuncSetAttribute(attn_dq_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
    g_attr_set = true;
  }

  int numKV = (int)((S + BN - 1) / BN);
  dim3 gridA((unsigned)numKV, (unsigned)H, (unsigned)B);
  attn_dkv_kernel<<<gridA, THREADS, SMEM_BYTES, stream>>>(
      Qp, Kp, Vp, dOp, Lp, Dbuf, dKp, dVp, (int)S);
  CUDA_CHECK(cudaGetLastError());

  int numQ = (int)((S + BM - 1) / BM);
  dim3 gridB((unsigned)numQ, (unsigned)H, (unsigned)B);
  attn_dq_kernel<<<gridB, THREADS, SMEM_BYTES, stream>>>(
      Qp, Kp, Vp, dOp, Lp, Dbuf, dQp, (int)S);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd