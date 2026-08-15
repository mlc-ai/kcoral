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
constexpr int THREADS = 256;
constexpr int NWARPS = THREADS / 32;   // 8
constexpr int KT = DH / 16;            // 8

// ---- Pass A tile (dK,dV): BM=128 query, BN=64 kv ----
constexpr int A_BM = 128, A_BN = 64;
constexpr int A_MT = A_BM/16, A_NT = A_BN/16, A_NLOC = A_BN/16;
constexpr int A_SMEM_BF16 = (A_BM*DH + A_BN*DH + A_BN*DH + A_BM*DH + A_BM*A_BN + A_BM*A_BN);
constexpr int A_SMEM_F32  = (NWARPS*256 + A_BM + A_BM);
constexpr int A_SMEM = A_SMEM_BF16*2 + A_SMEM_F32*4;

// ---- Pass B tile (dQ): BM=64 query, BN=64 kv ----
constexpr int B_BM = 64, B_BN = 64;
constexpr int B_MT = B_BM/16, B_NT = B_BN/16;
constexpr int B_SMEM_BF16 = (B_BM*DH + B_BN*DH + B_BN*DH + B_BM*DH + B_BM*B_BN + B_BM*B_BN);
constexpr int B_SMEM_F32  = (NWARPS*256 + B_BM + B_BM);
constexpr int B_SMEM = B_SMEM_BF16*2 + B_SMEM_F32*4;

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

// ============ Pass A: dK, dV ============
__global__ void __launch_bounds__(THREADS) attn_dkv_kernel(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* dOg, const float* Lg, const float* Dg,
    __nv_bfloat16* dK, __nv_bfloat16* dV, int S) {

  extern __shared__ char smem_raw[];
  __nv_bfloat16* Qs  = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  __nv_bfloat16* Ks  = Qs  + A_BM * DH;
  __nv_bfloat16* Vs  = Ks  + A_BN * DH;
  __nv_bfloat16* dOs = Vs  + A_BN * DH;
  __nv_bfloat16* Ps  = dOs + A_BM * DH;
  __nv_bfloat16* dSs = Ps  + A_BM * A_BN;
  float* stg   = reinterpret_cast<float*>(dSs + A_BM * A_BN);
  float* Lsh   = stg + NWARPS * 256;
  float* Dsh   = Lsh + A_BM;

  int bh = blockIdx.z * gridDim.y + blockIdx.y;
  int n0 = blockIdx.x * A_BN;
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
  int gid = lane >> 2, tg = lane & 3;
  int ro[8] = {gid, gid, gid+8, gid+8, gid, gid, gid+8, gid+8};
  int co[8] = {2*tg, 2*tg+1, 2*tg, 2*tg+1, 2*tg+8, 2*tg+9, 2*tg+8, 2*tg+9};

  load_tile(Ks, Kh, n0, S, A_BN);
  load_tile(Vs, Vh, n0, S, A_BN);

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> dv_acc[A_NLOC];
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> dk_acc[A_NLOC];
  #pragma unroll
  for (int li = 0; li < A_NLOC; li++) {
    wmma::fill_fragment(dv_acc[li], 0.0f);
    wmma::fill_fragment(dk_acc[li], 0.0f);
  }

  int numQ = (S + A_BM - 1) / A_BM;
  for (int qb = 0; qb < numQ; qb++) {
    int m0 = qb * A_BM;
    load_tile(Qs, Qh, m0, S, A_BM);
    load_tile(dOs, dOh, m0, S, A_BM);
    for (int i = threadIdx.x; i < A_BM; i += blockDim.x) {
      int s = m0 + i;
      Lsh[i] = (s < S) ? Lh[s] : 0.0f;
      Dsh[i] = (s < S) ? Dh[s] : 0.0f;
    }
    __syncthreads();

    // GEMM1: S=Q@K^T -> fused P
    {
      int nt = warp_id % A_NT;
      int mt_base = (warp_id / A_NT) * 4;
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
        wmma::mma_sync(c0, a0, b, c0); wmma::mma_sync(c1, a1, b, c1);
        wmma::mma_sync(c2, a2, b, c2); wmma::mma_sync(c3, a3, b, c3);
      }
      wmma::fragment<wmma::accumulator, 16, 16, 16, float>* cs[4] = {&c0,&c1,&c2,&c3};
      #pragma unroll
      for (int j = 0; j < 4; j++) {
        int mt = mt_base + j;
        #pragma unroll
        for (int e = 0; e < 8; e++) {
          int r = mt * 16 + ro[e]; int cl = nt * 16 + co[e];
          float s = cs[j]->x[e];
          int key = n0 + cl;
          float p = (key < S) ? __expf(scale * s - Lsh[r]) : 0.0f;
          Ps[r * A_BN + cl] = __float2bfloat16(p);
        }
      }
    }
    // GEMM2: dP=dO@V^T -> fused dS
    {
      int nt = warp_id % A_NT;
      int mt_base = (warp_id / A_NT) * 4;
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
        wmma::mma_sync(c0, a0, b, c0); wmma::mma_sync(c1, a1, b, c1);
        wmma::mma_sync(c2, a2, b, c2); wmma::mma_sync(c3, a3, b, c3);
      }
      wmma::fragment<wmma::accumulator, 16, 16, 16, float>* cs[4] = {&c0,&c1,&c2,&c3};
      #pragma unroll
      for (int j = 0; j < 4; j++) {
        int mt = mt_base + j;
        #pragma unroll
        for (int e = 0; e < 8; e++) {
          int r = mt * 16 + ro[e]; int cl = nt * 16 + co[e];
          float dp = cs[j]->x[e];
          float p = __bfloat162float(Ps[r * A_BN + cl]);
          float ds = scale * p * (dp - Dsh[r]);
          dSs[r * A_BN + cl] = __float2bfloat16(ds);
        }
      }
    }
    __syncthreads();

    // GEMM3: dV += P^T@dO ; GEMM4: dK += dS^T@Q
    {
      int dt = warp_id;
      #pragma unroll
      for (int mt = 0; mt < A_MT; mt++) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bdo, bq;
        wmma::load_matrix_sync(bdo, dOs + mt * 16 * DH + dt * 16, DH);
        wmma::load_matrix_sync(bq,  Qs  + mt * 16 * DH + dt * 16, DH);
        #pragma unroll
        for (int nt = 0; nt < A_NLOC; nt++) {
          wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> ap, ads;
          wmma::load_matrix_sync(ap,  Ps  + mt * 16 * A_BN + nt * 16, A_BN);
          wmma::load_matrix_sync(ads, dSs + mt * 16 * A_BN + nt * 16, A_BN);
          wmma::mma_sync(dv_acc[nt], ap,  bdo, dv_acc[nt]);
          wmma::mma_sync(dk_acc[nt], ads, bq,  dk_acc[nt]);
        }
      }
    }
    __syncthreads();
  }

  // store dV, dK
  {
    int dt = warp_id;
    #pragma unroll
    for (int nt = 0; nt < A_NLOC; nt++) {
      wmma::store_matrix_sync(stg + warp_id * 256, dv_acc[nt], 16, wmma::mem_row_major);
      __syncwarp();
      for (int e = lane; e < 256; e += 32) {
        int r = e / 16, cc = e % 16;
        int gn = n0 + nt * 16 + r; int gc = dt * 16 + cc;
        if (gn < S) dV[head_off + (int64_t)gn * DH + gc] = __float2bfloat16(stg[warp_id * 256 + e]);
      }
      __syncwarp();
      wmma::store_matrix_sync(stg + warp_id * 256, dk_acc[nt], 16, wmma::mem_row_major);
      __syncwarp();
      for (int e = lane; e < 256; e += 32) {
        int r = e / 16, cc = e % 16;
        int gn = n0 + nt * 16 + r; int gc = dt * 16 + cc;
        if (gn < S) dK[head_off + (int64_t)gn * DH + gc] = __float2bfloat16(stg[warp_id * 256 + e]);
      }
      __syncwarp();
    }
  }
}

// ============ Pass B: dQ ============
__global__ void __launch_bounds__(THREADS) attn_dq_kernel(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* dOg, const float* Lg, const float* Dg,
    __nv_bfloat16* dQ, int S) {

  extern __shared__ char smem_raw[];
  __nv_bfloat16* Qs  = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  __nv_bfloat16* Ks  = Qs  + B_BM * DH;
  __nv_bfloat16* Vs  = Ks  + B_BN * DH;
  __nv_bfloat16* dOs = Vs  + B_BN * DH;
  __nv_bfloat16* Ps  = dOs + B_BM * DH;
  __nv_bfloat16* dSs = Ps  + B_BM * B_BN;
  float* stg   = reinterpret_cast<float*>(dSs + B_BM * B_BN);
  float* Lsh   = stg + NWARPS * 256;
  float* Dsh   = Lsh + B_BM;

  int bh = blockIdx.z * gridDim.y + blockIdx.y;
  int m0 = blockIdx.x * B_BM;
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
  int gid = lane >> 2, tg = lane & 3;
  int ro[8] = {gid, gid, gid+8, gid+8, gid, gid, gid+8, gid+8};
  int co[8] = {2*tg, 2*tg+1, 2*tg, 2*tg+1, 2*tg+8, 2*tg+9, 2*tg+8, 2*tg+9};

  load_tile(Qs, Qh, m0, S, B_BM);
  load_tile(dOs, dOh, m0, S, B_BM);
  for (int i = threadIdx.x; i < B_BM; i += blockDim.x) {
    int s = m0 + i;
    Lsh[i] = (s < S) ? Lh[s] : 0.0f;
    Dsh[i] = (s < S) ? Dh[s] : 0.0f;
  }

  // persistent dQ accumulators: warp owns dt=warp_id (DH tile), B_MT row tiles
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> dq_acc[B_MT];
  #pragma unroll
  for (int li = 0; li < B_MT; li++) wmma::fill_fragment(dq_acc[li], 0.0f);

  __syncthreads();

  int numKV = (S + B_BN - 1) / B_BN;
  for (int kb = 0; kb < numKV; kb++) {
    int n0 = kb * B_BN;
    load_tile(Ks, Kh, n0, S, B_BN);
    load_tile(Vs, Vh, n0, S, B_BN);
    __syncthreads();

    // GEMM1: S=Q@K^T -> fused P  (2 accumulators/warp)
    {
      int nt = warp_id % B_NT;
      int mtg = warp_id / B_NT;   // 0 or 1
      int mt0 = mtg, mt1 = mtg + 2;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> c0, c1;
      wmma::fill_fragment(c0, 0.0f); wmma::fill_fragment(c1, 0.0f);
      #pragma unroll
      for (int kt = 0; kt < KT; kt++) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
        wmma::load_matrix_sync(b, Ks + nt * 16 * DH + kt * 16, DH);
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a0, a1;
        wmma::load_matrix_sync(a0, Qs + mt0 * 16 * DH + kt * 16, DH);
        wmma::load_matrix_sync(a1, Qs + mt1 * 16 * DH + kt * 16, DH);
        wmma::mma_sync(c0, a0, b, c0); wmma::mma_sync(c1, a1, b, c1);
      }
      wmma::fragment<wmma::accumulator, 16, 16, 16, float>* cs[2] = {&c0,&c1};
      int mts[2] = {mt0, mt1};
      #pragma unroll
      for (int j = 0; j < 2; j++) {
        int mt = mts[j];
        #pragma unroll
        for (int e = 0; e < 8; e++) {
          int r = mt * 16 + ro[e]; int cl = nt * 16 + co[e];
          float s = cs[j]->x[e];
          int key = n0 + cl;
          float p = (key < S) ? __expf(scale * s - Lsh[r]) : 0.0f;
          Ps[r * B_BN + cl] = __float2bfloat16(p);
        }
      }
    }
    // GEMM2: dP=dO@V^T -> fused dS
    {
      int nt = warp_id % B_NT;
      int mtg = warp_id / B_NT;
      int mt0 = mtg, mt1 = mtg + 2;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> c0, c1;
      wmma::fill_fragment(c0, 0.0f); wmma::fill_fragment(c1, 0.0f);
      #pragma unroll
      for (int kt = 0; kt < KT; kt++) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
        wmma::load_matrix_sync(b, Vs + nt * 16 * DH + kt * 16, DH);
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a0, a1;
        wmma::load_matrix_sync(a0, dOs + mt0 * 16 * DH + kt * 16, DH);
        wmma::load_matrix_sync(a1, dOs + mt1 * 16 * DH + kt * 16, DH);
        wmma::mma_sync(c0, a0, b, c0); wmma::mma_sync(c1, a1, b, c1);
      }
      wmma::fragment<wmma::accumulator, 16, 16, 16, float>* cs[2] = {&c0,&c1};
      int mts[2] = {mt0, mt1};
      #pragma unroll
      for (int j = 0; j < 2; j++) {
        int mt = mts[j];
        #pragma unroll
        for (int e = 0; e < 8; e++) {
          int r = mt * 16 + ro[e]; int cl = nt * 16 + co[e];
          float dp = cs[j]->x[e];
          float p = __bfloat162float(Ps[r * B_BN + cl]);
          float ds = scale * p * (dp - Dsh[r]);
          dSs[r * B_BN + cl] = __float2bfloat16(ds);
        }
      }
    }
    __syncthreads();

    // GEMM5: dQ += dS@K  (warp owns dt=warp_id, B_MT row accumulators, reduce nt)
    {
      int dt = warp_id;
      #pragma unroll
      for (int nt = 0; nt < B_NT; nt++) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b;
        wmma::load_matrix_sync(b, Ks + nt * 16 * DH + dt * 16, DH);
        #pragma unroll
        for (int mt = 0; mt < B_MT; mt++) {
          wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
          wmma::load_matrix_sync(a, dSs + mt * 16 * B_BN + nt * 16, B_BN);
          wmma::mma_sync(dq_acc[mt], a, b, dq_acc[mt]);
        }
      }
    }
    __syncthreads();
  }

  // store dQ
  {
    int dt = warp_id;
    #pragma unroll
    for (int mt = 0; mt < B_MT; mt++) {
      wmma::store_matrix_sync(stg + warp_id * 256, dq_acc[mt], 16, wmma::mem_row_major);
      __syncwarp();
      for (int e = lane; e < 256; e += 32) {
        int r = e / 16, cc = e % 16;
        int gm = m0 + mt * 16 + r; int gc = dt * 16 + cc;
        if (gm < S) dQ[head_off + (int64_t)gm * DH + gc] = __float2bfloat16(stg[warp_id * 256 + e]);
      }
      __syncwarp();
    }
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
        cudaFuncAttributeMaxDynamicSharedMemorySize, A_SMEM));
    CUDA_CHECK(cudaFuncSetAttribute(attn_dq_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, B_SMEM));
    g_attr_set = true;
  }

  int numKV = (int)((S + A_BN - 1) / A_BN);
  dim3 gridA((unsigned)numKV, (unsigned)H, (unsigned)B);
  attn_dkv_kernel<<<gridA, THREADS, A_SMEM, stream>>>(
      Qp, Kp, Vp, dOp, Lp, Dbuf, dKp, dVp, (int)S);
  CUDA_CHECK(cudaGetLastError());

  int numQ = (int)((S + B_BM - 1) / B_BM);
  dim3 gridB((unsigned)numQ, (unsigned)H, (unsigned)B);
  attn_dq_kernel<<<gridB, THREADS, B_SMEM, stream>>>(
      Qp, Kp, Vp, dOp, Lp, Dbuf, dQp, (int)S);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd