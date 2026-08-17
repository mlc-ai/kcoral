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
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace attn_causal {

constexpr int D = 128;
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int THREADS = 128;

// Shared memory layout (dynamic):
//   Qs : BM*D  bf16
//   Ks : BN*D  bf16
//   Vs : BN*D  bf16
//   Ss : BM*BN fp32
//   Ps : BM*BN bf16
//   Os : BM*D  fp32
//   ms,ls,cs : BM fp32 each

__global__ void attn_kernel(const __nv_bfloat16* __restrict__ Q,
                            const __nv_bfloat16* __restrict__ K,
                            const __nv_bfloat16* __restrict__ V,
                            __nv_bfloat16* __restrict__ O,
                            float* __restrict__ LSE,
                            int S, int H) {
  const int b   = blockIdx.z;
  const int h   = blockIdx.y;
  const int qb  = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;

  const long head_base = ((long)(b * H + h)) * (long)S * D;
  const __nv_bfloat16* Qbh = Q + head_base;
  const __nv_bfloat16* Kbh = K + head_base;
  const __nv_bfloat16* Vbh = V + head_base;
  __nv_bfloat16* Obh = O + head_base;
  float* LSEbh = LSE + ((long)(b * H + h)) * S;

  extern __shared__ char smem_raw[];
  __nv_bfloat16* Qs = (__nv_bfloat16*)smem_raw;
  __nv_bfloat16* Ks = Qs + BM * D;
  __nv_bfloat16* Vs = Ks + BN * D;
  float*         Ss = (float*)(Vs + BN * D);
  __nv_bfloat16* Ps = (__nv_bfloat16*)(Ss + BM * BN);
  float*         Os = (float*)(Ps + BM * BN);
  float*         ms = Os + BM * D;
  float*         ls = ms + BM;
  float*         cs = ls + BM;

  const float scale = 0.08838834764831843f; // 1/sqrt(128)

  // init
  if (tid < BM) { ms[tid] = -INFINITY; ls[tid] = 0.0f; }
  for (int idx = tid; idx < BM * D; idx += THREADS) Os[idx] = 0.0f;

  // Load Q block (vectorized float4 = 8 bf16)
  for (int idx = tid; idx < BM * (D / 8); idx += THREADS) {
    int r = idx / (D / 8);
    int v = idx % (D / 8);
    int gq = qb * BM + r;
    float4 val;
    if (gq < S) val = reinterpret_cast<const float4*>(Qbh + (long)gq * D)[v];
    else        { val.x = val.y = val.z = val.w = 0.0f; }
    reinterpret_cast<float4*>(Qs + r * D)[v] = val;
  }
  __syncthreads();

  const int kb_max = qb; // inclusive (causal)

  for (int kb = 0; kb <= kb_max; kb++) {
    // Load K,V blocks
    for (int idx = tid; idx < BN * (D / 8); idx += THREADS) {
      int c = idx / (D / 8);
      int v = idx % (D / 8);
      int gk = kb * BN + c;
      float4 kval, vval;
      if (gk < S) {
        kval = reinterpret_cast<const float4*>(Kbh + (long)gk * D)[v];
        vval = reinterpret_cast<const float4*>(Vbh + (long)gk * D)[v];
      } else {
        kval.x = kval.y = kval.z = kval.w = 0.0f;
        vval.x = vval.y = vval.z = vval.w = 0.0f;
      }
      reinterpret_cast<float4*>(Ks + c * D)[v] = kval;
      reinterpret_cast<float4*>(Vs + c * D)[v] = vval;
    }
    __syncthreads();

    // S = Q @ K^T  (A=Q row-major, B=K col-major)
    {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[BN / 16];
      #pragma unroll
      for (int nt = 0; nt < BN / 16; nt++) wmma::fill_fragment(c_frag[nt], 0.0f);
      #pragma unroll
      for (int kt = 0; kt < D / 16; kt++) {
        wmma::load_matrix_sync(a_frag, Qs + (warp * 16) * D + kt * 16, D);
        #pragma unroll
        for (int nt = 0; nt < BN / 16; nt++) {
          wmma::load_matrix_sync(b_frag, Ks + (nt * 16) * D + kt * 16, D);
          wmma::mma_sync(c_frag[nt], a_frag, b_frag, c_frag[nt]);
        }
      }
      #pragma unroll
      for (int nt = 0; nt < BN / 16; nt++)
        wmma::store_matrix_sync(Ss + (warp * 16) * BN + nt * 16, c_frag[nt], BN, wmma::mem_row_major);
    }
    __syncthreads();

    // Softmax (one thread per row)
    if (tid < BM) {
      int r = tid;
      int gq = qb * BM + r;
      bool row_valid = (gq < S);
      float m_old = ms[r];
      float m_new = m_old;
      if (row_valid) {
        float m_local = -INFINITY;
        #pragma unroll
        for (int c = 0; c < BN; c++) {
          int gk = kb * BN + c;
          if (gk < S && gk <= gq) {
            float s = Ss[r * BN + c] * scale;
            m_local = fmaxf(m_local, s);
          }
        }
        m_new = fmaxf(m_old, m_local);
      }
      float corr = 0.0f;
      if (row_valid) {
        corr = __expf(m_old - m_new); // m_old = -inf -> 0
        float sum = 0.0f;
        #pragma unroll
        for (int c = 0; c < BN; c++) {
          int gk = kb * BN + c;
          float p = 0.0f;
          if (gk < S && gk <= gq)
            p = __expf(Ss[r * BN + c] * scale - m_new);
          sum += p;
          Ps[r * BN + c] = __float2bfloat16(p);
        }
        ls[r] = ls[r] * corr + sum;
        ms[r] = m_new;
      } else {
        #pragma unroll
        for (int c = 0; c < BN; c++) Ps[r * BN + c] = __float2bfloat16(0.0f);
      }
      cs[r] = corr;
    }
    __syncthreads();

    // Pre-scale running O by correction factor
    for (int idx = tid; idx < BM * D; idx += THREADS) {
      int r = idx / D;
      Os[idx] = Os[idx] * cs[r];
    }
    __syncthreads();

    // O += P @ V   (accumulate into scaled O held in Os)
    {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> pa;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> vb;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> o_frag[D / 16];
      #pragma unroll
      for (int nt = 0; nt < D / 16; nt++)
        wmma::load_matrix_sync(o_frag[nt], Os + (warp * 16) * D + nt * 16, D, wmma::mem_row_major);
      #pragma unroll
      for (int kt = 0; kt < BN / 16; kt++) {
        wmma::load_matrix_sync(pa, Ps + (warp * 16) * BN + kt * 16, BN);
        #pragma unroll
        for (int nt = 0; nt < D / 16; nt++) {
          wmma::load_matrix_sync(vb, Vs + (kt * 16) * D + nt * 16, D);
          wmma::mma_sync(o_frag[nt], pa, vb, o_frag[nt]);
        }
      }
      #pragma unroll
      for (int nt = 0; nt < D / 16; nt++)
        wmma::store_matrix_sync(Os + (warp * 16) * D + nt * 16, o_frag[nt], D, wmma::mem_row_major);
    }
    __syncthreads();
  }

  // Finalize: O /= l  and write LSE
  for (int idx = tid; idx < BM * D; idx += THREADS) {
    int r = idx / D;
    int d = idx % D;
    int gq = qb * BM + r;
    if (gq < S) {
      float l = ls[r];
      float o = Os[idx] / l;
      Obh[(long)gq * D + d] = __float2bfloat16(o);
    }
  }
  if (tid < BM) {
    int gq = qb * BM + tid;
    if (gq < S) {
      LSEbh[gq] = ms[tid] + logf(ls[tid]);
    }
  }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));

  int B = (int)Q.size(0);
  int H = (int)Q.size(1);
  int S = (int)Q.size(2);
  // int Dd = (int)Q.size(3); // == 128

  const __nv_bfloat16* Qp = static_cast<const __nv_bfloat16*>(Q.data_ptr());
  const __nv_bfloat16* Kp = static_cast<const __nv_bfloat16*>(K.data_ptr());
  const __nv_bfloat16* Vp = static_cast<const __nv_bfloat16*>(V.data_ptr());
  __nv_bfloat16* Op = static_cast<__nv_bfloat16*>(O.data_ptr());
  float* LSEp = static_cast<float*>(LSE.data_ptr());

  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

  int nqb = (S + BM - 1) / BM;
  dim3 grid(nqb, H, B);

  size_t smem = (size_t)BM * D * 2 * 3      // Q,K,V bf16
              + (size_t)BM * BN * 4          // S fp32
              + (size_t)BM * BN * 2          // P bf16
              + (size_t)BM * D * 4           // O fp32
              + (size_t)BM * 3 * 4;          // m,l,cs

  static bool attr_set = false;
  if (!attr_set) {
    CUDA_CHECK(cudaFuncSetAttribute(attn_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    attr_set = true;
  }

  attn_kernel<<<grid, THREADS, smem, stream>>>(Qp, Kp, Vp, Op, LSEp, S, H);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, attn_causal::run);

}  // namespace attn_causal