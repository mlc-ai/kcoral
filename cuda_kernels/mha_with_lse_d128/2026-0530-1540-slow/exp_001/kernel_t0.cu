#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace mha_lse {

constexpr int D   = 128;
constexpr int DH  = D / 2;      // number of bf16x2 per row
constexpr int BM  = 128;        // query rows per block
constexpr int BN  = 16;         // key rows per kv tile

__global__ void __launch_bounds__(BM)
mha_kernel(const __nv_bfloat16* __restrict__ Q,
           const __nv_bfloat16* __restrict__ K,
           const __nv_bfloat16* __restrict__ V,
           __nv_bfloat16* __restrict__ O,
           float* __restrict__ LSE,
           int B, int H, int S) {
  const int q_block = blockIdx.x;
  const int bh      = blockIdx.y;        // 0 .. B*H-1
  const int tid     = threadIdx.x;       // 0 .. 127
  const int row     = q_block * BM + tid;

  const __nv_bfloat16* Qh = Q + (size_t)bh * S * D;
  const __nv_bfloat16* Kh = K + (size_t)bh * S * D;
  const __nv_bfloat16* Vh = V + (size_t)bh * S * D;
  __nv_bfloat16* Oh       = O + (size_t)bh * S * D;
  float* LSEh             = LSE + (size_t)bh * S;

  __shared__ __nv_bfloat162 qs[BM][DH + 1];  // padded to avoid bank conflicts
  __shared__ __nv_bfloat162 ks[BN][DH];
  __shared__ __nv_bfloat162 vs[BN][DH];

  const __nv_bfloat162 zero2 = __floats2bfloat162_rn(0.f, 0.f);

  // Load this thread's query row into shared
  if (row < S) {
    const __nv_bfloat162* qrow =
        reinterpret_cast<const __nv_bfloat162*>(Qh + (size_t)row * D);
    #pragma unroll
    for (int k = 0; k < DH; ++k) qs[tid][k] = qrow[k];
  } else {
    #pragma unroll
    for (int k = 0; k < DH; ++k) qs[tid][k] = zero2;
  }

  float acc[D];
  #pragma unroll
  for (int d = 0; d < D; ++d) acc[d] = 0.f;

  float m2 = -INFINITY;   // running max (base-2 units)
  float l  = 0.f;         // running denominator

  const float scale  = rsqrtf((float)D);
  const float scale2 = scale * 1.4426950408889634f;   // * log2(e)
  const float ln2    = 0.6931471805599453f;

  __syncthreads();

  for (int kv = 0; kv < S; kv += BN) {
    // Cooperative load of K/V tile
    const int total = BN * DH;
    for (int idx = tid; idx < total; idx += BM) {
      int j  = idx / DH;
      int k  = idx % DH;
      int kj = kv + j;
      if (kj < S) {
        ks[j][k] = reinterpret_cast<const __nv_bfloat162*>(Kh + (size_t)kj * D)[k];
        vs[j][k] = reinterpret_cast<const __nv_bfloat162*>(Vh + (size_t)kj * D)[k];
      } else {
        ks[j][k] = zero2;
        vs[j][k] = zero2;
      }
    }
    __syncthreads();

    // Compute scores S = scale * (q . k) for this tile
    float s[BN];
    float mb = -INFINITY;
    #pragma unroll
    for (int j = 0; j < BN; ++j) {
      float dot = 0.f;
      #pragma unroll
      for (int k = 0; k < DH; ++k) {
        float2 af = __bfloat1622float2(qs[tid][k]);
        float2 bf = __bfloat1622float2(ks[j][k]);
        dot += af.x * bf.x + af.y * bf.y;
      }
      float sc = dot * scale2;
      if (kv + j >= S) sc = -INFINITY;     // mask out-of-range keys
      s[j] = sc;
      mb = fmaxf(mb, sc);
    }

    // Online softmax update
    float m2_new = fmaxf(m2, mb);
    float corr   = exp2f(m2 - m2_new);     // 0 when m2 == -inf (first block)

    #pragma unroll
    for (int d = 0; d < D; ++d) acc[d] *= corr;
    l *= corr;

    #pragma unroll
    for (int j = 0; j < BN; ++j) {
      float p = exp2f(s[j] - m2_new);      // 0 for masked keys
      l += p;
      #pragma unroll
      for (int k = 0; k < DH; ++k) {
        float2 vf = __bfloat1622float2(vs[j][k]);
        acc[2 * k]     += p * vf.x;
        acc[2 * k + 1] += p * vf.y;
      }
    }
    m2 = m2_new;
    __syncthreads();
  }

  // Write outputs
  if (row < S) {
    float inv_l = 1.f / l;
    __nv_bfloat162* orow = reinterpret_cast<__nv_bfloat162*>(Oh + (size_t)row * D);
    #pragma unroll
    for (int k = 0; k < DH; ++k) {
      float x = acc[2 * k]     * inv_l;
      float y = acc[2 * k + 1] * inv_l;
      orow[k] = __floats2bfloat162_rn(x, y);
    }
    LSEh[row] = ln2 * (m2 + log2f(l));
  }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));

  int B = (int)Q.size(0);
  int H = (int)Q.size(1);
  int S = (int)Q.size(2);
  // int Dd = (int)Q.size(3); // == 128

  const __nv_bfloat16* q = static_cast<const __nv_bfloat16*>(Q.data_ptr());
  const __nv_bfloat16* k = static_cast<const __nv_bfloat16*>(K.data_ptr());
  const __nv_bfloat16* v = static_cast<const __nv_bfloat16*>(V.data_ptr());
  __nv_bfloat16* o       = static_cast<__nv_bfloat16*>(O.data_ptr());
  float* lse             = static_cast<float*>(LSE.data_ptr());

  dim3 grid((S + BM - 1) / BM, B * H);
  dim3 block(BM);

  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

  mha_kernel<<<grid, block, 0, stream>>>(q, k, v, o, lse, B, H, S);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_lse::run);

}  // namespace mha_lse