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

constexpr int D        = 128;
constexpr int DH       = 64;     // bf16x2 per row
constexpr int BM_ROWS  = 128;    // query rows per block
constexpr int TPR      = 4;      // threads per row
constexpr int NTHREAD  = BM_ROWS * TPR;   // 512
constexpr int SUB      = DH / TPR;        // 16 bf16x2 per thread
constexpr int BN       = 32;     // key rows per kv tile

__device__ __forceinline__ float fast_exp2(float x) {
  float y;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__global__ void __launch_bounds__(NTHREAD)
mha_kernel(const __nv_bfloat16* __restrict__ Q,
           const __nv_bfloat16* __restrict__ K,
           const __nv_bfloat16* __restrict__ V,
           __nv_bfloat16* __restrict__ O,
           float* __restrict__ LSE,
           int B, int H, int S) {
  const int bh   = blockIdx.y;
  const int tid  = threadIdx.x;
  const int g    = tid & (TPR - 1);          // 0..3  : which dim-quarter
  const int rid  = tid >> 2;                  // 0..127: row within block
  const int row  = blockIdx.x * BM_ROWS + rid;
  const int doff = g * SUB;                    // bf16x2 dim offset (0,16,32,48)

  const __nv_bfloat16* Qh = Q + (size_t)bh * S * D;
  const float4* Kh4 = reinterpret_cast<const float4*>(K + (size_t)bh * S * D);
  const float4* Vh4 = reinterpret_cast<const float4*>(V + (size_t)bh * S * D);
  __nv_bfloat16* Oh = O + (size_t)bh * S * D;
  float* LSEh       = LSE + (size_t)bh * S;

  __shared__ __nv_bfloat162 ks[BN][DH];
  __shared__ __nv_bfloat162 vs[BN][DH];
  float4* ks4 = reinterpret_cast<float4*>(&ks[0][0]);
  float4* vs4 = reinterpret_cast<float4*>(&vs[0][0]);

  // Load this thread's slice of the query row (16 bf16x2 = 32 dims).
  __nv_bfloat162 q2[SUB];
  if (row < S) {
    const __nv_bfloat162* qrow =
        reinterpret_cast<const __nv_bfloat162*>(Qh + (size_t)row * D);
    #pragma unroll
    for (int i = 0; i < SUB; ++i) q2[i] = qrow[doff + i];
  } else {
    #pragma unroll
    for (int i = 0; i < SUB; ++i) q2[i] = __floats2bfloat162_rn(0.f, 0.f);
  }

  float accx[SUB], accy[SUB];
  #pragma unroll
  for (int i = 0; i < SUB; ++i) { accx[i] = 0.f; accy[i] = 0.f; }

  float m2 = -INFINITY;
  float l  = 0.f;

  const float scale2 = rsqrtf((float)D) * 1.4426950408889634f;
  const float ln2    = 0.6931471805599453f;
  const float4 z4    = make_float4(0.f, 0.f, 0.f, 0.f);

  for (int kv = 0; kv < S; kv += BN) {
    // ---- cooperative tile load (128-bit float4) ----
    #pragma unroll
    for (int idx = tid; idx < BN * 16; idx += NTHREAD) {
      int j  = idx >> 4;
      int c  = idx & 15;
      int kj = kv + j;
      if (kj < S) {
        ks4[idx] = Kh4[(size_t)kj * 16 + c];
        vs4[idx] = Vh4[(size_t)kj * 16 + c];
      } else {
        ks4[idx] = z4;
        vs4[idx] = z4;
      }
    }
    __syncthreads();

    // ---- QK^T scores ----
    float s[BN];
    #pragma unroll
    for (int j = 0; j < BN; ++j) {
      float partial = 0.f;
      #pragma unroll
      for (int i = 0; i < SUB; ++i) {
        float2 qf = __bfloat1622float2(q2[i]);
        float2 kf = __bfloat1622float2(ks[j][doff + i]);
        partial += qf.x * kf.x + qf.y * kf.y;
      }
      partial += __shfl_xor_sync(0xffffffffu, partial, 1);
      partial += __shfl_xor_sync(0xffffffffu, partial, 2);
      float sc = partial * scale2;
      if (kv + j >= S) sc = -INFINITY;
      s[j] = sc;
    }

    float mb = -INFINITY;
    #pragma unroll
    for (int j = 0; j < BN; ++j) mb = fmaxf(mb, s[j]);

    // ---- online softmax update ----
    float m2_new = fmaxf(m2, mb);
    float corr   = fast_exp2(m2 - m2_new);
    #pragma unroll
    for (int i = 0; i < SUB; ++i) { accx[i] *= corr; accy[i] *= corr; }
    l *= corr;

    // ---- PV accumulation ----
    #pragma unroll
    for (int j = 0; j < BN; ++j) {
      float p = fast_exp2(s[j] - m2_new);
      l += p;
      #pragma unroll
      for (int i = 0; i < SUB; ++i) {
        float2 vf = __bfloat1622float2(vs[j][doff + i]);
        accx[i] += p * vf.x;
        accy[i] += p * vf.y;
      }
    }
    m2 = m2_new;
    __syncthreads();
  }

  // ---- write outputs ----
  if (row < S) {
    float inv = 1.f / l;
    __nv_bfloat162* orow = reinterpret_cast<__nv_bfloat162*>(Oh + (size_t)row * D);
    #pragma unroll
    for (int i = 0; i < SUB; ++i) {
      orow[doff + i] = __floats2bfloat162_rn(accx[i] * inv, accy[i] * inv);
    }
    if (g == 0) LSEh[row] = ln2 * (m2 + log2f(l));
  }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));

  int B = (int)Q.size(0);
  int H = (int)Q.size(1);
  int S = (int)Q.size(2);

  const __nv_bfloat16* q = static_cast<const __nv_bfloat16*>(Q.data_ptr());
  const __nv_bfloat16* k = static_cast<const __nv_bfloat16*>(K.data_ptr());
  const __nv_bfloat16* v = static_cast<const __nv_bfloat16*>(V.data_ptr());
  __nv_bfloat16* o       = static_cast<__nv_bfloat16*>(O.data_ptr());
  float* lse             = static_cast<float*>(LSE.data_ptr());

  dim3 grid((S + BM_ROWS - 1) / BM_ROWS, B * H);
  dim3 block(NTHREAD);

  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

  mha_kernel<<<grid, block, 0, stream>>>(q, k, v, o, lse, B, H, S);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_lse::run);

}  // namespace mha_lse