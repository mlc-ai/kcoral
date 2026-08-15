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

namespace mha_kernel {

constexpr int BM = 256;   // query rows per block (== threads)
constexpr int BN = 64;    // key columns per KV tile
constexpr int D  = 128;   // head dim (fixed by spec)
constexpr int NT = 256;   // threads per block

__global__ void __launch_bounds__(NT, 1) attn_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S) {

  const float scale = rsqrtf((float)D);  // 1/sqrt(128)

  int b     = blockIdx.z;
  int h     = blockIdx.y;
  int qtile = blockIdx.x;
  int tid   = threadIdx.x;
  int row   = qtile * BM + tid;

  long base = ((long)(b * H + h) * (long)S) * (long)D;

  __shared__ __nv_bfloat16 Ksh[BN * D];
  __shared__ __nv_bfloat16 Vsh[BN * D];

  // Load this thread's Q row into registers (bf16).
  __nv_bfloat16 qrow[D];
  bool valid = row < S;
  if (valid) {
    const uint4* Qg = reinterpret_cast<const uint4*>(Q + base + (long)row * D);
    #pragma unroll
    for (int i = 0; i < D / 8; ++i) {
      uint4 v = Qg[i];
      const __nv_bfloat16* vb = reinterpret_cast<const __nv_bfloat16*>(&v);
      #pragma unroll
      for (int k = 0; k < 8; ++k) qrow[i * 8 + k] = vb[k];
    }
  } else {
    #pragma unroll
    for (int i = 0; i < D; ++i) qrow[i] = __float2bfloat16(0.0f);
  }

  // Output accumulator (fp32) in registers.
  float acc[D];
  #pragma unroll
  for (int i = 0; i < D; ++i) acc[i] = 0.0f;
  float m = -INFINITY;  // running max
  float l = 0.0f;       // running sum

  int ntiles = (S + BN - 1) / BN;
  for (int t = 0; t < ntiles; ++t) {
    int kbase  = t * BN;
    int nvalid = min(BN, S - kbase);

    // Cooperative load of K and V tiles into shared memory (vectorized).
    {
      const uint4* Kg = reinterpret_cast<const uint4*>(K + base + (long)kbase * D);
      const uint4* Vg = reinterpret_cast<const uint4*>(V + base + (long)kbase * D);
      uint4* Ks4 = reinterpret_cast<uint4*>(Ksh);
      uint4* Vs4 = reinterpret_cast<uint4*>(Vsh);
      int n4 = (nvalid * D) / 8;
      for (int i = tid; i < n4; i += NT) {
        Ks4[i] = Kg[i];
        Vs4[i] = Vg[i];
      }
    }
    __syncthreads();

    // Online softmax over the keys in this tile.
    for (int j = 0; j < nvalid; ++j) {
      const uint4* Kj = reinterpret_cast<const uint4*>(&Ksh[j * D]);
      float s = 0.0f;
      #pragma unroll
      for (int i = 0; i < D / 8; ++i) {
        uint4 kv = Kj[i];
        const __nv_bfloat16* kb = reinterpret_cast<const __nv_bfloat16*>(&kv);
        #pragma unroll
        for (int k = 0; k < 8; ++k)
          s += __bfloat162float(qrow[i * 8 + k]) * __bfloat162float(kb[k]);
      }
      s *= scale;

      float m_new = fmaxf(m, s);
      float alpha = __expf(m - m_new);
      float p     = __expf(s - m_new);
      l = l * alpha + p;

      const uint4* Vj = reinterpret_cast<const uint4*>(&Vsh[j * D]);
      #pragma unroll
      for (int i = 0; i < D / 8; ++i) {
        uint4 vv = Vj[i];
        const __nv_bfloat16* vb = reinterpret_cast<const __nv_bfloat16*>(&vv);
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
          int d = i * 8 + k;
          acc[d] = acc[d] * alpha + p * __bfloat162float(vb[k]);
        }
      }
      m = m_new;
    }
    __syncthreads();
  }

  if (valid) {
    float inv_l = 1.0f / l;
    uint4* Og = reinterpret_cast<uint4*>(O + base + (long)row * D);
    #pragma unroll
    for (int i = 0; i < D / 8; ++i) {
      uint4 v;
      __nv_bfloat16* vb = reinterpret_cast<__nv_bfloat16*>(&v);
      #pragma unroll
      for (int k = 0; k < 8; ++k) vb[k] = __float2bfloat16(acc[i * 8 + k] * inv_l);
      Og[i] = v;
    }
    LSE[(long)(b * H + h) * (long)S + row] = m + logf(l);
  }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int B = (int)Q.size(0);
  int H = (int)Q.size(1);
  int S = (int)Q.size(2);
  // D == 128 by spec

  const __nv_bfloat16* Qp = static_cast<const __nv_bfloat16*>(Q.data_ptr());
  const __nv_bfloat16* Kp = static_cast<const __nv_bfloat16*>(K.data_ptr());
  const __nv_bfloat16* Vp = static_cast<const __nv_bfloat16*>(V.data_ptr());
  __nv_bfloat16* Op = static_cast<__nv_bfloat16*>(O.data_ptr());
  float* Lp = static_cast<float*>(LSE.data_ptr());

  dim3 grid((S + BM - 1) / BM, H, B);
  dim3 block(NT);
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
  attn_kernel<<<grid, block, 0, stream>>>(Qp, Kp, Vp, Op, Lp, B, H, S);
  CUDA_CHECK(cudaGetLastError());
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_kernel::run);

}  // namespace mha_kernel