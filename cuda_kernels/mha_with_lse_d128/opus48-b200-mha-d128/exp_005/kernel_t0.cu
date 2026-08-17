#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
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

namespace mha {

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int D  = 128;
constexpr int THREADS = 128;

__device__ __forceinline__ float redMax8(float v){
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 4));
  return v;
}
__device__ __forceinline__ float redSum8(float v){
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  v += __shfl_xor_sync(0xffffffffu, v, 4);
  return v;
}

__global__ __launch_bounds__(THREADS) void attn_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S, float scale)
{
  int bh = blockIdx.y;
  int b = bh / H;
  int h = bh % H;
  int q_tile = blockIdx.x;
  int q0 = q_tile * BM;
  int tid = threadIdx.x;
  int mid = tid >> 3;   // 0..15  (query-row group)
  int nid = tid & 7;    // 0..7   (column group)

  extern __shared__ char smem[];
  __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem);
  __nv_bfloat16* sK = sQ + BM * D;
  __nv_bfloat16* sV = sK + BN * D;
  float* sP = reinterpret_cast<float*>(sV + BN * D);

  const __nv_bfloat16* Qbh = Q + (int64_t)(b * H + h) * S * D;
  const __nv_bfloat16* Kbh = K + (int64_t)(b * H + h) * S * D;
  const __nv_bfloat16* Vbh = V + (int64_t)(b * H + h) * S * D;

  // ---- Load Q tile ----
  int q_valid = min(BM, S - q0);
  {
    const int4* gp = reinterpret_cast<const int4*>(Qbh + (int64_t)q0 * D);
    int4* sp = reinterpret_cast<int4*>(sQ);
    int total = BM * (D / 8); // 1024
    for (int idx = tid; idx < total; idx += THREADS) {
      int row = idx >> 4;
      int c16 = idx & 15;
      if (row < q_valid) sp[idx] = gp[(int64_t)row * 16 + c16];
      else { int4 z; z.x=z.y=z.z=z.w=0; sp[idx]=z; }
    }
  }

  // ---- Per-thread state ----
  float m_[4], l_[4], acc[4][16];
  #pragma unroll
  for (int r = 0; r < 4; r++) {
    m_[r] = -INFINITY; l_[r] = 0.f;
    #pragma unroll
    for (int e = 0; e < 16; e++) acc[r][e] = 0.f;
  }

  int n_kv = (S + BN - 1) / BN;
  for (int kt = 0; kt < n_kv; kt++) {
    int kv0 = kt * BN;
    int k_valid = min(BN, S - kv0);

    // ---- Load K,V tile ----
    {
      const int4* gpk = reinterpret_cast<const int4*>(Kbh + (int64_t)kv0 * D);
      const int4* gpv = reinterpret_cast<const int4*>(Vbh + (int64_t)kv0 * D);
      int4* spk = reinterpret_cast<int4*>(sK);
      int4* spv = reinterpret_cast<int4*>(sV);
      int total = BN * (D / 8);
      for (int idx = tid; idx < total; idx += THREADS) {
        int row = idx >> 4;
        int c16 = idx & 15;
        if (row < k_valid) {
          spk[idx] = gpk[(int64_t)row * 16 + c16];
          spv[idx] = gpv[(int64_t)row * 16 + c16];
        } else {
          int4 z; z.x=z.y=z.z=z.w=0; spk[idx]=z; spv[idx]=z;
        }
      }
    }
    __syncthreads();

    // ---- GEMM1: S = Q @ K^T ----
    float sreg[4][8];
    #pragma unroll
    for (int r = 0; r < 4; r++)
      #pragma unroll
      for (int c = 0; c < 8; c++) sreg[r][c] = 0.f;

    #pragma unroll 1
    for (int d0 = 0; d0 < D; d0 += 8) {
      float qf[4][8];
      #pragma unroll
      for (int r = 0; r < 4; r++) {
        int4 t = *reinterpret_cast<const int4*>(&sQ[(mid*4+r)*D + d0]);
        __nv_bfloat16* p = reinterpret_cast<__nv_bfloat16*>(&t);
        #pragma unroll
        for (int dd = 0; dd < 8; dd++) qf[r][dd] = __bfloat162float(p[dd]);
      }
      #pragma unroll
      for (int c = 0; c < 8; c++) {
        int4 t = *reinterpret_cast<const int4*>(&sK[(nid*8+c)*D + d0]);
        __nv_bfloat16* p = reinterpret_cast<__nv_bfloat16*>(&t);
        #pragma unroll
        for (int dd = 0; dd < 8; dd++) {
          float kk = __bfloat162float(p[dd]);
          #pragma unroll
          for (int r = 0; r < 4; r++) sreg[r][c] += qf[r][dd] * kk;
        }
      }
    }

    // scale + mask
    #pragma unroll
    for (int r = 0; r < 4; r++)
      #pragma unroll
      for (int c = 0; c < 8; c++) {
        int j = nid*8 + c;
        float v = sreg[r][c] * scale;
        if (j >= k_valid) v = -INFINITY;
        sreg[r][c] = v;
      }

    // ---- online softmax ----
    #pragma unroll
    for (int r = 0; r < 4; r++) {
      float tmax = -INFINITY;
      #pragma unroll
      for (int c = 0; c < 8; c++) tmax = fmaxf(tmax, sreg[r][c]);
      tmax = redMax8(tmax);
      float mold = m_[r];
      float mnew = fmaxf(mold, tmax);
      float corr = __expf(mold - mnew);
      l_[r] *= corr;
      #pragma unroll
      for (int e = 0; e < 16; e++) acc[r][e] *= corr;
      float psum = 0.f;
      #pragma unroll
      for (int c = 0; c < 8; c++) {
        float p = __expf(sreg[r][c] - mnew);
        psum += p;
        sP[(mid*4+r)*BN + (nid*8+c)] = p;
      }
      psum = redSum8(psum);
      l_[r] += psum;
      m_[r] = mnew;
    }
    __syncthreads();

    // ---- GEMM2: acc += P @ V ----
    #pragma unroll 1
    for (int j = 0; j < BN; j++) {
      float pr[4];
      #pragma unroll
      for (int r = 0; r < 4; r++) pr[r] = sP[(mid*4+r)*BN + j];
      int4 v0 = *reinterpret_cast<const int4*>(&sV[j*D + nid*16]);
      int4 v1 = *reinterpret_cast<const int4*>(&sV[j*D + nid*16 + 8]);
      __nv_bfloat16* p0 = reinterpret_cast<__nv_bfloat16*>(&v0);
      __nv_bfloat16* p1 = reinterpret_cast<__nv_bfloat16*>(&v1);
      #pragma unroll
      for (int e = 0; e < 8; e++) {
        float vv = __bfloat162float(p0[e]);
        #pragma unroll
        for (int r = 0; r < 4; r++) acc[r][e] += pr[r] * vv;
      }
      #pragma unroll
      for (int e = 0; e < 8; e++) {
        float vv = __bfloat162float(p1[e]);
        #pragma unroll
        for (int r = 0; r < 4; r++) acc[r][8+e] += pr[r] * vv;
      }
    }
    __syncthreads();
  }

  // ---- write O and LSE ----
  #pragma unroll
  for (int r = 0; r < 4; r++) {
    int gi = q0 + mid*4 + r;
    if (gi < S) {
      float linv = 1.0f / l_[r];
      #pragma unroll
      for (int half = 0; half < 2; half++) {
        union { int4 v; __nv_bfloat16 hbuf[8]; } out;
        #pragma unroll
        for (int e2 = 0; e2 < 8; e2++)
          out.hbuf[e2] = __float2bfloat16(acc[r][half*8 + e2] * linv);
        int d0 = nid*16 + half*8;
        *reinterpret_cast<int4*>(&O[((int64_t)(b*H+h)*S + gi)*D + d0]) = out.v;
      }
      if (nid == 0) {
        LSE[(int64_t)(b*H+h)*S + gi] = m_[r] + logf(l_[r]);
      }
    }
  }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int B = (int)Q.size(0);
  int H = (int)Q.size(1);
  int S = (int)Q.size(2);
  int Dd = (int)Q.size(3);

  const __nv_bfloat16* q = static_cast<const __nv_bfloat16*>(Q.data_ptr());
  const __nv_bfloat16* k = static_cast<const __nv_bfloat16*>(K.data_ptr());
  const __nv_bfloat16* v = static_cast<const __nv_bfloat16*>(V.data_ptr());
  __nv_bfloat16* o = static_cast<__nv_bfloat16*>(O.data_ptr());
  float* lse = static_cast<float*>(LSE.data_ptr());

  float scale = 1.0f / sqrtf((float)Dd);

  dim3 grid((S + BM - 1) / BM, B * H);
  dim3 block(THREADS);
  size_t shmem = (size_t)(BM*D + BN*D + BN*D) * sizeof(__nv_bfloat16)
               + (size_t)(BM*BN) * sizeof(float);

  CUDA_CHECK(cudaFuncSetAttribute(attn_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem));

  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

  attn_kernel<<<grid, block, shmem, stream>>>(q, k, v, o, lse, B, H, S, scale);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha::run);

}  // namespace mha