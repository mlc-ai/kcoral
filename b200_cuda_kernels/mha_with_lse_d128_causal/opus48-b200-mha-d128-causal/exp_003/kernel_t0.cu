#include <cuda_bf16.h>
#include <cuda_runtime.h>
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

namespace mha_kernel_ns {

constexpr int D  = 128;
constexpr int BM = 64;
constexpr int BN = 64;

// Shared memory layout (dynamic):
//   Qs [0,16384)  Ks [16384,32768)  Vs [32768,49152)
//   Ss [49152,65536) float  Os [65536,98304) float  Ps [98304,106496) bf16
__global__ void __launch_bounds__(128, 2) mha_fwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int S) {

  extern __shared__ char smem[];
  __nv_bfloat16* Qs = reinterpret_cast<__nv_bfloat16*>(smem + 0);
  __nv_bfloat16* Ks = reinterpret_cast<__nv_bfloat16*>(smem + 16384);
  __nv_bfloat16* Vs = reinterpret_cast<__nv_bfloat16*>(smem + 32768);
  float*         Ss = reinterpret_cast<float*>(smem + 49152);
  float*         Os = reinterpret_cast<float*>(smem + 65536);
  __nv_bfloat16* Ps = reinterpret_cast<__nv_bfloat16*>(smem + 98304);

  __shared__ float corr_sh[BM];
  __shared__ float inv_l_sh[BM];

  int tid     = threadIdx.x;
  int warp_id = tid >> 5;
  int lane    = tid & 31;

  int b = blockIdx.z;
  int h = blockIdx.y;
  int q_start = blockIdx.x * BM;

  long hb        = (long)b * gridDim.y + h;
  long head_base = hb * (long)S * D;
  const __nv_bfloat16* Qh = Q + head_base + (long)q_start * D;
  const __nv_bfloat16* Kh = K + head_base;
  const __nv_bfloat16* Vh = V + head_base;
  __nv_bfloat16* Oh = O + head_base + (long)q_start * D;
  float* LSEh = LSE + hb * (long)S + q_start;

  const float scale = 0.08838834764831845f; // 1/sqrt(128)

  // ---- Load Q tile ----
  {
    const int4* Qh4 = reinterpret_cast<const int4*>(Qh);
    int4* Qs4 = reinterpret_cast<int4*>(Qs);
    const int units = BM * (D/8);
    for (int i = tid; i < units; i += blockDim.x) {
      int r = i / (D/8);
      int c = i % (D/8);
      int row = q_start + r;
      int4 v = make_int4(0,0,0,0);
      if (row < S) v = Qh4[(long)r*(D/8) + c];
      Qs4[i] = v;
    }
  }
  // ---- init Os = 0 ----
  {
    float4* Os4 = reinterpret_cast<float4*>(Os);
    const int units = BM * D / 4;
    for (int i = tid; i < units; i += blockDim.x) Os4[i] = make_float4(0,0,0,0);
  }

  float m_i = -INFINITY;
  float l_i = 0.0f;

  int kv_end = (q_start + BM < S) ? (q_start + BM) : S;

  for (int kv_start = 0; kv_start < kv_end; kv_start += BN) {
    __syncthreads();
    // ---- Load K,V tiles ----
    {
      const int4* Kh4 = reinterpret_cast<const int4*>(Kh);
      const int4* Vh4 = reinterpret_cast<const int4*>(Vh);
      int4* Ks4 = reinterpret_cast<int4*>(Ks);
      int4* Vs4 = reinterpret_cast<int4*>(Vs);
      const int units = BN * (D/8);
      for (int i = tid; i < units; i += blockDim.x) {
        int r = i / (D/8);
        int c = i % (D/8);
        int key = kv_start + r;
        int4 kv = make_int4(0,0,0,0);
        int4 vv = make_int4(0,0,0,0);
        if (key < S) { kv = Kh4[(long)key*(D/8)+c]; vv = Vh4[(long)key*(D/8)+c]; }
        Ks4[i] = kv;
        Vs4[i] = vv;
      }
    }
    __syncthreads();

    // ---- S = Q @ K^T ----
    {
      wmma::fragment<wmma::accumulator,16,16,16,float> acc_s[BN/16];
      #pragma unroll
      for (int n=0;n<BN/16;n++) wmma::fill_fragment(acc_s[n], 0.0f);
      #pragma unroll
      for (int kk=0; kk<D/16; kk++) {
        wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> fa;
        wmma::load_matrix_sync(fa, &Qs[(warp_id*16)*D + kk*16], D);
        #pragma unroll
        for (int n=0;n<BN/16;n++) {
          wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> fb;
          wmma::load_matrix_sync(fb, &Ks[(n*16)*D + kk*16], D);
          wmma::mma_sync(acc_s[n], fa, fb, acc_s[n]);
        }
      }
      #pragma unroll
      for (int n=0;n<BN/16;n++)
        wmma::store_matrix_sync(&Ss[(warp_id*16)*BN + n*16], acc_s[n], BN, wmma::mem_row_major);
    }
    __syncthreads();

    // ---- Softmax (lanes 0..15 per warp, one row each) ----
    if (lane < 16) {
      int r = lane;
      int row_local  = warp_id*16 + r;
      int row_global = q_start + row_local;
      float* srow = &Ss[row_local*BN];
      float local_max = -INFINITY;
      #pragma unroll
      for (int c=0;c<BN;c++) {
        int key = kv_start + c;
        float s = (key <= row_global) ? srow[c]*scale : -INFINITY;
        srow[c] = s;
        local_max = fmaxf(local_max, s);
      }
      float m_new = fmaxf(m_i, local_max);
      float corr  = __expf(m_i - m_new);
      float local_sum = 0.0f;
      __nv_bfloat16* prow = &Ps[row_local*BN];
      #pragma unroll
      for (int c=0;c<BN;c++) {
        float p = __expf(srow[c] - m_new);
        prow[c] = __float2bfloat16(p);
        local_sum += p;
      }
      l_i = l_i * corr + local_sum;
      m_i = m_new;
      corr_sh[row_local] = corr;
    }
    __syncthreads();

    // ---- Scale Os by correction (cooperative) ----
    {
      float4* Os4 = reinterpret_cast<float4*>(Os);
      const int units = BM * D / 4;
      for (int i = tid; i < units; i += blockDim.x) {
        int r = i / (D/4);
        float c = corr_sh[r];
        float4 v = Os4[i];
        v.x*=c; v.y*=c; v.z*=c; v.w*=c;
        Os4[i] = v;
      }
    }
    __syncthreads();

    // ---- O += P @ V ----
    {
      wmma::fragment<wmma::accumulator,16,16,16,float> acc_o[D/16];
      #pragma unroll
      for (int n=0;n<D/16;n++)
        wmma::load_matrix_sync(acc_o[n], &Os[(warp_id*16)*D + n*16], D, wmma::mem_row_major);
      #pragma unroll
      for (int kk=0; kk<BN/16; kk++) {
        wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> fp;
        wmma::load_matrix_sync(fp, &Ps[(warp_id*16)*BN + kk*16], BN);
        #pragma unroll
        for (int n=0;n<D/16;n++) {
          wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> fv;
          wmma::load_matrix_sync(fv, &Vs[(kk*16)*D + n*16], D);
          wmma::mma_sync(acc_o[n], fp, fv, acc_o[n]);
        }
      }
      #pragma unroll
      for (int n=0;n<D/16;n++)
        wmma::store_matrix_sync(&Os[(warp_id*16)*D + n*16], acc_o[n], D, wmma::mem_row_major);
    }
  }

  __syncthreads();

  // ---- finalize: inv_l and LSE ----
  if (lane < 16) {
    int r = lane;
    int row_local  = warp_id*16 + r;
    int row_global = q_start + row_local;
    float inv = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
    inv_l_sh[row_local] = inv;
    if (row_global < S) {
      LSEh[row_local] = m_i + __logf(l_i);
    }
  }
  __syncthreads();

  // ---- write O (normalized, coalesced) ----
  {
    const int units = BM * (D/8);
    for (int i = tid; i < units; i += blockDim.x) {
      int r = i / (D/8);
      int c = i % (D/8);
      int row = q_start + r;
      if (row >= S) continue;
      float inv = inv_l_sh[r];
      const float* op = &Os[r*D + c*8];
      __nv_bfloat16 out8[8];
      #pragma unroll
      for (int k=0;k<8;k++) out8[k] = __float2bfloat16(op[k]*inv);
      *reinterpret_cast<int4*>(Oh + (long)r*D + c*8) = *reinterpret_cast<const int4*>(out8);
    }
  }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int B = (int)Q.size(0);
  int H = (int)Q.size(1);
  int S = (int)Q.size(2);
  if (S == 0) return;

  const __nv_bfloat16* Qp = static_cast<const __nv_bfloat16*>(Q.data_ptr());
  const __nv_bfloat16* Kp = static_cast<const __nv_bfloat16*>(K.data_ptr());
  const __nv_bfloat16* Vp = static_cast<const __nv_bfloat16*>(V.data_ptr());
  __nv_bfloat16* Op = static_cast<__nv_bfloat16*>(O.data_ptr());
  float* LSEp = static_cast<float*>(LSE.data_ptr());

  const int smem = 106496;
  CUDA_CHECK(cudaFuncSetAttribute(mha_fwd_kernel,
             cudaFuncAttributeMaxDynamicSharedMemorySize, smem));

  dim3 grid((S + BM - 1)/BM, H, B);
  dim3 block(128);
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

  mha_fwd_kernel<<<grid, block, smem, stream>>>(Qp, Kp, Vp, Op, LSEp, S);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_kernel_ns::run);

}  // namespace mha_kernel_ns