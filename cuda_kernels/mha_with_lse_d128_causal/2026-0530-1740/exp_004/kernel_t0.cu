#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <math.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

using namespace nvcuda;
using bf16 = __nv_bfloat16;

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__);} } while(0)

namespace mha_kernel {

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int D  = 128;

// Shared memory byte offsets
constexpr int OFF_Q  = 0;
constexpr int OFF_K  = OFF_Q + BM*D*2;     // 16384
constexpr int OFF_V  = OFF_K + BN*D*2;     // 32768
constexpr int OFF_S  = OFF_V + BN*D*2;     // 49152
constexpr int OFF_P  = OFF_S + BM*BN*4;    // 65536
constexpr int OFF_O  = OFF_P + BM*BN*2;    // 73728
constexpr int OFF_MR = OFF_O + BM*D*4;     // 106496
constexpr int OFF_LR = OFF_MR + BM*4;      // 106752
constexpr int OFF_SC = OFF_LR + BM*4;      // 107008
constexpr int SHMEM_BYTES = OFF_SC + BM*4; // 107264

__global__ __launch_bounds__(128) void mha_kernel_fn(
    const bf16* __restrict__ Q, const bf16* __restrict__ K,
    const bf16* __restrict__ V, bf16* __restrict__ O,
    float* __restrict__ LSE, int B, int H, int S) {

  extern __shared__ __align__(16) char smem[];
  bf16*  Qsh   = reinterpret_cast<bf16*>(smem + OFF_Q);
  bf16*  Ksh   = reinterpret_cast<bf16*>(smem + OFF_K);
  bf16*  Vsh   = reinterpret_cast<bf16*>(smem + OFF_V);
  float* Ssh   = reinterpret_cast<float*>(smem + OFF_S);
  bf16*  Psh   = reinterpret_cast<bf16*>(smem + OFF_P);
  float* Oacc  = reinterpret_cast<float*>(smem + OFF_O);
  float* m_run = reinterpret_cast<float*>(smem + OFF_MR);
  float* l_run = reinterpret_cast<float*>(smem + OFF_LR);
  float* sc    = reinterpret_cast<float*>(smem + OFF_SC);

  int tid  = threadIdx.x;
  int warp = tid >> 5;
  int qb   = blockIdx.x;
  int h    = blockIdx.y;
  int b    = blockIdx.z;
  int qs   = qb * BM;

  const float scale = 0.08838834764831843f; // 1/sqrt(128)

  long bh = (long)(b*H + h);
  const bf16* Qbase = Q + bh*(long)S*D;
  const bf16* Kbase = K + bh*(long)S*D;
  const bf16* Vbase = V + bh*(long)S*D;
  bf16* Obase = O + bh*(long)S*D;
  float* LSEbase = LSE + bh*(long)S;

  // init accumulators
  for (int i = tid; i < BM*D; i += 128) Oacc[i] = 0.0f;
  if (tid < BM) { m_run[tid] = -1e30f; l_run[tid] = 0.0f; }

  // load Q tile (persists across KV blocks)
  for (int vi = tid; vi < BM*D/8; vi += 128) {
    int row = vi / (D/8);
    int col = (vi % (D/8)) * 8;
    int gq  = qs + row;
    float4 val;
    if (gq < S) val = *reinterpret_cast<const float4*>(Qbase + (long)gq*D + col);
    else        val = make_float4(0,0,0,0);
    *reinterpret_cast<float4*>(Qsh + row*D + col) = val;
  }
  __syncthreads();

  int q_max_row = (qs + BM - 1 < S - 1) ? (qs + BM - 1) : (S - 1);
  int kvb_max   = q_max_row / BN;

  for (int kvb = 0; kvb <= kvb_max; ++kvb) {
    int kv_start = kvb * BN;

    // load K, V tiles (zero-pad OOB)
    for (int vi = tid; vi < BN*D/8; vi += 128) {
      int row = vi / (D/8);
      int col = (vi % (D/8)) * 8;
      int gk  = kv_start + row;
      float4 vk, vv;
      if (gk < S) {
        vk = *reinterpret_cast<const float4*>(Kbase + (long)gk*D + col);
        vv = *reinterpret_cast<const float4*>(Vbase + (long)gk*D + col);
      } else { vk = make_float4(0,0,0,0); vv = make_float4(0,0,0,0); }
      *reinterpret_cast<float4*>(Ksh + row*D + col) = vk;
      *reinterpret_cast<float4*>(Vsh + row*D + col) = vv;
    }
    __syncthreads();

    // ---- QK^T : warp `warp` handles rows [16*warp, +16) ----
    {
      wmma::fragment<wmma::accumulator,16,16,16,float> cS[BN/16];
      for (int j=0;j<BN/16;j++) wmma::fill_fragment(cS[j], 0.0f);
      for (int kk=0; kk<D/16; kk++) {
        wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::row_major> a;
        wmma::load_matrix_sync(a, Qsh + warp*16*D + kk*16, D);
        for (int j=0;j<BN/16;j++) {
          wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::col_major> bb;
          wmma::load_matrix_sync(bb, Ksh + j*16*D + kk*16, D);
          wmma::mma_sync(cS[j], a, bb, cS[j]);
        }
      }
      for (int j=0;j<BN/16;j++)
        wmma::store_matrix_sync(Ssh + warp*16*BN + j*16, cS[j], BN, wmma::mem_row_major);
    }
    __syncthreads();

    // ---- softmax per row (online) ----
    if (tid < BM) {
      int r  = tid;
      int gq = qs + r;
      if (gq >= S) {
        for (int c=0;c<BN;c++) Psh[r*BN+c] = __float2bfloat16(0.0f);
        sc[r] = 0.0f;
      } else {
        float m_prev = m_run[r];
        float l_prev = l_run[r];
        float bmax = -1e30f;
        for (int c=0;c<BN;c++) {
          int gk = kv_start + c;
          float s;
          if (gk <= gq && gk < S) s = Ssh[r*BN+c] * scale;
          else                    s = -1e30f;
          Ssh[r*BN+c] = s;
          bmax = fmaxf(bmax, s);
        }
        float m_new = fmaxf(m_prev, bmax);
        float corr  = __expf(m_prev - m_new);   // -> 0 on first block (m_prev=-1e30)
        float bl = 0.0f;
        for (int c=0;c<BN;c++) {
          float p = __expf(Ssh[r*BN+c] - m_new); // masked (-1e30) -> 0
          Psh[r*BN+c] = __float2bfloat16(p);
          bl += p;
        }
        l_run[r] = l_prev*corr + bl;
        m_run[r] = m_new;
        sc[r]    = corr;
      }
    }
    __syncthreads();

    // ---- rescale O accumulator ----
    for (int i = tid; i < BM*D; i += 128) {
      int r = i / D;
      Oacc[i] *= sc[r];
    }
    __syncthreads();

    // ---- P @ V (accumulate into Oacc) ----
    {
      for (int j=0;j<D/16;j++) {
        wmma::fragment<wmma::accumulator,16,16,16,float> cO;
        wmma::load_matrix_sync(cO, Oacc + warp*16*D + j*16, D, wmma::mem_row_major);
        for (int kk=0; kk<BN/16; kk++) {
          wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::row_major> a;
          wmma::load_matrix_sync(a, Psh + warp*16*BN + kk*16, BN);
          wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::row_major> bb;
          wmma::load_matrix_sync(bb, Vsh + kk*16*D + j*16, D);
          wmma::mma_sync(cO, a, bb, cO);
        }
        wmma::store_matrix_sync(Oacc + warp*16*D + j*16, cO, D, wmma::mem_row_major);
      }
    }
    __syncthreads();
  }

  // ---- write O (normalized) ----
  for (int i = tid; i < BM*D; i += 128) {
    int r = i / D;
    int d = i % D;
    int gq = qs + r;
    if (gq < S) {
      float inv = 1.0f / l_run[r];
      Obase[(long)gq*D + d] = __float2bfloat16(Oacc[i]*inv);
    }
  }
  // ---- write LSE ----
  if (tid < BM) {
    int gq = qs + tid;
    if (gq < S) LSEbase[gq] = m_run[tid] + logf(l_run[tid]);
  }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int Bv = Q.size(0), Hv = Q.size(1), Sv = Q.size(2);

  const bf16* Qp = static_cast<const bf16*>(Q.data_ptr());
  const bf16* Kp = static_cast<const bf16*>(K.data_ptr());
  const bf16* Vp = static_cast<const bf16*>(V.data_ptr());
  bf16* Op = static_cast<bf16*>(O.data_ptr());
  float* Lp = static_cast<float*>(LSE.data_ptr());

  int num_qb = (Sv + BM - 1) / BM;
  dim3 grid(num_qb, Hv, Bv);
  dim3 block(128);

  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

  CUDA_CHECK(cudaFuncSetAttribute(mha_kernel_fn,
      cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_BYTES));

  mha_kernel_fn<<<grid, block, SHMEM_BYTES, stream>>>(Qp, Kp, Vp, Op, Lp, Bv, Hv, Sv);
  CUDA_CHECK(cudaGetLastError());
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_kernel::run);

}  // namespace mha_kernel