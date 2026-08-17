#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <math.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace mha_bwd {

using bf16 = __nv_bfloat16;
using namespace nvcuda;
using FragC = wmma::fragment<wmma::accumulator,16,16,16,float>;

constexpr int SMEM_BYTES = 98816;
constexpr int NT = 256; // threads per block (8 warps)

// ---- C[64x64] = X[64x128] @ Y[64x128]^T   (Y accessed col-major) ----
__device__ __forceinline__ void mm_xyt(const bf16* Xbuf, const bf16* Ybuf, float* Cbuf, int warp){
  int mt = warp >> 1;   // 0..3  -> rows mt*16
  int nt = warp & 1;    // 0..1  -> cols nt*32
  wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::row_major> a;
  wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::col_major> b0,b1;
  FragC c0,c1; wmma::fill_fragment(c0,0.f); wmma::fill_fragment(c1,0.f);
  #pragma unroll
  for(int k=0;k<128;k+=16){
    wmma::load_matrix_sync(a,  Xbuf + mt*16*128 + k, 128);
    wmma::load_matrix_sync(b0, Ybuf + k + (nt*32)*128, 128);
    wmma::load_matrix_sync(b1, Ybuf + k + (nt*32+16)*128, 128);
    wmma::mma_sync(c0,a,b0,c0);
    wmma::mma_sync(c1,a,b1,c1);
  }
  wmma::store_matrix_sync(Cbuf + mt*16*64 + nt*32,    c0, 64, wmma::mem_row_major);
  wmma::store_matrix_sync(Cbuf + mt*16*64 + nt*32+16, c1, 64, wmma::mem_row_major);
}

// ---- acc[64x128] += A^T @ B  ; A is 64x64 stored row-major, loaded col-major (=> A^T), B is 64x128 row-major ----
__device__ __forceinline__ void mm_acc_colA(const bf16* Abuf, const bf16* Bbuf, FragC (&acc)[2][2], int warp){
  int wm = warp >> 2;  // 0..1 rows wm*32
  int wn = warp & 3;   // 0..3 cols wn*32
  wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::col_major> a0,a1;
  wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::row_major> b0,b1;
  #pragma unroll
  for(int k=0;k<64;k+=16){
    wmma::load_matrix_sync(a0, Abuf + (wm*32)    + k*64, 64);
    wmma::load_matrix_sync(a1, Abuf + (wm*32+16) + k*64, 64);
    wmma::load_matrix_sync(b0, Bbuf + k*128 + wn*32,     128);
    wmma::load_matrix_sync(b1, Bbuf + k*128 + wn*32+16,  128);
    wmma::mma_sync(acc[0][0],a0,b0,acc[0][0]);
    wmma::mma_sync(acc[0][1],a0,b1,acc[0][1]);
    wmma::mma_sync(acc[1][0],a1,b0,acc[1][0]);
    wmma::mma_sync(acc[1][1],a1,b1,acc[1][1]);
  }
}

// ---- acc[64x128] += A @ B ; A 64x64 row-major, B 64x128 row-major ----
__device__ __forceinline__ void mm_acc_rowA(const bf16* Abuf, const bf16* Bbuf, FragC (&acc)[2][2], int warp){
  int wm = warp >> 2;
  int wn = warp & 3;
  wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::row_major> a0,a1;
  wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::row_major> b0,b1;
  #pragma unroll
  for(int k=0;k<64;k+=16){
    wmma::load_matrix_sync(a0, Abuf + (wm*32)*64    + k, 64);
    wmma::load_matrix_sync(a1, Abuf + (wm*32+16)*64 + k, 64);
    wmma::load_matrix_sync(b0, Bbuf + k*128 + wn*32,     128);
    wmma::load_matrix_sync(b1, Bbuf + k*128 + wn*32+16,  128);
    wmma::mma_sync(acc[0][0],a0,b0,acc[0][0]);
    wmma::mma_sync(acc[0][1],a0,b1,acc[0][1]);
    wmma::mma_sync(acc[1][0],a1,b0,acc[1][0]);
    wmma::mma_sync(acc[1][1],a1,b1,acc[1][1]);
  }
}

__device__ __forceinline__ void store_acc(FragC (&acc)[2][2], float* staging, int warp){
  int wm = warp >> 2;
  int wn = warp & 3;
  wmma::store_matrix_sync(staging + (wm*32)*128    + wn*32,    acc[0][0], 128, wmma::mem_row_major);
  wmma::store_matrix_sync(staging + (wm*32)*128    + wn*32+16, acc[0][1], 128, wmma::mem_row_major);
  wmma::store_matrix_sync(staging + (wm*32+16)*128 + wn*32,    acc[1][0], 128, wmma::mem_row_major);
  wmma::store_matrix_sync(staging + (wm*32+16)*128 + wn*32+16, acc[1][1], 128, wmma::mem_row_major);
}

// ---- delta: D_i = sum_d O_id * dO_id ----
__global__ void compute_delta(const bf16* O, const bf16* dO, float* Dg, long long rows){
  int warp = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  long long row = (long long)blockIdx.x*(blockDim.x>>5) + warp;
  if(row >= rows) return;
  const bf16* o = O  + row*128;
  const bf16* g = dO + row*128;
  float s = 0.f;
  #pragma unroll
  for(int k=lane;k<128;k+=32) s += __bfloat162float(o[k])*__bfloat162float(g[k]);
  #pragma unroll
  for(int off=16;off>0;off>>=1) s += __shfl_down_sync(0xffffffffu, s, off);
  if(lane==0) Dg[row] = s;
}

// ============ dK, dV kernel ============
__global__ void bwd_dkdv_kernel(const bf16* Q,const bf16* K,const bf16* V,const bf16* dO,
                                const float* Lg,const float* Dg,bf16* dK,bf16* dV,int S,float scale){
  extern __shared__ char smem[];
  bf16*  Kbuf  = (bf16*)(smem);
  bf16*  Vbuf  = (bf16*)(smem+16384);
  bf16*  Qbuf  = (bf16*)(smem+32768);
  bf16*  dObuf = (bf16*)(smem+49152);
  float* Sbuf  = (float*)(smem+65536);
  bf16*  Pbuf  = (bf16*)(smem+81920);
  bf16*  dSbuf = (bf16*)(smem+90112);
  float* Lbuf  = (float*)(smem+98304);
  float* Dbuf  = (float*)(smem+98560);
  float* staging = (float*)(smem+32768); // reuse Q/dO region

  int bh  = blockIdx.y;
  int kv0 = blockIdx.x*64;
  int warp = threadIdx.x >> 5;

  const bf16* Kbh = K + (size_t)bh*S*128;
  const bf16* Vbh = V + (size_t)bh*S*128;
  const bf16* Qbh = Q + (size_t)bh*S*128;
  const bf16* dObh= dO+ (size_t)bh*S*128;
  const float* Lbh= Lg + (size_t)bh*S;
  const float* Dbh= Dg + (size_t)bh*S;

  // load K,V
  for(int idx=threadIdx.x; idx<64*128; idx+=NT){
    int r=idx>>7, c=idx&127, gr=kv0+r;
    Kbuf[idx] = (gr<S)? Kbh[(size_t)gr*128+c] : __float2bfloat16(0.f);
    Vbuf[idx] = (gr<S)? Vbh[(size_t)gr*128+c] : __float2bfloat16(0.f);
  }
  __syncthreads();

  FragC dVacc[2][2], dKacc[2][2];
  #pragma unroll
  for(int i=0;i<2;i++)
    #pragma unroll
    for(int j=0;j<2;j++){ wmma::fill_fragment(dVacc[i][j],0.f); wmma::fill_fragment(dKacc[i][j],0.f); }

  int nq=(S+63)/64;
  for(int qt=0; qt<nq; ++qt){
    int q0=qt*64;
    for(int idx=threadIdx.x; idx<64*128; idx+=NT){
      int r=idx>>7, c=idx&127, gr=q0+r;
      Qbuf[idx]  = (gr<S)? Qbh[(size_t)gr*128+c]  : __float2bfloat16(0.f);
      dObuf[idx] = (gr<S)? dObh[(size_t)gr*128+c] : __float2bfloat16(0.f);
    }
    for(int idx=threadIdx.x; idx<64; idx+=NT){
      int gr=q0+idx;
      Lbuf[idx] = (gr<S)? Lbh[gr] : 0.f;
      Dbuf[idx] = (gr<S)? Dbh[gr] : 0.f;
    }
    __syncthreads();

    mm_xyt(Qbuf, Kbuf, Sbuf, warp);      // S = Q@K^T
    __syncthreads();

    for(int idx=threadIdx.x; idx<64*64; idx+=NT){
      int i=idx>>6, j=idx&63;
      float p=0.f;
      if(q0+i<S && kv0+j<S) p = __expf(scale*Sbuf[idx] - Lbuf[i]);
      Pbuf[idx] = __float2bfloat16(p);
    }
    __syncthreads();

    mm_xyt(dObuf, Vbuf, Sbuf, warp);     // dP = dO@V^T
    __syncthreads();

    for(int idx=threadIdx.x; idx<64*64; idx+=NT){
      int i=idx>>6;
      float p  = __bfloat162float(Pbuf[idx]);
      float dp = Sbuf[idx];
      dSbuf[idx] = __float2bfloat16(scale*p*(dp - Dbuf[i]));
    }
    __syncthreads();

    mm_acc_colA(Pbuf,  dObuf, dVacc, warp); // dV += P^T @ dO
    mm_acc_colA(dSbuf, Qbuf,  dKacc, warp); // dK += dS^T @ Q
    __syncthreads();
  }

  // store dV
  store_acc(dVacc, staging, warp);
  __syncthreads();
  for(int idx=threadIdx.x; idx<64*128; idx+=NT){
    int r=idx>>7, c=idx&127, gr=kv0+r;
    if(gr<S) dV[(size_t)bh*S*128 + (size_t)gr*128 + c] = __float2bfloat16(staging[idx]);
  }
  __syncthreads();
  // store dK
  store_acc(dKacc, staging, warp);
  __syncthreads();
  for(int idx=threadIdx.x; idx<64*128; idx+=NT){
    int r=idx>>7, c=idx&127, gr=kv0+r;
    if(gr<S) dK[(size_t)bh*S*128 + (size_t)gr*128 + c] = __float2bfloat16(staging[idx]);
  }
}

// ============ dQ kernel ============
__global__ void bwd_dq_kernel(const bf16* Q,const bf16* K,const bf16* V,const bf16* dO,
                              const float* Lg,const float* Dg,bf16* dQ,int S,float scale){
  extern __shared__ char smem[];
  bf16*  Kbuf  = (bf16*)(smem);
  bf16*  Vbuf  = (bf16*)(smem+16384);
  bf16*  Qbuf  = (bf16*)(smem+32768);
  bf16*  dObuf = (bf16*)(smem+49152);
  float* Sbuf  = (float*)(smem+65536);
  bf16*  Pbuf  = (bf16*)(smem+81920);
  bf16*  dSbuf = (bf16*)(smem+90112);
  float* Lbuf  = (float*)(smem+98304);
  float* Dbuf  = (float*)(smem+98560);
  float* staging = (float*)(smem); // reuse K/V region

  int bh = blockIdx.y;
  int q0 = blockIdx.x*64;
  int warp = threadIdx.x >> 5;

  const bf16* Kbh = K + (size_t)bh*S*128;
  const bf16* Vbh = V + (size_t)bh*S*128;
  const bf16* Qbh = Q + (size_t)bh*S*128;
  const bf16* dObh= dO+ (size_t)bh*S*128;
  const float* Lbh= Lg + (size_t)bh*S;
  const float* Dbh= Dg + (size_t)bh*S;

  for(int idx=threadIdx.x; idx<64*128; idx+=NT){
    int r=idx>>7, c=idx&127, gr=q0+r;
    Qbuf[idx]  = (gr<S)? Qbh[(size_t)gr*128+c]  : __float2bfloat16(0.f);
    dObuf[idx] = (gr<S)? dObh[(size_t)gr*128+c] : __float2bfloat16(0.f);
  }
  for(int idx=threadIdx.x; idx<64; idx+=NT){
    int gr=q0+idx;
    Lbuf[idx] = (gr<S)? Lbh[gr] : 0.f;
    Dbuf[idx] = (gr<S)? Dbh[gr] : 0.f;
  }
  __syncthreads();

  FragC dQacc[2][2];
  #pragma unroll
  for(int i=0;i<2;i++)
    #pragma unroll
    for(int j=0;j<2;j++) wmma::fill_fragment(dQacc[i][j],0.f);

  int nkv=(S+63)/64;
  for(int kt=0; kt<nkv; ++kt){
    int kv0=kt*64;
    for(int idx=threadIdx.x; idx<64*128; idx+=NT){
      int r=idx>>7, c=idx&127, gr=kv0+r;
      Kbuf[idx] = (gr<S)? Kbh[(size_t)gr*128+c] : __float2bfloat16(0.f);
      Vbuf[idx] = (gr<S)? Vbh[(size_t)gr*128+c] : __float2bfloat16(0.f);
    }
    __syncthreads();

    mm_xyt(Qbuf, Kbuf, Sbuf, warp);      // S = Q@K^T
    __syncthreads();

    for(int idx=threadIdx.x; idx<64*64; idx+=NT){
      int i=idx>>6, j=idx&63;
      float p=0.f;
      if(q0+i<S && kv0+j<S) p = __expf(scale*Sbuf[idx] - Lbuf[i]);
      Pbuf[idx] = __float2bfloat16(p);
    }
    __syncthreads();

    mm_xyt(dObuf, Vbuf, Sbuf, warp);     // dP = dO@V^T
    __syncthreads();

    for(int idx=threadIdx.x; idx<64*64; idx+=NT){
      int i=idx>>6;
      float p  = __bfloat162float(Pbuf[idx]);
      float dp = Sbuf[idx];
      dSbuf[idx] = __float2bfloat16(scale*p*(dp - Dbuf[i]));
    }
    __syncthreads();

    mm_acc_rowA(dSbuf, Kbuf, dQacc, warp); // dQ += dS @ K
    __syncthreads();
  }

  store_acc(dQacc, staging, warp);
  __syncthreads();
  for(int idx=threadIdx.x; idx<64*128; idx+=NT){
    int r=idx>>7, c=idx&127, gr=q0+r;
    if(gr<S) dQ[(size_t)bh*S*128 + (size_t)gr*128 + c] = __float2bfloat16(staging[idx]);
  }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV){
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int64_t B = Q.size(0);
  int64_t H = Q.size(1);
  int64_t S = Q.size(2);
  int BH = (int)(B*H);
  float scale = 1.0f/sqrtf(128.0f);

  const bf16* Qp = static_cast<const bf16*>(Q.data_ptr());
  const bf16* Kp = static_cast<const bf16*>(K.data_ptr());
  const bf16* Vp = static_cast<const bf16*>(V.data_ptr());
  const bf16* Op = static_cast<const bf16*>(O.data_ptr());
  const bf16* dOp= static_cast<const bf16*>(dO.data_ptr());
  const float* Lp= static_cast<const float*>(L.data_ptr());
  bf16* dQp = static_cast<bf16*>(dQ.data_ptr());
  bf16* dKp = static_cast<bf16*>(dK.data_ptr());
  bf16* dVp = static_cast<bf16*>(dV.data_ptr());

  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

  long long rows = (long long)BH*S;
  float* Dg = nullptr;
  CUDA_CHECK(cudaMallocAsync(&Dg, sizeof(float)*rows, stream));

  // delta
  {
    int threads = 128;
    long long rpb = threads/32;
    long long blocks = (rows + rpb - 1)/rpb;
    compute_delta<<<(unsigned int)blocks, threads, 0, stream>>>(Op, dOp, Dg, rows);
    CUDA_CHECK(cudaGetLastError());
  }

  static bool attr_set = false;
  if(!attr_set){
    CUDA_CHECK(cudaFuncSetAttribute(bwd_dkdv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
    CUDA_CHECK(cudaFuncSetAttribute(bwd_dq_kernel,   cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
    attr_set = true;
  }

  int nkv = (int)((S+63)/64);
  int nq  = (int)((S+63)/64);

  dim3 g1((unsigned)nkv, (unsigned)BH);
  dim3 g2((unsigned)nq,  (unsigned)BH);

  bwd_dkdv_kernel<<<g1, NT, SMEM_BYTES, stream>>>(Qp,Kp,Vp,dOp,Lp,Dg,dKp,dVp,(int)S,scale);
  CUDA_CHECK(cudaGetLastError());
  bwd_dq_kernel<<<g2, NT, SMEM_BYTES, stream>>>(Qp,Kp,Vp,dOp,Lp,Dg,dQp,(int)S,scale);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaFreeAsync(Dg, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd