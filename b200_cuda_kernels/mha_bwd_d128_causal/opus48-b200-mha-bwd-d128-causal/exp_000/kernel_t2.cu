#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <mma.h>
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
    }                                                              \
} while(0)

namespace mha_bwd_d128_causal {

using namespace nvcuda;
typedef __nv_bfloat16 bf16;

__device__ __forceinline__ float b2f(bf16 x){ return __bfloat162float(x); }
__device__ __forceinline__ bf16  f2b(float x){ return __float2bfloat16(x); }
__device__ __forceinline__ void split_bf16(float x, bf16& hi, bf16& lo){
  hi = f2b(x);
  float r = x - __bfloat162float(hi);
  lo = f2b(r);
}

// D_i = sum_c dO[i,c] * O[i,c]  (one warp per row, d=128)
__global__ void delta_kernel(const bf16* __restrict__ dO, const bf16* __restrict__ O,
                             float* __restrict__ delta, int64_t numRows, int d){
  int64_t warp = ((int64_t)blockIdx.x*blockDim.x + threadIdx.x) >> 5;
  int lane = threadIdx.x & 31;
  if(warp >= numRows) return;
  const bf16* dOp = dO + warp*(int64_t)d;
  const bf16* Op  = O  + warp*(int64_t)d;
  float s = 0.f;
  for(int c=lane; c<d; c+=32) s += b2f(dOp[c]) * b2f(Op[c]);
  #pragma unroll
  for(int off=16; off>0; off>>=1) s += __shfl_down_sync(0xffffffffu, s, off);
  if(lane==0) delta[warp] = s;
}

__device__ __forceinline__ void load_tile(const bf16* __restrict__ g, bf16* sh, int row0, int S){
  int tid = threadIdx.x;
  const int4* g4 = reinterpret_cast<const int4*>(g);
  int4* sh4 = reinterpret_cast<int4*>(sh);
  #pragma unroll
  for(int idx=tid; idx<1024; idx+=256){
    int r = idx >> 4;
    int cchunk = idx & 15;
    int gs = row0 + r;
    int4 val;
    if(gs < S) val = g4[(int64_t)gs*16 + cchunk];
    else       val = make_int4(0,0,0,0);
    sh4[idx] = val;
  }
}

// ---------------- dK, dV kernel ----------------
__global__ void __launch_bounds__(256) dkv_kernel(
    const bf16* __restrict__ Qa, const bf16* __restrict__ Ka,
    const bf16* __restrict__ Va, const bf16* __restrict__ dOa,
    const float* __restrict__ La, const float* __restrict__ Da,
    bf16* __restrict__ dKa, bf16* __restrict__ dVa,
    int S, float scale)
{
  int kb=blockIdx.x, bh=blockIdx.y, numBlk=gridDim.x;
  int kv0=kb*64;
  int64_t base=(int64_t)bh*S*128, lbase=(int64_t)bh*S;
  const bf16* Qg=Qa+base; const bf16* Kg=Ka+base; const bf16* Vg=Va+base; const bf16* dOg=dOa+base;
  const float* Lg=La+lbase; const float* Dg=Da+lbase;
  bf16* dKg=dKa+base; bf16* dVg=dVa+base;

  extern __shared__ char smem[];
  bf16*  Ksh   =(bf16*)(smem+0);
  bf16*  Vsh   =(bf16*)(smem+16384);
  bf16*  Qsh   =(bf16*)(smem+32768);
  bf16*  dOsh  =(bf16*)(smem+49152);
  float* Ssh   =(float*)(smem+65536);
  float* dPsh  =(float*)(smem+81920);
  bf16*  Psh_hi=(bf16*)(smem+98304);
  bf16*  Psh_lo=(bf16*)(smem+106496);
  bf16*  dSh   =(bf16*)(smem+114688);
  bf16*  dSl   =(bf16*)(smem+122880);
  float* Lsh   =(float*)(smem+131072);
  float* Dsh   =(float*)(smem+131328);
  float* stageF = Ssh;

  int tid=threadIdx.x, warp=tid>>5, ct=warp;

  load_tile(Kg, Ksh, kv0, S);
  load_tile(Vg, Vsh, kv0, S);

  wmma::fragment<wmma::accumulator,16,16,16,float> accDV[4], accDK[4];
  #pragma unroll
  for(int jt=0;jt<4;jt++){ wmma::fill_fragment(accDV[jt],0.0f); wmma::fill_fragment(accDK[jt],0.0f); }
  __syncthreads();

  for(int qb=kb; qb<numBlk; qb++){
    int q0=qb*64;
    load_tile(Qg,Qsh,q0,S);
    load_tile(dOg,dOsh,q0,S);
    for(int r=tid;r<64;r+=256){ int gs=q0+r; Lsh[r]=(gs<S)?Lg[gs]:0.f; Dsh[r]=(gs<S)?Dg[gs]:0.f; }
    __syncthreads();

    // Phase 1: S = Q@K^T (warps 0-3) ; dP = dO@V^T (warps 4-7)
    if(warp<4){
      int it=warp;
      wmma::fragment<wmma::accumulator,16,16,16,float> acc[4];
      #pragma unroll
      for(int jt=0;jt<4;jt++) wmma::fill_fragment(acc[jt],0.0f);
      #pragma unroll
      for(int kt=0;kt<8;kt++){
        wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::row_major> a;
        wmma::load_matrix_sync(a, Qsh + it*2048 + kt*16, 128);
        #pragma unroll
        for(int jt=0;jt<4;jt++){
          wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::col_major> b;
          wmma::load_matrix_sync(b, Ksh + jt*2048 + kt*16, 128);
          wmma::mma_sync(acc[jt],a,b,acc[jt]);
        }
      }
      #pragma unroll
      for(int jt=0;jt<4;jt++) wmma::store_matrix_sync(Ssh + it*16*64 + jt*16, acc[jt],64, wmma::mem_row_major);
    } else {
      int it=warp-4;
      wmma::fragment<wmma::accumulator,16,16,16,float> acc[4];
      #pragma unroll
      for(int jt=0;jt<4;jt++) wmma::fill_fragment(acc[jt],0.0f);
      #pragma unroll
      for(int kt=0;kt<8;kt++){
        wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::row_major> a;
        wmma::load_matrix_sync(a, dOsh + it*2048 + kt*16, 128);
        #pragma unroll
        for(int jt=0;jt<4;jt++){
          wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::col_major> b;
          wmma::load_matrix_sync(b, Vsh + jt*2048 + kt*16, 128);
          wmma::mma_sync(acc[jt],a,b,acc[jt]);
        }
      }
      #pragma unroll
      for(int jt=0;jt<4;jt++) wmma::store_matrix_sync(dPsh + it*16*64 + jt*16, acc[jt],64, wmma::mem_row_major);
    }
    __syncthreads();

    // Phase 2: softmax grad, split into hi/lo bf16
    for(int idx=tid; idx<4096; idx+=256){
      int i=idx>>6, j=idx&63;
      int gi=q0+i, gj=kv0+j;
      float p, ds;
      if(gi<S && gj<S && gj<=gi){
        p = __expf(Ssh[idx]*scale - Lsh[i]);
        ds = p*(dPsh[idx]-Dsh[i]);
      } else { p=0.f; ds=0.f; }
      bf16 ph,pl,dh,dl;
      split_bf16(p, ph, pl);
      split_bf16(ds, dh, dl);
      Psh_hi[idx]=ph; Psh_lo[idx]=pl;
      dSh[idx]=dh;    dSl[idx]=dl;
    }
    __syncthreads();

    // Phase 3: dV += P^T@dO ; dK += dS^T@Q  (2x bf16 precision)
    #pragma unroll
    for(int kt=0;kt<4;kt++){
      wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::row_major> bdO, bQ;
      wmma::load_matrix_sync(bdO, dOsh + kt*2048 + ct*16, 128);
      wmma::load_matrix_sync(bQ,  Qsh  + kt*2048 + ct*16, 128);
      #pragma unroll
      for(int jt=0;jt<4;jt++){
        wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::col_major> aP, adS;
        wmma::load_matrix_sync(aP,  Psh_hi + kt*1024 + jt*16, 64);
        wmma::mma_sync(accDV[jt], aP, bdO, accDV[jt]);
        wmma::load_matrix_sync(aP,  Psh_lo + kt*1024 + jt*16, 64);
        wmma::mma_sync(accDV[jt], aP, bdO, accDV[jt]);
        wmma::load_matrix_sync(adS, dSh + kt*1024 + jt*16, 64);
        wmma::mma_sync(accDK[jt], adS, bQ, accDK[jt]);
        wmma::load_matrix_sync(adS, dSl + kt*1024 + jt*16, 64);
        wmma::mma_sync(accDK[jt], adS, bQ, accDK[jt]);
      }
    }
    __syncthreads();
  }

  // store dV
  #pragma unroll
  for(int jt=0;jt<4;jt++) wmma::store_matrix_sync(stageF + jt*16*128 + ct*16, accDV[jt],128, wmma::mem_row_major);
  __syncthreads();
  for(int idx=tid; idx<8192; idx+=256){
    int r=idx>>7, c=idx&127; int gs=kv0+r;
    if(gs<S) dVg[(int64_t)gs*128+c]=f2b(stageF[idx]);
  }
  __syncthreads();
  // store dK (scaled)
  #pragma unroll
  for(int jt=0;jt<4;jt++){
    #pragma unroll
    for(int e=0;e<accDK[jt].num_elements;e++) accDK[jt].x[e]*=scale;
    wmma::store_matrix_sync(stageF + jt*16*128 + ct*16, accDK[jt],128, wmma::mem_row_major);
  }
  __syncthreads();
  for(int idx=tid; idx<8192; idx+=256){
    int r=idx>>7, c=idx&127; int gs=kv0+r;
    if(gs<S) dKg[(int64_t)gs*128+c]=f2b(stageF[idx]);
  }
}

// ---------------- dQ kernel ----------------
__global__ void __launch_bounds__(256) dq_kernel(
    const bf16* __restrict__ Qa, const bf16* __restrict__ Ka,
    const bf16* __restrict__ Va, const bf16* __restrict__ dOa,
    const float* __restrict__ La, const float* __restrict__ Da,
    bf16* __restrict__ dQa,
    int S, float scale)
{
  int qb=blockIdx.x, bh=blockIdx.y;
  int q0=qb*64;
  int64_t base=(int64_t)bh*S*128, lbase=(int64_t)bh*S;
  const bf16* Qg=Qa+base; const bf16* Kg=Ka+base; const bf16* Vg=Va+base; const bf16* dOg=dOa+base;
  const float* Lg=La+lbase; const float* Dg=Da+lbase;
  bf16* dQg=dQa+base;

  extern __shared__ char smem[];
  bf16*  Ksh  =(bf16*)(smem+0);
  bf16*  Vsh  =(bf16*)(smem+16384);
  bf16*  Qsh  =(bf16*)(smem+32768);
  bf16*  dOsh =(bf16*)(smem+49152);
  float* Ssh  =(float*)(smem+65536);
  float* dPsh =(float*)(smem+81920);
  bf16*  dSh  =(bf16*)(smem+98304);
  bf16*  dSl  =(bf16*)(smem+106496);
  float* Lsh  =(float*)(smem+114688);
  float* Dsh  =(float*)(smem+114944);
  float* stageF = Ssh;

  int tid=threadIdx.x, warp=tid>>5, ct=warp;

  load_tile(Qg,Qsh,q0,S);
  load_tile(dOg,dOsh,q0,S);
  for(int r=tid;r<64;r+=256){ int gs=q0+r; Lsh[r]=(gs<S)?Lg[gs]:0.f; Dsh[r]=(gs<S)?Dg[gs]:0.f; }

  wmma::fragment<wmma::accumulator,16,16,16,float> accDQ[4];
  #pragma unroll
  for(int it=0;it<4;it++) wmma::fill_fragment(accDQ[it],0.0f);
  __syncthreads();

  for(int kb=0; kb<=qb; kb++){
    int kv0=kb*64;
    load_tile(Kg,Ksh,kv0,S);
    load_tile(Vg,Vsh,kv0,S);
    __syncthreads();

    // Phase 1: S, dP
    if(warp<4){
      int it=warp;
      wmma::fragment<wmma::accumulator,16,16,16,float> acc[4];
      #pragma unroll
      for(int jt=0;jt<4;jt++) wmma::fill_fragment(acc[jt],0.0f);
      #pragma unroll
      for(int kt=0;kt<8;kt++){
        wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::row_major> a;
        wmma::load_matrix_sync(a, Qsh + it*2048 + kt*16, 128);
        #pragma unroll
        for(int jt=0;jt<4;jt++){
          wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::col_major> b;
          wmma::load_matrix_sync(b, Ksh + jt*2048 + kt*16, 128);
          wmma::mma_sync(acc[jt],a,b,acc[jt]);
        }
      }
      #pragma unroll
      for(int jt=0;jt<4;jt++) wmma::store_matrix_sync(Ssh + it*16*64 + jt*16, acc[jt],64, wmma::mem_row_major);
    } else {
      int it=warp-4;
      wmma::fragment<wmma::accumulator,16,16,16,float> acc[4];
      #pragma unroll
      for(int jt=0;jt<4;jt++) wmma::fill_fragment(acc[jt],0.0f);
      #pragma unroll
      for(int kt=0;kt<8;kt++){
        wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::row_major> a;
        wmma::load_matrix_sync(a, dOsh + it*2048 + kt*16, 128);
        #pragma unroll
        for(int jt=0;jt<4;jt++){
          wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::col_major> b;
          wmma::load_matrix_sync(b, Vsh + jt*2048 + kt*16, 128);
          wmma::mma_sync(acc[jt],a,b,acc[jt]);
        }
      }
      #pragma unroll
      for(int jt=0;jt<4;jt++) wmma::store_matrix_sync(dPsh + it*16*64 + jt*16, acc[jt],64, wmma::mem_row_major);
    }
    __syncthreads();

    // Phase 2: dS split
    for(int idx=tid; idx<4096; idx+=256){
      int i=idx>>6, j=idx&63;
      int gi=q0+i, gj=kv0+j;
      float p, ds;
      if(gi<S && gj<S && gj<=gi){
        p = __expf(Ssh[idx]*scale - Lsh[i]);
        ds = p*(dPsh[idx]-Dsh[i]);
      } else { ds=0.f; }
      bf16 dh,dl; split_bf16(ds, dh, dl);
      dSh[idx]=dh; dSl[idx]=dl;
    }
    __syncthreads();

    // Phase 3: dQ += dS@K (2x bf16 precision)
    #pragma unroll
    for(int jt=0;jt<4;jt++){
      wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::row_major> bK;
      wmma::load_matrix_sync(bK, Ksh + jt*2048 + ct*16, 128);
      #pragma unroll
      for(int it=0;it<4;it++){
        wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::row_major> aS;
        wmma::load_matrix_sync(aS, dSh + it*16*64 + jt*16, 64);
        wmma::mma_sync(accDQ[it], aS, bK, accDQ[it]);
        wmma::load_matrix_sync(aS, dSl + it*16*64 + jt*16, 64);
        wmma::mma_sync(accDQ[it], aS, bK, accDQ[it]);
      }
    }
    __syncthreads();
  }

  // store dQ (scaled)
  #pragma unroll
  for(int it=0;it<4;it++){
    #pragma unroll
    for(int e=0;e<accDQ[it].num_elements;e++) accDQ[it].x[e]*=scale;
    wmma::store_matrix_sync(stageF + it*16*128 + ct*16, accDQ[it],128, wmma::mem_row_major);
  }
  __syncthreads();
  for(int idx=tid; idx<8192; idx+=256){
    int r=idx>>7, c=idx&127; int gs=q0+r;
    if(gs<S) dQg[(int64_t)gs*128+c]=f2b(stageF[idx]);
  }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV)
{
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int B=(int)Q.size(0), H=(int)Q.size(1), S=(int)Q.size(2), d=(int)Q.size(3);
  int BH=B*H;
  int numBlk=(S+63)/64;
  float scale=1.0f/sqrtf((float)d);

  const bf16* Qp=(const bf16*)Q.data_ptr();
  const bf16* Kp=(const bf16*)K.data_ptr();
  const bf16* Vp=(const bf16*)V.data_ptr();
  const bf16* Op=(const bf16*)O.data_ptr();
  const bf16* dOp=(const bf16*)dO.data_ptr();
  const float* Lp=(const float*)L.data_ptr();
  bf16* dQp=(bf16*)dQ.data_ptr();
  bf16* dKp=(bf16*)dK.data_ptr();
  bf16* dVp=(bf16*)dV.data_ptr();

  cudaStream_t stream=(cudaStream_t)TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id);

  float* delta=nullptr;
  CUDA_CHECK(cudaMalloc(&delta,(size_t)BH*S*sizeof(float)));

  int64_t numRows=(int64_t)BH*S;
  int dblk=256, wpb=dblk/32;
  int64_t dgrid=(numRows+wpb-1)/wpb;
  delta_kernel<<<(unsigned)dgrid,dblk,0,stream>>>(dOp,Op,delta,numRows,d);
  CUDA_CHECK(cudaGetLastError());

  size_t sh_dkv = 131584;
  size_t sh_dq  = 115200;
  CUDA_CHECK(cudaFuncSetAttribute((const void*)dkv_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh_dkv));
  CUDA_CHECK(cudaFuncSetAttribute((const void*)dq_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh_dq));

  dim3 grid((unsigned)numBlk,(unsigned)BH);
  dkv_kernel<<<grid,256,sh_dkv,stream>>>(Qp,Kp,Vp,dOp,Lp,delta,dKp,dVp,S,scale);
  CUDA_CHECK(cudaGetLastError());
  dq_kernel<<<grid,256,sh_dq,stream>>>(Qp,Kp,Vp,dOp,Lp,delta,dQp,S,scale);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaFree(delta));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_d128_causal::run);

}  // namespace mha_bwd_d128_causal