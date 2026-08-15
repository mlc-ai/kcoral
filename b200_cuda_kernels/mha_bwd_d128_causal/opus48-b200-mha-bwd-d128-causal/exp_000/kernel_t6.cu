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

__device__ __forceinline__ void load_tile_h(__half* sh, const bf16* g, int row0, int S){
  int tid=threadIdx.x;
  const int4* g4=(const int4*)g; int4* sh4=(int4*)sh;
  int fr=S-row0; if(fr>64)fr=64;
  #pragma unroll
  for(int idx=tid; idx<1024; idx+=256){
    int row=idx>>4, ch=idx&15;
    int4 raw = (row<fr)? g4[(int64_t)(row0+row)*16+ch] : make_int4(0,0,0,0);
    const bf16* b=(const bf16*)&raw;
    __half h[8];
    #pragma unroll
    for(int i=0;i<8;i++) h[i]=__float2half(b2f(b[i]));
    sh4[idx]=*(int4*)h;
  }
}

// D_i = sum_c dO[i,c] * O[i,c]
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

// ---------------- dK, dV kernel (emits dS half to global) ----------------
__global__ void __launch_bounds__(256) dkv_kernel(
    const bf16* __restrict__ Qa, const bf16* __restrict__ Ka,
    const bf16* __restrict__ Va, const bf16* __restrict__ dOa,
    const float* __restrict__ La, const float* __restrict__ Da,
    bf16* __restrict__ dKa, bf16* __restrict__ dVa,
    __half* __restrict__ dSall, int triPairs,
    int S, float scale)
{
  int kb=blockIdx.x, bh=blockIdx.y, numBlk=gridDim.x;
  int kv0=kb*64;
  int64_t base=(int64_t)bh*S*128, lbase=(int64_t)bh*S;
  const bf16* Qg=Qa+base; const bf16* Kg=Ka+base; const bf16* Vg=Va+base; const bf16* dOg=dOa+base;
  const float* Lg=La+lbase; const float* Dg=Da+lbase;
  bf16* dKg=dKa+base; bf16* dVg=dVa+base;

  extern __shared__ char smem[];
  __half* Ksh =(__half*)(smem+0);
  __half* Vsh =(__half*)(smem+16384);
  __half* Qsh =(__half*)(smem+32768);
  __half* dOsh=(__half*)(smem+49152);
  float*  Ssh =(float*)(smem+65536);
  float*  dPsh=(float*)(smem+81920);
  __half* Psh =(__half*)(smem+65536);
  __half* dSsh=(__half*)(smem+73728);
  float*  Lsh =(float*)(smem+98304);
  float*  Dsh =(float*)(smem+98560);
  float*  stageF=(float*)(smem+0);

  int tid=threadIdx.x, warp=tid>>5, ct=warp;

  load_tile_h(Ksh, Kg, kv0, S);
  load_tile_h(Vsh, Vg, kv0, S);

  wmma::fragment<wmma::accumulator,16,16,16,float> accDV[4], accDK[4];
  #pragma unroll
  for(int jt=0;jt<4;jt++){ wmma::fill_fragment(accDV[jt],0.0f); wmma::fill_fragment(accDK[jt],0.0f); }

  for(int qb=kb; qb<numBlk; qb++){
    int q0=qb*64;
    load_tile_h(Qsh, Qg, q0, S);
    load_tile_h(dOsh, dOg, q0, S);
    for(int r=tid;r<64;r+=256){ int gs=q0+r; Lsh[r]=(gs<S)?Lg[gs]:0.f; Dsh[r]=(gs<S)?Dg[gs]:0.f; }
    __syncthreads();

    // Phase 1: S=Q@K^T (warps0-3), dP=dO@V^T (warps4-7)
    if(warp<4){
      int it=warp;
      wmma::fragment<wmma::accumulator,16,16,16,float> acc[4];
      #pragma unroll
      for(int jt=0;jt<4;jt++) wmma::fill_fragment(acc[jt],0.0f);
      #pragma unroll
      for(int kt=0;kt<8;kt++){
        wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> a;
        wmma::load_matrix_sync(a, Qsh + it*2048 + kt*16, 128);
        #pragma unroll
        for(int jt=0;jt<4;jt++){
          wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> b;
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
        wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> a;
        wmma::load_matrix_sync(a, dOsh + it*2048 + kt*16, 128);
        #pragma unroll
        for(int jt=0;jt<4;jt++){
          wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> b;
          wmma::load_matrix_sync(b, Vsh + jt*2048 + kt*16, 128);
          wmma::mma_sync(acc[jt],a,b,acc[jt]);
        }
      }
      #pragma unroll
      for(int jt=0;jt<4;jt++) wmma::store_matrix_sync(dPsh + it*16*64 + jt*16, acc[jt],64, wmma::mem_row_major);
    }
    __syncthreads();

    // Phase 2: softmax grad -> fp16; also store dS to global
    int64_t bp = ((int64_t)bh*triPairs + ((int64_t)qb*(qb+1)/2 + kb))*4096;
    float sReg[16], pReg[16];
    { int c=0; for(int idx=tid; idx<4096; idx+=256){ sReg[c]=Ssh[idx]; pReg[c]=dPsh[idx]; c++; } }
    __syncthreads();
    { int c=0;
      for(int idx=tid; idx<4096; idx+=256){
        int i=idx>>6, j=idx&63; int gi=q0+i, gj=kv0+j;
        float p, ds;
        if(gi<S && gj<S && gj<=gi){ p=__expf(sReg[c]*scale-Lsh[i]); ds=p*(pReg[c]-Dsh[i]); }
        else { p=0.f; ds=0.f; }
        __half ph=__float2half(p), dh=__float2half(ds);
        Psh[idx]=ph; dSsh[idx]=dh; dSall[bp+idx]=dh;
        c++;
      }
    }
    __syncthreads();

    // Phase 3: dV += P^T@dO ; dK += dS^T@Q
    #pragma unroll
    for(int kt=0;kt<4;kt++){
      wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::row_major> bdO, bQ;
      wmma::load_matrix_sync(bdO, dOsh + kt*2048 + ct*16, 128);
      wmma::load_matrix_sync(bQ,  Qsh  + kt*2048 + ct*16, 128);
      #pragma unroll
      for(int jt=0;jt<4;jt++){
        wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::col_major> aP, adS;
        wmma::load_matrix_sync(aP,  Psh + kt*1024 + jt*16, 64);
        wmma::mma_sync(accDV[jt], aP, bdO, accDV[jt]);
        wmma::load_matrix_sync(adS, dSsh + kt*1024 + jt*16, 64);
        wmma::mma_sync(accDK[jt], adS, bQ, accDK[jt]);
      }
    }
    __syncthreads();
  }

  #pragma unroll
  for(int jt=0;jt<4;jt++) wmma::store_matrix_sync(stageF + jt*16*128 + ct*16, accDV[jt],128, wmma::mem_row_major);
  __syncthreads();
  for(int idx=tid; idx<8192; idx+=256){
    int r=idx>>7, c=idx&127; int gs=kv0+r;
    if(gs<S) dVg[(int64_t)gs*128+c]=f2b(stageF[idx]);
  }
  __syncthreads();
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

// ---------------- dQ kernel (reads precomputed dS half) ----------------
__global__ void __launch_bounds__(256) dq_kernel(
    const bf16* __restrict__ Ka,
    const __half* __restrict__ dSall, int triPairs,
    bf16* __restrict__ dQa,
    int S, float scale)
{
  int qb=blockIdx.x, bh=blockIdx.y;
  int q0=qb*64;
  int64_t base=(int64_t)bh*S*128;
  const bf16* Kg=Ka+base;
  bf16* dQg=dQa+base;

  extern __shared__ char smem[];
  __half* Ksh =(__half*)(smem+0);
  __half* dSsh=(__half*)(smem+16384);
  float*  stageF=(float*)(smem+0);

  int tid=threadIdx.x, warp=tid>>5, ct=warp;

  wmma::fragment<wmma::accumulator,16,16,16,float> accDQ[4];
  #pragma unroll
  for(int it=0;it<4;it++) wmma::fill_fragment(accDQ[it],0.0f);

  for(int kb=0; kb<=qb; kb++){
    int kv0=kb*64;
    load_tile_h(Ksh, Kg, kv0, S);
    int64_t bp = ((int64_t)bh*triPairs + ((int64_t)qb*(qb+1)/2 + kb))*4096;
    const int4* src=(const int4*)(dSall+bp); int4* dst=(int4*)dSsh;
    for(int idx=tid; idx<512; idx+=256) dst[idx]=src[idx];
    __syncthreads();

    // dQ += dS@K
    #pragma unroll
    for(int jt=0;jt<4;jt++){
      wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::row_major> bK;
      wmma::load_matrix_sync(bK, Ksh + jt*2048 + ct*16, 128);
      #pragma unroll
      for(int it=0;it<4;it++){
        wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> aS;
        wmma::load_matrix_sync(aS, dSsh + it*16*64 + jt*16, 64);
        wmma::mma_sync(accDQ[it], aS, bK, accDQ[it]);
      }
    }
    __syncthreads();
  }

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

static __half* g_dS=nullptr; static size_t g_dSsz=0;
static float* g_delta=nullptr; static size_t g_deltasz=0;

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV)
{
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int B=(int)Q.size(0), H=(int)Q.size(1), S=(int)Q.size(2), d=(int)Q.size(3);
  int BH=B*H;
  int numBlk=(S+63)/64;
  int triPairs=numBlk*(numBlk+1)/2;
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

  size_t need_delta=(size_t)BH*S*sizeof(float);
  if(need_delta>g_deltasz){ if(g_delta) cudaFree(g_delta); CUDA_CHECK(cudaMalloc(&g_delta,need_delta)); g_deltasz=need_delta; }
  size_t need_dS=(size_t)BH*triPairs*4096*sizeof(__half);
  if(need_dS>g_dSsz){ if(g_dS) cudaFree(g_dS); CUDA_CHECK(cudaMalloc(&g_dS,need_dS)); g_dSsz=need_dS; }
  float* delta=g_delta; __half* dSall=g_dS;

  int64_t numRows=(int64_t)BH*S;
  int dblk=256, wpb=dblk/32;
  int64_t dgrid=(numRows+wpb-1)/wpb;
  delta_kernel<<<(unsigned)dgrid,dblk,0,stream>>>(dOp,Op,delta,numRows,d);
  CUDA_CHECK(cudaGetLastError());

  size_t sh_dkv = 98816;
  size_t sh_dq  = 32768;
  CUDA_CHECK(cudaFuncSetAttribute((const void*)dkv_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh_dkv));
  CUDA_CHECK(cudaFuncSetAttribute((const void*)dq_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh_dq));

  dim3 grid((unsigned)numBlk,(unsigned)BH);
  dkv_kernel<<<grid,256,sh_dkv,stream>>>(Qp,Kp,Vp,dOp,Lp,delta,dKp,dVp,dSall,triPairs,S,scale);
  CUDA_CHECK(cudaGetLastError());
  dq_kernel<<<grid,256,sh_dq,stream>>>(Kp,dSall,triPairs,dQp,S,scale);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_d128_causal::run);

}  // namespace mha_bwd_d128_causal