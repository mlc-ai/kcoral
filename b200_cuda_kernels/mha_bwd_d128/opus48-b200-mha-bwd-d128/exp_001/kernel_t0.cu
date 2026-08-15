#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <math.h>
#include <cstdio>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
  fprintf(stderr,"CUDA err %s @%s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__);} } while(0)

namespace mha_bwd {

constexpr int DHEAD=128;
constexpr int BM=64;
constexpr int BN=64;

__device__ __forceinline__ uint32_t pack2(__nv_bfloat16 lo,__nv_bfloat16 hi){
  return (uint32_t)__bfloat16_as_ushort(lo) | ((uint32_t)__bfloat16_as_ushort(hi)<<16);
}

template<bool TRANS>
__device__ __forceinline__ void loadA(const __nv_bfloat16* tile,int ld,int m0,int k0,uint32_t a[4],int lane){
  int tg=lane>>2, tin=lane&3;
  #define AE(r,c) (TRANS? tile[(c)*ld+(r)] : tile[(r)*ld+(c)])
  a[0]=pack2(AE(m0+tg,   k0+2*tin),   AE(m0+tg,   k0+2*tin+1));
  a[1]=pack2(AE(m0+tg+8, k0+2*tin),   AE(m0+tg+8, k0+2*tin+1));
  a[2]=pack2(AE(m0+tg,   k0+8+2*tin), AE(m0+tg,   k0+8+2*tin+1));
  a[3]=pack2(AE(m0+tg+8, k0+8+2*tin), AE(m0+tg+8, k0+8+2*tin+1));
  #undef AE
}

template<bool KMAJOR>
__device__ __forceinline__ void loadB(const __nv_bfloat16* tile,int ld,int k0,int n0,uint32_t b[2],int lane){
  int tg=lane>>2, tin=lane&3;
  #define BE(k,n) (KMAJOR? tile[(n)*ld+(k)] : tile[(k)*ld+(n)])
  b[0]=pack2(BE(k0+2*tin,   n0+tg), BE(k0+2*tin+1,   n0+tg));
  b[1]=pack2(BE(k0+8+2*tin, n0+tg), BE(k0+8+2*tin+1, n0+tg));
  #undef BE
}

__device__ __forceinline__ void mma16816(float c[4],const uint32_t a[4],const uint32_t b[2]){
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
    : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
    : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
}

__device__ __forceinline__ void load_tile(__nv_bfloat16* smem,const __nv_bfloat16* gmem,int row0,int nrows,int S,int tid,int nthreads){
  const int vpr=DHEAD/8;
  int total=nrows*vpr;
  for(int idx=tid; idx<total; idx+=nthreads){
    int r=idx/vpr, cc=idx%vpr;
    int gr=row0+r;
    uint4 v;
    if(gr<S) v=*reinterpret_cast<const uint4*>(gmem+(size_t)gr*DHEAD+cc*8);
    else { v.x=v.y=v.z=v.w=0u; }
    *reinterpret_cast<uint4*>(smem+r*DHEAD+cc*8)=v;
  }
}

__device__ __forceinline__ void compute_P_dS(
   const __nv_bfloat16* sQ,const __nv_bfloat16* sK,
   const __nv_bfloat16* sO,const __nv_bfloat16* sV,
   __nv_bfloat16* sP,__nv_bfloat16* sDS,
   const float* sL,const float* sD,
   int q0,int kv0,int S,float scale,int warp,int lane,int nw){
  int tg=lane>>2, tin=lane&3;
  for(int t=warp; t<32; t+=nw){
    int mt=t/8, nt=t%8; int m0=mt*16, n0=nt*8;
    float cs[4]={0,0,0,0}, cp[4]={0,0,0,0};
    #pragma unroll
    for(int k0=0;k0<DHEAD;k0+=16){
      uint32_t a[4],b[2];
      loadA<false>(sQ,DHEAD,m0,k0,a,lane);
      loadB<true>(sK,DHEAD,k0,n0,b,lane);
      mma16816(cs,a,b);
    }
    #pragma unroll
    for(int k0=0;k0<DHEAD;k0+=16){
      uint32_t a[4],b[2];
      loadA<false>(sO,DHEAD,m0,k0,a,lane);
      loadB<true>(sV,DHEAD,k0,n0,b,lane);
      mma16816(cp,a,b);
    }
    #pragma unroll
    for(int e=0;e<4;e++){
      int rr=m0+tg+((e>=2)?8:0);
      int cc=n0+2*tin+(e&1);
      int gq=q0+rr, gk=kv0+cc;
      float pval=0.f, dsval=0.f;
      if(gq<S && gk<S){
        float Lv=sL[rr], Dv=sD[rr];
        pval=__expf(scale*cs[e]-Lv);
        dsval=scale*pval*(cp[e]-Dv);
      }
      sP[rr*BN+cc]=__float2bfloat16(pval);
      sDS[rr*BN+cc]=__float2bfloat16(dsval);
    }
  }
}

__global__ void compute_D_kernel(const __nv_bfloat16* O,const __nv_bfloat16* dO,float* Dout,int total){
  int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<total){
    const __nv_bfloat16* o=O+(size_t)idx*DHEAD;
    const __nv_bfloat16* g=dO+(size_t)idx*DHEAD;
    float s=0.f;
    #pragma unroll
    for(int e=0;e<DHEAD;e++) s+=__bfloat162float(o[e])*__bfloat162float(g[e]);
    Dout[idx]=s;
  }
}

__global__ __launch_bounds__(256,2) void bwd_dkdv_kernel(
   const __nv_bfloat16* Q,const __nv_bfloat16* K,const __nv_bfloat16* V,
   const __nv_bfloat16* dO,const float* L,const float* Del,
   __nv_bfloat16* dK,__nv_bfloat16* dV,int B,int H,int S,float scale){
  int bh=blockIdx.y, kv_tile=blockIdx.x;
  int tid=threadIdx.x, lane=tid&31, warp=tid>>5, nw=blockDim.x>>5;
  int kv0=kv_tile*BN;

  extern __shared__ char smem[];
  __nv_bfloat16* sK=(__nv_bfloat16*)smem;
  __nv_bfloat16* sV=sK+BN*DHEAD;
  __nv_bfloat16* sQ=sV+BN*DHEAD;
  __nv_bfloat16* sO=sQ+BM*DHEAD;
  __nv_bfloat16* sP=sO+BM*DHEAD;
  __nv_bfloat16* sDS=sP+BM*BN;
  float* sL=(float*)(sDS+BM*BN);
  float* sD=sL+BM;

  size_t base=(size_t)bh*S*DHEAD;
  const __nv_bfloat16* Kb=K+base; const __nv_bfloat16* Vb=V+base;
  const __nv_bfloat16* Qb=Q+base; const __nv_bfloat16* Ob=dO+base;
  const float* Lb=L+(size_t)bh*S; const float* Db=Del+(size_t)bh*S;

  load_tile(sK,Kb,kv0,BN,S,tid,blockDim.x);
  load_tile(sV,Vb,kv0,BN,S,tid,blockDim.x);

  float accV[8][4], accK[8][4];
  #pragma unroll
  for(int p=0;p<8;p++)
    #pragma unroll
    for(int e=0;e<4;e++){accV[p][e]=0.f;accK[p][e]=0.f;}

  int nqt=(S+BM-1)/BM;
  for(int qt=0; qt<nqt; qt++){
    int q0=qt*BM;
    __syncthreads();
    load_tile(sQ,Qb,q0,BM,S,tid,blockDim.x);
    load_tile(sO,Ob,q0,BM,S,tid,blockDim.x);
    for(int i=tid;i<BM;i+=blockDim.x){
      int gr=q0+i; sL[i]=(gr<S)?Lb[gr]:0.f; sD[i]=(gr<S)?Db[gr]:0.f;
    }
    __syncthreads();
    compute_P_dS(sQ,sK,sO,sV,sP,sDS,sL,sD,q0,kv0,S,scale,warp,lane,nw);
    __syncthreads();
    #pragma unroll
    for(int p=0;p<8;p++){
      int t=warp+8*p; int mt=t/16, nt=t%16; int m0=mt*16, n0=nt*8;
      #pragma unroll
      for(int k0=0;k0<BM;k0+=16){
        uint32_t a[4],b[2];
        loadA<true>(sP,BN,m0,k0,a,lane);
        loadB<false>(sO,DHEAD,k0,n0,b,lane);
        mma16816(accV[p],a,b);
      }
      #pragma unroll
      for(int k0=0;k0<BM;k0+=16){
        uint32_t a[4],b[2];
        loadA<true>(sDS,BN,m0,k0,a,lane);
        loadB<false>(sQ,DHEAD,k0,n0,b,lane);
        mma16816(accK[p],a,b);
      }
    }
  }

  int tg=lane>>2, tin=lane&3;
  #pragma unroll
  for(int p=0;p<8;p++){
    int t=warp+8*p; int mt=t/16, nt=t%16; int m0=mt*16, n0=nt*8;
    #pragma unroll
    for(int e=0;e<4;e++){
      int rr=m0+tg+((e>=2)?8:0);
      int cc=n0+2*tin+(e&1);
      int gk=kv0+rr;
      if(gk<S){
        dV[base+(size_t)gk*DHEAD+cc]=__float2bfloat16(accV[p][e]);
        dK[base+(size_t)gk*DHEAD+cc]=__float2bfloat16(accK[p][e]);
      }
    }
  }
}

__global__ __launch_bounds__(256,2) void bwd_dq_kernel(
   const __nv_bfloat16* Q,const __nv_bfloat16* K,const __nv_bfloat16* V,
   const __nv_bfloat16* dO,const float* L,const float* Del,
   __nv_bfloat16* dQ,int B,int H,int S,float scale){
  int bh=blockIdx.y, q_tile=blockIdx.x;
  int tid=threadIdx.x, lane=tid&31, warp=tid>>5, nw=blockDim.x>>5;
  int q0=q_tile*BM;

  extern __shared__ char smem[];
  __nv_bfloat16* sQ=(__nv_bfloat16*)smem;
  __nv_bfloat16* sO=sQ+BM*DHEAD;
  __nv_bfloat16* sK=sO+BM*DHEAD;
  __nv_bfloat16* sV=sK+BN*DHEAD;
  __nv_bfloat16* sP=sV+BN*DHEAD;
  __nv_bfloat16* sDS=sP+BM*BN;
  float* sL=(float*)(sDS+BM*BN);
  float* sD=sL+BM;

  size_t base=(size_t)bh*S*DHEAD;
  const __nv_bfloat16* Kb=K+base; const __nv_bfloat16* Vb=V+base;
  const __nv_bfloat16* Qb=Q+base; const __nv_bfloat16* Ob=dO+base;
  const float* Lb=L+(size_t)bh*S; const float* Db=Del+(size_t)bh*S;

  load_tile(sQ,Qb,q0,BM,S,tid,blockDim.x);
  load_tile(sO,Ob,q0,BM,S,tid,blockDim.x);
  for(int i=tid;i<BM;i+=blockDim.x){
    int gr=q0+i; sL[i]=(gr<S)?Lb[gr]:0.f; sD[i]=(gr<S)?Db[gr]:0.f;
  }

  float accQ[8][4];
  #pragma unroll
  for(int p=0;p<8;p++)
    #pragma unroll
    for(int e=0;e<4;e++) accQ[p][e]=0.f;

  int nkt=(S+BN-1)/BN;
  for(int kt=0;kt<nkt;kt++){
    int kv0=kt*BN;
    __syncthreads();
    load_tile(sK,Kb,kv0,BN,S,tid,blockDim.x);
    load_tile(sV,Vb,kv0,BN,S,tid,blockDim.x);
    __syncthreads();
    compute_P_dS(sQ,sK,sO,sV,sP,sDS,sL,sD,q0,kv0,S,scale,warp,lane,nw);
    __syncthreads();
    #pragma unroll
    for(int p=0;p<8;p++){
      int t=warp+8*p; int mt=t/16, nt=t%16; int m0=mt*16, n0=nt*8;
      #pragma unroll
      for(int k0=0;k0<BN;k0+=16){
        uint32_t a[4],b[2];
        loadA<false>(sDS,BN,m0,k0,a,lane);
        loadB<false>(sK,DHEAD,k0,n0,b,lane);
        mma16816(accQ[p],a,b);
      }
    }
  }

  int tg=lane>>2, tin=lane&3;
  #pragma unroll
  for(int p=0;p<8;p++){
    int t=warp+8*p; int mt=t/16, nt=t%16; int m0=mt*16, n0=nt*8;
    #pragma unroll
    for(int e=0;e<4;e++){
      int rr=m0+tg+((e>=2)?8:0);
      int cc=n0+2*tin+(e&1);
      int gq=q0+rr;
      if(gq<S) dQ[base+(size_t)gq*DHEAD+cc]=__float2bfloat16(accQ[p][e]);
    }
  }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV){
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int Bd=(int)Q.size(0), Hd=(int)Q.size(1), Sd=(int)Q.size(2);
  float scale=1.0f/sqrtf((float)DHEAD);
  cudaStream_t stream=(cudaStream_t)TVMFFIEnvGetStream(Q.device().device_type,Q.device().device_id);

  const __nv_bfloat16* Qp=(const __nv_bfloat16*)Q.data_ptr();
  const __nv_bfloat16* Kp=(const __nv_bfloat16*)K.data_ptr();
  const __nv_bfloat16* Vp=(const __nv_bfloat16*)V.data_ptr();
  const __nv_bfloat16* Op=(const __nv_bfloat16*)O.data_ptr();
  const __nv_bfloat16* dOp=(const __nv_bfloat16*)dO.data_ptr();
  const float* Lp=(const float*)L.data_ptr();
  __nv_bfloat16* dQp=(__nv_bfloat16*)dQ.data_ptr();
  __nv_bfloat16* dKp=(__nv_bfloat16*)dK.data_ptr();
  __nv_bfloat16* dVp=(__nv_bfloat16*)dV.data_ptr();

  size_t nrows=(size_t)Bd*Hd*Sd;
  static float* Dbuf=nullptr; static size_t Dcap=0;
  if(nrows>Dcap){ if(Dbuf) cudaFree(Dbuf); CUDA_CHECK(cudaMalloc(&Dbuf,nrows*sizeof(float))); Dcap=nrows; }

  compute_D_kernel<<<(unsigned)((nrows+255)/256),256,0,stream>>>(Op,dOp,Dbuf,(int)nrows);
  CUDA_CHECK(cudaGetLastError());

  int smem=82432;
  cudaFuncSetAttribute(bwd_dkdv_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);
  cudaFuncSetAttribute(bwd_dq_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);

  int nkv=(Sd+BN-1)/BN;
  dim3 g1(nkv, Bd*Hd);
  bwd_dkdv_kernel<<<g1,256,smem,stream>>>(Qp,Kp,Vp,dOp,Lp,Dbuf,dKp,dVp,Bd,Hd,Sd,scale);
  CUDA_CHECK(cudaGetLastError());

  int nq=(Sd+BM-1)/BM;
  dim3 g2(nq, Bd*Hd);
  bwd_dq_kernel<<<g2,256,smem,stream>>>(Qp,Kp,Vp,dOp,Lp,Dbuf,dQp,Bd,Hd,Sd,scale);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

} // namespace mha_bwd