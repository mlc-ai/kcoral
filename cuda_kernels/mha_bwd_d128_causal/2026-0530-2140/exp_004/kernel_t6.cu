#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); exit(1);} } while(0)

namespace mha_bwd {
using tvm::ffi::TensorView;

constexpr int D=128;
constexpr int LD=136;
constexpr int LDP=72;

__device__ __forceinline__ uint32_t su32(const void* p){ return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void ldm_x4(uint32_t a[4],uint32_t addr){
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
    :"=r"(a[0]),"=r"(a[1]),"=r"(a[2]),"=r"(a[3]):"r"(addr));
}
__device__ __forceinline__ void ldm_x2(uint32_t b[2],uint32_t addr){
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1},[%2];"
    :"=r"(b[0]),"=r"(b[1]):"r"(addr));
}
__device__ __forceinline__ void ldm_x2t(uint32_t b[2],uint32_t addr){
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];"
    :"=r"(b[0]),"=r"(b[1]):"r"(addr));
}
__device__ __forceinline__ void mma(float* c,const uint32_t* a,const uint32_t* b){
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
    :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
    :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
}
__device__ __forceinline__ void ssplit(__nv_bfloat16* H,__nv_bfloat16* Lo,int idx,float v){
  __nv_bfloat16 hi=__float2bfloat16(v);
  H[idx]=hi; Lo[idx]=__float2bfloat16(v-__bfloat162float(hi));
}

__global__ void compute_D_kernel(const __nv_bfloat16* __restrict__ dO,
                                 const __nv_bfloat16* __restrict__ O,
                                 float* __restrict__ Dout,size_t nrows){
  size_t row=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
  size_t stride=(size_t)gridDim.x*blockDim.x;
  for(;row<nrows;row+=stride){
    const __nv_bfloat16* dop=dO+row*D; const __nv_bfloat16* op=O+row*D;
    float acc=0.f;
    #pragma unroll
    for(int e=0;e<D;e+=8){
      uint4 a=*reinterpret_cast<const uint4*>(dop+e);
      uint4 c=*reinterpret_cast<const uint4*>(op+e);
      __nv_bfloat16* pa=reinterpret_cast<__nv_bfloat16*>(&a);
      __nv_bfloat16* pc=reinterpret_cast<__nv_bfloat16*>(&c);
      #pragma unroll
      for(int k=0;k<8;k++) acc+=__bfloat162float(pa[k])*__bfloat162float(pc[k]);
    }
    Dout[row]=acc;
  }
}

__device__ __forceinline__ void load_tile(const __nv_bfloat16* g,int row0,int S,__nv_bfloat16* sm,int tid){
  #pragma unroll
  for(int v=tid; v<64*16; v+=256){
    int r=v>>4, c=(v&15)<<3;
    int gr=row0+r;
    uint4 val=make_uint4(0,0,0,0);
    if(gr<S) val=*reinterpret_cast<const uint4*>(g+(size_t)gr*D+c);
    *reinterpret_cast<uint4*>(sm+r*LD+c)=val;
  }
}

// ---------------- PASS 1: dK, dV ----------------
__global__ __launch_bounds__(256,2) void pass1(
    const __nv_bfloat16* __restrict__ Q,const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,const float* __restrict__ Dv,
    __nv_bfloat16* __restrict__ dK,__nv_bfloat16* __restrict__ dV,
    int B,int H,int S,float scale)
{
  int kb=blockIdx.x,h=blockIdx.y,b=blockIdx.z;
  int nq=(S+63)/64;
  extern __shared__ char smem[];
  __nv_bfloat16* Ks=(__nv_bfloat16*)smem;
  __nv_bfloat16* Vs=Ks+64*LD;
  __nv_bfloat16* Qs=Vs+64*LD;
  __nv_bfloat16* Os=Qs+64*LD;
  __nv_bfloat16* PTh=Os+64*LD;
  __nv_bfloat16* dSTh=PTh+64*LDP;
  __nv_bfloat16* dSTl=dSTh+64*LDP;
  float* Lq=(float*)(dSTl+64*LDP);
  float* Dq=Lq+64;

  int tid=threadIdx.x, lane=tid&31, warp=tid>>5;
  int wm=warp&3, ws=warp>>2;
  int kbw=wm*16, qblk=ws*32, eblk=ws*64;

  size_t bh=((size_t)(b*H+h))*S;
  const __nv_bfloat16* Kb=K+bh*D; const __nv_bfloat16* Vb=V+bh*D;
  const __nv_bfloat16* Qb=Q+bh*D; const __nv_bfloat16* Ob=dO+bh*D;
  const float* Lb=L+bh; const float* Db=Dv+bh;

  load_tile(Kb,kb*64,S,Ks,tid);
  load_tile(Vb,kb*64,S,Vs,tid);

  float dVacc[8][4], dKacc[8][4];
  #pragma unroll
  for(int i=0;i<8;i++)
    #pragma unroll
    for(int j=0;j<4;j++){ dVacc[i][j]=0.f; dKacc[i][j]=0.f; }
  __syncthreads();

  for(int qb=kb; qb<nq; qb++){
    load_tile(Qb,qb*64,S,Qs,tid);
    load_tile(Ob,qb*64,S,Os,tid);
    if(tid<64){ int qg=qb*64+tid; Lq[tid]=qg<S?Lb[qg]:0.f; Dq[tid]=qg<S?Db[qg]:0.f; }
    __syncthreads();

    // Phase A: S^T = K@Q^T -> P (unsplit, stored in PTh)
    float cS[4][4];
    #pragma unroll
    for(int n=0;n<4;n++){cS[n][0]=cS[n][1]=cS[n][2]=cS[n][3]=0.f;}
    #pragma unroll
    for(int kt=0;kt<8;kt++){
      uint32_t aK[4];
      ldm_x4(aK, su32(&Ks[(kbw+lane%16)*LD + kt*16+(lane/16)*8]));
      #pragma unroll
      for(int nt=0;nt<4;nt++){
        uint32_t bQ[2];
        ldm_x2(bQ, su32(&Qs[(qblk+nt*8+lane%8)*LD + kt*16+((lane/8)&1)*8]));
        mma(cS[nt],aK,bQ);
      }
    }
    #pragma unroll
    for(int nt=0;nt<4;nt++){
      int q0=qblk+nt*8+(lane%4)*2;
      int keyA=kbw+lane/4, keyB=keyA+8;
      int kgA=kb*64+keyA, kgB=kb*64+keyB;
      int qg0=qb*64+q0, qg1=qg0+1;
      float p0=(kgA<S&&qg0<S&&kgA<=qg0)?__expf(scale*cS[nt][0]-Lq[q0]):0.f;
      float p1=(kgA<S&&qg1<S&&kgA<=qg1)?__expf(scale*cS[nt][1]-Lq[q0+1]):0.f;
      float p2=(kgB<S&&qg0<S&&kgB<=qg0)?__expf(scale*cS[nt][2]-Lq[q0]):0.f;
      float p3=(kgB<S&&qg1<S&&kgB<=qg1)?__expf(scale*cS[nt][3]-Lq[q0+1]):0.f;
      PTh[keyA*LDP+q0]  =__float2bfloat16(p0);
      PTh[keyA*LDP+q0+1]=__float2bfloat16(p1);
      PTh[keyB*LDP+q0]  =__float2bfloat16(p2);
      PTh[keyB*LDP+q0+1]=__float2bfloat16(p3);
    }
    // Phase B: dP^T = V@dO^T -> dS^T (split). Reload P from PTh (saves regs)
    float cP[4][4];
    #pragma unroll
    for(int n=0;n<4;n++){cP[n][0]=cP[n][1]=cP[n][2]=cP[n][3]=0.f;}
    #pragma unroll
    for(int kt=0;kt<8;kt++){
      uint32_t aV[4];
      ldm_x4(aV, su32(&Vs[(kbw+lane%16)*LD + kt*16+(lane/16)*8]));
      #pragma unroll
      for(int nt=0;nt<4;nt++){
        uint32_t bO[2];
        ldm_x2(bO, su32(&Os[(qblk+nt*8+lane%8)*LD + kt*16+((lane/8)&1)*8]));
        mma(cP[nt],aV,bO);
      }
    }
    #pragma unroll
    for(int nt=0;nt<4;nt++){
      int q0=qblk+nt*8+(lane%4)*2;
      int keyA=kbw+lane/4, keyB=keyA+8;
      float p0=__bfloat162float(PTh[keyA*LDP+q0]);
      float p1=__bfloat162float(PTh[keyA*LDP+q0+1]);
      float p2=__bfloat162float(PTh[keyB*LDP+q0]);
      float p3=__bfloat162float(PTh[keyB*LDP+q0+1]);
      ssplit(dSTh,dSTl,keyA*LDP+q0,  p0*(cP[nt][0]-Dq[q0]));
      ssplit(dSTh,dSTl,keyA*LDP+q0+1,p1*(cP[nt][1]-Dq[q0+1]));
      ssplit(dSTh,dSTl,keyB*LDP+q0,  p2*(cP[nt][2]-Dq[q0]));
      ssplit(dSTh,dSTl,keyB*LDP+q0+1,p3*(cP[nt][3]-Dq[q0+1]));
    }
    __syncthreads();

    // Phase C: dV += P^T @ dO (single bf16 P)
    #pragma unroll
    for(int qt=0;qt<4;qt++){
      uint32_t aPh[4];
      ldm_x4(aPh, su32(&PTh[(kbw+lane%16)*LDP + qt*16+(lane/16)*8]));
      #pragma unroll
      for(int nt=0;nt<8;nt++){
        uint32_t bO[2];
        ldm_x2t(bO, su32(&Os[(qt*16+((lane/8)&1)*8+lane%8)*LD + eblk+nt*8]));
        mma(dVacc[nt],aPh,bO);
      }
    }
    // Phase D: dK += dS^T @ Q (split dS)
    #pragma unroll
    for(int qt=0;qt<4;qt++){
      uint32_t aDh[4],aDl[4];
      ldm_x4(aDh, su32(&dSTh[(kbw+lane%16)*LDP + qt*16+(lane/16)*8]));
      ldm_x4(aDl, su32(&dSTl[(kbw+lane%16)*LDP + qt*16+(lane/16)*8]));
      #pragma unroll
      for(int nt=0;nt<8;nt++){
        uint32_t bQ[2];
        ldm_x2t(bQ, su32(&Qs[(qt*16+((lane/8)&1)*8+lane%8)*LD + eblk+nt*8]));
        mma(dKacc[nt],aDh,bQ);
        mma(dKacc[nt],aDl,bQ);
      }
    }
    __syncthreads();
  }

  #pragma unroll
  for(int nt=0;nt<8;nt++){
    int e0=eblk+nt*8+(lane%4)*2;
    int keyA=kbw+lane/4, keyB=keyA+8;
    int kgA=kb*64+keyA, kgB=kb*64+keyB;
    if(kgA<S){
      dV[(bh+kgA)*D+e0]=__float2bfloat16(dVacc[nt][0]);
      dV[(bh+kgA)*D+e0+1]=__float2bfloat16(dVacc[nt][1]);
      dK[(bh+kgA)*D+e0]=__float2bfloat16(scale*dKacc[nt][0]);
      dK[(bh+kgA)*D+e0+1]=__float2bfloat16(scale*dKacc[nt][1]);
    }
    if(kgB<S){
      dV[(bh+kgB)*D+e0]=__float2bfloat16(dVacc[nt][2]);
      dV[(bh+kgB)*D+e0+1]=__float2bfloat16(dVacc[nt][3]);
      dK[(bh+kgB)*D+e0]=__float2bfloat16(scale*dKacc[nt][2]);
      dK[(bh+kgB)*D+e0+1]=__float2bfloat16(scale*dKacc[nt][3]);
    }
  }
}

// ---------------- PASS 2: dQ ----------------
__global__ __launch_bounds__(256,2) void pass2(
    const __nv_bfloat16* __restrict__ Q,const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,const float* __restrict__ Dv,
    __nv_bfloat16* __restrict__ dQ,
    int B,int H,int S,float scale)
{
  int qb=blockIdx.x,h=blockIdx.y,b=blockIdx.z;
  extern __shared__ char smem[];
  __nv_bfloat16* Qs=(__nv_bfloat16*)smem;
  __nv_bfloat16* Os=Qs+64*LD;
  __nv_bfloat16* Ks=Os+64*LD;
  __nv_bfloat16* Vs=Ks+64*LD;
  __nv_bfloat16* dSh=Vs+64*LD;
  __nv_bfloat16* dSl=dSh+64*LDP;
  float* Lq=(float*)(dSl+64*LDP);
  float* Dq=Lq+64;

  int tid=threadIdx.x, lane=tid&31, warp=tid>>5;
  int wm=warp&3, ws=warp>>2;
  int qbw=wm*16, kblk=ws*32, eblk=ws*64;

  size_t bh=((size_t)(b*H+h))*S;
  const __nv_bfloat16* Qb=Q+bh*D; const __nv_bfloat16* Ob=dO+bh*D;
  const __nv_bfloat16* Kb=K+bh*D; const __nv_bfloat16* Vb=V+bh*D;
  const float* Lb=L+bh; const float* Db=Dv+bh;

  load_tile(Qb,qb*64,S,Qs,tid);
  load_tile(Ob,qb*64,S,Os,tid);
  if(tid<64){ int qg=qb*64+tid; Lq[tid]=qg<S?Lb[qg]:0.f; Dq[tid]=qg<S?Db[qg]:0.f; }

  float dQacc[8][4];
  #pragma unroll
  for(int i=0;i<8;i++)
    #pragma unroll
    for(int j=0;j<4;j++) dQacc[i][j]=0.f;
  __syncthreads();

  for(int kb=0; kb<=qb; kb++){
    load_tile(Kb,kb*64,S,Ks,tid);
    load_tile(Vb,kb*64,S,Vs,tid);
    __syncthreads();

    // Phase A: S = Q@K^T -> P (in regs)
    float cS[4][4];
    #pragma unroll
    for(int n=0;n<4;n++){cS[n][0]=cS[n][1]=cS[n][2]=cS[n][3]=0.f;}
    #pragma unroll
    for(int kt=0;kt<8;kt++){
      uint32_t aQ[4];
      ldm_x4(aQ, su32(&Qs[(qbw+lane%16)*LD + kt*16+(lane/16)*8]));
      #pragma unroll
      for(int nt=0;nt<4;nt++){
        uint32_t bK[2];
        ldm_x2(bK, su32(&Ks[(kblk+nt*8+lane%8)*LD + kt*16+((lane/8)&1)*8]));
        mma(cS[nt],aQ,bK);
      }
    }
    float Preg[4][4];
    #pragma unroll
    for(int nt=0;nt<4;nt++){
      int k0=kblk+nt*8+(lane%4)*2;
      int qA=qbw+lane/4, qB=qA+8;
      int qgA=qb*64+qA, qgB=qb*64+qB;
      int kg0=kb*64+k0, kg1=kg0+1;
      Preg[nt][0]=(qgA<S&&kg0<S&&kg0<=qgA)?__expf(scale*cS[nt][0]-Lq[qA]):0.f;
      Preg[nt][1]=(qgA<S&&kg1<S&&kg1<=qgA)?__expf(scale*cS[nt][1]-Lq[qA]):0.f;
      Preg[nt][2]=(qgB<S&&kg0<S&&kg0<=qgB)?__expf(scale*cS[nt][2]-Lq[qB]):0.f;
      Preg[nt][3]=(qgB<S&&kg1<S&&kg1<=qgB)?__expf(scale*cS[nt][3]-Lq[qB]):0.f;
    }
    // Phase B: dP = dO@V^T -> dS (split)
    float cP[4][4];
    #pragma unroll
    for(int n=0;n<4;n++){cP[n][0]=cP[n][1]=cP[n][2]=cP[n][3]=0.f;}
    #pragma unroll
    for(int kt=0;kt<8;kt++){
      uint32_t aO[4];
      ldm_x4(aO, su32(&Os[(qbw+lane%16)*LD + kt*16+(lane/16)*8]));
      #pragma unroll
      for(int nt=0;nt<4;nt++){
        uint32_t bV[2];
        ldm_x2(bV, su32(&Vs[(kblk+nt*8+lane%8)*LD + kt*16+((lane/8)&1)*8]));
        mma(cP[nt],aO,bV);
      }
    }
    #pragma unroll
    for(int nt=0;nt<4;nt++){
      int k0=kblk+nt*8+(lane%4)*2;
      int qA=qbw+lane/4, qB=qA+8;
      ssplit(dSh,dSl,qA*LDP+k0,  Preg[nt][0]*(cP[nt][0]-Dq[qA]));
      ssplit(dSh,dSl,qA*LDP+k0+1,Preg[nt][1]*(cP[nt][1]-Dq[qA]));
      ssplit(dSh,dSl,qB*LDP+k0,  Preg[nt][2]*(cP[nt][2]-Dq[qB]));
      ssplit(dSh,dSl,qB*LDP+k0+1,Preg[nt][3]*(cP[nt][3]-Dq[qB]));
    }
    __syncthreads();

    // Phase C: dQ += dS @ K (split dS)
    #pragma unroll
    for(int kt=0;kt<4;kt++){
      uint32_t aSh[4],aSl[4];
      ldm_x4(aSh, su32(&dSh[(qbw+lane%16)*LDP + kt*16+(lane/16)*8]));
      ldm_x4(aSl, su32(&dSl[(qbw+lane%16)*LDP + kt*16+(lane/16)*8]));
      #pragma unroll
      for(int nt=0;nt<8;nt++){
        uint32_t bK[2];
        ldm_x2t(bK, su32(&Ks[(kt*16+((lane/8)&1)*8+lane%8)*LD + eblk+nt*8]));
        mma(dQacc[nt],aSh,bK);
        mma(dQacc[nt],aSl,bK);
      }
    }
    __syncthreads();
  }

  #pragma unroll
  for(int nt=0;nt<8;nt++){
    int e0=eblk+nt*8+(lane%4)*2;
    int qA=qbw+lane/4, qB=qA+8;
    int qgA=qb*64+qA, qgB=qb*64+qB;
    if(qgA<S){
      dQ[(bh+qgA)*D+e0]=__float2bfloat16(scale*dQacc[nt][0]);
      dQ[(bh+qgA)*D+e0+1]=__float2bfloat16(scale*dQacc[nt][1]);
    }
    if(qgB<S){
      dQ[(bh+qgB)*D+e0]=__float2bfloat16(scale*dQacc[nt][2]);
      dQ[(bh+qgB)*D+e0+1]=__float2bfloat16(scale*dQacc[nt][3]);
    }
  }
}

void run(TensorView Q, TensorView K, TensorView V, TensorView O, TensorView dO, TensorView L,
         TensorView dQ, TensorView dK, TensorView dV) {
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int B=(int)Q.size(0),H=(int)Q.size(1),S=(int)Q.size(2),d=(int)Q.size(3);
  cudaStream_t stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type,Q.device().device_id));

  const __nv_bfloat16* Qp=static_cast<const __nv_bfloat16*>(Q.data_ptr());
  const __nv_bfloat16* Kp=static_cast<const __nv_bfloat16*>(K.data_ptr());
  const __nv_bfloat16* Vp=static_cast<const __nv_bfloat16*>(V.data_ptr());
  const __nv_bfloat16* Op=static_cast<const __nv_bfloat16*>(O.data_ptr());
  const __nv_bfloat16* dOp=static_cast<const __nv_bfloat16*>(dO.data_ptr());
  const float* Lp=static_cast<const float*>(L.data_ptr());
  __nv_bfloat16* dQp=static_cast<__nv_bfloat16*>(dQ.data_ptr());
  __nv_bfloat16* dKp=static_cast<__nv_bfloat16*>(dK.data_ptr());
  __nv_bfloat16* dVp=static_cast<__nv_bfloat16*>(dV.data_ptr());

  float scale=1.0f/sqrtf((float)d);
  size_t nrows=(size_t)B*H*S;

  float* Dvec=nullptr; CUDA_CHECK(cudaMalloc(&Dvec,nrows*sizeof(float)));
  {
    int thr=256; size_t need=(nrows+thr-1)/thr;
    unsigned blk=(unsigned)std::min(need,(size_t)65535u); if(blk==0) blk=1;
    compute_D_kernel<<<blk,thr,0,stream>>>(dOp,Op,Dvec,nrows);
  }

  int nkv=(S+63)/64, nq=(S+63)/64;
  int smem1=(int)((4*64*LD+3*64*LDP)*sizeof(__nv_bfloat16)+2*64*sizeof(float));
  int smem2=(int)((4*64*LD+2*64*LDP)*sizeof(__nv_bfloat16)+2*64*sizeof(float));
  CUDA_CHECK(cudaFuncSetAttribute(pass1,cudaFuncAttributeMaxDynamicSharedMemorySize,smem1));
  CUDA_CHECK(cudaFuncSetAttribute(pass2,cudaFuncAttributeMaxDynamicSharedMemorySize,smem2));

  dim3 g1(nkv,H,B),g2(nq,H,B);
  pass1<<<g1,256,smem1,stream>>>(Qp,Kp,Vp,dOp,Lp,Dvec,dKp,dVp,B,H,S,scale);
  pass2<<<g2,256,smem2,stream>>>(Qp,Kp,Vp,dOp,Lp,Dvec,dQp,B,H,S,scale);

  CUDA_CHECK(cudaStreamSynchronize(stream));
  cudaFree(Dvec);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);
}  // namespace mha_bwd