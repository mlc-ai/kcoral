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
constexpr int BM=128;   // query tile (inner-loop rows for dkdv / block rows for dq)
constexpr int BN=64;    // key tile
constexpr int NTHREADS=512;

typedef __nv_bfloat16 bf16;

__device__ __forceinline__ int swz(int row,int col,int LD){
  int chunk=col>>3; int within=col&7;
  return row*LD + ((chunk ^ (row&7))<<3) + within;
}
__device__ __forceinline__ void ld_A(const bf16* s,int m0,int k0,int LD,int lane,uint32_t a[4]){
  int row=(lane&7)+((lane>>3)&1)*8;
  int col=(lane>>4)*8;
  uint32_t addr=(uint32_t)__cvta_generic_to_shared(s+swz(m0+row,k0+col,LD));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
    :"=r"(a[0]),"=r"(a[1]),"=r"(a[2]),"=r"(a[3]):"r"(addr));
}
__device__ __forceinline__ void ld_A_trans(const bf16* s,int Am0,int Ak0,int LD,int lane,uint32_t a[4]){
  int row=Ak0+(lane&7)+((lane>>4)&1)*8;
  int col=Am0+((lane>>3)&1)*8;
  uint32_t addr=(uint32_t)__cvta_generic_to_shared(s+swz(row,col,LD));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3},[%4];"
    :"=r"(a[0]),"=r"(a[1]),"=r"(a[2]),"=r"(a[3]):"r"(addr));
}
__device__ __forceinline__ void ld_B(const bf16* s,int k0,int n0,int LD,int lane,uint32_t b[2]){
  int row=n0+(lane&7);
  int col=k0+((lane>>3)&1)*8;
  uint32_t addr=(uint32_t)__cvta_generic_to_shared(s+swz(row,col,LD));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1},[%2];"
    :"=r"(b[0]),"=r"(b[1]):"r"(addr));
}
__device__ __forceinline__ void ld_B_trans(const bf16* s,int k0,int n0,int LD,int lane,uint32_t b[2]){
  int row=k0+(lane&7)+((lane>>3)&1)*8;
  int col=n0;
  uint32_t addr=(uint32_t)__cvta_generic_to_shared(s+swz(row,col,LD));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];"
    :"=r"(b[0]),"=r"(b[1]):"r"(addr));
}
__device__ __forceinline__ void mma(float c[4],const uint32_t a[4],const uint32_t b[2]){
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
    :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
    :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
}

__device__ __forceinline__ void cp_async16(uint32_t sa,const void* gp){
  asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n"::"r"(sa),"l"(gp));
}
__device__ __forceinline__ void cp_commit(){ asm volatile("cp.async.commit_group;\n"); }
template<int N> __device__ __forceinline__ void cp_wait(){ asm volatile("cp.async.wait_group %0;\n"::"n"(N):"memory"); }

__device__ __forceinline__ void load_tile_async(bf16* s,const bf16* g,int row0,int nrows,int S,int tid,int nth){
  const int CH=DHEAD/8;
  int total=nrows*CH;
  for(int idx=tid; idx<total; idx+=nth){
    int r=idx/CH, c=idx%CH; int gr=row0+r;
    int phys=r*DHEAD + ((c^(r&7))<<3);
    if(gr<S){
      uint32_t sa=(uint32_t)__cvta_generic_to_shared(s+phys);
      cp_async16(sa, g+(size_t)gr*DHEAD + c*8);
    } else { *(uint4*)(s+phys)=make_uint4(0,0,0,0); }
  }
}

// output [BM q x BN key] -> sP=P (exp), sDS=dS. Warp tiling: 4 qgroups x 4 kgroups, each 32q x 16key.
__device__ __forceinline__ void compute_pds(
   const bf16* sQ,const bf16* sK,const bf16* sO,const bf16* sV,
   bf16* sP,bf16* sDS,const float* sL,const float* sD,
   int q0,int kv0,int S,float scale,int warp,int lane){
  int qg=warp&3, kg=warp>>2;
  int qbase=qg*32, kbase=kg*16;
  float cs[2][2][4], cp[2][2][4];
  #pragma unroll
  for(int mi=0;mi<2;mi++)
   #pragma unroll
   for(int ni=0;ni<2;ni++)
    #pragma unroll
    for(int e=0;e<4;e++){cs[mi][ni][e]=0.f;cp[mi][ni][e]=0.f;}
  #pragma unroll
  for(int k0=0;k0<DHEAD;k0+=16){
    uint32_t aq[2][4],ao[2][4],bk[2][2],bv[2][2];
    #pragma unroll
    for(int mi=0;mi<2;mi++){ ld_A(sQ,qbase+mi*16,k0,DHEAD,lane,aq[mi]); ld_A(sO,qbase+mi*16,k0,DHEAD,lane,ao[mi]); }
    #pragma unroll
    for(int ni=0;ni<2;ni++){ ld_B(sK,k0,kbase+ni*8,DHEAD,lane,bk[ni]); ld_B(sV,k0,kbase+ni*8,DHEAD,lane,bv[ni]); }
    #pragma unroll
    for(int mi=0;mi<2;mi++)
     #pragma unroll
     for(int ni=0;ni<2;ni++){ mma(cs[mi][ni],aq[mi],bk[ni]); mma(cp[mi][ni],ao[mi],bv[ni]); }
  }
  int tg=lane>>2, tin=lane&3;
  #pragma unroll
  for(int mi=0;mi<2;mi++)
   #pragma unroll
   for(int ni=0;ni<2;ni++)
    #pragma unroll
    for(int e=0;e<4;e++){
      int r=qbase+mi*16+tg+((e>=2)?8:0);
      int c=kbase+ni*8+2*tin+(e&1);
      int gq=q0+r, gk=kv0+c;
      float pv=0.f,dv=0.f;
      if(gq<S&&gk<S){
        float Lv=sL[r],Dv=sD[r];
        pv=__expf(scale*cs[mi][ni][e]-Lv);
        dv=scale*pv*(cp[mi][ni][e]-Dv);
      }
      sP[swz(r,c,BN)]=__float2bfloat16(pv);
      sDS[swz(r,c,BN)]=__float2bfloat16(dv);
    }
}

__global__ void compute_D_kernel(const bf16* O,const bf16* dO,float* Dout,int total){
  int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx>=total)return;
  const bf16* o=O+(size_t)idx*DHEAD;
  const bf16* g=dO+(size_t)idx*DHEAD;
  float s=0.f;
  #pragma unroll
  for(int c=0;c<DHEAD;c+=8){
    uint4 vo=*(const uint4*)(o+c);
    uint4 vg=*(const uint4*)(g+c);
    const bf16* po=(const bf16*)&vo; const bf16* pg=(const bf16*)&vg;
    #pragma unroll
    for(int j=0;j<8;j++) s+=__bfloat162float(po[j])*__bfloat162float(pg[j]);
  }
  Dout[idx]=s;
}

__global__ __launch_bounds__(NTHREADS,1) void bwd_dkdv_kernel(
   const bf16* Q,const bf16* K,const bf16* V,const bf16* dO,
   const float* L,const float* Del,bf16* dK,bf16* dV,
   int B,int H,int S,float scale){
  int bh=blockIdx.y, kv_tile=blockIdx.x;
  int tid=threadIdx.x, lane=tid&31, warp=tid>>5;
  int kv0=kv_tile*BN;

  extern __shared__ __align__(16) char smem[];
  bf16* sQ0=(bf16*)smem;
  bf16* sQ1=sQ0+BM*DHEAD;
  bf16* sO0=sQ1+BM*DHEAD;
  bf16* sO1=sO0+BM*DHEAD;
  bf16* sK =sO1+BM*DHEAD;
  bf16* sV =sK +BN*DHEAD;
  bf16* sP =sV +BN*DHEAD;
  bf16* sDS=sP +BM*BN;
  float* sL=(float*)(sDS+BM*BN);
  float* sD=sL+BM;

  size_t base=(size_t)bh*S*DHEAD;
  const bf16* Qb=Q+base; const bf16* Kb=K+base; const bf16* Vb=V+base; const bf16* Ob=dO+base;
  const float* Lb=L+(size_t)bh*S; const float* Db=Del+(size_t)bh*S;

  load_tile_async(sK,Kb,kv0,BN,S,tid,NTHREADS);
  load_tile_async(sV,Vb,kv0,BN,S,tid,NTHREADS);
  cp_commit(); cp_wait<0>();
  __syncthreads();

  int nqt=(S+BM-1)/BM;
  load_tile_async(sQ0,Qb,0,BM,S,tid,NTHREADS);
  load_tile_async(sO0,Ob,0,BM,S,tid,NTHREADS);
  cp_commit();

  float accV[2][2][4], accK[2][2][4];
  #pragma unroll
  for(int mi=0;mi<2;mi++)
   #pragma unroll
   for(int ni=0;ni<2;ni++)
    #pragma unroll
    for(int e=0;e<4;e++){accV[mi][ni][e]=0.f;accK[mi][ni][e]=0.f;}

  int kg2=warp&1, dg=warp>>1;
  int keybase=kg2*32, dbase=dg*16;

  for(int qt=0; qt<nqt; qt++){
    int buf=qt&1;
    bf16* sQc = buf? sQ1:sQ0;
    bf16* sOc = buf? sO1:sO0;
    if(qt+1<nqt){
      int nb=(qt+1)&1;
      bf16* sQn=nb?sQ1:sQ0; bf16* sOn=nb?sO1:sO0;
      load_tile_async(sQn,Qb,(qt+1)*BM,BM,S,tid,NTHREADS);
      load_tile_async(sOn,Ob,(qt+1)*BM,BM,S,tid,NTHREADS);
      cp_commit(); cp_wait<1>();
    } else { cp_wait<0>(); }

    int q0=qt*BM;
    for(int i=tid;i<BM;i+=NTHREADS){int gr=q0+i; sL[i]=(gr<S)?Lb[gr]:0.f; sD[i]=(gr<S)?Db[gr]:0.f;}
    __syncthreads();
    compute_pds(sQc,sK,sOc,sV,sP,sDS,sL,sD,q0,kv0,S,scale,warp,lane);
    __syncthreads();
    // dV += P^T@dO, dK += dS^T@Q ; k-loop over queries (BM)
    #pragma unroll
    for(int k0=0;k0<BM;k0+=16){
      uint32_t apv[2][4],adk[2][4],bdo[2][2],bq[2][2];
      #pragma unroll
      for(int mi=0;mi<2;mi++){ ld_A_trans(sP,keybase+mi*16,k0,BN,lane,apv[mi]); ld_A_trans(sDS,keybase+mi*16,k0,BN,lane,adk[mi]); }
      #pragma unroll
      for(int ni=0;ni<2;ni++){ ld_B_trans(sOc,k0,dbase+ni*8,DHEAD,lane,bdo[ni]); ld_B_trans(sQc,k0,dbase+ni*8,DHEAD,lane,bq[ni]); }
      #pragma unroll
      for(int mi=0;mi<2;mi++)
       #pragma unroll
       for(int ni=0;ni<2;ni++){ mma(accV[mi][ni],apv[mi],bdo[ni]); mma(accK[mi][ni],adk[mi],bq[ni]); }
    }
  }

  int tg=lane>>2, tin=lane&3;
  #pragma unroll
  for(int mi=0;mi<2;mi++)
   #pragma unroll
   for(int ni=0;ni<2;ni++)
    #pragma unroll
    for(int e=0;e<4;e++){
      int r=keybase+mi*16+tg+((e>=2)?8:0);
      int c=dbase+ni*8+2*tin+(e&1);
      int gk=kv0+r;
      if(gk<S){
        dV[base+(size_t)gk*DHEAD+c]=__float2bfloat16(accV[mi][ni][e]);
        dK[base+(size_t)gk*DHEAD+c]=__float2bfloat16(accK[mi][ni][e]);
      }
    }
}

__global__ __launch_bounds__(NTHREADS,1) void bwd_dq_kernel(
   const bf16* Q,const bf16* K,const bf16* V,const bf16* dO,
   const float* L,const float* Del,bf16* dQ,
   int B,int H,int S,float scale){
  int bh=blockIdx.y, q_tile=blockIdx.x;
  int tid=threadIdx.x, lane=tid&31, warp=tid>>5;
  int q0=q_tile*BM;

  extern __shared__ __align__(16) char smem[];
  bf16* sQ =(bf16*)smem;
  bf16* sO =sQ+BM*DHEAD;
  bf16* sK0=sO+BM*DHEAD;
  bf16* sK1=sK0+BN*DHEAD;
  bf16* sV0=sK1+BN*DHEAD;
  bf16* sV1=sV0+BN*DHEAD;
  bf16* sP =sV1+BN*DHEAD;
  bf16* sDS=sP+BM*BN;
  float* sL=(float*)(sDS+BM*BN);
  float* sD=sL+BM;

  size_t base=(size_t)bh*S*DHEAD;
  const bf16* Qb=Q+base; const bf16* Kb=K+base; const bf16* Vb=V+base; const bf16* Ob=dO+base;
  const float* Lb=L+(size_t)bh*S; const float* Db=Del+(size_t)bh*S;

  load_tile_async(sQ,Qb,q0,BM,S,tid,NTHREADS);
  load_tile_async(sO,Ob,q0,BM,S,tid,NTHREADS);
  cp_commit(); cp_wait<0>();
  for(int i=tid;i<BM;i+=NTHREADS){int gr=q0+i; sL[i]=(gr<S)?Lb[gr]:0.f; sD[i]=(gr<S)?Db[gr]:0.f;}
  __syncthreads();

  int nkt=(S+BN-1)/BN;
  load_tile_async(sK0,Kb,0,BN,S,tid,NTHREADS);
  load_tile_async(sV0,Vb,0,BN,S,tid,NTHREADS);
  cp_commit();

  float accQ[2][4][4];
  #pragma unroll
  for(int mi=0;mi<2;mi++)
   #pragma unroll
   for(int ni=0;ni<4;ni++)
    #pragma unroll
    for(int e=0;e<4;e++) accQ[mi][ni][e]=0.f;

  int qg=warp&3, dg=warp>>2;
  int qbase=qg*32, dbase=dg*32;

  for(int kt=0; kt<nkt; kt++){
    int buf=kt&1;
    bf16* sKc=buf?sK1:sK0; bf16* sVc=buf?sV1:sV0;
    if(kt+1<nkt){
      int nb=(kt+1)&1;
      bf16* sKn=nb?sK1:sK0; bf16* sVn=nb?sV1:sV0;
      load_tile_async(sKn,Kb,(kt+1)*BN,BN,S,tid,NTHREADS);
      load_tile_async(sVn,Vb,(kt+1)*BN,BN,S,tid,NTHREADS);
      cp_commit(); cp_wait<1>();
    } else { cp_wait<0>(); }

    int kv0=kt*BN;
    __syncthreads();
    compute_pds(sQ,sKc,sO,sVc,sP,sDS,sL,sD,q0,kv0,S,scale,warp,lane);
    __syncthreads();
    // dQ += dS@K ; k-loop over keys (BN)
    #pragma unroll
    for(int k0=0;k0<BN;k0+=16){
      uint32_t ads[2][4],bk[4][2];
      #pragma unroll
      for(int mi=0;mi<2;mi++) ld_A(sDS,qbase+mi*16,k0,BN,lane,ads[mi]);
      #pragma unroll
      for(int ni=0;ni<4;ni++) ld_B_trans(sKc,k0,dbase+ni*8,DHEAD,lane,bk[ni]);
      #pragma unroll
      for(int mi=0;mi<2;mi++)
       #pragma unroll
       for(int ni=0;ni<4;ni++) mma(accQ[mi][ni],ads[mi],bk[ni]);
    }
  }

  int tg=lane>>2, tin=lane&3;
  #pragma unroll
  for(int mi=0;mi<2;mi++)
   #pragma unroll
   for(int ni=0;ni<4;ni++)
    #pragma unroll
    for(int e=0;e<4;e++){
      int r=qbase+mi*16+tg+((e>=2)?8:0);
      int c=dbase+ni*8+2*tin+(e&1);
      int gq=q0+r;
      if(gq<S) dQ[base+(size_t)gq*DHEAD+c]=__float2bfloat16(accQ[mi][ni][e]);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV){
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int Bd=(int)Q.size(0), Hd=(int)Q.size(1), Sd=(int)Q.size(2);
  float scale=1.0f/sqrtf((float)DHEAD);
  cudaStream_t stream=(cudaStream_t)TVMFFIEnvGetStream(Q.device().device_type,Q.device().device_id);

  const bf16* Qp=(const bf16*)Q.data_ptr();
  const bf16* Kp=(const bf16*)K.data_ptr();
  const bf16* Vp=(const bf16*)V.data_ptr();
  const bf16* Op=(const bf16*)O.data_ptr();
  const bf16* dOp=(const bf16*)dO.data_ptr();
  const float* Lp=(const float*)L.data_ptr();
  bf16* dQp=(bf16*)dQ.data_ptr();
  bf16* dKp=(bf16*)dK.data_ptr();
  bf16* dVp=(bf16*)dV.data_ptr();

  size_t nrows=(size_t)Bd*Hd*Sd;
  static float* Dbuf=nullptr; static size_t Dcap=0;
  if(nrows>Dcap){ if(Dbuf) cudaFree(Dbuf); CUDA_CHECK(cudaMalloc(&Dbuf,nrows*sizeof(float))); Dcap=nrows; }

  compute_D_kernel<<<(unsigned)((nrows+255)/256),256,0,stream>>>(Op,dOp,Dbuf,(int)nrows);
  CUDA_CHECK(cudaGetLastError());

  int smem_dkdv = (4*BM*DHEAD + 2*BN*DHEAD + 2*BM*BN)*2 + 2*BM*4;
  int smem_dq   = (2*BM*DHEAD + 4*BN*DHEAD + 2*BM*BN)*2 + 2*BM*4;
  cudaFuncSetAttribute(bwd_dkdv_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,smem_dkdv);
  cudaFuncSetAttribute(bwd_dq_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,smem_dq);

  int nkv=(Sd+BN-1)/BN;
  dim3 g1(nkv, Bd*Hd);
  bwd_dkdv_kernel<<<g1,NTHREADS,smem_dkdv,stream>>>(Qp,Kp,Vp,dOp,Lp,Dbuf,dKp,dVp,Bd,Hd,Sd,scale);
  CUDA_CHECK(cudaGetLastError());

  int nq=(Sd+BM-1)/BM;
  dim3 g2(nq, Bd*Hd);
  bwd_dq_kernel<<<g2,NTHREADS,smem_dq,stream>>>(Qp,Kp,Vp,dOp,Lp,Dbuf,dQp,Bd,Hd,Sd,scale);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

} // namespace mha_bwd