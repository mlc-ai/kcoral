#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do{ cudaError_t e=(call); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s %s:%d\n",cudaGetErrorString(e),__FILE__,__LINE__);exit(1);} }while(0)

namespace mha_kernel {
constexpr int D=128, BM=64, BN=64, LDS=136;
constexpr float LOG2E=1.4426950408889634f, LN2=0.6931471805599453f;

__device__ __forceinline__ uint32_t pack2bf16(float x,float y){
  __nv_bfloat162 v=__floats2bfloat162_rn(x,y);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ void mma16816(float&d0,float&d1,float&d2,float&d3,
  uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,uint32_t b0,uint32_t b1,
  float c0,float c1,float c2,float c3){
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
   :"=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
   :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"f"(c0),"f"(c1),"f"(c2),"f"(c3));
}
__device__ __forceinline__ float qmax(float v){v=fmaxf(v,__shfl_xor_sync(0xffffffffu,v,1));v=fmaxf(v,__shfl_xor_sync(0xffffffffu,v,2));return v;}
__device__ __forceinline__ float qsum(float v){v+=__shfl_xor_sync(0xffffffffu,v,1);v+=__shfl_xor_sync(0xffffffffu,v,2);return v;}

__device__ __forceinline__ void cp16(void* d,const void* s){
  unsigned a=(unsigned)__cvta_generic_to_shared(d);
  asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n"::"r"(a),"l"(s):"memory");
}
__device__ __forceinline__ void ldm_x4(uint32_t&r0,uint32_t&r1,uint32_t&r2,uint32_t&r3,const void* p){
  unsigned a=(unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n":"=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3):"r"(a));
}
__device__ __forceinline__ void ldm_trans_x4(uint32_t&r0,uint32_t&r1,uint32_t&r2,uint32_t&r3,const void* p){
  unsigned a=(unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3},[%4];\n":"=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3):"r"(a));
}

__device__ __forceinline__ void load_tile(__nv_bfloat16* dst,const __nv_bfloat16* src,int rowstart,int S,int tid){
  #pragma unroll
  for(int v=tid; v<BN*16; v+=128){
    int row=v>>4, col=(v&15)<<3;
    int grow=rowstart+row;
    __nv_bfloat16* d=&dst[row*LDS+col];
    if(grow<S) cp16(d,&src[(int64_t)grow*D+col]);
    else *reinterpret_cast<uint4*>(d)=make_uint4(0,0,0,0);
  }
}

__global__ void __launch_bounds__(128) attn(
  const __nv_bfloat16* __restrict__ Q,const __nv_bfloat16* __restrict__ K,
  const __nv_bfloat16* __restrict__ V,__nv_bfloat16* __restrict__ O,
  float* __restrict__ LSE,int S,int H,float scale){

  extern __shared__ __align__(16) char smem[];
  __nv_bfloat16* sQ=reinterpret_cast<__nv_bfloat16*>(smem);
  __nv_bfloat16* sK=sQ+BM*LDS;
  __nv_bfloat16* sV=sK+2*BN*LDS;

  int qb=blockIdx.x, h=blockIdx.y, b=blockIdx.z;
  int qstart=qb*BM;
  int tid=threadIdx.x, warp=tid>>5, lane=tid&31, groupID=lane>>2, tig=lane&3;
  int64_t bh=(int64_t)(b*H+h);
  const __nv_bfloat16* Qbh=Q+bh*S*D;
  const __nv_bfloat16* Kbh=K+bh*S*D;
  const __nv_bfloat16* Vbh=V+bh*S*D;
  __nv_bfloat16* Obh=O+bh*S*D;
  float* LSEbh=LSE+bh*S;
  float SC=scale*LOG2E;

  float Oacc[16][4];
  #pragma unroll
  for(int i=0;i<16;i++){Oacc[i][0]=Oacc[i][1]=Oacc[i][2]=Oacc[i][3]=0.f;}
  float m_g=-INFINITY,m_g8=-INFINITY,l_g=0.f,l_g8=0.f;

  load_tile(sQ,Qbh,qstart,S,tid);
  load_tile(&sK[0],Kbh,0,S,tid);
  load_tile(&sV[0],Vbh,0,S,tid);
  asm volatile("cp.async.commit_group;\n":::"memory");
  asm volatile("cp.async.wait_group 0;\n":::"memory");
  __syncthreads();

  int num_kb=(S+BN-1)/BN;
  int qrow=warp*16+(lane&15);
  int qcolbase=((lane>>4)<<3);
  for(int kb=0;kb<num_kb;kb++){
    int buf=kb&1, nbuf=(kb+1)&1;
    if(kb+1<num_kb){
      load_tile(&sK[nbuf*BN*LDS],Kbh,(kb+1)*BN,S,tid);
      load_tile(&sV[nbuf*BN*LDS],Vbh,(kb+1)*BN,S,tid);
      asm volatile("cp.async.commit_group;\n":::"memory");
    }
    __nv_bfloat16* skb=&sK[buf*BN*LDS];
    __nv_bfloat16* svb=&sV[buf*BN*LDS];
    int kstart=kb*BN;

    float Sacc[8][4];
    #pragma unroll
    for(int n=0;n<8;n++){Sacc[n][0]=Sacc[n][1]=Sacc[n][2]=Sacc[n][3]=0.f;}
    int col0=2*tig;
    #pragma unroll
    for(int kt2=0;kt2<8;kt2++){
      uint32_t a0,a1,a2,a3;
      ldm_x4(a0,a1,a2,a3,&sQ[qrow*LDS + kt2*16 + qcolbase]);
      int col=kt2*16+col0;
      #pragma unroll
      for(int n=0;n<8;n++){
        int kc=n*8+groupID;
        uint32_t b0=*reinterpret_cast<uint32_t*>(&skb[kc*LDS+col]);
        uint32_t b1=*reinterpret_cast<uint32_t*>(&skb[kc*LDS+col+8]);
        mma16816(Sacc[n][0],Sacc[n][1],Sacc[n][2],Sacc[n][3],a0,a1,a2,a3,b0,b1,
                 Sacc[n][0],Sacc[n][1],Sacc[n][2],Sacc[n][3]);
      }
    }
    float y[8][4]; float lmax=-INFINITY,lmax8=-INFINITY;
    #pragma unroll
    for(int n=0;n<8;n++){
      int gk0=kstart+n*8+2*tig, gk1=gk0+1;
      float y0=(gk0<S)?Sacc[n][0]*SC:-INFINITY;
      float y1=(gk1<S)?Sacc[n][1]*SC:-INFINITY;
      float y2=(gk0<S)?Sacc[n][2]*SC:-INFINITY;
      float y3=(gk1<S)?Sacc[n][3]*SC:-INFINITY;
      y[n][0]=y0;y[n][1]=y1;y[n][2]=y2;y[n][3]=y3;
      lmax=fmaxf(lmax,fmaxf(y0,y1)); lmax8=fmaxf(lmax8,fmaxf(y2,y3));
    }
    lmax=qmax(lmax); lmax8=qmax(lmax8);
    float mn=fmaxf(m_g,lmax), mn8=fmaxf(m_g8,lmax8);
    float cr=exp2f(m_g-mn), cr8=exp2f(m_g8-mn8);
    #pragma unroll
    for(int i=0;i<16;i++){Oacc[i][0]*=cr;Oacc[i][1]*=cr;Oacc[i][2]*=cr8;Oacc[i][3]*=cr8;}
    l_g*=cr; l_g8*=cr8;
    uint32_t pa[8][2]; float sg=0.f,sg8=0.f;
    #pragma unroll
    for(int n=0;n<8;n++){
      float p0=exp2f(y[n][0]-mn),p1=exp2f(y[n][1]-mn),p2=exp2f(y[n][2]-mn8),p3=exp2f(y[n][3]-mn8);
      sg+=p0+p1; sg8+=p2+p3;
      pa[n][0]=pack2bf16(p0,p1); pa[n][1]=pack2bf16(p2,p3);
    }
    sg=qsum(sg); sg8=qsum(sg8);
    l_g+=sg; l_g8+=sg8; m_g=mn; m_g8=mn8;

    // ---- P @ V using ldmatrix.trans for V ----
    #pragma unroll
    for(int c=0;c<8;c++){
      int dt0=2*c, dt1=2*c+1;
      int dbase=c*16;
      int colo=dbase + ((lane>>4)<<3);
      #pragma unroll
      for(int kt=0;kt<BN/16;kt++){
        uint32_t r0,r1,r2,r3;
        int row=kt*16+(lane&15);
        ldm_trans_x4(r0,r1,r2,r3,&svb[row*LDS+colo]);
        uint32_t pa0=pa[2*kt][0],pa1=pa[2*kt][1],pa2=pa[2*kt+1][0],pa3=pa[2*kt+1][1];
        mma16816(Oacc[dt0][0],Oacc[dt0][1],Oacc[dt0][2],Oacc[dt0][3],pa0,pa1,pa2,pa3,r0,r1,
                 Oacc[dt0][0],Oacc[dt0][1],Oacc[dt0][2],Oacc[dt0][3]);
        mma16816(Oacc[dt1][0],Oacc[dt1][1],Oacc[dt1][2],Oacc[dt1][3],pa0,pa1,pa2,pa3,r2,r3,
                 Oacc[dt1][0],Oacc[dt1][1],Oacc[dt1][2],Oacc[dt1][3]);
      }
    }
    if(kb+1<num_kb){
      asm volatile("cp.async.wait_group 0;\n":::"memory");
      __syncthreads();
    }
  }
  float ig=(l_g>0.f)?1.f/l_g:0.f, ig8=(l_g8>0.f)?1.f/l_g8:0.f;
  int qg=qstart+warp*16+groupID, qg8=qg+8;
  #pragma unroll
  for(int dt=0;dt<16;dt++){
    int d0=dt*8+2*tig, d1=d0+1;
    if(qg<S){ Obh[(int64_t)qg*D+d0]=__float2bfloat16(Oacc[dt][0]*ig); Obh[(int64_t)qg*D+d1]=__float2bfloat16(Oacc[dt][1]*ig);}
    if(qg8<S){ Obh[(int64_t)qg8*D+d0]=__float2bfloat16(Oacc[dt][2]*ig8); Obh[(int64_t)qg8*D+d1]=__float2bfloat16(Oacc[dt][3]*ig8);}
  }
  if(tig==0){
    if(qg<S) LSEbh[qg]=m_g*LN2+logf(l_g);
    if(qg8<S) LSEbh[qg8]=m_g8*LN2+logf(l_g8);
  }
}

void run(tvm::ffi::TensorView Q,tvm::ffi::TensorView K,tvm::ffi::TensorView V,
         tvm::ffi::TensorView O,tvm::ffi::TensorView LSE){
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int Bsz=(int)Q.size(0),Hsz=(int)Q.size(1),S=(int)Q.size(2),Dsz=(int)Q.size(3);
  const __nv_bfloat16* Qp=static_cast<const __nv_bfloat16*>(Q.data_ptr());
  const __nv_bfloat16* Kp=static_cast<const __nv_bfloat16*>(K.data_ptr());
  const __nv_bfloat16* Vp=static_cast<const __nv_bfloat16*>(V.data_ptr());
  __nv_bfloat16* Op=static_cast<__nv_bfloat16*>(O.data_ptr());
  float* Lp=static_cast<float*>(LSE.data_ptr());
  float scale=1.f/sqrtf((float)Dsz);
  dim3 grid((S+BM-1)/BM,Hsz,Bsz);
  int smem=(BM*LDS + 2*BN*LDS + 2*BN*LDS)*sizeof(__nv_bfloat16);
  static bool set=false;
  if(!set){ cudaFuncSetAttribute(attn,cudaFuncAttributeMaxDynamicSharedMemorySize,smem); set=true;}
  cudaStream_t st=static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type,Q.device().device_id));
  attn<<<grid,128,smem,st>>>(Qp,Kp,Vp,Op,Lp,S,Hsz,scale);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(st));
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_kernel::run);
}  // namespace mha_kernel