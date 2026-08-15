#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdio.h>
#include <math.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
    fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__); exit(1);} } while(0)

namespace mha {
using bf16 = __nv_bfloat16;

__device__ __forceinline__ uint32_t ld_u32(const void* p){ return *reinterpret_cast<const uint32_t*>(p); }
__device__ __forceinline__ uint32_t pk(uint16_t lo, uint16_t hi){ return (uint32_t)lo | ((uint32_t)hi<<16); }
__device__ __forceinline__ uint16_t f2b(float x){ bf16 b=__float2bfloat16(x); return *reinterpret_cast<uint16_t*>(&b); }

__device__ __forceinline__ void mma16816(float &d0,float&d1,float&d2,float&d3,
   uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3, uint32_t b0,uint32_t b1){
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)
    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}

#define BM 64
#define BN 64
#define Dd 128
#define PAD 8
#define LDQ (Dd+PAD)    // 136
#define LDV (BN+PAD)     // 72
#define NTHREADS 128

__device__ __forceinline__ void cpasync16(void* s,const void* g){
  uint32_t sa=(uint32_t)__cvta_generic_to_shared(s);
  asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n"::"r"(sa),"l"(g):"memory");
}
__device__ __forceinline__ void cp_commit(){asm volatile("cp.async.commit_group;\n":::"memory");}
template<int N> __device__ __forceinline__ void cp_wait(){asm volatile("cp.async.wait_group %0;\n"::"n"(N):"memory");}

__device__ __forceinline__ void load_tile(bf16* smem,const bf16* g,int rows,int row_off,int S){
  const int vpr=Dd/8;
  for(int v=threadIdx.x; v<rows*vpr; v+=NTHREADS){
    int r=v/vpr, c=(v%vpr)*8;
    int gr=row_off+r; if(gr>=S) gr=S-1; if(gr<0) gr=0;
    cpasync16(&smem[r*LDQ+c], g+(long)gr*Dd+c);
  }
}

__global__ __launch_bounds__(128) void attn(const bf16*Q,const bf16*K,const bf16*V,bf16*O,float*LSE,int S){
  extern __shared__ char smem_raw[];
  bf16* Qs=(bf16*)smem_raw;
  bf16* Ks=Qs+BM*LDQ;
  bf16* Vs=Ks+2*BN*LDQ;
  bf16* Vt=Vs+2*BN*LDQ;

  int bh=blockIdx.y, qstart=blockIdx.x*BM;
  const bf16* Qb=Q+(long)bh*S*Dd;
  const bf16* Kb=K+(long)bh*S*Dd;
  const bf16* Vb=V+(long)bh*S*Dd;
  bf16* Ob=O+(long)bh*S*Dd;
  float* LSEb=LSE+(long)bh*S;
  int warp=threadIdx.x/32, lane=threadIdx.x%32, gid=lane/4, tid=lane%4;

  load_tile(Qs, Qb, BM, qstart, S);
  load_tile(Ks+0*BN*LDQ, Kb, BN, 0, S);
  load_tile(Vs+0*BN*LDQ, Vb, BN, 0, S);
  cp_commit();

  float o[16][4];
  #pragma unroll
  for(int i=0;i<16;i++)o[i][0]=o[i][1]=o[i][2]=o[i][3]=0.f;
  float mrun0=-1e30f,mrun1=-1e30f,lrun0=0.f,lrun1=0.f;
  const float scale=rsqrtf((float)Dd);
  int nkv=(S+BN-1)/BN;

  for(int kv=0;kv<nkv;kv++){
    int cur=kv&1, nxt=(kv+1)&1;
    int kvstart=kv*BN;
    if(kv+1<nkv){
      load_tile(Ks+nxt*BN*LDQ, Kb, BN, (kv+1)*BN, S);
      load_tile(Vs+nxt*BN*LDQ, Vb, BN, (kv+1)*BN, S);
      cp_commit();
      cp_wait<1>();
    } else {
      cp_wait<0>();
    }
    __syncthreads();
    bf16* Ksc=Ks+cur*BN*LDQ;
    bf16* Vsc=Vs+cur*BN*LDQ;
    // transpose Vsc[n][d] -> Vt[d][n]
    for(int v=threadIdx.x; v<BN*(Dd/8); v+=NTHREADS){
      int n=v/(Dd/8), d0=(v%(Dd/8))*8;
      uint4 tmp=*reinterpret_cast<const uint4*>(&Vsc[n*LDQ+d0]);
      bf16* t=reinterpret_cast<bf16*>(&tmp);
      #pragma unroll
      for(int i=0;i<8;i++) Vt[(d0+i)*LDV+n]=t[i];
    }
    __syncthreads();

    // ---- QK^T ----
    float s[8][4];
    #pragma unroll
    for(int nt=0;nt<8;nt++)s[nt][0]=s[nt][1]=s[nt][2]=s[nt][3]=0.f;
    #pragma unroll
    for(int kt=0;kt<8;kt++){
      int qr0=16*warp+gid, qr1=qr0+8;
      uint32_t a0=ld_u32(&Qs[qr0*LDQ+16*kt+2*tid]);
      uint32_t a1=ld_u32(&Qs[qr1*LDQ+16*kt+2*tid]);
      uint32_t a2=ld_u32(&Qs[qr0*LDQ+16*kt+2*tid+8]);
      uint32_t a3=ld_u32(&Qs[qr1*LDQ+16*kt+2*tid+8]);
      #pragma unroll
      for(int nt=0;nt<8;nt++){
        uint32_t b0=ld_u32(&Ksc[(8*nt+gid)*LDQ+16*kt+2*tid]);
        uint32_t b1=ld_u32(&Ksc[(8*nt+gid)*LDQ+16*kt+2*tid+8]);
        mma16816(s[nt][0],s[nt][1],s[nt][2],s[nt][3],a0,a1,a2,a3,b0,b1);
      }
    }
    #pragma unroll
    for(int nt=0;nt<8;nt++){
      int key0=kvstart+8*nt+2*tid,key1=key0+1;
      s[nt][0]*=scale;s[nt][1]*=scale;s[nt][2]*=scale;s[nt][3]*=scale;
      if(key0>=S){s[nt][0]=-1e30f;s[nt][2]=-1e30f;}
      if(key1>=S){s[nt][1]=-1e30f;s[nt][3]=-1e30f;}
    }
    float m0=-1e30f,m1=-1e30f;
    #pragma unroll
    for(int nt=0;nt<8;nt++){m0=fmaxf(m0,fmaxf(s[nt][0],s[nt][1]));m1=fmaxf(m1,fmaxf(s[nt][2],s[nt][3]));}
    m0=fmaxf(m0,__shfl_xor_sync(0xffffffffu,m0,1));m0=fmaxf(m0,__shfl_xor_sync(0xffffffffu,m0,2));
    m1=fmaxf(m1,__shfl_xor_sync(0xffffffffu,m1,1));m1=fmaxf(m1,__shfl_xor_sync(0xffffffffu,m1,2));
    float nm0=fmaxf(mrun0,m0),nm1=fmaxf(mrun1,m1);
    float sc0=__expf(mrun0-nm0),sc1=__expf(mrun1-nm1);
    uint16_t pp[8][4];
    float l0=0.f,l1=0.f;
    #pragma unroll
    for(int nt=0;nt<8;nt++){
      float p0=__expf(s[nt][0]-nm0),p1=__expf(s[nt][1]-nm0);
      float p2=__expf(s[nt][2]-nm1),p3=__expf(s[nt][3]-nm1);
      l0+=p0+p1;l1+=p2+p3;
      pp[nt][0]=f2b(p0);pp[nt][1]=f2b(p1);pp[nt][2]=f2b(p2);pp[nt][3]=f2b(p3);
    }
    l0+=__shfl_xor_sync(0xffffffffu,l0,1);l0+=__shfl_xor_sync(0xffffffffu,l0,2);
    l1+=__shfl_xor_sync(0xffffffffu,l1,1);l1+=__shfl_xor_sync(0xffffffffu,l1,2);
    #pragma unroll
    for(int dt=0;dt<16;dt++){o[dt][0]*=sc0;o[dt][1]*=sc0;o[dt][2]*=sc1;o[dt][3]*=sc1;}
    lrun0=lrun0*sc0+l0;lrun1=lrun1*sc1+l1;mrun0=nm0;mrun1=nm1;
    // ---- P @ V ----
    #pragma unroll
    for(int kt=0;kt<4;kt++){
      uint32_t a0=pk(pp[2*kt][0],pp[2*kt][1]);
      uint32_t a1=pk(pp[2*kt][2],pp[2*kt][3]);
      uint32_t a2=pk(pp[2*kt+1][0],pp[2*kt+1][1]);
      uint32_t a3=pk(pp[2*kt+1][2],pp[2*kt+1][3]);
      #pragma unroll
      for(int dt=0;dt<16;dt++){
        uint32_t b0=ld_u32(&Vt[(8*dt+gid)*LDV+16*kt+2*tid]);
        uint32_t b1=ld_u32(&Vt[(8*dt+gid)*LDV+16*kt+2*tid+8]);
        mma16816(o[dt][0],o[dt][1],o[dt][2],o[dt][3],a0,a1,a2,a3,b0,b1);
      }
    }
    __syncthreads();
  }
  float inv0=1.f/lrun0, inv1=1.f/lrun1;
  int row0=qstart+16*warp+gid, row1=row0+8;
  #pragma unroll
  for(int dt=0;dt<16;dt++){
    int d0=8*dt+2*tid,d1=d0+1;
    if(row0<S){Ob[(long)row0*Dd+d0]=__float2bfloat16(o[dt][0]*inv0);Ob[(long)row0*Dd+d1]=__float2bfloat16(o[dt][1]*inv0);}
    if(row1<S){Ob[(long)row1*Dd+d0]=__float2bfloat16(o[dt][2]*inv1);Ob[(long)row1*Dd+d1]=__float2bfloat16(o[dt][3]*inv1);}
  }
  if(tid==0){
    if(row0<S)LSEb[row0]=mrun0+logf(lrun0);
    if(row1<S)LSEb[row1]=mrun1+logf(lrun1);
  }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE){
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int B=(int)Q.size(0), H=(int)Q.size(1), S=(int)Q.size(2);
  const bf16* Qd=(const bf16*)Q.data_ptr();
  const bf16* Kd=(const bf16*)K.data_ptr();
  const bf16* Vd=(const bf16*)V.data_ptr();
  bf16* Od=(bf16*)O.data_ptr();
  float* LSEd=(float*)LSE.data_ptr();

  int smem_bytes = (BM*LDQ + 2*BN*LDQ + 2*BN*LDQ + Dd*LDV) * (int)sizeof(bf16);
  static bool cfg=false;
  if(!cfg){ CUDA_CHECK(cudaFuncSetAttribute(attn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes)); cfg=true; }

  dim3 grid((S+BM-1)/BM, B*H);
  dim3 block(NTHREADS);
  cudaStream_t stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
  attn<<<grid,block,smem_bytes,stream>>>(Qd,Kd,Vd,Od,LSEd,S);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha::run);

}  // namespace mha