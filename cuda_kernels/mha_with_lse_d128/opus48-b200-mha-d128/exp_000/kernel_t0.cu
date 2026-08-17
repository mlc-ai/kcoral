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
__device__ __forceinline__ uint16_t ld_u16(const void* p){ return *reinterpret_cast<const uint16_t*>(p); }
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

__device__ __forceinline__ void load_tile(bf16* smem, const bf16* g, int rows, int row_off, int S){
  const int D=128; const int vpr=D/8; int total=rows*vpr;
  for(int v=threadIdx.x; v<total; v+=blockDim.x){
    int r=v/vpr; int c=(v%vpr)*8; int gr=row_off+r;
    float4 val;
    if(gr<S) val=*reinterpret_cast<const float4*>(g+(long)gr*D+c);
    else val=make_float4(0.f,0.f,0.f,0.f);
    *reinterpret_cast<float4*>(smem+r*D+c)=val;
  }
}

__global__ __launch_bounds__(128) void attn_kernel(const bf16* Q,const bf16* K,const bf16* V,
                                                   bf16* O, float* LSE, int S){
  const int D=128, BM=64, BN=64;
  int bh=blockIdx.y; int qstart=blockIdx.x*BM;
  const bf16* Qb=Q+(long)bh*S*D;
  const bf16* Kb=K+(long)bh*S*D;
  const bf16* Vb=V+(long)bh*S*D;
  bf16* Ob=O+(long)bh*S*D;
  float* LSEb=LSE+(long)bh*S;

  __shared__ __align__(16) bf16 Qs[BM][D];
  __shared__ __align__(16) bf16 Ks[BN][D];
  __shared__ __align__(16) bf16 Vs[BN][D];

  int warp=threadIdx.x/32; int lane=threadIdx.x%32; int gid=lane/4; int tid=lane%4;

  load_tile(&Qs[0][0], Qb, BM, qstart, S);
  __syncthreads();

  float o[16][4];
  #pragma unroll
  for(int dt=0;dt<16;dt++){o[dt][0]=o[dt][1]=o[dt][2]=o[dt][3]=0.f;}
  float mrun0=-1e30f,mrun1=-1e30f,lrun0=0.f,lrun1=0.f;
  const float scale=rsqrtf((float)D);
  int num_kv=(S+BN-1)/BN;

  for(int kv=0; kv<num_kv; kv++){
    int kvstart=kv*BN;
    load_tile(&Ks[0][0], Kb, BN, kvstart, S);
    load_tile(&Vs[0][0], Vb, BN, kvstart, S);
    __syncthreads();

    // ---- QK^T ----
    float s[8][4];
    #pragma unroll
    for(int nt=0;nt<8;nt++){s[nt][0]=s[nt][1]=s[nt][2]=s[nt][3]=0.f;}
    #pragma unroll
    for(int kt=0;kt<8;kt++){
      int qr0=16*warp+gid, qr1=16*warp+gid+8;
      uint32_t a0=ld_u32(&Qs[qr0][16*kt+2*tid]);
      uint32_t a1=ld_u32(&Qs[qr1][16*kt+2*tid]);
      uint32_t a2=ld_u32(&Qs[qr0][16*kt+2*tid+8]);
      uint32_t a3=ld_u32(&Qs[qr1][16*kt+2*tid+8]);
      #pragma unroll
      for(int nt=0;nt<8;nt++){
        uint32_t b0=ld_u32(&Ks[8*nt+gid][16*kt+2*tid]);
        uint32_t b1=ld_u32(&Ks[8*nt+gid][16*kt+2*tid+8]);
        mma16816(s[nt][0],s[nt][1],s[nt][2],s[nt][3], a0,a1,a2,a3,b0,b1);
      }
    }
    // scale + causal-free mask (key bounds)
    #pragma unroll
    for(int nt=0;nt<8;nt++){
      int key0=kvstart+8*nt+2*tid, key1=key0+1;
      s[nt][0]*=scale; s[nt][1]*=scale; s[nt][2]*=scale; s[nt][3]*=scale;
      if(key0>=S){s[nt][0]=-1e30f;s[nt][2]=-1e30f;}
      if(key1>=S){s[nt][1]=-1e30f;s[nt][3]=-1e30f;}
    }
    // rowmax reduction (over 4 threads sharing a row)
    float m0=-1e30f,m1=-1e30f;
    #pragma unroll
    for(int nt=0;nt<8;nt++){m0=fmaxf(m0,fmaxf(s[nt][0],s[nt][1])); m1=fmaxf(m1,fmaxf(s[nt][2],s[nt][3]));}
    m0=fmaxf(m0,__shfl_xor_sync(0xffffffff,m0,1)); m0=fmaxf(m0,__shfl_xor_sync(0xffffffff,m0,2));
    m1=fmaxf(m1,__shfl_xor_sync(0xffffffff,m1,1)); m1=fmaxf(m1,__shfl_xor_sync(0xffffffff,m1,2));
    float nm0=fmaxf(mrun0,m0), nm1=fmaxf(mrun1,m1);
    float sc0=__expf(mrun0-nm0), sc1=__expf(mrun1-nm1);
    // P = exp(S - m) ; block sums
    uint16_t pp[8][4];
    float l0=0.f,l1=0.f;
    #pragma unroll
    for(int nt=0;nt<8;nt++){
      float p0=__expf(s[nt][0]-nm0), p1=__expf(s[nt][1]-nm0);
      float p2=__expf(s[nt][2]-nm1), p3=__expf(s[nt][3]-nm1);
      l0+=p0+p1; l1+=p2+p3;
      pp[nt][0]=f2b(p0); pp[nt][1]=f2b(p1); pp[nt][2]=f2b(p2); pp[nt][3]=f2b(p3);
    }
    l0+=__shfl_xor_sync(0xffffffff,l0,1); l0+=__shfl_xor_sync(0xffffffff,l0,2);
    l1+=__shfl_xor_sync(0xffffffff,l1,1); l1+=__shfl_xor_sync(0xffffffff,l1,2);
    // rescale O accumulator, update running stats
    #pragma unroll
    for(int dt=0;dt<16;dt++){o[dt][0]*=sc0;o[dt][1]*=sc0;o[dt][2]*=sc1;o[dt][3]*=sc1;}
    lrun0=lrun0*sc0+l0; lrun1=lrun1*sc1+l1; mrun0=nm0; mrun1=nm1;
    // ---- P @ V ----
    #pragma unroll
    for(int kt=0;kt<4;kt++){
      uint32_t a0=pk(pp[2*kt][0],pp[2*kt][1]);
      uint32_t a1=pk(pp[2*kt][2],pp[2*kt][3]);
      uint32_t a2=pk(pp[2*kt+1][0],pp[2*kt+1][1]);
      uint32_t a3=pk(pp[2*kt+1][2],pp[2*kt+1][3]);
      #pragma unroll
      for(int dt=0;dt<16;dt++){
        uint32_t b0=pk(ld_u16(&Vs[16*kt+2*tid][8*dt+gid]),   ld_u16(&Vs[16*kt+2*tid+1][8*dt+gid]));
        uint32_t b1=pk(ld_u16(&Vs[16*kt+2*tid+8][8*dt+gid]), ld_u16(&Vs[16*kt+2*tid+9][8*dt+gid]));
        mma16816(o[dt][0],o[dt][1],o[dt][2],o[dt][3], a0,a1,a2,a3,b0,b1);
      }
    }
    __syncthreads();
  }
  // ---- finalize ----
  float inv0=1.0f/lrun0, inv1=1.0f/lrun1;
  int row0=qstart+16*warp+gid, row1=qstart+16*warp+gid+8;
  #pragma unroll
  for(int dt=0;dt<16;dt++){
    int d0=8*dt+2*tid, d1=d0+1;
    if(row0<S){ Ob[(long)row0*D+d0]=__float2bfloat16(o[dt][0]*inv0); Ob[(long)row0*D+d1]=__float2bfloat16(o[dt][1]*inv0);}
    if(row1<S){ Ob[(long)row1*D+d0]=__float2bfloat16(o[dt][2]*inv1); Ob[(long)row1*D+d1]=__float2bfloat16(o[dt][3]*inv1);}
  }
  if(tid==0){
    if(row0<S) LSEb[row0]=mrun0+logf(lrun0);
    if(row1<S) LSEb[row1]=mrun1+logf(lrun1);
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
  dim3 grid((S+63)/64, B*H);
  dim3 block(128);
  cudaStream_t stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
  attn_kernel<<<grid,block,0,stream>>>(Qd,Kd,Vd,Od,LSEd,S);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha::run);

}  // namespace mha