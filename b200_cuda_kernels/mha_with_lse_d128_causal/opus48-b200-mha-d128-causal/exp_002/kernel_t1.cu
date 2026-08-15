#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace attn_causal {

constexpr int D = 128;
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int THREADS = 128;
#define FULL 0xffffffffu

__device__ __forceinline__ unsigned int ld2(const __nv_bfloat16* p, int idx){
  return *reinterpret_cast<const unsigned int*>(p + idx);
}
__device__ __forceinline__ unsigned int pk(float a, float b){
  __nv_bfloat16 x=__float2bfloat16(a), y=__float2bfloat16(b);
  unsigned short xs=*reinterpret_cast<unsigned short*>(&x);
  unsigned short ys=*reinterpret_cast<unsigned short*>(&y);
  return (unsigned int)xs | ((unsigned int)ys<<16);
}
__device__ __forceinline__ unsigned int pkraw(__nv_bfloat16 a, __nv_bfloat16 b){
  unsigned short as=*reinterpret_cast<unsigned short*>(&a);
  unsigned short bs=*reinterpret_cast<unsigned short*>(&b);
  return (unsigned int)as | ((unsigned int)bs<<16);
}

#define MMA(d0,d1,d2,d3,a0,a1,a2,a3,b0,b1) \
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " \
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n" \
    : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3) \
    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1))

__global__ void attn_kernel(const __nv_bfloat16* __restrict__ Q,
                            const __nv_bfloat16* __restrict__ K,
                            const __nv_bfloat16* __restrict__ V,
                            __nv_bfloat16* __restrict__ O,
                            float* __restrict__ LSE, int S, int H){
  const int b=blockIdx.z, h=blockIdx.y, qb=blockIdx.x;
  const int tid=threadIdx.x;
  const int w=tid>>5, lane=tid&31, g=lane>>2, tig=lane&3;

  const long hb = ((long)(b*H+h))*(long)S*D;
  const __nv_bfloat16* Qbh=Q+hb;
  const __nv_bfloat16* Kbh=K+hb;
  const __nv_bfloat16* Vbh=V+hb;
  __nv_bfloat16* Obh=O+hb;
  float* LSEbh = LSE + ((long)(b*H+h))*S;

  extern __shared__ __nv_bfloat16 smem[];
  __nv_bfloat16* Qs=smem;
  __nv_bfloat16* Ks=Qs+BM*D;
  __nv_bfloat16* Vs=Ks+BN*D;

  // Load Q tile
  { const float4* Qg=reinterpret_cast<const float4*>(Qbh);
    for(int i=tid;i<BM*(D/8);i+=THREADS){int row=i>>4,v=i&15;int gq=qb*BM+row;
      float4 val; if(gq<S) val=Qg[(long)gq*16+v]; else val=make_float4(0,0,0,0);
      reinterpret_cast<float4*>(Qs)[row*16+v]=val;}
  }
  __syncthreads();

  float Of[16][4];
  #pragma unroll
  for(int no=0;no<16;no++){Of[no][0]=Of[no][1]=Of[no][2]=Of[no][3]=0.f;}
  float run_m_g=-INFINITY, run_l_g=0.f, run_m_g8=-INFINITY, run_l_g8=0.f;
  const float sc=0.08838834764831843f; // 1/sqrt(128)
  const int base_row=qb*BM + w*16;
  const int gq_g=base_row+g, gq_g8=base_row+g+8;

  for(int kb=0;kb<=qb;kb++){
    // Load K,V tiles
    { const float4* Kg=reinterpret_cast<const float4*>(Kbh);
      const float4* Vg=reinterpret_cast<const float4*>(Vbh);
      for(int i=tid;i<BN*(D/8);i+=THREADS){int col=i>>4,v=i&15;int gk=kb*BN+col;
        float4 kv,vv;
        if(gk<S){kv=Kg[(long)gk*16+v];vv=Vg[(long)gk*16+v];}
        else{kv=make_float4(0,0,0,0);vv=kv;}
        reinterpret_cast<float4*>(Ks)[col*16+v]=kv;
        reinterpret_cast<float4*>(Vs)[col*16+v]=vv;}
    }
    __syncthreads();

    // ---- QK^T : S[16x64] in C-fragment registers ----
    float Sf[8][4];
    #pragma unroll
    for(int nt=0;nt<8;nt++){Sf[nt][0]=Sf[nt][1]=Sf[nt][2]=Sf[nt][3]=0.f;}
    #pragma unroll
    for(int kt=0;kt<8;kt++){
      unsigned int a0=ld2(Qs,(w*16+g  )*D + kt*16+tig*2);
      unsigned int a1=ld2(Qs,(w*16+g+8)*D + kt*16+tig*2);
      unsigned int a2=ld2(Qs,(w*16+g  )*D + kt*16+tig*2+8);
      unsigned int a3=ld2(Qs,(w*16+g+8)*D + kt*16+tig*2+8);
      #pragma unroll
      for(int nt=0;nt<8;nt++){
        unsigned int b0=ld2(Ks,(nt*8+g)*D + kt*16+tig*2);
        unsigned int b1=ld2(Ks,(nt*8+g)*D + kt*16+tig*2+8);
        MMA(Sf[nt][0],Sf[nt][1],Sf[nt][2],Sf[nt][3], a0,a1,a2,a3, b0,b1);
      }
    }

    // ---- Softmax (warp-local) ----
    float lmax_g=-INFINITY, lmax_g8=-INFINITY;
    #pragma unroll
    for(int nt=0;nt<8;nt++){
      int c0=kb*BN+nt*8+tig*2, c1=c0+1;
      if(c0<S&&c0<=gq_g)  lmax_g =fmaxf(lmax_g ,Sf[nt][0]*sc);
      if(c1<S&&c1<=gq_g)  lmax_g =fmaxf(lmax_g ,Sf[nt][1]*sc);
      if(c0<S&&c0<=gq_g8) lmax_g8=fmaxf(lmax_g8,Sf[nt][2]*sc);
      if(c1<S&&c1<=gq_g8) lmax_g8=fmaxf(lmax_g8,Sf[nt][3]*sc);
    }
    lmax_g =fmaxf(lmax_g ,__shfl_xor_sync(FULL,lmax_g ,1)); lmax_g =fmaxf(lmax_g ,__shfl_xor_sync(FULL,lmax_g ,2));
    lmax_g8=fmaxf(lmax_g8,__shfl_xor_sync(FULL,lmax_g8,1)); lmax_g8=fmaxf(lmax_g8,__shfl_xor_sync(FULL,lmax_g8,2));

    float mno_g =fmaxf(run_m_g ,lmax_g );
    float mno_g8=fmaxf(run_m_g8,lmax_g8);
    float corr_g =(mno_g ==-INFINITY)?1.f:(run_m_g ==-INFINITY?0.f:__expf(run_m_g -mno_g ));
    float corr_g8=(mno_g8==-INFINITY)?1.f:(run_m_g8==-INFINITY?0.f:__expf(run_m_g8-mno_g8));

    unsigned int plo[8], phi[8];
    float psum_g=0.f, psum_g8=0.f;
    #pragma unroll
    for(int nt=0;nt<8;nt++){
      int c0=kb*BN+nt*8+tig*2, c1=c0+1;
      float p0=(c0<S&&c0<=gq_g )?__expf(Sf[nt][0]*sc-mno_g ):0.f;
      float p1=(c1<S&&c1<=gq_g )?__expf(Sf[nt][1]*sc-mno_g ):0.f;
      float p2=(c0<S&&c0<=gq_g8)?__expf(Sf[nt][2]*sc-mno_g8):0.f;
      float p3=(c1<S&&c1<=gq_g8)?__expf(Sf[nt][3]*sc-mno_g8):0.f;
      psum_g+=p0+p1; psum_g8+=p2+p3;
      plo[nt]=pk(p0,p1); phi[nt]=pk(p2,p3);
    }
    psum_g +=__shfl_xor_sync(FULL,psum_g ,1); psum_g +=__shfl_xor_sync(FULL,psum_g ,2);
    psum_g8+=__shfl_xor_sync(FULL,psum_g8,1); psum_g8+=__shfl_xor_sync(FULL,psum_g8,2);
    run_l_g =run_l_g *corr_g +psum_g;
    run_l_g8=run_l_g8*corr_g8+psum_g8;
    run_m_g =mno_g; run_m_g8=mno_g8;

    #pragma unroll
    for(int no=0;no<16;no++){Of[no][0]*=corr_g;Of[no][1]*=corr_g;Of[no][2]*=corr_g8;Of[no][3]*=corr_g8;}

    // ---- PV : O[16x128] += P[16x64] @ V[64x128] ----
    #pragma unroll
    for(int ktp=0;ktp<4;ktp++){
      unsigned int A0=plo[2*ktp], A1=phi[2*ktp], A2=plo[2*ktp+1], A3=phi[2*ktp+1];
      #pragma unroll
      for(int no=0;no<16;no++){
        __nv_bfloat16 v0=Vs[(ktp*16+tig*2  )*D + no*8+g];
        __nv_bfloat16 v1=Vs[(ktp*16+tig*2+1)*D + no*8+g];
        __nv_bfloat16 v2=Vs[(ktp*16+tig*2+8)*D + no*8+g];
        __nv_bfloat16 v3=Vs[(ktp*16+tig*2+9)*D + no*8+g];
        unsigned int B0=pkraw(v0,v1), B1=pkraw(v2,v3);
        MMA(Of[no][0],Of[no][1],Of[no][2],Of[no][3], A0,A1,A2,A3, B0,B1);
      }
    }
    __syncthreads();
  }

  // ---- Finalize ----
  float inv_l_g =(run_l_g >0.f)?1.f/run_l_g :0.f;
  float inv_l_g8=(run_l_g8>0.f)?1.f/run_l_g8:0.f;
  #pragma unroll
  for(int no=0;no<16;no++){
    int d0=no*8+tig*2, d1=d0+1;
    if(gq_g<S){
      Obh[(long)gq_g*D + d0]=__float2bfloat16(Of[no][0]*inv_l_g);
      Obh[(long)gq_g*D + d1]=__float2bfloat16(Of[no][1]*inv_l_g);
    }
    if(gq_g8<S){
      Obh[(long)gq_g8*D + d0]=__float2bfloat16(Of[no][2]*inv_l_g8);
      Obh[(long)gq_g8*D + d1]=__float2bfloat16(Of[no][3]*inv_l_g8);
    }
  }
  if(tig==0){
    if(gq_g <S) LSEbh[gq_g ]=run_m_g +logf(run_l_g );
    if(gq_g8<S) LSEbh[gq_g8]=run_m_g8+logf(run_l_g8);
  }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE){
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int B=(int)Q.size(0);
  int H=(int)Q.size(1);
  int S=(int)Q.size(2);

  const __nv_bfloat16* Qp=static_cast<const __nv_bfloat16*>(Q.data_ptr());
  const __nv_bfloat16* Kp=static_cast<const __nv_bfloat16*>(K.data_ptr());
  const __nv_bfloat16* Vp=static_cast<const __nv_bfloat16*>(V.data_ptr());
  __nv_bfloat16* Op=static_cast<__nv_bfloat16*>(O.data_ptr());
  float* LSEp=static_cast<float*>(LSE.data_ptr());

  cudaStream_t stream=static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

  int nqb=(S+BM-1)/BM;
  dim3 grid(nqb, H, B);
  size_t smem=(size_t)3*BM*D*2; // Qs,Ks,Vs bf16

  static bool attr_set=false;
  if(!attr_set){
    CUDA_CHECK(cudaFuncSetAttribute(attn_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));
    attr_set=true;
  }

  attn_kernel<<<grid,THREADS,smem,stream>>>(Qp,Kp,Vp,Op,LSEp,S,H);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, attn_causal::run);

}  // namespace attn_causal