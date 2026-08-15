#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <math.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { \
  cudaError_t _e=(call); \
  if(_e!=cudaSuccess){fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__);exit(1);} \
}while(0)

namespace mha_kernel {

constexpr int D = 128;
constexpr int BM = 64;
constexpr int BN = 64;

__device__ __forceinline__ uint32_t pack2f(float a, float b){
  __nv_bfloat162 v = __floats2bfloat162_rn(a,b);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ uint32_t pack2h(__nv_bfloat16 a, __nv_bfloat16 b){
  __nv_bfloat162 v = __halves2bfloat162(a,b);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ void mma16816(float&c0,float&c1,float&c2,float&c3,
  uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,uint32_t b0,uint32_t b1){
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
    :"+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
    :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t&r0,uint32_t&r1,uint32_t&r2,uint32_t&r3,uint32_t a){
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3},[%4];\n"
    :"=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3):"r"(a));
}
__device__ __forceinline__ void cp_async16(__nv_bfloat16* s,const __nv_bfloat16* g,int bytes){
  uint32_t sa=(uint32_t)__cvta_generic_to_shared(s);
  asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;\n"::"r"(sa),"l"(g),"r"(bytes):"memory");
}
__device__ __forceinline__ void cp_commit(){ asm volatile("cp.async.commit_group;\n":::"memory"); }
template<int N> __device__ __forceinline__ void cp_wait(){ asm volatile("cp.async.wait_group %0;\n"::"n"(N):"memory"); }

__device__ __forceinline__ void load_tile_cp(const __nv_bfloat16* src,int row_start,int S,__nv_bfloat16* dst){
  int tid=threadIdx.x;
  #pragma unroll
  for(int i=0;i<8;i++){
    int lin=tid+i*128; int row=lin>>4; int c4=lin&15;
    int grow=row_start+row; int in=(grow<S);
    const __nv_bfloat16* g=src+(long)(in?grow:(S-1))*D+c4*8;
    cp_async16(dst+row*D+c4*8, g, in?16:0);
  }
}

__global__ __launch_bounds__(128) void attn_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int S, int H){
  int b=blockIdx.z, h=blockIdx.y, qtile=blockIdx.x;
  int q_start=qtile*BM;
  long base=((long)(b*H+h)*S)*D;
  const __nv_bfloat16* Qbh=Q+base;
  const __nv_bfloat16* Kbh=K+base;
  const __nv_bfloat16* Vbh=V+base;
  __nv_bfloat16* Obh=O+base;
  float* LSEbh=LSE+(long)(b*H+h)*S;

  extern __shared__ __nv_bfloat16 smem[];
  __nv_bfloat16* Qsh=smem;
  __nv_bfloat16* Ksh=Qsh+BM*D;      // 2 buffers of BN*D
  __nv_bfloat16* Vsh=Ksh+2*BN*D;    // 2 buffers of BN*D

  int warp=threadIdx.x>>5, lane=threadIdx.x&31, gid=lane>>2, tidg=lane&3;

  float O_[16][4];
  #pragma unroll
  for(int nt=0;nt<16;nt++){O_[nt][0]=0;O_[nt][1]=0;O_[nt][2]=0;O_[nt][3]=0;}
  float m_a=-INFINITY,m_b=-INFINITY,l_a=0.f,l_b=0.f;
  const float scale=rsqrtf((float)D);

  int qmax=q_start+BM-1; int last_key=qmax<(S-1)?qmax:(S-1); int num_kv=last_key/BN+1;
  int qrow0=warp*16+gid, qrow1=warp*16+gid+8;
  int q0g=q_start+qrow0, q1g=q_start+qrow1;

  // prefetch Q + KV tile 0
  load_tile_cp(Qbh,q_start,S,Qsh);
  load_tile_cp(Kbh,0,S,Ksh);
  load_tile_cp(Vbh,0,S,Vsh);
  cp_commit();

  for(int kvt=0;kvt<num_kv;kvt++){
    int buf=kvt&1;
    if(kvt+1<num_kv){
      int nb=(kvt+1)&1;
      load_tile_cp(Kbh,(kvt+1)*BN,S,Ksh+nb*BN*D);
      load_tile_cp(Vbh,(kvt+1)*BN,S,Vsh+nb*BN*D);
      cp_commit();
      cp_wait<1>();
    } else {
      cp_wait<0>();
    }
    __syncthreads();
    __nv_bfloat16* Kb=Ksh+buf*BN*D;
    __nv_bfloat16* Vb=Vsh+buf*BN*D;
    int kv_start=kvt*BN;

    // Q fragments (resident)
    uint32_t Qf[8][4];
    #pragma unroll
    for(int kt=0;kt<8;kt++){
      Qf[kt][0]=*reinterpret_cast<uint32_t*>(&Qsh[qrow0*D+16*kt+2*tidg]);
      Qf[kt][1]=*reinterpret_cast<uint32_t*>(&Qsh[qrow1*D+16*kt+2*tidg]);
      Qf[kt][2]=*reinterpret_cast<uint32_t*>(&Qsh[qrow0*D+16*kt+8+2*tidg]);
      Qf[kt][3]=*reinterpret_cast<uint32_t*>(&Qsh[qrow1*D+16*kt+8+2*tidg]);
    }
    // QK^T
    float Sf[8][4];
    #pragma unroll
    for(int nt=0;nt<8;nt++){
      float c0=0,c1=0,c2=0,c3=0;
      int krow=8*nt+gid;
      #pragma unroll
      for(int kt=0;kt<8;kt++){
        uint32_t b0=*reinterpret_cast<uint32_t*>(&Kb[krow*D+16*kt+2*tidg]);
        uint32_t b1=*reinterpret_cast<uint32_t*>(&Kb[krow*D+16*kt+8+2*tidg]);
        mma16816(c0,c1,c2,c3, Qf[kt][0],Qf[kt][1],Qf[kt][2],Qf[kt][3], b0,b1);
      }
      Sf[nt][0]=c0*scale;Sf[nt][1]=c1*scale;Sf[nt][2]=c2*scale;Sf[nt][3]=c3*scale;
    }
    // causal + range mask
    #pragma unroll
    for(int nt=0;nt<8;nt++){
      int key0=kv_start+8*nt+2*tidg, key1=key0+1;
      if(!(key0<S&&key0<=q0g))Sf[nt][0]=-INFINITY;
      if(!(key1<S&&key1<=q0g))Sf[nt][1]=-INFINITY;
      if(!(key0<S&&key0<=q1g))Sf[nt][2]=-INFINITY;
      if(!(key1<S&&key1<=q1g))Sf[nt][3]=-INFINITY;
    }
    // online softmax
    float rmax_a=-INFINITY,rmax_b=-INFINITY;
    #pragma unroll
    for(int nt=0;nt<8;nt++){
      rmax_a=fmaxf(rmax_a,fmaxf(Sf[nt][0],Sf[nt][1]));
      rmax_b=fmaxf(rmax_b,fmaxf(Sf[nt][2],Sf[nt][3]));
    }
    rmax_a=fmaxf(rmax_a,__shfl_xor_sync(0xffffffff,rmax_a,1));
    rmax_a=fmaxf(rmax_a,__shfl_xor_sync(0xffffffff,rmax_a,2));
    rmax_b=fmaxf(rmax_b,__shfl_xor_sync(0xffffffff,rmax_b,1));
    rmax_b=fmaxf(rmax_b,__shfl_xor_sync(0xffffffff,rmax_b,2));
    float new_m_a=fmaxf(m_a,rmax_a),new_m_b=fmaxf(m_b,rmax_b);
    float corr_a=(new_m_a==-INFINITY)?1.f:__expf(m_a-new_m_a);
    float corr_b=(new_m_b==-INFINITY)?1.f:__expf(m_b-new_m_b);
    float msub_a=(new_m_a==-INFINITY)?0.f:new_m_a;
    float msub_b=(new_m_b==-INFINITY)?0.f:new_m_b;
    uint32_t Pa[8],Pb[8];
    float part_a=0.f,part_b=0.f;
    #pragma unroll
    for(int nt=0;nt<8;nt++){
      float pa0=__expf(Sf[nt][0]-msub_a),pa1=__expf(Sf[nt][1]-msub_a);
      float pb0=__expf(Sf[nt][2]-msub_b),pb1=__expf(Sf[nt][3]-msub_b);
      part_a+=pa0+pa1;part_b+=pb0+pb1;
      Pa[nt]=pack2f(pa0,pa1);Pb[nt]=pack2f(pb0,pb1);
    }
    part_a+=__shfl_xor_sync(0xffffffff,part_a,1);part_a+=__shfl_xor_sync(0xffffffff,part_a,2);
    part_b+=__shfl_xor_sync(0xffffffff,part_b,1);part_b+=__shfl_xor_sync(0xffffffff,part_b,2);
    l_a=l_a*corr_a+part_a;l_b=l_b*corr_b+part_b;m_a=new_m_a;m_b=new_m_b;
    #pragma unroll
    for(int nt=0;nt<16;nt++){O_[nt][0]*=corr_a;O_[nt][1]*=corr_a;O_[nt][2]*=corr_b;O_[nt][3]*=corr_b;}

    // P @ V using ldmatrix.trans for V fragments
    #pragma unroll
    for(int kt=0;kt<4;kt++){
      uint32_t A0=Pa[2*kt],A1=Pb[2*kt],A2=Pa[2*kt+1],A3=Pb[2*kt+1];
      #pragma unroll
      for(int dp=0;dp<8;dp++){
        int vrow=lane%16, vcol=(lane/16)*8;
        uint32_t vaddr=(uint32_t)__cvta_generic_to_shared(&Vb[(16*kt+vrow)*D+16*dp+vcol]);
        uint32_t vr0,vr1,vr2,vr3;
        ldmatrix_x4_trans(vr0,vr1,vr2,vr3,vaddr);
        mma16816(O_[2*dp][0],O_[2*dp][1],O_[2*dp][2],O_[2*dp][3], A0,A1,A2,A3, vr0,vr1);
        mma16816(O_[2*dp+1][0],O_[2*dp+1][1],O_[2*dp+1][2],O_[2*dp+1][3], A0,A1,A2,A3, vr2,vr3);
      }
    }
    __syncthreads();
  }

  float inv_a=(l_a>0.f)?1.f/l_a:0.f, inv_b=(l_b>0.f)?1.f/l_b:0.f;
  #pragma unroll
  for(int nt=0;nt<16;nt++){
    int d0=8*nt+2*tidg;
    if(q0g<S){
      __nv_bfloat16 o0=__float2bfloat16(O_[nt][0]*inv_a);
      __nv_bfloat16 o1=__float2bfloat16(O_[nt][1]*inv_a);
      *reinterpret_cast<uint32_t*>(&Obh[(long)q0g*D+d0])=pack2h(o0,o1);
    }
    if(q1g<S){
      __nv_bfloat16 o2=__float2bfloat16(O_[nt][2]*inv_b);
      __nv_bfloat16 o3=__float2bfloat16(O_[nt][3]*inv_b);
      *reinterpret_cast<uint32_t*>(&Obh[(long)q1g*D+d0])=pack2h(o2,o3);
    }
  }
  if(tidg==0){
    if(q0g<S)LSEbh[q0g]=(l_a>0.f)?(m_a+logf(l_a)):(-INFINITY);
    if(q1g<S)LSEbh[q1g]=(l_b>0.f)?(m_b+logf(l_b)):(-INFINITY);
  }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE){
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int Bsz=Q.size(0), Hh=Q.size(1), S=Q.size(2);
  const __nv_bfloat16* Qp=static_cast<const __nv_bfloat16*>(Q.data_ptr());
  const __nv_bfloat16* Kp=static_cast<const __nv_bfloat16*>(K.data_ptr());
  const __nv_bfloat16* Vp=static_cast<const __nv_bfloat16*>(V.data_ptr());
  __nv_bfloat16* Op=static_cast<__nv_bfloat16*>(O.data_ptr());
  float* Lp=static_cast<float*>(LSE.data_ptr());

  int num_qtiles=(S+BM-1)/BM;
  dim3 grid(num_qtiles,Hh,Bsz);
  dim3 block(128);
  size_t smem=(size_t)(BM*D + 2*BN*D + 2*BN*D)*sizeof(__nv_bfloat16);

  cudaStream_t stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type,Q.device().device_id));
  cudaFuncSetAttribute(attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem);
  attn_kernel<<<grid,block,smem,stream>>>(Qp,Kp,Vp,Op,Lp,S,Hh);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_kernel::run);

}  // namespace mha_kernel