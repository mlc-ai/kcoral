#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <math.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
    fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__); exit(1);} }while(0)

namespace mha_kernel {

constexpr int Dc = 128;
constexpr int BQ = 64;
constexpr int BK = 32;
constexpr int SD = Dc + 8;   // padded stride (bank-conflict free, int4-aligned)

__device__ __forceinline__ void mma16816(float &d0,float &d1,float &d2,float &d3,
    uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,uint32_t b0,uint32_t b1){
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)
    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
__device__ __forceinline__ uint32_t packf(float x,float y){
  __nv_bfloat162 r = __floats2bfloat162_rn(x,y);
  return *reinterpret_cast<uint32_t*>(&r);
}
__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t &d0,uint32_t &d1,const void* addr){
  uint32_t a=(uint32_t)__cvta_generic_to_shared(addr);
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.b16 {%0,%1}, [%2];\n"
    : "=r"(d0),"=r"(d1) : "r"(a));
}
__device__ __forceinline__ void cp_async_cg16(void* smem,const void* gmem){
  unsigned s=(unsigned)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s),"l"(gmem):"memory");
}
__device__ __forceinline__ void cp_async_commit(){ asm volatile("cp.async.commit_group;\n":::"memory"); }
template<int N> __device__ __forceinline__ void cp_async_wait(){ asm volatile("cp.async.wait_group %0;\n"::"n"(N):"memory"); }

__device__ __forceinline__ void load_kv(__nv_bfloat16* Ks,__nv_bfloat16* Vs,
    const __nv_bfloat16* Kbase,const __nv_bfloat16* Vbase,int kb,int S,int tid){
  #pragma unroll
  for(int it=0; it<(BK*16)/128; it++){
    int v=tid+it*128; int kk=v>>4; int d=(v&15)*8;
    int key=kb*BK+kk;
    if(key<S){
      cp_async_cg16(&Ks[kk*SD+d], Kbase+(size_t)key*Dc+d);
      cp_async_cg16(&Vs[kk*SD+d], Vbase+(size_t)key*Dc+d);
    } else {
      *reinterpret_cast<int4*>(&Ks[kk*SD+d]) = make_int4(0,0,0,0);
      *reinterpret_cast<int4*>(&Vs[kk*SD+d]) = make_int4(0,0,0,0);
    }
  }
}

__global__ __launch_bounds__(128) void mha_kernel_fn(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int S, int H){
  extern __shared__ __nv_bfloat16 smem[];
  __nv_bfloat16* Qs = smem;
  __nv_bfloat16* Ksb[2]; __nv_bfloat16* Vsb[2];
  Ksb[0]=Qs+BQ*SD; Ksb[1]=Ksb[0]+BK*SD;
  Vsb[0]=Ksb[1]+BK*SD; Vsb[1]=Vsb[0]+BK*SD;

  int b=blockIdx.z, h=blockIdx.y, qbase=blockIdx.x*BQ;
  int tid=threadIdx.x, warp=tid>>5, lane=tid&31;
  int gid=lane>>2, tig=lane&3;

  const float scale = 0.08838834764831845f;

  size_t headOff=(size_t)(b*H+h)*S;
  const __nv_bfloat16* Qbase=Q+headOff*Dc;
  const __nv_bfloat16* Kbase=K+headOff*Dc;
  const __nv_bfloat16* Vbase=V+headOff*Dc;
  __nv_bfloat16* Obase=O+headOff*Dc;
  float* LSEbase=LSE+headOff;

  // ---- load Q tile (zero OOB) ----
  #pragma unroll
  for(int it=0; it<(BQ*16)/128; it++){
    int v=tid+it*128; int kk=v>>4; int d=(v&15)*8; int row=qbase+kk;
    int4 val=(row<S)?*reinterpret_cast<const int4*>(Qbase+(size_t)row*Dc+d):make_int4(0,0,0,0);
    *reinterpret_cast<int4*>(&Qs[kk*SD+d])=val;
  }
  __syncthreads();

  // ---- persistent Q fragments ----
  uint32_t qf[8][4];
  {
    int r0=warp*16+gid, r8=r0+8;
    #pragma unroll
    for(int kt=0;kt<8;kt++){
      int d0=kt*16+tig*2, d8=d0+8;
      qf[kt][0]=*reinterpret_cast<const uint32_t*>(&Qs[r0*SD+d0]);
      qf[kt][1]=*reinterpret_cast<const uint32_t*>(&Qs[r8*SD+d0]);
      qf[kt][2]=*reinterpret_cast<const uint32_t*>(&Qs[r0*SD+d8]);
      qf[kt][3]=*reinterpret_cast<const uint32_t*>(&Qs[r8*SD+d8]);
    }
  }

  float o[16][4];
  #pragma unroll
  for(int i=0;i<16;i++){o[i][0]=o[i][1]=o[i][2]=o[i][3]=0.f;}
  float m0=-INFINITY,m1=-INFINITY,l0=0.f,l1=0.f;

  int num_kb=(S+BK-1)/BK;
  load_kv(Ksb[0],Vsb[0],Kbase,Vbase,0,S,tid);
  cp_async_commit();

  for(int kb=0; kb<num_kb; kb++){
    int cur=kb&1, nxt=(kb+1)&1;
    bool has_next=(kb+1<num_kb);
    if(has_next){ load_kv(Ksb[nxt],Vsb[nxt],Kbase,Vbase,kb+1,S,tid); cp_async_commit(); }
    if(has_next) cp_async_wait<1>(); else cp_async_wait<0>();
    __syncthreads();

    __nv_bfloat16* Ks=Ksb[cur];
    __nv_bfloat16* Vs=Vsb[cur];

    // ---- QK^T ----
    float sc[4][4];
    #pragma unroll
    for(int nt=0;nt<4;nt++){sc[nt][0]=sc[nt][1]=sc[nt][2]=sc[nt][3]=0.f;}
    #pragma unroll
    for(int nt=0;nt<4;nt++){
      int key=nt*8+gid;
      #pragma unroll
      for(int kt=0;kt<8;kt++){
        int d0=kt*16+tig*2;
        uint32_t b0=*reinterpret_cast<const uint32_t*>(&Ks[key*SD+d0]);
        uint32_t b1=*reinterpret_cast<const uint32_t*>(&Ks[key*SD+d0+8]);
        mma16816(sc[nt][0],sc[nt][1],sc[nt][2],sc[nt][3],
                 qf[kt][0],qf[kt][1],qf[kt][2],qf[kt][3], b0,b1);
      }
    }
    // ---- scale + mask ----
    #pragma unroll
    for(int nt=0;nt<4;nt++){
      int kcol0=nt*8+tig*2; int g0=kb*BK+kcol0, g1=g0+1;
      sc[nt][0]*=scale; sc[nt][1]*=scale; sc[nt][2]*=scale; sc[nt][3]*=scale;
      if(g0>=S){ sc[nt][0]=-INFINITY; sc[nt][2]=-INFINITY; }
      if(g1>=S){ sc[nt][1]=-INFINITY; sc[nt][3]=-INFINITY; }
    }
    // ---- row max ----
    float lm0=-INFINITY, lm1=-INFINITY;
    #pragma unroll
    for(int nt=0;nt<4;nt++){
      lm0=fmaxf(lm0,fmaxf(sc[nt][0],sc[nt][1]));
      lm1=fmaxf(lm1,fmaxf(sc[nt][2],sc[nt][3]));
    }
    lm0=fmaxf(lm0,__shfl_xor_sync(0xffffffff,lm0,1));
    lm0=fmaxf(lm0,__shfl_xor_sync(0xffffffff,lm0,2));
    lm1=fmaxf(lm1,__shfl_xor_sync(0xffffffff,lm1,1));
    lm1=fmaxf(lm1,__shfl_xor_sync(0xffffffff,lm1,2));
    float nm0=fmaxf(m0,lm0), nm1=fmaxf(m1,lm1);
    float al0=__expf(m0-nm0), al1=__expf(m1-nm1);
    // ---- P = exp(sc - m) ----
    float ls0=0.f, ls1=0.f;
    #pragma unroll
    for(int nt=0;nt<4;nt++){
      sc[nt][0]=__expf(sc[nt][0]-nm0); ls0+=sc[nt][0];
      sc[nt][1]=__expf(sc[nt][1]-nm0); ls0+=sc[nt][1];
      sc[nt][2]=__expf(sc[nt][2]-nm1); ls1+=sc[nt][2];
      sc[nt][3]=__expf(sc[nt][3]-nm1); ls1+=sc[nt][3];
    }
    ls0+=__shfl_xor_sync(0xffffffff,ls0,1); ls0+=__shfl_xor_sync(0xffffffff,ls0,2);
    ls1+=__shfl_xor_sync(0xffffffff,ls1,1); ls1+=__shfl_xor_sync(0xffffffff,ls1,2);
    l0=l0*al0+ls0; l1=l1*al1+ls1;
    m0=nm0; m1=nm1;
    // ---- rescale O ----
    #pragma unroll
    for(int i=0;i<16;i++){ o[i][0]*=al0; o[i][1]*=al0; o[i][2]*=al1; o[i][3]*=al1; }
    // ---- O += P @ V (ldmatrix.trans for V) ----
    #pragma unroll
    for(int kt=0;kt<2;kt++){
      uint32_t a0=packf(sc[2*kt][0],  sc[2*kt][1]);
      uint32_t a1=packf(sc[2*kt][2],  sc[2*kt][3]);
      uint32_t a2=packf(sc[2*kt+1][0],sc[2*kt+1][1]);
      uint32_t a3=packf(sc[2*kt+1][2],sc[2*kt+1][3]);
      #pragma unroll
      for(int ntv=0;ntv<16;ntv++){
        const __nv_bfloat16* vaddr=&Vs[(kt*16 + (lane & 15))*SD + ntv*8];
        uint32_t b0,b1;
        ldmatrix_x2_trans(b0,b1,vaddr);
        mma16816(o[ntv][0],o[ntv][1],o[ntv][2],o[ntv][3], a0,a1,a2,a3, b0,b1);
      }
    }
    __syncthreads();
  }

  // ---- finalize ----
  float inv0=(l0>0.f)?1.f/l0:0.f;
  float inv1=(l1>0.f)?1.f/l1:0.f;
  int row0=qbase+warp*16+gid, row1=row0+8;
  #pragma unroll
  for(int ntv=0;ntv<16;ntv++){
    int d0=ntv*8+tig*2, d1=d0+1;
    if(row0<S){
      Obase[(size_t)row0*Dc+d0]=__float2bfloat16(o[ntv][0]*inv0);
      Obase[(size_t)row0*Dc+d1]=__float2bfloat16(o[ntv][1]*inv0);
    }
    if(row1<S){
      Obase[(size_t)row1*Dc+d0]=__float2bfloat16(o[ntv][2]*inv1);
      Obase[(size_t)row1*Dc+d1]=__float2bfloat16(o[ntv][3]*inv1);
    }
  }
  if(tig==0){
    if(row0<S) LSEbase[row0]=m0+logf(l0);
    if(row1<S) LSEbase[row1]=m1+logf(l1);
  }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE){
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int B=Q.size(0), H=Q.size(1), S=Q.size(2);

  const __nv_bfloat16* Qp=static_cast<const __nv_bfloat16*>(Q.data_ptr());
  const __nv_bfloat16* Kp=static_cast<const __nv_bfloat16*>(K.data_ptr());
  const __nv_bfloat16* Vp=static_cast<const __nv_bfloat16*>(V.data_ptr());
  __nv_bfloat16* Op=static_cast<__nv_bfloat16*>(O.data_ptr());
  float* Lp=static_cast<float*>(LSE.data_ptr());

  cudaStream_t stream=
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

  dim3 grid((S+BQ-1)/BQ, H, B);
  dim3 block(128);
  size_t smem=(size_t)(BQ*SD + 2*BK*SD + 2*BK*SD)*sizeof(__nv_bfloat16);

  static bool attr_set=false;
  if(!attr_set){
    CUDA_CHECK(cudaFuncSetAttribute(mha_kernel_fn,
        cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));
    attr_set=true;
  }

  mha_kernel_fn<<<grid,block,smem,stream>>>(Qp,Kp,Vp,Op,Lp,S,H);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_kernel::run);

}  // namespace mha_kernel