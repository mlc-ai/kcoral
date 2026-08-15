#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <math.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__); exit(1);} } while(0)

namespace mha_causal {

constexpr int Dh=128, BM=128, BN=128;

__device__ __forceinline__ uint32_t saddr(const void* p){ return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp16(uint32_t d,const void*s){ asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n"::"r"(d),"l"(s)); }
__device__ __forceinline__ void cpcommit(){ asm volatile("cp.async.commit_group;\n"); }
template<int N> __device__ __forceinline__ void cpwait(){ asm volatile("cp.async.wait_group %0;\n"::"n"(N):"memory"); }
__device__ __forceinline__ float fexp2(float x){ float y; asm volatile("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x)); return y; }
__device__ __forceinline__ void mbar_init(uint64_t* b,uint32_t c){ asm volatile("mbarrier.init.shared.b64 [%0], %1;"::"r"(saddr(b)),"r"(c)); }
__device__ __forceinline__ void mbar_wait(uint64_t* b,uint32_t ph){
  asm volatile("{\n.reg .pred P;\nW%=:\nmbarrier.try_wait.parity.shared.b64 P,[%0],%1;\n@!P bra W%=;\n}\n"::"r"(saddr(b)),"r"(ph));
}
__device__ __forceinline__ uint64_t make_desc(const void* p,uint32_t lbo,uint32_t sbo){
  uint64_t d=0; uint32_t a=saddr(p);
  d |= ((uint64_t)(a & 0x3FFFF)) >> 4;
  d |= ((uint64_t)((lbo>>4)&0x3FFF)) << 16;
  d |= ((uint64_t)((sbo>>4)&0x3FFF)) << 32;
  d |= ((uint64_t)1) << 46;   // version=1, swizzle none
  return d;
}
__device__ __forceinline__ void umma1(uint32_t td,uint64_t da,uint64_t db,uint32_t id,int acc){
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p,%4,0;\n"
    "tcgen05.mma.cta_group::1.kind::f16 [%0],%1,%2,%3,p;\n}\n"
    ::"r"(td),"l"(da),"l"(db),"r"(id),"r"(acc));
}
__device__ __forceinline__ void commit1(uint64_t* b){
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"::"r"(saddr(b)));
}
__device__ __forceinline__ void ld4(uint32_t col,uint32_t&r0,uint32_t&r1,uint32_t&r2,uint32_t&r3){
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3},[%4];":"=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3):"r"(col));
}
__device__ __forceinline__ void st4(uint32_t col,uint32_t r0,uint32_t r1,uint32_t r2,uint32_t r3){
  asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0],{%1,%2,%3,%4};"::"r"(col),"r"(r0),"r"(r1),"r"(r2),"r"(r3):"memory");
}
__device__ __forceinline__ void ldwait(){ asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory"); }
__device__ __forceinline__ void stwait(){ asm volatile("tcgen05.wait::st.sync.aligned;":::"memory"); }
__device__ __forceinline__ void fpa(){ asm volatile("fence.proxy.async;\n":::"memory"); }
__device__ __forceinline__ void tc_before(){ asm volatile("tcgen05.fence::before_thread_sync;":::"memory"); }
__device__ __forceinline__ void tc_after(){ asm volatile("tcgen05.fence::after_thread_sync;":::"memory"); }

__global__ __launch_bounds__(128,1)
void attn_kernel(const __nv_bfloat16* Q,const __nv_bfloat16* K,const __nv_bfloat16* V,
                 __nv_bfloat16* O,float* LSE,int S){
  extern __shared__ char smem_raw[];
  __nv_bfloat16* smem_Q =(__nv_bfloat16*)smem_raw;   // tiled K-major
  __nv_bfloat16* smem_K = smem_Q + BM*Dh;            // tiled K-major (single buf)
  __nv_bfloat16* smem_V = smem_K + BN*Dh;            // row-major staging
  __nv_bfloat16* smem_Vt= smem_V + BN*Dh;            // tiled K-major (transposed V)
  __nv_bfloat16* smem_P = smem_Vt + Dh*BN;           // tiled K-major (P weights)
  uint64_t* mbar=(uint64_t*)(smem_P + BM*BN);
  uint32_t* tmem_ptr=(uint32_t*)(mbar + 4);

  int bh=blockIdx.y;
  int q0=blockIdx.x*BM;
  int tid=threadIdx.x, warp=tid>>5;

  const __nv_bfloat16* Qb=Q+(size_t)bh*S*Dh;
  const __nv_bfloat16* Kb=K+(size_t)bh*S*Dh;
  const __nv_bfloat16* Vb=V+(size_t)bh*S*Dh;

  if(warp==0) asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"::"r"(saddr(tmem_ptr)),"r"(512));
  if(tid==0){ mbar_init(&mbar[0],1); mbar_init(&mbar[1],1); }
  asm volatile("fence.mbarrier_init.release.cluster;\n":::"memory");
  __syncthreads();
  uint32_t tbase=*tmem_ptr;
  uint32_t S0=tbase, S1=tbase+128, O_t=tbase+256;
  uint64_t* mbq=&mbar[0]; uint64_t* mbp=&mbar[1];

  const float scale2=0.08838834764831845f*1.4426950408889634f;
  const float ln2=0.6931471805599453f;
  const float NEG=-1e30f;

  // Q -> smem_Q tiled
  for(int i=tid;i<BM*Dh/8;i+=128){
    int r=i/(Dh/8), c=i%(Dh/8); int gr=q0+r;
    __nv_bfloat16* dst=&smem_Q[c*1024 + r*8];
    if(gr<S) cp16(saddr(dst), &Qb[(size_t)gr*Dh + c*8]); else *(uint4*)dst=make_uint4(0,0,0,0);
  }
  cpcommit(); cpwait<0>(); __syncthreads();

  int max_q=q0+BM-1; if(max_q>S-1) max_q=S-1;
  int num_kt=max_q/BN + 1;
  int qrow=q0+tid;
  uint32_t idesc=(1u<<4)|(1u<<7)|(1u<<10)|((BN/8)<<17)|((BM/16)<<24);

  float m=NEG, l=0.f;
  uint32_t phq=0, php=0;

  // prologue: load K[0], issue QK[0]->S0
  for(int i=tid;i<BN*Dh/8;i+=128){
    int r=i/(Dh/8), c=i%(Dh/8); int gk=r;
    __nv_bfloat16* dst=&smem_K[c*1024 + r*8];
    if(gk<S) cp16(saddr(dst), &Kb[(size_t)gk*Dh + c*8]); else *(uint4*)dst=make_uint4(0,0,0,0);
  }
  cpcommit(); cpwait<0>(); __syncthreads();
  if(tid==0){
    fpa(); tc_after();
    #pragma unroll
    for(int kk=0;kk<8;kk++)
      umma1(S0, make_desc(&smem_Q[kk*2048],2048,128), make_desc(&smem_K[kk*2048],2048,128), idesc, kk>0);
    tc_before(); commit1(mbq);
  }

  for(int kt=0;kt<num_kt;kt++){
    int cur=kt&1; uint32_t Scur=cur?S1:S0; uint32_t Snxt=cur?S0:S1;
    int kv0=kt*BN;

    mbar_wait(mbq, phq); phq^=1;   // S[cur] ready

    // load K[kt+1] (single buffer, K[kt] consumed) and V[kt]
    if(kt+1<num_kt){
      int kb=(kt+1)*BN;
      for(int i=tid;i<BN*Dh/8;i+=128){
        int r=i/(Dh/8), c=i%(Dh/8); int gk=kb+r;
        __nv_bfloat16* dst=&smem_K[c*1024 + r*8];
        if(gk<S) cp16(saddr(dst), &Kb[(size_t)gk*Dh + c*8]); else *(uint4*)dst=make_uint4(0,0,0,0);
      }
    }
    for(int i=tid;i<BN*Dh/8;i+=128){
      int r=i/(Dh/8), c=i%(Dh/8); int gk=kv0+r;
      __nv_bfloat16* dst=&smem_V[r*Dh + c*8];
      if(gk<S) cp16(saddr(dst), &Vb[(size_t)gk*Dh + c*8]); else *(uint4*)dst=make_uint4(0,0,0,0);
    }
    cpcommit(); cpwait<0>(); __syncthreads();

    bool need_mask = (kv0+BN-1 >= q0) || (kv0+BN > S);

    // ---- softmax pass A: row max (short, not overlapped) ----
    tc_after();
    float rmax=NEG;
    #pragma unroll
    for(int col=0;col<BN;col+=4){
      uint32_t r0,r1,r2,r3; ld4(Scur+col,r0,r1,r2,r3); ldwait();
      float s0=__uint_as_float(r0)*scale2, s1=__uint_as_float(r1)*scale2;
      float s2=__uint_as_float(r2)*scale2, s3=__uint_as_float(r3)*scale2;
      if(need_mask){ int b=kv0+col;
        if(!(b+0<=qrow&&b+0<S)) s0=NEG; if(!(b+1<=qrow&&b+1<S)) s1=NEG;
        if(!(b+2<=qrow&&b+2<S)) s2=NEG; if(!(b+3<=qrow&&b+3<S)) s3=NEG; }
      rmax=fmaxf(rmax,fmaxf(fmaxf(s0,s1),fmaxf(s2,s3)));
    }
    float mnew=fmaxf(m,rmax);
    float alpha=fexp2(m-mnew);

    // ---- issue QK[kt+1] (overlaps rescale + pass B + transpose) ----
    if(kt+1<num_kt && tid==0){
      fpa(); tc_after();
      #pragma unroll
      for(int kk=0;kk<8;kk++)
        umma1(Snxt, make_desc(&smem_Q[kk*2048],2048,128), make_desc(&smem_K[kk*2048],2048,128), idesc, kk>0);
      tc_before(); commit1(mbq);
    }

    // ---- wait previous PV, rescale O ----
    if(kt>0){ mbar_wait(mbp, php); php^=1; }
    if(kt>0){
      #pragma unroll
      for(int col=0;col<Dh;col+=4){
        uint32_t r0,r1,r2,r3; ld4(O_t+col,r0,r1,r2,r3); ldwait();
        st4(O_t+col, __float_as_uint(__uint_as_float(r0)*alpha), __float_as_uint(__uint_as_float(r1)*alpha),
                     __float_as_uint(__uint_as_float(r2)*alpha), __float_as_uint(__uint_as_float(r3)*alpha));
      }
      stwait();
    }

    // ---- softmax pass B: exp -> smem_P (tiled), rowsum ----
    float rsum=0.f;
    #pragma unroll
    for(int col=0;col<BN;col+=4){
      uint32_t r0,r1,r2,r3; ld4(Scur+col,r0,r1,r2,r3); ldwait();
      float s0=__uint_as_float(r0)*scale2, s1=__uint_as_float(r1)*scale2;
      float s2=__uint_as_float(r2)*scale2, s3=__uint_as_float(r3)*scale2;
      float p0,p1,p2,p3;
      if(need_mask){ int b=kv0+col;
        p0=(b+0<=qrow&&b+0<S)?fexp2(s0-mnew):0.f; p1=(b+1<=qrow&&b+1<S)?fexp2(s1-mnew):0.f;
        p2=(b+2<=qrow&&b+2<S)?fexp2(s2-mnew):0.f; p3=(b+3<=qrow&&b+3<S)?fexp2(s3-mnew):0.f;
      }else{ p0=fexp2(s0-mnew); p1=fexp2(s1-mnew); p2=fexp2(s2-mnew); p3=fexp2(s3-mnew); }
      rsum+=p0+p1+p2+p3;
      int base=(col>>3)*1024 + tid*8 + (col&7);
      smem_P[base+0]=__float2bfloat16(p0); smem_P[base+1]=__float2bfloat16(p1);
      smem_P[base+2]=__float2bfloat16(p2); smem_P[base+3]=__float2bfloat16(p3);
    }
    l = alpha*l + rsum; m = mnew;

    // ---- transpose V: smem_V (row major) -> smem_Vt (tiled K-major) ----
    #pragma unroll
    for(int cc=0; cc<BN/8; cc++){
      uint16_t buf[8];
      #pragma unroll
      for(int j=0;j<8;j++){ int kv=cc*8+j; buf[j]=*reinterpret_cast<uint16_t*>(&smem_V[kv*Dh + tid]); }
      *reinterpret_cast<uint4*>(&smem_Vt[cc*1024 + tid*8])=*reinterpret_cast<uint4*>(buf);
    }
    __syncthreads();

    // ---- PV: O += P @ V ----
    if(tid==0){
      fpa(); tc_after();
      #pragma unroll
      for(int kk=0;kk<8;kk++)
        umma1(O_t, make_desc(&smem_P[kk*2048],2048,128), make_desc(&smem_Vt[kk*2048],2048,128), idesc, (kt>0)||(kk>0));
      tc_before(); commit1(mbp);
    }
  }

  mbar_wait(mbp, php); php^=1;
  tc_after();

  float invl=(l>0.f)?(1.f/l):0.f;
  size_t ob=(size_t)bh*S*Dh;
  #pragma unroll
  for(int col=0;col<Dh;col+=4){
    uint32_t r0,r1,r2,r3; ld4(O_t+col,r0,r1,r2,r3); ldwait();
    if(qrow<S){
      __nv_bfloat16* out=&O[ob+(size_t)qrow*Dh+col];
      out[0]=__float2bfloat16(__uint_as_float(r0)*invl); out[1]=__float2bfloat16(__uint_as_float(r1)*invl);
      out[2]=__float2bfloat16(__uint_as_float(r2)*invl); out[3]=__float2bfloat16(__uint_as_float(r3)*invl);
    }
  }
  if(qrow<S) LSE[(size_t)bh*S+qrow]=m*ln2+logf(l);

  __syncthreads();
  if(warp==0) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"::"r"(tbase),"r"(512));
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE){
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int64_t B=Q.size(0), Hh=Q.size(1), S=Q.size(2);
  const __nv_bfloat16* q=static_cast<const __nv_bfloat16*>(Q.data_ptr());
  const __nv_bfloat16* k=static_cast<const __nv_bfloat16*>(K.data_ptr());
  const __nv_bfloat16* v=static_cast<const __nv_bfloat16*>(V.data_ptr());
  __nv_bfloat16* o=static_cast<__nv_bfloat16*>(O.data_ptr());
  float* lse=static_cast<float*>(LSE.data_ptr());
  cudaStream_t stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type,Q.device().device_id));

  size_t smem=(size_t)(BM*Dh + 3*BN*Dh + BM*BN)*2 + 128;
  static bool set=false;
  if(!set){ cudaFuncSetAttribute(attn_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem); set=true; }

  dim3 grid((unsigned)((S+BM-1)/BM),(unsigned)(B*Hh));
  attn_kernel<<<grid,128,smem,stream>>>(q,k,v,o,lse,(int)S);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_causal::run);

}  // namespace mha_causal