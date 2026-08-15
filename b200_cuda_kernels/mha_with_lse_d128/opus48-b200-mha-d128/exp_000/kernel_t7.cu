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

__device__ __forceinline__ float uf(uint32_t x){ return __uint_as_float(x); }
__device__ __forceinline__ uint16_t f2b(float x){ bf16 b=__float2bfloat16(x); return *reinterpret_cast<uint16_t*>(&b); }
__device__ __forceinline__ float ex2m(float x){ float y; asm("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x)); return y; }
__device__ __forceinline__ float ex2p(float x){
  float ff=floorf(x); float f=x-ff;
  float p=0.0771f; p=fmaf(p,f,0.2276f); p=fmaf(p,f,0.6951f); p=fmaf(p,f,1.0f);
  int n=(int)ff; int bits=__float_as_int(p)+(n<<23); return __int_as_float(bits);
}

__device__ __forceinline__ void cp16(void* s,const void* g){
  uint32_t sa=(uint32_t)__cvta_generic_to_shared(s);
  asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n"::"r"(sa),"l"(g):"memory");
}
__device__ __forceinline__ void cp_commit(){asm volatile("cp.async.commit_group;\n":::"memory");}
__device__ __forceinline__ void cp_wait0(){asm volatile("cp.async.wait_group 0;\n":::"memory");}

__device__ __forceinline__ void ld_x4(uint32_t t,uint32_t*r0,uint32_t*r1,uint32_t*r2,uint32_t*r3){
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
    :"=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3):"r"(t));
}
__device__ __forceinline__ void st_x4(uint32_t t,uint32_t r0,uint32_t r1,uint32_t r2,uint32_t r3){
  asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};"
    ::"r"(t),"r"(r0),"r"(r1),"r"(r2),"r"(r3));
}
__device__ __forceinline__ void wait_ld(){asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory");}
__device__ __forceinline__ void wait_st(){asm volatile("tcgen05.wait::st.sync.aligned;":::"memory");}
__device__ __forceinline__ void fence_before(){asm volatile("tcgen05.fence::before_thread_sync;":::"memory");}
__device__ __forceinline__ void fence_after(){asm volatile("tcgen05.fence::after_thread_sync;":::"memory");}
__device__ __forceinline__ void fence_pa(){asm volatile("fence.proxy.async;\n":::"memory");}

__device__ __forceinline__ uint64_t make_desc(const void* p){
  uint64_t d=0; uint32_t a=(uint32_t)__cvta_generic_to_shared(p);
  d |= (uint64_t)((a & 0x3FFFF)>>4);
  d |= (uint64_t)((2048u & 0x3FFFF)>>4) << 16;
  d |= (uint64_t)((128u  & 0x3FFFF)>>4) << 32;
  d |= (uint64_t)1 << 46;
  return d;
}
__device__ __forceinline__ uint32_t make_idesc(){
  uint32_t d=0; d|=(1u<<4); d|=(1u<<7); d|=(1u<<10); d|=(16u<<17); d|=(8u<<24); return d;
}
__device__ __forceinline__ void umma(uint32_t t,uint64_t da,uint64_t db,uint32_t id,int en){
  asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
    ::"r"(t),"l"(da),"l"(db),"r"(id),"r"(en));
}
__device__ __forceinline__ void umma_commit(uint64_t* mbar){
  uint32_t a=(uint32_t)__cvta_generic_to_shared(mbar);
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"::"r"(a));
}
__device__ __forceinline__ void mbar_init(uint64_t* b,uint32_t c){
  asm volatile("mbarrier.init.shared.b64 [%0], %1;"::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(c));
}
__device__ __forceinline__ void mbar_wait(uint64_t* b,uint32_t ph){
  asm volatile("{\n.reg .pred P;\nW%=:\nmbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n@!P bra W%=;\n}\n"
    ::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(ph));
}
__device__ __forceinline__ void tmem_alloc1(uint32_t* dst,int nc){
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
    ::"r"((uint32_t)__cvta_generic_to_shared(dst)),"r"(nc));
}
__device__ __forceinline__ void tmem_relinquish1(){asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");}
__device__ __forceinline__ void tmem_dealloc1(uint32_t a,int nc){asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"::"r"(a),"r"(nc));}

__device__ __forceinline__ void rescale_O(uint32_t Obase,float corr){
  #pragma unroll
  for(int c=0;c<128;c+=4){
    uint32_t r0,r1,r2,r3; ld_x4(Obase+c,&r0,&r1,&r2,&r3); wait_ld();
    r0=__float_as_uint(uf(r0)*corr); r1=__float_as_uint(uf(r1)*corr);
    r2=__float_as_uint(uf(r2)*corr); r3=__float_as_uint(uf(r3)*corr);
    st_x4(Obase+c,r0,r1,r2,r3);
  }
  wait_st();
}

__global__ __launch_bounds__(128) void attn(const bf16* Q,const bf16* K,const bf16* V,bf16* O,float* LSE,int S){
  extern __shared__ char smem[];
  bf16* Qs =(bf16*)(smem+0);
  bf16* Bx =(bf16*)(smem+32768);
  bf16* Cx =(bf16*)(smem+65536);
  uint64_t* mbar=(uint64_t*)(smem+98304);
  uint32_t* tmslot=(uint32_t*)(smem+98304+16);

  int tid=threadIdx.x;
  int bh=blockIdx.y, qstart=blockIdx.x*128;
  const bf16* Qb=Q+(long)bh*S*128;
  const bf16* Kb=K+(long)bh*S*128;
  const bf16* Vb=V+(long)bh*S*128;
  bf16* Ob=O+(long)bh*S*128;
  float* LSEb=LSE+(long)bh*S;
  const float scale_log2 = 0.08838834764831843f * 1.4426950408889634f;
  const float LN2 = 0.6931471805599453f;
  const float TAU = 16.0f;
  uint32_t idesc=make_idesc();

  if(tid<32) tmem_alloc1(tmslot,256);
  if(tid<32) tmem_relinquish1();
  if(tid==0){ mbar_init(&mbar[0],1); mbar_init(&mbar[1],1); }
  asm volatile("fence.mbarrier_init.release.cluster;\n":::"memory");
  __syncthreads();
  uint32_t tbase=tmslot[0];
  uint32_t Sbase=tbase, Obase=tbase+128;

  for(int idx=tid; idx<128*16; idx+=128){
    int mm=idx>>4,c=idx&15; int qg=qstart+mm; if(qg>=S)qg=S-1;
    cp16(&Qs[(c<<10)+(mm<<3)], &Qb[(long)qg*128+8*c]);
  }

  int nkv=(S+127)/128;
  int qph=0,pph=0;
  float m=-1e30f, l=0.f, mo=-1e30f;

  for(int blk=0; blk<nkv; blk++){
    int kvstart=blk*128;
    for(int idx=tid; idx<128*16; idx+=128){
      int key=idx>>4,c=idx&15; int kg=kvstart+key; if(kg>=S)kg=S-1;
      cp16(&Bx[(c<<10)+(key<<3)], &Kb[(long)kg*128+8*c]);
    }
    for(int idx=tid; idx<128*16; idx+=128){
      int key=idx>>4,c=idx&15; int kg=kvstart+key; if(kg>=S)kg=S-1;
      cp16(&Cx[key*128+8*c], &Vb[(long)kg*128+8*c]);
    }
    cp_commit(); cp_wait0(); __syncthreads();
    fence_pa(); __syncthreads();

    if(tid==0){
      #pragma unroll
      for(int j=0;j<8;j++) umma(Sbase, make_desc(Qs+j*2048), make_desc(Bx+j*2048), idesc, j?1:0);
      umma_commit(&mbar[0]);
    }
    mbar_wait(&mbar[0], qph); qph^=1;

    // transpose Vps(Cx) -> Vts(Bx)
    for(int idx=tid; idx<128*16; idx+=128){
      int d=idx>>4,b=idx&15; uint16_t v[8];
      #pragma unroll
      for(int e=0;e<8;e++) v[e]=((uint16_t*)Cx)[(8*b+e)*128 + d];
      *(uint4*)&Bx[(b<<10)+(d<<3)] = *(uint4*)v;
    }
    __syncthreads();

    // ---- softmax : single batched TMEM read of full row ----
    uint32_t R[128];
    #pragma unroll
    for(int i=0;i<32;i++) ld_x4(Sbase+4*i,&R[4*i],&R[4*i+1],&R[4*i+2],&R[4*i+3]);
    wait_ld();
    float mb=-1e30f;
    #pragma unroll
    for(int c=0;c<128;c++){
      int key=kvstart+c; float x=uf(R[c])*scale_log2; if(key>=S)x=-1e30f;
      R[c]=__float_as_uint(x); mb=fmaxf(mb,x);
    }
    float m_new=fmaxf(m,mb);
    bool first=(mo<-1e29f);
    bool need=(!first)&&(m_new-mo>TAU);
    float corr=1.f;
    if(first){ mo=m_new; }
    else if(need){ corr=ex2m(mo-m_new); mo=m_new; }

    float psum=0.f;
    #pragma unroll
    for(int b=0;b<16;b++){
      uint16_t pv[8];
      #pragma unroll
      for(int e=0;e<8;e++){
        float arg=uf(R[8*b+e])-mo; float ex;
        if(arg<-80.f) ex=0.f;
        else if(e&1)  ex=ex2p(arg);   // FMA-emulated
        else          ex=ex2m(arg);   // MUFU
        psum+=ex; pv[e]=f2b(ex);
      }
      *(uint4*)&Cx[(b<<10)+(tid<<3)] = *(uint4*)pv;
    }
    float e_mo=ex2m(mo-m_new);
    float e_m =(m<-1e29f)?0.f:ex2m(m-m_new);
    l = l*e_m + e_mo*psum;
    m = m_new;

    unsigned wneed=__any_sync(0xffffffffu, need);
    if(wneed) rescale_O(Obase, corr);

    fence_pa();
    fence_before();
    __syncthreads();

    if(tid==0){
      fence_after();
      #pragma unroll
      for(int j=0;j<8;j++) umma(Obase, make_desc(Cx+j*2048), make_desc(Bx+j*2048), idesc, (blk==0&&j==0)?0:1);
      umma_commit(&mbar[1]);
    }
    mbar_wait(&mbar[1], pph); pph^=1;
    __syncthreads();
  }

  int qg=qstart+tid;
  float ofac=ex2m(mo-m)/l;
  #pragma unroll
  for(int c=0;c<128;c+=4){
    uint32_t r0,r1,r2,r3; ld_x4(Obase+c,&r0,&r1,&r2,&r3); wait_ld();
    if(qg<S){
      uint16_t o[4];
      o[0]=f2b(uf(r0)*ofac); o[1]=f2b(uf(r1)*ofac);
      o[2]=f2b(uf(r2)*ofac); o[3]=f2b(uf(r3)*ofac);
      *(uint2*)&Ob[(long)qg*128+c] = *(uint2*)o;
    }
  }
  if(qg<S) LSEb[qg]=m*LN2+logf(l);

  __syncthreads();
  if(tid<32) tmem_dealloc1(tbase,256);
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

  int smem_bytes = 98304 + 32;
  static bool cfg=false;
  if(!cfg){ CUDA_CHECK(cudaFuncSetAttribute(attn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes)); cfg=true; }

  dim3 grid((S+127)/128, B*H);
  dim3 block(128);
  cudaStream_t stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
  attn<<<grid,block,smem_bytes,stream>>>(Qd,Kd,Vd,Od,LSEd,S);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha::run);

}  // namespace mha