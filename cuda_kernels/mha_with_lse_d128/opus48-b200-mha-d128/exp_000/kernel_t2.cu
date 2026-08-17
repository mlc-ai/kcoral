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

// canonical K-major no-swizzle offset for a 128x128 tile: off = (col/8)*1024 + row*8 + (col%8)
__device__ __forceinline__ uint32_t caddr(int row,int col){ return ((col>>3)<<10) + (row<<3) + (col&7); }

__device__ __forceinline__ void cp16(void* s,const void* g){
  uint32_t sa=(uint32_t)__cvta_generic_to_shared(s);
  asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n"::"r"(sa),"l"(g):"memory");
}
__device__ __forceinline__ void cp_commit(){asm volatile("cp.async.commit_group;\n":::"memory");}
__device__ __forceinline__ void cp_wait0(){asm volatile("cp.async.wait_group 0;\n":::"memory");}

__device__ __forceinline__ void ld_x4(uint32_t taddr,uint32_t*r0,uint32_t*r1,uint32_t*r2,uint32_t*r3){
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
    :"=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3):"r"(taddr));
}
__device__ __forceinline__ void st_x4(uint32_t taddr,uint32_t r0,uint32_t r1,uint32_t r2,uint32_t r3){
  asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};"
    ::"r"(taddr),"r"(r0),"r"(r1),"r"(r2),"r"(r3));
}
__device__ __forceinline__ void wait_ld(){asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory");}
__device__ __forceinline__ void wait_st(){asm volatile("tcgen05.wait::st.sync.aligned;":::"memory");}
__device__ __forceinline__ void fence_before(){asm volatile("tcgen05.fence::before_thread_sync;":::"memory");}
__device__ __forceinline__ void fence_after(){asm volatile("tcgen05.fence::after_thread_sync;":::"memory");}
__device__ __forceinline__ void fence_pa(){asm volatile("fence.proxy.async;\n":::"memory");}

__device__ __forceinline__ uint64_t make_desc(const void* p){
  uint64_t d=0; uint32_t a=(uint32_t)__cvta_generic_to_shared(p);
  d |= (uint64_t)((a & 0x3FFFF)>>4);
  d |= (uint64_t)((2048u & 0x3FFFF)>>4) << 16; // LBO=2048
  d |= (uint64_t)((128u  & 0x3FFFF)>>4) << 32; // SBO=128
  d |= (uint64_t)1 << 46;   // version SM100
  d |= (uint64_t)0 << 61;   // swizzle none
  return d;
}
__device__ __forceinline__ uint32_t make_idesc(){
  uint32_t d=0;
  d |= (1u<<4);   // c=fp32
  d |= (1u<<7);   // a=bf16
  d |= (1u<<10);  // b=bf16
  d |= (16u<<17); // N=128
  d |= (8u<<24);  // M=128
  return d;
}
__device__ __forceinline__ void umma(uint32_t taddr,uint64_t da,uint64_t db,uint32_t idesc,int en){
  asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
    ::"r"(taddr),"l"(da),"l"(db),"r"(idesc),"r"(en));
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

#define BM 128
#define BN 128
#define DH 128

__device__ __forceinline__ void ld8(uint32_t base8,float* f){
  uint32_t r0,r1,r2,r3,r4,r5,r6,r7;
  ld_x4(base8,&r0,&r1,&r2,&r3);
  ld_x4(base8+4,&r4,&r5,&r6,&r7);
  wait_ld();
  f[0]=uf(r0);f[1]=uf(r1);f[2]=uf(r2);f[3]=uf(r3);
  f[4]=uf(r4);f[5]=uf(r5);f[6]=uf(r6);f[7]=uf(r7);
}

__global__ __launch_bounds__(128) void attn(const bf16* Q,const bf16* K,const bf16* V,bf16* O,float* LSE,int S){
  extern __shared__ char smem[];
  bf16* Qs =(bf16*)(smem+0);
  bf16* Ks =(bf16*)(smem+32768);
  bf16* Vps=(bf16*)(smem+65536);
  bf16* Vts=(bf16*)(smem+98304);
  bf16* Ps =(bf16*)(smem+131072);
  uint64_t* mbar=(uint64_t*)(smem+163840);
  uint32_t* tmslot=(uint32_t*)(smem+163840+16);

  int tid=threadIdx.x;
  int bh=blockIdx.y, qstart=blockIdx.x*BM;
  const bf16* Qb=Q+(long)bh*S*DH;
  const bf16* Kb=K+(long)bh*S*DH;
  const bf16* Vb=V+(long)bh*S*DH;
  bf16* Ob=O+(long)bh*S*DH;
  float* LSEb=LSE+(long)bh*S;
  const float scale=0.08838834764831845f;

  // TMEM alloc (256 cols: S in [0,128), O in [128,256))
  if(tid<32) tmem_alloc1(tmslot,256);
  if(tid<32) tmem_relinquish1();
  if(tid==0) mbar_init(mbar,1);
  asm volatile("fence.mbarrier_init.release.cluster;\n":::"memory");
  __syncthreads();
  uint32_t tbase=tmslot[0];
  uint32_t S_base=tbase, O_base=tbase+128;

  // load Q once
  for(int idx=tid; idx<128*16; idx+=128){
    int m=idx>>4, c=idx&15; int qg=qstart+m; if(qg>=S)qg=S-1; if(qg<0)qg=0;
    cp16(&Qs[(c<<10)+(m<<3)], &Qb[(long)qg*DH+8*c]);
  }

  int nkv=(S+BN-1)/BN;
  int ph=0;
  uint32_t idesc=make_idesc();
  float mold=-1e30f, lold=0.f;

  for(int blk=0; blk<nkv; blk++){
    int kvstart=blk*BN;
    // load K (canonical), V (plain)
    for(int idx=tid; idx<128*16; idx+=128){
      int key=idx>>4, c=idx&15; int kg=kvstart+key; if(kg>=S)kg=S-1; if(kg<0)kg=0;
      cp16(&Ks[(c<<10)+(key<<3)], &Kb[(long)kg*DH+8*c]);
    }
    for(int idx=tid; idx<128*16; idx+=128){
      int key=idx>>4, c=idx&15; int kg=kvstart+key; if(kg>=S)kg=S-1; if(kg<0)kg=0;
      cp16(&Vps[key*DH+8*c], &Vb[(long)kg*DH+8*c]);
    }
    cp_commit(); cp_wait0(); __syncthreads();

    // transpose Vps[key][d] -> Vts canonical (row=d,col=key)
    for(int idx=tid; idx<128*16; idx+=128){
      int d=idx>>4, b=idx&15;
      uint16_t v[8];
      #pragma unroll
      for(int e=0;e<8;e++) v[e]=((uint16_t*)Vps)[(8*b+e)*DH + d];
      *(uint4*)&Vts[(b<<10)+(d<<3)] = *(uint4*)v;
    }
    __syncthreads();
    fence_pa(); __syncthreads();

    // QK
    if(tid==0){
      #pragma unroll
      for(int j=0;j<8;j++){
        uint64_t da=make_desc(Qs+j*2048);
        uint64_t db=make_desc(Ks+j*2048);
        umma(S_base, da, db, idesc, j==0?0:1);
      }
      umma_commit(mbar);
    }
    mbar_wait(mbar,ph); ph^=1;

    // ---- softmax over S row (thread = query row = tid) ----
    // pass1: rowmax
    float mblk=-1e30f;
    #pragma unroll
    for(int b=0;b<16;b++){
      float f[8]; ld8(S_base+8*b, f);
      #pragma unroll
      for(int e=0;e<8;e++){
        int key=kvstart+8*b+e; float x=f[e]*scale; if(key>=S)x=-1e30f;
        mblk=fmaxf(mblk,x);
      }
    }
    float mnew=fmaxf(mold,mblk);
    float corr=__expf(mold-mnew);
    // pass2: exp -> Ps, sum
    float lblk=0.f;
    #pragma unroll
    for(int b=0;b<16;b++){
      float f[8]; ld8(S_base+8*b, f);
      uint16_t pv[8];
      #pragma unroll
      for(int e=0;e<8;e++){
        int key=kvstart+8*b+e; float x=f[e]*scale; if(key>=S)x=-1e30f;
        float ex=__expf(x-mnew); lblk+=ex; pv[e]=f2b(ex);
      }
      *(uint4*)&Ps[(b<<10)+(tid<<3)] = *(uint4*)pv;
    }
    float lnew=lold*corr+lblk;
    mold=mnew; lold=lnew;
    __syncthreads();
    fence_pa(); __syncthreads();

    // rescale O by corr (block>0)
    if(blk>0){
      uint32_t cu=__float_as_uint(corr);
      #pragma unroll
      for(int c=0;c<128;c+=4){
        uint32_t r0,r1,r2,r3; ld_x4(O_base+c,&r0,&r1,&r2,&r3); wait_ld();
        r0=__float_as_uint(uf(r0)*corr); r1=__float_as_uint(uf(r1)*corr);
        r2=__float_as_uint(uf(r2)*corr); r3=__float_as_uint(uf(r3)*corr);
        st_x4(O_base+c,r0,r1,r2,r3);
      }
      wait_st(); (void)cu;
      fence_before();
    }
    __syncthreads();

    // PV
    if(tid==0){
      if(blk>0) fence_after();
      #pragma unroll
      for(int j=0;j<8;j++){
        uint64_t da=make_desc(Ps+j*2048);
        uint64_t db=make_desc(Vts+j*2048);
        int en=(blk==0 && j==0)?0:1;
        umma(O_base, da, db, idesc, en);
      }
      umma_commit(mbar);
    }
    mbar_wait(mbar,ph); ph^=1;
    __syncthreads();
  }

  // epilogue
  int qg=qstart+tid;
  float invl=1.0f/lold;
  #pragma unroll
  for(int c=0;c<128;c+=4){
    uint32_t r0,r1,r2,r3; ld_x4(O_base+c,&r0,&r1,&r2,&r3); wait_ld();
    if(qg<S){
      uint16_t o[4];
      o[0]=f2b(uf(r0)*invl); o[1]=f2b(uf(r1)*invl);
      o[2]=f2b(uf(r2)*invl); o[3]=f2b(uf(r3)*invl);
      *(uint2*)&Ob[(long)qg*DH+c] = *(uint2*)o;
    }
  }
  if(qg<S) LSEb[qg]=mold+logf(lold);

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

  int smem_bytes = 163840 + 256;
  static bool cfg=false;
  if(!cfg){ CUDA_CHECK(cudaFuncSetAttribute(attn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes)); cfg=true; }

  dim3 grid((S+BM-1)/BM, B*H);
  dim3 block(128);
  cudaStream_t stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
  attn<<<grid,block,smem_bytes,stream>>>(Qd,Kd,Vd,Od,LSEd,S);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha::run);

}  // namespace mha