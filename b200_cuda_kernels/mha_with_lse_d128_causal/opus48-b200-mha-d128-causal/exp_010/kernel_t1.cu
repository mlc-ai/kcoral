#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <math.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__); exit(1);} } while(0)

namespace mha {
typedef __nv_bfloat16 bf16;

__device__ __forceinline__ void tmem_alloc(uint32_t* dst,int ncols){
  uint32_t a=(uint32_t)__cvta_generic_to_shared(dst);
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"::"r"(a),"r"(ncols));
}
__device__ __forceinline__ void tmem_dealloc(uint32_t addr,int ncols){
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"::"r"(addr),"r"(ncols));
}
__device__ __forceinline__ void tmem_relinquish(){
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
}
__device__ __forceinline__ void umma(uint32_t tc,uint64_t da,uint64_t db,uint32_t id,uint32_t acc){
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p,%4,0;\n"
    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
    ::"r"(tc),"l"(da),"l"(db),"r"(id),"r"(acc));
}
__device__ __forceinline__ void umma_commit(uint64_t* bar){
  uint32_t a=(uint32_t)__cvta_generic_to_shared(bar);
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cta.b64 [%0];"::"r"(a));
}
__device__ __forceinline__ void tmem_ld8(uint32_t addr,uint32_t*r){
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7},[%8];"
   :"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]):"r"(addr));
}
__device__ __forceinline__ void tmem_st8(uint32_t addr,uint32_t*r){
  asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%8],{%0,%1,%2,%3,%4,%5,%6,%7};"
   ::"r"(r[0]),"r"(r[1]),"r"(r[2]),"r"(r[3]),"r"(r[4]),"r"(r[5]),"r"(r[6]),"r"(r[7]),"r"(addr):"memory");
}
__device__ __forceinline__ void tmem_wait_ld(){ asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory"); }
__device__ __forceinline__ void tmem_wait_st(){ asm volatile("tcgen05.wait::st.sync.aligned;":::"memory"); }
__device__ __forceinline__ void tc_fence_before(){ asm volatile("tcgen05.fence::before_thread_sync;":::"memory"); }
__device__ __forceinline__ void tc_fence_after(){ asm volatile("tcgen05.fence::after_thread_sync;":::"memory"); }
__device__ __forceinline__ void mbar_init(uint64_t*b,uint32_t c){
  asm volatile("mbarrier.init.shared.b64 [%0],%1;"::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(c));
}
__device__ __forceinline__ void fence_mbar_init(){ asm volatile("fence.mbarrier_init.release.cluster;":::"memory"); }
__device__ __forceinline__ void mbar_wait(uint64_t*b,uint32_t ph){
  asm volatile("{\n.reg .pred P;\nW%=:\nmbarrier.try_wait.parity.shared.b64 P,[%0],%1;\n@!P bra W%=;\n}\n"
   ::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(ph));
}
__device__ __forceinline__ void fence_async(){ asm volatile("fence.proxy.async.shared::cta;":::"memory"); }
__device__ __forceinline__ uint64_t make_desc(void* p,uint32_t lbo,uint32_t sbo){
  uint64_t d=0; uint32_t a=(uint32_t)__cvta_generic_to_shared(p);
  d |= (uint64_t)((a&0x3FFFF)>>4);
  d |= (uint64_t)((lbo&0x3FFFF)>>4)<<16;
  d |= (uint64_t)((sbo&0x3FFFF)>>4)<<32;
  d |= (uint64_t)1<<46;
  d |= (uint64_t)2<<61;
  return d;
}
__device__ __forceinline__ uint32_t idesc(uint32_t M,uint32_t N,uint32_t am,uint32_t bm){
  uint32_t d=0; d|=(1u<<4); d|=(1u<<7); d|=(1u<<10);
  d|=(am<<15); d|=(bm<<16); d|=((N/8)<<17); d|=((M/16)<<24);
  return d;
}

#define NEG (-1e30f)

__global__ __launch_bounds__(128) void kernel(
    const bf16* __restrict__ Q,const bf16* __restrict__ K,const bf16* __restrict__ V,
    bf16* __restrict__ O,float* __restrict__ LSE,int S){
  extern __shared__ char smem_raw[];
  uintptr_t bp = ((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023;
  bf16* sQ =(bf16*)(bp+0);
  bf16* sK =(bf16*)(bp+32768);
  bf16* sVt=(bf16*)(bp+65536);
  bf16* sP =(bf16*)(bp+98304);
  uint64_t* mbar=(uint64_t*)(bp+131072);
  uint32_t* tslot=(uint32_t*)(bp+131080);

  int bh=blockIdx.y, q_block=blockIdx.x, q0=q_block*128;
  int tid=threadIdx.x, warp=tid>>5;
  const float scale2 = 0.08838834764831843f * 1.4426950408889634f;
  const float LN2 = 0.6931471805599453f;

  if(warp==0) tmem_alloc(tslot,256);
  if(tid==0){ mbar_init(mbar,1); fence_mbar_init(); }
  __syncthreads();
  uint32_t tbase=tslot[0];
  uint32_t S_addr=tbase, O_addr=tbase+128;

  const bf16* Qg=Q+(size_t)bh*S*128;
  const bf16* Kg=K+(size_t)bh*S*128;
  const bf16* Vg=V+(size_t)bh*S*128;
  bf16* Og=O+(size_t)bh*S*128;
  float* LSEg=LSE+(size_t)bh*S;

  // load Q swizzled
  for(int i=tid;i<128*16;i+=128){
    int row=i>>4,c8=i&15,d0=c8*8;
    int chunk=d0>>6,chunk16=(row&7)^(c8&7);
    int off=chunk*8192+row*64+chunk16*8;
    int4 v; if(q0+row<S) v=*(const int4*)&Qg[(size_t)(q0+row)*128+d0]; else {v.x=v.y=v.z=v.w=0;}
    *(int4*)&sQ[off]=v;
  }
  __syncthreads();

  float m=NEG,l=0.f;
  uint32_t phase=0;
  uint32_t idQK=idesc(128,128,0,0);
  uint32_t idPV=idesc(128,128,0,0);

  for(int kvb=0;kvb<=q_block;kvb++){
    int kv0=kvb*128; bool diag=(kvb==q_block);
    // load K swizzled
    for(int i=tid;i<128*16;i+=128){
      int row=i>>4,c8=i&15,d0=c8*8;
      int chunk=d0>>6,chunk16=(row&7)^(c8&7);
      int off=chunk*8192+row*64+chunk16*8;
      int4 v; if(kv0+row<S) v=*(const int4*)&Kg[(size_t)(kv0+row)*128+d0]; else {v.x=v.y=v.z=v.w=0;}
      *(int4*)&sK[off]=v;
    }
    // load V transposed into sVt swizzled [d,kv]
    for(int i=tid;i<128*16;i+=128){
      int row=i>>4,c8=i&15,d0=c8*8;
      int4 v; if(kv0+row<S) v=*(const int4*)&Vg[(size_t)(kv0+row)*128+d0]; else {v.x=v.y=v.z=v.w=0;}
      const bf16* vb=(const bf16*)&v;
      int kv=row,chunk=kv>>6,kvloc=kv&63,klo=kvloc>>3,kb=kvloc&7;
      #pragma unroll
      for(int j=0;j<8;j++){
        int d=d0+j; int off=chunk*8192+d*64+(((d&7)^klo)*8)+kb;
        sVt[off]=vb[j];
      }
    }
    __syncthreads();
    fence_async();
    __syncthreads();

    if(tid==0){
      tc_fence_after();
      #pragma unroll
      for(int c=0;c<2;c++)
        #pragma unroll
        for(int k=0;k<4;k++){
          uint64_t da=make_desc(&sQ[c*8192+k*16],1,1024);
          uint64_t db=make_desc(&sK[c*8192+k*16],1,1024);
          umma(S_addr,da,db,idQK,(c==0&&k==0)?0:1);
        }
      umma_commit(mbar);
    }
    mbar_wait(mbar,phase); phase^=1;

    // softmax (row = tid)
    int qr=q0+tid;
    float bmax=NEG;
    for(int col=0;col<128;col+=8){
      uint32_t r[8]; tmem_ld8(S_addr+col,r); tmem_wait_ld();
      #pragma unroll
      for(int j=0;j<8;j++){
        float s=__uint_as_float(r[j])*scale2;
        int kv=kv0+col+j;
        if(diag && kv>qr) s=NEG;
        bmax=fmaxf(bmax,s);
      }
    }
    float m_new=fmaxf(m,bmax);
    float corr=exp2f(m-m_new);
    float bsum=0.f;
    for(int col=0;col<128;col+=8){
      uint32_t r[8]; tmem_ld8(S_addr+col,r); tmem_wait_ld();
      union{ int4 v; bf16 b[8]; } pk;
      #pragma unroll
      for(int j=0;j<8;j++){
        float s=__uint_as_float(r[j])*scale2;
        int kv=kv0+col+j;
        float p;
        if(diag && kv>qr) p=0.f; else { p=exp2f(s-m_new); bsum+=p; }
        pk.b[j]=__float2bfloat16(p);
      }
      int chunk=col>>6,kvloc=col&63,chunk16=(tid&7)^(kvloc>>3);
      int off=chunk*8192+tid*64+chunk16*8;
      *(int4*)&sP[off]=pk.v;
    }
    l=l*corr+bsum;

    if(kvb>0){
      for(int col=0;col<128;col+=8){
        uint32_t r[8]; tmem_ld8(O_addr+col,r); tmem_wait_ld();
        #pragma unroll
        for(int j=0;j<8;j++) r[j]=__float_as_uint(__uint_as_float(r[j])*corr);
        tmem_st8(O_addr+col,r);
      }
      tmem_wait_st();
    }

    fence_async();
    tc_fence_before();
    __syncthreads();

    if(tid==0){
      tc_fence_after();
      #pragma unroll
      for(int c=0;c<2;c++)
        #pragma unroll
        for(int k=0;k<4;k++){
          uint64_t da=make_desc(&sP[c*8192+k*16],1,1024);
          uint64_t db=make_desc(&sVt[c*8192+k*16],1,1024);
          umma(O_addr,da,db,idPV,(kvb==0&&c==0&&k==0)?0:1);
        }
      umma_commit(mbar);
    }
    mbar_wait(mbar,phase); phase^=1;
    __syncthreads();
    m=m_new;
  }

  // epilogue
  int qr=q0+tid;
  float invl=1.f/l;
  for(int col=0;col<128;col+=8){
    uint32_t r[8]; tmem_ld8(O_addr+col,r); tmem_wait_ld();
    union{ int4 v; bf16 b[8]; } ok;
    #pragma unroll
    for(int j=0;j<8;j++) ok.b[j]=__float2bfloat16(__uint_as_float(r[j])*invl);
    if(qr<S) *(int4*)&Og[(size_t)qr*128+col]=ok.v;
  }
  if(qr<S) LSEg[qr]=m*LN2+logf(l);
  __syncthreads();
  if(warp==0){ tmem_dealloc(tbase,256); tmem_relinquish(); }
}

void run(tvm::ffi::TensorView Q,tvm::ffi::TensorView K,tvm::ffi::TensorView V,
         tvm::ffi::TensorView O,tvm::ffi::TensorView LSE){
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int B=(int)Q.size(0),H=(int)Q.size(1),S=(int)Q.size(2);
  int bh=B*H;
  const bf16* Qp=(const bf16*)Q.data_ptr();
  const bf16* Kp=(const bf16*)K.data_ptr();
  const bf16* Vp=(const bf16*)V.data_ptr();
  bf16* Op=(bf16*)O.data_ptr();
  float* Lp=(float*)LSE.data_ptr();
  int smem=133120;
  static bool once=false;
  CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
  (void)once;
  dim3 grid((S+127)/128, bh);
  dim3 block(128);
  cudaStream_t stream=(cudaStream_t)TVMFFIEnvGetStream(Q.device().device_type,Q.device().device_id);
  kernel<<<grid,block,smem,stream>>>(Qp,Kp,Vp,Op,Lp,S);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha::run);
}  // namespace mha