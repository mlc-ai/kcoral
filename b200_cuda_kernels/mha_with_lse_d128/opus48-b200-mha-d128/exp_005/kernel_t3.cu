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
#define CU_CHECK(call) do { CUresult _e=(call); if(_e!=CUDA_SUCCESS){ const char* s; cuGetErrorString(_e,&s); \
  fprintf(stderr,"CU error %s at %s:%d\n",s,__FILE__,__LINE__); exit(1);} }while(0)

namespace mha {

constexpr int BM=128, BN=128, D=128, THREADS=128;
constexpr int SPAN=16384;

__device__ __forceinline__ void init_bar(uint64_t* b, uint32_t c){
  asm volatile("mbarrier.init.shared.b64 [%0], %1;"::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(c));
}
__device__ __forceinline__ void fence_bar_init(){ asm volatile("fence.mbarrier_init.release.cluster;":::"memory"); }
__device__ __forceinline__ void arrive_expect(uint64_t* b, uint32_t tx){
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
    ::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(tx):"memory");
}
__device__ __forceinline__ void bar_wait(uint64_t* b, uint32_t ph){
  asm volatile("{\n.reg .pred P;\nWAIT_%=:\nmbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n@!P bra WAIT_%=;\n}\n"
    ::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(ph));
}
__device__ __forceinline__ void tma_load(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1){
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
    ::"r"((uint32_t)__cvta_generic_to_shared(smem)),"l"((uint64_t)d),
      "r"((uint32_t)__cvta_generic_to_shared(bar)),"r"(c0),"r"(c1):"memory");
}
__device__ __forceinline__ void fence_async(){ asm volatile("fence.proxy.async;":::"memory"); }

__device__ __forceinline__ void tmem_alloc1(uint32_t* dst, int nc){
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
    ::"r"((uint32_t)__cvta_generic_to_shared(dst)),"r"(nc));
}
__device__ __forceinline__ void tmem_dealloc1(uint32_t a, int nc){
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"::"r"(a),"r"(nc));
}
__device__ __forceinline__ void umma1(uint32_t d, uint64_t a, uint64_t b, uint32_t id, uint32_t acc){
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p,%4,0;\n"
    "tcgen05.mma.cta_group::1.kind::f16 [%0],%1,%2,%3,p;\n}\n"
    ::"r"(d),"l"(a),"l"(b),"r"(id),"r"(acc));
}
__device__ __forceinline__ void umma_commit1(uint64_t* bar){
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
    ::"r"((uint32_t)__cvta_generic_to_shared(bar)));
}
__device__ __forceinline__ void ld8(uint32_t t, uint32_t* r){
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
   :"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]):"r"(t));
}
__device__ __forceinline__ void st8(uint32_t t, const uint32_t* r){
  asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
   ::"r"(t),"r"(r[0]),"r"(r[1]),"r"(r[2]),"r"(r[3]),"r"(r[4]),"r"(r[5]),"r"(r[6]),"r"(r[7]));
}
__device__ __forceinline__ void wait_ld(){ asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory"); }
__device__ __forceinline__ void wait_st(){ asm volatile("tcgen05.wait::st.sync.aligned;":::"memory"); }
__device__ __forceinline__ void fence_after(){ asm volatile("tcgen05.fence::after_thread_sync;":::"memory"); }
__device__ __forceinline__ void fence_before(){ asm volatile("tcgen05.fence::before_thread_sync;":::"memory"); }

__device__ __forceinline__ uint64_t make_desc(void* p, uint32_t lbo, uint32_t sbo, int swz){
  uint64_t d=0; uint32_t a=(uint32_t)__cvta_generic_to_shared(p);
  d |= (uint64_t)((a&0x3FFFF)>>4);
  d |= ((uint64_t)((lbo&0x3FFFF)>>4))<<16;
  d |= ((uint64_t)((sbo&0x3FFFF)>>4))<<32;
  d |= (uint64_t)1<<46;
  d |= (uint64_t)swz<<61;
  return d;
}
__device__ __forceinline__ uint32_t make_idesc(uint32_t M,uint32_t N,uint32_t amaj,uint32_t bmaj){
  uint32_t d=0;
  d|=(1u<<4); d|=(1u<<7); d|=(1u<<10);
  d|=(amaj<<15); d|=(bmaj<<16);
  d|=((N/8)<<17); d|=((M/16)<<24);
  return d;
}

__device__ __forceinline__ void issue_qk(uint8_t* sQ, uint8_t* sK, uint32_t Sdst){
  uint32_t id=make_idesc(128,128,0,0);
  int step=0;
  #pragma unroll
  for(int sp=0;sp<2;sp++)
    #pragma unroll
    for(int j=0;j<4;j++){
      uint64_t da=make_desc(sQ+sp*SPAN+j*32,0,1024,2);
      uint64_t db=make_desc(sK+sp*SPAN+j*32,0,1024,2);
      umma1(Sdst, da, db, id, step==0?0:1);
      step++;
    }
}

__global__ __launch_bounds__(THREADS) void attn(
    const __grid_constant__ CUtensorMap tmaQ,
    const __grid_constant__ CUtensorMap tmaK,
    const __grid_constant__ CUtensorMap tmaV,
    __nv_bfloat16* __restrict__ O, float* __restrict__ LSE,
    int B,int H,int S,float scale)
{
  int q0 = blockIdx.x*BM;
  int bh = blockIdx.y;
  int tid = threadIdx.x;

  extern __shared__ __align__(1024) char smem[];
  uint8_t* base=(uint8_t*)smem;
  uint8_t* sQ =base;
  uint8_t* sK0=base+32768;
  uint8_t* sK1=base+65536;
  uint8_t* sV0=base+98304;
  uint8_t* sV1=base+131072;
  uint8_t* sP =base+163840;
  uint64_t* kv_bar0=(uint64_t*)(base+196608);
  uint64_t* kv_bar1=(uint64_t*)(base+196616);
  uint64_t* qk_bar =(uint64_t*)(base+196624);
  uint64_t* pv_bar =(uint64_t*)(base+196632);
  uint64_t* bar_q  =(uint64_t*)(base+196640);
  uint32_t* tmem_smem=(uint32_t*)(base+196648);

  if(tid==0){ init_bar(kv_bar0,1); init_bar(kv_bar1,1); init_bar(qk_bar,1); init_bar(pv_bar,1); init_bar(bar_q,1); }
  fence_bar_init();
  if(tid<32) tmem_alloc1(tmem_smem,512);
  __syncthreads();
  uint32_t tbase=tmem_smem[0];
  uint32_t Obase=tbase;
  uint32_t S0addr=tbase+128;
  uint32_t S1addr=tbase+256;

  int nkv=(S+BN-1)/BN;
  int qrow=bh*S+q0;

  uint32_t qph=0, pph=0, kvph0=0, kvph1=0;

  // ---- prologue ----
  if(tid==0){
    arrive_expect(bar_q, 2*SPAN);
    tma_load(&tmaQ,bar_q,sQ,0,qrow);
    tma_load(&tmaQ,bar_q,sQ+SPAN,64,qrow);
    arrive_expect(kv_bar0, 4*SPAN);
    int kr0=bh*S;
    tma_load(&tmaK,kv_bar0,sK0,0,kr0);   tma_load(&tmaK,kv_bar0,sK0+SPAN,64,kr0);
    tma_load(&tmaV,kv_bar0,sV0,0,kr0);   tma_load(&tmaV,kv_bar0,sV0+SPAN,64,kr0);
    if(nkv>1){
      arrive_expect(kv_bar1, 4*SPAN);
      int kr1=bh*S+BN;
      tma_load(&tmaK,kv_bar1,sK1,0,kr1); tma_load(&tmaK,kv_bar1,sK1+SPAN,64,kr1);
      tma_load(&tmaV,kv_bar1,sV1,0,kr1); tma_load(&tmaV,kv_bar1,sV1+SPAN,64,kr1);
    }
    bar_wait(bar_q,0);
    bar_wait(kv_bar0,kvph0); kvph0^=1;
    fence_after();
    issue_qk(sQ,sK0,S0addr);
    umma_commit1(qk_bar);
  }
  __syncthreads();

  float m_=-INFINITY, l_=0.f;

  for(int t=0;t<nkv;t++){
    uint32_t curS = (t&1)? S1addr : S0addr;
    uint32_t nxtS = (t&1)? S0addr : S1addr;
    uint8_t* Vbuf = (t&1)? sV1 : sV0;
    int kvalid=min(BN, S-t*BN);

    bar_wait(qk_bar, qph); qph^=1;   // S[curS] ready
    fence_after();

    // overlap: issue QK[t+1] into nxtS
    if(tid==0 && t+1<nkv){
      uint8_t* Knext = ((t+1)&1)? sK1 : sK0;
      if(((t+1)&1)==0){ bar_wait(kv_bar0,kvph0); kvph0^=1; }
      else            { bar_wait(kv_bar1,kvph1); kvph1^=1; }
      issue_qk(sQ,Knext,nxtS);
      umma_commit1(qk_bar);
    }

    // ---- softmax (each thread owns one full row) ----
    uint32_t sr[128];
    #pragma unroll
    for(int c=0;c<128;c+=8) ld8(curS+c, sr+c);
    wait_ld();

    float rmax=-INFINITY;
    #pragma unroll
    for(int i=0;i<128;i++){ float v=__uint_as_float(sr[i])*scale; if(i<kvalid) rmax=fmaxf(rmax,v); }
    float mold=m_;
    float mnew=fmaxf(mold,rmax);
    float cval=__expf(mold-mnew);

    // exact conditional O rescale (only when running max grows)
    int need=__any_sync(0xffffffffu, mnew>mold);
    if(t>0 && need){
      #pragma unroll
      for(int c=0;c<128;c+=8){
        uint32_t o8[8]; ld8(Obase+c,o8); wait_ld();
        uint32_t w[8];
        #pragma unroll
        for(int i=0;i<8;i++) w[i]=__float_as_uint(__uint_as_float(o8[i])*cval);
        st8(Obase+c,w);
      }
      wait_st();
    }

    // exp -> P (bf16, smem), rowsum
    float rsum=0.f; int m=tid;
    #pragma unroll
    for(int c=0;c<128;c+=8){
      union{ int4 v; __nv_bfloat16 h[8]; } pk;
      #pragma unroll
      for(int i=0;i<8;i++){
        int n=c+i;
        float v=__uint_as_float(sr[c+i])*scale;
        float pp=(n<kvalid)?__expf(v-mnew):0.f;
        rsum+=pp;
        pk.h[i]=__float2bfloat16(pp);
      }
      *(int4*)(sP + (c/8)*2048 + (m/8)*128 + (m%8)*16) = pk.v;
    }
    l_=l_*cval+rsum; m_=mnew;

    fence_async();
    fence_before();
    __syncthreads();

    // ---- PV[t] -> O ----
    if(tid==0){
      fence_after();
      uint32_t id=make_idesc(128,64,0,1);
      #pragma unroll
      for(int h2=0;h2<2;h2++){
        #pragma unroll
        for(int j=0;j<8;j++){
          uint64_t da=make_desc(sP + j*4096, 2048,128,0);
          uint64_t db=make_desc(Vbuf + h2*SPAN + j*2048, 16384,1024,2);
          int acc=(t==0 && j==0)?0:1;
          umma1(Obase + h2*64, da, db, id, acc);
        }
      }
      umma_commit1(pv_bar);
    }
    bar_wait(pv_bar, pph); pph^=1;

    // prefetch KV for t+2
    if(tid==0 && t+2<nkv){
      int b=(t+2)&1;
      uint8_t* Kb=b?sK1:sK0; uint8_t* Vb=b?sV1:sV0;
      uint64_t* kb=b?kv_bar1:kv_bar0;
      int kr=bh*S+(t+2)*BN;
      arrive_expect(kb,4*SPAN);
      tma_load(&tmaK,kb,Kb,0,kr); tma_load(&tmaK,kb,Kb+SPAN,64,kr);
      tma_load(&tmaV,kb,Vb,0,kr); tma_load(&tmaV,kb,Vb+SPAN,64,kr);
    }
    __syncthreads();
  }

  // ---- epilogue ----
  fence_after();
  int gi=q0+tid;
  if(gi<S){
    float linv=1.0f/l_;
    #pragma unroll
    for(int c=0;c<128;c+=8){
      uint32_t r8[8]; ld8(Obase+c,r8); wait_ld();
      union{ int4 v; __nv_bfloat16 h[8]; } ob;
      #pragma unroll
      for(int i=0;i<8;i++) ob.h[i]=__float2bfloat16(__uint_as_float(r8[i])*linv);
      *(int4*)(O + ((int64_t)bh*S + gi)*D + c) = ob.v;
    }
    LSE[(int64_t)bh*S+gi]=m_+logf(l_);
  }
  __syncthreads();
  if(tid<32) tmem_dealloc1(tbase,512);
}

static CUresult make_tma(CUtensorMap* d, void* g, uint64_t inner, uint64_t outer, uint32_t bi, uint32_t bo){
  uint64_t gd[2]={inner,outer};
  uint64_t gs[1]={inner*2};
  uint32_t bd[2]={bi,bo};
  uint32_t es[2]={1,1};
  return cuTensorMapEncodeTiled(d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, g, gd, gs, bd, es,
    CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
    CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE){
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int B=(int)Q.size(0), H=(int)Q.size(1), S=(int)Q.size(2);
  __nv_bfloat16* q=(__nv_bfloat16*)Q.data_ptr();
  __nv_bfloat16* k=(__nv_bfloat16*)K.data_ptr();
  __nv_bfloat16* v=(__nv_bfloat16*)V.data_ptr();
  __nv_bfloat16* o=(__nv_bfloat16*)O.data_ptr();
  float* lse=(float*)LSE.data_ptr();
  float scale=1.0f/sqrtf((float)D);

  uint64_t rows=(uint64_t)B*H*S;
  CUtensorMap tmaQ,tmaK,tmaV;
  CU_CHECK(make_tma(&tmaQ, q, D, rows, 64, 128));
  CU_CHECK(make_tma(&tmaK, k, D, rows, 64, 128));
  CU_CHECK(make_tma(&tmaV, v, D, rows, 64, 128));

  dim3 grid((S+BM-1)/BM, B*H);
  size_t shmem=197120;
  CUDA_CHECK(cudaFuncSetAttribute(attn, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)shmem));

  cudaStream_t stream=(cudaStream_t)TVMFFIEnvGetStream(Q.device().device_type,Q.device().device_id);
  attn<<<grid, THREADS, shmem, stream>>>(tmaQ,tmaK,tmaV,o,lse,B,H,S,scale);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha::run);

} // namespace mha