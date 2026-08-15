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
constexpr int BN=128, D=128, SPAN=16384;

__device__ __forceinline__ void init_bar(uint64_t* b, uint32_t c){
  asm volatile("mbarrier.init.shared.b64 [%0], %1;"::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(c));}
__device__ __forceinline__ void fence_bar_init(){ asm volatile("fence.mbarrier_init.release.cluster;":::"memory"); }
__device__ __forceinline__ void arrive_expect(uint64_t* b, uint32_t tx){
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
    ::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(tx):"memory");}
__device__ __forceinline__ void bar_wait(uint64_t* b, uint32_t ph){
  asm volatile("{\n.reg .pred P;\nWAIT_%=:\nmbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n@!P bra WAIT_%=;\n}\n"
    ::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(ph));}
__device__ __forceinline__ void tma_load(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1){
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
    ::"r"((uint32_t)__cvta_generic_to_shared(smem)),"l"((uint64_t)d),
      "r"((uint32_t)__cvta_generic_to_shared(bar)),"r"(c0),"r"(c1):"memory");}
__device__ __forceinline__ void fence_async(){ asm volatile("fence.proxy.async;":::"memory"); }
__device__ __forceinline__ void tmem_alloc1(uint32_t* dst, int nc){
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
    ::"r"((uint32_t)__cvta_generic_to_shared(dst)),"r"(nc));}
__device__ __forceinline__ void tmem_dealloc1(uint32_t a, int nc){
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"::"r"(a),"r"(nc));}
__device__ __forceinline__ void umma1(uint32_t d, uint64_t a, uint64_t b, uint32_t id, uint32_t acc){
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p,%4,0;\n"
    "tcgen05.mma.cta_group::1.kind::f16 [%0],%1,%2,%3,p;\n}\n"::"r"(d),"l"(a),"l"(b),"r"(id),"r"(acc));}
__device__ __forceinline__ void umma_commit1(uint64_t* bar){
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
    ::"r"((uint32_t)__cvta_generic_to_shared(bar)));}
__device__ __forceinline__ void ld8(uint32_t t, uint32_t* r){
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
   :"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]):"r"(t));}
__device__ __forceinline__ void st8(uint32_t t, const uint32_t* r){
  asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
   ::"r"(t),"r"(r[0]),"r"(r[1]),"r"(r[2]),"r"(r[3]),"r"(r[4]),"r"(r[5]),"r"(r[6]),"r"(r[7]));}
__device__ __forceinline__ void wait_ld(){ asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory"); }
__device__ __forceinline__ void wait_st(){ asm volatile("tcgen05.wait::st.sync.aligned;":::"memory"); }
__device__ __forceinline__ void fence_after(){ asm volatile("tcgen05.fence::after_thread_sync;":::"memory"); }
__device__ __forceinline__ void fence_before(){ asm volatile("tcgen05.fence::before_thread_sync;":::"memory"); }
__device__ __forceinline__ void nbar(int id){ asm volatile("barrier.sync.aligned %0, %1;"::"r"(id),"r"(128)); }

__device__ __forceinline__ uint64_t make_desc(void* p, uint32_t lbo, uint32_t sbo, int swz){
  uint64_t d=0; uint32_t a=(uint32_t)__cvta_generic_to_shared(p);
  d |= (uint64_t)((a&0x3FFFF)>>4);
  d |= ((uint64_t)((lbo&0x3FFFF)>>4))<<16;
  d |= ((uint64_t)((sbo&0x3FFFF)>>4))<<32;
  d |= (uint64_t)1<<46; d |= (uint64_t)swz<<61; return d;}
__device__ __forceinline__ uint32_t make_idesc(uint32_t M,uint32_t N,uint32_t amaj,uint32_t bmaj){
  uint32_t d=0; d|=(1u<<4); d|=(1u<<7); d|=(1u<<10);
  d|=(amaj<<15); d|=(bmaj<<16); d|=((N/8)<<17); d|=((M/16)<<24); return d;}

__global__ __launch_bounds__(256,1) void attn(
    const __grid_constant__ CUtensorMap tmaQ,
    const __grid_constant__ CUtensorMap tmaK,
    const __grid_constant__ CUtensorMap tmaV,
    __nv_bfloat16* __restrict__ O, float* __restrict__ LSE,
    int B,int H,int S,float scale)
{
  int q0 = blockIdx.x*256;
  int bh = blockIdx.y;
  int tid = threadIdx.x;
  int wg = tid>>7;
  int lw = tid&127;
  bool leader = (lw==0);

  extern __shared__ __align__(1024) char smem[];
  uint8_t* base=(uint8_t*)smem;
  uint8_t* sQ = base + (wg? 32768:0);
  uint8_t* sK = base+65536;
  uint8_t* sV = base+98304;
  uint8_t* sP = base + (wg? 163840:131072);
  uint64_t* bar_q =(uint64_t*)(base+196608);
  uint64_t* bar_kv=(uint64_t*)(base+196616);
  uint64_t* qkbar =(uint64_t*)(base+196624);
  uint64_t* pvbar =(uint64_t*)(base+196640);
  uint32_t* tmem_smem=(uint32_t*)(base+196656);

  if(tid==0){ init_bar(bar_q,1); init_bar(bar_kv,1); init_bar(qkbar,1); init_bar(qkbar+1,1);
              init_bar(pvbar,1); init_bar(pvbar+1,1); }
  fence_bar_init();
  if(tid<32) tmem_alloc1(tmem_smem,512);
  __syncthreads();
  uint32_t tbase=tmem_smem[0];
  uint32_t Obase = tbase + wg*128;
  uint32_t Sbase = tbase + 256 + wg*128;

  // prologue: load Q for both tiles
  if(tid==0){
    arrive_expect(bar_q, 4*SPAN);
    int ra=bh*S+q0, rb=bh*S+q0+128;
    tma_load(&tmaQ,bar_q,base,       0, ra); tma_load(&tmaQ,bar_q,base+SPAN,      64, ra);
    tma_load(&tmaQ,bar_q,base+32768, 0, rb); tma_load(&tmaQ,bar_q,base+32768+SPAN,64, rb);
  }
  bar_wait(bar_q,0);

  float m_=-INFINITY, l_=0.f;
  int nkv=(S+BN-1)/BN;
  uint32_t kvph=0, qkph=0, pvph=0;

  for(int t=0;t<nkv;t++){
    int kv0=t*BN, kr=bh*S+kv0;
    int kvalid=min(BN, S-kv0);
    if(tid==0){
      arrive_expect(bar_kv, 4*SPAN);
      tma_load(&tmaK,bar_kv,sK,0,kr); tma_load(&tmaK,bar_kv,sK+SPAN,64,kr);
      tma_load(&tmaV,bar_kv,sV,0,kr); tma_load(&tmaV,bar_kv,sV+SPAN,64,kr);
    }
    bar_wait(bar_kv,kvph); kvph^=1;

    // QK -> S
    if(leader){
      fence_after();
      uint32_t id=make_idesc(128,128,0,0); int step=0;
      #pragma unroll
      for(int sp=0;sp<2;sp++)
        #pragma unroll
        for(int j=0;j<4;j++){
          uint64_t da=make_desc(sQ+sp*SPAN+j*32,0,1024,2);
          uint64_t db=make_desc(sK+sp*SPAN+j*32,0,1024,2);
          umma1(Sbase, da, db, id, step==0?0:1); step++;
        }
      umma_commit1(qkbar+wg);
    }
    bar_wait(qkbar+wg, qkph);
    fence_after();

    // softmax (each thread owns full row)
    uint32_t sr[128];
    #pragma unroll
    for(int c=0;c<128;c+=8) ld8(Sbase+c, sr+c);
    wait_ld();
    float rmax=-INFINITY;
    #pragma unroll
    for(int i=0;i<128;i++){ float v=__uint_as_float(sr[i])*scale; if(i<kvalid) rmax=fmaxf(rmax,v); }
    float mold=m_, mnew=fmaxf(mold,rmax), cval=__expf(mold-mnew);
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
    float rsum=0.f;
    #pragma unroll
    for(int c=0;c<128;c+=8){
      union{ int4 v; __nv_bfloat16 h[8]; } pk;
      #pragma unroll
      for(int i=0;i<8;i++){ int n=c+i; float v=__uint_as_float(sr[c+i])*scale;
        float pp=(n<kvalid)?__expf(v-mnew):0.f; rsum+=pp; pk.h[i]=__float2bfloat16(pp); }
      *(int4*)(sP + (c/8)*2048 + (lw/8)*128 + (lw%8)*16) = pk.v;
    }
    l_=l_*cval+rsum; m_=mnew;

    fence_async(); fence_before();
    nbar(1+wg);

    if(leader){
      fence_after();
      uint32_t id=make_idesc(128,64,0,1);
      #pragma unroll
      for(int h2=0;h2<2;h2++)
        #pragma unroll
        for(int j=0;j<8;j++){
          uint64_t da=make_desc(sP + j*4096, 2048,128,0);
          uint64_t db=make_desc(sV + h2*SPAN + j*2048, 16384,1024,2);
          umma1(Obase + h2*64, da, db, id, (t==0&&j==0)?0:1);
        }
      umma_commit1(pvbar+wg);
    }
    bar_wait(pvbar+wg, pvph);
    qkph^=1; pvph^=1;
    __syncthreads();
  }

  // epilogue
  fence_after();
  int gi=q0+wg*128+lw;
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
  uint64_t gd[2]={inner,outer}; uint64_t gs[1]={inner*2}; uint32_t bd[2]={bi,bo}; uint32_t es[2]={1,1};
  return cuTensorMapEncodeTiled(d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, g, gd, gs, bd, es,
    CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
    CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE){
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int B=(int)Q.size(0), H=(int)Q.size(1), S=(int)Q.size(2);
  __nv_bfloat16* q=(__nv_bfloat16*)Q.data_ptr(); __nv_bfloat16* k=(__nv_bfloat16*)K.data_ptr();
  __nv_bfloat16* v=(__nv_bfloat16*)V.data_ptr(); __nv_bfloat16* o=(__nv_bfloat16*)O.data_ptr();
  float* lse=(float*)LSE.data_ptr();
  float scale=1.0f/sqrtf((float)D);
  uint64_t rows=(uint64_t)B*H*S;
  CUtensorMap tmaQ,tmaK,tmaV;
  CU_CHECK(make_tma(&tmaQ, q, D, rows, 64, 128));
  CU_CHECK(make_tma(&tmaK, k, D, rows, 64, 128));
  CU_CHECK(make_tma(&tmaV, v, D, rows, 64, 128));
  dim3 grid((S+255)/256, B*H);
  size_t shmem=196672;
  CUDA_CHECK(cudaFuncSetAttribute(attn, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)shmem));
  cudaStream_t stream=(cudaStream_t)TVMFFIEnvGetStream(Q.device().device_type,Q.device().device_id);
  attn<<<grid, 256, shmem, stream>>>(tmaQ,tmaK,tmaV,o,lse,B,H,S,scale);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha::run);
} // namespace mha