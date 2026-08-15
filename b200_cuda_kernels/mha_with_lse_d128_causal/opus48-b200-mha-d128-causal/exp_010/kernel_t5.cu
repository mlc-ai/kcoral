#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <math.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__); exit(1);} } while(0)
#define CU_CHECK(call) do { CUresult _e=(call); if(_e!=CUDA_SUCCESS){const char*s; cuGetErrorString(_e,&s); fprintf(stderr,"CU error %s at %s:%d\n",s,__FILE__,__LINE__); exit(1);} } while(0)

namespace mha {
typedef __nv_bfloat16 bf16;

__device__ __forceinline__ uint32_t cvta(void*p){return (uint32_t)__cvta_generic_to_shared(p);}
__device__ __forceinline__ void tmem_alloc(uint32_t* dst,int ncols){
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"::"r"(cvta(dst)),"r"(ncols));}
__device__ __forceinline__ void tmem_dealloc(uint32_t addr,int ncols){
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"::"r"(addr),"r"(ncols));}
__device__ __forceinline__ void tmem_relinquish(){
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");}
__device__ __forceinline__ void umma(uint32_t tc,uint64_t da,uint64_t db,uint32_t id,uint32_t acc){
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p,%4,0;\n"
    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
    ::"r"(tc),"l"(da),"l"(db),"r"(id),"r"(acc));}
__device__ __forceinline__ void umma_commit(uint64_t* bar){
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"::"r"(cvta(bar)));}
__device__ __forceinline__ void tmem_ld8(uint32_t addr,uint32_t*r){
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7},[%8];"
   :"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]):"r"(addr));}
__device__ __forceinline__ void tmem_st8(uint32_t addr,uint32_t*r){
  asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%8],{%0,%1,%2,%3,%4,%5,%6,%7};"
   ::"r"(r[0]),"r"(r[1]),"r"(r[2]),"r"(r[3]),"r"(r[4]),"r"(r[5]),"r"(r[6]),"r"(r[7]),"r"(addr):"memory");}
__device__ __forceinline__ void tmem_wait_ld(){ asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory"); }
__device__ __forceinline__ void tmem_wait_st(){ asm volatile("tcgen05.wait::st.sync.aligned;":::"memory"); }
__device__ __forceinline__ void tc_fence_before(){ asm volatile("tcgen05.fence::before_thread_sync;":::"memory"); }
__device__ __forceinline__ void tc_fence_after(){ asm volatile("tcgen05.fence::after_thread_sync;":::"memory"); }
__device__ __forceinline__ void mbar_init(uint64_t*b,uint32_t c){
  asm volatile("mbarrier.init.shared.b64 [%0],%1;"::"r"(cvta(b)),"r"(c));}
__device__ __forceinline__ void fence_mbar_init(){ asm volatile("fence.mbarrier_init.release.cluster;":::"memory"); }
__device__ __forceinline__ void mbar_wait(uint64_t*b,uint32_t ph){
  asm volatile("{\n.reg .pred P;\nW%=:\nmbarrier.try_wait.parity.shared.b64 P,[%0],%1;\n@!P bra W%=;\n}\n"
   ::"r"(cvta(b)),"r"(ph));}
__device__ __forceinline__ void mbar_expect(uint64_t*b,uint32_t tx){
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(cvta(b)),"r"(tx):"memory");}
__device__ __forceinline__ void fence_async(){ asm volatile("fence.proxy.async.shared::cta;":::"memory"); }
__device__ __forceinline__ void tma_load(const CUtensorMap*d,uint64_t*bar,void*smem,int c0,int c1){
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3,%4}], [%2];"
    ::"r"(cvta(smem)),"l"((uint64_t)d),"r"(cvta(bar)),"r"(c0),"r"(c1):"memory");}
__device__ __forceinline__ uint64_t make_desc(void* p,uint32_t lbo,uint32_t sbo){
  uint64_t d=0; uint32_t a=cvta(p);
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

__device__ __forceinline__ void do_prefetch(const CUtensorMap*dK,const CUtensorMap*dV,
    uint64_t*bar,bf16*sK,bf16*sV,int row){
  mbar_expect(bar,65536);
  tma_load(dK,bar,sK,0,row);
  tma_load(dK,bar,sK+8192,64,row);
  tma_load(dV,bar,sV,0,row);
  tma_load(dV,bar,sV+8192,64,row);
}

__global__ __launch_bounds__(128) void kernel(
    const __grid_constant__ CUtensorMap dQ,
    const __grid_constant__ CUtensorMap dK,
    const __grid_constant__ CUtensorMap dV,
    bf16* __restrict__ O,float* __restrict__ LSE,int S){
  extern __shared__ char smem_raw[];
  uintptr_t bp = ((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023;
  bf16* sQ =(bf16*)(bp+0);
  bf16* sK0=(bf16*)(bp+32768);
  bf16* sK1=(bf16*)(bp+65536);
  bf16* sV0=(bf16*)(bp+98304);
  bf16* sV1=(bf16*)(bp+131072);
  bf16* sP =(bf16*)(bp+163840);
  uint64_t* bLoad=(uint64_t*)(bp+196608);
  uint64_t* bS   =(uint64_t*)(bp+196624);
  uint64_t* bO   =(uint64_t*)(bp+196640);
  uint64_t* bQ   =(uint64_t*)(bp+196648);
  uint32_t* tslot=(uint32_t*)(bp+196656);
  bf16* sKb[2]={sK0,sK1};
  bf16* sVb[2]={sV0,sV1};

  int bh=blockIdx.y, q_block=blockIdx.x, q0=q_block*128;
  int tid=threadIdx.x, warp=tid>>5;
  const float scale2 = 0.08838834764831843f * 1.4426950408889634f;
  const float LN2 = 0.6931471805599453f;

  if(warp==0) tmem_alloc(tslot,512);
  if(tid==0){ mbar_init(&bLoad[0],1); mbar_init(&bLoad[1],1);
              mbar_init(&bS[0],1); mbar_init(&bS[1],1);
              mbar_init(&bO[0],1); mbar_init(&bQ[0],1);
              fence_mbar_init(); }
  __syncthreads();
  uint32_t tbase=tslot[0];
  uint32_t Sa[2]={tbase, tbase+128};
  uint32_t Oa=tbase+256;

  bf16* Og=O+(size_t)bh*S*128;
  float* LSEg=LSE+(size_t)bh*S;
  int base_row = bh*S;

  uint32_t idQK=idesc(128,128,0,0);
  uint32_t idPV=idesc(128,64,0,1);

  // load Q
  if(tid==0){ mbar_expect(&bQ[0],32768);
    tma_load(&dQ,&bQ[0],sQ,0,base_row+q0);
    tma_load(&dQ,&bQ[0],sQ+8192,64,base_row+q0); }
  mbar_wait(&bQ[0],0);
  __syncthreads();

  uint32_t phLoad[2]={0,0}, phS[2]={0,0}, phO=0;

  // prologue prefetch
  if(tid==0) do_prefetch(&dK,&dV,&bLoad[0],sKb[0],sVb[0],base_row+0);
  if(q_block>=1 && tid==0) do_prefetch(&dK,&dV,&bLoad[1],sKb[1],sVb[1],base_row+128);
  mbar_wait(&bLoad[0],phLoad[0]); phLoad[0]^=1;
  if(tid==0){
    tc_fence_after();
    #pragma unroll
    for(int c=0;c<2;c++)
      #pragma unroll
      for(int k=0;k<4;k++){
        uint64_t da=make_desc(&sQ[c*8192+k*16],1,1024);
        uint64_t db=make_desc(&sKb[0][c*8192+k*16],1,1024);
        umma(Sa[0],da,db,idQK,(c==0&&k==0)?0:1);
      }
    umma_commit(&bS[0]);
  }

  float m=NEG,l=0.f;

  for(int i=0;i<=q_block;i++){
    int cur=i&1, nxt=(i+1)&1;
    int kv0=i*128; bool diag=(i==q_block);

    // issue QK(i+1) overlapping softmax(i)
    if(i+1<=q_block){
      mbar_wait(&bLoad[nxt],phLoad[nxt]); phLoad[nxt]^=1;
      if(tid==0){
        tc_fence_after();
        #pragma unroll
        for(int c=0;c<2;c++)
          #pragma unroll
          for(int k=0;k<4;k++){
            uint64_t da=make_desc(&sQ[c*8192+k*16],1,1024);
            uint64_t db=make_desc(&sKb[nxt][c*8192+k*16],1,1024);
            umma(Sa[nxt],da,db,idQK,(c==0&&k==0)?0:1);
          }
        umma_commit(&bS[nxt]);
      }
    }

    // wait QK(i)
    mbar_wait(&bS[cur],phS[cur]); phS[cur]^=1;

    // softmax(i)
    int qr=q0+tid;
    float bmax=NEG;
    for(int col=0;col<128;col+=8){
      uint32_t r[8]; tmem_ld8(Sa[cur]+col,r); tmem_wait_ld();
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
      uint32_t r[8]; tmem_ld8(Sa[cur]+col,r); tmem_wait_ld();
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

    // rescale O
    if(i>0){
      for(int col=0;col<128;col+=8){
        uint32_t r[8]; tmem_ld8(Oa+col,r); tmem_wait_ld();
        #pragma unroll
        for(int j=0;j<8;j++) r[j]=__float_as_uint(__uint_as_float(r[j])*corr);
        tmem_st8(Oa+col,r);
      }
      tmem_wait_st();
    }

    fence_async();
    tc_fence_before();
    __syncthreads();

    // PV(i)
    if(tid==0){
      tc_fence_after();
      #pragma unroll
      for(int half=0;half<2;half++)
        #pragma unroll
        for(int k=0;k<8;k++){
          uint64_t da=make_desc(&sP[(k/4)*8192+(k%4)*16],1,1024);
          uint64_t db=make_desc(&sVb[cur][half*8192+k*1024],2048,1024);
          umma(Oa+half*64,da,db,idPV,(i==0&&k==0)?0:1);
        }
      umma_commit(&bO[0]);
    }
    mbar_wait(&bO[0],phO); phO^=1;

    // prefetch KV[i+2] into stage cur
    if(i+2<=q_block && tid==0)
      do_prefetch(&dK,&dV,&bLoad[cur],sKb[cur],sVb[cur],base_row+(i+2)*128);

    m=m_new;
    __syncthreads();
  }

  // epilogue
  int qr=q0+tid;
  float invl=1.f/l;
  for(int col=0;col<128;col+=8){
    uint32_t r[8]; tmem_ld8(Oa+col,r); tmem_wait_ld();
    union{ int4 v; bf16 b[8]; } ok;
    #pragma unroll
    for(int j=0;j<8;j++) ok.b[j]=__float2bfloat16(__uint_as_float(r[j])*invl);
    if(qr<S) *(int4*)&Og[(size_t)qr*128+col]=ok.v;
  }
  if(qr<S) LSEg[qr]=m*LN2+logf(l);
  __syncthreads();
  if(warp==0){ tmem_dealloc(tbase,512); tmem_relinquish(); }
}

static CUresult make_tma(CUtensorMap* m, void* gptr, uint64_t outer){
  uint64_t gdim[2]={128, outer};
  uint64_t gstr[1]={128*2};
  uint32_t bdim[2]={64,128};
  uint32_t estr[2]={1,1};
  return cuTensorMapEncodeTiled(m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, gptr,
     gdim, gstr, bdim, estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
     CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
     CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

void run(tvm::ffi::TensorView Q,tvm::ffi::TensorView K,tvm::ffi::TensorView V,
         tvm::ffi::TensorView O,tvm::ffi::TensorView LSE){
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int B=(int)Q.size(0),H=(int)Q.size(1),S=(int)Q.size(2);
  int bh=B*H;
  uint64_t outer=(uint64_t)bh*S;
  CUtensorMap dQ,dK,dV;
  CU_CHECK(make_tma(&dQ,(void*)Q.data_ptr(),outer));
  CU_CHECK(make_tma(&dK,(void*)K.data_ptr(),outer));
  CU_CHECK(make_tma(&dV,(void*)V.data_ptr(),outer));
  bf16* Op=(bf16*)O.data_ptr();
  float* Lp=(float*)LSE.data_ptr();
  int smem=200704;
  CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
  dim3 grid((S+127)/128, bh);
  dim3 block(128);
  cudaStream_t stream=(cudaStream_t)TVMFFIEnvGetStream(Q.device().device_type,Q.device().device_id);
  kernel<<<grid,block,smem,stream>>>(dQ,dK,dV,Op,Lp,S);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha::run);
}  // namespace mha