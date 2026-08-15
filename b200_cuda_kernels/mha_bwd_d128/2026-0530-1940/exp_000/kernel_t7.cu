#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

namespace mha_bwd {

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__); exit(1);} } while(0)
#define CU_CHECK(call) do { CUresult _e=(call); if(_e!=CUDA_SUCCESS){ \
  const char* s; cuGetErrorString(_e,&s); fprintf(stderr,"CU error %s at %s:%d\n",s,__FILE__,__LINE__); exit(1);} } while(0)

#define BR 128
#define HD 128
#define BN 64
#define QTILE 32768   // 128*128*2
#define KVTILE 16384  // 64*128*2
#define LOG2E 1.4426950408889634f

__device__ __forceinline__ void init_barrier(uint64_t* b,uint32_t c){
  asm volatile("mbarrier.init.shared.b64 [%0], %1;"::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(c));}
__device__ __forceinline__ void arrive_expect(uint64_t* b,uint32_t tx){
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
    ::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(tx):"memory");}
__device__ __forceinline__ void bar_wait(uint64_t* b,uint32_t ph){
  asm volatile("{\n.reg .pred P;\nW%=:\nmbarrier.try_wait.parity.shared.b64 P,[%0],%1;\n@!P bra W%=;\n}\n"
    ::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(ph));}
__device__ __forceinline__ void tma3d(const CUtensorMap* d,uint64_t* bar,void* smem,int c0,int c1,int c2){
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1,{%3,%4,%5}],[%2];"
    ::"r"((uint32_t)__cvta_generic_to_shared(smem)),"l"((uint64_t)d),
      "r"((uint32_t)__cvta_generic_to_shared(bar)),"r"(c0),"r"(c1),"r"(c2):"memory");}
__device__ __forceinline__ void fence_async(){ asm volatile("fence.proxy.async.shared::cta;\n":::"memory"); }

__device__ __forceinline__ void tmem_alloc(uint32_t* dst,int n){
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
    ::"r"((uint32_t)__cvta_generic_to_shared(dst)),"r"(n));}
__device__ __forceinline__ void tmem_dealloc(uint32_t a,int n){
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"::"r"(a),"r"(n));}
__device__ __forceinline__ void umma(uint32_t d,uint64_t a,uint64_t b,uint32_t id,uint32_t ac){
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p,%4,0;\n"
    "tcgen05.mma.cta_group::1.kind::f16 [%0],%1,%2,%3,p;\n}\n"
    ::"r"(d),"l"(a),"l"(b),"r"(id),"r"(ac):"memory");}
__device__ __forceinline__ void commit(uint64_t* b){
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
    ::"r"((uint32_t)__cvta_generic_to_shared(b)):"memory");}
__device__ __forceinline__ void ld_wait(){ asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory"); }
__device__ __forceinline__ void tmem_ld32(uint32_t ta,float* r){
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 "
   "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
   "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
   :"=f"(r[0]),"=f"(r[1]),"=f"(r[2]),"=f"(r[3]),"=f"(r[4]),"=f"(r[5]),"=f"(r[6]),"=f"(r[7]),
    "=f"(r[8]),"=f"(r[9]),"=f"(r[10]),"=f"(r[11]),"=f"(r[12]),"=f"(r[13]),"=f"(r[14]),"=f"(r[15]),
    "=f"(r[16]),"=f"(r[17]),"=f"(r[18]),"=f"(r[19]),"=f"(r[20]),"=f"(r[21]),"=f"(r[22]),"=f"(r[23]),
    "=f"(r[24]),"=f"(r[25]),"=f"(r[26]),"=f"(r[27]),"=f"(r[28]),"=f"(r[29]),"=f"(r[30]),"=f"(r[31])
   :"r"(ta));}
__device__ __forceinline__ float fexp2(float x){ float y; asm volatile("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x)); return y; }

__device__ __forceinline__ uint64_t mkdesc(uint32_t addr,uint32_t lbo,uint32_t sbo){
  uint64_t d=0;
  d |= (uint64_t)((addr&0x3FFFFu)>>4);
  d |= ((uint64_t)((lbo&0x3FFFFu)>>4))<<16;
  d |= ((uint64_t)((sbo&0x3FFFFu)>>4))<<32;
  d |= ((uint64_t)1)<<46;
  return d;
}
__device__ __forceinline__ uint32_t mkinstr(int M,int N,int am,int bm){
  uint32_t d=0; d|=(1u<<4); d|=(1u<<7); d|=(1u<<10);
  d|=((uint32_t)am<<15); d|=((uint32_t)bm<<16);
  d|=((uint32_t)(N>>3)<<17); d|=((uint32_t)(M>>4)<<24);
  return d;
}
__device__ __forceinline__ void run_mma(uint32_t td,
   uint32_t bA,uint32_t sA,uint32_t lA,uint32_t boA,
   uint32_t bB,uint32_t sB,uint32_t lB,uint32_t boB,
   uint32_t id,int nk,bool clear){
  for(int k=0;k<nk;k++){
    uint64_t da=mkdesc(bA+k*sA,lA,boA);
    uint64_t db=mkdesc(bB+k*sB,lB,boB);
    umma(td,da,db,id,(k==0&&clear)?0u:1u);
  }
}

__global__ void compute_D(const __nv_bfloat16* dO,const __nv_bfloat16* O,float* D,int tot){
  int gw=(blockIdx.x*blockDim.x+threadIdx.x)>>5, lane=threadIdx.x&31;
  if(gw>=tot) return;
  const __nv_bfloat16* a=dO+(size_t)gw*HD; const __nv_bfloat16* b=O+(size_t)gw*HD;
  float s=0.f;
  #pragma unroll
  for(int k=lane;k<HD;k+=32) s+=__bfloat162float(a[k])*__bfloat162float(b[k]);
  #pragma unroll
  for(int o=16;o>0;o>>=1) s+=__shfl_down_sync(0xffffffff,s,o);
  if(lane==0) D[gw]=s;
}

// ===================== dV kernel (block per kv-tile, loop over q) =====================
__global__ void __launch_bounds__(128,2) dv_kernel(
  const __grid_constant__ CUtensorMap dQ_,const __grid_constant__ CUtensorMap dK_,
  const __grid_constant__ CUtensorMap ddO_,const float* L,
  __nv_bfloat16* dV,int S,int num_q,int num_kv,float scale){
  extern __shared__ __align__(1024) char sm[];
  __nv_bfloat16* sQ=(__nv_bfloat16*)(sm+0);
  __nv_bfloat16* sdO=(__nv_bfloat16*)(sm+32768);
  __nv_bfloat16* sK=(__nv_bfloat16*)(sm+65536);
  __nv_bfloat16* sP=(__nv_bfloat16*)(sm+81920);
  float* sL=(float*)(sm+98304);
  uint64_t* tb=(uint64_t*)(sm+98816);
  uint64_t* cb=(uint64_t*)(sm+98824);
  uint32_t* tmp=(uint32_t*)(sm+98832);

  int blk=blockIdx.x, jb=blk%num_kv, bh=blk/num_kv;
  int kv_start=jb*BN, tid=threadIdx.x, warp=tid>>5;
  const float* Lb=L+(size_t)bh*S;
  float scale_l2=scale*LOG2E;

  if(warp==0) tmem_alloc(tmp,256);
  if(tid==0){ init_barrier(tb,1); init_barrier(cb,1); }
  __syncthreads();
  uint32_t TB=*tmp, TM_S=TB, TM_dV=TB+64;
  uint32_t aQ=(uint32_t)__cvta_generic_to_shared(sQ);
  uint32_t adO=(uint32_t)__cvta_generic_to_shared(sdO);
  uint32_t aK=(uint32_t)__cvta_generic_to_shared(sK);
  uint32_t aP=(uint32_t)__cvta_generic_to_shared(sP);
  uint32_t idS=mkinstr(BR,BN,0,0), idV=mkinstr(BN,HD,0,1);
  uint32_t pt=0,pc=0;

  if(tid==0){ arrive_expect(tb,KVTILE); tma3d(&dK_,tb,sK,0,bh*S+kv_start,0); }
  bar_wait(tb,pt); pt^=1; __syncthreads();

  for(int ib=0; ib<num_q; ib++){
    int q_start=ib*BR;
    if(tid==0){ arrive_expect(tb,2*QTILE);
      tma3d(&dQ_,tb,sQ,0,bh*S+q_start,0); tma3d(&ddO_,tb,sdO,0,bh*S+q_start,0); }
    { int gr=q_start+tid; sL[tid]=(gr<S)?Lb[gr]:0.f; }
    bar_wait(tb,pt); pt^=1; __syncthreads();

    if(tid==0){ run_mma(TM_S, aQ,4096,2048,128, aK,2048,1024,128, idS,8,true); commit(cb); }
    bar_wait(cb,pc); pc^=1;

    { int i=tid; float lm=sL[i]*LOG2E; int gi=q_start+i; int b=(i>>3)*512+(i&7);
      #pragma unroll
      for(int c=0;c<2;c++){ float rs[32]; tmem_ld32(TM_S+c*32,rs); ld_wait();
        #pragma unroll
        for(int j=0;j<32;j++){ int n=c*32+j; int gj=kv_start+n;
          float p=(gi<S&&gj<S)?fexp2(rs[j]*scale_l2-lm):0.f;
          sP[b+n*8]=__float2bfloat16(p); } } }
    fence_async(); __syncthreads();

    if(tid==0){ run_mma(TM_dV, aP,2048,1024,128, adO,256,128,2048, idV,8, ib==0); commit(cb); }
    bar_wait(cb,pc); pc^=1; __syncthreads();
  }

  { int row=tid; int gr=kv_start+row;
    #pragma unroll
    for(int c=0;c<4;c++){ float r[32]; tmem_ld32(TM_dV+c*32,r); ld_wait();
      if(row<BN && gr<S){ __nv_bfloat16* o=dV+((size_t)bh*S+gr)*HD+c*32;
        #pragma unroll
        for(int j=0;j<32;j++) o[j]=__float2bfloat16(r[j]); } } }
  __syncthreads();
  if(warp==0) tmem_dealloc(TB,256);
}

// ===================== dK kernel (block per kv-tile, loop over q) =====================
__global__ void __launch_bounds__(128,2) dk_kernel(
  const __grid_constant__ CUtensorMap dQ_,const __grid_constant__ CUtensorMap dK_,
  const __grid_constant__ CUtensorMap dV_,const __grid_constant__ CUtensorMap ddO_,
  const float* L,const float* Dv,__nv_bfloat16* dK,int S,int num_q,int num_kv,float scale){
  extern __shared__ __align__(1024) char sm[];
  __nv_bfloat16* sQ=(__nv_bfloat16*)(sm+0);
  __nv_bfloat16* sdO=(__nv_bfloat16*)(sm+32768);   // reused as sDS after dP
  __nv_bfloat16* sK=(__nv_bfloat16*)(sm+65536);
  __nv_bfloat16* sV=(__nv_bfloat16*)(sm+81920);
  float* sL=(float*)(sm+98304);
  float* sD=(float*)(sm+98816);
  uint64_t* tb=(uint64_t*)(sm+99328);
  uint64_t* cb=(uint64_t*)(sm+99336);
  uint32_t* tmp=(uint32_t*)(sm+99344);
  __nv_bfloat16* sDS=sdO;

  int blk=blockIdx.x, jb=blk%num_kv, bh=blk/num_kv;
  int kv_start=jb*BN, tid=threadIdx.x, warp=tid>>5;
  const float* Lb=L+(size_t)bh*S; const float* Db=Dv+(size_t)bh*S;
  float scale_l2=scale*LOG2E;

  if(warp==0) tmem_alloc(tmp,256);
  if(tid==0){ init_barrier(tb,1); init_barrier(cb,1); }
  __syncthreads();
  uint32_t TB=*tmp, TM_S=TB, TM_dP=TB+64, TM_dK=TB+128;
  uint32_t aQ=(uint32_t)__cvta_generic_to_shared(sQ);
  uint32_t adO=(uint32_t)__cvta_generic_to_shared(sdO);
  uint32_t aK=(uint32_t)__cvta_generic_to_shared(sK);
  uint32_t aV=(uint32_t)__cvta_generic_to_shared(sV);
  uint32_t aDS=(uint32_t)__cvta_generic_to_shared(sDS);
  uint32_t idS=mkinstr(BR,BN,0,0), idK=mkinstr(BN,HD,0,1);
  uint32_t pt=0,pc=0;

  if(tid==0){ arrive_expect(tb,2*KVTILE);
    tma3d(&dK_,tb,sK,0,bh*S+kv_start,0); tma3d(&dV_,tb,sV,0,bh*S+kv_start,0); }
  bar_wait(tb,pt); pt^=1; __syncthreads();

  for(int ib=0; ib<num_q; ib++){
    int q_start=ib*BR;
    if(tid==0){ arrive_expect(tb,2*QTILE);
      tma3d(&dQ_,tb,sQ,0,bh*S+q_start,0); tma3d(&ddO_,tb,sdO,0,bh*S+q_start,0); }
    { int gr=q_start+tid; sL[tid]=(gr<S)?Lb[gr]:0.f; sD[tid]=(gr<S)?Db[gr]:0.f; }
    bar_wait(tb,pt); pt^=1; __syncthreads();

    if(tid==0){
      run_mma(TM_S , aQ ,4096,2048,128, aK,2048,1024,128, idS,8,true);
      run_mma(TM_dP, adO,4096,2048,128, aV,2048,1024,128, idS,8,true);
      commit(cb);
    }
    bar_wait(cb,pc); pc^=1;

    { int i=tid; float lm=sL[i]*LOG2E, dm=sD[i]; int gi=q_start+i; int b=(i>>3)*512+(i&7);
      #pragma unroll
      for(int c=0;c<2;c++){
        float rs[32]; tmem_ld32(TM_S+c*32,rs); ld_wait();
        float rp[32]; tmem_ld32(TM_dP+c*32,rp); ld_wait();
        #pragma unroll
        for(int j=0;j<32;j++){ int n=c*32+j; int gj=kv_start+n;
          float p=(gi<S&&gj<S)?fexp2(rs[j]*scale_l2-lm):0.f;
          sDS[b+n*8]=__float2bfloat16(p*(rp[j]-dm)); } } }
    fence_async(); __syncthreads();

    if(tid==0){ run_mma(TM_dK, aDS,2048,1024,128, aQ,256,128,2048, idK,8, ib==0); commit(cb); }
    bar_wait(cb,pc); pc^=1; __syncthreads();
  }

  { int row=tid; int gr=kv_start+row;
    #pragma unroll
    for(int c=0;c<4;c++){ float r[32]; tmem_ld32(TM_dK+c*32,r); ld_wait();
      if(row<BN && gr<S){ __nv_bfloat16* o=dK+((size_t)bh*S+gr)*HD+c*32;
        #pragma unroll
        for(int j=0;j<32;j++) o[j]=__float2bfloat16(r[j]*scale); } } }
  __syncthreads();
  if(warp==0) tmem_dealloc(TB,256);
}

// ===================== dQ kernel (block per q-tile, loop over kv) =====================
__global__ void __launch_bounds__(128,2) dq_kernel(
  const __grid_constant__ CUtensorMap dQ_,const __grid_constant__ CUtensorMap dK_,
  const __grid_constant__ CUtensorMap dV_,const __grid_constant__ CUtensorMap ddO_,
  const float* L,const float* Dv,__nv_bfloat16* dQ,int S,int num_q,int num_kv,float scale){
  extern __shared__ __align__(1024) char sm[];
  __nv_bfloat16* sQ=(__nv_bfloat16*)(sm+0);
  __nv_bfloat16* sdO=(__nv_bfloat16*)(sm+32768);
  __nv_bfloat16* sK=(__nv_bfloat16*)(sm+65536);
  __nv_bfloat16* sV=(__nv_bfloat16*)(sm+81920);   // reused as sDS after dP
  float* sL=(float*)(sm+98304);
  float* sD=(float*)(sm+98816);
  uint64_t* tb=(uint64_t*)(sm+99328);
  uint64_t* cb=(uint64_t*)(sm+99336);
  uint32_t* tmp=(uint32_t*)(sm+99344);
  __nv_bfloat16* sDS=sV;

  int blk=blockIdx.x, ib=blk%num_q, bh=blk/num_q;
  int q_start=ib*BR, tid=threadIdx.x, warp=tid>>5;
  const float* Lb=L+(size_t)bh*S; const float* Db=Dv+(size_t)bh*S;
  float scale_l2=scale*LOG2E;

  if(warp==0) tmem_alloc(tmp,256);
  if(tid==0){ init_barrier(tb,1); init_barrier(cb,1); }
  __syncthreads();
  uint32_t TB=*tmp, TM_S=TB, TM_dP=TB+64, TM_dQ=TB+128;
  uint32_t aQ=(uint32_t)__cvta_generic_to_shared(sQ);
  uint32_t adO=(uint32_t)__cvta_generic_to_shared(sdO);
  uint32_t aK=(uint32_t)__cvta_generic_to_shared(sK);
  uint32_t aV=(uint32_t)__cvta_generic_to_shared(sV);
  uint32_t aDS=(uint32_t)__cvta_generic_to_shared(sDS);
  uint32_t idS=mkinstr(BR,BN,0,0), idQ=mkinstr(BR,HD,0,1);
  uint32_t pt=0,pc=0;

  if(tid==0){ arrive_expect(tb,2*QTILE);
    tma3d(&dQ_,tb,sQ,0,bh*S+q_start,0); tma3d(&ddO_,tb,sdO,0,bh*S+q_start,0); }
  { int gr=q_start+tid; sL[tid]=(gr<S)?Lb[gr]:0.f; sD[tid]=(gr<S)?Db[gr]:0.f; }
  bar_wait(tb,pt); pt^=1; __syncthreads();

  for(int jb=0; jb<num_kv; jb++){
    int kv_start=jb*BN;
    if(tid==0){ arrive_expect(tb,2*KVTILE);
      tma3d(&dK_,tb,sK,0,bh*S+kv_start,0); tma3d(&dV_,tb,sV,0,bh*S+kv_start,0); }
    bar_wait(tb,pt); pt^=1; __syncthreads();

    if(tid==0){
      run_mma(TM_S , aQ ,4096,2048,128, aK,2048,1024,128, idS,8,true);
      run_mma(TM_dP, adO,4096,2048,128, aV,2048,1024,128, idS,8,true);
      commit(cb);
    }
    bar_wait(cb,pc); pc^=1;

    { int i=tid; float lm=sL[i]*LOG2E, dm=sD[i]; int gi=q_start+i;
      #pragma unroll
      for(int c=0;c<2;c++){
        float rs[32]; tmem_ld32(TM_S+c*32,rs); ld_wait();
        float rp[32]; tmem_ld32(TM_dP+c*32,rp); ld_wait();
        #pragma unroll
        for(int j=0;j<32;j++){ int n=c*32+j; int gj=kv_start+n; int n8=n>>3,nr=n&7;
          float p=(gi<S&&gj<S)?fexp2(rs[j]*scale_l2-lm):0.f;
          sDS[n8*1024 + i*8 + nr]=__float2bfloat16(p*(rp[j]-dm)); } } }
    fence_async(); __syncthreads();

    if(tid==0){ run_mma(TM_dQ, aDS,4096,2048,128, aK,256,128,1024, idQ,4, jb==0); commit(cb); }
    bar_wait(cb,pc); pc^=1; __syncthreads();
  }

  { int row=tid; int gr=q_start+row;
    #pragma unroll
    for(int c=0;c<4;c++){ float r[32]; tmem_ld32(TM_dQ+c*32,r); ld_wait();
      if(gr<S){ __nv_bfloat16* o=dQ+((size_t)bh*S+gr)*HD+c*32;
        #pragma unroll
        for(int j=0;j<32;j++) o[j]=__float2bfloat16(r[j]*scale); } } }
  __syncthreads();
  if(warp==0) tmem_dealloc(TB,256);
}

static void mkdesc3d(CUtensorMap* d,void* p,uint64_t S_total,uint32_t rows){
  cuuint64_t gdim[3]={8,(cuuint64_t)S_total,(cuuint64_t)(HD/8)};
  cuuint64_t gstr[2]={(cuuint64_t)HD*2, 16};
  cuuint32_t bdim[3]={8,(cuuint32_t)rows,(cuuint32_t)(HD/8)};
  cuuint32_t estr[3]={1,1,1};
  CU_CHECK(cuTensorMapEncodeTiled(d,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,3,p,gdim,gstr,
    bdim,estr,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,
    CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV){
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int B=(int)Q.size(0), H=(int)Q.size(1), S=(int)Q.size(2);
  int BH=B*H;

  __nv_bfloat16* Qp=(__nv_bfloat16*)Q.data_ptr();
  __nv_bfloat16* Kp=(__nv_bfloat16*)K.data_ptr();
  __nv_bfloat16* Vp=(__nv_bfloat16*)V.data_ptr();
  __nv_bfloat16* Op=(__nv_bfloat16*)O.data_ptr();
  __nv_bfloat16* dOp=(__nv_bfloat16*)dO.data_ptr();
  const float* Lp=(const float*)L.data_ptr();
  __nv_bfloat16* dQp=(__nv_bfloat16*)dQ.data_ptr();
  __nv_bfloat16* dKp=(__nv_bfloat16*)dK.data_ptr();
  __nv_bfloat16* dVp=(__nv_bfloat16*)dV.data_ptr();

  cudaStream_t stream=(cudaStream_t)TVMFFIEnvGetStream(Q.device().device_type,Q.device().device_id);

  float* Dvec=nullptr;
  CUDA_CHECK(cudaMallocAsync(&Dvec,sizeof(float)*(size_t)BH*S,stream));

  int tot=BH*S, thr=256, wpb=thr/32, blk=(tot+wpb-1)/wpb;
  compute_D<<<blk,thr,0,stream>>>(dOp,Op,Dvec,tot);

  uint64_t St=(uint64_t)BH*S;
  CUtensorMap tQ,tK,tV,tdO;
  mkdesc3d(&tQ,Qp,St,BR); mkdesc3d(&tdO,dOp,St,BR);
  mkdesc3d(&tK,Kp,St,BN); mkdesc3d(&tV,Vp,St,BN);

  int num_q=(S+BR-1)/BR, num_kv=(S+BN-1)/BN;
  float scale=1.0f/sqrtf((float)HD);

  int SMEM_DV=98848, SMEM_DK=99360, SMEM_DQ=99360;
  CUDA_CHECK(cudaFuncSetAttribute(dv_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM_DV));
  CUDA_CHECK(cudaFuncSetAttribute(dk_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM_DK));
  CUDA_CHECK(cudaFuncSetAttribute(dq_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM_DQ));

  dv_kernel<<<BH*num_kv,128,SMEM_DV,stream>>>(tQ,tK,tdO,Lp,dVp,S,num_q,num_kv,scale);
  dk_kernel<<<BH*num_kv,128,SMEM_DK,stream>>>(tQ,tK,tV,tdO,Lp,Dvec,dKp,S,num_q,num_kv,scale);
  dq_kernel<<<BH*num_q ,128,SMEM_DQ,stream>>>(tQ,tK,tV,tdO,Lp,Dvec,dQp,S,num_q,num_kv,scale);

  CUDA_CHECK(cudaFreeAsync(Dvec,stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd