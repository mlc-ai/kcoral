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
#define BN 128
#define HD 128
#define TILEB 32768   // 128*128*2

// ---- mbarrier / TMA ----
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

// ---- tcgen05 ----
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
  #pragma unroll
  for(int k=0;k<nk;k++){
    uint64_t da=mkdesc(bA+k*sA,lA,boA);
    uint64_t db=mkdesc(bB+k*sB,lB,boB);
    umma(td,da,db,id,(k==0&&clear)?0u:1u);
  }
}

// ---- D = rowsum(dO*O) ----
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

// ===================== dKV kernel =====================
#define K_sQ 0
#define K_sdO 32768
#define K_sK 65536
#define K_sV 98304
#define K_sP 131072
#define K_sdS 163840
#define K_sL 196608
#define K_sD 197120
#define K_tb 197632
#define K_cb 197640
#define K_tm 197648
#define SMEM_DKV 197664

__global__ void __launch_bounds__(128,1) dkv_kernel(
  const __grid_constant__ CUtensorMap dQ_,const __grid_constant__ CUtensorMap dK_,
  const __grid_constant__ CUtensorMap dV_,const __grid_constant__ CUtensorMap ddO_,
  const float* L,const float* Dv,__nv_bfloat16* dK,__nv_bfloat16* dV,
  int S,int num_q,int num_kv,float scale){
  extern __shared__ __align__(1024) char sm[];
  __nv_bfloat16* sQ=(__nv_bfloat16*)(sm+K_sQ);
  __nv_bfloat16* sdO=(__nv_bfloat16*)(sm+K_sdO);
  __nv_bfloat16* sK=(__nv_bfloat16*)(sm+K_sK);
  __nv_bfloat16* sV=(__nv_bfloat16*)(sm+K_sV);
  __nv_bfloat16* sP=(__nv_bfloat16*)(sm+K_sP);
  __nv_bfloat16* sdS=(__nv_bfloat16*)(sm+K_sdS);
  float* sL=(float*)(sm+K_sL); float* sD=(float*)(sm+K_sD);
  uint64_t* tb=(uint64_t*)(sm+K_tb); uint64_t* cb=(uint64_t*)(sm+K_cb);
  uint32_t* tmp=(uint32_t*)(sm+K_tm);

  int blk=blockIdx.x, jb=blk%num_kv, bh=blk/num_kv;
  int kv_start=jb*BN, tid=threadIdx.x, warp=tid>>5;
  const float* Lb=L+(size_t)bh*S; const float* Db=Dv+(size_t)bh*S;

  if(warp==0) tmem_alloc(tmp,512);
  if(tid==0){ init_barrier(tb,1); init_barrier(cb,1); }
  __syncthreads();
  uint32_t TB=*tmp, TM_S=TB, TM_dV=TB+128, TM_dK=TB+256;

  uint32_t aQ=(uint32_t)__cvta_generic_to_shared(sQ);
  uint32_t adO=(uint32_t)__cvta_generic_to_shared(sdO);
  uint32_t aK=(uint32_t)__cvta_generic_to_shared(sK);
  uint32_t aV=(uint32_t)__cvta_generic_to_shared(sV);
  uint32_t aP=(uint32_t)__cvta_generic_to_shared(sP);
  uint32_t adS=(uint32_t)__cvta_generic_to_shared(sdS);
  uint32_t idS=mkinstr(BR,BN,0,0), idVK=mkinstr(BN,HD,0,1);

  uint32_t pt=0,pc=0;
  if(tid==0){ arrive_expect(tb,2*TILEB);
    tma3d(&dK_,tb,sK,0,bh*S+kv_start,0); tma3d(&dV_,tb,sV,0,bh*S+kv_start,0); }
  bar_wait(tb,pt); pt^=1; __syncthreads();

  for(int ib=0; ib<num_q; ib++){
    int q_start=ib*BR;
    if(tid==0){ arrive_expect(tb,2*TILEB);
      tma3d(&dQ_,tb,sQ,0,bh*S+q_start,0); tma3d(&ddO_,tb,sdO,0,bh*S+q_start,0); }
    { int gr=q_start+tid; sL[tid]=(gr<S)?Lb[gr]:0.f; sD[tid]=(gr<S)?Db[gr]:0.f; }
    bar_wait(tb,pt); pt^=1; __syncthreads();

    // matmul1: S = Q@K^T
    if(tid==0){ run_mma(TM_S, aQ,4096,2048,128, aK,4096,2048,128, idS,8,true); commit(cb); }
    bar_wait(cb,pc); pc^=1;
    { int i=tid; float lm=sL[i]; int i8=i>>3, ir=i&7; int gi=q_start+i;
      #pragma unroll
      for(int c=0;c<4;c++){ float r[32]; tmem_ld32(TM_S+c*32,r); ld_wait();
        #pragma unroll
        for(int j=0;j<32;j++){ int n=c*32+j; int gj=kv_start+n;
          float p=(gi<S&&gj<S)?__expf(scale*r[j]-lm):0.f;
          sP[i8*(BN*8)+n*8+ir]=__float2bfloat16(p); } } }
    __syncthreads();

    // matmul2: dP = dO@V^T
    if(tid==0){ run_mma(TM_S, adO,4096,2048,128, aV,4096,2048,128, idS,8,true); commit(cb); }
    bar_wait(cb,pc); pc^=1;
    { int i=tid; float dm=sD[i]; int i8=i>>3, ir=i&7;
      #pragma unroll
      for(int c=0;c<4;c++){ float r[32]; tmem_ld32(TM_S+c*32,r); ld_wait();
        #pragma unroll
        for(int j=0;j<32;j++){ int n=c*32+j;
          float p=__bfloat162float(sP[i8*(BN*8)+n*8+ir]);
          sdS[i8*(BN*8)+n*8+ir]=__float2bfloat16(p*(r[j]-dm)); } } }
    fence_async(); __syncthreads();

    // matmul3: dV += P^T@dO ; matmul4: dK += dS^T@Q
    if(tid==0){
      run_mma(TM_dV, aP,4096,2048,128, adO,256,128,2048, idVK,8, ib==0);
      run_mma(TM_dK, adS,4096,2048,128, aQ,256,128,2048, idVK,8, ib==0);
      commit(cb);
    }
    bar_wait(cb,pc); pc^=1; __syncthreads();
  }

  // store dV (no scale)
  { int row=tid; int gr=kv_start+row;
    #pragma unroll
    for(int c=0;c<4;c++){ float r[32]; tmem_ld32(TM_dV+c*32,r); ld_wait();
      if(gr<S){ __nv_bfloat16* o=dV+((size_t)bh*S+gr)*HD+c*32;
        #pragma unroll
        for(int j=0;j<32;j++) o[j]=__float2bfloat16(r[j]); } } }
  __syncthreads();
  // store dK (scale)
  { int row=tid; int gr=kv_start+row;
    #pragma unroll
    for(int c=0;c<4;c++){ float r[32]; tmem_ld32(TM_dK+c*32,r); ld_wait();
      if(gr<S){ __nv_bfloat16* o=dK+((size_t)bh*S+gr)*HD+c*32;
        #pragma unroll
        for(int j=0;j<32;j++) o[j]=__float2bfloat16(r[j]*scale); } } }
  __syncthreads();
  if(warp==0) tmem_dealloc(TB,512);
}

// ===================== dQ kernel =====================
#define Q_sQ 0
#define Q_sdO 32768
#define Q_sK 65536
#define Q_sV 98304
#define Q_sP 131072
#define Q_sL 163840
#define Q_sD 164352
#define Q_tb 164864
#define Q_cb 164872
#define Q_tm 164880
#define SMEM_DQ 164896

__global__ void __launch_bounds__(128,1) dq_kernel(
  const __grid_constant__ CUtensorMap dQ_,const __grid_constant__ CUtensorMap dK_,
  const __grid_constant__ CUtensorMap dV_,const __grid_constant__ CUtensorMap ddO_,
  const float* L,const float* Dv,__nv_bfloat16* dQ,
  int S,int num_q,int num_kv,float scale){
  extern __shared__ __align__(1024) char sm[];
  __nv_bfloat16* sQ=(__nv_bfloat16*)(sm+Q_sQ);
  __nv_bfloat16* sdO=(__nv_bfloat16*)(sm+Q_sdO);
  __nv_bfloat16* sK=(__nv_bfloat16*)(sm+Q_sK);
  __nv_bfloat16* sV=(__nv_bfloat16*)(sm+Q_sV);
  __nv_bfloat16* sP=(__nv_bfloat16*)(sm+Q_sP);
  float* sL=(float*)(sm+Q_sL); float* sD=(float*)(sm+Q_sD);
  uint64_t* tb=(uint64_t*)(sm+Q_tb); uint64_t* cb=(uint64_t*)(sm+Q_cb);
  uint32_t* tmp=(uint32_t*)(sm+Q_tm);

  int blk=blockIdx.x, ib=blk%num_q, bh=blk/num_q;
  int q_start=ib*BR, tid=threadIdx.x, warp=tid>>5;
  const float* Lb=L+(size_t)bh*S; const float* Db=Dv+(size_t)bh*S;

  if(warp==0) tmem_alloc(tmp,256);
  if(tid==0){ init_barrier(tb,1); init_barrier(cb,1); }
  __syncthreads();
  uint32_t TB=*tmp, TM_S=TB, TM_dQ=TB+128;

  uint32_t aQ=(uint32_t)__cvta_generic_to_shared(sQ);
  uint32_t adO=(uint32_t)__cvta_generic_to_shared(sdO);
  uint32_t aK=(uint32_t)__cvta_generic_to_shared(sK);
  uint32_t aV=(uint32_t)__cvta_generic_to_shared(sV);
  uint32_t aP=(uint32_t)__cvta_generic_to_shared(sP);
  uint32_t idS=mkinstr(BR,BN,0,0), idQ=mkinstr(BR,HD,0,1);

  uint32_t pt=0,pc=0;
  if(tid==0){ arrive_expect(tb,2*TILEB);
    tma3d(&dQ_,tb,sQ,0,bh*S+q_start,0); tma3d(&ddO_,tb,sdO,0,bh*S+q_start,0); }
  { int gr=q_start+tid; sL[tid]=(gr<S)?Lb[gr]:0.f; sD[tid]=(gr<S)?Db[gr]:0.f; }
  bar_wait(tb,pt); pt^=1; __syncthreads();

  for(int jb=0; jb<num_kv; jb++){
    int kv_start=jb*BN;
    if(tid==0){ arrive_expect(tb,2*TILEB);
      tma3d(&dK_,tb,sK,0,bh*S+kv_start,0); tma3d(&dV_,tb,sV,0,bh*S+kv_start,0); }
    bar_wait(tb,pt); pt^=1; __syncthreads();

    // matmul1: S=Q@K^T
    if(tid==0){ run_mma(TM_S, aQ,4096,2048,128, aK,4096,2048,128, idS,8,true); commit(cb); }
    bar_wait(cb,pc); pc^=1;
    { int i=tid; float lm=sL[i]; int gi=q_start+i;
      #pragma unroll
      for(int c=0;c<4;c++){ float r[32]; tmem_ld32(TM_S+c*32,r); ld_wait();
        #pragma unroll
        for(int j=0;j<32;j++){ int n=c*32+j; int gj=kv_start+n; int n8=n>>3,nr=n&7;
          float p=(gi<S&&gj<S)?__expf(scale*r[j]-lm):0.f;
          sP[n8*(BR*8)+i*8+nr]=__float2bfloat16(p); } } }
    __syncthreads();

    // matmul2: dP=dO@V^T
    if(tid==0){ run_mma(TM_S, adO,4096,2048,128, aV,4096,2048,128, idS,8,true); commit(cb); }
    bar_wait(cb,pc); pc^=1;
    { int i=tid; float dm=sD[i];
      #pragma unroll
      for(int c=0;c<4;c++){ float r[32]; tmem_ld32(TM_S+c*32,r); ld_wait();
        #pragma unroll
        for(int j=0;j<32;j++){ int n=c*32+j; int n8=n>>3,nr=n&7;
          float p=__bfloat162float(sP[n8*(BR*8)+i*8+nr]);
          sP[n8*(BR*8)+i*8+nr]=__float2bfloat16(p*(r[j]-dm)); } } }
    fence_async(); __syncthreads();

    // matmul5: dQ += dS@K
    if(tid==0){ run_mma(TM_dQ, aP,4096,2048,128, aK,256,128,2048, idQ,8, jb==0); commit(cb); }
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

static void mkdesc3d(CUtensorMap* d,void* p,uint64_t S_total){
  cuuint64_t gdim[3]={8,(cuuint64_t)S_total,(cuuint64_t)(HD/8)};
  cuuint64_t gstr[2]={(cuuint64_t)HD*2, 16};
  cuuint32_t bdim[3]={8,(cuuint32_t)BR,(cuuint32_t)(HD/8)};
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
  mkdesc3d(&tQ,Qp,St); mkdesc3d(&tK,Kp,St); mkdesc3d(&tV,Vp,St); mkdesc3d(&tdO,dOp,St);

  int num_q=(S+BR-1)/BR, num_kv=(S+BN-1)/BN;
  float scale=1.0f/sqrtf((float)HD);

  CUDA_CHECK(cudaFuncSetAttribute(dkv_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM_DKV));
  CUDA_CHECK(cudaFuncSetAttribute(dq_kernel ,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM_DQ));

  dkv_kernel<<<BH*num_kv,128,SMEM_DKV,stream>>>(tQ,tK,tV,tdO,Lp,Dvec,dKp,dVp,S,num_q,num_kv,scale);
  dq_kernel <<<BH*num_q ,128,SMEM_DQ ,stream>>>(tQ,tK,tV,tdO,Lp,Dvec,dQp,S,num_q,num_kv,scale);

  CUDA_CHECK(cudaFreeAsync(Dvec,stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd