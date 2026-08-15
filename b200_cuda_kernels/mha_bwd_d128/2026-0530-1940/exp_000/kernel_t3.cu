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
#define TILE_BYTES (HD*128*2)  // 32768

// smem offsets
#define O_sQ   0
#define O_sdO  32768
#define O_sK   65536
#define O_sV   98304
#define O_sP   131072
#define O_sdS  163840
#define O_sL   196608
#define O_sD   197120
#define O_tbar 197632
#define O_cbar 197640
#define O_tmem 197648
#define SMEM_SIZE 197664

// ---------- mbarrier / TMA ----------
__device__ __forceinline__ void init_barrier(uint64_t* bar,uint32_t c){
  asm volatile("mbarrier.init.shared.b64 [%0], %1;"::"r"((uint32_t)__cvta_generic_to_shared(bar)),"r"(c));
}
__device__ __forceinline__ void arrive_expect(uint64_t* bar,uint32_t tx){
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
    ::"r"((uint32_t)__cvta_generic_to_shared(bar)),"r"(tx):"memory");
}
__device__ __forceinline__ void bar_wait(uint64_t* bar,uint32_t ph){
  asm volatile("{\n.reg .pred P;\nWAIT_%=:\nmbarrier.try_wait.parity.shared.b64 P,[%0],%1;\n@!P bra WAIT_%=;\n}\n"
    ::"r"((uint32_t)__cvta_generic_to_shared(bar)),"r"(ph));
}
__device__ __forceinline__ void tma2d(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1){
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1,{%3,%4}],[%2];"
    ::"r"((uint32_t)__cvta_generic_to_shared(smem)),"l"((uint64_t)d),
      "r"((uint32_t)__cvta_generic_to_shared(bar)),"r"(c0),"r"(c1):"memory");
}
__device__ __forceinline__ void fence_async(){ asm volatile("fence.proxy.async.shared::cta;\n":::"memory"); }

// ---------- tcgen05 ----------
__device__ __forceinline__ void tmem_alloc_cg1(uint32_t* dst,int n){
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
    ::"r"((uint32_t)__cvta_generic_to_shared(dst)),"r"(n));
}
__device__ __forceinline__ void tmem_dealloc_cg1(uint32_t addr,int n){
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"::"r"(addr),"r"(n));
}
__device__ __forceinline__ void umma_cg1(uint32_t d,uint64_t a,uint64_t b,uint32_t idesc,uint32_t accum){
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p,%4,0;\n"
    "tcgen05.mma.cta_group::1.kind::f16 [%0],%1,%2,%3,p;\n}\n"
    ::"r"(d),"l"(a),"l"(b),"r"(idesc),"r"(accum):"memory");
}
__device__ __forceinline__ void commit_cg1(uint64_t* bar){
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
    ::"r"((uint32_t)__cvta_generic_to_shared(bar)):"memory");
}
__device__ __forceinline__ void tmem_ld_wait(){ asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory"); }
__device__ __forceinline__ void tmem_ld32(uint32_t ta,float* r){
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 "
   "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
   "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
   : "=f"(r[0]),"=f"(r[1]),"=f"(r[2]),"=f"(r[3]),"=f"(r[4]),"=f"(r[5]),"=f"(r[6]),"=f"(r[7]),
     "=f"(r[8]),"=f"(r[9]),"=f"(r[10]),"=f"(r[11]),"=f"(r[12]),"=f"(r[13]),"=f"(r[14]),"=f"(r[15]),
     "=f"(r[16]),"=f"(r[17]),"=f"(r[18]),"=f"(r[19]),"=f"(r[20]),"=f"(r[21]),"=f"(r[22]),"=f"(r[23]),
     "=f"(r[24]),"=f"(r[25]),"=f"(r[26]),"=f"(r[27]),"=f"(r[28]),"=f"(r[29]),"=f"(r[30]),"=f"(r[31])
   : "r"(ta));
}

// ---------- descriptors ----------
__device__ __forceinline__ uint64_t make_desc(uint32_t saddr,uint32_t C){
  uint64_t lbo=16, sbo=(uint64_t)16*C;
  uint64_t d=0;
  d |= ((uint64_t)(saddr & 0x3FFFFu))>>4;
  d |= (((uint64_t)(lbo & 0x3FFFF))>>4)<<16;
  d |= (((uint64_t)(sbo & 0x3FFFF))>>4)<<32;
  d |= (uint64_t)1<<46;
  return d;
}
__device__ __forceinline__ uint32_t make_instr(int M,int N,int amaj,int bmaj){
  uint32_t d=0;
  d |= (1u<<4); d |= (1u<<7); d |= (1u<<10);
  d |= ((uint32_t)amaj<<15); d |= ((uint32_t)bmaj<<16);
  d |= ((uint32_t)(N>>3)<<17); d |= ((uint32_t)(M>>4)<<24);
  return d;
}
// 8 K-steps of K=16 each
__device__ __forceinline__ void mma8(uint32_t tmem_d,uint32_t sA,uint32_t sB,
    uint32_t aks,uint32_t bks,uint32_t idesc,bool clear){
  #pragma unroll
  for(int k=0;k<8;k++){
    uint64_t da=make_desc(sA + k*aks, 128);
    uint64_t db=make_desc(sB + k*bks, 128);
    umma_cg1(tmem_d, da, db, idesc, (k==0&&clear)?0u:1u);
  }
}

// ---------- D = rowsum(dO*O) ----------
__global__ void compute_D_kernel(const __nv_bfloat16* dO,const __nv_bfloat16* O,float* D,int total_rows){
  int gw=(blockIdx.x*blockDim.x+threadIdx.x)>>5;
  int lane=threadIdx.x&31;
  if(gw>=total_rows) return;
  const __nv_bfloat16* dOr=dO+(size_t)gw*HD;
  const __nv_bfloat16* Or =O +(size_t)gw*HD;
  float s=0.f;
  #pragma unroll
  for(int k=lane;k<HD;k+=32) s+=__bfloat162float(dOr[k])*__bfloat162float(Or[k]);
  #pragma unroll
  for(int o=16;o>0;o>>=1) s+=__shfl_down_sync(0xffffffff,s,o);
  if(lane==0) D[gw]=s;
}

// ---------- elementwise helpers ----------
__device__ __forceinline__ void compute_P(uint32_t TM_S,__nv_bfloat16* sP,float* sL,
    int kv_start,int q_start,int S,float scale){
  int m=threadIdx.x; int gi=q_start+m; float lm=sL[m];
  #pragma unroll
  for(int c=0;c<4;c++){
    float r[32]; tmem_ld32(TM_S + c*32, r); tmem_ld_wait();
    #pragma unroll
    for(int j=0;j<32;j++){ int n=c*32+j; int gj=kv_start+n;
      float p=(gi<S && gj<S)? __expf(scale*r[j]-lm):0.f;
      sP[m*BN+n]=__float2bfloat16(p);
    }
  }
}
__device__ __forceinline__ void compute_dS(uint32_t TM_S,__nv_bfloat16* sP,__nv_bfloat16* sdS,float* sD){
  int m=threadIdx.x; float dm=sD[m];
  #pragma unroll
  for(int c=0;c<4;c++){
    float r[32]; tmem_ld32(TM_S + c*32, r); tmem_ld_wait();
    #pragma unroll
    for(int j=0;j<32;j++){ int n=c*32+j;
      float p=__bfloat162float(sP[m*BN+n]);
      sdS[m*BN+n]=__float2bfloat16(p*(r[j]-dm));
    }
  }
}
__device__ __forceinline__ void store_acc(uint32_t TM,__nv_bfloat16* out,int bh,int row_start,int S,float scale){
  int n=threadIdx.x; int gr=row_start+n;
  #pragma unroll
  for(int c=0;c<4;c++){
    float r[32]; tmem_ld32(TM + c*32, r); tmem_ld_wait();
    if(gr<S){
      __nv_bfloat16* o=out+((size_t)bh*S+gr)*HD + c*32;
      #pragma unroll
      for(int j=0;j<32;j++) o[j]=__float2bfloat16(r[j]*scale);
    }
  }
}

// ================= dK / dV kernel =================
__global__ void __launch_bounds__(128,1) dkv_kernel(
    const __grid_constant__ CUtensorMap descQ,const __grid_constant__ CUtensorMap descK,
    const __grid_constant__ CUtensorMap descV,const __grid_constant__ CUtensorMap descdO,
    const float* L,const float* Dvec,__nv_bfloat16* dK,__nv_bfloat16* dV,
    int S,int num_q,int num_kv,float scale){
  extern __shared__ __align__(1024) char smem[];
  __nv_bfloat16* sQ =(__nv_bfloat16*)(smem+O_sQ);
  __nv_bfloat16* sdO=(__nv_bfloat16*)(smem+O_sdO);
  __nv_bfloat16* sK =(__nv_bfloat16*)(smem+O_sK);
  __nv_bfloat16* sV =(__nv_bfloat16*)(smem+O_sV);
  __nv_bfloat16* sP =(__nv_bfloat16*)(smem+O_sP);
  __nv_bfloat16* sdS=(__nv_bfloat16*)(smem+O_sdS);
  float* sL=(float*)(smem+O_sL);
  float* sD=(float*)(smem+O_sD);
  uint64_t* tbar=(uint64_t*)(smem+O_tbar);
  uint64_t* cbar=(uint64_t*)(smem+O_cbar);
  uint32_t* tmem_p=(uint32_t*)(smem+O_tmem);

  int blk=blockIdx.x; int jb=blk%num_kv, bh=blk/num_kv;
  int kv_start=jb*BN; int tid=threadIdx.x; int warp=tid>>5;
  const float* Lb=L+(size_t)bh*S; const float* Db=Dvec+(size_t)bh*S;

  if(warp==0) tmem_alloc_cg1(tmem_p,512);
  if(tid==0){ init_barrier(tbar,1); init_barrier(cbar,1); }
  __syncthreads();
  uint32_t TB=*tmem_p;
  uint32_t TM_S=TB+0, TM_dV=TB+128, TM_dK=TB+256;

  uint32_t sQa=(uint32_t)__cvta_generic_to_shared(sQ);
  uint32_t sdOa=(uint32_t)__cvta_generic_to_shared(sdO);
  uint32_t sKa=(uint32_t)__cvta_generic_to_shared(sK);
  uint32_t sVa=(uint32_t)__cvta_generic_to_shared(sV);
  uint32_t sPa=(uint32_t)__cvta_generic_to_shared(sP);
  uint32_t sdSa=(uint32_t)__cvta_generic_to_shared(sdS);

  uint32_t idS=make_instr(BR,BN,0,0);
  uint32_t idVK=make_instr(BN,HD,1,1);

  uint32_t pt=0, pc=0;
  // load K,V
  if(tid==0){ arrive_expect(tbar,2*TILE_BYTES);
    tma2d(&descK,tbar,sK,0,bh*S+kv_start); tma2d(&descV,tbar,sV,0,bh*S+kv_start); }
  bar_wait(tbar,pt); pt^=1;
  __syncthreads();

  for(int ib=0; ib<num_q; ib++){
    int q_start=ib*BR;
    if(tid==0){ arrive_expect(tbar,2*TILE_BYTES);
      tma2d(&descQ,tbar,sQ,0,bh*S+q_start); tma2d(&descdO,tbar,sdO,0,bh*S+q_start); }
    int gr=q_start+tid; sL[tid]=(gr<S)?Lb[gr]:0.f; sD[tid]=(gr<S)?Db[gr]:0.f;
    bar_wait(tbar,pt); pt^=1;
    __syncthreads();

    // S = Q@K^T
    if(tid==0){ mma8(TM_S,sQa,sKa,32,32,idS,true); commit_cg1(cbar); }
    bar_wait(cbar,pc); pc^=1;
    compute_P(TM_S,sP,sL,kv_start,q_start,S,scale);
    __syncthreads();

    // dP = dO@V^T
    if(tid==0){ mma8(TM_S,sdOa,sVa,32,32,idS,true); commit_cg1(cbar); }
    bar_wait(cbar,pc); pc^=1;
    compute_dS(TM_S,sP,sdS,sD);
    fence_async();
    __syncthreads();

    // dV += P^T@dO ; dK += dS^T@Q   (MN-major)
    if(tid==0){
      mma8(TM_dV,sPa,sdOa,4096,4096,idVK,ib==0);
      mma8(TM_dK,sdSa,sQa,4096,4096,idVK,ib==0);
      commit_cg1(cbar);
    }
    bar_wait(cbar,pc); pc^=1;
    __syncthreads();
  }

  store_acc(TM_dV,dV,bh,kv_start,S,1.0f);
  store_acc(TM_dK,dK,bh,kv_start,S,scale);
  __syncthreads();
  if(warp==0) tmem_dealloc_cg1(TB,512);
}

// ================= dQ kernel =================
__global__ void __launch_bounds__(128,1) dq_kernel(
    const __grid_constant__ CUtensorMap descQ,const __grid_constant__ CUtensorMap descK,
    const __grid_constant__ CUtensorMap descV,const __grid_constant__ CUtensorMap descdO,
    const float* L,const float* Dvec,__nv_bfloat16* dQ,
    int S,int num_q,int num_kv,float scale){
  extern __shared__ __align__(1024) char smem[];
  __nv_bfloat16* sQ =(__nv_bfloat16*)(smem+O_sQ);
  __nv_bfloat16* sdO=(__nv_bfloat16*)(smem+O_sdO);
  __nv_bfloat16* sK =(__nv_bfloat16*)(smem+O_sK);
  __nv_bfloat16* sV =(__nv_bfloat16*)(smem+O_sV);
  __nv_bfloat16* sP =(__nv_bfloat16*)(smem+O_sP);
  __nv_bfloat16* sdS=(__nv_bfloat16*)(smem+O_sdS);
  float* sL=(float*)(smem+O_sL);
  float* sD=(float*)(smem+O_sD);
  uint64_t* tbar=(uint64_t*)(smem+O_tbar);
  uint64_t* cbar=(uint64_t*)(smem+O_cbar);
  uint32_t* tmem_p=(uint32_t*)(smem+O_tmem);

  int blk=blockIdx.x; int ib=blk%num_q, bh=blk/num_q;
  int q_start=ib*BR; int tid=threadIdx.x; int warp=tid>>5;
  const float* Lb=L+(size_t)bh*S; const float* Db=Dvec+(size_t)bh*S;

  if(warp==0) tmem_alloc_cg1(tmem_p,256);
  if(tid==0){ init_barrier(tbar,1); init_barrier(cbar,1); }
  __syncthreads();
  uint32_t TB=*tmem_p;
  uint32_t TM_S=TB+0, TM_dQ=TB+128;

  uint32_t sQa=(uint32_t)__cvta_generic_to_shared(sQ);
  uint32_t sdOa=(uint32_t)__cvta_generic_to_shared(sdO);
  uint32_t sKa=(uint32_t)__cvta_generic_to_shared(sK);
  uint32_t sVa=(uint32_t)__cvta_generic_to_shared(sV);
  uint32_t sdSa=(uint32_t)__cvta_generic_to_shared(sdS);

  uint32_t idS=make_instr(BR,BN,0,0);
  uint32_t idQ=make_instr(BR,HD,0,1);

  uint32_t pt=0, pc=0;
  // load Q,dO once
  if(tid==0){ arrive_expect(tbar,2*TILE_BYTES);
    tma2d(&descQ,tbar,sQ,0,bh*S+q_start); tma2d(&descdO,tbar,sdO,0,bh*S+q_start); }
  int gr=q_start+tid; sL[tid]=(gr<S)?Lb[gr]:0.f; sD[tid]=(gr<S)?Db[gr]:0.f;
  bar_wait(tbar,pt); pt^=1;
  __syncthreads();

  for(int jb=0; jb<num_kv; jb++){
    int kv_start=jb*BN;
    if(tid==0){ arrive_expect(tbar,2*TILE_BYTES);
      tma2d(&descK,tbar,sK,0,bh*S+kv_start); tma2d(&descV,tbar,sV,0,bh*S+kv_start); }
    bar_wait(tbar,pt); pt^=1;
    __syncthreads();

    // S = Q@K^T
    if(tid==0){ mma8(TM_S,sQa,sKa,32,32,idS,true); commit_cg1(cbar); }
    bar_wait(cbar,pc); pc^=1;
    compute_P(TM_S,sP,sL,kv_start,q_start,S,scale);
    __syncthreads();

    // dP = dO@V^T
    if(tid==0){ mma8(TM_S,sdOa,sVa,32,32,idS,true); commit_cg1(cbar); }
    bar_wait(cbar,pc); pc^=1;
    compute_dS(TM_S,sP,sdS,sD);
    fence_async();
    __syncthreads();

    // dQ += dS@K  (A=dS K-major, B=K N-major)
    if(tid==0){ mma8(TM_dQ,sdSa,sKa,32,4096,idQ,jb==0); commit_cg1(cbar); }
    bar_wait(cbar,pc); pc^=1;
    __syncthreads();
  }

  store_acc(TM_dQ,dQ,bh,q_start,S,scale);
  __syncthreads();
  if(warp==0) tmem_dealloc_cg1(TB,256);
}

static void make_desc_h(CUtensorMap* d,void* p,uint64_t rows){
  cuuint64_t gdim[2]={(cuuint64_t)HD,(cuuint64_t)rows};
  cuuint64_t gstr[1]={(cuuint64_t)HD*2};
  cuuint32_t bdim[2]={(cuuint32_t)HD,(cuuint32_t)BR};
  cuuint32_t estr[2]={1,1};
  CU_CHECK(cuTensorMapEncodeTiled(d,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,p,gdim,gstr,
    bdim,estr,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,
    CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV){
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int B=(int)Q.size(0), H=(int)Q.size(1), S=(int)Q.size(2);
  int BH=B*H;

  __nv_bfloat16* Qp =(__nv_bfloat16*)Q.data_ptr();
  __nv_bfloat16* Kp =(__nv_bfloat16*)K.data_ptr();
  __nv_bfloat16* Vp =(__nv_bfloat16*)V.data_ptr();
  __nv_bfloat16* Op =(__nv_bfloat16*)O.data_ptr();
  __nv_bfloat16* dOp=(__nv_bfloat16*)dO.data_ptr();
  const float* Lp=(const float*)L.data_ptr();
  __nv_bfloat16* dQp=(__nv_bfloat16*)dQ.data_ptr();
  __nv_bfloat16* dKp=(__nv_bfloat16*)dK.data_ptr();
  __nv_bfloat16* dVp=(__nv_bfloat16*)dV.data_ptr();

  cudaStream_t stream=(cudaStream_t)TVMFFIEnvGetStream(Q.device().device_type,Q.device().device_id);

  float* Dvec=nullptr;
  CUDA_CHECK(cudaMallocAsync(&Dvec,sizeof(float)*(size_t)BH*S,stream));

  int total_rows=BH*S, threadsD=256, wpb=threadsD/32;
  int blocksD=(total_rows+wpb-1)/wpb;
  compute_D_kernel<<<blocksD,threadsD,0,stream>>>(dOp,Op,Dvec,total_rows);

  uint64_t rows=(uint64_t)BH*S;
  CUtensorMap tQ,tK,tV,tdO;
  make_desc_h(&tQ,Qp,rows); make_desc_h(&tK,Kp,rows);
  make_desc_h(&tV,Vp,rows); make_desc_h(&tdO,dOp,rows);

  int num_q=(S+BR-1)/BR, num_kv=(S+BN-1)/BN;
  float scale=1.0f/sqrtf((float)HD);

  CUDA_CHECK(cudaFuncSetAttribute(dkv_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM_SIZE));
  CUDA_CHECK(cudaFuncSetAttribute(dq_kernel ,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM_SIZE));

  dkv_kernel<<<BH*num_kv,128,SMEM_SIZE,stream>>>(tQ,tK,tV,tdO,Lp,Dvec,dKp,dVp,S,num_q,num_kv,scale);
  dq_kernel <<<BH*num_q ,128,SMEM_SIZE,stream>>>(tQ,tK,tV,tdO,Lp,Dvec,dQp,S,num_q,num_kv,scale);

  CUDA_CHECK(cudaFreeAsync(Dvec,stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd