#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__); exit(1);} } while(0)
#define CU_CHECK(call) do { CUresult _e=(call); if(_e!=CUDA_SUCCESS){ const char*s; cuGetErrorString(_e,&s); \
  fprintf(stderr,"CU error %s at %s:%d\n",s,__FILE__,__LINE__); exit(1);} } while(0)

namespace gemm_ov {

constexpr int BMc=128, BNc=128, BN=256, BK=64, STAGES=4, THREADS=256, NCOLS=512, GM=8;

__device__ __forceinline__ uint32_t sa_(const void*p){return (uint32_t)__cvta_generic_to_shared(p);}
__device__ __forceinline__ int imin(int a,int b){return a<b?a:b;}
__device__ __forceinline__ uint32_t mapa0(void* p){
  uint32_t a=sa_(p), r, z=0;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;":"=r"(r):"r"(a),"r"(z));
  return r;
}
__device__ __forceinline__ void init_bar(uint64_t*b,uint32_t c){ asm volatile("mbarrier.init.shared.b64 [%0], %1;"::"r"(sa_(b)),"r"(c)); }
__device__ __forceinline__ void fence_bar_init(){ asm volatile("fence.mbarrier_init.release.cluster;\n":::"memory"); }
__device__ __forceinline__ void cluster_sync(){ asm volatile("barrier.cluster.arrive;\nbarrier.cluster.wait;\n":::"memory"); }
__device__ __forceinline__ void bar_arrive_expect(uint64_t*b,uint32_t tx){ asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"::"r"(sa_(b)),"r"(tx):"memory"); }
__device__ __forceinline__ void arrive_local(uint64_t*b){ asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"::"r"(sa_(b)):"memory"); }
__device__ __forceinline__ void arrive_addr(uint32_t a){ asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];"::"r"(a):"memory"); }
__device__ __forceinline__ void bar_wait(uint64_t*b,uint32_t ph){
  asm volatile("{\n.reg .pred P;\nW%=:\nmbarrier.try_wait.parity.shared.b64 P,[%0],%1;\n@!P bra W%=;\n}\n"::"r"(sa_(b)),"r"(ph));
}
__device__ __forceinline__ void tma_2d(const CUtensorMap*d,uint32_t bar_addr,void*sm,int c0,int c1){
  uint32_t s=sa_(sm);
  asm volatile("cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2,%3}], [%4];"
    ::"r"(s),"l"((uint64_t)d),"r"(c0),"r"(c1),"r"(bar_addr):"memory");
}
__device__ __forceinline__ void tmem_alloc(uint32_t*dst,int nc){ asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"::"r"(sa_(dst)),"r"(nc)); }
__device__ __forceinline__ void tmem_dealloc(uint32_t addr,int nc){ asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"::"r"(addr),"r"(nc)); }
__device__ __forceinline__ uint64_t mk_desc(void*p,uint32_t lbo,uint32_t sbo){
  uint32_t a=sa_(p); uint64_t d=0;
  d|=(uint64_t)((a&0x3FFFF)>>4); d|=(uint64_t)((lbo&0x3FFFF)>>4)<<16;
  d|=(uint64_t)((sbo&0x3FFFF)>>4)<<32; d|=(uint64_t)1<<46; d|=(uint64_t)2<<61; return d;
}
__device__ __forceinline__ uint32_t mk_idesc(uint32_t M,uint32_t N){
  uint32_t d=0; d|=(1u<<4); d|=(1u<<7); d|=(1u<<10); d|=((N/8)<<17); d|=((M/16)<<24); return d;
}
__device__ __forceinline__ void umma(uint32_t tc,uint64_t da,uint64_t db,uint32_t id,uint32_t acc){
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p,%4,0;\ntcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}\n"
    ::"r"(tc),"l"(da),"l"(db),"r"(id),"r"(acc));
}
__device__ __forceinline__ void umma_commit(uint64_t*b){
  asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
    ::"r"(sa_(b)),"h"((uint16_t)0x3));
}
__device__ __forceinline__ void tmem_ld8(uint32_t addr,uint32_t*r){
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
    :"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]):"r"(addr));
}
__device__ __forceinline__ void tmem_ld_wait(){ asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory"); }
__device__ __forceinline__ void tcgen05_fence_after(){ asm volatile("tcgen05.fence::after_thread_sync;":::"memory"); }
__device__ __forceinline__ void nb_sync(int id,int cnt){ asm volatile("barrier.sync.aligned %0, %1;"::"r"(id),"r"(cnt)); }

__device__ __forceinline__ void map_tile(int tile,int numM,int numN,int&m,int&n){
  int nig=GM*numN; int gid=tile/nig; int fm=gid*GM; int gsm=imin(numM-fm,GM);
  m=fm+(tile%gsm); n=(tile%nig)/gsm;
}

__global__ __launch_bounds__(THREADS) void kernel(
   const __grid_constant__ CUtensorMap tma_A, const __grid_constant__ CUtensorMap tma_B,
   __nv_bfloat16* C, int M, int N, int K){
  extern __shared__ char smem_ext[];
  uint32_t sbb=sa_(smem_ext); uint32_t adj=((sbb+1023u)&~1023u)-sbb; char* raw=smem_ext+adj;
  __nv_bfloat16* sA =(__nv_bfloat16*)raw;
  __nv_bfloat16* sB =(__nv_bfloat16*)(raw + STAGES*BMc*BK*2);
  __nv_bfloat16* sout=(__nv_bfloat16*)(raw + STAGES*BMc*BK*2 + STAGES*BNc*BK*2);
  uint64_t* full=(uint64_t*)(raw + 3*65536);
  uint64_t* empty=full+STAGES;
  uint64_t* acc_ready=empty+STAGES;
  uint64_t* acc_free=acc_ready+2;
  uint32_t* tmem_ptr=(uint32_t*)(acc_free+2);

  int tid=threadIdx.x;
  int rank=blockIdx.x&1;
  int cluster_id=blockIdx.x>>1;
  int numClusters=gridDim.x>>1;
  int numM=(M+BN-1)/BN, numN=N/BN;
  int totalTiles=numM*numN;
  int numK=K/BK;

  if(tid==0){
    #pragma unroll
    for(int s=0;s<STAGES;s++){ init_bar(&full[s],1); init_bar(&empty[s],1); }
    init_bar(&acc_ready[0],1); init_bar(&acc_ready[1],1);
    init_bar(&acc_free[0],2); init_bar(&acc_free[1],2);
    fence_bar_init();
  }
  if(tid<32){ tmem_alloc(tmem_ptr,NCOLS); }
  __syncthreads();
  cluster_sync();
  uint32_t tmem_base=tmem_ptr[0];
  uint32_t idesc=mk_idesc(BN,BN);
  const uint32_t TOTAL=(uint32_t)(2*(BMc*BK*2 + BNc*BK*2));

  if(tid<128){
    // ---- mainloop warpgroup ----
    if(tid==0 && rank==0){
      // MMA
      for(int ti=0;;ti++){
        int tile=cluster_id+ti*numClusters; if(tile>=totalTiles) break;
        int p=ti&1, uj=ti>>1;
        if(ti>=2) bar_wait(&acc_free[p], (uint32_t)((uj-1)&1));
        for(int kt=0;kt<numK;kt++){
          int g=ti*numK+kt, buf=g%STAGES;
          bar_wait(&full[buf],(uint32_t)((g/STAGES)&1));
          uint32_t acc=(kt==0)?0u:1u;
          #pragma unroll
          for(int kk=0;kk<4;kk++){
            uint64_t da=mk_desc(&sA[buf*BMc*BK+kk*16],0,1024);
            uint64_t db=mk_desc(&sB[buf*BNc*BK+kk*16],0,1024);
            umma(tmem_base+p*256, da, db, idesc, (kk==0)?acc:1u);
          }
          if(kt<numK-1) umma_commit(&empty[buf]);
          else          umma_commit(&acc_ready[p]);
        }
      }
    } else if(tid==32){
      // TMA producer (both ranks)
      for(int ti=0;;ti++){
        int tile=cluster_id+ti*numClusters; if(tile>=totalTiles) break;
        int m_tile,n_tile; map_tile(tile,numM,numN,m_tile,n_tile);
        int MbaseCTA=m_tile*BN+rank*BMc;
        int NbaseCTA=n_tile*BN+rank*BNc;
        for(int kt=0;kt<numK;kt++){
          int g=ti*numK+kt, buf=g%STAGES;
          if(g>=STAGES) bar_wait(&empty[buf],(uint32_t)(((g/STAGES)-1)&1));
          uint32_t bar_addr=(rank==0)? sa_(&full[buf]) : mapa0(&full[buf]);
          if(rank==0) bar_arrive_expect(&full[buf],TOTAL);
          tma_2d(&tma_A,bar_addr,&sA[buf*BMc*BK],kt*BK,MbaseCTA);
          tma_2d(&tma_B,bar_addr,&sB[buf*BNc*BK],kt*BK,NbaseCTA);
        }
      }
    }
  } else {
    // ---- epilogue warpgroup (both ranks) ----
    int el=tid-128;
    for(int ti=0;;ti++){
      int tile=cluster_id+ti*numClusters; if(tile>=totalTiles) break;
      int p=ti&1, uj=ti>>1;
      bar_wait(&acc_ready[p],(uint32_t)(uj&1));
      tcgen05_fence_after();
      int buf_last=(ti*numK+numK-1)%STAGES;
      if(el==0) arrive_local(&empty[buf_last]);   // free smem buffer promptly
      // drain TMEM[p] -> smem
      uint32_t tb=tmem_base+p*256;
      for(int col=0;col<BN;col+=8){
        uint32_t r[8]; tmem_ld8(tb+col,r); tmem_ld_wait();
        int b=el*BN+col;
        #pragma unroll
        for(int i=0;i<8;i++) sout[b+i]=__float2bfloat16(__uint_as_float(r[i]));
      }
      nb_sync(8,128);
      if(el==0) arrive_addr(mapa0(&acc_free[p]));  // free TMEM buffer
      // coalesced smem -> global
      int m_tile,n_tile; map_tile(tile,numM,numN,m_tile,n_tile);
      int m_base=m_tile*BN+rank*BMc;
      int n_base=n_tile*BN;
      #pragma unroll
      for(int it=0;it<32;it++){
        int lin=it*128+el; int row=lin>>5; int cg=lin&31; int col=cg*8;
        int grow=m_base+row;
        if(grow<M){
          int4 v=*reinterpret_cast<int4*>(&sout[row*BN+col]);
          *reinterpret_cast<int4*>(C+(size_t)grow*N+n_base+col)=v;
        }
      }
      nb_sync(8,128);
    }
  }

  __syncthreads();
  cluster_sync();
  if(tid<32) tmem_dealloc(tmem_base,NCOLS);
}

static CUresult make_tma(CUtensorMap* d, void* g, uint64_t inner, uint64_t outer, uint32_t bi, uint32_t bo){
  uint64_t gd[2]={inner,outer}; uint64_t gs[1]={inner*2}; uint32_t bd[2]={bi,bo}; uint32_t es[2]={1,1};
  return cuTensorMapEncodeTiled(d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, g, gd, gs, bd, es,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C){
  CUDA_CHECK(cudaSetDevice(A.device().device_id));
  int M=(int)A.size(0), K=(int)A.size(1), N=(int)B.size(0);
  __nv_bfloat16* Ap=(__nv_bfloat16*)A.data_ptr();
  __nv_bfloat16* Bp=(__nv_bfloat16*)B.data_ptr();
  __nv_bfloat16* Cp=(__nv_bfloat16*)C.data_ptr();
  CUtensorMap tA,tB;
  CU_CHECK(make_tma(&tA,Ap,K,M,BK,BMc));
  CU_CHECK(make_tma(&tB,Bp,K,N,BK,BNc));

  size_t smem = 1024 + 3*65536 + 96 + 16;
  CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));

  int smcount=148;
  cudaDeviceGetAttribute(&smcount, cudaDevAttrMultiProcessorCount, A.device().device_id);
  int numM=(M+BN-1)/BN, numN=N/BN, totalTiles=numM*numN;
  int maxClusters=smcount/2;
  int numClusters=maxClusters; if(numClusters>totalTiles) numClusters=totalTiles;
  if(numClusters<1) numClusters=1;

  dim3 grid(numClusters*2);
  cudaStream_t stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type,A.device().device_id));
  cudaLaunchConfig_t cfg={}; cfg.gridDim=grid; cfg.blockDim=dim3(THREADS); cfg.dynamicSmemBytes=smem; cfg.stream=stream;
  cudaLaunchAttribute at[1]; at[0].id=cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x=2; at[0].val.clusterDim.y=1; at[0].val.clusterDim.z=1;
  cfg.attrs=at; cfg.numAttrs=1;
  CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, tA, tB, Cp, M, N, K));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_ov::run);

}  // namespace gemm_ov