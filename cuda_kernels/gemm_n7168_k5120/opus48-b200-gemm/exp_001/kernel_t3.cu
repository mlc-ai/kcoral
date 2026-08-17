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

namespace gemm_cg2 {

constexpr int BMc=128, BNc=128, BN=256, BK=64, STAGES=4, THREADS=128, NCOLS=256;

__device__ __forceinline__ uint32_t sa_(const void*p){return (uint32_t)__cvta_generic_to_shared(p);}
__device__ __forceinline__ uint32_t mapa0(void* p){
  uint32_t a=sa_(p), r, z=0;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;":"=r"(r):"r"(a),"r"(z));
  return r;
}
__device__ __forceinline__ void init_bar(uint64_t*b,uint32_t c){ asm volatile("mbarrier.init.shared.b64 [%0], %1;"::"r"(sa_(b)),"r"(c)); }
__device__ __forceinline__ void fence_bar_init(){ asm volatile("fence.mbarrier_init.release.cluster;\n":::"memory"); }
__device__ __forceinline__ void cluster_sync(){ asm volatile("barrier.cluster.arrive;\nbarrier.cluster.wait;\n":::"memory"); }
__device__ __forceinline__ void bar_arrive_expect(uint64_t*b,uint32_t tx){ asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"::"r"(sa_(b)),"r"(tx):"memory"); }
__device__ __forceinline__ void bar_wait(uint64_t*b,uint32_t ph){
  asm volatile("{\n.reg .pred P;\nW%=:\nmbarrier.try_wait.parity.shared.b64 P,[%0],%1;\n@!P bra W%=;\n}\n"::"r"(sa_(b)),"r"(ph));
}
__device__ __forceinline__ void tma_2d_cg2_addr(const CUtensorMap*d,uint32_t bar_addr,void*sm,int c0,int c1){
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
__device__ __forceinline__ void umma_cg2(uint32_t tc,uint64_t da,uint64_t db,uint32_t id,uint32_t acc){
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

__global__ __launch_bounds__(THREADS) void kernel(
   const __grid_constant__ CUtensorMap tma_A, const __grid_constant__ CUtensorMap tma_B,
   __nv_bfloat16* C, int M, int N, int K){
  extern __shared__ char smem_ext[];
  uint32_t sb=sa_(smem_ext); uint32_t adj=((sb+1023u)&~1023u)-sb; char* raw=smem_ext+adj;
  __nv_bfloat16* sA=(__nv_bfloat16*)raw;
  __nv_bfloat16* sB=sA+STAGES*BMc*BK;
  uint64_t* full=(uint64_t*)(sB+STAGES*BNc*BK);
  uint64_t* empty=full+STAGES;
  uint64_t* mma_done=empty+STAGES;
  uint32_t* tmem_ptr=(uint32_t*)(mma_done+1);

  int tid=threadIdx.x, warp=tid>>5;
  int rank=blockIdx.x&1, n_tile=blockIdx.x>>1, m_tile=blockIdx.y;
  int MbaseCTA=m_tile*(2*BMc)+rank*BMc;
  int Nbase=n_tile*BN;
  int NbaseCTA=Nbase+rank*BNc;
  int numK=K/BK;

  if(tid==0){
    #pragma unroll
    for(int s=0;s<STAGES;s++){ init_bar(&full[s],1); init_bar(&empty[s],1); }
    init_bar(mma_done,1); fence_bar_init();
  }
  if(warp==0){ tmem_alloc(tmem_ptr,NCOLS); }
  __syncthreads();
  cluster_sync();
  uint32_t tmem_base=tmem_ptr[0];

  const uint32_t TOTAL=(uint32_t)(2*(BMc*BK*2 + BNc*BK*2)); // all 4 transfers -> rank0's barrier
  uint32_t idesc=mk_idesc(2*BMc, BN);

  if(rank==0 && tid==0){
    for(int kt=0;kt<numK;kt++){
      int buf=kt%STAGES; int ph=(kt/STAGES)&1;
      bar_wait(&full[buf],ph);
      #pragma unroll
      for(int kk=0;kk<4;kk++){
        uint64_t da=mk_desc(&sA[buf*BMc*BK+kk*16],0,1024);
        uint64_t db=mk_desc(&sB[buf*BNc*BK+kk*16],0,1024);
        uint32_t acc=(kt==0&&kk==0)?0u:1u;
        umma_cg2(tmem_base,da,db,idesc,acc);
      }
      if(kt<numK-1) umma_commit(&empty[buf]); else umma_commit(mma_done);
    }
  } else if(tid==32){
    for(int kt=0;kt<numK;kt++){
      int buf=kt%STAGES;
      if(kt>=STAGES){ int ph=((kt/STAGES)-1)&1; bar_wait(&empty[buf],ph); }
      uint32_t bar_addr = (rank==0)? sa_(&full[buf]) : mapa0(&full[buf]);
      if(rank==0) bar_arrive_expect(&full[buf],TOTAL);
      tma_2d_cg2_addr(&tma_A,bar_addr,&sA[buf*BMc*BK],kt*BK,MbaseCTA);
      tma_2d_cg2_addr(&tma_B,bar_addr,&sB[buf*BNc*BK],kt*BK,NbaseCTA);
    }
  }

  bar_wait(mma_done,0);
  __syncthreads();

  __nv_bfloat16* sout=(__nv_bfloat16*)raw;
  #pragma unroll
  for(int col=0;col<BN;col+=8){
    uint32_t r[8]; tmem_ld8(tmem_base+col,r); tmem_ld_wait();
    int base=tid*BN+col;
    #pragma unroll
    for(int i=0;i<8;i++) sout[base+i]=__float2bfloat16(__uint_as_float(r[i]));
  }
  __syncthreads();
  cluster_sync();
  if(warp==0){ tmem_dealloc(tmem_base,NCOLS); }

  int warp_id=tid>>5, lane=tid&31;
  int GrowBase=m_tile*(2*BMc)+rank*BMc;
  #pragma unroll
  for(int step=0;step<BMc/4;step++){
    int row=step*4+warp_id; int grow=GrowBase+row;
    int cstart=lane*8; int gcol=Nbase+cstart;
    if(grow<M){
      int4 v=*reinterpret_cast<int4*>(&sout[row*BN+cstart]);
      *reinterpret_cast<int4*>(C+(size_t)grow*N+gcol)=v;
    }
  }
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
  size_t smem=1024 + (size_t)STAGES*(BMc*BK+BNc*BK)*2 + (size_t)(2*STAGES+1)*8 + 64;
  CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));
  dim3 grid(2*(N/BN), (M+2*BMc-1)/(2*BMc));
  cudaStream_t stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type,A.device().device_id));
  cudaLaunchConfig_t cfg={}; cfg.gridDim=grid; cfg.blockDim=dim3(THREADS); cfg.dynamicSmemBytes=smem; cfg.stream=stream;
  cudaLaunchAttribute at[1]; at[0].id=cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x=2; at[0].val.clusterDim.y=1; at[0].val.clusterDim.z=1;
  cfg.attrs=at; cfg.numAttrs=1;
  CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, tA, tB, Cp, M, N, K));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_cg2::run);

}  // namespace gemm_cg2