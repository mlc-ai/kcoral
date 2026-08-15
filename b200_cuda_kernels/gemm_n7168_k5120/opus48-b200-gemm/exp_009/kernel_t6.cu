#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s at %s:%d\n", cudaGetErrorString(_e),__FILE__,__LINE__); exit(1);} } while(0)
#define CU_CHECK(call) do { CUresult _e=(call); if(_e!=CUDA_SUCCESS){ const char* s; cuGetErrorString(_e,&s); \
  fprintf(stderr,"CU error %s at %s:%d\n", s,__FILE__,__LINE__); exit(1);} } while(0)

namespace gemm_kernel {

constexpr int BM_CTA=128, BN_CTA=128;
constexpr int MTOT=256, NTOT=256;
constexpr int KCH=64, NS=6;
constexpr int KITER=KCH/16;
constexpr int THREADS=128;
constexpr int GROUP_M=8;

__device__ __forceinline__ uint32_t cvta(const void* p){ return (uint32_t)__cvta_generic_to_shared(p); }

__device__ __forceinline__ bool elect(){
  uint32_t pred;
  asm volatile("{\n.reg .pred p;\nelect.sync _|p, 0xFFFFFFFF;\nselp.b32 %0,1,0,p;\n}\n":"=r"(pred));
  return pred!=0;
}
__device__ __forceinline__ uint32_t cluster_rank(){
  uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;":"=r"(r)); return r;
}
__device__ __forceinline__ void cluster_sync(){
  asm volatile("barrier.cluster.arrive;\nbarrier.cluster.wait;\n":::"memory");
}
__device__ __forceinline__ void init_bar(uint64_t* bar,uint32_t cnt){
  asm volatile("mbarrier.init.shared.b64 [%0],%1;"::"r"(cvta(bar)),"r"(cnt));
}
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* bar,uint32_t tx){
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"::"r"(cvta(bar)),"r"(tx):"memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar,uint32_t phase){
  asm volatile("{\n.reg .pred P;\nWT_%=:\nmbarrier.try_wait.parity.shared.b64 P,[%0],%1;\n@!P bra WT_%=;\n}\n"
    ::"r"(cvta(bar)),"r"(phase));
}
__device__ __forceinline__ void tma_load_cg2(const CUtensorMap* d,uint64_t* bar,void* smem,int32_t c0,int32_t c1){
  uint32_t sa=cvta(smem);
  uint32_t ba=cvta(&bar[0]) & 0xFEFFFFFF;
  asm volatile(
    "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
    " [%0], [%1, {%2, %3}], [%4];"
    :: "r"(sa),"l"((uint64_t)d),"r"(c0),"r"(c1),"r"(ba):"memory");
}
__device__ __forceinline__ void tmem_alloc(uint32_t* dst,int ncols){
  asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"::"r"(cvta(dst)),"r"(ncols));
}
__device__ __forceinline__ void tmem_dealloc(uint32_t addr,int ncols){
  asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"::"r"(addr),"r"(ncols));
}
__device__ __forceinline__ void umma(uint32_t tmem_c,uint64_t da,uint64_t db,uint32_t idesc,uint32_t accum){
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p,%4,0;\n"
    "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}\n"
    ::"r"(tmem_c),"l"(da),"l"(db),"r"(idesc),"r"(accum));
}
__device__ __forceinline__ void umma_commit_2sm(uint64_t* bar){
  uint32_t a=cvta(&bar[0]);
  asm volatile(
    "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
    :: "r"(a),"h"((uint16_t)0x3):"memory");
}
__device__ __forceinline__ uint64_t smem_desc(void* p){
  uint64_t d=0; uint32_t a=cvta(p);
  d |= (uint64_t)((a&0x3FFFF)>>4);
  d |= (uint64_t)0 << 16;
  d |= (uint64_t)((1024u&0x3FFFF)>>4) << 32;
  d |= (uint64_t)1 << 46;
  d |= (uint64_t)2 << 61;
  return d;
}
__device__ __forceinline__ uint32_t make_idesc(uint32_t M,uint32_t N){
  uint32_t d=0;
  d |= (1u<<4); d |= (1u<<7); d |= (1u<<10);
  d |= (0u<<15); d |= (0u<<16);
  d |= ((N/8)<<17); d |= ((M/16)<<24);
  return d;
}

__global__ void __launch_bounds__(THREADS)
kernel(const __grid_constant__ CUtensorMap dA, const __grid_constant__ CUtensorMap dB,
       __nv_bfloat16* __restrict__ C, int M,int N,int K){
  extern __shared__ char smem_raw[];
  uint64_t* full_bar=(uint64_t*)smem_raw;
  uint64_t* empty_bar=full_bar+NS;
  uint64_t* mma_done=empty_bar+NS;
  uint32_t* tmem_ptr=(uint32_t*)(mma_done+1);
  uintptr_t tb=(uintptr_t)(tmem_ptr+1);
  tb=(tb+1023)&~(uintptr_t)1023;
  __nv_bfloat16* As=(__nv_bfloat16*)tb;
  __nv_bfloat16* Bs=As+NS*BM_CTA*KCH;
  __nv_bfloat16* smem_out=As;

  int tid=threadIdx.x, warpid=tid>>5;
  uint32_t rank=cluster_rank();
  bool is_leader=(rank==0);

  // grouped rasterization for L2 reuse
  int num_pid_m=(M+MTOT-1)/MTOT;
  int num_pid_n=N/NTOT;
  int cid=blockIdx.y;
  int num_pid_in_group=GROUP_M*num_pid_n;
  int group_id=cid/num_pid_in_group;
  int first_pid_m=group_id*GROUP_M;
  int group_size_m=num_pid_m-first_pid_m; if(group_size_m>GROUP_M) group_size_m=GROUP_M;
  int pair_m=first_pid_m+(cid%group_size_m);
  int block_n=(cid%num_pid_in_group)/group_size_m;

  if(tid==0){
    for(int s=0;s<NS;s++){ init_bar(&full_bar[s],1); init_bar(&empty_bar[s],1); }
    init_bar(mma_done,1);
  }
  cluster_sync();
  if(warpid==0){ tmem_alloc(tmem_ptr,NTOT); }
  cluster_sync();
  uint32_t tmem_base=*tmem_ptr;

  const uint32_t idesc=make_idesc(MTOT,NTOT);
  const int KC=K/KCH;
  const uint32_t TX=(uint32_t)2*(BM_CTA*KCH+BN_CTA*KCH)*2;

  int m_row=pair_m*MTOT + (int)rank*BM_CTA;
  int n_col=block_n*NTOT + (int)rank*BN_CTA;

  if(warpid==0 && elect()){
    for(int p=0;p<KC;p++){
      int buf=p%NS;
      if(p>=NS) mbar_wait(&empty_bar[buf],(uint32_t)((p/NS-1)&1));
      if(is_leader) mbar_arrive_expect_tx(&full_bar[buf],TX);
      int kco=p*KCH;
      __nv_bfloat16* aptr=As+buf*BM_CTA*KCH;
      __nv_bfloat16* bptr=Bs+buf*BN_CTA*KCH;
      tma_load_cg2(&dA,&full_bar[buf],aptr,kco,m_row);
      tma_load_cg2(&dB,&full_bar[buf],bptr,kco,n_col);
    }
  } else if(is_leader && warpid==1 && elect()){
    for(int c=0;c<KC;c++){
      int buf=c%NS;
      mbar_wait(&full_bar[buf],(uint32_t)((c/NS)&1));
      __nv_bfloat16* aptr=As+buf*BM_CTA*KCH;
      __nv_bfloat16* bptr=Bs+buf*BN_CTA*KCH;
      #pragma unroll
      for(int kk=0;kk<KITER;kk++){
        uint64_t da=smem_desc(aptr+16*kk);
        uint64_t db=smem_desc(bptr+16*kk);
        uint32_t accum=(c==0&&kk==0)?0u:1u;
        umma(tmem_base,da,db,idesc,accum);
      }
      umma_commit_2sm(&empty_bar[buf]);
    }
    umma_commit_2sm(mma_done);
  }

  mbar_wait(mma_done,0);
  __syncthreads();

  #pragma unroll
  for(int col=0;col<NTOT;col+=4){
    uint32_t r0,r1,r2,r3;
    uint32_t a=tmem_base+col;
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3},[%4];"
      :"=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3):"r"(a));
    asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory");
    int base=tid*NTOT+col;
    smem_out[base+0]=__float2bfloat16(__uint_as_float(r0));
    smem_out[base+1]=__float2bfloat16(__uint_as_float(r1));
    smem_out[base+2]=__float2bfloat16(__uint_as_float(r2));
    smem_out[base+3]=__float2bfloat16(__uint_as_float(r3));
  }
  __syncthreads();
  int warp=tid>>5, lane=tid&31;
  int grow_base=pair_m*MTOT + (int)rank*BM_CTA;
  int gcol_base=block_n*NTOT;
  #pragma unroll
  for(int s=0;s<BM_CTA/4;s++){
    int row=s*4+warp;
    int grow=grow_base+row;
    int col0=lane*8;
    if(grow<M){
      uint4 data=*reinterpret_cast<uint4*>(&smem_out[row*NTOT+col0]);
      *reinterpret_cast<uint4*>(&C[(size_t)grow*N+gcol_base+col0])=data;
    }
  }
  cluster_sync();
  if(warpid==0) tmem_dealloc(tmem_base,NTOT);
}

static CUresult make_tma(CUtensorMap* d,void* ptr,uint64_t inner,uint64_t outer,uint32_t binner,uint32_t bouter){
  uint64_t gdim[2]={inner,outer};
  uint64_t gstr[1]={inner*2};
  uint32_t bdim[2]={binner,bouter};
  uint32_t estr[2]={1,1};
  return cuTensorMapEncodeTiled(d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, ptr, gdim, gstr, bdim, estr,
    CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C){
  CUDA_CHECK(cudaSetDevice(A.device().device_id));
  int M=A.size(0), K=A.size(1), N=B.size(0);
  __nv_bfloat16* a=static_cast<__nv_bfloat16*>(A.data_ptr());
  __nv_bfloat16* b=static_cast<__nv_bfloat16*>(B.data_ptr());
  __nv_bfloat16* c=static_cast<__nv_bfloat16*>(C.data_ptr());

  CUtensorMap dA,dB;
  CU_CHECK(make_tma(&dA,a,(uint64_t)K,(uint64_t)M,KCH,BM_CTA));
  CU_CHECK(make_tma(&dB,b,(uint64_t)K,(uint64_t)N,KCH,BN_CTA));

  int smem_bytes = 1024 + NS*(BM_CTA*KCH+BN_CTA*KCH)*2;
  CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

  int num_m_pairs = (M+MTOT-1)/MTOT;
  int num_n = N/NTOT;
  dim3 grid(2, num_m_pairs*num_n, 1);
  dim3 block(THREADS);
  cudaStream_t stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type,A.device().device_id));

  cudaLaunchConfig_t cfg={};
  cfg.gridDim=grid; cfg.blockDim=block; cfg.dynamicSmemBytes=smem_bytes; cfg.stream=stream;
  cudaLaunchAttribute attr[1];
  attr[0].id=cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x=2; attr[0].val.clusterDim.y=1; attr[0].val.clusterDim.z=1;
  cfg.attrs=attr; cfg.numAttrs=1;
  CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, dA, dB, c, M, N, K));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_kernel::run);

}  // namespace gemm_kernel