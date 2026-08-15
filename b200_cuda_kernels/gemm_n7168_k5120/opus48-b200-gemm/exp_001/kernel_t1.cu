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

namespace gemm_tc {

constexpr int BM=128, BN=256, BK=64, STAGES=4, THREADS=128, NCOLS=256;

__device__ __forceinline__ uint32_t smem_addr(const void*p){return (uint32_t)__cvta_generic_to_shared(p);}
__device__ __forceinline__ void init_bar(uint64_t*b,uint32_t c){
  asm volatile("mbarrier.init.shared.b64 [%0], %1;"::"r"(smem_addr(b)),"r"(c));
}
__device__ __forceinline__ void bar_arrive_expect(uint64_t*b,uint32_t tx){
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"::"r"(smem_addr(b)),"r"(tx):"memory");
}
__device__ __forceinline__ void bar_wait(uint64_t*b,uint32_t ph){
  asm volatile("{\n.reg .pred P;\nW%=:\nmbarrier.try_wait.parity.shared.b64 P,[%0],%1;\n@!P bra W%=;\n}\n"
    ::"r"(smem_addr(b)),"r"(ph));
}
__device__ __forceinline__ void tma_2d(const CUtensorMap*d,uint64_t*bar,void*sm,int c0,int c1){
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3,%4}], [%2];"
    ::"r"(smem_addr(sm)),"l"((uint64_t)d),"r"(smem_addr(bar)),"r"(c0),"r"(c1):"memory");
}
__device__ __forceinline__ void tmem_alloc(uint32_t*dst,int nc){
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"::"r"(smem_addr(dst)),"r"(nc));
}
__device__ __forceinline__ void tmem_dealloc(uint32_t addr,int nc){
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"::"r"(addr),"r"(nc));
}
__device__ __forceinline__ uint64_t mk_desc(void*p,uint32_t lbo,uint32_t sbo){
  uint32_t a=smem_addr(p); uint64_t d=0;
  d|=(uint64_t)((a&0x3FFFF)>>4);
  d|=(uint64_t)((lbo&0x3FFFF)>>4)<<16;
  d|=(uint64_t)((sbo&0x3FFFF)>>4)<<32;
  d|=(uint64_t)1<<46;
  d|=(uint64_t)2<<61;   // 128B swizzle
  return d;
}
__device__ __forceinline__ uint32_t mk_idesc(uint32_t M,uint32_t N){
  uint32_t d=0; d|=(1u<<4); d|=(1u<<7); d|=(1u<<10);
  d|=((N/8)<<17); d|=((M/16)<<24); return d;
}
__device__ __forceinline__ void umma(uint32_t tc,uint64_t da,uint64_t db,uint32_t id,uint32_t acc){
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p,%4,0;\n"
    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
    ::"r"(tc),"l"(da),"l"(db),"r"(id),"r"(acc));
}
__device__ __forceinline__ void umma_commit(uint64_t*b){
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
    ::"r"(smem_addr(b)));
}
__device__ __forceinline__ void tmem_ld4(uint32_t addr,uint32_t&r0,uint32_t&r1,uint32_t&r2,uint32_t&r3){
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
    :"=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3):"r"(addr));
}
__device__ __forceinline__ void tmem_ld_wait(){asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory");}

__global__ __launch_bounds__(THREADS) void kernel(
   const __grid_constant__ CUtensorMap tma_A,
   const __grid_constant__ CUtensorMap tma_B,
   __nv_bfloat16* C, int M, int N, int K){
  extern __shared__ char smem_ext[];
  // Align base to 1024 bytes for 128B swizzle correctness (base_offset=0)
  uint32_t sb = smem_addr(smem_ext);
  uint32_t adj = ((sb + 1023u) & ~1023u) - sb;
  char* smem_raw = smem_ext + adj;

  __nv_bfloat16* sA=(__nv_bfloat16*)smem_raw;
  __nv_bfloat16* sB=sA+STAGES*BM*BK;
  uint64_t* full=(uint64_t*)(sB+STAGES*BN*BK);
  uint64_t* empty=full+STAGES;
  uint64_t* mbar_mma=empty+STAGES;
  uint32_t* tmem_ptr=(uint32_t*)(mbar_mma+1);

  int tid=threadIdx.x;
  int warp=tid>>5;
  int bn=blockIdx.x;
  int bm=blockIdx.y;
  int numK=K/BK;

  if(tid==0){
    #pragma unroll
    for(int s=0;s<STAGES;s++){init_bar(&full[s],1);init_bar(&empty[s],1);}
    init_bar(mbar_mma,1);
  }
  if(warp==0){ tmem_alloc(tmem_ptr,NCOLS); }
  __syncthreads();
  uint32_t tmem_base=tmem_ptr[0];

  const int bytesA=BM*BK*2;
  const int bytesB=BN*BK*2;
  uint32_t idesc=mk_idesc(BM,BN);

  if(tid==32){ // producer
    for(int kt=0;kt<numK;kt++){
      int buf=kt%STAGES;
      if(kt>=STAGES){
        int ph=((kt/STAGES)-1)&1;
        bar_wait(&empty[buf],ph);
      }
      bar_arrive_expect(&full[buf],bytesA+bytesB);
      tma_2d(&tma_A,&full[buf],&sA[buf*BM*BK],kt*BK,bm*BM);
      tma_2d(&tma_B,&full[buf],&sB[buf*BN*BK],kt*BK,bn*BN);
    }
  } else if(tid==0){ // consumer / MMA
    for(int kt=0;kt<numK;kt++){
      int buf=kt%STAGES;
      int ph=(kt/STAGES)&1;
      bar_wait(&full[buf],ph);
      #pragma unroll
      for(int kk=0;kk<4;kk++){
        uint64_t da=mk_desc(&sA[buf*BM*BK+kk*16],0,1024);
        uint64_t db=mk_desc(&sB[buf*BN*BK+kk*16],0,1024);
        uint32_t acc=(kt==0&&kk==0)?0u:1u;
        umma(tmem_base,da,db,idesc,acc);
      }
      if(kt<numK-1) umma_commit(&empty[buf]);
      else umma_commit(mbar_mma);
    }
    bar_wait(mbar_mma,0);
  }
  __syncthreads();

  // Epilogue phase 1: TMEM -> smem
  __nv_bfloat16* sout=(__nv_bfloat16*)smem_raw;
  #pragma unroll 1
  for(int col=0;col<BN;col+=4){
    uint32_t r0,r1,r2,r3;
    tmem_ld4(tmem_base+col,r0,r1,r2,r3);
    tmem_ld_wait();
    int base=tid*BN+col;
    sout[base+0]=__float2bfloat16(__uint_as_float(r0));
    sout[base+1]=__float2bfloat16(__uint_as_float(r1));
    sout[base+2]=__float2bfloat16(__uint_as_float(r2));
    sout[base+3]=__float2bfloat16(__uint_as_float(r3));
  }
  __syncthreads();
  if(warp==0){ tmem_dealloc(tmem_base,NCOLS); }

  // Epilogue phase 2: smem -> global (coalesced int4 = 8 bf16/lane)
  int warp_id=tid>>5, lane=tid&31;
  #pragma unroll
  for(int step=0;step<BM/4;step++){
    int row=step*4+warp_id;
    int grow=bm*BM+row;
    int cstart=lane*8;
    int gcol=bn*BN+cstart;
    if(grow<M){
      int4 v=*reinterpret_cast<int4*>(&sout[row*BN+cstart]);
      *reinterpret_cast<int4*>(C+(size_t)grow*N+gcol)=v;
    }
  }
}

static CUresult make_tma_desc(CUtensorMap* d, void* gptr, uint64_t inner, uint64_t outer,
                              uint32_t box_inner, uint32_t box_outer){
  uint64_t globalDim[2]={inner,outer};
  uint64_t globalStrides[1]={inner*2};
  uint32_t boxDim[2]={box_inner,box_outer};
  uint32_t elemStrides[2]={1,1};
  return cuTensorMapEncodeTiled(d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, gptr,
      globalDim, globalStrides, boxDim, elemStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C){
  CUDA_CHECK(cudaSetDevice(A.device().device_id));
  int M=(int)A.size(0), K=(int)A.size(1), N=(int)B.size(0);
  __nv_bfloat16* Ap=(__nv_bfloat16*)A.data_ptr();
  __nv_bfloat16* Bp=(__nv_bfloat16*)B.data_ptr();
  __nv_bfloat16* Cp=(__nv_bfloat16*)C.data_ptr();

  CUtensorMap tA, tB;
  CU_CHECK(make_tma_desc(&tA, Ap, K, M, BK, BM));
  CU_CHECK(make_tma_desc(&tB, Bp, K, N, BK, BN));

  size_t smem = 1024 + (size_t)STAGES*(BM*BK+BN*BK)*2 + (size_t)(2*STAGES+1)*8 + 16;
  CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));

  dim3 grid(N/BN, (M+BM-1)/BM);
  cudaStream_t stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
  kernel<<<grid, THREADS, smem, stream>>>(tA, tB, Cp, M, N, K);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_tc::run);

}  // namespace gemm_tc