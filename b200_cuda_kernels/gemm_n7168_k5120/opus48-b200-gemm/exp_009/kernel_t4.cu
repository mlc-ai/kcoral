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

constexpr int BM=128, BN=256, KCH=64, NS=4;
constexpr int KITER=KCH/16;   // 4
constexpr int THREADS=128;

__device__ __forceinline__ uint32_t cvta(const void* p){ return (uint32_t)__cvta_generic_to_shared(p); }

__device__ __forceinline__ bool elect(){
  uint32_t pred;
  asm volatile("{\n.reg .pred p;\nelect.sync _|p, 0xFFFFFFFF;\nselp.b32 %0,1,0,p;\n}\n":"=r"(pred));
  return pred!=0;
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
__device__ __forceinline__ void tma_load_2d(const CUtensorMap* d,uint64_t* bar,void* smem,int32_t c0,int32_t c1){
  asm volatile(
    "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
    :: "r"(cvta(smem)),"l"((uint64_t)d),"r"(cvta(&bar[0])),"r"(c0),"r"(c1):"memory");
}
__device__ __forceinline__ void tmem_alloc(uint32_t* dst,int ncols){
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"::"r"(cvta(dst)),"r"(ncols));
}
__device__ __forceinline__ void tmem_dealloc(uint32_t addr,int ncols){
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"::"r"(addr),"r"(ncols));
}
__device__ __forceinline__ void umma(uint32_t tmem_c,uint64_t da,uint64_t db,uint32_t idesc,uint32_t accum){
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p,%4,0;\n"
    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
    ::"r"(tmem_c),"l"(da),"l"(db),"r"(idesc),"r"(accum));
}
__device__ __forceinline__ void umma_commit(uint64_t* bar){
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"::"r"(cvta(bar)):"memory");
}
__device__ __forceinline__ uint64_t smem_desc(void* p){
  uint64_t d=0; uint32_t a=cvta(p);
  d |= (uint64_t)((a&0x3FFFF)>>4);
  d |= (uint64_t)0 << 16;                  // LBO unused for swizzle
  d |= (uint64_t)((1024u&0x3FFFF)>>4) << 32; // SBO=1024
  d |= (uint64_t)1 << 46;                   // fixed 0b001
  d |= (uint64_t)2 << 61;                   // 128B swizzle
  return d;
}
__device__ __forceinline__ uint32_t make_idesc(uint32_t M,uint32_t N){
  uint32_t d=0;
  d |= (1u<<4);   // dtype F32
  d |= (1u<<7);   // atype BF16
  d |= (1u<<10);  // btype BF16
  d |= (0u<<15);  // A K-major
  d |= (0u<<16);  // B K-major
  d |= ((N/8)<<17);
  d |= ((M/16)<<24);
  return d;
}

__device__ __forceinline__ void run_epilogue(__nv_bfloat16* D,__nv_bfloat16* smem_out,
    uint32_t tmem_base,int M,int N,int block_m,int block_n){
  int tid=threadIdx.x;
  #pragma unroll
  for(int col=0;col<BN;col+=4){
    uint32_t r0,r1,r2,r3;
    uint32_t a=tmem_base+col;
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3},[%4];"
      :"=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3):"r"(a));
    asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory");
    int base=tid*BN+col;
    smem_out[base+0]=__float2bfloat16(__uint_as_float(r0));
    smem_out[base+1]=__float2bfloat16(__uint_as_float(r1));
    smem_out[base+2]=__float2bfloat16(__uint_as_float(r2));
    smem_out[base+3]=__float2bfloat16(__uint_as_float(r3));
  }
  __syncthreads();
  int warp=tid>>5, lane=tid&31;
  #pragma unroll
  for(int s=0;s<BM/4;s++){
    int row=s*4+warp;
    int grow=block_m*BM+row;
    int col0=lane*8;
    int gcol=block_n*BN+col0;
    if(grow<M){
      uint4 data=*reinterpret_cast<uint4*>(&smem_out[row*BN+col0]);
      *reinterpret_cast<uint4*>(&D[(size_t)grow*N+gcol])=data;
    }
  }
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
  __nv_bfloat16* Bs=As+NS*BM*KCH;
  __nv_bfloat16* smem_out=As;

  int tid=threadIdx.x, warpid=tid>>5;
  int block_m=blockIdx.y, block_n=blockIdx.x;

  if(tid==0){
    for(int s=0;s<NS;s++){ init_bar(&full_bar[s],1); init_bar(&empty_bar[s],1); }
    init_bar(mma_done,1);
  }
  __syncthreads();
  if(warpid==0){ tmem_alloc(tmem_ptr,BN); }
  __syncthreads();
  uint32_t tmem_base=*tmem_ptr;

  const uint32_t idesc=make_idesc(BM,BN);
  const int KC=K/KCH;
  const uint32_t TX=(uint32_t)(BM*KCH+BN*KCH)*2;

  if(warpid==0 && elect()){
    // producer
    for(int p=0;p<KC;p++){
      int buf=p%NS;
      if(p>=NS) mbar_wait(&empty_bar[buf],(uint32_t)((p/NS-1)&1));
      mbar_arrive_expect_tx(&full_bar[buf],TX);
      int kco=p*KCH;
      __nv_bfloat16* aptr=As+buf*BM*KCH;
      __nv_bfloat16* bptr=Bs+buf*BN*KCH;
      tma_load_2d(&dA,&full_bar[buf],aptr,kco,block_m*BM);
      tma_load_2d(&dB,&full_bar[buf],bptr,kco,block_n*BN);
    }
  } else if(warpid==1 && elect()){
    // consumer
    for(int c=0;c<KC;c++){
      int buf=c%NS;
      mbar_wait(&full_bar[buf],(uint32_t)((c/NS)&1));
      __nv_bfloat16* aptr=As+buf*BM*KCH;
      __nv_bfloat16* bptr=Bs+buf*BN*KCH;
      #pragma unroll
      for(int kk=0;kk<KITER;kk++){
        uint64_t da=smem_desc(aptr+16*kk);
        uint64_t db=smem_desc(bptr+16*kk);
        uint32_t accum=(c==0&&kk==0)?0u:1u;
        umma(tmem_base,da,db,idesc,accum);
      }
      umma_commit(&empty_bar[buf]);
    }
    umma_commit(mma_done);
  }
  mbar_wait(mma_done,0);
  __syncthreads();

  run_epilogue(C,smem_out,tmem_base,M,N,block_m,block_n);
  __syncthreads();
  if(warpid==0) tmem_dealloc(tmem_base,BN);
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
  CU_CHECK(make_tma(&dA,a,(uint64_t)K,(uint64_t)M,KCH,BM));
  CU_CHECK(make_tma(&dB,b,(uint64_t)K,(uint64_t)N,KCH,BN));

  int smem_bytes = 1024 + NS*(BM*KCH+BN*KCH)*2;
  CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

  dim3 grid(N/BN,(M+BM-1)/BM);
  dim3 block(THREADS);
  cudaStream_t stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type,A.device().device_id));
  kernel<<<grid,block,smem_bytes,stream>>>(dA,dB,c,M,N,K);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_kernel::run);

}  // namespace gemm_kernel