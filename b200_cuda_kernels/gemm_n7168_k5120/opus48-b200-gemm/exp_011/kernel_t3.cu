#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__);} } while(0)
#define CU_CHECK(call) do { CUresult _e=(call); if(_e!=CUDA_SUCCESS){ \
  const char* s; cuGetErrorString(_e,&s); fprintf(stderr,"CU error %s at %s:%d\n",s,__FILE__,__LINE__);} } while(0)

namespace gemm_ns {
using bf16 = __nv_bfloat16;

constexpr int NN=7168, KK=5120;
constexpr int BM=128;    // per-CTA M rows
constexpr int BN=256;    // combined N per pair
constexpr int BNH=128;   // per-CTA B cols (N half)
constexpr int BK=64;
constexpr int STAGES=5;
constexpr int T=KK/BK;   // 80
constexpr int NUM_N=NN/BN; // 28
constexpr int GROUP_M=8;

__device__ __forceinline__ uint32_t sh(const void*p){return (uint32_t)__cvta_generic_to_shared(p);}
__device__ __forceinline__ bool elect_one(){uint32_t p; asm volatile("{\n.reg .pred q;\nelect.sync _|q,0xFFFFFFFF;\nselp.b32 %0,1,0,q;\n}\n":"=r"(p)); return p!=0;}
__device__ __forceinline__ uint32_t cluster_rank(){uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;":"=r"(r)); return r;}
__device__ __forceinline__ void bar_init(uint64_t*b,uint32_t c){asm volatile("mbarrier.init.shared.b64 [%0], %1;"::"r"(sh(b)),"r"(c));}
__device__ __forceinline__ void fence_bar_init(){asm volatile("fence.mbarrier_init.release.cluster;\n":::"memory");}
__device__ __forceinline__ void bar_arrive_tx(uint64_t*b,uint32_t tx){asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"::"r"(sh(b)),"r"(tx):"memory");}
__device__ __forceinline__ void bar_wait(uint64_t*b,uint32_t ph){
  asm volatile("{\n.reg .pred P;\nW%=:\nmbarrier.try_wait.parity.shared.b64 P,[%0],%1;\n@!P bra W%=;\n}\n"::"r"(sh(b)),"r"(ph));
}
__device__ __forceinline__ uint32_t mapa_rank0(uint32_t a){uint32_t r; uint32_t z=0; asm volatile("mapa.shared::cluster.u32 %0,%1,%2;":"=r"(r):"r"(a),"r"(z)); return r;}
__device__ __forceinline__ void cluster_sync(){asm volatile("barrier.cluster.arrive;\nbarrier.cluster.wait;\n":::"memory");}

__device__ __forceinline__ void tma_load_ownbar(const CUtensorMap*d,uint64_t*bar,void*smem,int c0,int c1){
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3,%4}], [%2];"
    ::"r"(sh(smem)),"l"((uint64_t)d),"r"(sh(bar)),"r"(c0),"r"(c1):"memory");
}
__device__ __forceinline__ void tma_load_rembar(const CUtensorMap*d,uint32_t rembar,void*smem,int c0,int c1){
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3,%4}], [%2];"
    ::"r"(sh(smem)),"l"((uint64_t)d),"r"(rembar),"r"(c0),"r"(c1):"memory");
}
__device__ __forceinline__ uint64_t make_desc(const void*p,uint32_t lbo,uint32_t sbo){
  uint64_t d=0; uint32_t a=sh(p);
  d|=(uint64_t)(a&0x3FFFF)>>4;
  d|=(uint64_t)((lbo&0x3FFFF)>>4)<<16;
  d|=(uint64_t)((sbo&0x3FFFF)>>4)<<32;
  d|=(uint64_t)1<<46;
  d|=(uint64_t)2<<61; // SWIZZLE_128B
  return d;
}
__device__ __forceinline__ uint32_t make_idesc(uint32_t M,uint32_t N){
  uint32_t d=0; d|=(1u<<4); d|=(1u<<7); d|=(1u<<10); d|=((N/8)<<17); d|=((M/16)<<24); return d;
}
__device__ __forceinline__ void tmem_alloc(uint32_t*dst,int nc){asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"::"r"(sh(dst)),"r"(nc));}
__device__ __forceinline__ void tmem_dealloc(uint32_t a,int nc){asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"::"r"(a),"r"(nc));}
__device__ __forceinline__ void umma_cg2(uint32_t tc,uint64_t da,uint64_t db,uint32_t id,uint32_t acc){
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p,%4,0;\ntcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}\n"
    ::"r"(tc),"l"(da),"l"(db),"r"(id),"r"(acc));
}
__device__ __forceinline__ void umma_commit_cg2(uint64_t*b){
  uint32_t a=sh(b);
  asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
    ::"r"(a),"h"((uint16_t)0x3):"memory");
}

__global__ __launch_bounds__(128) void gemm_kernel(
    const __grid_constant__ CUtensorMap dA,
    const __grid_constant__ CUtensorMap dB,
    bf16* __restrict__ C, int M, int num_m, int num_n){
  extern __shared__ __align__(1024) char smem[];
  bf16* A_s=(bf16*)smem;                 // STAGES*BM*BK
  bf16* B_s=A_s + STAGES*BM*BK;          // STAGES*BNH*BK

  __shared__ __align__(8) uint64_t full_b[STAGES];
  __shared__ __align__(8) uint64_t empty_b[STAGES];
  __shared__ uint32_t tmem_base_s;

  int tid=threadIdx.x, warp=tid>>5;
  uint32_t rank=cluster_rank();

  int cid=blockIdx.x>>1;
  int npg=GROUP_M*num_n;
  int gid=cid/npg;
  int first_m=gid*GROUP_M;
  int gsize=num_m-first_m; if(gsize>GROUP_M) gsize=GROUP_M;
  int pid_m=first_m+(cid%gsize);
  int pid_n=(cid%npg)/gsize;
  int bm_base=pid_m*BN;   // combined M tile base (256 rows)
  int bn_base=pid_n*BN;
  int my_a_row=bm_base + rank*BM;   // this CTA's 128 A rows
  int my_b_col=bn_base + rank*BNH;  // this CTA's 128 B cols

  if(warp==0) tmem_alloc(&tmem_base_s,BN); // 256 columns
  if(tid==0){
    #pragma unroll
    for(int s=0;s<STAGES;s++){ bar_init(&full_b[s],1); bar_init(&empty_b[s],1); }
  }
  fence_bar_init();
  __syncthreads();
  cluster_sync();
  uint32_t tmem_base=tmem_base_s;

  const uint32_t idesc=make_idesc(BN,BN); // M=256 combined, N=256 combined
  const uint32_t TX=(2*BM*BK + 2*BNH*BK)*2; // all 4 loads (both CTAs)

  if(warp==0){
    // producer
    if(elect_one()){
      if(rank==0){
        for(int kt=0;kt<T;kt++){
          int st=kt%STAGES;
          if(kt>=STAGES){ uint32_t p=(((kt-STAGES)/STAGES)&1); bar_wait(&empty_b[st],p); }
          bar_arrive_tx(&full_b[st],TX);
          bf16* Ad=A_s+st*BM*BK;
          bf16* Bd=B_s+st*BNH*BK;
          tma_load_ownbar(&dA,&full_b[st],Ad,kt*BK,my_a_row);
          tma_load_ownbar(&dB,&full_b[st],Bd,kt*BK,my_b_col);
        }
      } else {
        for(int kt=0;kt<T;kt++){
          int st=kt%STAGES;
          if(kt>=STAGES){ uint32_t p=(((kt-STAGES)/STAGES)&1); bar_wait(&empty_b[st],p); }
          uint32_t rb=mapa_rank0(sh(&full_b[st]));
          bf16* Ad=A_s+st*BM*BK;
          bf16* Bd=B_s+st*BNH*BK;
          tma_load_rembar(&dA,rb,Ad,kt*BK,my_a_row);
          tma_load_rembar(&dB,rb,Bd,kt*BK,my_b_col);
        }
      }
    }
  } else if(warp==1 && rank==0){
    // consumer (leader only)
    if(elect_one()){
      for(int kt=0;kt<T;kt++){
        int st=kt%STAGES;
        uint32_t p=((kt/STAGES)&1);
        bar_wait(&full_b[st],p);
        bf16* Ad=A_s+st*BM*BK;
        bf16* Bd=B_s+st*BNH*BK;
        #pragma unroll
        for(int ks=0;ks<BK/16;ks++){
          uint64_t da=make_desc(Ad+ks*16,1,1024);
          uint64_t db=make_desc(Bd+ks*16,1,1024);
          uint32_t acc=(kt==0&&ks==0)?0u:1u;
          umma_cg2(tmem_base,da,db,idesc,acc);
        }
        umma_commit_cg2(&empty_b[st]);
      }
      uint32_t lst=(T-1)%STAGES;
      uint32_t lp=(((T-1)/STAGES)&1);
      bar_wait(&empty_b[lst],lp);
    }
  }

  __syncthreads();
  cluster_sync();
  asm volatile("tcgen05.fence::after_thread_sync;\n":::"memory");

  // epilogue: each CTA reads its own 128x256 TMEM tile
  bf16* out_s=(bf16*)smem;
  #pragma unroll
  for(int col=0;col<BN;col+=8){
    uint32_t r0,r1,r2,r3,r4,r5,r6,r7;
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];\n"
      :"=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7)
      :"r"(tmem_base+(uint32_t)col));
    asm volatile("tcgen05.wait::ld.sync.aligned;\n":::"memory");
    bf16* o=out_s+tid*BN+col;
    o[0]=__float2bfloat16(__uint_as_float(r0));
    o[1]=__float2bfloat16(__uint_as_float(r1));
    o[2]=__float2bfloat16(__uint_as_float(r2));
    o[3]=__float2bfloat16(__uint_as_float(r3));
    o[4]=__float2bfloat16(__uint_as_float(r4));
    o[5]=__float2bfloat16(__uint_as_float(r5));
    o[6]=__float2bfloat16(__uint_as_float(r6));
    o[7]=__float2bfloat16(__uint_as_float(r7));
  }
  __syncthreads();

  const int VEC=BN/8; // 32 uint4 per row
  for(int i=tid;i<BM*VEC;i+=128){
    int row=i/VEC, vc=i%VEC, col=vc*8;
    int grow=my_a_row+row;
    if(grow<M){
      uint4 v=*reinterpret_cast<uint4*>(&out_s[row*BN+col]);
      *reinterpret_cast<uint4*>(&C[(int64_t)grow*NN+bn_base+col])=v;
    }
  }
  __syncthreads();
  cluster_sync();
  if(warp==0) tmem_dealloc(tmem_base,BN);
}

static CUresult make_tma(CUtensorMap*d,void*ptr,uint64_t inner,uint64_t outer,uint32_t bi,uint32_t bo){
  uint64_t gdim[2]={inner,outer};
  uint64_t gstride[1]={inner*2};
  uint32_t bdim[2]={bi,bo};
  uint32_t estr[2]={1,1};
  return cuTensorMapEncodeTiled(d,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,ptr,gdim,gstride,bdim,estr,
    CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,
    CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C){
  CUDA_CHECK(cudaSetDevice(A.device().device_id));
  int M=(int)A.size(0);
  if(M<=0) return;
  bf16* Ap=static_cast<bf16*>(A.data_ptr());
  bf16* Bp=static_cast<bf16*>(B.data_ptr());
  bf16* Cp=static_cast<bf16*>(C.data_ptr());

  CUtensorMap dA,dB;
  CU_CHECK(make_tma(&dA,Ap,KK,(uint64_t)M,BK,BM));
  CU_CHECK(make_tma(&dB,Bp,KK,NN,BK,BNH));

  int num_m=(M+BN-1)/BN;  // number of 256-row tiles
  int total_clusters=num_m*NUM_N;
  size_t smem=(size_t)STAGES*(BM*BK+BNH*BK)*sizeof(bf16);
  cudaFuncSetAttribute(gemm_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem);

  cudaStream_t stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type,A.device().device_id));

  cudaLaunchConfig_t config={};
  config.gridDim=dim3(total_clusters*2,1,1);
  config.blockDim=dim3(128,1,1);
  config.dynamicSmemBytes=smem;
  config.stream=stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id=cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x=2;
  attrs[0].val.clusterDim.y=1;
  attrs[0].val.clusterDim.z=1;
  config.attrs=attrs;
  config.numAttrs=1;
  cudaLaunchKernelEx(&config,gemm_kernel,dA,dB,Cp,M,num_m,NUM_N);
  CUDA_CHECK(cudaGetLastError());
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_ns::run);
} // namespace gemm_ns