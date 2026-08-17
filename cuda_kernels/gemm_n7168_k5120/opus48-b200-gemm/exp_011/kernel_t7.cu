#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <algorithm>
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
constexpr int BNH=128;   // per-CTA N cols (combined 256)
constexpr int BK=64;
constexpr int STAGES=4;
constexpr int T=KK/BK;   // 80
constexpr int NUM_N=NN/256; // 28
constexpr int GROUP_M=8;

__device__ __forceinline__ uint32_t sh(const void*p){return (uint32_t)__cvta_generic_to_shared(p);}
__device__ __forceinline__ bool elect_one(){uint32_t p; asm volatile("{\n.reg .pred q;\nelect.sync _|q,0xFFFFFFFF;\nselp.b32 %0,1,0,q;\n}\n":"=r"(p)); return p!=0;}
__device__ __forceinline__ uint32_t cluster_rank(){uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;":"=r"(r)); return r;}
__device__ __forceinline__ void bar_init(uint64_t*b,uint32_t c){asm volatile("mbarrier.init.shared.b64 [%0], %1;"::"r"(sh(b)),"r"(c));}
__device__ __forceinline__ void fence_bar_init(){asm volatile("fence.mbarrier_init.release.cluster;\n":::"memory");}
__device__ __forceinline__ void bar_arrive_tx(uint64_t*b,uint32_t tx){asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"::"r"(sh(b)),"r"(tx):"memory");}
__device__ __forceinline__ void mbar_arrive(uint64_t*b){asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"::"r"(sh(b)):"memory");}
__device__ __forceinline__ void mbar_arrive_rank0(uint64_t*b){uint32_t a=sh(b),r,z=0; asm volatile("mapa.shared::cluster.u32 %0,%1,%2;":"=r"(r):"r"(a),"r"(z)); asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];"::"r"(r):"memory");}
__device__ __forceinline__ void bar_wait(uint64_t*b,uint32_t ph){
  asm volatile("{\n.reg .pred P;\nW%=:\nmbarrier.try_wait.parity.shared.b64 P,[%0],%1;\n@!P bra W%=;\n}\n"::"r"(sh(b)),"r"(ph));
}
__device__ __forceinline__ void cluster_sync(){asm volatile("barrier.cluster.arrive;\nbarrier.cluster.wait;\n":::"memory");}
__device__ __forceinline__ void nbar(int id,int cnt){asm volatile("barrier.sync.aligned %0, %1;"::"r"(id),"r"(cnt));}

__device__ __forceinline__ void tma_load_cg2(const CUtensorMap*d,uint64_t*bar,void*smem,int c0,int c1){
  uint32_t sa=sh(smem); uint32_t ba=sh(bar)&0xFEFFFFFF;
  asm volatile("cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2,%3}], [%4];"
    ::"r"(sa),"l"((uint64_t)d),"r"(c0),"r"(c1),"r"(ba):"memory");
}
__device__ __forceinline__ uint64_t make_desc(const void*p,uint32_t lbo,uint32_t sbo){
  uint64_t d=0; uint32_t a=sh(p);
  d|=(uint64_t)(a&0x3FFFF)>>4;
  d|=(uint64_t)((lbo&0x3FFFF)>>4)<<16;
  d|=(uint64_t)((sbo&0x3FFFF)>>4)<<32;
  d|=(uint64_t)1<<46; d|=(uint64_t)2<<61;
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
__device__ __forceinline__ void commit_mc(uint64_t*b){
  uint32_t a=sh(b);
  asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
    ::"r"(a),"h"((uint16_t)0x3):"memory");
}

__device__ __forceinline__ void coords(int gt,int num_m,int num_n,uint32_t rank,int&a_row,int&b_col,int&bn_base){
  int npg=GROUP_M*num_n;
  int grp=gt/npg;
  int first_m=grp*GROUP_M;
  int gsz=num_m-first_m; if(gsz>GROUP_M)gsz=GROUP_M;
  int idx=gt%npg;
  int pid_m=first_m+idx%gsz;
  int pid_n=idx/gsz;
  int bm=pid_m*256, bn=pid_n*256;
  a_row=bm+rank*BM; b_col=bn+rank*BNH; bn_base=bn;
}

__global__ __launch_bounds__(256) void gemm_kernel(
    const __grid_constant__ CUtensorMap dA,
    const __grid_constant__ CUtensorMap dB,
    bf16* __restrict__ C, int M, int num_m, int num_n, int numClusters, int total_tiles){
  extern __shared__ __align__(1024) char smem[];
  bf16* A_s =(bf16*)smem;                 // STAGES*BM*BK
  bf16* B_s =A_s + STAGES*BM*BK;          // STAGES*BNH*BK
  bf16* epi_s=B_s + STAGES*BNH*BK;        // 128*256

  __shared__ __align__(8) uint64_t full_b[STAGES];
  __shared__ __align__(8) uint64_t empty_b[STAGES];
  __shared__ __align__(8) uint64_t tmem_full[2];
  __shared__ __align__(8) uint64_t tmem_free[2];
  __shared__ uint32_t tmem_base_s;

  int tid=threadIdx.x, warp=tid>>5;
  uint32_t rank=cluster_rank();
  int clusterId=blockIdx.x>>1;

  if(warp==0) tmem_alloc(&tmem_base_s,512);
  if(tid==0){
    #pragma unroll
    for(int s=0;s<STAGES;s++){ bar_init(&full_b[s],1); bar_init(&empty_b[s],1); }
    bar_init(&tmem_full[0],1); bar_init(&tmem_full[1],1);
    bar_init(&tmem_free[0],2); bar_init(&tmem_free[1],2);
  }
  fence_bar_init();
  __syncthreads();
  cluster_sync();
  uint32_t tmem_base=tmem_base_s;

  const uint32_t idesc=make_idesc(256,256);
  const uint32_t TX=(2*BM*BK + 2*BNH*BK)*2; // 4 loads across both CTAs

  if(warp==0){
    // producer (both CTAs)
    if(elect_one()){
      int tl=0,gt;
      while((gt=clusterId+tl*numClusters)<total_tiles){
        int a_row,b_col,bn; coords(gt,num_m,num_n,rank,a_row,b_col,bn);
        for(int k=0;k<T;k++){
          int gstep=tl*T+k; int st=gstep%STAGES;
          if(gstep>=STAGES){ uint32_t p=((gstep-STAGES)/STAGES)&1; bar_wait(&empty_b[st],p); }
          if(rank==0) bar_arrive_tx(&full_b[st],TX);
          bf16* Ad=A_s+st*BM*BK; bf16* Bd=B_s+st*BNH*BK;
          tma_load_cg2(&dA,&full_b[st],Ad,k*BK,a_row);
          tma_load_cg2(&dB,&full_b[st],Bd,k*BK,b_col);
        }
        tl++;
      }
    }
  } else if(warp==1 && rank==0){
    // consumer / MMA (leader only)
    if(elect_one()){
      int tl=0,gt;
      while((gt=clusterId+tl*numClusters)<total_tiles){
        int b=tl&1, r=tl>>1;
        if(tl>=2){ bar_wait(&tmem_free[b], (uint32_t)((r-1)&1)); }
        for(int k=0;k<T;k++){
          int gstep=tl*T+k; int st=gstep%STAGES;
          bar_wait(&full_b[st], (uint32_t)((gstep/STAGES)&1));
          bf16* Ad=A_s+st*BM*BK; bf16* Bd=B_s+st*BNH*BK;
          #pragma unroll
          for(int ks=0;ks<BK/16;ks++){
            uint64_t da=make_desc(Ad+ks*16,1,1024);
            uint64_t db=make_desc(Bd+ks*16,1,1024);
            uint32_t acc=(k==0&&ks==0)?0u:1u;
            umma_cg2(tmem_base+b*256, da,db,idesc,acc);
          }
          commit_mc(&empty_b[st]);
          if(k==T-1) commit_mc(&tmem_full[b]);
        }
        tl++;
      }
    }
  } else if(warp>=4){
    // epilogue warpgroup (both CTAs)
    int ltid=tid-128;
    int tl=0,gt;
    while((gt=clusterId+tl*numClusters)<total_tiles){
      int a_row,b_col,bn; coords(gt,num_m,num_n,rank,a_row,b_col,bn);
      int b=tl&1, r=tl>>1;
      bar_wait(&tmem_full[b], (uint32_t)(r&1));
      #pragma unroll
      for(int c0=0;c0<256;c0+=32){
        uint32_t rg[32];
        #pragma unroll
        for(int j=0;j<32;j+=8){
          asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];\n"
            :"=r"(rg[j]),"=r"(rg[j+1]),"=r"(rg[j+2]),"=r"(rg[j+3]),
             "=r"(rg[j+4]),"=r"(rg[j+5]),"=r"(rg[j+6]),"=r"(rg[j+7])
            :"r"(tmem_base+(uint32_t)(b*256+c0+j)));
        }
        asm volatile("tcgen05.wait::ld.sync.aligned;\n":::"memory");
        #pragma unroll
        for(int j=0;j<32;j++) epi_s[ltid*256 + c0 + j]=__float2bfloat16(__uint_as_float(rg[j]));
      }
      nbar(1,128);
      if(ltid==0){ if(rank==0) mbar_arrive(&tmem_free[b]); else mbar_arrive_rank0(&tmem_free[b]); }
      // coalesced global writes
      for(int i=ltid;i<128*32;i+=128){
        int row=i>>5, vc=i&31, col=vc*8;
        int grow=a_row+row;
        if(grow<M){
          uint4 v=*reinterpret_cast<uint4*>(&epi_s[row*256+col]);
          *reinterpret_cast<uint4*>(&C[(int64_t)grow*NN+bn+col])=v;
        }
      }
      nbar(1,128);
      tl++;
    }
  }

  __syncthreads();
  cluster_sync();
  if(warp==0) tmem_dealloc(tmem_base,512);
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

  int num_m=(M+255)/256;
  int num_n=NUM_N;
  int total_tiles=num_m*num_n;
  int numClusters=std::min(total_tiles,74);
  if(numClusters<1) numClusters=1;

  size_t smem=(size_t)(STAGES*(BM*BK+BNH*BK) + 128*256)*sizeof(bf16);
  cudaFuncSetAttribute(gemm_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem);

  cudaStream_t stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type,A.device().device_id));

  cudaLaunchConfig_t config={};
  config.gridDim=dim3(numClusters*2,1,1);
  config.blockDim=dim3(256,1,1);
  config.dynamicSmemBytes=smem;
  config.stream=stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id=cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x=2; attrs[0].val.clusterDim.y=1; attrs[0].val.clusterDim.z=1;
  config.attrs=attrs; config.numAttrs=1;
  cudaLaunchKernelEx(&config,gemm_kernel,dA,dB,Cp,M,num_m,num_n,numClusters,total_tiles);
  CUDA_CHECK(cudaGetLastError());
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_ns::run);
} // namespace gemm_ns