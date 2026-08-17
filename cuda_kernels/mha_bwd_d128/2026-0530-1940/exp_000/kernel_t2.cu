#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <mma.h>
#include <math.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

namespace mha_bwd {
namespace wmma = nvcuda::wmma;

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__); exit(1);} } while(0)
#define CU_CHECK(call) do { CUresult _e=(call); if(_e!=CUDA_SUCCESS){ \
  const char* s; cuGetErrorString(_e,&s); fprintf(stderr,"CU error %s at %s:%d\n",s,__FILE__,__LINE__); exit(1);} } while(0)

#define BR 64
#define BN 64
#define HD 128
#define TILE_BYTES (HD*64*2)  // 16384

using FragA_row = wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major>;
using FragA_col = wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::col_major>;
using FragB_row = wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major>;
using FragB_col = wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major>;
using FragC     = wmma::fragment<wmma::accumulator,16,16,16,float>;

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

__global__ void compute_D_kernel(const __nv_bfloat16* dO, const __nv_bfloat16* O, float* D, int total_rows){
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

// ===================== dV kernel =====================
__global__ void __launch_bounds__(128,3) dv_kernel(
    const __grid_constant__ CUtensorMap descQ, const __grid_constant__ CUtensorMap descK,
    const __grid_constant__ CUtensorMap descdO, const float* L,
    __nv_bfloat16* dV, int S, int num_q, int num_kv, float scale){
  extern __shared__ __align__(16) char smem[];
  __nv_bfloat16* sQ =(__nv_bfloat16*)(smem+0);
  __nv_bfloat16* sdO=(__nv_bfloat16*)(smem+16384);
  __nv_bfloat16* sK =(__nv_bfloat16*)(smem+32768);
  float* sScore=(float*)(smem+49152);
  __nv_bfloat16* sP =(__nv_bfloat16*)(smem+65536);
  float* sL=(float*)(smem+73728);
  uint64_t* bar=(uint64_t*)(smem+74112);
  float* dOut=(float*)smem;

  int blk=blockIdx.x;
  int jb=blk%num_kv, bh=blk/num_kv;
  int kv_start=jb*BN, warp=threadIdx.x>>5;
  const float* Lb=L+(size_t)bh*S;

  if(threadIdx.x==0) init_barrier(bar,1);
  __syncthreads();
  uint32_t phase=0;
  if(threadIdx.x==0){ arrive_expect(bar,TILE_BYTES); tma2d(&descK,bar,sK,0,bh*S+kv_start); }
  bar_wait(bar,phase); phase^=1;

  FragC dV_acc[8];
  #pragma unroll
  for(int c=0;c<8;c++) wmma::fill_fragment(dV_acc[c],0.f);

  for(int ib=0; ib<num_q; ib++){
    int q_start=ib*BR;
    if(threadIdx.x==0){ arrive_expect(bar,2*TILE_BYTES);
      tma2d(&descQ,bar,sQ,0,bh*S+q_start); tma2d(&descdO,bar,sdO,0,bh*S+q_start); }
    bar_wait(bar,phase); phase^=1;
    for(int r=threadIdx.x;r<BR;r+=128){ int gr=q_start+r; sL[r]=(gr<S)?Lb[gr]:0.f; }
    __syncthreads();

    #pragma unroll
    for(int c=0;c<BN/16;c++){
      FragC acc; wmma::fill_fragment(acc,0.f);
      #pragma unroll
      for(int k=0;k<HD/16;k++){
        FragA_row a; wmma::load_matrix_sync(a,sQ+(warp*16)*HD+k*16,HD);
        FragB_col b; wmma::load_matrix_sync(b,sK+(c*16)*HD+k*16,HD);
        wmma::mma_sync(acc,a,b,acc);
      }
      wmma::store_matrix_sync(sScore+(warp*16)*BN+c*16,acc,BN,wmma::mem_row_major);
    }
    __syncthreads();
    for(int idx=threadIdx.x;idx<BR*BN;idx+=128){
      int i=idx/BN,j=idx%BN; int gj=kv_start+j,gi=q_start+i;
      float p=(gj<S&&gi<S)?__expf(scale*sScore[idx]-sL[i]):0.f;
      sP[idx]=__float2bfloat16(p);
    }
    __syncthreads();
    #pragma unroll
    for(int c=0;c<HD/16;c++){
      #pragma unroll
      for(int k=0;k<BR/16;k++){
        FragA_col a; wmma::load_matrix_sync(a,sP+(k*16)*BN+warp*16,BN);
        FragB_row b; wmma::load_matrix_sync(b,sdO+(k*16)*HD+c*16,HD);
        wmma::mma_sync(dV_acc[c],a,b,dV_acc[c]);
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for(int c=0;c<8;c++) wmma::store_matrix_sync(dOut+(warp*16)*HD+c*16,dV_acc[c],HD,wmma::mem_row_major);
  __syncthreads();
  for(int idx=threadIdx.x;idx<BN*HD;idx+=128){
    int r=idx/HD,c=idx%HD; int gr=kv_start+r;
    if(gr<S) dV[(size_t)bh*S*HD+(size_t)gr*HD+c]=__float2bfloat16(dOut[idx]);
  }
}

// ===================== dK kernel =====================
__global__ void __launch_bounds__(128,2) dk_kernel(
    const __grid_constant__ CUtensorMap descQ, const __grid_constant__ CUtensorMap descK,
    const __grid_constant__ CUtensorMap descV, const __grid_constant__ CUtensorMap descdO,
    const float* L, const float* Dvec,
    __nv_bfloat16* dK, int S, int num_q, int num_kv, float scale){
  extern __shared__ __align__(16) char smem[];
  __nv_bfloat16* sQ =(__nv_bfloat16*)(smem+0);
  __nv_bfloat16* sdO=(__nv_bfloat16*)(smem+16384);
  __nv_bfloat16* sK =(__nv_bfloat16*)(smem+32768);
  __nv_bfloat16* sV =(__nv_bfloat16*)(smem+49152);
  float* sScore=(float*)(smem+65536);
  __nv_bfloat16* sP =(__nv_bfloat16*)(smem+81920);
  float* sL=(float*)(smem+90112);
  float* sD=(float*)(smem+90368);
  uint64_t* bar=(uint64_t*)(smem+90624);
  float* dOut=(float*)smem;

  int blk=blockIdx.x;
  int jb=blk%num_kv, bh=blk/num_kv;
  int kv_start=jb*BN, warp=threadIdx.x>>5;
  const float* Lb=L+(size_t)bh*S;
  const float* Db=Dvec+(size_t)bh*S;

  if(threadIdx.x==0) init_barrier(bar,1);
  __syncthreads();
  uint32_t phase=0;
  if(threadIdx.x==0){ arrive_expect(bar,2*TILE_BYTES);
    tma2d(&descK,bar,sK,0,bh*S+kv_start); tma2d(&descV,bar,sV,0,bh*S+kv_start); }
  bar_wait(bar,phase); phase^=1;

  FragC dK_acc[8];
  #pragma unroll
  for(int c=0;c<8;c++) wmma::fill_fragment(dK_acc[c],0.f);

  for(int ib=0; ib<num_q; ib++){
    int q_start=ib*BR;
    if(threadIdx.x==0){ arrive_expect(bar,2*TILE_BYTES);
      tma2d(&descQ,bar,sQ,0,bh*S+q_start); tma2d(&descdO,bar,sdO,0,bh*S+q_start); }
    bar_wait(bar,phase); phase^=1;
    for(int r=threadIdx.x;r<BR;r+=128){ int gr=q_start+r; sL[r]=(gr<S)?Lb[gr]:0.f; sD[r]=(gr<S)?Db[gr]:0.f; }
    __syncthreads();

    // S = Q@K^T
    #pragma unroll
    for(int c=0;c<BN/16;c++){
      FragC acc; wmma::fill_fragment(acc,0.f);
      #pragma unroll
      for(int k=0;k<HD/16;k++){
        FragA_row a; wmma::load_matrix_sync(a,sQ+(warp*16)*HD+k*16,HD);
        FragB_col b; wmma::load_matrix_sync(b,sK+(c*16)*HD+k*16,HD);
        wmma::mma_sync(acc,a,b,acc);
      }
      wmma::store_matrix_sync(sScore+(warp*16)*BN+c*16,acc,BN,wmma::mem_row_major);
    }
    __syncthreads();
    // P
    for(int idx=threadIdx.x;idx<BR*BN;idx+=128){
      int i=idx/BN,j=idx%BN; int gj=kv_start+j,gi=q_start+i;
      float p=(gj<S&&gi<S)?__expf(scale*sScore[idx]-sL[i]):0.f;
      sP[idx]=__float2bfloat16(p);
    }
    __syncthreads();
    // dP = dO@V^T
    #pragma unroll
    for(int c=0;c<BN/16;c++){
      FragC acc; wmma::fill_fragment(acc,0.f);
      #pragma unroll
      for(int k=0;k<HD/16;k++){
        FragA_row a; wmma::load_matrix_sync(a,sdO+(warp*16)*HD+k*16,HD);
        FragB_col b; wmma::load_matrix_sync(b,sV+(c*16)*HD+k*16,HD);
        wmma::mma_sync(acc,a,b,acc);
      }
      wmma::store_matrix_sync(sScore+(warp*16)*BN+c*16,acc,BN,wmma::mem_row_major);
    }
    __syncthreads();
    // dS = P*(dP - D)  -> overwrite sP
    for(int idx=threadIdx.x;idx<BR*BN;idx+=128){
      int i=idx/BN; float p=__bfloat162float(sP[idx]);
      sP[idx]=__float2bfloat16(p*(sScore[idx]-sD[i]));
    }
    __syncthreads();
    // dK += dS^T @ Q
    #pragma unroll
    for(int c=0;c<HD/16;c++){
      #pragma unroll
      for(int k=0;k<BR/16;k++){
        FragA_col a; wmma::load_matrix_sync(a,sP+(k*16)*BN+warp*16,BN);
        FragB_row b; wmma::load_matrix_sync(b,sQ+(k*16)*HD+c*16,HD);
        wmma::mma_sync(dK_acc[c],a,b,dK_acc[c]);
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for(int c=0;c<8;c++) wmma::store_matrix_sync(dOut+(warp*16)*HD+c*16,dK_acc[c],HD,wmma::mem_row_major);
  __syncthreads();
  for(int idx=threadIdx.x;idx<BN*HD;idx+=128){
    int r=idx/HD,c=idx%HD; int gr=kv_start+r;
    if(gr<S) dK[(size_t)bh*S*HD+(size_t)gr*HD+c]=__float2bfloat16(dOut[idx]*scale);
  }
}

// ===================== dQ kernel =====================
__global__ void __launch_bounds__(128,2) dq_kernel(
    const __grid_constant__ CUtensorMap descQ, const __grid_constant__ CUtensorMap descK,
    const __grid_constant__ CUtensorMap descV, const __grid_constant__ CUtensorMap descdO,
    const float* L, const float* Dvec,
    __nv_bfloat16* dQ, int S, int num_q, int num_kv, float scale){
  extern __shared__ __align__(16) char smem[];
  __nv_bfloat16* sQ =(__nv_bfloat16*)(smem+0);
  __nv_bfloat16* sdO=(__nv_bfloat16*)(smem+16384);
  __nv_bfloat16* sK =(__nv_bfloat16*)(smem+32768);
  __nv_bfloat16* sV =(__nv_bfloat16*)(smem+49152);
  float* sScore=(float*)(smem+65536);
  __nv_bfloat16* sP =(__nv_bfloat16*)(smem+81920);
  float* sL=(float*)(smem+90112);
  float* sD=(float*)(smem+90368);
  uint64_t* bar=(uint64_t*)(smem+90624);
  float* dOut=(float*)smem;

  int blk=blockIdx.x;
  int ib=blk%num_q, bh=blk/num_q;
  int q_start=ib*BR, warp=threadIdx.x>>5;
  const float* Lb=L+(size_t)bh*S;
  const float* Db=Dvec+(size_t)bh*S;

  if(threadIdx.x==0) init_barrier(bar,1);
  __syncthreads();
  uint32_t phase=0;
  if(threadIdx.x==0){ arrive_expect(bar,2*TILE_BYTES);
    tma2d(&descQ,bar,sQ,0,bh*S+q_start); tma2d(&descdO,bar,sdO,0,bh*S+q_start); }
  bar_wait(bar,phase); phase^=1;
  for(int r=threadIdx.x;r<BR;r+=128){ int gr=q_start+r; sL[r]=(gr<S)?Lb[gr]:0.f; sD[r]=(gr<S)?Db[gr]:0.f; }
  __syncthreads();

  FragC dQ_acc[8];
  #pragma unroll
  for(int c=0;c<8;c++) wmma::fill_fragment(dQ_acc[c],0.f);

  for(int jb=0; jb<num_kv; jb++){
    int kv_start=jb*BN;
    if(threadIdx.x==0){ arrive_expect(bar,2*TILE_BYTES);
      tma2d(&descK,bar,sK,0,bh*S+kv_start); tma2d(&descV,bar,sV,0,bh*S+kv_start); }
    bar_wait(bar,phase); phase^=1;
    __syncthreads();

    // S = Q@K^T
    #pragma unroll
    for(int c=0;c<BN/16;c++){
      FragC acc; wmma::fill_fragment(acc,0.f);
      #pragma unroll
      for(int k=0;k<HD/16;k++){
        FragA_row a; wmma::load_matrix_sync(a,sQ+(warp*16)*HD+k*16,HD);
        FragB_col b; wmma::load_matrix_sync(b,sK+(c*16)*HD+k*16,HD);
        wmma::mma_sync(acc,a,b,acc);
      }
      wmma::store_matrix_sync(sScore+(warp*16)*BN+c*16,acc,BN,wmma::mem_row_major);
    }
    __syncthreads();
    // P
    for(int idx=threadIdx.x;idx<BR*BN;idx+=128){
      int i=idx/BN,j=idx%BN; int gj=kv_start+j,gi=q_start+i;
      float p=(gj<S&&gi<S)?__expf(scale*sScore[idx]-sL[i]):0.f;
      sP[idx]=__float2bfloat16(p);
    }
    __syncthreads();
    // dP = dO@V^T
    #pragma unroll
    for(int c=0;c<BN/16;c++){
      FragC acc; wmma::fill_fragment(acc,0.f);
      #pragma unroll
      for(int k=0;k<HD/16;k++){
        FragA_row a; wmma::load_matrix_sync(a,sdO+(warp*16)*HD+k*16,HD);
        FragB_col b; wmma::load_matrix_sync(b,sV+(c*16)*HD+k*16,HD);
        wmma::mma_sync(acc,a,b,acc);
      }
      wmma::store_matrix_sync(sScore+(warp*16)*BN+c*16,acc,BN,wmma::mem_row_major);
    }
    __syncthreads();
    // dS = P*(dP - D)
    for(int idx=threadIdx.x;idx<BR*BN;idx+=128){
      int i=idx/BN; float p=__bfloat162float(sP[idx]);
      sP[idx]=__float2bfloat16(p*(sScore[idx]-sD[i]));
    }
    __syncthreads();
    // dQ += dS @ K
    #pragma unroll
    for(int c=0;c<HD/16;c++){
      #pragma unroll
      for(int k=0;k<BN/16;k++){
        FragA_row a; wmma::load_matrix_sync(a,sP+(warp*16)*BN+k*16,BN);
        FragB_row b; wmma::load_matrix_sync(b,sK+(k*16)*HD+c*16,HD);
        wmma::mma_sync(dQ_acc[c],a,b,dQ_acc[c]);
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for(int c=0;c<8;c++) wmma::store_matrix_sync(dOut+(warp*16)*HD+c*16,dQ_acc[c],HD,wmma::mem_row_major);
  __syncthreads();
  for(int idx=threadIdx.x;idx<BR*HD;idx+=128){
    int r=idx/HD,c=idx%HD; int gr=q_start+r;
    if(gr<S) dQ[(size_t)bh*S*HD+(size_t)gr*HD+c]=__float2bfloat16(dOut[idx]*scale);
  }
}

static void make_desc(CUtensorMap* d, void* p, uint64_t rows){
  cuuint64_t gdim[2]={(cuuint64_t)HD,(cuuint64_t)rows};
  cuuint64_t gstr[1]={(cuuint64_t)HD*2};
  cuuint32_t bdim[2]={(cuuint32_t)HD,64};
  cuuint32_t estr[2]={1,1};
  CU_CHECK(cuTensorMapEncodeTiled(d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, p, gdim, gstr,
    bdim, estr, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
    CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
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

  int total_rows=BH*S;
  int threadsD=256, wpb=threadsD/32;
  int blocksD=(total_rows+wpb-1)/wpb;
  compute_D_kernel<<<blocksD,threadsD,0,stream>>>(dOp,Op,Dvec,total_rows);

  uint64_t rows=(uint64_t)BH*S;
  CUtensorMap tmaQ,tmaK,tmaV,tmadO;
  make_desc(&tmaQ,Qp,rows); make_desc(&tmaK,Kp,rows);
  make_desc(&tmaV,Vp,rows); make_desc(&tmadO,dOp,rows);

  int num_q=(S+BR-1)/BR, num_kv=(S+BN-1)/BN;
  float scale=1.0f/sqrtf((float)HD);

  size_t smem_dv=74240, smem_dk=90752, smem_dq=90752;
  CUDA_CHECK(cudaFuncSetAttribute(dv_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem_dv));
  CUDA_CHECK(cudaFuncSetAttribute(dk_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem_dk));
  CUDA_CHECK(cudaFuncSetAttribute(dq_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem_dq));

  dv_kernel<<<BH*num_kv,128,smem_dv,stream>>>(tmaQ,tmaK,tmadO,Lp,dVp,S,num_q,num_kv,scale);
  dk_kernel<<<BH*num_kv,128,smem_dk,stream>>>(tmaQ,tmaK,tmaV,tmadO,Lp,Dvec,dKp,S,num_q,num_kv,scale);
  dq_kernel<<<BH*num_q ,128,smem_dq,stream>>>(tmaQ,tmaK,tmaV,tmadO,Lp,Dvec,dQp,S,num_q,num_kv,scale);

  CUDA_CHECK(cudaFreeAsync(Dvec,stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd