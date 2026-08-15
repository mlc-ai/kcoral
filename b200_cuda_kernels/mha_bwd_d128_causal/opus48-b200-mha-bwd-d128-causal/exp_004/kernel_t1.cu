#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <cmath>
#include <type_traits>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace mha_bwd_cuda {

using namespace nvcuda;
using row_major = wmma::row_major;
using col_major = wmma::col_major;
using bf16 = __nv_bfloat16;
using AccFrag = wmma::fragment<wmma::accumulator,16,16,16,float>;

constexpr int BN = 64;
constexpr int BM = 64;
constexpr int DIM = 128;
constexpr int THREADS = 256;
constexpr int NW = 8;

template<typename LayA>
__device__ __forceinline__ const bf16* aptr(const bf16* A, int lda, int mi, int k){
  if constexpr (std::is_same<LayA, row_major>::value) return A + (mi*16)*lda + (k*16);
  else return A + (mi*16) + (k*16)*lda;
}
template<typename LayB>
__device__ __forceinline__ const bf16* bptr(const bf16* B, int ldb, int k, int ni){
  if constexpr (std::is_same<LayB, row_major>::value) return B + (k*16)*ldb + (ni*16);
  else return B + (k*16) + (ni*16)*ldb;
}

template<int M,int N,int K,typename LayA,typename LayB>
__device__ __forceinline__ void gemm_smem(const bf16* A,int lda,const bf16* B,int ldb,
                                           float* C,int ldc,int warp_id){
  constexpr int MT=M/16,NT=N/16,KT=K/16; constexpr int TOT=MT*NT;
  for(int t=warp_id;t<TOT;t+=NW){
    int mi=t/NT,ni=t%NT;
    AccFrag c; wmma::fill_fragment(c,0.f);
    #pragma unroll
    for(int k=0;k<KT;k++){
      wmma::fragment<wmma::matrix_a,16,16,16,bf16,LayA> af;
      wmma::fragment<wmma::matrix_b,16,16,16,bf16,LayB> bf_;
      wmma::load_matrix_sync(af, aptr<LayA>(A,lda,mi,k), lda);
      wmma::load_matrix_sync(bf_, bptr<LayB>(B,ldb,k,ni), ldb);
      wmma::mma_sync(c,af,bf_,c);
    }
    wmma::store_matrix_sync(C+mi*16*ldc+ni*16, c, ldc, wmma::mem_row_major);
  }
}

template<int M,int N,int K,typename LayA,typename LayB,int NLOC>
__device__ __forceinline__ void gemm_accum(const bf16* A,int lda,const bf16* B,int ldb,
                                            AccFrag* acc,int warp_id){
  constexpr int NT=N/16,KT=K/16;
  #pragma unroll
  for(int li=0;li<NLOC;li++){
    int t=warp_id+li*NW; int mi=t/NT, ni=t%NT;
    #pragma unroll
    for(int k=0;k<KT;k++){
      wmma::fragment<wmma::matrix_a,16,16,16,bf16,LayA> af;
      wmma::fragment<wmma::matrix_b,16,16,16,bf16,LayB> bf_;
      wmma::load_matrix_sync(af, aptr<LayA>(A,lda,mi,k), lda);
      wmma::load_matrix_sync(bf_, bptr<LayB>(B,ldb,k,ni), ldb);
      wmma::mma_sync(acc[li],af,bf_,acc[li]);
    }
  }
}

template<int N,int NLOC>
__device__ __forceinline__ void store_accum(AccFrag* acc,float* C,int ldc,int warp_id){
  constexpr int NT=N/16;
  #pragma unroll
  for(int li=0;li<NLOC;li++){
    int t=warp_id+li*NW; int mi=t/NT, ni=t%NT;
    wmma::store_matrix_sync(C+mi*16*ldc+ni*16, acc[li], ldc, wmma::mem_row_major);
  }
}

__device__ __forceinline__ void load_tile(bf16* dst, const bf16* base, int row0, int rows, int S, int tid){
  if(row0+rows<=S){
    const int4* s=reinterpret_cast<const int4*>(base+(size_t)row0*DIM);
    int4* d=reinterpret_cast<int4*>(dst);
    int cnt = rows*DIM/8;
    for(int i=tid;i<cnt;i+=THREADS) d[i]=s[i];
  } else {
    const bf16 z=__float2bfloat16(0.f);
    for(int idx=tid; idx<rows*DIM; idx+=THREADS){ int r=idx/DIM,e=idx%DIM;int gr=row0+r; dst[idx]=(gr<S)?base[(size_t)gr*DIM+e]:z; }
  }
}

__global__ void compute_D_kernel(const bf16* dO, const bf16* O, float* Dg, long total_rows){
  int row = blockIdx.x*(blockDim.x/32) + (threadIdx.x/32);
  int lane = threadIdx.x%32;
  if(row >= total_rows) return;
  const bf16* dOr = dO + (size_t)row*DIM;
  const bf16* Or  = O  + (size_t)row*DIM;
  float sum=0.f;
  const int4* d4=reinterpret_cast<const int4*>(dOr);
  const int4* o4=reinterpret_cast<const int4*>(Or);
  for(int e=lane;e<DIM/8;e+=32){
    int4 dv=d4[e], ov=o4[e];
    const bf16* db=reinterpret_cast<const bf16*>(&dv);
    const bf16* ob=reinterpret_cast<const bf16*>(&ov);
    #pragma unroll
    for(int k=0;k<8;k++) sum += __bfloat162float(db[k])*__bfloat162float(ob[k]);
  }
  #pragma unroll
  for(int o=16;o>0;o>>=1) sum += __shfl_down_sync(0xffffffff,sum,o);
  if(lane==0) Dg[row]=sum;
}

// ---------------- dK / dV kernel: parallel over key blocks ----------------
__global__ __launch_bounds__(THREADS) void kernel_dkdv(
    const bf16* Q, const bf16* K, const bf16* V, const bf16* dO,
    const float* L, const float* Dg,
    bf16* dKout, bf16* dVout, int S, float scale){

  extern __shared__ char smem[];
  char* p = smem;
  bf16* Ksh=(bf16*)p; p+=16384;
  bf16* Vsh=(bf16*)p; p+=16384;
  bf16* Qsh=(bf16*)p; p+=16384;
  bf16* dOsh=(bf16*)p; p+=16384;
  float* STsh=(float*)p; p+=16384;
  float* dPsh=(float*)p; p+=16384;
  bf16* Pbf=(bf16*)p; p+=8192;
  bf16* dSbf=(bf16*)p; p+=8192;
  float* Lsh=(float*)p; p+=256;
  float* Dsh=(float*)p; p+=256;
  float* stage=(float*)Ksh;

  int tid=threadIdx.x, warp_id=tid/32;
  int bh=blockIdx.y, kb=blockIdx.x;
  int k0=kb*BN;
  if(k0>=S) return;

  const bf16* Kbase=K+(size_t)bh*S*DIM;
  const bf16* Vbase=V+(size_t)bh*S*DIM;
  const bf16* Qbase=Q+(size_t)bh*S*DIM;
  const bf16* dObase=dO+(size_t)bh*S*DIM;
  const float* Lbase=L+(size_t)bh*S;
  const float* Dgbase=Dg+(size_t)bh*S;
  bf16* dKoutbase=dKout+(size_t)bh*S*DIM;
  bf16* dVoutbase=dVout+(size_t)bh*S*DIM;

  load_tile(Ksh,Kbase,k0,BN,S,tid);
  load_tile(Vsh,Vbase,k0,BN,S,tid);

  AccFrag dVacc[4], dKacc[4];
  #pragma unroll
  for(int i=0;i<4;i++){ wmma::fill_fragment(dVacc[i],0.f); wmma::fill_fragment(dKacc[i],0.f); }
  __syncthreads();

  int numQ=(S+BM-1)/BM;
  for(int qb=kb; qb<numQ; qb++){
    int q0=qb*BM;
    load_tile(Qsh,Qbase,q0,BM,S,tid);
    load_tile(dOsh,dObase,q0,BM,S,tid);
    for(int i=tid;i<BM;i+=THREADS){ int gi=q0+i; Lsh[i]=(gi<S)?Lbase[gi]:0.f; Dsh[i]=(gi<S)?Dgbase[gi]:0.f; }
    __syncthreads();

    gemm_smem<BN,BM,DIM,row_major,col_major>(Ksh,DIM,Qsh,DIM,STsh,BM,warp_id);
    gemm_smem<BN,BM,DIM,row_major,col_major>(Vsh,DIM,dOsh,DIM,dPsh,BM,warp_id);
    __syncthreads();

    bool diag=(qb==kb);
    for(int idx=tid; idx<BN*BM; idx+=THREADS){
      int j=idx/BM, i=idx%BM;
      int gj=k0+j, gi=q0+i;
      bool valid=(gj<S)&&(gi<S)&&(!diag||gi>=gj);
      float s=STsh[idx];
      float pv = valid?__expf(scale*s - Lsh[i]):0.f;
      float dp=dPsh[idx];
      float ds= valid?scale*pv*(dp-Dsh[i]):0.f;
      Pbf[idx]=__float2bfloat16(pv);
      dSbf[idx]=__float2bfloat16(ds);
    }
    __syncthreads();

    gemm_accum<BN,DIM,BM,row_major,row_major,4>(Pbf,BM,dOsh,DIM,dVacc,warp_id);
    gemm_accum<BN,DIM,BM,row_major,row_major,4>(dSbf,BM,Qsh,DIM,dKacc,warp_id);
    __syncthreads();
  }

  store_accum<DIM,4>(dVacc, stage, DIM, warp_id);
  __syncthreads();
  for(int idx=tid; idx<BN*DIM; idx+=THREADS){ int r=idx/DIM,e=idx%DIM;int gr=k0+r; if(gr<S) dVoutbase[(size_t)gr*DIM+e]=__float2bfloat16(stage[idx]); }
  __syncthreads();
  store_accum<DIM,4>(dKacc, stage, DIM, warp_id);
  __syncthreads();
  for(int idx=tid; idx<BN*DIM; idx+=THREADS){ int r=idx/DIM,e=idx%DIM;int gr=k0+r; if(gr<S) dKoutbase[(size_t)gr*DIM+e]=__float2bfloat16(stage[idx]); }
}

// ---------------- dQ kernel: parallel over query blocks ----------------
__global__ __launch_bounds__(THREADS) void kernel_dq(
    const bf16* Q, const bf16* K, const bf16* V, const bf16* dO,
    const float* L, const float* Dg,
    bf16* dQout, int S, float scale){

  extern __shared__ char smem[];
  char* p = smem;
  bf16* Qsh=(bf16*)p; p+=16384;
  bf16* dOsh=(bf16*)p; p+=16384;
  bf16* Ksh=(bf16*)p; p+=16384;
  bf16* Vsh=(bf16*)p; p+=16384;
  float* STsh=(float*)p; p+=16384;
  float* dPsh=(float*)p; p+=16384;
  bf16* dSbf=(bf16*)p; p+=8192;
  float* Lsh=(float*)p; p+=256;
  float* Dsh=(float*)p; p+=256;
  float* stage=(float*)Ksh;

  int tid=threadIdx.x, warp_id=tid/32;
  int bh=blockIdx.y, qb=blockIdx.x;
  int q0=qb*BM;
  if(q0>=S) return;

  const bf16* Kbase=K+(size_t)bh*S*DIM;
  const bf16* Vbase=V+(size_t)bh*S*DIM;
  const bf16* Qbase=Q+(size_t)bh*S*DIM;
  const bf16* dObase=dO+(size_t)bh*S*DIM;
  const float* Lbase=L+(size_t)bh*S;
  const float* Dgbase=Dg+(size_t)bh*S;
  bf16* dQoutbase=dQout+(size_t)bh*S*DIM;

  load_tile(Qsh,Qbase,q0,BM,S,tid);
  load_tile(dOsh,dObase,q0,BM,S,tid);
  for(int i=tid;i<BM;i+=THREADS){ int gi=q0+i; Lsh[i]=(gi<S)?Lbase[gi]:0.f; Dsh[i]=(gi<S)?Dgbase[gi]:0.f; }

  AccFrag dQacc[4];
  #pragma unroll
  for(int i=0;i<4;i++) wmma::fill_fragment(dQacc[i],0.f);
  __syncthreads();

  for(int kb=0; kb<=qb; kb++){
    int k0=kb*BN;
    load_tile(Ksh,Kbase,k0,BN,S,tid);
    load_tile(Vsh,Vbase,k0,BN,S,tid);
    __syncthreads();

    gemm_smem<BN,BM,DIM,row_major,col_major>(Ksh,DIM,Qsh,DIM,STsh,BM,warp_id);
    gemm_smem<BN,BM,DIM,row_major,col_major>(Vsh,DIM,dOsh,DIM,dPsh,BM,warp_id);
    __syncthreads();

    bool diag=(kb==qb);
    for(int idx=tid; idx<BN*BM; idx+=THREADS){
      int j=idx/BM, i=idx%BM;
      int gj=k0+j, gi=q0+i;
      bool valid=(gj<S)&&(gi<S)&&(!diag||gi>=gj);
      float s=STsh[idx];
      float pv = valid?__expf(scale*s - Lsh[i]):0.f;
      float dp=dPsh[idx];
      float ds= valid?scale*pv*(dp-Dsh[i]):0.f;
      dSbf[idx]=__float2bfloat16(ds);
    }
    __syncthreads();

    gemm_accum<BM,DIM,BN,col_major,row_major,4>(dSbf,BM,Ksh,DIM,dQacc,warp_id);
    __syncthreads();
  }

  store_accum<DIM,4>(dQacc, stage, DIM, warp_id);
  __syncthreads();
  for(int idx=tid; idx<BM*DIM; idx+=THREADS){ int r=idx/DIM,e=idx%DIM;int gr=q0+r; if(gr<S) dQoutbase[(size_t)gr*DIM+e]=__float2bfloat16(stage[idx]); }
}

static float* s_D=nullptr; static size_t s_Dsz=0;
static bool s_attr=false;

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV){
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int64_t B=Q.size(0), H=Q.size(1), S=Q.size(2);
  int64_t BH=B*H;
  cudaStream_t stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

  const bf16* Qp=static_cast<const bf16*>(Q.data_ptr());
  const bf16* Kp=static_cast<const bf16*>(K.data_ptr());
  const bf16* Vp=static_cast<const bf16*>(V.data_ptr());
  const bf16* Op=static_cast<const bf16*>(O.data_ptr());
  const bf16* dOp=static_cast<const bf16*>(dO.data_ptr());
  const float* Lp=static_cast<const float*>(L.data_ptr());
  bf16* dQp=static_cast<bf16*>(dQ.data_ptr());
  bf16* dKp=static_cast<bf16*>(dK.data_ptr());
  bf16* dVp=static_cast<bf16*>(dV.data_ptr());

  if(S<=0) return;

  size_t Dbytes=(size_t)BH*S*sizeof(float);
  if(s_Dsz<Dbytes){ if(s_D) cudaFree(s_D); CUDA_CHECK(cudaMalloc(&s_D,Dbytes)); s_Dsz=Dbytes; }

  long total_rows=(long)BH*S;
  int rpb=THREADS/32;
  dim3 gridD((total_rows+rpb-1)/rpb);
  compute_D_kernel<<<gridD,THREADS,0,stream>>>(dOp,Op,s_D,total_rows);
  CUDA_CHECK(cudaGetLastError());

  int SMEM_DKDV=115200;
  int SMEM_DQ=107008;
  if(!s_attr){
    CUDA_CHECK(cudaFuncSetAttribute((const void*)kernel_dkdv, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_DKDV));
    CUDA_CHECK(cudaFuncSetAttribute((const void*)kernel_dq, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_DQ));
    s_attr=true;
  }

  float scale=1.0f/sqrtf((float)DIM);
  int numKB=(int)((S+BN-1)/BN);
  int numQB=(int)((S+BM-1)/BM);

  dim3 gridKV(numKB,(unsigned)BH);
  kernel_dkdv<<<gridKV,THREADS,SMEM_DKDV,stream>>>(Qp,Kp,Vp,dOp,Lp,s_D,dKp,dVp,(int)S,scale);
  CUDA_CHECK(cudaGetLastError());

  dim3 gridQ(numQB,(unsigned)BH);
  kernel_dq<<<gridQ,THREADS,SMEM_DQ,stream>>>(Qp,Kp,Vp,dOp,Lp,s_D,dQp,(int)S,scale);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_cuda::run);

}  // namespace mha_bwd_cuda