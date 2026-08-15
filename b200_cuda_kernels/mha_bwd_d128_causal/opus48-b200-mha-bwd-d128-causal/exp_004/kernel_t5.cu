#include <cuda_bf16.h>
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

constexpr int BN=64, BM=64, DIM=128, THREADS=256, NW=8;
constexpr int LDK=136, LDS=72, LDM=72;
constexpr int SMEM=107008;

template<typename L> struct isrow { static constexpr bool v=std::is_same<L,row_major>::value; };

template<int M,int N,int K,typename LA,typename LB>
__device__ __forceinline__ void gemm_out(const bf16* A,int lda,const bf16* B,int ldb,float* C,int ldc,int wid){
  constexpr int MT=M/16,NT=N/16,KT=K/16,TOT=MT*NT;
  for(int t=wid;t<TOT;t+=NW){
    int mi=t/NT,ni=t%NT;
    AccFrag c; wmma::fill_fragment(c,0.f);
    #pragma unroll
    for(int k=0;k<KT;k++){
      wmma::fragment<wmma::matrix_a,16,16,16,bf16,LA> af;
      wmma::fragment<wmma::matrix_b,16,16,16,bf16,LB> bfr;
      const bf16* ap=isrow<LA>::v?(A+mi*16*lda+k*16):(A+k*16*lda+mi*16);
      const bf16* bp=isrow<LB>::v?(B+k*16*ldb+ni*16):(B+ni*16*ldb+k*16);
      wmma::load_matrix_sync(af,ap,lda);
      wmma::load_matrix_sync(bfr,bp,ldb);
      wmma::mma_sync(c,af,bfr,c);
    }
    wmma::store_matrix_sync(C+mi*16*ldc+ni*16,c,ldc,wmma::mem_row_major);
  }
}

template<int M,int N,int K,typename LA,typename LB,int NLOC>
__device__ __forceinline__ void gemm_acc(const bf16* A,int lda,const bf16* B,int ldb,AccFrag* acc,int wid){
  constexpr int NT=N/16,KT=K/16;
  #pragma unroll
  for(int li=0;li<NLOC;li++){
    int t=wid+li*NW; int mi=t/NT,ni=t%NT;
    #pragma unroll
    for(int k=0;k<KT;k++){
      wmma::fragment<wmma::matrix_a,16,16,16,bf16,LA> af;
      wmma::fragment<wmma::matrix_b,16,16,16,bf16,LB> bfr;
      const bf16* ap=isrow<LA>::v?(A+mi*16*lda+k*16):(A+k*16*lda+mi*16);
      const bf16* bp=isrow<LB>::v?(B+k*16*ldb+ni*16):(B+ni*16*ldb+k*16);
      wmma::load_matrix_sync(af,ap,lda);
      wmma::load_matrix_sync(bfr,bp,ldb);
      wmma::mma_sync(acc[li],af,bfr,acc[li]);
    }
  }
}

template<int N,int NLOC>
__device__ __forceinline__ void store_acc(AccFrag* acc,float* C,int ldc,int wid){
  constexpr int NT=N/16;
  #pragma unroll
  for(int li=0;li<NLOC;li++){
    int t=wid+li*NW; int mi=t/NT,ni=t%NT;
    wmma::store_matrix_sync(C+mi*16*ldc+ni*16,acc[li],ldc,wmma::mem_row_major);
  }
}

__device__ __forceinline__ void load_tile(bf16* dst,int ld,const bf16* base,int row0,int rows,int S,int tid){
  constexpr int PPR=DIM/8;
  int cnt=rows*PPR;
  for(int i=tid;i<cnt;i+=THREADS){
    int r=i/PPR,c=i%PPR;
    int gr=row0+r; if(gr>=S) gr=S-1; if(gr<0) gr=0;
    int4 v=*(reinterpret_cast<const int4*>(base+(size_t)gr*DIM)+c);
    *reinterpret_cast<int4*>(dst+(size_t)r*ld+c*8)=v;
  }
}

__global__ void compute_D_kernel(const bf16* dO,const bf16* O,float* Dg,long total_rows){
  int row=blockIdx.x*(blockDim.x/32)+(threadIdx.x/32);
  int lane=threadIdx.x%32;
  if(row>=total_rows) return;
  const int4* d4=reinterpret_cast<const int4*>(dO+(size_t)row*DIM);
  const int4* o4=reinterpret_cast<const int4*>(O +(size_t)row*DIM);
  float sum=0.f;
  for(int e=lane;e<DIM/8;e+=32){
    int4 dv=d4[e],ov=o4[e];
    const bf16* db=reinterpret_cast<const bf16*>(&dv);
    const bf16* ob=reinterpret_cast<const bf16*>(&ov);
    #pragma unroll
    for(int k=0;k<8;k++) sum+=__bfloat162float(db[k])*__bfloat162float(ob[k]);
  }
  #pragma unroll
  for(int o=16;o>0;o>>=1) sum+=__shfl_down_sync(0xffffffff,sum,o);
  if(lane==0) Dg[row]=sum;
}

// ---------------- dK/dV (parallel over key blocks) ----------------
__global__ __launch_bounds__(THREADS,2) void kernel_dkdv(
    const bf16* Q,const bf16* K,const bf16* V,const bf16* dO,
    const float* L,const float* Dg, bf16* dKout,bf16* dVout,int S,float scale){
  extern __shared__ char smem[];
  bf16* Ksh=(bf16*)(smem+0);
  bf16* Vsh=(bf16*)(smem+17408);
  bf16* Qsh=(bf16*)(smem+34816);
  bf16* dOsh=(bf16*)(smem+52224);
  float* Sf=(float*)(smem+69632);
  bf16* PT=(bf16*)(smem+88064);
  bf16* dST=(bf16*)(smem+97280);
  float* Lsh=(float*)(smem+106496);
  float* Dsh=(float*)(smem+106752);
  float* stage=(float*)smem;

  int tid=threadIdx.x, wid=tid/32;
  int bh=blockIdx.y, kb=blockIdx.x;
  int k0=kb*BN;
  if(k0>=S) return;

  const bf16* Kb=K+(size_t)bh*S*DIM;
  const bf16* Vb=V+(size_t)bh*S*DIM;
  const bf16* Qb=Q+(size_t)bh*S*DIM;
  const bf16* dOb=dO+(size_t)bh*S*DIM;
  const float* Lb=L+(size_t)bh*S;
  const float* Db=Dg+(size_t)bh*S;
  bf16* dKb=dKout+(size_t)bh*S*DIM;
  bf16* dVb=dVout+(size_t)bh*S*DIM;

  load_tile(Ksh,LDK,Kb,k0,BN,S,tid);
  load_tile(Vsh,LDK,Vb,k0,BN,S,tid);

  AccFrag dVacc[4],dKacc[4];
  #pragma unroll
  for(int i=0;i<4;i++){ wmma::fill_fragment(dVacc[i],0.f); wmma::fill_fragment(dKacc[i],0.f); }
  __syncthreads();

  int numQ=(S+BM-1)/BM;
  for(int qb=kb; qb<numQ; qb++){
    int q0=qb*BM;
    load_tile(Qsh,LDK,Qb,q0,BM,S,tid);
    load_tile(dOsh,LDK,dOb,q0,BM,S,tid);
    for(int i=tid;i<BM;i+=THREADS){ int gi=q0+i; Lsh[i]=(gi<S)?Lb[gi]:0.f; Dsh[i]=(gi<S)?Db[gi]:0.f; }
    __syncthreads();

    gemm_out<BN,BM,DIM,row_major,col_major>(Ksh,LDK,Qsh,LDK,Sf,LDS,wid);
    __syncthreads();

    bool diag=(qb==kb);
    for(int idx=tid; idx<BN*BM; idx+=THREADS){
      int j=idx/BM, i=idx%BM;
      int gj=k0+j, gi=q0+i;
      bool valid=(gj<S)&&(gi<S)&&(!diag||gi>=gj);
      float pv=valid?__expf(scale*Sf[j*LDS+i]-Lsh[i]):0.f;
      PT[j*LDM+i]=__float2bfloat16(pv);
    }
    __syncthreads();

    gemm_out<BN,BM,DIM,row_major,col_major>(Vsh,LDK,dOsh,LDK,Sf,LDS,wid);
    __syncthreads();

    for(int idx=tid; idx<BN*BM; idx+=THREADS){
      int j=idx/BM, i=idx%BM;
      float p=__bfloat162float(PT[j*LDM+i]);
      float ds=scale*p*(Sf[j*LDS+i]-Dsh[i]);
      dST[j*LDM+i]=__float2bfloat16(ds);
    }
    __syncthreads();

    gemm_acc<BN,DIM,BM,row_major,row_major,4>(PT,LDM,dOsh,LDK,dVacc,wid);
    gemm_acc<BN,DIM,BM,row_major,row_major,4>(dST,LDM,Qsh,LDK,dKacc,wid);
    __syncthreads();
  }

  store_acc<DIM,4>(dVacc,stage,128,wid);
  __syncthreads();
  for(int idx=tid; idx<BN*DIM; idx+=THREADS){ int r=idx/DIM,e=idx%DIM;int gr=k0+r; if(gr<S) dVb[(size_t)gr*DIM+e]=__float2bfloat16(stage[r*128+e]); }
  __syncthreads();
  store_acc<DIM,4>(dKacc,stage,128,wid);
  __syncthreads();
  for(int idx=tid; idx<BN*DIM; idx+=THREADS){ int r=idx/DIM,e=idx%DIM;int gr=k0+r; if(gr<S) dKb[(size_t)gr*DIM+e]=__float2bfloat16(stage[r*128+e]); }
}

// ---------------- dQ (parallel over query blocks) ----------------
__global__ __launch_bounds__(THREADS,2) void kernel_dq(
    const bf16* Q,const bf16* K,const bf16* V,const bf16* dO,
    const float* L,const float* Dg, bf16* dQout,int S,float scale){
  extern __shared__ char smem[];
  bf16* Qsh=(bf16*)(smem+0);
  bf16* dOsh=(bf16*)(smem+17408);
  bf16* Ksh=(bf16*)(smem+34816);
  bf16* Vsh=(bf16*)(smem+52224);
  float* Sf=(float*)(smem+69632);
  bf16* Pbf=(bf16*)(smem+88064);
  bf16* dST=(bf16*)(smem+97280);
  float* Lsh=(float*)(smem+106496);
  float* Dsh=(float*)(smem+106752);
  float* stage=(float*)smem;

  int tid=threadIdx.x, wid=tid/32;
  int bh=blockIdx.y, qb=blockIdx.x;
  int q0=qb*BM;
  if(q0>=S) return;

  const bf16* Kb=K+(size_t)bh*S*DIM;
  const bf16* Vb=V+(size_t)bh*S*DIM;
  const bf16* Qb=Q+(size_t)bh*S*DIM;
  const bf16* dOb=dO+(size_t)bh*S*DIM;
  const float* Lb=L+(size_t)bh*S;
  const float* Db=Dg+(size_t)bh*S;
  bf16* dQb=dQout+(size_t)bh*S*DIM;

  load_tile(Qsh,LDK,Qb,q0,BM,S,tid);
  load_tile(dOsh,LDK,dOb,q0,BM,S,tid);
  for(int i=tid;i<BM;i+=THREADS){ int gi=q0+i; Lsh[i]=(gi<S)?Lb[gi]:0.f; Dsh[i]=(gi<S)?Db[gi]:0.f; }

  AccFrag dQacc[4];
  #pragma unroll
  for(int i=0;i<4;i++) wmma::fill_fragment(dQacc[i],0.f);
  __syncthreads();

  for(int kb=0; kb<=qb; kb++){
    int k0=kb*BN;
    load_tile(Ksh,LDK,Kb,k0,BN,S,tid);
    load_tile(Vsh,LDK,Vb,k0,BN,S,tid);
    __syncthreads();

    gemm_out<BN,BM,DIM,row_major,col_major>(Ksh,LDK,Qsh,LDK,Sf,LDS,wid);
    __syncthreads();

    bool diag=(kb==qb);
    for(int idx=tid; idx<BN*BM; idx+=THREADS){
      int j=idx/BM, i=idx%BM;
      int gj=k0+j, gi=q0+i;
      bool valid=(gj<S)&&(gi<S)&&(!diag||gi>=gj);
      float pv=valid?__expf(scale*Sf[j*LDS+i]-Lsh[i]):0.f;
      Pbf[j*LDM+i]=__float2bfloat16(pv);
    }
    __syncthreads();

    gemm_out<BN,BM,DIM,row_major,col_major>(Vsh,LDK,dOsh,LDK,Sf,LDS,wid);
    __syncthreads();

    for(int idx=tid; idx<BN*BM; idx+=THREADS){
      int j=idx/BM, i=idx%BM;
      float p=__bfloat162float(Pbf[j*LDM+i]);
      float ds=scale*p*(Sf[j*LDS+i]-Dsh[i]);
      dST[j*LDM+i]=__float2bfloat16(ds);
    }
    __syncthreads();

    gemm_acc<BM,DIM,BN,col_major,row_major,4>(dST,LDM,Ksh,LDK,dQacc,wid);
    __syncthreads();
  }

  store_acc<DIM,4>(dQacc,stage,128,wid);
  __syncthreads();
  for(int idx=tid; idx<BM*DIM; idx+=THREADS){ int r=idx/DIM,e=idx%DIM;int gr=q0+r; if(gr<S) dQb[(size_t)gr*DIM+e]=__float2bfloat16(stage[r*128+e]); }
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
  int rpb=256/32;
  dim3 gridD((total_rows+rpb-1)/rpb);
  compute_D_kernel<<<gridD,256,0,stream>>>(dOp,Op,s_D,total_rows);
  CUDA_CHECK(cudaGetLastError());

  if(!s_attr){
    CUDA_CHECK(cudaFuncSetAttribute((const void*)kernel_dkdv, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    CUDA_CHECK(cudaFuncSetAttribute((const void*)kernel_dq, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    s_attr=true;
  }

  float scale=1.0f/sqrtf((float)DIM);
  int numKB=(int)((S+BN-1)/BN);
  int numQB=(int)((S+BM-1)/BM);

  dim3 gridKV(numKB,(unsigned)BH);
  kernel_dkdv<<<gridKV,THREADS,SMEM,stream>>>(Qp,Kp,Vp,dOp,Lp,s_D,dKp,dVp,(int)S,scale);
  CUDA_CHECK(cudaGetLastError());

  dim3 gridQ(numQB,(unsigned)BH);
  kernel_dq<<<gridQ,THREADS,SMEM,stream>>>(Qp,Kp,Vp,dOp,Lp,s_D,dQp,(int)S,scale);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_cuda::run);

}  // namespace mha_bwd_cuda