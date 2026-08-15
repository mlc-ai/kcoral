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
using bf16 = __nv_bfloat16;
using AccFrag = wmma::fragment<wmma::accumulator,16,16,16,float>;

constexpr int BN = 64;
constexpr int BM = 64;
constexpr int DIM = 128;
constexpr int THREADS = 512;
constexpr int NW = 16;
constexpr int LDK = 136;   // padded ld for [.][DIM]
constexpr int LDT = 72;    // padded ld for [.][64]

// offsets in bytes
constexpr int O_K=0, O_V=17408, O_Q=34816, O_dO=52224, O_QT=69632, O_dOT=88064;
constexpr int O_ST=106496, O_dP=124928, O_P=143360, O_dS=152576, O_dST=161792;
constexpr int O_L=171008, O_D=171264, SMEM=171520;

template<typename Lay>
__device__ __forceinline__ const bf16* apt(const bf16* A,int ld,int mi,int k){
  return A + (mi*16)*ld + (k*16);
}
template<typename Lay>
__device__ __forceinline__ const bf16* bpt(const bf16* B,int ld,int k,int ni){
  return B + (k*16)*ld + (ni*16);
}

template<int M,int N,int K>
__device__ __forceinline__ void gemm_w(const bf16* A,int lda,const bf16* B,int ldb,
                                        float* C,int ldc,int warp_id){
  constexpr int MT=M/16,NT=N/16,KT=K/16; constexpr int TOT=MT*NT;
  for(int t=warp_id;t<TOT;t+=NW){
    int mi=t/NT,ni=t%NT;
    AccFrag c; wmma::fill_fragment(c,0.f);
    #pragma unroll
    for(int k=0;k<KT;k++){
      wmma::fragment<wmma::matrix_a,16,16,16,bf16,row_major> af;
      wmma::fragment<wmma::matrix_b,16,16,16,bf16,row_major> bfr;
      wmma::load_matrix_sync(af, A+(mi*16)*lda+(k*16), lda);
      wmma::load_matrix_sync(bfr, B+(k*16)*ldb+(ni*16), ldb);
      wmma::mma_sync(c,af,bfr,c);
    }
    wmma::store_matrix_sync(C+mi*16*ldc+ni*16, c, ldc, wmma::mem_row_major);
  }
}

template<int M,int N,int K,int NLOC>
__device__ __forceinline__ void gemm_acc(const bf16* A,int lda,const bf16* B,int ldb,
                                          AccFrag* acc,int warp_id){
  constexpr int NT=N/16,KT=K/16;
  #pragma unroll
  for(int li=0;li<NLOC;li++){
    int t=warp_id+li*NW; int mi=t/NT, ni=t%NT;
    #pragma unroll
    for(int k=0;k<KT;k++){
      wmma::fragment<wmma::matrix_a,16,16,16,bf16,row_major> af;
      wmma::fragment<wmma::matrix_b,16,16,16,bf16,row_major> bfr;
      wmma::load_matrix_sync(af, A+(mi*16)*lda+(k*16), lda);
      wmma::load_matrix_sync(bfr, B+(k*16)*ldb+(ni*16), ldb);
      wmma::mma_sync(acc[li],af,bfr,acc[li]);
    }
  }
}

template<int N,int NLOC>
__device__ __forceinline__ void store_acc(AccFrag* acc,float* C,int ldc,int warp_id){
  constexpr int NT=N/16;
  #pragma unroll
  for(int li=0;li<NLOC;li++){
    int t=warp_id+li*NW; int mi=t/NT, ni=t%NT;
    wmma::store_matrix_sync(C+mi*16*ldc+ni*16, acc[li], ldc, wmma::mem_row_major);
  }
}

__device__ __forceinline__ void load_pad(bf16* dst,int ldd,const bf16* base,int row0,int rows,int S,int tid){
  constexpr int PPR=DIM/8;
  int cnt=rows*PPR;
  for(int i=tid;i<cnt;i+=THREADS){
    int r=i/PPR, c=i%PPR;
    int gr=row0+r;
    int4 v;
    if(gr<S) v=*(reinterpret_cast<const int4*>(base+(size_t)gr*DIM)+c);
    else v=make_int4(0,0,0,0);
    *reinterpret_cast<int4*>(dst + r*ldd + c*8) = v;
  }
}

__global__ void compute_D_kernel(const bf16* dO, const bf16* O, float* Dg, long total_rows){
  int row = blockIdx.x*(blockDim.x/32) + (threadIdx.x/32);
  int lane = threadIdx.x%32;
  if(row >= total_rows) return;
  const int4* d4=reinterpret_cast<const int4*>(dO + (size_t)row*DIM);
  const int4* o4=reinterpret_cast<const int4*>(O  + (size_t)row*DIM);
  float sum=0.f;
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

__global__ __launch_bounds__(THREADS) void kernel_bwd(
    const bf16* Q, const bf16* K, const bf16* V, const bf16* dO,
    const float* L, const float* Dg,
    float* dQscr, bf16* dKout, bf16* dVout, int S, float scale){

  extern __shared__ char smem[];
  bf16* Ksh=(bf16*)(smem+O_K);
  bf16* Vsh=(bf16*)(smem+O_V);
  bf16* Qsh=(bf16*)(smem+O_Q);
  bf16* dOsh=(bf16*)(smem+O_dO);
  bf16* QTsh=(bf16*)(smem+O_QT);
  bf16* dOTsh=(bf16*)(smem+O_dOT);
  float* STf=(float*)(smem+O_ST);
  float* dPf=(float*)(smem+O_dP);
  bf16* Pbf=(bf16*)(smem+O_P);
  bf16* dSbf=(bf16*)(smem+O_dS);
  bf16* dSTbf=(bf16*)(smem+O_dST);
  float* Lsh=(float*)(smem+O_L);
  float* Dsh=(float*)(smem+O_D);
  float* dQtile=(float*)(smem+O_ST);   // overlays STf+dPf, ld=LDK
  float* stage=(float*)(smem+O_ST);    // post-loop staging, ld=128

  int tid=threadIdx.x, warp_id=tid/32;
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

  load_pad(Ksh,LDK,Kb,k0,BN,S,tid);
  load_pad(Vsh,LDK,Vb,k0,BN,S,tid);

  AccFrag dVacc[2], dKacc[2];
  #pragma unroll
  for(int i=0;i<2;i++){ wmma::fill_fragment(dVacc[i],0.f); wmma::fill_fragment(dKacc[i],0.f); }

  int numQ=(S+BM-1)/BM;
  __syncthreads();

  for(int qb=kb; qb<numQ; qb++){
    int q0=qb*BM;
    load_pad(Qsh,LDK,Qb,q0,BM,S,tid);
    load_pad(dOsh,LDK,dOb,q0,BM,S,tid);
    for(int i=tid;i<BM;i+=THREADS){ int gi=q0+i; Lsh[i]=(gi<S)?Lb[gi]:0.f; Dsh[i]=(gi<S)?Db[gi]:0.f; }
    __syncthreads();

    // transpose Q, dO
    for(int i=tid;i<BM*DIM;i+=THREADS){
      int m=i/DIM, d=i%DIM;
      QTsh[d*LDT+m]=Qsh[m*LDK+d];
      dOTsh[d*LDT+m]=dOsh[m*LDK+d];
    }
    __syncthreads();

    gemm_w<BN,BM,DIM>(Ksh,LDK,QTsh,LDT,STf,LDT,warp_id);
    gemm_w<BN,BM,DIM>(Vsh,LDK,dOTsh,LDT,dPf,LDT,warp_id);
    __syncthreads();

    bool diag=(qb==kb);
    for(int t=tid;t<BN*BM;t+=THREADS){
      int j=t>>6, i=t&63;
      int pos=j*LDT+i;
      int gj=k0+j, gi=q0+i;
      bool valid=(gj<S)&&(gi<S)&&(!diag||gi>=gj);
      float pv = valid?__expf(scale*STf[pos]-Lsh[i]):0.f;
      float ds = valid?scale*pv*(dPf[pos]-Dsh[i]):0.f;
      Pbf[pos]=__float2bfloat16(pv);
      dSbf[pos]=__float2bfloat16(ds);
      dSTbf[i*LDT+j]=__float2bfloat16(ds);
    }
    __syncthreads();

    gemm_acc<BN,DIM,BM,2>(Pbf,LDT,dOsh,LDK,dVacc,warp_id);
    gemm_acc<BN,DIM,BM,2>(dSbf,LDT,Qsh,LDK,dKacc,warp_id);
    gemm_w<BM,DIM,BN>(dSTbf,LDT,Ksh,LDK,dQtile,LDK,warp_id);
    __syncthreads();

    for(int t=tid;t<BM*DIM;t+=THREADS){
      int r=t/DIM, e=t%DIM;
      int gi=q0+r;
      if(gi<S) atomicAdd(&dQscr[((size_t)bh*S+gi)*DIM+e], dQtile[r*LDK+e]);
    }
    __syncthreads();
  }

  store_acc<DIM,2>(dVacc, stage, 128, warp_id);
  __syncthreads();
  for(int t=tid;t<BN*DIM;t+=THREADS){ int r=t/DIM,e=t%DIM;int gr=k0+r; if(gr<S) dVb[(size_t)gr*DIM+e]=__float2bfloat16(stage[r*128+e]); }
  __syncthreads();
  store_acc<DIM,2>(dKacc, stage, 128, warp_id);
  __syncthreads();
  for(int t=tid;t<BN*DIM;t+=THREADS){ int r=t/DIM,e=t%DIM;int gr=k0+r; if(gr<S) dKb[(size_t)gr*DIM+e]=__float2bfloat16(stage[r*128+e]); }
}

__global__ void convert_dQ_kernel(const float* scr, bf16* dQ, size_t total){
  size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
  if(i<total) dQ[i]=__float2bfloat16(scr[i]);
}

static float* s_D=nullptr; static size_t s_Dsz=0;
static float* s_dQ=nullptr; static size_t s_dQsz=0;
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
  size_t dQbytes=(size_t)BH*S*DIM*sizeof(float);
  if(s_Dsz<Dbytes){ if(s_D) cudaFree(s_D); CUDA_CHECK(cudaMalloc(&s_D,Dbytes)); s_Dsz=Dbytes; }
  if(s_dQsz<dQbytes){ if(s_dQ) cudaFree(s_dQ); CUDA_CHECK(cudaMalloc(&s_dQ,dQbytes)); s_dQsz=dQbytes; }

  CUDA_CHECK(cudaMemsetAsync(s_dQ,0,dQbytes,stream));

  long total_rows=(long)BH*S;
  int rpb=256/32;
  dim3 gridD((total_rows+rpb-1)/rpb);
  compute_D_kernel<<<gridD,256,0,stream>>>(dOp,Op,s_D,total_rows);
  CUDA_CHECK(cudaGetLastError());

  if(!s_attr){
    CUDA_CHECK(cudaFuncSetAttribute((const void*)kernel_bwd, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    s_attr=true;
  }

  float scale=1.0f/sqrtf((float)DIM);
  int numKB=(int)((S+BN-1)/BN);
  dim3 grid(numKB,(unsigned)BH);
  kernel_bwd<<<grid,THREADS,SMEM,stream>>>(Qp,Kp,Vp,dOp,Lp,s_D,s_dQ,dKp,dVp,(int)S,scale);
  CUDA_CHECK(cudaGetLastError());

  size_t total=(size_t)BH*S*DIM;
  int ct=256;
  convert_dQ_kernel<<<(unsigned)((total+ct-1)/ct),ct,0,stream>>>(s_dQ,dQp,total);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_cuda::run);

}  // namespace mha_bwd_cuda