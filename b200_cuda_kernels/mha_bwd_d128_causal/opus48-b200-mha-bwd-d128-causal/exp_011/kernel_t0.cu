#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <math.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
    fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__);exit(1);} } while(0)

namespace mha_bwd {
using bf16 = __nv_bfloat16;
using tvm::ffi::TensorView;

constexpr int HD = 128;
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int NT = 256;
constexpr int ACC = BN*HD/NT; // 32

__global__ void delta_kernel(const bf16* __restrict__ dO, const bf16* __restrict__ O,
                             float* __restrict__ Delta, int64_t total_rows) {
  int64_t row = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (row < total_rows) {
    const bf16* d = dO + row*HD;
    const bf16* o = O + row*HD;
    float acc = 0.f;
    #pragma unroll
    for (int e=0;e<HD;e++) acc += __bfloat162float(d[e]) * __bfloat162float(o[e]);
    Delta[row] = acc;
  }
}

__global__ void dkv_kernel(
    const bf16* __restrict__ Q, const bf16* __restrict__ K, const bf16* __restrict__ V,
    const bf16* __restrict__ dO, const float* __restrict__ L, const float* __restrict__ Delta,
    bf16* __restrict__ dK, bf16* __restrict__ dV, int S, float scale) {
  extern __shared__ char smem[];
  bf16* sQ  = (bf16*)smem;
  bf16* sdO = sQ + BM*HD;
  bf16* sK  = sdO + BM*HD;
  bf16* sV  = sK + BN*HD;
  float* sL = (float*)(sV + BN*HD);
  float* sD = sL + BM;
  float* sA = sD + BM;      // ST then PT
  float* sB = sA + BN*BM;   // dPT then dST

  int tid = threadIdx.x;
  int kb  = blockIdx.x;
  int bh  = blockIdx.y;
  int k0  = kb*BN;
  int64_t base  = (int64_t)bh * S * HD;
  int64_t Lbase = (int64_t)bh * S;
  const bf16* Qb=Q+base; const bf16* Kb=K+base; const bf16* Vb=V+base; const bf16* dOb=dO+base;
  bf16 zero = __float2bfloat16(0.f);

  for (int idx=tid; idx<BN*HD; idx+=NT) {
    int r=idx/HD, c=idx%HD; int gk=k0+r;
    sK[idx] = (gk<S)? Kb[(int64_t)gk*HD+c] : zero;
    sV[idx] = (gk<S)? Vb[(int64_t)gk*HD+c] : zero;
  }

  float dKacc[ACC]; float dVacc[ACC];
  #pragma unroll
  for (int c=0;c<ACC;c++){dKacc[c]=0.f;dVacc[c]=0.f;}

  int num_q = (S+BM-1)/BM;
  for (int qb=kb; qb<num_q; ++qb) {
    int q0 = qb*BM;
    __syncthreads();
    for (int idx=tid; idx<BM*HD; idx+=NT) {
      int r=idx/HD, c=idx%HD; int gq=q0+r;
      sQ[idx]  = (gq<S)? Qb[(int64_t)gq*HD+c]:zero;
      sdO[idx] = (gq<S)? dOb[(int64_t)gq*HD+c]:zero;
    }
    for (int i=tid;i<BM;i+=NT){ int gq=q0+i; sL[i]=(gq<S)?L[Lbase+gq]:0.f; sD[i]=(gq<S)?Delta[Lbase+gq]:0.f; }
    __syncthreads();

    for (int p=tid;p<BN*BM;p+=NT){
      int j=p/BM, i=p%BM;
      float st=0.f, dpt=0.f;
      #pragma unroll
      for(int e=0;e<HD;e++){
        st  += __bfloat162float(sK[j*HD+e])*__bfloat162float(sQ[i*HD+e]);
        dpt += __bfloat162float(sV[j*HD+e])*__bfloat162float(sdO[i*HD+e]);
      }
      sA[p]=st*scale; sB[p]=dpt;
    }
    __syncthreads();

    for (int p=tid;p<BN*BM;p+=NT){
      int j=p/BM, i=p%BM; int gq=q0+i, gk=k0+j;
      if (gq<S && gk<S && gq>=gk){
        float pt = __expf(sA[p]-sL[i]);
        float ds = pt*(sB[p]-sD[i]);
        sA[p]=pt; sB[p]=ds;
      } else { sA[p]=0.f; sB[p]=0.f; }
    }
    __syncthreads();

    #pragma unroll
    for (int c=0;c<ACC;c++){
      int p2=tid+c*NT; int j=p2/HD, e=p2%HD;
      float sv=0.f, sk=0.f;
      for(int i=0;i<BM;i++){
        sv += sA[j*BM+i]*__bfloat162float(sdO[i*HD+e]);
        sk += sB[j*BM+i]*__bfloat162float(sQ[i*HD+e]);
      }
      dVacc[c]+=sv; dKacc[c]+=sk;
    }
  }
  __syncthreads();
  #pragma unroll
  for (int c=0;c<ACC;c++){
    int p2=tid+c*NT; int j=p2/HD, e=p2%HD; int gk=k0+j;
    if (gk<S){
      dK[base+(int64_t)gk*HD+e]=__float2bfloat16(dKacc[c]*scale);
      dV[base+(int64_t)gk*HD+e]=__float2bfloat16(dVacc[c]);
    }
  }
}

__global__ void dq_kernel(
    const bf16* __restrict__ Q, const bf16* __restrict__ K, const bf16* __restrict__ V,
    const bf16* __restrict__ dO, const float* __restrict__ L, const float* __restrict__ Delta,
    bf16* __restrict__ dQ, int S, float scale) {
  extern __shared__ char smem[];
  bf16* sQ  = (bf16*)smem;
  bf16* sdO = sQ + BM*HD;
  bf16* sK  = sdO + BM*HD;
  bf16* sV  = sK + BN*HD;
  float* sL = (float*)(sV + BN*HD);
  float* sD = sL + BM;
  float* sA = sD + BM;      // S then dS
  float* sB = sA + BM*BN;   // dP

  int tid=threadIdx.x;
  int qb=blockIdx.x;
  int bh=blockIdx.y;
  int q0=qb*BM;
  int64_t base=(int64_t)bh*S*HD;
  int64_t Lbase=(int64_t)bh*S;
  const bf16* Qb=Q+base; const bf16* Kb=K+base; const bf16* Vb=V+base; const bf16* dOb=dO+base;
  bf16 zero=__float2bfloat16(0.f);

  for(int idx=tid;idx<BM*HD;idx+=NT){int r=idx/HD,c=idx%HD;int gq=q0+r;
    sQ[idx]=(gq<S)?Qb[(int64_t)gq*HD+c]:zero;
    sdO[idx]=(gq<S)?dOb[(int64_t)gq*HD+c]:zero;}
  for(int i=tid;i<BM;i+=NT){int gq=q0+i; sL[i]=(gq<S)?L[Lbase+gq]:0.f; sD[i]=(gq<S)?Delta[Lbase+gq]:0.f;}

  float dQacc[ACC];
  #pragma unroll
  for(int c=0;c<ACC;c++) dQacc[c]=0.f;

  for(int kb=0;kb<=qb;++kb){
    int k0=kb*BN;
    __syncthreads();
    for(int idx=tid;idx<BN*HD;idx+=NT){int r=idx/HD,c=idx%HD;int gk=k0+r;
      sK[idx]=(gk<S)?Kb[(int64_t)gk*HD+c]:zero;
      sV[idx]=(gk<S)?Vb[(int64_t)gk*HD+c]:zero;}
    __syncthreads();

    for(int p=tid;p<BM*BN;p+=NT){
      int i=p/BN, j=p%BN;
      float s=0.f, dp=0.f;
      #pragma unroll
      for(int e=0;e<HD;e++){
        s  += __bfloat162float(sQ[i*HD+e])*__bfloat162float(sK[j*HD+e]);
        dp += __bfloat162float(sdO[i*HD+e])*__bfloat162float(sV[j*HD+e]);
      }
      sA[p]=s*scale; sB[p]=dp;
    }
    __syncthreads();

    for(int p=tid;p<BM*BN;p+=NT){
      int i=p/BN, j=p%BN; int gq=q0+i, gk=k0+j;
      if(gq<S&&gk<S&&gq>=gk){
        float pt=__expf(sA[p]-sL[i]);
        sA[p]=pt*(sB[p]-sD[i]);
      } else sA[p]=0.f;
    }
    __syncthreads();

    #pragma unroll
    for(int c=0;c<ACC;c++){
      int p2=tid+c*NT; int i=p2/HD, e=p2%HD;
      float acc=0.f;
      for(int j=0;j<BN;j++) acc += sA[i*BN+j]*__bfloat162float(sK[j*HD+e]);
      dQacc[c]+=acc;
    }
  }
  __syncthreads();
  #pragma unroll
  for(int c=0;c<ACC;c++){
    int p2=tid+c*NT; int i=p2/HD, e=p2%HD; int gq=q0+i;
    if(gq<S) dQ[base+(int64_t)gq*HD+e]=__float2bfloat16(dQacc[c]*scale);
  }
}

void run(TensorView Q, TensorView K, TensorView V, TensorView O, TensorView dO, TensorView L,
         TensorView dQ, TensorView dK, TensorView dV) {
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int B=(int)Q.size(0), H=(int)Q.size(1), S=(int)Q.size(2);
  int64_t BH=(int64_t)B*H;

  const bf16* Qp=(const bf16*)Q.data_ptr();
  const bf16* Kp=(const bf16*)K.data_ptr();
  const bf16* Vp=(const bf16*)V.data_ptr();
  const bf16* Op=(const bf16*)O.data_ptr();
  const bf16* dOp=(const bf16*)dO.data_ptr();
  const float* Lp=(const float*)L.data_ptr();
  bf16* dQp=(bf16*)dQ.data_ptr();
  bf16* dKp=(bf16*)dK.data_ptr();
  bf16* dVp=(bf16*)dV.data_ptr();

  float scale=1.0f/sqrtf((float)HD);
  cudaStream_t stream=(cudaStream_t)TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id);

  float* Delta=nullptr;
  CUDA_CHECK(cudaMalloc(&Delta,(size_t)BH*S*sizeof(float)));

  int64_t total_rows=BH*S;
  int dthreads=256;
  int64_t dblocks=(total_rows+dthreads-1)/dthreads;
  delta_kernel<<<(unsigned)dblocks,dthreads,0,stream>>>(dOp,Op,Delta,total_rows);
  CUDA_CHECK(cudaGetLastError());

  int smem = 4*BM*HD*(int)sizeof(bf16) + 2*BM*(int)sizeof(float) + 2*BM*BN*(int)sizeof(float);
  CUDA_CHECK(cudaFuncSetAttribute(dkv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  CUDA_CHECK(cudaFuncSetAttribute(dq_kernel,  cudaFuncAttributeMaxDynamicSharedMemorySize, smem));

  int num_kv=(S+BN-1)/BN;
  int num_q =(S+BM-1)/BM;
  dim3 gkv((unsigned)num_kv, (unsigned)BH);
  dim3 gq ((unsigned)num_q,  (unsigned)BH);

  dkv_kernel<<<gkv,NT,smem,stream>>>(Qp,Kp,Vp,dOp,Lp,Delta,dKp,dVp,S,scale);
  CUDA_CHECK(cudaGetLastError());
  dq_kernel<<<gq,NT,smem,stream>>>(Qp,Kp,Vp,dOp,Lp,Delta,dQp,S,scale);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaFree(Delta));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd