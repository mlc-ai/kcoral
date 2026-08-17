#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <math.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { \
    cudaError_t _e=(call); \
    if(_e!=cudaSuccess){fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__);exit(1);} \
}while(0)

namespace mha_bwd {

using bf16 = __nv_bfloat16;
__device__ __forceinline__ float b2f(bf16 x){ return __bfloat162float(x); }

constexpr int D = 128;
constexpr int TILE = 64;   // BN=BM=BQ=BK
constexpr int NT = 256;

// Delta_i = sum_k dO_i[k]*O_i[k]
__global__ void delta_kernel(const bf16* __restrict__ O, const bf16* __restrict__ dO,
                             float* __restrict__ Delta, long rows){
  long warp = ((long)blockIdx.x*blockDim.x + threadIdx.x)/32;
  int lane = threadIdx.x % 32;
  if (warp >= rows) return;
  const bf16* o = O + warp*D;
  const bf16* g = dO + warp*D;
  float acc=0.f;
  #pragma unroll
  for (int k=lane;k<D;k+=32) acc += b2f(o[k])*b2f(g[k]);
  #pragma unroll
  for (int off=16; off>0; off>>=1) acc += __shfl_down_sync(0xffffffffu, acc, off);
  if (lane==0) Delta[warp]=acc;
}

// Computes dK, dV : outer over KV blocks, inner over Q tiles
__global__ void dkdv_kernel(
    const bf16* __restrict__ Q, const bf16* __restrict__ K, const bf16* __restrict__ V,
    const bf16* __restrict__ dO, const float* __restrict__ L, const float* __restrict__ Delta,
    bf16* __restrict__ dK, bf16* __restrict__ dV, int S, float scale)
{
  const int BN = TILE, BM = TILE;
  int bh = blockIdx.y;
  int kv0 = blockIdx.x * BN;
  if (kv0 >= S) return;

  const bf16* Qb = Q + (size_t)bh*S*D;
  const bf16* Kb = K + (size_t)bh*S*D;
  const bf16* Vb = V + (size_t)bh*S*D;
  const bf16* dOb= dO+ (size_t)bh*S*D;
  const float* Lb = L + (size_t)bh*S;
  const float* Db = Delta + (size_t)bh*S;
  bf16* dKb = dK + (size_t)bh*S*D;
  bf16* dVb = dV + (size_t)bh*S*D;

  extern __shared__ char smem[];
  bf16* sK = (bf16*)smem;
  bf16* sV = sK + BN*D;
  bf16* sQ = sV + BN*D;
  bf16* sdO= sQ + BM*D;
  float* sP = (float*)(sdO + BM*D);
  float* sdS= sP + BN*BM;
  float* sL = sdS + BN*BM;
  float* sD = sL + BM;

  int tid = threadIdx.x;

  for (int idx=tid; idx<BN*D; idx+=NT){
     int n=idx/D, k=idx%D; int kj=kv0+n;
     if (kj<S){ sK[idx]=Kb[(size_t)kj*D+k]; sV[idx]=Vb[(size_t)kj*D+k]; }
     else { sK[idx]=__float2bfloat16(0.f); sV[idx]=__float2bfloat16(0.f);}
  }

  int nn_grp=tid/16, kk_grp=tid%16;
  int n0=nn_grp*4, k0=kk_grp*8;
  float dKacc[4][8], dVacc[4][8];
  #pragma unroll
  for(int i=0;i<4;i++)
   #pragma unroll
   for(int j=0;j<8;j++){dKacc[i][j]=0.f;dVacc[i][j]=0.f;}

  int qstart = (kv0/BM)*BM;
  __syncthreads();

  for (int q0=qstart; q0<S; q0+=BM){
     for (int idx=tid; idx<BM*D; idx+=NT){
        int m=idx/D, k=idx%D; int qi=q0+m;
        if (qi<S){ sQ[idx]=Qb[(size_t)qi*D+k]; sdO[idx]=dOb[(size_t)qi*D+k]; }
        else { sQ[idx]=__float2bfloat16(0.f); sdO[idx]=__float2bfloat16(0.f);}
     }
     for (int m=tid;m<BM;m+=NT){ int qi=q0+m; sL[m]=(qi<S)?Lb[qi]:0.f; sD[m]=(qi<S)?Db[qi]:0.f; }
     __syncthreads();

     // P^T = exp(scale*K.Q - L), causal mask
     for (int idx=tid; idx<BN*BM; idx+=NT){
        int n=idx/BM, m=idx%BM;
        float raw=0.f; const bf16* kr=&sK[n*D]; const bf16* qr=&sQ[m*D];
        #pragma unroll 8
        for(int k=0;k<D;k++) raw += b2f(kr[k])*b2f(qr[k]);
        int kj=kv0+n, qi=q0+m;
        sP[idx] = (kj<S && qi<S && kj<=qi) ? __expf(scale*raw - sL[m]) : 0.f;
     }
     __syncthreads();

     // dV += P^T @ dO
     for (int m=0;m<BM;m++){
        float os[8], ps[4];
        #pragma unroll
        for(int kk=0;kk<8;kk++) os[kk]=b2f(sdO[m*D + k0+kk]);
        #pragma unroll
        for(int nn=0;nn<4;nn++) ps[nn]=sP[(n0+nn)*BM + m];
        #pragma unroll
        for(int nn=0;nn<4;nn++)
         #pragma unroll
         for(int kk=0;kk<8;kk++) dVacc[nn][kk]+=ps[nn]*os[kk];
     }

     // dS^T = P^T*(V.dO - D)
     for (int idx=tid; idx<BN*BM; idx+=NT){
        int n=idx/BM, m=idx%BM;
        float dp=0.f; const bf16* vr=&sV[n*D]; const bf16* orr=&sdO[m*D];
        #pragma unroll 8
        for(int k=0;k<D;k++) dp += b2f(vr[k])*b2f(orr[k]);
        sdS[idx]=sP[idx]*(dp - sD[m]);
     }
     __syncthreads();

     // dK += dS^T @ Q
     for (int m=0;m<BM;m++){
        float qs[8], ss[4];
        #pragma unroll
        for(int kk=0;kk<8;kk++) qs[kk]=b2f(sQ[m*D + k0+kk]);
        #pragma unroll
        for(int nn=0;nn<4;nn++) ss[nn]=sdS[(n0+nn)*BM + m];
        #pragma unroll
        for(int nn=0;nn<4;nn++)
         #pragma unroll
         for(int kk=0;kk<8;kk++) dKacc[nn][kk]+=ss[nn]*qs[kk];
     }
     __syncthreads();
  }

  #pragma unroll
  for(int nn=0;nn<4;nn++){
     int kj=kv0+n0+nn;
     if (kj<S){
        #pragma unroll
        for(int kk=0;kk<8;kk++){
           int k=k0+kk;
           dKb[(size_t)kj*D+k]=__float2bfloat16(scale*dKacc[nn][kk]);
           dVb[(size_t)kj*D+k]=__float2bfloat16(dVacc[nn][kk]);
        }
     }
  }
}

// Computes dQ : outer over Q blocks, inner over KV tiles
__global__ void dq_kernel(
    const bf16* __restrict__ Q, const bf16* __restrict__ K, const bf16* __restrict__ V,
    const bf16* __restrict__ dO, const float* __restrict__ L, const float* __restrict__ Delta,
    bf16* __restrict__ dQ, int S, float scale)
{
  const int BQ = TILE, BK = TILE;
  int bh = blockIdx.y;
  int q0b = blockIdx.x * BQ;
  if (q0b >= S) return;

  const bf16* Qb = Q + (size_t)bh*S*D;
  const bf16* Kb = K + (size_t)bh*S*D;
  const bf16* Vb = V + (size_t)bh*S*D;
  const bf16* dOb= dO+ (size_t)bh*S*D;
  const float* Lb = L + (size_t)bh*S;
  const float* Db = Delta + (size_t)bh*S;
  bf16* dQb = dQ + (size_t)bh*S*D;

  extern __shared__ char smem[];
  bf16* sQ = (bf16*)smem;
  bf16* sdO= sQ + BQ*D;
  bf16* sK = sdO + BQ*D;
  bf16* sV = sK + BK*D;
  float* sS = (float*)(sV + BK*D);
  float* sdS= sS + BQ*BK;
  float* sL = sdS + BQ*BK;
  float* sD = sL + BQ;

  int tid=threadIdx.x;

  for (int idx=tid; idx<BQ*D; idx+=NT){
     int m=idx/D, k=idx%D; int qi=q0b+m;
     if (qi<S){ sQ[idx]=Qb[(size_t)qi*D+k]; sdO[idx]=dOb[(size_t)qi*D+k]; }
     else { sQ[idx]=__float2bfloat16(0.f); sdO[idx]=__float2bfloat16(0.f);}
  }
  for (int m=tid;m<BQ;m+=NT){ int qi=q0b+m; sL[m]=(qi<S)?Lb[qi]:0.f; sD[m]=(qi<S)?Db[qi]:0.f; }

  int mm_grp=tid/16, kk_grp=tid%16;
  int m0=mm_grp*4, k0=kk_grp*8;
  float dQacc[4][8];
  #pragma unroll
  for(int i=0;i<4;i++)
   #pragma unroll
   for(int j=0;j<8;j++) dQacc[i][j]=0.f;

  int qend = q0b + BQ - 1;
  __syncthreads();

  for (int k0k=0; k0k<S && k0k<=qend; k0k+=BK){
     for (int idx=tid; idx<BK*D; idx+=NT){
        int n=idx/D, k=idx%D; int kj=k0k+n;
        if (kj<S){ sK[idx]=Kb[(size_t)kj*D+k]; sV[idx]=Vb[(size_t)kj*D+k]; }
        else { sK[idx]=__float2bfloat16(0.f); sV[idx]=__float2bfloat16(0.f);}
     }
     __syncthreads();

     // P[m][n] = exp(scale*Q.K - L), causal
     for (int idx=tid; idx<BQ*BK; idx+=NT){
        int m=idx/BK, n=idx%BK;
        float raw=0.f; const bf16* qr=&sQ[m*D]; const bf16* kr=&sK[n*D];
        #pragma unroll 8
        for(int k=0;k<D;k++) raw += b2f(qr[k])*b2f(kr[k]);
        int qi=q0b+m, kj=k0k+n;
        sS[idx] = (kj<S && qi<S && kj<=qi) ? __expf(scale*raw - sL[m]) : 0.f;
     }
     __syncthreads();

     // dS[m][n] = P*(dO.V - D)
     for (int idx=tid; idx<BQ*BK; idx+=NT){
        int m=idx/BK, n=idx%BK;
        float dp=0.f; const bf16* orr=&sdO[m*D]; const bf16* vr=&sV[n*D];
        #pragma unroll 8
        for(int k=0;k<D;k++) dp += b2f(orr[k])*b2f(vr[k]);
        sdS[idx]=sS[idx]*(dp - sD[m]);
     }
     __syncthreads();

     // dQ += dS @ K
     for (int n=0;n<BK;n++){
        float ks[8], ds[4];
        #pragma unroll
        for(int kk=0;kk<8;kk++) ks[kk]=b2f(sK[n*D + k0+kk]);
        #pragma unroll
        for(int mm=0;mm<4;mm++) ds[mm]=sdS[(m0+mm)*BK + n];
        #pragma unroll
        for(int mm=0;mm<4;mm++)
         #pragma unroll
         for(int kk=0;kk<8;kk++) dQacc[mm][kk]+=ds[mm]*ks[kk];
     }
     __syncthreads();
  }

  #pragma unroll
  for(int mm=0;mm<4;mm++){
     int qi=q0b+m0+mm;
     if (qi<S){
        #pragma unroll
        for(int kk=0;kk<8;kk++){
           int k=k0+kk;
           dQb[(size_t)qi*D+k]=__float2bfloat16(scale*dQacc[mm][kk]);
        }
     }
  }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV)
{
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int64_t B = Q.size(0), H = Q.size(1), S = Q.size(2);
  int64_t BH = B*H;
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

  const bf16* Qp=(const bf16*)Q.data_ptr();
  const bf16* Kp=(const bf16*)K.data_ptr();
  const bf16* Vp=(const bf16*)V.data_ptr();
  const bf16* Op=(const bf16*)O.data_ptr();
  const bf16* dOp=(const bf16*)dO.data_ptr();
  const float* Lp=(const float*)L.data_ptr();
  bf16* dQp=(bf16*)dQ.data_ptr();
  bf16* dKp=(bf16*)dK.data_ptr();
  bf16* dVp=(bf16*)dV.data_ptr();

  float scale = 1.0f/sqrtf((float)D);

  float* Delta=nullptr;
  CUDA_CHECK(cudaMalloc(&Delta, sizeof(float)*(size_t)BH*S));

  long rows = (long)BH*S;
  long dblocks = (rows*32 + NT - 1)/NT; // NT/32 rows per block
  delta_kernel<<<(unsigned)dblocks, NT, 0, stream>>>(Op, dOp, Delta, rows);
  CUDA_CHECK(cudaGetLastError());

  size_t smem_bytes = (size_t)(2*TILE*D + 2*TILE*D)*sizeof(bf16)
                    + (size_t)(2*TILE*TILE + 2*TILE)*sizeof(float);

  CUDA_CHECK(cudaFuncSetAttribute(dkdv_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));
  CUDA_CHECK(cudaFuncSetAttribute(dq_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));

  dim3 grid((unsigned)((S+TILE-1)/TILE), (unsigned)BH);
  dkdv_kernel<<<grid, NT, smem_bytes, stream>>>(Qp,Kp,Vp,dOp,Lp,Delta,dKp,dVp,(int)S,scale);
  CUDA_CHECK(cudaGetLastError());
  dq_kernel<<<grid, NT, smem_bytes, stream>>>(Qp,Kp,Vp,dOp,Lp,Delta,dQp,(int)S,scale);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaFree(Delta));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd