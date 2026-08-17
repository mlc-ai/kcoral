#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__); exit(1);} } while(0)

namespace mha_bwd {

constexpr int D = 128;
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int THREADS = 256;

using tvm::ffi::TensorView;

// Delta[row] = sum_c O[row][c] * dO[row][c]
__global__ void delta_kernel(const __nv_bfloat16* __restrict__ O,
                             const __nv_bfloat16* __restrict__ dOg,
                             float* __restrict__ Delta,
                             int64_t total_rows){
  int64_t row = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
  if(row >= total_rows) return;
  const __nv_bfloat16* o = O + row*D;
  const __nv_bfloat16* g = dOg + row*D;
  float s=0.f;
  #pragma unroll
  for(int c=0;c<D;c++){
    s += __bfloat162float(o[c]) * __bfloat162float(g[c]);
  }
  Delta[row]=s;
}

// load [64][D] tile from global (row_start..+64) into smem, zero-pad rows >= S
__device__ __forceinline__ void loadTile(__nv_bfloat16* smem, const __nv_bfloat16* gptr,
                                         int row_start, int S, int tid){
  const int ROWS = 64;
  const int VPR = D*2/16; // int4 per row = 16
  #pragma unroll
  for(int idx=tid; idx<ROWS*VPR; idx+=THREADS){
    int r = idx / VPR; int v = idx % VPR;
    int gp = row_start + r;
    int4 data;
    if(gp < S) data = reinterpret_cast<const int4*>(gptr + (int64_t)gp*D)[v];
    else data = make_int4(0,0,0,0);
    reinterpret_cast<int4*>(smem + r*D)[v] = data;
  }
}

__global__ void bwd_kernel(
   const __nv_bfloat16* __restrict__ Q,
   const __nv_bfloat16* __restrict__ K,
   const __nv_bfloat16* __restrict__ V,
   const __nv_bfloat16* __restrict__ dOg,
   const float* __restrict__ Lmat,
   const float* __restrict__ Delta,
   float* __restrict__ dQf,
   __nv_bfloat16* __restrict__ dK,
   __nv_bfloat16* __restrict__ dV,
   int S, int H, float scale)
{
  extern __shared__ char smem[];
  __nv_bfloat16* sQ  = (__nv_bfloat16*)smem;          // BM*D
  __nv_bfloat16* sK  = sQ + BM*D;                     // BN*D
  __nv_bfloat16* sV  = sK + BN*D;                     // BN*D
  __nv_bfloat16* sdO = sV + BN*D;                     // BM*D
  float* sP  = (float*)(sdO + BM*D);                  // BM*BN
  float* sdS = sP + BM*BN;                            // BM*BN
  float* sL  = sdS + BM*BN;                           // BM
  float* sD  = sL + BM;                               // BM

  int kvb = blockIdx.x;
  int h   = blockIdx.y;
  int b   = blockIdx.z;
  int kv_start = kvb*BN;
  if(kv_start >= S) return;

  int64_t head_off = ((int64_t)(b*H + h))*S*D;
  const __nv_bfloat16* Qh  = Q + head_off;
  const __nv_bfloat16* Kh  = K + head_off;
  const __nv_bfloat16* Vh  = V + head_off;
  const __nv_bfloat16* dOh = dOg + head_off;
  const float* Lh = Lmat + (int64_t)(b*H+h)*S;
  const float* Dh = Delta + (int64_t)(b*H+h)*S;
  float* dQfh = dQf + head_off;
  __nv_bfloat16* dKh = dK + head_off;
  __nv_bfloat16* dVh = dV + head_off;

  int tid = threadIdx.x;
  int ty = tid >> 4;  // 0..15
  int tx = tid & 15;  // 0..15

  loadTile(sK, Kh, kv_start, S, tid);
  loadTile(sV, Vh, kv_start, S, tid);

  float dV_acc[4][8]; float dK_acc[4][8];
  #pragma unroll
  for(int i=0;i<4;i++)
    #pragma unroll
    for(int j=0;j<8;j++){ dV_acc[i][j]=0.f; dK_acc[i][j]=0.f; }

  __syncthreads();

  for(int q_start = kv_start; q_start < S; q_start += BM){
    loadTile(sQ, Qh, q_start, S, tid);
    loadTile(sdO, dOh, q_start, S, tid);
    if(tid < BM){
      int qp = q_start+tid;
      sL[tid] = (qp<S)? Lh[qp] : 0.f;
      sD[tid] = (qp<S)? Dh[qp] : 0.f;
    }
    __syncthreads();

    // Phase A: S = Q@K^T -> P
    float sacc[4][4];
    #pragma unroll
    for(int i=0;i<4;i++)
      #pragma unroll
      for(int j=0;j<4;j++) sacc[i][j]=0.f;
    #pragma unroll 4
    for(int k=0;k<D;k++){
      float qv[4], kv[4];
      #pragma unroll
      for(int i=0;i<4;i++) qv[i]=__bfloat162float(sQ[(ty*4+i)*D+k]);
      #pragma unroll
      for(int j=0;j<4;j++) kv[j]=__bfloat162float(sK[(tx*4+j)*D+k]);
      #pragma unroll
      for(int i=0;i<4;i++)
        #pragma unroll
        for(int j=0;j<4;j++) sacc[i][j]+=qv[i]*kv[j];
    }
    #pragma unroll
    for(int i=0;i<4;i++){
      int m=ty*4+i; int qp=q_start+m;
      #pragma unroll
      for(int j=0;j<4;j++){
        int n=tx*4+j; int kp=kv_start+n;
        float val=0.f;
        if(qp<S && kp<S && qp>=kp){
          val = __expf(scale*sacc[i][j] - sL[m]);
        }
        sP[m*BN+n]=val;
      }
    }
    __syncthreads();

    // Phase B: dP = dO@V^T -> dS
    float dacc[4][4];
    #pragma unroll
    for(int i=0;i<4;i++)
      #pragma unroll
      for(int j=0;j<4;j++) dacc[i][j]=0.f;
    #pragma unroll 4
    for(int k=0;k<D;k++){
      float dv4[4], vv[4];
      #pragma unroll
      for(int i=0;i<4;i++) dv4[i]=__bfloat162float(sdO[(ty*4+i)*D+k]);
      #pragma unroll
      for(int j=0;j<4;j++) vv[j]=__bfloat162float(sV[(tx*4+j)*D+k]);
      #pragma unroll
      for(int i=0;i<4;i++)
        #pragma unroll
        for(int j=0;j<4;j++) dacc[i][j]+=dv4[i]*vv[j];
    }
    #pragma unroll
    for(int i=0;i<4;i++){
      int m=ty*4+i;
      #pragma unroll
      for(int j=0;j<4;j++){
        int n=tx*4+j;
        float p=sP[m*BN+n];
        sdS[m*BN+n]= p*(dacc[i][j]-sD[m]);
      }
    }
    __syncthreads();

    // Phase C: dV += P^T@dO ; dK += dS^T@Q  (over m)
    for(int m=0;m<BM;m++){
      float pr[4], dsr[4];
      #pragma unroll
      for(int i=0;i<4;i++){ pr[i]=sP[m*BN + ty*4+i]; dsr[i]=sdS[m*BN+ty*4+i]; }
      float dor[8], qr[8];
      #pragma unroll
      for(int j=0;j<8;j++){ dor[j]=__bfloat162float(sdO[m*D + tx*8+j]); qr[j]=__bfloat162float(sQ[m*D+tx*8+j]); }
      #pragma unroll
      for(int i=0;i<4;i++)
        #pragma unroll
        for(int j=0;j<8;j++){ dV_acc[i][j]+=pr[i]*dor[j]; dK_acc[i][j]+=dsr[i]*qr[j]; }
    }

    // Phase D: dQ tile = scale * dS@K -> atomic add
    float dq[4][8];
    #pragma unroll
    for(int i=0;i<4;i++)
      #pragma unroll
      for(int j=0;j<8;j++) dq[i][j]=0.f;
    for(int n=0;n<BN;n++){
      float dsr[4];
      #pragma unroll
      for(int i=0;i<4;i++) dsr[i]=sdS[(ty*4+i)*BN + n];
      float kr[8];
      #pragma unroll
      for(int j=0;j<8;j++) kr[j]=__bfloat162float(sK[n*D + tx*8+j]);
      #pragma unroll
      for(int i=0;i<4;i++)
        #pragma unroll
        for(int j=0;j<8;j++) dq[i][j]+=dsr[i]*kr[j];
    }
    #pragma unroll
    for(int i=0;i<4;i++){
      int m=ty*4+i; int qp=q_start+m;
      if(qp<S){
        #pragma unroll
        for(int j=0;j<8;j++){
          int c=tx*8+j;
          atomicAdd(&dQfh[(int64_t)qp*D + c], scale*dq[i][j]);
        }
      }
    }
    __syncthreads();
  }

  // write dK, dV
  #pragma unroll
  for(int i=0;i<4;i++){
    int n=ty*4+i; int kp=kv_start+n;
    if(kp<S){
      #pragma unroll
      for(int j=0;j<8;j++){
        int c=tx*8+j;
        dVh[(int64_t)kp*D + c] = __float2bfloat16(dV_acc[i][j]);
        dKh[(int64_t)kp*D + c] = __float2bfloat16(scale*dK_acc[i][j]);
      }
    }
  }
}

__global__ void convert_kernel(const float* __restrict__ dQf, __nv_bfloat16* __restrict__ dQ, int64_t n){
  int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) dQ[i]=__float2bfloat16(dQf[i]);
}

void run(TensorView Q, TensorView K, TensorView V, TensorView O, TensorView dO, TensorView L,
         TensorView dQ, TensorView dK, TensorView dV){
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int B=(int)Q.size(0), H=(int)Q.size(1), S=(int)Q.size(2);
  float scale = 1.0f / sqrtf((float)D);
  cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

  auto Qp=(const __nv_bfloat16*)Q.data_ptr();
  auto Kp=(const __nv_bfloat16*)K.data_ptr();
  auto Vp=(const __nv_bfloat16*)V.data_ptr();
  auto Op=(const __nv_bfloat16*)O.data_ptr();
  auto dOp=(const __nv_bfloat16*)dO.data_ptr();
  auto Lp=(const float*)L.data_ptr();
  auto dQp=(__nv_bfloat16*)dQ.data_ptr();
  auto dKp=(__nv_bfloat16*)dK.data_ptr();
  auto dVp=(__nv_bfloat16*)dV.data_ptr();

  int64_t totalRows=(int64_t)B*H*S;
  int64_t totalElems=totalRows*D;

  float* Delta=nullptr; float* dQf=nullptr;
  CUDA_CHECK(cudaMalloc(&Delta, totalRows*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dQf, totalElems*sizeof(float)));
  CUDA_CHECK(cudaMemsetAsync(dQf, 0, totalElems*sizeof(float), stream));

  {
    int t=256; int64_t bl=(totalRows+t-1)/t;
    delta_kernel<<<(unsigned)bl,t,0,stream>>>(Op,dOp,Delta,totalRows);
  }

  const size_t smem = (size_t)(2*BM + 2*BN)*D*sizeof(__nv_bfloat16)
                    + (size_t)2*BM*BN*sizeof(float)
                    + (size_t)2*BM*sizeof(float); // = 98816 bytes

  CUDA_CHECK(cudaFuncSetAttribute(bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));

  int num_kv=(S+BN-1)/BN;
  dim3 grid((unsigned)num_kv, (unsigned)H, (unsigned)B);
  bwd_kernel<<<grid, THREADS, smem, stream>>>(Qp,Kp,Vp,dOp,Lp,Delta,dQf,dKp,dVp,S,H,scale);

  {
    int t=256; int64_t bl=(totalElems+t-1)/t;
    convert_kernel<<<(unsigned)bl,t,0,stream>>>(dQf,dQp,totalElems);
  }

  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaFree(Delta));
  CUDA_CHECK(cudaFree(dQf));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd