#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { \
  cudaError_t _e=(call); \
  if(_e!=cudaSuccess){ fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__); } \
} while(0)

namespace mha_d128 {

constexpr int BM=64, BN=64, D=128, NT=128;

__device__ __forceinline__ void load_tile(__nv_bfloat16* dst, const __nv_bfloat16* base,
                                          int row_start, int S, int tid) {
  // BM*D = 8192 bf16 = 1024 int4 loads, 128 threads -> 8 each
  #pragma unroll
  for (int it=0; it<8; ++it) {
    int v = tid + it*NT;          // 0..1023
    int elem = v*8;
    int row = elem >> 7;          // /128
    int col = elem & 127;         // %128
    int grow = row_start + row;
    int4 val;
    if (grow < S) val = *reinterpret_cast<const int4*>(base + (long)grow*D + col);
    else val = make_int4(0,0,0,0);
    *reinterpret_cast<int4*>(dst + row*D + col) = val;
  }
}

__global__ __launch_bounds__(128) void mha_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S, float scale) {
  extern __shared__ char smem_raw[];
  __nv_bfloat16* sQ=reinterpret_cast<__nv_bfloat16*>(smem_raw);
  __nv_bfloat16* sK=sQ+BM*D;
  __nv_bfloat16* sV=sK+BN*D;
  float* sS=reinterpret_cast<float*>(sV+BN*D);
  float* sM=sS+BM*BN;
  float* sL=sM+BM;
  float* sCorr=sL+BM;

  const int tid=threadIdx.x;
  const int b=blockIdx.z, h=blockIdx.y, qtile=blockIdx.x;
  const int q_start=qtile*BM;
  const int rg=tid>>3, cg=tid&7;        // rg 0..15, cg 0..7
  const int r0=rg*4, sc0=cg*8, oc0=cg*16;

  long bh=(long)(b*H+h);
  const __nv_bfloat16* Qbase=Q+bh*S*D;
  const __nv_bfloat16* Kbase=K+bh*S*D;
  const __nv_bfloat16* Vbase=V+bh*S*D;
  __nv_bfloat16* Obase=O+bh*S*D;
  float* LSEbase=LSE+bh*S;

  float oacc[4][16];
  #pragma unroll
  for(int i=0;i<4;i++)
    #pragma unroll
    for(int j=0;j<16;j++) oacc[i][j]=0.f;

  if(tid<BM){sM[tid]=-1e30f;sL[tid]=0.f;}
  load_tile(sQ,Qbase,q_start,S,tid);
  __syncthreads();

  int num_kt=(S+BN-1)/BN;
  for(int kt=0;kt<num_kt;++kt){
    int k_start=kt*BN;
    load_tile(sK,Kbase,k_start,S,tid);
    load_tile(sV,Vbase,k_start,S,tid);
    __syncthreads();

    // ---- S = Q @ K^T (thread: 4 rows x 8 cols) ----
    float acc[4][8];
    #pragma unroll
    for(int i=0;i<4;i++)
      #pragma unroll
      for(int j=0;j<8;j++) acc[i][j]=0.f;

    #pragma unroll 4
    for(int d0=0;d0<D;d0+=8){
      __align__(16) __nv_bfloat16 qf[4][8];
      __align__(16) __nv_bfloat16 kf[8][8];
      #pragma unroll
      for(int i=0;i<4;i++)
        *reinterpret_cast<int4*>(&qf[i][0])=*reinterpret_cast<const int4*>(&sQ[(r0+i)*D+d0]);
      #pragma unroll
      for(int j=0;j<8;j++)
        *reinterpret_cast<int4*>(&kf[j][0])=*reinterpret_cast<const int4*>(&sK[(sc0+j)*D+d0]);
      #pragma unroll
      for(int dd=0;dd<8;dd++){
        float kv[8];
        #pragma unroll
        for(int j=0;j<8;j++) kv[j]=__bfloat162float(kf[j][dd]);
        #pragma unroll
        for(int i=0;i<4;i++){
          float q=__bfloat162float(qf[i][dd]);
          #pragma unroll
          for(int j=0;j<8;j++) acc[i][j]+=q*kv[j];
        }
      }
    }
    #pragma unroll
    for(int i=0;i<4;i++)
      #pragma unroll
      for(int j=0;j<8;j++){
        int gk=k_start+sc0+j;
        sS[(r0+i)*BN+(sc0+j)]=(gk<S)?acc[i][j]*scale:-1e30f;
      }
    __syncthreads();

    // ---- online softmax (one thread per row) ----
    if(tid<BM){
      int r=tid;
      float m_old=sM[r], l_old=sL[r];
      float lm=-1e30f;
      #pragma unroll 8
      for(int j=0;j<BN;j++) lm=fmaxf(lm,sS[r*BN+j]);
      float m_new=fmaxf(m_old,lm);
      float corr=__expf(m_old-m_new);
      float sum=0.f;
      #pragma unroll 8
      for(int j=0;j<BN;j++){
        float p=__expf(sS[r*BN+j]-m_new);
        sS[r*BN+j]=p; sum+=p;
      }
      sM[r]=m_new; sL[r]=l_old*corr+sum; sCorr[r]=corr;
    }
    __syncthreads();

    // ---- rescale O accumulator ----
    float ci[4];
    #pragma unroll
    for(int i=0;i<4;i++) ci[i]=sCorr[r0+i];
    #pragma unroll
    for(int i=0;i<4;i++)
      #pragma unroll
      for(int j=0;j<16;j++) oacc[i][j]*=ci[i];

    // ---- O += P @ V (thread: 4 rows x 16 cols) ----
    #pragma unroll 4
    for(int n=0;n<BN;++n){
      float pv[4];
      #pragma unroll
      for(int i=0;i<4;i++) pv[i]=sS[(r0+i)*BN+n];
      __align__(16) __nv_bfloat16 vf[16];
      *reinterpret_cast<int4*>(&vf[0])=*reinterpret_cast<const int4*>(&sV[n*D+oc0]);
      *reinterpret_cast<int4*>(&vf[8])=*reinterpret_cast<const int4*>(&sV[n*D+oc0+8]);
      #pragma unroll
      for(int i=0;i<4;i++){
        #pragma unroll
        for(int j=0;j<16;j++) oacc[i][j]+=pv[i]*__bfloat162float(vf[j]);
      }
    }
    __syncthreads();
  }

  // ---- epilogue ----
  #pragma unroll
  for(int i=0;i<4;i++){
    int grow=q_start+r0+i;
    if(grow<S){
      float inv=1.f/sL[r0+i];
      #pragma unroll
      for(int j=0;j<16;j++)
        Obase[(long)grow*D+oc0+j]=__float2bfloat16(oacc[i][j]*inv);
      if(cg==0) LSEbase[grow]=sM[r0+i]+logf(sL[r0+i]);
    }
  }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE){
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int B=(int)Q.size(0), H=(int)Q.size(1), S=(int)Q.size(2), Dd=(int)Q.size(3);
  const __nv_bfloat16* Qp=(const __nv_bfloat16*)Q.data_ptr();
  const __nv_bfloat16* Kp=(const __nv_bfloat16*)K.data_ptr();
  const __nv_bfloat16* Vp=(const __nv_bfloat16*)V.data_ptr();
  __nv_bfloat16* Op=(__nv_bfloat16*)O.data_ptr();
  float* Lp=(float*)LSE.data_ptr();
  float scale=1.0f/sqrtf((float)Dd);

  dim3 grid((S+BM-1)/BM, H, B);
  dim3 block(NT);
  int smem=BM*D*2+BN*D*2+BN*D*2+BM*BN*4+BM*4*3;  // 66304 bytes
  cudaFuncSetAttribute(mha_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

  cudaStream_t stream=(cudaStream_t)TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id);
  mha_kernel<<<grid,block,smem,stream>>>(Qp,Kp,Vp,Op,Lp,B,H,S,scale);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_d128::run);

}  // namespace mha_d128