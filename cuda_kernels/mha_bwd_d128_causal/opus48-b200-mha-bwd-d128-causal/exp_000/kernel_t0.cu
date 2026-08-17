#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <math.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
    }                                                              \
} while(0)

namespace mha_bwd_d128_causal {

typedef __nv_bfloat16 bf16;

__device__ __forceinline__ float b2f(bf16 x){ return __bfloat162float(x); }
__device__ __forceinline__ bf16  f2b(float x){ return __float2bfloat16(x); }

// D_i = sum_c dO[i,c] * O[i,c]  (one warp per row, d=128)
__global__ void delta_kernel(const bf16* __restrict__ dO, const bf16* __restrict__ O,
                             float* __restrict__ delta, int64_t numRows, int d){
  int64_t warp = ((int64_t)blockIdx.x*blockDim.x + threadIdx.x) >> 5;
  int lane = threadIdx.x & 31;
  if(warp >= numRows) return;
  const bf16* dOp = dO + warp*(int64_t)d;
  const bf16* Op  = O  + warp*(int64_t)d;
  float s = 0.f;
  for(int c=lane; c<d; c+=32) s += b2f(dOp[c]) * b2f(Op[c]);
  #pragma unroll
  for(int off=16; off>0; off>>=1) s += __shfl_down_sync(0xffffffffu, s, off);
  if(lane==0) delta[warp] = s;
}

// -------- dK, dV kernel : grid(numBlk, BH), 256 threads --------
// Each CTA owns one KV block (64 rows). Loops over Q blocks (causal).
__global__ void __launch_bounds__(256,2) dkv_kernel(
    const bf16* __restrict__ Qa, const bf16* __restrict__ Ka,
    const bf16* __restrict__ Va, const bf16* __restrict__ dOa,
    const float* __restrict__ La, const float* __restrict__ Da,
    bf16* __restrict__ dKa, bf16* __restrict__ dVa,
    int S, float scale)
{
  int kb = blockIdx.x, bh = blockIdx.y, numBlk = gridDim.x;
  int kv0 = kb*64;
  int64_t base  = (int64_t)bh * S * 128;
  int64_t lbase = (int64_t)bh * S;
  const bf16* Qg=Qa+base; const bf16* Kg=Ka+base; const bf16* Vg=Va+base; const bf16* dOg=dOa+base;
  const float* Lg=La+lbase; const float* Dg=Da+lbase;
  bf16* dKg=dKa+base; bf16* dVg=dVa+base;

  extern __shared__ char smem[];
  bf16* Ksh =(bf16*)smem;
  bf16* Vsh = Ksh + 64*128;
  bf16* Qsh = Vsh + 64*128;
  bf16* dOsh= Qsh + 64*128;
  float* Psh =(float*)(dOsh + 64*128);
  float* dSsh= Psh + 64*64;
  float* Lsh = dSsh + 64*64;
  float* Dsh = Lsh + 64;

  int tid = threadIdx.x;
  bf16 zero = f2b(0.f);

  // load K, V (once)
  for(int idx=tid; idx<64*128; idx+=256){
    int r=idx>>7, c=idx&127; int gs=kv0+r;
    bf16 kk=zero, vv=zero;
    if(gs<S){ kk=Kg[(int64_t)gs*128+c]; vv=Vg[(int64_t)gs*128+c]; }
    Ksh[idx]=kk; Vsh[idx]=vv;
  }

  int tr=tid>>4, tc=tid&15;   // 16x16 grid
  int i0=tr*4, j0=tc*4;       // phase A: query rows / key cols
  int r0=tr*4, c0=tc*8;       // phase B: kv rows / d cols

  float aV[4][8], aK[4][8];
  #pragma unroll
  for(int a=0;a<4;a++)
    #pragma unroll
    for(int b=0;b<8;b++){ aV[a][b]=0.f; aK[a][b]=0.f; }

  __syncthreads();

  for(int qb=kb; qb<numBlk; qb++){
    int q0=qb*64;
    for(int idx=tid; idx<64*128; idx+=256){
      int r=idx>>7, c=idx&127; int gs=q0+r;
      bf16 qq=zero, dd=zero;
      if(gs<S){ qq=Qg[(int64_t)gs*128+c]; dd=dOg[(int64_t)gs*128+c]; }
      Qsh[idx]=qq; dOsh[idx]=dd;
    }
    for(int r=tid;r<64;r+=256){ int gs=q0+r; Lsh[r]=(gs<S)?Lg[gs]:0.f; Dsh[r]=(gs<S)?Dg[gs]:0.f; }
    __syncthreads();

    // phase A : S = Q@K^T , dP = dO@V^T
    float Sr[4][4], dPr[4][4];
    #pragma unroll
    for(int a=0;a<4;a++)
      #pragma unroll
      for(int b=0;b<4;b++){ Sr[a][b]=0.f; dPr[a][b]=0.f; }
    for(int k=0;k<128;k++){
      float qv[4],dov[4],kv[4],vv[4];
      #pragma unroll
      for(int a=0;a<4;a++){ qv[a]=b2f(Qsh[(i0+a)*128+k]); dov[a]=b2f(dOsh[(i0+a)*128+k]); }
      #pragma unroll
      for(int b=0;b<4;b++){ kv[b]=b2f(Ksh[(j0+b)*128+k]); vv[b]=b2f(Vsh[(j0+b)*128+k]); }
      #pragma unroll
      for(int a=0;a<4;a++)
        #pragma unroll
        for(int b=0;b<4;b++){ Sr[a][b]+=qv[a]*kv[b]; dPr[a][b]+=dov[a]*vv[b]; }
    }
    #pragma unroll
    for(int a=0;a<4;a++){
      int gi=q0+i0+a; float Li=Lsh[i0+a], Di=Dsh[i0+a];
      #pragma unroll
      for(int b=0;b<4;b++){
        int gj=kv0+j0+b;
        float p = (gi<S && gj<S && gj<=gi) ? __expf(Sr[a][b]*scale - Li) : 0.f;
        float ds = p*(dPr[a][b]-Di);
        Psh [(i0+a)*64+(j0+b)] = p;
        dSsh[(i0+a)*64+(j0+b)] = ds;
      }
    }
    __syncthreads();

    // phase B : dV += P^T@dO , dK += dS^T@Q
    for(int i=0;i<64;i++){
      float pv[4],dsv[4],doc[8],qc[8];
      #pragma unroll
      for(int a=0;a<4;a++){ pv[a]=Psh[i*64+(r0+a)]; dsv[a]=dSsh[i*64+(r0+a)]; }
      #pragma unroll
      for(int b=0;b<8;b++){ doc[b]=b2f(dOsh[i*128+(c0+b)]); qc[b]=b2f(Qsh[i*128+(c0+b)]); }
      #pragma unroll
      for(int a=0;a<4;a++)
        #pragma unroll
        for(int b=0;b<8;b++){ aV[a][b]+=pv[a]*doc[b]; aK[a][b]+=dsv[a]*qc[b]; }
    }
    __syncthreads();
  }

  #pragma unroll
  for(int a=0;a<4;a++){
    int gj=kv0+r0+a;
    if(gj<S){
      #pragma unroll
      for(int b=0;b<8;b++){
        int c=c0+b;
        dVg[(int64_t)gj*128+c]=f2b(aV[a][b]);
        dKg[(int64_t)gj*128+c]=f2b(aK[a][b]*scale);
      }
    }
  }
}

// -------- dQ kernel : grid(numBlk, BH), 256 threads --------
// Each CTA owns one Q block (64 rows). Loops over KV blocks (causal).
__global__ void __launch_bounds__(256,2) dq_kernel(
    const bf16* __restrict__ Qa, const bf16* __restrict__ Ka,
    const bf16* __restrict__ Va, const bf16* __restrict__ dOa,
    const float* __restrict__ La, const float* __restrict__ Da,
    bf16* __restrict__ dQa,
    int S, float scale)
{
  int qb=blockIdx.x, bh=blockIdx.y;
  int q0=qb*64;
  int64_t base  = (int64_t)bh * S * 128;
  int64_t lbase = (int64_t)bh * S;
  const bf16* Qg=Qa+base; const bf16* Kg=Ka+base; const bf16* Vg=Va+base; const bf16* dOg=dOa+base;
  const float* Lg=La+lbase; const float* Dg=Da+lbase;
  bf16* dQg=dQa+base;

  extern __shared__ char smem[];
  bf16* Ksh =(bf16*)smem;
  bf16* Vsh = Ksh + 64*128;
  bf16* Qsh = Vsh + 64*128;
  bf16* dOsh= Qsh + 64*128;
  float* dSsh=(float*)(dOsh + 64*128);
  float* Lsh = dSsh + 64*64;
  float* Dsh = Lsh + 64;

  int tid=threadIdx.x;
  bf16 zero=f2b(0.f);

  // load Q, dO, L, D (once)
  for(int idx=tid; idx<64*128; idx+=256){
    int r=idx>>7, c=idx&127; int gs=q0+r;
    bf16 qq=zero, dd=zero;
    if(gs<S){ qq=Qg[(int64_t)gs*128+c]; dd=dOg[(int64_t)gs*128+c]; }
    Qsh[idx]=qq; dOsh[idx]=dd;
  }
  for(int r=tid;r<64;r+=256){ int gs=q0+r; Lsh[r]=(gs<S)?Lg[gs]:0.f; Dsh[r]=(gs<S)?Dg[gs]:0.f; }

  int tr=tid>>4, tc=tid&15;
  int i0=tr*4, j0=tc*4;
  int r0=tr*4, c0=tc*8;

  float aQ[4][8];
  #pragma unroll
  for(int a=0;a<4;a++)
    #pragma unroll
    for(int b=0;b<8;b++) aQ[a][b]=0.f;

  __syncthreads();

  for(int kb=0; kb<=qb; kb++){
    int kv0=kb*64;
    for(int idx=tid; idx<64*128; idx+=256){
      int r=idx>>7, c=idx&127; int gs=kv0+r;
      bf16 kk=zero, vv=zero;
      if(gs<S){ kk=Kg[(int64_t)gs*128+c]; vv=Vg[(int64_t)gs*128+c]; }
      Ksh[idx]=kk; Vsh[idx]=vv;
    }
    __syncthreads();

    float Sr[4][4], dPr[4][4];
    #pragma unroll
    for(int a=0;a<4;a++)
      #pragma unroll
      for(int b=0;b<4;b++){ Sr[a][b]=0.f; dPr[a][b]=0.f; }
    for(int k=0;k<128;k++){
      float qv[4],dov[4],kv[4],vv[4];
      #pragma unroll
      for(int a=0;a<4;a++){ qv[a]=b2f(Qsh[(i0+a)*128+k]); dov[a]=b2f(dOsh[(i0+a)*128+k]); }
      #pragma unroll
      for(int b=0;b<4;b++){ kv[b]=b2f(Ksh[(j0+b)*128+k]); vv[b]=b2f(Vsh[(j0+b)*128+k]); }
      #pragma unroll
      for(int a=0;a<4;a++)
        #pragma unroll
        for(int b=0;b<4;b++){ Sr[a][b]+=qv[a]*kv[b]; dPr[a][b]+=dov[a]*vv[b]; }
    }
    #pragma unroll
    for(int a=0;a<4;a++){
      int gi=q0+i0+a; float Li=Lsh[i0+a], Di=Dsh[i0+a];
      #pragma unroll
      for(int b=0;b<4;b++){
        int gj=kv0+j0+b;
        float p = (gi<S && gj<S && gj<=gi) ? __expf(Sr[a][b]*scale - Li) : 0.f;
        float ds = p*(dPr[a][b]-Di);
        dSsh[(i0+a)*64+(j0+b)] = ds;
      }
    }
    __syncthreads();

    // dQ += dS@K
    for(int j=0;j<64;j++){
      float dsv[4],kc[8];
      #pragma unroll
      for(int a=0;a<4;a++) dsv[a]=dSsh[(r0+a)*64+j];
      #pragma unroll
      for(int b=0;b<8;b++) kc[b]=b2f(Ksh[j*128+(c0+b)]);
      #pragma unroll
      for(int a=0;a<4;a++)
        #pragma unroll
        for(int b=0;b<8;b++) aQ[a][b]+=dsv[a]*kc[b];
    }
    __syncthreads();
  }

  #pragma unroll
  for(int a=0;a<4;a++){
    int gi=q0+r0+a;
    if(gi<S){
      #pragma unroll
      for(int b=0;b<8;b++){
        int c=c0+b;
        dQg[(int64_t)gi*128+c]=f2b(aQ[a][b]*scale);
      }
    }
  }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV)
{
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int B=(int)Q.size(0), H=(int)Q.size(1), S=(int)Q.size(2), d=(int)Q.size(3);
  int BH=B*H;
  int numBlk=(S+63)/64;
  float scale=1.0f/sqrtf((float)d);

  const bf16* Qp=(const bf16*)Q.data_ptr();
  const bf16* Kp=(const bf16*)K.data_ptr();
  const bf16* Vp=(const bf16*)V.data_ptr();
  const bf16* Op=(const bf16*)O.data_ptr();
  const bf16* dOp=(const bf16*)dO.data_ptr();
  const float* Lp=(const float*)L.data_ptr();
  bf16* dQp=(bf16*)dQ.data_ptr();
  bf16* dKp=(bf16*)dK.data_ptr();
  bf16* dVp=(bf16*)dV.data_ptr();

  cudaStream_t stream=(cudaStream_t)TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id);

  float* delta=nullptr;
  CUDA_CHECK(cudaMalloc(&delta,(size_t)BH*S*sizeof(float)));

  int64_t numRows=(int64_t)BH*S;
  int dblk=256, wpb=dblk/32;
  int64_t dgrid=(numRows+wpb-1)/wpb;
  delta_kernel<<<(unsigned)dgrid,dblk,0,stream>>>(dOp,Op,delta,numRows,d);
  CUDA_CHECK(cudaGetLastError());

  size_t sh1=(size_t)(4*64*128*2 + 2*64*64*4 + 2*64*4); // 98816
  size_t sh2=(size_t)(4*64*128*2 + 1*64*64*4 + 2*64*4); // 82432
  CUDA_CHECK(cudaFuncSetAttribute((const void*)dkv_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh1));
  CUDA_CHECK(cudaFuncSetAttribute((const void*)dq_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh2));

  dim3 grid((unsigned)numBlk,(unsigned)BH);
  dkv_kernel<<<grid,256,sh1,stream>>>(Qp,Kp,Vp,dOp,Lp,delta,dKp,dVp,S,scale);
  CUDA_CHECK(cudaGetLastError());
  dq_kernel<<<grid,256,sh2,stream>>>(Qp,Kp,Vp,dOp,Lp,delta,dQp,S,scale);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaFree(delta));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_d128_causal::run);

}  // namespace mha_bwd_d128_causal