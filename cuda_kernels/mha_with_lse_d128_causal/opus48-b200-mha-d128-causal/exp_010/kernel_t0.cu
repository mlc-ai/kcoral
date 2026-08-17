#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <math.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__); exit(1);} } while(0)

namespace mha_kernel {

#define DD 128
#define BM 64
#define BN 64

__device__ __forceinline__ void mma_m16n8k16(
  float &d0,float &d1,float &d2,float &d3,
  uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,
  uint32_t b0,uint32_t b1,
  float c0,float c1,float c2,float c3){
  asm volatile(
   "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
   "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
   : "=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
   : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
     "f"(c0),"f"(c1),"f"(c2),"f"(c3));
}

__device__ __forceinline__ uint32_t pack2bf16(float a,float b){
  __nv_bfloat16 x=__float2bfloat16(a);
  __nv_bfloat16 y=__float2bfloat16(b);
  uint16_t xi=*reinterpret_cast<uint16_t*>(&x);
  uint16_t yi=*reinterpret_cast<uint16_t*>(&y);
  return (uint32_t)xi | ((uint32_t)yi<<16);
}
__device__ __forceinline__ float groupMax(float v){
  v=fmaxf(v,__shfl_xor_sync(0xffffffff,v,1));
  v=fmaxf(v,__shfl_xor_sync(0xffffffff,v,2));
  return v;
}
__device__ __forceinline__ float groupSum(float v){
  v+=__shfl_xor_sync(0xffffffff,v,1);
  v+=__shfl_xor_sync(0xffffffff,v,2);
  return v;
}

__global__ __launch_bounds__(128) void mha_kernel_fn(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int S){
  const float scale2 = 0.08838834764831845f * 1.4426950408889634f; // (1/sqrt(128))*log2e
  const float LN2 = 0.6931471805599453f;
  const float NEG = -1e30f;

  int bh = blockIdx.y;
  int q_block = blockIdx.x;
  int q0 = q_block*BM;
  if(q0>=S) return;

  int tid = threadIdx.x;
  int warp = tid>>5;
  int lane = tid&31;
  int groupID = lane>>2;
  int tg = lane&3;

  __shared__ __align__(16) __nv_bfloat16 sQK[BN*DD]; // Q first, then K per iter
  __shared__ __align__(16) __nv_bfloat16 sV[BN*DD];

  const __nv_bfloat16* Qg = Q + (size_t)bh*S*DD;
  const __nv_bfloat16* Kg = K + (size_t)bh*S*DD;
  const __nv_bfloat16* Vg = V + (size_t)bh*S*DD;
  __nv_bfloat16* Og = O + (size_t)bh*S*DD;
  float* LSEg = LSE + (size_t)bh*S;

  // Load Q tile into sQK
  #pragma unroll
  for(int i=tid;i<(BM*DD)/8;i+=128){
    int idx=i*8; int row=idx>>7; int col=idx&127;
    int gs=q0+row;
    int4 val;
    if(gs<S) val=*(const int4*)&Qg[(size_t)gs*DD+col];
    else { val.x=val.y=val.z=val.w=0; }
    *(int4*)&sQK[row*DD+col]=val;
  }
  __syncthreads();

  // Load Q fragments (A operand), reused across all kv blocks
  uint32_t qfrag[8][4];
  #pragma unroll
  for(int kk=0;kk<8;kk++){
    int bc=16*kk;
    int r0=warp*16+groupID;
    int r1=warp*16+groupID+8;
    qfrag[kk][0]=*(const uint32_t*)&sQK[r0*DD+bc+2*tg];
    qfrag[kk][1]=*(const uint32_t*)&sQK[r1*DD+bc+2*tg];
    qfrag[kk][2]=*(const uint32_t*)&sQK[r0*DD+bc+8+2*tg];
    qfrag[kk][3]=*(const uint32_t*)&sQK[r1*DD+bc+8+2*tg];
  }
  __syncthreads(); // ensure all read Q before overwriting sQK

  float acc[16][4];
  #pragma unroll
  for(int jj=0;jj<16;jj++){acc[jj][0]=acc[jj][1]=acc[jj][2]=acc[jj][3]=0.f;}
  float m0=NEG,m1=NEG,l0=0.f,l1=0.f;

  int qrow0=q0+warp*16+groupID;
  int qrow1=q0+warp*16+groupID+8;

  const uint16_t* Vu=(const uint16_t*)sV;

  for(int kvb=0;kvb<=q_block;kvb++){
    int kv0=kvb*BN;
    #pragma unroll
    for(int i=tid;i<(BN*DD)/8;i+=128){
      int idx=i*8; int row=idx>>7; int col=idx&127;
      int gs=kv0+row;
      int4 vk,vv;
      if(gs<S){ vk=*(const int4*)&Kg[(size_t)gs*DD+col]; vv=*(const int4*)&Vg[(size_t)gs*DD+col]; }
      else { vk.x=vk.y=vk.z=vk.w=0; vv=vk; }
      *(int4*)&sQK[row*DD+col]=vk;
      *(int4*)&sV[row*DD+col]=vv;
    }
    __syncthreads();

    // matmul1: S = Q @ K^T
    float sfrag[8][4];
    #pragma unroll
    for(int j=0;j<8;j++){
      float d0=0,d1=0,d2=0,d3=0;
      #pragma unroll
      for(int kk=0;kk<8;kk++){
        int kcol=16*kk;
        int krow=8*j+groupID;
        uint32_t b0=*(const uint32_t*)&sQK[krow*DD+kcol+2*tg];
        uint32_t b1=*(const uint32_t*)&sQK[krow*DD+kcol+8+2*tg];
        mma_m16n8k16(d0,d1,d2,d3, qfrag[kk][0],qfrag[kk][1],qfrag[kk][2],qfrag[kk][3], b0,b1, d0,d1,d2,d3);
      }
      sfrag[j][0]=d0; sfrag[j][1]=d1; sfrag[j][2]=d2; sfrag[j][3]=d3;
    }

    // scale + causal mask (only diagonal block needs masking)
    bool diag = (kvb==q_block);
    #pragma unroll
    for(int j=0;j<8;j++){
      int col0=kv0+8*j+2*tg;
      if(diag){
        sfrag[j][0]=(col0  <=qrow0)?sfrag[j][0]*scale2:NEG;
        sfrag[j][1]=(col0+1<=qrow0)?sfrag[j][1]*scale2:NEG;
        sfrag[j][2]=(col0  <=qrow1)?sfrag[j][2]*scale2:NEG;
        sfrag[j][3]=(col0+1<=qrow1)?sfrag[j][3]*scale2:NEG;
      } else {
        sfrag[j][0]*=scale2; sfrag[j][1]*=scale2; sfrag[j][2]*=scale2; sfrag[j][3]*=scale2;
      }
    }

    // online softmax
    float bm0=NEG,bm1=NEG;
    #pragma unroll
    for(int j=0;j<8;j++){
      bm0=fmaxf(bm0,fmaxf(sfrag[j][0],sfrag[j][1]));
      bm1=fmaxf(bm1,fmaxf(sfrag[j][2],sfrag[j][3]));
    }
    bm0=groupMax(bm0); bm1=groupMax(bm1);
    float mn0=fmaxf(m0,bm0), mn1=fmaxf(m1,bm1);
    float sf0=exp2f(m0-mn0), sf1=exp2f(m1-mn1);
    float ps0=0.f,ps1=0.f;
    #pragma unroll
    for(int j=0;j<8;j++){
      float p0=exp2f(sfrag[j][0]-mn0);
      float p1=exp2f(sfrag[j][1]-mn0);
      float p2=exp2f(sfrag[j][2]-mn1);
      float p3=exp2f(sfrag[j][3]-mn1);
      sfrag[j][0]=p0; sfrag[j][1]=p1; sfrag[j][2]=p2; sfrag[j][3]=p3;
      ps0+=p0+p1; ps1+=p2+p3;
    }
    ps0=groupSum(ps0); ps1=groupSum(ps1);
    l0=l0*sf0+ps0; l1=l1*sf1+ps1;
    #pragma unroll
    for(int jj=0;jj<16;jj++){
      acc[jj][0]*=sf0; acc[jj][1]*=sf0; acc[jj][2]*=sf1; acc[jj][3]*=sf1;
    }
    m0=mn0; m1=mn1;

    // matmul2: acc += P @ V
    #pragma unroll
    for(int kk=0;kk<4;kk++){
      int j0=2*kk, j1=2*kk+1;
      uint32_t p0=pack2bf16(sfrag[j0][0],sfrag[j0][1]);
      uint32_t p1=pack2bf16(sfrag[j0][2],sfrag[j0][3]);
      uint32_t p2=pack2bf16(sfrag[j1][0],sfrag[j1][1]);
      uint32_t p3=pack2bf16(sfrag[j1][2],sfrag[j1][3]);
      int vr=16*kk+2*tg;
      #pragma unroll
      for(int jj=0;jj<16;jj++){
        int vcol=8*jj+groupID;
        uint16_t vb0=Vu[(vr)*DD+vcol];
        uint16_t vb1=Vu[(vr+1)*DD+vcol];
        uint16_t vb2=Vu[(vr+8)*DD+vcol];
        uint16_t vb3=Vu[(vr+9)*DD+vcol];
        uint32_t B0=(uint32_t)vb0|((uint32_t)vb1<<16);
        uint32_t B1=(uint32_t)vb2|((uint32_t)vb3<<16);
        mma_m16n8k16(acc[jj][0],acc[jj][1],acc[jj][2],acc[jj][3], p0,p1,p2,p3, B0,B1,
                     acc[jj][0],acc[jj][1],acc[jj][2],acc[jj][3]);
      }
    }
    __syncthreads();
  }

  // epilogue: normalize and write O, LSE
  float inv0 = 1.f/l0, inv1 = 1.f/l1;
  #pragma unroll
  for(int jj=0;jj<16;jj++){
    int d0=8*jj+2*tg;
    if(qrow0<S){
      Og[(size_t)qrow0*DD+d0  ]=__float2bfloat16(acc[jj][0]*inv0);
      Og[(size_t)qrow0*DD+d0+1]=__float2bfloat16(acc[jj][1]*inv0);
    }
    if(qrow1<S){
      Og[(size_t)qrow1*DD+d0  ]=__float2bfloat16(acc[jj][2]*inv1);
      Og[(size_t)qrow1*DD+d0+1]=__float2bfloat16(acc[jj][3]*inv1);
    }
  }
  if(tg==0){
    if(qrow0<S) LSEg[qrow0]=m0*LN2+logf(l0);
    if(qrow1<S) LSEg[qrow1]=m1*LN2+logf(l1);
  }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE){
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int B = (int)Q.size(0);
  int H = (int)Q.size(1);
  int S = (int)Q.size(2);

  const __nv_bfloat16* Qp=(const __nv_bfloat16*)Q.data_ptr();
  const __nv_bfloat16* Kp=(const __nv_bfloat16*)K.data_ptr();
  const __nv_bfloat16* Vp=(const __nv_bfloat16*)V.data_ptr();
  __nv_bfloat16* Op=(__nv_bfloat16*)O.data_ptr();
  float* Lp=(float*)LSE.data_ptr();

  dim3 grid((S+BM-1)/BM, B*H);
  dim3 block(128);
  cudaStream_t stream = (cudaStream_t)TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id);
  mha_kernel_fn<<<grid,block,0,stream>>>(Qp,Kp,Vp,Op,Lp,S);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_kernel::run);

}  // namespace mha_kernel