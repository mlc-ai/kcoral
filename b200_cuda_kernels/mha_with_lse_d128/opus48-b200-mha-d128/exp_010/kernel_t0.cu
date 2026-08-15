#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <math.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
    fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__); exit(1);} }while(0)

namespace mha_kernel {

constexpr int Dc = 128;
constexpr int BQ = 64;
constexpr int BK = 64;
constexpr int SD = Dc + 8;   // padded stride for Q/K smem (bank-conflict free, int4-aligned)
constexpr int SK = BK + 8;   // padded stride for transposed V smem

__device__ __forceinline__ void mma16816(float &d0,float &d1,float &d2,float &d3,
    uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,uint32_t b0,uint32_t b1){
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)
    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}

__device__ __forceinline__ uint32_t packf(float x,float y){
  __nv_bfloat162 r = __floats2bfloat162_rn(x,y);
  return *reinterpret_cast<uint32_t*>(&r);
}

__global__ __launch_bounds__(128) void mha_kernel_fn(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int S, int H){
  extern __shared__ __nv_bfloat16 smem[];
  __nv_bfloat16* Qs  = smem;
  __nv_bfloat16* Ks  = Qs + BQ*SD;
  __nv_bfloat16* Vs2 = Ks + BK*SD;   // transposed: Vs2[d*SK + key] = V[key][d]

  int b = blockIdx.z;
  int h = blockIdx.y;
  int qbase = blockIdx.x * BQ;
  int tid = threadIdx.x;
  int warp = tid >> 5;
  int lane = tid & 31;
  int gid = lane >> 2;   // 0..7
  int tig = lane & 3;    // 0..3

  const float scale = 0.08838834764831845f; // 1/sqrt(128)

  size_t headOff = (size_t)(b*H + h) * S;
  const __nv_bfloat16* Qbase = Q + headOff*Dc;
  const __nv_bfloat16* Kbase = K + headOff*Dc;
  const __nv_bfloat16* Vbase = V + headOff*Dc;
  __nv_bfloat16* Obase = O + headOff*Dc;
  float* LSEbase = LSE + headOff;

  // ---- load Q tile ----
  for(int v=tid; v<BQ*16; v+=128){
    int kk=v>>4; int d=(v&15)*8;
    int row=qbase+kk;
    int4 val = (row<S) ? *reinterpret_cast<const int4*>(Qbase+(size_t)row*Dc+d)
                       : make_int4(0,0,0,0);
    *reinterpret_cast<int4*>(&Qs[kk*SD+d]) = val;
  }
  __syncthreads();

  // ---- load Q fragments (persistent) ----
  uint32_t qf[8][4];
  {
    int r0 = warp*16 + gid;
    int r8 = r0 + 8;
    #pragma unroll
    for(int kt=0;kt<8;kt++){
      int d0 = kt*16 + tig*2;
      int d8 = d0 + 8;
      qf[kt][0]=*reinterpret_cast<const uint32_t*>(&Qs[r0*SD+d0]);
      qf[kt][1]=*reinterpret_cast<const uint32_t*>(&Qs[r8*SD+d0]);
      qf[kt][2]=*reinterpret_cast<const uint32_t*>(&Qs[r0*SD+d8]);
      qf[kt][3]=*reinterpret_cast<const uint32_t*>(&Qs[r8*SD+d8]);
    }
  }

  float o[16][4];
  #pragma unroll
  for(int i=0;i<16;i++){o[i][0]=o[i][1]=o[i][2]=o[i][3]=0.f;}
  float m0=-INFINITY,m1=-INFINITY,l0=0.f,l1=0.f;

  int num_kb = (S + BK - 1)/BK;
  for(int kb=0; kb<num_kb; kb++){
    // ---- load K (normal) and V (transposed) ----
    for(int v=tid; v<BK*16; v+=128){
      int kk=v>>4; int d=(v&15)*8;
      int key=kb*BK+kk;
      int4 kvv, vvv;
      if(key<S){
        kvv=*reinterpret_cast<const int4*>(Kbase+(size_t)key*Dc+d);
        vvv=*reinterpret_cast<const int4*>(Vbase+(size_t)key*Dc+d);
      } else { kvv=make_int4(0,0,0,0); vvv=make_int4(0,0,0,0);}
      *reinterpret_cast<int4*>(&Ks[kk*SD+d])=kvv;
      __nv_bfloat16* vb=reinterpret_cast<__nv_bfloat16*>(&vvv);
      #pragma unroll
      for(int i=0;i<8;i++) Vs2[(d+i)*SK+kk]=vb[i];
    }
    __syncthreads();

    // ---- QK^T ----
    float sc[8][4];
    #pragma unroll
    for(int nt=0;nt<8;nt++){sc[nt][0]=sc[nt][1]=sc[nt][2]=sc[nt][3]=0.f;}
    #pragma unroll
    for(int nt=0;nt<8;nt++){
      int key=nt*8+gid;
      #pragma unroll
      for(int kt=0;kt<8;kt++){
        int d0=kt*16+tig*2;
        uint32_t b0=*reinterpret_cast<const uint32_t*>(&Ks[key*SD+d0]);
        uint32_t b1=*reinterpret_cast<const uint32_t*>(&Ks[key*SD+d0+8]);
        mma16816(sc[nt][0],sc[nt][1],sc[nt][2],sc[nt][3],
                 qf[kt][0],qf[kt][1],qf[kt][2],qf[kt][3], b0,b1);
      }
    }
    // ---- scale + mask ----
    #pragma unroll
    for(int nt=0;nt<8;nt++){
      int kcol0=nt*8+tig*2;
      int g0=kb*BK+kcol0, g1=g0+1;
      sc[nt][0]*=scale; sc[nt][1]*=scale; sc[nt][2]*=scale; sc[nt][3]*=scale;
      if(g0>=S){ sc[nt][0]=-INFINITY; sc[nt][2]=-INFINITY; }
      if(g1>=S){ sc[nt][1]=-INFINITY; sc[nt][3]=-INFINITY; }
    }
    // ---- row max ----
    float lm0=-INFINITY, lm1=-INFINITY;
    #pragma unroll
    for(int nt=0;nt<8;nt++){
      lm0=fmaxf(lm0,fmaxf(sc[nt][0],sc[nt][1]));
      lm1=fmaxf(lm1,fmaxf(sc[nt][2],sc[nt][3]));
    }
    lm0=fmaxf(lm0,__shfl_xor_sync(0xffffffff,lm0,1));
    lm0=fmaxf(lm0,__shfl_xor_sync(0xffffffff,lm0,2));
    lm1=fmaxf(lm1,__shfl_xor_sync(0xffffffff,lm1,1));
    lm1=fmaxf(lm1,__shfl_xor_sync(0xffffffff,lm1,2));
    float nm0=fmaxf(m0,lm0), nm1=fmaxf(m1,lm1);
    float al0=__expf(m0-nm0), al1=__expf(m1-nm1);
    // ---- P = exp(sc - m) ----
    float ls0=0.f, ls1=0.f;
    #pragma unroll
    for(int nt=0;nt<8;nt++){
      sc[nt][0]=__expf(sc[nt][0]-nm0); ls0+=sc[nt][0];
      sc[nt][1]=__expf(sc[nt][1]-nm0); ls0+=sc[nt][1];
      sc[nt][2]=__expf(sc[nt][2]-nm1); ls1+=sc[nt][2];
      sc[nt][3]=__expf(sc[nt][3]-nm1); ls1+=sc[nt][3];
    }
    ls0+=__shfl_xor_sync(0xffffffff,ls0,1); ls0+=__shfl_xor_sync(0xffffffff,ls0,2);
    ls1+=__shfl_xor_sync(0xffffffff,ls1,1); ls1+=__shfl_xor_sync(0xffffffff,ls1,2);
    l0=l0*al0+ls0; l1=l1*al1+ls1;
    m0=nm0; m1=nm1;
    // ---- rescale O ----
    #pragma unroll
    for(int i=0;i<16;i++){ o[i][0]*=al0; o[i][1]*=al0; o[i][2]*=al1; o[i][3]*=al1; }
    // ---- O += P @ V ----
    #pragma unroll
    for(int kt=0;kt<4;kt++){
      uint32_t a0=packf(sc[2*kt][0],  sc[2*kt][1]);
      uint32_t a1=packf(sc[2*kt][2],  sc[2*kt][3]);
      uint32_t a2=packf(sc[2*kt+1][0],sc[2*kt+1][1]);
      uint32_t a3=packf(sc[2*kt+1][2],sc[2*kt+1][3]);
      int key0=16*kt+tig*2, key8=key0+8;
      #pragma unroll
      for(int ntv=0;ntv<16;ntv++){
        int d=ntv*8+gid;
        uint32_t b0=*reinterpret_cast<const uint32_t*>(&Vs2[d*SK+key0]);
        uint32_t b1=*reinterpret_cast<const uint32_t*>(&Vs2[d*SK+key8]);
        mma16816(o[ntv][0],o[ntv][1],o[ntv][2],o[ntv][3], a0,a1,a2,a3, b0,b1);
      }
    }
    __syncthreads();
  }

  // ---- finalize ----
  float inv0 = (l0>0.f)?1.f/l0:0.f;
  float inv1 = (l1>0.f)?1.f/l1:0.f;
  int row0=qbase+warp*16+gid;
  int row1=row0+8;
  #pragma unroll
  for(int ntv=0;ntv<16;ntv++){
    int d0=ntv*8+tig*2, d1=d0+1;
    if(row0<S){
      Obase[(size_t)row0*Dc+d0]=__float2bfloat16(o[ntv][0]*inv0);
      Obase[(size_t)row0*Dc+d1]=__float2bfloat16(o[ntv][1]*inv0);
    }
    if(row1<S){
      Obase[(size_t)row1*Dc+d0]=__float2bfloat16(o[ntv][2]*inv1);
      Obase[(size_t)row1*Dc+d1]=__float2bfloat16(o[ntv][3]*inv1);
    }
  }
  if(tig==0){
    if(row0<S) LSEbase[row0]=m0+logf(l0);
    if(row1<S) LSEbase[row1]=m1+logf(l1);
  }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE){
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int B = Q.size(0);
  int H = Q.size(1);
  int S = Q.size(2);

  const __nv_bfloat16* Qp=static_cast<const __nv_bfloat16*>(Q.data_ptr());
  const __nv_bfloat16* Kp=static_cast<const __nv_bfloat16*>(K.data_ptr());
  const __nv_bfloat16* Vp=static_cast<const __nv_bfloat16*>(V.data_ptr());
  __nv_bfloat16* Op=static_cast<__nv_bfloat16*>(O.data_ptr());
  float* Lp=static_cast<float*>(LSE.data_ptr());

  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

  dim3 grid((S+BQ-1)/BQ, H, B);
  dim3 block(128);
  size_t smem = (size_t)(BQ*SD + BK*SD + Dc*SK) * sizeof(__nv_bfloat16);

  static bool attr_set = false;
  if(!attr_set){
    CUDA_CHECK(cudaFuncSetAttribute(mha_kernel_fn,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    attr_set = true;
  }

  mha_kernel_fn<<<grid, block, smem, stream>>>(Qp,Kp,Vp,Op,Lp,S,H);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_kernel::run);

}  // namespace mha_kernel