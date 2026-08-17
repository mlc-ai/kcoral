#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
    fprintf(stderr,"CUDA error %s at %s:%d\n", cudaGetErrorString(_e),__FILE__,__LINE__); } } while(0)

namespace mha_kernel {

constexpr int Dh = 128;
constexpr int BM = 64;
constexpr int BN = 64;

__device__ __forceinline__ void mma_m16n8k16(
    float &c0,float &c1,float &c2,float &c3,
    uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,
    uint32_t b0,uint32_t b1) {
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}

__device__ __forceinline__ uint32_t pack_bf16x2(__nv_bfloat16 a, __nv_bfloat16 b){
  uint16_t ba = *reinterpret_cast<uint16_t*>(&a);
  uint16_t bb = *reinterpret_cast<uint16_t*>(&b);
  return (uint32_t)ba | ((uint32_t)bb << 16);
}

__global__ void attn_kernel(const __nv_bfloat16* __restrict__ Q,
                            const __nv_bfloat16* __restrict__ K,
                            const __nv_bfloat16* __restrict__ V,
                            __nv_bfloat16* __restrict__ O,
                            float* __restrict__ LSE,
                            int S, int H) {
  int b = blockIdx.z;
  int h = blockIdx.y;
  int qb = blockIdx.x;

  int tid = threadIdx.x;
  int warp = tid >> 5;
  int lane = tid & 31;
  int gid = lane >> 2;   // groupID 0..7
  int tig = lane & 3;    // threadID_in_group 0..3

  size_t bh_off = ((size_t)(b*H + h) * S) * Dh;
  const __nv_bfloat16* Qbh = Q + bh_off;
  const __nv_bfloat16* Kbh = K + bh_off;
  const __nv_bfloat16* Vbh = V + bh_off;
  __nv_bfloat16* Obh = O + bh_off;
  size_t lse_base = (size_t)(b*H + h) * S;

  extern __shared__ char smem[];
  __nv_bfloat16* sQ = (__nv_bfloat16*)smem;
  __nv_bfloat16* sK = sQ + BM*Dh;
  __nv_bfloat16* sV = sK + BN*Dh;
  float* sS = (float*)(sV + BN*Dh);
  __nv_bfloat16* sP = (__nv_bfloat16*)(sS + BM*BN);
  float* sM = (float*)(sP + BM*BN);
  float* sL = sM + BM;
  float* sFac = sL + BM;

  const float scale = rsqrtf((float)Dh); // 1/sqrt(128)

  float oacc[Dh/8][4];
  #pragma unroll
  for (int nt=0; nt<Dh/8; nt++){ oacc[nt][0]=0.f;oacc[nt][1]=0.f;oacc[nt][2]=0.f;oacc[nt][3]=0.f; }

  // load Q tile
  for (int i=tid; i<BM*Dh/8; i+=128){
    int row = i/(Dh/8);
    int c8 = (i%(Dh/8))*8;
    int qpos = qb*BM + row;
    int4 v = make_int4(0,0,0,0);
    if (qpos < S) v = *(const int4*)(Qbh + (size_t)qpos*Dh + c8);
    *((int4*)sQ + i) = v;
  }
  if (tid < BM){ sM[tid]=-INFINITY; sL[tid]=0.f; }
  __syncthreads();

  for (int kb=0; kb<=qb; kb++){
    int kbase = kb*BN;
    // load K,V
    for (int i=tid; i<BN*Dh/8; i+=128){
      int row=i/(Dh/8); int c8=(i%(Dh/8))*8;
      int kpos=kbase+row;
      int4 vk=make_int4(0,0,0,0), vv=make_int4(0,0,0,0);
      if (kpos<S){ vk=*(const int4*)(Kbh+(size_t)kpos*Dh+c8); vv=*(const int4*)(Vbh+(size_t)kpos*Dh+c8); }
      *((int4*)sK+i)=vk;
      *((int4*)sV+i)=vv;
    }
    __syncthreads();

    // QK^T  ->  sS
    #pragma unroll
    for (int nt=0; nt<BN/8; nt++){
      float c0=0,c1=0,c2=0,c3=0;
      #pragma unroll
      for (int kt=0; kt<Dh/16; kt++){
        int kcol = kt*16 + tig*2;
        int rA0 = warp*16 + gid;
        int rA1 = warp*16 + gid + 8;
        uint32_t a0=*(const uint32_t*)(sQ + rA0*Dh + kcol);
        uint32_t a1=*(const uint32_t*)(sQ + rA1*Dh + kcol);
        uint32_t a2=*(const uint32_t*)(sQ + rA0*Dh + kcol + 8);
        uint32_t a3=*(const uint32_t*)(sQ + rA1*Dh + kcol + 8);
        int nrow = nt*8 + gid;
        uint32_t b0=*(const uint32_t*)(sK + nrow*Dh + kcol);
        uint32_t b1=*(const uint32_t*)(sK + nrow*Dh + kcol + 8);
        mma_m16n8k16(c0,c1,c2,c3,a0,a1,a2,a3,b0,b1);
      }
      int col0=nt*8+tig*2;
      int rlo=warp*16+gid, rhi=warp*16+gid+8;
      sS[rlo*BN+col0]   = c0*scale;
      sS[rlo*BN+col0+1] = c1*scale;
      sS[rhi*BN+col0]   = c2*scale;
      sS[rhi*BN+col0+1] = c3*scale;
    }
    __syncthreads();

    // softmax (one thread per row)
    if (tid < BM){
      int r=tid; int qpos=qb*BM+r;
      if (qpos>=S){ sFac[r]=1.f; }
      else {
        float m_old=sM[r], l_old=sL[r];
        float m_blk=-INFINITY;
        for (int c=0;c<BN;c++){
          int kpos=kbase+c;
          if (kpos<=qpos){ float s=sS[r*BN+c]; m_blk=fmaxf(m_blk,s); }
        }
        float m_new=fmaxf(m_old,m_blk);
        float factor = (m_old==-INFINITY)?0.f:__expf(m_old-m_new);
        float sumP=0.f;
        for (int c=0;c<BN;c++){
          int kpos=kbase+c;
          float p=0.f;
          if (kpos<=qpos){ float s=sS[r*BN+c]; p=__expf(s-m_new); }
          sumP+=p;
          sP[r*BN+c]=__float2bfloat16(p);
        }
        float l_new=l_old*factor+sumP;
        sM[r]=m_new; sL[r]=l_new; sFac[r]=factor;
      }
    }
    __syncthreads();

    // rescale running O accumulator
    float fac_lo=sFac[warp*16+gid];
    float fac_hi=sFac[warp*16+gid+8];
    #pragma unroll
    for (int nt=0; nt<Dh/8; nt++){
      oacc[nt][0]*=fac_lo; oacc[nt][1]*=fac_lo;
      oacc[nt][2]*=fac_hi; oacc[nt][3]*=fac_hi;
    }

    // P @ V  -> accumulate into oacc
    const uint16_t* sV16=(const uint16_t*)sV;
    #pragma unroll
    for (int nt=0; nt<Dh/8; nt++){
      #pragma unroll
      for (int kt=0; kt<BN/16; kt++){
        int kcol=kt*16+tig*2;
        int rA0=warp*16+gid, rA1=warp*16+gid+8;
        uint32_t a0=*(const uint32_t*)(sP + rA0*BN + kcol);
        uint32_t a1=*(const uint32_t*)(sP + rA1*BN + kcol);
        uint32_t a2=*(const uint32_t*)(sP + rA0*BN + kcol + 8);
        uint32_t a3=*(const uint32_t*)(sP + rA1*BN + kcol + 8);
        int ncol=nt*8+gid;
        int kr0=kt*16+tig*2;
        int kr1=kt*16+tig*2+8;
        uint32_t b0=(uint32_t)sV16[kr0*Dh+ncol] | ((uint32_t)sV16[(kr0+1)*Dh+ncol]<<16);
        uint32_t b1=(uint32_t)sV16[kr1*Dh+ncol] | ((uint32_t)sV16[(kr1+1)*Dh+ncol]<<16);
        mma_m16n8k16(oacc[nt][0],oacc[nt][1],oacc[nt][2],oacc[nt][3],a0,a1,a2,a3,b0,b1);
      }
    }
    __syncthreads();
  }

  // finalize: normalize & write O, write LSE
  int rlo=warp*16+gid, rhi=warp*16+gid+8;
  int qpos_lo=qb*BM+rlo, qpos_hi=qb*BM+rhi;
  float inv_lo = (qpos_lo<S)?(1.f/sL[rlo]):0.f;
  float inv_hi = (qpos_hi<S)?(1.f/sL[rhi]):0.f;
  #pragma unroll
  for (int nt=0; nt<Dh/8; nt++){
    int col0=nt*8+tig*2;
    if (qpos_lo<S){
      __nv_bfloat16 v0=__float2bfloat16(oacc[nt][0]*inv_lo);
      __nv_bfloat16 v1=__float2bfloat16(oacc[nt][1]*inv_lo);
      *(uint32_t*)(Obh + (size_t)qpos_lo*Dh + col0)=pack_bf16x2(v0,v1);
    }
    if (qpos_hi<S){
      __nv_bfloat16 v2=__float2bfloat16(oacc[nt][2]*inv_hi);
      __nv_bfloat16 v3=__float2bfloat16(oacc[nt][3]*inv_hi);
      *(uint32_t*)(Obh + (size_t)qpos_hi*Dh + col0)=pack_bf16x2(v2,v3);
    }
  }
  if (tid<BM){
    int qpos=qb*BM+tid;
    if (qpos<S) LSE[lse_base+qpos]=sM[tid]+logf(sL[tid]);
  }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE){
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int B = Q.size(0);
  int H = Q.size(1);
  int S = Q.size(2);

  const __nv_bfloat16* Qp=(const __nv_bfloat16*)Q.data_ptr();
  const __nv_bfloat16* Kp=(const __nv_bfloat16*)K.data_ptr();
  const __nv_bfloat16* Vp=(const __nv_bfloat16*)V.data_ptr();
  __nv_bfloat16* Op=(__nv_bfloat16*)O.data_ptr();
  float* Lp=(float*)LSE.data_ptr();

  int numQB=(S+BM-1)/BM;
  dim3 grid(numQB,H,B);
  dim3 block(128);
  size_t smem = (size_t)(BM*Dh + BN*Dh + BN*Dh)*sizeof(__nv_bfloat16)
              + (size_t)(BM*BN)*sizeof(float)
              + (size_t)(BM*BN)*sizeof(__nv_bfloat16)
              + (size_t)(3*BM)*sizeof(float);
  cudaFuncSetAttribute(attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);

  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
  attn_kernel<<<grid,block,smem,stream>>>(Qp,Kp,Vp,Op,Lp,S,H);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_kernel::run);

} // namespace mha_kernel