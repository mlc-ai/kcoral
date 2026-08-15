#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__); exit(1);} }while(0)

namespace mha_kernel {

constexpr int D    = 128;
constexpr int BM   = 64;
constexpr int BN   = 64;
constexpr float LOG2E = 1.4426950408889634f;
constexpr float LN2   = 0.6931471805599453f;

__device__ __forceinline__ uint32_t pack2bf16(float x, float y){
  __nv_bfloat162 v = __floats2bfloat162_rn(x, y);
  return *reinterpret_cast<uint32_t*>(&v);
}

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

__device__ __forceinline__ float quad_max(float v){
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
  return v;
}
__device__ __forceinline__ float quad_sum(float v){
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  return v;
}

__global__ void __launch_bounds__(128) mha_kernel_fn(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int S, int H, float scale){

  extern __shared__ __align__(16) char smem_raw[];
  __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  __nv_bfloat16* sK = sQ + BM*D;
  __nv_bfloat16* sV = sK + BN*D;

  int qblock = blockIdx.x;
  int h = blockIdx.y;
  int b = blockIdx.z;
  int qstart = qblock*BM;

  int tid = threadIdx.x;
  int warp = tid >> 5;
  int lane = tid & 31;
  int groupID = lane >> 2;
  int tig = lane & 3;

  int64_t bh = (int64_t)(b*H + h);
  const __nv_bfloat16* Qbh = Q + bh*S*D;
  const __nv_bfloat16* Kbh = K + bh*S*D;
  const __nv_bfloat16* Vbh = V + bh*S*D;
  __nv_bfloat16* Obh = O + bh*S*D;
  float* LSEbh = LSE + bh*S;

  float SC = scale * LOG2E;

  float Oacc[16][4];
  #pragma unroll
  for(int dt=0; dt<16; dt++){ Oacc[dt][0]=0;Oacc[dt][1]=0;Oacc[dt][2]=0;Oacc[dt][3]=0; }
  float m_g = -INFINITY, m_g8 = -INFINITY;
  float l_g = 0.f, l_g8 = 0.f;

  // Load Q tile
  for(int v = tid; v < (BM*D)/8; v += blockDim.x){
    int row = v >> 4;            // 16 vec per row
    int col = (v & 15) << 3;
    int qrow = qstart + row;
    uint4 data;
    if(qrow < S) data = *reinterpret_cast<const uint4*>(&Qbh[(int64_t)qrow*D + col]);
    else { data.x=data.y=data.z=data.w=0; }
    *reinterpret_cast<uint4*>(&sQ[row*D + col]) = data;
  }

  int num_kb = (S + BN - 1)/BN;
  for(int kb=0; kb<num_kb; kb++){
    int kstart = kb*BN;
    __syncthreads();
    for(int v = tid; v < (BN*D)/8; v += blockDim.x){
      int row = v >> 4;
      int col = (v & 15) << 3;
      int krow = kstart + row;
      uint4 kd, vd;
      if(krow < S){
        kd = *reinterpret_cast<const uint4*>(&Kbh[(int64_t)krow*D + col]);
        vd = *reinterpret_cast<const uint4*>(&Vbh[(int64_t)krow*D + col]);
      } else {
        kd.x=kd.y=kd.z=kd.w=0; vd.x=vd.y=vd.z=vd.w=0;
      }
      *reinterpret_cast<uint4*>(&sK[row*D + col]) = kd;
      *reinterpret_cast<uint4*>(&sV[row*D + col]) = vd;
    }
    __syncthreads();

    // ---- QK^T ----
    float Sacc[8][4];
    #pragma unroll
    for(int nt=0;nt<8;nt++){Sacc[nt][0]=0;Sacc[nt][1]=0;Sacc[nt][2]=0;Sacc[nt][3]=0;}
    int rg  = warp*16 + groupID;
    int rg8 = rg + 8;
    #pragma unroll
    for(int kt2=0; kt2<8; kt2++){
      int col = kt2*16 + 2*tig;
      uint32_t a0 = *reinterpret_cast<uint32_t*>(&sQ[rg*D  + col]);
      uint32_t a1 = *reinterpret_cast<uint32_t*>(&sQ[rg8*D + col]);
      uint32_t a2 = *reinterpret_cast<uint32_t*>(&sQ[rg*D  + col + 8]);
      uint32_t a3 = *reinterpret_cast<uint32_t*>(&sQ[rg8*D + col + 8]);
      #pragma unroll
      for(int nt=0; nt<8; nt++){
        int keycol = nt*8 + groupID;
        uint32_t b0 = *reinterpret_cast<uint32_t*>(&sK[keycol*D + col]);
        uint32_t b1 = *reinterpret_cast<uint32_t*>(&sK[keycol*D + col + 8]);
        mma_m16n8k16(Sacc[nt][0],Sacc[nt][1],Sacc[nt][2],Sacc[nt][3],
                     a0,a1,a2,a3,b0,b1,
                     Sacc[nt][0],Sacc[nt][1],Sacc[nt][2],Sacc[nt][3]);
      }
    }

    // scale + mask + rowmax (base-2)
    float y[8][4];
    float lmax_g = -INFINITY, lmax_g8 = -INFINITY;
    #pragma unroll
    for(int nt=0; nt<8; nt++){
      int gk0 = kstart + nt*8 + 2*tig;
      int gk1 = gk0 + 1;
      float y0 = (gk0 < S) ? Sacc[nt][0]*SC : -INFINITY;
      float y1 = (gk1 < S) ? Sacc[nt][1]*SC : -INFINITY;
      float y2 = (gk0 < S) ? Sacc[nt][2]*SC : -INFINITY;
      float y3 = (gk1 < S) ? Sacc[nt][3]*SC : -INFINITY;
      y[nt][0]=y0; y[nt][1]=y1; y[nt][2]=y2; y[nt][3]=y3;
      lmax_g  = fmaxf(lmax_g,  fmaxf(y0,y1));
      lmax_g8 = fmaxf(lmax_g8, fmaxf(y2,y3));
    }
    lmax_g  = quad_max(lmax_g);
    lmax_g8 = quad_max(lmax_g8);

    float m_new_g  = fmaxf(m_g,  lmax_g);
    float m_new_g8 = fmaxf(m_g8, lmax_g8);
    float corr_g  = exp2f(m_g  - m_new_g);
    float corr_g8 = exp2f(m_g8 - m_new_g8);

    #pragma unroll
    for(int dt=0; dt<16; dt++){
      Oacc[dt][0]*=corr_g;  Oacc[dt][1]*=corr_g;
      Oacc[dt][2]*=corr_g8; Oacc[dt][3]*=corr_g8;
    }
    l_g  *= corr_g;  l_g8 *= corr_g8;

    uint32_t pa[8][2];
    float sum_g=0.f, sum_g8=0.f;
    #pragma unroll
    for(int nt=0; nt<8; nt++){
      float p0 = exp2f(y[nt][0]-m_new_g);
      float p1 = exp2f(y[nt][1]-m_new_g);
      float p2 = exp2f(y[nt][2]-m_new_g8);
      float p3 = exp2f(y[nt][3]-m_new_g8);
      sum_g  += p0+p1;  sum_g8 += p2+p3;
      pa[nt][0] = pack2bf16(p0,p1);
      pa[nt][1] = pack2bf16(p2,p3);
    }
    sum_g  = quad_sum(sum_g);
    sum_g8 = quad_sum(sum_g8);
    l_g  += sum_g;  l_g8 += sum_g8;
    m_g  = m_new_g; m_g8 = m_new_g8;

    // ---- P @ V ----
    #pragma unroll
    for(int kt=0; kt<BN/16; kt++){
      uint32_t pa0 = pa[2*kt][0];
      uint32_t pa1 = pa[2*kt][1];
      uint32_t pa2 = pa[2*kt+1][0];
      uint32_t pa3 = pa[2*kt+1][1];
      int keyrow = kt*16 + 2*tig;
      #pragma unroll
      for(int dt=0; dt<16; dt++){
        int dcol = dt*8 + groupID;
        uint16_t v0 = *reinterpret_cast<uint16_t*>(&sV[(keyrow  )*D + dcol]);
        uint16_t v1 = *reinterpret_cast<uint16_t*>(&sV[(keyrow+1)*D + dcol]);
        uint16_t v2 = *reinterpret_cast<uint16_t*>(&sV[(keyrow+8)*D + dcol]);
        uint16_t v3 = *reinterpret_cast<uint16_t*>(&sV[(keyrow+9)*D + dcol]);
        uint32_t b0 = (uint32_t)v0 | ((uint32_t)v1 << 16);
        uint32_t b1 = (uint32_t)v2 | ((uint32_t)v3 << 16);
        mma_m16n8k16(Oacc[dt][0],Oacc[dt][1],Oacc[dt][2],Oacc[dt][3],
                     pa0,pa1,pa2,pa3,b0,b1,
                     Oacc[dt][0],Oacc[dt][1],Oacc[dt][2],Oacc[dt][3]);
      }
    }
  }

  // finalize
  float inv_g  = (l_g >0.f)? 1.0f/l_g  : 0.f;
  float inv_g8 = (l_g8>0.f)? 1.0f/l_g8 : 0.f;
  int qrow_g  = qstart + warp*16 + groupID;
  int qrow_g8 = qrow_g + 8;
  #pragma unroll
  for(int dt=0; dt<16; dt++){
    int d0 = dt*8 + 2*tig;
    int d1 = d0+1;
    if(qrow_g < S){
      Obh[(int64_t)qrow_g*D + d0] = __float2bfloat16(Oacc[dt][0]*inv_g);
      Obh[(int64_t)qrow_g*D + d1] = __float2bfloat16(Oacc[dt][1]*inv_g);
    }
    if(qrow_g8 < S){
      Obh[(int64_t)qrow_g8*D + d0] = __float2bfloat16(Oacc[dt][2]*inv_g8);
      Obh[(int64_t)qrow_g8*D + d1] = __float2bfloat16(Oacc[dt][3]*inv_g8);
    }
  }
  if(tig==0){
    if(qrow_g  < S) LSEbh[qrow_g]  = m_g  * LN2 + logf(l_g);
    if(qrow_g8 < S) LSEbh[qrow_g8] = m_g8 * LN2 + logf(l_g8);
  }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE){
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int Bsz = (int)Q.size(0);
  int Hsz = (int)Q.size(1);
  int S   = (int)Q.size(2);
  int Dsz = (int)Q.size(3);

  const __nv_bfloat16* Qp = static_cast<const __nv_bfloat16*>(Q.data_ptr());
  const __nv_bfloat16* Kp = static_cast<const __nv_bfloat16*>(K.data_ptr());
  const __nv_bfloat16* Vp = static_cast<const __nv_bfloat16*>(V.data_ptr());
  __nv_bfloat16* Op = static_cast<__nv_bfloat16*>(O.data_ptr());
  float* LSEp = static_cast<float*>(LSE.data_ptr());

  float scale = 1.0f / sqrtf((float)Dsz);

  dim3 grid((S + BM - 1)/BM, Hsz, Bsz);
  int smem = 3 * BN * D * (int)sizeof(__nv_bfloat16); // BM == BN

  static bool attr_set = false;
  if(!attr_set){
    cudaFuncSetAttribute(mha_kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    attr_set = true;
  }

  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

  mha_kernel_fn<<<grid, 128, smem, stream>>>(Qp, Kp, Vp, Op, LSEp, S, Hsz, scale);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_kernel::run);

}  // namespace mha_kernel