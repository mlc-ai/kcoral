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

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int Dh = 128;
constexpr int NTHREADS = 128;
constexpr int VEC = 8; // bf16 per int4
constexpr int QSTRIDE = Dh + 8;   // 136
constexpr int KSTRIDE = Dh + 8;   // 136
constexpr int VSTRIDE = Dh + 8;   // 136
constexpr int PSTRIDE = BN + 8;   // 72

__device__ __forceinline__ void mma_m16n8k16(
    float &d0, float &d1, float &d2, float &d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1) {
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)
    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}

__global__ __launch_bounds__(128,1) void attn_kernel(
    const __nv_bfloat16* __restrict__ Qg,
    const __nv_bfloat16* __restrict__ Kg,
    const __nv_bfloat16* __restrict__ Vg,
    __nv_bfloat16* __restrict__ Og,
    float* __restrict__ LSEg,
    int B, int H, int S) {

  const int b = blockIdx.z;
  const int h = blockIdx.y;
  const int m0 = blockIdx.x * BM;

  const int t = threadIdx.x;
  const int warp = t >> 5;
  const int lane = t & 31;
  const int groupID = lane >> 2;   // 0..7
  const int tid = lane & 3;        // 0..3

  const float scale = rsqrtf((float)Dh);
  const float NEG = -1e30f;

  extern __shared__ char smem_raw[];
  __nv_bfloat16* Q_smem = (__nv_bfloat16*)smem_raw;
  __nv_bfloat16* K_smem = Q_smem + BM*QSTRIDE;
  __nv_bfloat16* V_smem = K_smem + BN*KSTRIDE;
  __nv_bfloat16* P_smem = V_smem + BN*VSTRIDE;

  const long head_off = ((long)b*H + h) * (long)S * Dh;
  const int vpr = Dh/VEC; // 16 vecs per row

  // ---- Load Q tile ----
  for (int i=t; i<BM*vpr; i+=NTHREADS) {
    int row=i/vpr, colv=i%vpr, col=colv*VEC;
    int grow=m0+row;
    int4 val = (grow<S) ? *(const int4*)&Qg[head_off + (long)grow*Dh + col]
                        : make_int4(0,0,0,0);
    *(int4*)&Q_smem[row*QSTRIDE + col] = val;
  }
  __syncthreads();

  // ---- Preload Q fragments (persist across KV loop) ----
  uint32_t Qf[8][4];
  {
    int rA = (warp*16 + groupID)*QSTRIDE;
    int rB = (warp*16 + groupID + 8)*QSTRIDE;
    #pragma unroll
    for (int kt=0; kt<8; kt++) {
      Qf[kt][0] = *(uint32_t*)&Q_smem[rA + kt*16 + tid*2];
      Qf[kt][1] = *(uint32_t*)&Q_smem[rB + kt*16 + tid*2];
      Qf[kt][2] = *(uint32_t*)&Q_smem[rA + kt*16 + tid*2 + 8];
      Qf[kt][3] = *(uint32_t*)&Q_smem[rB + kt*16 + tid*2 + 8];
    }
  }

  float O[16][4];
  #pragma unroll
  for (int dt=0; dt<16; dt++){ O[dt][0]=0;O[dt][1]=0;O[dt][2]=0;O[dt][3]=0; }
  float mA=NEG, mB=NEG, lA=0.f, lB=0.f;

  int num_kv = (S + BN - 1)/BN;
  for (int kv=0; kv<num_kv; kv++) {
    int kv_start = kv*BN;

    // ---- Load K,V ----
    for (int i=t; i<BN*vpr; i+=NTHREADS) {
      int row=i/vpr, colv=i%vpr, col=colv*VEC;
      int grow=kv_start+row;
      int4 kval, vval;
      if (grow<S) {
        kval = *(const int4*)&Kg[head_off + (long)grow*Dh + col];
        vval = *(const int4*)&Vg[head_off + (long)grow*Dh + col];
      } else { kval=make_int4(0,0,0,0); vval=make_int4(0,0,0,0); }
      *(int4*)&K_smem[row*KSTRIDE+col] = kval;
      *(int4*)&V_smem[row*VSTRIDE+col] = vval;
    }
    __syncthreads();

    // ---- S = Q @ K^T ----
    float Sreg[8][4];
    #pragma unroll
    for (int nt=0; nt<8; nt++) {
      float d0=0,d1=0,d2=0,d3=0;
      int krow = (nt*8 + groupID)*KSTRIDE;
      #pragma unroll
      for (int kt=0; kt<8; kt++) {
        uint32_t b0 = *(uint32_t*)&K_smem[krow + kt*16 + tid*2];
        uint32_t b1 = *(uint32_t*)&K_smem[krow + kt*16 + tid*2 + 8];
        mma_m16n8k16(d0,d1,d2,d3, Qf[kt][0],Qf[kt][1],Qf[kt][2],Qf[kt][3], b0,b1);
      }
      int c0 = kv_start + nt*8 + tid*2;
      int c1 = c0+1;
      Sreg[nt][0] = (c0<S)? d0*scale : NEG;
      Sreg[nt][1] = (c1<S)? d1*scale : NEG;
      Sreg[nt][2] = (c0<S)? d2*scale : NEG;
      Sreg[nt][3] = (c1<S)? d3*scale : NEG;
    }

    // ---- block max ----
    float bmA=NEG, bmB=NEG;
    #pragma unroll
    for (int nt=0; nt<8; nt++){
      bmA = fmaxf(bmA, fmaxf(Sreg[nt][0], Sreg[nt][1]));
      bmB = fmaxf(bmB, fmaxf(Sreg[nt][2], Sreg[nt][3]));
    }
    bmA = fmaxf(bmA, __shfl_xor_sync(0xffffffff,bmA,1));
    bmA = fmaxf(bmA, __shfl_xor_sync(0xffffffff,bmA,2));
    bmB = fmaxf(bmB, __shfl_xor_sync(0xffffffff,bmB,1));
    bmB = fmaxf(bmB, __shfl_xor_sync(0xffffffff,bmB,2));

    float newA = fmaxf(mA,bmA), newB=fmaxf(mB,bmB);
    float corrA = __expf(mA-newA), corrB=__expf(mB-newB);
    #pragma unroll
    for (int dt=0; dt<16; dt++){ O[dt][0]*=corrA;O[dt][1]*=corrA;O[dt][2]*=corrB;O[dt][3]*=corrB; }
    lA*=corrA; lB*=corrB;

    // ---- P = exp(S - m), store to smem ----
    float sA=0.f,sB=0.f;
    #pragma unroll
    for (int nt=0; nt<8; nt++){
      float p0=__expf(Sreg[nt][0]-newA);
      float p1=__expf(Sreg[nt][1]-newA);
      float p2=__expf(Sreg[nt][2]-newB);
      float p3=__expf(Sreg[nt][3]-newB);
      sA+=p0+p1; sB+=p2+p3;
      int rA=(warp*16+groupID)*PSTRIDE + nt*8 + tid*2;
      int rB=(warp*16+groupID+8)*PSTRIDE + nt*8 + tid*2;
      P_smem[rA]   = __float2bfloat16(p0);
      P_smem[rA+1] = __float2bfloat16(p1);
      P_smem[rB]   = __float2bfloat16(p2);
      P_smem[rB+1] = __float2bfloat16(p3);
    }
    sA += __shfl_xor_sync(0xffffffff,sA,1); sA += __shfl_xor_sync(0xffffffff,sA,2);
    sB += __shfl_xor_sync(0xffffffff,sB,1); sB += __shfl_xor_sync(0xffffffff,sB,2);
    lA+=sA; lB+=sB; mA=newA; mB=newB;

    __syncwarp();

    // ---- O += P @ V ----
    #pragma unroll
    for (int dt=0; dt<16; dt++){
      float o0=O[dt][0],o1=O[dt][1],o2=O[dt][2],o3=O[dt][3];
      int ncol = dt*8 + groupID;
      #pragma unroll
      for (int kt=0; kt<4; kt++){
        int prA=(warp*16+groupID)*PSTRIDE + kt*16 + tid*2;
        int prB=(warp*16+groupID+8)*PSTRIDE + kt*16 + tid*2;
        uint32_t a0=*(uint32_t*)&P_smem[prA];
        uint32_t a1=*(uint32_t*)&P_smem[prB];
        uint32_t a2=*(uint32_t*)&P_smem[prA+8];
        uint32_t a3=*(uint32_t*)&P_smem[prB+8];
        uint16_t v0=*(uint16_t*)&V_smem[(kt*16+tid*2)*VSTRIDE + ncol];
        uint16_t v1=*(uint16_t*)&V_smem[(kt*16+tid*2+1)*VSTRIDE + ncol];
        uint16_t v2=*(uint16_t*)&V_smem[(kt*16+tid*2+8)*VSTRIDE + ncol];
        uint16_t v3=*(uint16_t*)&V_smem[(kt*16+tid*2+9)*VSTRIDE + ncol];
        uint32_t b0=(uint32_t)v0 | ((uint32_t)v1<<16);
        uint32_t b1=(uint32_t)v2 | ((uint32_t)v3<<16);
        mma_m16n8k16(o0,o1,o2,o3, a0,a1,a2,a3, b0,b1);
      }
      O[dt][0]=o0;O[dt][1]=o1;O[dt][2]=o2;O[dt][3]=o3;
    }
    __syncthreads();
  }

  // ---- finalize ----
  float invA = (lA>0.f)? 1.f/lA : 0.f;
  float invB = (lB>0.f)? 1.f/lB : 0.f;
  int grA = m0 + warp*16 + groupID;
  int grB = m0 + warp*16 + groupID + 8;
  #pragma unroll
  for (int dt=0; dt<16; dt++){
    int col = dt*8 + tid*2;
    if (grA < S) {
      Og[head_off + (long)grA*Dh + col]   = __float2bfloat16(O[dt][0]*invA);
      Og[head_off + (long)grA*Dh + col+1] = __float2bfloat16(O[dt][1]*invA);
    }
    if (grB < S) {
      Og[head_off + (long)grB*Dh + col]   = __float2bfloat16(O[dt][2]*invB);
      Og[head_off + (long)grB*Dh + col+1] = __float2bfloat16(O[dt][3]*invB);
    }
  }
  if (tid==0) {
    long lse_head = ((long)b*H + h)*(long)S;
    if (grA < S) LSEg[lse_head + grA] = mA + logf(lA);
    if (grB < S) LSEg[lse_head + grB] = mB + logf(lB);
  }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int Bn = (int)Q.size(0);
  int Hn = (int)Q.size(1);
  int Sn = (int)Q.size(2);

  const __nv_bfloat16* Qp = static_cast<const __nv_bfloat16*>(Q.data_ptr());
  const __nv_bfloat16* Kp = static_cast<const __nv_bfloat16*>(K.data_ptr());
  const __nv_bfloat16* Vp = static_cast<const __nv_bfloat16*>(V.data_ptr());
  __nv_bfloat16* Op = static_cast<__nv_bfloat16*>(O.data_ptr());
  float* LSEp = static_cast<float*>(LSE.data_ptr());

  int num_m = (Sn + BM - 1)/BM;
  dim3 grid(num_m, Hn, Bn);

  int smem_bytes = (BM*QSTRIDE + BN*KSTRIDE + BN*VSTRIDE + BM*PSTRIDE) * (int)sizeof(__nv_bfloat16);

  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

  CUDA_CHECK(cudaFuncSetAttribute(attn_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

  attn_kernel<<<grid, NTHREADS, smem_bytes, stream>>>(Qp, Kp, Vp, Op, LSEp, Bn, Hn, Sn);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_kernel::run);

}  // namespace mha_kernel