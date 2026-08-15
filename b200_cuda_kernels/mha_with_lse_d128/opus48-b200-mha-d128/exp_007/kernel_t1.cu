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

constexpr int BM = 128;
constexpr int BN = 64;
constexpr int Dh = 128;
constexpr int NWARP = 8;
constexpr int NTHREADS = 256;
constexpr int QSTRIDE = Dh + 8;   // 136
constexpr int KSTRIDE = Dh + 8;   // 136
constexpr int VSTRIDE = Dh + 8;   // 136
constexpr int PSTRIDE = BN + 8;   // 72
constexpr float LOG2E = 1.4426950408889634f;
constexpr float LN2   = 0.6931471805599453f;

__device__ __forceinline__ float ex2(float x){ float y; asm volatile("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x)); return y; }

__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem){
  unsigned s=(unsigned)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n"::"r"(s),"l"(gmem));
}
template<int N> __device__ __forceinline__ void cp_wait(){ asm volatile("cp.async.wait_group %0;\n"::"n"(N):"memory"); }
__device__ __forceinline__ void cp_commit(){ asm volatile("cp.async.commit_group;\n":::"memory"); }

__device__ __forceinline__ void ldmatrix_trans_x4(uint32_t &r0,uint32_t &r1,uint32_t &r2,uint32_t &r3,uint32_t addr){
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3},[%4];"
    :"=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3):"r"(addr));
}

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

__global__ __launch_bounds__(256,1) void attn_kernel(
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
  const int gid = lane >> 2;   // 0..7
  const int tid = lane & 3;    // 0..3

  const float combined = rsqrtf((float)Dh) * LOG2E;
  const float NEG = -1e30f;

  extern __shared__ char smem[];
  __nv_bfloat16* Q_smem = (__nv_bfloat16*)smem;
  __nv_bfloat16* K_smem = Q_smem + BM*QSTRIDE;
  __nv_bfloat16* V_smem = K_smem + 2*BN*KSTRIDE;
  __nv_bfloat16* P_smem = V_smem + 2*BN*VSTRIDE;

  const long head_off = ((long)b*H + h) * (long)S * Dh;
  const int vpr = Dh/8; // 16

  // ---- Load Q tile once ----
  for (int i=t; i<BM*vpr; i+=NTHREADS) {
    int row=i/vpr, col=(i%vpr)*8;
    int grow=m0+row;
    int4 val = (grow<S) ? *(const int4*)&Qg[head_off + (long)grow*Dh + col] : make_int4(0,0,0,0);
    *(int4*)&Q_smem[row*QSTRIDE + col] = val;
  }
  __syncthreads();

  // ---- Preload Q fragments ----
  uint32_t Qf[8][4];
  {
    int rA = (warp*16 + gid)*QSTRIDE;
    int rB = (warp*16 + gid + 8)*QSTRIDE;
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
  float mA2=NEG, mB2=NEG, lA=0.f, lB=0.f;

  int num_kv = (S + BN - 1)/BN;

  // ---- KV loader lambda-like macro ----
  auto load_kv = [&](int buf, int kv_start){
    for (int i=t; i<BN*vpr; i+=NTHREADS) {
      int row=i/vpr, col=(i%vpr)*8;
      int grow=kv_start+row;
      __nv_bfloat16* kdst=&K_smem[buf*BN*KSTRIDE + row*KSTRIDE + col];
      __nv_bfloat16* vdst=&V_smem[buf*BN*VSTRIDE + row*VSTRIDE + col];
      if (grow<S){
        cp_async_16(kdst, &Kg[head_off + (long)grow*Dh + col]);
        cp_async_16(vdst, &Vg[head_off + (long)grow*Dh + col]);
      } else {
        *(int4*)kdst = make_int4(0,0,0,0);
        *(int4*)vdst = make_int4(0,0,0,0);
      }
    }
  };

  int cur = 0;
  load_kv(0, 0);
  cp_commit();

  for (int kv=0; kv<num_kv; kv++) {
    int nxt = cur ^ 1;
    if (kv+1 < num_kv) { load_kv(nxt, (kv+1)*BN); cp_commit(); cp_wait<1>(); }
    else { cp_wait<0>(); }
    __syncthreads();

    int kv_start = kv*BN;
    int bufK = cur*BN*KSTRIDE;
    int bufV = cur*BN*VSTRIDE;

    // ---- S = Q @ K^T ----
    float Sr[8][4];
    #pragma unroll
    for (int nt=0; nt<8; nt++) {
      float d0=0,d1=0,d2=0,d3=0;
      int krow = (nt*8 + gid)*KSTRIDE;
      #pragma unroll
      for (int kt=0; kt<8; kt++) {
        uint32_t b0 = *(uint32_t*)&K_smem[bufK + krow + kt*16 + tid*2];
        uint32_t b1 = *(uint32_t*)&K_smem[bufK + krow + kt*16 + tid*2 + 8];
        mma_m16n8k16(d0,d1,d2,d3, Qf[kt][0],Qf[kt][1],Qf[kt][2],Qf[kt][3], b0,b1);
      }
      int c0 = kv_start + nt*8 + tid*2, c1 = c0+1;
      Sr[nt][0] = (c0<S)? d0*combined : NEG;
      Sr[nt][1] = (c1<S)? d1*combined : NEG;
      Sr[nt][2] = (c0<S)? d2*combined : NEG;
      Sr[nt][3] = (c1<S)? d3*combined : NEG;
    }

    // ---- block max ----
    float bmA=NEG, bmB=NEG;
    #pragma unroll
    for (int nt=0; nt<8; nt++){
      bmA = fmaxf(bmA, fmaxf(Sr[nt][0], Sr[nt][1]));
      bmB = fmaxf(bmB, fmaxf(Sr[nt][2], Sr[nt][3]));
    }
    bmA = fmaxf(bmA, __shfl_xor_sync(0xffffffff,bmA,1));
    bmA = fmaxf(bmA, __shfl_xor_sync(0xffffffff,bmA,2));
    bmB = fmaxf(bmB, __shfl_xor_sync(0xffffffff,bmB,1));
    bmB = fmaxf(bmB, __shfl_xor_sync(0xffffffff,bmB,2));

    float nA=fmaxf(mA2,bmA), nB=fmaxf(mB2,bmB);
    float cA=ex2(mA2-nA), cB=ex2(mB2-nB);
    #pragma unroll
    for (int dt=0; dt<16; dt++){ O[dt][0]*=cA;O[dt][1]*=cA;O[dt][2]*=cB;O[dt][3]*=cB; }
    lA*=cA; lB*=cB;

    // ---- P = 2^(S - m), store to smem ----
    float sA=0.f,sB=0.f;
    #pragma unroll
    for (int nt=0; nt<8; nt++){
      float p0=ex2(Sr[nt][0]-nA);
      float p1=ex2(Sr[nt][1]-nA);
      float p2=ex2(Sr[nt][2]-nB);
      float p3=ex2(Sr[nt][3]-nB);
      sA+=p0+p1; sB+=p2+p3;
      int keyc=nt*8 + tid*2;
      int rA=(warp*16+gid)*PSTRIDE + keyc;
      int rB=(warp*16+gid+8)*PSTRIDE + keyc;
      P_smem[rA]   = __float2bfloat16(p0);
      P_smem[rA+1] = __float2bfloat16(p1);
      P_smem[rB]   = __float2bfloat16(p2);
      P_smem[rB+1] = __float2bfloat16(p3);
    }
    sA += __shfl_xor_sync(0xffffffff,sA,1); sA += __shfl_xor_sync(0xffffffff,sA,2);
    sB += __shfl_xor_sync(0xffffffff,sB,1); sB += __shfl_xor_sync(0xffffffff,sB,2);
    lA+=sA; lB+=sB; mA2=nA; mB2=nB;

    __syncwarp();

    // ---- O += P @ V (V via ldmatrix.trans) ----
    #pragma unroll
    for (int kt=0; kt<4; kt++){
      int prA=(warp*16+gid)*PSTRIDE + kt*16 + tid*2;
      int prB=(warp*16+gid+8)*PSTRIDE + kt*16 + tid*2;
      uint32_t a0=*(uint32_t*)&P_smem[prA];
      uint32_t a1=*(uint32_t*)&P_smem[prB];
      uint32_t a2=*(uint32_t*)&P_smem[prA+8];
      uint32_t a3=*(uint32_t*)&P_smem[prB+8];
      int bn0=kt*16;
      int m=lane>>3, r=lane&7;
      int rowV=bn0 + ((m&1)*8) + r;
      #pragma unroll
      for (int dpair=0; dpair<8; dpair++){
        int colV=dpair*16 + ((m>>1)*8);
        uint32_t addr=(uint32_t)__cvta_generic_to_shared(&V_smem[bufV + rowV*VSTRIDE + colV]);
        uint32_t v0,v1,v2,v3;
        ldmatrix_trans_x4(v0,v1,v2,v3, addr);
        int dt0=2*dpair, dt1=2*dpair+1;
        mma_m16n8k16(O[dt0][0],O[dt0][1],O[dt0][2],O[dt0][3], a0,a1,a2,a3, v0,v1);
        mma_m16n8k16(O[dt1][0],O[dt1][1],O[dt1][2],O[dt1][3], a0,a1,a2,a3, v2,v3);
      }
    }
    __syncthreads();
    cur = nxt;
  }

  // ---- finalize ----
  float invA = (lA>0.f)? 1.f/lA : 0.f;
  float invB = (lB>0.f)? 1.f/lB : 0.f;
  int grA = m0 + warp*16 + gid;
  int grB = grA + 8;
  #pragma unroll
  for (int dt=0; dt<16; dt++){
    int col = dt*8 + tid*2;
    if (grA < S) {
      __nv_bfloat162 v; v.x=__float2bfloat16(O[dt][0]*invA); v.y=__float2bfloat16(O[dt][1]*invA);
      *(__nv_bfloat162*)&Og[head_off + (long)grA*Dh + col] = v;
    }
    if (grB < S) {
      __nv_bfloat162 v; v.x=__float2bfloat16(O[dt][2]*invB); v.y=__float2bfloat16(O[dt][3]*invB);
      *(__nv_bfloat162*)&Og[head_off + (long)grB*Dh + col] = v;
    }
  }
  if (tid==0) {
    long lse_head = ((long)b*H + h)*(long)S;
    if (grA < S) LSEg[lse_head + grA] = mA2*LN2 + logf(lA);
    if (grB < S) LSEg[lse_head + grB] = mB2*LN2 + logf(lB);
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

  int smem_bytes = (BM*QSTRIDE + 2*BN*KSTRIDE + 2*BN*VSTRIDE + BM*PSTRIDE) * (int)sizeof(__nv_bfloat16);

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