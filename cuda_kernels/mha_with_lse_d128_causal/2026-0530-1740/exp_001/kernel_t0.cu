#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
    fprintf(stderr,"CUDA error %s at %s:%d\n", cudaGetErrorString(_e),__FILE__,__LINE__);} } while(0)

namespace mha_kernel {

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int D  = 128;
constexpr int Dpad  = 136;  // 128 + 8 padding (bank-conflict avoidance)
constexpr int BNpad = 72;   // 64 + 8 padding

__device__ __forceinline__ float ex2f(float x){
  float y; asm volatile("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x)); return y;
}
__device__ __forceinline__ uint32_t ldb32(const __nv_bfloat16* p){
  return *reinterpret_cast<const uint32_t*>(p);
}
__device__ __forceinline__ uint32_t pack2bf16(float a, float b){
  __nv_bfloat162 v = __floats2bfloat162_rn(a,b);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ void mma16816(float &d0,float &d1,float &d2,float &d3,
   uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3, uint32_t b0,uint32_t b1){
  asm volatile(
   "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
   "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
   : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)
   : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}

__global__ __launch_bounds__(128,1)
void attn_kernel(const __nv_bfloat16* __restrict__ Q,
                 const __nv_bfloat16* __restrict__ K,
                 const __nv_bfloat16* __restrict__ V,
                 __nv_bfloat16* __restrict__ O,
                 float* __restrict__ LSE,
                 int B, int H, int S){
  const float SCALE  = 0.08838834764831845f; // 1/sqrt(128)
  const float LOG2E  = 1.4426950408889634f;
  const float SCALE2 = SCALE*LOG2E;
  const float LN2    = 0.6931471805599453f;

  extern __shared__ char smem_raw[];
  __nv_bfloat16* Qs = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  __nv_bfloat16* Ks = Qs + BM*Dpad;
  __nv_bfloat16* Vs = Ks + BN*Dpad;   // V transposed: Vs[d*BNpad + n] = V[n][d]

  int b = blockIdx.z;
  int h = blockIdx.y;
  int qb_base = blockIdx.x*BM;

  int tid  = threadIdx.x;
  int warp = tid >> 5;
  int lane = tid & 31;
  int gid  = lane >> 2;   // groupID (0..7)
  int tig  = lane & 3;    // threadID_in_group (0..3)

  size_t bh = (size_t)(b*H + h);
  const __nv_bfloat16* Qg = Q + bh*(size_t)S*D;
  const __nv_bfloat16* Kg = K + bh*(size_t)S*D;
  const __nv_bfloat16* Vg = V + bh*(size_t)S*D;
  __nv_bfloat16* Og = O + bh*(size_t)S*D;
  float* LSEg = LSE + bh*(size_t)S;

  // ---- Load Q tile once ----
  #pragma unroll
  for (int idx = tid; idx < BM*16; idx += 128){
    int row = idx >> 4, cvec = idx & 15, col = cvec*8;
    int gs = qb_base + row;
    int4 val;
    if (gs < S) val = *reinterpret_cast<const int4*>(Qg + (size_t)gs*D + col);
    else { val.x=val.y=val.z=val.w=0; }
    *reinterpret_cast<int4*>(&Qs[row*Dpad + col]) = val;
  }
  __syncthreads();

  // ---- Accumulators ----
  float Oacc[16][4];
  #pragma unroll
  for (int i=0;i<16;i++){ Oacc[i][0]=0.f;Oacc[i][1]=0.f;Oacc[i][2]=0.f;Oacc[i][3]=0.f; }
  float m0=-1e30f, m1=-1e30f, l0=0.f, l1=0.f;

  int q0g = qb_base + warp*16 + gid;
  int q1g = qb_base + warp*16 + gid + 8;
  int Qrow0 = warp*16 + gid;
  int Qrow1 = warp*16 + gid + 8;

  int q_last = qb_base + BM - 1;
  if (q_last > S-1) q_last = S-1;
  int kv_last = q_last / BN;

  for (int kbi=0; kbi<=kv_last; kbi++){
    int kb_base = kbi*BN;
    // ---- Load K ----
    #pragma unroll
    for (int idx = tid; idx < BN*16; idx += 128){
      int row = idx >> 4, cvec = idx & 15, col = cvec*8;
      int gs = kb_base + row;
      int4 val;
      if (gs < S) val = *reinterpret_cast<const int4*>(Kg + (size_t)gs*D + col);
      else { val.x=val.y=val.z=val.w=0; }
      *reinterpret_cast<int4*>(&Ks[row*Dpad + col]) = val;
    }
    // ---- Load V transposed ----
    #pragma unroll
    for (int idx = tid; idx < BN*16; idx += 128){
      int n = idx >> 4, cvec = idx & 15, d0 = cvec*8;
      int gs = kb_base + n;
      int4 val;
      if (gs < S) val = *reinterpret_cast<const int4*>(Vg + (size_t)gs*D + d0);
      else { val.x=val.y=val.z=val.w=0; }
      __nv_bfloat16* pv = reinterpret_cast<__nv_bfloat16*>(&val);
      #pragma unroll
      for (int j=0;j<8;j++) Vs[(d0+j)*BNpad + n] = pv[j];
    }
    __syncthreads();

    // ---- GEMM1: S = Q @ K^T  (contract D=128) ----
    float Sreg[8][4];
    #pragma unroll
    for (int nt=0;nt<8;nt++){ Sreg[nt][0]=0;Sreg[nt][1]=0;Sreg[nt][2]=0;Sreg[nt][3]=0; }
    #pragma unroll
    for (int kt=0;kt<8;kt++){
      uint32_t a0 = ldb32(&Qs[Qrow0*Dpad + kt*16 + tig*2]);
      uint32_t a1 = ldb32(&Qs[Qrow1*Dpad + kt*16 + tig*2]);
      uint32_t a2 = ldb32(&Qs[Qrow0*Dpad + kt*16 + tig*2 + 8]);
      uint32_t a3 = ldb32(&Qs[Qrow1*Dpad + kt*16 + tig*2 + 8]);
      #pragma unroll
      for (int nt=0;nt<8;nt++){
        int n = nt*8 + gid;
        uint32_t b0 = ldb32(&Ks[n*Dpad + kt*16 + tig*2]);
        uint32_t b1 = ldb32(&Ks[n*Dpad + kt*16 + tig*2 + 8]);
        mma16816(Sreg[nt][0],Sreg[nt][1],Sreg[nt][2],Sreg[nt][3], a0,a1,a2,a3, b0,b1);
      }
    }

    // ---- scale + causal mask + rowmax ----
    float tmax0=-1e30f, tmax1=-1e30f;
    #pragma unroll
    for (int nt=0;nt<8;nt++){
      int kc_a = kb_base + nt*8 + tig*2;
      int kc_b = kc_a + 1;
      float s0 = Sreg[nt][0]*SCALE2;
      float s1 = Sreg[nt][1]*SCALE2;
      float s2 = Sreg[nt][2]*SCALE2;
      float s3 = Sreg[nt][3]*SCALE2;
      if (kc_a > q0g || kc_a >= S) s0=-1e30f;
      if (kc_b > q0g || kc_b >= S) s1=-1e30f;
      if (kc_a > q1g || kc_a >= S) s2=-1e30f;
      if (kc_b > q1g || kc_b >= S) s3=-1e30f;
      Sreg[nt][0]=s0;Sreg[nt][1]=s1;Sreg[nt][2]=s2;Sreg[nt][3]=s3;
      tmax0 = fmaxf(tmax0, fmaxf(s0,s1));
      tmax1 = fmaxf(tmax1, fmaxf(s2,s3));
    }
    tmax0 = fmaxf(tmax0, __shfl_xor_sync(0xffffffff,tmax0,1));
    tmax0 = fmaxf(tmax0, __shfl_xor_sync(0xffffffff,tmax0,2));
    tmax1 = fmaxf(tmax1, __shfl_xor_sync(0xffffffff,tmax1,1));
    tmax1 = fmaxf(tmax1, __shfl_xor_sync(0xffffffff,tmax1,2));

    float mn0 = fmaxf(m0, tmax0);
    float mn1 = fmaxf(m1, tmax1);
    float corr0 = ex2f(m0 - mn0);
    float corr1 = ex2f(m1 - mn1);
    #pragma unroll
    for (int nd=0;nd<16;nd++){
      Oacc[nd][0]*=corr0; Oacc[nd][1]*=corr0;
      Oacc[nd][2]*=corr1; Oacc[nd][3]*=corr1;
    }
    l0*=corr0; l1*=corr1;
    m0=mn0; m1=mn1;

    // ---- P = exp2(S - m), pack to bf16 (register-only layout for GEMM2 A operand) ----
    uint32_t Pp[8][2];
    #pragma unroll
    for (int nt=0;nt<8;nt++){
      float p0 = ex2f(Sreg[nt][0]-mn0);
      float p1 = ex2f(Sreg[nt][1]-mn0);
      float p2 = ex2f(Sreg[nt][2]-mn1);
      float p3 = ex2f(Sreg[nt][3]-mn1);
      l0 += p0+p1;
      l1 += p2+p3;
      Pp[nt][0]=pack2bf16(p0,p1);
      Pp[nt][1]=pack2bf16(p2,p3);
    }

    // ---- GEMM2: O += P @ V  (contract BN=64) ----
    #pragma unroll
    for (int kt2=0;kt2<4;kt2++){
      uint32_t a0=Pp[2*kt2][0];
      uint32_t a1=Pp[2*kt2][1];
      uint32_t a2=Pp[2*kt2+1][0];
      uint32_t a3=Pp[2*kt2+1][1];
      #pragma unroll
      for (int nd=0;nd<16;nd++){
        int d = nd*8 + gid;
        uint32_t b0=ldb32(&Vs[d*BNpad + kt2*16 + tig*2]);
        uint32_t b1=ldb32(&Vs[d*BNpad + kt2*16 + tig*2 + 8]);
        mma16816(Oacc[nd][0],Oacc[nd][1],Oacc[nd][2],Oacc[nd][3], a0,a1,a2,a3, b0,b1);
      }
    }
    __syncthreads();
  }

  // ---- finalize: reduce rowsum across quad, normalize, write ----
  float ll0 = l0, ll1 = l1;
  ll0 += __shfl_xor_sync(0xffffffff,ll0,1); ll0 += __shfl_xor_sync(0xffffffff,ll0,2);
  ll1 += __shfl_xor_sync(0xffffffff,ll1,1); ll1 += __shfl_xor_sync(0xffffffff,ll1,2);
  float inv0 = 1.0f/ll0;
  float inv1 = 1.0f/ll1;

  #pragma unroll
  for (int nd=0;nd<16;nd++){
    int d0 = nd*8 + tig*2;
    float o0 = Oacc[nd][0]*inv0;
    float o1 = Oacc[nd][1]*inv0;
    float o2 = Oacc[nd][2]*inv1;
    float o3 = Oacc[nd][3]*inv1;
    if (q0g < S){
      __nv_bfloat162 v = __floats2bfloat162_rn(o0,o1);
      *reinterpret_cast<__nv_bfloat162*>(&Og[(size_t)q0g*D + d0]) = v;
    }
    if (q1g < S){
      __nv_bfloat162 v = __floats2bfloat162_rn(o2,o3);
      *reinterpret_cast<__nv_bfloat162*>(&Og[(size_t)q1g*D + d0]) = v;
    }
  }

  if (tig==0){
    if (q0g < S) LSEg[q0g] = m0*LN2 + logf(ll0);
    if (q1g < S) LSEg[q1g] = m1*LN2 + logf(ll1);
  }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE){
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int B = (int)Q.size(0);
  int H = (int)Q.size(1);
  int S = (int)Q.size(2);

  const __nv_bfloat16* Qp = static_cast<const __nv_bfloat16*>(Q.data_ptr());
  const __nv_bfloat16* Kp = static_cast<const __nv_bfloat16*>(K.data_ptr());
  const __nv_bfloat16* Vp = static_cast<const __nv_bfloat16*>(V.data_ptr());
  __nv_bfloat16* Op = static_cast<__nv_bfloat16*>(O.data_ptr());
  float* LSEp = static_cast<float*>(LSE.data_ptr());

  int num_q_blocks = (S + BM - 1)/BM;
  dim3 grid(num_q_blocks, H, B);
  dim3 block(128);
  size_t smem = (size_t)(BM*Dpad + BN*Dpad + D*BNpad)*sizeof(__nv_bfloat16);

  cudaFuncSetAttribute(attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);

  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
  attn_kernel<<<grid, block, smem, stream>>>(Qp,Kp,Vp,Op,LSEp,B,H,S);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_kernel::run);

}  // namespace mha_kernel