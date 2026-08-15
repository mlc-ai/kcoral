#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <math.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace mha_causal {

constexpr int Dh = 128;
constexpr int BM = 64;
constexpr int BN = 64;

__device__ __forceinline__ uint32_t pack2(float a, float b){
  __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&v);
}

__device__ __forceinline__ void mma16816(float &d0,float &d1,float &d2,float &d3,
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

__global__ __launch_bounds__(128, 2)
void attn_kernel(const __nv_bfloat16* __restrict__ Q,
                 const __nv_bfloat16* __restrict__ K,
                 const __nv_bfloat16* __restrict__ V,
                 __nv_bfloat16* __restrict__ O,
                 float* __restrict__ LSE,
                 int S){
  extern __shared__ __nv_bfloat16 smem[];
  __nv_bfloat16* Qs = smem;
  __nv_bfloat16* Ks = Qs + BM*Dh;
  __nv_bfloat16* Vs = Ks + BN*Dh;

  int bh    = blockIdx.y;
  int qtile = blockIdx.x;
  int q0    = qtile * BM;
  if (q0 >= S) return;

  int tid = threadIdx.x;
  int warp = tid >> 5;
  int lane = tid & 31;
  int groupID = lane >> 2;
  int tig = lane & 3;

  const __nv_bfloat16* Qbase = Q + (size_t)bh * S * Dh;
  const __nv_bfloat16* Kbase = K + (size_t)bh * S * Dh;
  const __nv_bfloat16* Vbase = V + (size_t)bh * S * Dh;

  // Load Q tile
  for (int idx = tid; idx < BM*Dh; idx += blockDim.x){
    int r = idx / Dh; int c = idx % Dh;
    int grow = q0 + r;
    Qs[idx] = (grow < S) ? Qbase[(size_t)grow*Dh + c] : __float2bfloat16(0.f);
  }

  const float scale  = 0.08838834764831845f; // 1/sqrt(128)
  const float log2e  = 1.4426950408889634f;
  const float scale2 = scale * log2e;
  const float ln2    = 0.6931471805599453f;
  const float NEG    = -1e30f;

  float Oacc[16][4];
  #pragma unroll
  for (int i=0;i<16;i++){ Oacc[i][0]=0.f; Oacc[i][1]=0.f; Oacc[i][2]=0.f; Oacc[i][3]=0.f; }
  float m2_lo = NEG, m2_hi = NEG;
  float l_lo = 0.f, l_hi = 0.f;

  int qrow_lo = q0 + warp*16 + groupID;
  int qrow_hi = qrow_lo + 8;

  int maxq = q0 + BM - 1;
  if (maxq > S-1) maxq = S-1;
  int num_kt = maxq / BN + 1;

  int rlo = warp*16 + groupID;   // shared-memory row for lo
  int rhi = rlo + 8;

  for (int kt = 0; kt < num_kt; kt++){
    int kv0 = kt * BN;

    // Load K,V tiles
    for (int idx = tid; idx < BN*Dh; idx += blockDim.x){
      int r = idx / Dh; int c = idx % Dh;
      int krow = kv0 + r;
      Ks[idx] = (krow < S) ? Kbase[(size_t)krow*Dh + c] : __float2bfloat16(0.f);
      Vs[idx] = (krow < S) ? Vbase[(size_t)krow*Dh + c] : __float2bfloat16(0.f);
    }
    __syncthreads();

    // ---- S = Q K^T ----
    float Sreg[8][4];
    #pragma unroll
    for (int nt = 0; nt < 8; nt++){
      float acc0=0.f, acc1=0.f, acc2=0.f, acc3=0.f;
      int krow = nt*8 + groupID;
      #pragma unroll
      for (int kk = 0; kk < 8; kk++){
        int cc = kk*16 + tig*2;
        uint32_t a0 = *reinterpret_cast<const uint32_t*>(&Qs[rlo*Dh + cc]);
        uint32_t a1 = *reinterpret_cast<const uint32_t*>(&Qs[rhi*Dh + cc]);
        uint32_t a2 = *reinterpret_cast<const uint32_t*>(&Qs[rlo*Dh + cc + 8]);
        uint32_t a3 = *reinterpret_cast<const uint32_t*>(&Qs[rhi*Dh + cc + 8]);
        uint32_t b0 = *reinterpret_cast<const uint32_t*>(&Ks[krow*Dh + cc]);
        uint32_t b1 = *reinterpret_cast<const uint32_t*>(&Ks[krow*Dh + cc + 8]);
        mma16816(acc0,acc1,acc2,acc3, a0,a1,a2,a3, b0,b1, acc0,acc1,acc2,acc3);
      }
      Sreg[nt][0]=acc0; Sreg[nt][1]=acc1; Sreg[nt][2]=acc2; Sreg[nt][3]=acc3;
    }

    // ---- scale + causal mask (values become base-2 exponent domain) ----
    #pragma unroll
    for (int nt = 0; nt < 8; nt++){
      int c0col = kv0 + nt*8 + tig*2;
      int c1col = c0col + 1;
      Sreg[nt][0] = (c0col<=qrow_lo && c0col<S) ? Sreg[nt][0]*scale2 : NEG;
      Sreg[nt][1] = (c1col<=qrow_lo && c1col<S) ? Sreg[nt][1]*scale2 : NEG;
      Sreg[nt][2] = (c0col<=qrow_hi && c0col<S) ? Sreg[nt][2]*scale2 : NEG;
      Sreg[nt][3] = (c1col<=qrow_hi && c1col<S) ? Sreg[nt][3]*scale2 : NEG;
    }

    // ---- row max ----
    float rmax_lo = NEG, rmax_hi = NEG;
    #pragma unroll
    for (int nt = 0; nt < 8; nt++){
      rmax_lo = fmaxf(rmax_lo, fmaxf(Sreg[nt][0], Sreg[nt][1]));
      rmax_hi = fmaxf(rmax_hi, fmaxf(Sreg[nt][2], Sreg[nt][3]));
    }
    rmax_lo = fmaxf(rmax_lo, __shfl_xor_sync(0xffffffff, rmax_lo, 1));
    rmax_lo = fmaxf(rmax_lo, __shfl_xor_sync(0xffffffff, rmax_lo, 2));
    rmax_hi = fmaxf(rmax_hi, __shfl_xor_sync(0xffffffff, rmax_hi, 1));
    rmax_hi = fmaxf(rmax_hi, __shfl_xor_sync(0xffffffff, rmax_hi, 2));

    float m2n_lo = fmaxf(m2_lo, rmax_lo);
    float m2n_hi = fmaxf(m2_hi, rmax_hi);
    float alpha_lo = exp2f(m2_lo - m2n_lo);
    float alpha_hi = exp2f(m2_hi - m2n_hi);

    // ---- P = exp2(S - m), row sum ----
    float rsum_lo=0.f, rsum_hi=0.f;
    #pragma unroll
    for (int nt = 0; nt < 8; nt++){
      float p0 = exp2f(Sreg[nt][0]-m2n_lo); rsum_lo+=p0; Sreg[nt][0]=p0;
      float p1 = exp2f(Sreg[nt][1]-m2n_lo); rsum_lo+=p1; Sreg[nt][1]=p1;
      float p2 = exp2f(Sreg[nt][2]-m2n_hi); rsum_hi+=p2; Sreg[nt][2]=p2;
      float p3 = exp2f(Sreg[nt][3]-m2n_hi); rsum_hi+=p3; Sreg[nt][3]=p3;
    }
    rsum_lo += __shfl_xor_sync(0xffffffff, rsum_lo, 1);
    rsum_lo += __shfl_xor_sync(0xffffffff, rsum_lo, 2);
    rsum_hi += __shfl_xor_sync(0xffffffff, rsum_hi, 1);
    rsum_hi += __shfl_xor_sync(0xffffffff, rsum_hi, 2);

    l_lo = alpha_lo*l_lo + rsum_lo;
    l_hi = alpha_hi*l_hi + rsum_hi;
    m2_lo = m2n_lo; m2_hi = m2n_hi;

    // ---- rescale O accumulator ----
    #pragma unroll
    for (int dt = 0; dt < 16; dt++){
      Oacc[dt][0]*=alpha_lo; Oacc[dt][1]*=alpha_lo;
      Oacc[dt][2]*=alpha_hi; Oacc[dt][3]*=alpha_hi;
    }

    // ---- build A fragments (P as bf16) for the 4 k-slices ----
    uint32_t Afr[4][4];
    #pragma unroll
    for (int ks = 0; ks < 4; ks++){
      Afr[ks][0]=pack2(Sreg[2*ks][0],   Sreg[2*ks][1]);
      Afr[ks][1]=pack2(Sreg[2*ks][2],   Sreg[2*ks][3]);
      Afr[ks][2]=pack2(Sreg[2*ks+1][0], Sreg[2*ks+1][1]);
      Afr[ks][3]=pack2(Sreg[2*ks+1][2], Sreg[2*ks+1][3]);
    }

    // ---- O += P @ V ----
    #pragma unroll
    for (int dt = 0; dt < 16; dt++){
      float c0=Oacc[dt][0], c1=Oacc[dt][1], c2=Oacc[dt][2], c3=Oacc[dt][3];
      int d = dt*8 + groupID;
      #pragma unroll
      for (int ks = 0; ks < 4; ks++){
        int r0 = ks*16 + tig*2;
        uint32_t b0 = (uint32_t)(*reinterpret_cast<const uint16_t*>(&Vs[r0*Dh + d]))
                    | ((uint32_t)(*reinterpret_cast<const uint16_t*>(&Vs[(r0+1)*Dh + d]))<<16);
        int r1 = r0 + 8;
        uint32_t b1 = (uint32_t)(*reinterpret_cast<const uint16_t*>(&Vs[r1*Dh + d]))
                    | ((uint32_t)(*reinterpret_cast<const uint16_t*>(&Vs[(r1+1)*Dh + d]))<<16);
        mma16816(c0,c1,c2,c3, Afr[ks][0],Afr[ks][1],Afr[ks][2],Afr[ks][3], b0,b1, c0,c1,c2,c3);
      }
      Oacc[dt][0]=c0; Oacc[dt][1]=c1; Oacc[dt][2]=c2; Oacc[dt][3]=c3;
    }

    __syncthreads();
  }

  // ---- finalize ----
  float inv_lo = 1.f / l_lo;
  float inv_hi = 1.f / l_hi;
  size_t obase = (size_t)bh * S * Dh;
  #pragma unroll
  for (int dt = 0; dt < 16; dt++){
    int col0 = dt*8 + tig*2;
    int col1 = col0 + 1;
    if (qrow_lo < S){
      O[obase + (size_t)qrow_lo*Dh + col0] = __float2bfloat16(Oacc[dt][0]*inv_lo);
      O[obase + (size_t)qrow_lo*Dh + col1] = __float2bfloat16(Oacc[dt][1]*inv_lo);
    }
    if (qrow_hi < S){
      O[obase + (size_t)qrow_hi*Dh + col0] = __float2bfloat16(Oacc[dt][2]*inv_hi);
      O[obase + (size_t)qrow_hi*Dh + col1] = __float2bfloat16(Oacc[dt][3]*inv_hi);
    }
  }
  if (tig == 0){
    if (qrow_lo < S) LSE[(size_t)bh*S + qrow_lo] = m2_lo*ln2 + logf(l_lo);
    if (qrow_hi < S) LSE[(size_t)bh*S + qrow_hi] = m2_hi*ln2 + logf(l_hi);
  }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE){
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int64_t B = Q.size(0);
  int64_t Hh = Q.size(1);
  int64_t S = Q.size(2);

  const __nv_bfloat16* q = static_cast<const __nv_bfloat16*>(Q.data_ptr());
  const __nv_bfloat16* k = static_cast<const __nv_bfloat16*>(K.data_ptr());
  const __nv_bfloat16* v = static_cast<const __nv_bfloat16*>(V.data_ptr());
  __nv_bfloat16* o = static_cast<__nv_bfloat16*>(O.data_ptr());
  float* lse = static_cast<float*>(LSE.data_ptr());

  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

  dim3 grid((unsigned)((S + BM - 1)/BM), (unsigned)(B*Hh));
  dim3 block(128);
  size_t smem = (size_t)(BM + 2*BN) * Dh * sizeof(__nv_bfloat16);

  static bool attr_set = false;
  if (!attr_set){
    cudaFuncSetAttribute(attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    attr_set = true;
  }

  attn_kernel<<<grid, block, smem, stream>>>(q, k, v, o, lse, (int)S);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_causal::run);

}  // namespace mha_causal