#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { \
  cudaError_t _e = (call); \
  if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
  } \
} while(0)

namespace flash_attn {

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int HD = 128;

__device__ __forceinline__ uint32_t pack2(float lo, float hi){
  __nv_bfloat16 a = __float2bfloat16(lo);
  __nv_bfloat16 b = __float2bfloat16(hi);
  uint16_t al = *reinterpret_cast<uint16_t*>(&a);
  uint16_t bl = *reinterpret_cast<uint16_t*>(&b);
  return ((uint32_t)bl << 16) | (uint32_t)al;
}
__device__ __forceinline__ uint32_t pack_h(__nv_bfloat16 a, __nv_bfloat16 b){
  uint16_t al = *reinterpret_cast<uint16_t*>(&a);
  uint16_t bl = *reinterpret_cast<uint16_t*>(&b);
  return ((uint32_t)bl << 16) | (uint32_t)al;
}
__device__ __forceinline__ void mma_op(
    float &d0,float &d1,float &d2,float &d3,
    uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,
    uint32_t b0,uint32_t b1){
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)
    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
__device__ __forceinline__ void cp_async16(void* s, const void* g){
  unsigned sa = (unsigned)__cvta_generic_to_shared(s);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(g));
}
__device__ __forceinline__ void cp_commit(){ asm volatile("cp.async.commit_group;\n"); }
template<int N> __device__ __forceinline__ void cp_wait(){ asm volatile("cp.async.wait_group %0;\n"::"n"(N)); }

__device__ __forceinline__ void load_kv(
    __nv_bfloat16* Kbuf, __nv_bfloat16* Vbuf,
    const __nv_bfloat16* gK, const __nv_bfloat16* gV,
    int kbase, int S, int tid){
  #pragma unroll
  for (int it=0; it<8; ++it){
    int idx = tid + it*128;
    int r = idx >> 4;
    int c = (idx & 15) * 8;
    int grow = kbase + r;
    __nv_bfloat16* dk = &Kbuf[r*HD + c];
    __nv_bfloat16* dv = &Vbuf[r*HD + c];
    if (grow < S){
      cp_async16(dk, &gK[(int64_t)grow*HD + c]);
      cp_async16(dv, &gV[(int64_t)grow*HD + c]);
    } else {
      *reinterpret_cast<float4*>(dk) = make_float4(0,0,0,0);
      *reinterpret_cast<float4*>(dv) = make_float4(0,0,0,0);
    }
  }
}

__global__ __launch_bounds__(128) void fa_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int H, int S){

  int qtile = blockIdx.x;
  int h = blockIdx.y;
  int b = blockIdx.z;
  int qstart = qtile * BM;
  if (qstart >= S) return;

  int tid = threadIdx.x;
  int warp = tid >> 5;
  int lane = tid & 31;
  int g = lane >> 2;   // 0..7
  int t = lane & 3;    // 0..3

  extern __shared__ __nv_bfloat16 dyn[];
  __nv_bfloat16* sK = dyn;                       // [2][BN][HD]
  __nv_bfloat16* sV = dyn + 2*BN*HD;             // [2][BN][HD]

  const int64_t headoff = ((int64_t)(b*H + h) * S) * HD;
  const __nv_bfloat16* gQ = Q + headoff;
  const __nv_bfloat16* gK = K + headoff;
  const __nv_bfloat16* gV = V + headoff;

  int rowg  = qstart + warp*16 + g;
  int rowg8 = rowg + 8;

  uint32_t regQ[8][4];
  #pragma unroll
  for (int kb=0; kb<8; ++kb){
    int d0 = 16*kb + 2*t;
    int d8 = d0 + 8;
    regQ[kb][0] = (rowg  < S) ? *reinterpret_cast<const uint32_t*>(&gQ[(int64_t)rowg *HD + d0]) : 0u;
    regQ[kb][1] = (rowg8 < S) ? *reinterpret_cast<const uint32_t*>(&gQ[(int64_t)rowg8*HD + d0]) : 0u;
    regQ[kb][2] = (rowg  < S) ? *reinterpret_cast<const uint32_t*>(&gQ[(int64_t)rowg *HD + d8]) : 0u;
    regQ[kb][3] = (rowg8 < S) ? *reinterpret_cast<const uint32_t*>(&gQ[(int64_t)rowg8*HD + d8]) : 0u;
  }

  float acc[16][4];
  #pragma unroll
  for (int jd=0;jd<16;++jd){ acc[jd][0]=acc[jd][1]=acc[jd][2]=acc[jd][3]=0.f; }

  float m_g=-1e30f, l_g=0.f, m_g8=-1e30f, l_g8=0.f;

  const float scale = rsqrtf((float)HD);
  const float LOG2E = 1.4426950408889634f;
  const float scale_log2 = scale * LOG2E;

  int last_q = qstart + BM - 1;
  int s_last = S - 1;
  int q_row_max = last_q < s_last ? last_q : s_last;
  int kt_last = q_row_max / BN;

  // prologue
  load_kv(sK + 0*BN*HD, sV + 0*BN*HD, gK, gV, 0, S, tid);
  cp_commit();

  for (int kt=0; kt<=kt_last; ++kt){
    int buf = kt & 1;
    if (kt < kt_last){
      int nb = (kt+1) & 1;
      load_kv(sK + nb*BN*HD, sV + nb*BN*HD, gK, gV, (kt+1)*BN, S, tid);
      cp_commit();
      cp_wait<1>();
    } else {
      cp_wait<0>();
    }
    __syncthreads();

    __nv_bfloat16* Kbuf = sK + buf*BN*HD;
    __nv_bfloat16* Vbuf = sV + buf*BN*HD;
    int kbase = kt*BN;

    // QK^T
    float C[8][4];
    #pragma unroll
    for (int j=0;j<8;++j){ C[j][0]=C[j][1]=C[j][2]=C[j][3]=0.f; }
    #pragma unroll
    for (int j=0;j<8;++j){
      #pragma unroll
      for (int kb=0;kb<8;++kb){
        int d0 = 16*kb + 2*t;
        uint32_t b0 = *reinterpret_cast<const uint32_t*>(&Kbuf[(8*j+g)*HD + d0]);
        uint32_t b1 = *reinterpret_cast<const uint32_t*>(&Kbuf[(8*j+g)*HD + d0+8]);
        mma_op(C[j][0],C[j][1],C[j][2],C[j][3],
               regQ[kb][0],regQ[kb][1],regQ[kb][2],regQ[kb][3], b0,b1);
      }
    }

    // softmax
    float sc_g[16], sc_g8[16];
    float lmax_g=-1e30f, lmax_g8=-1e30f;
    #pragma unroll
    for (int j=0;j<8;++j){
      int col0 = kbase + 8*j + 2*t;
      int col1 = col0 + 1;
      float v0 = C[j][0]*scale_log2; if (col0 > rowg ) v0=-1e30f;
      float v1 = C[j][1]*scale_log2; if (col1 > rowg ) v1=-1e30f;
      float w0 = C[j][2]*scale_log2; if (col0 > rowg8) w0=-1e30f;
      float w1 = C[j][3]*scale_log2; if (col1 > rowg8) w1=-1e30f;
      sc_g[2*j]=v0; sc_g[2*j+1]=v1; sc_g8[2*j]=w0; sc_g8[2*j+1]=w1;
      lmax_g  = fmaxf(lmax_g,  fmaxf(v0,v1));
      lmax_g8 = fmaxf(lmax_g8, fmaxf(w0,w1));
    }
    lmax_g  = fmaxf(lmax_g,  __shfl_xor_sync(0xffffffffu, lmax_g, 1));
    lmax_g  = fmaxf(lmax_g,  __shfl_xor_sync(0xffffffffu, lmax_g, 2));
    lmax_g8 = fmaxf(lmax_g8, __shfl_xor_sync(0xffffffffu, lmax_g8, 1));
    lmax_g8 = fmaxf(lmax_g8, __shfl_xor_sync(0xffffffffu, lmax_g8, 2));

    float mnew_g  = fmaxf(m_g,  lmax_g);
    float mnew_g8 = fmaxf(m_g8, lmax_g8);
    float corr_g  = exp2f(m_g  - mnew_g);
    float corr_g8 = exp2f(m_g8 - mnew_g8);

    float psum_g=0.f, psum_g8=0.f;
    #pragma unroll
    for (int k=0;k<16;++k){
      float p = exp2f(sc_g[k]  - mnew_g);  sc_g[k]=p;  psum_g  += p;
      float q = exp2f(sc_g8[k] - mnew_g8); sc_g8[k]=q; psum_g8 += q;
    }
    psum_g  += __shfl_xor_sync(0xffffffffu, psum_g, 1);
    psum_g  += __shfl_xor_sync(0xffffffffu, psum_g, 2);
    psum_g8 += __shfl_xor_sync(0xffffffffu, psum_g8, 1);
    psum_g8 += __shfl_xor_sync(0xffffffffu, psum_g8, 2);

    l_g  = l_g *corr_g  + psum_g;
    l_g8 = l_g8*corr_g8 + psum_g8;
    m_g = mnew_g; m_g8 = mnew_g8;

    #pragma unroll
    for (int jd=0;jd<16;++jd){
      acc[jd][0]*=corr_g; acc[jd][1]*=corr_g;
      acc[jd][2]*=corr_g8; acc[jd][3]*=corr_g8;
    }

    uint32_t regP[4][4];
    #pragma unroll
    for (int kb2=0;kb2<4;++kb2){
      regP[kb2][0]=pack2(sc_g[4*kb2],    sc_g[4*kb2+1]);
      regP[kb2][1]=pack2(sc_g8[4*kb2],   sc_g8[4*kb2+1]);
      regP[kb2][2]=pack2(sc_g[4*kb2+2],  sc_g[4*kb2+3]);
      regP[kb2][3]=pack2(sc_g8[4*kb2+2], sc_g8[4*kb2+3]);
    }

    // PV
    #pragma unroll
    for (int jd=0;jd<16;++jd){
      int col = 8*jd + g;
      #pragma unroll
      for (int kb2=0;kb2<4;++kb2){
        int row0 = 16*kb2 + 2*t;
        uint32_t vb0 = pack_h(Vbuf[row0*HD+col],     Vbuf[(row0+1)*HD+col]);
        uint32_t vb1 = pack_h(Vbuf[(row0+8)*HD+col], Vbuf[(row0+9)*HD+col]);
        mma_op(acc[jd][0],acc[jd][1],acc[jd][2],acc[jd][3],
               regP[kb2][0],regP[kb2][1],regP[kb2][2],regP[kb2][3], vb0,vb1);
      }
    }

    __syncthreads();
  }

  float inv_l_g  = 1.0f / l_g;
  float inv_l_g8 = 1.0f / l_g8;
  #pragma unroll
  for (int jd=0;jd<16;++jd){
    int col0 = 8*jd + 2*t;
    int col1 = col0 + 1;
    if (rowg < S){
      O[headoff + (int64_t)rowg*HD + col0] = __float2bfloat16(acc[jd][0]*inv_l_g);
      O[headoff + (int64_t)rowg*HD + col1] = __float2bfloat16(acc[jd][1]*inv_l_g);
    }
    if (rowg8 < S){
      O[headoff + (int64_t)rowg8*HD + col0] = __float2bfloat16(acc[jd][2]*inv_l_g8);
      O[headoff + (int64_t)rowg8*HD + col1] = __float2bfloat16(acc[jd][3]*inv_l_g8);
    }
  }

  const float LN2 = 0.6931471805599453f;
  if (t == 0){
    int64_t lbase = (int64_t)(b*H+h)*S;
    if (rowg  < S) LSE[lbase + rowg ] = LN2*(m_g  + log2f(l_g));
    if (rowg8 < S) LSE[lbase + rowg8] = LN2*(m_g8 + log2f(l_g8));
  }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE){
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int H = (int)Q.size(1);
  int S = (int)Q.size(2);
  int B = (int)Q.size(0);

  const __nv_bfloat16* qptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
  const __nv_bfloat16* kptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
  const __nv_bfloat16* vptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
  __nv_bfloat16* optr = static_cast<__nv_bfloat16*>(O.data_ptr());
  float* lptr = static_cast<float*>(LSE.data_ptr());

  int num_q_tiles = (S + BM - 1) / BM;
  dim3 grid(num_q_tiles, H, B);
  dim3 block(128);

  size_t smem_bytes = (size_t)2*BN*HD*sizeof(__nv_bfloat16)*2; // K + V double-buffered
  static bool attr_set = false;
  if (!attr_set){
    cudaFuncSetAttribute(fa_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
    attr_set = true;
  }

  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
  fa_kernel<<<grid, block, smem_bytes, stream>>>(qptr, kptr, vptr, optr, lptr, H, S);
  CUDA_CHECK(cudaGetLastError());
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flash_attn::run);

}  // namespace flash_attn