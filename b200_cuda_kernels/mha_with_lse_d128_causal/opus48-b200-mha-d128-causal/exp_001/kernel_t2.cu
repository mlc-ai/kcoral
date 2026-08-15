#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <math.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                       \
    cudaError_t _e = (call);                                        \
    if (_e != cudaSuccess) {                                        \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                 \
                cudaGetErrorString(_e), __FILE__, __LINE__);        \
        exit(1);                                                    \
    }                                                               \
} while(0)

namespace mha {

constexpr int D       = 128;
constexpr int Br      = 128;
constexpr int Bc      = 64;
constexpr int WARPS   = 8;
constexpr int THREADS = WARPS * 32;
constexpr int QSTR    = 136;
constexpr int KSTR    = 136;
constexpr int VSTR    = 136;
constexpr int STAGES  = 3;
constexpr int SMEM_BYTES = (Br*QSTR + STAGES*Bc*KSTR + STAGES*Bc*VSTR) * 2;

__device__ __forceinline__ uint32_t saddr(const void* p){
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void mma16816(
    float& d0,float& d1,float& d2,float& d3,
    uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,
    uint32_t b0,uint32_t b1,
    float c0,float c1,float c2,float c3){
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
    :"=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
    :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
     "f"(c0),"f"(c1),"f"(c2),"f"(c3));
}
__device__ __forceinline__ void ldm_x2_trans(uint32_t& r0,uint32_t& r1,uint32_t addr){
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];"
    :"=r"(r0),"=r"(r1):"r"(addr));
}
__device__ __forceinline__ uint32_t pk(float lo, float hi){
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ void cpasync16z(void* dst, const void* src, int ssize){
  unsigned s = (unsigned)__cvta_generic_to_shared(dst);
  asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;\n"
    :: "r"(s), "l"(src), "r"(ssize) : "memory");
}
__device__ __forceinline__ void cp_commit(){ asm volatile("cp.async.commit_group;\n"); }
template<int N> __device__ __forceinline__ void cp_wait(){
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__global__ void __launch_bounds__(THREADS)
attn(const __nv_bfloat16* __restrict__ Q,
     const __nv_bfloat16* __restrict__ K,
     const __nv_bfloat16* __restrict__ V,
     __nv_bfloat16* __restrict__ O,
     float* __restrict__ LSE,
     int H, int S, float scale){
  int b = blockIdx.z, h = blockIdx.y, qb = blockIdx.x;
  int qstart = qb * Br;
  int tid = threadIdx.x, w = tid >> 5, t = tid & 31;
  long hb = ((long)(b * H + h)) * S;
  const __nv_bfloat16* Qh = Q + hb * D;
  const __nv_bfloat16* Kh = K + hb * D;
  const __nv_bfloat16* Vh = V + hb * D;
  __nv_bfloat16* Oh = O + hb * D;

  extern __shared__ char smem[];
  __nv_bfloat16* Qsh = reinterpret_cast<__nv_bfloat16*>(smem);
  __nv_bfloat16* Ksh = Qsh + Br*QSTR;
  __nv_bfloat16* Vsh = Ksh + STAGES*Bc*KSTR;

  const int4* Qh4 = reinterpret_cast<const int4*>(Qh);
  int4 zero4; zero4.x=zero4.y=zero4.z=zero4.w=0;

  // ---- load Q tile ----
  for(int i=tid; i<Br*16; i+=THREADS){
    int row=i>>4, d8=i&15; int gr=qstart+row;
    int4 v = (gr<S) ? Qh4[(long)gr*16 + d8] : zero4;
    *reinterpret_cast<int4*>(&Qsh[row*QSTR + d8*8]) = v;
  }

  float O0[16],O1[16],O2[16],O3[16];
  #pragma unroll
  for(int j=0;j<16;j++){O0[j]=O1[j]=O2[j]=O3[j]=0.f;}
  float m0=-1e30f,m1=-1e30f,l0=0.f,l1=0.f;

  int row0g = qstart + w*16 + (t>>2);
  int row1g = row0g + 8;
  int maxrow = qstart + Br - 1; if(maxrow>S-1) maxrow=S-1;
  int num_tiles = maxrow/Bc + 1;

  auto loadKV = [&](int kb, int st){
    int kstart = kb*Bc;
    for(int i=tid; i<Bc*16; i+=THREADS){
      int key=i>>4, d8=i&15; int gk=kstart+key;
      int sz=(gk<S)?16:0;
      const void* srcK=&Kh[(long)(gk<S?gk:0)*D + d8*8];
      const void* srcV=&Vh[(long)(gk<S?gk:0)*D + d8*8];
      cpasync16z(&Ksh[(st*Bc+key)*KSTR + d8*8], srcK, sz);
      cpasync16z(&Vsh[(st*Bc+key)*VSTR + d8*8], srcV, sz);
    }
  };

  __syncthreads();

  // prologue
  #pragma unroll
  for(int s=0;s<STAGES-1;s++){ if(s<num_tiles) loadKV(s, s%STAGES); cp_commit(); }

  int rw = w*16 + (t>>2);

  for(int kb=0; kb<num_tiles; kb++){
    int cur = kb % STAGES;
    int nxt = kb + STAGES-1;
    if(nxt<num_tiles) loadKV(nxt, nxt%STAGES);
    cp_commit();
    cp_wait<STAGES-1>();
    __syncthreads();

    int kstart = kb*Bc;

    // ---- QK^T ----
    float S0[8],S1[8],S2[8],S3[8];
    #pragma unroll
    for(int jn=0;jn<8;jn++){S0[jn]=S1[jn]=S2[jn]=S3[jn]=0.f;}
    #pragma unroll
    for(int kk=0;kk<8;kk++){
      int col=kk*16+(t&3)*2;
      uint32_t a0=*reinterpret_cast<uint32_t*>(&Qsh[rw*QSTR + col]);
      uint32_t a1=*reinterpret_cast<uint32_t*>(&Qsh[(rw+8)*QSTR + col]);
      uint32_t a2=*reinterpret_cast<uint32_t*>(&Qsh[rw*QSTR + col + 8]);
      uint32_t a3=*reinterpret_cast<uint32_t*>(&Qsh[(rw+8)*QSTR + col + 8]);
      #pragma unroll
      for(int jn=0;jn<8;jn++){
        int n=jn*8+(t>>2);
        uint32_t b0=*reinterpret_cast<uint32_t*>(&Ksh[(cur*Bc+n)*KSTR + col]);
        uint32_t b1=*reinterpret_cast<uint32_t*>(&Ksh[(cur*Bc+n)*KSTR + col + 8]);
        mma16816(S0[jn],S1[jn],S2[jn],S3[jn],a0,a1,a2,a3,b0,b1,
                 S0[jn],S1[jn],S2[jn],S3[jn]);
      }
    }

    // ---- scale + causal mask ----
    #pragma unroll
    for(int jn=0;jn<8;jn++){
      int c0=kstart+jn*8+(t&3)*2, c1=c0+1;
      S0[jn]*=scale; S1[jn]*=scale; S2[jn]*=scale; S3[jn]*=scale;
      if(c0>row0g) S0[jn]=-1e30f;
      if(c1>row0g) S1[jn]=-1e30f;
      if(c0>row1g) S2[jn]=-1e30f;
      if(c1>row1g) S3[jn]=-1e30f;
    }

    float pmax0=-1e30f,pmax1=-1e30f;
    #pragma unroll
    for(int jn=0;jn<8;jn++){
      pmax0=fmaxf(pmax0,fmaxf(S0[jn],S1[jn]));
      pmax1=fmaxf(pmax1,fmaxf(S2[jn],S3[jn]));
    }
    pmax0=fmaxf(pmax0,__shfl_xor_sync(0xffffffff,pmax0,1));
    pmax0=fmaxf(pmax0,__shfl_xor_sync(0xffffffff,pmax0,2));
    pmax1=fmaxf(pmax1,__shfl_xor_sync(0xffffffff,pmax1,1));
    pmax1=fmaxf(pmax1,__shfl_xor_sync(0xffffffff,pmax1,2));

    float nm0=fmaxf(m0,pmax0), nm1=fmaxf(m1,pmax1);
    float corr0=__expf(m0-nm0), corr1=__expf(m1-nm1);
    #pragma unroll
    for(int j=0;j<16;j++){O0[j]*=corr0;O1[j]*=corr0;O2[j]*=corr1;O3[j]*=corr1;}

    float rs0=0.f, rs1=0.f;
    #pragma unroll
    for(int jn=0;jn<8;jn++){
      S0[jn]=__expf(S0[jn]-nm0); S1[jn]=__expf(S1[jn]-nm0);
      S2[jn]=__expf(S2[jn]-nm1); S3[jn]=__expf(S3[jn]-nm1);
      rs0+=S0[jn]+S1[jn]; rs1+=S2[jn]+S3[jn];
    }
    rs0+=__shfl_xor_sync(0xffffffff,rs0,1); rs0+=__shfl_xor_sync(0xffffffff,rs0,2);
    rs1+=__shfl_xor_sync(0xffffffff,rs1,1); rs1+=__shfl_xor_sync(0xffffffff,rs1,2);
    l0=l0*corr0+rs0; l1=l1*corr1+rs1;
    m0=nm0; m1=nm1;

    uint32_t A0[4],A1[4],A2[4],A3[4];
    #pragma unroll
    for(int kk=0;kk<4;kk++){
      A0[kk]=pk(S0[2*kk],   S1[2*kk]);
      A1[kk]=pk(S2[2*kk],   S3[2*kk]);
      A2[kk]=pk(S0[2*kk+1], S1[2*kk+1]);
      A3[kk]=pk(S2[2*kk+1], S3[2*kk+1]);
    }

    // ---- PV via ldmatrix.trans ----
    #pragma unroll
    for(int jn=0;jn<16;jn++){
      int n0=jn*8;
      #pragma unroll
      for(int kk=0;kk<4;kk++){
        int krow=kk*16+(t&15);
        uint32_t addr=saddr(&Vsh[(cur*Bc+krow)*VSTR + n0]);
        uint32_t b0,b1; ldm_x2_trans(b0,b1,addr);
        mma16816(O0[jn],O1[jn],O2[jn],O3[jn],
                 A0[kk],A1[kk],A2[kk],A3[kk],b0,b1,
                 O0[jn],O1[jn],O2[jn],O3[jn]);
      }
    }
    __syncthreads();
  }

  // ---- epilogue ----
  float inv0=1.f/l0, inv1=1.f/l1;
  if(row0g<S){
    #pragma unroll
    for(int jn=0;jn<16;jn++){
      int col=jn*8+(t&3)*2;
      __nv_bfloat162 o=__floats2bfloat162_rn(O0[jn]*inv0,O1[jn]*inv0);
      *reinterpret_cast<uint32_t*>(&Oh[(long)row0g*D+col])=*reinterpret_cast<uint32_t*>(&o);
    }
  }
  if(row1g<S){
    #pragma unroll
    for(int jn=0;jn<16;jn++){
      int col=jn*8+(t&3)*2;
      __nv_bfloat162 o=__floats2bfloat162_rn(O2[jn]*inv1,O3[jn]*inv1);
      *reinterpret_cast<uint32_t*>(&Oh[(long)row1g*D+col])=*reinterpret_cast<uint32_t*>(&o);
    }
  }
  if((t&3)==0){
    if(row0g<S) LSE[hb+row0g]=m0+logf(l0);
    if(row1g<S) LSE[hb+row1g]=m1+logf(l1);
  }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int Bsz=(int)Q.size(0);
  int H=(int)Q.size(1);
  int S=(int)Q.size(2);
  int Dd=(int)Q.size(3);

  const __nv_bfloat16* Qp=static_cast<const __nv_bfloat16*>(Q.data_ptr());
  const __nv_bfloat16* Kp=static_cast<const __nv_bfloat16*>(K.data_ptr());
  const __nv_bfloat16* Vp=static_cast<const __nv_bfloat16*>(V.data_ptr());
  __nv_bfloat16* Op=static_cast<__nv_bfloat16*>(O.data_ptr());
  float* LSEp=static_cast<float*>(LSE.data_ptr());

  float scale=1.0f/sqrtf((float)Dd);

  static bool attr_set=false;
  if(!attr_set){
    CUDA_CHECK(cudaFuncSetAttribute(attn,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
    attr_set=true;
  }

  dim3 grid((S+Br-1)/Br, H, Bsz);
  dim3 block(THREADS);
  cudaStream_t stream=static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

  attn<<<grid, block, SMEM_BYTES, stream>>>(Qp,Kp,Vp,Op,LSEp,H,S,scale);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha::run);

}  // namespace mha