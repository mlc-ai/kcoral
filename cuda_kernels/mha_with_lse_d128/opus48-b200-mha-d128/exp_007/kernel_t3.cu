#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <math.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__); exit(1);} }while(0)

namespace mha_kernel {

constexpr int BM=128, BN=128, Dh=128, NT=128;
constexpr float NEG=-1e30f;
constexpr int TILE_BYTES = (Dh/8)*BM*16;   // 32768
constexpr int LBO_K=BM*16, SBO_K=128;      // K-major (Q/K/P): LBO=2048 SBO=128
constexpr int LBO_V=128,   SBO_V=BN*16;    // MN-major (V):    LBO=128  SBO=2048

__device__ __forceinline__ uint64_t make_desc(void* p, uint32_t lbo, uint32_t sbo){
  uint32_t addr=(uint32_t)__cvta_generic_to_shared(p);
  uint64_t d=0;
  d |= (uint64_t)((addr & 0x3FFFF) >> 4);
  d |= ((uint64_t)((lbo & 0x3FFFF) >> 4)) << 16;
  d |= ((uint64_t)((sbo & 0x3FFFF) >> 4)) << 32;
  d |= ((uint64_t)1) << 46;   // fixed 0b001
  return d;                   // swizzle none (bits 61-63 = 0)
}
__device__ __forceinline__ uint32_t make_idesc(uint32_t M,uint32_t N,uint32_t amaj,uint32_t bmaj){
  uint32_t d=0;
  d|=(1u<<4); d|=(1u<<7); d|=(1u<<10);
  d|=(amaj<<15); d|=(bmaj<<16);
  d|=((N/8)<<17); d|=((M/16)<<24);
  return d;
}
__device__ __forceinline__ void cp_async16(void* smem,const void* g){
  uint32_t s=(uint32_t)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n"::"r"(s),"l"(g));
}
__device__ __forceinline__ void cp_commit(){ asm volatile("cp.async.commit_group;\n":::"memory"); }
__device__ __forceinline__ void cp_wait0(){ asm volatile("cp.async.wait_group 0;\n":::"memory"); }

__device__ __forceinline__ void init_bar(uint64_t* b,uint32_t c){
  asm volatile("mbarrier.init.shared.b64 [%0], %1;"::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(c));
}
__device__ __forceinline__ void fence_bar_init(){ asm volatile("fence.mbarrier_init.release.cluster;\n":::"memory"); }
__device__ __forceinline__ void bar_wait(uint64_t* b,uint32_t ph){
  asm volatile("{\n.reg .pred P;\nW%=:\nmbarrier.try_wait.parity.shared.b64 P,[%0],%1;\n@!P bra W%=;\n}\n"
    ::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(ph));
}
__device__ __forceinline__ void fence_proxy_async(){ asm volatile("fence.proxy.async;\n":::"memory"); }

__device__ __forceinline__ void tmem_alloc1(uint32_t* dst,int ncols){
  uint32_t a=(uint32_t)__cvta_generic_to_shared(dst);
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0],%1;"::"r"(a),"r"(ncols));
}
__device__ __forceinline__ void tmem_dealloc1(uint32_t addr,int ncols){
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0,%1;"::"r"(addr),"r"(ncols));
}
__device__ __forceinline__ void umma1(uint32_t d,uint64_t a,uint64_t b,uint32_t idesc,uint32_t accum){
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p,%4,0;\n"
    "tcgen05.mma.cta_group::1.kind::f16 [%0],%1,%2,%3,p;\n}\n"
    ::"r"(d),"l"(a),"l"(b),"r"(idesc),"r"(accum));
}
__device__ __forceinline__ void umma_commit1(uint64_t* bar){
  uint32_t a=(uint32_t)__cvta_generic_to_shared(bar);
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"::"r"(a));
}
__device__ __forceinline__ void ldx4(uint32_t addr,uint32_t&a,uint32_t&b,uint32_t&c,uint32_t&d){
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3},[%4];"
    :"=r"(a),"=r"(b),"=r"(c),"=r"(d):"r"(addr));
}
__device__ __forceinline__ void stx4(uint32_t addr,uint32_t a,uint32_t b,uint32_t c,uint32_t d){
  asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0],{%1,%2,%3,%4};"
    ::"r"(addr),"r"(a),"r"(b),"r"(c),"r"(d));
}
__device__ __forceinline__ void waitld(){ asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory"); }
__device__ __forceinline__ void waitst(){ asm volatile("tcgen05.wait::st.sync.aligned;":::"memory"); }
__device__ __forceinline__ uint32_t packbf(float a,float b){
  __nv_bfloat162 v=__floats2bfloat162_rn(a,b);
  return *reinterpret_cast<uint32_t*>(&v);
}

__launch_bounds__(128,1)
__global__ void attn(const __nv_bfloat16* __restrict__ Qg,const __nv_bfloat16* __restrict__ Kg,
                     const __nv_bfloat16* __restrict__ Vg,__nv_bfloat16* __restrict__ Og,
                     float* __restrict__ LSEg,int B,int H,int S){
  const int b=blockIdx.z, h=blockIdx.y, m0=blockIdx.x*BM;
  const int tid=threadIdx.x, warp=tid>>5;
  const bool leader=(tid==0);
  const float scale=rsqrtf((float)Dh);

  extern __shared__ char smem[];
  char* qs=smem;
  char* ks=qs+TILE_BYTES;
  char* vs=ks+TILE_BYTES;
  char* ps=vs+TILE_BYTES;
  __shared__ uint64_t mbar[2];
  __shared__ uint32_t tbase_s[1];

  const long head_off=((long)(b*H+h))*(long)S*Dh;

  if(leader){ init_bar(&mbar[0],1); init_bar(&mbar[1],1); }
  fence_bar_init();
  __syncthreads();
  if(warp==0) tmem_alloc1(&tbase_s[0],256);
  __syncthreads();
  uint32_t tbase=tbase_s[0];
  uint32_t Sbase=tbase, Obase=tbase+128;

  // ---- load Q once (K-major [D/8, BM, 8]) ----
  for(int i=tid;i<BM*(Dh/8);i+=NT){
    int m=i/(Dh/8), kb=i%(Dh/8);
    int grow=m0+m;
    char* dst=qs + kb*LBO_K + m*16;
    if(grow<S) cp_async16(dst,&Qg[head_off+(long)grow*Dh+kb*8]);
    else *(int4*)dst=make_int4(0,0,0,0);
  }
  cp_commit();

  float m_i=NEG, l_i=0.f;
  uint32_t ph1=0, ph2=0;
  int num_kv=(S+BN-1)/BN;

  for(int kv=0; kv<num_kv; kv++){
    int kv_start=kv*BN;
    // ---- load K (K-major) and V (MN-major) ----
    for(int i=tid;i<BN*(Dh/8);i+=NT){
      int n=i/(Dh/8), kb=i%(Dh/8);
      int grow=kv_start+n;
      char* kd=ks + kb*LBO_K + n*16;
      if(grow<S) cp_async16(kd,&Kg[head_off+(long)grow*Dh+kb*8]);
      else *(int4*)kd=make_int4(0,0,0,0);
    }
    for(int i=tid;i<BN*(Dh/8);i+=NT){
      int k=i/(Dh/8), nb=i%(Dh/8);
      int grow=kv_start+k;
      char* vd=vs + nb*SBO_V + k*16;
      if(grow<S) cp_async16(vd,&Vg[head_off+(long)grow*Dh+nb*8]);
      else *(int4*)vd=make_int4(0,0,0,0);
    }
    cp_commit();
    cp_wait0();
    __syncthreads();
    fence_proxy_async();
    __syncthreads();

    // ---- matmul1: S = Q @ K^T ----
    if(leader){
      uint32_t id1=make_idesc(BM,BN,0,0);
      #pragma unroll
      for(int j=0;j<8;j++){
        uint64_t da=make_desc(qs+j*(BM*32),LBO_K,SBO_K);
        uint64_t db=make_desc(ks+j*(BN*32),LBO_K,SBO_K);
        umma1(Sbase,da,db,id1,j==0?0:1);
      }
      umma_commit1(&mbar[0]);
    }
    bar_wait(&mbar[0],ph1); ph1^=1;
    __syncthreads();

    // ---- softmax pass1: row max ----
    float bmax=NEG;
    #pragma unroll
    for(int cb=0;cb<16;cb++){
      uint32_t r0,r1,r2,r3,r4,r5,r6,r7;
      ldx4(Sbase+cb*8,   r0,r1,r2,r3);
      ldx4(Sbase+cb*8+4, r4,r5,r6,r7);
      waitld();
      float f[8]={__uint_as_float(r0),__uint_as_float(r1),__uint_as_float(r2),__uint_as_float(r3),
                  __uint_as_float(r4),__uint_as_float(r5),__uint_as_float(r6),__uint_as_float(r7)};
      #pragma unroll
      for(int e=0;e<8;e++){
        int col=cb*8+e;
        float s=(kv_start+col<S)? f[e]*scale : NEG;
        bmax=fmaxf(bmax,s);
      }
    }
    float nm=fmaxf(m_i,bmax);
    float corr=__expf(m_i-nm);

    // ---- softmax pass2: p=exp(s-nm), write P (K-major [BN/8, BM, 8]) ----
    float bsum=0.f;
    #pragma unroll
    for(int cb=0;cb<16;cb++){
      uint32_t r0,r1,r2,r3,r4,r5,r6,r7;
      ldx4(Sbase+cb*8,   r0,r1,r2,r3);
      ldx4(Sbase+cb*8+4, r4,r5,r6,r7);
      waitld();
      float f[8]={__uint_as_float(r0),__uint_as_float(r1),__uint_as_float(r2),__uint_as_float(r3),
                  __uint_as_float(r4),__uint_as_float(r5),__uint_as_float(r6),__uint_as_float(r7)};
      float p[8];
      #pragma unroll
      for(int e=0;e<8;e++){
        int col=cb*8+e;
        float s=(kv_start+col<S)? f[e]*scale : NEG;
        p[e]=__expf(s-nm);
        bsum+=p[e];
      }
      uint4 pk;
      pk.x=packbf(p[0],p[1]); pk.y=packbf(p[2],p[3]);
      pk.z=packbf(p[4],p[5]); pk.w=packbf(p[6],p[7]);
      *(uint4*)(ps + cb*LBO_K + tid*16)=pk;
    }
    l_i=l_i*corr+bsum;
    m_i=nm;

    // ---- rescale O in TMEM by corr (kv>0) ----
    if(kv>0){
      #pragma unroll
      for(int cb=0;cb<32;cb++){
        uint32_t o0,o1,o2,o3;
        ldx4(Obase+cb*4,o0,o1,o2,o3);
        waitld();
        o0=__float_as_uint(__uint_as_float(o0)*corr);
        o1=__float_as_uint(__uint_as_float(o1)*corr);
        o2=__float_as_uint(__uint_as_float(o2)*corr);
        o3=__float_as_uint(__uint_as_float(o3)*corr);
        stx4(Obase+cb*4,o0,o1,o2,o3);
      }
      waitst();
    }
    __syncthreads();
    fence_proxy_async();
    __syncthreads();

    // ---- matmul2: O += P @ V ----
    if(leader){
      uint32_t id2=make_idesc(BM,Dh,0,1);
      #pragma unroll
      for(int j=0;j<8;j++){
        uint64_t da=make_desc(ps+j*(BM*32),LBO_K,SBO_K);
        uint64_t db=make_desc(vs+j*256,   LBO_V,SBO_V);
        uint32_t acc=(kv==0 && j==0)?0:1;
        umma1(Obase,da,db,id2,acc);
      }
      umma_commit1(&mbar[1]);
    }
    bar_wait(&mbar[1],ph2); ph2^=1;
    __syncthreads();
  }

  // ---- finalize: O/l, write output + LSE ----
  float inv=(l_i>0.f)?1.f/l_i:0.f;
  int grow=m0+tid;
  #pragma unroll
  for(int cb=0;cb<16;cb++){
    uint32_t o0,o1,o2,o3,o4,o5,o6,o7;
    ldx4(Obase+cb*8,   o0,o1,o2,o3);
    ldx4(Obase+cb*8+4, o4,o5,o6,o7);
    waitld();
    float f[8]={__uint_as_float(o0),__uint_as_float(o1),__uint_as_float(o2),__uint_as_float(o3),
                __uint_as_float(o4),__uint_as_float(o5),__uint_as_float(o6),__uint_as_float(o7)};
    uint4 pk;
    pk.x=packbf(f[0]*inv,f[1]*inv); pk.y=packbf(f[2]*inv,f[3]*inv);
    pk.z=packbf(f[4]*inv,f[5]*inv); pk.w=packbf(f[6]*inv,f[7]*inv);
    if(grow<S) *(uint4*)&Og[head_off+(long)grow*Dh+cb*8]=pk;
  }
  if(grow<S){
    long lse_off=((long)(b*H+h))*(long)S + grow;
    LSEg[lse_off]=m_i+logf(l_i);
  }
  __syncthreads();
  if(warp==0) tmem_dealloc1(tbase,256);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int Bn=(int)Q.size(0), Hn=(int)Q.size(1), Sn=(int)Q.size(2);
  const __nv_bfloat16* Qp=static_cast<const __nv_bfloat16*>(Q.data_ptr());
  const __nv_bfloat16* Kp=static_cast<const __nv_bfloat16*>(K.data_ptr());
  const __nv_bfloat16* Vp=static_cast<const __nv_bfloat16*>(V.data_ptr());
  __nv_bfloat16* Op=static_cast<__nv_bfloat16*>(O.data_ptr());
  float* Lp=static_cast<float*>(LSE.data_ptr());

  int num_m=(Sn+BM-1)/BM;
  dim3 grid(num_m,Hn,Bn);
  int smem_bytes=4*TILE_BYTES;

  cudaStream_t stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type,Q.device().device_id));
  CUDA_CHECK(cudaFuncSetAttribute(attn,cudaFuncAttributeMaxDynamicSharedMemorySize,smem_bytes));
  attn<<<grid,NT,smem_bytes,stream>>>(Qp,Kp,Vp,Op,Lp,Bn,Hn,Sn);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_kernel::run);

}  // namespace mha_kernel