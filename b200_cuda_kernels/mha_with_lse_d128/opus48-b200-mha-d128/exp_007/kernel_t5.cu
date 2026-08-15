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

constexpr int BM=128, BN=64, Dh=128, NT=128;
constexpr float NEG=-1e30f;

__device__ __forceinline__ uint64_t make_desc(void* p,uint32_t lbo,uint32_t sbo){
  uint32_t a=(uint32_t)__cvta_generic_to_shared(p); uint64_t d=0;
  d|=(uint64_t)((a&0x3FFFF)>>4);
  d|=((uint64_t)((lbo&0x3FFFF)>>4))<<16;
  d|=((uint64_t)((sbo&0x3FFFF)>>4))<<32;
  d|=((uint64_t)1)<<46;
  return d;
}
__device__ __forceinline__ uint32_t make_idesc(uint32_t M,uint32_t N,uint32_t am,uint32_t bm){
  uint32_t d=0; d|=(1u<<4); d|=(1u<<7); d|=(1u<<10);
  d|=(am<<15); d|=(bm<<16); d|=((N/8)<<17); d|=((M/16)<<24); return d;
}
__device__ __forceinline__ void cp_async16(void* s,const void* g){
  uint32_t a=(uint32_t)__cvta_generic_to_shared(s);
  asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n"::"r"(a),"l"(g));
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
__device__ __forceinline__ void tmem_alloc1(uint32_t* d,int n){
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0],%1;"
    ::"r"((uint32_t)__cvta_generic_to_shared(d)),"r"(n));
}
__device__ __forceinline__ void tmem_dealloc1(uint32_t a,int n){
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0,%1;"::"r"(a),"r"(n));
}
__device__ __forceinline__ void umma1(uint32_t d,uint64_t a,uint64_t b,uint32_t id,uint32_t acc){
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p,%4,0;\n"
    "tcgen05.mma.cta_group::1.kind::f16 [%0],%1,%2,%3,p;\n}\n"
    ::"r"(d),"l"(a),"l"(b),"r"(id),"r"(acc));
}
__device__ __forceinline__ void umma_commit1(uint64_t* b){
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
    ::"r"((uint32_t)__cvta_generic_to_shared(b)));
}
__device__ __forceinline__ void ldx4(uint32_t ad,uint32_t&a,uint32_t&b,uint32_t&c,uint32_t&d){
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3},[%4];"
    :"=r"(a),"=r"(b),"=r"(c),"=r"(d):"r"(ad));
}
__device__ __forceinline__ void stx4(uint32_t ad,uint32_t a,uint32_t b,uint32_t c,uint32_t d){
  asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0],{%1,%2,%3,%4};"
    ::"r"(ad),"r"(a),"r"(b),"r"(c),"r"(d));
}
__device__ __forceinline__ void waitld(){ asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory"); }
__device__ __forceinline__ void waitst(){ asm volatile("tcgen05.wait::st.sync.aligned;":::"memory"); }
__device__ __forceinline__ uint32_t packbf(float a,float b){
  __nv_bfloat162 v=__floats2bfloat162_rn(a,b); return *reinterpret_cast<uint32_t*>(&v);
}

__launch_bounds__(128,2)
__global__ void attn(const __nv_bfloat16* __restrict__ Qg,const __nv_bfloat16* __restrict__ Kg,
                     const __nv_bfloat16* __restrict__ Vg,__nv_bfloat16* __restrict__ Og,
                     float* __restrict__ LSEg,int B,int H,int S){
  const int b=blockIdx.z,h=blockIdx.y,m0=blockIdx.x*BM;
  const int tid=threadIdx.x,warp=tid>>5;
  const bool leader=(tid==0);
  const float scale=rsqrtf((float)Dh);

  extern __shared__ char smem[];
  char* qs=smem;               // 32768
  char* ks=qs+32768;           // 16384
  char* vs=ks+16384;           // 16384
  char* ps=vs+16384;           // 16384
  __shared__ uint64_t mbar[2];
  __shared__ uint32_t tbase_s[1];

  const long head_off=((long)(b*H+h))*(long)S*Dh;
  const int num_kv=(S+BN-1)/BN;

  if(leader){ init_bar(&mbar[0],1); init_bar(&mbar[1],1); }
  fence_bar_init(); __syncthreads();
  if(warp==0) tmem_alloc1(&tbase_s[0],256);
  __syncthreads();
  uint32_t tbase=tbase_s[0];
  uint32_t Sbase=tbase, Obase=tbase+64;

  // ---- load Q once ----
  for(int i=tid;i<BM*16;i+=NT){int m=i>>4,kb=i&15;int gr=m0+m;
    char* d=qs+kb*2048+m*16;
    if(gr<S) cp_async16(d,&Qg[head_off+(long)gr*Dh+kb*8]); else *(int4*)d=make_int4(0,0,0,0);}
  cp_commit();

  float m_i=NEG,l_i=0.f;
  uint32_t ph0=0,ph1=0;

  for(int j=0;j<num_kv;j++){
    int kvs=j*BN;
    // ---- load K,V ----
    for(int i=tid;i<BN*16;i+=NT){int n=i>>4,kb=i&15;int gr=kvs+n;
      char* d=ks+kb*1024+n*16;
      if(gr<S) cp_async16(d,&Kg[head_off+(long)gr*Dh+kb*8]); else *(int4*)d=make_int4(0,0,0,0);}
    for(int i=tid;i<BN*16;i+=NT){int k=i>>4,nb=i&15;int gr=kvs+k;
      char* d=vs+nb*1024+k*16;
      if(gr<S) cp_async16(d,&Vg[head_off+(long)gr*Dh+nb*8]); else *(int4*)d=make_int4(0,0,0,0);}
    cp_commit(); cp_wait0();
    __syncthreads(); fence_proxy_async(); __syncthreads();

    // ---- MMA1: S = Q@K^T ----
    if(leader){
      uint32_t id=make_idesc(BM,BN,0,0);
      #pragma unroll
      for(int k=0;k<8;k++){
        uint64_t da=make_desc(qs+k*4096,2048,128);
        uint64_t db=make_desc(ks+k*2048,1024,128);
        umma1(Sbase,da,db,id,k==0?0:1);
      }
      umma_commit1(&mbar[0]);
    }
    bar_wait(&mbar[0],ph0); ph0^=1;

    // ---- softmax (per-thread row) ----
    float s[64];
    #pragma unroll
    for(int cb=0;cb<8;cb++){
      ldx4(Sbase+cb*8,   *(uint32_t*)&s[cb*8+0],*(uint32_t*)&s[cb*8+1],*(uint32_t*)&s[cb*8+2],*(uint32_t*)&s[cb*8+3]);
      ldx4(Sbase+cb*8+4, *(uint32_t*)&s[cb*8+4],*(uint32_t*)&s[cb*8+5],*(uint32_t*)&s[cb*8+6],*(uint32_t*)&s[cb*8+7]);
    }
    waitld();
    float rmax=NEG;
    #pragma unroll
    for(int e=0;e<64;e++){ float v=(kvs+e<S)? s[e]*scale : NEG; s[e]=v; rmax=fmaxf(rmax,v); }
    float nm=fmaxf(m_i,rmax);
    float corr=__expf(m_i-nm);
    float rsum=0.f;
    #pragma unroll
    for(int e=0;e<64;e++){ float p=__expf(s[e]-nm); s[e]=p; rsum+=p; }
    #pragma unroll
    for(int kb=0;kb<8;kb++){
      uint4 pk;
      pk.x=packbf(s[kb*8+0],s[kb*8+1]); pk.y=packbf(s[kb*8+2],s[kb*8+3]);
      pk.z=packbf(s[kb*8+4],s[kb*8+5]); pk.w=packbf(s[kb*8+6],s[kb*8+7]);
      *(uint4*)(ps+kb*2048+tid*16)=pk;
    }
    l_i=l_i*corr+rsum; m_i=nm;

    // ---- rescale O (j>0) ----
    if(j>0){
      #pragma unroll
      for(int c=0;c<4;c++){
        uint32_t o[32];
        #pragma unroll
        for(int q=0;q<8;q++) ldx4(Obase+c*32+q*4,o[q*4],o[q*4+1],o[q*4+2],o[q*4+3]);
        waitld();
        #pragma unroll
        for(int e=0;e<32;e++) o[e]=__float_as_uint(__uint_as_float(o[e])*corr);
        #pragma unroll
        for(int q=0;q<8;q++) stx4(Obase+c*32+q*4,o[q*4],o[q*4+1],o[q*4+2],o[q*4+3]);
      }
      waitst();
    }
    __syncthreads(); fence_proxy_async(); __syncthreads();

    // ---- MMA2: O += P@V ----
    if(leader){
      uint32_t id=make_idesc(BM,Dh,0,1);
      #pragma unroll
      for(int k=0;k<4;k++){
        uint64_t da=make_desc(ps+k*4096,2048,128);
        uint64_t db=make_desc(vs+k*256,128,1024);
        umma1(Obase,da,db,id,(j==0&&k==0)?0:1);
      }
      umma_commit1(&mbar[1]);
    }
    bar_wait(&mbar[1],ph1); ph1^=1;
  }

  // ---- finalize ----
  float inv=(l_i>0.f)?1.f/l_i:0.f;
  int grow=m0+tid;
  #pragma unroll
  for(int c=0;c<4;c++){
    uint32_t o[32];
    #pragma unroll
    for(int q=0;q<8;q++) ldx4(Obase+c*32+q*4,o[q*4],o[q*4+1],o[q*4+2],o[q*4+3]);
    waitld();
    #pragma unroll
    for(int g=0;g<4;g++){
      int col=c*32+g*8;
      uint4 pk;
      pk.x=packbf(__uint_as_float(o[g*8+0])*inv,__uint_as_float(o[g*8+1])*inv);
      pk.y=packbf(__uint_as_float(o[g*8+2])*inv,__uint_as_float(o[g*8+3])*inv);
      pk.z=packbf(__uint_as_float(o[g*8+4])*inv,__uint_as_float(o[g*8+5])*inv);
      pk.w=packbf(__uint_as_float(o[g*8+6])*inv,__uint_as_float(o[g*8+7])*inv);
      if(grow<S) *(uint4*)&Og[head_off+(long)grow*Dh+col]=pk;
    }
  }
  if(grow<S){ long lo=((long)(b*H+h))*(long)S+grow; LSEg[lo]=m_i+logf(l_i); }
  __syncthreads();
  if(warp==0) tmem_dealloc1(tbase,256);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int Bn=(int)Q.size(0),Hn=(int)Q.size(1),Sn=(int)Q.size(2);
  const __nv_bfloat16* Qp=static_cast<const __nv_bfloat16*>(Q.data_ptr());
  const __nv_bfloat16* Kp=static_cast<const __nv_bfloat16*>(K.data_ptr());
  const __nv_bfloat16* Vp=static_cast<const __nv_bfloat16*>(V.data_ptr());
  __nv_bfloat16* Op=static_cast<__nv_bfloat16*>(O.data_ptr());
  float* Lp=static_cast<float*>(LSE.data_ptr());
  int num_m=(Sn+BM-1)/BM;
  dim3 grid(num_m,Hn,Bn);
  int smem_bytes=81920;
  cudaStream_t stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type,Q.device().device_id));
  CUDA_CHECK(cudaFuncSetAttribute(attn,cudaFuncAttributeMaxDynamicSharedMemorySize,smem_bytes));
  attn<<<grid,NT,smem_bytes,stream>>>(Qp,Kp,Vp,Op,Lp,Bn,Hn,Sn);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_kernel::run);

}  // namespace mha_kernel