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
constexpr int TILE_BYTES=(Dh/8)*BM*16;   // 32768
constexpr int LBO_K=BM*16, SBO_K=128;    // K-major Q/K/P
constexpr int LBO_V=128,   SBO_V=BN*16;  // MN-major V

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
template<int N> __device__ __forceinline__ void cp_waitg(){ asm volatile("cp.async.wait_group %0;\n"::"n"(N):"memory"); }
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

__launch_bounds__(128,1)
__global__ void attn(const __nv_bfloat16* __restrict__ Qg,const __nv_bfloat16* __restrict__ Kg,
                     const __nv_bfloat16* __restrict__ Vg,__nv_bfloat16* __restrict__ Og,
                     float* __restrict__ LSEg,int B,int H,int S){
  const int b=blockIdx.z,h=blockIdx.y,m0=blockIdx.x*BM;
  const int tid=threadIdx.x,warp=tid>>5;
  const bool leader=(tid==0);
  const float scale=rsqrtf((float)Dh);

  extern __shared__ char smem[];
  char* qs=smem;
  char* ks=qs+TILE_BYTES;          // 2 buffers
  char* vs=ks+2*TILE_BYTES;        // 2 buffers
  char* ps=vs+2*TILE_BYTES;
  __shared__ uint64_t mbar[3];
  __shared__ uint32_t tbase_s[1];

  const long head_off=((long)(b*H+h))*(long)S*Dh;
  const int num_kv=(S+BN-1)/BN;

  if(leader){ init_bar(&mbar[0],1); init_bar(&mbar[1],1); init_bar(&mbar[2],1); }
  fence_bar_init(); __syncthreads();
  if(warp==0) tmem_alloc1(&tbase_s[0],512);
  __syncthreads();
  uint32_t tbase=tbase_s[0];
  uint32_t S0=tbase, S1=tbase+128, Oc=tbase+256;

  auto loadQ=[&](){
    for(int i=tid;i<BM*16;i+=NT){int m=i>>4,kb=i&15;int gr=m0+m;
      char* d=qs+kb*LBO_K+m*16;
      if(gr<S) cp_async16(d,&Qg[head_off+(long)gr*Dh+kb*8]); else *(int4*)d=make_int4(0,0,0,0);}
  };
  auto loadK=[&](int buf,int kvs){
    for(int i=tid;i<BN*16;i+=NT){int n=i>>4,kb=i&15;int gr=kvs+n;
      char* d=ks+buf*TILE_BYTES+kb*LBO_K+n*16;
      if(gr<S) cp_async16(d,&Kg[head_off+(long)gr*Dh+kb*8]); else *(int4*)d=make_int4(0,0,0,0);}
  };
  auto loadV=[&](int buf,int kvs){
    for(int i=tid;i<BN*16;i+=NT){int k=i>>4,nb=i&15;int gr=kvs+k;
      char* d=vs+buf*TILE_BYTES+nb*SBO_V+k*16;
      if(gr<S) cp_async16(d,&Vg[head_off+(long)gr*Dh+nb*8]); else *(int4*)d=make_int4(0,0,0,0);}
  };
  auto mma1=[&](int sbuf,int kbuf){
    uint32_t id=make_idesc(BM,BN,0,0);
    #pragma unroll
    for(int k=0;k<8;k++){
      uint64_t da=make_desc(qs+k*4096,LBO_K,SBO_K);
      uint64_t db=make_desc(ks+kbuf*TILE_BYTES+k*4096,LBO_K,SBO_K);
      umma1(sbuf?S1:S0,da,db,id,k==0?0:1);
    }
    umma_commit1(&mbar[sbuf]);
  };
  auto mma2=[&](int vbuf,bool firstO){
    uint32_t id=make_idesc(BM,Dh,0,1);
    #pragma unroll
    for(int k=0;k<8;k++){
      uint64_t da=make_desc(ps+k*4096,LBO_K,SBO_K);
      uint64_t db=make_desc(vs+vbuf*TILE_BYTES+k*256,LBO_V,SBO_V);
      umma1(Oc,da,db,id,(firstO&&k==0)?0:1);
    }
    umma_commit1(&mbar[2]);
  };

  // ---- prologue loads ----
  loadQ(); cp_commit();
  loadK(0,0); cp_commit();
  loadV(0,0); cp_commit();
  if(num_kv>1){ loadK(1,BN); cp_commit(); }
  cp_waitg<0>();
  __syncthreads(); fence_proxy_async(); __syncthreads();
  if(leader) mma1(0,0);   // MMA1(0) -> S0

  float m_i=NEG,l_i=0.f;
  uint32_t phS0=0,phS1=0,phO=0;

  for(int j=0;j<num_kv;j++){
    int cur=j&1;
    int kv_start=j*BN;
    uint32_t Sbase=cur?S1:S0;

    if(cur==0){ bar_wait(&mbar[0],phS0); phS0^=1; }
    else      { bar_wait(&mbar[1],phS1); phS1^=1; }

    // issue MMA1(j+1), overlaps softmax(j)
    if(j+1<num_kv){
      if(j+2<num_kv){ loadK(j&1,(j+2)*BN); cp_commit(); cp_waitg<1>(); }
      else { cp_waitg<0>(); }
      __syncthreads(); fence_proxy_async(); __syncthreads();
      int nb=(j+1)&1;
      if(leader) mma1(nb,nb);
    }

    // softmax pass1: row max
    float rmax=NEG;
    #pragma unroll
    for(int c=0;c<4;c++){
      uint32_t vv[32];
      #pragma unroll
      for(int q=0;q<8;q++) ldx4(Sbase+c*32+q*4,vv[q*4],vv[q*4+1],vv[q*4+2],vv[q*4+3]);
      waitld();
      #pragma unroll
      for(int e=0;e<32;e++){int col=c*32+e;float s=(kv_start+col<S)?__uint_as_float(vv[e])*scale:NEG;rmax=fmaxf(rmax,s);}
    }
    float nm=fmaxf(m_i,rmax);
    float corr=__expf(m_i-nm);

    // wait MMA2(j-1), rescale O
    if(j>0){ bar_wait(&mbar[2],phO); phO^=1; }
    if(j>0){
      #pragma unroll
      for(int c=0;c<4;c++){
        uint32_t oo[32];
        #pragma unroll
        for(int q=0;q<8;q++) ldx4(Oc+c*32+q*4,oo[q*4],oo[q*4+1],oo[q*4+2],oo[q*4+3]);
        waitld();
        #pragma unroll
        for(int e=0;e<32;e++) oo[e]=__float_as_uint(__uint_as_float(oo[e])*corr);
        #pragma unroll
        for(int q=0;q<8;q++) stx4(Oc+c*32+q*4,oo[q*4],oo[q*4+1],oo[q*4+2],oo[q*4+3]);
      }
      waitst();
    }

    // prefetch V(j+1)
    if(j+1<num_kv){ loadV((j+1)&1,(j+1)*BN); cp_commit(); cp_waitg<1>(); }
    else { cp_waitg<0>(); }

    // softmax pass2: p=exp(s-nm) -> P (smem), sum
    float rsum=0.f;
    #pragma unroll
    for(int c=0;c<4;c++){
      uint32_t vv[32];
      #pragma unroll
      for(int q=0;q<8;q++) ldx4(Sbase+c*32+q*4,vv[q*4],vv[q*4+1],vv[q*4+2],vv[q*4+3]);
      waitld();
      float pp[32];
      #pragma unroll
      for(int e=0;e<32;e++){int col=c*32+e;float s=(kv_start+col<S)?__uint_as_float(vv[e])*scale:NEG;pp[e]=__expf(s-nm);rsum+=pp[e];}
      #pragma unroll
      for(int g=0;g<4;g++){
        int cb=c*4+g;
        uint4 pk;
        pk.x=packbf(pp[g*8+0],pp[g*8+1]); pk.y=packbf(pp[g*8+2],pp[g*8+3]);
        pk.z=packbf(pp[g*8+4],pp[g*8+5]); pk.w=packbf(pp[g*8+6],pp[g*8+7]);
        *(uint4*)(ps+cb*LBO_K+tid*16)=pk;
      }
    }
    l_i=l_i*corr+rsum; m_i=nm;

    // issue MMA2(j)
    __syncthreads(); fence_proxy_async(); __syncthreads();
    if(leader) mma2(cur,j==0);
  }

  // epilogue
  bar_wait(&mbar[2],phO); phO^=1;
  float inv=(l_i>0.f)?1.f/l_i:0.f;
  int grow=m0+tid;
  #pragma unroll
  for(int c=0;c<4;c++){
    uint32_t oo[32];
    #pragma unroll
    for(int q=0;q<8;q++) ldx4(Oc+c*32+q*4,oo[q*4],oo[q*4+1],oo[q*4+2],oo[q*4+3]);
    waitld();
    #pragma unroll
    for(int g=0;g<4;g++){
      int col=c*32+g*8;
      uint4 pk;
      pk.x=packbf(__uint_as_float(oo[g*8+0])*inv,__uint_as_float(oo[g*8+1])*inv);
      pk.y=packbf(__uint_as_float(oo[g*8+2])*inv,__uint_as_float(oo[g*8+3])*inv);
      pk.z=packbf(__uint_as_float(oo[g*8+4])*inv,__uint_as_float(oo[g*8+5])*inv);
      pk.w=packbf(__uint_as_float(oo[g*8+6])*inv,__uint_as_float(oo[g*8+7])*inv);
      if(grow<S) *(uint4*)&Og[head_off+(long)grow*Dh+col]=pk;
    }
  }
  if(grow<S){ long lo=((long)(b*H+h))*(long)S+grow; LSEg[lo]=m_i+logf(l_i); }
  __syncthreads();
  if(warp==0) tmem_dealloc1(tbase,512);
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
  int smem_bytes=6*TILE_BYTES;
  cudaStream_t stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type,Q.device().device_id));
  CUDA_CHECK(cudaFuncSetAttribute(attn,cudaFuncAttributeMaxDynamicSharedMemorySize,smem_bytes));
  attn<<<grid,NT,smem_bytes,stream>>>(Qp,Kp,Vp,Op,Lp,Bn,Hn,Sn);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_kernel::run);

}  // namespace mha_kernel